"""
Adaptive sweep benchmark: find the replica count and CUDA MPS setting that
give good exchange rates and the best throughput.

Algorithm
---------
Phase 1 — exchange-rate search (small probes, run in parallel)
    Each probe is a tiny HREMD ladder (default 4 replicas) at the *cold end*
    of the temperature range, spaced geometrically by a ratio
    ``r = T[i+1] / T[i]``.  The cold end is the bottleneck for geometric
    ladders, so if exchange is acceptable there it is acceptable everywhere.

    On a single node, one probe runs per GPU at the same time (each probe's
    replicas share its GPU via CUDA MPS), so every round tests ``n_gpus``
    different spacings.

    All observed neighbour pairs are pooled and fit to a simple model of
    replica-exchange acceptance for Gaussian energy distributions::

        P_exchange ≈ erfc(k · ln r)

    where ``k`` is a single system-dependent constant (roughly
    ∝ sqrt(heat capacity)).  The fit predicts the ratio ``r*`` that gives the
    target exchange rate, and the next round's probes are centred on it.
    If a round shows almost no exchanges the spacing shrinks aggressively; if
    exchanges are near-certain it widens.  Usually converges in 2–3 rounds.

    Full ladder size: ``N = ceil(ln(T_max/T_min) / ln r*) + 1``

Phase 2 — throughput + validation (full ladder)
    Runs the predicted N-replica ladder across all GPUs for each MPS value
    (default 1–4).  This measures ns/day and also validates the exchange
    prediction on the real ladder (pooled across the runs).

Usage::

    chacra sweep-benchmark -p system.xml -s structure.pdb -j 4
    chacra sweep-benchmark -p system.xml -s structure.pdb -j 4 \\
        --min-temp 290 --max-temp 450 --target-exchange 0.20
    # Skip the search if you already know the replica count:
    chacra sweep-benchmark -p system.xml -s structure.pdb -j 4 -n 24
"""

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np

from chacra.mpi import build_mpi_command, configure_mps_env

# Scratch lives in the CWD so it is on a shared filesystem for multi-node runs.
_SCRATCH = pathlib.Path(".chacra_sweep_scratch")

# Clamp on the searchable ratio range.
_R_MIN = 1.0005
_R_MAX = 1.07

_verfc = np.vectorize(math.erfc)


# ═════════════════════════════════════════════════════════════════════════════
# MPI worker (runs inside mpirun)
# ═════════════════════════════════════════════════════════════════════════════


def _exchange_counts(samples_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (accepted, proposed) swap counts for adjacent state pairs."""
    from chacra.trajectories.process_hremd import load_femto_data

    df = load_femto_data(samples_path)
    acc = np.vstack(df["n_accepted_swaps"].values)
    att = np.vstack(df["n_proposed_swaps"].values)
    n_states = acc.shape[1]
    acc = np.vstack(acc[-1])
    att = np.vstack(att[-1])
    i = np.arange(n_states - 1)
    j = i + 1
    return acc[i, j].astype(float), att[i, j].astype(float)


def _probe_worker(args):
    """Run one short HREMD and write timing + exchange counts to JSON."""
    # A pinned probe owns exactly one physical GPU.  This must be set before
    # any CUDA initialisation, and divide_gpus() must be skipped because it
    # would overwrite CUDA_VISIBLE_DEVICES with a rank-relative index.
    if args._gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id

    import femto.md.config
    import femto.md.constants
    import femto.md.hremd
    import femto.md.rest
    import femto.md.utils.mpi
    import femto.md.utils.openmm
    import MDAnalysis as mda
    import mdtop
    import openmm
    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    if args._gpu_id is None:
        femto.md.utils.mpi.divide_gpus()

    temps = [float(t) for t in args._probe_temps.split(",")]

    with open(args.system_file) as f:
        system = XmlSerializer.deserialize(f.read())

    u = mda.Universe(args.structure_file)
    solute_idxs = set(u.select_atoms(args.lambda_selection).atoms.ix)

    rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)
    femto.md.rest.apply_rest(system, solute_idxs, rest_config)

    pdb = PDBFile(args.structure_file)
    structure = mdtop.Topology.from_file(args.structure_file)

    integrator = LangevinMiddleIntegrator(
        temps[0], 1 / unit.picosecond, args.timestep * unit.femtosecond
    )
    integrator.setRandomNumberSeed(12345)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temps[0], 12345)
    base_state = simulation.context.getState(
        getPositions=True, getVelocities=True, getForces=True,
        getEnergy=True, enforcePeriodicBox=True,
    )
    del simulation

    output_dir = pathlib.Path(args._output_dir)

    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        if mpi_comm.rank == 0 and output_dir.exists():
            shutil.rmtree(output_dir)
        mpi_comm.barrier()

    rest_temperatures = temps * openmm.unit.kelvin
    rest_betas = [
        1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * t) for t in rest_temperatures
    ]
    states = [
        {femto.md.rest.REST_CTX_PARAM: rb / rest_betas[0]} for rb in rest_betas
    ]
    states = [
        femto.md.utils.openmm.evaluate_ctx_parameters(s, system) for s in states
    ]

    integrator_config = femto.md.config.LangevinIntegrator(
        timestep=args.timestep * openmm.unit.femtosecond,
    )
    final_integrator = femto.md.utils.openmm.create_integrator(
        integrator_config, rest_temperatures[0]
    )
    final_integrator.setRandomNumberSeed(12345)

    simulation = femto.md.utils.openmm.create_simulation(
        system, structure, coords=base_state, integrator=final_integrator,
        state=states[0], platform=femto.md.constants.OpenMMPlatform.CUDA,
    )

    # No trajectories / checkpoints during probes — pure timing + exchange data.
    hremd_config = femto.md.config.HREMD(
        n_warmup_steps=args.warmup_steps,
        n_steps_per_cycle=args.steps_per_cycle,
        n_cycles=args.cycles,
        trajectory_interval=args.cycles + 1,
        checkpoint_interval=args.cycles + 1,
    )

    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        mpi_comm.barrier()
        t0 = time.time()
        femto.md.hremd.run_hremd(simulation, states, hremd_config, output_dir=output_dir)
        mpi_comm.barrier()
        wall = time.time() - t0

        if mpi_comm.rank != 0:
            return

        steps = args.cycles * args.steps_per_cycle + args.warmup_steps
        ns_per_replica = steps * args.timestep / 1e6
        results = {
            "temps": temps,
            "wall_time_sec": wall,
            "ns_per_replica": ns_per_replica,
            "ns_per_day_per_replica": ns_per_replica / wall * 86400,
        }

        samples_path = output_dir / "samples.arrow"
        if samples_path.exists():
            try:
                accepted, proposed = _exchange_counts(str(samples_path))
                results["accepted"] = accepted.tolist()
                results["proposed"] = proposed.tolist()
            except Exception as e:  # noqa: BLE001
                results["exchange_error"] = str(e)

        with open(args._results_file, "w") as f:
            json.dump(results, f, indent=2)

        shutil.rmtree(output_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
# Probe launching
# ═════════════════════════════════════════════════════════════════════════════


def _launch(
    temps: list[float],
    *,
    tag: str,
    n_ranks: int,
    gpu_id: str | None,
    opts: argparse.Namespace,
) -> tuple[subprocess.Popen, pathlib.Path]:
    """Start an MPI HREMD run in the background.  Returns (process, probe_dir)."""
    # More ranks than replicas would leave idle ranks — never do that.
    n_ranks = max(1, min(n_ranks, len(temps)))

    probe_dir = _SCRATCH / tag
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)

    cmd = build_mpi_command(n_ranks, mpi_command=opts.mpi_command) + [
        "chacra", "sweep-benchmark", "--_is-probe-worker",
        "-p", opts.system_file,
        "-s", opts.structure_file,
        "-j", "1",
        "--timestep", str(opts.timestep),
        "--lambda_selection", opts.lambda_selection,
        "--cycles", str(opts.cycles),
        "--steps-per-cycle", str(opts.steps_per_cycle),
        "--warmup-steps", str(opts.warmup_steps),
        "--_probe-temps", ",".join(f"{t:.4f}" for t in temps),
        "--_results-file", str(probe_dir / "results.json"),
        "--_output-dir", str(probe_dir / "hremd-output"),
    ]
    if gpu_id is not None:
        cmd += ["--_gpu-id", gpu_id]

    env = os.environ.copy()
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    if n_ranks > 1:
        # Ranks sharing a GPU — same convention as run-hremd.
        ranks_per_gpu = n_ranks if gpu_id is not None else max(1, n_ranks // opts.n_jobs)
        if ranks_per_gpu > 1:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, 200 // ranks_per_gpu))

    log = open(probe_dir / "mpi.log", "wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._chacra_log = log  # keep handle alive; closed in _collect
    return proc, probe_dir


def _collect(proc: subprocess.Popen, probe_dir: pathlib.Path, timeout: float) -> dict:
    """Wait for a launched run and return its results dict (with ``ok`` key)."""
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = None
    finally:
        proc._chacra_log.close()

    results_file = probe_dir / "results.json"
    if rc == 0 and results_file.exists():
        with open(results_file) as f:
            res = json.load(f)
        res["ok"] = True
        return res

    tail = ""
    log_path = probe_dir / "mpi.log"
    if log_path.exists():
        tail = log_path.read_text(errors="replace")[-4000:]
    err = "timed out" if rc is None else f"exit code {rc}"
    return {"ok": False, "error": err, "log_tail": tail}


def _error_line(log_tail: str) -> str:
    """Pick the most informative line from a failed probe's log.

    mpirun ends its error output with a row of dashes, so the last line is
    useless — prefer the last line that looks like a Python exception.
    """
    lines = [ln.strip() for ln in log_tail.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "Error" in ln or "Exception" in ln:
            return ln
    for ln in reversed(lines):
        if set(ln) - set("-="):
            return ln
    return ""


def _gpu_ids(n_gpus: int) -> list[str]:
    """Physical GPU ids to pin probes to, respecting an existing CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        return ids[:n_gpus]
    return [str(i) for i in range(n_gpus)]


# ═════════════════════════════════════════════════════════════════════════════
# Exchange model
# ═════════════════════════════════════════════════════════════════════════════


def _fit_k(obs: list[tuple[float, float, float]]) -> float | None:
    """Fit ``P = erfc(k·x)`` to (x=ln r, accepted, proposed) observations."""
    obs = [o for o in obs if o[2] > 0]
    if not obs:
        return None
    x = np.array([o[0] for o in obs])
    p = np.array([o[1] / o[2] for o in obs])
    w = np.array([o[2] for o in obs])
    ks = np.geomspace(1.0, 5000.0, 3000)
    pred = _verfc(np.outer(ks, x))
    err = ((pred - p) ** 2 * w).sum(axis=1)
    return float(ks[int(np.argmin(err))])


def _erfcinv(p: float) -> float:
    lo, hi = 0.0, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ratio_for_target(k: float, target: float) -> float:
    r = math.exp(_erfcinv(target) / k)
    return min(max(r, _R_MIN), _R_MAX)


def _n_replicas(r: float, t_min: float, t_max: float) -> int:
    return math.ceil(math.log(t_max / t_min) / math.log(r)) + 1


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — adaptive search
# ═════════════════════════════════════════════════════════════════════════════


def _adaptive_search(opts) -> tuple[float, list[dict]]:
    gpu_ids = _gpu_ids(opts.n_jobs)
    parallel = (not opts.serial_probes) and len(gpu_ids) > 1
    n_par = len(gpu_ids) if parallel else 1
    m = opts.probe_replicas

    # Round 1: span a broad range of spacings (≈1–20 K gaps near 300 K).
    if n_par > 1:
        ratios = list(np.geomspace(1.004, 1.035, n_par))
    else:
        ratios = [1.015]

    obs: list[tuple[float, float, float]] = []
    history: list[dict] = []
    r_star: float | None = None

    print(f"\n{'─' * 72}")
    print(f"  Phase 1: adaptive exchange search  (target P_x ≈ {opts.target_px:.2f})")
    mode = f"{n_par} probes in parallel (1 per GPU)" if parallel else "serial probes"
    print(f"  {m} replicas/probe at the cold end ({opts.min_temp:.0f} K), {mode}")
    print(f"{'─' * 72}")
    print(f"  {'Rnd':>3}  {'Ratio':>7}  {'ΔT@Tmin':>8}  {'Mean Px':>8}  {'Min Px':>7}  {'→ N':>5}")

    for rnd in range(1, opts.max_rounds + 1):
        launched = []
        for idx, r in enumerate(ratios):
            temps = [opts.min_temp * r**i for i in range(m)]
            if parallel:
                gpu, n_ranks = gpu_ids[idx], m
            else:
                gpu, n_ranks = None, opts.n_jobs
            proc, pdir = _launch(
                temps, tag=f"r{rnd}_p{idx}", n_ranks=n_ranks, gpu_id=gpu, opts=opts,
            )
            launched.append((r, proc, pdir))
            if not parallel:
                # serial: wait before starting the next probe
                launched[-1] = (r, _collect(proc, pdir, opts.timeout), None)

        results = []
        for r, proc_or_res, pdir in launched:
            res = proc_or_res if pdir is None else _collect(proc_or_res, pdir, opts.timeout)
            results.append((r, res))

        round_px = []
        for r, res in results:
            dT = opts.min_temp * (r - 1)
            if not res.get("ok") or "accepted" not in res:
                msg = res.get("error") or res.get("exchange_error", "no exchange data")
                print(f"  {rnd:>3}  {r:7.4f}  {dT:7.2f}K  FAILED ({msg})")
                if res.get("log_tail"):
                    print("       " + _error_line(res["log_tail"])[:150])
                history.append({"round": rnd, "ratio": r, "ok": False, "error": msg})
                continue
            acc = np.array(res["accepted"])
            prop = np.array(res["proposed"])
            px = np.divide(acc, prop, out=np.zeros_like(acc), where=prop > 0)
            for a, n in zip(acc, prop):
                obs.append((math.log(r), float(a), float(n)))
            round_px.extend(px.tolist())
            print(
                f"  {rnd:>3}  {r:7.4f}  {dT:7.2f}K  {px.mean():8.3f}  {px.min():7.3f}"
                f"  {_n_replicas(r, opts.min_temp, opts.max_temp):>5}"
            )
            history.append({
                "round": rnd, "ratio": r, "ok": True, "temps": res["temps"],
                "exchange_probs": px.tolist(),
                "ns_per_day_per_replica": res.get("ns_per_day_per_replica"),
            })

        if not round_px:
            # Everything failed — most often too-wide spacing blowing up.
            ratios = [math.exp(math.log(r) * 0.5) for r in ratios]
            continue

        # Degenerate rounds: steer explicitly before trusting the fit.
        lnrs = [math.log(r) for r in ratios]
        if max(round_px) < 0.02:
            print("       ↳ almost no exchanges — shrinking spacing")
            base = min(lnrs)
            ratios = [math.exp(base * f) for f in np.linspace(0.2, 0.6, n_par)]
            continue
        if min(round_px) > 0.90:
            print("       ↳ exchanges near-certain — widening spacing")
            base = max(lnrs)
            ratios = [min(_R_MAX, math.exp(base * f)) for f in np.linspace(1.5, 3.0, n_par)]
            continue

        k = _fit_k(obs)
        new_r = _ratio_for_target(k, opts.target_px)
        print(
            f"       ↳ fit k={k:.1f} → r*={new_r:.4f} "
            f"(ΔT@Tmin={opts.min_temp * (new_r - 1):.2f} K, N={_n_replicas(new_r, opts.min_temp, opts.max_temp)})"
        )

        converged = (
            r_star is not None
            and abs(math.log(new_r) - math.log(r_star)) / math.log(r_star) < 0.05
        )
        r_star = new_r
        if converged:
            break

        # Next round: bracket r* in ln-space.
        if n_par > 1:
            ratios = [math.exp(math.log(r_star) * f) for f in np.linspace(0.75, 1.3, n_par)]
        else:
            ratios = [r_star]

    if r_star is None:
        raise RuntimeError(
            "Exchange search failed to produce a usable estimate. "
            "Check the probe logs (rerun with --keep-scratch)."
        )
    return r_star, history


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 — throughput + validation on the full ladder
# ═════════════════════════════════════════════════════════════════════════════


def _throughput_sweep(n_replicas: int, opts) -> tuple[list[dict], dict | None]:
    temps = list(np.geomspace(opts.min_temp, opts.max_temp, n_replicas))
    mps_range = [int(x) for x in opts.mps_range.split(",")]

    print(f"\n{'─' * 72}")
    print(f"  Phase 2: throughput + validation  ({n_replicas} replicas, {opts.n_jobs} GPU(s))")
    print(f"{'─' * 72}")
    print(f"  {'MPS':>4}  {'Ranks':>5}  {'ns/day/rep':>11}  {'Agg ns/day':>11}  {'Min Px':>7}  {'Wall':>7}")

    runs = []
    pooled_acc = np.zeros(n_replicas - 1)
    pooled_prop = np.zeros(n_replicas - 1)

    for mps in mps_range:
        n_ranks = opts.n_jobs * mps
        if n_ranks > n_replicas:
            print(f"  {mps:>4}  {n_ranks:>5}  skipped (more ranks than replicas)")
            continue
        proc, pdir = _launch(temps, tag=f"tp_mps{mps}", n_ranks=n_ranks, gpu_id=None, opts=opts)
        res = _collect(proc, pdir, opts.timeout)
        if not res.get("ok"):
            print(f"  {mps:>4}  {n_ranks:>5}  FAILED ({res.get('error')})")
            runs.append({"mps_replicas": mps, "ok": False, "error": res.get("error")})
            continue

        ns_day = res["ns_per_day_per_replica"]
        min_px = None
        if "accepted" in res:
            acc, prop = np.array(res["accepted"]), np.array(res["proposed"])
            pooled_acc += acc
            pooled_prop += prop
            px = np.divide(acc, prop, out=np.zeros_like(acc), where=prop > 0)
            min_px = float(px.min())
        min_str = f"{min_px:7.3f}" if min_px is not None else "      —"
        print(
            f"  {mps:>4}  {n_ranks:>5}  {ns_day:11.2f}  {ns_day * n_replicas:11.2f}"
            f"  {min_str}  {res['wall_time_sec']:6.0f}s"
        )
        runs.append({
            "mps_replicas": mps, "ok": True, "ranks": n_ranks,
            "ns_per_day_per_replica": ns_day,
            "agg_ns_per_day": ns_day * n_replicas,
            "wall_time_sec": res["wall_time_sec"],
        })

    validation = None
    if pooled_prop.sum() > 0:
        px = np.divide(pooled_acc, pooled_prop, out=np.zeros_like(pooled_acc), where=pooled_prop > 0)
        validation = {"temps": temps, "exchange_probs": px.tolist()}
    return runs, validation


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════


def _print_summary(r_star, n_replicas, runs, validation, opts):
    print(f"\n{'═' * 72}")
    print("  SWEEP BENCHMARK RESULTS")
    print(f"{'═' * 72}")
    print(f"  Temperature range : {opts.min_temp:.0f} – {opts.max_temp:.0f} K")
    if r_star is not None:
        print(f"  Spacing ratio     : {r_star:.4f}  (ΔT at {opts.min_temp:.0f} K ≈ {opts.min_temp * (r_star - 1):.2f} K)")
    print(f"  Replicas          : {n_replicas}")

    if validation:
        px = np.array(validation["exchange_probs"])
        temps = validation["temps"]
        worst = int(px.argmin())
        print(f"\n  Exchange on full ladder (pooled over Phase 2 runs):")
        print(f"    mean {px.mean():.3f}   min {px.min():.3f}   max {px.max():.3f}")
        print(f"    worst pair: {temps[worst]:.1f} K ↔ {temps[worst + 1]:.1f} K")
        if px.min() < 0.7 * opts.target_px:
            print(f"    ⚠  Worst pair well below target — consider ~{int(n_replicas * 1.15) + 1} replicas,")
            print(f"       or rerun with more --cycles for better statistics.")

    ok_runs = [r for r in runs if r.get("ok")]
    if ok_runs:
        best = max(ok_runs, key=lambda r: r["agg_ns_per_day"])
        base = next((r for r in ok_runs if r["mps_replicas"] == 1), None)
        print(f"\n  ★ Recommended:  -n {n_replicas} --mps-replicas {best['mps_replicas']}")
        print(f"      {best['ns_per_day_per_replica']:.2f} ns/day/replica, "
              f"{best['agg_ns_per_day']:.1f} ns/day aggregate")
        if base and best is not base:
            gain = best["agg_ns_per_day"] / base["agg_ns_per_day"]
            print(f"      ({gain:.2f}× vs --mps-replicas 1)")
    print(f"{'═' * 72}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Adaptively find the HREMD replica count and --mps-replicas "
                    "setting that give good exchange rates and the best throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--system_file", required=True, help="System XML file.")
    parser.add_argument("-s", "--structure_file", required=True, help="Structure PDB file.")
    parser.add_argument("-j", "--n_jobs", type=int, required=True, help="Number of GPUs.")
    parser.add_argument("--min-temp", type=float, default=290.0, dest="min_temp",
                        help="Minimum effective temperature (K).")
    parser.add_argument("--max-temp", type=float, default=450.0, dest="max_temp",
                        help="Maximum effective temperature (K).")
    parser.add_argument("--target-exchange", type=float, default=0.20, dest="target_px",
                        help="Target exchange probability for the worst (coldest) pair.")
    parser.add_argument("-n", "--n-replicas", type=int, default=None, dest="n_replicas",
                        help="Skip the exchange search and benchmark this replica count.")
    parser.add_argument("--probe-replicas", type=int, default=4, dest="probe_replicas",
                        help="Replicas per Phase-1 probe (2–4 recommended).")
    parser.add_argument("--max-rounds", type=int, default=4, dest="max_rounds",
                        help="Maximum Phase-1 search rounds.")
    parser.add_argument("--mps-range", type=str, default="1,2,3,4", dest="mps_range",
                        help="Comma-separated --mps-replicas values for Phase 2.")
    parser.add_argument("--cycles", type=int, default=50,
                        help="HREMD cycles per probe (more = better statistics).")
    parser.add_argument("--steps-per-cycle", type=int, default=1000, dest="steps_per_cycle",
                        help="MD steps per cycle.")
    parser.add_argument("--warmup-steps", type=int, default=5000, dest="warmup_steps",
                        help="Equilibration steps before exchanges start in each probe.")
    parser.add_argument("--timestep", type=int, default=2, help="Timestep (fs).")
    parser.add_argument("--lambda_selection", type=str, default="protein",
                        help="MDAnalysis selection for REST2 scaling.")
    parser.add_argument("--mpi-command", type=str, default=None, dest="mpi_command",
                        help="MPI launcher (auto-detected if omitted).")
    parser.add_argument("--serial-probes", action="store_true", dest="serial_probes",
                        help="Run Phase-1 probes one at a time across all GPUs instead of "
                             "one probe per GPU in parallel (use if your launcher can't "
                             "run concurrent jobs, e.g. some srun setups).")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="Per-run timeout in seconds.")
    parser.add_argument("--save-results", type=str, default="sweep_results.json",
                        dest="save_results", help="Write detailed results to this JSON file.")
    parser.add_argument("--keep-scratch", action="store_true", dest="keep_scratch",
                        help="Keep probe logs in .chacra_sweep_scratch/ for debugging.")

    # Internal worker flags
    parser.add_argument("--_is-probe-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_probe-temps", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_results-file", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_output-dir", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_gpu-id", type=str, default=None, help=argparse.SUPPRESS)

    opts = parser.parse_args()

    if opts._is_probe_worker:
        _probe_worker(opts)
        return

    print("=" * 72)
    print("  ChACRA HREMD Sweep Benchmark")
    print("=" * 72)
    print(f"  System      : {opts.system_file}")
    print(f"  Structure   : {opts.structure_file}")
    print(f"  GPUs        : {opts.n_jobs}")
    print(f"  Temp range  : {opts.min_temp:.0f} – {opts.max_temp:.0f} K")
    print(f"  Cycles/run  : {opts.cycles} × {opts.steps_per_cycle} steps "
          f"(+{opts.warmup_steps} warmup)")

    _SCRATCH.mkdir(exist_ok=True)

    # MPS lets a probe's replicas share one GPU concurrently, and is needed
    # for Phase 2 runs with --mps-replicas > 1.
    mps_started = False
    fmpi = None
    try:
        import femto.md.utils.mpi as fmpi
        if not fmpi.is_mps_running():
            configure_mps_env()
            fmpi.start_mps()
            mps_started = True
    except Exception as e:  # noqa: BLE001
        print(f"  Note: CUDA MPS not started ({e}); shared-GPU runs will time-slice.")

    t_start = time.time()
    try:
        r_star, history = None, []
        if opts.n_replicas is None:
            r_star, history = _adaptive_search(opts)
            n_replicas = _n_replicas(r_star, opts.min_temp, opts.max_temp)
        else:
            n_replicas = opts.n_replicas

        runs, validation = _throughput_sweep(n_replicas, opts)
        elapsed = time.time() - t_start

        _print_summary(r_star, n_replicas, runs, validation, opts)
        print(f"  Total sweep time: {elapsed / 60:.1f} min")

        if opts.save_results:
            with open(opts.save_results, "w") as f:
                json.dump({
                    "min_temp": opts.min_temp, "max_temp": opts.max_temp,
                    "target_px": opts.target_px, "ratio": r_star,
                    "n_replicas": n_replicas, "search_history": history,
                    "throughput": runs, "validation": validation,
                    "elapsed_sec": elapsed,
                }, f, indent=2)
            print(f"  Detailed results: {opts.save_results}\n")
    finally:
        if mps_started and fmpi is not None:
            fmpi.stop_mps()
        if not opts.keep_scratch:
            shutil.rmtree(_SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
