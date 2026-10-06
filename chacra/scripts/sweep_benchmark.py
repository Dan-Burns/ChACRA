"""
Sweep benchmark: single-GPU MPS throughput + static exchange-rate sweep.

Gives you the two numbers needed to plan an HREMD run on *any* amount of
hardware — how many replicas you need, and how many replicas to pack onto
each GPU — without committing to a particular GPU count up front.

Phase 1 — single-GPU MPS throughput
    Runs a short OpenMM MD simulation (default 10,000 steps) on one GPU with
    1, 2, 3, 4 concurrent replicas sharing it via CUDA MPS, and reports ns/day
    per replica and aggregate ns/day per GPU.  The best row is your ``--mps-replicas``.

Phase 2 — static exchange sweep
    Runs a 4-replica ladder at the *cold end* of the temperature range with
    neighbour gap ΔT (default 4, 7, 10 K).  The cold end is the
    bottleneck for a geometric ladder, so a gap that exchanges well there
    exchanges at least as well everywhere.

    The neighbour-pair counts are pooled and the mean acceptance probability
    is reported. You can choose a gap that produces a suitable acceptance rate.

    Full ladder size for a gap:  ``N = ceil(ln(T_max/T_min) / ln r) + 1``

Optional — full ladder (``--test-full-ladder``)
    Once you've decided how to distribute replicas, validate the real ladder
    (``-n`` replicas across ``-j`` GPUs × ``--mps-replicas``).

Usage::

    chacra sweep-benchmark -p system.xml -s structure.pdb
    chacra sweep-benchmark -p system.xml -s structure.pdb \\
        --min-temp 290 --max-temp 450 --target-exchange 0.20
    # Then validate a concrete layout:
    chacra sweep-benchmark -p system.xml -s structure.pdb --full-ladder-only \\
        -n 24 -j 4 --mps-replicas 2
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

    from chacra.simulation import build_hremd_base_state
    system, structure, base_state = build_hremd_base_state(
        system_file=args.system_file,
        structure_file=args.structure_file,
        lambda_selection=args.lambda_selection,
        temperature=temps[0],
        timestep=args.timestep,
    )

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


def _throughput_worker(args):
    """Run pure MD simulation on each rank (no HREMD swaps) to measure raw throughput under MPS."""
    if args._gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id

    import femto.md.config
    import femto.md.constants
    import femto.md.rest
    import femto.md.utils.mpi
    import femto.md.utils.openmm
    import MDAnalysis as mda
    import mdtop
    import openmm
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    if args._gpu_id is None:
        femto.md.utils.mpi.divide_gpus()

    from chacra.simulation import build_hremd_base_state
    system, structure, base_state = build_hremd_base_state(
        system_file=args.system_file,
        structure_file=args.structure_file,
        lambda_selection=args.lambda_selection,
        temperature=args.min_temp,
        timestep=args.timestep,
    )

    temp = args.min_temp * openmm.unit.kelvin
    state = femto.md.utils.openmm.evaluate_ctx_parameters(
        {femto.md.rest.REST_CTX_PARAM: 1.0}, system
    )

    integrator_config = femto.md.config.LangevinIntegrator(
        timestep=args.timestep * openmm.unit.femtosecond,
    )
    integrator = femto.md.utils.openmm.create_integrator(integrator_config, temp)
    integrator.setRandomNumberSeed(12345)

    simulation = femto.md.utils.openmm.create_simulation(
        system, structure, coords=base_state, integrator=integrator,
        state=state, platform=femto.md.constants.OpenMMPlatform.CUDA,
    )

    steps = args._benchmark_steps

    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        # Warmup a few steps to JIT kernels
        simulation.step(100)
        mpi_comm.barrier()

        t0 = time.time()
        simulation.step(steps)
        mpi_comm.barrier()
        wall = time.time() - t0

        if mpi_comm.rank != 0:
            return

        ns_per_replica = steps * args.timestep / 1e6
        ns_per_day = ns_per_replica / wall * 86400
        results = {
            "wall_time_sec": wall,
            "steps": steps,
            "ns_per_day_per_replica": ns_per_day,
        }
        with open(args._results_file, "w") as f:
            json.dump(results, f, indent=2)


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
    cycles: int,
    warmup_steps: int,
    n_gpus: int = 1,
) -> tuple[subprocess.Popen, pathlib.Path]:
    """Start an MPI HREMD run in the background.  Returns (process, probe_dir)."""
    # More ranks than replicas would leave idle ranks — never do that.
    n_ranks = max(1, min(n_ranks, len(temps)))

    probe_dir = _SCRATCH / tag
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)

    import sys
    cmd = build_mpi_command(n_ranks, mpi_command=opts.mpi_command) + [
        sys.executable, __file__, "--_is-probe-worker",
        "-p", opts.system_file,
        "-s", opts.structure_file,
        "--timestep", str(opts.timestep),
        "--lambda_selection", opts.lambda_selection,
        "--cycles", str(cycles),
        "--steps-per-cycle", str(opts.steps_per_cycle),
        "--warmup-steps", str(warmup_steps),
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
    # Ranks sharing a GPU — same convention as run-hremd.
    ranks_per_gpu = n_ranks if gpu_id is not None else max(1, n_ranks // n_gpus)
    if ranks_per_gpu > 1:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, 200 // ranks_per_gpu))

    log = open(probe_dir / "mpi.log", "wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._chacra_log = log  # keep handle alive; closed in _collect
    return proc, probe_dir


def _launch_throughput(
    *,
    n_ranks: int,
    steps: int,
    tag: str,
    gpu_id: str | None,
    opts: argparse.Namespace,
) -> tuple[subprocess.Popen, pathlib.Path]:
    """Start an MPI run to benchmark pure MD throughput on n_ranks sharing a GPU."""
    probe_dir = _SCRATCH / tag
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)

    import sys
    cmd = build_mpi_command(n_ranks, mpi_command=opts.mpi_command) + [
        sys.executable, __file__, "--_is-throughput-worker",
        "-p", opts.system_file,
        "-s", opts.structure_file,
        "--min-temp", str(opts.min_temp),
        "--timestep", str(opts.timestep),
        "--lambda_selection", opts.lambda_selection,
        "--_benchmark-steps", str(steps),
        "--_results-file", str(probe_dir / "results.json"),
    ]
    if gpu_id is not None:
        cmd += ["--_gpu-id", gpu_id]

    env = os.environ.copy()
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    if n_ranks > 1:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, 200 // n_ranks))

    log = open(probe_dir / "mpi.log", "wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._chacra_log = log
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


def _run(temps, *, tag, n_ranks, gpu_id, opts, cycles, warmup_steps, n_gpus=1) -> dict:
    proc, pdir = _launch(
        temps, tag=tag, n_ranks=n_ranks, gpu_id=gpu_id, opts=opts,
        cycles=cycles, warmup_steps=warmup_steps, n_gpus=n_gpus,
    )
    return _collect(proc, pdir, opts.timeout)


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


def _print_failure(prefix: str, res: dict) -> None:
    print(f"{prefix}FAILED ({res.get('error') or res.get('exchange_error', 'no exchange data')})")
    if res.get("log_tail"):
        print("        " + _error_line(res["log_tail"])[:150])


def _get_available_gpus(opts) -> list[str]:
    """List of available physical GPUs."""
    if opts.gpu is not None:
        return [str(opts.gpu)]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if ids:
            return ids
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True
        )
        ids = [x.strip() for x in out.splitlines() if x.strip()]
        if ids:
            return ids
    except Exception:
        pass
    return ["0"]


# ═════════════════════════════════════════════════════════════════════════════
# Exchange model
# ═════════════════════════════════════════════════════════════════════════════


def _gap_to_ratio(gap: float, t_min: float) -> float:
    return 1.0 + gap / t_min


def _n_replicas(r: float, t_min: float, t_max: float) -> int:
    return math.ceil(math.log(t_max / t_min) / math.log(r)) + 1


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — single-GPU MPS throughput
# ═════════════════════════════════════════════════════════════════════════════


def _mps_sweep(opts, gpus: list[str]) -> list[dict]:
    mps_values = [int(x) for x in opts.mps_range.split(",") if x.strip()]
    steps = opts.mps_steps

    print(f"\n{'─' * 72}")
    print(f"  Phase 1: single-GPU MPS throughput  (GPUs {','.join(gpus)}, {steps} steps/run)")
    print(f"  Measuring {steps} simulation steps across 1 to {max(mps_values)} concurrent replicas")
    print(f"{'─' * 72}")
    print(f"  {'Replicas/GPU':>12}  {'ns/day/replica':>14}  {'ns/day/GPU':>11}  {'Speedup':>8}  {'Wall':>6}")

    runs: list[dict] = []
    
    import concurrent.futures
    
    def run_one(k, gpu_id):
        proc, pdir = _launch_throughput(
            n_ranks=k, steps=steps, tag=f"mps_{k}", gpu_id=gpu_id, opts=opts
        )
        res = _collect(proc, pdir, opts.timeout)
        return k, res

    futures = []
    # Submit tasks across all GPUs
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for i, k in enumerate(mps_values):
            gpu_id = gpus[i % len(gpus)]
            futures.append(executor.submit(run_one, k, gpu_id))
            
        base_agg = None
        # Collect results in order so they print cleanly
        for f in futures:
            k, res = f.result()
            if not res.get("ok"):
                _print_failure(f"  {k:>12}  ", res)
                runs.append({"mps_replicas": k, "ok": False, "error": res.get("error")})
                continue
    
            per_rep = res["ns_per_day_per_replica"]
            agg = per_rep * k
            if base_agg is None and k == 1:
                base_agg = agg
            speedup = (agg / base_agg) if base_agg else 1.0
            print(f"  {k:>12}  {per_rep:14.2f}  {agg:11.2f}  {speedup:7.2f}×  {res['wall_time_sec']:5.0f}s")
            runs.append({
                "mps_replicas": k, "ok": True,
                "ns_per_day_per_replica": per_rep, "ns_per_day_per_gpu": agg,
                "speedup": speedup, "wall_time_sec": res["wall_time_sec"],
            })
    return runs


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 — exchange sweep
# ═════════════════════════════════════════════════════════════════════════════


def _exchange_sweep(opts, gpus: list[str]) -> list[dict]:
    m = opts.probe_replicas
    t_min = opts.min_temp

    deltas = [float(x.strip()) for x in opts.temp_deltas.split(",")]

    print(f"\n{'─' * 72}")
    print(f"  Phase 2: static exchange sweep")
    print(f"  {m} replicas per probe on GPUs {','.join(gpus)} at the cold end ({t_min:.0f} K), "
          f"{opts.cycles} cycles/probe")
    print(f"{'─' * 72}")
    print(f"  {'ΔT':>7}  {'Mean Px':>8}  {'Min Px':>7}  {'Max Px':>7}  {'N':>4}")

    probes: list[dict] = []
    
    import concurrent.futures
    
    def run_one(gap, gpu_id):
        r = _gap_to_ratio(gap, t_min)
        temps = [t_min * r**i for i in range(m)]
        res = _run(temps, tag=f"x_{gap}K", n_ranks=m, gpu_id=gpu_id, opts=opts,
                   cycles=opts.cycles, warmup_steps=opts.warmup_steps)
        return gap, r, res

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for i, gap in enumerate(deltas):
            gpu_id = gpus[i % len(gpus)]
            futures.append(executor.submit(run_one, gap, gpu_id))
            
        for f in futures:
            gap, r, res = f.result()
            if not res.get("ok") or "accepted" not in res:
                _print_failure(f"  {gap:6.2f}K  ", res)
                probes.append({"gap": gap, "ok": False, "error": res.get("error")})
                continue
    
            acc = np.array(res["accepted"])
            prop = np.array(res["proposed"])
            px = np.divide(acc, prop, out=np.zeros_like(acc), where=prop > 0)
            n_full = _n_replicas(r, t_min, opts.max_temp)
            print(f"  {gap:6.2f}K  {px.mean():8.3f}  {px.min():7.3f}  {px.max():7.3f}  {n_full:>4}")
    
            probes.append({
                "gap": gap, "ratio": r, "ok": True,
                "exchange_probs": px.tolist(), "accepted": acc.tolist(),
                "proposed": prop.tolist(), "pooled_px": float(acc.sum() / max(prop.sum(), 1)),
                "n_replicas_full": n_full,
            })

    return probes


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════


def _print_summary(opts, mps_runs, probes):
    t_min, t_max = opts.min_temp, opts.max_temp
    print(f"\n{'═' * 72}")
    print("  SWEEP BENCHMARK RESULTS")
    print(f"{'═' * 72}")
    if not getattr(opts, "mps_only", False):
        print(f"  Temperature range : {t_min:.0f} – {t_max:.0f} K")

    ok_mps = [r for r in mps_runs if r.get("ok")]
    if ok_mps:
        print("\n  Single-GPU throughput (CUDA MPS):")
        print(f"    {'Replicas/GPU':>12}  {'ns/day/replica':>14}  {'ns/day/GPU':>11}  {'Speedup':>8}")
        for r in ok_mps:
            print(f"    {r['mps_replicas']:>12}  {r['ns_per_day_per_replica']:14.2f}"
                  f"  {r['ns_per_day_per_gpu']:11.2f}  {r['speedup']:7.2f}×")

    ok_probes = sorted((p for p in probes if p.get("ok")), key=lambda p: p["gap"])
    if ok_probes:
        print("\n  Measured exchange (cold end):")
        print(f"    {'ΔT':>7}  {'Pooled Px':>9}  {'Min pair':>8}  {'Max pair':>8}  {'N for range':>11}")
        for p in ok_probes:
            print(f"    {p['gap']:6.2f}K  {p['pooled_px']:9.3f}  "
                  f"  {min(p['exchange_probs']):8.3f}  {max(p['exchange_probs']):8.3f}  {p['n_replicas_full']:>11}")

    if not getattr(opts, "mps_only", False) and not ok_probes:
        print("\n  ★ Exchange sweep produced no usable data — rerun with --keep-scratch.")

    if ok_mps:
        top = max(ok_mps, key=lambda r: r["ns_per_day_per_gpu"])
        print(f"\n  ★ Recommendations:")
        print(f"    • Best throughput: --mps-replicas {top['mps_replicas']}  "
              f"({top['ns_per_day_per_gpu']:.1f} ns/day/GPU, {top['speedup']:.2f}× vs 1)")

    print(f"{'═' * 72}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Measure single-GPU MPS throughput and find the temperature "
                    "gap that gives the target exchange rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--system_file", required=True, help="System XML file.")
    parser.add_argument("-s", "--structure_file", required=True, help="Structure PDB file.")
    parser.add_argument("--min-temp", type=float, default=290.0, dest="min_temp",
                        help="Minimum effective temperature (K).")
    parser.add_argument("--max-temp", type=float, default=450.0, dest="max_temp",
                        help="Maximum effective temperature (K).")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU for the single-GPU phases (default: first visible).")

    g1 = parser.add_argument_group("Phase 1 — MPS throughput")
    g1.add_argument("--mps-only", "--mps-throughput-only", action="store_true", dest="mps_only",
                    help="Only measure single-GPU MPS throughput and exit.")
    g1.add_argument("--mps-range", type=str, default="1,2,3,4", dest="mps_range",
                    help="Comma-separated replicas-per-GPU values to time.")
    g1.add_argument("--mps-steps", type=int, default=10000, dest="mps_steps",
                    help="MD steps per throughput run.")

    g2 = parser.add_argument_group("Phase 2 — exchange sweep")
    g2.add_argument("--exchange-only", action="store_true", dest="exchange_only",
                    help="Only run Phase 2 static exchange sweep (skip Phase 1 MPS sweep).")
    g2.add_argument("--temp-deltas", type=str, default="4,7,10", dest="temp_deltas",
                    help="Comma-separated neighbour ΔT to test (K).")
    g2.add_argument("--probe-replicas", type=int, default=4, dest="probe_replicas",
                    help="Replicas per exchange probe (all on one GPU).")
    g2.add_argument("--cycles", type=int, default=100,
                    help="HREMD cycles per exchange probe.")
    g2.add_argument("--warmup-steps", type=int, default=5000, dest="warmup_steps",
                    help="Equilibration steps before exchanges start.")

    gc = parser.add_argument_group("Common")
    gc.add_argument("--steps-per-cycle", type=int, default=1000, dest="steps_per_cycle",
                    help="MD steps per exchange cycle.")
    gc.add_argument("--timestep", type=int, default=2, help="Timestep (fs).")
    gc.add_argument("--lambda_selection", type=str, default="protein",
                    help="MDAnalysis selection for REST2 scaling.")
    gc.add_argument("--mpi-command", type=str, default=None, dest="mpi_command",
                    help="MPI launcher (auto-detected if omitted).")
    gc.add_argument("--timeout", type=float, default=1800, help="Per-run timeout (s).")
    gc.add_argument("--save-results", type=str, default="sweep_results.json",
                    dest="save_results", help="Write detailed results to this JSON file.")
    gc.add_argument("--keep-scratch", action="store_true", dest="keep_scratch",
                    help="Keep probe logs in .chacra_sweep_scratch/ for debugging.")

    # Internal worker flags
    parser.add_argument("--_is-throughput-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_benchmark-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_is-probe-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_probe-temps", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_results-file", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_output-dir", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_gpu-id", type=str, default=None, help=argparse.SUPPRESS)

    opts = parser.parse_args()

    if opts._is_throughput_worker:
        _throughput_worker(opts)
        return
    if opts._is_probe_worker:
        _probe_worker(opts)
        return

    if opts.mps_only and opts.exchange_only:
        parser.error("cannot specify both --mps-only and --exchange-only")

    gpus = _get_available_gpus(opts)
    print("=" * 72)
    print("  ChACRA HREMD Sweep Benchmark")
    print("=" * 72)
    print(f"  System      : {opts.system_file}")
    print(f"  Structure   : {opts.structure_file}")
    if opts.mps_only:
        print("  Mode        : Phase 1 MPS throughput only")
    elif opts.exchange_only:
        print("  Mode        : Phase 2 exchange sweep only")
        print(f"  Temp range  : {opts.min_temp:.0f} – {opts.max_temp:.0f} K")
    else:
        print(f"  Temp range  : {opts.min_temp:.0f} – {opts.max_temp:.0f} K")

    _SCRATCH.mkdir(exist_ok=True)

    # MPS lets several ranks share one GPU concurrently.
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
    results: dict = {"min_temp": opts.min_temp, "max_temp": opts.max_temp}
    try:
        mps_runs = []
        probes = []

        if opts.mps_only:
            mps_runs = _mps_sweep(opts, gpus)
            results["mps_scaling"] = mps_runs
            _print_summary(opts, mps_runs, probes)
        elif opts.exchange_only:
            probes = _exchange_sweep(opts, gpus)
            results.update(exchange_probes=probes)
            _print_summary(opts, mps_runs, probes)
        else:
            # Phase 1 uses all available GPUs, then Phase 2 uses all available GPUs
            mps_runs = _mps_sweep(opts, gpus)
            probes = _exchange_sweep(opts, gpus)
            
            results.update(mps_scaling=mps_runs, exchange_probes=probes)
            _print_summary(opts, mps_runs, probes)

        elapsed = time.time() - t_start
        results["elapsed_sec"] = elapsed
        print(f"  Total sweep time: {elapsed / 60:.1f} min")

        if opts.save_results:
            with open(opts.save_results, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Detailed results: {opts.save_results}\n")
    finally:
        if mps_started and fmpi is not None:
            fmpi.stop_mps()
        if not opts.keep_scratch:
            shutil.rmtree(_SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
