"""
Benchmark HREMD simulation performance.

Runs a short HREMD simulation and reports throughput metrics and exchange
statistics to help users decide on replica counts and oversubscription levels.

Usage::

    chacra benchmark-hremd -p system.xml -s structure.pdb -n 8 -j 2
    chacra benchmark-hremd -p system.xml -s structure.pdb -n 8 -j 2 -r 2
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime

from chacra.mpi import build_mpi_command


def _run_worker(args):
    """MPI worker: runs the actual HREMD benchmark."""
    import femto.md.config
    import femto.md.constants
    import femto.md.hremd
    import femto.md.rest
    import femto.md.utils.mpi
    import femto.md.utils.openmm
    import MDAnalysis as mda
    import mdtop
    import numpy as np
    import openmm
    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    from chacra.trajectories.process_hremd import (
        get_exchange_probabilities,
        load_femto_data,
    )

    femto.md.utils.mpi.divide_gpus()

    from chacra.simulation import build_hremd_base_state
    system, structure, base_state = build_hremd_base_state(
        system_file=args.system_file,
        structure_file=args.structure_file,
        lambda_selection=args.lambda_selection,
        temperature=args.min_temp,
        timestep=args.timestep,
    )

    output_dir = pathlib.Path("benchmark-hremd-outputs")

    # Clean previous benchmark output (rank 0 only)
    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        if mpi_comm.rank == 0 and output_dir.exists():
            shutil.rmtree(output_dir)
        mpi_comm.barrier()
        time.sleep(0.3)

    # Temperature ladder
    temps = list(np.geomspace(args.min_temp, args.max_temp, args.n_systems))
    rest_temperatures = temps * openmm.unit.kelvin
    rest_betas = [
        1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * t)
        for t in rest_temperatures
    ]

    states = [
        {femto.md.rest.REST_CTX_PARAM: rest_beta / rest_betas[0]}
        for rest_beta in rest_betas
    ]
    states = [
        femto.md.utils.openmm.evaluate_ctx_parameters(state, system)
        for state in states
    ]

    # Create production simulation
    integrator_config = femto.md.config.LangevinIntegrator(
        timestep=args.timestep * openmm.unit.femtosecond,
    )
    final_integrator = femto.md.utils.openmm.create_integrator(
        integrator_config, rest_temperatures[0]
    )
    final_integrator.setRandomNumberSeed(12345)

    simulation = femto.md.utils.openmm.create_simulation(
        system,
        structure,
        coords=base_state,
        integrator=final_integrator,
        state=states[0],
        platform=femto.md.constants.OpenMMPlatform.CUDA,
    )

    hremd_config = femto.md.config.HREMD(
        n_warmup_steps=args.warmup_steps,
        n_steps_per_cycle=args.steps_per_cycle,
        n_cycles=args.n_cycles,
        trajectory_interval=args.save_interval,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Monkey-patch to capture exchange proposal time
    original_propose_swaps = femto.md.hremd._propose_swaps
    exchange_time = 0.0

    def timed_propose_swaps(*a, **kw):
        nonlocal exchange_time
        t0 = time.time()
        res = original_propose_swaps(*a, **kw)
        exchange_time += time.time() - t0
        return res

    femto.md.hremd._propose_swaps = timed_propose_swaps

    # Run benchmark
    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        mpi_comm.barrier()
        start_time = time.time()

        if mpi_comm.rank == 0:
            print("=" * 50)
            print("       STARTING HREMD BENCHMARK")
            print("=" * 50)
            print(f"Start Time         : {datetime.now().strftime('%H:%M:%S')}")
            print(f"Replicas           : {args.n_systems}")
            print(f"GPUs               : {args.n_jobs}")
            print(f"MPS replicas/GPU   : {args.mps_replicas}×")
            print(f"Total MPI ranks    : {args.n_jobs * args.mps_replicas}")
            print(f"Cycles             : {args.n_cycles}")
            print(f"Steps/Cycle        : {args.steps_per_cycle}")
            print(f"Timestep           : {args.timestep} fs")
            print(f"Total Steps/Replica: {args.n_cycles * args.steps_per_cycle + args.warmup_steps}")
            print(f"Warmup Steps       : {args.warmup_steps}")
            print(f"Temperatures (K)   : {', '.join(f'{t:.1f}' for t in temps)}")
            print("=" * 50 + "\n")

        femto.md.hremd.run_hremd(
            simulation,
            states,
            hremd_config,
            output_dir=output_dir,
            save_final_coords=False,
        )

        mpi_comm.barrier()
        end_time = time.time()

        if mpi_comm.rank == 0:
            total_time_sec = end_time - start_time
            total_time_hr = total_time_sec / 3600.0

            total_steps_per_sys = args.n_cycles * args.steps_per_cycle + args.warmup_steps
            ns_per_system = total_steps_per_sys * args.timestep / 1e6
            total_aggregate_ns = ns_per_system * args.n_systems

            ns_per_hour = ns_per_system / total_time_hr
            agg_ns_per_hour = total_aggregate_ns / total_time_hr

            # ── Exchange statistics ──
            samples_path = output_dir / "samples.arrow"
            exchange_probs = None
            if samples_path.exists():
                try:
                    exchange_probs = get_exchange_probabilities(str(samples_path))
                except Exception:
                    pass

            # ── Print results ──
            print("\n" + "=" * 50)
            print("       HREMD BENCHMARK RESULTS")
            print("=" * 50)
            print(f"End Time                     : {datetime.now().strftime('%H:%M:%S')}")
            print(f"Total Wall-clock Time        : {total_time_sec:.1f} s  ({total_time_sec/60:.1f} min)")
            print(f"Exchange Proposal Time       : {exchange_time:.3f} s  ({(exchange_time/total_time_sec)*100:.1f}%)")
            print()
            print("── Throughput ──")
            print(f"  ns per replica             : {ns_per_system:.6f} ns")
            print(f"  Aggregate ns               : {total_aggregate_ns:.6f} ns")
            print(f"  Throughput (ns/day/replica) : {ns_per_hour * 24:.2f} ns/day")
            print(f"  Aggregate (ns/day)         : {agg_ns_per_hour * 24:.2f} ns/day")
            print()

            if exchange_probs is not None:
                print("── Exchange Probabilities (adjacent pairs) ──")
                for i, prob in enumerate(exchange_probs):
                    bar = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))
                    label = f"  {i:2d} ↔ {i+1:2d}"
                    print(f"{label} : {bar} {prob:.3f}")
                mean_prob = np.mean(exchange_probs)
                min_prob = np.min(exchange_probs)
                print()
                print(f"  Mean exchange probability  : {mean_prob:.3f}")
                print(f"  Min  exchange probability  : {min_prob:.3f}")
                if min_prob < 0.1:
                    print("  ⚠  Low exchange rate detected — consider adding replicas")
                elif mean_prob > 0.4:
                    print("  💡 High exchange rates — you may be able to use fewer replicas")
            print("=" * 50)

            # Clean up benchmark outputs
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print("\nBenchmark outputs cleaned up.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HREMD simulation performance and exchange statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--system_file", type=str, required=True,
        help="Path to the system XML file.",
    )
    parser.add_argument(
        "-s", "--structure_file", type=str, required=True,
        help="Path to the structure PDB file.",
    )
    parser.add_argument(
        "-n", "--n_systems", type=int, required=True,
        help="The number of replicas.",
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, required=True,
        help="The number of GPUs to use.",
    )
    parser.add_argument(
        "-c", "--n_cycles", type=int, default=500,
        help="Number of exchange cycles to run.",
    )
    parser.add_argument(
        "-d", "--steps_per_cycle", type=int, default=1000,
        help="MD steps between exchange attempts.",
    )
    parser.add_argument(
        "-r", "--mps-replicas", type=int, default=1, dest="mps_replicas",
        help="Number of replicas to run simultaneously per GPU using CUDA MPS.",
    )
    # Keep the old name as a hidden alias for backward compatibility
    parser.add_argument(
        "-o", "--oversubscribe", type=int, default=None,
        dest="_oversubscribe_compat",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min_temp", type=float, default=290,
        help="Minimum effective temperature (K).",
    )
    parser.add_argument(
        "--max_temp", type=float, default=450,
        help="Maximum effective temperature (K).",
    )
    parser.add_argument(
        "--timestep", type=int, default=2,
        help="Timestep in femtoseconds.",
    )
    parser.add_argument(
        "--lambda_selection", type=str, default="protein",
        help="MDAnalysis selection for lambda scaling.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
        help="Warmup steps before exchanges begin.",
    )
    parser.add_argument(
        "--save_interval", type=int, default=10,
        help="Trajectory save interval (cycles).",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=100,
        help="Checkpoint interval (cycles).",
    )
    parser.add_argument(
        "--mpi-command", type=str, default=None, dest="mpi_command",
        help="Path to MPI launcher. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--is-mpi-worker", action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # Backward compat: honour the old --oversubscribe / -o flag
    if args._oversubscribe_compat is not None:
        args.mps_replicas = args._oversubscribe_compat

    if args.is_mpi_worker:
        _run_worker(args)
        return

    # ── Launcher process ──
    n_total_ranks = args.n_jobs * args.mps_replicas

    # Build MPI command that calls back into this console script
    mpi_prefix = build_mpi_command(
        n_total_ranks,
        mpi_command=args.mpi_command,
    )
    cmd = mpi_prefix + [
        "chacra", "benchmark-hremd", "--is-mpi-worker",
    ]
    # Forward all original args (except --is-mpi-worker which we just added)
    cmd.extend(sys.argv[1:])

    run_env = os.environ.copy()
    if args.mps_replicas > 1:
        import femto.md.utils.mpi as _fmpi

        thread_pct = max(1, 200 // args.mps_replicas)
        run_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_pct)
        print(
            f"CUDA MPS: {args.mps_replicas} replicas/GPU, "
            f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={thread_pct}%"
        )
        if not _fmpi.is_mps_running():
            from chacra.mpi import configure_mps_env

            configure_mps_env()
            print("Starting CUDA MPS daemon...")
            _fmpi.start_mps()

    print(f"Launching benchmark: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, env=run_env, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
