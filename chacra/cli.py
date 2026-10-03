"""
Unified CLI entry point for ChACRA.

Usage::

    chacra <command> [options]

Commands::

    run-hremd           Run an HREMD simulation
    run-femto           Run femto HREMD (low-level, called by run-hremd via MPI)
    process-output      Process HREMD output (state trajectories → contacts → analysis)
    benchmark-hremd     Benchmark HREMD throughput and exchange statistics
    sweep-benchmark     Find optimal replica count and --mps-replicas adaptively
    make-simulation     Solvate a structure and create an OpenMM system
    project             Set up the ChACRA project directory
    get-state-contacts  Run contact calculations on existing state trajectories
    windowed-freqs      Compute contact frequencies from a frame window or percentile cutoffs
"""

import importlib
import sys

COMMANDS = {
    "run-hremd":          "chacra.scripts.run_hremd",
    "run-femto":          "chacra.scripts.run_femto_hremd",
    "process-output":     "chacra.scripts.process_hremd_output",
    "check-convergence":  "chacra.scripts.check_convergence",
    "benchmark-hremd":    "chacra.scripts.benchmark_hremd",
    "sweep-benchmark":    "chacra.scripts.sweep_benchmark",
    "make-simulation":    "chacra.scripts.make_simulation",
    "project":            "chacra.scripts.project_setup",
    "get-state-contacts": "chacra.scripts.get_state_contacts",
    "windowed-freqs":     "chacra.scripts.windowed_freqs",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: chacra <command> [options]\n")
        print("Available commands:")
        for cmd in COMMANDS:
            print(f"  {cmd}")
        print("\nRun 'chacra <command> --help' for command-specific options.")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"chacra: unknown command '{cmd}'\n")
        print("Available commands:")
        for c in COMMANDS:
            print(f"  {c}")
        sys.exit(1)

    # Rewrite sys.argv so the subcommand's argparse sees the right prog name
    sys.argv = [f"chacra {cmd}"] + sys.argv[2:]

    mod = importlib.import_module(COMMANDS[cmd])
    mod.main()


if __name__ == "__main__":
    main()
