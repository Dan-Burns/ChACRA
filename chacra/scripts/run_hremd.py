import argparse
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from chacra.trajectories.process_hremd import *


def main():
    parser = argparse.ArgumentParser(description="Run HREMD simulation.")

    # Add arguments for system file and structure
    parser.add_argument(
        "-p", "--system_file",
        type=str,
        required=True,
        help="Path to the system XML file.",
    )
    parser.add_argument(
        "-s", "--structure_file",
        type=str,
        required=True,
        help="Path to the structure PDB file.",
    )
    parser.add_argument(
        "-c", "--n_cycles",
        type=int,
        default=10000,
        help="The number of replica_exchange attempts for this run.",
    )
    parser.add_argument(
        "-j", "--n_jobs",
        type=int,
        required=True,
        help="The number of mpi processes to start.",
    )
    parser.add_argument(
        "-d", "--steps_per_cycle",
        type=int,
        default=1000,
        help="The number of timesteps between replica exchange attempts.",
    )
    parser.add_argument(
        "-l", "--min_temp",
        type=float,
        default=290,
        help="The minimum effective temperature (in kelvin) of the replica exchange ensemble.\
                        This is the same as the solvent temperature for all replicas",
    )
    parser.add_argument(
        "-x", "--max_temp",
        type=float,
        default=450,
        help="The maximum effective temperature (in kelvin) of the replica exchange ensemble.",
    )
    parser.add_argument(
        "-n", "--n_systems", type=int, default=None, help="The number of replicas."
    )
    parser.add_argument(
        "-i", "--save_interval",
        type=int,
        default=10,
        help="Save trajectory data at this cycle interval.",
    )
    parser.add_argument(
        "-k", "--checkpoint_interval",
        type=int,
        default=500,
        help="Save checkpoints at this cycle interval.",
    )
    parser.add_argument(
        "-w", "--warmup_steps",
        type=int,
        default=0,
        help="The number of warmup steps to run before starting replica exchange attempts.\
                        Only necessary on the first run.",
    )
    parser.add_argument(
        "-b", "--lambda_selection",
        type=str,
        default="protein",
        help="The MDAnalysis selection to which the lambda scaling will be applied.",
    )
    parser.add_argument(
        "--output_selection",
        type=str,
        default="protein",
        help="MDAnalysis selection of atoms to write for state trajectories."
        )
    parser.add_argument(
        "--timestep", type=int, required=False,
        default=2,
        help="""
        Timestep in femtoseconds. Hydrogen mass repartitioning is recoommended 
        for timesteps larger than 2 fs.
        """,
    )
    args = parser.parse_args()

    current_run = (
        len(
            [
                folder
                for folder in os.listdir("./replica_trajectories")
                if folder.startswith("run")
                and os.path.isdir(f"./replica_trajectories/{folder}")
            ]
        )
        + 1
    )
    
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TQDM_MININTERVAL", "600")  # once every 10 minutes if enabled
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # check that default output fold does not exist or is empty
    # if current_run is 1:
    #     if os.path.exists('hremd-outputs/trajectories') or os.path.exists()
    if current_run > 1:
        # Load data from the previous run
        df = load_femto_data(
            f"replica_trajectories/run_{current_run - 1}/samples.arrow"
        )
        # Get the number of replicas from the data if not specified
        if args.n_systems is None:
            n_systems = get_num_states(df)
        elif args.n_systems == get_num_states(df):
            n_systems = args.n_systems
        else:
            n_systems = get_num_states(df)
            print(
                f"The number of replicas specified ({args.n_systems}) does not equal that of previous runs ({n_systems})."
            )
            print(
                f"Using {n_systems} replicas. Start a new hremd project if you want to run with {args.n_systems} replicas."
            )
    else:  # current_run == 1
        if args.n_systems is None:
            print(
                "This is the first run. You need to specify how many systems to create."
            )
            exit()
        else:
            n_systems = args.n_systems

    
    # Get the number of cycles run thus far
    # Get the number of cycles run thus far
    if current_run == 1:
        total_cycles = args.n_cycles
    else:
        cycles_completed = (
            int((df["step"].values / args.steps_per_cycle)[-1]) + 1
        )
        total_cycles = cycles_completed + args.n_cycles

    # Define the MPI command
    mpi_command = [
        "mpirun",
        "-np",
        str(args.n_jobs),
        "run-femto",
        "--system_file",
        args.system_file,
        "--structure_file",
        args.structure_file,
        "--n_cycles",
        str(total_cycles),
        "--steps_per_cycle",
        str(args.steps_per_cycle),
        "--min_temp",
        str(args.min_temp),
        "--max_temp",
        str(args.max_temp),
        "--n_systems",
        str(n_systems),
        "--save_interval",
        str(args.save_interval),
        "--checkpoint_interval",
        str(args.checkpoint_interval),
        "--warmup_steps",
        str(args.warmup_steps),
        "--lambda_selection",
        args.lambda_selection,
        "--timestep",
        args.timestep
    ]
    
    
    times = {}
    
    times["start"] = datetime.now().strftime("%H:%M")

    try:
        log_dir = Path(f"./analysis_output/run_{current_run}")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "hremd_stdout.log", "wb") as out, open(log_dir / "hremd_stderr.log", "wb") as err:
            result = subprocess.run(
                mpi_command,
                stdout=out,
                stderr=err,
                check=True,  # Raises CalledProcessError for non-zero exit codes
                env=os.environ.copy()
            )

        print("Replica exchange completed:", result.returncode)

        # Make the run output directories and move the output files
        os.makedirs(f"./replica_trajectories/run_{current_run}")
        shutil.move(
            "./hremd-outputs/trajectories/",
            f"./replica_trajectories/run_{current_run}/",
        )

        shutil.move(
            "./hremd-outputs/samples.arrow",
            f"./replica_trajectories/run_{current_run}/samples.arrow",
        )
        shutil.copy(
            "./hremd-outputs/checkpoint.pkl",
            f"./replica_trajectories/run_{current_run}/checkpoint.pkl",
        )
        
        times["end"] = datetime.now().strftime("%H:%M")

        # Process the replicas to state trajectories and run analyses
        analysis_command = [
            "process-output",
            "--run",
            str(current_run),
            "--n_jobs",
            str(args.n_jobs),
            "--structure_file",
            args.structure_file,
            "--save_interval",
            str(args.save_interval),
            "--output_selection",
            args.output_selection,
            "--min_temp",
            str(args.min_temp),
            "--max_temp",
            str(args.max_temp),
            "--n_systems",
            str(n_systems),
        ]
        subprocess.run(analysis_command)

        # write out run info

        with open(f"analysis_output/run_{current_run}/stats.txt", "w") as f:
            f.write(f"{times}\n")
            f.write(f"{times['start']} to {times['end']}\n")
            f.write(f"n_systems : {n_systems}\n")
            f.write(f"n_steps : {args.steps_per_cycle*args.n_cycles}\n")
            f.write(f"save_interval : {args.save_interval}\n")
            f.write(f"checkpoint_interval : {args.checkpoint_interval}\n")

    except subprocess.CalledProcessError as e:
        print("Error: The subprocess call failed.")
        print("Return Code:", e.returncode)
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)

    except Exception:
        print("An unexpected error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
