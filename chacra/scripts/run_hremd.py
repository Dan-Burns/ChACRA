import subprocess
import traceback
from ChACRA.chacra.trajectories.process_hremd import *
import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Run HREMD simulation.')

# Add arguments for system file and structure
parser.add_argument('--system_file', type=str, required=True, 
                    help='Path to the system XML file.')
parser.add_argument('--structure_file', type=str, required=True, 
                    help='Path to the structure PDB file.')
parser.add_argument('--n_cycles', type=int, default=10000,
                    help='The number of replica_exchange attempts for this run.')
parser.add_argument('--n_jobs', type=int, required=True,
                    help='The number of mpi processes to start.')
parser.add_argument('--steps_per_cycle', type=int, default=1000,
                    help='The number of timesteps between replica exchange attempts.')
parser.add_argument('--min_temp', type=float, default=290,
                    help='The minimum effective temperature (in kelvin) of the replica exchange ensemble.\
                    This is the same as the solvent temperature for all replicas')
parser.add_argument('--max_temp', type=float, default=450,
                    help='The maximum effective temperature (in kelvin) of the replica exchange ensemble.')
parser.add_argument('--n_systems', type=int, default=None,
                    help='The number of replicas.')
parser.add_argument('--save_interval', type=int, default=10,
                    help='Save trajectory data at this cycle interval.')
parser.add_argument('--checkpoint_interval', type=int, default=100,
                    help='Save checkpoints at this cycle interval.')
parser.add_argument('--warmup_steps', type=int, default=0,
                    help='The number of warmup steps to run before starting replica exchange attempts.\
                    Only necessary on the first run.')
parser.add_argument('--lambda_selection', type=str, default='protein',
                    help='The MDAnalysis selection to which the lambda scaling will be applied.')
args = parser.parse_args()

current_run = len([folder for folder in os.listdir('./replica_trajectories') 
               if folder.startswith('run') 
               and os.path.isdir(f'./replica_trajectories/{folder}')]) + 1



# check that default output fold does not exist or is empty
# if current_run is 1:
#     if os.path.exists('hremd-outputs/trajectories') or os.path.exists()
if current_run > 1:
    # Load data from the previous run
    df = load_femto_data(f'replica_trajectories/run_{current_run - 1}/samples.arrow')
    # Get the number of replicas from the data if not specified
    if args.n_systems is None:
        n_systems = get_num_states(df)
    elif args.n_systems == get_num_states(df):
        n_systems = args.n_systems
    else:
        n_systems = get_num_states(df)
        print(f"The number of replicas specified ({args.n_systems}) does not equal that of previous runs ({n_systems}).")
        print(f"Using {n_systems} replicas. Start a new hremd project if you want to run with {args.n_systems} replicas.")
else:  # current_run == 1
    if args.n_systems is None:
        print("This is the first run. You need to specify how many systems to create.")
        exit()
    else:
        n_systems = args.n_systems
    
# Get the number of cycles run thus far
cycles_completed = int((df['step'].values/args.steps_per_cycle)[-1])+1
total_cycles = cycles_completed + args.n_cycles

# Define the MPI command
mpi_command = [
    "mpirun",  
    "-np", str(args.n_jobs),  
    "python", "run_femto_hremd.py",  
    '--system_file', args.system_file,  
    '--structure_file', args.structure_file,   
    '--n_cycles', str(total_cycles),
    '--steps_per_cycle', str(args.steps_per_cycle),
    '--min_temp', str(args.min_temp),
    '--max_temp', str(args.max_temp),
    '--n_systems', str(n_systems),
    '--save_interval', str(args.save_interval),
    '--checkpoint_interval', str(args.checkpoint_interval),
    '--warmup_steps', str(args.warmup_steps),
    '--lambda_selection', args.lambda_selection
]

try:
    # Run the command
    result = subprocess.run(
        mpi_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
   
        check=True  # Raises CalledProcessError for non-zero exit codes
    )

    # Print the output
    print("Standard Output:", result.stdout)
    os.makedirs(f'./replica_trajectories/run_{current_run}')
    shutil.move('./hremd-outputs/trajectories/', f'./replica_trajectories/run_{current_run}/')
    
    shutil.move('./hremd-outputs/samples.arrow', f'./replica_trajectories/run_{current_run}/samples.arrow')
    shutil.copy('./hremd-outputs/checkpoint.pkl', f'./replica_trajectories/run_{current_run}/checkpoint.pkl')
   
    # Call Processing/ Analysis script here
    analysis_command = [
                        "python", "process_hremd_output.py",
                        "--run", str(current_run),
                        "--n_jobs", str(args.n_jobs),
                        "--structure_file", args.structure_file,
                        "--save_interval", str(args.save_interval)
    ]
    subprocess.run(analysis_command)

except subprocess.CalledProcessError as e:
    print("Error: The subprocess call failed.")
    print("Return Code:", e.returncode)
    print("Standard Output:", e.stdout)
    print("Standard Error:", e.stderr)

except Exception as e:
    print("An unexpected error occurred:")
    traceback.print_exc()