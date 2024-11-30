import subprocess
import os
import re
import argparse

parser = argparse.ArgumentParser(description='Run getcontacts on state trajectories.')
parser.add_argument('--run', type=int, required=True,
                    help='The run ID to process.')
parser.add_argument('--n_jobs', type=int, default=-1,
                    help='The number of jobs to use for parallel calculations.')
parser.add_argument('--structure_file', type=str, required=True,
                    help='The path to the topology file.')


args = parser.parse_args()
run = args.run
topology = args.structure_file
traj_dir = f"./state_trajectories/run_{run}"
contacts_out = f' ./contact_output/run_{run}/contacts' 
freqs_out = f' ./contact_output/run_{run}/freqs'                        
sele1 = 'protein' 
sele2 = 'protein'
itypes = 'all'
jobs = args.n_jobs


if jobs == -1:
    jobs = int(os.cpu_count()/2)

trajectories = os.listdir(traj_dir)
trajectories.sort(key=lambda x: int(re.split(r'\.|_', x)[1]))


# Loop through replicas and execute commands
for i, traj in enumerate(trajectories):  # Includes 0 to n_reps
    # Run get_dynamic_contacts.py
    
    subprocess.run([
        "conda", "run", "-n", "getcontacts",
        "get_dynamic_contacts.py",
        "--topology", topology,
        "--trajectory", f"{traj_dir}/{traj}",
        "--output", f"{contacts_out}/cont_state_{i}.tsv",
        "--cores", str(jobs),
        "--itypes", itypes,
        "--distout",
        "--sele", sele1,
        "--sele2", sele2
        ], 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


    #print("Standard Output:", result.stderr)
    
    # Run get_contact_frequencies.py
    subprocess.run([
        "get_contact_frequencies.py",
        "--input_files", f"{contacts_out}/cont_state_{i}.tsv",
        "--output_file", f"{freqs_out}/freqs_state_{i}.tsv"
    ])