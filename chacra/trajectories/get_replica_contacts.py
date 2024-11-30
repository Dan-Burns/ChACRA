import subprocess
import os
import re

## REMOVE
# Run as script instead 
def get_replica_contacts(topology, traj_dir, contacts_out='contacts', freqs_out='freqs', 
                         sele1='protein', sele2='protein', itypes='all', jobs=-1):
    '''
    Expecting the trajectories in traj_dir to be named in the format "state_1.xtc".
    
    '''

    if jobs == -1:
        jobs = int(os.cpu_count()/2)

    trajectories = os.listdir(traj_dir)
    trajectories.sort(key=lambda x: int(re.split(r'\.|_', x)[1]))


    # Create output directories if they don't exist
    subprocess.run(["mkdir", "-p", contacts_out])
    subprocess.run(["mkdir", "-p", freqs_out])

    # Loop through replicas and execute commands
    for i, traj in enumerate(trajectories):  # Includes 0 to n_reps
        # Run get_dynamic_contacts.py
        subprocess.run([
            "get_dynamic_contacts.py",
            "--topology", topology,
            "--trajectory", f"{traj_dir}/{traj}",
            "--output", f"{contacts_out}/cont_state_{i}.tsv",
            "--cores", str(jobs),
            "--itypes", itypes,
            "--distout",
            "--sele", sele1,
            "--sele2", sele2
        ])
        
        # Run get_contact_frequencies.py
        subprocess.run([
            "get_contact_frequencies.py",
            "--input_files", f"{contacts_out}/cont_state_{i}.tsv",
            "--output_file", f"{freqs_out}/freqs_state_{i}.tsv"
        ])