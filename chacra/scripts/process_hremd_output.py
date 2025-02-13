import os
from ChACRA.chacra.trajectories.process_hremd import *
from ChACRA.chacra.ContactFrequencies import make_contact_frequency_dictionary
import argparse
import os
import pandas as pd
import re
import subprocess



parser = argparse.ArgumentParser(description='Process the HREMD output.')
parser.add_argument('--run', type=int, required=True,
                    help='The run ID to process.')
parser.add_argument('--n_jobs', type=int, default=-1,
                    help='The number of jobs to use for parallel calculations.')
parser.add_argument('--structure_file', type=str, required=True,
                    help='The path to the topology file.')
parser.add_argument('--save_interval', type=int, default=10,
                    help='Save trajectory data at this cycle interval.')
parser.add_argument('--output_selection', type=str, default='protein',
                    help='MDAnalysis selection of atoms to write for processing state trajectories.')

args = parser.parse_args()
run = args.run
structure_file = args.structure_file

# save just the protein for reference structure calculations
structure_name = re.split(r'\/|\.',structure_file)[-2]
if f"{structure_name}_protein.pdb" not in os.listdir("./structures"):
    protein = mda.Universe(structure_file).select_atoms('protein')
    protein.write(f"./structures/{structure_name}_protein.pdb")

os.makedirs(f'./state_trajectories/run_{run}', exist_ok=True)
os.makedirs(f'./analysis_output/run_{run}', exist_ok=True)
os.makedirs(f'./contact_output/run_{run}/contacts', exist_ok=True)
os.makedirs(f'./contact_output/run_{run}/freqs', exist_ok=True)

hremd_data = f'./replica_trajectories/run_{run}/samples.arrow'
df = load_femto_data(hremd_data)

n_states = get_num_states(df)

# write out a pdb with the same selection
u = mda.Universe(structure_file)
sel = u.select_atoms(args.output_selection)

selection_file = f"./structures/state_trajectory_selection.pdb"

sel.write(selection_file)

write_state_trajectories(structure_file,
                        f'./replica_trajectories/run_{run}/trajectories',
                        hremd_data,
                        args.save_interval,
                        f'./state_trajectories/run_{run}',
                        selection=args.output_selection,
                        ref=selection_file
)



exchange_probs = get_exchange_probabilities(df)
np.save(f"./analysis_output/run_{run}/exchange_probabilities", exchange_probs)
with open(f"./analysis_output/run_{run}/exchange_probabilities.txt", 'w') as f:
    for i, prob in enumerate(exchange_probs):
        f.write(f"{i}\n\t{prob}\n")

# run getcontacts
for i in range(n_states):
    command = [
    "bash", "get_state_contacts.sh",
    selection_file,
    f'./state_trajectories/run_{run}/state_{i}.xtc',
    f'./contact_output/run_{run}',
    str(i),
    str(args.n_jobs), 
        ]
    subprocess.run(command)

# Go through previous runs' contact frequency files and generate a 
# dataframe of the current frequencies for all the combined runs.
if run > 1:
    cdfs = {} # contact dataframes
    frame_counts = {} # the number of trajectory frames each contact frequency dataframe represents
    
    for i in range(1,run+1):
        contact_files = [f'./contact_output/run_{i}/freqs/{file}' for file in sorted(
                            os.listdir(f'./contact_output/run_{i}/freqs'),
                            key=lambda x: int(re.split(r'_|\.',x)[-2]))
                              if file.endswith('.tsv')]
        frame_counts[i] = freq_frames(contact_files[0])

        cdfs[i] = pd.DataFrame(make_contact_frequency_dictionary(contact_files))

    total_frames = sum(frame_counts.values())
    adjusted_cdfs = {}
    for i, df in cdfs.items():
        adjusted_cdfs[i] = cdfs[i]*(frame_counts[i]/total_frames)

    combined = pd.concat([cdf for cdf in adjusted_cdfs.values()], axis=0).fillna(0)

    # Sum the DataFrames row-wise, for shared columns only
    result = combined.groupby(combined.index).sum().reset_index(drop=True)
    result.to_pickle(f"./analysis_output/run_{run}/total_contacts.pd")

