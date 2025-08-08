import argparse
import os
import re
import subprocess

import pandas as pd

from chacra.ContactFrequencies import make_contact_dataframe, ContactFrequencies
from chacra.trajectories.process_hremd import *
from chacra.plot import *

from chacra.visualize.pymol import to_pymol


def main():
    parser = argparse.ArgumentParser(
        description="Process the HREMD output. "
        "Replica trajectories will be separated into the individual "
        "thermodynamic state trajectories and saved in state_trajectories/"
        "run_x/. Contacts will be calculated and the cumulative contact "
        "dataframe will be saved in analysis_output/run_x."
    )
    parser.add_argument(
        "--run", type=int, required=True, help="The run ID to process."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of jobs to use for parallel calculations.",
    )
    parser.add_argument(
        "--structure_file",
        type=str,
        required=True,
        help="The path to the topology file.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save trajectory data at this cycle interval.",
    )
    parser.add_argument(
        "--output_selection",
        type=str,
        default="protein",
        help="MDAnalysis selection of atoms to write for "
        "processing state trajectories.",
    )
    parser.add_argument(
        "--min_temp",
        type=float,
        default=None,
        help="The minimum effective temperature (in kelvin)"
    )
    parser.add_argument(
        "--max_temp",
        type=float,
        default=None,
        help="The maximum effective temperature (in kelvin)"
    )
    parser.add_argument(
        "--n_systems",
        type=int,
        default=None,
        help="The number of replicas.",
    )

    args = parser.parse_args()
    run = args.run
    structure_file = args.structure_file
    temps = np.geomspace(args.min_temp,
                         args.max_temp,
                         args.n_systems
                         ).astype(int)

    # save just the protein for reference structure calculations
    structure_name = re.split(r"\/|\.", structure_file)[-2]
    if f"{structure_name}_protein.pdb" not in os.listdir("./structures"):
        protein = mda.Universe(structure_file).select_atoms("protein")
        protein.write(f"./structures/{structure_name}_protein.pdb")

    os.makedirs(f"./state_trajectories/run_{run}", exist_ok=True)
    os.makedirs(f"./analysis_output/run_{run}", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/contacts", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/freqs", exist_ok=True)

    hremd_data = f"./replica_trajectories/run_{run}/samples.arrow"
    df = load_femto_data(hremd_data)

    n_states = get_num_states(df)

    # write out a pdb with the same selection
    u = mda.Universe(structure_file)
    sel = u.select_atoms(args.output_selection)

    selection_file = f"./structures/{structure_name}_protein.pdb"

    sel.write(selection_file)

    replica_handler = ReplicaHandler(
        structure=structure_file,
        traj_dir=f"./replica_trajectories/run_{run}/trajectories",
        hremd_data=hremd_data,
        save_interval=args.save_interval,
    )
    replica_handler.write_state_trajectories(
        output_dir=f"./state_trajectories/run_{run}",
        selection=args.output_selection,
        ref=selection_file,
    )

    exchange_probs = get_exchange_probabilities(df)
    np.save(
        f"./analysis_output/run_{run}/exchange_probabilities", exchange_probs
    )
    with open(
        f"./analysis_output/run_{run}/exchange_probabilities.txt", "w"
    ) as f:
        for i, prob in enumerate(exchange_probs):
            f.write(f"{i}\n\t{prob}\n")

    # run getcontacts
    for i in range(n_states):
        command = [
            "get-state-contacts",
            selection_file,
            f"./state_trajectories/run_{run}/state_{i}.xtc",
            f"./contact_output/run_{run}",
            str(i),
            str(args.n_jobs),
        ]
        subprocess.run(command)

    if run == 1:
        contact_files = [
            f"./contact_output/run_{run}/freqs/{file}"
            for file in sorted(
                os.listdir(f"./contact_output/run_{run}/freqs"),
                key=lambda x: int(re.split(r"_|\.", x)[-2]),
            )
            if file.endswith(".tsv")
        ]

        cdf = make_contact_dataframe(contact_files)
        cdf.to_pickle(f"./analysis_output/run_{run}/total_contacts.pd")
        cf = ContactFrequencies(cdf, temps=np.round(temps))
    # Go through previous runs' contact frequency files and generate a
    # dataframe of the current frequencies for all the combined runs.
    if run > 1:
        cdfs = {}  # contact dataframes
        frame_counts = {}  # number of frames each dataframe represents

        for i in range(1, run + 1):
            contact_files = [
                f"./contact_output/run_{i}/freqs/{file}"
                for file in sorted(
                    os.listdir(f"./contact_output/run_{i}/freqs"),
                    key=lambda x: int(re.split(r"_|\.", x)[-2]),
                )
                if file.endswith(".tsv")
            ]
            frame_counts[i] = freq_frames(contact_files[0])

            cdfs[i] = make_contact_dataframe(contact_files)

        total_frames = sum(frame_counts.values())
        adjusted_cdfs = {}
        for i, df in cdfs.items():
            adjusted_cdfs[i] = cdfs[i] * (frame_counts[i] / total_frames)

        combined = pd.concat(
            [cdf for cdf in adjusted_cdfs.values()], axis=0
        ).fillna(0)

        # Sum the DataFrames row-wise, for shared columns only
        result = combined.groupby(combined.index).sum().reset_index(drop=True)
        result.to_pickle(f"./analysis_output/run_{run}/total_contacts.pd")

        cf = ContactFrequencies(result, temps=np.round(temps))

    top_ten = {pc: cf.cpca.sorted_norm_loadings(pc)[f'PC{pc}'][:10].index
               for pc in cf.cpca.top_chacras}
    df_top = pd.DataFrame(top_ten)
    df_top.to_csv(
        f"./analysis_output/run_{run}/top_chacra_contacts.csv", index=False
    )
    plot_chacras(cf.cpca, n_pcs=cf.cpca.top_chacras[-1],
                 contacts=cf.freqs, temps=temps,
                 temp_scale="K",
                 filename=f"./analysis_output/run_{run}/top_chacra_contacts.png")

    plot_energies(get_state_energies(df),
                  filename=f"./analysis_output/run_{run}/state_energies.png",
                  n_bins=50)
    plot_difference_of_roots(
        cf.cpca, n_pcs=cf.cpca.top_chacras[-1],
        filename=f"./analysis_output/run_{run}/difference_of_roots.png",
    )
    plot_explained_variance(
        cf.cpca,
        filename=f"./analysis_output/run_{run}/explained_variance.png",
    )

    to_visualize = []
    for pc in cf.cpca.top_chacras:
        to_visualize.extend(cf.cpca.get_chacra_center(pc, cutoff=.7).index)

    to_pymol(to_visualize,
             cf.freqs,
             cf.cpca,
             pc_range=(cf.cpca.top_chacras[0],cf.cpca.top_chacras[-1]),
             variable_sphere_scale=True)

if __name__ == "__main__":
    main()
