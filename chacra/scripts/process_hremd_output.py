"""
Process HREMD output: separate state trajectories, run contacts,
compute frequencies, and produce ChACRA analysis.

Pipeline stages
───────────────
1. State trajectories    → state_trajectories/run_N/state_*.xtc
2. Exchange probabilities → analysis_output/run_N/exchange_probabilities.npy
3. Contact calculations  → contact_output/run_N/contacts/cont_state_*.{parquet,tsv}
4. Frequency calculation → contact_output/run_N/freqs/freqs_state_*.*
5. ChACRA analysis       → analysis_output/run_N/ (plots, .pml, total_contacts)

Stages 1–4 are skipped when their outputs already exist.  Once a
stage is identified as incomplete every downstream stage re-runs
even if its own outputs look complete (cascade rule).  Stage 5
always re-runs to guarantee fresh results.

Use ``--force`` to ignore all skip checks and rerun everything.

Temps and replica count are read from ``chacra_run.json`` when available
so that repeated restarts do not require re-supplying ``--min_temp``,
``--max_temp``, and ``--n_systems``.
"""

import argparse
import gc
import os
import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from chacra.ContactFrequencies import make_contact_dataframe, ContactFrequencies
from chacra.trajectories.process_hremd import (
    load_femto_data,
    get_num_states,
    ReplicaHandler,
    get_exchange_probabilities,
    get_state_energies,
)
import GPUtil
from chacra.plot import plot_chacras, plot_difference_of_roots, plot_explained_variance
from chacra.convergence import (
    convergence_report,
    print_convergence_report,
    save_convergence_report,
    plot_convergence_history,
    plot_exchange_diagnostics,
)
from chacra.visualize.pymol import to_pymol
from chacra.run_config import RunConfig

import MDAnalysis as mda


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _update_latest_symlink(analysis_dir: str, run: int) -> None:
    """
    Create / update ``analysis_output/latest`` → ``analysis_output/run_N``.
    Uses a relative symlink so the project directory is portable.
    """
    latest = Path(analysis_dir) / "latest"
    target = Path(f"run_{run}")  # relative
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target)


def _sorted_contact_files(freq_dir: str) -> list[str]:
    """Return contact files (.parquet or .tsv) sorted by state index."""
    all_files = os.listdir(freq_dir)
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    target_files = parquet_files if parquet_files else [f for f in all_files if f.endswith(".tsv")]

    return [
        f"{freq_dir}/{f}"
        for f in sorted(
            target_files,
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
        )
    ]


def _contact_exists(run: int, state_idx: int) -> bool:
    """Check if a per-state contact file exists (parquet or tsv)."""
    return (
        os.path.exists(f"contact_output/run_{run}/contacts/cont_state_{state_idx}.parquet")
        or os.path.exists(f"contact_output/run_{run}/contacts/cont_state_{state_idx}.tsv")
    )


def _freq_exists(run: int, state_idx: int) -> bool:
    """Check if a per-state frequency file exists (parquet or tsv)."""
    return (
        os.path.exists(f"contact_output/run_{run}/freqs/freqs_state_{state_idx}_condensed.parquet")
        or os.path.exists(f"contact_output/run_{run}/freqs/freqs_state_{state_idx}.tsv")
    )


def _accumulate_contacts(
    run: int,
    current_run_df: pd.DataFrame,
    selection_file: str,
) -> pd.DataFrame:
    """
    Merge *current_run_df* with the accumulated contact data from all prior runs.
    Uses MDAnalysis trajectory length to determine weights.
    """
    if run == 1:
        return current_run_df

    prior_parquet = Path(f"./analysis_output/run_{run - 1}/total_contacts.parquet")
    if not prior_parquet.exists():
        print(
            f"  [process-output] No prior parquet found at {prior_parquet}; "
            "falling back to unweighted direct merge."
        )
        return current_run_df

    prior_df = pd.read_parquet(prior_parquet)

    # Recover frame counts via MDAnalysis (only needs 1 IO call per run)
    current_xtc = f"./state_trajectories/run_{run}/state_0.xtc"
    current_frames = len(mda.Universe(selection_file, current_xtc).trajectory) if os.path.exists(current_xtc) else 1

    prior_total_frames = 0
    for i in range(1, run):
        prior_xtc = f"./state_trajectories/run_{i}/state_0.xtc"
        if os.path.exists(prior_xtc):
            prior_total_frames += len(mda.Universe(selection_file, prior_xtc).trajectory)

    total_frames = prior_total_frames + current_frames

    # Align columns — union with zero-fill
    all_cols = prior_df.columns.union(current_run_df.columns)
    prior_df = prior_df.reindex(columns=all_cols, fill_value=0.0)
    current_run_df = current_run_df.reindex(columns=all_cols, fill_value=0.0)

    # Re-weight
    if prior_total_frames > 0 and total_frames > 0:
        result = (
            prior_df * (prior_total_frames / total_frames)
            + current_run_df * (current_frames / total_frames)
        )
    else:
        result = current_run_df
    return result


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process the HREMD output.  Replica trajectories are separated into "
            "individual thermodynamic-state trajectories (state_trajectories/run_N/), "
            "contacts are calculated (contact_output/run_N/), and cumulative analysis "
            "outputs are written to analysis_output/run_N/.  Stages with existing "
            "outputs are skipped automatically; use --force to rerun everything."
        )
    )
    parser.add_argument(
        "--run", type=int, required=True, help="The run ID to process."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force all stages to re-run, ignoring existing outputs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to chacra_run.json.  If present, temps/n_systems are read "
             "from this file and CLI flags below become optional.",
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
        default=None,
        help="Path to the topology / structure file.",
    )
    parser.add_argument(
        "--system_file",
        type=str,
        default=None,
        help="Path to the OpenMM system XML (used for H-bond topology in ultracontacts).",
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
        help="MDAnalysis atom selection for writing state trajectories.",
    )
    # Legacy / override args — not required when chacra_run.json is present
    parser.add_argument(
        "--min_temp",
        type=float,
        default=None,
        help="Minimum effective temperature (K).  Read from config if omitted.",
    )
    parser.add_argument(
        "--max_temp",
        type=float,
        default=None,
        help="Maximum effective temperature (K).  Read from config if omitted.",
    )
    parser.add_argument(
        "--n_systems",
        type=int,
        default=None,
        help="Number of replicas.  Read from config if omitted.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------- #
    # Resolve configuration                                                   #
    # ---------------------------------------------------------------------- #
    config_path = args.config or (
        "chacra_run.json" if os.path.exists("chacra_run.json") else None
    )
    run_config = RunConfig(config_path)
    run_config.apply_to_namespace(args)

    run = args.run
    structure_file = args.structure_file

    if structure_file is None:
        parser.error(
            "--structure_file is required (or set 'structure_file' in chacra_run.json)."
        )

    # Build the temperature array from the stored list or re-derive it
    if run_config.get("temps") is not None:
        temps = np.array(run_config.get("temps"))
    elif args.min_temp is not None and args.max_temp is not None and args.n_systems is not None:
        temps = np.geomspace(args.min_temp, args.max_temp, args.n_systems)
    else:
        parser.error(
            "Temperature information is missing.  Either provide --config pointing "
            "to chacra_run.json, or supply --min_temp, --max_temp, and --n_systems."
        )

    n_states = len(temps)

    # ---------------------------------------------------------------------- #
    # Directory setup                                                         #
    # ---------------------------------------------------------------------- #
    os.makedirs(f"./state_trajectories/run_{run}", exist_ok=True)
    os.makedirs(f"./analysis_output/run_{run}", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/contacts", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/freqs", exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Protein-only reference structure                                        #
    # ---------------------------------------------------------------------- #
    structure_name = re.split(r"\/|\.", structure_file)[-2]
    selection_file = f"./structures/{structure_name}_protein.pdb"
    if not os.path.exists(selection_file):
        u = mda.Universe(structure_file)
        u.select_atoms(args.output_selection).write(selection_file)

    # ---------------------------------------------------------------------- #
    # Determine what needs to run — cascade rule                              #
    # ---------------------------------------------------------------------- #
    # Once a stage is identified as incomplete, all downstream stages must
    # re-run even if their outputs already exist.
    cascade = args.force

    # Stage 1 check: all state trajectory xtc files exist?
    stage1_complete = all(
        os.path.exists(f"./state_trajectories/run_{run}/state_{i}.xtc")
        for i in range(n_states)
    )
    run_stage1 = cascade or not stage1_complete
    if run_stage1:
        cascade = True

    # Stage 2 check: exchange probabilities exist?
    stage2_complete = os.path.exists(
        f"./analysis_output/run_{run}/exchange_probabilities.npy"
    )
    run_stage2 = cascade or not stage2_complete
    if run_stage2:
        cascade = True

    # Stage 3 check: per-state contact files
    missing_contacts = [
        i for i in range(n_states)
        if not _contact_exists(run, i)
    ]
    stage3_complete = len(missing_contacts) == 0
    run_stage3 = cascade or not stage3_complete
    if run_stage3 and not cascade:
        # Stage 3 is the first incomplete stage — cascade starts here
        cascade = True

    # Stage 4 check: per-state frequency files
    missing_freqs = [
        i for i in range(n_states)
        if not _freq_exists(run, i)
    ]
    stage4_complete = len(missing_freqs) == 0
    run_stage4 = cascade or not stage4_complete
    if run_stage4 and not cascade:
        cascade = True

    # Stage 5: always re-run
    run_stage5 = True

    # ---------------------------------------------------------------------- #
    # Stage 1: State trajectory assembly                                      #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 1/5: State trajectories")
    if run_stage1:
        hremd_data = f"./replica_trajectories/run_{run}/samples.arrow"
        if not os.path.exists(hremd_data):
            sys.exit(f"  [ERROR] {hremd_data} not found. Cannot generate state trajectories.")

        df = load_femto_data(hremd_data)

        # Energy plots (cheap, uses femto data)
        from chacra.plot import plot_energies
        plot_energies(
            get_state_energies(df),
            filename=f"./analysis_output/run_{run}/state_energies.png",
            n_bins=50,
        )

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
        print(f"  [DONE] Wrote {n_states} state trajectories.")
    else:
        df = None  # not loaded — not needed
        print(f"  [SKIP] All {n_states} state trajectories already exist.")

    # ---------------------------------------------------------------------- #
    # Stage 2: Exchange probabilities                                         #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 2/5: Exchange probabilities")
    if run_stage2:
        # Load femto data if stage 1 was skipped
        if df is None:
            hremd_data = f"./replica_trajectories/run_{run}/samples.arrow"
            if not os.path.exists(hremd_data):
                sys.exit(f"  [ERROR] {hremd_data} not found. Cannot compute exchange probabilities.")
            df = load_femto_data(hremd_data)

        exchange_probs = get_exchange_probabilities(df)
        np.save(f"./analysis_output/run_{run}/exchange_probabilities", exchange_probs)
        with open(f"./analysis_output/run_{run}/exchange_probabilities.txt", "w") as f:
            for i, prob in enumerate(exchange_probs):
                f.write(f"{i}\n\t{prob:.4f}\n")
        print(f"  [DONE] Exchange probabilities saved.")
    else:
        print(f"  [SKIP] Exchange probabilities already exist.")

    # Free femto data — no longer needed after stage 2
    if df is not None:
        del df
    if "replica_handler" in dir():
        del replica_handler
    gc.collect()

    # ---------------------------------------------------------------------- #
    # Stage 3: Contact calculations (per-state)                              #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 3/5: Contact calculations")

    gpus = GPUtil.getGPUs()
    n_gpus = len(gpus)
    use_ultracontacts = n_gpus > 0
    openmm_sys = getattr(args, "system_file", None) or run_config.get("system_file")

    if run_stage3:
        # Which states need contacts?
        if cascade and stage3_complete and not args.force:
            # Cascade from upstream — but contacts are all present.
            # No upstream data changed if we didn't re-run stage 1,
            # but cascade rule says rerun anyway for safety.
            states_to_contact = list(range(n_states))
        elif args.force:
            states_to_contact = list(range(n_states))
        else:
            states_to_contact = missing_contacts

        n_skip = n_states - len(states_to_contact)
        if n_skip > 0:
            print(f"  [SKIP] {n_skip}/{n_states} states already have contacts.")
        print(f"  [RUN]  Computing contacts for {len(states_to_contact)} states.")

        if use_ultracontacts:
            print(f"  Engine: ultracontacts ({n_gpus} GPUs)")
            # Process in GPU-sized chunks
            chunks = [
                states_to_contact[i:i + n_gpus]
                for i in range(0, len(states_to_contact), n_gpus)
            ]
            for chunk in chunks:
                processes = []
                for gpu_offset, state_idx in enumerate(chunk):
                    contacts_out = f"contact_output/run_{run}/contacts/cont_state_{state_idx}.parquet"
                    cmd = [
                        "ultracontacts", "contacts",
                        "--topology", selection_file,
                        "--trajectory", f"./state_trajectories/run_{run}/state_{state_idx}.xtc",
                        "--output", contacts_out,
                        "--stride", "1",
                    ]
                    if openmm_sys and os.path.exists(openmm_sys):
                        cmd.extend(["--openmm-system", str(openmm_sys)])

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_offset)

                    proc = subprocess.Popen(
                        cmd, env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    processes.append((state_idx, proc))

                for state_idx, proc in processes:
                    _, stderr = proc.communicate()
                    if proc.returncode != 0:
                        err_msg = stderr.decode().strip() if stderr else ""
                        print(
                            f"  [WARN] ultracontacts failed for state {state_idx} "
                            f"(exit code {proc.returncode}). {err_msg}"
                        )
        else:
            print(f"  Engine: getcontacts (CPU)")
            for state_idx in states_to_contact:
                contacts_out = f"contact_output/run_{run}/contacts/cont_state_{state_idx}.tsv"
                try:
                    subprocess.run(
                        [
                            "get-dynamic-contacts",
                            "--topology", selection_file,
                            "--trajectory", f"./state_trajectories/run_{run}/state_{state_idx}.xtc",
                            "--output", str(contacts_out),
                            "--cores", str(args.n_jobs),
                            "--itypes", "all",
                            "--distout",
                            "--sele", "protein",
                            "--sele2", "protein",
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"  [WARN] getcontacts failed for state {state_idx} "
                        f"(exit code {e.returncode}). Continuing."
                    )
        print(f"  [DONE] Contact calculations complete.")
    else:
        print(f"  [SKIP] All {n_states} states already have contacts.")

    # ---------------------------------------------------------------------- #
    # Stage 4: Frequency calculation (per-state)                             #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 4/5: Frequency calculation")

    if run_stage4:
        # Which states need frequencies?
        if cascade and stage4_complete and not args.force:
            states_to_freq = list(range(n_states))
        elif args.force:
            states_to_freq = list(range(n_states))
        else:
            states_to_freq = missing_freqs

        n_skip = n_states - len(states_to_freq)
        if n_skip > 0:
            print(f"  [SKIP] {n_skip}/{n_states} states already have frequencies.")
        print(f"  [RUN]  Computing frequencies for {len(states_to_freq)} states.")

        freq_failures = []
        for state_idx in states_to_freq:
            # Determine which contact file exists for this state
            contact_parquet = f"contact_output/run_{run}/contacts/cont_state_{state_idx}.parquet"
            contact_tsv = f"contact_output/run_{run}/contacts/cont_state_{state_idx}.tsv"

            if os.path.exists(contact_parquet):
                # ultracontacts frequencies (CPU — no GPU needed)
                freq_out = f"contact_output/run_{run}/freqs/freqs_state_{state_idx}_condensed.parquet"
                try:
                    subprocess.run(
                        [
                            "ultracontacts", "frequencies",
                            "--input", contact_parquet,
                            "--output", freq_out,
                            "--condensed",
                        ],
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"  [WARN] ultracontacts frequencies failed for state {state_idx} "
                        f"(exit code {e.returncode}). Continuing."
                    )
                    freq_failures.append(state_idx)

            elif os.path.exists(contact_tsv):
                # getcontacts frequency calculation
                freq_out = f"contact_output/run_{run}/freqs/freqs_state_{state_idx}.tsv"
                try:
                    subprocess.run(
                        [
                            "get-contact-frequencies",
                            "--input_files", str(contact_tsv),
                            "--output_file", str(freq_out),
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"  [WARN] get-contact-frequencies failed for state {state_idx} "
                        f"(exit code {e.returncode}). Continuing."
                    )
                    freq_failures.append(state_idx)
            else:
                print(
                    f"  [WARN] No contact file found for state {state_idx}. "
                    f"Cannot compute frequencies."
                )
                freq_failures.append(state_idx)

        if freq_failures:
            print(f"  [WARN] Frequency calculation failed for states: {freq_failures}")
        print(f"  [DONE] Frequency calculation complete.")
    else:
        print(f"  [SKIP] All {n_states} states already have frequencies.")

    # ---------------------------------------------------------------------- #
    # Stage 5: Contact frequency aggregation + ChACRA analysis               #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 5/5: ChACRA analysis")

    # Hard-fail if any frequency files are missing
    still_missing = [i for i in range(n_states) if not _freq_exists(run, i)]
    if still_missing:
        sys.exit(
            f"  [ERROR] Cannot proceed with analysis — frequency files missing "
            f"for states: {still_missing}.\n"
            f"  Fix the upstream issue and rerun: process-output --run {run}"
        )

    current_freq_dir = f"./contact_output/run_{run}/freqs"
    current_files = _sorted_contact_files(current_freq_dir)
    current_run_df = make_contact_dataframe(current_files)

    # Save a per-run summary parquet (raw, unweighted) for archival
    current_run_df.to_parquet(
        f"./contact_output/run_{run}/freqs_summary.parquet", index=True
    )

    # Compute (or update) the cumulative weighted contact frequencies
    cdf = _accumulate_contacts(run, current_run_df, selection_file)
    del current_run_df
    gc.collect()

    # Persist the cumulative result
    cdf.to_parquet(f"./analysis_output/run_{run}/total_contacts.parquet", index=True)

    cf = ContactFrequencies(cdf, temps=np.round(temps), n_jobs=args.n_jobs)

    top_ten = {
        pc: cf.cpca.sorted_norm_loadings(pc)[f"PC{pc}"][:10].index.tolist()
        for pc in cf.cpca.top_chacras
    }
    pd.DataFrame(top_ten).to_csv(
        f"./analysis_output/run_{run}/top_chacra_contacts.csv", index=False
    )

    fig = plot_chacras(
        cf.cpca,
        n_pcs=cf.cpca.top_chacras[-1],
        contacts=cf.freqs,
        temps=temps,
        temp_scale="K",
        filename=f"./analysis_output/run_{run}/chacra_modes.png",
    )
    fig.clf()

    fig = plot_difference_of_roots(
        cf.cpca,
        n_pcs=cf.cpca.top_chacras[-1],
        filename=f"./analysis_output/run_{run}/difference_of_roots.png",
    )
    fig.clf()

    plot_explained_variance(
        cf.cpca,
        filename=f"./analysis_output/run_{run}/explained_variance.png",
    )

    to_visualize = []
    for pc in cf.cpca.top_chacras:
        to_visualize.extend(cf.cpca.get_chacra_center(pc, cutoff=0.7).index)

    to_pymol(
        to_visualize,
        cf.freqs,
        cf.cpca,
        output_file=f"./analysis_output/run_{run}/top_chacras.pml",
        pc_range=(cf.cpca.top_chacras[0], cf.cpca.top_chacras[-1]),
        variable_sphere_scale=True,
    )

    # ---------------------------------------------------------------------- #
    # Convergence diagnostics                                                 #
    # ---------------------------------------------------------------------- #
    # Load exchange probs for this run (saved in stage 2)
    exch_path = f"./analysis_output/run_{run}/exchange_probabilities.npy"
    exchange_probs = np.load(exch_path) if os.path.exists(exch_path) else None

    k_convergence = len(cf.cpca.top_chacras) if cf.cpca.top_chacras else 3
    report = convergence_report(
        run=run,
        n_states=n_states,
        exchange_probs=exchange_probs,
        k=k_convergence,
        n_jobs=args.n_jobs,
    )
    save_convergence_report(report, f"./analysis_output/run_{run}")
    print_convergence_report(report)

    # Exchange diagnostics plot
    if exchange_probs is not None:
        fig = plot_exchange_diagnostics(
            exchange_probs,
            filename=f"./analysis_output/run_{run}/exchange_diagnostics.png",
        )
        fig.clf()

    # Convergence history plot (across all runs)
    fig = plot_convergence_history(
        filename=f"./analysis_output/convergence_history.png",
    )
    if fig is not None:
        fig.clf()

    # ---------------------------------------------------------------------- #
    # Finalize                                                                #
    # ---------------------------------------------------------------------- #
    _update_latest_symlink("./analysis_output", run)

    # Persist config
    run_config.write()

    print(f"\n[process-output] Done.  See analysis_output/run_{run}/ (or analysis_output/latest/).")


if __name__ == "__main__":
    main()
