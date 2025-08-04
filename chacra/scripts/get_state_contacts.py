import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: get-state-contacts <structure> <trajectory> <contacts_folder> "
            "<state> <n_jobs>"
        )
        sys.exit(1)

    structure, trajectory, contacts_folder, state, n_jobs = sys.argv[1:]

    contacts_folder = str(Path(contacts_folder))
    contacts_folder.mkdir(parents=True, exist_ok=True)
    (contacts_folder / "contacts").mkdir(exist_ok=True)
    (contacts_folder / "freqs").mkdir(exist_ok=True)

    contacts_out = contacts_folder / f"contacts/cont_state_{state}.tsv"
    freqs_out = contacts_folder / f"freqs/freqs_state_{state}.tsv"

    subprocess.run(
        [
            "get-dynamic-contacts",
            "--topology",
            structure,
            "--trajectory",
            trajectory,
            "--output",
            contacts_out,
            "--cores",
            str(n_jobs),
            "--itypes",
            "all",
            "--distout",
            "--sele",
            "protein",
            "--sele2",
            "protein",
        ],
        check=True,
    )

    subprocess.run(
        [
            "get-contact-frequencies",
            "--input_files",
            contacts_out,
            "--output_file",
            freqs_out,
        ],
        check=True,
    )
