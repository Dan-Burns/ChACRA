import os
import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Set up ChACRA project directories.")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Include the example file.",
        default=False
        )
    
    args = parser.parse_args()

    dirs = [
        "analysis_output",
        "contact_output",
        "notebooks",
        "replica_trajectories",
        "state_trajectories",
        "structures",
        "system",
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    if args.example:
        EXAMPLE_PDB = Path(__file__).parent.parent / "examples" / "1tnf.pdb"
        shutil.copy(EXAMPLE_PDB, "structures/1tnf.pdb")

if __name__ == "__main__":
    main()
