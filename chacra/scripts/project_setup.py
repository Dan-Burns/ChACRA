import os


def main():
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


if __name__ == "__main__":
    main()
