"""
pytest fixtures for the ChACRA test suite.

All fixtures are designed to work without any HREMD simulation data.
Contact frequencies are synthesised from random data; TSV files are
written to temporary directories so tests are hermetically isolated.
"""

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

N_TEMPS = 20
N_CONTACTS = 40
MIN_TEMP = 290
MAX_TEMP = 450

# Realistic contact id strings:  chain:resname:resid-chain:resname:resid
# We use residue pairs separated by > 4 positions so exclude_neighbors works.
_CHAINS = ["A", "B"]
_RESNAMES = ["ALA", "GLY", "VAL", "LEU", "ILE", "PHE", "SER", "THR"]


def _make_contact_ids(n: int, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    contacts = []
    seen = set()
    while len(contacts) < n:
        chain_a = rng.choice(_CHAINS)
        chain_b = rng.choice(_CHAINS)
        resn_a = rng.choice(_RESNAMES)
        resn_b = rng.choice(_RESNAMES)
        rid_a = int(rng.integers(1, 80))
        rid_b = int(rng.integers(rid_a + 5, 120))  # always separated by >= 5
        contact = f"{chain_a}:{resn_a}:{rid_a}-{chain_b}:{resn_b}:{rid_b}"
        if contact not in seen:
            seen.add(contact)
            contacts.append(contact)
    return contacts


CONTACT_IDS = _make_contact_ids(N_CONTACTS)
TEMPS = np.geomspace(MIN_TEMP, MAX_TEMP, N_TEMPS).astype(int).tolist()


# ------------------------------------------------------------------ #
# Core data fixture                                                    #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """
    A (N_TEMPS × N_CONTACTS) contact-frequency DataFrame with:
    - geomspace temperatures as the index
    - realistic contact-id column names
    - values in [0, 1] to mimic contact probabilities
    """
    rng = np.random.default_rng(0)
    data = rng.random((N_TEMPS, N_CONTACTS))
    df = pd.DataFrame(data, index=TEMPS, columns=CONTACT_IDS)
    df.index.name = None
    return df


# ------------------------------------------------------------------ #
# TSV file fixture (mimics getcontacts output format)                 #
# ------------------------------------------------------------------ #

_TSV_HEADER = textwrap.dedent("""\
    # total_frames:{n_frames}
    # Columns: residue1 residue2 contact_frequency
""")


def _write_tsv(path: Path, contacts: dict[str, float], n_frames: int = 500) -> None:
    """Write a minimal getcontacts-style frequency TSV file."""
    lines = [_TSV_HEADER.format(n_frames=n_frames)]
    for contact, freq in contacts.items():
        a, b = contact.split("-")
        lines.append(f"{a}\t{b}\t{freq:.4f}\n")
    path.write_text("".join(lines))


@pytest.fixture(scope="session")
def synthetic_tsv_dir(tmp_path_factory) -> Path:
    """
    Write N_TEMPS minimal getcontacts TSV files to a temp directory.
    Each file has the same contacts but slightly varied frequencies
    (simulating different thermodynamic states).
    """
    base = tmp_path_factory.mktemp("tsv_files")
    rng = np.random.default_rng(1)
    for i in range(N_TEMPS):
        freqs = {c: float(rng.random()) for c in CONTACT_IDS}
        _write_tsv(base / f"freqs_state_{i}.tsv", freqs)
    return base


@pytest.fixture(scope="session")
def synthetic_parquet_dir(tmp_path_factory) -> Path:
    """
    Write N_TEMPS minimal ultracontacts condensed parquet files to a temp directory.
    Uses default naming pattern: freqs_state_{i}_condensed.parquet
    """
    base = tmp_path_factory.mktemp("parquet_files")
    rng = np.random.default_rng(2)
    for i in range(N_TEMPS):
        data = {c: [float(rng.random())] for c in CONTACT_IDS}
        df = pd.DataFrame(data)
        df.to_parquet(base / f"freqs_state_{i}_condensed.parquet")
    return base


# ------------------------------------------------------------------ #
# ContactFrequencies / ContactPCA fixture                             #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def contact_frequencies(synthetic_df):
    """
    A ContactFrequencies object built from *synthetic_df*.
    Uses n_jobs=1 (no subprocess workers) so the fixture works reliably
    inside pytest without joblib's loky backend spawning child processes.
    """
    from chacra.ContactFrequencies import ContactFrequencies
    return ContactFrequencies(synthetic_df, N_permutations=30, n_jobs=1)


@pytest.fixture(scope="session")
def contact_pca(contact_frequencies):
    """ContactPCA attribute of the session-scoped ContactFrequencies object."""
    return contact_frequencies.cpca


# ------------------------------------------------------------------ #
# RunConfig / JSON fixture                                            #
# ------------------------------------------------------------------ #


@pytest.fixture()
def tmp_config(tmp_path) -> dict:
    """
    Writes a minimal ``chacra_run.json`` to *tmp_path* and returns a dict
    with both the config data and the path for use in tests.
    """
    data = {
        "system_file": "system/test_system.xml",
        "structure_file": "structures/test.pdb",
        "n_systems": N_TEMPS,
        "min_temp": float(MIN_TEMP),
        "max_temp": float(MAX_TEMP),
        "temps": TEMPS,
        "n_cycles": 500,
        "steps_per_cycle": 1000,
        "save_interval": 10,
        "checkpoint_interval": 100,
        "warmup_steps": 0,
        "lambda_selection": "protein",
        "output_selection": "protein",
        "timestep": 2,
        "n_jobs": 4,
        "current_run": 1,
    }
    config_path = tmp_path / "chacra_run.json"
    config_path.write_text(json.dumps(data, indent=2))
    return {"path": config_path, "data": data}
