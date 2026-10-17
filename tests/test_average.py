import os

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest

from chacra.average import (
    average_multimer,
    chain_to_seg,
    everything_from_averaged,
    find_identical_subunits,
    get_equivalent_interactions,
    has_only_identical_subunits,
    trim_multimer_termini,
)


def _make_pdb_line(idx, chain, resid, resname, x, y, z):
    return (
        f"ATOM  {idx:5d}  CA  {resname:3s} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
    )


@pytest.fixture
def dimer_pdb(tmp_path):
    """
    Homodimer with C2 symmetry along the y-axis at z = 3.8.
    Residue 2 is at the twofold symmetry axis (contacts itself).
    Residues 1 and 3 are symmetrically paired.
    """
    pdb_path = tmp_path / "dimer.pdb"
    lines = [
        _make_pdb_line(1, "A", 1, "ALA", 0.0, 0.0, 0.0),
        _make_pdb_line(2, "A", 2, "ALA", 0.0, 1.0, 3.8),
        _make_pdb_line(3, "A", 3, "ALA", 0.0, 0.0, 7.6),
        _make_pdb_line(4, "B", 1, "ALA", 5.0, 0.0, 7.6),
        _make_pdb_line(5, "B", 2, "ALA", 5.0, -1.0, 3.8),
        _make_pdb_line(6, "B", 3, "ALA", 5.0, 0.0, 0.0),
        "END",
    ]
    pdb_path.write_text("\n".join(lines))
    return str(pdb_path)


@pytest.fixture
def linear_tetramer_pdb(tmp_path):
    """
    Linear homotetramer A-B-C-D:
    A and D are terminal subunits (env 1).
    B and C are internal subunits (env 2).
    """
    pdb_path = tmp_path / "linear_tetramer.pdb"
    lines = []
    atom_idx = 1
    for ch_idx, ch in enumerate(["A", "B", "C", "D"]):
        x_base = ch_idx * 15.0
        for res in [1, 2, 3]:
            lines.append(
                _make_pdb_line(atom_idx, ch, res, "ALA", x_base, 0.0, (res - 1) * 3.8)
            )
            atom_idx += 1
    lines.append("END")
    pdb_path.write_text("\n".join(lines))
    return str(pdb_path)


class TestFindIdenticalSubunits:
    def test_homodimer(self, dimer_pdb):
        u = mda.Universe(dimer_pdb)
        assert has_only_identical_subunits(u) is True
        subunits = find_identical_subunits(u)
        assert len(subunits) == 1
        assert sorted(subunits[0]) == [0, 1]

    def test_heteromultimer_with_unique_subunit(self, tmp_path):
        pdb_path = tmp_path / "a2b.pdb"
        lines = [
            _make_pdb_line(1, "A", 1, "ALA", 0.0, 0.0, 0.0),
            _make_pdb_line(2, "A", 2, "ALA", 0.0, 0.0, 3.8),
            _make_pdb_line(3, "B", 1, "ALA", 5.0, 0.0, 0.0),
            _make_pdb_line(4, "B", 2, "ALA", 5.0, 0.0, 3.8),
            _make_pdb_line(5, "C", 1, "GLY", 10.0, 0.0, 0.0),
            _make_pdb_line(6, "C", 2, "GLY", 10.0, 0.0, 3.8),
            "END",
        ]
        pdb_path.write_text("\n".join(lines))
        u = mda.Universe(str(pdb_path))
        assert has_only_identical_subunits(u) is False

        subunits = find_identical_subunits(u)
        assert len(subunits) == 2
        # One group has 2 segments (A, B) and one group has 1 segment (C)
        sizes = sorted([len(v) for v in subunits.values()])
        assert sizes == [1, 2]


class TestTrimMultimerTermini:
    def test_ragged_termini_trimming(self, tmp_path):
        # Chain A has residues 1..5: ALA, GLY, VAL, LEU, PRO
        # Chain B has residues 2..4: GLY, VAL, LEU (missing 1 and 5)
        pdb_path = tmp_path / "ragged.pdb"
        lines = [
            _make_pdb_line(1, "A", 1, "ALA", 0.0, 0.0, 0.0),
            _make_pdb_line(2, "A", 2, "GLY", 0.0, 0.0, 3.8),
            _make_pdb_line(3, "A", 3, "VAL", 0.0, 0.0, 7.6),
            _make_pdb_line(4, "A", 4, "LEU", 0.0, 0.0, 11.4),
            _make_pdb_line(5, "A", 5, "PRO", 0.0, 0.0, 15.2),
            _make_pdb_line(6, "B", 2, "GLY", 5.0, 0.0, 3.8),
            _make_pdb_line(7, "B", 3, "VAL", 5.0, 0.0, 7.6),
            _make_pdb_line(8, "B", 4, "LEU", 5.0, 0.0, 11.4),
            "END",
        ]
        pdb_path.write_text("\n".join(lines))

        df = pd.DataFrame(
            {
                "A:ALA:1-A:GLY:2": [0.5],
                "A:GLY:2-A:VAL:3": [0.8],
                "A:LEU:4-A:PRO:5": [0.9],
                "B:GLY:2-B:VAL:3": [0.8],
                "A:ALA:1-B:GLY:2": [0.4],
                "A:GLY:2-B:GLY:2": [0.7],
            }
        )

        out_pdb = tmp_path / "trimmed.pdb"
        trimmed_u, trimmed_df = trim_multimer_termini(
            str(pdb_path), df, output_structure=str(out_pdb)
        )

        # Both chains in trimmed universe should now have exactly 3 residues:
        # GLY, VAL, LEU
        assert len(trimmed_u.segments[0].residues) == 3
        assert len(trimmed_u.segments[1].residues) == 3
        assert has_only_identical_subunits(trimmed_u) is True

        # Columns involving Chain A residue 1 or 5 should be removed
        assert "A:ALA:1-A:GLY:2" not in trimmed_df.columns
        assert "A:LEU:4-A:PRO:5" not in trimmed_df.columns
        assert "A:ALA:1-B:GLY:2" not in trimmed_df.columns

        # Valid columns should be retained
        assert "A:GLY:2-A:VAL:3" in trimmed_df.columns
        assert "B:GLY:2-B:VAL:3" in trimmed_df.columns
        assert "A:GLY:2-B:GLY:2" in trimmed_df.columns
        assert os.path.exists(str(out_pdb))


class TestAverageMultimerDimer:
    def test_self_symmetric_contact_frequency(self, dimer_pdb):
        """
        Verify that self-symmetric contacts across a 2-fold axis are NOT halved.
        """
        df = pd.DataFrame(
            {
                "A:ALA:1-B:ALA:3": [1.0],
                "A:ALA:3-B:ALA:1": [1.0],
                "A:ALA:2-B:ALA:2": [1.0],
                "A:ALA:1-A:ALA:2": [1.0],
                "B:ALA:1-B:ALA:2": [1.0],
            }
        )

        avg_df, std_df = average_multimer(dimer_pdb, df, return_stdev=True)

        # Self-symmetric contact A:ALA:2-B:ALA:2 must be 1.0, not 0.5!
        assert "A:ALA:2-B:ALA:2" in avg_df.columns
        assert np.isclose(avg_df["A:ALA:2-B:ALA:2"].iloc[0], 1.0)
        assert np.isclose(std_df["A:ALA:2-B:ALA:2"].iloc[0], 0.0)

        # Non-self-symmetric interface contacts A1-B3 & A3-B1 should average to 1.0
        assert "A:ALA:1-B:ALA:3" in avg_df.columns
        assert np.isclose(avg_df["A:ALA:1-B:ALA:3"].iloc[0], 1.0)

        # Intra-subunit contacts A1-A2 & B1-B2 should average to 1.0
        assert "A:ALA:1-A:ALA:2" in avg_df.columns
        assert np.isclose(avg_df["A:ALA:1-A:ALA:2"].iloc[0], 1.0)


class TestCyclicTrimer:
    def test_1tnf_averaging_and_as_map(self):
        pdb_path = "chacra/examples/1tnf_truncated.pdb"
        if not os.path.exists(pdb_path):
            pytest.skip("1tnf_truncated.pdb not found.")

        # Test with dummy contacts from 1tnf (alphabetized as getcontacts produces them)
        df = pd.DataFrame(
            {
                "A:PRO:2-A:VAL:1": [1.0],
                "B:PRO:2-B:VAL:1": [1.0],
                "C:PRO:2-C:VAL:1": [1.0],
                "A:LEU:57-B:TYR:59": [0.9],
                "B:LEU:57-C:TYR:59": [0.9],
                "A:TYR:59-C:LEU:57": [0.9],
            }
        )

        avg_df = average_multimer(pdb_path, df)
        assert "A:PRO:2-A:VAL:1" in avg_df.columns
        assert "A:LEU:57-B:TYR:59" in avg_df.columns
        assert np.isclose(avg_df["A:PRO:2-A:VAL:1"].iloc[0], 1.0)
        assert np.isclose(avg_df["A:LEU:57-B:TYR:59"].iloc[0], 0.9)

        # Test everything_from_averaged with as_map=True
        u = mda.Universe(pdb_path)
        mapped = everything_from_averaged(avg_df, u, as_map=True)
        assert isinstance(mapped, dict)
        assert "A:LEU:57-B:TYR:59" in mapped
        assert set(mapped["A:LEU:57-B:TYR:59"]) == {
            "A:LEU:57-B:TYR:59",
            "B:LEU:57-C:TYR:59",
            "A:TYR:59-C:LEU:57",
        }

        # Test everything_from_averaged with as_map=False
        replicated_df = everything_from_averaged(avg_df, u, as_map=False)
        assert isinstance(replicated_df, pd.DataFrame)
        assert "B:LEU:57-C:TYR:59" in replicated_df.columns
        assert np.isclose(replicated_df["B:LEU:57-C:TYR:59"].iloc[0], 0.9)


class TestDistinctOrientationEnvironments:
    def test_linear_tetramer_separation(self, linear_tetramer_pdb):
        """
        Verify that in a linear tetramer, terminal pairs and internal pairs
        are not improperly merged into a single false 4-subunit equivalence orbit.
        """
        u = mda.Universe(linear_tetramer_pdb)
        subunits = find_identical_subunits(u)
        chain_seg = chain_to_seg(u)

        eq = get_equivalent_interactions(
            u, subunits, chain_seg, representative_chains=["A"]
        )

        # Interface between adjacent chains: (A, B), (B, C), (C, D)
        # Should be recognized without forcing D to pair with an unphysical neighbor
        assert ("A", "B") in eq
        adj_pairs = eq[("A", "B")]
        assert ("A", "B") in adj_pairs
        assert ("B", "C") in adj_pairs
        assert ("C", "D") in adj_pairs
        # Subunit D has no equivalent partner to its right, so adj_pairs
        # must have length 3, not 4!
        assert len(adj_pairs) == 3
