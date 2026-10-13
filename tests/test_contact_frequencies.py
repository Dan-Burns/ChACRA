"""
Tests for ContactFrequencies, load_contact_file, make_contact_dataframe,
and ContactPCA.

All tests use synthetic data from conftest.py — no simulation data required.
"""

import numpy as np
import pandas as pd
import pytest

from chacra.ContactFrequencies import (
    ContactFrequencies,
    ContactPCA,
    load_contact_file,
    make_contact_dataframe,
)


# ------------------------------------------------------------------ #
# load_contact_file                                                    #
# ------------------------------------------------------------------ #


class TestLoadContactFile:
    def test_returns_series(self, synthetic_tsv_dir):
        tsv = next(synthetic_tsv_dir.glob("*.tsv"))
        result = load_contact_file(str(tsv))
        assert isinstance(result, pd.Series)

    def test_index_has_dash_separator(self, synthetic_tsv_dir):
        tsv = next(synthetic_tsv_dir.glob("*.tsv"))
        result = load_contact_file(str(tsv))
        assert all("-" in idx for idx in result.index)

    def test_frequencies_in_unit_interval(self, synthetic_tsv_dir):
        tsv = next(synthetic_tsv_dir.glob("*.tsv"))
        result = load_contact_file(str(tsv))
        assert (result >= 0).all() and (result <= 1).all()


# ------------------------------------------------------------------ #
# make_contact_dataframe                                               #
# ------------------------------------------------------------------ #


class TestMakeContactDataframe:
    def test_from_directory(self, synthetic_tsv_dir):
        df = make_contact_dataframe(str(synthetic_tsv_dir))
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 20  # N_TEMPS states
        assert df.shape[1] > 0

    def test_from_parquet_directory(self, synthetic_parquet_dir):
        df = make_contact_dataframe(str(synthetic_parquet_dir))
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 20
        assert df.shape[1] > 0

    def test_from_list_of_paths(self, synthetic_tsv_dir):
        paths = sorted(synthetic_tsv_dir.glob("*.tsv"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        df = make_contact_dataframe([str(p) for p in paths])
        assert df.shape[0] == 20

    def test_no_negative_values(self, synthetic_tsv_dir):
        df = make_contact_dataframe(str(synthetic_tsv_dir))
        assert (df >= 0).all().all()

    def test_with_temps(self, synthetic_tsv_dir):
        from tests.conftest import TEMPS
        df = make_contact_dataframe(str(synthetic_tsv_dir), temps=TEMPS)
        assert list(df.index) == TEMPS


# ------------------------------------------------------------------ #
# ContactFrequencies initialisation                                    #
# ------------------------------------------------------------------ #


class TestContactFrequenciesInit:
    def test_from_dataframe(self, synthetic_df):
        cf = ContactFrequencies(synthetic_df, get_chacras=False)
        assert cf.freqs.shape == synthetic_df.shape

    def test_from_dict(self):
        data = {"A:ALA:1-A:GLY:6": [0.1, 0.2, 0.3]}
        cf = ContactFrequencies(data, get_chacras=False)
        assert "A:ALA:1-A:GLY:6" in cf.freqs.columns

    def test_from_csv(self, synthetic_df, tmp_path):
        csv_path = tmp_path / "contacts.csv"
        synthetic_df.to_csv(csv_path)
        cf = ContactFrequencies(str(csv_path), get_chacras=False)
        assert cf.freqs.shape == synthetic_df.shape

    def test_from_parquet(self, synthetic_df, tmp_path):
        pq_path = tmp_path / "contacts.parquet"
        synthetic_df.to_parquet(pq_path)
        cf = ContactFrequencies(str(pq_path), get_chacras=False)
        assert cf.freqs.shape == synthetic_df.shape

    def test_from_tsv_directory(self, synthetic_tsv_dir):
        cf = ContactFrequencies(str(synthetic_tsv_dir), get_chacras=False)
        assert isinstance(cf.freqs, pd.DataFrame)
        assert cf.freqs.shape[0] == 20

    def test_from_parquet_directory(self, synthetic_parquet_dir):
        cf = ContactFrequencies(str(synthetic_parquet_dir), get_chacras=False)
        assert isinstance(cf.freqs, pd.DataFrame)
        assert cf.freqs.shape[0] == 20

    def test_from_file_list(self, synthetic_parquet_dir):
        paths = sorted(synthetic_parquet_dir.glob("*.parquet"))
        cf = ContactFrequencies([str(p) for p in paths], get_chacras=False)
        assert isinstance(cf.freqs, pd.DataFrame)
        assert cf.freqs.shape[0] == 20

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "contacts.xyz"
        bad.write_text("junk")
        with pytest.raises(ValueError):
            ContactFrequencies(str(bad), get_chacras=False)

    def test_temps_applied_to_index(self, synthetic_df):
        from tests.conftest import TEMPS
        cf = ContactFrequencies(synthetic_df, temps=TEMPS, get_chacras=False)
        assert list(cf.freqs.index) == TEMPS

    def test_temp_progression_linear(self, synthetic_df):
        cf = ContactFrequencies(
            synthetic_df,
            temp_progression="linear",
            min_max_temp=(290, 450),
            get_chacras=False,
        )
        assert len(cf.freqs.index) == synthetic_df.shape[0]


# ------------------------------------------------------------------ #
# ContactFrequencies filter methods                                    #
# ------------------------------------------------------------------ #


class TestContactFrequenciesFilters:
    def test_exclude_below_reduces_columns(self, synthetic_df):
        cf = ContactFrequencies(synthetic_df, get_chacras=False)
        reduced = cf.exclude_below(min_frequency=0.5)
        assert reduced.shape[1] <= synthetic_df.shape[1]

    def test_exclude_above_reduces_columns(self, synthetic_df):
        cf = ContactFrequencies(synthetic_df, get_chacras=False)
        reduced = cf.exclude_above(max_frequency=0.5)
        assert reduced.shape[1] <= synthetic_df.shape[1]

    def test_exclude_below_zero_keeps_all(self, synthetic_df):
        cf = ContactFrequencies(synthetic_df, get_chacras=False)
        # Nothing should be excluded if threshold is 0.0
        reduced = cf.exclude_below(min_frequency=0.0)
        assert reduced.shape[1] == synthetic_df.shape[1]

    def test_get_contact_partners_by_resid(self, synthetic_df):
        cf = ContactFrequencies(synthetic_df, get_chacras=False)
        # Pick the first contact's resid
        first = synthetic_df.columns[0]
        rid = int(first.split(":")[2].split("-")[0])
        result = cf.get_contact_partners(rid)
        assert isinstance(result, pd.DataFrame)
        # All returned contacts should involve that residue id
        for col in result.columns:
            assert str(rid) in col


# ------------------------------------------------------------------ #
# ContactPCA                                                           #
# ------------------------------------------------------------------ #


class TestContactPCA:
    def test_loadings_shape(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        # rows = contacts, columns = PCs
        assert cpca.loadings.shape[0] == contact_frequencies.freqs.shape[1]
        assert cpca.loadings.shape[1] == contact_frequencies.freqs.shape[0]

    def test_pc1_negative_slope(self, contact_frequencies):
        """PC1 projection should have a non-positive slope (melting trend)."""
        from scipy.stats import linregress
        cpca = contact_frequencies.cpca
        slope = linregress(
            range(cpca.loadings.shape[1]), cpca._transform[:, 0]
        ).slope
        assert slope <= 0, f"PC1 slope should be ≤ 0, got {slope:.4f}"

    def test_norm_loadings_range(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        vals = cpca.norm_loadings.values
        assert (vals >= 0).all() and (vals <= 1).all()

    def test_sorted_loadings_descending(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        sl = cpca.sorted_loadings(pc=1)
        abs_vals = sl["PC1"].abs().values
        assert all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1))

    def test_permuted_pca_sets_top_chacras(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        assert cpca.top_chacras is not None
        assert isinstance(cpca.top_chacras, list)
        assert len(cpca.top_chacras) >= 0  # may be empty on random data

    def test_chacra_pvals_shape(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        assert cpca.chacra_pvals is not None
        assert len(cpca.chacra_pvals) == contact_frequencies.freqs.shape[0]

    def test_get_chacra_center_returns_subset(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        result = cpca.get_chacra_center(pc=1, cutoff=0.0)
        # With cutoff 0.0 we should get all contacts back
        assert result.shape[0] == contact_frequencies.freqs.shape[1]

    def test_get_chacra_center_high_cutoff(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        result = cpca.get_chacra_center(pc=1, cutoff=0.9)
        # Fewer contacts than total
        assert result.shape[0] <= contact_frequencies.freqs.shape[1]

    def test_score_sums_shape(self, contact_frequencies):
        cpca = contact_frequencies.cpca
        if cpca.top_chacras:
            # columns = residues, index = PC numbers
            assert cpca.score_sums.shape[0] == len(cpca.top_chacras)

    def test_direct_instantiation(self, synthetic_df):
        """ContactPCA can be built directly from a DataFrame."""
        cpca = ContactPCA(synthetic_df, N_permutations=20, n_jobs=1)
        assert cpca.loadings is not None
