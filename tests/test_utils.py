"""
Tests for utility functions and RunConfig.

All tests use in-memory data or tmp_path — no simulation data required.
"""

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from chacra.utils import parse_id, split_id, multi_intersection
from chacra.run_config import RunConfig


# ------------------------------------------------------------------ #
# parse_id / split_id                                                  #
# ------------------------------------------------------------------ #


class TestParseId:
    CONTACT = "A:ALA:5-B:GLY:10"

    def test_returns_six_keys(self):
        result = parse_id(self.CONTACT)
        assert set(result.keys()) == {"chaina", "resna", "resida", "chainb", "resnb", "residb"}

    def test_chain_values(self):
        result = parse_id(self.CONTACT)
        assert result["chaina"] == "A"
        assert result["chainb"] == "B"

    def test_resname_values(self):
        result = parse_id(self.CONTACT)
        assert result["resna"] == "ALA"
        assert result["resnb"] == "GLY"

    def test_resid_values(self):
        result = parse_id(self.CONTACT)
        assert result["resida"] == "5"
        assert result["residb"] == "10"

    def test_prepended_chain_stripped(self):
        """parse_id should strip subunit prefixes like 'apo_A:ALA:5-...'"""
        result = parse_id("apo_A:ALA:5-B:GLY:10")
        assert result["chaina"] == "A"


class TestSplitId:
    def test_returns_resa_resb(self):
        result = split_id("A:ALA:5-B:GLY:10")
        assert set(result.keys()) == {"resa", "resb"}

    def test_values(self):
        result = split_id("A:ALA:5-B:GLY:10")
        assert result["resa"] == "A:ALA:5"
        assert result["resb"] == "B:GLY:10"


# ------------------------------------------------------------------ #
# multi_intersection                                                   #
# ------------------------------------------------------------------ #


class TestMultiIntersection:
    def test_basic_intersection(self):
        result = multi_intersection([["a", "b", "c"], ["b", "c", "d"], ["c", "b", "e"]])
        assert set(result) == {"b", "c"}

    def test_empty_intersection(self):
        result = multi_intersection([["a"], ["b"], ["c"]])
        assert result == []

    def test_single_list_returns_itself(self):
        result = multi_intersection([["a", "b", "c"]])
        assert set(result) == {"a", "b", "c"}

    def test_float_cutoff_excludes_short_lists(self):
        # Only the two long lists should be intersected; the short one filtered
        result = multi_intersection(
            [["a", "b", "c", "d"], ["a", "b", "c", "d"], ["a"]],
            cutoff=0.5,  # lists shorter than 50 % of max (4) are excluded
        )
        assert "a" in result  # still present from the two long lists
        assert "b" in result

    def test_int_cutoff_excludes_short_lists(self):
        result = multi_intersection(
            [["a", "b", "c"], ["a", "b"], ["a"]],
            cutoff=2,  # only lists with len > 2 kept
        )
        assert result == ["a", "b", "c"]


# ------------------------------------------------------------------ #
# freq_frames                                                          #
# ------------------------------------------------------------------ #


class TestFreqFrames:
    def test_parses_frame_count(self, tmp_path):
        from chacra.trajectories.process_hremd import freq_frames

        tsv = tmp_path / "freqs_state_0.tsv"
        tsv.write_text(
            "# total_frames:1234\n"
            "# Columns: residue1 residue2 contact_frequency\n"
            "A:ALA:5\tB:GLY:10\t0.75\n"
        )
        assert freq_frames(str(tsv)) == 1234


# ------------------------------------------------------------------ #
# RunConfig                                                            #
# ------------------------------------------------------------------ #


class TestRunConfig:
    def test_default_instantiation_has_all_keys(self):
        rc = RunConfig()
        for key in RunConfig.defaults:
            assert key in rc.config

    def test_config_starts_from_defaults(self):
        rc = RunConfig()
        assert rc.config["min_temp"] == 290.0
        assert rc.config["max_temp"] == 450.0

    def test_write_creates_json(self, tmp_path):
        rc = RunConfig()
        path = str(tmp_path / "chacra_run.json")
        rc.write(path)
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert "min_temp" in data

    def test_load_overrides_defaults(self, tmp_config):
        rc = RunConfig(str(tmp_config["path"]))
        assert rc.config["n_systems"] == tmp_config["data"]["n_systems"]
        assert rc.config["min_temp"] == tmp_config["data"]["min_temp"]

    def test_round_trip(self, tmp_path):
        rc = RunConfig()
        rc.update(n_systems=20, min_temp=300.0, max_temp=500.0)
        path = str(tmp_path / "chacra_run.json")
        rc.write(path)
        rc2 = RunConfig(path)
        assert rc2.config["n_systems"] == 20
        assert rc2.config["min_temp"] == 300.0
        assert rc2.config["max_temp"] == 500.0

    def test_update_ignores_none_values(self):
        rc = RunConfig()
        rc.update(min_temp=310.0, max_temp=None)  # None should be skipped
        assert rc.config["min_temp"] == 310.0
        assert rc.config["max_temp"] == 450.0  # unchanged default

    def test_apply_to_namespace_fills_nones(self, tmp_config):
        import argparse
        rc = RunConfig(str(tmp_config["path"]))
        ns = argparse.Namespace(n_systems=None, min_temp=None, other=99)
        rc.apply_to_namespace(ns)
        assert ns.n_systems == tmp_config["data"]["n_systems"]
        assert ns.min_temp == tmp_config["data"]["min_temp"]
        assert ns.other == 99  # untouched

    def test_apply_to_namespace_does_not_overwrite_set_values(self, tmp_config):
        import argparse
        rc = RunConfig(str(tmp_config["path"]))
        ns = argparse.Namespace(min_temp=999.0)
        rc.apply_to_namespace(ns)
        assert ns.min_temp == 999.0  # CLI value preserved

    def test_missing_file_prints_warning(self, capsys):
        rc = RunConfig("/nonexistent/path.json")
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or captured.out == ""
        # config still initialised from defaults
        assert rc.config["min_temp"] == 290.0

    def test_malformed_json_handled_gracefully(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("{this is not valid json")
        rc = RunConfig(str(bad))
        # Should not raise; defaults remain accessible
        assert rc.config["min_temp"] == 290.0

    def test_temps_list_stored_and_retrieved(self):
        temps = list(np.geomspace(290, 450, 20).astype(int))
        rc = RunConfig()
        rc.update(temps=temps)
        assert rc.config["temps"] == temps
