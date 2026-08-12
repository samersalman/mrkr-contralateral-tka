"""Tests for src/subgroups.py - the ONE subgroup-family declaration and its parser.

``src/subgroups.py`` had no test file until 2026-08-11, which is precisely when it stopped
being a self-contained tally module and became the place where the subgroup families are
declared for the whole pipeline. ``src/eval_models.py`` imports :func:`load_families` and
:func:`family_mask` from it, so a change here now moves the published equity audit and the
v6 imaging robustness table as well as ``outputs/subgroup_counts.csv``.

Three things are pinned, and each protects something outside this module:

* **the six equity families and their order**, because that list IS the row set of the
  published ``outputs/tables/{val,test}_subgroups.csv``;
* **the regression anchors**, because they are the re-gate event counts that partition the
  533 primary events, and the new families are anchored to the same standard as the old;
* **the rule vocabulary**, because a rule that silently evaluates to all-False is a stratum
  that quietly vanishes from a table rather than failing loudly.

NOTHING here writes ``outputs/subgroup_counts.csv``. The full-pipeline test exercises
``build_patient_frame`` -> ``build_rows`` -> ``validate`` directly and asserts on the frame,
so the published CSV keeps its mtime. It is skipped when the Parquet inputs are absent.

Run::

    PYTHONPATH="$PWD" ~/.venvs/mrkr-torch/bin/python -m pytest tests/test_subgroups.py -q
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.subgroups as sg
from src.config import PROJECT_ROOT, load_config

QUIET = logging.getLogger("test_subgroups")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False

CONFIG = "config/feasibility.yaml"

#: The primary final landmark cohort (recovery_any / 730-day). Re-gate verified.
N_COHORT, N_EVENTS = 3709, 533


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def _synthetic_frame() -> pd.DataFrame:
    """A patient frame carrying every column any declared clinical rule can name."""
    return pd.DataFrame({
        "empi_anon": [f"{900000 + i}" for i in range(8)],
        "sex": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
        "race": ["African American or Black", "Caucasian or White", "Asian", "Other",
                 "Caucasian or White", "Asian", "African American or Black", "Other"],
        "age_at_index": [40.0, 64.9, 65.0, 70.0, 55.0, 80.0, 64.0, 66.0],
        "obesity": [1, 0, 1, 0, 1, 0, 1, 0],
        "weight_bearing_frontal": [True, False, True, False, True, False, True, False],
        "view_set": ["frontal", "frontal+lateral", "frontal", "frontal+lateral+sunrise",
                     "lateral", "frontal", "frontal+other", "sunrise"],
        "side_source": ["coded", "recovered", "coded", "recovered",
                        "coded", "recovered", "coded", "recovered"],
        "acquisition_year": [2007, 2011, 2012, 2015, 2016, 2018, 2019, 2021],
        "primary_event": [1, 0, 1, 0, 1, 0, 1, 0],
    })


# =========================================================================== #
# 1. THE DECLARATION - schema, scopes, labels                                  #
# =========================================================================== #
def test_every_family_lives_in_config_and_nowhere_else(cfg):
    """No family, level, label or rule may be written in Python."""
    src = Path(sg.__file__).read_text(encoding="utf-8")
    for hard_coded in ('add("sex"', 'add("race"', 'add("obesity"',
                       'add("imaging_weight_bearing"', 'add("imaging_views"'):
        assert hard_coded not in src, (
            f"{hard_coded} is back in src/subgroups.py; the family list lives in "
            f"config/feasibility.yaml -> subgroups.families and nowhere else")
    ev = Path(PROJECT_ROOT / "src" / "eval_models.py").read_text(encoding="utf-8")
    assert '("Sex", "Female"' not in ev and '("Race group", "Black"' not in ev, (
        "the hard-coded family list is back in src/eval_models.subgroup_levels")


def test_the_equity_scope_is_frozen_at_the_six_original_families(cfg):
    fams = sg.load_families(cfg, "equity")
    assert tuple(f.key for f in fams) == sg.FROZEN_EQUITY_FAMILIES
    assert [f.report_label for f in fams] == [
        "Sex", "Age group", "Race group", "Obesity",
        "Frontal radiograph technique", "Imaging views"]


def test_adding_a_family_to_the_equity_scope_is_refused(cfg):
    """The published {val,test}_subgroups.csv row set may not grow by a config edit."""
    bad = load_config(CONFIG)
    fams = [dict(f) for f in bad["subgroups"]["families"]]
    for f in fams:
        if f["key"] == "laterality_source":
            f["scopes"] = ["cohort", "equity", "robustness"]
    bad["subgroups"] = dict(bad["subgroups"])
    bad["subgroups"]["families"] = fams
    with pytest.raises(AssertionError, match="frozen"):
        sg.load_families(bad, "equity")


def test_the_published_equity_row_set_matches_the_published_csv(cfg):
    path = PROJECT_ROOT / "outputs" / "tables" / "test_subgroups.csv"
    if not path.exists():                                   # pragma: no cover
        pytest.skip("outputs/tables/test_subgroups.csv is not present")
    published = pd.read_csv(path)[["subgroup", "level"]].apply(tuple, axis=1).tolist()
    declared = [(f.report_label, lv.report_label)
                for f in sg.load_families(cfg, "equity") for lv in f.levels]
    assert declared == published, (
        "config subgroups.families no longer reproduces the published equity table's rows "
        "in the published order")


def test_reviewer_facing_labels_carry_no_underscores(cfg):
    for scope in sg.FAMILY_SCOPES:
        for fam in sg.load_families(cfg, scope):
            assert "_" not in fam.report_label
            for lv in fam.levels:
                assert "_" not in lv.report_label


def test_the_age_cutoff_is_substituted_into_both_label_sets(cfg):
    cut = int(cfg["subgroups"]["age_cutoff"])
    fam = next(f for f in sg.load_families(cfg, "cohort") if f.key == "age_at_index")
    assert [lv.report_label for lv in fam.levels] == [
        f"Under {cut} years", f"{cut} years or older"]
    assert [lv.cohort_label for lv in fam.levels] == [f"<{cut}", f">={cut}"]


def test_an_imaging_family_may_not_reach_the_cohort_or_equity_scopes(cfg):
    for scope in ("cohort", "equity"):
        assert all(f.source == "clinical" for f in sg.load_families(cfg, scope)), (
            "a source:imaging family needs the per-image preprocess_labels join, which "
            "src/subgroups.py deliberately does not perform")
    bad = load_config(CONFIG)
    fams = [dict(f) for f in bad["subgroups"]["families"]]
    for f in fams:
        if f["key"] == "image_masking":
            f["scopes"] = ["cohort", "robustness"]
    bad["subgroups"] = dict(bad["subgroups"]); bad["subgroups"]["families"] = fams
    with pytest.raises(AssertionError, match="source:imaging"):
        sg.load_families(bad, "cohort")


def test_an_unknown_operator_is_a_schema_error_not_an_empty_stratum(cfg):
    bad = load_config(CONFIG)
    fams = [dict(f) for f in bad["subgroups"]["families"]]
    fams[0] = dict(fams[0])
    fams[0]["levels"] = [dict(lv) for lv in fams[0]["levels"]]
    fams[0]["levels"][0] = dict(fams[0]["levels"][0])
    fams[0]["levels"][0]["rule"] = {"column": "sex", "op": "contains", "value": "F"}
    bad["subgroups"] = dict(bad["subgroups"]); bad["subgroups"]["families"] = fams
    with pytest.raises(AssertionError, match="not one of"):
        sg.load_families(bad, "cohort")


def test_an_unknown_scope_is_refused(cfg):
    with pytest.raises(AssertionError, match="unknown family scope"):
        sg.load_families(cfg, "equity_v2")


def test_every_declared_scope_yields_at_least_one_family(cfg):
    for scope in sg.FAMILY_SCOPES:
        assert sg.load_families(cfg, scope), f"scope {scope} yields nothing"


# =========================================================================== #
# 2. THE RULE VOCABULARY                                                       #
# =========================================================================== #
def test_every_rule_operator_selects_what_it_says(cfg):
    f = _synthetic_frame()
    m = sg.family_mask
    assert m({"column": "sex", "op": "eq", "value": "Female"}, f, cfg).sum() == 4
    assert m({"column": "view_set", "op": "ne", "value": "frontal"}, f, cfg).sum() == 5
    assert m({"column": "age_at_index", "op": "lt", "value": 65}, f, cfg).sum() == 4
    assert m({"column": "age_at_index", "op": "ge", "value": 65}, f, cfg).sum() == 4
    assert m({"column": "acquisition_year", "op": "le", "value": 2015}, f, cfg).sum() == 4
    assert m({"column": "acquisition_year", "op": "gt", "value": 2015}, f, cfg).sum() == 4
    assert m({"column": "acquisition_year", "op": "between",
              "value": [2012, 2016]}, f, cfg).sum() == 3
    assert m({"column": "weight_bearing_frontal", "op": "is_true"}, f, cfg).sum() == 4
    assert m({"column": "weight_bearing_frontal", "op": "is_false"}, f, cfg).sum() == 4


def test_value_from_resolves_through_the_subgroups_block(cfg):
    f = _synthetic_frame()
    cut = int(cfg["subgroups"]["age_cutoff"])
    by_ref = sg.family_mask({"column": "age_at_index", "op": "lt",
                             "value_from": "age_cutoff"}, f, cfg)
    by_literal = sg.family_mask({"column": "age_at_index", "op": "lt", "value": cut}, f, cfg)
    assert by_ref.tolist() == by_literal.tolist()
    black = sg.family_mask({"column": "race", "op": "eq",
                            "value_from": "race_groups.black"}, f, cfg)
    assert black.sum() == 2


def test_a_missing_value_never_satisfies_any_rule(cfg):
    f = _synthetic_frame()
    f["weight_bearing_frontal"] = f["weight_bearing_frontal"].astype(object)
    f.loc[0, "age_at_index"] = np.nan
    f.loc[1, "weight_bearing_frontal"] = None
    f.loc[2, "acquisition_year"] = np.nan
    lo = sg.family_mask({"column": "age_at_index", "op": "lt", "value": 65}, f, cfg)
    hi = sg.family_mask({"column": "age_at_index", "op": "ge", "value": 65}, f, cfg)
    assert not lo[0] and not hi[0], "a NaN age must fall into NEITHER age bucket"
    yes = sg.family_mask({"column": "weight_bearing_frontal", "op": "is_true"}, f, cfg)
    no = sg.family_mask({"column": "weight_bearing_frontal", "op": "is_false"}, f, cfg)
    assert not yes[1] and not no[1], "a null flag must fall into NEITHER level"
    btw = sg.family_mask({"column": "acquisition_year", "op": "between",
                          "value": [2000, 2100]}, f, cfg)
    assert not btw[2]


def test_a_rule_naming_an_absent_column_fails_loudly(cfg):
    with pytest.raises(AssertionError, match="not on the frame"):
        sg.family_mask({"column": "manufacturer", "op": "eq", "value": "GE"},
                       _synthetic_frame(), cfg)


def test_every_partition_family_partitions_the_synthetic_frame(cfg):
    """A partition declared in config must actually be one, on any frame."""
    f = _synthetic_frame()
    for fam in sg.load_families(cfg, "cohort"):
        if not fam.partition:
            continue
        covered = np.zeros(len(f), dtype=bool)
        for lv in fam.levels:
            mask = sg.family_mask(lv.rule, f, cfg)
            assert not (covered & mask).any(), f"{fam.key} levels overlap"
            covered |= mask
        assert covered.all(), f"{fam.key} leaves a row in no level"


def test_the_three_era_schemes_are_three_different_partitions(cfg):
    f = _synthetic_frame()
    schemes = {fam.key: [sg.family_mask(lv.rule, f, cfg).tolist() for lv in fam.levels]
               for fam in sg.load_families(cfg, "robustness")
               if fam.key.startswith("acquisition_era")}
    assert len(schemes) == 3
    assert len({tuple(map(tuple, v)) for v in schemes.values()}) == 3, (
        "two era schemes produce the same partition; one of them is redundant")


# =========================================================================== #
# 3. THE ERA CAVEAT - it must travel WITH the numbers                          #
# =========================================================================== #
def test_every_era_family_carries_the_unconfirmed_date_caveat(cfg):
    eras = [f for f in sg.load_families(cfg, "robustness")
            if f.key.startswith("acquisition_era")]
    assert eras, "no acquisition-era family is declared"
    for fam in eras:
        note = fam.note
        assert note, f"{fam.key} carries no note; the caveat would live only in prose"
        for phrase in ("per-patient random shift", "section 17", "not been obtained",
                       "D17", "D35"):
            assert phrase.lower() in note.lower(), (
                f"{fam.key} note does not mention {phrase!r}")


def test_the_era_caveat_is_identical_across_the_three_schemes(cfg):
    notes = {f.note for f in sg.load_families(cfg, "robustness")
             if f.key.startswith("acquisition_era")}
    assert len(notes) == 1, "the three era schemes disagree about their own caveat"


def test_the_laterality_source_family_separates_the_two_inferred_exposures(cfg):
    fam = next(f for f in sg.load_families(cfg, "robustness")
               if f.key == "laterality_source")
    assert "not a neural-network output" in fam.note
    assert "D2" in fam.note


def test_the_unavailable_strata_are_declared_rather_than_omitted(cfg):
    keys = {u["key"] for u in cfg["subgroups"]["unavailable_strata"]}
    assert {"equipment", "manufacturer", "site"} <= keys, (
        "the editor asked for equipment and site; a stratum that cannot be built must be "
        "stated as unavailable, not dropped")
    assert "horizontal_flip" in keys
    for u in cfg["subgroups"]["unavailable_strata"]:
        assert len(str(u["reason"])) > 40, f"{u['key']} has no substantive reason"


# =========================================================================== #
# 4. THE TALLY AND ITS ANCHORS                                                 #
# =========================================================================== #
def test_the_stability_bins_are_the_protocol_ones(cfg):
    stability = sg.make_stability(cfg)
    assert stability(49) == "<50" and stability(50) == "50-99"
    assert stability(99) == "50-99" and stability(100) == ">=100"


def test_build_rows_emits_one_row_per_declared_level_in_declaration_order(cfg):
    f = _synthetic_frame()
    df = sg.build_rows(f, cfg, sg.make_stability(cfg))
    expected = [(fam.cohort_label, lv.cohort_label)
                for fam in sg.load_families(cfg, "cohort") for lv in fam.levels]
    assert list(zip(df.subgroup_family, df.subgroup)) == expected
    assert list(df.columns) == sg.SUBGROUP_COUNT_COLUMNS
    assert not any("empi" in c.lower() for c in df.columns)


def test_the_anchor_table_still_holds_every_original_anchor():
    """The seven pre-2026-08-11 anchors are unchanged, not merely present."""
    original = {
        ("sex", "Female", "n_events"): 342,
        ("sex", "Male", "n_events"): 191,
        ("age_at_index", "<65", "n_events"): 228,
        ("age_at_index", ">=65", "n_events"): 305,
        ("race", "Black", "n_events"): 156,
        ("race", "White", "n_events"): 328,
        ("race", "Asian", "n_events"): 13,
    }
    for key, value in original.items():
        assert sg.ANCHORS.get(key) == value, f"anchor {key} was loosened or dropped"
    assert original[("sex", "Female", "n_events")] + original[("sex", "Male", "n_events")] \
        == N_EVENTS


def test_every_new_family_is_anchored_not_merely_partition_checked(cfg):
    anchored = {fam for fam, _sub, _col in sg.ANCHORS}
    for fam in sg.load_families(cfg, "cohort"):
        if fam.key in ("obesity", "imaging_weight_bearing", "imaging_views"):
            continue          # deliberately anchor-free; partition-checked only
        assert fam.cohort_label in anchored, (
            f"{fam.key} carries no regression anchor; the new families are pinned to the "
            f"same standard as the original six")


def test_every_anchor_names_a_level_that_actually_exists(cfg):
    declared = {(fam.cohort_label, lv.cohort_label)
                for fam in sg.load_families(cfg, "cohort") for lv in fam.levels}
    for fam, sub, _col in sg.ANCHORS:
        assert (fam, sub) in declared, f"anchor names {fam}/{sub}, which is not declared"


def test_validate_fails_on_a_moved_anchor(cfg):
    df = pd.DataFrame([
        {"subgroup_family": fam.cohort_label, "subgroup": lv.cohort_label,
         "n_patients": 1, "n_events": 1, "event_pct": 100.0, "stability_flag": "<50"}
        for fam in sg.load_families(cfg, "cohort") for lv in fam.levels])
    with pytest.raises(AssertionError, match="ANCHOR FAIL"):
        sg.validate(df, N_COHORT, N_EVENTS, QUIET, cfg=cfg)


# =========================================================================== #
# 5. THE REAL COHORT - anchors against the artefacts, writing nothing          #
# =========================================================================== #
def _inputs_present(cfg) -> bool:
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    try:
        return ((coh / "final_cohort.parquet").exists()
                and cfg.parquet_path("demographics").exists()
                and cfg.parquet_path("icd").exists())
    except Exception:                                       # pragma: no cover
        return False


def test_the_real_cohort_reproduces_every_anchor_and_every_partition(cfg):
    if not _inputs_present(cfg):                            # pragma: no cover
        pytest.skip("the typed Parquet inputs are not present on this machine")
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    try:
        pt = sg.build_patient_frame(con, cfg, QUIET)
    finally:
        con.close()
    assert len(pt) == N_COHORT and int(pt["primary_event"].sum()) == N_EVENTS
    assert {"side_source", "acquisition_year"} <= set(pt.columns)
    assert int(pt["acquisition_year"].min()) == 2007
    assert int(pt["acquisition_year"].max()) == 2021
    df = sg.build_rows(pt, cfg, sg.make_stability(cfg))
    sg.validate(df, N_COHORT, N_EVENTS, QUIET, cfg=cfg)      # every anchor, every partition


def test_the_original_thirteen_rows_are_byte_identical_to_the_published_file(cfg):
    """The refactor may not move outputs/subgroup_counts.csv by a single character."""
    path = PROJECT_ROOT / "outputs" / "subgroup_counts.csv"
    if not path.exists() or not _inputs_present(cfg):       # pragma: no cover
        pytest.skip("outputs/subgroup_counts.csv or the Parquet inputs are absent")
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    try:
        pt = sg.build_patient_frame(con, cfg, QUIET)
    finally:
        con.close()
    fresh = sg.build_rows(pt, cfg, sg.make_stability(cfg))
    published = pd.read_csv(path)
    head = fresh.iloc[:len(published)].reset_index(drop=True)
    pd.testing.assert_frame_equal(head, published, check_dtype=False)
