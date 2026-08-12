"""Tests for src/v6_analyses.py - the v6 post-hoc analyses A1 and A2.

Two halves.

The first needs no data at all: the Kellgren-Lawrence bin edges, the 50-event floor, the
multiplicity isolation and the output namespace are properties of the module's declarations
and are checked directly. These are the guards that matter, because each one protects
something outside this module - a published table's adjusted p values, a protocol section,
or a file that another agent owns.

The second runs the real pipeline once, at a reduced bootstrap count, against the published
artefacts, and asserts the things a smaller bootstrap cannot move: patient counts, event
counts, point estimates, stratum membership, which cells the floor suppresses, and the
arithmetic tying the likelihood-ratio test to the two log-likelihoods it is computed from.
Nothing here asserts an interval bound, because those legitimately depend on the replicate
count; the production tables are written with the protocol's 2,000.

Run::

    PYTHONPATH="$PWD" ~/.venvs/mrkr-torch/bin/python -m pytest tests/test_v6_analyses.py -q
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import src.v6_analyses as v6
from src.config import load_config
from src.eval_models import COMPARISON_COLUMNS
from src.model_clinical import SUPPRESS_BELOW_EVENTS, cloglog
from src.train_model import apply_recalibration

QUIET = logging.getLogger("test_v6_analyses")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False

# The KLG-eligible test population, measured by T1 against outputs/tables/ and re-measured
# by the fixture below. 2 x 50 > 98 is the whole reason A2(iii) is mostly suppressed.
KLG_N, KLG_EVENTS = 707, 98

# Stratum anchors under the declared primary scheme.
KL3_ANCHORS = {"KL 0-1": (144, 3), "KL 2": (266, 30), "KL 3-4": (297, 65)}

# The two values M0's paired estimate must NOT take on the M2 intersection: 0.682366 is its
# marginal estimate on all 741 patients and 0.680394 is its estimate paired against M4 on
# 740. On M2's 734 it is a third number, and printing either of the other two as the
# reference level of this contrast is the exact error T1 flagged.
M0_MARGINAL = 0.682366
M0_PAIRED_AGAINST_M4 = 0.680394
M2_MARGINAL_AUC = 0.836759
M2_MARGINAL_C = 0.778775


# =========================================================================== #
# 1. DECLARATIONS - no data required                                           #
# =========================================================================== #
def test_the_bin_edges_are_declared_in_the_module_not_chosen_by_pandas():
    assert v6.KL_EDGES == (1.5, 2.5)
    for scheme, spec in v6.KL_SCHEMES.items():
        assert spec["edges"], f"{scheme} declares no edges"
        assert all(float(e) == e for e in spec["edges"])
        assert spec["tie_rule"] in {"high", "low"}
        assert len(spec["strata"]) == len(spec["edges"]) + 1


def test_the_primary_scheme_partitions_every_observed_grade():
    grades = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    seen = np.zeros(grades.size, dtype=bool)
    for _order, _label, _rule, mask in v6.kl_stratum_masks(grades, "kl3"):
        assert not (seen & mask).any(), "strata overlap"
        seen |= mask
    assert seen.all(), "a grade fell into no stratum"


def test_a_missing_grade_falls_into_no_stratum():
    grades = np.array([1.0, np.nan, 3.0])
    covered = np.zeros(3, dtype=bool)
    for _o, _l, _r, m in v6.kl_stratum_masks(grades, "kl3"):
        covered |= m
    assert covered.tolist() == [True, False, True]


def test_the_tie_rule_is_the_declared_one_and_the_sensitivity_reverses_it():
    grades = np.array([1.5, 2.5])
    high = {lab: mask.tolist() for _o, lab, _r, mask in v6.kl_stratum_masks(grades, "kl3")}
    low = {lab: mask.tolist()
           for _o, lab, _r, mask in v6.kl_stratum_masks(grades, "kl3_tielow")}
    # 1.5 -> KL 2 and 2.5 -> KL 3-4 under the primary (left-closed) rule
    assert high["KL 2"] == [True, False] and high["KL 3-4"] == [False, True]
    # the sensitivity sends each tie down one stratum
    assert low["KL 0-1"] == [True, False] and low["KL 2"] == [False, True]


def test_the_event_floor_is_the_frozen_protocol_value_and_is_not_redefined_here():
    assert SUPPRESS_BELOW_EVENTS == 50
    cfg = load_config("config/feasibility.yaml")
    assert int(cfg["model_eval"]["suppress_below_events"]) == SUPPRESS_BELOW_EVENTS
    src = (v6.__file__ and open(v6.__file__, encoding="utf-8").read())
    assert "SUPPRESS_BELOW_EVENTS =" not in src, (
        "the module must import the floor, never define its own copy")
    assert "monkeypatch" not in src and "setattr(em" not in src


def test_no_partition_of_the_klg_population_can_leave_two_estimable_strata():
    # Stated as a test because it is the reason A2(iii) is mostly suppressed, and a future
    # reader will otherwise assume the suppression is a bug.
    assert 2 * SUPPRESS_BELOW_EVENTS > KLG_EVENTS


def test_the_new_families_are_disjoint_from_every_published_family():
    published = set(pd.read_csv("outputs/tables/test_comparisons.csv")["family"].unique())
    new = set(v6.POSTHOC_FAMILIES) | {v6.WITHIN_STRATUM_FAMILY}
    assert not (published & new), (
        f"family name collision {published & new}: Benjamini-Hochberg is applied inside a "
        f"family, so reusing a published family name would change published adjusted p values")


def test_every_output_is_namespaced_and_collides_with_nothing_published():
    import os
    published = {n for n in os.listdir("outputs/tables") if not n.startswith("v6_")}
    assert "test_comparisons.csv" in published, "the published table this must not overwrite"
    for name in v6.OUTPUT_BASENAMES.values():
        assert name.startswith("v6_") and name.endswith(".csv")
        assert name not in published


def test_the_recalibration_cannot_change_the_cox_score_once_it_is_standardized():
    """The invariance the module's docstring claims, checked rather than asserted in prose.

    The frozen transform is affine on the cloglog scale, so standardising the cloglog score
    removes it entirely. That is why A2 cannot be wrong through a double-applied (or a
    missing) recalibration.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.9, size=500)
    recal = {"intercept": 0.3469, "slope": 1.6383}      # m4_fusion at 1825 d, train_arms.json

    def z(x):
        c = cloglog(x)
        return (c - c.mean()) / c.std(ddof=1)

    assert np.allclose(z(p), z(apply_recalibration(p, recal)), atol=1e-10)


# =========================================================================== #
# 2. THE REAL PIPELINE, RUN ONCE                                               #
# =========================================================================== #
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = load_config("config/feasibility.yaml")
    out = tmp_path_factory.mktemp("v6")
    tables = v6.run(cfg, QUIET, out, n_boot=100)
    return tables, out


def test_the_analysis_population_is_the_klg_eligible_test_split(built):
    tables, _ = built
    cox = tables["cox"]
    for arm in v6.A2_ARMS:
        rows = cox[cox["arm"] == arm]
        assert set(rows["n_patients"]) == {KLG_N}
        assert set(rows["n_events"]) == {KLG_EVENTS}


def test_the_strata_hold_the_counts_the_manuscript_will_print(built):
    tables, _ = built
    st = tables["strata"]
    kl3 = st[st["scheme"] == "kl3"].set_index("stratum")
    for label, (n, ev) in KL3_ANCHORS.items():
        assert int(kl3.loc[label, "n_patients"]) == n
        assert int(kl3.loc[label, "n_events"]) == ev
    assert int(kl3["n_patients"].sum()) == KLG_N
    assert int(kl3["n_events"].sum()) == KLG_EVENTS
    # exactly one stratum clears the floor, and it is the advanced one
    cleared = kl3[kl3["cleared_event_floor"].astype(bool)].index.tolist()
    assert cleared == ["KL 3-4"]


def test_every_written_row_says_it_is_post_hoc(built):
    tables, _ = built
    for name, df in tables.items():
        assert len(df) > 0, name
        assert df["note"].astype(str).str.contains("POST HOC EXPLORATORY").all(), name


def test_no_new_contrast_claims_to_be_primary(built):
    tables, _ = built
    comp = tables["comparisons"]
    assert list(comp.columns) == COMPARISON_COLUMNS
    assert not comp["is_primary"].astype(bool).any()


def test_the_headline_contrast_is_paired_on_the_intersection_not_on_a_marginal_level(built):
    tables, _ = built
    comp = tables["comparisons"]
    row = comp[(comp["model"] == "m2_frontal") & (comp["reference"] == "m0")
               & (comp["metric"] == "auc")].iloc[0]
    assert int(row["n_paired"]) == 734
    # M2 scores exactly the 734 it is paired on, so its paired level IS its marginal one
    assert row["estimate_model"] == pytest.approx(M2_MARGINAL_AUC, abs=5e-6)
    # M0's is a third value, neither its marginal nor its M4-paired estimate
    assert abs(row["estimate_reference"] - M0_MARGINAL) > 1e-3
    assert abs(row["estimate_reference"] - M0_PAIRED_AGAINST_M4) > 1e-3
    assert row["difference"] == pytest.approx(row["estimate_model"] - row["estimate_reference"],
                                              abs=1e-9)


def test_the_klg_contrast_is_paired_on_the_klg_eligible_set(built):
    tables, _ = built
    comp = tables["comparisons"]
    for ref in ("m1", "m1_klg"):
        rows = comp[(comp["reference"] == ref) & (comp["family"] != v6.WITHIN_STRATUM_FAMILY)]
        assert len(rows) > 0
        assert set(rows["n_paired"]) == {KLG_N}


def test_harrell_c_rows_exist_beside_the_auroc_rows(built):
    tables, _ = built
    comp = tables["comparisons"]
    pairs_auc = {(r.model, r.reference) for r in comp[comp["metric"] == "auc"].itertuples()}
    pairs_c = {(r.model, r.reference) for r in comp[comp["metric"] == "harrell_c"].itertuples()}
    assert pairs_auc == pairs_c and pairs_auc
    c_row = comp[(comp["model"] == "m2_frontal") & (comp["reference"] == "m0")
                 & (comp["metric"] == "harrell_c")].iloc[0]
    assert c_row["estimate_model"] == pytest.approx(M2_MARGINAL_C, abs=5e-6)


def test_benjamini_hochberg_runs_inside_each_new_family_alone(built):
    tables, _ = built
    comp = tables["comparisons"]
    for family, g in comp.groupby("family"):
        est = g[np.isfinite(g["p_two_sided"])]
        if est.empty:
            continue
        m = len(est)
        for _, r in est.iterrows():
            assert r["p_adjusted"] >= r["p_two_sided"] - 1e-12
            assert r["p_adjusted"] <= min(1.0, r["p_two_sided"] * m) + 1e-12
            assert r["fdr_method"] == "bh"


def test_running_this_module_writes_only_its_own_v6_files(built):
    tables, out = built
    written = sorted(p.name for p in out.iterdir())
    assert written == sorted(v6.OUTPUT_BASENAMES.values()), (
        "the module wrote something outside its declared v6_ namespace")
    published = pd.read_csv("outputs/tables/test_comparisons.csv")
    assert len(published) == 8, "the published contrast table gained or lost a row"
    assert not published["family"].isin(list(v6.POSTHOC_FAMILIES)
                                        + [v6.WITHIN_STRATUM_FAMILY]).any()


def test_the_within_stratum_auroc_honours_the_floor_and_flags_wide_intervals(built):
    tables, _ = built
    au = tables["auroc"]
    assert (au.loc[au["n_events"] < SUPPRESS_BELOW_EVENTS, "suppressed"]).all()
    assert not (au.loc[au["n_events"] >= SUPPRESS_BELOW_EVENTS, "suppressed"]).any()
    supp = au[au["suppressed"].astype(bool)]
    assert supp["estimate"].isna().all() and supp["ci_lo"].isna().all()
    assert supp["suppression_reason"].str.contains("section 21").all()
    live = au[~au["suppressed"].astype(bool)]
    assert len(live) > 0
    assert set(live["stratum"]) == {"KL 3-4"}
    ok = live["ci_hi"] - live["ci_lo"]
    assert np.allclose(live["ci_width"], ok)
    assert (live["wide_interval"] == (live["ci_width"] >= v6.WIDE_INTERVAL_WIDTH)).all()


def test_a_suppressed_within_stratum_contrast_carries_no_estimate_at_all(built):
    tables, _ = built
    comp = tables["comparisons"]
    w = comp[comp["family"] == v6.WITHIN_STRATUM_FAMILY]
    assert len(w) == 6                                   # 3 strata x 2 metrics
    dead = w[w["note"].str.contains("SUPPRESSED")]
    assert len(dead) == 4                                # KL 0-1 and KL 2, both metrics
    for col in ("estimate_model", "estimate_reference", "difference", "ci_lo", "ci_hi",
                "p_two_sided", "p_adjusted"):
        assert dead[col].isna().all(), col


def test_the_tertiles_partition_each_stratum(built):
    tables, _ = built
    t = tables["tertiles"]
    for (arm, stratum), g in t.groupby(["arm", "stratum"]):
        assert len(g) == v6.N_TERTILES
        assert int(g["n_patients"].sum()) == int(g["n_stratum_patients"].iloc[0])
        assert int(g["n_events"].sum()) == int(g["n_stratum_events"].iloc[0])
        # equal-count bins, to within one patient
        assert int(g["n_patients"].max()) - int(g["n_patients"].min()) <= 1
        # tertiles are ascending in predicted risk and do not overlap
        hi = g.sort_values("tertile")["max_predicted_risk"].to_numpy()
        lo = g.sort_values("tertile")["min_predicted_risk"].to_numpy()
        assert np.all(hi[:-1] <= lo[1:])


def test_the_tertiles_cover_the_whole_klg_population_in_the_primary_scheme(built):
    tables, _ = built
    t = tables["tertiles"]
    for arm in v6.A2_ARMS:
        strata = t[(t["arm"] == arm) & (t["stratum"] != "All KL grades")]
        assert int(strata["n_patients"].sum()) == KLG_N
        assert int(strata["n_events"].sum()) == KLG_EVENTS


def test_the_likelihood_ratio_test_is_the_two_log_likelihoods_it_reports(built):
    from scipy import stats

    tables, _ = built
    cox = tables["cox"]
    for _, r in cox.iterrows():
        assert r["lr_chi2"] == pytest.approx(2.0 * (r["ll_full"] - r["ll_reference"]), abs=1e-6)
        assert r["lr_df"] == 1.0
        assert r["lr_p"] == pytest.approx(float(stats.chi2.sf(max(r["lr_chi2"], 0.0), 1)),
                                          rel=1e-6, abs=1e-300)


def test_a_tiny_parametric_p_survives_the_six_decimal_rounding_of_the_csv(built):
    """The CSV writer rounds every float to 6 dp, which would print 5e-16 as 0.0."""
    _tables, out = built
    cox = pd.read_csv(out / v6.OUTPUT_BASENAMES["cox"])
    score = cox[cox["is_score_term"].astype(bool)]
    assert (score["lr_p"] == 0.0).any(), "the fixture no longer exercises the rounding case"
    for _, r in score.iterrows():
        assert float(r["lr_p_text"]) > 0.0
        assert float(r["lr_p_text"]) == pytest.approx(
            float(v6._p_text(float(r["lr_p_text"]))), rel=1e-12)
    tert = pd.read_csv(out / v6.OUTPUT_BASENAMES["tertiles"])
    flat = tert[tert["logrank_p"] == 0.0]
    assert len(flat) > 0
    assert (flat["logrank_p_text"].astype(float) > 0.0).all()


def test_every_cox_specification_carries_the_score_term_and_a_hazard_ratio_per_sd(built):
    tables, _ = built
    cox = tables["cox"]
    for (arm, spec), g in cox.groupby(["arm", "specification"]):
        s = g[g["is_score_term"].astype(bool)]
        assert len(s) == 1, (arm, spec)
        r = s.iloc[0]
        assert r["hr"] == pytest.approx(float(np.exp(r["coef"])), rel=1e-8)
        assert r["hr_lo"] < r["hr"] < r["hr_hi"]
        assert r["score_sd"] > 0
    assert set(cox["specification"]) == {"kl_linear", "kl_categorical", "kl_stratified",
                                         "score_only"}


def test_the_primary_cox_specification_enters_the_grade_as_recorded(built):
    tables, _ = built
    cox = tables["cox"]
    lin = cox[cox["specification"] == "kl_linear"]
    assert set(lin["term"]) == {"kl", "score_z"}
    assert lin["kl_form"].str.contains("primary").all()


def test_two_runs_write_identical_bytes(tmp_path):
    cfg = load_config("config/feasibility.yaml")
    a, b = tmp_path / "a", tmp_path / "b"
    v6.run(cfg, QUIET, a, n_boot=40)
    v6.run(cfg, QUIET, b, n_boot=40)
    for name in v6.OUTPUT_BASENAMES.values():
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
