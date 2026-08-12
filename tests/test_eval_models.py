"""Unit tests for src/eval_models.py - synthetic fixtures only, no checkpoint required.

The evaluator cannot be exercised against real hazards until ``src/train_model.py`` has run,
so every test here builds its own roster, its own arms and its own npz files. They are small
and hand-worked on purpose: the IPCW AUROC, the paired difference and the Benjamini-Hochberg
adjustment are checked against numbers computed on paper, so a silent change to an estimator
fails here rather than inside a metric nobody can audit.

Four contracts are pinned rather than merely tested, because breaking one of them breaks
something outside this module:

* the output column sets are compared against ``src/make_manuscript.py``'s own fixture
  functions, so the manuscript generator's expectations are enforced by a test instead of by
  hope;
* the sealed-test guard reuses ``src.train_model.assert_development_splits`` and is asserted
  to be the same object, so there is exactly one definition of "not the test split";
* the bootstrap uses ONE shared draw, proven by feeding two arms identical risks and
  requiring an exactly zero difference in every replicate;
* two runs on the same inputs produce identical frames and identical CSV bytes.

Run::

    ~/.venvs/mrkr-torch/bin/python -m pytest tests/test_eval_models.py -q
"""
from __future__ import annotations

import functools
import inspect
import json
import logging

import numpy as np
import pandas as pd
import pytest

import src.eval_models as em
import src.train_model as tm
from src.config import DEFAULT_CONFIG, Config, load_config
from src.model_clinical import SEALED_SPLIT, ipcw_auc, ipcw_labels_weights, percentile_ci

HORIZONS = [365, 730, 1825]
QUIET = logging.getLogger("test_eval_models")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False


# --------------------------------------------------------------------------- #
# Synthetic fixtures. A roster is 1:1 with the real one in shape, not in size. #
# --------------------------------------------------------------------------- #
def make_roster(n: int = 60, seed: int = 0, flat_censoring: bool = False) -> em.Roster:
    """A validation roster with the columns the subgroup audit reads."""
    rng = np.random.default_rng(seed)
    pids = np.array([f"{100000 + i}" for i in range(n)], dtype="<U8")
    event = (rng.uniform(size=n) < 0.35).astype(int)
    time = np.where(event == 1, rng.uniform(60.0, 1700.0, size=n),
                    rng.choice([600.0, 1200.0, 1826.0], size=n))
    frame = pd.DataFrame({
        "empi_anon": pids, "split": "val",
        "time_from_landmark": time, "event_indicator": event,
        "sex": np.where(rng.uniform(size=n) < 0.6, "Female", "Male"),
        "race": rng.choice(["Caucasian or White", "African American or Black", "Asian"], size=n),
        "age_at_index": rng.integers(45, 85, size=n).astype(float),
        "obesity": (rng.uniform(size=n) < 0.3).astype(int),
        "weight_bearing_frontal": rng.uniform(size=n) < 0.9,
        "view_set": rng.choice(["frontal", "frontal+lateral"], size=n),
    })
    if flat_censoring:
        g_grid, g_vals = np.array([0.0]), np.array([1.0])
    else:
        g_grid, g_vals = tm.reverse_km(time, event)
    return em.Roster(pids=pids, time=time, event=event, frame=frame,
                     g_grid=g_grid, g_vals=g_vals)


def make_arm(name: str, roster: em.Roster, *, signal: float = 1.0, seed: int = 1,
             drop: int = 0, horizons=HORIZONS) -> em.ArmScores:
    """One arm's roster-aligned scores. ``drop`` removes the last patients it can score."""
    rng = np.random.default_rng(seed)
    n = len(roster)
    present = np.ones(n, dtype=bool)
    if drop:
        present[-drop:] = False
    lin = signal * (-roster.time / 900.0) + 0.4 * rng.normal(size=n)
    rank = np.where(present, lin, np.nan)
    risk = {}
    for i, h in enumerate(horizons):
        p = 1.0 / (1.0 + np.exp(-(lin - 2.0 + 0.4 * i)))
        risk[h] = np.where(present, p, np.nan)
    return em.ArmScores(arm=name, label=f"label {name}", source="discrete_time",
                        present=present, rank=rank, risk=risk, val_nll=0.42)


def make_engine(roster: em.Roster, n_boot: int = 40, seed: int = 7,
                horizons=HORIZONS) -> em.BootstrapEngine:
    draw = em.bootstrap_draw(len(roster), n_boot, seed)
    return em.BootstrapEngine(roster, draw, horizons, QUIET)


def eval_config(primary, families, fdr: str = "bh") -> Config:
    """The slice of config that :func:`build_comparisons` reads, and nothing else."""
    return Config({"model_eval": {"primary_contrast": primary, "comparison_families": families,
                                  "fdr_method": fdr}})


# =========================================================================== #
# 1. THE OUTPUT CONTRACT src/make_manuscript.py CONSUMES                       #
# =========================================================================== #
def _manuscript_module():
    pytest.importorskip("docx", reason="src.make_manuscript imports python-docx at module level")
    import src.make_manuscript as mm
    return mm


def _manuscript_fixtures():
    mm = _manuscript_module()
    cfg = load_config(DEFAULT_CONFIG)
    inp = mm.Inputs(cfg=cfg, dry_run=True)
    inp.metrics = mm._fixture_metrics(cfg, inp)          # trained arms must exist first, so
    return mm, cfg, inp                                   # _fixture_comparisons has pairs to add


def test_val_metrics_columns_are_exactly_what_the_manuscript_declares():
    mm, cfg, inp = _manuscript_fixtures()
    hz = [int(h) for h in cfg["model_eval"]["horizons_days"]]
    assert set(em.metrics_columns(hz)) == set(inp.metrics.columns)
    # order too, not just membership: the manuscript reads by name but a stable order is
    # what makes two runs byte-identical and a diff readable.
    assert em.metrics_columns(hz) == list(inp.metrics.columns)


def test_val_comparisons_columns_are_exactly_what_the_manuscript_declares():
    mm, cfg, inp = _manuscript_fixtures()
    fixture = mm._fixture_comparisons(cfg, inp)
    assert not fixture.empty, "the manuscript fixture produced no contrast to compare against"
    assert set(em.COMPARISON_COLUMNS) == set(fixture.columns)
    assert em.COMPARISON_COLUMNS == list(fixture.columns)


def test_val_subgroups_columns_are_exactly_what_the_manuscript_declares():
    mm, cfg, inp = _manuscript_fixtures()
    fixture = mm._fixture_subgroups(cfg, inp)
    assert set(em.SUBGROUP_COLUMNS) == set(fixture.columns)
    assert em.SUBGROUP_COLUMNS == list(fixture.columns)


def test_val_results_json_carries_every_key_the_manuscript_reads():
    mm, cfg, inp = _manuscript_fixtures()
    declared = set(mm._fixture_results_json(cfg, inp))
    roster = make_roster(30)
    arms = {"m0": make_arm("m0", roster)}
    got = em.build_results_json(cfg, roster, arms, [], [365, 730, 1825], 2000, 1, "m0", {})
    assert declared <= set(got), f"missing key(s) {sorted(declared - set(got))}"
    # the default split is validation, and on validation the cohort keys are frozen: the
    # manuscript reads n_val / n_val_events off val_results.json by those exact names.
    assert set(got["cohort"]) == {"n_val", "n_val_events"}
    assert set(got["bootstrap"]) >= {"n", "seed", "note"}


def test_manuscript_figures_finds_the_columns_it_needs():
    """The figure module asserts an 'arm' column and reads auc_{h}* off the metrics table.

    ``metrics_columns`` is split-independent, so this pins the schema of BOTH files it can
    be handed. That distinction used to be academic. Now that the sealed read has been
    performed, the reported figures are drawn from ``test_metrics.csv`` rather than from
    ``val_metrics.csv``, so a column dropped here breaks the manuscript that is actually
    published, not only a development artefact.
    """
    cols = em.metrics_columns(HORIZONS)
    assert "arm" in cols
    for h in HORIZONS:
        assert {f"auc_{h}", f"auc_{h}_lo", f"auc_{h}_hi"} <= set(cols)


def test_the_horizons_are_the_frozen_ones_and_1826_is_not_among_them():
    cfg = load_config(DEFAULT_CONFIG)
    hz = em.horizons_from_config(cfg)
    assert hz == [365, 730, 1825]
    assert 1826 not in hz, "1826 is administrative censoring; its T > t control set is empty"


def test_1826_really_has_no_controls_so_the_clamp_to_1825_is_not_cosmetic():
    t = np.array([500.0, 1826.0, 1826.0, 1000.0])
    e = np.array([1, 0, 0, 1])
    y, _ = ipcw_labels_weights(t, e, 1825.0, np.array([0.0]), np.array([1.0]))
    assert (y == 0).sum() == 2, "1825 must still have controls"
    y26 = np.where((t <= 1826.0) & (e == 1), 1, np.where(t > 1826.0, 0, -1))
    assert (y26 == 0).sum() == 0, "at 1826 the T > t control set is empty by construction"


# =========================================================================== #
# 2. IPCW AUROC AND THE POINT METRICS, HAND-WORKED                             #
# =========================================================================== #
def _hand_worked_arm():
    """Four patients, flat censoring curve, so every IPCW weight is exactly 1.

    times/events   A(100, event)  B(200, event)  C(800, censored)  D(900, censored)
    risk           A .9           B .4           C .3              D .5
    At horizon 365: cases {A, B}, controls {C, D}.
      A > C, A > D, B > C, B < D  ->  AUC = 3 / 4 = 0.75
    Harrell C: comparable pairs A-B, A-C, A-D, B-C, B-D (5); concordant A-B, A-C, A-D, B-C
    (4) -> C = 4 / 5 = 0.80.
    """
    pids = np.array(["100000", "100001", "100002", "100003"], dtype="<U8")
    time = np.array([100.0, 200.0, 800.0, 900.0])
    event = np.array([1, 1, 0, 0])
    frame = pd.DataFrame({"empi_anon": pids, "split": "val",
                          "time_from_landmark": time, "event_indicator": event})
    roster = em.Roster(pids=pids, time=time, event=event, frame=frame,
                       g_grid=np.array([0.0]), g_vals=np.array([1.0]))
    risk = np.array([0.9, 0.4, 0.3, 0.5])
    arm = em.ArmScores(arm="toy", label="toy", source="discrete_time",
                       present=np.ones(4, dtype=bool), rank=risk, risk={365: risk})
    return roster, arm


def test_ipcw_auroc_matches_the_hand_worked_four_patient_example():
    roster, arm = _hand_worked_arm()
    m = em.arm_metrics(arm, roster, np.arange(4), [365], full=True)
    assert m["auc@365"] == pytest.approx(0.75)
    assert m["harrell_c"] == pytest.approx(0.80)
    assert m["n_patients"] == 4 and m["n_events"] == 2


def test_arm_metrics_agrees_with_calling_the_estimators_directly():
    """arm_metrics must be plumbing, not a second implementation."""
    roster, arm = _hand_worked_arm()
    y, w = ipcw_labels_weights(roster.time, roster.event, 365.0, roster.g_grid, roster.g_vals)
    assert em.arm_metrics(arm, roster, np.arange(4), [365], full=False)["auc@365"] == \
        pytest.approx(ipcw_auc(y, w, arm.risk[365]))


def test_a_tie_in_risk_scores_one_half():
    roster, arm = _hand_worked_arm()
    tied = np.array([0.5, 0.4, 0.3, 0.5])          # A now ties with D
    arm = em.ArmScores(arm="toy", label="toy", source="discrete_time",
                       present=np.ones(4, dtype=bool), rank=tied, risk={365: tied})
    # A > C, A ties D (0.5), B > C, B < D -> (1 + 0.5 + 1 + 0) / 4
    assert em.arm_metrics(arm, roster, np.arange(4), [365], full=False)["auc@365"] == \
        pytest.approx(0.625)


def test_a_sample_with_fewer_than_two_events_yields_nan_not_a_degenerate_number():
    roster, arm = _hand_worked_arm()
    m = em.arm_metrics(arm, roster, np.array([0, 2, 3]), [365], full=False)   # one event
    assert np.isnan(m["harrell_c"]) and np.isnan(m["auc@365"])
    assert m["n_events"] == 1


def test_only_the_patients_an_arm_scores_reach_the_metric():
    roster = make_roster(40, seed=3)
    arm = make_arm("a", roster, drop=8)
    pt = em.arm_metrics(arm, roster, np.flatnonzero(arm.present), HORIZONS, full=True)
    assert pt["n_patients"] == 32 == arm.n_patients
    assert pt["n_events"] == int(roster.event[arm.present].sum())


# =========================================================================== #
# 3. THE ONE SHARED BOOTSTRAP DRAW (protocol section 18)                       #
# =========================================================================== #
def test_the_draw_is_one_seeded_matrix_reused_by_everything():
    a = em.bootstrap_draw(371, 2000, 20250720)
    b = em.bootstrap_draw(371, 2000, 20250720)
    assert a.shape == (2000, 371)
    assert np.array_equal(a, b), "the same seed must give the same draw"
    assert not np.array_equal(a, em.bootstrap_draw(371, 2000, 20250721))
    assert a.min() >= 0 and a.max() < 371


def test_two_arms_with_identical_risks_differ_by_exactly_zero_in_every_replicate():
    """The point of a SHARED draw: replicate b scores both arms on the same patients."""
    roster = make_roster(50, seed=11)
    left = make_arm("left", roster, seed=5)
    right = em.ArmScores(arm="right", label="right", source="discrete_time",
                         present=left.present.copy(), rank=left.rank.copy(),
                         risk={h: v.copy() for h, v in left.risk.items()})
    engine = make_engine(roster, n_boot=60, seed=2)
    mask = left.present & right.present
    for key in em.boot_metric_keys(HORIZONS):
        d = engine.boot(left, mask)[key] - engine.boot(right, mask)[key]
        assert np.isfinite(d).sum() > 0, "no replicate was estimable; the test proves nothing"
        assert np.nanmax(np.abs(d)) == 0.0, (
            f"{key} differed between two identical arms; the draw is not shared")


def test_two_independent_draws_would_not_give_a_zero_difference():
    """The control for the test above: without a shared draw the difference is noise."""
    roster = make_roster(50, seed=11)
    left = make_arm("left", roster, seed=5)
    right = em.ArmScores(arm="right", label="right", source="discrete_time",
                         present=left.present.copy(), rank=left.rank.copy(),
                         risk={h: v.copy() for h, v in left.risk.items()})
    a = em.BootstrapEngine(roster, em.bootstrap_draw(len(roster), 60, 2), HORIZONS, QUIET)
    b = em.BootstrapEngine(roster, em.bootstrap_draw(len(roster), 60, 999), HORIZONS, QUIET)
    d = a.boot(left, left.present)["harrell_c"] - b.boot(right, right.present)["harrell_c"]
    assert np.nanmax(np.abs(d)) > 0.0


def test_an_identical_pair_is_a_zero_difference_contrast_with_a_p_of_one():
    roster = make_roster(50, seed=11)
    left = make_arm("left", roster, seed=5)
    right = em.ArmScores(arm="right", label="right", source="discrete_time",
                         present=left.present.copy(), rank=left.rank.copy(),
                         risk={h: v.copy() for h, v in left.risk.items()})
    engine = make_engine(roster, n_boot=60, seed=2)
    row = em.contrast_row("f", "left", "right", metric="auc", horizon=1825,
                          arms={"left": left, "right": right}, engine=engine, is_primary=False)
    assert row["difference"] == pytest.approx(0.0)
    assert row["ci_lo"] == pytest.approx(0.0) and row["ci_hi"] == pytest.approx(0.0)
    assert row["p_two_sided"] == pytest.approx(1.0)


def test_the_engine_caches_per_arm_and_patient_set():
    roster = make_roster(40, seed=4)
    arm = make_arm("a", roster)
    engine = make_engine(roster, n_boot=20)
    assert engine.point(arm, arm.present) is engine.point(arm, arm.present)
    other = arm.present.copy(); other[:5] = False
    assert engine.point(arm, other) is not engine.point(arm, arm.present)


def test_two_sided_bootstrap_p_is_floored_at_one_over_the_valid_replicates():
    p, n = em.two_sided_bootstrap_p(np.full(50, 0.4))       # every replicate strictly positive
    assert n == 50 and p == pytest.approx(1.0 / 50)
    p, n = em.two_sided_bootstrap_p(np.array([1.0, -1.0, 1.0, -1.0]))
    assert n == 4 and p == pytest.approx(1.0)
    assert np.isnan(em.two_sided_bootstrap_p(np.full(5, np.nan))[0])


# =========================================================================== #
# 4. PAIRING ON THE INTERSECTION OF TWO ARMS' PATIENT SETS                     #
# =========================================================================== #
def test_a_contrast_is_paired_on_the_intersection_and_says_so():
    roster = make_roster(60, seed=8)
    wide = make_arm("wide", roster, seed=1)                     # all 60
    narrow = make_arm("narrow", roster, seed=2, drop=9)         # 51
    engine = make_engine(roster, n_boot=30)
    row = em.contrast_row("f", "narrow", "wide", metric="auc", horizon=1825,
                          arms={"wide": wide, "narrow": narrow}, engine=engine, is_primary=False)
    assert row["n_paired"] == 51 == int((wide.present & narrow.present).sum())
    assert "51 patients both arms score" in row["note"]
    # both estimates are recomputed on the intersection, so neither is the marginal one
    marginal = engine.point(wide, wide.present)["auc@1825"]
    assert row["estimate_reference"] != pytest.approx(marginal)


def test_a_contrast_against_an_arm_that_is_not_scored_is_skipped_not_faked():
    roster = make_roster(30)
    arms = {"a": make_arm("a", roster)}
    engine = make_engine(roster, n_boot=10)
    assert em.contrast_row("f", "a", "missing", metric="auc", horizon=1825, arms=arms,
                           engine=engine, is_primary=False) is None
    assert em.contrast_row("f", "missing", "a", metric="auc", horizon=1825, arms=arms,
                           engine=engine, is_primary=True) is None


# =========================================================================== #
# 5. BENJAMINI-HOCHBERG **WITHIN** EACH FAMILY                                 #
# =========================================================================== #
def test_benjamini_hochberg_matches_the_hand_computed_adjustment():
    # p = .01 .02 .03 .04 (m = 4). Raw q = .04, .04, .04, .04 after the monotone sweep:
    #   .01*4/1=.040  .02*4/2=.040  .03*4/3=.040  .04*4/4=.040
    assert em.benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04] * 4)
    # p = .01 .40 .50 (m = 3): .03, .60, .50 -> monotone from the top -> .03, .50, .50
    assert em.benjamini_hochberg([0.01, 0.40, 0.50]) == pytest.approx([0.03, 0.5, 0.5])


def test_benjamini_hochberg_leaves_a_single_test_alone_and_never_exceeds_one():
    assert em.benjamini_hochberg([0.037]) == pytest.approx([0.037])
    assert float(em.benjamini_hochberg([0.9, 0.95])[1]) <= 1.0


def test_benjamini_hochberg_ignores_nans_and_does_not_count_them_in_m():
    out = em.benjamini_hochberg([0.01, np.nan, 0.02])
    assert np.isnan(out[1])
    assert out[[0, 2]] == pytest.approx(em.benjamini_hochberg([0.01, 0.02]))


def test_fdr_is_applied_inside_each_family_and_never_pooled_across_them():
    roster = make_roster(70, seed=21)
    names = ["m0", "a", "b", "c", "d"]
    arms = {n: make_arm(n, roster, signal=0.4 + 0.35 * i, seed=10 + i)
            for i, n in enumerate(names)}
    engine = make_engine(roster, n_boot=60, seed=3)
    cfg = eval_config({"model": "a", "reference": "m0", "horizon_days": 1825},
                      {"solo": [["b", "m0"]],
                       "trio": [["a", "m0"], ["c", "m0"], ["d", "m0"]]})
    df = em.build_comparisons(cfg, arms, engine, QUIET)

    solo = df[df.family == "solo"]
    trio = df[df.family == "trio"]
    assert len(solo) == 1 and len(trio) == 3
    # A family of one is its own multiplicity: BH with m = 1 is the identity. If the four
    # secondary contrasts had been pooled, this row would have been inflated by up to 4x.
    assert float(solo.iloc[0]["p_adjusted"]) == pytest.approx(float(solo.iloc[0]["p_two_sided"]))
    # Inside the family of three the adjustment is real and never shrinks a p value.
    assert (trio["p_adjusted"].to_numpy() >= trio["p_two_sided"].to_numpy() - 1e-12).all()
    assert trio["p_adjusted"].to_numpy() == pytest.approx(
        em.benjamini_hochberg(trio["p_two_sided"].to_numpy()))
    assert set(df["fdr_method"][df.family != "primary"]) == {"bh"}


def test_the_primary_contrast_is_unadjusted_and_flagged():
    roster = make_roster(60, seed=22)
    arms = {n: make_arm(n, roster, signal=0.5 + 0.4 * i, seed=30 + i)
            for i, n in enumerate(["m0", "m4_fusion", "m3_image"])}
    engine = make_engine(roster, n_boot=40, seed=4)
    cfg = eval_config({"model": "m4_fusion", "reference": "m0", "horizon_days": 1825},
                      {"modality": [["m3_image", "m0"]]})
    df = em.build_comparisons(cfg, arms, engine, QUIET)
    primary = df[df.is_primary]
    assert len(primary) == 1
    r = primary.iloc[0]
    assert r["family"] == "primary" and r["model"] == "m4_fusion" and r["reference"] == "m0"
    assert r["metric"] == "auc" and int(r["horizon_days"]) == 1825
    assert np.isnan(r["p_adjusted"]) and r["fdr_method"] == ""
    assert list(df.columns) == em.COMPARISON_COLUMNS


def test_comparison_families_are_written_in_a_deterministic_order():
    roster = make_roster(50, seed=23)
    arms = {n: make_arm(n, roster, seed=40 + i) for i, n in enumerate(["m0", "a", "b"])}
    engine = make_engine(roster, n_boot=20, seed=5)
    cfg = eval_config({"model": "a", "reference": "m0", "horizon_days": 1825},
                      {"zulu": [["b", "m0"]], "alpha": [["a", "m0"]]})
    fam = em.build_comparisons(cfg, arms, engine, QUIET)["family"].tolist()
    assert fam == ["primary", "alpha", "zulu"], "families must be emitted sorted by name"


def test_an_unknown_fdr_method_is_refused_rather_than_silently_ignored():
    roster = make_roster(30)
    arms = {"m0": make_arm("m0", roster), "a": make_arm("a", roster)}
    engine = make_engine(roster, n_boot=10)
    cfg = eval_config({"model": "a", "reference": "m0", "horizon_days": 1825},
                      {"f": [["a", "m0"]]}, fdr="holm")
    with pytest.raises(AssertionError, match="unknown model_eval.fdr_method"):
        em.build_comparisons(cfg, arms, engine, QUIET)


# =========================================================================== #
# 6. PROTOCOL SECTION 21 SUPPRESSION                                          #
# =========================================================================== #
def _subgroup_roster(n: int, n_events: int, seed: int = 0) -> em.Roster:
    """A roster whose events are concentrated in Female, so that level can clear the floor."""
    rng = np.random.default_rng(seed)
    pids = np.array([f"{200000 + i}" for i in range(n)], dtype="<U8")
    event = np.zeros(n, dtype=int); event[:n_events] = 1
    time = np.where(event == 1, rng.uniform(90.0, 1500.0, size=n), 1826.0)
    frame = pd.DataFrame({
        "empi_anon": pids, "split": "val", "time_from_landmark": time,
        "event_indicator": event,
        "sex": np.where(np.arange(n) < int(0.8 * n), "Female", "Male"),
        "race": "Caucasian or White",
        "age_at_index": np.where(np.arange(n) < n // 2, 55.0, 75.0),
        "obesity": 0, "weight_bearing_frontal": True, "view_set": "frontal",
    })
    g_grid, g_vals = tm.reverse_km(time, event)
    return em.Roster(pids=pids, time=time, event=event, frame=frame,
                     g_grid=g_grid, g_vals=g_vals)


def test_a_level_below_fifty_events_is_suppressed_with_a_reason():
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(120, 20, seed=1)
    arm = make_arm("m4_fusion", roster, seed=2)
    df = em.build_subgroups(cfg, roster, arm, make_engine(roster, n_boot=20), 1825, QUIET)
    assert df["suppressed"].all(), "20 events cannot clear the 50-event floor anywhere"
    assert df["estimate"].isna().all() and df["ci_lo"].isna().all()
    assert df["suppression_reason"].str.contains("protocol section 21").all()
    assert list(df.columns) == em.SUBGROUP_COLUMNS


def test_a_level_at_or_above_fifty_events_is_estimated_not_suppressed():
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(300, 140, seed=3)          # 140 events, all inside Female
    arm = make_arm("m4_fusion", roster, seed=4)
    df = em.build_subgroups(cfg, roster, arm, make_engine(roster, n_boot=30), 1825, QUIET)
    female = df[(df.subgroup == "Sex") & (df.level == "Female")].iloc[0]
    assert int(female["n_events"]) >= em.SUPPRESS_BELOW_EVENTS
    assert not bool(female["suppressed"]) and female["suppression_reason"] == ""
    assert np.isfinite(female["estimate"]) and np.isfinite(female["ci_lo"])
    male = df[(df.subgroup == "Sex") & (df.level == "Male")].iloc[0]
    assert bool(male["suppressed"]) and np.isnan(male["estimate"])


def test_the_suppression_floor_is_the_frozen_protocol_value():
    cfg = load_config(DEFAULT_CONFIG)
    assert int(cfg["model_eval"]["suppress_below_events"]) == em.SUPPRESS_BELOW_EVENTS == 50
    bad = load_config(DEFAULT_CONFIG)
    bad["model_eval"] = dict(bad["model_eval"]); bad["model_eval"]["suppress_below_events"] = 5
    roster = _subgroup_roster(60, 10)
    with pytest.raises(AssertionError, match="protocol section 21"):
        em.build_subgroups(bad, roster, make_arm("a", roster), make_engine(roster, 5), 1825, QUIET)


def test_subgroup_levels_come_from_the_config_block_and_carry_no_underscores():
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(80, 12)
    levels = em.subgroup_levels(cfg, roster.frame)
    names = {s for s, _, _ in levels}
    assert {"Sex", "Age group", "Race group", "Obesity", "Imaging views"} <= names
    cut = int(cfg["subgroups"]["age_cutoff"])
    assert any(lv == f"Under {cut} years" for _, lv, _ in levels)
    for sg, lv, _ in levels:
        assert "_" not in sg and "_" not in lv, "reviewer-facing names carry no underscores"


def test_no_arm_scored_yet_gives_a_fully_suppressed_table_rather_than_a_crash():
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(80, 60)
    df = em.build_subgroups(cfg, roster, None, make_engine(roster, 5), 1825, QUIET)
    assert df["suppressed"].all() and df["n_patients"].sum() == 0
    assert df["suppression_reason"].str.contains("no arm is scored yet").all()


# =========================================================================== #
# 7. THE SEALED-TEST GUARD                                                    #
# =========================================================================== #
def test_the_guard_is_the_one_from_train_model_not_a_second_copy():
    assert em.assert_development_splits is tm.assert_development_splits


def test_assert_validation_only_refuses_the_sealed_split():
    with pytest.raises(AssertionError, match="REFUSED"):
        em.assert_validation_only(["val", SEALED_SPLIT])
    with pytest.raises(AssertionError, match="REFUSED"):
        em.assert_validation_only([SEALED_SPLIT])


def test_assert_validation_only_also_refuses_the_train_split():
    assert em.assert_validation_only(["val"]) == ["val"]
    with pytest.raises(AssertionError, match="evaluates the 'val' split only"):
        em.assert_validation_only(["train", "val"])


def test_the_config_guard_must_still_be_switched_on():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg["model_eval"]["forbid_test_split"] is True
    em.assert_forbid_test_split_is_on(cfg)
    off = load_config(DEFAULT_CONFIG)
    off["model_eval"] = dict(off["model_eval"]); off["model_eval"]["forbid_test_split"] = False
    with pytest.raises(AssertionError, match="sealed-split guard"):
        em.assert_forbid_test_split_is_on(off)


def test_the_bootstrap_replicate_count_may_not_be_the_development_value():
    """Protocol Table 7 is 2,000 here; model_clinical's 500 is a development-report value."""
    cfg = load_config(DEFAULT_CONFIG)
    assert em.PROTOCOL_BOOTSTRAP_N == 2000
    assert int(cfg["model_eval"]["bootstrap_n"]) == em.PROTOCOL_BOOTSTRAP_N
    assert int(cfg["model_clinical"]["bootstrap_n"]) == 500
    assert int(cfg["model_eval"]["bootstrap_n"]) != int(cfg["model_clinical"]["bootstrap_n"])


def test_main_refuses_a_config_that_inherited_the_development_replicate_count(tmp_path):
    """The anchor is in the module, so editing config cannot lower it - only --bootstrap-n can."""
    import yaml
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["model_eval"]["bootstrap_n"] = raw["model_clinical"]["bootstrap_n"]      # the 500 trap
    raw["paths"]["run_log"] = str(tmp_path / "run.log")     # never touch the project's log
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(AssertionError, match="protocol Table 7 pre-specifies"):
        em.main(["--config", str(p)])


def test_the_roster_refuses_a_patient_it_does_not_contain():
    roster = make_roster(20)
    assert roster.positions_of(roster.pids[[3, 7]]).tolist() == [3, 7]
    with pytest.raises(AssertionError, match="not in the validation roster"):
        roster.positions_of(np.array(["999999"]))


def test_the_development_anchors_are_the_frozen_ones():
    assert tm.EXPECTED_DEV_ROWS == 2968                       # train 2,597 + val 371
    assert tm.EXPECTED_DEV_PATIENTS_WITH_CROPS == 2966
    assert em.EXPECTED_VAL_PATIENTS == 371 and em.EXPECTED_VAL_EVENTS == 54
    em.assert_development_anchors(2968, {"cohort": {"patients_with_crops": 2966}}, QUIET)
    with pytest.raises(AssertionError, match="locked cohort moved"):
        em.assert_development_anchors(2967, None, QUIET)
    with pytest.raises(AssertionError, match="image cohort moved"):
        em.assert_development_anchors(2968, {"cohort": {"patients_with_crops": 2900}}, QUIET)


def test_a_missing_hand_over_index_fails_with_the_command_that_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        em.load_train_arms(tmp_path / "train_arms.json")
    msg = str(exc.value)
    assert "src.train_model" in msg and "--stage stage1" in msg
    assert "sealed test split" in msg


def test_a_hand_over_index_without_arms_is_refused(tmp_path):
    p = tmp_path / "train_arms.json"
    p.write_text(json.dumps({"module": "train_model"}))
    with pytest.raises(AssertionError, match="no 'arms' object"):
        em.load_train_arms(p)


# --------------------------------------------------------------------------- #
# 7b. THE SEALED READ MUST BE ON THE RECORD, AND THE MODELS MUST NOT HAVE MOVED #
#                                                                              #
# forbid_test_split only stops the sealed split being READ. Once the read has  #
# happened it is silent, so retraining an arm afterwards and re-rendering is    #
# caught by nothing. assert_sealed_read_is_recorded is what catches it.        #
# --------------------------------------------------------------------------- #
def _real_cohort_dir():
    cfg = load_config(DEFAULT_CONFIG)
    return cfg.path(cfg["paths"]["cohort_dir"])


def _real_sealed_record() -> dict:
    """The genuine ``test_scoring.json``, so the guard is tested against real spellings."""
    p = _real_cohort_dir() / em.SEALED_READ_RECORD
    if not p.exists():
        pytest.skip("the sealed read has not been performed in this checkout")
    return json.loads(p.read_text())


def _tmp_cohort(tmp_path, record: dict, frozen_hash: str) -> Config:
    """A tmp_path copy of the two artefacts the guard reads, pointed at by cohort_dir."""
    (tmp_path / em.SEALED_READ_RECORD).write_text(json.dumps(record, indent=2))
    (tmp_path / "train_arms.json").write_text(json.dumps(
        {"module": "train_model", "training_contract_hash": frozen_hash, "arms": {}}, indent=2))
    cfg = load_config(DEFAULT_CONFIG)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["cohort_dir"] = str(tmp_path)
    return cfg


def test_the_sealed_read_guard_passes_on_the_real_artefacts():
    coh = _real_cohort_dir()
    record = _real_sealed_record()
    got = em.assert_sealed_read_is_recorded(load_config(DEFAULT_CONFIG))
    assert got, "the guard returned an empty contract hash"
    assert got == record["training_contract_hash"]
    assert got == json.loads((coh / "train_arms.json").read_text())["training_contract_hash"]


def test_the_guard_matches_the_key_names_and_spelling_score_test_actually_writes():
    """Read off the artefact, never guessed: src/score_test.py writes these literals."""
    record = _real_sealed_record()
    assert {"sealed_read", "training_contract_hash"} <= set(record)
    assert em.SEALED_READ_RECORD == "test_scoring.json"
    assert em.SEALED_READ_PERFORMED == "PERFORMED"
    assert record["sealed_read"].startswith(em.SEALED_READ_PERFORMED)


def test_the_sealed_read_guard_refuses_a_contract_hash_that_moved(tmp_path):
    """A model changed AFTER the sealed read: the one failure nothing else here catches."""
    record = _real_sealed_record()
    scored = record["training_contract_hash"]
    with pytest.raises(AssertionError) as exc:
        em.assert_sealed_read_is_recorded(_tmp_cohort(tmp_path, record, "deadbeefdeadbeef"))
    msg = str(exc.value)
    assert scored in msg and "deadbeefdeadbeef" in msg
    assert "never scored out of sample" in msg
    assert "forbid_test_split" in msg, (
        "the message must say why this guard is stronger than the config flag")
    # the identical record passes once the hashes agree again, so the test is about the
    # mutation and not about the fixture being unreadable
    assert em.assert_sealed_read_is_recorded(_tmp_cohort(tmp_path, record, scored)) == scored


def test_the_sealed_read_guard_refuses_a_record_that_does_not_say_performed(tmp_path):
    base = {"module": "score_test", "training_contract_hash": "abc123"}
    for record in ({**base},                                     # sealed_read key absent
                   {**base, "sealed_read": ""},
                   {**base, "sealed_read": "not performed"},
                   {**base, "sealed_read": "SEALED, never loaded"},
                   {**base, "sealed_read": "performed"}):        # wrong spelling, not the token
        with pytest.raises(AssertionError, match="PERFORMED"):
            em.assert_sealed_read_is_recorded(_tmp_cohort(tmp_path, record, "abc123"))
    ok = {**base, "sealed_read": "PERFORMED. This is the single permitted read of the locked "
                                 "test split."}
    assert em.assert_sealed_read_is_recorded(_tmp_cohort(tmp_path, ok, "abc123")) == "abc123"


def test_the_sealed_read_guard_refuses_a_record_with_no_contract_hash_at_all(tmp_path):
    record = {"module": "score_test", "sealed_read": "PERFORMED. one read."}
    with pytest.raises(AssertionError, match="no training_contract_hash"):
        em.assert_sealed_read_is_recorded(_tmp_cohort(tmp_path, record, "abc123"))


def test_a_missing_sealed_read_record_names_the_script_that_produces_it(tmp_path):
    cfg = load_config(DEFAULT_CONFIG)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["cohort_dir"] = str(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        em.assert_sealed_read_is_recorded(cfg)
    msg = str(exc.value)
    assert "src.score_test" in msg and "--confirm-sealed-read" in msg


# =========================================================================== #
# 8. LOADING A TRAINED ARM FROM ITS npz                                        #
# =========================================================================== #
def _write_npz(path, roster, pos, *, arm="m4_fusion", edges=None, time=None):
    n = pos.size
    rng = np.random.default_rng(9)
    haz = np.clip(0.02 + 0.01 * rng.normal(size=(n, tm.N_INTERVALS)), 1e-4, 0.3)
    np.savez(path, hazards=haz,
             hazards_per_seed=np.stack([haz, haz]),
             seeds=np.array([1, 2], dtype=np.int64),
             empi_anon=roster.pids[pos].astype("U"),
             time=(roster.time[pos] if time is None else time),
             event=roster.event[pos].astype(np.int64),
             at_risk=np.ones((n, tm.N_INTERVALS)), target=np.zeros((n, tm.N_INTERVALS)),
             n_scored=np.full(n, tm.N_INTERVALS, dtype=np.int64),
             edges=(tm.EDGES if edges is None else edges),
             arm=np.asarray(arm), mode=np.asarray("fusion"),
             arch=np.asarray("convnext_tiny"), design=np.asarray("m0"),
             views=np.asarray(["frontal"], dtype=object).astype("U"))
    return haz


def test_a_trained_arm_loads_and_the_frozen_recalibration_is_applied(tmp_path):
    roster = make_roster(30, seed=12)
    pos = np.arange(24)
    haz = _write_npz(tmp_path / "val_hazards_m4_fusion.npz", roster, pos)
    recal = {str(float(h)): {"intercept": 0.2, "slope": 1.3} for h in HORIZONS}
    summary = {"hazards_npz": "val_hazards_m4_fusion.npz", "label": "M4", "mode": "fusion",
               "recalibration": recal, "ensemble_val_nll": 0.41, "seeds": [1, 2]}
    sc = em.trained_arm_scores("m4_fusion", summary, tmp_path, roster, HORIZONS, QUIET)
    assert sc is not None and sc.n_patients == 24
    assert sc.present[:24].all() and not sc.present[24:].any()
    expected = tm.apply_recalibration(tm.risk_at_horizon(haz, 1825.0), recal["1825.0"])
    assert sc.risk[1825][pos] == pytest.approx(expected)
    assert np.isnan(sc.risk[1825][24:]).all(), "an unscored patient must carry NaN, not a value"
    assert sc.rank[pos] == pytest.approx(tm.risk_score(haz))


def test_an_arm_without_a_recalibration_block_is_left_uncalibrated(tmp_path):
    roster = make_roster(20, seed=13)
    pos = np.arange(20)
    haz = _write_npz(tmp_path / "val_hazards_a.npz", roster, pos, arm="a")
    sc = em.trained_arm_scores("a", {"hazards_npz": "val_hazards_a.npz"}, tmp_path, roster,
                               HORIZONS, QUIET)
    assert sc.risk[730][pos] == pytest.approx(tm.risk_at_horizon(haz, 730.0))


def test_an_arm_whose_npz_is_absent_is_skipped_not_fatal(tmp_path):
    roster = make_roster(20)
    assert em.trained_arm_scores("ghost", {"hazards_npz": "nope.npz"}, tmp_path, roster,
                                 HORIZONS, QUIET) is None


def test_an_npz_written_on_a_different_grid_is_refused(tmp_path):
    roster = make_roster(20, seed=14)
    _write_npz(tmp_path / "val_hazards_a.npz", roster, np.arange(20), arm="a",
               edges=np.linspace(0.0, 1000.0, tm.N_INTERVALS + 1))
    with pytest.raises(AssertionError, match="different interval grid"):
        em.trained_arm_scores("a", {"hazards_npz": "val_hazards_a.npz"}, tmp_path, roster,
                              HORIZONS, QUIET)


def test_an_npz_whose_follow_up_disagrees_with_the_cohort_is_refused(tmp_path):
    roster = make_roster(20, seed=15)
    _write_npz(tmp_path / "val_hazards_a.npz", roster, np.arange(20), arm="a",
               time=roster.time + 5.0)
    with pytest.raises(AssertionError, match="disagree with the locked cohort"):
        em.trained_arm_scores("a", {"hazards_npz": "val_hazards_a.npz"}, tmp_path, roster,
                              HORIZONS, QUIET)


def test_an_npz_indexed_under_the_wrong_arm_name_is_refused(tmp_path):
    roster = make_roster(20, seed=16)
    _write_npz(tmp_path / "val_hazards_a.npz", roster, np.arange(20), arm="somebody_else")
    with pytest.raises(AssertionError, match="is indexed under"):
        em.trained_arm_scores("a", {"hazards_npz": "val_hazards_a.npz"}, tmp_path, roster,
                              HORIZONS, QUIET)


def test_an_npz_carrying_a_patient_outside_the_validation_roster_is_refused(tmp_path):
    roster = make_roster(20, seed=17)
    path = tmp_path / "val_hazards_a.npz"
    _write_npz(path, roster, np.arange(20), arm="a")
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files}
    # distinct ids, none of them on the roster: stands in for a sealed-split leak
    payload["empi_anon"] = np.array([f"{900000 + i}" for i in range(20)], dtype="<U8")
    np.savez(path, **payload)
    with pytest.raises(AssertionError, match="not in the validation roster"):
        em.trained_arm_scores("a", {"hazards_npz": "val_hazards_a.npz"}, tmp_path, roster,
                              HORIZONS, QUIET)


def test_an_npz_carrying_a_duplicated_patient_is_refused(tmp_path):
    roster = make_roster(20, seed=18)
    path = tmp_path / "val_hazards_a.npz"
    _write_npz(path, roster, np.arange(20), arm="a")
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files}
    payload["empi_anon"] = np.array([roster.pids[0]] * 20, dtype="<U8")
    np.savez(path, **payload)
    with pytest.raises(AssertionError, match="duplicated empi_anon"):
        em.trained_arm_scores("a", {"hazards_npz": "val_hazards_a.npz"}, tmp_path, roster,
                              HORIZONS, QUIET)


# =========================================================================== #
# 9. ONE PATH RULE FOR BOTH SPLITS                                             #
#                                                                              #
# split_path() is imported by src/manuscript_figures.py and                    #
# src/make_manuscript.py. If it and main() ever disagree, a reader silently     #
# resolves a file the writer never wrote, so both halves are pinned here.      #
# =========================================================================== #
SPLIT_PATH_KEYS = ("metrics_csv", "comparisons_csv", "subgroups_csv", "convergence_csv",
                   "results_json")

# Every model_eval value that names a FILE, listed by hand rather than derived, so the
# test below is an independent statement of which keys are paths instead of a restatement
# of split_path's own rule. net_benefit_csv is the sixth: split_path already resolves it,
# the decision-curve module that writes it lands next.
MODEL_EVAL_PATH_KEYS = SPLIT_PATH_KEYS + ("net_benefit_csv",)


def _declared_split_paths(cfg: Config) -> list[str]:
    """Every model_eval value that names a ``val_*`` artefact this module writes.

    Derived from the config rather than listed, so a sixth output added beside the five
    (decision-curve analysis is next) is covered by these tests the day it lands.
    """
    return sorted(k for k, v in cfg["model_eval"].items()
                  if isinstance(v, str) and "val_" in v)


def test_split_path_covers_every_val_output_model_eval_declares():
    cfg = load_config(DEFAULT_CONFIG)
    assert set(SPLIT_PATH_KEYS) <= set(_declared_split_paths(cfg))


def test_split_path_on_val_is_todays_configured_path_unchanged():
    """The development path must not move by one byte; the val artefacts are reproducible."""
    cfg = load_config(DEFAULT_CONFIG)
    for key in _declared_split_paths(cfg):
        assert em.split_path(cfg, key, "val") == cfg.path(cfg["model_eval"][key])


def test_split_path_on_test_is_the_test_prefixed_sibling_beside_it():
    cfg = load_config(DEFAULT_CONFIG)
    for key in _declared_split_paths(cfg):
        v = em.split_path(cfg, key, "val")
        t = em.split_path(cfg, key, SEALED_SPLIT)
        assert v.name.startswith("val_"), f"{key} is not a val_* artefact: {v.name}"
        assert t.name == "test_" + v.name[len("val_"):]
        assert t.parent == v.parent, "the sealed output sits BESIDE the validation one"
        assert t != v, "a sealed read must never overwrite a validation artefact"


def test_split_path_rewrites_one_val_prefix_in_the_basename_and_nothing_else():
    """The bound is the writer's own: replace(..., 1), on p.name only."""
    cfg = Config({"model_eval": {"k": "outputs/val_dir/val_metrics_val_extra.csv"}})
    p = em.split_path(cfg, "k", SEALED_SPLIT)
    assert p.name == "test_metrics_val_extra.csv", "exactly one substitution"
    assert p.parent.name == "val_dir", "a directory named val_* is left alone"


def test_split_path_refuses_a_split_it_does_not_know_and_a_key_that_is_not_there():
    cfg = load_config(DEFAULT_CONFIG)
    with pytest.raises(AssertionError, match="split must be"):
        em.split_path(cfg, "metrics_csv", "train")
    with pytest.raises(AssertionError, match="no path key"):
        em.split_path(cfg, "not_a_key", "val")


def test_split_path_refuses_every_model_eval_key_that_names_a_setting_not_a_path():
    """A key that IS in model_eval but is not a filename must fail as loudly as an unknown one.

    ``model_eval`` holds six paths among a dozen settings and two of them differ by one
    token: ``net_benefit`` sits directly beside ``net_benefit_csv``. ``Config.path``
    ``str()``s whatever it is handed, so before the value check
    ``split_path(cfg, "net_benefit", "test")`` returned
    ``<repo>/{'threshold_min_pct': 1, ...}`` - a perfectly ordinary-looking Path that no
    caller would question. ``fdr_method`` is the nastier case: a plain string, so a
    type test alone would have let it through.

    The settings are derived by subtracting the pinned path keys, so a seventh key added
    to the block has to be classified as one or the other before this passes.
    """
    cfg = load_config(DEFAULT_CONFIG)
    me = cfg["model_eval"]
    assert set(MODEL_EVAL_PATH_KEYS) <= set(me), "a pinned path key left model_eval"
    settings = sorted(set(me) - set(MODEL_EVAL_PATH_KEYS))
    assert {"net_benefit", "horizons_days", "primary_contrast", "forbid_test_split",
            "fdr_method"} <= set(settings), "the named non-path keys must still be tested"
    for key in settings:
        for split in ("val", SEALED_SPLIT):
            with pytest.raises(AssertionError, match="is not a path"):
                em.split_path(cfg, key, split)
    for key in MODEL_EVAL_PATH_KEYS:            # and every real path still resolves
        assert em.split_path(cfg, key, "val").name == me[key].rsplit("/", 1)[-1]


def test_net_benefit_csv_is_a_val_prefixed_path_that_split_path_rewrites():
    """The sixth output. Nothing writes it yet, so nothing else would catch a bad value.

    The decision-curve module must write ``test_net_benefit.csv`` beside
    ``val_net_benefit.csv`` under the same single rewrite as the other five, and the
    manuscript and figure modules must look for it there. Pinning it before the writer
    exists means the writer inherits a path contract rather than inventing one.
    """
    cfg = load_config(DEFAULT_CONFIG)
    configured = cfg["model_eval"]["net_benefit_csv"]
    assert configured == "outputs/tables/val_net_benefit.csv"
    v = em.split_path(cfg, "net_benefit_csv", "val")
    t = em.split_path(cfg, "net_benefit_csv", SEALED_SPLIT)
    assert v == cfg.path(configured)
    assert v.name == "val_net_benefit.csv" and t.name == "test_net_benefit.csv"
    assert t.parent == v.parent == cfg.path(cfg["paths"]["tables_dir"])


@pytest.mark.parametrize("split", ["val", "test"])
def test_split_path_resolves_the_files_the_writer_actually_wrote(split):
    """Byte-for-byte agreement with main(): the paths it derives are the files on disk."""
    cfg = load_config(DEFAULT_CONFIG)
    paths = {k: em.split_path(cfg, k, split) for k in SPLIT_PATH_KEYS}
    if not any(p.exists() for p in paths.values()):
        pytest.skip(f"src.eval_models --split {split} has not been run in this checkout")
    absent = sorted(k for k, p in paths.items() if not p.exists())
    assert not absent, (
        f"split_path resolves {absent} to files src.eval_models --split {split} did not "
        f"write: {[str(paths[k]) for k in absent]}")


def test_main_derives_its_outputs_from_split_path_and_keeps_no_second_copy():
    src = inspect.getsource(em.main)
    assert "split_path(" in src, "main() must route its output paths through split_path"
    assert 'replace("val_"' not in src and "replace('val_'" not in src, (
        "main() carries its own copy of the sealed-read path rewrite; there must be exactly "
        "one, in split_path(), because two downstream modules import it to find these files")


# =========================================================================== #
# 9c. CONFIG VALUES AND THE CODE THAT READS THEM STILL AGREE                   #
#                                                                              #
# This section once said that nothing read the six section_target_words        #
# budgets or model_image.local.test_shard_dir. That is no longer true and the  #
# claim is corrected here rather than left to mislead: the budgets are gated   #
# by test_every_section_of_the_reported_split_is_within_its_budget in          #
# tests/test_make_manuscript.py, and test_shard_dir is read through            #
# manuscript_figures.SHARD_DIR_KEY to count the sealed split's crops for       #
# figure 1. (The net_benefit block joined them when                            #
# src.eval_models.net_benefit_settings started reading it; section 12 below    #
# pins it against the estimator.) src/score_test.py is the one real holdout:   #
# it still requires --shard-dir and ignores the configured value.              #
#                                                                              #
# A config value can drift out of agreement with the code it was written       #
# against, and a reader is then wrong on its very first run, silently, because #
# the config looks deliberate. These tests pin the agreements that are not     #
# checked anywhere else. They belong here rather than beside the manuscript    #
# tests because this file already owns "the config and the code still agree".  #
# =========================================================================== #
def test_section_target_words_keys_are_exactly_the_imrad_headings():
    """Six keys, spelled and ordered as make_manuscript.IMRAD_HEADINGS spells them.

    Nothing looks these budgets up yet, so nothing notices if a heading is renamed: the
    per-section target would simply never match a section and the budget would become
    decorative while still reading as a constraint. Order is pinned as well as membership,
    because the reader this is written for will iterate the headings and index the dict.
    """
    mm = _manuscript_module()
    got = tuple(load_config(DEFAULT_CONFIG)["manuscript"]["section_target_words"])
    assert got == mm.IMRAD_HEADINGS, (
        f"section_target_words keys {got} have drifted from IMRAD_HEADINGS "
        f"{mm.IMRAD_HEADINGS}; the per-section budget would silently stop applying")


def test_the_combined_word_target_is_the_sum_of_the_methods_and_results_sections():
    """verify() checks only ``target_words``; the per-section numbers must not contradict it.

    ``target_words`` is Methods plus Results and is the ONE word budget any code enforces
    (src/make_manuscript.py:1996). ``section_target_words`` restates the same two numbers
    per section and ``target_words_methods`` / ``target_words_results`` restate them again
    for the verify() printout. Three statements of one budget is two chances to disagree,
    and a Results cut sized against the wrong one lands outside the tolerance the render
    actually checks.
    """
    ms = load_config(DEFAULT_CONFIG)["manuscript"]
    sec = ms["section_target_words"]
    assert int(ms["target_words"]) == int(sec["Methods"]) + int(sec["Results"])
    assert int(ms["target_words_methods"]) == int(sec["Methods"])
    assert int(ms["target_words_results"]) == int(sec["Results"])


def test_the_net_benefit_horizon_is_the_primary_contrast_horizon():
    """``horizon_days: *horizon_5y`` is a YAML alias, and an alias can be unlinked in one edit.

    Replacing the alias with a literal 1825 loses the link and nothing else; replacing it
    with a different number is the failure that matters, because the decision curve would
    then be drawn at a horizon the primary estimand is not evaluated at, and the manuscript
    would print the two beside each other as though they matched. The RESOLVED values are
    compared rather than the YAML text, so this states the requirement, not the mechanism.
    """
    me = load_config(DEFAULT_CONFIG)["model_eval"]
    primary = int(me["primary_contrast"]["horizon_days"])
    assert int(me["net_benefit"]["horizon_days"]) == primary, (
        "the decision-curve horizon has come unlinked from the primary contrast's")
    assert primary == 1825, "the 5-year horizon is protocol section 18's primary estimand"
    assert primary in [int(h) for h in me["horizons_days"]]


# =========================================================================== #
# 9b. OUTPUTS ARE AGGREGATE ONLY AND DETERMINISTIC                             #
# =========================================================================== #
def test_an_identifier_column_never_reaches_outputs():
    df = pd.DataFrame({"empi_anon": ["1"], "auc_365": [0.6]})
    with pytest.raises(AssertionError, match="aggregate only"):
        em.assert_aggregate_only(df, ["1"], "val_metrics.csv")
    for bad in ("StudyInstanceUID_anon", "patient_id", "mrn"):
        with pytest.raises(AssertionError, match="aggregate only"):
            em.assert_aggregate_only(pd.DataFrame({bad: ["x"]}), [], "t")


def test_an_identifier_value_never_reaches_outputs_even_in_a_harmless_column():
    df = pd.DataFrame({"label": ["M4 fusion", "100007"], "auc_365": [0.6, 0.7]})
    with pytest.raises(AssertionError, match="patient identifier"):
        em.assert_aggregate_only(df, ["100007", "100008"], "val_metrics.csv")
    em.assert_aggregate_only(df, ["999999"], "val_metrics.csv")       # no overlap, no raise


def test_write_table_refuses_a_schema_that_drifted(tmp_path):
    df = pd.DataFrame({"b": [1.0], "a": ["x"]})
    with pytest.raises(AssertionError, match="column order drifted"):
        em.write_table(tmp_path / "t.csv", df, ["a", "b"], [], "t")


def test_the_written_csv_carries_no_timestamp_and_repeats_byte_for_byte(tmp_path):
    roster = make_roster(60, seed=31)
    arms = {n: make_arm(n, roster, signal=0.6 + 0.3 * i, seed=50 + i, drop=3 * i)
            for i, n in enumerate(["m0", "m4_fusion"])}
    order = ["m0", "m4_fusion"]
    outs = []
    for run in range(2):
        engine = make_engine(roster, n_boot=40, seed=20250720)
        met = em.build_metrics(arms, engine, HORIZONS, order, QUIET)
        p = tmp_path / f"metrics_{run}.csv"
        em.write_table(p, met, em.metrics_columns(HORIZONS), roster.pids, "val_metrics.csv")
        outs.append(p.read_bytes())
    assert outs[0] == outs[1], "two runs of the same inputs must be byte-identical"
    text = outs[0].decode()
    assert "20" + "26-" not in text and "Z\n" not in text, "no timestamp belongs in a CSV"


def test_two_runs_give_identical_frames_for_every_output():
    roster = make_roster(60, seed=32)
    names = ["m0", "m4_fusion", "m3_image"]
    arms = {n: make_arm(n, roster, signal=0.5 + 0.4 * i, seed=60 + i, drop=2 * i)
            for i, n in enumerate(names)}
    cfg_full = load_config(DEFAULT_CONFIG)
    cfg = eval_config({"model": "m4_fusion", "reference": "m0", "horizon_days": 1825},
                      {"modality": [["m3_image", "m0"], ["m4_fusion", "m3_image"]]})
    frames = []
    for _ in range(2):
        engine = make_engine(roster, n_boot=40, seed=20250720)
        frames.append((em.build_metrics(arms, engine, HORIZONS, names, QUIET),
                       em.build_comparisons(cfg, arms, engine, QUIET),
                       em.build_subgroups(cfg_full, roster, arms["m4_fusion"], engine,
                                          1825, QUIET)))
    for a, b in zip(*frames):
        pd.testing.assert_frame_equal(a, b)


def test_rounding_is_fixed_so_a_float_cannot_wobble_between_runs():
    df = pd.DataFrame({"x": [1.0 / 3.0], "n": [3], "s": ["a"]})
    out = em.round_floats(df)
    assert out["x"].iloc[0] == pytest.approx(round(1.0 / 3.0, em.ROUND_DECIMALS))
    assert out["n"].dtype == df["n"].dtype and out["s"].iloc[0] == "a"


def test_val_metrics_rows_follow_the_declared_ladder_order():
    roster = make_roster(40, seed=33)
    arms = {n: make_arm(n, roster, seed=70 + i) for i, n in enumerate(["m4_fusion", "m0"])}
    engine = make_engine(roster, n_boot=15)
    order = ["m0", "m1", "m0d_clinical", "m4_fusion"]      # m1 and m0d are not scored
    met = em.build_metrics(arms, engine, HORIZONS, order, QUIET)
    assert met["arm"].tolist() == ["m0", "m4_fusion"], "absent arms are omitted, not reordered"
    assert list(met.columns) == em.metrics_columns(HORIZONS)
    assert met["n_patients"].dtype.kind in "iu" and met["n_events"].dtype.kind in "iu"


def test_an_empty_ladder_still_writes_the_schema(tmp_path):
    roster = make_roster(20)
    engine = make_engine(roster, n_boot=5)
    met = em.build_metrics({}, engine, HORIZONS, ["m0"], QUIET)
    assert met.empty and list(met.columns) == em.metrics_columns(HORIZONS)
    em.write_table(tmp_path / "m.csv", met, em.metrics_columns(HORIZONS), roster.pids, "m")
    assert (tmp_path / "m.csv").read_text().strip() == ",".join(em.metrics_columns(HORIZONS))


# =========================================================================== #
# 10. THE REPORT AND THE RESULTS HEADER                                        #
# =========================================================================== #
def test_the_report_renders_without_a_single_estimable_contrast():
    roster = make_roster(40, seed=34)
    arms = {"m0": make_arm("m0", roster)}
    engine = make_engine(roster, n_boot=15)
    met = em.build_metrics(arms, engine, HORIZONS, ["m0"], QUIET)
    cmp_ = pd.DataFrame(columns=em.COMPARISON_COLUMNS)
    sub = pd.DataFrame(columns=em.SUBGROUP_COLUMNS)
    lines = em.build_report(met, cmp_, sub, HORIZONS, ["m4_fusion"], 2000)
    text = "\n".join(lines)
    assert "m4_fusion" in text and "2000 shared bootstrap replicates" in text


def test_the_run_log_heading_names_the_split_that_was_actually_evaluated():
    """The fourth "the test split was never read" claim, and the last one in this module.

    Three others were removed from ``test_results.json``; this one survived in
    ``outputs/logs/run.log``, where a sealed run printed "test split sealed" on the very
    line heading the test-split numbers. The log is not an artefact a reader cites, but a
    module that contradicts itself in one place teaches a reader to discount the statements
    that do matter. Only the heading changes: the tables below it are split-independent.
    """
    roster = make_roster(40, seed=35)
    arms = {"m0": make_arm("m0", roster)}
    engine = make_engine(roster, n_boot=15)
    met = em.build_metrics(arms, engine, HORIZONS, ["m0"], QUIET)
    cmp_ = pd.DataFrame(columns=em.COMPARISON_COLUMNS)
    sub = pd.DataFrame(columns=em.SUBGROUP_COLUMNS)

    def report(**kw):
        return em.build_report(met, cmp_, sub, HORIZONS, [], 2000, **kw)

    val, sealed = report(), report(split=SEALED_SPLIT)
    assert val == report(split="val"), "the default must stay the validation behaviour"
    assert val[0] == ("## Validation evaluation (2000 shared bootstrap replicates, "
                      "test split sealed)")
    assert sealed[0] == ("## Test evaluation (2000 shared bootstrap replicates, THE single "
                         "permitted sealed read)")
    assert "test split sealed" not in sealed[0] and "Validation" not in sealed[0]
    assert val[1:] == sealed[1:], "only the heading may depend on the split"
    with pytest.raises(AssertionError, match="no report heading"):
        report(split="train")


def _results_header(split: str, missing=("r1_densenet_frontal",)) -> dict:
    cfg = load_config(DEFAULT_CONFIG)
    roster = make_roster(30)
    arms = {"m0": make_arm("m0", roster),
            "m4_fusion": make_arm("m4_fusion", roster)}
    arms["m4_fusion"].recalibration = {"1825.0": {"intercept": 0.0, "slope": 1.0}}
    return em.build_results_json(cfg, roster, arms, list(missing), HORIZONS,
                                 2000, 20250720, "m4_fusion", {}, split=split)


def test_the_validation_results_header_is_unchanged_and_still_says_never_loaded():
    """On the development split nothing about the sealed set moved, so nor may this text."""
    js = _results_header("val")
    assert "SEALED" in js["test_split"] and "never" in js["test_split"].lower()
    assert js["test_split"] == em.TEST_SPLIT_STATEMENT["val"]
    assert set(js["cohort"]) == {"n_val", "n_val_events"}
    assert js["bootstrap"] == {"n": 2000, "seed": 20250720,
                               "note": "one patient-level resample per replicate, "
                                       "shared across arms"}
    assert js["arms_not_scored"] == ["r1_densenet_frontal"]
    assert js["recalibration"]["m0"]["per_horizon"] is False
    assert js["recalibration"]["m4_fusion"]["per_horizon"] is True
    assert json.loads(json.dumps(js, default=str))          # must be serialisable as written


def test_the_sealed_results_header_no_longer_claims_the_split_was_never_loaded():
    """test_results.json said "SEALED, never loaded" about a read that had been performed."""
    js = _results_header(SEALED_SPLIT)
    assert js["test_split"] == em.TEST_SPLIT_STATEMENT[SEALED_SPLIT]
    assert "never" not in js["test_split"].lower()
    assert "sealed, never loaded" not in js["test_split"].lower()
    assert js["test_split"].startswith("READ")
    assert "single permitted read" in js["test_split"]
    assert json.loads(json.dumps(js, default=str))


def test_a_sealed_cohort_is_not_labelled_n_val():
    """741 test patients were written as ``n_val: 741``. A test cohort is not a val cohort."""
    cfg = load_config(DEFAULT_CONFIG)
    roster = make_roster(30)
    js = em.build_results_json(cfg, roster, {"m0": make_arm("m0", roster)}, [], HORIZONS,
                               2000, 20250720, "m0", {}, split=SEALED_SPLIT)
    assert set(js["cohort"]) == {"n_test", "n_test_events"}
    assert js["cohort"]["n_test"] == len(roster)
    assert js["cohort"]["n_test_events"] == int(roster.event.sum())


def test_the_split_statement_is_keyed_by_split_with_no_split_free_default():
    """A single string is how the sealed run came to announce that it had never happened."""
    assert isinstance(em.TEST_SPLIT_STATEMENT, dict)
    assert set(em.TEST_SPLIT_STATEMENT) == {"val", SEALED_SPLIT}
    assert em.TEST_SPLIT_STATEMENT["val"].startswith("SEALED, never loaded.")
    with pytest.raises(AssertionError, match="no run-header statement"):
        roster = make_roster(10)
        em.build_results_json(load_config(DEFAULT_CONFIG), roster, {}, [], HORIZONS,
                              2000, 1, None, {}, split="train")


# --------------------------------------------------------------------------- #
# The convergence gate (D28). An arm that never fitted a model is not a        #
# comparator, and a constant predictor's AUROC of exactly 0.500 must not be    #
# reported as "no better than chance".                                         #
# --------------------------------------------------------------------------- #
def _history(arm: str, train_first: float, train_last: float,
             val_last: float, val_min: float, n_seeds: int = 5) -> pd.DataFrame:
    """Minimal train_history.csv shape: first/last training NLL and the val minimum."""
    rows = []
    for seed in range(n_seeds):
        rows += [dict(arm=arm, seed=seed, epoch=0, train_nll=train_first, val_nll=val_min + 0.01),
                 dict(arm=arm, seed=seed, epoch=1, train_nll=(train_first + train_last) / 2,
                      val_nll=val_min),
                 dict(arm=arm, seed=seed, epoch=2, train_nll=train_last, val_nll=val_last)]
    return pd.DataFrame(rows)


def _cfg_with_history(tmp_path, frame: pd.DataFrame) -> Config:
    cfg = load_config(DEFAULT_CONFIG)
    p = tmp_path / "train_history.csv"
    frame.to_csv(p, index=False)
    cfg["model_image"]["local"]["history_csv"] = str(p)
    return cfg


def test_an_arm_whose_training_loss_never_falls_is_flagged_did_not_converge(tmp_path):
    # m2_frontal really did move by 9.6e-06 over five seeds; anything under the floor is
    # a constant predictor, whose IPCW AUROC is exactly 0.5 with a zero-width interval.
    cfg = _cfg_with_history(tmp_path, _history("flat", 0.60997, 0.60996, 0.6367, 0.6329))
    df = em.convergence_diagnostics(cfg, QUIET)
    assert df.loc[0, "status"] == em.STATUS_NO_CONVERGE
    assert df.loc[0, "train_nll_drop"] < 0.001
    assert "constant" in df.loc[0, "reason"]


def test_an_arm_that_memorises_train_while_val_rises_is_flagged_severe_overfit(tmp_path):
    # r1_densenet_frontal: train 0.610 -> 0.382 while val ends 0.239 above its own minimum.
    cfg = _cfg_with_history(tmp_path, _history("memo", 0.6099, 0.3824, 0.8311, 0.5773))
    df = em.convergence_diagnostics(cfg, QUIET)
    assert df.loc[0, "status"] == em.STATUS_OVERFIT
    assert df.loc[0, "val_overfit_gap"] > 0.10
    assert "optimistic" in df.loc[0, "reason"]


def test_a_healthy_arm_passes_the_gate(tmp_path):
    cfg = _cfg_with_history(tmp_path, _history("good", 0.6084, 0.5896, 0.6291, 0.6235))
    df = em.convergence_diagnostics(cfg, QUIET)
    assert df.loc[0, "status"] == em.STATUS_OK
    assert df.loc[0, "reason"] == ""


def test_a_missing_history_file_disables_the_gate_rather_than_crashing(tmp_path):
    cfg = load_config(DEFAULT_CONFIG)
    cfg["model_image"]["local"]["history_csv"] = str(tmp_path / "absent.csv")
    df = em.convergence_diagnostics(cfg, QUIET)
    assert df.empty and list(df.columns) == em.CONVERGENCE_COLUMNS


def test_suppression_blanks_the_estimate_and_states_the_reason():
    conv = pd.DataFrame([dict(arm="bad", n_seeds=5, train_nll_drop=1e-6, val_overfit_gap=0.0,
                              status=em.STATUS_NO_CONVERGE, reason="did not fit")],
                        columns=em.CONVERGENCE_COLUMNS)
    rows = [dict(family="views", model="good", reference="bad", metric="auc", horizon_days=1825,
                 n_paired=370, estimate_model=0.69, estimate_reference=0.50, difference=0.19,
                 ci_lo=0.09, ci_hi=0.28, p_two_sided=0.0005, p_adjusted=float("nan"),
                 fdr_method="", is_primary=False, note="")]
    out = em.suppress_unfit_contrasts(rows, conv, QUIET)[0]
    for k in ("difference", "ci_lo", "ci_hi", "p_two_sided"):
        assert np.isnan(out[k]), k
    assert "SUPPRESSED" in out["note"] and "did not fit" in out["note"]
    # the row survives and still names the comparison, exactly like a suppressed subgroup
    assert out["model"] == "good" and out["reference"] == "bad" and out["n_paired"] == 370


def test_a_suppressed_contrast_does_not_consume_its_family_multiplicity():
    """The whole point of suppressing BEFORE Benjamini-Hochberg.

    Two real contrasts plus one training failure must be adjusted as m = 2, not m = 3:
    a failed arm is not a hypothesis anyone tested.
    """
    conv = pd.DataFrame([dict(arm="bad", n_seeds=5, train_nll_drop=1e-6, val_overfit_gap=0.0,
                              status=em.STATUS_NO_CONVERGE, reason="did not fit")],
                        columns=em.CONVERGENCE_COLUMNS)

    def row(model, ref, p):
        return dict(family="f", model=model, reference=ref, metric="auc", horizon_days=1825,
                    n_paired=370, estimate_model=0.6, estimate_reference=0.5, difference=0.1,
                    ci_lo=0.0, ci_hi=0.2, p_two_sided=p, p_adjusted=float("nan"),
                    fdr_method="", is_primary=False, note="")

    rows = [row("a", "m0", 0.01), row("b", "m0", 0.04), row("c", "bad", 0.0005)]
    adj = em.benjamini_hochberg(
        [r["p_two_sided"] for r in em.suppress_unfit_contrasts(rows, conv, QUIET)])
    assert np.isnan(adj[2])
    # m = 2: 0.01 * 2/1 = 0.02 and 0.04 * 2/2 = 0.04. Pooling at m = 3 would give 0.03/0.06.
    assert adj[0] == pytest.approx(0.02, abs=1e-12)
    assert adj[1] == pytest.approx(0.04, abs=1e-12)


# =========================================================================== #
# 12. DECISION-CURVE ANALYSIS - THE CENSORING-AWARE NET-BENEFIT ESTIMATOR      #
#                                                                              #
# Every number in this section was worked on paper first. The estimator is     #
#                                                                              #
#     w(p_t)  = p_t / (1 - p_t)                                                #
#     NB(p_t) = F_A(h) * (n_A/n) - (1 - F_A(h)) * (n_A/n) * w(p_t)             #
#                                                                              #
# with F_A the Kaplan-Meier cumulative incidence at the horizon INSIDE the     #
# flagged set A = {i : risk_i >= p_t}. Treat-all is the same call with         #
# risk=None. The four-patient fixtures below are small enough that F_A is a    #
# product of two or three fractions, so a reviewer can redo the arithmetic     #
# without running anything.                                                    #
# =========================================================================== #
NB_HORIZON = 1825.0


def _dca_four():
    """Four patients, one horizon, follow-up that reaches past it.

    times/events   P1(100, event)  P2(1200, event)  P3(1500, censored)  P4(1900, censored)
    risk           P1 .90          P2 .40           P3 .30              P4 .50

    Kaplan-Meier inside each flagged set, at h = 1825:

    * p_t = 0.35 flags {P1, P2, P4} (P3's 0.30 falls short). Deaths at 100 (3 at risk) and
      1200 (2 at risk); P4's 1900 is past the horizon, so S = (2/3)(1/2) = 1/3 and
      F_A = 2/3. n_A/n = 3/4, w = .35/.65 = 7/13.
          TP = (2/3)(3/4) = 1/2, FP = (1/3)(3/4) = 1/4, TP + FP = 3/4 = n_A/n
          NB = 1/2 - (1/4)(7/13) = 26/52 - 7/52 = **19/52 = 0.365384615...**
    * p_t = 0.45 flags {P1, P4}. One death at 100 out of 2 at risk, so F_A = 1/2,
      n_A/n = 1/2, w = .45/.55 = 9/11.
          NB = 1/4 - (1/4)(9/11) = (1/4)(2/11) = **1/22 = 0.0454545...**
    * p_t = 0.50 flags the same pair with w = 1, so **NB = 1/4 - 1/4 = 0 exactly**.
    * TREAT-ALL at p_t = 0.20: deaths at 100 (4 at risk) and 1200 (3 at risk), P3 censored
      at 1500 leaves the product alone, P4 is past the horizon, so S = (3/4)(2/3) = 1/2 and
      F_E = 1/2. w = .2/.8 = 1/4.
          NB = 1/2 - (1/2)(1/4) = **0.375**
      and at p_t = F_E = 0.50, w = 1, so treat-all is **0 exactly** - the zero crossing sits
      at the prevalence, which is the whole reason the censoring correction matters.
    """
    time = np.array([100.0, 1200.0, 1500.0, 1900.0])
    event = np.array([1, 1, 0, 0])
    risk = np.array([0.90, 0.40, 0.30, 0.50])
    return time, event, risk


def _dca_four_no_events_above():
    """The negative floor. Risks are ordered AGAINST the outcome, so the flagged set is
    all non-events.

    times/events  P1(100, event)  P2(200, event)  P3(1900, censored)  P4(1900, censored)
    risk          P1 .10          P2 .20          P3 .60              P4 .70

    p_t = 0.50 flags {P3, P4}, both censored past the horizon, so no factor enters the
    product: S = 1, F_A = 0. n_A/n = 1/2, w = 1.
        TP = 0, FP = 1/2, NB = 0 - (1/2)(1) = **-0.5 = -(n_A/n) * w(p_t)** exactly.
    """
    time = np.array([100.0, 200.0, 1900.0, 1900.0])
    event = np.array([1, 1, 0, 0])
    risk = np.array([0.10, 0.20, 0.60, 0.70])
    return time, event, risk


def _dca_four_short_follow_up():
    """Nobody reaches the horizon. Same risks and same KM factors as :func:`_dca_four`'s
    0.35 flagged set, but the last follow-up day is 900.

    times/events  P1(100, event)  P2(200, event)  P3(800, censored)  P4(900, censored)
    risk          P1 .90          P2 .40          P3 .30             P4 .50

    p_t = 0.35 flags {P1, P2, P4}: deaths at 100 (3 at risk) and 200 (2 at risk), P4
    censored at 900 adds nothing, so S = 1/3 CARRIED FORWARD to day 1825 and F_A = 2/3.
    The arithmetic is therefore identical - **NB = 19/52** - and the only difference is
    that km_last_obs_day records 900.
    """
    time = np.array([100.0, 200.0, 800.0, 900.0])
    event = np.array([1, 1, 0, 0])
    risk = np.array([0.90, 0.40, 0.30, 0.50])
    return time, event, risk


def _prevalence_cohort(n: int, n_events: int):
    """A cohort whose 5-year Kaplan-Meier risk is n_events/n to within 1e-12.

    Distinct event days 1..n_events, then administrative censoring past the horizon. With
    no censoring before the last event the product-limit estimate telescopes:
    S = (n-1)/n * (n-2)/(n-1) * ... = (n - n_events)/n.
    """
    time = np.arange(1.0, n + 1.0)
    time[n_events:] = 1826.0 + np.arange(n - n_events)
    event = np.zeros(n, dtype=int)
    event[:n_events] = 1
    return time, event


def _censored_cohort(n: int, seed: int):
    """A realistic screening cohort: a spread of predicted risks, exponential event times
    matched to them, and censoring drawn INDEPENDENTLY OF RISK - the assumption IPCW needs.
    """
    rng = np.random.default_rng(seed)
    risk = rng.beta(2.0, 8.0, size=n) * 0.9 + 0.02
    lam = -np.log(1.0 - np.clip(risk, 1e-6, 0.99)) / NB_HORIZON
    t_event = rng.exponential(1.0 / lam)
    t_cens = np.minimum(rng.exponential(1500.0, size=n), 1826.0 * 1.6)
    return np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int), risk


# --------------------------------------------------------------------------- #
# 12a. The grid: an integer-percent basis, and a ceiling config cannot raise.  #
# --------------------------------------------------------------------------- #
def test_the_threshold_grid_is_the_integer_percent_basis_exactly():
    """Every entry is the nearest double to k/100, and 0.07 is 0.07.

    A float ``np.arange(0.01, 0.36, 0.01)`` accumulates its step, so at least one entry is
    not the nearest double to k/100 - on this build it is the last one. Since the flagged
    set is ``risk >= p_t``, an entry one ulp high flags a different set of patients, and
    WHICH entry drifts is a property of the numpy build rather than of the analysis.
    """
    grid = em.threshold_grid(1, 35)
    assert grid.size == 35
    assert grid[6] == 0.07, "the seventh threshold must be exactly the double 0.07"
    assert grid[0] == 0.01 and grid[-1] == 0.35
    for k in range(1, 36):
        assert grid[k - 1] == k / 100.0, f"threshold {k}% is not the nearest double to k/100"
    drifted = em.threshold_grid(1, 35) != np.arange(0.01, 0.36, 0.01)
    assert drifted.any(), (
        "a float np.arange happens to agree with the integer basis everywhere on this numpy "
        "build; the integer basis is what makes that a fact rather than luck")


def test_the_threshold_ceiling_is_frozen_in_the_module_and_config_cannot_exceed_it():
    assert em.NB_THRESHOLD_CEILING_PCT == 50
    configured = int(load_config(DEFAULT_CONFIG)["model_eval"]["net_benefit"]["threshold_max_pct"])
    assert configured <= em.NB_THRESHOLD_CEILING_PCT
    em.threshold_grid(1, em.NB_THRESHOLD_CEILING_PCT)                 # the ceiling itself is fine
    with pytest.raises(AssertionError, match="above the frozen ceiling"):
        em.threshold_grid(1, em.NB_THRESHOLD_CEILING_PCT + 1)


def test_the_grid_refuses_a_float_bound_and_an_inverted_range():
    with pytest.raises(AssertionError, match="integer-percent basis"):
        em.threshold_grid(1.0, 35)
    with pytest.raises(AssertionError, match="integer-percent basis"):
        em.threshold_grid(1, 35.0)
    with pytest.raises(AssertionError, match="at least 1%"):
        em.threshold_grid(0, 35)
    with pytest.raises(AssertionError, match="at least 1%"):
        em.threshold_grid(20, 10)


def test_net_benefit_settings_reads_the_configured_block_and_nothing_else():
    cfg = load_config(DEFAULT_CONFIG)
    s = em.net_benefit_settings(cfg)
    nb = cfg["model_eval"]["net_benefit"]
    assert np.array_equal(s["thresholds"], em.threshold_grid(nb["threshold_min_pct"],
                                                             nb["threshold_max_pct"]))
    assert s["arms"] == list(nb["arms"]) and s["reference"] == nb["reference"]
    assert s["horizon_days"] == int(nb["horizon_days"]) == 1825
    assert s["sparse_events_min"] == 15
    assert (s["plot_min_pct"], s["plot_max_pct"]) == (2, 30)
    assert np.array_equal(s["threshold_pcts"], np.arange(1, 36))


def test_the_settings_block_is_read_directly_because_split_path_refuses_it():
    """The two keys differ by one token. ``net_benefit`` is settings, ``net_benefit_csv``
    is the output path, and resolving the first as a filename used to produce
    ``<repo>/{'threshold_min_pct': 1, ...}`` with no error at all."""
    cfg = load_config(DEFAULT_CONFIG)
    em.net_benefit_settings(cfg)                                   # the settings read works
    with pytest.raises(AssertionError, match="is not a path"):
        em.split_path(cfg, "net_benefit", "test")
    assert em.split_path(cfg, "net_benefit_csv", "test").name == "test_net_benefit.csv"


@pytest.mark.parametrize("edit,match", [
    ({"plot_max_pct": 40}, "not inside the estimated range"),
    ({"plot_min_pct": 0}, "not inside the estimated range"),
    ({"reference": "m9_nonexistent"}, "is not among the arms"),
    ({"arms": []}, "arms is empty"),
    ({"sparse_events_min": 0}, "marks nothing"),
    ({"threshold_max_pct": 60}, "above the frozen ceiling"),
])
def test_net_benefit_settings_refuses_a_block_that_contradicts_itself(edit, match):
    cfg = load_config(DEFAULT_CONFIG)
    cfg["model_eval"] = dict(cfg["model_eval"])
    cfg["model_eval"]["net_benefit"] = {**cfg["model_eval"]["net_benefit"], **edit}
    with pytest.raises(AssertionError, match=match):
        em.net_benefit_settings(cfg)


# --------------------------------------------------------------------------- #
# 12b. The estimator, hand-worked on four patients.                            #
# --------------------------------------------------------------------------- #
def test_net_benefit_matches_the_hand_worked_four_patient_example():
    time, event, risk = _dca_four()
    at = lambda p: em.net_benefit_at(time, event, risk, p, horizon=NB_HORIZON)

    r35 = at(0.35)
    assert r35["n_above"] == 3 and r35["events_above"] == 2
    assert r35["km_risk_above"] == pytest.approx(2.0 / 3.0, abs=1e-15)
    assert r35["km_last_obs_day"] == 1900.0
    assert r35["weight"] == pytest.approx(7.0 / 13.0, abs=1e-15)
    assert r35["tp"] == 0.5 and r35["fp"] == 0.25
    assert r35["net_benefit"] == pytest.approx(19.0 / 52.0, abs=1e-15)
    assert r35["note"] == ""

    r45 = at(0.45)
    assert r45["n_above"] == 2 and r45["km_risk_above"] == 0.5
    assert r45["net_benefit"] == pytest.approx(1.0 / 22.0, abs=1e-15)

    # w = 1 at p_t = 0.5, and half the flagged pair has the event: TP and FP cancel exactly.
    assert at(0.50)["net_benefit"] == 0.0


def test_treat_all_matches_the_hand_worked_four_patient_example():
    time, event, _ = _dca_four()
    all20 = em.net_benefit_at(time, event, None, 0.20, horizon=NB_HORIZON)
    assert all20["n_above"] == 4 and all20["n_scored"] == 4
    assert all20["km_risk_above"] == 0.5
    assert all20["net_benefit"] == 0.375
    # F_E = 0.5, so treat-all crosses zero at p_t = 0.5 exactly.
    assert em.net_benefit_at(time, event, None, 0.50, horizon=NB_HORIZON)["net_benefit"] == 0.0


def test_an_empty_flagged_set_is_zero_exactly_and_never_nan():
    """NaN would truncate the curve and hide the degeneration.

    A model whose predicted risks top out early stops flagging anybody, at which point the
    rule HAS BECOME treat-none: nobody is treated, TP = FP = 0, and 0 is the answer rather
    than the absence of one.
    """
    time, event, risk = _dca_four()                       # max predicted risk is 0.90
    row = em.net_benefit_at(time, event, risk, 0.95, horizon=NB_HORIZON)
    assert row["n_above"] == 0 and row["events_above"] == 0
    assert row["net_benefit"] == 0.0
    assert row["tp"] == 0.0 and row["fp"] == 0.0
    assert row["km_risk_above"] == 0.0 and row["km_last_obs_day"] == 0.0
    assert not np.isnan(row["net_benefit"])
    assert "treat-none" in row["note"]


def test_a_flagged_set_with_no_observed_event_reports_the_negative_floor():
    """Blanking exactly the thresholds at which a model flags non-events would delete the
    evidence of harm, which is the one thing a decision curve exists to show."""
    time, event, risk = _dca_four_no_events_above()
    row = em.net_benefit_at(time, event, risk, 0.50, horizon=NB_HORIZON)
    assert row["n_above"] == 2 and row["events_above"] == 0
    assert row["km_risk_above"] == 0.0
    assert row["net_benefit"] == -0.5
    assert row["net_benefit"] == -(row["n_above"] / row["n_scored"]) * row["weight"]
    assert row["km_last_obs_day"] == 1900.0 and row["note"] == "", (
        "follow-up reached past the horizon, so nothing was carried forward")


def test_nobody_followed_to_the_horizon_carries_the_km_forward_and_records_the_day():
    time, event, risk = _dca_four_short_follow_up()
    row = em.net_benefit_at(time, event, risk, 0.35, horizon=NB_HORIZON)
    assert row["km_last_obs_day"] == 900.0 < NB_HORIZON
    assert row["km_risk_above"] == pytest.approx(2.0 / 3.0, abs=1e-15)
    assert row["net_benefit"] == pytest.approx(19.0 / 52.0, abs=1e-15)
    assert "carried forward from day 900" in row["note"]
    assert "biases net benefit down" in row["note"], (
        "the direction of the bias is the reason this is reportable rather than fatal")


def test_tp_and_fp_partition_the_flagged_set_exactly():
    """One Kaplan-Meier fit, so no censored patient is dropped or double-counted.

    The partition is EXACT in patient units - ``F_A * n_A + (n_A - F_A * n_A) == n_A`` to
    the last bit, because n_A is an exact integer - and the per-patient rates TP and FP
    therefore reproduce n_A/n to within one ulp. (Bit-exact addition of the two rates is
    not attainable: TP + FP is a sum of two rounded quotients, and IEEE double addition is
    not associative with the rounding that produced them.)
    """
    time, event, risk = _censored_cohort(600, seed=11)
    for p in em.threshold_grid(1, 35):
        row = em.net_benefit_at(time, event, risk, float(p), horizon=NB_HORIZON)
        n_above, km, n = row["n_above"], row["km_risk_above"], row["n_scored"]
        tp_patients = km * n_above
        fp_patients = n_above - tp_patients
        assert tp_patients + fp_patients == float(n_above), "the flagged set is not partitioned"
        frac = n_above / n
        assert abs((row["tp"] + row["fp"]) - frac) <= np.spacing(max(frac, 5e-324))
        assert row["net_benefit"] == pytest.approx(row["tp"] - row["fp"] * row["weight"],
                                                   abs=1e-15)


def test_treat_all_runs_through_the_identical_code_path():
    """``risk=None`` is not a second formula; it means "flag everyone".

    Feeding a risk array that flags everyone must reproduce treat-all field for field, so
    the reference curve cannot drift away from the model curves it is subtracted from.
    """
    time, event, _ = _dca_four()
    everyone = np.ones(time.size)
    for p in em.threshold_grid(1, 35):
        assert em.net_benefit_at(time, event, None, float(p), horizon=NB_HORIZON) == \
            em.net_benefit_at(time, event, everyone, float(p), horizon=NB_HORIZON)
    src = inspect.getsource(em.net_benefit_at)
    assert src.count("km_cif_numpy(") == 1, (
        "there must be exactly one Kaplan-Meier call in the estimator; a second one is how "
        "treat-all and the model curves come to disagree")


def test_the_treat_all_curve_crosses_zero_at_the_prevalence():
    """The single most consequential number on the plot.

    Treat-all is ``F_E - (1 - F_E) * w(p_t)``, which is zero exactly at ``p_t = F_E``. The
    naive 5-year event rate on the test split is 106/741 = 14.3% because it counts the 473
    censored patients (63.8%) as event-free; the Kaplan-Meier value is 20.04% on test and
    20.02% on development, so a naive curve would put this crossing 5.7 percentage points
    too low. Both verified prevalences are reproduced here from cohorts built to carry them.
    """
    for n, n_events, prevalence in ((5000, 1001, 0.2002), (5000, 1002, 0.2004)):
        time, event = _prevalence_cohort(n, n_events)
        at = lambda p: em.net_benefit_at(time, event, None, p, horizon=NB_HORIZON)
        assert at(prevalence)["km_risk_above"] == pytest.approx(prevalence, abs=1e-12)
        assert at(prevalence)["net_benefit"] == pytest.approx(0.0, abs=1e-12)
        assert at(prevalence - 0.01)["net_benefit"] > 0, "treat-all pays below the prevalence"
        assert at(prevalence + 0.01)["net_benefit"] < 0, "and is net-harmful above it"
    # the naive rate would have put the crossing here instead
    naive = 106 / 741
    assert 0.2004 - naive == pytest.approx(0.0574, abs=5e-4)


def test_a_non_finite_risk_is_refused_rather_than_inflating_the_denominator():
    """``ArmScores.risk`` holds NaN for every patient the arm cannot score, and
    ``NaN >= p_t`` is False - so such a patient would never be flagged yet would still sit
    in n. That is a silently wrong curve, so it is an assertion instead."""
    time, event, risk = _dca_four()
    risk = risk.copy()
    risk[2] = np.nan
    with pytest.raises(AssertionError, match="non-finite predicted risk"):
        em.net_benefit_at(time, event, risk, 0.10, horizon=NB_HORIZON)


def test_the_estimator_never_calls_the_lifelines_km_that_raises_on_an_empty_set():
    """``model_clinical.km_risk`` raises ValueError on empty input, which is precisely the
    empty-flagged-set case this estimator has to answer. It is also 100x slower, and the
    curve needs ~1e5 fits."""
    with pytest.raises(Exception):
        from src.model_clinical import km_risk
        km_risk(np.array([]), np.array([], dtype=int), NB_HORIZON)
    for fn in (em.net_benefit_at, em.net_benefit_curve, em.net_benefit_ipcw_curve):
        assert "km_risk(" not in inspect.getsource(fn), fn.__name__


def test_the_flagged_set_km_agrees_with_the_lifelines_estimator():
    """``km_cif_numpy`` is a stand-in for ``km_risk``, not a different estimator."""
    from src.model_clinical import km_risk
    time, event, risk = _censored_cohort(300, seed=5)
    for p in (0.05, 0.10, 0.20, 0.30):
        row = em.net_benefit_at(time, event, risk, p, horizon=NB_HORIZON)
        flag = risk >= p
        assert row["n_above"] > 0
        obs, _, _ = km_risk(time[flag], event[flag], NB_HORIZON)
        assert row["km_risk_above"] == pytest.approx(obs, abs=1e-12)


def test_the_denominator_is_the_patients_screened_and_may_be_stated_explicitly():
    time, event, risk = _dca_four()
    default = em.net_benefit_at(time, event, risk, 0.35, horizon=NB_HORIZON)
    assert default["n_scored"] == 4
    stated = em.net_benefit_at(time, event, risk, 0.35, horizon=NB_HORIZON, n_scored=8)
    assert stated["n_scored"] == 8
    assert stated["net_benefit"] == pytest.approx(default["net_benefit"] / 2.0, abs=1e-15)
    with pytest.raises(AssertionError, match="not a usable denominator"):
        em.net_benefit_at(time, event, risk, 0.35, horizon=NB_HORIZON, n_scored=2)


# --------------------------------------------------------------------------- #
# 12c. The curve: the estimator at every threshold, plus treat-all beside it.  #
# --------------------------------------------------------------------------- #
def test_the_curve_is_the_estimator_at_every_threshold_and_nothing_more():
    time, event, risk = _dca_four()
    grid = em.threshold_grid(1, 35)
    rows = em.net_benefit_curve(time, event, risk, grid, horizon=NB_HORIZON,
                                sparse_events_min=15)
    assert len(rows) == grid.size
    for row, p in zip(rows, grid):
        direct = em.net_benefit_at(time, event, risk, float(p), horizon=NB_HORIZON)
        for key, value in direct.items():
            if key == "note":
                assert row[key].startswith(value)
            else:
                assert row[key] == value, key
        assert row["threshold_pct"] == int(round(float(p) * 100))


def test_treat_all_is_evaluated_on_the_same_patients_at_the_same_threshold():
    """``nb_treat_all_same_set`` is never a comparison across two denominators."""
    time, event, risk = _dca_four()
    rows = em.net_benefit_curve(time, event, risk, em.threshold_grid(1, 35),
                                horizon=NB_HORIZON, sparse_events_min=15)
    for row in rows:
        ta = em.net_benefit_at(time, event, None, row["threshold"], horizon=NB_HORIZON)
        assert row["nb_treat_all_same_set"] == ta["net_benefit"]
        assert ta["n_scored"] == row["n_scored"]
        assert row["diff_vs_treat_all"] == row["net_benefit"] - ta["net_benefit"]


def test_net_reduction_per_100_is_the_hand_worked_twenty_five_per_hundred():
    """At p_t = 0.35 on the four-patient fixture the model and treat-all find the SAME
    expected true positives (2 patients: F_A * n_A = (2/3)(3) = 2 = F_E * n = (1/2)(4)),
    and the model flags one fewer false positive (1 against 2). Net benefit therefore
    differs by exactly w/n, and the net reduction is one avoided intervention in four
    patients = **25 per 100**, at no cost in missed events.
    """
    time, event, risk = _dca_four()
    rows = em.net_benefit_curve(time, event, risk, em.threshold_grid(35, 35),
                                horizon=NB_HORIZON, sparse_events_min=15)
    row = rows[0]
    assert row["diff_vs_treat_all"] == pytest.approx(row["weight"] / 4.0, abs=1e-15)
    assert row["net_reduction_per_100"] == pytest.approx(25.0, abs=1e-12)


def test_the_treat_all_curve_differs_from_treat_all_by_exactly_zero():
    time, event, _ = _dca_four()
    rows = em.net_benefit_curve(time, event, None, em.threshold_grid(1, 35),
                                horizon=NB_HORIZON, sparse_events_min=15)
    for row in rows:
        assert row["diff_vs_treat_all"] == 0.0
        assert row["net_reduction_per_100"] == 0.0
        assert row["nb_treat_all_same_set"] == row["net_benefit"]


def test_treat_all_can_be_skipped_without_changing_the_estimate():
    time, event, risk = _censored_cohort(200, seed=3)
    grid = em.threshold_grid(1, 35)
    with_ta = em.net_benefit_curve(time, event, risk, grid, horizon=NB_HORIZON,
                                   sparse_events_min=15)
    without = em.net_benefit_curve(time, event, risk, grid, horizon=NB_HORIZON,
                                   sparse_events_min=15, treat_all=False)
    assert [r["net_benefit"] for r in with_ta] == [r["net_benefit"] for r in without]
    assert "nb_treat_all_same_set" not in without[0]


def test_a_sparse_row_is_flagged_not_suppressed():
    """The row keeps its estimate; only the figure truncates a curve where the flag trips."""
    time, event, risk = _censored_cohort(200, seed=9)
    grid = em.threshold_grid(1, 35)
    rows = em.net_benefit_curve(time, event, risk, grid, horizon=NB_HORIZON,
                                sparse_events_min=15)
    assert not rows[0]["sparse"] and rows[-1]["sparse"], (
        "this fixture must span both regimes: 22 events at p_t = 0.01, only 2 at 0.35")
    for row in rows:
        assert row["sparse"] == (row["events_above"] < 15)
        assert np.isfinite(row["net_benefit"]), "a sparse row is flagged, never blanked"
        if row["sparse"] and row["n_above"]:
            assert "sparse" in row["note"] and "estimate stands" in row["note"]
    # the floor has one source of truth: it is passed in, never defaulted
    loose = em.net_benefit_curve(time, event, risk, grid, horizon=NB_HORIZON,
                                 sparse_events_min=1)
    assert sum(r["sparse"] for r in loose) < sum(r["sparse"] for r in rows)


def test_the_curve_refuses_a_grid_that_is_not_the_integer_percent_basis():
    time, event, risk = _dca_four()
    bad = np.array([0.10, 0.125, 0.20])
    with pytest.raises(AssertionError, match="integer-percent basis"):
        em.net_benefit_curve(time, event, risk, bad, horizon=NB_HORIZON, sparse_events_min=15)


def test_the_curve_refuses_a_grid_that_is_not_ascending():
    time, event, risk = _dca_four()
    with pytest.raises(AssertionError, match="strictly ascending"):
        em.net_benefit_curve(time, event, risk, np.array([0.20, 0.10]), horizon=NB_HORIZON,
                             sparse_events_min=15)


def test_a_model_whose_risks_top_out_early_degenerates_into_treat_none_not_into_nan():
    """m0's maximum predicted risk is 0.4715, so above that its curve is identically zero.
    The tail of the CSV must say so rather than go missing."""
    time, event, risk = _dca_four()
    rows = em.net_benefit_curve(time, event, np.minimum(risk, 0.12), em.threshold_grid(1, 35),
                                horizon=NB_HORIZON, sparse_events_min=15)
    tail = [r for r in rows if r["threshold"] > 0.12]
    assert tail and all(r["n_above"] == 0 and r["net_benefit"] == 0.0 for r in tail)
    assert all(np.isfinite(r["net_benefit"]) for r in rows)


# --------------------------------------------------------------------------- #
# 12d. The IPCW sensitivity column.                                            #
# --------------------------------------------------------------------------- #
def test_the_two_estimators_give_the_identical_treat_all_curve():
    """For treat-all the IPCW estimator of the marginal risk IS the Kaplan-Meier estimator
    whenever G is the reverse Kaplan-Meier of the same sample, so the two net-benefit
    curves agree to floating point. The forms can only separate on a flagged SUBSET."""
    time, event, _ = _censored_cohort(2000, seed=2)
    g_grid, g_vals = tm.reverse_km(time, event)
    grid = em.threshold_grid(1, 35)
    km = np.array([em.net_benefit_at(time, event, None, float(p),
                                     horizon=NB_HORIZON)["net_benefit"] for p in grid])
    ipcw = em.net_benefit_ipcw_curve(time, event, None, grid, horizon=NB_HORIZON,
                                     g_grid=g_grid, g_vals=g_vals)
    assert np.max(np.abs(km - ipcw)) < 1e-12


def _worst_agreement(n: int, seed: int) -> float:
    """max |KM - IPCW| over the plotted range on one simulated cohort.

    Censoring is drawn INDEPENDENTLY of the predicted risk, which is exactly what IPCW with
    the marginal reverse Kaplan-Meier requires and what the real cohort cannot promise
    (follow-up is administrative EHR contact, which plausibly tracks the comorbidity and
    radiographic severity the model reads off the film). So this is the estimators' BEST
    case, and the disagreement measured here is a floor, not a ceiling.
    """
    grid = em.threshold_grid(1, 35)
    time, event, risk = _censored_cohort(n, seed=seed)
    g_grid, g_vals = tm.reverse_km(time, event)
    km = np.array([em.net_benefit_at(time, event, risk, float(p),
                                     horizon=NB_HORIZON)["net_benefit"] for p in grid])
    ipcw = em.net_benefit_ipcw_curve(time, event, risk, grid, horizon=NB_HORIZON,
                                     g_grid=g_grid, g_vals=g_vals)
    return float(np.max(np.abs(km - ipcw)[grid <= 0.30]))


def test_the_old_zero_point_zero_zero_five_agreement_bound_was_an_artefact_of_sample_size():
    """WHY THIS TEST REPLACED ITS PREDECESSOR.

    ``net_benefit_ipcw_curve``'s docstring used to claim the two estimators "agree to within
    0.005 for p_t <= 0.30", and the test that certified it drew 6,000 patients. The study's
    sealed split holds 741. At 741 the bound fails more often than it holds, EVEN when
    censoring is drawn independently of risk, so the old test proved nothing about this
    cohort while making the false claim look defended.

    Measured: 17 of 20 seeds breach 0.005 at n = 400, 11 of 20 at n = 741, and none of 20 at
    n = 6,000. What is asserted is the shape of that - monotone in n, a majority breach at
    the study's own n, none in the regime the retired test used - rather than the three
    counts, which are properties of a particular generator stream.
    """
    breaches = {n: sum(_worst_agreement(n, s) >= 0.005 for s in range(20))
                for n in (400, 741, 6000)}
    assert breaches[6000] == 0, (
        f"at n = 6,000 the two estimators do agree to 0.005 ({breaches[6000]}/20 breaches); "
        f"that regime is the ONLY thing the retired test ever established")
    assert breaches[741] >= 10, (
        f"only {breaches[741]}/20 seeds breach 0.005 at the sealed split's n = 741, against "
        f"11/20 measured. If this has genuinely improved, re-measure the real artefacts and "
        f"rewrite net_benefit_ipcw_curve's docstring - do not relax this bound")
    assert breaches[400] > breaches[741] > breaches[6000], (
        f"the disagreement must fall with n, which is what makes it a sample-size effect "
        f"rather than a defect in either estimator; got {breaches}")


def test_the_ipcw_curve_is_defined_wherever_the_primary_estimator_is():
    time, event, risk = _dca_four()
    grid = em.threshold_grid(1, 35)
    g_grid, g_vals = tm.reverse_km(time, event)
    out = em.net_benefit_ipcw_curve(time, event, risk, grid, horizon=NB_HORIZON,
                                    g_grid=g_grid, g_vals=g_vals)
    assert out.shape == grid.shape and np.isfinite(out).all()
    # above the maximum predicted risk nobody is flagged, so both estimators report 0
    empty = em.net_benefit_ipcw_curve(time, event, risk, np.array([0.95]), horizon=NB_HORIZON,
                                      g_grid=g_grid, g_vals=g_vals)
    assert empty[0] == 0.0


# --------------------------------------------------------------------------- #
# 12da. HOW FAR APART THE TWO ESTIMATORS ACTUALLY ARE, ON THE REAL ARTEFACTS.  #
#                                                                              #
# net_benefit_ipcw_curve's docstring makes a quantitative claim about THIS      #
# study's data, so it is checked against this study's data, at this study's n.  #
# Its PROSE is deliberately not pinned: a test that asserted the docstring      #
# contained the string "0.005" is exactly what kept a false agreement claim     #
# alive through a code review. What is pinned is the measured quantity the      #
# words describe, so the words cannot drift away from it in silence.            #
#                                                                              #
# Nothing here runs src.eval_models or writes anything. The frozen Cox arms are #
# replayed and the trained arms are read from their hazard npz exactly as       #
# main() does, so these are the risks the decision-curve engine will see -      #
# WITH the frozen recalibration on the image arms, which is not optional: it    #
# moves the flagged set at a given threshold and therefore the whole curve.     #
# --------------------------------------------------------------------------- #
NB_DCA_ARMS = ("m2_frontal", "m1", "m0", "m4_fusion")
NB_DCA_REFERENCE = "m0"

# max |KM - IPCW| over p_t <= 0.30, per arm. MEASURED, not derived.
NB_MEASURED_MAX_GAP = {
    "val": {"m2_frontal": 0.0283, "m1": 0.0129, "m0": 0.0200, "m4_fusion": 0.0216},
    "test": {"m2_frontal": 0.0078, "m1": 0.0077, "m0": 0.0161, "m4_fusion": 0.0098},
}
# (KM, IPCW) at the headline p_t = 0.20.
NB_MEASURED_AT_20 = {
    "val": {"m2_frontal": (0.0729, 0.0925), "m1": (0.0651, 0.0704),
            "m0": (0.0334, 0.0467), "m4_fusion": (0.0696, 0.0804)},
    "test": {"m2_frontal": (0.0904, 0.0952), "m1": (0.0671, 0.0717),
             "m0": (0.0396, 0.0557), "m4_fusion": (0.0710, 0.0741)},
}
# The same two quantities for the PAIRED contrast against m0, on the intersection.
NB_MEASURED_CONTRAST_MAX_GAP = {
    "val": {"m2_frontal": 0.0209, "m1": 0.0143, "m4_fusion": 0.0146},
    "test": {"m2_frontal": 0.0132, "m1": 0.0119, "m4_fusion": 0.0130},
}
NB_MEASURED_CONTRAST_AT_20 = {
    "val": {"m2_frontal": (0.0388, 0.0457), "m1": (0.0347, 0.0284),
            "m4_fusion": (0.0362, 0.0337)},
    "test": {"m2_frontal": (0.0495, 0.0390), "m1": (0.0276, 0.0158),
             "m4_fusion": (0.0313, 0.0184)},
}
NB_TOL = 5e-4          # the estimates are deterministic; this is float-noise room, not slack


@functools.lru_cache(maxsize=2)
def _real_dca(split: str):
    """``(roster, {arm: ArmScores})`` for the four decision-curve arms on ``split``.

    Skips rather than fails when the artefacts are not in this checkout - ``derived-data/``
    is git-ignored, so a fresh clone has no hazard files. Reading the sealed split is
    additionally gated on the sealed read being ON THE RECORD, which is the repository's own
    rule and not a new one invented here.
    """
    cfg = load_config(DEFAULT_CONFIG)
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    needed = ["m0_clinical_model.json", "m1_klg_model.json", "clinical_imputation_params.json",
              "features_clinical.parquet", "train_arms.json",
              f"{split}_hazards_m2_frontal.npz", f"{split}_hazards_m4_fusion.npz"]
    absent = [name for name in needed if not (coh / name).exists()]
    if absent:
        pytest.skip(f"the {split} decision-curve artefacts are not in this checkout: {absent}")
    if split == SEALED_SPLIT:
        em.assert_sealed_read_is_recorded(cfg)
    contracts = tm.FrozenContracts(coh)
    train_arms = em.load_train_arms(coh / "train_arms.json")
    roster, _ = em.load_roster(contracts, QUIET, split=split)
    scores = {}
    for arm in NB_DCA_ARMS:
        scores[arm] = (
            em.cox_arm_scores(arm, contracts, roster, HORIZONS, QUIET, split=split)
            if arm in em.COX_ARMS else
            em.trained_arm_scores(arm, train_arms["arms"][arm], coh, roster, HORIZONS,
                                  QUIET, split=split))
        assert scores[arm] is not None, arm
    return roster, scores


def _real_curves(roster, scores, arm, grid, mask=None):
    """``(km, ipcw)`` for one arm, on its own patients unless ``mask`` narrows them."""
    m = scores[arm].present if mask is None else (mask & scores[arm].present)
    t, e, n = roster.time[m], roster.event[m], int(m.sum())
    r = scores[arm].risk[int(NB_HORIZON)][m]
    km = np.array([row["net_benefit"] for row in
                   em.net_benefit_curve(t, e, r, grid, horizon=NB_HORIZON,
                                        sparse_events_min=15, n_scored=n, treat_all=False)])
    ipcw = em.net_benefit_ipcw_curve(t, e, r, grid, horizon=NB_HORIZON, g_grid=roster.g_grid,
                                     g_vals=roster.g_vals, n_scored=n)
    return km, ipcw


@pytest.mark.parametrize("split", ["val", SEALED_SPLIT])
def test_absolute_net_benefit_is_estimator_dependent_at_this_sample_size(split):
    """The retired claim was "the two agree to within 0.005 for p_t <= 0.30". They do not.

    Worst arm: 0.0161 on the sealed split (m0, at p_t = 0.20) and 0.0283 on validation
    (m2_frontal, at 0.26), against a claimed 0.005. At p_t = 0.20 the m0 gap is 41% of m0's
    own estimate. These are the numbers net_benefit_ipcw_curve's docstring quotes; if one of
    them moves, the docstring is wrong until someone rewrites it.
    """
    roster, scores = _real_dca(split)
    grid = em.threshold_grid(1, 35)
    plotted = grid <= 0.30
    at20 = int(np.flatnonzero(np.rint(grid * 100).astype(int) == 20)[0])
    worst = 0.0
    for arm in NB_DCA_ARMS:
        km, ipcw = _real_curves(roster, scores, arm, grid)
        gap = np.abs(km - ipcw)
        assert float(gap[plotted].max()) == pytest.approx(
            NB_MEASURED_MAX_GAP[split][arm], abs=NB_TOL), (
            f"{split}/{arm}: max |KM - IPCW| over p_t <= 0.30 is "
            f"{gap[plotted].max():.4f}, not the {NB_MEASURED_MAX_GAP[split][arm]:.4f} the "
            f"docstring quotes. Re-measure and rewrite the docstring; do not widen this")
        exp_km, exp_ipcw = NB_MEASURED_AT_20[split][arm]
        assert float(km[at20]) == pytest.approx(exp_km, abs=NB_TOL)
        assert float(ipcw[at20]) == pytest.approx(exp_ipcw, abs=NB_TOL)
        worst = max(worst, float(gap[plotted].max()))
    assert worst > 0.005, (
        f"on {split} the two estimators now agree to within 0.005 (worst {worst:.4f}), which "
        f"is the claim this repository retired as false. Something moved: re-measure every "
        f"number in net_benefit_ipcw_curve's docstring before trusting this")


@pytest.mark.parametrize("split", ["val", SEALED_SPLIT])
def test_the_arm_versus_arm_contrast_moves_less_than_the_curves_but_does_not_stand_still(split):
    """The frozen-G bias is largely common to two arms on the same patients, so it partly
    cancels in a difference - PARTLY. On the sealed split the worst contrast gap is 0.0132
    against 0.0161 for the curves themselves, and at p_t = 0.20 m2_frontal minus m0 is
    +0.0495 under the primary estimator against +0.0390 under IPCW, still 21% of it. The
    contrast is the quantity to report, but not because the two estimators agree about it.
    """
    roster, scores = _real_dca(split)
    grid = em.threshold_grid(1, 35)
    plotted = grid <= 0.30
    at20 = int(np.flatnonzero(np.rint(grid * 100).astype(int) == 20)[0])
    ref = scores[NB_DCA_REFERENCE].present
    worst_contrast = 0.0
    for arm in NB_DCA_ARMS:
        if arm == NB_DCA_REFERENCE:
            continue
        both = scores[arm].present & ref
        km_a, ipcw_a = _real_curves(roster, scores, arm, grid, mask=both)
        km_r, ipcw_r = _real_curves(roster, scores, NB_DCA_REFERENCE, grid, mask=both)
        km_d, ipcw_d = km_a - km_r, ipcw_a - ipcw_r
        gap = np.abs(km_d - ipcw_d)
        assert float(gap[plotted].max()) == pytest.approx(
            NB_MEASURED_CONTRAST_MAX_GAP[split][arm], abs=NB_TOL), (
            f"{split}/{arm} minus {NB_DCA_REFERENCE} on {int(both.sum())} paired patients: "
            f"max |KM - IPCW| over p_t <= 0.30 is {gap[plotted].max():.4f}, not "
            f"{NB_MEASURED_CONTRAST_MAX_GAP[split][arm]:.4f}")
        exp_km, exp_ipcw = NB_MEASURED_CONTRAST_AT_20[split][arm]
        assert float(km_d[at20]) == pytest.approx(exp_km, abs=NB_TOL)
        assert float(ipcw_d[at20]) == pytest.approx(exp_ipcw, abs=NB_TOL)
        worst_contrast = max(worst_contrast, float(gap[plotted].max()))
    worst_absolute = max(NB_MEASURED_MAX_GAP[split].values())
    assert worst_contrast < worst_absolute, (
        f"the contrast ({worst_contrast:.4f}) is no steadier than the curves it is built "
        f"from ({worst_absolute:.4f}) on {split}, so the docstring's reason for reporting "
        f"the contrast rather than the level no longer holds")
    assert worst_contrast > 0.005, (
        f"the contrast now agrees to {worst_contrast:.4f} on {split}; the docstring says it "
        f"cancels the frozen-G bias only PARTLY, and that sentence needs re-measuring")


def test_the_frozen_censoring_curve_is_what_moves_the_two_estimators_apart():
    """The disagreement is imported from a TRAINING quantity, not discovered in the data.

    Treat-all flags everyone, so no subsetting is involved at all; under a G estimated from
    the sample itself the IPCW estimator of the marginal risk IS the Kaplan-Meier estimator
    (the test above this section pins that identity to floating point on a cohort with no
    ties). Weight the sealed split with M0's frozen ``censoring_km_train`` instead and the
    implied 5-year risk is 0.2125 against a Kaplan-Meier 0.2004, which moves the treat-all
    reference line itself by up to 0.0252 and shifts its zero crossing by a whole percentage
    point of threshold. Re-estimating G on the split closes that to 4e-5, some 500 times
    smaller; the residual is tie handling, not estimation, because follow-up is in whole
    days and 33 of the 106 sealed-split events fall on a day that also carries a censoring.
    """
    roster, _ = _real_dca(SEALED_SPLIT)
    grid = em.threshold_grid(1, 35)
    plotted = grid <= 0.30
    t, e = roster.time, roster.event
    km = np.array([em.net_benefit_at(t, e, None, float(p), horizon=NB_HORIZON)["net_benefit"]
                   for p in grid])
    frozen = em.net_benefit_ipcw_curve(t, e, None, grid, horizon=NB_HORIZON,
                                       g_grid=roster.g_grid, g_vals=roster.g_vals)
    own_grid, own_vals = tm.reverse_km(t, e)
    own = em.net_benefit_ipcw_curve(t, e, None, grid, horizon=NB_HORIZON,
                                    g_grid=own_grid, g_vals=own_vals)
    own_gap = float(np.abs(km - own)[plotted].max())
    frozen_gap = float(np.abs(km - frozen)[plotted].max())
    assert int(np.isin(t[e == 1], t[e == 0]).sum()) == 33, "the tie count the docstring cites"
    assert own_gap == pytest.approx(4.5e-5, abs=1e-5), (
        f"with the split's own reverse Kaplan-Meier the treat-all curves are the same "
        f"estimator up to tie handling; they now differ by {own_gap:.2e}")
    assert frozen_gap == pytest.approx(0.0252, abs=NB_TOL)
    assert frozen_gap > 100 * own_gap, (
        "the frozen curve is meant to be the dominant source of the disagreement; if it is "
        "not, points 2 and 4 of net_benefit_ipcw_curve's docstring need re-measuring")
    # the implied prevalence, which is where treat-all crosses zero
    y, w = ipcw_labels_weights(t, e, NB_HORIZON, roster.g_grid, roster.g_vals)
    implied = float(w[y == 1].sum() / (w[y == 1].sum() + w[y == 0].sum()))
    observed, _ = tm.km_cif_numpy(t, e, NB_HORIZON)
    assert implied == pytest.approx(0.2125, abs=NB_TOL)
    assert observed == pytest.approx(0.2004, abs=NB_TOL)
    pcts = np.rint(grid * 100).astype(int)
    assert int(pcts[np.flatnonzero(km <= 0)[0]]) == 21
    assert int(pcts[np.flatnonzero(frozen <= 0)[0]]) == 22


NB_MEASURED_GAP_OVER_HALF_AT_20 = {"m2_frontal": 0.33, "m1": 0.43, "m4_fusion": 0.38}


def test_the_ipcw_contrast_stays_inside_the_primary_estimators_own_interval():
    """The load-bearing claim: the estimator gap is smaller than the sampling uncertainty.

    Rebuilds the paired bootstrap the decision-curve engine uses - one shared seeded draw of
    roster positions, ``take = idx[usable[idx]]``, ``percentile_ci`` - for all three
    contrasts against m0 on the sealed split. Across those 90 contrast-by-threshold points
    the IPCW point estimate sits inside the primary estimator's 95% interval at 89, the gap
    never reaches that interval's half-width, and wherever the primary contrast excludes
    zero the IPCW contrast has the same sign. That, and not agreement in level, is why the
    figure's conclusions do not depend on the estimator.

    Deliberately the sealed split and not both: this is the split the manuscript reports,
    and a second 2,000-replicate pass would double the file's runtime for a weaker claim.
    """
    roster, scores = _real_dca(SEALED_SPLIT)
    cfg = load_config(DEFAULT_CONFIG)
    grid = em.threshold_grid(1, 35)
    plotted = grid <= 0.30
    pcts = np.rint(grid * 100).astype(int)
    at20 = int(np.flatnonzero(pcts == 20)[0])
    draw = em.bootstrap_draw(len(roster), int(cfg["model_eval"]["bootstrap_n"]),
                             int(cfg["model_eval"]["bootstrap_seed"]))
    r_ref = scores[NB_DCA_REFERENCE].risk[int(NB_HORIZON)]

    outside_all, checked = [], 0
    for arm in NB_DCA_ARMS:
        if arm == NB_DCA_REFERENCE:
            continue
        usable = scores[arm].present & scores[NB_DCA_REFERENCE].present
        km_a, ipcw_a = _real_curves(roster, scores, arm, grid, mask=usable)
        km_r, ipcw_r = _real_curves(roster, scores, NB_DCA_REFERENCE, grid, mask=usable)
        km_d, ipcw_d = km_a - km_r, ipcw_a - ipcw_r
        r_arm = scores[arm].risk[int(NB_HORIZON)]
        boot = np.full((len(draw), grid.size), np.nan)
        for b, idx in enumerate(draw):
            take = idx[usable[idx]]
            if take.size < 2:
                continue
            t, e, n = roster.time[take], roster.event[take], take.size
            one = np.array([row["net_benefit"] for row in em.net_benefit_curve(
                t, e, r_arm[take], grid, horizon=NB_HORIZON, sparse_events_min=15,
                n_scored=n, treat_all=False)])
            two = np.array([row["net_benefit"] for row in em.net_benefit_curve(
                t, e, r_ref[take], grid, horizon=NB_HORIZON, sparse_events_min=15,
                n_scored=n, treat_all=False)])
            boot[b] = one - two
        lo = np.array([percentile_ci(boot[:, j])[0] for j in range(grid.size)])
        hi = np.array([percentile_ci(boot[:, j])[1] for j in range(grid.size)])
        half = (hi - lo) / 2.0
        gap = np.abs(km_d - ipcw_d)

        inside = (ipcw_d >= lo) & (ipcw_d <= hi)
        outside_all += [(arm, int(p)) for p, ok in zip(pcts[plotted], inside[plotted]) if not ok]
        checked += int(plotted.sum())
        wide = plotted & (half > 1e-9)
        assert float((gap[wide] / half[wide]).max()) < 1.0, (
            f"{arm}: the estimator gap reaches {(gap[wide] / half[wide]).max():.2f} of the "
            f"interval half-width, so it is no longer smaller than the sampling uncertainty "
            f"the figure already shows")
        assert float(gap[at20] / half[at20]) == pytest.approx(
            NB_MEASURED_GAP_OVER_HALF_AT_20[arm], abs=0.02)
        significant = plotted & (lo > 0)
        assert significant.sum() >= 8 and bool(np.all(ipcw_d[significant] > 0)), (
            f"{arm}: the two estimators disagree about the SIGN of the contrast at a "
            f"threshold where the paired bootstrap interval excludes zero; the manuscript "
            f"may not then report the contrast as estimator-independent")

    assert checked == 90 and outside_all == [("m2_frontal", 1)], (
        f"the IPCW contrast leaves the primary estimator's own interval at {outside_all} "
        f"instead of only at m2_frontal / p_t = 0.01, where it misses by 0.0001")


# --------------------------------------------------------------------------- #
# 12e. The CSV schema.                                                         #
# --------------------------------------------------------------------------- #
def test_net_benefit_columns_carry_every_pinned_field_in_a_fixed_order():
    """``write_table`` asserts column order exactly, and two downstream modules read this
    file, so membership AND order are the contract."""
    cols = em.NET_BENEFIT_COLUMNS
    assert len(cols) == len(set(cols)), "a duplicated column name"
    required = ["n_above", "events_above", "km_risk_above", "km_last_obs_day",
                "net_benefit", "net_benefit_lo", "net_benefit_hi", "net_benefit_ipcw",
                "nb_treat_all_same_set", "diff_vs_treat_all", "diff_vs_treat_all_lo",
                "diff_vs_treat_all_hi", "diff_vs_treat_all_p", "net_reduction_per_100",
                "diff_vs_reference", "diff_vs_reference_lo", "diff_vs_reference_hi",
                "diff_vs_reference_p", "n_paired", "n_replicates_valid", "sparse",
                "suppressed", "note"]
    assert set(required) <= set(cols), f"missing {sorted(set(required) - set(cols))}"
    # the row must also say what it is a row OF
    assert {"split", "arm", "threshold", "horizon_days", "reference"} <= set(cols)
    assert cols[-1] == "note", "the free-text column stays last, as in every other schema"
    assert not any(tok in c.lower() for c in cols
                   for tok in em.FORBIDDEN_OUTPUT_COLUMN_TOKENS), (
        "outputs/ is aggregate only (protocol section 28)")


def test_the_estimator_fills_the_columns_it_claims_and_leaves_the_rest_to_the_engine():
    """Which columns are the pure functions' job, and which wait for the shared draw."""
    time, event, risk = _dca_four()
    row = em.net_benefit_curve(time, event, risk, em.threshold_grid(1, 35),
                               horizon=NB_HORIZON, sparse_events_min=15)[0]
    filled = set(row) & set(em.NET_BENEFIT_COLUMNS)
    assert filled == {"threshold", "threshold_pct", "horizon_days", "n_scored", "n_above",
                      "events_above", "km_risk_above", "km_last_obs_day", "net_benefit",
                      "nb_treat_all_same_set", "diff_vs_treat_all", "net_reduction_per_100",
                      "sparse", "note"}
    left = set(em.NET_BENEFIT_COLUMNS) - filled
    assert left == {"split", "arm", "label", "net_benefit_lo", "net_benefit_hi",
                    "net_benefit_ipcw", "diff_vs_treat_all_lo", "diff_vs_treat_all_hi",
                    "diff_vs_treat_all_p", "reference", "diff_vs_reference",
                    "diff_vs_reference_lo", "diff_vs_reference_hi", "diff_vs_reference_p",
                    "n_paired", "n_replicates_valid", "suppressed"}
    assert "weight" not in em.NET_BENEFIT_COLUMNS and "tp" not in em.NET_BENEFIT_COLUMNS, (
        "w, TP and FP are diagnostics recoverable from km_risk_above and n_above")


def test_a_net_benefit_frame_survives_the_aggregate_only_writer(tmp_path):
    time, event, risk = _dca_four()
    rows = em.net_benefit_curve(time, event, risk, em.threshold_grid(1, 35),
                                horizon=NB_HORIZON, sparse_events_min=15)
    for r in rows:                       # what the engine will add
        r.update(split="test", arm="m2_frontal", label="M2 frontal", reference="m0",
                 net_benefit_lo=np.nan, net_benefit_hi=np.nan, net_benefit_ipcw=np.nan,
                 diff_vs_treat_all_lo=np.nan, diff_vs_treat_all_hi=np.nan,
                 diff_vs_treat_all_p=np.nan, diff_vs_reference=np.nan,
                 diff_vs_reference_lo=np.nan, diff_vs_reference_hi=np.nan,
                 diff_vs_reference_p=np.nan, n_paired=4, n_replicates_valid=2000,
                 suppressed=False)
    df = pd.DataFrame(rows, columns=em.NET_BENEFIT_COLUMNS)
    em.write_table(tmp_path / "nb.csv", df, em.NET_BENEFIT_COLUMNS,
                   np.array(["100000", "100001"]), "net benefit")
    written = pd.read_csv(tmp_path / "nb.csv")
    assert list(written.columns) == em.NET_BENEFIT_COLUMNS
    assert len(written) == 35 and written["threshold_pct"].tolist() == list(range(1, 36))


# --------------------------------------------------------------------------- #
# 12f. THE ENGINE. The curves above are pure functions; NetBenefitEngine drives #
#      them from the ONE shared draw and build_net_benefit assembles the CSV.   #
#                                                                              #
# Three properties carry the weight here and each has a defect behind it:       #
#                                                                              #
#  * the SAME draw object, and a SEPARATE cache. BootstrapEngine's key is       #
#    (arm, packbits(mask)) with no horizon and no metric-set component, so a    #
#    net-benefit entry would collide with a metrics entry for the same arm and  #
#    mask; and its boot_metric_keys seeds {k: nan} and fills only the names     #
#    arm_metrics knows, so an added key returns silently all-NaN into a         #
#    val_metrics.csv schema three tests pin.                                    #
#  * every arm on ITS OWN patients (741 / 740 / 734 / 707), every difference on #
#    the intersection. Harmonising the ladder onto the 707 that every arm       #
#    scores would throw away 8 of 106 events to buy nothing.                    #
#  * the RECALIBRATED risks ArmScores publishes. Measuring on the raw ones      #
#    moves the flagged set at a given threshold and therefore the whole curve;  #
#    doing so once already produced a figure that was wrong by a factor of two. #
# --------------------------------------------------------------------------- #
def nb_config(arms, reference: str, *, min_pct: int = 1, max_pct: int = 35,
              sparse: int = 15, horizon: int = 1825) -> Config:
    """The slice of config :func:`net_benefit_settings` reads, and nothing else."""
    return Config({"model_eval": {"net_benefit": {
        "threshold_min_pct": min_pct, "threshold_max_pct": max_pct,
        "plot_min_pct": min_pct, "plot_max_pct": max_pct,
        "sparse_events_min": sparse, "arms": list(arms), "reference": reference,
        "horizon_days": horizon}}})


def make_nb_engine(roster: em.Roster, cfg: Config, *, n_boot: int = 40,
                   seed: int = 7) -> em.NetBenefitEngine:
    s = em.net_benefit_settings(cfg)
    draw = em.bootstrap_draw(len(roster), n_boot, seed)
    return em.NetBenefitEngine(roster, draw, s["thresholds"], s["horizon_days"],
                               s["sparse_events_min"], QUIET)


def _nb_fixture(n: int = 80, *, drop_arm: int = 10, drop_ref_head: int = 5):
    """A roster, one arm and a reference whose patient sets OVERLAP without nesting."""
    roster = make_roster(n, seed=3)
    a = make_arm("a", roster, drop=drop_arm, seed=1)
    ref = make_arm("m0", roster, seed=2)
    ref.present[:drop_ref_head] = False        # the reference misses patients the arm scores
    return roster, {"a": a, "m0": ref}


def test_the_net_benefit_engine_runs_on_the_very_same_draw_object():
    """Not an equal matrix - the SAME object. A second draw would silently unpair every
    difference the figure draws, and the two would still look identical in the CSV."""
    roster = make_roster(60)
    engine = make_engine(roster)
    cfg = nb_config(["a", "m0"], "m0")
    s = em.net_benefit_settings(cfg)
    nb = em.NetBenefitEngine(roster, engine.draw, s["thresholds"], s["horizon_days"],
                             s["sparse_events_min"], QUIET)
    assert nb.draw is engine.draw
    assert "nb_engine.draw is engine.draw" in inspect.getsource(em.main), (
        "main() must assert the two engines share the draw, not merely pass it")


def test_the_two_engines_share_a_cache_key_and_must_not_share_a_cache():
    """The reason net benefit is a second engine rather than more metrics in the first.

    The keys COLLIDE by construction: both are (arm, packbits(mask)), with no horizon and
    no metric-set component. That is fine only because the caches are different objects
    holding different value types.
    """
    roster = make_roster(60)
    arm = make_arm("a", roster)
    engine = make_engine(roster)
    cfg = nb_config(["a", "m0"], "m0")
    nbe = em.NetBenefitEngine(roster, engine.draw, *[em.net_benefit_settings(cfg)[k]
                                                     for k in ("thresholds", "horizon_days",
                                                               "sparse_events_min")], QUIET)
    assert em.NetBenefitEngine._key(arm, arm.present) == em.BootstrapEngine._key(arm, arm.present)
    assert nbe._point is not engine._point and nbe._boot is not engine._boot
    assert isinstance(engine.point(arm, arm.present), dict)
    assert isinstance(nbe.point(arm, arm.present), list)


def test_net_benefit_never_enters_the_pinned_metrics_schema_or_its_replicate_loop():
    """``boot_metric_keys`` multiplies the 2,000-replicate loop per arm per mask and feeds
    ``val_metrics.csv``; an added key there returns all-NaN with no error at all."""
    assert em.boot_metric_keys(HORIZONS) == ["harrell_c", "uno_c"] + [f"auc@{h}"
                                                                     for h in HORIZONS]
    assert not any("net_benefit" in c or "threshold" in c
                   for c in em.metrics_columns(HORIZONS) + em.point_metric_keys(HORIZONS))
    assert "net_benefit" not in inspect.getsource(em.arm_metrics)


def test_a_replicate_drops_the_patients_the_arm_cannot_score_exactly_as_the_metrics_engine_does():
    """``take = idx[usable[idx]]``, verbatim. Replicate sizes therefore differ per arm, and
    any other rule would unpair the difference the shared draw exists to provide."""
    assert "idx[usable[idx]]" in inspect.getsource(em.BootstrapEngine.boot)
    assert "idx[usable[idx]]" in inspect.getsource(em.NetBenefitEngine.boot)

    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m0"], "m0")
    eng = make_nb_engine(roster, cfg, n_boot=25)
    a = arms["a"]
    got = eng.boot(a, a.present, treat_all=True)
    grid = em.net_benefit_settings(cfg)["thresholds"]

    want_nb = np.full((len(eng.draw), grid.size), np.nan)
    want_da = np.full_like(want_nb, np.nan)
    usable = a.present
    for b, idx in enumerate(eng.draw):
        take = idx[usable[idx]]
        if take.size < 2:
            continue
        rows = em.net_benefit_curve(roster.time[take], roster.event[take],
                                    a.risk[1825][take], grid, horizon=1825.0,
                                    sparse_events_min=15, n_scored=int(take.size))
        want_nb[b] = [r["net_benefit"] for r in rows]
        want_da[b] = [r["diff_vs_treat_all"] for r in rows]
    assert np.allclose(got["net_benefit"], want_nb, equal_nan=True)
    assert np.allclose(got["diff_vs_treat_all"], want_da, equal_nan=True)


def test_skipping_treat_all_halves_the_fits_without_moving_the_estimate():
    """``treat_all=False`` is a speed switch for the reference arm's pass on a contrast
    intersection, where the treat-all line belongs to the other arm's row. It may not
    change a single net benefit."""
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m0"], "m0")
    with_all = make_nb_engine(roster, cfg, n_boot=20).boot(arms["a"], arms["a"].present,
                                                           treat_all=True)
    without = make_nb_engine(roster, cfg, n_boot=20).boot(arms["a"], arms["a"].present,
                                                          treat_all=False)
    assert np.allclose(with_all["net_benefit"], without["net_benefit"], equal_nan=True)
    assert "diff_vs_treat_all" not in without


def test_a_cached_pass_that_skipped_treat_all_is_recomputed_when_it_is_needed():
    """The saving may never turn into a missing column."""
    roster, arms = _nb_fixture()
    eng = make_nb_engine(roster, nb_config(["a", "m0"], "m0"), n_boot=12)
    first = eng.boot(arms["a"], arms["a"].present, treat_all=False)
    assert "diff_vs_treat_all" not in first
    second = eng.boot(arms["a"], arms["a"].present, treat_all=True)
    assert "diff_vs_treat_all" in second
    assert np.allclose(first["net_benefit"], second["net_benefit"], equal_nan=True)


def test_every_paired_difference_is_on_the_intersection_and_the_note_says_so():
    """The arms rest on different populations on purpose; the row must not pretend otherwise."""
    roster, arms = _nb_fixture(80, drop_arm=10, drop_ref_head=5)
    cfg = nb_config(["a", "m0"], "m0")
    eng = make_nb_engine(roster, cfg, n_boot=20)
    df = em.build_net_benefit(cfg, arms, eng, QUIET, split="val")
    row = df[df["arm"] == "a"].iloc[0]
    assert int(row["n_scored"]) == 70 and int(row["n_paired"]) == 65
    assert "paired on the 65 patients both arms score" in row["note"]
    assert "a scores 70" in row["note"] and "m0 scores 75" in row["note"]
    # and the difference really is the two arms recomputed THERE, not two marginal curves
    both = arms["a"].present & arms["m0"].present
    d = (np.array([r["net_benefit"] for r in eng.point(arms["a"], both)])
         - np.array([r["net_benefit"] for r in eng.point(arms["m0"], both)]))
    got = df[df["arm"] == "a"].sort_values("threshold_pct")["diff_vs_reference"].to_numpy()
    assert np.allclose(got, d)
    # the arm's own net_benefit is NOT on the intersection - it is on its own 70
    own = np.array([r["net_benefit"] for r in eng.point(arms["a"], arms["a"].present)])
    assert np.allclose(df[df["arm"] == "a"].sort_values("threshold_pct")["net_benefit"], own)


def test_the_reference_arms_own_row_is_an_exact_zero_difference_with_a_p_of_one():
    """It runs through the identical code path against itself, so the figure can draw the
    reference as the flat zero line it is instead of special-casing a NaN."""
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m0"], "m0")
    df = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=20), QUIET,
                              split="val")
    ref = df[df["arm"] == "m0"]
    assert len(ref) == 35
    for col in ("diff_vs_reference", "diff_vs_reference_lo", "diff_vs_reference_hi"):
        assert (ref[col] == 0.0).all(), col
    assert (ref["diff_vs_reference_p"] == 1.0).all()
    assert (df["reference"] == "m0").all()


def test_no_multiplicity_adjustment_is_applied_across_the_thresholds():
    """35 nested flagged sets are 35 views of ONE curve, not 35 hypotheses. The schema
    carries no adjusted p at all, so nobody can add one without changing the contract."""
    assert "p_adjusted" not in em.NET_BENEFIT_COLUMNS
    assert "fdr_method" not in em.NET_BENEFIT_COLUMNS
    assert "benjamini_hochberg" not in inspect.getsource(em.build_net_benefit)
    assert "pointwise" in em.NET_BENEFIT_MULTIPLICITY_NOTE
    assert "NO multiplicity adjustment" in em.NET_BENEFIT_MULTIPLICITY_NOTE


def test_an_arm_the_ladder_has_not_trained_is_skipped_rather_than_faked():
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "not_trained", "m0"], "m0")
    df = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=10), QUIET,
                              split="val")
    assert df["arm"].unique().tolist() == ["a", "m0"]
    assert list(df.columns) == em.NET_BENEFIT_COLUMNS


def test_a_missing_reference_writes_the_schema_rather_than_half_a_table():
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m9"], "m9")
    df = em.build_net_benefit(cfg, {"a": arms["a"]}, make_nb_engine(roster, cfg, n_boot=10),
                              QUIET, split="val")
    assert df.empty and list(df.columns) == em.NET_BENEFIT_COLUMNS


# --------------------------------------------------------------------------- #
# 12g. The convergence gate, reused rather than reinvented.                    #
# --------------------------------------------------------------------------- #
def _conv(arm: str, status: str) -> pd.DataFrame:
    return pd.DataFrame([dict(arm=arm, n_seeds=5, train_nll_drop=1e-6, val_overfit_gap=0.9,
                              status=status, reason="stated reason")],
                        columns=em.CONVERGENCE_COLUMNS)


def test_the_gate_is_the_one_function_the_contrasts_use_with_its_defaults_unchanged():
    """Parameterised, not copied. If the defaults move, ``val_comparisons.csv`` moves."""
    assert "suppress_unfit_contrasts(" in inspect.getsource(em.build_net_benefit)
    assert "suppress_unfit_contrasts(" in inspect.getsource(em.build_comparisons)
    p = inspect.signature(em.suppress_unfit_contrasts).parameters
    assert p["arm_keys"].default == ("model", "reference")
    assert p["blank_keys"].default == ("difference", "ci_lo", "ci_hi", "p_two_sided",
                                       "p_adjusted")
    assert p["flag_key"].default is None


def test_the_gate_refuses_to_invent_a_column_the_row_does_not_carry():
    """Its blanking loop ASSIGNS, so a wrong key set would add a column and break
    write_table's exact column-order assert at the far end of the run."""
    row = dict(family="f", model="good", reference="bad", note="")
    with pytest.raises(AssertionError, match="does not carry"):
        em.suppress_unfit_contrasts([row], _conv("bad", em.STATUS_NO_CONVERGE), QUIET,
                                    blank_keys=("no_such_column",))


def test_a_row_whose_arm_is_its_own_reference_states_the_reason_once():
    """The decision-curve reference's own row names the same arm twice."""
    row = dict(arm="bad", reference="bad", note="", net_benefit=1.0, suppressed=False)
    out = em.suppress_unfit_contrasts([row], _conv("bad", em.STATUS_NO_CONVERGE), QUIET,
                                      arm_keys=("arm", "reference"),
                                      blank_keys=("net_benefit",), flag_key="suppressed")[0]
    assert out["note"].count("stated reason") == 1
    assert out["suppressed"] is True and np.isnan(out["net_benefit"])


def test_a_curve_from_an_arm_that_never_fitted_a_model_is_suppressed_on_every_split():
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m0"], "m0")
    for sealed in (False, True):
        df = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=10), QUIET,
                                  split=("test" if sealed else "val"),
                                  convergence=_conv("a", em.STATUS_NO_CONVERGE), sealed=sealed)
        assert df.loc[df["arm"] == "a", "suppressed"].all(), sealed
        assert not df.loc[df["arm"] == "m0", "suppressed"].any(), sealed


def test_severe_overfit_suppresses_the_curve_on_validation_and_not_on_the_sealed_split():
    """That flag exists because a validation metric taken at a validation-selected epoch is
    circular. The sealed split took no part in that selection."""
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m0"], "m0")
    conv = _conv("a", em.STATUS_OVERFIT)
    on_val = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=10), QUIET,
                                  split="val", convergence=conv, sealed=False)
    on_test = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=10), QUIET,
                                   split="test", convergence=conv, sealed=True)
    assert on_val.loc[on_val["arm"] == "a", "suppressed"].all()
    assert not on_test["suppressed"].any()


def test_a_suppressed_curve_loses_every_estimate_and_keeps_every_count():
    """Same precedent as a suppressed subgroup: the row stays and says why. Every column
    that estimates net benefit goes, INCLUDING the IPCW sensitivity column - a row whose
    sensitivity column still reads +0.09 has not been suppressed - and INCLUDING
    ``km_risk_above``, which is a Kaplan-Meier fit on the set the model chose to flag and is
    the last term of the identity that reproduces the estimate (see the reconstruction test
    below). The descriptive counts stay, and so does treat-all: it flags everyone, so it
    carries no model output at all.
    """
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m0"], "m0")
    df = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=10), QUIET,
                              split="val", convergence=_conv("a", em.STATUS_NO_CONVERGE))
    bad = df[df["arm"] == "a"]
    for col in em.NET_BENEFIT_SUPPRESSED_KEYS:
        assert bad[col].isna().all(), col
    assert "net_benefit_ipcw" in em.NET_BENEFIT_SUPPRESSED_KEYS
    assert "km_risk_above" in em.NET_BENEFIT_SUPPRESSED_KEYS, (
        "F_A is not a descriptive count; with threshold, n_above and n_scored it IS the "
        "estimate, so a row that keeps it has not been suppressed")
    for col in ("n_scored", "n_above", "events_above", "km_last_obs_day",
                "n_paired", "nb_treat_all_same_set"):
        assert bad[col].notna().all(), col
    assert bad["note"].str.contains("SUPPRESSED").all()
    assert bad["note"].str.contains("stated reason").all()
    assert list(df.columns) == em.NET_BENEFIT_COLUMNS, "the gate must not add a column"


def nb_reconstruct(frame: pd.DataFrame) -> np.ndarray:
    """Net benefit rebuilt from published columns alone - the estimator's own identity.

    ``net_benefit_at`` computes TP and FP in PATIENT units from one Kaplan-Meier fit:
    ``TP = F_A * n_A``, ``FP = n_A - TP``, ``NB = (TP - FP * w) / n`` with
    ``w = p_t / (1 - p_t)``. So four columns of the CSV - ``threshold``, ``km_risk_above``,
    ``n_above``, ``n_scored`` - reproduce the point estimate exactly, and breaking that is
    what suppression has to mean.
    """
    p = frame["threshold"].to_numpy(dtype=float)
    w = p / (1.0 - p)
    f = frame["km_risk_above"].to_numpy(dtype=float)
    share = frame["n_above"].to_numpy(dtype=float) / frame["n_scored"].to_numpy(dtype=float)
    return f * share - (1.0 - f) * share * w


def test_a_suppressed_row_does_not_permit_the_reconstruction_of_its_estimate():
    """THE property, not the column list: a row is suppressed only when its number is gone.

    The defect this pins. The gate blanked the estimate columns and deliberately kept
    ``km_risk_above`` beside the counts, on the reading that F_A is descriptive. It is not:
    it is a Kaplan-Meier fit on the set the MODEL chose to flag, and it is the last term of
    the estimator's own identity. On the sealed table those four surviving columns returned
    ``net_benefit`` to 7.5e-7 and ``diff_vs_treat_all`` to 9.0e-7 on every one of the 140
    rows - the file is written rounded to six decimals, so that is exact to the last digit
    anyone can read. A validation render suppresses m2_frontal and m4_fusion for
    severe_overfit, so ``val_net_benefit.csv`` would have published precisely the estimates
    the gate had just withheld.

    A column-list assertion cannot catch that, because no single column carried the
    estimate. So this test ATTEMPTS the reconstruction: it proves the identity is real on an
    unsuppressed row, then requires the same arithmetic to fail on a suppressed one.
    """
    roster, arms = _nb_fixture()
    cfg = nb_config(["a", "m0"], "m0")
    live = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=10), QUIET,
                                split="val")
    gated = em.build_net_benefit(cfg, arms, make_nb_engine(roster, cfg, n_boot=10), QUIET,
                                 split="val", convergence=_conv("a", em.STATUS_NO_CONVERGE))
    truth = live[live["arm"] == "a"].sort_values("threshold_pct").reset_index(drop=True)
    blank = gated[gated["arm"] == "a"].sort_values("threshold_pct").reset_index(drop=True)
    assert len(truth) == 35 and blank["suppressed"].all()
    withheld = truth["net_benefit"].to_numpy(dtype=float)

    # 1. The identity is REAL, or the rest of this test asserts nothing at all: on the
    #    unsuppressed rows it returns the published estimate to the last bit.
    assert np.allclose(nb_reconstruct(truth), withheld, rtol=0, atol=1e-15), (
        "the reconstruction this test tries to defeat does not reproduce a live estimate, "
        "so it is testing the wrong arithmetic")

    # 2. The same four columns of the SUPPRESSED row do not evaluate at all.
    assert np.isnan(nb_reconstruct(blank)).all(), (
        "the estimator's identity still evaluates on a suppressed row: the withheld net "
        "benefit is one line of arithmetic away in the published CSV")

    # 3. And nothing that survives closes it. Wherever the rule is a real decision - it
    #    flags somebody, and not everybody - the surviving columns place the estimate
    #    somewhere in [-(n_A/n) * w, n_A/n], which is an interval and not a value: at least
    #    1e-3 wide against a file that publishes six decimals.
    decision = ((blank["n_above"] > 0) & (blank["n_above"] < blank["n_scored"])).to_numpy()
    assert int(decision.sum()) >= 20, "the fixture must exercise mostly genuine decisions"
    w = (blank["threshold"] / (1.0 - blank["threshold"])).to_numpy(dtype=float)
    share = (blank["n_above"] / blank["n_scored"]).to_numpy(dtype=float)
    lo, hi = -share * w, share                      # F_A = 0 and F_A = 1, the whole of it
    assert ((lo <= withheld) & (withheld <= hi))[decision].all()
    assert ((hi - lo)[decision] > 1e-3).all(), (
        "the surviving counts pin the withheld estimate to within the precision the file "
        "publishes, so blanking the estimate columns achieved nothing")
    for col in [c for c in em.NET_BENEFIT_COLUMNS
                if c not in em.NET_BENEFIT_SUPPRESSED_KEYS]:
        got = pd.to_numeric(blank[col], errors="coerce").to_numpy(dtype=float)
        assert not np.any(np.abs(got - withheld)[decision] <= 1e-6), (
            f"{col} reproduces the withheld estimate to six decimals on a suppressed row")

    # The two degenerate thresholds are excluded above KNOWINGLY, not overlooked, and are
    # asserted here so that stays true. At neither of them has the model made a decision:
    # below its minimum predicted risk it flags everyone and its curve IS treat-all, and
    # above its maximum it flags nobody and the rule has become treat-none, whose net
    # benefit is 0 by construction rather than an estimate.
    everyone = truth[blank["n_above"].to_numpy() == blank["n_scored"].to_numpy()]
    assert np.allclose(everyone["net_benefit"], everyone["nb_treat_all_same_set"])
    assert (truth.loc[blank["n_above"].to_numpy() == 0, "net_benefit"] == 0.0).all()


def test_the_counts_a_suppressed_row_keeps_do_not_stand_in_for_the_risk_it_loses():
    """Why ``events_above`` and ``n_above`` survive the gate, measured rather than argued.

    The obvious substitute for the blanked F_A is the naive rate ``events_above/n_above``,
    and on this cohort it is not one: 63.8% of the split is censored before day 1825, so the
    naive rate runs materially below the Kaplan-Meier value at every threshold and in the
    same direction. Feeding it through the estimator's identity therefore yields a
    different, biased estimator - one any reader could compute from any censored cohort -
    rather than a recovery of this arm's curve.
    """
    df = _written_nb(SEALED_SPLIT)
    p = df["threshold"].to_numpy(dtype=float)
    w = p / (1.0 - p)
    share = df["n_above"].to_numpy(dtype=float) / df["n_scored"].to_numpy(dtype=float)
    naive_f = df["events_above"].to_numpy(dtype=float) / df["n_above"].to_numpy(dtype=float)
    assert (df["n_above"] > 0).all(), "no flagged set is empty on the sealed table"

    # the exact identity, on a table with nothing suppressed: this is what the gate breaks
    assert np.allclose(nb_reconstruct(df), df["net_benefit"].to_numpy(dtype=float), atol=1e-6)

    gap = df["km_risk_above"].to_numpy(dtype=float) - naive_f
    assert (gap > 0.04).all(), (
        "the naive event rate is not below the Kaplan-Meier risk at every threshold, so "
        "censoring is not doing the work this justification rests on")
    naive_nb = naive_f * share - (1.0 - naive_f) * share * w
    err = np.abs(naive_nb - df["net_benefit"].to_numpy(dtype=float))
    assert err.min() > 1e-3 and np.median(err) > 0.04, (
        f"the naive rate reconstructs net benefit to within {err.min():.2g}; keeping "
        f"events_above beside n_above would then republish the suppressed estimate")


# --------------------------------------------------------------------------- #
# 12h. The engine on the REAL artefacts, and the table it actually wrote.      #
# --------------------------------------------------------------------------- #
def _real_nb_frame(split: str, n_boot: int = 8) -> pd.DataFrame:
    """``build_net_benefit`` on the real arms. ``n_boot`` is small because the POINT
    estimates and the denominators - what these tests check - do not depend on it."""
    roster, scores = _real_dca(split)
    cfg = load_config(DEFAULT_CONFIG)
    s = em.net_benefit_settings(cfg)
    eng = em.NetBenefitEngine(roster, em.bootstrap_draw(len(roster), n_boot, 1),
                              s["thresholds"], s["horizon_days"], s["sparse_events_min"],
                              QUIET)
    return em.build_net_benefit(cfg, scores, eng, QUIET, split=split,
                                sealed=(split == SEALED_SPLIT))


def test_every_arm_is_measured_on_its_own_patients_and_never_harmonised():
    """741 / 740 / 734 / 707, with 106 / 106 / 98 events. Restricting the ladder to the 707
    every arm scores would discard 8 of 106 events for nothing."""
    df = _real_nb_frame(SEALED_SPLIT)
    assert list(df.columns) == em.NET_BENEFIT_COLUMNS
    assert len(df) == len(NB_DCA_ARMS) * 35
    assert df.groupby("arm", sort=False)["n_scored"].first().to_dict() == {
        "m2_frontal": 734, "m1": 707, "m0": 741, "m4_fusion": 740}
    assert df["n_scored"].nunique() > 1, "the arms have been harmonised onto one population"
    assert df["net_benefit"].notna().all(), (
        "every degenerate case is DEFINED - an empty flagged set is 0 and a flagged set with "
        "no observed event is the negative floor - so no threshold may be NaN")


def test_the_engine_reproduces_the_measured_curves_section_12da_pins():
    """The same numbers, from the engine rather than from a hand-built loop: each arm on its
    own ``present`` set, ``n_scored = present.sum()``, and ``ArmScores.risk[horizon]`` with
    the frozen recalibration on the image arms."""
    at20 = _real_nb_frame(SEALED_SPLIT)
    at20 = at20[at20["threshold_pct"] == 20].set_index("arm")
    for arm, (km, ipcw) in NB_MEASURED_AT_20["test"].items():
        assert float(at20.loc[arm, "net_benefit"]) == pytest.approx(km, abs=NB_TOL), arm
        assert float(at20.loc[arm, "net_benefit_ipcw"]) == pytest.approx(ipcw, abs=NB_TOL), arm
    for arm, (km, _) in NB_MEASURED_CONTRAST_AT_20["test"].items():
        assert float(at20.loc[arm, "diff_vs_reference"]) == pytest.approx(km, abs=NB_TOL), arm


def test_the_engine_measures_the_recalibrated_risks_and_that_choice_moves_the_curve():
    """The defect this pins actually happened: measuring on unrecalibrated risks produced an
    agreement figure that was wrong by a factor of two and nearly reached the manuscript.

    The frozen cloglog transform (slope 1.562) is strictly monotone, so it does not reorder
    a single patient and the FAMILY of flagged sets is unchanged - but at a GIVEN threshold
    the flagged set is different, and a decision curve is read at given thresholds. Measured
    on the sealed split over the plotted range, it moves m2_frontal's curve by up to
    **0.0319** (at p_t = 0.24), which is four times that arm's whole KM-versus-IPCW
    disagreement (0.0078) - the estimator choice this study writes three paragraphs about.
    """
    roster, scores = _real_dca(SEALED_SPLIT)
    cfg = load_config(DEFAULT_CONFIG)
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    sc = scores["m2_frontal"]
    assert sc.recalibration is not None, "the image arms carry a frozen recalibration"
    with np.load(coh / f"{SEALED_SPLIT}_hazards_m2_frontal.npz", allow_pickle=False) as z:
        hazards = np.asarray(z["hazards"], dtype=float)
        pids = np.asarray(z["empi_anon"]).astype(str)
        edges = np.asarray(z["edges"], dtype=float)
    raw = np.full(len(roster), np.nan)
    raw[roster.positions_of(pids)] = tm.risk_at_horizon(hazards, NB_HORIZON, edges=edges)

    m = sc.present
    grid = em.threshold_grid(1, 35)
    at20 = int(np.flatnonzero(np.rint(grid * 100).astype(int) == 20)[0])

    def curve(r):
        return np.array([row["net_benefit"] for row in em.net_benefit_curve(
            roster.time[m], roster.event[m], r[m], grid, horizon=NB_HORIZON,
            sparse_events_min=15, n_scored=int(m.sum()), treat_all=False)])

    recal, plain = curve(sc.risk[int(NB_HORIZON)]), curve(raw)
    assert float(recal[at20]) == pytest.approx(NB_MEASURED_AT_20["test"]["m2_frontal"][0],
                                               abs=NB_TOL)
    # strictly monotone: not one patient is reordered, so the family of flagged sets is the
    # same family - which is exactly why the shift below is horizontal and not a rescaling.
    m_take = np.flatnonzero(m)
    assert np.array_equal(np.argsort(raw[m_take], kind="mergesort"),
                          np.argsort(sc.risk[int(NB_HORIZON)][m_take], kind="mergesort"))
    plotted = grid <= 0.30
    moved = float(np.abs(recal - plain)[plotted].max())
    assert moved == pytest.approx(0.0319, abs=NB_TOL), (
        f"the recalibration moves m2_frontal's curve by {moved:.4f} over the plotted range, "
        f"not the 0.0319 measured; if it has stopped moving it, check that ArmScores.risk "
        f"still carries the frozen transform")
    assert moved > NB_MEASURED_MAX_GAP["test"]["m2_frontal"], (
        "recalibration is meant to matter MORE than the KM-versus-IPCW estimator choice "
        "this study argues about at length; if it no longer does, re-measure both")
    got = _real_nb_frame(SEALED_SPLIT)
    got = got[got["arm"] == "m2_frontal"].sort_values("threshold_pct")["net_benefit"]
    assert np.allclose(got.to_numpy(), recal), "the engine is not reading ArmScores.risk"


# The headline probe, MEASURED from the written table with the pre-specified 2,000-replicate
# draw. threshold_pct -> (diff vs treat-all, diff vs m0) for m2_frontal on the sealed split.
NB_PROBE_M2_TEST = {12: (0.031337, 0.018107), 18: (0.078110, 0.036918),
                    20: (0.086360, 0.049474), 26: (0.135385, 0.061731)}
# and its pointwise 95% intervals at the headline threshold
NB_PROBE_M2_CI_AT_20 = {"diff_vs_treat_all": (0.050489, 0.119726),
                        "diff_vs_reference": (0.016951, 0.081559)}


def _written_nb(split: str) -> pd.DataFrame:
    path = em.split_path(load_config(DEFAULT_CONFIG), "net_benefit_csv", split)
    if not path.exists():
        pytest.skip(f"{path.name} has not been written in this checkout")
    return pd.read_csv(path)


def test_the_written_table_is_the_four_arms_by_thirty_five_thresholds():
    df = _written_nb(SEALED_SPLIT)
    assert list(df.columns) == em.NET_BENEFIT_COLUMNS
    assert df["arm"].tolist() == [a for a in NB_DCA_ARMS for _ in range(35)]
    assert df["threshold_pct"].tolist() == list(range(1, 36)) * len(NB_DCA_ARMS)
    assert (df["split"] == SEALED_SPLIT).all()
    assert (df["horizon_days"] == 1825).all()
    assert (df["reference"] == NB_DCA_REFERENCE).all()
    assert not df["suppressed"].any(), (
        "nothing is suppressed on the sealed split: only did_not_converge suppresses there "
        "and no decision-curve arm carries it")
    roster, _ = _real_dca(SEALED_SPLIT)
    em.assert_aggregate_only(df, roster.pids, "test_net_benefit.csv")


def test_the_written_table_reproduces_the_measured_probe():
    """The numbers the plan probed before any of this was written, to the last digit the
    file carries. They are deterministic: one seeded draw, fixed rounding, no timestamp."""
    df = _written_nb(SEALED_SPLIT)
    m2 = df[df["arm"] == "m2_frontal"].set_index("threshold_pct")
    for pct, (vs_all, vs_ref) in NB_PROBE_M2_TEST.items():
        assert float(m2.loc[pct, "diff_vs_treat_all"]) == pytest.approx(vs_all, abs=1e-6), pct
        assert float(m2.loc[pct, "diff_vs_reference"]) == pytest.approx(vs_ref, abs=1e-6), pct
    for col, (lo, hi) in NB_PROBE_M2_CI_AT_20.items():
        assert float(m2.loc[20, f"{col}_lo"]) == pytest.approx(lo, abs=1e-6), col
        assert float(m2.loc[20, f"{col}_hi"]) == pytest.approx(hi, abs=1e-6), col
    assert (m2["n_replicates_valid"] == em.PROTOCOL_BOOTSTRAP_N).all(), (
        "every replicate is estimable for m2_frontal; a shortfall would have to be stated "
        "in the note column")


def test_main_writes_the_net_benefit_table_through_split_path():
    src = inspect.getsource(em.main)
    assert 'NetBenefitEngine(' in src and 'build_net_benefit(' in src
    assert '_out("net_benefit_csv")' in src, (
        "the decision curve must be written through split_path like every other output, so "
        "the sealed run writes test_net_benefit.csv beside the validation one")
    assert "NET_BENEFIT_COLUMNS" in src


# =========================================================================== #
# 12. V6 REVISION - A3 IMAGING ROBUSTNESS STRATA                               #
#                                                                              #
# Added 2026-08-11 with the config-driven family declaration. Everything here  #
# is an EXTENSION: no test above was modified, and the published equity table  #
# is asserted to be unchanged rather than merely believed to be.               #
# =========================================================================== #
import src.subgroups as sgm                                       # noqa: E402
from pathlib import Path as _Path                                 # noqa: E402

TEST_EVENTS = 106            # the sealed split's event count
TEST_CONTROLS = 162          # patients observed event-free beyond 1,825 days

#: Per-arm train-validation gap AT THE RETAINED CHECKPOINT, independently recomputed from
#: outputs/tables/train_history.csv during the v6 defect pass and quoted in the T5 brief.
#: These are the numbers a supplementary learning-curve figure must be consistent with.
GAP_AT_RETAINED = {
    "m2_frontal": 0.027812, "m3_image": 0.052354, "m4_frontal": 0.023332,
    "m4_fusion": 0.051978, "m0d_clinical": 0.035980, "m1_klg": 0.052365,
    "r1_densenet_frontal": 0.009344,
}


def _robustness_frame_stub(n: int = 12) -> pd.DataFrame:
    """A frame carrying every column a robustness rule can name."""
    return pd.DataFrame({
        "empi_anon": [f"{800000 + i}" for i in range(n)],
        "sex": "Female", "race": "Asian", "age_at_index": 70.0, "obesity": 0,
        "weight_bearing_frontal": True, "view_set": "frontal",
        "side_source": ["coded", "recovered"] * (n // 2),
        "acquisition_year": np.linspace(2007, 2021, n).round().astype(int),
        "image_masked_pct_max": np.where(np.arange(n) % 2 == 0, 0.2275, 0.31),
        "image_crop_confidence_min": np.where(np.arange(n) % 3 == 0, 0.7, 1.0),
        "image_crop_method_any_intensity_profile": (np.arange(n) % 3 == 0),
        "image_inverted_any": (np.arange(n) % 4 == 0),
        "image_half_selected_any": (np.arange(n) % 5 != 0),
    })


# --- the refactor: one declaration, two consumers --------------------------- #
def test_subgroup_levels_now_reads_the_shared_declaration():
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(80, 12)
    from_eval = [(s, lv) for s, lv, _ in em.subgroup_levels(cfg, roster.frame)]
    from_config = [(f.report_label, l.report_label)
                   for f in sgm.load_families(cfg, "equity") for l in f.levels]
    assert from_eval == from_config
    assert em.family_mask is sgm.family_mask and em.load_families is sgm.load_families


def test_the_equity_table_is_unchanged_by_the_refactor():
    """The thirteen published rows, in the published order, with no additions."""
    path = _Path("outputs/tables/test_subgroups.csv")
    if not path.exists():                                    # pragma: no cover
        pytest.skip("outputs/tables/test_subgroups.csv is absent")
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(80, 12)
    got = [(s, lv) for s, lv, _ in em.subgroup_levels(cfg, roster.frame)]
    pub = pd.read_csv(path)[["subgroup", "level"]].apply(tuple, axis=1).tolist()
    assert got == pub


def test_a_robustness_family_never_leaks_into_the_equity_scope():
    cfg = load_config(DEFAULT_CONFIG)
    equity = {f.key for f in sgm.load_families(cfg, "equity")}
    robust = {f.key for f in sgm.load_families(cfg, "robustness")}
    assert equity == set(sgm.FROZEN_EQUITY_FAMILIES)
    assert robust - equity, "the robustness scope adds nothing"
    assert "acquisition_era_calendar" not in equity


# --- the floor, unweakened -------------------------------------------------- #
def test_the_robustness_table_uses_the_same_floor_and_defines_no_copy():
    src = _Path(em.__file__).read_text(encoding="utf-8")
    assert "SUPPRESS_BELOW_EVENTS = " not in src, (
        "the floor is imported from src.model_clinical and never redefined here")
    assert src.count("suppress_below_events") >= 2
    cfg = load_config(DEFAULT_CONFIG)
    assert int(cfg["model_eval"]["suppress_below_events"]) == em.SUPPRESS_BELOW_EVENTS == 50


def test_a_lowered_floor_is_refused_by_the_robustness_builder():
    bad = load_config(DEFAULT_CONFIG)
    bad["model_eval"] = dict(bad["model_eval"])
    bad["model_eval"]["suppress_below_events"] = 5
    roster = _subgroup_roster(60, 10)
    with pytest.raises(AssertionError, match="protocol section 21"):
        em.build_imaging_robustness(bad, roster, {}, make_engine(roster, 5), 1825, QUIET)


def test_below_the_floor_a_cell_is_suppressed_with_the_section_21_reason():
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(120, 20, seed=5)
    fam = sgm.load_families(cfg, "robustness")[0]
    m = np.ones(len(roster), dtype=bool)
    row = em._robustness_row(fam, fam.levels[0], "m2_frontal", "not applicable", m, roster,
                             make_arm("m2_frontal", roster), make_engine(roster, 5), 1825,
                             em.SUPPRESS_BELOW_EVENTS, em.PRIMARY_METRIC)
    assert row["suppressed"] and row["suppression_reason"].startswith("protocol section 21")
    assert np.isnan(row["estimate"]) and np.isnan(row["ci_lo"])


def test_no_partition_of_the_sealed_split_can_leave_three_estimable_levels():
    """Stated as a test because it is why every three-level era scheme mostly suppresses."""
    assert 3 * em.SUPPRESS_BELOW_EVENTS > TEST_EVENTS
    assert 2 * em.SUPPRESS_BELOW_EVENTS <= TEST_EVENTS      # two is arithmetically possible


# --- a second, different reason a cell has no number ------------------------ #
def test_a_level_with_no_control_beyond_the_horizon_is_suppressed_as_not_estimable():
    """An IPCW AUROC needs a case AND a control. Neither the floor nor a wide interval
    describes a level that has zero patients observed past the horizon."""
    cfg = load_config(DEFAULT_CONFIG)
    n = 200
    pids = np.array([f"{700000 + i}" for i in range(n)], dtype="<U8")
    event = np.zeros(n, dtype=int); event[:60] = 1
    time = np.where(event == 1, 900.0, 1200.0)               # nobody survives past 1825
    frame = pd.DataFrame({"empi_anon": pids, "split": "val", "time_from_landmark": time,
                          "event_indicator": event, "sex": "Female", "race": "Asian",
                          "age_at_index": 70.0, "obesity": 0,
                          "weight_bearing_frontal": True, "view_set": "frontal"})
    g_grid, g_vals = tm.reverse_km(time, event)
    roster = em.Roster(pids=pids, time=time, event=event, frame=frame,
                       g_grid=g_grid, g_vals=g_vals)
    fam = sgm.load_families(cfg, "robustness")[0]
    row = em._robustness_row(fam, fam.levels[0], "m2_frontal", "not applicable",
                             np.ones(n, dtype=bool), roster, make_arm("m2_frontal", roster),
                             make_engine(roster, 5), 1825, em.SUPPRESS_BELOW_EVENTS,
                             em.PRIMARY_METRIC)
    assert row["n_events"] == 60 >= em.SUPPRESS_BELOW_EVENTS, "it clears the floor"
    assert row["n_controls_beyond_horizon"] == 0
    assert row["suppressed"] and row["suppression_reason"].startswith("not estimable")
    assert "undefined, not imprecise" in row["suppression_reason"]
    assert np.isnan(row["estimate"])


def test_harrell_c_is_reported_because_it_survives_where_the_auroc_does_not():
    assert em.ROBUSTNESS_METRICS == (em.PRIMARY_METRIC, "harrell_c")
    cfg = load_config(DEFAULT_CONFIG)
    n = 200
    pids = np.array([f"{700000 + i}" for i in range(n)], dtype="<U8")
    event = np.zeros(n, dtype=int); event[:60] = 1
    rng = np.random.default_rng(11)
    time = np.where(event == 1, rng.uniform(100.0, 900.0, n), 1200.0)
    frame = pd.DataFrame({"empi_anon": pids, "split": "val", "time_from_landmark": time,
                          "event_indicator": event, "sex": "Female", "race": "Asian",
                          "age_at_index": 70.0, "obesity": 0,
                          "weight_bearing_frontal": True, "view_set": "frontal"})
    g_grid, g_vals = tm.reverse_km(time, event)
    roster = em.Roster(pids=pids, time=time, event=event, frame=frame,
                       g_grid=g_grid, g_vals=g_vals)
    fam = sgm.load_families(cfg, "robustness")[0]
    row = em._robustness_row(fam, fam.levels[0], "m2_frontal", "not applicable",
                             np.ones(n, dtype=bool), roster, make_arm("m2_frontal", roster),
                             make_engine(roster, 8), 1825, em.SUPPRESS_BELOW_EVENTS,
                             "harrell_c")
    assert not row["suppressed"] and np.isfinite(row["estimate"])
    assert em.HARRELL_C_NOTE in row["note"], "the horizon column is a schema artifact"


def test_the_wide_interval_flag_is_the_same_one_the_a1_a2_module_declared():
    import src.v6_analyses as v6
    assert em.WIDE_INTERVAL_WIDTH == v6.WIDE_INTERVAL_WIDTH == 0.15


# --- the era caveat travels in the table ------------------------------------ #
def test_every_era_row_carries_both_caveats_in_its_own_note():
    cfg = load_config(DEFAULT_CONFIG)
    roster = _subgroup_roster(120, 20, seed=7)
    for fam in sgm.load_families(cfg, "robustness"):
        if not fam.key.startswith("acquisition_era"):
            continue
        row = em._robustness_row(fam, fam.levels[0], "m2_frontal", "not applicable",
                                 np.ones(len(roster), dtype=bool), roster,
                                 make_arm("m2_frontal", roster), make_engine(roster, 5),
                                 1825, em.SUPPRESS_BELOW_EVENTS, em.PRIMARY_METRIC)
        note = row["note"].lower()
        assert "per-patient random shift" in note
        assert "section 17" in note and "not been obtained" in note
        assert "d17" in note and "d35" in note


def test_the_unavailable_strata_table_answers_the_editor_rather_than_dropping_the_request():
    cfg = load_config(DEFAULT_CONFIG)
    df = em.build_metadata_availability(cfg)
    assert list(df.columns) == em.METADATA_AVAILABILITY_COLUMNS
    assert not df["available"].any()
    joined = " ".join(df["stratum"]).lower()
    for want in ("equipment", "manufacturer", "site"):
        assert want in joined
    assert any("horizontal flip" in s.lower() for s in df["stratum"])
    assert all(len(r) > 40 for r in df["reason"])


# --- the write namespace ---------------------------------------------------- #
def test_the_v6_entry_point_can_only_write_v6_files():
    names = set(em.V6_ROBUSTNESS_BASENAMES.values()) | set(em.V6_LEARNING_CURVE_BASENAMES.values())
    assert names and all(n.startswith("v6_") and n.endswith(".csv") for n in names)
    cfg = load_config(DEFAULT_CONFIG)
    published = {em.split_path(cfg, k, s).name
                 for k in ("metrics_csv", "comparisons_csv", "subgroups_csv",
                           "convergence_csv", "net_benefit_csv")
                 for s in ("val", SEALED_SPLIT)}
    assert not (names & published), "a v6 output would overwrite a published table"
    src = inspect.getsource(em.run_v6)
    assert 'name.startswith("v6_")' in src, "the namespace guard must be in the writer"
    assert "_out(" not in src and "split_path(" not in src, (
        "run_v6 must not resolve a published output path at all")


def test_the_v6_subcommand_does_not_change_the_published_entry_point():
    src = _Path(em.__file__).read_text(encoding="utf-8")
    tail = src[src.rindex('if __name__ == "__main__":'):]
    assert "main_v6(sys.argv[2:])" in tail and "SystemExit(main())" in tail
    assert em.V6_SUBCOMMAND == "v6"
    ap_src = inspect.getsource(em.main)
    assert "v6" not in ap_src.split("def main")[0]


# =========================================================================== #
# 13. V6 REVISION - A4 LEARNING CURVES                                         #
# =========================================================================== #
def _history_config(tmp_path, rows: list[dict], patience: int = 8, max_epochs: int = 40):
    p = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    cfg = load_config(DEFAULT_CONFIG)
    cfg["model_image"] = dict(cfg["model_image"])
    cfg["model_image"]["early_stopping"] = {"monitor": "val_nll", "patience": patience}
    cfg["model_image"]["max_epochs"] = max_epochs
    cfg["model_image"]["local"] = dict(cfg["model_image"]["local"])
    cfg["model_image"]["local"]["history_csv"] = str(p)
    cfg["model_eval"] = dict(cfg["model_eval"])
    return cfg


def _series(arm="a", seed=1, best=3, patience=8, train0=0.60, val0=0.65):
    """A history whose validation minimum is at ``best`` and which then runs ``patience``
    more epochs, exactly as early stopping produces."""
    rows = []
    for e in range(best + patience + 1):
        val = val0 - 0.01 * min(e, best) + 0.02 * max(0, e - best)
        train = train0 - 0.01 * e
        rows.append(dict(arm=arm, seed=seed, epoch=e, train_nll=train, val_nll=val,
                         lr=1e-4, secs=1.0, improved=(e <= best)))
    return rows


def test_the_retained_epoch_is_marked_on_exactly_one_row_per_series(tmp_path):
    cfg = _history_config(tmp_path, _series(best=3) + _series(seed=2, best=5))
    curves, per_seed, per_arm = em.build_learning_curves(cfg, QUIET)
    assert list(curves.columns) == em.LEARNING_CURVE_COLUMNS
    assert list(per_seed.columns) == em.LEARNING_CURVE_SEED_COLUMNS
    assert list(per_arm.columns) == em.LEARNING_CURVE_ARM_COLUMNS
    for (_arm, _seed), g in curves.groupby(["arm", "seed"]):
        assert int(g["is_retained_epoch"].sum()) == 1
        assert int(g.loc[g["is_retained_epoch"], "val_nll"].iloc[0] * 1e9) == \
            int(g["val_nll"].min() * 1e9)
    assert per_seed["retained_epoch"].tolist() == [3, 5]
    assert per_seed["epochs_after_retained"].tolist() == [8, 8]


def test_epochs_from_retained_is_signed_so_a_figure_can_align_the_series(tmp_path):
    cfg = _history_config(tmp_path, _series(best=3))
    curves, _s, _a = em.build_learning_curves(cfg, QUIET)
    assert curves["epochs_from_retained"].min() == -3
    assert curves["epochs_from_retained"].max() == 8
    assert (curves.loc[curves["is_retained_epoch"], "epochs_from_retained"] == 0).all()


def test_the_overfit_gap_is_not_the_train_validation_gap_and_the_table_says_so(tmp_path):
    cfg = _history_config(tmp_path, _series(best=3))
    _c, per_seed, per_arm = em.build_learning_curves(cfg, QUIET)
    r = per_seed.iloc[0]
    assert r["gap_at_retained"] != r["val_overfit_gap_last_minus_min"]
    assert r["val_overfit_gap_last_minus_min"] == pytest.approx(0.02 * 8, abs=1e-9)
    for note in (r["note"], per_arm.iloc[0]["note"]):
        assert "val_nll[last] - min(val_nll)" in note
        assert "patience=8" in note
    assert "LOWER BOUND" in per_arm.iloc[0]["note"]
    assert "dropout" in per_arm.iloc[0]["note"] and "augmentation" in per_arm.iloc[0]["note"]


def test_a_history_whose_last_improved_epoch_is_not_the_minimum_is_refused(tmp_path):
    rows = _series(best=3)
    rows[6]["improved"] = True                    # a later epoch claims improvement
    cfg = _history_config(tmp_path, rows)
    with pytest.raises(AssertionError, match="retained-epoch rule is wrong"):
        em.build_learning_curves(cfg, QUIET)


def test_a_run_that_stopped_on_neither_patience_nor_max_epochs_is_refused(tmp_path):
    rows = _series(best=3)[:-2]                   # truncated: 6 epochs past the minimum
    cfg = _history_config(tmp_path, rows)
    with pytest.raises(AssertionError, match="neither the configured patience"):
        em.build_learning_curves(cfg, QUIET)


# --- against the real artefacts --------------------------------------------- #
def _real_history_cfg():
    cfg = load_config(DEFAULT_CONFIG)
    return cfg if cfg.path(cfg["model_image"]["local"]["history_csv"]).exists() else None


def test_every_real_series_stopped_exactly_patience_epochs_past_the_checkpoint():
    cfg = _real_history_cfg()
    if cfg is None:                                          # pragma: no cover
        pytest.skip("outputs/tables/train_history.csv is absent")
    _c, per_seed, _a = em.build_learning_curves(cfg, QUIET)
    patience = int(cfg["model_image"]["early_stopping"]["patience"])
    assert len(per_seed) == 35, "7 arms x 5 seeds"
    assert (per_seed["epochs_after_retained"] == patience).all(), (
        "this is the whole reason val_overfit_gap is not a train-validation gap")


def test_the_per_arm_gap_at_the_retained_checkpoint_matches_the_verified_values():
    cfg = _real_history_cfg()
    if cfg is None:                                          # pragma: no cover
        pytest.skip("outputs/tables/train_history.csv is absent")
    _c, _s, per_arm = em.build_learning_curves(cfg, QUIET)
    got = dict(zip(per_arm["arm"], per_arm["mean_gap_at_retained"]))
    assert set(got) == set(GAP_AT_RETAINED)
    for arm, want in GAP_AT_RETAINED.items():
        assert got[arm] == pytest.approx(want, abs=5e-6), arm


def test_the_published_val_overfit_gap_is_reproduced_not_reinterpreted():
    cfg = _real_history_cfg()
    path = _Path("outputs/tables/test_convergence.csv")
    if cfg is None or not path.exists():                     # pragma: no cover
        pytest.skip("train_history.csv or test_convergence.csv is absent")
    _c, _s, per_arm = em.build_learning_curves(cfg, QUIET)
    pub = pd.read_csv(path).set_index("arm")
    for _, r in per_arm.iterrows():
        assert float(r["mean_val_overfit_gap"]) == pytest.approx(
            float(pub.loc[r["arm"], "val_overfit_gap"]), abs=1e-6), r["arm"]
        assert str(r["convergence_status"]) == str(pub.loc[r["arm"], "status"])


def test_the_gap_at_the_retained_checkpoint_is_far_smaller_than_the_published_one():
    """The defect the supplementary figure exists to correct: 'the training split was
    memorised' rests on last-minus-min, which is a different quantity."""
    cfg = _real_history_cfg()
    if cfg is None:                                          # pragma: no cover
        pytest.skip("outputs/tables/train_history.csv is absent")
    _c, _s, per_arm = em.build_learning_curves(cfg, QUIET)
    image = per_arm[per_arm["convergence_status"] == em.STATUS_OVERFIT]
    assert len(image) == 4, "the four severe_overfit arms"
    assert (image["mean_gap_at_retained"] < image["mean_val_overfit_gap"]).all()
    assert image["mean_gap_at_retained"].max() < 0.06
