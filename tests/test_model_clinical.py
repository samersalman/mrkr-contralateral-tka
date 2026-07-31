"""Unit tests for the PURE helpers in src/model_clinical.py — synthetic inputs only.

Covers the four things that would silently corrupt M0 or breach the seal:
the sealed-test guard, the TRAIN-ONLY spline-knot rule, the IPCW weight computation
(checked against a fully hand-worked reverse-Kaplan-Meier example), and the calibration
binning. No patient data is read: the split-guard tests write a synthetic Parquet into a
tmp_path, and the config-contract tests read config/feasibility.yaml only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG, load_config
from src.model_clinical import (
    CALIBRATION_BINS,
    CV_N_REPEATS,
    CV_N_SPLITS,
    DEV_SPLITS,
    EMPHASISE_CI_BELOW_EVENTS,
    EXPECTED_IDENTIFIED_PARAMS,
    EXPECTED_M1_IDENTIFIED_PARAMS,
    EXPECTED_M1_N_PARAMS,
    EXPECTED_N_PARAMS,
    EXPECTED_SPLIT_EVENTS,
    EXPECTED_SPLIT_N,
    SEALED_SPLIT,
    SELECTION_LABELS,
    SELECTION_METRICS,
    SUPPRESS_BELOW_EVENTS,
    aliased_columns,
    build_design,
    calibration_slope_intercept,
    censoring_curve,
    clamp_horizon_days,
    cloglog,
    cv_penalizer_stability,
    design_identifiability,
    fit_age_spline,
    fit_m1_klg,
    harrell_c,
    ipcw_auc,
    ipcw_labels_weights,
    load_development_frame,
    parquet_num_rows,
    percentile_ci,
    replay_from_json,
    risk_bins,
    select_penalizer,
    spline_basis,
    step_value,
    sums_to_constant,
    suppression,
    tune_and_select_penalizer,
)

# Hand-worked reverse-Kaplan-Meier example, used by several tests below.
#   times  = [1, 2, 3, 4, 5]   events = [1, 0, 1, 0, 1]   (1 = outcome, 0 = censored)
# Flipping the indicator makes censoring the "event": censorings occur at t = 2 and t = 4.
#   G(1) = 1
#   G(2) = 1 * (1 - 1/4) = 0.75      (4 at risk at t=2, 1 censoring)
#   G(3) = 0.75
#   G(4) = 0.75 * (1 - 1/2) = 0.375  (2 at risk at t=4, 1 censoring)
#   G(5) = 0.375
HW_TIMES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
HW_EVENTS = np.array([1, 0, 1, 0, 1])
HW_G = {1.0: 1.0, 2.0: 0.75, 3.0: 0.75, 4.0: 0.375, 5.0: 0.375}


def _frame(splits, n_per=4, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in splits:
        for i in range(n_per):
            rows.append(dict(empi_anon=f"{s}{i}", split=s,
                             age_at_index_imp=float(rng.integers(45, 85)),
                             event_indicator=int(i % 2),
                             time_from_landmark=float(100 * (i + 1))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# SEALED-SPLIT GUARD — a reviewer will check this specifically                  #
# --------------------------------------------------------------------------- #
def test_load_development_frame_refuses_test_rows(tmp_path):
    p = tmp_path / "feat.parquet"
    _frame(["train", "val", "test"]).to_parquet(p, index=False)
    dev = load_development_frame(p)
    assert SEALED_SPLIT not in set(dev["split"]), "test rows reached the model"
    assert set(dev["split"]) == set(DEV_SPLITS)
    assert len(dev) == 8, "exactly the train+val rows should survive"


def test_load_development_frame_never_returns_a_test_row_even_if_only_test_exists(tmp_path):
    p = tmp_path / "only_test.parquet"
    _frame(["test"]).to_parquet(p, index=False)
    dev = load_development_frame(p)
    assert len(dev) == 0, "a test-only table must yield zero rows, not the test rows"


def test_parquet_num_rows_counts_all_rows_without_loading_values(tmp_path):
    p = tmp_path / "feat.parquet"
    _frame(["train", "val", "test"]).to_parquet(p, index=False)
    assert parquet_num_rows(p) == 12
    assert len(load_development_frame(p)) == 8, "the invariant count must not load test rows"


def test_forbid_test_split_false_is_an_explicit_opt_out(tmp_path):
    p = tmp_path / "feat.parquet"
    _frame(["train", "val", "test"]).to_parquet(p, index=False)
    assert len(load_development_frame(p, forbid_test=False)) == 12


def test_config_forbids_the_test_split():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg["model_clinical"]["forbid_test_split"] is True
    assert cfg["model_clinical"]["tuning_split"] == "val", "tuning must never touch test"


# --------------------------------------------------------------------------- #
# SPLINE KNOTS — must come from TRAIN rows only                                 #
# --------------------------------------------------------------------------- #
def test_spline_knots_are_train_min_max_and_median():
    train_age = np.arange(40.0, 90.0)              # 40..89, 50 distinct values
    spec = fit_age_spline(train_age, 3)
    assert spec["lower_bound"] == 40.0 and spec["upper_bound"] == 89.0
    assert spec["knots_fit_on"] == "train"
    assert len(spec["interior_knots"]) == 1
    assert spec["interior_knots"][0] == pytest.approx(64.5)


def test_val_rows_cannot_move_the_knots():
    train_age = np.arange(40.0, 90.0)
    val_age = np.array([95.0, 99.0, 100.0])        # far outside the training range
    spec_train_only = fit_age_spline(train_age, 3)
    spec_contaminated = fit_age_spline(np.concatenate([train_age, val_age]), 3)
    assert spec_train_only["all_knots"] != spec_contaminated["all_knots"], \
        "the test itself is broken if contamination does not move the knots"
    # The operative rule: the basis for the val ages depends only on the train-fitted spec.
    b = spline_basis(val_age, spec_train_only)
    assert list(b.columns) == spec_train_only["basis_columns"]
    assert b.shape == (3, 3)


def test_spline_basis_is_reproducible_from_the_persisted_knots_alone():
    spec = fit_age_spline(np.arange(40.0, 90.0), 3)
    ages = np.array([41.0, 52.5, 70.0, 88.0])
    a = spline_basis(ages, spec).to_numpy()
    # Rebuild the spec from JSON-round-tripped scalars, as T7 would.
    reloaded = {k: spec[k] for k in ("variable", "interior_knots", "lower_bound",
                                     "upper_bound", "basis_columns", "df")}
    b = spline_basis(ages, reloaded).to_numpy()
    assert np.abs(a - b).max() == 0.0


def _synthetic_design(n=60, n_extra=10, seed=1):
    """A design shaped like M0's: one age column + n_extra binary model columns.

    M0 (protocol Table 7) holds 11 model columns — the 11 `<predictor>_imp` columns of
    ``features_clinical.primary_predictors``, with `pain_score_max_missing` dropped as
    exactly aliased and no `klg_contra_*` column (inferred KLG is protocol Table 6's
    secondary comparator and belongs to M1). So: 1 age + 10 others.
    """
    model_columns = ["age_at_index_imp"] + [f"x{i}" for i in range(n_extra)]
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"age_at_index_imp": rng.uniform(40, 89, n)})
    for c in model_columns[1:]:
        df[c] = rng.integers(0, 2, n).astype(float)
    spec = fit_age_spline(df["age_at_index_imp"].to_numpy(), 3)
    return df, spec, model_columns


def test_design_matrix_has_the_prespecified_parameter_count():
    df, spec, model_columns = _synthetic_design()
    assert len(model_columns) == 11, \
        "M0 uses 11 model columns: 11 imputed predictors, the aliased pain indicator " \
        "dropped, and no klg_contra_* column (protocol Table 6/7 put inferred KLG in M1)"
    X = build_design(df, spec, model_columns)
    assert X.shape[1] == EXPECTED_N_PARAMS == 13
    assert list(X.columns)[:3] == spec["basis_columns"], "spline columns must come first"
    assert "age_at_index_imp" not in X.columns, "raw age must be replaced by the basis"


def test_build_design_rejects_missing_values():
    df = pd.DataFrame({"age_at_index_imp": [50.0, 60.0, 70.0], "x": [1.0, np.nan, 0.0]})
    spec = fit_age_spline(df["age_at_index_imp"].to_numpy(), 3)
    with pytest.raises(AssertionError):
        build_design(df, spec, ["age_at_index_imp", "x"])


def test_aliased_columns_finds_perfect_collinearity():
    X = pd.DataFrame({"a": [1.0, 0.0, 1.0, 0.0], "b": [0.0, 1.0, 0.0, 1.0],
                      "c": [1.0, 2.0, 4.0, 3.0]})
    pairs = aliased_columns(X)
    assert len(pairs) == 1 and set(pairs[0][:2]) == {"a", "b"}
    assert pairs[0][2] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# IDENTIFIABILITY — rank, not correlation. A pairwise scan cannot see a         #
# partition of unity, which is exactly the relation the age spline creates.     #
# --------------------------------------------------------------------------- #
def test_pairwise_alias_scan_is_blind_to_a_three_column_partition_of_unity():
    """The failure the rank check exists to catch: three columns summing to 1, no pair."""
    rng = np.random.default_rng(7)
    a = rng.uniform(0.1, 0.5, 40)
    b = rng.uniform(0.1, 0.4, 40)
    X = pd.DataFrame({"p1": a, "p2": b, "p3": 1.0 - a - b, "z": rng.normal(size=40)})
    assert aliased_columns(X) == [], "no PAIR is perfectly correlated — that is the point"
    ident = design_identifiability(X)
    assert ident["n_columns"] == 4
    assert ident["rank"] == 4, "the columns are linearly independent; only the CONSTANT is spanned"
    assert ident["rank_with_intercept"] == 4, "adding an intercept adds nothing"
    assert ident["spans_constant"] is True
    assert ident["identified_parameters"] == 3, "a no-intercept model loses one direction"


def test_design_identifiability_counts_a_rank_deficiency_and_the_constant_separately():
    rng = np.random.default_rng(11)
    a = rng.uniform(0.1, 0.5, 50)
    b = rng.uniform(0.1, 0.4, 50)
    d = rng.integers(0, 2, 50).astype(float)
    # p1+p2+p3 = 1 (constant) AND dup == d (an ordinary rank deficiency).
    X = pd.DataFrame({"p1": a, "p2": b, "p3": 1.0 - a - b, "d": d, "dup": d})
    ident = design_identifiability(X)
    assert ident["n_columns"] == 5
    assert ident["rank"] == 4 and ident["rank_deficiency"] == 1
    assert ident["identified_parameters"] == 3
    assert [set(p[:2]) for p in ident["aliased_column_pairs"]] == [{"d", "dup"}]


def test_full_rank_design_without_a_constant_identifies_every_column():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    ident = design_identifiability(X)
    assert ident["rank"] == 4 and ident["rank_with_intercept"] == 5
    assert ident["spans_constant"] is False
    assert ident["identified_parameters"] == 4


def test_the_cr_spline_basis_really_is_a_partition_of_unity():
    """The concrete fact behind EXPECTED_IDENTIFIED_PARAMS: cr() columns sum to 1."""
    df, spec, model_columns = _synthetic_design()
    X = build_design(df, spec, model_columns)
    assert sums_to_constant(X, spec["basis_columns"])
    s = X[spec["basis_columns"]].to_numpy().sum(axis=1)
    assert np.abs(s - 1.0).max() < 1e-12
    ident = design_identifiability(X)
    assert ident["n_columns"] == EXPECTED_N_PARAMS
    assert ident["spans_constant"] is True
    assert ident["identified_parameters"] == EXPECTED_IDENTIFIED_PARAMS == EXPECTED_N_PARAMS - 1


def test_adding_the_aliased_pain_indicator_costs_a_further_parameter():
    """knee_pain_any_imp + pain_score_max_missing = 1 was the second unidentified direction."""
    df, spec, model_columns = _synthetic_design()
    df["knee_pain_any_imp"] = np.r_[np.ones(40), np.zeros(20)]
    df["pain_score_max_missing"] = 1.0 - df["knee_pain_any_imp"]
    with_alias = model_columns + ["knee_pain_any_imp", "pain_score_max_missing"]
    without = model_columns + ["knee_pain_any_imp"]
    Xa, Xb = build_design(df, spec, with_alias), build_design(df, spec, without)
    ia, ib = design_identifiability(Xa), design_identifiability(Xb)
    assert ia["n_columns"] == ib["n_columns"] + 1
    assert ia["rank"] == ib["rank"], "the extra column adds no rank"
    assert ia["identified_parameters"] == ib["identified_parameters"], \
        "dropping the aliased indicator loses nothing that was identified"


def test_sums_to_constant_rejects_a_block_that_does_not():
    X = pd.DataFrame({"a": [0.2, 0.4, 0.5], "b": [0.8, 0.5, 0.5]})
    assert sums_to_constant(X, ["a", "b"]) is False
    assert sums_to_constant(pd.DataFrame({"a": [0.2, 0.4], "b": [0.8, 0.6]}), ["a", "b"]) is True
    assert sums_to_constant(X, ["a"]) is False, "a single column is not a partition"


# --------------------------------------------------------------------------- #
# M1 = "M0 plus inferred KLG" on the KLG-ELIGIBLE SUBSET (protocol Table 7 and  #
# Secondary objective 2). The property that matters is that a KLG-missing       #
# patient never reaches the fit, so no severity grade is ever imputed.          #
# --------------------------------------------------------------------------- #
def _m1_dev_frame(n=400, n_missing_klg=60, seed=5):
    """A development frame shaped like features_clinical.parquet, M0 = 11 model columns."""
    rng = np.random.default_rng(seed)
    others = [f"x{i}_imp" for i in range(10)]
    df = pd.DataFrame({"age_at_index_imp": rng.uniform(45, 85, n)})
    for c in others:
        df[c] = rng.integers(0, 2, n).astype(float)
    klg = rng.integers(0, 5, n).astype(float)
    klg[:n_missing_klg] = np.nan
    df["klg_contra"] = klg
    df["klg_contra_missing"] = df["klg_contra"].isna().astype(int)
    df["klg_contra_imp"] = df["klg_contra"].fillna(2.0)
    lp = 0.4 * np.nan_to_num(klg, nan=2.0) + 0.02 * (df["age_at_index_imp"] - 65)
    df["time_from_landmark"] = np.clip(rng.exponential(900 / np.exp(lp - lp.mean())), 30, 1826)
    df["event_indicator"] = (rng.random(n) < 0.3).astype(int)
    df["split"] = np.where(np.arange(n) % 4 == 0, "val", "train")
    frozen = {"model_columns": ["age_at_index_imp"] + others,
              "m1_model_columns": ["age_at_index_imp"] + others + ["klg_contra_imp"],
              "m1_eligibility": {"n_eligible": int((~np.isnan(klg)).sum())}}
    return df, frozen


def test_fit_m1_klg_never_fits_on_a_klg_missing_patient():
    df, frozen = _m1_dev_frame()
    horizons = clamp_horizon_days([1.0, 5.0], 365.25, 1826.0)
    n_val_all = int((df["split"] == "val").sum())
    m0_lp = np.zeros(n_val_all)
    m0_risk = {h["horizon_days"]: np.full(n_val_all, 0.2) for h in horizons}
    import logging

    out = fit_m1_klg(df, frozen, horizons, [0.1, 1.0], 0.0, 3, 20,
                     np.random.default_rng(0), m0_lp, m0_risk, logging.getLogger("t"))
    n_elig = int((df["klg_contra_missing"] == 0).sum())
    assert out["n_train"] + out["n_val"] == n_elig, "an ineligible patient reached M1"
    assert out["n_train_dropped"] + out["n_val_dropped"] == len(df) - n_elig
    assert out["added_columns"] == ["klg_contra_imp"]
    assert out["fits"]["M1_klg"]["n_parameters"] == EXPECTED_M1_N_PARAMS
    assert out["fits"]["M0_refit_eligible"]["n_parameters"] == EXPECTED_N_PARAMS
    assert out["fits"]["M1_klg"]["identified_parameters"] == EXPECTED_M1_IDENTIFIED_PARAMS
    # The published-M0 arm must be restricted to exactly the eligible validation rows.
    assert len(out["fits"]["M0_as_fitted"]["lp"]) == out["n_val"]
    # A paired difference exists against both references, for every reported metric.
    for ref in ("M0_refit_eligible", "M0_as_fitted"):
        assert f"M1_klg-{ref}|cindex" in out["paired"]


def test_fit_m1_klg_refuses_a_frozen_contract_that_is_not_a_superset_of_m0():
    df, frozen = _m1_dev_frame()
    frozen["m1_model_columns"] = ["klg_contra_imp"]          # not a superset
    horizons = clamp_horizon_days([5.0], 365.25, 1826.0)
    import logging

    with pytest.raises(AssertionError, match="strict superset"):
        fit_m1_klg(df, frozen, horizons, [1.0], 0.0, 3, 2, np.random.default_rng(0),
                   np.zeros(int((df["split"] == "val").sum())), {}, logging.getLogger("t"))


# --------------------------------------------------------------------------- #
# PENALIZER SELECTION (deviation D24). The criterion moved from the 54-event    #
# validation C-index to the 373-event cross-validated C-index. BOTH criteria    #
# must still be computed and both winners reported, and the grid-edge warning   #
# must now be evaluated against whichever criterion actually selected.          #
# --------------------------------------------------------------------------- #
def _grid(val, cv, pens=(0.001, 0.01, 1.0)) -> pd.DataFrame:
    return pd.DataFrame({"penalizer": [float(p) for p in pens],
                         "train_cindex": [0.6] * len(pens),
                         "train_partial_loglik": [-100.0] * len(pens),
                         "val_cindex": list(val), "cv_mean_cindex": list(cv),
                         "cv_sd_cindex": [0.03] * len(pens), "n_folds": [25] * len(pens)})


def test_select_penalizer_uses_the_configured_criterion_not_the_other_one():
    """The two criteria disagree at OPPOSITE ends — exactly the D23/D24 situation."""
    g = _grid(val=[0.600, 0.601, 0.609], cv=[0.591, 0.590, 0.583])
    by_val = select_penalizer(g, "val_cindex")
    by_cv = select_penalizer(g, "cv_mean_cindex")
    assert by_val["penalizer"] == 1.0, "val_cindex still selects the old winner"
    assert by_cv["penalizer"] == 0.001, "cv_mean_cindex must select the CV winner"
    # Whichever selects, BOTH winners are reported so the loser stays auditable.
    for sel in (by_val, by_cv):
        assert sel["val_selected_penalizer"] == 1.0
        assert sel["cv_selected_penalizer"] == 0.001
        assert sel["criteria_agree"] is False
    assert by_cv["selection_metric"] == "cv_mean_cindex"
    assert by_cv["criterion"] == SELECTION_LABELS["cv_mean_cindex"]
    assert by_cv["selected_value"] == pytest.approx(0.591)


def test_select_penalizer_flags_the_grid_edge_of_the_SELECTING_criterion():
    """A CV winner at the lower edge must be flagged the same way a val winner was."""
    g = _grid(val=[0.600, 0.601, 0.609], cv=[0.591, 0.590, 0.583])
    assert select_penalizer(g, "val_cindex")["grid_edge_side"] == "upper"
    assert select_penalizer(g, "cv_mean_cindex")["grid_edge_side"] == "lower"
    for m in SELECTION_METRICS:
        assert select_penalizer(g, m)["at_grid_edge"] is True
    # An interior winner is not an edge under either criterion.
    interior = _grid(val=[0.60, 0.62, 0.61], cv=[0.58, 0.60, 0.59])
    for m in SELECTION_METRICS:
        s = select_penalizer(interior, m)
        assert s["penalizer"] == 0.01 and s["at_grid_edge"] is False
        assert s["grid_edge_side"] is None
    assert select_penalizer(interior, "val_cindex")["criteria_agree"] is True


def test_select_penalizer_breaks_ties_toward_less_shrinkage():
    g = _grid(val=[0.61, 0.61, 0.55], cv=[0.55, 0.59, 0.59])
    assert select_penalizer(g, "val_cindex")["penalizer"] == 0.001
    assert select_penalizer(g, "cv_mean_cindex")["penalizer"] == 0.01


def test_select_penalizer_refuses_an_unknown_criterion_or_a_half_computed_grid():
    g = _grid(val=[0.60, 0.61, 0.62], cv=[0.59, 0.58, 0.57])
    with pytest.raises(AssertionError, match="selection_metric"):
        select_penalizer(g, "auroc_at_2y")
    # Dropping the losing criterion is a silent-mixing bug, not an optimisation.
    with pytest.raises(AssertionError, match="missing the 'cv_mean_cindex' column"):
        select_penalizer(g.drop(columns=["cv_mean_cindex"]), "val_cindex")
    with pytest.raises(AssertionError, match="nothing to select on"):
        select_penalizer(_grid(val=[0.6, 0.6, 0.6], cv=[np.nan] * 3), "cv_mean_cindex")


def _tuning_frames(n=270, seed=11):
    """Synthetic train/val frames shaped for build_design and cv_penalizer_stability."""
    rng = np.random.default_rng(seed)
    cols = ["age_at_index_imp", "x0_imp", "x1_imp"]

    def mk(m):
        d = pd.DataFrame({"age_at_index_imp": rng.uniform(45.0, 85.0, m),
                          "x0_imp": rng.normal(size=m), "x1_imp": rng.normal(size=m)})
        lp = 0.8 * d["x0_imp"].to_numpy() - 0.3 * d["x1_imp"].to_numpy()
        d["time_from_landmark"] = np.clip(
            rng.exponential(700.0 / np.exp(lp - lp.mean())), 30.0, 1826.0)
        d["event_indicator"] = (rng.random(m) < 0.35).astype(int)
        return d

    return mk(n), mk(max(n // 3, 40)), cols


def test_tune_and_select_penalizer_computes_both_criteria_on_every_fit():
    tr, va, cols = _tuning_frames()
    spline = fit_age_spline(tr["age_at_index_imp"].to_numpy(dtype=float), 3)
    Xtr, Xva = build_design(tr, spline, cols), build_design(va, spline, cols)
    args = (tr, cols, Xtr, tr["time_from_landmark"].to_numpy(dtype=float),
            tr["event_indicator"].to_numpy(int), Xva,
            va["time_from_landmark"].to_numpy(dtype=float),
            va["event_indicator"].to_numpy(int), [0.01, 1.0], 0.0, 3, 0)
    g_cv, s_cv = tune_and_select_penalizer(*args, "cv_mean_cindex")
    g_val, s_val = tune_and_select_penalizer(*args, "val_cindex")
    # Both criteria are present on BOTH runs — the losing one is never skipped.
    for g in (g_cv, g_val):
        for m in SELECTION_METRICS:
            assert m in g.columns and g[m].notna().all()
        assert (g["n_folds"] == CV_N_SPLITS * CV_N_REPEATS).all()
    # Same grid either way; only the selector differs.
    assert s_cv["cv_selected_penalizer"] == s_val["cv_selected_penalizer"]
    assert s_cv["val_selected_penalizer"] == s_val["val_selected_penalizer"]
    assert s_cv["penalizer"] == s_cv["cv_selected_penalizer"]
    assert s_val["penalizer"] == s_val["val_selected_penalizer"]
    # The selection criterion decides which event count the choice rests on.
    assert s_cv["n_selection_events"] == int(tr["event_indicator"].sum())
    assert s_val["n_selection_events"] == int(va["event_indicator"].sum())


def test_cv_penalizer_stability_drops_a_constant_column_instead_of_crashing():
    """The complete-case sensitivity makes knee_pain_any constant; CV must survive it."""
    tr, _, cols = _tuning_frames(n=150, seed=3)
    tr["const_imp"] = 1.0                                    # constant by construction
    out = cv_penalizer_stability(
        tr, tr["time_from_landmark"].to_numpy(dtype=float),
        tr["event_indicator"].to_numpy(int), [0.1], 0.0, 3, cols + ["const_imp"], 0)
    assert len(out) == 1 and np.isfinite(out["cv_mean_cindex"].iloc[0])
    # Passing the caller's own drop list reaches the same place.
    out2 = cv_penalizer_stability(
        tr, tr["time_from_landmark"].to_numpy(dtype=float),
        tr["event_indicator"].to_numpy(int), [0.1], 0.0, 3, cols + ["const_imp"], 0,
        drop_design_columns=["const_imp"])
    assert np.isfinite(out2["cv_mean_cindex"].iloc[0])


def test_fit_m1_klg_selects_by_cross_validated_cindex_when_asked():
    """D24 applies the SAME criterion to the M1 ladder; both winners are recorded."""
    df, frozen = _m1_dev_frame()
    horizons = clamp_horizon_days([1.0, 5.0], 365.25, 1826.0)
    n_val_all = int((df["split"] == "val").sum())
    m0_lp = np.zeros(n_val_all)
    m0_risk = {h["horizon_days"]: np.full(n_val_all, 0.2) for h in horizons}
    import logging

    out = fit_m1_klg(df, frozen, horizons, [0.01, 1.0], 0.0, 3, 20,
                     np.random.default_rng(0), m0_lp, m0_risk, logging.getLogger("t"),
                     selection_metric="cv_mean_cindex", seed=7)
    assert out["selection_metric"] == "cv_mean_cindex"
    for name in ("M1_klg", "M0_refit_eligible"):
        sel = out["fits"][name]["selection"]
        assert sel["selection_metric"] == "cv_mean_cindex"
        assert sel["penalizer"] == sel["cv_selected_penalizer"] == \
            out["fits"][name]["penalizer"]
        assert sel["val_selected_penalizer"] in (0.01, 1.0), "the loser is still reported"
        # Both criteria survive into the persisted per-penalizer grid.
        gr = pd.DataFrame(out["fits"][name]["penalizer_grid"])
        for m in SELECTION_METRICS:
            assert m in gr.columns and gr[m].notna().all()
    # The published-M0 arm is not refitted, so it carries no selection of its own.
    assert out["fits"]["M0_as_fitted"]["selection"] is None


# --------------------------------------------------------------------------- #
# IPCW — against the hand-worked reverse-KM example above                       #
# --------------------------------------------------------------------------- #
def test_reverse_km_matches_the_hand_worked_censoring_curve():
    grid, vals = censoring_curve(HW_TIMES, HW_EVENTS)
    for t, expected in HW_G.items():
        assert float(step_value(grid, vals, t)[0]) == pytest.approx(expected)


def test_step_value_left_limit_differs_from_the_right_limit_at_a_jump():
    grid, vals = censoring_curve(HW_TIMES, HW_EVENTS)
    assert float(step_value(grid, vals, 4.0, left=True)[0]) == pytest.approx(0.75)   # G(4-)
    assert float(step_value(grid, vals, 4.0, left=False)[0]) == pytest.approx(0.375)  # G(4)
    assert float(step_value(grid, vals, 0.0, left=True)[0]) == 1.0, "G before time 0 is 1"


def test_ipcw_weights_match_the_hand_worked_example():
    """Horizon t = 3 on the hand-worked data.

    cases    : T=1 (event) w = 1 / G(1-) = 1 / 1    = 1
               T=3 (event) w = 1 / G(3-) = 1 / 0.75 = 4/3
    controls : T=4, T=5     w = 1 / G(3)  = 1 / 0.75 = 4/3
    excluded : T=2 (censored before the horizon) -> weight 0
    """
    grid, vals = censoring_curve(HW_TIMES, HW_EVENTS)
    y, w = ipcw_labels_weights(HW_TIMES, HW_EVENTS, 3.0, grid, vals)
    assert list(y) == [1, -1, 1, 0, 0]
    assert w[0] == pytest.approx(1.0)
    assert w[2] == pytest.approx(4 / 3)
    assert w[3] == pytest.approx(4 / 3) and w[4] == pytest.approx(4 / 3)
    assert w[1] == 0.0, "a patient censored before the horizon carries no weight"


def test_ipcw_auc_matches_the_hand_worked_value():
    """With risks [0.9, 0.1, 0.15, 0.2, 0.3] and the weights above the AUC is exactly 3/7.

    numerator   = w0*w3*1{0.9>0.2} + w0*w4*1{0.9>0.3} + w2*w3*1{0.15>0.2} + w2*w4*1{0.15>0.3}
                = 1*(4/3) + 1*(4/3) + 0 + 0 = 8/3
    denominator = (w0 + w2) * (w3 + w4) = (1 + 4/3) * (4/3 + 4/3) = 56/9
    """
    grid, vals = censoring_curve(HW_TIMES, HW_EVENTS)
    y, w = ipcw_labels_weights(HW_TIMES, HW_EVENTS, 3.0, grid, vals)
    risk = np.array([0.9, 0.1, 0.15, 0.2, 0.3])
    assert ipcw_auc(y, w, risk) == pytest.approx(3 / 7)


def test_ipcw_auc_scores_ties_as_one_half():
    y = np.array([1, 0]); w = np.array([1.0, 1.0])
    assert ipcw_auc(y, w, np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert ipcw_auc(y, w, np.array([0.9, 0.1])) == pytest.approx(1.0)
    assert ipcw_auc(y, w, np.array([0.1, 0.9])) == pytest.approx(0.0)


def test_ipcw_auc_is_nan_when_an_arm_is_empty():
    y = np.array([1, 1]); w = np.array([1.0, 1.0])
    assert np.isnan(ipcw_auc(y, w, np.array([0.2, 0.8])))


def test_administrative_horizon_leaves_no_controls():
    """The 5-year problem in miniature: everyone administratively censored on the same day.

    At the administrative horizon nobody is observed event-free BEYOND it, so the
    cumulative/dynamic AUROC has no control arm and is undefined — which is why the module
    clamps the 5-year horizon to the last day strictly inside follow-up.
    """
    times = np.array([3.0, 10.0, 10.0, 10.0])
    events = np.array([1, 0, 0, 0])
    grid, vals = censoring_curve(times, events)
    assert float(step_value(grid, vals, 10.0)[0]) == 0.0, "G is 0 at the administrative horizon"
    y, w = ipcw_labels_weights(times, events, 10.0, grid, vals)
    assert (y == 0).sum() == 0, "no controls at the administrative horizon"
    assert np.isnan(ipcw_auc(y, w, np.array([0.9, 0.1, 0.2, 0.3])))
    # One day earlier the estimator is well defined again.
    y9, w9 = ipcw_labels_weights(times, events, 9.0, grid, vals)
    assert (y9 == 0).sum() == 3 and np.isfinite(ipcw_auc(y9, w9, np.array([0.9, .1, .2, .3])))


def test_ipcw_weights_refuse_a_zero_censoring_probability_with_controls_present():
    """G comes from TRAIN but the labels come from VAL, so the two can disagree; a control
    weighted by 1/G with G == 0 must fail loudly rather than emit an infinite weight."""
    g_grid, g_vals = censoring_curve(np.array([3.0, 10.0, 10.0, 10.0]), np.array([1, 0, 0, 0]))
    assert float(step_value(g_grid, g_vals, 10.0)[0]) == 0.0
    with pytest.raises(AssertionError):
        ipcw_labels_weights(np.array([5.0, 12.0]), np.array([1, 0]), 10.0, g_grid, g_vals)


def test_ipcw_upweights_a_cohort_with_more_censoring():
    """More censoring before the horizon -> smaller G -> larger control weights."""
    light = (np.array([1.0, 2.0, 3.0, 9.0, 9.0]), np.array([1, 1, 1, 0, 0]))
    heavy = (np.array([1.0, 2.0, 3.0, 9.0, 9.0]), np.array([1, 0, 0, 0, 0]))
    w_light = ipcw_labels_weights(*light, 4.0, *censoring_curve(*light))[1]
    w_heavy = ipcw_labels_weights(*heavy, 4.0, *censoring_curve(*heavy))[1]
    assert w_heavy[-1] > w_light[-1]


def test_harrell_c_is_one_for_a_perfect_score_and_zero_for_a_reversed_one():
    t = np.array([1.0, 2.0, 3.0, 4.0]); e = np.array([1, 1, 1, 1])
    assert harrell_c(t, e, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(1.0)
    assert harrell_c(t, e, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# CALIBRATION BINNING                                                           #
# --------------------------------------------------------------------------- #
def test_risk_bins_are_equal_sized_and_ascending():
    pred = np.linspace(0.01, 0.9, 100)
    b = risk_bins(pred, CALIBRATION_BINS)
    assert set(b) == set(range(CALIBRATION_BINS))
    counts = np.bincount(b)
    assert counts.max() - counts.min() <= 1, "bins must hold equal patient counts"
    means = [pred[b == k].mean() for k in range(CALIBRATION_BINS)]
    assert means == sorted(means), "bin means must increase with the bin index"


def test_risk_bins_are_order_invariant_for_distinct_predictions():
    rng = np.random.default_rng(3)
    pred = rng.uniform(0, 1, 40)
    order = rng.permutation(40)
    a = risk_bins(pred, 5)
    b = risk_bins(pred[order], 5)
    assert (a[order] == b).all()


def test_risk_bins_handle_a_ragged_count():
    b = risk_bins(np.linspace(0, 1, 13), 5)
    assert set(b) == {0, 1, 2, 3, 4}
    assert np.bincount(b).max() - np.bincount(b).min() <= 1


def test_risk_bins_refuse_fewer_observations_than_bins():
    with pytest.raises(AssertionError):
        risk_bins(np.array([0.1, 0.2]), 5)


def test_cloglog_is_monotone_and_finite_at_the_boundaries():
    v = cloglog(np.array([0.0, 0.01, 0.5, 0.99, 1.0]))
    assert np.isfinite(v).all()
    assert (np.diff(v) > 0).all()


def test_cloglog_is_the_cloglog_function_and_not_the_logit():
    """Pins the LINK. log(-log(1-p)) and log(p/(1-p)) agree nowhere except by accident."""
    p = np.array([0.2, 0.5, 0.8])
    assert cloglog(p) == pytest.approx(np.log(-np.log(1.0 - p)), abs=1e-12)
    logit = np.log(p / (1.0 - p))
    assert np.abs(cloglog(p) - logit).min() > 0.1, "the two links must be distinguishable here"
    # cloglog(1 - e^-1) = log(-log(e^-1)) = log(1) = 0 exactly; logit there is +0.4587.
    assert cloglog(np.array([1 - np.exp(-1.0)]))[0] == pytest.approx(0.0, abs=1e-9)


def test_calibration_slope_uses_a_cloglog_glm_not_a_logit_one():
    """Mutation guard: swapping the GLM family to logit must change the answer.

    The expected values are recomputed here from statsmodels under BOTH links, so the test
    pins the link rather than a magic number, and fails if the module silently switches.
    """
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(5)
    n = 400
    pred = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < pred).astype(int)
    w = rng.uniform(1.0, 2.5, n)
    slope, intercept = calibration_slope_intercept(y, w, pred)

    x = cloglog(pred)
    ref_cll = sm.GLM(y.astype(float), sm.add_constant(x, has_constant="add"),
                     family=sm.families.Binomial(link=sm.families.links.CLogLog()),
                     var_weights=w).fit()
    ref_logit = sm.GLM(y.astype(float), sm.add_constant(x, has_constant="add"),
                       family=sm.families.Binomial(link=sm.families.links.Logit()),
                       var_weights=w).fit()
    assert slope == pytest.approx(float(ref_cll.params[1]), abs=1e-8)
    assert abs(float(ref_cll.params[1]) - float(ref_logit.params[1])) > 0.05, \
        "the two links must give visibly different slopes for this test to have teeth"
    assert np.isfinite(intercept)


def test_calibration_slope_is_one_for_a_perfectly_calibrated_ph_outcome():
    """A correctly specified cloglog model recovers slope 1; a logit fit would not."""
    rng = np.random.default_rng(9)
    n = 20000
    pred = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < pred).astype(int)
    slope, intercept = calibration_slope_intercept(y, np.ones(n), pred)
    assert slope == pytest.approx(1.0, abs=0.06)
    assert intercept == pytest.approx(0.0, abs=0.06)


# --------------------------------------------------------------------------- #
# HORIZON CLAMP — the shared definition; sample_size_riley imports this one     #
# --------------------------------------------------------------------------- #
def test_horizons_are_clamped_inside_observed_followup():
    """Removing the clamp would put the 5-year horizon on the administrative censoring day,
    where there are zero controls and the cumulative/dynamic AUROC is undefined."""
    h = clamp_horizon_days([1, 2, 5], 365.25, 1826.0)
    assert [int(x["horizon_days"]) for x in h] == [365, 730, 1825]
    assert [x["horizon_days_nominal"] for x in h] == [365, 730, 1826]
    assert [x["clamped"] for x in h] == [False, False, True]


def test_the_clamped_horizon_is_the_last_day_strictly_inside_followup():
    for max_obs in (1826.0, 1500.0, 900.5):
        h = clamp_horizon_days([5], 365.25, max_obs)[0]
        assert h["horizon_days"] <= int(max_obs) - 1
        assert h["horizon_days"] == min(1826, int(max_obs) - 1)


def test_an_unclamped_horizon_is_reported_as_unclamped():
    h = clamp_horizon_days([1, 2], 365.25, 5000.0)
    assert [int(x["horizon_days"]) for x in h] == [365, 730]
    assert not any(x["clamped"] for x in h)


def test_clamped_horizon_matches_the_config_and_the_locked_1825():
    cfg = load_config(DEFAULT_CONFIG)
    h = clamp_horizon_days(cfg["model_clinical"]["horizons_years"],
                           float(cfg["timeline"]["days_per_year"]), 1826.0)
    assert [int(x["horizon_days"]) for x in h] == [365, 730, 1825], \
        "src/sample_size_riley.py imports this helper; both must land on day 1825"


# --------------------------------------------------------------------------- #
# JSON REPLAY — the contract T7 reproduces M0 from                              #
# --------------------------------------------------------------------------- #
def test_replay_from_json_reproduces_the_documented_formulas():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 1.0, 0.0]})
    mj = {"design_columns": ["a", "b"],
          "coefficients": {"a": 0.5, "b": -0.25},
          "centering_means": {"a": 1.0, "b": 0.5},
          "baseline_survival": {"times": [0.0, 10.0, 20.0], "survival": [1.0, 0.9, 0.8]},
          "horizons": [{"horizon_days": 15.0}]}
    lp, risk = replay_from_json(mj, X)
    expected_lp = (X["a"] - 1.0) * 0.5 + (X["b"] - 0.5) * -0.25
    assert lp == pytest.approx(expected_lp.to_numpy(), abs=1e-15)
    # right-continuous baseline: the last stored time <= 15 is 10, so S0 = 0.9
    assert risk[15.0] == pytest.approx(1.0 - 0.9 ** np.exp(lp), abs=1e-15)


def test_replay_from_json_refuses_a_reordered_design():
    X = pd.DataFrame({"b": [1.0], "a": [0.0]})
    mj = {"design_columns": ["a", "b"], "coefficients": {"a": 0.5, "b": -0.25},
          "centering_means": {"a": 0.0, "b": 0.0},
          "baseline_survival": {"times": [0.0], "survival": [1.0]},
          "horizons": [{"horizon_days": 1.0}]}
    with pytest.raises(AssertionError):
        replay_from_json(mj, X)


def test_percentile_ci_brackets_the_median():
    v = np.linspace(0.0, 1.0, 1001)
    lo, hi = percentile_ci(v, 0.05)
    assert lo == pytest.approx(0.025, abs=1e-3) and hi == pytest.approx(0.975, abs=1e-3)
    assert np.isnan(percentile_ci(np.array([np.nan]))[0])


# --------------------------------------------------------------------------- #
# PROTOCOL SECTION 21 SUPPRESSION RULE                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_events,expected", [(0, True), (49, True), (50, False), (99, False),
                                               (100, False)])
def test_suppression_threshold(n_events, expected):
    supp, note = suppression(n_events)
    assert supp is expected
    if n_events < SUPPRESS_BELOW_EVENTS:
        assert "suppressed" in note
    elif n_events < EMPHASISE_CI_BELOW_EVENTS:
        assert "CI" in note
    else:
        assert note == ""


# --------------------------------------------------------------------------- #
# CONFIG CONTRACT — these constants are what downstream tasks rely on           #
# --------------------------------------------------------------------------- #
def test_config_model_clinical_contract():
    mc = load_config(DEFAULT_CONFIG)["model_clinical"]
    assert len(mc["penalizer_grid"]) == 7 and mc["l1_ratio"] == 0.0
    assert mc["age_rcs_df"] == 3 and mc["horizons_years"] == [1, 2, 5]
    # Deviation D24: selection moved from the 54-event validation C-index to the
    # 373-event cross-validated C-index. tuning_split stays "val" — the validation grid
    # is still computed and reported, it is simply no longer the selector.
    assert mc["selection_metric"] == "cv_mean_cindex" and mc["bootstrap_n"] == 500
    assert mc["selection_metric"] in SELECTION_METRICS
    assert mc["tuning_split"] == "val"
    assert mc["predictor_selection"] == "prespecified_all", "section 19 forbids screening"
    assert mc["imputation_source"] == "features_clinical_frozen", "section 20: never refit"
    assert set(mc["sensitivity"]) >= {"complete_case", "drop_pain_predictors", "race_included"}


def test_penalizer_grid_was_not_widened_by_d24():
    """D24 changed the CRITERION, not the pre-specified grid (protocol section 27)."""
    mc = load_config(DEFAULT_CONFIG)["model_clinical"]
    assert [float(p) for p in mc["penalizer_grid"]] == [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def test_locked_split_anchors_are_not_configurable():
    assert EXPECTED_SPLIT_N == {"train": 2597, "val": 371, "test": 741}
    assert EXPECTED_SPLIT_EVENTS == {"train": 373, "val": 54, "test": 106}
    assert sum(EXPECTED_SPLIT_N.values()) == 3709


def test_parameter_anchors_distinguish_columns_from_identified_parameters():
    """M0: 13 design columns, 12 identified — the difference is the cr() partition of unity."""
    assert EXPECTED_N_PARAMS == 13
    assert EXPECTED_IDENTIFIED_PARAMS == 12
    assert EXPECTED_IDENTIFIED_PARAMS == EXPECTED_N_PARAMS - 1


def test_m1_anchors_are_m0_plus_exactly_one_column():
    """Protocol Table 7: M1 = 'M0 plus inferred KLG' — one column, one more identified."""
    assert EXPECTED_M1_N_PARAMS == EXPECTED_N_PARAMS + 1 == 14
    assert EXPECTED_M1_IDENTIFIED_PARAMS == EXPECTED_IDENTIFIED_PARAMS + 1 == 13
    assert EXPECTED_M1_IDENTIFIED_PARAMS == EXPECTED_M1_N_PARAMS - 1
