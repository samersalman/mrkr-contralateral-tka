"""Unit tests for src/train_model.py - synthetic inputs only, no patient data read.

The numpy half of the trainer was hand-checked interval by interval in notebook cells 19,
21 and 23 before it was ported. These tests re-assert the SAME hand-worked numbers, so a
silent change to the label construction, the censoring-aware likelihood, the hazard-to-risk
conversion or the IPCW weighting fails here rather than in a metric nobody can audit.

They also pin the three things that would breach a contract rather than merely be wrong:
the sealed-test guard, the protocol section 13 augmentation caps, and the exactly-zero
masked border after the affine.

Torch-dependent tests start with ``pytest.importorskip("torch")`` so the whole file also
runs clean under system Python 3.14, which has no torch:

    python3 -m pytest tests/ -q                              (torch tests skip)
    ~/.venvs/mrkr-torch/bin/python -m pytest tests/test_train_model.py -q
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

import src.model_clinical as mc
from src.config import DEFAULT_CONFIG, load_config
from src.train_model import (
    DEV_SPLITS,
    EDGES,
    EXPECTED_DEV_ROWS,
    EXPECTED_SPLIT_CROPS,
    GRID_MAX_DAYS,
    HISTORY_COLUMNS,
    INTERVAL_DAYS,
    MAX_ROTATION_DEG,
    N_INTERVALS,
    SEED_VARIABILITY_COLUMNS,
    TrainSettings,
    apply_recalibration,
    assert_border_is_zero,
    assert_development_splits,
    average_hazard,
    build_clinical_design,
    calibration_slope_intercept,
    cloglog,
    discretize_survival,
    dt_nll_numpy,
    fit_recalibration,
    harrell_c,
    harrell_c_numpy,
    hazards_to_survival,
    inv_cloglog,
    ipcw_auc,
    ipcw_brier,
    ipcw_labels_weights,
    km_cif_numpy,
    load_sidecar,
    materialize_split,
    parse_augmentation,
    read_json_retrying,
    resolve_arms,
    reverse_km,
    risk_at_horizon,
    risk_score,
    shard_urls,
    step_value,
    uno_c,
    write_history,
    write_seed_variability,
)


# --------------------------------------------------------------------------- #
# 1. The frozen discrete-time grid                                             #
# --------------------------------------------------------------------------- #
def test_interval_grid_matches_the_frozen_contract():
    assert N_INTERVALS == 10
    assert GRID_MAX_DAYS == 1826.0
    assert EDGES.shape == (11,)
    assert EDGES[0] == 0.0 and EDGES[-1] == 1826.0
    assert INTERVAL_DAYS == pytest.approx(182.6)
    assert EDGES[2] == pytest.approx(365.2)


def test_config_agrees_with_the_frozen_grid_and_caps():
    cfg = load_config(DEFAULT_CONFIG)
    mi = cfg["model_image"]
    assert int(mi["survival_head"]["n_intervals"]) == N_INTERVALS
    assert int(mi["survival_head"]["interval_months"]) == 6
    assert str(mi["ensemble"]) == "average_hazard"
    assert str(mi["optimizer"]) == "adamw" and str(mi["lr_schedule"]) == "cosine"
    assert int(mi["early_stopping"]["patience"]) == 8
    assert int(mi["warmup_epochs"]) == 2 and float(mi["grad_clip_norm"]) == 1.0
    assert int(mi["batch_size"]) == 32 and int(mi["max_epochs"]) == 40
    assert len(mi["local"]["seeds"]) == int(mi["n_seeds"]) == 5


# --------------------------------------------------------------------------- #
# 2. Discrete-time labels - notebook cell 19's HAND-WORKED expectations         #
# --------------------------------------------------------------------------- #
HAND_TIMES = np.array([100.0, 300.0, 365.2, 1826.0, 1826.0, 182.6, 0.0, 50.0])
HAND_EVENTS = np.array([1, 0, 1, 0, 1, 0, 1, 0])
HAND_K = [0, -1, 2, -1, 9, -1, 0, -1]
HAND_N_SCORED = [1, 1, 3, 10, 10, 1, 1, 0]
HAND_LABELS = ["event in interval 0", "censored mid-interval", "event exactly on edge 365.2",
               "censored at day 1826", "event at day 1826", "censored exactly on edge 182.6",
               "event at day 0", "censored inside interval 0"]


def test_discretize_survival_matches_the_hand_worked_labels():
    at_risk, target, k_event, n_scored = discretize_survival(HAND_TIMES, HAND_EVENTS)
    for i, label in enumerate(HAND_LABELS):
        assert k_event[i] == HAND_K[i], label
        assert n_scored[i] == HAND_N_SCORED[i], label
    assert at_risk.shape == (len(HAND_TIMES), N_INTERVALS)
    assert target.shape == (len(HAND_TIMES), N_INTERVALS)


def test_event_exactly_on_an_interval_edge_lands_in_the_next_interval():
    _, _, k, ns = discretize_survival(np.array([365.2]), np.array([1]))
    assert k[0] == 2 and ns[0] == 3


def test_event_on_the_last_edge_is_clamped_into_the_final_interval():
    _, target, k, ns = discretize_survival(np.array([1826.0]), np.array([1]))
    assert k[0] == N_INTERVALS - 1 and ns[0] == N_INTERVALS
    assert target[0, -1] == 1.0 and target.sum() == 1.0


def test_a_patient_censored_inside_interval_zero_scores_nothing():
    at_risk, target, k, ns = discretize_survival(np.array([50.0]), np.array([0]))
    assert k[0] == -1 and ns[0] == 0
    assert at_risk.sum() == 0.0 and target.sum() == 0.0


def test_exactly_one_target_interval_per_event_and_it_is_at_risk():
    at_risk, target, _, _ = discretize_survival(HAND_TIMES, HAND_EVENTS)
    assert (target.sum(axis=1) == (HAND_EVENTS == 1)).all()
    assert (target * (1.0 - at_risk)).sum() == 0.0


def test_discretize_survival_refuses_times_beyond_the_grid():
    with pytest.raises(AssertionError, match="beyond"):
        discretize_survival(np.array([2000.0]), np.array([0]))
    with pytest.raises(AssertionError, match="negative"):
        discretize_survival(np.array([-1.0]), np.array([0]))
    with pytest.raises(AssertionError, match="0/1"):
        discretize_survival(np.array([10.0]), np.array([2]))


# --------------------------------------------------------------------------- #
# 3. The censoring-aware NLL - cell 19's HAND-COMPUTED values                   #
# --------------------------------------------------------------------------- #
def test_dt_nll_numpy_matches_the_hand_computed_two_patient_case():
    h = np.zeros((2, N_INTERVALS)); h[:, 0] = 0.1; h[:, 1] = 0.2
    a = np.zeros((2, N_INTERVALS)); a[0, :2] = 1; a[1, :2] = 1
    y = np.zeros((2, N_INTERVALS)); y[0, 1] = 1
    hand_a = -(math.log(0.9) + math.log(0.2))    # survived interval 0, event in interval 1
    hand_b = -(math.log(0.9) + math.log(0.8))    # censored after two full intervals
    mean, per = dt_nll_numpy(h, a, y)
    assert per[0] == pytest.approx(hand_a, abs=1e-12)
    assert per[1] == pytest.approx(hand_b, abs=1e-12)
    assert mean == pytest.approx((hand_a + hand_b) / 2, abs=1e-12)


def test_a_perfectly_predicted_event_costs_almost_nothing():
    eps = 1e-6
    h = np.full((1, N_INTERVALS), eps); h[0, 2] = 1 - eps
    a, y, _, _ = discretize_survival(np.array([500.0]), np.array([1]))
    nll, _ = dt_nll_numpy(h, a, y)
    assert nll < 1e-5


def test_a_censored_patient_scores_only_the_intervals_it_survived():
    """Cell 19's property test: hazards after the censoring time cannot change the loss."""
    h = np.full((1, N_INTERVALS), 0.3)
    a, y, _, n_scored = discretize_survival(np.array([600.0]), np.array([0]))
    assert n_scored[0] == 3
    nll, _ = dt_nll_numpy(h, a, y)
    assert nll == pytest.approx(-3 * math.log(0.7), abs=1e-12)
    h2 = h.copy(); h2[0, 3:] = 0.99
    assert dt_nll_numpy(h2, a, y)[0] == pytest.approx(nll, abs=1e-12)


def test_sample_weights_reweight_the_mean_but_not_the_per_patient_terms():
    h = np.full((4, N_INTERVALS), 0.2)
    a, y, _, _ = discretize_survival(np.array([100., 100., 900., 900.]), np.array([1, 1, 0, 0]))
    unweighted, per_u = dt_nll_numpy(h, a, y)
    w = np.array([0.5, 0.5, 1.5, 1.5])
    weighted, per_w = dt_nll_numpy(h, a, y, w)
    assert np.allclose(per_u, per_w)
    assert weighted == pytest.approx(float((w * per_u).sum() / w.sum()), abs=1e-12)
    assert weighted != pytest.approx(unweighted, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. Hazards -> survival -> risk, and the ensemble rule - cell 21               #
# --------------------------------------------------------------------------- #
CELL21_HAZARDS = np.array([[0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]])


def test_hazards_to_survival_matches_the_hand_computed_product():
    S = hazards_to_survival(CELL21_HAZARDS)
    assert S.shape == (1, N_INTERVALS + 1)
    assert S[0, 0] == 1.0
    assert S[0, 2] == pytest.approx(0.95 * 0.96, abs=1e-12)


def test_risk_at_horizon_interpolates_inside_the_interval():
    frac = (365.0 - EDGES[1]) / INTERVAL_DAYS
    hand = 1 - 0.95 * 0.96 ** frac
    assert risk_at_horizon(CELL21_HAZARDS, 365.0)[0] == pytest.approx(hand, abs=1e-12)


def test_risk_at_an_exact_edge_needs_no_interpolation():
    assert risk_at_horizon(CELL21_HAZARDS, 365.2)[0] == pytest.approx(1 - 0.95 * 0.96, abs=1e-12)


def test_risk_at_1825_uses_the_last_interval_fraction():
    S = hazards_to_survival(CELL21_HAZARDS)
    frac = (1825.0 - EDGES[9]) / INTERVAL_DAYS
    hand = 1 - S[0, 9] * 0.99 ** frac
    assert risk_at_horizon(CELL21_HAZARDS, 1825.0)[0] == pytest.approx(hand, abs=1e-12)


def test_risk_at_horizon_refuses_a_horizon_off_the_grid():
    with pytest.raises(AssertionError, match="outside the discrete grid"):
        risk_at_horizon(CELL21_HAZARDS, 2000.0)


def test_the_ensemble_averages_hazards_not_risks():
    """Protocol section 16: average the HAZARDS, then convert once. The two differ."""
    h1 = np.full((1, N_INTERVALS), 0.10)
    h2 = np.full((1, N_INTERVALS), 0.30)
    hb = average_hazard([h1, h2])
    assert hb[0, 0] == pytest.approx(0.2, abs=1e-12)
    r_from_hazard = risk_at_horizon(hb, 1826.0)[0]
    r_from_risk = 0.5 * (risk_at_horizon(h1, 1826.0)[0] + risk_at_horizon(h2, 1826.0)[0])
    assert r_from_hazard == pytest.approx(1 - 0.8 ** 10, abs=1e-12)
    assert abs(r_from_hazard - r_from_risk) > 1e-3


def test_average_hazard_refuses_anything_but_a_list_of_2d_arrays():
    with pytest.raises(AssertionError, match="hazard arrays"):
        average_hazard([np.zeros(N_INTERVALS), np.zeros(N_INTERVALS)])


def test_risk_score_is_monotone_in_the_hazards():
    low = np.full((1, N_INTERVALS), 0.01)
    high = np.full((1, N_INTERVALS), 0.20)
    assert risk_score(high)[0] > risk_score(low)[0] > 0


# --------------------------------------------------------------------------- #
# 5. IPCW - cell 23's toy case, and identity with src.model_clinical            #
# --------------------------------------------------------------------------- #
TOY_G_GRID = np.array([0.0, 50.0, 150.0])
TOY_G_VALS = np.array([1.0, 0.8, 0.5])
TOY_T = np.array([40.0, 80.0, 90.0, 200.0, 300.0])
TOY_E = np.array([1, 1, 0, 0, 1])
TOY_R = np.array([0.9, 0.4, 0.5, 0.3, 0.2])


def test_the_ipcw_estimators_are_the_ones_from_model_clinical_not_copies():
    """A second implementation could drift; there is only one, re-exported."""
    assert step_value is mc.step_value
    assert ipcw_labels_weights is mc.ipcw_labels_weights
    assert ipcw_auc is mc.ipcw_auc
    assert calibration_slope_intercept is mc.calibration_slope_intercept
    assert cloglog is mc.cloglog
    assert harrell_c is mc.harrell_c


def test_ipcw_labels_and_weights_on_the_hand_worked_toy_case():
    y, w = ipcw_labels_weights(TOY_T, TOY_E, 100.0, TOY_G_GRID, TOY_G_VALS)
    assert y.tolist() == [1, 1, -1, 0, 0]
    # case T=40 -> 1/G(40-)=1.00 ; case T=80 -> 1/G(80-)=1/0.8=1.25 ;
    # controls -> 1/G(100)=1/0.8=1.25 ; censored before the horizon -> weight 0
    assert np.allclose(w, [1.0, 1.25, 0.0, 1.25, 1.25])


def test_ipcw_auc_on_the_hand_worked_toy_case():
    y, w = ipcw_labels_weights(TOY_T, TOY_E, 100.0, TOY_G_GRID, TOY_G_VALS)
    num = 1 * 1.25 + 1 * 1.25 + 1.25 * 1.25 + 1.25 * 1.25
    den = (1 + 1.25) * (1.25 + 1.25)
    assert ipcw_auc(y, w, TOY_R) == pytest.approx(num / den, abs=1e-12)


def test_a_case_tied_with_a_control_scores_one_half():
    y, w = ipcw_labels_weights(TOY_T, TOY_E, 100.0, TOY_G_GRID, TOY_G_VALS)
    tied = np.array([0.3, 0.4, 0.5, 0.3, 0.2])
    num = 1 * 1.25 * 0.5 + 1 * 1.25 + 1.25 * 1.25 + 1.25 * 1.25
    den = (1 + 1.25) * (1.25 + 1.25)
    assert ipcw_auc(y, w, tied) == pytest.approx(num / den, abs=1e-12)


def test_uno_c_on_the_hand_worked_three_patient_case():
    t = np.array([10., 20., 30.]); e = np.array([1, 0, 1]); r = np.array([0.9, 0.5, 0.7])
    assert uno_c(t, e, r, TOY_G_GRID, TOY_G_VALS) == pytest.approx(1.0, abs=1e-12)


def test_ipcw_brier_is_a_weighted_squared_error_over_usable_patients_only():
    y, w = ipcw_labels_weights(TOY_T, TOY_E, 100.0, TOY_G_GRID, TOY_G_VALS)
    p = np.array([1.0, 1.0, 0.42, 0.0, 0.0])       # perfect on every usable patient
    assert ipcw_brier(y, w, p) == pytest.approx(0.0, abs=1e-12)
    p_bad = np.array([0.0, 0.0, 0.42, 1.0, 1.0])
    assert ipcw_brier(y, w, p_bad) == pytest.approx(1.0, abs=1e-12)


def test_reverse_km_agrees_with_the_lifelines_backed_censoring_curve():
    rng = np.random.default_rng(11)
    t = np.round(rng.uniform(1, 900, size=200), 0)
    e = (rng.uniform(size=200) < 0.35).astype(int)
    grid_np, vals_np = reverse_km(t, e)
    grid_lf, vals_lf = mc.censoring_curve(t, e)
    probes = np.array([0.0, 1.0, 50.0, 123.0, 400.0, 899.0, 900.0])
    got = step_value(grid_np, vals_np, probes)
    want = step_value(grid_lf, vals_lf, probes)
    assert np.allclose(got, want, atol=1e-12)


def test_cloglog_and_its_inverse_round_trip():
    p = np.array([0.001, 0.05, 0.2, 0.5, 0.9, 0.99])
    assert np.allclose(inv_cloglog(cloglog(p)), p, atol=1e-9)


# --------------------------------------------------------------------------- #
# 6. Harrell's C - the ported numpy version must agree with the lifelines one   #
# --------------------------------------------------------------------------- #
def test_harrell_c_on_the_hand_worked_three_patient_case():
    t = np.array([10., 20., 30.]); e = np.array([1, 0, 1]); r = np.array([0.9, 0.5, 0.7])
    assert harrell_c_numpy(t, e, r) == pytest.approx(1.0, abs=1e-12)
    assert harrell_c(t, e, r) == pytest.approx(1.0, abs=1e-12)


def test_harrell_c_numpy_agrees_with_the_lifelines_estimator_on_ties():
    """The two estimators must be interchangeable, including on time and risk ties."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(15, 120))
        t = np.round(rng.uniform(0, GRID_MAX_DAYS, size=n), 0)     # deliberate time ties
        e = (rng.uniform(size=n) < 0.3).astype(int)
        r = np.round(rng.normal(size=n), 1)                        # deliberate risk ties
        assert harrell_c_numpy(t, e, r) == pytest.approx(harrell_c(t, e, r), abs=1e-12)


def test_reversing_the_risk_ordering_reflects_the_c_index_about_one_half():
    rng = np.random.default_rng(3)
    t = rng.uniform(0, GRID_MAX_DAYS, size=80)
    e = (rng.uniform(size=80) < 0.4).astype(int)
    r = rng.normal(size=80)
    assert harrell_c_numpy(t, e, r) + harrell_c_numpy(t, e, -r) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 7. Forward KM cumulative incidence - the fast numpy twin of km_risk           #
#    The decision curve refits KM ~1e5 times, so it cannot use lifelines.       #
# --------------------------------------------------------------------------- #
def test_km_cif_on_the_hand_worked_three_patient_case():
    """t=[10,20,30], e=[1,0,1]. Day 10: 1/3 of a risk set of 3 fails -> S=2/3, F=1/3."""
    t = np.array([10., 20., 30.]); e = np.array([1, 0, 1])
    assert km_cif_numpy(t, e, 25.0) == (pytest.approx(1 / 3, abs=1e-12), 30.0)
    # Day 30: the last patient is the whole risk set and fails -> S=0 exactly.
    assert km_cif_numpy(t, e, 30.0) == (pytest.approx(1.0, abs=1e-12), 30.0)
    assert km_cif_numpy(t, e, 5.0) == (0.0, 30.0)          # before any observation


def test_km_cif_numpy_agrees_with_the_lifelines_estimator_on_ties():
    """The two estimators must be interchangeable, including on time ties."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(15, 120))
        t = np.round(rng.uniform(0, GRID_MAX_DAYS, size=n), 0)     # deliberate time ties
        e = (rng.uniform(size=n) < 0.3).astype(int)
        for h in (365.0, 1095.0, 1825.0):
            assert km_cif_numpy(t, e, h)[0] == pytest.approx(mc.km_risk(t, e, h)[0], abs=1e-12)


def test_an_event_exactly_on_the_horizon_is_included():
    """F is right-continuous: F(h) counts the events ON day h, one day earlier it does not."""
    t = np.array([50., 100., 150.]); e = np.array([0, 1, 0])
    assert km_cif_numpy(t, e, 100.0)[0] == pytest.approx(0.5, abs=1e-12)
    assert km_cif_numpy(t, e, 99.0)[0] == 0.0
    assert mc.km_risk(t, e, 100.0)[0] == pytest.approx(0.5, abs=1e-12)


def test_a_censoring_tied_with_an_event_stays_in_that_days_risk_set():
    """Censoring is conventionally AFTER the events it ties with - the reverse_km rule."""
    t = np.array([10., 10., 20., 30.]); e = np.array([1, 0, 0, 1])
    # risk set at day 10 is 4, not 3: the patient censored on day 10 is still in it.
    assert km_cif_numpy(t, e, 15.0)[0] == pytest.approx(1 - 3 / 4, abs=1e-12)
    assert km_cif_numpy(t, e, 15.0)[0] == pytest.approx(mc.km_risk(t, e, 15.0)[0], abs=1e-12)
    # Same rule reverse_km applies with the roles swapped: G steps at the tied censoring.
    _, g_vals = reverse_km(t, e)
    assert g_vals[1] == pytest.approx(1 - 1 / 4, abs=1e-12)


def test_the_curve_is_carried_forward_past_the_last_observed_day():
    """Nobody followed to the horizon -> the last observed value, never NaN, and the day."""
    t = np.array([100., 200.]); e = np.array([1, 0])
    cif, last = km_cif_numpy(t, e, 1825.0)
    assert last == 200.0 < 1825.0                     # this is the km_last_obs_day column
    assert cif == pytest.approx(0.5, abs=1e-12)       # F(200) carried forward to day 1825
    assert cif == km_cif_numpy(t, e, 200.0)[0]
    assert cif == pytest.approx(mc.km_risk(t, e, 1825.0)[0], abs=1e-12)


def test_last_obs_day_is_the_largest_follow_up_time_whatever_the_horizon():
    """The caller detects carry-forward with ``last_obs_day < horizon``, so it must not lie."""
    rng = np.random.default_rng(17)
    t = np.round(rng.uniform(1, 900, size=50), 0)
    e = (rng.uniform(size=50) < 0.4).astype(int)
    for h in (0.0, 10.0, 450.0, 900.0, 1825.0):
        assert km_cif_numpy(t, e, h)[1] == float(t.max())


def test_km_cif_is_defined_on_every_degenerate_input():
    """All finite, no NaN - including the two inputs lifelines' km_risk cannot take."""
    assert km_cif_numpy([], [], 1825.0) == (0.0, 0.0)             # empty: reverse_km's [0],[1]
    with pytest.raises(Exception):                                    # km_risk cannot
        mc.km_risk(np.array([]), np.array([]), 1825.0)
    assert km_cif_numpy([100., 200., 300.], [0, 0, 0], 1825.0) == (0.0, 300.0)   # all censored
    assert km_cif_numpy([5.], [1], 10.0) == (1.0, 5.0)                # single patient, event
    assert km_cif_numpy([5.], [0], 10.0) == (0.0, 5.0)                # single patient, censored
    assert km_cif_numpy([1., 2., 3.], [1, 1, 1], 5.0) == (1.0, 3.0)   # risk set empties: S=0
    assert km_cif_numpy([100., 200.], [1, 1], -5.0) == (0.0, 200.0)   # horizon before day 0


def test_km_cif_numpy_is_far_cheaper_than_the_lifelines_path():
    """~1e5 fits at lifelines' ~2.2 ms would be minutes; the ceiling is deliberately loose."""
    rng = np.random.default_rng(23)
    n = 741                                            # the real test-split size
    t = np.round(rng.uniform(0, 2600, size=n), 0)
    e = (rng.uniform(size=n) < 0.2).astype(int)

    def per_call(fn, reps):
        fn()                                           # warm up; ignore first-call overhead
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    fast = per_call(lambda: km_cif_numpy(t, e, 1825.0), 200)
    slow = per_call(lambda: mc.km_risk(t, e, 1825.0), 20)
    assert fast < 2e-3, f"{fast * 1e6:.0f} us/call - too slow for ~1e5 decision-curve fits"
    assert fast < slow / 5, f"numpy {fast * 1e6:.0f} us vs lifelines {slow * 1e6:.0f} us"


# --------------------------------------------------------------------------- #
# 8. Recalibration, fitted on validation only                                   #
# --------------------------------------------------------------------------- #
def test_identity_recalibration_leaves_a_risk_unchanged():
    p = np.array([0.05, 0.2, 0.6])
    out = apply_recalibration(p, {"intercept": 0.0, "slope": 1.0})
    assert np.allclose(out, p, atol=1e-9)


def test_fit_recalibration_returns_one_entry_per_horizon_and_is_finite():
    rng = np.random.default_rng(5)
    n = 400
    t = rng.uniform(10, GRID_MAX_DAYS, size=n)
    e = (rng.uniform(size=n) < 0.3).astype(int)
    hz = np.clip(rng.uniform(0.01, 0.06, size=(n, N_INTERVALS)), 1e-4, 0.5)
    g_grid, g_vals = reverse_km(t, e)
    recal = fit_recalibration(hz, t, e, [365.0, 730.0, 1825.0], g_grid, g_vals)
    assert sorted(recal) == ["1825.0", "365.0", "730.0"]
    for h, r in recal.items():
        assert np.isfinite(r["intercept"]) and np.isfinite(r["slope"]), h
        assert r["n_cases"] >= 0 and r["n_controls"] >= 0


# --------------------------------------------------------------------------- #
# 9. THE SEALED-TEST GUARD - the trainer must refuse the test split             #
# --------------------------------------------------------------------------- #
def test_assert_development_splits_refuses_the_sealed_split():
    assert assert_development_splits(["train", "val"]) == ["train", "val"]
    with pytest.raises(AssertionError, match="REFUSED"):
        assert_development_splits(["train", "test"])
    with pytest.raises(AssertionError, match="REFUSED"):
        assert_development_splits(["test"])


def test_every_shard_reader_refuses_the_sealed_split(tmp_path):
    with pytest.raises(AssertionError, match="REFUSED"):
        shard_urls(tmp_path, "test")
    with pytest.raises(AssertionError, match="REFUSED"):
        materialize_split("test", shard_dir=tmp_path, cache_dir=tmp_path,
                          labels=pd.DataFrame({"split": ["test"], "key": ["k"]}), out_size=512)


def test_load_sidecar_refuses_a_sidecar_carrying_test_rows(tmp_path):
    pd.DataFrame({"empi_anon": ["a", "b"], "key": ["a_0", "b_0"], "split": ["train", "test"],
                  "view": ["frontal", "frontal"]}).to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(AssertionError, match="REFUSED"):
        load_sidecar(tmp_path)


def test_build_clinical_design_refuses_the_sealed_split():
    class _Contracts:                                   # never reaches the parquet
        pass
    with pytest.raises(AssertionError, match="REFUSED"):
        build_clinical_design(_Contracts(), ["train", "test"], "m0")


def test_the_parquet_predicate_is_the_one_model_clinical_uses():
    """The guard is pushed DOWN into the reader, not applied after materialising rows."""
    from src.train_model import load_development_frame as ldf
    assert ldf is mc.load_development_frame
    assert mc.SEALED_SPLIT == "test" and tuple(DEV_SPLITS) == ("train", "val")


def test_the_sealed_split_guard_is_still_switched_on_in_config():
    cfg = load_config(DEFAULT_CONFIG)
    assert bool(cfg["model_image"]["local"]["forbid_test_split"]) is True


def test_cohort_anchors_are_the_frozen_ones():
    assert EXPECTED_DEV_ROWS == 2968
    assert EXPECTED_SPLIT_CROPS == {"train": 4254, "val": 601}


# --------------------------------------------------------------------------- #
# 10. Augmentation caps - protocol section 13                                   #
# --------------------------------------------------------------------------- #
def test_the_configured_augmentation_is_inside_the_protocol_caps():
    cfg = load_config(DEFAULT_CONFIG)
    aug = parse_augmentation(cfg["model_image"]["augmentation"])
    assert aug["rotation_deg"] == 5.0 and aug["rotation_deg"] <= MAX_ROTATION_DEG
    assert aug["translate"] == 0.05 and aug["scale"] == 0.05
    assert aug["intensity"] == 0.1


@pytest.mark.parametrize("bad", ["random_resized_crop", "RandomResizedCrop",
                                 "horizontal_flip", "hflip", "elastic_0.1",
                                 "perspective_0.2", "shear_5deg", "random_erasing"])
def test_forbidden_augmentations_are_refused(bad):
    with pytest.raises(AssertionError, match="forbidden by protocol section 13"):
        parse_augmentation(["rotation_5deg", bad])


@pytest.mark.parametrize("bad,msg", [("rotation_10deg", "rotation at 5"),
                                     ("translate_0.2", "translation and scale"),
                                     ("scale_0.5", "translation and scale")])
def test_augmentation_over_the_cap_is_refused(bad, msg):
    with pytest.raises(AssertionError, match=msg):
        parse_augmentation([bad])


def test_assert_border_is_zero_catches_a_polluted_border():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[8:-8, 8:-8] = 200
    assert_border_is_zero(img, 8)
    img[0, 0] = 1
    with pytest.raises(AssertionError, match="masked border"):
        assert_border_is_zero(img, 8)


# --------------------------------------------------------------------------- #
# 11. Output schemas and the CLI arm resolver                                   #
# --------------------------------------------------------------------------- #
def test_the_output_column_schemas_are_the_pinned_ones():
    assert HISTORY_COLUMNS == ["arm", "seed", "epoch", "train_nll", "val_nll", "lr", "secs",
                               "improved"]
    assert SEED_VARIABILITY_COLUMNS == ["arm", "label", "n_seeds", "val_nll_mean", "val_nll_sd",
                                        "val_nll_min", "val_nll_max", "best_epochs",
                                        "n_epochs_run", "ensemble_val_nll"]


def test_write_history_replaces_only_the_arms_it_was_given(tmp_path):
    p = tmp_path / "train_history.csv"
    write_history(p, [{"arm": "a", "seed": 1, "epoch": 0, "train_nll": 1.0, "val_nll": 2.0,
                       "lr": 1e-4, "secs": 3.0, "improved": True}])
    write_history(p, [{"arm": "b", "seed": 1, "epoch": 0, "train_nll": 1.0, "val_nll": 2.0,
                       "lr": 1e-4, "secs": 3.0, "improved": False}])
    df = pd.read_csv(p)
    assert list(df.columns) == HISTORY_COLUMNS
    assert sorted(df["arm"]) == ["a", "b"]
    assert "empi_anon" not in df.columns


def test_write_seed_variability_matches_the_pinned_schema(tmp_path):
    p = tmp_path / "seed_variability.csv"
    write_seed_variability(p, [{"arm": "a", "label": "A", "seeds": [1, 2],
                                "val_nll_mean": 1.0, "val_nll_sd": 0.1, "val_nll_min": 0.9,
                                "val_nll_max": 1.1, "best_epochs": [3, 4],
                                "n_epochs_run": [11, 12], "ensemble_val_nll": 0.95}])
    df = pd.read_csv(p)
    assert list(df.columns) == SEED_VARIABILITY_COLUMNS
    assert int(df.loc[0, "n_seeds"]) == 2 and df.loc[0, "best_epochs"] == "[3, 4]"


def test_resolve_arms_covers_the_config_ladder():
    cfg = load_config(DEFAULT_CONFIG)
    settings = _settings_without_torch(cfg)
    assert resolve_arms(settings, "stage1", None) == ["m0d_clinical", "m1_klg", "m4_fusion",
                                                      "m3_image"]
    assert resolve_arms(settings, "stage2", None) == ["m2_frontal", "m4_frontal",
                                                      "r1_densenet_frontal"]
    assert resolve_arms(settings, None, "m4_fusion,m3_image") == ["m4_fusion", "m3_image"]
    assert len(resolve_arms(settings, None, None)) == 7
    with pytest.raises(AssertionError, match="unknown arm"):
        resolve_arms(settings, None, "not_an_arm")
    with pytest.raises(AssertionError, match="unknown stage"):
        resolve_arms(settings, "stage9", None)


def _settings_without_torch(cfg):
    """TrainSettings reads only config, so it works under either interpreter."""
    return TrainSettings(cfg)


def test_training_contract_hash_is_stable_and_changes_with_a_knob():
    cfg = load_config(DEFAULT_CONFIG)
    a = TrainSettings(cfg).contract_hash()
    b = TrainSettings(cfg).contract_hash()
    assert a == b and len(a) == 16
    c = TrainSettings(cfg, max_epochs_override=41).contract_hash()
    assert c != a
    d = TrainSettings(cfg, grad_accum_override=4).contract_hash()
    assert d != a


def test_gradient_accumulation_splits_the_batch_without_changing_it():
    """Protocol section 13: the optimisation batch stays 32 patients; only the forward
    pass is split. Downscaling the 512x512 crop is never an option."""
    cfg = load_config(DEFAULT_CONFIG)
    base = TrainSettings(cfg)
    assert base.grad_accum_steps == 1 and base.micro_batch_size == base.batch_size == 32
    split = TrainSettings(cfg, grad_accum_override=4)
    assert split.batch_size == 32 and split.grad_accum_steps == 4
    assert split.micro_batch_size == 8
    assert split.micro_batch_size * split.grad_accum_steps == split.batch_size


def test_read_json_retrying_reads_a_complete_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1}))
    assert read_json_retrying(p) == {"a": 1}


def test_read_json_retrying_fails_loudly_on_a_permanently_broken_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    with pytest.raises(AssertionError, match="could not parse"):
        read_json_retrying(p, attempts=2, delay=0.0)


# =========================================================================== #
# TORCH-DEPENDENT TESTS - skipped under system Python 3.14                     #
# =========================================================================== #
def test_torch_nll_agrees_with_the_numpy_reference_to_1e6():
    pytest.importorskip("torch")
    from src.train_model import dt_nll_torch
    import torch

    rng = np.random.default_rng(7)
    logits = rng.normal(size=(64, N_INTERVALS)) * 2.0
    t = rng.uniform(0, GRID_MAX_DAYS, size=64)
    e = (rng.uniform(size=64) < 0.3).astype(int)
    at_risk, target, _, _ = discretize_survival(t, e)
    hazards = 1.0 / (1.0 + np.exp(-logits))
    np_mean, np_per = dt_nll_numpy(hazards, at_risk, target)
    pt_mean, pt_per = dt_nll_torch(torch.tensor(logits), torch.tensor(at_risk),
                                   torch.tensor(target))
    assert abs(np_mean - float(pt_mean)) < 1e-6
    assert float(np.abs(np_per - pt_per.numpy()).max()) < 1e-6


def test_weighted_torch_nll_agrees_with_the_numpy_reference():
    pytest.importorskip("torch")
    from src.train_model import dt_nll_torch
    import torch

    rng = np.random.default_rng(7)
    logits = rng.normal(size=(64, N_INTERVALS)) * 2.0
    t = rng.uniform(0, GRID_MAX_DAYS, size=64)
    e = (rng.uniform(size=64) < 0.3).astype(int)
    at_risk, target, _, _ = discretize_survival(t, e)
    hazards = 1.0 / (1.0 + np.exp(-logits))
    w = rng.uniform(0.5, 2.0, size=64)
    np_mean = dt_nll_numpy(hazards, at_risk, target, w)[0]
    pt_mean = float(dt_nll_torch(torch.tensor(logits), torch.tensor(at_risk),
                                 torch.tensor(target), torch.tensor(w))[0])
    assert abs(np_mean - pt_mean) < 1e-6


def test_numpy_torch_nll_agreement_helper_is_within_tolerance():
    pytest.importorskip("torch")
    from src.train_model import numpy_torch_nll_agreement
    d_mean, d_per = numpy_torch_nll_agreement()
    assert d_mean < 1e-6 and d_per < 1e-6


def test_device_resolution_prefers_cuda_then_mps_then_cpu():
    pytest.importorskip("torch")
    import torch
    from src.train_model import resolve_device

    device, amp = resolve_device(["cuda", "mps", "cpu"], ["cuda"])
    if torch.cuda.is_available():
        assert device.type == "cuda" and amp is True
    elif torch.backends.mps.is_available():
        assert device.type == "mps" and amp is False
    else:
        assert device.type == "cpu" and amp is False
    cpu, cpu_amp = resolve_device(["cpu"], ["cuda"])
    assert cpu.type == "cpu" and cpu_amp is False


def test_the_border_is_exactly_zero_after_augmentation():
    """Protocol section 13: rotation drags pixels into the masked band; it is re-zeroed."""
    pytest.importorskip("torch")
    from src.train_model import probe_augmentation

    band = 31
    img = np.zeros((512, 512), dtype=np.uint8)
    rng = np.random.default_rng(1)
    img[band:-band, band:-band] = rng.integers(1, 256, size=(512 - 2 * band, 512 - 2 * band),
                                               dtype=np.uint8)
    aug = parse_augmentation(["rotation_5deg", "translate_0.05", "scale_0.05",
                              "brightness_contrast_0.1"])
    for seed in range(8):
        out = probe_augmentation(img, aug, band, seed=seed)
        assert_border_is_zero(out.numpy(), band)
        assert out.numpy()[band:-band, band:-band].max() > 0     # the crop survived


def test_augmentation_without_the_re_zero_would_leak_into_the_border():
    """The re-zero is load-bearing: a plain rotation DOES pollute the band."""
    pytest.importorskip("torch")
    import torch
    import torchvision.transforms.functional as TF

    band = 31
    img = np.zeros((512, 512), dtype=np.uint8)
    img[band:-band, band:-band] = 255
    rotated = TF.affine(torch.from_numpy(img).unsqueeze(0), angle=5.0, translate=[0, 0],
                        scale=1.0, shear=[0.0, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=0).squeeze(0)
    with pytest.raises(AssertionError, match="masked border"):
        assert_border_is_zero(rotated.numpy(), band)


def test_masked_attention_pool_ignores_padded_slots():
    """Cell 31: padding to a longer set must not change the pooled embedding."""
    pytest.importorskip("torch")
    import torch
    from src.train_model import MaskedAttentionPool

    torch.manual_seed(0)
    pool = MaskedAttentionPool(8).eval()
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        out_padded, attn = pool(x, torch.tensor([[True, False, False]]))
        out_single, _ = pool(x[:, :1], torch.tensor([[True]]))
        out_two, _ = pool(x, torch.tensor([[True, True, False]]))
    assert torch.allclose(out_padded, out_single, atol=1e-6)
    assert abs(float(attn.sum()) - 1.0) < 1e-6
    assert float(attn[0, 1:].abs().sum()) == 0.0
    assert float((out_two - out_padded).abs().max()) > 1e-6


@pytest.mark.parametrize("mode,expect_encoder", [("fusion", True), ("image_only", True),
                                                 ("clinical_only", False)])
def test_survival_fusion_net_mode_wiring(mode, expect_encoder):
    pytest.importorskip("torch")
    from src.train_model import SurvivalFusionNet

    base = np.linspace(0.005, 0.02, N_INTERVALS)
    net = SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=13, n_views=3, mode=mode,
                            arch="convnext_tiny", pretrained=False, base_hazard=base)
    assert hasattr(net, "encoder") is expect_encoder
    assert hasattr(net, "img_norm") is expect_encoder
    assert net.head[-1].out_features == N_INTERVALS


@pytest.mark.parametrize("training", [False, True])
def test_encoding_only_the_masked_slots_is_equivalent_for_convnext(training):
    """The 3x compute the padded slots cost buys nothing: the output is the same tensor."""
    pytest.importorskip("torch")
    import torch
    from src.train_model import SurvivalFusionNet

    torch.manual_seed(0)
    kw = dict(n_intervals=N_INTERVALS, n_clinical=13, n_views=3, mode="fusion",
              arch="convnext_tiny", pretrained=False)
    dense = SurvivalFusionNet(encode_masked_only=False, **kw)
    gathered = SurvivalFusionNet(encode_masked_only=True, **kw)
    gathered.load_state_dict(dense.state_dict())
    dense.train(training); gathered.train(training)

    B, E, S = 3, 5, 64
    rng = np.random.default_rng(2)
    images = torch.from_numpy(rng.integers(0, 256, size=(B, E, S, S), dtype=np.uint8))
    mask = torch.tensor([[True, True, False, False, False],
                         [True, False, False, False, False],
                         [True, True, True, False, False]])
    images = images * mask[..., None, None]                  # padded slots are all-zero
    view_id = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 2, 0, 0]])
    with torch.no_grad():
        a, attn_a = dense.embed_images(images, view_id, mask)
        b, attn_b = gathered.embed_images(images, view_id, mask)
    assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
    assert torch.allclose(attn_a, attn_b, atol=1e-6)


def test_the_masked_gather_still_propagates_gradient_to_the_encoder():
    pytest.importorskip("torch")
    import torch
    from src.train_model import SurvivalFusionNet

    torch.manual_seed(0)
    net = SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=13, n_views=3, mode="image_only",
                            arch="convnext_tiny", pretrained=False)
    B, E, S = 2, 3, 64
    images = torch.randint(0, 256, (B, E, S, S), dtype=torch.uint8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    images = images * mask[..., None, None]
    view_id = torch.zeros(B, E, dtype=torch.long)
    emb, _ = net.embed_images(images, view_id, mask)
    emb.sum().backward()
    grads = [p.grad for n, p in net.encoder.named_parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_the_head_bias_starts_at_the_marginal_hazards():
    pytest.importorskip("torch")
    from src.train_model import SurvivalFusionNet

    base = np.linspace(0.005, 0.02, N_INTERVALS)
    net = SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=13, n_views=3,
                            mode="clinical_only", pretrained=False, base_hazard=base)
    b = net.head[-1].bias.detach().numpy()
    assert np.allclose(1 / (1 + np.exp(-b)), base, atol=1e-5)


def test_a_model_with_neither_branch_is_refused():
    pytest.importorskip("torch")
    from src.train_model import SurvivalFusionNet

    with pytest.raises(AssertionError):
        SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=13, n_views=3, mode="nonsense",
                          pretrained=False)


def test_base_hazard_from_a_toy_dataset_is_events_over_at_risk():
    pytest.importorskip("torch")
    from src.train_model import base_hazard_from

    class _DS:
        pass

    t = np.array([100.0, 400.0, 1000.0, 1826.0])
    e = np.array([1, 1, 0, 0])
    ds = _DS()
    ds.at_risk, ds.target, _, _ = discretize_survival(t, e)
    h = base_hazard_from(ds)
    assert h.shape == (N_INTERVALS,)
    assert h[0] == pytest.approx(1.0 / 4.0, abs=1e-12)
    assert (h > 0).all() and (h < 1).all()


def test_cosine_schedule_warms_up_then_decays():
    pytest.importorskip("torch")
    import torch
    from src.train_model import make_scheduler

    cfg = load_config(DEFAULT_CONFIG)
    settings = TrainSettings(cfg, max_epochs_override=10)
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=settings.lr)
    steps_per_epoch = 5
    sched = make_scheduler(opt, steps_per_epoch, settings)
    lrs = []
    for _ in range(settings.max_epochs * steps_per_epoch):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    warm = settings.warmup_epochs * steps_per_epoch
    assert lrs[0] < lrs[warm - 1]                       # linear warm-up
    assert lrs[warm - 1] == pytest.approx(settings.lr, rel=1e-9)
    assert lrs[-1] < lrs[warm]                          # cosine decay afterwards
    assert min(lrs) >= 0.0


def test_gradient_accumulation_reproduces_the_unsplit_optimiser_trajectory():
    """train_one_epoch with accum=k must reproduce accum=1 step for step and weight for weight.

    ``test_gradient_accumulation_splits_the_batch_without_changing_it`` pins only the
    TrainSettings arithmetic (micro x accum == 32). This one runs the real loop, with the
    production optimiser and the production ``make_scheduler`` wiring, because stage 1 is
    being trained with ``--grad-accum 8``: the property that actually has to hold is that
    splitting the 32-patient optimisation batch into eight micro-batches leaves the
    optimiser trajectory unchanged, not merely that the arithmetic multiplies out.

    PRECONDITION, and the reason the toy cohort is an exact multiple of the optimisation
    batch: with a ragged final batch the two configurations legitimately diverge, because the
    end-of-epoch flush at ``(step + 1) == len(loader)`` divides a short micro-batch by
    ``accum`` as though it were full. That is deferred finding F2, it is NOT covered here, and
    setting ``n_patients`` to a non-multiple of 32 will fail this test for that reason rather
    than because accumulation broke.

    Deliberately tiny and CPU-only: this shares the machine with a live training run and must
    not allocate on the accelerator.
    """
    pytest.importorskip("torch")
    import copy

    import torch
    from torch.utils.data import DataLoader, Dataset

    from src.train_model import make_scheduler, train_one_epoch

    cfg = load_config(DEFAULT_CONFIG)
    device = torch.device("cpu")
    n_patients, n_features, n_epochs = 96, 6, 2
    assert n_patients % TrainSettings(cfg).batch_size == 0, "see PRECONDITION above (F2)"

    rng = np.random.default_rng(11)
    feats = rng.normal(size=(n_patients, n_features)).astype(np.float32)
    t = rng.uniform(30.0, GRID_MAX_DAYS, size=n_patients)
    e = (rng.uniform(size=n_patients) < 0.35).astype(int)
    at_risk, target, _, _ = discretize_survival(t, e)

    class _ToyCohort(Dataset):
        """The three attributes train_one_epoch touches: length, batch dict, loss_weight."""

        def __init__(self):
            self.loss_weight = np.ones(n_patients)   # what make_loader sets without oversampling

        def __len__(self):
            return n_patients

        def __getitem__(self, i):
            return {"x": torch.from_numpy(feats[i]),
                    "at_risk": torch.from_numpy(at_risk[i].astype(np.float32)),
                    "target": torch.from_numpy(target[i].astype(np.float32)),
                    "idx": torch.tensor(i)}

    class _ToyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(n_features, N_INTERVALS)

        def forward(self, batch):
            return self.fc(batch["x"])

    torch.manual_seed(0)
    init_state = copy.deepcopy(_ToyHead().state_dict())

    def run(accum):
        settings = TrainSettings(cfg, grad_accum_override=accum, max_epochs_override=n_epochs)
        ds = _ToyCohort()
        loader = DataLoader(ds, batch_size=settings.micro_batch_size, shuffle=False)
        model = _ToyHead()
        model.load_state_dict(copy.deepcopy(init_state))
        opt = torch.optim.AdamW(model.parameters(), lr=settings.lr,
                                weight_decay=settings.weight_decay)
        # Counted with the optimiser's own post-step hook rather than by shadowing opt.step,
        # which LambdaLR both re-wraps and warns about. sched.last_epoch is LambdaLR's own
        # count of explicit step() calls (init leaves it at 0).
        n_opt = {"n": 0}
        opt.register_step_post_hook(lambda *_a, **_k: n_opt.__setitem__("n", n_opt["n"] + 1))
        # Exactly the production expression: scheduler steps are OPTIMISER steps per epoch.
        sched = make_scheduler(
            opt, max(1, math.ceil(len(loader) / max(1, settings.grad_accum_steps))), settings)
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        losses = [train_one_epoch(model, loader, opt, sched, scaler, ds, device=device,
                                  settings=settings, amp=False) for _ in range(n_epochs)]
        counts = {"opt": n_opt["n"], "sched": int(sched.last_epoch)}
        return counts, len(loader), losses, [p.detach().clone() for p in model.parameters()]

    plain_counts, plain_batches, plain_losses, plain_params = run(1)
    split_counts, split_batches, split_losses, split_params = run(8)

    # The split really happened: eight times as many forward passes, same optimiser steps.
    assert split_batches == 8 * plain_batches == 24
    assert plain_counts["opt"] == n_epochs * (n_patients // 32) == 6
    assert split_counts["opt"] == plain_counts["opt"]
    assert split_counts["sched"] == plain_counts["sched"] == plain_counts["opt"]

    # Same trajectory, not merely the same number of steps.
    for a, b in zip(plain_losses, split_losses):
        assert abs(a - b) < 1e-6
    assert len(split_params) == len(plain_params) > 0
    # Calibration: this toy head reproduces exactly (0.0) and the production float32 model was
    # measured at 1.5e-8, while the ragged-tail divergence this test excludes (F2) is ~2e-4 on
    # the same setup. 1e-6 therefore sits far below the failure mode and far above float noise.
    worst = max(float((p - q).abs().max()) for p, q in zip(plain_params, split_params))
    assert worst < 1e-6, f"accum=8 drifted from accum=1 by {worst:.3e} on a divisible epoch"


def test_require_torch_is_a_no_op_when_torch_is_installed():
    pytest.importorskip("torch")
    from src.train_model import TORCH_AVAILABLE, require_torch
    assert TORCH_AVAILABLE is True
    require_torch()
