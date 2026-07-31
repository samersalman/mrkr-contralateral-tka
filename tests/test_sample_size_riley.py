"""Unit tests for the PURE formula helpers in src/sample_size_riley.py.

Nothing here reads a project data file. The Riley criteria are pinned against
hand-computed arithmetic AND against the published `pmsampsize` survival worked
example (`pmsampsize(type="s", csrsquared=0.051, parameters=30, rate=0.065,
timepoint=2, meanfup=2.07)`), which the reference implementation reports as
n1 = 5143, n2 = 1039, max R2_CS = 0.555, Nagelkerke = 0.092. The simulation
helpers are checked for correctness (against lifelines and against closed forms)
and for reproducibility under the config seed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.sample_size_riley import (
    ADMIN_HORIZON_DAYS,
    EXPECTED_DESIGN_COLUMNS,
    EXPECTED_IDENTIFIED_PARAMS,
    IPCW_WEIGHT_FLOOR,
    Z975,
    auc_cd_analytic,
    baseline_hazard_grid,
    candidate_parameter_count,
    cox_slope_univariate,
    hanley_mcneil_se,
    km_numpy,
    marginal_risk,
    max_parameters_supported,
    max_r2_cs_survival,
    min_r2_fraction_supported,
    n_for_risk_precision,
    n_for_shrinkage,
    observed_inputs,
    published_cohort_aggregates,
    required_n_for_halfwidth,
    riley_survival,
    shrinkage_for_small_optimism,
    simulate_replicates,
    solve_cum_hazard,
    sigma_for_auc,
    step_eval,
    summarise_replicates,
    uno_auc_cd,
)

SEED = 20250720                                  # config reproducibility.random_seed

# The published pmsampsize survival example.
PM_R2, PM_P, PM_RATE, PM_T, PM_FUP = 0.051, 30, 0.065, 2.0, 2.07


# --------------------------------------------------------------------------- #
# max R2_CS (Riley 2019 eq. 23)                                                #
# --------------------------------------------------------------------------- #
def test_max_r2_cs_matches_hand_arithmetic():
    # phi = 0.065 * 2.07 = 0.13455;  ln(phi) = -2.0058215
    # 2 * (0.13455 * -2.0058215 - 0.13455) = -0.8088663;  exp(...) = 0.4453628
    # max R2_CS = 1 - 0.4453628 = 0.5546372
    phi = PM_RATE * PM_FUP
    assert phi == pytest.approx(0.13455, abs=1e-9)
    assert max_r2_cs_survival(phi) == pytest.approx(0.5546372, abs=1e-6)


def test_max_r2_cs_matches_pmsampsize_reference_rounding():
    """pmsampsize builds it from events=ceil(rate*meanfup*10000) at n=10000."""
    n_ref = 10000
    events = math.ceil(PM_RATE * PM_FUP * n_ref)
    ln_null = events * math.log(events / n_ref) - events
    reference = 1.0 - math.exp(2.0 * ln_null / n_ref)
    assert round(reference, 3) == 0.555
    assert max_r2_cs_survival(PM_RATE * PM_FUP) == pytest.approx(reference, abs=1e-4)


def test_max_r2_cs_increases_with_event_fraction_up_to_a_half():
    vals = [max_r2_cs_survival(p) for p in (0.02, 0.05, 0.10, 0.20, 0.40)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert all(0.0 < v < 1.0 for v in vals)


def test_max_r2_cs_rejects_impossible_fractions():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(AssertionError):
            max_r2_cs_survival(bad)


# --------------------------------------------------------------------------- #
# Criterion 1 - shrinkage                                                      #
# --------------------------------------------------------------------------- #
def test_criterion1_reproduces_published_example():
    # n1 = 30 / ((0.9-1) * ln(1 - 0.051/0.9)) = 30 / 0.00583317 = 5142.87 -> 5143
    n1 = n_for_shrinkage(PM_P, PM_R2, 0.9)
    assert n1 == pytest.approx(5142.87, abs=0.5)
    assert math.ceil(n1) == 5143


def test_criterion1_hand_arithmetic_for_this_study():
    # P = 15, max R2_CS = 0.570433 at phi = 533/3709, f = 0.15 -> R2 = 0.0855649
    # n1 = 15 / ((0.9-1)*ln(1 - 0.0855649/0.9)) = 15 / 0.00998927 = 1501.6 -> 1502
    max_r2 = max_r2_cs_survival(533 / 3709)
    assert max_r2 == pytest.approx(0.570433, abs=1e-5)
    n1 = n_for_shrinkage(15, 0.15 * max_r2, 0.9)
    assert math.ceil(n1) == 1502


def test_criterion1_is_linear_in_parameters():
    a = n_for_shrinkage(10, 0.08, 0.9)
    b = n_for_shrinkage(20, 0.08, 0.9)
    assert b == pytest.approx(2 * a, rel=1e-12)


def test_criterion1_monotone_in_parameters_and_r2():
    base = n_for_shrinkage(15, 0.08, 0.9)
    assert n_for_shrinkage(16, 0.08, 0.9) > base            # more parameters -> larger n
    assert n_for_shrinkage(15, 0.12, 0.9) < base            # higher R2 -> smaller n
    assert n_for_shrinkage(15, 0.08, 0.95) > base           # stricter shrinkage -> larger n


def test_criterion1_rejects_r2_above_shrinkage():
    with pytest.raises(AssertionError):
        n_for_shrinkage(15, 0.95, 0.9)


# --------------------------------------------------------------------------- #
# Criterion 2 - small optimism                                                 #
# --------------------------------------------------------------------------- #
def test_criterion2_reproduces_published_example():
    max_r2 = max_r2_cs_survival(PM_RATE * PM_FUP)
    s2 = shrinkage_for_small_optimism(PM_R2, max_r2, 0.05)
    assert s2 == pytest.approx(0.64773, abs=5e-5)
    assert math.ceil(n_for_shrinkage(PM_P, PM_R2, s2)) == 1039


def test_nagelkerke_of_published_example():
    assert PM_R2 / max_r2_cs_survival(PM_RATE * PM_FUP) == pytest.approx(0.092, abs=5e-4)


def test_criterion2_shrinkage_collapses_to_f_over_f_plus_delta():
    """With R2 = f * maxR2, S2 = f/(f+delta) exactly, independent of maxR2."""
    for max_r2 in (0.2, 0.5704, 0.8):
        for f, delta in ((0.15, 0.05), (0.10, 0.05), (0.30, 0.02)):
            got = shrinkage_for_small_optimism(f * max_r2, max_r2, delta)
            assert got == pytest.approx(f / (f + delta), rel=1e-12)


def test_criterion2_hand_arithmetic_for_this_study():
    # f = 0.15, delta = 0.05 -> S2 = 0.75 exactly; R2 = 0.0855649
    # n2 = 15 / ((0.75-1)*ln(1-0.0855649/0.75)) = 15/0.0302915 = 495.2 -> 496
    max_r2 = max_r2_cs_survival(533 / 3709)
    s2 = shrinkage_for_small_optimism(0.15 * max_r2, max_r2, 0.05)
    assert s2 == pytest.approx(0.75, rel=1e-12)
    assert math.ceil(n_for_shrinkage(15, 0.15 * max_r2, s2)) == 496


# --------------------------------------------------------------------------- #
# Criterion 3 - precision of the overall risk                                  #
# --------------------------------------------------------------------------- #
def test_criterion3_closed_form_hits_the_target_margin():
    """The solved n must make the margin of error land exactly on the target."""
    from src.sample_size_riley import risk_ci_exponential
    rate, fup, t, mape = 0.057238, 2.51066, 5.0, 0.05
    n = n_for_risk_precision(rate, fup, t, mape)
    lo, pt, hi = risk_ci_exponential(rate, fup, t, n)
    assert hi - pt == pytest.approx(mape, abs=1e-9)
    assert lo < pt < hi


def test_criterion3_hand_arithmetic_for_this_study():
    # S_exp(5) = exp(-0.057238*5) = 0.751124; mape/S = 0.066567
    # ln(1-0.066567) = -0.0688864; SE_max = 0.0688864/(1.96*5) = 0.00702922
    # n = 0.057238 / (2.51066 * 0.00702922^2) = 461.5 -> 462
    n = n_for_risk_precision(0.057238, 2.51066, 5.0, 0.05)
    assert math.ceil(n) == 462


def test_criterion3_monotone():
    base = n_for_risk_precision(0.057238, 2.51066, 5.0, 0.05)
    assert n_for_risk_precision(0.057238, 2.51066, 5.0, 0.03) > base   # tighter margin
    assert n_for_risk_precision(0.057238, 5.0, 5.0, 0.05) < base       # longer follow-up


def test_criterion3_rejects_unreachable_margin():
    # a margin wider than the survival probability itself can never be attained
    with pytest.raises(AssertionError):
        n_for_risk_precision(0.5, 2.0, 5.0, 0.5)


# --------------------------------------------------------------------------- #
# The assembled calculation                                                    #
# --------------------------------------------------------------------------- #
def test_riley_survival_reproduces_the_published_example_end_to_end():
    r = riley_survival(parameters=PM_P, r2_cs_adj=PM_R2, rate=PM_RATE, timepoint=PM_T,
                       mean_followup=PM_FUP, shrinkage=0.9, max_optimism=0.05, mape=0.05)
    assert r["n_criterion1"] == 5143
    assert r["n_criterion2"] == 1039
    assert round(r["max_r2_cs"], 3) == 0.555
    assert round(r["nagelkerke_r2"], 3) == 0.092
    assert r["n_required"] == max(r["n_criterion1"], r["n_criterion2"], r["n_criterion3"])


def test_riley_survival_for_this_study():
    max_r2 = max_r2_cs_survival(533 / 3709)
    r = riley_survival(parameters=15, r2_cs_adj=0.15 * max_r2, rate=0.057238, timepoint=5.0,
                       mean_followup=2.51066, shrinkage=0.9, max_optimism=0.05, mape=0.05,
                       event_fraction=533 / 3709)
    assert (r["n_criterion1"], r["n_criterion2"], r["n_criterion3"]) == (1502, 496, 462)
    assert r["n_required"] == 1502
    assert r["binding_criterion"] == "1_shrinkage"
    assert r["events_required"] == pytest.approx(1502 * 533 / 3709, rel=1e-12)
    assert r["epp_required"] == pytest.approx(r["events_required"] / 15, rel=1e-12)


def test_required_n_falls_as_the_assumed_r2_rises():
    max_r2 = max_r2_cs_survival(533 / 3709)
    ns = [riley_survival(15, f * max_r2, 0.057238, 5.0, 2.51066, 0.9, 0.05, 0.05,
                         event_fraction=533 / 3709)["n_required"]
          for f in (0.10, 0.15, 0.20, 0.30)]
    assert all(b <= a for a, b in zip(ns, ns[1:])), ns
    assert ns == [2291, 1502, 1107, 712]


def test_required_n_rises_with_more_parameters():
    max_r2 = max_r2_cs_survival(533 / 3709)
    ns = [riley_survival(p, 0.15 * max_r2, 0.057238, 5.0, 2.51066, 0.9, 0.05, 0.05,
                         event_fraction=533 / 3709)["n_required"] for p in (5, 15, 30, 60)]
    assert all(b > a for a, b in zip(ns, ns[1:])), ns


def test_riley_rejects_r2_above_the_maximum():
    max_r2 = max_r2_cs_survival(533 / 3709)
    with pytest.raises(AssertionError):
        riley_survival(15, 1.05 * max_r2, 0.057238, 5.0, 2.51066, 0.95, 0.05, 0.05,
                       event_fraction=533 / 3709)


# --------------------------------------------------------------------------- #
# Headroom helpers                                                             #
# --------------------------------------------------------------------------- #
def test_min_r2_fraction_supported_is_the_break_even_point():
    max_r2 = max_r2_cs_survival(533 / 3709)
    f = min_r2_fraction_supported(15, 2968, max_r2, 0.9, 0.05)
    need_at_f = n_for_shrinkage(15, f * max_r2, 0.9)
    assert need_at_f == pytest.approx(2968, rel=1e-4)
    # just below the break-even the requirement exceeds what is available
    assert n_for_shrinkage(15, 0.98 * f * max_r2, 0.9) > 2968


def test_max_parameters_supported_is_consistent_with_criterion1():
    max_r2 = max_r2_cs_survival(533 / 3709)
    p_max = max_parameters_supported(2968, 0.15 * max_r2, max_r2, 0.9, 0.05)
    assert n_for_shrinkage(p_max, 0.15 * max_r2, 0.9) <= 2968
    assert n_for_shrinkage(p_max + 1, 0.15 * max_r2, 0.9) > 2968


# --------------------------------------------------------------------------- #
# Survival utilities                                                           #
# --------------------------------------------------------------------------- #
def test_km_numpy_matches_a_hand_computed_curve():
    # times 1,2,3,4 with events 1,0,1,1 -> S = 3/4, 3/4, (3/4)(1/2), (3/4)(1/2)(0)
    t = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([1, 0, 1, 1])
    ut, s, gvar = km_numpy(t, d)
    assert list(ut) == [1.0, 2.0, 3.0, 4.0]
    assert s == pytest.approx([0.75, 0.75, 0.375, 0.0])
    # Greenwood at t=1: S^2 * d/(n(n-d)) = 0.5625 * 1/(4*3) = 0.046875
    assert gvar[0] == pytest.approx(0.046875)


def test_km_numpy_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(SEED)
    t = np.round(rng.exponential(400, 300), 1) + 1.0
    d = (rng.random(300) < 0.4).astype(int)
    ut, s, _ = km_numpy(t, d)
    ref = lifelines.KaplanMeierFitter().fit(t, d)
    assert np.allclose(s, ref.predict(ut).to_numpy(), atol=1e-12)


def test_step_eval_left_limit_versus_right_continuous():
    ut = np.array([1.0, 2.0, 3.0])
    s = np.array([0.9, 0.5, 0.1])
    assert step_eval(ut, s, [0.5])[0] == 1.0                    # before the first time
    assert step_eval(ut, s, [2.0])[0] == pytest.approx(0.5)     # right-continuous
    assert step_eval(ut, s, [2.0], left_limit=True)[0] == pytest.approx(0.9)


def test_cox_slope_matches_lifelines_on_untied_data():
    """No ties, so Breslow and Efron coincide and lifelines is a valid reference."""
    lifelines = pytest.importorskip("lifelines")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(SEED)
    n = 400
    x = rng.normal(0, 1, n)
    t = rng.exponential(1.0 / np.exp(0.6 * x))
    d = (rng.random(n) < 0.6).astype(int)
    beta, se, ok = cox_slope_univariate(t, d, x)
    assert ok is True
    ref = lifelines.CoxPHFitter().fit(pd.DataFrame(dict(t=t, e=d, x=x)), "t", "e")
    assert beta == pytest.approx(float(ref.params_["x"]), abs=1e-6)
    assert se == pytest.approx(float(ref.standard_errors_["x"]), abs=1e-6)


def test_cox_slope_recovers_a_known_coefficient():
    rng = np.random.default_rng(SEED)
    n = 20000
    x = rng.normal(0, 1, n)
    t = rng.exponential(1.0 / np.exp(0.8 * x))
    beta, se, ok = cox_slope_univariate(t, np.ones(n, int), x)
    assert ok is True
    assert beta == pytest.approx(0.8, abs=4 * se)


def test_cox_slope_returns_nan_without_events():
    beta, se, ok = cox_slope_univariate(np.arange(1.0, 6.0), np.zeros(5, int), np.arange(5.0))
    assert math.isnan(beta) and math.isnan(se) and ok is False


def test_cox_slope_reports_non_convergence_instead_of_the_last_iterate():
    """Starved of iterations the solver must say so, not hand back a half-finished beta."""
    rng = np.random.default_rng(SEED)
    n = 300
    x = rng.normal(0, 1, n)
    t = rng.exponential(1.0 / np.exp(1.5 * x))
    d = np.ones(n, int)
    beta_ok, _, ok = cox_slope_univariate(t, d, x)
    assert ok is True and math.isfinite(beta_ok)
    beta_bad, se_bad, ok_bad = cox_slope_univariate(t, d, x, maxit=1)
    assert ok_bad is False, "one Newton step cannot have converged"
    assert math.isnan(beta_bad) and math.isnan(se_bad), \
        "a non-converged fit must be NaN, not the last iterate"


# --------------------------------------------------------------------------- #
# uno_auc_cd — HAND-WORKED weighted example.                                    #
#                                                                               #
#   times  = [2, 2, 3, 5, 7]   events = [1, 0, 1, 0, 1]   horizon = 4           #
#   Censoring KM (flip the indicator; censorings at t = 2 and t = 5):           #
#     unique times [2, 3, 5, 7], at risk [5, 3, 2, 1], censorings [1, 0, 1, 0]  #
#     G = [0.8, 0.8, 0.4, 0.4]                                                  #
#   Cases (T <= 4 and event):  T=2 -> w = 1/G(2-) = 1/1   = 1                   #
#                              T=3 -> w = 1/G(3-) = 1/0.8 = 1.25                #
#   Controls (T > 4):          T=5 (censored) AND T=7 (EVENT after the horizon, #
#                              therefore event-free AT 4 and a legitimate       #
#                              control) -> n_ctrl = 2                           #
#   lp = [10, 0, 1, 5, 0]:  case@2 beats both controls (2), case@3 beats one (1)#
#   AUC = (1*2 + 1.25*1) / ((1 + 1.25) * 2) = 3.25 / 4.5 = 13/18                #
#                                                                               #
# This single value kills four mutants: inverting the case weight (0.7778),     #
# disabling IPCW via the floor (0.75), restoring `event == 0` on controls       #
# (0.4444) and reading G right-continuously instead of as a left limit (0.75).  #
# --------------------------------------------------------------------------- #
HW_T = np.array([2.0, 2.0, 3.0, 5.0, 7.0])
HW_D = np.array([1, 0, 1, 0, 1])
HW_LP = np.array([10.0, 0.0, 1.0, 5.0, 0.0])
HW_HORIZON = 4.0


def test_uno_auc_matches_the_hand_worked_weighted_value():
    auc, n_case, n_ctrl = uno_auc_cd(HW_T, HW_D, HW_LP, HW_HORIZON)
    assert (n_case, n_ctrl) == (2, 2)
    assert auc == pytest.approx(13 / 18)


def test_uno_auc_censoring_curve_matches_the_hand_worked_arithmetic():
    """The G values the hand-worked weights are built on, checked independently."""
    gt, gv, _ = km_numpy(HW_T, 1 - HW_D)
    assert list(gt) == [2.0, 3.0, 5.0, 7.0]
    assert gv == pytest.approx([0.8, 0.8, 0.4, 0.4])
    assert step_eval(gt, gv, [2.0], left_limit=True)[0] == 1.0        # G(2-) — case weight 1
    assert step_eval(gt, gv, [2.0])[0] == pytest.approx(0.8)          # G(2) right-continuous
    assert step_eval(gt, gv, [3.0], left_limit=True)[0] == pytest.approx(0.8)


def test_uno_auc_case_weights_are_inverse_G_not_G():
    """Mutation guard (a): with w = G instead of 1/G the answer would be 2.8/3.6."""
    auc, _, _ = uno_auc_cd(HW_T, HW_D, HW_LP, HW_HORIZON)
    assert auc == pytest.approx(13 / 18)
    assert auc != pytest.approx(2.8 / 3.6, abs=1e-6)


def test_uno_auc_weight_floor_is_inert_by_default_and_disables_ipcw_when_raised():
    """Mutation guard (b): the floor is a guard, not an operating parameter."""
    assert IPCW_WEIGHT_FLOOR == 1e-3
    assert uno_auc_cd(HW_T, HW_D, HW_LP, HW_HORIZON)[0] == pytest.approx(13 / 18)
    assert uno_auc_cd(HW_T, HW_D, HW_LP, HW_HORIZON,
                      weight_floor=IPCW_WEIGHT_FLOOR)[0] == pytest.approx(13 / 18)
    # floor = 1 clamps every G to 1, i.e. all weights equal -> the unweighted AUC 3/4.
    off, _, _ = uno_auc_cd(HW_T, HW_D, HW_LP, HW_HORIZON, weight_floor=1.0)
    assert off == pytest.approx(0.75)
    assert off != pytest.approx(13 / 18, abs=1e-6)


def test_uno_auc_counts_a_late_event_patient_as_a_control():
    """Mutation guard (c): controls are T > t, with NO condition on the event indicator.

    A patient whose event happens after the horizon is event-free AT the horizon and is
    the single most informative control there is. Requiring `event == 0` drops exactly
    those patients and biases the AUROC upward.
    """
    auc, _, n_ctrl = uno_auc_cd(HW_T, HW_D, HW_LP, HW_HORIZON)
    assert n_ctrl == 2, "T=5 (censored) and T=7 (event AFTER the horizon) are both controls"
    assert auc == pytest.approx(13 / 18)

    def mutant(time, event, lp, horizon):
        """The old definition: controls restricted to (T >= horizon) AND event == 0."""
        case = (time <= horizon) & (event == 1)
        ctrl = (time >= horizon) & (event == 0)
        gt, gv, _ = km_numpy(time, 1 - event)
        w = 1.0 / np.maximum(step_eval(gt, gv, time[case], left_limit=True),
                             IPCW_WEIGHT_FLOOR)
        conc = (lp[case][:, None] > lp[ctrl][None, :]).astype(float)
        return float((w[:, None] * conc).sum() / (w.sum() * int(ctrl.sum()))), int(ctrl.sum())

    auc_mut, n_ctrl_mut = mutant(HW_T, HW_D, HW_LP, HW_HORIZON)
    assert n_ctrl_mut == 1, "the mutant drops the late-event control"
    assert auc_mut == pytest.approx(1 / 2.25)
    assert auc_mut != pytest.approx(13 / 18, abs=1e-6)


def test_uno_auc_case_weight_uses_the_left_limit_of_G():
    """Mutation guard (d): G evaluated right-continuously at a case time under-weights it.

    In the hand-worked data a censoring and an event share time 2. The left limit gives
    that case weight 1; the right-continuous value would give 1/0.8 = 1.25 and make both
    case weights equal, collapsing the answer to the unweighted 0.75.
    """
    auc, _, _ = uno_auc_cd(HW_T, HW_D, HW_LP, HW_HORIZON)
    assert auc == pytest.approx(13 / 18)
    assert auc != pytest.approx(0.75, abs=1e-6)


def test_uno_auc_control_condition_is_strictly_greater_than_the_horizon():
    """A patient observed only up to exactly t is NOT known event-free after t.

    This is why the 5-year horizon must be clamped off the administrative censoring day:
    there, every T equals the horizon, the control arm is empty and the estimator is
    undefined.
    """
    t = np.array([1.0, 4.0, 4.0, 4.0])
    d = np.array([1, 0, 0, 0])
    auc, n_case, n_ctrl = uno_auc_cd(t, d, np.array([3.0, 2.0, 1.0, 0.0]), 4.0)
    assert (n_case, n_ctrl) == (1, 0), "T == horizon is not a control"
    assert math.isnan(auc)
    # One day earlier the estimator is well defined again.
    auc9, n_case9, n_ctrl9 = uno_auc_cd(t, d, np.array([3.0, 2.0, 1.0, 0.0]), 3.0)
    assert (n_case9, n_ctrl9) == (1, 3) and auc9 == pytest.approx(1.0)


def test_uno_auc_is_one_for_a_perfect_predictor():
    t = np.array([100.0, 200.0, 300.0, 900.0, 900.0, 900.0])
    d = np.array([1, 1, 1, 0, 0, 0])
    lp = np.array([5.0, 4.0, 3.0, 0.0, -1.0, -2.0])          # every case above every control
    auc, n_case, n_ctrl = uno_auc_cd(t, d, lp, 899.0)
    assert (n_case, n_ctrl) == (3, 3)
    assert auc == pytest.approx(1.0)


def test_uno_auc_is_a_half_for_identical_predictions():
    t = np.array([100.0, 200.0, 900.0, 900.0])
    d = np.array([1, 1, 0, 0])
    auc, _, _ = uno_auc_cd(t, d, np.zeros(4), 899.0)
    assert auc == pytest.approx(0.5)


def test_uno_auc_nan_without_cases_or_controls():
    t = np.array([900.0, 900.0, 900.0])
    auc, n_case, n_ctrl = uno_auc_cd(t, np.zeros(3, int), np.arange(3.0), 899.0)
    assert math.isnan(auc) and n_case == 0 and n_ctrl == 3


def test_hanley_mcneil_se_shrinks_with_more_cases():
    a = hanley_mcneil_se(0.7, 100, 200)
    b = hanley_mcneil_se(0.7, 400, 800)
    assert b < a
    assert b == pytest.approx(a / 2.0, rel=0.05)              # roughly 1/sqrt(n)


def test_required_n_scaling():
    assert required_n_for_halfwidth(741, 0.10, 0.05) == 2964   # 741 * 4
    assert required_n_for_halfwidth(741, 0.04, 0.05) == 741    # already met -> unchanged


# --------------------------------------------------------------------------- #
# Data-generating process for the test-precision simulation                    #
# --------------------------------------------------------------------------- #
def test_marginal_risk_is_monotone_and_bounded():
    vals = [marginal_risk(h, 0.7) for h in (0.01, 0.1, 0.5, 2.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert 0.0 < vals[0] < vals[-1] < 1.0


def test_solve_cum_hazard_inverts_marginal_risk():
    for sigma in (0.3, 0.7, 1.2):
        for risk in (0.05, 0.2, 0.5):
            h = solve_cum_hazard(risk, sigma)
            assert marginal_risk(h, sigma) == pytest.approx(risk, abs=1e-8)


def test_auc_is_a_half_without_spread_and_rises_with_sigma():
    """As sigma -> 0 every patient has the same risk, so the AUROC collapses to 0.5.

    The approach is O(sigma): for small sigma the case/control mean separation is
    sigma^2 * F'(0) / (F(0)(1-F(0))), so AUROC - 0.5 ~ sigma * 1.116 / sqrt(2) / sqrt(2pi)
    at a 20% risk (about 0.0031 at sigma = 0.01, 0.00031 at sigma = 0.001).
    """
    risk = 0.2
    aucs = [auc_cd_analytic(solve_cum_hazard(risk, s), s) for s in (0.001, 0.5, 1.0, 2.0)]
    assert aucs[0] == pytest.approx(0.5, abs=1e-3)
    assert all(b > a for a, b in zip(aucs, aucs[1:]))
    small = [auc_cd_analytic(solve_cum_hazard(risk, s), s) - 0.5 for s in (0.01, 0.001)]
    assert small[0] / small[1] == pytest.approx(10.0, rel=0.05)     # linear in sigma


def test_sigma_for_auc_inverts_the_analytic_auc():
    for target in (0.65, 0.70, 0.75, 0.80):
        sigma = sigma_for_auc(target, 0.20032)
        assert auc_cd_analytic(solve_cum_hazard(0.20032, sigma), sigma) == \
            pytest.approx(target, abs=1e-8)


def test_baseline_hazard_grid_reproduces_the_marginal_survival():
    surv = np.array([0.98, 0.93, 0.88, 0.82, 0.80])
    h0 = baseline_hazard_grid(0.7, surv)
    assert np.all(np.diff(h0) > 0)
    for h, s in zip(h0, surv):
        assert marginal_risk(h, 0.7) == pytest.approx(1.0 - s, abs=1e-6)


# --------------------------------------------------------------------------- #
# Simulation: correctness and reproducibility under the config seed            #
# --------------------------------------------------------------------------- #
# Evaluation horizon for the toy DGP. It is day 1825, NOT the administrative censoring
# day 1826: at 1826 every patient has T == horizon, so the cumulative/dynamic control arm
# (T > horizon) is empty and the estimator is undefined. Same reason the real pipeline
# clamps (src.model_clinical.clamp_horizon_days).
TOY_HORIZON = 1825.0


def _toy_dgp(target_auc: float = 0.70, risk: float = 0.20):
    """A small synthetic event curve + censoring curve on a 1826-day follow-up."""
    t_grid = np.linspace(30.0, 1826.0, 60)
    surv = np.exp(-np.linspace(0.0, -math.log(1.0 - risk), 60))
    surv[-1] = 1.0 - risk
    surv = np.minimum.accumulate(surv)
    sigma = sigma_for_auc(target_auc, risk)
    h0 = baseline_hazard_grid(sigma, surv)
    cens_t = np.array([400.0, 900.0, 1400.0, 1826.0])
    cens_s = np.array([0.80, 0.60, 0.40, 0.0])
    return sigma, h0, t_grid, cens_t, cens_s


def test_simulation_is_reproducible_under_the_config_seed():
    sigma, h0, t_grid, cens_t, cens_s = _toy_dgp()
    kw = dict(n_test=741, sigma=sigma, h0_grid=h0, t_grid=t_grid, cens_t=cens_t,
              cens_s=cens_s, horizon=TOY_HORIZON, n_sim=40)
    a = simulate_replicates(np.random.default_rng(SEED), **kw)
    b = simulate_replicates(np.random.default_rng(SEED), **kw)
    assert a.equals(b)
    c = simulate_replicates(np.random.default_rng(SEED + 1), **kw)
    assert not a["auroc"].equals(c["auroc"])


def test_simulated_calibration_slope_is_unbiased_and_auroc_recovers_the_target():
    sigma, h0, t_grid, cens_t, cens_s = _toy_dgp(target_auc=0.70, risk=0.20)
    reps = simulate_replicates(np.random.default_rng(SEED), 741, sigma, h0, t_grid,
                               cens_t, cens_s, TOY_HORIZON, 400)
    s = summarise_replicates(reps)
    mc_se_slope = s["slope_sd"] / math.sqrt(s["n_valid_slope"])
    mc_se_auroc = s["auroc_sd"] / math.sqrt(s["n_valid_auroc"])
    assert s["slope_mean"] == pytest.approx(1.0, abs=4 * mc_se_slope)
    assert s["auroc_mean"] == pytest.approx(0.70, abs=0.02 + 4 * mc_se_auroc)


def test_simulated_precision_scales_as_one_over_sqrt_n():
    sigma, h0, t_grid, cens_t, cens_s = _toy_dgp()
    small = summarise_replicates(simulate_replicates(
        np.random.default_rng(SEED), 741, sigma, h0, t_grid, cens_t, cens_s, TOY_HORIZON, 400))
    big = summarise_replicates(simulate_replicates(
        np.random.default_rng(SEED), 2964, sigma, h0, t_grid, cens_t, cens_s, TOY_HORIZON, 400))
    assert big["slope_halfwidth"] == pytest.approx(small["slope_halfwidth"] / 2.0, rel=0.15)
    assert big["auroc_halfwidth"] == pytest.approx(small["auroc_halfwidth"] / 2.0, rel=0.20)


def test_monte_carlo_and_wald_slope_halfwidths_agree():
    sigma, h0, t_grid, cens_t, cens_s = _toy_dgp()
    s = summarise_replicates(simulate_replicates(
        np.random.default_rng(SEED), 741, sigma, h0, t_grid, cens_t, cens_s, TOY_HORIZON, 400))
    assert s["slope_halfwidth"] == pytest.approx(s["slope_halfwidth_wald"], rel=0.12)


def test_z975_constant():
    stats = pytest.importorskip("scipy.stats")
    assert Z975 == pytest.approx(float(stats.norm.ppf(0.975)), abs=1e-12)


# --------------------------------------------------------------------------- #
# Candidate parameter count (derived, not hard-coded)                          #
# --------------------------------------------------------------------------- #
def _write_params(tmp_path, n_cols=12):
    import json

    (tmp_path / "clinical_imputation_params.json").write_text(json.dumps(
        {"model_columns": ["age_at_index_imp"] + [f"v{i}_imp" for i in range(n_cols - 1)]}))


def test_candidate_parameter_count_falls_back_to_columns_without_the_model_json(tmp_path):
    """No fitted model yet -> the COLUMN count, flagged as an upper bound on P."""
    _write_params(tmp_path, 13)
    cfg = {"model_clinical": {"age_rcs_df": 3},
           "sample_size": {"extra_image_parameters": 0}}
    p, detail = candidate_parameter_count(cfg, tmp_path)
    assert p == 15                                              # 13 - 1 + 3 + 0
    assert detail["n_model_columns"] == 13
    assert detail["n_design_columns"] == 15
    assert detail["identified_parameters"] is None
    assert "upper bound" in detail["parameters_source"]
    assert detail["age_columns"] == ["age_at_index_imp"]


def test_candidate_parameter_count_prefers_the_identified_count_from_the_model_json(tmp_path):
    """The Riley input is what the likelihood can ESTIMATE, not how many columns exist."""
    import json

    # M0 (protocol Table 7) holds 11 model columns -> 13 design columns, 12 identified.
    _write_params(tmp_path, 11)
    (tmp_path / "m0_clinical_model.json").write_text(json.dumps(
        {"n_parameters": 13, "identified_parameters": 12}))
    cfg = {"model_clinical": {"age_rcs_df": 3},
           "sample_size": {"extra_image_parameters": 0}}
    p, detail = candidate_parameter_count(cfg, tmp_path)
    assert p == EXPECTED_IDENTIFIED_PARAMS == 12
    assert detail["n_design_columns"] == EXPECTED_DESIGN_COLUMNS == 13
    assert detail["parameters_source"].endswith("identified_parameters")
    assert p < detail["n_design_columns"], "the spline partition of unity costs one parameter"


def test_candidate_parameter_count_rejects_a_stale_model_json(tmp_path):
    """A model fitted on a different column set must not silently supply P."""
    import json

    _write_params(tmp_path, 12)
    (tmp_path / "m0_clinical_model.json").write_text(json.dumps(
        {"n_parameters": 15, "identified_parameters": 13}))
    cfg = {"model_clinical": {"age_rcs_df": 3},
           "sample_size": {"extra_image_parameters": 0}}
    with pytest.raises(AssertionError):
        candidate_parameter_count(cfg, tmp_path)


def test_candidate_parameter_count_honours_extra_image_parameters(tmp_path):
    _write_params(tmp_path, 13)
    cfg = {"model_clinical": {"age_rcs_df": 4},
           "sample_size": {"extra_image_parameters": 8}}
    p, _ = candidate_parameter_count(cfg, tmp_path)
    assert p == 13 - 1 + 4 + 8


def test_riley_requirement_falls_when_p_drops_from_15_to_the_identified_13():
    """The corrected P is smaller, so the old P = 15 requirement was CONSERVATIVE."""
    max_r2 = max_r2_cs_survival(427 / 2968)
    n15 = riley_survival(15, 0.15 * max_r2, 0.057180, 5.0, 2.5159, 0.9, 0.05, 0.05,
                         event_fraction=427 / 2968)["n_required"]
    n13 = riley_survival(13, 0.15 * max_r2, 0.057180, 5.0, 2.5159, 0.9, 0.05, 0.05,
                         event_fraction=427 / 2968)["n_required"]
    assert n13 < n15
    assert n13 / n15 == pytest.approx(13 / 15, rel=0.01), "criterion 1 is linear in P"


# --------------------------------------------------------------------------- #
# SEALED SPLIT — the module states it never reads the test rows                #
# --------------------------------------------------------------------------- #
def test_the_sealed_split_loader_is_the_one_model_clinical_uses():
    """Both modules must go through the same predicate, or the claim is not enforced."""
    import src.model_clinical as mc
    import src.sample_size_riley as ssr

    assert ssr.load_development_frame is mc.load_development_frame
    assert ssr.clamp_horizon_days is mc.clamp_horizon_days


def test_development_loader_drops_test_rows(tmp_path):
    pd = pytest.importorskip("pandas")
    from src.sample_size_riley import load_development_frame

    p = tmp_path / "f.parquet"
    pd.DataFrame({"split": ["train"] * 3 + ["val"] * 2 + ["test"] * 4,
                  "time_from_landmark": np.arange(9.0) + 1,
                  "event_indicator": [1, 0, 1, 0, 1, 1, 1, 1, 1]}).to_parquet(p, index=False)
    dev = load_development_frame(p, forbid_test=True)
    assert len(dev) == 5 and set(dev["split"]) == {"train", "val"}


def test_published_cohort_aggregates_match_the_locked_counts():
    """Full-cohort context comes from the Phase-1 aggregate files, not from sealed rows."""
    from src.config import DEFAULT_CONFIG, load_config

    cfg = load_config(DEFAULT_CONFIG)
    pub = published_cohort_aggregates(cfg)
    assert pub["n"] == 3709 and pub["n_events"] == 533
    assert pub["by_split"]["test"] == {"n": 741, "n_events": 106}
    assert pub["by_split"]["train"]["n"] + pub["by_split"]["val"]["n"] == 2968
    assert "split_summary.csv" in pub["source"] and "event_counts.csv" in pub["source"]


def test_candidate_parameter_count_requires_exactly_one_age_column(tmp_path):
    import json

    (tmp_path / "clinical_imputation_params.json").write_text(json.dumps(
        {"model_columns": ["age_at_index_imp", "age_at_index_sq", "sex_female_imp"]}))
    cfg = {"model_clinical": {"age_rcs_df": 3},
           "sample_size": {"extra_image_parameters": 0}}
    with pytest.raises(AssertionError):
        candidate_parameter_count(cfg, tmp_path)


# --------------------------------------------------------------------------- #
# One name per maturity statistic (B3). Three quantities have all been called    #
# "complete 5-year observation"; observed_inputs must return them separately.    #
# --------------------------------------------------------------------------- #
def _maturity_frame():
    """8 patients: 2 events, 3 event-free through the administrative horizon, 3 lost."""
    import pandas as pd

    return pd.DataFrame({
        "time_from_landmark": [300.0, 1000.0, 1826.0, 1826.0, 1826.0, 200.0, 900.0, 1700.0],
        "event_indicator":    [1,      1,       0,      0,      0,      0,     0,     0],
        # complete_5y is a RECORD-COVERAGE flag, deliberately not a function of the two above
        "complete_5y":        [False,  True,    True,   True,   True,   False, False, False],
    })


def test_observed_inputs_names_the_three_maturity_counts_apart():
    df = _maturity_frame()
    out = observed_inputs(df, 1825.0)
    assert out["n_status_determined_5y"] == 5, "2 events + 3 event-free at the horizon"
    assert out["n_admin_censored_at_horizon"] == 3
    assert out["n_full_5y_record_coverage"] == 4, "complete_5y counts RECORD coverage"
    assert out["n_followup_reaches_day_1825"] == 3
    assert out["n_followup_reaches_day_1825"] == out["n_reaching_timepoint"]
    # The three are genuinely different numbers, which is the whole point of naming them.
    assert len({out["n_status_determined_5y"], out["n_full_5y_record_coverage"],
                out["n_followup_reaches_day_1825"]}) == 3


def test_status_determined_decomposes_into_events_plus_administrative_censoring():
    out = observed_inputs(_maturity_frame(), 1825.0)
    assert (out["n_status_determined_5y"]
            == out["n_events"] + out["n_admin_censored_at_horizon"])


def test_the_administrative_horizon_is_a_day_later_than_the_evaluation_horizon():
    """1826 is where censoring lands; 1825 is where the estimators are evaluated."""
    assert ADMIN_HORIZON_DAYS == 1826.0
    assert ADMIN_HORIZON_DAYS == round(5 * 365.25)
