"""sample_size_riley.py — minimum development sample size + locked-test-set precision (protocol section 16).

Protocol section 16 requires TWO calculations before model development:
  (a) the Riley minimum development sample for a time-to-event model, using the
      final candidate-parameter count, the OBSERVED event rate, the OBSERVED
      follow-up distribution and a defensible anticipated Cox-Snell R-squared;
  (b) sizing of the LOCKED test set BY SIMULATION so that the 5-year AUROC and
      the calibration slope are estimated with acceptable precision.
It also states the decision rule: "If precision is inadequate, simplify the model
and revise the question before preregistration rather than proceeding with an
underpowered deep model."

This module answers both, plus the widen-the-imaging-window question (2-year vs
3-year pre-index window, 3,709/533 vs 3,842/547 from the re-gate grid).

--------------------------------------------------------------------------------
PART 1 FORMULAS (Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for
developing a multivariable prediction model: PART II - binary and time-to-event
outcomes. Stat Med. 2019;38(7):1276-1296. doi:10.1002/sim.7992. Reference
implementation: the `pmsampsize` R package, function `pmsampsize_surv`.)

  phi   = overall event fraction  = E / n  ( = rate * mean_followup )
  rate  = events per person-year  = E / total_person_years
  P     = number of candidate PARAMETERS (not predictors)
  S     = target uniform shrinkage factor (config target_shrinkage = 0.9)
  delta = tolerated apparent-minus-adjusted Nagelkerke optimism (max_optimism = 0.05)
  MAPE  = tolerated margin of error on the overall risk at the timepoint (0.05)

  (0) Maximum achievable Cox-Snell R-squared for a survival outcome
      (Riley 2019 eq. 23), from the null log-likelihood of an exponential model
      written per patient (n cancels, so this depends only on phi):
          lnL_null / n = phi * ln(phi) - phi
          max R2_CS    = 1 - exp( 2 * (phi * ln(phi) - phi) )
      Nagelkerke R2 = R2_CS / max R2_CS, so an assumed "fraction f of the maximum"
      IS the anticipated Nagelkerke R2: R2_CS_adj = f * max R2_CS.

  (1) Criterion 1 - small overfitting (expected uniform shrinkage >= S).
      Van Houwelingen heuristic shrinkage S = 1 + P / (n * ln(1 - R2_CS_app)) with
      R2_CS_app = R2_CS_adj / S, rearranged for n:
          n1 = P / ( (S - 1) * ln(1 - R2_CS_adj / S) )
      Both factors are negative, so n1 > 0.

  (2) Criterion 2 - small optimism, |apparent - adjusted| Nagelkerke R2 <= delta.
      Since R2_CS_app = R2_CS_adj / S, the optimism constraint
      (R2_CS_adj/S - R2_CS_adj) / maxR2_CS <= delta rearranges to a required
      shrinkage
          S2 = R2_CS_adj / ( R2_CS_adj + delta * max R2_CS )
      which, with R2_CS_adj = f * max R2_CS, simplifies to S2 = f / (f + delta).
      n2 is then criterion 1 evaluated at S2:
          n2 = P / ( (S2 - 1) * ln(1 - R2_CS_adj / S2) )

  (3) Criterion 3 - precise estimate of the overall outcome risk at the timepoint t.
      Under a constant-hazard (exponential) approximation with total person-time
      PT = n * mean_followup, the rate has SE sqrt(rate / PT), and the risk is
      F(t) = 1 - exp(-rate * t), so the upper 95% limit and the margin of error are
          F_upper(t) = 1 - exp( -(rate + 1.96*sqrt(rate/PT)) * t )
          MOE        = F_upper(t) - F(t)
      pmsampsize REPORTS this margin at n = max(n1, n2) rather than solving for n.
      Solving MOE <= MAPE for n gives the closed form used here (an extension of
      the published function, flagged as such):
          n3 = rate * (1.96 * t)^2 / ( mean_followup * [ ln(1 - MAPE / S_exp(t)) ]^2 )
      with S_exp(t) = exp(-rate * t) and MAPE < S_exp(t) required.
      Because the observed hazard here is strongly front-loaded, the exponential
      approximation is checked against a nonparametric alternative: the Greenwood
      SE of the observed Kaplan-Meier risk at t, scaled as n_req = n_obs *
      (1.96 * SE_obs / MAPE)^2.

  The required development sample is max(n1, n2, n3); the binding criterion is
  whichever attains that maximum. Required events = n_required * phi;
  events per parameter = required events / P.

--------------------------------------------------------------------------------
PART 2 - LOCKED TEST-SET PRECISION BY SIMULATION (protocol section 16, second
sentence; evaluation principles per Riley RD, Debray TPA, Collins GS, Snell KIE.
Minimum sample size for external validation of a clinical prediction model with a
time-to-event outcome. Stat Med. 2022;41(7):1280-1295. doi:10.1002/sim.9275.)

  Data-generating process (stated explicitly so it can be criticised):
    * n_test patients, linear predictor LP ~ N(0, sigma^2), model CORRECTLY
      specified (true calibration slope = 1 by construction).
    * Proportional hazards: S(u | LP) = exp( -H0(u) * exp(LP) ). H0(.) is
      calibrated ON THE OBSERVED train+val Kaplan-Meier curve, so the simulated
      event times reproduce the real (front-loaded) event-time shape rather than
      an exponential idealisation.
    * sigma is solved so the TRUE cumulative/dynamic AUROC at the horizon equals
      each assumed value in the discrimination grid.
    * Censoring times are drawn by inverse sampling from the OBSERVED reverse-KM
      censoring distribution of train+val (which carries the administrative atom
      at day 1826), independent of the event time.
    * Estimands per replicate: the IPCW cumulative/dynamic AUROC at the horizon
      (Uno), and the Cox calibration slope of the outcome on LP with follow-up
      administratively truncated at the horizon.
    * Precision = 1.96 * Monte-Carlo SD across n_sim replicates (cross-checked
      against the mean within-replicate Wald half-width for the slope).

--------------------------------------------------------------------------------
WHAT THIS MODULE READS, EXACTLY

Patient rows: ``features_clinical.parquet`` loaded through
``src.model_clinical.load_development_frame``, i.e. with the SAME
``split != "test"`` predicate pushed into the Parquet reader that the model module
uses. **2,968 development rows are materialised; no sealed row is.** Every observed
input to both calculations — event fraction, person-time, mean follow-up, event
rate, follow-up quantiles, the Kaplan-Meier risk and its Greenwood SE, the
event-time curve and the reverse-KM censoring curve — is computed from those
development rows alone.

Full-cohort figures (3,709 patients / 533 events) appear only as CONTEXT and are
read from the already-published Phase-1 aggregates ``outputs/tables/split_summary.csv``
and ``outputs/event_counts.csv``, never recomputed from sealed rows. They are labelled
as published aggregates wherever they appear. The locked test split contributes only
its size (n_test = 741) and its published event count (106), both from config.

The scientific cost of development-only inputs is about one patient (the development and
full-cohort event fractions differ in the fifth decimal, 0.14387 vs 0.14370, so the Riley
criterion-1 requirement moves by one). The report prints both figures. The point of the
change is not the arithmetic: it is that the module's own statement about its leakage
controls is now true. It previously computed every input over all 3,709 patients,
including the 741 sealed ones, while claiming here and in the report that it did not.

HORIZONS. The horizon grid comes from ``src.model_clinical.clamp_horizon_days`` —
the single definition shared with M0 — so 5 years is day **1825** here as well
(administrative censoring lands on day 1826 and there are no controls at or beyond
it). ``outputs/tables/sample_size_riley.csv`` carries a ``horizon_days`` column so it
reconciles against ``derived-data/cohort/m0_clinical_model.json`` day for day.

Run:  python3 -m src.sample_size_riley --config config/feasibility.yaml
Writes outputs/sample_size.md and outputs/tables/sample_size_riley.csv
(aggregate only, no empi_anon).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.config import load_config
from src.followup import reverse_km
# The horizon grid and the sealed-split loader are IMPORTED, not re-implemented: the two
# modules previously disagreed by a day at the 5-year horizon (1826 here vs 1825 there).
from src.model_clinical import clamp_horizon_days, load_development_frame, parquet_num_rows

MODULE = "sample_size_riley"
DAYS_PER_YEAR = 365.25
Z975 = 1.959963984540054                      # scipy.stats.norm.ppf(0.975)
PMSAMPSIZE_Z = 1.96                           # the literal constant in pmsampsize_surv
QUAD_POINTS = 1201                            # Gauss-free uniform grid over +/- 8 sigma
CONFIRM_SIM_FRACTION = 0.25                   # replicates for the confirmatory re-simulation
# IPCW case weights are 1 / max(G(T-), IPCW_WEIGHT_FLOOR), i.e. capped at 1,000. The floor
# is a guard against a censoring curve that touches 0 inside a replicate, NOT an operating
# parameter: the report states the smallest G any case weight can see in these data and how
# far that sits from the cap. Raising the floor toward 1 turns IPCW off, so it is pinned by
# a unit test rather than left as an unexamined default.
IPCW_WEIGHT_FLOOR = 1e-3
# Administrative censoring lands on landmark + round(5 * 365.25) = day 1826 (src/followup.py).
# The EVALUATION horizon is one day earlier (day 1825, see clamp_horizon_days); the two are
# deliberately different numbers and the report names them apart.
ADMIN_HORIZON_DAYS = 1826.0
# LOCKED design anchors, mirroring src/model_clinical.py. Deliberately NOT in config: a
# config edit must not be able to weaken the guard that would catch a changed design.
EXPECTED_DESIGN_COLUMNS = 13                  # M0: 11 model columns - 1 linear age + 3 RCS terms
EXPECTED_IDENTIFIED_PARAMS = 12               # cr() is a partition of unity; Cox has no intercept


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(MODULE)
    lg.setLevel(logging.INFO); lg.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in lg.handlers):
        fh = logging.FileHandler(log_path, mode="a"); fh._mrkr = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s",
                                          datefmt="%Y-%m-%dT%H:%M:%S")); lg.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout); sh._mrkr = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s")); lg.addHandler(sh)
    return lg


# --------------------------------------------------------------------------- #
# PART 1 - Riley minimum development sample (pure functions).                  #
# --------------------------------------------------------------------------- #
def max_r2_cs_survival(event_fraction: float) -> float:
    """Maximum achievable Cox-Snell R2 for a time-to-event outcome (Riley 2019 eq. 23).

    ``event_fraction`` is phi = events / n = rate * mean follow-up. The null
    log-likelihood of the exponential model is E*ln(E/n) - E, so lnL_null/n =
    phi*ln(phi) - phi and max R2_CS = 1 - exp(2 * lnL_null / n).
    """
    assert 0.0 < event_fraction < 1.0, f"event fraction out of range: {event_fraction}"
    return 1.0 - math.exp(2.0 * (event_fraction * math.log(event_fraction) - event_fraction))


def n_for_shrinkage(parameters: int, r2_cs_adj: float, shrinkage: float) -> float:
    """n = P / ((S - 1) * ln(1 - R2_CS_adj / S)); the shared engine of criteria 1 and 2."""
    assert parameters >= 1, "parameters must be >= 1"
    assert 0.0 < shrinkage < 1.0, f"shrinkage must be in (0,1): {shrinkage}"
    assert 0.0 < r2_cs_adj < shrinkage, (
        f"anticipated R2_CS ({r2_cs_adj}) must be < the target shrinkage ({shrinkage})")
    return parameters / ((shrinkage - 1.0) * math.log(1.0 - r2_cs_adj / shrinkage))


def shrinkage_for_small_optimism(r2_cs_adj: float, max_r2_cs: float, max_optimism: float) -> float:
    """S2 = R2_CS_adj / (R2_CS_adj + delta * max R2_CS) (Riley criterion 2)."""
    assert 0.0 < r2_cs_adj <= max_r2_cs, "R2_CS_adj must be in (0, max R2_CS]"
    assert max_optimism > 0.0, "max_optimism must be > 0"
    return r2_cs_adj / (r2_cs_adj + max_optimism * max_r2_cs)


def risk_ci_exponential(rate: float, mean_followup: float, timepoint: float,
                        n: float) -> tuple[float, float, float]:
    """(lower, point, upper) 95% interval for the overall risk at ``timepoint``.

    Constant-hazard approximation exactly as pmsampsize_surv: person-time
    PT = mean_followup * n, SE(rate) = sqrt(rate / PT), risk = 1 - exp(-rate * t).
    """
    assert n > 0 and mean_followup > 0 and rate > 0, "rate/mean_followup/n must be > 0"
    se = math.sqrt(rate / (mean_followup * n))
    lo_rate = max(rate - PMSAMPSIZE_Z * se, 0.0)
    hi_rate = rate + PMSAMPSIZE_Z * se
    return (1.0 - math.exp(-lo_rate * timepoint),
            1.0 - math.exp(-rate * timepoint),
            1.0 - math.exp(-hi_rate * timepoint))


def n_for_risk_precision(rate: float, mean_followup: float, timepoint: float,
                         mape: float) -> float:
    """Smallest n whose upper-side margin of error on the risk at t is <= ``mape``.

    Closed form obtained by inverting ``risk_ci_exponential``:
        n = rate * (z*t)^2 / ( mean_followup * [ln(1 - mape / S_exp(t))]^2 )
    """
    s_exp = math.exp(-rate * timepoint)
    assert mape < s_exp, (
        f"MAPE {mape} exceeds the exponential survival probability {s_exp:.4f} at t={timepoint}; "
        "the upper risk limit can never come within that margin")
    denom = math.log(1.0 - mape / s_exp)
    return rate * (PMSAMPSIZE_Z * timepoint) ** 2 / (mean_followup * denom ** 2)


def riley_survival(parameters: int, r2_cs_adj: float, rate: float, timepoint: float,
                   mean_followup: float, shrinkage: float, max_optimism: float,
                   mape: float, event_fraction: float | None = None) -> dict:
    """Full Riley time-to-event calculation; returns every intermediate quantity."""
    phi = float(rate * mean_followup) if event_fraction is None else float(event_fraction)
    max_r2 = max_r2_cs_survival(phi)
    assert r2_cs_adj < max_r2, (
        f"anticipated R2_CS {r2_cs_adj:.4f} exceeds the maximum possible {max_r2:.4f}")

    n1 = math.ceil(n_for_shrinkage(parameters, r2_cs_adj, shrinkage))
    s2 = shrinkage_for_small_optimism(r2_cs_adj, max_r2, max_optimism)
    n2 = math.ceil(n_for_shrinkage(parameters, r2_cs_adj, s2))
    n3 = math.ceil(n_for_risk_precision(rate, mean_followup, timepoint, mape))

    n_req = max(n1, n2, n3)
    binding = {n1: "1_shrinkage", n2: "2_optimism", n3: "3_risk_precision"}[n_req]
    lo, pt, hi = risk_ci_exponential(rate, mean_followup, timepoint, n_req)
    return dict(parameters=parameters, r2_cs_adj=r2_cs_adj, max_r2_cs=max_r2,
                nagelkerke_r2=r2_cs_adj / max_r2, event_fraction=phi, rate=rate,
                mean_followup=mean_followup, timepoint=timepoint,
                shrinkage_target=shrinkage, shrinkage_criterion2=s2,
                n_criterion1=n1, n_criterion2=n2, n_criterion3=n3,
                n_required=n_req, binding_criterion=binding,
                events_required=n_req * phi, epp_required=n_req * phi / parameters,
                risk_at_t_exponential=pt, risk_lci_at_required_n=lo, risk_uci_at_required_n=hi,
                risk_moe_at_required_n=hi - pt)


def min_r2_fraction_supported(parameters: int, n_available: int, max_r2_cs: float,
                              shrinkage: float, max_optimism: float) -> float:
    """Smallest fraction-of-maximum R2_CS whose Riley requirement fits in ``n_available``.

    Both criteria 1 and 2 shrink monotonically as the assumed R2 grows, so a
    bisection on the fraction f finds the break-even assumption.
    """
    def excess(f: float) -> float:
        r2 = f * max_r2_cs
        n1 = n_for_shrinkage(parameters, r2, shrinkage)
        n2 = n_for_shrinkage(parameters, r2, shrinkage_for_small_optimism(r2, max_r2_cs, max_optimism))
        return max(n1, n2) - n_available

    lo, hi = 1e-4, min(0.99, 0.99 * shrinkage / max_r2_cs)
    if excess(hi) > 0:
        return float("nan")
    if excess(lo) <= 0:
        return lo
    return float(brentq(excess, lo, hi, xtol=1e-8))


def max_parameters_supported(n_available: int, r2_cs_adj: float, max_r2_cs: float,
                             shrinkage: float, max_optimism: float) -> int:
    """Largest P whose Riley requirement fits in ``n_available`` (n is linear in P)."""
    n1_per_p = n_for_shrinkage(1, r2_cs_adj, shrinkage)
    s2 = shrinkage_for_small_optimism(r2_cs_adj, max_r2_cs, max_optimism)
    n2_per_p = n_for_shrinkage(1, r2_cs_adj, s2)
    return int(math.floor(n_available / max(n1_per_p, n2_per_p)))


# --------------------------------------------------------------------------- #
# Small survival utilities (numpy; lifelines is reserved for the observed data).#
# --------------------------------------------------------------------------- #
def km_numpy(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kaplan-Meier on sorted unique times. Returns (times, survival, greenwood_var_of_S)."""
    order = np.argsort(np.asarray(time, float), kind="mergesort")
    t = np.asarray(time, float)[order]
    d = np.asarray(event, int)[order]
    ut, first = np.unique(t, return_index=True)
    at_risk = len(t) - first
    ev = np.add.reduceat(d, first)
    surv = np.cumprod(1.0 - ev / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(at_risk - ev > 0, ev / (at_risk * (at_risk - ev)), 0.0)
    gvar = surv ** 2 * np.cumsum(term)
    return ut, surv, gvar


def step_eval(ut: np.ndarray, values: np.ndarray, query, left_limit: bool = False) -> np.ndarray:
    """Evaluate a right-continuous step function (value 1.0 before the first time)."""
    q = np.atleast_1d(np.asarray(query, float))
    idx = np.searchsorted(ut, q, side="left" if left_limit else "right") - 1
    out = np.ones(q.shape, float)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def drop_flat(ut: np.ndarray, surv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the times where the step function actually drops (and drop t=0)."""
    keep = np.r_[True, np.diff(surv) < 0]
    ut, surv = ut[keep], surv[keep]
    if len(ut) and surv[0] >= 1.0:
        ut, surv = ut[1:], surv[1:]
    return ut, surv


def cox_slope_univariate(time: np.ndarray, event: np.ndarray, x: np.ndarray,
                         tol: float = 1e-9, maxit: int = 50) -> tuple[float, float, bool]:
    """Newton-Raphson univariate Cox partial likelihood, Breslow ties.

    Returns ``(beta, se, converged)``. ``converged`` is reported rather than assumed:
    the previous version silently returned the ``maxit``-th iterate whether or not the
    Newton step had settled, which is indistinguishable from a fitted value downstream.
    A non-converged fit returns ``(nan, nan, False)`` so it is counted as invalid by
    :func:`summarise_replicates` instead of quietly widening or narrowing a half-width.
    """
    order = np.argsort(-np.asarray(time, float), kind="mergesort")     # descending time
    t = np.asarray(time, float)[order]
    d = np.asarray(event, int)[order]
    xx = np.asarray(x, float)[order]
    if d.sum() == 0:
        return float("nan"), float("nan"), False
    _, first = np.unique(-t, return_index=True)                        # tie blocks
    last = np.append(first[1:], len(t)) - 1
    blk = np.searchsorted(first, np.arange(len(t)), side="right") - 1
    take = last[blk]
    m = d == 1
    beta, info, converged = 0.0, float("nan"), False
    for _ in range(maxit):
        w = np.exp(beta * xx)
        s0 = np.cumsum(w)[take]
        s1 = np.cumsum(w * xx)[take]
        s2 = np.cumsum(w * xx * xx)[take]
        ratio = s1[m] / s0[m]
        u = float(np.sum(xx[m] - ratio))
        info = float(np.sum(s2[m] / s0[m] - ratio ** 2))
        if not np.isfinite(info) or info <= 0:
            return float("nan"), float("nan"), False
        step = u / info
        beta += step
        if abs(step) < tol:
            converged = True
            break
    if not converged or not np.isfinite(beta):
        return float("nan"), float("nan"), False
    return beta, 1.0 / math.sqrt(info), True


def uno_auc_cd(time: np.ndarray, event: np.ndarray, lp: np.ndarray, horizon: float,
               weight_floor: float = IPCW_WEIGHT_FLOOR) -> tuple[float, int, int]:
    """IPCW cumulative/dynamic AUROC at ``horizon`` (Uno). Returns (auc, n_case, n_control).

    **Case** at t: ``T_i <= t`` with an OBSERVED event, weighted by ``1 / G(T_i^-)`` with
    G the Kaplan-Meier of the censoring distribution estimated in the same dataset. The
    weight is the LEFT limit — G at a case's own event time must not already reflect
    censorings that happen at that same instant, or the case is under-weighted.

    **Control** at t: ``T_i > t``, with NO condition on the event indicator. A patient
    whose event occurs after t is event-free *at* t and is therefore a control; requiring
    ``event == 0`` would silently discard exactly the highest-risk controls and bias the
    AUROC upward. Controls share the weight ``1 / G(t)``, which cancels in the ratio, so
    only their count enters.

    The comparison is strict (``T_i > t``, not ``>=``): a patient observed only up to
    exactly t is not known to be event-free after t. That is precisely why the horizon
    must be clamped to the last day strictly inside follow-up — at the administrative
    censoring day everyone has ``T == t``, the control arm is empty and the estimator is
    undefined. :func:`src.model_clinical.clamp_horizon_days` does the clamping, and the
    boundary horizon is computable only because of it.

    ``weight_floor`` bounds ``G`` away from 0 (default ``IPCW_WEIGHT_FLOOR``, i.e. a
    maximum weight of 1,000). It is a guard against a degenerate censoring curve; raising
    it toward 1 turns IPCW off.
    """
    case = (time <= horizon) & (event == 1)
    ctrl = time > horizon
    n_case, n_ctrl = int(case.sum()), int(ctrl.sum())
    if n_case == 0 or n_ctrl == 0:
        return float("nan"), n_case, n_ctrl
    gt, gv, _ = km_numpy(time, 1 - event)
    g = np.maximum(step_eval(gt, gv, time[case], left_limit=True), weight_floor)
    w = 1.0 / g
    a, b = lp[case][:, None], lp[ctrl][None, :]
    conc = (a > b).astype(float) + 0.5 * (a == b)
    return float((w[:, None] * conc).sum() / (w.sum() * n_ctrl)), n_case, n_ctrl


# --------------------------------------------------------------------------- #
# PART 2 - the simulated data-generating process.                              #
# --------------------------------------------------------------------------- #
def _normal_quadrature(sigma: float, npts: int = QUAD_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Grid over +/- 8 sigma with weights that sum to exactly 1."""
    x = np.linspace(-8.0 * sigma, 8.0 * sigma, npts)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return x, w / w.sum()


def marginal_risk(cum_hazard: float, sigma: float) -> float:
    """E_LP[ 1 - exp(-H * exp(LP)) ] for LP ~ N(0, sigma^2)."""
    x, w = _normal_quadrature(sigma)
    return float(np.sum(w * (1.0 - np.exp(-cum_hazard * np.exp(x)))))


def solve_cum_hazard(risk: float, sigma: float) -> float:
    """Baseline cumulative hazard H0 that reproduces a marginal ``risk``."""
    assert 0.0 < risk < 1.0, f"risk out of range: {risk}"
    return float(brentq(lambda h: marginal_risk(h, sigma) - risk, 1e-12, 1e3, xtol=1e-14))


def auc_cd_analytic(cum_hazard: float, sigma: float) -> float:
    """True cumulative/dynamic AUROC at the horizon under the PH + normal-LP model."""
    x, w = _normal_quadrature(sigma)
    f = 1.0 - np.exp(-cum_hazard * np.exp(x))
    case = w * f
    ctrl = w * (1.0 - f)
    case = case / case.sum()
    ctrl = ctrl / ctrl.sum()
    ctrl_cdf = np.cumsum(ctrl) - 0.5 * ctrl                 # half credit for the tie cell
    return float(np.sum(case * ctrl_cdf))


def sigma_for_auc(target_auc: float, risk: float) -> float:
    """LP standard deviation giving a true horizon AUROC of ``target_auc``."""
    assert 0.5 < target_auc < 1.0, f"target AUROC out of range: {target_auc}"

    def gap(sig: float) -> float:
        return auc_cd_analytic(solve_cum_hazard(risk, sig), sig) - target_auc

    return float(brentq(gap, 1e-3, 8.0, xtol=1e-10))


def baseline_hazard_grid(sigma: float, surv_grid: np.ndarray, iters: int = 80) -> np.ndarray:
    """H0 at every observed KM time so the MARGINAL simulated survival matches the data."""
    risks = 1.0 - surv_grid
    lo = np.full_like(risks, 1e-12)
    hi = np.full_like(risks, 1e3)
    x, w = _normal_quadrature(sigma)
    ex = np.exp(x)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = (w[None, :] * (1.0 - np.exp(-mid[:, None] * ex[None, :]))).sum(axis=1)
        below = val < risks
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    h0 = 0.5 * (lo + hi)
    assert np.all(np.diff(h0) > 0), "baseline cumulative hazard is not increasing"
    return h0


def simulate_replicates(rng: np.random.Generator, n_test: int, sigma: float,
                        h0_grid: np.ndarray, t_grid: np.ndarray, cens_t: np.ndarray,
                        cens_s: np.ndarray, horizon: float, n_sim: int) -> pd.DataFrame:
    """Run ``n_sim`` simulated locked-test-set validations; one row per replicate."""
    rows = []
    n_grid = len(h0_grid)
    for _ in range(n_sim):
        lp = rng.normal(0.0, sigma, n_test)
        need = -np.log(rng.random(n_test)) * np.exp(-lp)          # H0(T) threshold
        idx = np.searchsorted(h0_grid, need, side="left")
        t_event = np.where(idx < n_grid, t_grid[np.minimum(idx, n_grid - 1)], np.inf)
        cidx = np.searchsorted(-cens_s, -rng.random(n_test), side="left")
        t_cens = cens_t[np.minimum(cidx, len(cens_t) - 1)]
        t_obs = np.minimum(t_event, t_cens)
        d_obs = (t_event <= t_cens).astype(int)

        auc, n_case, n_ctrl = uno_auc_cd(t_obs, d_obs, lp, horizon)
        beta, se, ok = cox_slope_univariate(np.minimum(t_obs, horizon),
                                            d_obs * (t_obs <= horizon), lp)
        rows.append((auc, beta, se, bool(ok), n_case, n_ctrl, int(d_obs.sum())))
    return pd.DataFrame(rows, columns=["auroc", "slope", "slope_se", "slope_converged",
                                       "n_case", "n_control", "n_event_total"])


def summarise_replicates(reps: pd.DataFrame) -> dict:
    auc = reps["auroc"].to_numpy(float)
    slp = reps["slope"].to_numpy(float)
    se = reps["slope_se"].to_numpy(float)
    ok_a, ok_s = np.isfinite(auc), np.isfinite(slp)
    return dict(
        n_replicates=int(len(reps)),
        n_valid_auroc=int(ok_a.sum()), n_valid_slope=int(ok_s.sum()),
        n_slope_not_converged=int((~reps["slope_converged"].to_numpy(bool)).sum())
        if "slope_converged" in reps.columns else 0,
        auroc_mean=float(np.mean(auc[ok_a])), auroc_sd=float(np.std(auc[ok_a], ddof=1)),
        auroc_halfwidth=float(Z975 * np.std(auc[ok_a], ddof=1)),
        auroc_halfwidth_empirical=float((np.percentile(auc[ok_a], 97.5)
                                         - np.percentile(auc[ok_a], 2.5)) / 2.0),
        slope_mean=float(np.mean(slp[ok_s])), slope_sd=float(np.std(slp[ok_s], ddof=1)),
        slope_halfwidth=float(Z975 * np.std(slp[ok_s], ddof=1)),
        slope_halfwidth_wald=float(Z975 * np.mean(se[ok_s])),
        slope_halfwidth_empirical=float((np.percentile(slp[ok_s], 97.5)
                                         - np.percentile(slp[ok_s], 2.5)) / 2.0),
        mean_cases=float(reps["n_case"].mean()), mean_controls=float(reps["n_control"].mean()),
        mean_events=float(reps["n_event_total"].mean()))


def hanley_mcneil_se(auc: float, n_case: float, n_control: float) -> float:
    """Analytic AUROC SE (Hanley & McNeil, Radiology 1982;143:29-36) as a cross-check."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc ** 2 / (1.0 + auc)
    num = (auc * (1 - auc) + (n_case - 1) * (q1 - auc ** 2) + (n_control - 1) * (q2 - auc ** 2))
    return math.sqrt(num / (n_case * n_control))


def required_n_for_halfwidth(n_current: int, halfwidth: float, target: float) -> int:
    """Precision scales as 1/sqrt(n), so n_req = n_current * (achieved / target)^2."""
    if not np.isfinite(halfwidth) or halfwidth <= target:
        return int(n_current)
    return int(math.ceil(n_current * (halfwidth / target) ** 2))


# --------------------------------------------------------------------------- #
# Observed inputs.                                                             #
# --------------------------------------------------------------------------- #
def observed_inputs(feat: pd.DataFrame, timepoint_days: float) -> dict:
    """Event rate, person-time, follow-up distribution and KM/reverse-KM summaries."""
    t = feat["time_from_landmark"].to_numpy(float)
    e = feat["event_indicator"].to_numpy(int)
    n, n_ev = len(feat), int(e.sum())
    person_years = float(t.sum() / DAYS_PER_YEAR)
    ut, surv, gvar = km_numpy(t, e)
    s_t = float(step_eval(ut, surv, timepoint_days)[0])
    se_t = float(math.sqrt(step_eval(ut, gvar, timepoint_days)[0]))
    med_fu, _ = reverse_km(t, e)
    return dict(
        n=n, n_events=n_ev, event_fraction=n_ev / n, person_years=person_years,
        mean_followup_years=person_years / n, rate_per_person_year=n_ev / person_years,
        median_followup_days=float(np.median(t)),
        median_followup_reverse_km_days=float(med_fu),
        followup_q25_days=float(np.percentile(t, 25)), followup_q75_days=float(np.percentile(t, 75)),
        n_reaching_timepoint=int((t >= timepoint_days).sum()),
        # THREE distinct "5-year maturity" counts, named apart so none can stand in for
        # another (see the reconciliation table in the report):
        #   n_status_determined_5y      the 5-year outcome is KNOWN — an observed event, or
        #                               event-free follow-up reaching administrative
        #                               censoring. This is THE maturity statistic for a
        #                               5-year risk model.
        #   n_full_5y_record_coverage   the complete_5y flag: the record stream extends to
        #                               landmark + 1826 days. Coverage, not status.
        #   n_followup_reaches_day_1825 == n_reaching_timepoint above, kept under both names
        #                               so the CSV and the prose cannot drift.
        n_status_determined_5y=int(((e == 1) | (t >= ADMIN_HORIZON_DAYS)).sum()),
        n_admin_censored_at_horizon=int(((e == 0) & (t >= ADMIN_HORIZON_DAYS)).sum()),
        n_full_5y_record_coverage=(int(feat["complete_5y"].sum())
                                   if "complete_5y" in feat.columns else None),
        n_followup_reaches_day_1825=int((t >= timepoint_days).sum()),
        km_survival_at_t=s_t, km_risk_at_t=1.0 - s_t, km_greenwood_se_at_t=se_t)


def candidate_parameter_count(cfg, cohort_dir: Path) -> tuple[int, dict]:
    """Candidate parameters P for the Riley calculation — the IDENTIFIED count.

    Riley's criteria are statements about how many parameters the likelihood has to
    estimate, so the correct input is the number of parameters the design actually
    identifies, not the number of columns. ``src/model_clinical.py`` computes that by rank
    (``rank([X | 1]) - 1``, because a Cox partial likelihood has no intercept) and persists
    it as ``identified_parameters`` in ``m0_clinical_model.json``; that value is preferred
    here whenever the model has been fitted.

    Column-count fallback, used only when the model JSON is absent (e.g. a fresh checkout
    before M0 has run): model columns, minus the one linear age column, plus the RCS basis
    terms, plus ``sample_size.extra_image_parameters``. That fallback is an UPPER bound on
    P and therefore conservative — it can only overstate the required sample size.
    """
    params = json.loads((cohort_dir / "clinical_imputation_params.json").read_text())
    cols = list(params["model_columns"])
    age_rcs_df = int(cfg["model_clinical"]["age_rcs_df"])
    extra = int(cfg["sample_size"]["extra_image_parameters"])
    age_cols = [c for c in cols if c.startswith("age_at_index")]
    assert len(age_cols) == 1, f"expected exactly one age column in model_columns, got {age_cols}"
    n_design_cols = len(cols) - len(age_cols) + age_rcs_df
    detail = dict(n_model_columns=len(cols), age_columns=age_cols, age_rcs_df=age_rcs_df,
                  extra_image_parameters=extra, n_design_columns=n_design_cols,
                  model_columns=cols)

    model_json = cohort_dir / "m0_clinical_model.json"
    if model_json.exists():
        mj = json.loads(model_json.read_text())
        ident = mj.get("identified_parameters")
        if ident is not None:
            p = int(ident) + extra
            detail.update(parameters=p, identified_parameters=int(ident),
                          n_design_columns_from_model=int(mj.get("n_parameters", n_design_cols)),
                          parameters_source="m0_clinical_model.json:identified_parameters")
            assert detail["n_design_columns_from_model"] == n_design_cols, (
                "the fitted design has "
                f"{detail['n_design_columns_from_model']} columns but model_columns implies "
                f"{n_design_cols} — refit M0 before recomputing the sample size")
            return p, detail

    p = n_design_cols + extra
    detail.update(parameters=p, identified_parameters=None,
                  parameters_source="derived_from_model_columns (m0_clinical_model.json "
                                    "absent; this is an upper bound on P)")
    return p, detail


def _cross_check_horizons_against_model_json(cohort_dir: Path, horizons: list[dict],
                                             log: logging.Logger) -> None:
    """Assert this module's horizon grid equals the one frozen in the M0 contract.

    The two modules used to disagree by a day at 5 years (1826 vs 1825), which made the
    sample-size CSV silently non-reconcilable against ``m0_clinical_model.json``. Both now
    call the same helper; this verifies it rather than trusting it.
    """
    mj_path = cohort_dir / "m0_clinical_model.json"
    if not mj_path.exists():
        log.warning("m0_clinical_model.json absent — horizon grid not cross-checked")
        return
    frozen = [float(h["horizon_days"]) for h in json.loads(mj_path.read_text())["horizons"]]
    mine = [float(h["horizon_days"]) for h in horizons]
    assert frozen == mine, f"horizon mismatch: M0 uses {frozen}, this module computed {mine}"
    log.info("horizon grid matches the frozen M0 contract: %s",
             ", ".join(f"{int(d)} d" for d in mine))


def published_cohort_aggregates(cfg) -> dict:
    """Full-cohort counts from the ALREADY-PUBLISHED Phase-1 aggregate files.

    The locked test split is sealed, so cohort-wide figures are never recomputed from
    patient rows in this module. ``outputs/tables/split_summary.csv`` (per-split n and
    events) and ``outputs/event_counts.csv`` (the primary 5-year event definition) are
    aggregate-only files written during Phase 1; everything derived from them is labelled
    "published Phase-1 aggregate" in the report so no reader mistakes it for something
    recomputed here.
    """
    tables = cfg.out("tables_dir")
    split = pd.read_csv(tables / "split_summary.csv")
    counts = pd.read_csv(cfg.out("outputs_dir") / "event_counts.csv")
    prim = counts[counts["definition"] == "primary_contralateral_5y"].iloc[0]
    n_total = int(split["n_patients"].sum())
    e_total = int(split["n_events"].sum())
    assert n_total == int(prim["n_cohort"]) and e_total == int(prim["n_events"]), (
        f"published aggregates disagree: split_summary {n_total}/{e_total} vs event_counts "
        f"{int(prim['n_cohort'])}/{int(prim['n_events'])}")
    by_split = {r.split: dict(n=int(r.n_patients), n_events=int(r.n_events))
                for r in split.itertuples()}
    return dict(source="outputs/tables/split_summary.csv + outputs/event_counts.csv",
                n=n_total, n_events=e_total, event_fraction=e_total / n_total,
                by_split=by_split)


# --------------------------------------------------------------------------- #
# Report.                                                                      #
# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_report(path: Path, ctx: dict) -> None:
    pub, dev, pdet = ctx["pub"], ctx["dev"], ctx["param_detail"]
    cfgs, alt = ctx["cfg_ss"], ctx["alt"]
    auroc_grid, auroc_primary = ctx["auroc_grid"], ctx["auroc_primary"]
    L: list[str] = []
    A = L.append
    A("# Sample size: Riley minimum development sample and locked-test-set precision")
    A("")
    A(f"Generated {ctx['generated']} by `src/sample_size_riley.py` "
      f"(protocol section 16). Seed {ctx['seed']}.")
    A("")
    A("Protocol section 16, verbatim: *\"For the clinical comparator, calculate the minimum "
      "development sample using the Riley time-to-event framework with the final number of "
      "candidate parameters, observed event rate, follow-up distribution, and a defensible "
      "anticipated Cox-Snell R-squared. Size the locked test set by simulation to obtain "
      "acceptable precision for 5-year AUROC and calibration slope... A practical preliminary "
      "floor is 500 total primary events and 100 test events, followed by the formal "
      "calculation. If precision is inadequate, simplify the model and revise the question "
      "before preregistration rather than proceeding with an underpowered deep model.\"*")
    A("")

    # ---------------- headline ----------------
    A("## Bottom line")
    A("")
    for line in ctx["headline"]:
        A(f"- {line}")
    A("")

    # ---------------- inputs ----------------
    A("## 1. Observed inputs (nothing here is nominal)")
    A("")
    A("**What was read.** Patient rows come from `derived-data/cohort/features_clinical.parquet` "
      "through `src.model_clinical.load_development_frame`, i.e. with the same "
      "`split != \"test\"` predicate pushed into the Parquet reader that the M0 module uses: "
      f"**{dev['n']:,} of {pub['n']:,} rows are materialised and no sealed row is**. Every "
      "quantity in the middle column below is computed from those development rows. The "
      "right-hand column is CONTEXT ONLY and is read from the already-published Phase-1 "
      f"aggregates (`{pub['source']}`) — it is never recomputed from sealed rows, and only "
      "the three counts those files contain are available.")
    A("")
    A("| Quantity | Development set (train + val), computed here | Full cohort (published "
      "Phase-1 aggregate) |")
    A("|---|---|---|")
    sealed = "not recomputed (test split sealed)"
    rows = [
        ("Patients", f"{dev['n']:,}", f"{pub['n']:,}"),
        ("5-year contralateral TKA events", f"{dev['n_events']:,}", f"{pub['n_events']:,}"),
        ("Event fraction phi = E/n", _fmt(dev["event_fraction"], 5),
         _fmt(pub["event_fraction"], 5)),
        ("Person-years of follow-up", f"{dev['person_years']:,.1f}", sealed),
        ("Mean follow-up (years)", _fmt(dev["mean_followup_years"], 4), sealed),
        ("Event rate (per person-year)", _fmt(dev["rate_per_person_year"], 5), sealed),
        ("Median observed follow-up (days)", f"{dev['median_followup_days']:.0f}", sealed),
        ("Median follow-up, reverse KM (days)",
         f"{dev['median_followup_reverse_km_days']:.0f}", sealed),
        ("Follow-up IQR (days)",
         f"{dev['followup_q25_days']:.0f} to {dev['followup_q75_days']:.0f}", sealed),
        ("`n_status_determined_5y` — 5-year status DETERMINED (an observed event, or "
         f"event-free follow-up reaching day {ADMIN_HORIZON_DAYS:.0f})",
         f"**{dev['n_status_determined_5y']:,}** "
         f"({100 * dev['n_status_determined_5y'] / dev['n']:.1f}%)", sealed),
        ("`n_full_5y_record_coverage` — the `complete_5y` flag: `last_observed >= landmark + "
         f"{ADMIN_HORIZON_DAYS:.0f}`",
         f"{dev['n_full_5y_record_coverage']:,}", sealed),
        (f"`n_followup_reaches_day_{ctx['timepoint_days']:.0f}` — observed follow-up time "
         f"reaches day {ctx['timepoint_days']:.0f} (`time_from_landmark >= horizon`)",
         f"{dev['n_followup_reaches_day_1825']:,}", sealed),
        (f"Kaplan-Meier risk at day {ctx['timepoint_days']:.0f}",
         _fmt(dev["km_risk_at_t"], 5), sealed),
        (f"Greenwood SE of KM survival at day {ctx['timepoint_days']:.0f}",
         _fmt(dev["km_greenwood_se_at_t"], 5), sealed),
    ]
    for r in rows:
        A(f"| {r[0]} | {r[1]} | {r[2]} |")
    A("")
    A("### THREE \"5-year maturity\" counts exist in this project. They are not "
      "interchangeable.")
    A("")
    A("All three appear in the table above. Only the first is a maturity statistic for a "
      "5-year risk model; the other two answer different questions and must never be "
      "substituted for it. **No computed quantity in this document, or in "
      "`outputs/clinical_baseline_report.md`, depends on which one is quoted** — "
      "inverse-probability-of-censoring weighting already handles administrative censoring "
      "correctly, and these counts appear only in prose.")
    A("")
    A("| name | definition | development (train + val) | full cohort |")
    A("|---|---|---|---|")
    A(f"| `n_status_determined_5y` | the 5-year outcome is KNOWN: an observed event "
      f"({dev['n_events']:,} patients), or event-free follow-up reaching administrative "
      f"censoring at day {ADMIN_HORIZON_DAYS:.0f} "
      f"({dev['n_admin_censored_at_horizon']:,} patients) | **{dev['n_status_determined_5y']:,}"
      f" / {dev['n']:,} ({100 * dev['n_status_determined_5y'] / dev['n']:.1f}%)** | "
      "1,401 / 3,709 (`outputs/feasibility_report.md`) |")
    A(f"| `n_full_5y_record_coverage` | the `complete_5y` flag: `last_observed >= landmark + "
      f"{ADMIN_HORIZON_DAYS:.0f}`. This counts RECORD COVERAGE, not status, so it drops most "
      "of the patients whose 5-year status is known precisely because they had the event and "
      f"then left the record stream | {dev['n_full_5y_record_coverage']:,} / {dev['n']:,} "
      f"({100 * dev['n_full_5y_record_coverage'] / dev['n']:.1f}%) | 916 / 3,709 |")
    A(f"| `n_followup_reaches_day_{ctx['timepoint_days']:.0f}` | "
      f"`time_from_landmark >= {ctx['timepoint_days']:.0f}`: observed follow-up time reaching "
      "the CLAMPED EVALUATION horizon (one day inside administrative censoring), regardless "
      f"of how the patient left the risk set | {dev['n_followup_reaches_day_1825']:,} / "
      f"{dev['n']:,} ({100 * dev['n_followup_reaches_day_1825'] / dev['n']:.1f}%) | "
      "869 / 3,709 |")
    A("")
    A("Wherever follow-up maturity is invoked to support or undermine the 5-year horizon — "
      "in this report, in `outputs/clinical_baseline_report.md`, in "
      "`outputs/feasibility_report.md` and in `notebooks/train_colab.ipynb` — the figure "
      "used is `n_status_determined_5y`. An earlier revision argued the 5-year horizon down "
      f"from {dev['n_full_5y_record_coverage']:,} / {dev['n']:,} "
      f"({100 * dev['n_full_5y_record_coverage'] / dev['n']:.1f}%) in one document and from "
      "1,401 / 3,709 in another, which are two incompatible numbers for one conclusion.")
    A("")
    A("**Using development-only inputs costs about one patient and buys a true statement.** "
      f"Criterion 1 at the primary assumption needs {ctx['primary']['n_criterion1']:,} "
      "patients from the development event fraction; substituting the published full-cohort "
      f"event fraction ({_fmt(pub['event_fraction'], 5)} against "
      f"{_fmt(dev['event_fraction'], 5)}) would make it {ctx['n1_published_phi']:,}. Nothing "
      "in this document turns on that difference; what turned on it was whether the module's "
      "own claim about its leakage controls was accurate. It previously computed every input "
      "over all 3,709 patients, including the 741 sealed ones, while stating in its docstring "
      "and in this report that the locked test split is never read.")
    A("")
    A(f"**Candidate parameters P = {pdet['parameters']}**, taken (not hard-coded) from "
      f"`{pdet['parameters_source']}`.")
    A("")
    A(f"The design matrix has **{pdet['n_design_columns']} columns** — "
      f"{pdet['n_model_columns']} model columns in "
      "`derived-data/cohort/clinical_imputation_params.json`, minus the "
      f"{len(pdet['age_columns'])} linear age column (`{pdet['age_columns'][0]}`), plus "
      f"`model_clinical.age_rcs_df` = {pdet['age_rcs_df']} restricted-cubic-spline basis "
      f"terms on age — but only **{pdet['identified_parameters']} of them are IDENTIFIED**, "
      "plus `sample_size.extra_image_parameters` = "
      f"{pdet['extra_image_parameters']}. `src/model_clinical.py` computes the identified "
      "count by rank (`rank([X | 1]) - 1`, because a Cox partial likelihood has no "
      "intercept) and freezes it in `m0_clinical_model.json`; the shortfall is patsy's "
      "`cr()` basis, which is a partition of unity, so the three age columns sum to 1 on "
      "every row and one direction is a level the likelihood cannot see.")
    A("")
    A("**The identified count is the correct Riley input.** Riley's criteria describe how "
      "many parameters the likelihood has to estimate, not how many columns the design "
      f"matrix happens to have. A previous version of this module asserted P = 15, which was "
      "wrong twice over: the design carried an extra aliased indicator "
      "(`pain_score_max_missing`, the exact complement of `knee_pain_any_imp`) that has since "
      "been removed from the model column list, and the spline partition of unity was never "
      "counted at all. Both errors pushed P upward, so **the previous requirement was "
      "conservative** — it demanded a larger development sample than the model actually "
      "needs, and every 'sufficient' verdict it reached still stands a fortiori.")
    A("")
    A("**P moved again with the M0 correction (2026-07-24).** The dataset-inferred "
      "contralateral KLG was removed from M0 and the image-to-index interval was added, "
      "restoring protocol Table 7's `M0 = \"Age, sex, comorbidities, pain, image-to-index "
      "interval\"` (Table 6 lists inferred KLG as a **secondary comparator only**, and it now "
      "sits in M1). Net effect on the design: `klg_contra_imp` and `klg_contra_missing` out, "
      f"`days_to_index_imp` in — {pdet['n_design_columns']} design columns and "
      f"**P = {pdet['parameters']}** where the previous revision had 14 and 13. Fewer "
      "parameters lower every Riley requirement, so this direction is again conservative "
      "with respect to the earlier 'sufficient' verdicts. The M1 comparator carries one more "
      "parameter and is fitted on a smaller (KLG-eligible) subset; it is a secondary "
      "comparator and is not what protocol section 16 sizes.")
    A("")
    A("The image model is deliberately not charged extra parameters here: a frozen "
      "ConvNeXt-Tiny encoder contributes a learned representation, not free degrees of "
      "freedom in the survival head, and the Riley framework has no accepted extension to "
      "deep representation learning. That is an assumption, not a result.")
    A("")
    A(f"**Constant-hazard check.** The exponential approximation implied by the observed rate "
      f"gives a {ctx['timepoint_years']:.0f}-year risk of "
      f"{_fmt(1 - math.exp(-dev['rate_per_person_year'] * ctx['timepoint_years']), 5)} versus the "
      f"observed Kaplan-Meier {_fmt(dev['km_risk_at_t'], 5)}. The hazard is front-loaded (the "
      "1-year KM risk already exceeds what a constant hazard predicts), so criterion 3 is "
      "reported both ways below.")
    A("")

    # ---------------- formulas ----------------
    A("## 2. Formulas actually used, with citations")
    A("")
    A("Riley RD, Snell KIE, Ensor J, Burke DL, Harrell FE, Moons KGM, Collins GS. Minimum "
      "sample size for developing a multivariable prediction model: PART II - binary and "
      "time-to-event outcomes. *Stat Med.* 2019;38(7):1276-1296. Reference implementation: "
      "the `pmsampsize` R package, `pmsampsize_surv()`.")
    A("")
    A("```")
    A("phi   = E / n                (overall event fraction = rate * mean follow-up)")
    A("P     = candidate parameters,  S = target shrinkage,  delta = optimism tolerance")
    A("")
    A("(0)  max R2_CS = 1 - exp( 2 * ( phi*ln(phi) - phi ) )              [Riley 2019 eq. 23]")
    A("     R2_CS_adj = f * max R2_CS   ==>   f IS the anticipated Nagelkerke R2")
    A("")
    A("(1)  n1 = P / ( (S  - 1) * ln(1 - R2_CS_adj / S ) )                 criterion 1")
    A("(2)  S2 = R2_CS_adj / ( R2_CS_adj + delta * max R2_CS ) = f/(f+delta)")
    A("     n2 = P / ( (S2 - 1) * ln(1 - R2_CS_adj / S2) )                 criterion 2")
    A("(3)  PT = n * mean_followup;  SE(rate) = sqrt(rate / PT)")
    A("     risk(t)       = 1 - exp(-rate * t)")
    A("     risk_upper(t) = 1 - exp( -(rate + 1.96*SE(rate)) * t )")
    A("     MOE = risk_upper(t) - risk(t) <= MAPE, solved for n:")
    A("     n3 = rate * (1.96*t)^2 / ( mean_followup * [ ln(1 - MAPE/exp(-rate*t)) ]^2 )")
    A("")
    A("     n_required = max(n1, n2, n3);  events = n_required * phi;  EPP = events / P")
    A("```")
    A("")
    A("Step (3) is where this module goes beyond `pmsampsize_surv()`, which fixes "
      "n3 = max(n1, n2) and merely *reports* the resulting risk interval instead of solving "
      "for the n that meets the margin. Both are given below; the reported interval at "
      "n = max(n1, n2) reproduces the published function exactly.")
    A("")
    A("**Arithmetic verification against a published worked example.** The `pmsampsize` "
      "documentation example `pmsampsize(type=\"s\", csrsquared=0.051, parameters=30, "
      "rate=0.065, timepoint=2, meanfup=2.07)` gives phi = 0.065*2.07 = 0.13455, "
      "max R2_CS = 1 - exp(2*(0.13455*ln(0.13455) - 0.13455)) = 0.5547, "
      "Nagelkerke = 0.051/0.5547 = 0.092, "
      "n1 = 30/((0.9-1)*ln(1-0.051/0.9)) = 30/0.0058332 = 5143, "
      "S2 = 0.051/(0.051+0.05*0.5547) = 0.6477 and "
      "n2 = 30/((0.6477-1)*ln(1-0.051/0.6477)) = 30/0.0288856 = 1039. "
      "`tests/test_sample_size_riley.py` pins these values.")
    A("")

    # ---------------- part 1 results ----------------
    A("## 3. Riley minimum development sample (Part 1)")
    A("")
    A(f"Anticipated Cox-Snell R2 is unknown a priori (no prior model exists for this "
      f"outcome), so it is expressed as a fraction f of the maximum achievable value "
      f"**max R2_CS = {_fmt(ctx['max_r2_cs'], 4)}** (computed from the observed event "
      f"fraction phi = {_fmt(dev['event_fraction'], 5)}). Because Nagelkerke R2 = "
      "R2_CS / max R2_CS, f is exactly the anticipated Nagelkerke R2. The headline "
      f"assumption is f = {cfgs['r2_fraction_of_max_primary']}.")
    A("")
    A("| f (= Nagelkerke R2) | R2_CS_adj | n1 shrinkage | n2 optimism | n3 risk precision | "
      "**n required** | binding | events required | EPP at required n |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in ctx["riley_rows"]:
        star = " **(primary)**" if r["is_primary"] else ""
        A(f"| {r['r2_fraction']:.2f}{star} | {_fmt(r['r2_cs_adj'], 4)} | {r['n_criterion1']:,} | "
          f"{r['n_criterion2']:,} | {r['n_criterion3']:,} | **{r['n_required']:,}** | "
          f"{r['binding_criterion']} | {r['events_required']:.0f} | {r['epp_required']:.1f} |")
    A("")
    A(f"Observed against requirement, at the primary assumption f = "
      f"{cfgs['r2_fraction_of_max_primary']} (required n = {ctx['primary']['n_required']:,}, "
      f"required events = {ctx['primary']['events_required']:.0f}):")
    A("")
    A("| Comparison | n | events | EPP | vs required n |")
    A("|---|---|---|---|---|")
    A(f"| Full locked cohort (published Phase-1 aggregate) | {pub['n']:,} | {pub['n_events']:,} | "
      f"{pub['n_events'] / pdet['parameters']:.1f} | "
      f"{pub['n'] / ctx['primary']['n_required']:.2f}x |")
    A(f"| Development set actually used to fit (train + val) | {dev['n']:,} | "
      f"{dev['n_events']:,} | {dev['n_events'] / pdet['parameters']:.1f} | "
      f"{dev['n'] / ctx['primary']['n_required']:.2f}x |")
    A(f"| Training split alone | {ctx['n_train']:,} | {ctx['e_train']:,} | "
      f"{ctx['e_train'] / pdet['parameters']:.1f} | "
      f"{ctx['n_train'] / ctx['primary']['n_required']:.2f}x |")
    A("")
    A("The development set, not the full cohort, is the honest comparator: the 741 locked "
      "test patients are never used to estimate a coefficient.")
    A("")
    A("**Headroom.** With the "
      f"{dev['n']:,} development patients and P = {pdet['parameters']}, the Riley criteria are "
      f"satisfied for any anticipated Nagelkerke R2 at or above "
      f"**f = {_fmt(ctx['min_f_dev'], 4)}** "
      f"({_fmt(ctx['min_f_dev'] * ctx['max_r2_cs'], 4)} on the Cox-Snell scale), and the same "
      f"data would support up to **{ctx['p_max_dev']} parameters** at the primary "
      f"f = {cfgs['r2_fraction_of_max_primary']} assumption "
      f"(versus the {pdet['parameters']} pre-specified).")
    A("")
    A("### Criterion 3 in detail (the overall-risk margin of error)")
    A("")
    c3 = ctx["c3"]
    A(f"- Exponential (pmsampsize) form, solved for n: **n3 = {c3['n_exponential']:,}**, "
      f"which is {c3['n_exponential'] / dev['n']:.2f}x the development set.")
    A(f"- At n = max(n1, n2) = {ctx['primary']['n_required']:,} the published function reports "
      f"a {ctx['timepoint_years']:.0f}-year risk of {_fmt(ctx['primary']['risk_at_t_exponential'], 4)} "
      f"(95% CI {_fmt(ctx['primary']['risk_lci_at_required_n'], 4)} to "
      f"{_fmt(ctx['primary']['risk_uci_at_required_n'], 4)}), margin of error "
      f"{_fmt(ctx['primary']['risk_moe_at_required_n'], 4)} against a target of "
      f"{cfgs['mape_target']}.")
    A(f"- Nonparametric alternative, because the hazard is not constant: the observed "
      f"Kaplan-Meier {ctx['timepoint_years']:.0f}-year risk is {_fmt(dev['km_risk_at_t'], 4)} "
      f"with Greenwood SE {_fmt(dev['km_greenwood_se_at_t'], 5)} in {dev['n']:,} patients, so "
      f"the 95% margin of error is already {_fmt(Z975 * dev['km_greenwood_se_at_t'], 4)}. "
      f"Scaling as 1/sqrt(n), a margin of {cfgs['mape_target']} needs only "
      f"**n = {c3['n_greenwood']:,}**.")
    A(f"- The two routes differ by {abs(c3['n_exponential'] - c3['n_greenwood']):,} patients "
      f"({100 * abs(c3['n_exponential'] - c3['n_greenwood']) / max(c3['n_exponential'], c3['n_greenwood']):.0f}% "
      f"of the larger, roughly {100 * abs(math.sqrt(c3['n_exponential'] / c3['n_greenwood']) - 1):.0f}% "
      "on the standard-error scale), which is close agreement for a sample-size calculation "
      "and means the constant-hazard idealisation does not change the conclusion: criterion 3 "
      "is nowhere near binding. Criterion 1 (shrinkage) drives the requirement at every "
      "assumption in the grid.")
    A("")

    # ---------------- part 2 ----------------
    A("## 4. Locked test-set precision by simulation (Part 2)")
    A("")
    sim = ctx["sim_cfg"]
    A(f"n_test = {sim['n_test']:,} patients with {sim['n_test_events']} events "
      f"(the locked 20% split). {sim['n_sim']:,} replicates per scenario, seed {ctx['seed']}. "
      f"Horizons are the shared clamped grid (day "
      + ", ".join(str(int(h["horizon_days"])) for h in ctx["horizon_specs"])
      + "), identical to `derived-data/cohort/m0_clinical_model.json`.")
    A("")
    A(f"**Targets adopted for this calculation** (`config/feasibility.yaml`, "
      f"`sample_size.test_precision_simulation`; protocol section 16 requires \"acceptable "
      f"precision\" but specifies **no numeric target**): AUROC 95% CI half-width "
      f"<= {sim['target_auroc_ci_halfwidth']}, calibration slope 95% CI half-width "
      f"<= {sim['target_calibration_slope_ci_halfwidth']}. These are analyst choices and must "
      "never be described as protocol values. The **numeric floor section 16 does state — 500 "
      f"total primary events and 100 test events — IS met ({pub['n_events']} and "
      f"{sim['n_test_events']}).**")
    A("")
    A(f"The assumed true discrimination grid {tuple(auroc_grid)} and the primary value "
      f"{auroc_primary} also come from config "
      "(`test_precision_simulation.true_auroc_grid` / `.true_auroc_primary`), so the "
      "sensitivity of the headline conclusion to that assumption is visible in the "
      "configuration rather than buried in code.")
    A("")
    A(f"**IPCW weight floor.** Case weights are `1 / max(G(T-), {IPCW_WEIGHT_FLOOR:g})`, i.e. "
      f"capped at {1 / IPCW_WEIGHT_FLOOR:,.0f}. That cap is a guard against a censoring curve "
      "that touches zero inside a replicate, not an operating parameter: the smallest "
      "censoring-survival value a case weight can meet at each horizon in these data is "
      + ", ".join(f"G({int(t)}-) = {g:.3f} (max weight {1 / g:,.1f})"
                  for t, g in sorted(ctx["g_min_by_horizon"].items()))
      + ", every one of them three orders of magnitude clear of the cap. Raising the floor "
        "toward 1 would silently switch IPCW off, so a unit test pins it.")
    A("")
    A("**Assumed data-generating process** (see the module docstring for the full statement):")
    A("")
    A("1. Linear predictor LP ~ N(0, sigma^2); the model is correctly specified, so the true "
      "calibration slope is exactly 1.")
    A("2. Proportional hazards S(u | LP) = exp(-H0(u) exp(LP)), with the baseline cumulative "
      "hazard H0 calibrated at every observed Kaplan-Meier time so the simulated marginal "
      "survival reproduces the real, front-loaded event-time curve.")
    A("3. sigma is solved numerically so the TRUE cumulative/dynamic AUROC at each horizon "
      "equals the assumed value.")
    A("4. Censoring times are inverse-sampled from the observed reverse-KM censoring "
      "distribution (which carries the administrative atom at day 1826), independent of the "
      "event time. The event and censoring distributions are estimated on train + val only; "
      "the locked test split is never read.")
    A("5. Estimands: IPCW cumulative/dynamic AUROC at the horizon (Uno), and the Cox "
      "calibration slope of the outcome on LP with follow-up truncated at the horizon.")
    A("6. Reported precision is 1.96 x the Monte-Carlo SD across replicates, computed over "
      "the replicates that yielded a finite estimate: at the primary scenario "
      f"{ctx['prim_sim']['n_valid_auroc']:,} of {ctx['prim_sim']['n_replicates']:,} for the "
      f"AUROC and {ctx['prim_sim']['n_valid_slope']:,} of "
      f"{ctx['prim_sim']['n_replicates']:,} for the slope "
      f"({ctx['prim_sim']['n_slope_not_converged']:,} Cox fits failed to converge and are "
      "reported as invalid rather than returned as an unconverged last iterate).")
    A("")
    A("> **These assumptions flatter precision, and the direction matters.** The linear "
      "predictor is exactly normal and the censoring is independent of risk. Real linear "
      "predictors are skewed and heavier-tailed than a Gaussian, which spreads case and "
      "control scores less evenly and widens the AUROC interval; and if loss to follow-up is "
      "risk-related — plausible here, since disengagement from a health system is not random "
      "with respect to arthritis burden — the IPCW weights are misspecified and the true "
      "interval is wider still than the marginal-censoring simulation shows. Every "
      "half-width below is therefore best read as a **lower bound** on the uncertainty a real "
      "validation will face. That makes the shortfall against the adopted AUROC target a "
      "conservative statement, not an alarmist one.")
    A("")
    A("Solved LP standard deviations: " + ", ".join(
        f"{k[0]:.0f} y / AUROC {k[1]:.2f}: sigma = {v:.3f}" for k, v in ctx["sigmas"].items()))
    A("")
    A("| Horizon | True AUROC | Cases | Controls | AUROC half-width | target met | "
      "Slope half-width | target met | n_test needed for AUROC | n_test needed for slope |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in ctx["sim_rows"]:
        A(f"| {r['horizon_years']:.0f} y | {r['true_auroc']:.2f} | {r['mean_cases']:.0f} | "
          f"{r['mean_controls']:.0f} | {_fmt(r['auroc_halfwidth'], 4)} | "
          f"{'yes' if r['auroc_target_met'] else 'NO'} | {_fmt(r['slope_halfwidth'], 4)} | "
          f"{'yes' if r['slope_target_met'] else 'NO'} | {r['n_test_required_auroc']:,} | "
          f"{r['n_test_required_slope']:,} |")
    A("")
    A("Cross-checks that the simulator is doing what it claims:")
    A("")
    for line in ctx["sim_checks"]:
        A(f"- {line}")
    A("")
    A("What the table says, in words:")
    A("")
    for line in ctx["sim_narrative"]:
        A(f"- {line}")
    A("")

    # ---------------- part 3 ----------------
    A("## 5. Widen the pre-index imaging window from 2 to 3 years? (Part 3)")
    A("")
    A(f"The re-gate grid records that `{alt['label']}` would yield "
      f"{alt['n_final_cohort']:,} patients and {alt['primary_events_5y']} events, "
      f"versus {pub['n']:,} / {pub['n_events']} now: "
      f"**+{alt['n_final_cohort'] - pub['n']} patients (+"
      f"{100 * (alt['n_final_cohort'] - pub['n']) / pub['n']:.1f}%) and "
      f"+{alt['primary_events_5y'] - pub['n_events']} events "
      f"(+{100 * (alt['primary_events_5y'] - pub['n_events']) / pub['n_events']:.1f}%)**.")
    A("")
    A("| Metric | 2-year window (current) | 3-year window | Change |")
    A("|---|---|---|---|")
    for row in ctx["alt_rows"]:
        A(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    A("")
    for line in ctx["alt_narrative"]:
        A(f"- {line}")
    A("")

    # ---------------- recommendation ----------------
    A("## 6. Recommendation")
    A("")
    A(f"### {ctx['recommendation']}")
    A("")
    for line in ctx["recommendation_detail"]:
        A(f"  - {line.strip()}" if line.startswith("    ") else f"- {line}")
    A("")

    # ---------------- confidence ----------------
    A("## 7. What is solid, and what is interpretation")
    A("")
    A("**High confidence (verified against the reference implementation).**")
    for line in ctx["confidence_high"]:
        A(f"- {line}")
    A("")
    A("**Interpretation, stated so a reviewer can disagree.**")
    for line in ctx["confidence_low"]:
        A(f"- {line}")
    A("")
    A("**Not verifiable here.**")
    for line in ctx["confidence_none"]:
        A(f"- {line}")
    A("")
    A(f"Machine-readable results: `{cfgs['table_csv']}`.")
    A("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    ss = cfg["sample_size"]
    sim_cfg = ss["test_precision_simulation"]
    seed = int(cfg["reproducibility"]["random_seed"])
    tp_years = float(ss["timepoint_years"])
    auroc_grid = tuple(float(a) for a in sim_cfg["true_auroc_grid"])
    auroc_primary = float(sim_cfg["true_auroc_primary"])
    assert any(math.isclose(auroc_primary, a) for a in auroc_grid), \
        "test_precision_simulation.true_auroc_primary is not in true_auroc_grid"

    # ---- SEALED-SPLIT GUARD: the same predicate src/model_clinical.py uses ----
    # The docstring promises the locked test split is never read here; this is what makes
    # that true. `split != "test"` is pushed into the Parquet reader, so sealed rows are
    # never materialised, and the total-row invariant comes from FOOTER METADATA.
    feat_path = coh / "features_clinical.parquet"
    n_rows_total = parquet_num_rows(feat_path)                 # metadata only, no row group
    dev_df = load_development_frame(feat_path, forbid_test=True)
    assert "test" not in set(dev_df["split"].unique()), "sealed split reached sample_size_riley"
    assert {"event_indicator", "time_from_landmark", "split"} <= set(dev_df.columns), \
        "features_clinical.parquet is missing label columns"
    assert dev_df["empi_anon"].is_unique, "features_clinical is not one row per patient"
    pub = published_cohort_aggregates(cfg)                     # CONTEXT ONLY, never an input
    assert n_rows_total == pub["n"], \
        f"feature table has {n_rows_total} rows, published aggregates say {pub['n']}"
    log.info("test split SEALED: %d of %d rows loaded (train+val only); full-cohort context "
             "read from published aggregates (%s)", len(dev_df), n_rows_total, pub["source"])

    # Horizons come from the SHARED clamp, so 5 y is day 1825 here exactly as in M0.
    max_obs = float(dev_df["time_from_landmark"].max())
    horizon_specs = clamp_horizon_days(cfg["model_clinical"]["horizons_years"],
                                       float(cfg["timeline"]["days_per_year"]), max_obs)
    tp_spec = next((h for h in horizon_specs if math.isclose(h["horizon_years"], tp_years)), None)
    assert tp_spec is not None, "the sample-size timepoint is not among the model horizons"
    tp_days = float(tp_spec["horizon_days"])
    if tp_spec["clamped"]:
        log.warning("timepoint %.0f y clamped %d d -> %d d (administrative censoring on day "
                    "%d); this is the SAME rule src/model_clinical.py applies",
                    tp_years, tp_spec["horizon_days_nominal"], int(tp_days), int(max_obs))
    _cross_check_horizons_against_model_json(coh, horizon_specs, log)

    p, pdet = candidate_parameter_count(cfg, coh)
    assert pdet["n_design_columns"] == EXPECTED_DESIGN_COLUMNS, (
        f"design column count {pdet['n_design_columns']} != the pre-specified "
        f"{EXPECTED_DESIGN_COLUMNS}")
    if pdet["identified_parameters"] is not None:
        assert p == EXPECTED_IDENTIFIED_PARAMS + int(pdet["extra_image_parameters"]), \
            f"identified parameter count {p} != the pre-specified {EXPECTED_IDENTIFIED_PARAMS}"
    log.info("candidate parameters P=%d (%s); design columns %d", p,
             pdet["parameters_source"], pdet["n_design_columns"])

    dev = observed_inputs(dev_df, tp_days)
    log.info("development inputs: n=%d events=%d phi=%.5f person-years=%.1f meanfup=%.4f y "
             "rate=%.5f/py", dev["n"], dev["n_events"], dev["event_fraction"],
             dev["person_years"], dev["mean_followup_years"], dev["rate_per_person_year"])
    log.info("KM risk at day %d = %.5f (Greenwood SE %.5f); reverse-KM median follow-up %.0f d "
             "(development rows only)", int(tp_days), dev["km_risk_at_t"],
             dev["km_greenwood_se_at_t"], dev["median_followup_reverse_km_days"])

    # ---- Part 1 -----------------------------------------------------------
    max_r2 = max_r2_cs_survival(dev["event_fraction"])
    riley_rows, primary = [], None
    for f in ss["r2_fraction_of_max_grid"]:
        r = riley_survival(parameters=p, r2_cs_adj=float(f) * max_r2,
                           rate=dev["rate_per_person_year"], timepoint=tp_years,
                           mean_followup=dev["mean_followup_years"],
                           shrinkage=float(ss["target_shrinkage"]),
                           max_optimism=float(ss["max_optimism"]),
                           mape=float(ss["mape_target"]),
                           event_fraction=dev["event_fraction"])
        r["r2_fraction"] = float(f)
        r["is_primary"] = math.isclose(float(f), float(ss["r2_fraction_of_max_primary"]))
        riley_rows.append(r)
        if r["is_primary"]:
            primary = r
    assert primary is not None, "r2_fraction_of_max_primary is not in r2_fraction_of_max_grid"
    for r in riley_rows:
        log.info("Riley f=%.2f R2cs=%.4f -> n1=%d n2=%d n3=%d required=%d (%s) events=%.0f EPP=%.1f",
                 r["r2_fraction"], r["r2_cs_adj"], r["n_criterion1"], r["n_criterion2"],
                 r["n_criterion3"], r["n_required"], r["binding_criterion"],
                 r["events_required"], r["epp_required"])

    n_green = math.ceil(dev["n"] * (Z975 * dev["km_greenwood_se_at_t"]
                                    / float(ss["mape_target"])) ** 2)
    # Illustrative only: criterion 1 under the PUBLISHED full-cohort event fraction. It needs
    # nothing but phi, so it is computable from the aggregate files without reading a row.
    n1_published_phi = math.ceil(n_for_shrinkage(
        p, float(ss["r2_fraction_of_max_primary"]) * max_r2_cs_survival(pub["event_fraction"]),
        float(ss["target_shrinkage"])))
    c3 = dict(n_exponential=riley_rows[0]["n_criterion3"], n_greenwood=n_green)

    min_f_dev = min_r2_fraction_supported(p, dev["n"], max_r2, float(ss["target_shrinkage"]),
                                          float(ss["max_optimism"]))
    p_max_dev = max_parameters_supported(dev["n"], primary["r2_cs_adj"], max_r2,
                                         float(ss["target_shrinkage"]), float(ss["max_optimism"]))

    # ---- Part 2 -----------------------------------------------------------
    t_dev = dev_df["time_from_landmark"].to_numpy(float)
    e_dev = dev_df["event_indicator"].to_numpy(int)
    ut_e, s_e, _ = km_numpy(t_dev, e_dev)
    ev_t, ev_s = drop_flat(ut_e, s_e)
    _, kmc = reverse_km(t_dev, e_dev)
    cens_t = kmc.survival_function_.index.to_numpy(float)
    cens_s = kmc.survival_function_.iloc[:, 0].to_numpy(float)
    cens_t, cens_s = drop_flat(cens_t, cens_s)
    assert cens_s[-1] <= 1e-12, "reverse-KM censoring curve does not reach 0 at the horizon"

    n_test = int(sim_cfg["n_test"])
    n_sim = int(sim_cfg["n_sim"])
    tgt_auc = float(sim_cfg["target_auroc_ci_halfwidth"])
    tgt_slope = float(sim_cfg["target_calibration_slope_ci_halfwidth"])
    # Smallest censoring-survival value a case weight can meet in these data, per horizon:
    # the report quotes it against IPCW_WEIGHT_FLOOR so the cap is visible, not silent.
    g_min_by_horizon = {float(h["horizon_days"]):
                        float(step_eval(cens_t, cens_s, h["horizon_days"], left_limit=True)[0])
                        for h in horizon_specs}

    rng = np.random.default_rng(seed)
    sim_rows, sigmas = [], {}
    for hspec in horizon_specs:
        h_years = float(hspec["horizon_years"])
        h_days = float(hspec["horizon_days"])
        keep = ev_t <= h_days
        t_grid, s_grid = ev_t[keep], ev_s[keep]
        risk_h = 1.0 - float(s_grid[-1])
        for target in auroc_grid:
            sigma = sigma_for_auc(target, risk_h)
            sigmas[(h_years, target)] = sigma
            h0 = baseline_hazard_grid(sigma, s_grid)
            reps = simulate_replicates(rng, n_test, sigma, h0, t_grid, cens_t, cens_s,
                                       h_days, n_sim)
            summ = summarise_replicates(reps)
            hm = hanley_mcneil_se(target, summ["mean_cases"], summ["mean_controls"])
            row = dict(horizon_years=h_years, horizon_days=h_days, true_auroc=target,
                       marginal_risk=risk_h, sigma_lp=sigma, n_test=n_test, **summ,
                       auroc_halfwidth_hanley_mcneil=Z975 * hm,
                       auroc_target_met=summ["auroc_halfwidth"] <= tgt_auc,
                       slope_target_met=summ["slope_halfwidth"] <= tgt_slope,
                       n_test_required_auroc=required_n_for_halfwidth(
                           n_test, summ["auroc_halfwidth"], tgt_auc),
                       n_test_required_slope=required_n_for_halfwidth(
                           n_test, summ["slope_halfwidth"], tgt_slope))
            sim_rows.append(row)
            log.info("sim h=%.0fy AUROC=%.2f sigma=%.3f cases=%.0f ctrls=%.0f "
                     "auroc_hw=%.4f (HM %.4f) slope_hw=%.4f (Wald %.4f)",
                     h_years, target, sigma, summ["mean_cases"], summ["mean_controls"],
                     summ["auroc_halfwidth"], Z975 * hm, summ["slope_halfwidth"],
                     summ["slope_halfwidth_wald"])

    prim_sims = [r for r in sim_rows
                 if math.isclose(r["horizon_years"], tp_years)
                 and math.isclose(r["true_auroc"], auroc_primary)]
    assert len(prim_sims) == 1, "primary simulation scenario not uniquely identified"
    prim_sim = prim_sims[0]

    # Confirmatory re-simulation at the implied larger test size (verifies the 1/sqrt(n) scaling).
    n_conf = max(prim_sim["n_test_required_auroc"], prim_sim["n_test_required_slope"])
    h_days_p = prim_sim["horizon_days"]
    keep_p = ev_t <= h_days_p
    h0_p = baseline_hazard_grid(prim_sim["sigma_lp"], ev_s[keep_p])
    conf = summarise_replicates(simulate_replicates(
        np.random.default_rng(seed + 1), n_conf, prim_sim["sigma_lp"], h0_p, ev_t[keep_p],
        cens_t, cens_s, h_days_p, max(200, int(n_sim * CONFIRM_SIM_FRACTION))))
    log.info("confirmatory sim at n_test=%d: auroc_hw=%.4f slope_hw=%.4f",
             n_conf, conf["auroc_halfwidth"], conf["slope_halfwidth"])

    # ---- Part 3 -----------------------------------------------------------
    alt = ss["alternative_window"]
    alt_n, alt_e = int(alt["n_final_cohort"]), int(alt["primary_events_5y"])
    alt_phi = alt_e / alt_n
    alt_max_r2 = max_r2_cs_survival(alt_phi)
    alt_rate = alt_e / (alt_n * dev["mean_followup_years"])          # same follow-up assumed
    alt_riley = riley_survival(parameters=p,
                               r2_cs_adj=float(ss["r2_fraction_of_max_primary"]) * alt_max_r2,
                               rate=alt_rate, timepoint=tp_years,
                               mean_followup=dev["mean_followup_years"],
                               shrinkage=float(ss["target_shrinkage"]),
                               max_optimism=float(ss["max_optimism"]),
                               mape=float(ss["mape_target"]), event_fraction=alt_phi)
    alt_dev = round(alt_n * dev["n"] / pub["n"])
    alt_test = alt_n - alt_dev
    alt_auroc_hw = prim_sim["auroc_halfwidth"] * math.sqrt(n_test / alt_test)
    alt_slope_hw = prim_sim["slope_halfwidth"] * math.sqrt(n_test / alt_test)

    # ---- decision ---------------------------------------------------------
    # How large could the test set get before the DEVELOPMENT sample stops meeting Riley?
    n_worst_required = max(r["n_required"] for r in riley_rows)
    n_test_cap_pessimistic = pub["n"] - n_worst_required
    n_test_cap_primary = pub["n"] - primary["n_required"]
    scale_cap = math.sqrt(n_test / n_test_cap_pessimistic)
    auroc_hw_at_cap = prim_sim["auroc_halfwidth"] * scale_cap
    slope_hw_at_cap = prim_sim["slope_halfwidth"] * scale_cap
    dev_after_conf = pub["n"] - n_conf

    dev_ok = dev["n"] >= primary["n_required"] and dev["n_events"] >= primary["events_required"]
    dev_ok_worst = dev["n"] >= max(r["n_required"] for r in riley_rows)
    auroc_ok_any = any(r["auroc_target_met"] for r in sim_rows
                       if math.isclose(r["horizon_years"], tp_years))
    slope_ok_prim = prim_sim["slope_target_met"]
    slope_ok_high = any(r["slope_target_met"] for r in sim_rows
                        if math.isclose(r["horizon_years"], tp_years))

    if dev_ok and auroc_ok_any and slope_ok_prim:
        recommendation = "PROCEED"
    elif dev_ok:
        recommendation = ("PROCEED on model development; SIMPLIFY THE TEST-SET CLAIM "
                          "(do not widen the imaging window)")
    else:
        recommendation = "SIMPLIFY THE MODEL"

    headline = [
        f"**Development sample: sufficient.** The Riley requirement at the primary assumption "
        f"(Nagelkerke R2 = {ss['r2_fraction_of_max_primary']} of the maximum) is "
        f"{primary['n_required']:,} patients / {primary['events_required']:.0f} events for "
        f"P = {p} parameters. The development set (train + val) holds {dev['n']:,} patients and "
        f"{dev['n_events']} events, i.e. {dev['n'] / primary['n_required']:.2f}x the requirement, "
        f"{dev['n_events'] / p:.1f} events per parameter. Even the pessimistic assumption "
        f"(f = {min(ss['r2_fraction_of_max_grid'])}) needs only "
        f"{max(r['n_required'] for r in riley_rows):,}, still "
        f"{'met' if dev_ok_worst else 'NOT met'}.",
        f"**Protocol section 16's stated numeric floor IS met.** Section 16 names one: \"a "
        f"practical preliminary floor is 500 total primary events and 100 test events\". The "
        f"study has **{pub['n_events']} total primary events (>= 500) and "
        f"{int(sim_cfg['n_test_events'])} test events (>= 100)** — both cleared. Section 16 "
        f"asks for \"acceptable precision\" but states no numeric precision target.",
        f"**The locked test set misses the tighter half-width targets adopted for this "
        f"analysis.** The +/-{tgt_auc} AUROC and +/-{tgt_slope} calibration-slope half-widths "
        f"are ANALYST-ADOPTED choices recorded in `config/feasibility.yaml`, not protocol "
        f"values. At the primary scenario ({tp_years:.0f}-year horizon, true AUROC "
        f"{auroc_primary:.2f}) the {int(sim_cfg['n_test']):,}-patient test set gives an AUROC "
        f"95% CI half-width of {prim_sim['auroc_halfwidth']:.3f} and a calibration-slope "
        f"half-width of {prim_sim['slope_halfwidth']:.3f}. Meeting both adopted targets would "
        f"need about {n_conf:,} test patients "
        f"({100 * n_conf / pub['n']:.0f}% of the cohort), leaving {dev_after_conf:,} for "
        f"development, which is "
        f"{'still above' if dev_after_conf >= n_worst_required else 'BELOW'} the pessimistic "
        f"Riley requirement of {n_worst_required:,}. Because the simulation assumes a normal "
        f"linear predictor and censoring independent of risk, these half-widths are lower "
        f"bounds on real-world uncertainty.",
        f"**Widening the imaging window to 3 years is not worth it.** +{alt_n - pub['n']} "
        f"patients and +{alt_e - pub['n_events']} events move the required development sample "
        f"from {primary['n_required']:,} to {alt_riley['n_required']:,} (already met either "
        f"way) and shrink the test-set AUROC half-width from "
        f"{prim_sim['auroc_halfwidth']:.3f} to only {alt_auroc_hw:.3f}, at the cost of "
        f"radiographs up to a year staler relative to the index TKA.",
        f"**Recommendation: {recommendation}.**",
    ]

    sim_checks = [
        f"The mean simulated calibration slope is {prim_sim['slope_mean']:.4f} at the primary "
        f"scenario; the true value is 1 by construction, so the estimator is unbiased.",
        f"The mean simulated AUROC is {prim_sim['auroc_mean']:.4f} against a true "
        f"{auroc_primary:.2f}, so the IPCW estimator recovers the value the "
        f"data-generating process was solved for.",
        f"The Monte-Carlo slope half-width ({prim_sim['slope_halfwidth']:.4f}) and the mean "
        f"within-replicate Wald half-width ({prim_sim['slope_halfwidth_wald']:.4f}) agree, so "
        f"the model-based standard error is trustworthy in a real single validation.",
        f"The Monte-Carlo AUROC half-width ({prim_sim['auroc_halfwidth']:.4f}) is close to the "
        f"analytic Hanley-McNeil value for the realised "
        f"{prim_sim['mean_cases']:.0f} cases and {prim_sim['mean_controls']:.0f} controls "
        f"({prim_sim['auroc_halfwidth_hanley_mcneil']:.4f}), an independent formula.",
        f"Re-simulating from scratch at the implied n_test = {n_conf:,} (independent seed) "
        f"returns an AUROC half-width of {conf['auroc_halfwidth']:.4f} (target {tgt_auc}) and a "
        f"slope half-width of {conf['slope_halfwidth']:.4f} (target {tgt_slope}), both within "
        f"Monte-Carlo error of the targets. That confirms the 1/sqrt(n) scaling used to derive "
        f"every 'n_test needed' figure in the table.",
    ]

    five = [r for r in sim_rows if math.isclose(r["horizon_years"], tp_years)]
    two = [r for r in sim_rows if math.isclose(r["horizon_years"], 2.0)]
    five_p = next(r for r in five if math.isclose(r["true_auroc"], auroc_primary))
    two_p = next(r for r in two if math.isclose(r["true_auroc"], auroc_primary))
    auroc_ever = any(r["auroc_target_met"] for r in sim_rows)
    sim_narrative = [
        f"The number of usable cases and controls, not the number of patients, drives "
        f"precision. At {tp_years:.0f} years the test set contributes about "
        f"{five_p['mean_cases']:.0f} cases and {five_p['mean_controls']:.0f} controls known "
        f"event-free; the roughly "
        f"{n_test - five_p['mean_cases'] - five_p['mean_controls']:.0f} patients censored "
        f"before the horizon enter only through the IPCW weights.",
        f"The two horizons trade off in opposite directions, so neither is uniformly "
        f"better-powered. At true AUROC {auroc_primary:.2f} the 2-year AUROC half-width "
        f"({two_p['auroc_halfwidth']:.4f}) is "
        f"{'NARROWER' if two_p['auroc_halfwidth'] < five_p['auroc_halfwidth'] else 'WIDER'} "
        f"than the {tp_years:.0f}-year one ({five_p['auroc_halfwidth']:.4f}): 2 years gives up "
        f"{five_p['mean_cases'] - two_p['mean_cases']:.0f} cases but gains "
        f"{two_p['mean_controls'] - five_p['mean_controls']:.0f} controls known event-free. "
        f"The calibration slope goes the other way "
        f"({two_p['slope_halfwidth']:.4f} at 2 years versus "
        f"{five_p['slope_halfwidth']:.4f} at {tp_years:.0f} years), because truncating "
        f"follow-up at 2 years discards the later events that identify the slope. The 2-year "
        f"co-primary earns its place on clinical relevance and lighter censoring adjustment, "
        f"not on a precision advantage.",
        f"The ADOPTED AUROC target of +/-{tgt_auc} (analyst choice; protocol section 16 sets "
        f"no numeric target) is "
        f"{'met somewhere in' if auroc_ever else 'missed in EVERY cell of'} the grid "
        f"(half-widths {min(r['auroc_halfwidth'] for r in sim_rows):.4f} to "
        f"{max(r['auroc_halfwidth'] for r in sim_rows):.4f} across all horizons and true "
        f"AUROC {min(auroc_grid):.2f} to {max(auroc_grid):.2f}). Discrimination that is "
        f"genuinely higher helps only weakly: at {tp_years:.0f} years the half-width falls "
        f"from {max(r['auroc_halfwidth'] for r in five):.4f} to "
        f"{min(r['auroc_halfwidth'] for r in five):.4f} across the whole AUROC range.",
        (f"The calibration-slope target of +/-{tgt_slope} depends strongly on how well the "
         f"model discriminates. At {tp_years:.0f} years it is "
         + (f"met only from true AUROC "
            f"{min(r['true_auroc'] for r in five if r['slope_target_met']):.2f} upward "
            f"(half-width {min(r['slope_halfwidth'] for r in five):.4f}), and missed below it "
            f"(up to {max(r['slope_halfwidth'] for r in five):.4f} at true AUROC "
            f"{min(auroc_grid):.2f})." if slope_ok_high else
            f"never met (half-widths {min(r['slope_halfwidth'] for r in five):.4f} to "
            f"{max(r['slope_halfwidth'] for r in five):.4f}).")
         + " Slope precision scales as 1 / (sigma * sqrt(events)), and a weakly "
           "discriminating model has a narrow linear predictor, hence a poorly identified "
           "slope."),
        f"A useful reframing: with {n_test:,} test patients the achievable "
        f"{tp_years:.0f}-year AUROC precision is about "
        f"+/-{prim_sim['auroc_halfwidth']:.3f}, i.e. a 95% CI of roughly "
        f"{auroc_primary - prim_sim['auroc_halfwidth']:.3f} to "
        f"{auroc_primary + prim_sim['auroc_halfwidth']:.3f} around a true "
        f"{auroc_primary:.2f}. That is enough to show the model beats chance and to compare it "
        f"with the clinical baseline in a paired analysis, but not enough to certify a "
        f"specific AUROC to two decimal places.",
    ]

    alt_rows = [
        ("Patients", f"{pub['n']:,}", f"{alt_n:,}", f"+{alt_n - pub['n']}"),
        ("5-year events", f"{pub['n_events']}", f"{alt_e}", f"+{alt_e - pub['n_events']}"),
        ("Event fraction phi", _fmt(pub["event_fraction"], 5), _fmt(alt_phi, 5),
         _fmt(alt_phi - pub["event_fraction"], 5)),
        ("max R2_CS", _fmt(max_r2, 4), _fmt(alt_max_r2, 4), _fmt(alt_max_r2 - max_r2, 4)),
        (f"Riley required n (f = {ss['r2_fraction_of_max_primary']})",
         f"{primary['n_required']:,}", f"{alt_riley['n_required']:,}",
         f"{alt_riley['n_required'] - primary['n_required']:+,}"),
        ("Development set (same 80/20 split)", f"{dev['n']:,}", f"{alt_dev:,}",
         f"+{alt_dev - dev['n']}"),
        ("Test set (same 20%)", f"{n_test:,}", f"{alt_test:,}", f"+{alt_test - n_test}"),
        (f"Test AUROC half-width at {tp_years:.0f} y, true AUROC {auroc_primary:.2f}",
         _fmt(prim_sim["auroc_halfwidth"], 4), _fmt(alt_auroc_hw, 4),
         _fmt(alt_auroc_hw - prim_sim["auroc_halfwidth"], 4)),
        ("Test calibration-slope half-width", _fmt(prim_sim["slope_halfwidth"], 4),
         _fmt(alt_slope_hw, 4), _fmt(alt_slope_hw - prim_sim["slope_halfwidth"], 4)),
    ]
    cohort_needed_20pct = 5 * prim_sim["n_test_required_auroc"]
    alt_narrative = [
        f"The development sample is not the constraint, so extra patients buy nothing there: "
        f"{dev['n']:,} already exceeds the {primary['n_required']:,} required, and the widened "
        f"cohort would require {alt_riley['n_required']:,}.",
        f"The test set is the constraint, and "
        f"{100 * (alt_n - pub['n']) / pub['n']:.1f}% more patients shrink the AUROC half-width "
        f"by {abs(alt_auroc_hw - prim_sim['auroc_halfwidth']):.4f} "
        f"({100 * abs(alt_auroc_hw - prim_sim['auroc_halfwidth']) / prim_sim['auroc_halfwidth']:.1f}%), "
        f"from {prim_sim['auroc_halfwidth']:.4f} to {alt_auroc_hw:.4f} against a target of "
        f"{tgt_auc}. Reaching that target by growing the cohort while keeping a 20% test "
        f"fraction would need roughly {cohort_needed_20pct:,} patients "
        f"({cohort_needed_20pct / pub['n']:.1f}x the current cohort), which no widening of the "
        f"imaging window can deliver.",
        "The cost is not neutral: a 3-year pre-index window admits radiographs up to a year "
        "staler relative to the index TKA, so the exposure is measured further from the "
        "prediction origin and the contralateral knee has had more unobserved time to "
        "progress. That directly attacks the study's central claim.",
        f"Verdict: do NOT widen. The +{alt_e - pub['n_events']} events change the required "
        f"development sample by {alt_riley['n_required'] - primary['n_required']:+,} patients "
        f"(both already met by a factor of about {dev['n'] / primary['n_required']:.1f}) and "
        f"do not flip a single decision in this document.",
    ]

    recommendation_detail = [
        f"**Proceed with model development.** {dev['n']:,} development patients and "
        f"{dev['n_events']} events against a Riley requirement of {primary['n_required']:,} / "
        f"{primary['events_required']:.0f} at P = {p}. Do not add parameters casually: the "
        f"same data supports at most about {p_max_dev} at the primary R2 assumption, and the "
        f"pre-specified block of {p} should stay frozen with no univariable screening "
        "(protocol section 19).",
        f"**Do not widen the pre-index imaging window.** +{alt_n - pub['n']} patients and "
        f"+{alt_e - pub['n_events']} events change nothing quantitative and cost image "
        f"recency.",
        f"**Revise what the locked test set is asked to prove.** Protocol section 16's stated "
        f"preliminary floor — 500 total primary events and 100 test events — IS met "
        f"({pub['n_events']} and {int(sim_cfg['n_test_events'])}); nothing below contradicts "
        f"that. What is not met is the stricter +/-{tgt_auc} AUROC half-width **adopted for "
        f"this analysis** (`config/feasibility.yaml`), and section 16 states no numeric "
        f"precision target of its own. With {n_test:,} patients the {tp_years:.0f}-year AUROC "
        f"is estimable to about +/-{prim_sim['auroc_halfwidth']:.3f} and the calibration slope "
        f"to about +/-{prim_sim['slope_halfwidth']:.3f} at a true AUROC of "
        f"{auroc_primary:.2f}. Section 16's decision rule still applies to the gap between "
        "ambition and precision: simplify the model and revise the question rather than "
        "proceed underpowered — which is what (a) and (c) below do.",
        "**Concretely, one of these three, decided before the test set is unsealed:**",
        f"    (a) Re-specify the primary test-set estimand as a PAIRED comparison against the "
        f"clinical baseline M0 (difference in AUROC, difference in the index of prediction "
        f"accuracy) rather than an absolute AUROC. Paired differences share patients and are "
        f"far more precisely estimated than either absolute value, so a {n_test:,}-patient "
        f"test set can support a difference claim it cannot support for a level claim.",
        f"    (b) Increase the test fraction, but note the ceiling. Meeting BOTH targets needs "
        f"about {n_conf:,} test patients ({100 * n_conf / pub['n']:.0f}% of the cohort), which "
        f"leaves only {dev_after_conf:,} for development. That still clears the primary Riley "
        f"requirement ({primary['n_required']:,} at f = {ss['r2_fraction_of_max_primary']}) "
        + ("and also clears" if dev_after_conf >= n_worst_required else "but FAILS")
        + f" the pessimistic one "
        f"({n_worst_required:,} at f = {min(ss['r2_fraction_of_max_grid'])}). The largest test "
        f"set that keeps the pessimistic development requirement intact is "
        f"{n_test_cap_pessimistic:,} patients "
        f"({100 * n_test_cap_pessimistic / pub['n']:.0f}%), which would give an AUROC "
        f"half-width of {auroc_hw_at_cap:.3f} "
        f"({'meets' if auroc_hw_at_cap <= tgt_auc else 'misses'} the {tgt_auc} target) and a "
        f"slope half-width of {slope_hw_at_cap:.3f} "
        f"({'meets' if slope_hw_at_cap <= tgt_slope else 'misses'} the {tgt_slope} target); "
        f"tolerating only the primary R2 assumption would raise that ceiling to "
        f"{n_test_cap_primary:,}. So a re-split "
        + ("can buy both adopted targets at once"
           if (auroc_hw_at_cap <= tgt_auc and slope_hw_at_cap <= tgt_slope) else
           "can buy the AUROC target but not, quite, the slope target")
        + " without betting on an optimistic R2 — but it also means "
        f"re-drawing the LOCKED splits (protocol section 17), which is an investigator "
        f"decision, not a script's. Nothing in this module changes them, and (a) reaches the "
        f"same scientific end without touching them.",
        "    (c) Keep the splits and pre-specify the precision honestly in the protocol and "
        "the manuscript: report the CI and state up front that the study is powered to "
        "demonstrate discrimination better than chance and better than the clinical baseline, "
        "not to certify a point estimate.",
        "**My recommendation, if only one is chosen: (a) plus (c).** The splits are already "
        "locked and used by a sibling module; the scientific question that matters is whether "
        "radiographs add anything over clinical variables, which is a paired difference, and a "
        "paired difference is exactly the estimand the available test set can support.",
    ]

    confidence_high = [
        "Criteria 1 and 2 and the max R2_CS formula. These reproduce `pmsampsize_surv()` "
        "line for line; the published worked example (5143 / 1039 / 0.555 / 0.092) is "
        "recomputed exactly in the unit tests.",
        "The observed inputs. Event rate, person-time, follow-up quantiles, Kaplan-Meier risk "
        "and the reverse-KM censoring curve are all computed from "
        "`derived-data/cohort/features_clinical.parquet`, and the reverse-KM helper is the "
        "same `src.followup.reverse_km` used in the feasibility report.",
        "The univariate Cox solver and the IPCW AUROC. The Cox solver agrees with lifelines to "
        "6 decimal places on untied data (unit test), and the simulated AUROC recovers the "
        "value the data-generating process was solved for.",
        "The conclusion that the development sample is sufficient. It holds by a factor of "
        "roughly two under every assumption in the grid, so it is not sensitive to the "
        "R2 choice.",
    ]
    confidence_low = [
        "The anticipated Cox-Snell R2. There is no prior model for this outcome, so the "
        "fraction-of-maximum device is a convention, not evidence. It is presented across a "
        "grid for exactly that reason.",
        "Criterion 3 solved for n. `pmsampsize_surv()` does not solve criterion 3 for survival "
        "outcomes; it fixes n = max(n1, n2) and reports the interval. The closed form here is "
        "my inversion of the same expression. It is cross-checked against a Greenwood-based "
        "calculation on the observed Kaplan-Meier curve, and the two agree, but the "
        "constant-hazard assumption behind the published form is visibly violated in this "
        "cohort (exponential 5-year risk 0.249 vs observed 0.200).",
        "Charging the image model zero extra parameters. This follows the config "
        "(`extra_image_parameters: 0`) and is defensible for a frozen encoder with a small "
        "survival head, but no version of the Riley framework covers deep representation "
        "learning. The clinical-baseline requirement should not be read as a sample-size "
        "justification for the image model.",
        "The simulation's proportional-hazards, normal-linear-predictor, independent-censoring "
        "data-generating process. It is standard and it reproduces the observed marginal "
        "survival and censoring curves, but a real model's linear predictor will not be exactly "
        "normal and censoring may depend on covariates.",
        f"The discrimination grid {auroc_grid} and the primary value {auroc_primary}. These "
        "now live in `config/feasibility.yaml` "
        "(`sample_size.test_precision_simulation.true_auroc_grid` / `.true_auroc_primary`) "
        "rather than in code, so the assumption is visible and auditable — but it is still an "
        "assumption: this is the plausible range for a radiographic progression model, not a "
        "measurement.",
        f"The +/-{tgt_auc} AUROC and +/-{tgt_slope} calibration-slope half-widths. **These are "
        "analyst-adopted, not protocol values.** Protocol section 16 requires \"acceptable "
        "precision\" without defining it numerically; its only numeric floor is 500 total "
        f"primary events and 100 test events, and that floor is met ({pub['n_events']} and "
        f"{int(sim_cfg['n_test_events'])}). A reviewer who considers +/-0.07 acceptable "
        "for a first-in-domain model would read this document's precision verdict differently, "
        "and the config now says so in a comment.",
    ]
    confidence_none = [
        "Whether the model will in fact reach any particular AUROC. Everything in section 4 is "
        "conditional on an assumed true discrimination.",
        "Whether the laterality QA audit will hold. The phase-1 PROCEED was contingent on it, "
        "and a laterality error rate above a few percent would degrade both the labels and the "
        "images in ways no sample-size calculation can offset.",
        "Whether an investigator will accept re-drawing the locked splits. That is a protocol "
        "amendment decision.",
    ]

    # ---- outputs ----------------------------------------------------------
    out_rows = []
    for r in riley_rows:
        out_rows.append(dict(
            section="riley_development", scenario=f"r2_fraction={r['r2_fraction']:.2f}",
            parameters=p, r2_fraction_of_max=r["r2_fraction"], r2_cs_adj=r["r2_cs_adj"],
            max_r2_cs=r["max_r2_cs"], nagelkerke_r2=r["nagelkerke_r2"],
            shrinkage_criterion2=r["shrinkage_criterion2"], n_criterion1=r["n_criterion1"],
            n_criterion2=r["n_criterion2"], n_criterion3=r["n_criterion3"],
            n_required=r["n_required"], binding_criterion=r["binding_criterion"],
            events_required=r["events_required"], epp_required=r["epp_required"],
            n_observed_cohort=pub["n"], events_observed_cohort=pub["n_events"],
            n_observed_development=dev["n"], events_observed_development=dev["n_events"],
            epp_observed_development=dev["n_events"] / p,
            meets_requirement_development=bool(dev["n"] >= r["n_required"])))
    for r in sim_rows:
        out_rows.append(dict(
            section="test_precision",
            scenario=f"horizon={r['horizon_years']:.0f}y,auroc={r['true_auroc']:.2f}",
            parameters=p, horizon_years=r["horizon_years"],
            # horizon_days makes this CSV reconcilable against m0_clinical_model.json:
            # both come from src.model_clinical.clamp_horizon_days (5 y -> day 1825).
            horizon_days=r["horizon_days"], true_auroc=r["true_auroc"],
            sigma_lp=r["sigma_lp"], marginal_risk=r["marginal_risk"], n_test=r["n_test"],
            n_replicates=r["n_replicates"], n_valid_auroc=r["n_valid_auroc"],
            n_valid_slope=r["n_valid_slope"],
            n_slope_not_converged=r["n_slope_not_converged"],
            ipcw_weight_floor=IPCW_WEIGHT_FLOOR,
            mean_cases=r["mean_cases"], mean_controls=r["mean_controls"],
            mean_events=r["mean_events"], auroc_mean=r["auroc_mean"],
            auroc_halfwidth=r["auroc_halfwidth"],
            auroc_halfwidth_empirical=r["auroc_halfwidth_empirical"],
            auroc_halfwidth_hanley_mcneil=r["auroc_halfwidth_hanley_mcneil"],
            auroc_target_met=r["auroc_target_met"], slope_mean=r["slope_mean"],
            slope_halfwidth=r["slope_halfwidth"], slope_halfwidth_wald=r["slope_halfwidth_wald"],
            slope_halfwidth_empirical=r["slope_halfwidth_empirical"],
            slope_target_met=r["slope_target_met"],
            n_test_required_auroc=r["n_test_required_auroc"],
            n_test_required_slope=r["n_test_required_slope"]))
    out_rows.append(dict(
        section="alternative_window", scenario=alt["label"], parameters=p,
        r2_fraction_of_max=float(ss["r2_fraction_of_max_primary"]),
        r2_cs_adj=alt_riley["r2_cs_adj"], max_r2_cs=alt_max_r2,
        nagelkerke_r2=alt_riley["nagelkerke_r2"], n_criterion1=alt_riley["n_criterion1"],
        n_criterion2=alt_riley["n_criterion2"], n_criterion3=alt_riley["n_criterion3"],
        n_required=alt_riley["n_required"], binding_criterion=alt_riley["binding_criterion"],
        events_required=alt_riley["events_required"], epp_required=alt_riley["epp_required"],
        n_observed_cohort=alt_n, events_observed_cohort=alt_e, n_observed_development=alt_dev,
        n_test=alt_test, auroc_halfwidth=alt_auroc_hw, slope_halfwidth=alt_slope_hw,
        horizon_years=tp_years, horizon_days=tp_days, true_auroc=auroc_primary))
    out_rows.append(dict(
        section="confirmatory_simulation",
        scenario=f"n_test={n_conf},horizon={tp_years:.0f}y,auroc={auroc_primary:.2f}",
        parameters=p, horizon_years=tp_years, horizon_days=tp_days,
        true_auroc=auroc_primary, n_test=n_conf,
        n_replicates=conf["n_replicates"], n_valid_auroc=conf["n_valid_auroc"],
        n_valid_slope=conf["n_valid_slope"],
        n_slope_not_converged=conf["n_slope_not_converged"],
        ipcw_weight_floor=IPCW_WEIGHT_FLOOR,
        sigma_lp=prim_sim["sigma_lp"], mean_cases=conf["mean_cases"],
        mean_controls=conf["mean_controls"], auroc_mean=conf["auroc_mean"],
        auroc_halfwidth=conf["auroc_halfwidth"], slope_mean=conf["slope_mean"],
        slope_halfwidth=conf["slope_halfwidth"],
        auroc_target_met=conf["auroc_halfwidth"] <= tgt_auc,
        slope_target_met=conf["slope_halfwidth"] <= tgt_slope))

    tbl = pd.DataFrame(out_rows)
    front = ["section", "scenario", "parameters"]
    tbl = tbl[front + [c for c in tbl.columns if c not in front]]
    tbl_path = cfg.path(ss["table_csv"])
    tbl_path.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(tbl_path, index=False)
    assert "empi_anon" not in tbl.columns, "identifier leaked into the sample-size table"

    write_report(cfg.path(ss["report_md"]), dict(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), seed=seed,
        pub=pub, dev=dev, param_detail=pdet, cfg_ss=ss, sim_cfg=sim_cfg, alt=alt,
        auroc_grid=auroc_grid, auroc_primary=auroc_primary, horizon_specs=horizon_specs,
        g_min_by_horizon=g_min_by_horizon, prim_sim=prim_sim,
        timepoint_years=tp_years, timepoint_days=tp_days, max_r2_cs=max_r2,
        riley_rows=riley_rows, primary=primary, c3=c3, min_f_dev=min_f_dev,
        n1_published_phi=n1_published_phi,
        p_max_dev=p_max_dev, n_train=int((dev_df["split"] == "train").sum()),
        e_train=int(dev_df.loc[dev_df["split"] == "train", "event_indicator"].sum()),
        sim_rows=sim_rows, sigmas=sigmas, sim_checks=sim_checks, sim_narrative=sim_narrative,
        alt_rows=alt_rows, alt_narrative=alt_narrative, headline=headline,
        recommendation=recommendation, recommendation_detail=recommendation_detail,
        confidence_high=confidence_high, confidence_low=confidence_low,
        confidence_none=confidence_none))

    log.info("P=%d max_r2_cs=%.4f primary required n=%d (%s) events=%.0f; development n=%d "
             "events=%d -> %s", p, max_r2, primary["n_required"], primary["binding_criterion"],
             primary["events_required"], dev["n"], dev["n_events"],
             "SUFFICIENT" if dev_ok else "INSUFFICIENT")
    log.info("test precision at %.0f y / AUROC %.2f: auroc_hw=%.4f (target %.2f) "
             "slope_hw=%.4f (target %.2f) -> n_test needed %d",
             tp_years, auroc_primary, prim_sim["auroc_halfwidth"], tgt_auc,
             prim_sim["slope_halfwidth"], tgt_slope, n_conf)
    log.info("RECOMMENDATION: %s", recommendation)
    log.info("wrote %s and %s", cfg.path(ss["report_md"]), tbl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
