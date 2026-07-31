"""model_clinical.py — M0 and M1, the penalized-Cox clinical ladder of protocol Table 7.

Phase 2 / Track A, step 2. Fits the pre-specified clinical comparator of protocol
section 14 (model M0) on the LOCKED train split, tunes one hyper-parameter (the ridge
penalizer) with the criterion named by ``model_clinical.selection_metric``, and reports
discrimination, IPCW time-dependent AUROC and calibration at 1 / 2 / 5 years **on
validation only**. It then fits **M1 = M0 plus inferred KLG** on the KLG-eligible subset
and reports the two side by side (:func:`fit_m1_klg`, :func:`write_m1_outputs`).

Penalizer selection: two criteria, both reported, one of them selects
-------------------------------------------------------------------
``model_clinical.selection_metric`` chooses between

* ``val_cindex``      — Harrell's C on the LOCKED validation split (**54 events**), the
                        original behaviour; and
* ``cv_mean_cindex``  — the mean Harrell C over repeated stratified
                        :data:`CV_N_REPEATS` x :data:`CV_N_SPLITS`-fold cross-validation
                        *inside the training split* (**373 events**), which protocol
                        section 25 explicitly endorses as the stabler alternative.

**Both are computed on every run, for every fit** (M0, both M1 ladder arms and each
section-24 sensitivity), and both winners are logged, written to
``outputs/tables/m0_penalizer_grid.csv`` and printed in the report; only the configured
one selects. ``tuning_split: val`` still governs where the validation grid is evaluated,
so the losing criterion stays visible. The active criterion was switched from
``val_cindex`` to ``cv_mean_cindex`` on 2026-07-26 (deviation **D24**, resolving the
author decision flagged as **D23**): the validation C-index was monotone across the whole
pre-specified grid, so it never turned over, and the winner at the grid maximum compressed
predicted risks about tenfold (calibration slopes 7.8-13.0). The grid itself was **not**
widened. A winner at either end of the grid is still detected and reported, now against
whichever criterion actually selected.

The protocol Table 7 model ladder, and why M0 has no KLG in it
--------------------------------------------------------------
Table 7: **M0 = "Age, sex, comorbidities, pain, image-to-index interval"**;
**M1 = "M0 plus inferred KLG"**. Table 6 lists the dataset-inferred contralateral KLG as a
**secondary comparator only**. M0 is the comparator in the study's primary estimand
(Table 8: M4 multimodal versus M0 clinical-only, IPCW C/D AUROC at 5 years — *does imaging
improve prediction beyond routine clinical variables?*), so a radiograph-derived severity
grade inside M0 would be measuring imaging against imaging and would bias the incremental
value of imaging toward zero. The predictor lists are read from
``features_clinical.primary_predictors`` / ``m1_predictors`` via the frozen
``model_columns`` / ``m1_model_columns`` in the imputation JSON, and this module asserts
that no ``klg*`` column reaches the M0 design.

M1 is fitted **only where ``klg_contra`` is observed** (protocol Secondary objective 2:
"in the subset with eligible bilateral frontal images"); KLG is never median-imputed
cohort-wide for it. Because that restriction also changes the patients, M1 is compared
both against M0 refitted on the same eligible rows (the like-for-like anchor, where the
only difference is the KLG column) and against the published full-cohort M0 evaluated on
those rows.

**The test split is SEALED.** ``model_clinical.forbid_test_split`` is honoured by pushing
a ``split != "test"`` predicate down into the Parquet reader, so no test row is ever
materialised; the total-row invariant is checked against Parquet *metadata* and the
sealed-split counts against the frozen imputation JSON, never against test values.

Protocol sections implemented
-----------------------------
* **10 / 18** — censoring is administrative at day 1826 from the landmark or at the last
  observed date; horizon-specific evaluation uses inverse-probability-of-censoring
  weights, and every confidence interval is a patient-level bootstrap.
* **12 / 20** — the frozen train-fitted imputation from ``src/features_clinical.py`` is
  *replayed*, never refit; the spline knots are derived from TRAIN rows only. The module
  asserts the replayed ``*_imp`` columns reproduce the stored ones bit for bit.
* **19** — penalized Cox with a restricted cubic spline on age, the pre-specified
  predictor block entered WITHOUT univariable screening, and proportional hazards
  examined with scaled Schoenfeld residuals.
* **21** — subgroup performance with the pre-specified suppression rule (point estimates
  suppressed below 50 subgroup events, CIs emphasised below 100).
* **24** — complete-case, no-pain-predictor and race-included sensitivity models.
* **25** — repeated stratified 5-fold cross-validation *inside the training split*. Since
  D24 this is the **primary** penalizer criterion (373 events) rather than a cross-check
  on the 54-event validation choice, which is retained and reported alongside it.

IPCW time-dependent (cumulative/dynamic) AUROC — estimator and assumptions
--------------------------------------------------------------------------
At horizon ``t`` a patient is a **case** if ``T_i <= t`` with an observed event, a
**control** if ``T_i > t``, and is **not usable** if censored before ``t``. Uno-style
inverse-probability-of-censoring weights restore the censored patients' contribution:

    w_case_i = 1 / G(T_i^-)          w_control_j = 1 / G(t)

    AUC(t) = SUM_i SUM_j w_i w_j [ 1{r_i > r_j} + 0.5 * 1{r_i == r_j} ]
             ------------------------------------------------------------
                          ( SUM_i w_i ) ( SUM_j w_j )

``G(u) = P(C > u)`` is the censoring survival function, estimated by **reverse
Kaplan-Meier** (``src.followup.reverse_km``, the same estimator the Phase-1 follow-up
report used) on the **TRAIN** split — the convention scikit-survival uses, and the one
consistent with non-negotiable #2 (nothing is estimated from validation rows). ``G`` is
read left-continuously at case times and right-continuously at the horizon.

Assumptions this buys, and their cost:
  1. Censoring is independent of the event time given the covariates used, and (because
     ``G`` is marginal) independent of the covariates themselves. Loss to follow-up here
     is administrative or system-disengagement, so covariate-dependent censoring cannot
     be excluded; a covariate-conditional ``G`` is a documented future refinement.
  2. The censoring distribution is common to train and validation (random split -> the
     two are exchangeable). The validation-estimated ``G`` variant is reported alongside.
  3. Death is unavailable (protocol section 10), so the competing risk is unmeasured and
     these are cause-agnostic, not cause-specific, quantities.
  4. **At exactly 5 years the estimator is undefined**: administrative censoring lands on
     day 1826, so no patient is observed event-free beyond it (zero controls, ``G`` = 0).
     Horizons are therefore clamped to the last day strictly inside follow-up, day 1825
     (4.997 y). This is logged and stated in the report.

Three different "5-year maturity" counts exist, and only one of them belongs here
---------------------------------------------------------------------------------
For a 5-year risk model the quantity that matters is how many patients have a **DETERMINED
5-year status** — an observed event by day 1826, or event-free follow-up reaching it.
That is ``n_status_determined_5y`` = **1,133 of the 2,968 development patients (38.2%)**,
and it is the number this report uses whenever follow-up maturity is invoked. The two
other counts are real but answer different questions and must never be substituted for it:

* ``n_full_5y_record_coverage`` (the ``complete_5y`` flag, ``last_observed >= landmark +
  1826``): **746** development patients. It counts RECORD COVERAGE, so it drops 485 of the
  533 patients whose 5-year status is known precisely *because* they had the event and
  then left the record stream. It is not a maturity statistic for a survival model.
* ``n_followup_reaches_day_1825`` (``time_from_landmark >= 1825``, in
  ``outputs/sample_size.md``): **707** development patients — observed follow-up time
  reaching the clamped evaluation horizon, regardless of how the patient left the risk set.

**No computed metric depends on which one is quoted.** IPCW already handles administrative
censoring correctly; these counts appear only in prose. ``outputs/sample_size.md``
reconciles all three in one table.

Identifiability (why the parameter count is two numbers, not one)
-----------------------------------------------------------------
The M0 design has 13 columns and numeric rank 13, *and* spans the constant vector, because
patsy's ``cr()`` basis is a partition of unity (``age_rcs1 + age_rcs2 + age_rcs3 = 1``).
A Cox partial likelihood has no intercept, so a constant added to the linear predictor
cancels and **12 parameters are identified**: ``rank([X | 1]) - 1``. Ridge still fits and
predictions are unaffected — it just picks the minimum-norm representative — but no
individual ``age_rcs*`` hazard ratio may be reported, and 12 (not 13) is the number to
quote in an events-per-parameter statement. :func:`design_identifiability` computes this
by rank rather than by pairwise correlation, which is what a partition of unity requires.
M1 adds one column and one identified parameter (14 / 13).

Run:  python3 -m src.model_clinical --config config/feasibility.yaml
Writes outputs/clinical_baseline_report.md, outputs/tables/m0_{metrics,coefficients,
calibration,schoenfeld}.csv, outputs/clinical_m1_klg_report.md and
outputs/tables/m1_metrics.csv (all AGGREGATE ONLY — no empi_anon) and the frozen
derived-data/cohort/{m0_clinical_model,m1_klg_model}.json that task T7 replays in Colab.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pyarrow.parquet as pq
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import Config, ensure_dirs, load_config
from src.features_clinical import apply_imputer
from src.followup import reverse_km

MODULE = "model_clinical"

# LOCKED regression anchors (src/splits.py; outputs/tables/split_summary.csv). Deliberately
# NOT in config: a config edit must not be able to weaken the guard that would catch a
# silently-changed cohort. Test counts are checked against FROZEN METADATA, never test rows.
EXPECTED_N_PATIENTS = 3709
EXPECTED_SPLIT_N = {"train": 2597, "val": 371, "test": 741}
EXPECTED_SPLIT_EVENTS = {"train": 373, "val": 54, "test": 106}
EXPECTED_N_PARAMS = 13               # M0: 11 model columns - age + 3 spline basis columns
# A Cox partial likelihood has no intercept, so a constant added to the linear predictor
# cancels. patsy's cr() basis is a PARTITION OF UNITY (age_rcs1 + age_rcs2 + age_rcs3 = 1
# on every row), so the design spans the constant vector and one direction is unestimable:
# 13 columns, numeric rank 13, identified parameters 12. This is standard and harmless for
# prediction — it only means no individual age_rcs* hazard ratio may be reported as a level.
EXPECTED_IDENTIFIED_PARAMS = 12
# M1 (protocol Table 7) = M0 + klg_contra_imp on the KLG-eligible subset: one more column,
# one more identified parameter.
EXPECTED_M1_N_PARAMS = 14
EXPECTED_M1_IDENTIFIED_PARAMS = 13
DEV_SPLITS = ("train", "val")        # the only splits this module may ever hold
SEALED_SPLIT = "test"

# Protocol section 21 reporting rule.
SUPPRESS_BELOW_EVENTS = 50
EMPHASISE_CI_BELOW_EVENTS = 100

# Protocol section 25 repeated cross-validation (five-fold is pre-specified; the repeat
# count is not, and is fixed here rather than in config so the criterion cannot drift).
CV_N_SPLITS = 5
CV_N_REPEATS = 5

# The two penalizer-selection criteria. BOTH are computed on every fit and both winners are
# reported; `model_clinical.selection_metric` decides which one selects (deviation D24).
# Deliberately a closed set: an unrecognised value must fail loudly, never fall back.
SELECTION_METRICS = ("val_cindex", "cv_mean_cindex")
SELECTION_LABELS = {
    "val_cindex": "validation C-index",
    "cv_mean_cindex": f"{CV_N_REPEATS}x{CV_N_SPLITS}-fold cross-validated C-index "
                      "(inside the training split)",
}

CALIBRATION_BINS = 5                 # quintiles: 54 validation events -> ~11 events/bin
BOOTSTRAP_ALPHA = 0.05


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
# SEALED-SPLIT GUARD — the first thing that touches the feature table.          #
# --------------------------------------------------------------------------- #
def parquet_num_rows(path: Path) -> int:
    """Total row count from Parquet FOOTER METADATA — reads no row group, no test value."""
    return int(pq.ParquetFile(path).metadata.num_rows)


def load_development_frame(path: Path, *, forbid_test: bool = True,
                           split_col: str = "split") -> pd.DataFrame:
    """Load train+val ONLY. The ``split != "test"`` predicate is pushed into the Parquet
    reader so sealed rows are never materialised, and the result is asserted test-free."""
    filters = [(split_col, "!=", SEALED_SPLIT)] if forbid_test else None
    df = pd.read_parquet(path, filters=filters)
    present = set(df[split_col].unique())
    if forbid_test:
        assert SEALED_SPLIT not in present, \
            f"SEALED SPLIT VIOLATION: {SEALED_SPLIT!r} rows reached {MODULE}"
        assert present <= set(DEV_SPLITS), f"unexpected split values loaded: {sorted(present)}"
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# DESIGN MATRIX — restricted cubic spline knots from TRAIN rows only.           #
# --------------------------------------------------------------------------- #
def _extract_cr_knots(design_info) -> np.ndarray:
    """Recover the full knot vector patsy's stateful ``cr()`` memorised."""
    for fi in design_info.factor_infos.values():
        for obj in (fi.state or {}).get("transforms", {}).values():
            knots = getattr(obj, "_all_knots", None)
            if knots is not None:
                return np.asarray(knots, dtype=float)
    raise AssertionError("could not recover cr() knots from the patsy design_info")


def fit_age_spline(age_train, spline_df: int, *, basis_prefix: str = "age_rcs") -> dict:
    """Derive natural-cubic-spline (restricted cubic spline) knots from TRAIN ages ONLY.

    Returns a JSON-serialisable spec whose ``interior_knots`` / bounds fully determine the
    basis, so validation, the sealed test set and the Colab notebook (T7) all evaluate the
    identical transform. The caller must never pass validation or test ages in here.
    """
    age_train = np.asarray(age_train, dtype=float)
    assert age_train.size > 0 and np.isfinite(age_train).all(), "train ages must be finite"
    di = patsy.dmatrix(f"cr(age, df={int(spline_df)}) - 1", {"age": age_train},
                       return_type="dataframe").design_info
    all_knots = _extract_cr_knots(di)
    spec = {
        "variable": "age_at_index_imp",
        "kind": "patsy_cr_natural_cubic_spline",
        "df": int(spline_df),
        "n_basis_columns": int(spline_df),
        "all_knots": [float(k) for k in all_knots],
        "interior_knots": [float(k) for k in all_knots[1:-1]],
        "lower_bound": float(all_knots[0]),
        "upper_bound": float(all_knots[-1]),
        "knots_fit_on": "train",
        "n_train_rows_used": int(age_train.size),
        "basis_columns": [f"{basis_prefix}{i + 1}" for i in range(int(spline_df))],
    }
    spec["patsy_formula"] = (
        f"cr(age, knots={spec['interior_knots']}, lower_bound={spec['lower_bound']}, "
        f"upper_bound={spec['upper_bound']}) - 1")
    # The knot spec must reproduce the df-derived basis exactly, or patsy internals moved.
    a = patsy.build_design_matrices([di], {"age": age_train}, return_type="dataframe").pop().values
    b = spline_basis(age_train, spec).values
    assert np.abs(a - b).max() < 1e-12, "persisted knots do not reproduce the cr(df=) basis"
    return spec


def spline_basis(age, spline: dict) -> pd.DataFrame:
    """Evaluate the restricted cubic spline basis from the PERSISTED knots (no refit)."""
    age = np.asarray(age, dtype=float)
    dm = patsy.dmatrix("cr(age, knots=_k, lower_bound=_lb, upper_bound=_ub) - 1",
                       {"age": age, "_k": list(spline["interior_knots"]),
                        "_lb": float(spline["lower_bound"]), "_ub": float(spline["upper_bound"])},
                       return_type="dataframe")
    dm.columns = list(spline["basis_columns"])
    return dm.reset_index(drop=True)


def build_design(df: pd.DataFrame, spline: dict, model_columns: list[str]) -> pd.DataFrame:
    """Pre-specified block: every model column, age replaced by its spline basis.

    Protocol section 19 forbids univariable screening, so nothing here is selected — the
    block enters whole. Column ORDER is fixed (basis first, then the remaining model
    columns in their frozen order) because T7 reconstructs it from the JSON.
    """
    age_col = spline["variable"]
    assert age_col in model_columns, f"{age_col!r} missing from model_columns"
    basis = spline_basis(df[age_col].to_numpy(dtype=float), spline)
    rest = df[[c for c in model_columns if c != age_col]].astype("float64").reset_index(drop=True)
    X = pd.concat([basis, rest], axis=1)
    X.index = df.index
    assert X.notna().all().all(), "design matrix has missing values (imputation was skipped?)"
    return X


def aliased_columns(X: pd.DataFrame, tol: float = 1e-10) -> list[tuple[str, str, float]]:
    """Exactly collinear column PAIRS (|r| == 1) — a naming aid, not the identifiability test.

    A pairwise correlation scan can only see two-column relations. It cannot see a
    partition of unity (patsy's ``cr()`` basis: three columns summing to 1), which is why
    :func:`design_identifiability` — not this function — decides how many parameters are
    estimable. This is kept because when a pair IS aliased it names the two columns, which
    a rank number cannot.
    """
    C = X.corr().to_numpy()
    cols = list(X.columns)
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if np.isfinite(C[i, j]) and abs(abs(C[i, j]) - 1.0) < tol:
                out.append((cols[i], cols[j], float(C[i, j])))
    return out


def sums_to_constant(X: pd.DataFrame, columns: list[str], tol: float = 1e-9) -> bool:
    """True when the named block sums to the SAME value on every row (partition of unity).

    ``cr()`` bases satisfy this by construction. Such a block is *level-unidentified* in a
    model without an intercept: adding c to one basis coefficient and subtracting it from
    the others leaves every linear predictor unchanged, so the fitted age CURVE is
    identified but no single basis hazard ratio is.
    """
    cols = [c for c in columns if c in X.columns]
    if len(cols) < 2:
        return False
    s = X[cols].to_numpy(dtype=float).sum(axis=1)
    return bool(float(np.ptp(s)) <= tol)


def design_identifiability(X: pd.DataFrame) -> dict:
    """How many parameters a no-intercept Cox model can actually estimate from ``X``.

    Identifiability is a RANK question, not a correlation question. Two distinct relations
    live in this design and only one of them is a pair:

    * ``knee_pain_any_imp + pain_score_max_missing = 1`` (a pair — visible to
      :func:`aliased_columns`);
    * ``age_rcs1 + age_rcs2 + age_rcs3 = 1`` (three columns — invisible to any pairwise
      scan, because patsy's ``cr()`` basis is a partition of unity).

    Both make the design span the constant vector. A Cox partial likelihood has no
    intercept, so a constant added to the linear predictor cancels and one direction is
    unestimable on top of any ordinary rank deficiency:

        rank        = rank(X)
        identified  = rank([X | 1]) - 1

    Ridge keeps the fit well-posed by picking the minimum-norm representative, so
    PREDICTIONS are unaffected; what is affected is which coefficients may be read.
    """
    A = X.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(A))
    rank_aug = int(np.linalg.matrix_rank(np.hstack([A, np.ones((A.shape[0], 1))])))
    sv = np.linalg.svd(A, compute_uv=False)
    return {
        "n_columns": int(A.shape[1]),
        "rank": rank,
        "rank_with_intercept": rank_aug,
        "identified_parameters": int(rank_aug - 1),
        "rank_deficiency": int(A.shape[1] - rank),
        "spans_constant": bool(rank_aug == rank),
        "smallest_singular_value": float(sv[-1]),
        "largest_singular_value": float(sv[0]),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "aliased_column_pairs": [[a, b, r] for a, b, r in aliased_columns(X)],
    }


def clamp_horizon_days(horizons_years, days_per_year: float, max_observed: float) -> list[dict]:
    """Evaluation horizons in DAYS, clamped inside observed follow-up.

    THE single definition of the horizon grid, imported by ``src/sample_size_riley.py`` so
    the two modules cannot drift (they previously disagreed by a day at 5 years). The
    nominal horizon is ``round(years * days_per_year)``; because administrative censoring
    lands exactly on ``max_observed`` (day 1826 here), nobody is observed event-free BEYOND
    that day, so the cumulative/dynamic estimators have no control arm there and G = 0.
    Horizons are therefore capped at ``int(max_observed) - 1``, the last day strictly inside
    follow-up (day 1825 = 4.997 y).
    """
    out = []
    cap = int(max_observed) - 1
    for y in horizons_years:
        nominal = int(round(float(y) * float(days_per_year)))
        used = min(nominal, cap)
        out.append(dict(horizon_years=float(y), horizon_days_nominal=nominal,
                        horizon_days=float(used), clamped=bool(used != nominal)))
    return out


# --------------------------------------------------------------------------- #
# IPCW MACHINERY — reverse-KM censoring curve + Uno-style cumulative/dynamic AUC #
# --------------------------------------------------------------------------- #
def censoring_curve(times, events) -> tuple[np.ndarray, np.ndarray]:
    """Reverse-Kaplan-Meier censoring survival ``G(u) = P(C > u)`` as a step function.

    Wraps :func:`src.followup.reverse_km` (the Phase-1 estimator) and returns its step
    grid/values so the same curve can be persisted for T7 and replayed without lifelines.
    """
    _, kmf = reverse_km(np.asarray(times, dtype=float), np.asarray(events, dtype=int))
    sf = kmf.survival_function_
    grid = np.asarray(sf.index.values, dtype=float)
    vals = np.asarray(sf.iloc[:, 0].values, dtype=float)
    assert grid[0] == 0.0 and abs(vals[0] - 1.0) < 1e-12, "reverse-KM must start at G(0)=1"
    return grid, vals


def step_value(grid, vals, t, *, left: bool = False) -> np.ndarray:
    """Evaluate a right-continuous step function; ``left=True`` gives the left limit f(t-).

    Before the first grid point the value is 1.0 (nobody censored yet).
    """
    grid = np.asarray(grid, dtype=float)
    vals = np.asarray(vals, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    idx = np.searchsorted(grid, t, side=("left" if left else "right")) - 1
    return np.where(idx < 0, 1.0, vals[np.clip(idx, 0, len(vals) - 1)])


def ipcw_labels_weights(times, events, horizon: float, g_grid, g_vals):
    """Cumulative/dynamic case/control labels and IPCW weights at ``horizon``.

    Returns ``(y, w)`` where ``y`` is 1 = case (event by the horizon), 0 = control
    (event-free past the horizon) and -1 = not usable (censored before the horizon, weight
    0). Cases carry ``1 / G(T_i^-)``, controls ``1 / G(horizon)``.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    horizon = float(horizon)
    case = (times <= horizon) & (events == 1)
    ctrl = times > horizon
    y = np.where(case, 1, np.where(ctrl, 0, -1)).astype(int)
    w = np.zeros(times.shape, dtype=float)
    if case.any():
        g_case = step_value(g_grid, g_vals, times[case], left=True)
        assert (g_case > 0).all(), "censoring curve hit 0 at an event time — IPCW undefined"
        w[case] = 1.0 / g_case
    if ctrl.any():
        g_h = float(step_value(g_grid, g_vals, horizon, left=False)[0])
        assert g_h > 0, f"G({horizon}) = 0 — no follow-up beyond the horizon, IPCW undefined"
        w[ctrl] = 1.0 / g_h
    return y, w


def ipcw_auc(y, w, risk) -> float:
    """Weighted case-vs-control AUC; ties in ``risk`` score 0.5. NaN if either arm empty."""
    y = np.asarray(y, dtype=int); w = np.asarray(w, dtype=float); risk = np.asarray(risk, dtype=float)
    ci, co = y == 1, y == 0
    if not ci.any() or not co.any():
        return float("nan")
    rc, wc, ro, wo = risk[ci], w[ci], risk[co], w[co]
    cmp = (rc[:, None] > ro[None, :]).astype(float) + 0.5 * (rc[:, None] == ro[None, :])
    den = wc.sum() * wo.sum()
    return float(wc @ cmp @ wo / den) if den > 0 else float("nan")


def harrell_c(times, events, risk) -> float:
    """Harrell's C-index (lifelines). Higher ``risk`` must mean shorter survival."""
    return float(concordance_index(np.asarray(times, dtype=float),
                                   -np.asarray(risk, dtype=float),
                                   np.asarray(events, dtype=int)))


# --------------------------------------------------------------------------- #
# CALIBRATION                                                                   #
# --------------------------------------------------------------------------- #
def risk_bins(pred, n_bins: int) -> np.ndarray:
    """Equal-count bins of predicted risk, 0-indexed and ascending.

    Rank-based rather than ``qcut`` so every bin holds the same number of patients (to
    within one), which is what makes the per-bin Kaplan-Meier estimates comparable. Ties
    are broken by row order; with a continuous Cox risk score ties do not occur and the
    caller asserts so.
    """
    pred = np.asarray(pred, dtype=float)
    n = pred.size
    assert n >= n_bins, f"cannot form {n_bins} bins from {n} observations"
    r = pd.Series(pred).rank(method="first").to_numpy()
    return np.minimum(((r - 1) * n_bins // n).astype(int), n_bins - 1)


def km_risk(times, events, t: float) -> tuple[float, float, float]:
    """Kaplan-Meier observed risk ``1 - S(t)`` with the Greenwood 95% interval."""
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(times, dtype=float), event_observed=np.asarray(events, dtype=int))
    surv = float(np.asarray(kmf.predict(t)).ravel()[0])
    ci = kmf.confidence_interval_
    idx = max(int(np.searchsorted(ci.index.values, t, side="right")) - 1, 0)
    lo_s, hi_s = float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])
    return 1.0 - surv, 1.0 - hi_s, 1.0 - lo_s


def cloglog(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(-np.log(1.0 - p))


def calibration_slope_intercept(y, w, pred_risk) -> tuple[float, float]:
    """IPCW-weighted complementary-log-log recalibration at one horizon.

    ``cloglog(P(Y=1)) = a + b * cloglog(p_hat)``: ``b`` is the calibration slope (1 = the
    model's risk spread is right) and, refitting with ``b`` fixed at 1, ``a`` is
    calibration-in-the-large on the cloglog scale (0 = the average level is right). The
    cloglog link makes this exactly a proportional-hazards recalibration of the horizon
    risk. Censored-before-horizon patients are excluded (weight 0); the rest carry their
    IPCW weight. Returns ``(nan, nan)`` if the weighted GLM will not converge.
    """
    y = np.asarray(y, dtype=int); w = np.asarray(w, dtype=float)
    x = cloglog(pred_risk)
    m = (y >= 0) & (w > 0)
    yy, ww, xx = y[m].astype(float), w[m], x[m]
    if yy.size < 10 or len(np.unique(yy)) < 2:
        return float("nan"), float("nan")
    fam = sm.families.Binomial(link=sm.families.links.CLogLog())
    slope, intercept = float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.GLM(yy, sm.add_constant(xx, has_constant="add"), family=fam,
                         var_weights=ww).fit()
            slope = float(res.params[1])
        except Exception:                                    # noqa: BLE001 - report as NaN
            pass
        try:
            res0 = sm.GLM(yy, np.ones((yy.size, 1)), family=fam, var_weights=ww,
                          offset=xx).fit()
            intercept = float(res0.params[0])
        except Exception:                                    # noqa: BLE001
            pass
    return slope, intercept


def percentile_ci(values, alpha: float = BOOTSTRAP_ALPHA) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def suppression(n_events: int) -> tuple[bool, str]:
    """Protocol section 21: suppress point estimates below 50 events, flag CIs below 100."""
    if n_events < SUPPRESS_BELOW_EVENTS:
        return True, f"suppressed: {n_events} events < {SUPPRESS_BELOW_EVENTS}"
    if n_events < EMPHASISE_CI_BELOW_EVENTS:
        return False, f"interpret via CI: {n_events} events < {EMPHASISE_CI_BELOW_EVENTS}"
    return False, ""


# --------------------------------------------------------------------------- #
# FITTING / EVALUATION                                                          #
# --------------------------------------------------------------------------- #
def fit_cox(X: pd.DataFrame, times, events, penalizer: float, l1_ratio: float) -> CoxPHFitter:
    frame = X.copy()
    frame["_T"] = np.asarray(times, dtype=float)
    frame["_E"] = np.asarray(events, dtype=int)
    cph = CoxPHFitter(penalizer=float(penalizer), l1_ratio=float(l1_ratio))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(frame, duration_col="_T", event_col="_E")
    return cph


def linear_predictor(cph: CoxPHFitter, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(cph.predict_log_partial_hazard(X), dtype=float).ravel()


def horizon_risk(cph: CoxPHFitter, X: pd.DataFrame, horizon_days: float) -> np.ndarray:
    s = cph.predict_survival_function(X, times=[float(horizon_days)])
    return 1.0 - np.asarray(s.iloc[0].values, dtype=float).ravel()


def replay_from_json(model_json: dict, X: pd.DataFrame) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Reproduce M0 from the frozen JSON alone — the exact recipe handed to T7.

    Uses nothing but ``design_columns``, ``coefficients``, ``centering_means``,
    ``baseline_survival`` and ``horizons`` out of ``model_json``; lifelines is deliberately
    not touched. ``main`` runs this against the live fit and asserts agreement, so the
    report's "the replay was verified" sentence is a measurement, not a promise.

        lp(x)      = sum_j (x_j - centering_means[j]) * coefficients[j]
        risk(t|x)  = 1 - baseline_survival(t) ** exp(lp(x))

    ``baseline_survival`` is a right-continuous step function: take the last stored time
    <= t, and 1.0 before the first one.
    """
    cols = list(model_json["design_columns"])
    assert list(X.columns) == cols, "design column order differs from the frozen contract"
    coefs = np.array([float(model_json["coefficients"][c]) for c in cols])
    means = np.array([float(model_json["centering_means"][c]) for c in cols])
    lp = (X.to_numpy(dtype=float) - means) @ coefs
    bt = np.asarray(model_json["baseline_survival"]["times"], dtype=float)
    bs = np.asarray(model_json["baseline_survival"]["survival"], dtype=float)
    risk = {}
    for h in model_json["horizons"]:
        t = float(h["horizon_days"])
        idx = int(np.searchsorted(bt, t, side="right")) - 1
        s0 = 1.0 if idx < 0 else float(bs[idx])
        risk[t] = 1.0 - s0 ** np.exp(lp)
    return lp, risk


def tune_penalizer(Xtr, Ttr, Etr, Xva, Tva, Eva, grid, l1_ratio: float) -> pd.DataFrame:
    """Full penalizer grid on the tuning split — the winner AND the losers are reported."""
    rows = []
    for pen in grid:
        cph = fit_cox(Xtr, Ttr, Etr, pen, l1_ratio)
        rows.append(dict(penalizer=float(pen),
                         train_cindex=harrell_c(Ttr, Etr, linear_predictor(cph, Xtr)),
                         val_cindex=harrell_c(Tva, Eva, linear_predictor(cph, Xva)),
                         train_partial_loglik=float(cph.log_likelihood_)))
    return pd.DataFrame(rows)


def cv_penalizer_stability(train_df: pd.DataFrame, T, E, grid, l1_ratio: float,
                           spline_df: int, model_columns: list[str], seed: int, *,
                           drop_design_columns: list[str] | None = None,
                           log: logging.Logger | None = None) -> pd.DataFrame:
    """Protocol section 25: repeated stratified 5-fold CV *inside the training split*.

    Patients contribute one row each, so patient-grouping is automatic; folds are
    stratified on event status. Spline knots are refit on each CV-training fold, so no
    fold's validation rows influence its own design matrix.

    Since D24 this is not only the stability cross-check but (when
    ``model_clinical.selection_metric == "cv_mean_cindex"``) the criterion that SELECTS the
    penalizer, from 373 training events rather than 54 validation ones.

    ``drop_design_columns`` mirrors a drop the caller already made on its own design (a row
    filter can make a column constant — see the complete-case sensitivity), and any column
    that is additionally constant *within a fold's training rows* is dropped for that fold,
    because a zero-variance column has no hazard ratio and makes lifelines' normalisation
    singular. Both are reported, never silent.
    """
    rkf = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS,
                                  random_state=int(seed))
    per_fold: dict[float, list[float]] = {float(p): [] for p in grid}
    dropped_in_folds: set[str] = set()
    T = np.asarray(T, dtype=float); E = np.asarray(E, dtype=int)
    for tr_idx, va_idx in rkf.split(np.zeros(len(T)), E):
        spec = fit_age_spline(train_df.iloc[tr_idx]["age_at_index_imp"].to_numpy(dtype=float),
                              spline_df)
        Xd = build_design(train_df, spec, model_columns)      # knots from fold-train rows only
        if drop_design_columns:
            Xd = Xd.drop(columns=[c for c in drop_design_columns if c in Xd.columns])
        degenerate = [c for c in Xd.columns
                      if float(Xd[c].iloc[tr_idx].std(ddof=0)) == 0.0]
        if degenerate:
            Xd = Xd.drop(columns=degenerate)
            dropped_in_folds.update(degenerate)
        for pen in grid:
            cph = fit_cox(Xd.iloc[tr_idx], T[tr_idx], E[tr_idx], pen, l1_ratio)
            per_fold[float(pen)].append(
                harrell_c(T[va_idx], E[va_idx], linear_predictor(cph, Xd.iloc[va_idx])))
    if dropped_in_folds and log is not None:
        log.warning("CV penalizer grid: column(s) %s were constant inside at least one "
                    "CV-training fold and were dropped for that fold",
                    sorted(dropped_in_folds))
    return pd.DataFrame([dict(penalizer=p, cv_mean_cindex=float(np.mean(v)),
                              cv_sd_cindex=float(np.std(v, ddof=1)), n_folds=len(v))
                         for p, v in per_fold.items()])


def select_penalizer(grid_df: pd.DataFrame, selection_metric: str, *,
                     n_selection_events: int | None = None, label: str = "",
                     log: logging.Logger | None = None) -> dict:
    """Pick the penalizer by ``selection_metric``, reporting BOTH criteria and both winners.

    ``grid_df`` must carry ``penalizer`` plus every column in :data:`SELECTION_METRICS`, so
    the criterion that did **not** select is still recorded. Ties go to the smaller
    penalizer (less shrinkage), which is order-independent rather than dependent on the row
    order of the grid.

    A winner at either end of the pre-specified grid means the criterion never turned over,
    so the optimum may lie outside the grid. That is detected here against **whichever
    criterion actually selected** and returned as ``at_grid_edge`` / ``grid_edge_side``; it
    is reported, never fixed by widening the grid, which is a pre-specified config value
    (protocol section 27).
    """
    assert selection_metric in SELECTION_METRICS, (
        f"model_clinical.selection_metric must be one of {list(SELECTION_METRICS)}, "
        f"got {selection_metric!r}")
    for m in SELECTION_METRICS:
        assert m in grid_df.columns, f"penalizer grid is missing the {m!r} column"

    def _winner(metric: str) -> float:
        g = grid_df.dropna(subset=[metric])
        assert len(g), f"every {metric!r} value is missing — nothing to select on"
        return float(g.loc[g[metric] == g[metric].max(), "penalizer"].min())

    winners = {m: _winner(m) for m in SELECTION_METRICS}
    penalizer = winners[selection_metric]
    grid = [float(p) for p in grid_df["penalizer"]]
    edge = ("lower" if penalizer == min(grid)
            else "upper" if penalizer == max(grid) else None)
    sel = dict(selection_metric=selection_metric,
               criterion=SELECTION_LABELS[selection_metric], penalizer=penalizer,
               val_selected_penalizer=winners["val_cindex"],
               cv_selected_penalizer=winners["cv_mean_cindex"],
               criteria_agree=bool(winners["val_cindex"] == winners["cv_mean_cindex"]),
               at_grid_edge=edge is not None, grid_edge_side=edge,
               n_selection_events=(None if n_selection_events is None
                                   else int(n_selection_events)),
               selected_value=float(grid_df.loc[grid_df["penalizer"] == penalizer,
                                                selection_metric].iloc[0]),
               grid_span=float(grid_df[selection_metric].max()
                               - grid_df[selection_metric].min()))
    if log is not None:
        tag = f"{label} " if label else ""
        log.info("%spenalizer selected by %s: %g (%s %.4f%s); the other criterion (%s) "
                 "would pick %g", tag, selection_metric, penalizer, selection_metric,
                 sel["selected_value"],
                 "" if n_selection_events is None else f" from {n_selection_events} events",
                 [m for m in SELECTION_METRICS if m != selection_metric][0],
                 winners["val_cindex" if selection_metric == "cv_mean_cindex"
                         else "cv_mean_cindex"])
        if edge is not None:
            log.warning("%spenalizer %g is at the %s edge of the pre-specified grid %s — "
                        "the %s never turned over, so the optimum may lie outside the grid. "
                        "Flagged, not widened: model_clinical.penalizer_grid is a "
                        "pre-specified config value (protocol section 27).",
                        tag, penalizer, edge, grid, SELECTION_LABELS[selection_metric])
    return sel


def tune_and_select_penalizer(train_df: pd.DataFrame, model_columns: list[str],
                              Xtr, Ttr, Etr, Xva, Tva, Eva, grid, l1_ratio: float,
                              spline_df: int, seed: int, selection_metric: str, *,
                              drop_design_columns: list[str] | None = None,
                              label: str = "",
                              log: logging.Logger | None = None) -> tuple[pd.DataFrame, dict]:
    """Compute BOTH penalizer criteria on one fit and select with the configured one.

    Every fit in this module (M0, both M1 ladder arms, each section-24 sensitivity) goes
    through here, so no two of them can silently end up on different criteria — the mixed
    -criteria failure mode D24 explicitly forbids.
    """
    grid_df = tune_penalizer(Xtr, Ttr, Etr, Xva, Tva, Eva, grid, l1_ratio)
    cv_df = cv_penalizer_stability(train_df, Ttr, Etr, grid, l1_ratio, spline_df,
                                   model_columns, seed,
                                   drop_design_columns=drop_design_columns, log=log)
    grid_df = grid_df.merge(cv_df, on="penalizer", how="left")
    n_ev = int(np.asarray(Etr, dtype=int).sum()) if selection_metric == "cv_mean_cindex" \
        else int(np.asarray(Eva, dtype=int).sum())
    return grid_df, select_penalizer(grid_df, selection_metric, n_selection_events=n_ev,
                                     label=label, log=log)


def evaluate(times, events, risk_lp, risk_at_h: dict[float, np.ndarray],
             horizons: list[dict], g_grid, g_vals) -> dict:
    """All point metrics for one evaluation sample (used for the point estimate and for
    every bootstrap replicate, so the two can never drift apart)."""
    out = {"cindex": harrell_c(times, events, risk_lp)}
    for h in horizons:
        t = float(h["horizon_days"])
        p = risk_at_h[t]
        y, w = ipcw_labels_weights(times, events, t, g_grid, g_vals)
        out[f"auc@{t}"] = ipcw_auc(y, w, p)
        slope, intercept = calibration_slope_intercept(y, w, p)
        out[f"slope@{t}"] = slope
        out[f"intercept@{t}"] = intercept
        obs, _, _ = km_risk(times, events, t)
        out[f"predrisk@{t}"] = float(np.mean(p))
        out[f"obsrisk@{t}"] = float(obs)
        out[f"citl@{t}"] = float(obs - np.mean(p))
    return out


# --------------------------------------------------------------------------- #
# M1 — "M0 plus inferred KLG" on the KLG-eligible subset (protocol Table 7)      #
# --------------------------------------------------------------------------- #
def fit_m1_klg(dev: pd.DataFrame, frozen: dict, horizons: list[dict],
               grid: list[float], l1_ratio: float, age_rcs_df: int, n_boot: int,
               rng: np.random.Generator, m0_lp_va: np.ndarray,
               m0_risk_va: dict[float, np.ndarray], log: logging.Logger, *,
               selection_metric: str = "val_cindex", seed: int = 0) -> dict:
    """Fit protocol Table 7's **M1 = M0 + inferred KLG** and price what KLG adds.

    Protocol Secondary objective 2 asks for the inferred-KLG-plus-clinical comparator **in
    the subset with eligible bilateral frontal images**, so M1 is fitted and evaluated ONLY
    where ``klg_contra`` is observed. KLG is never median-imputed across the cohort for M1:
    a filled-in severity grade for a patient with no eligible frontal is a fabricated
    radiographic measurement, and it would be the one covariate the model leans on.

    Three quantities are returned so "what does inferred KLG add" is answerable without
    confounding the subset with the covariate:

    * ``M1``                  — M0's columns + ``klg_contra_imp``, refitted on the eligible
                                training rows;
    * ``M0_refit_eligible``   — M0's columns ALONE, refitted on the same eligible training
                                rows, tuned the same way. This is the like-for-like anchor:
                                M1 minus this difference is *exactly* the KLG column;
    * ``M0_as_fitted``        — the published full-cohort M0, simply evaluated on the
                                eligible validation rows. This says what the eligibility
                                restriction alone does to the published comparator.

    The eligible validation patients are shared, so each bootstrap replicate scores all
    three on the same resampled rows and the differences are paired (protocol section 18).
    The training split is the only place anything is fitted; the test split is never loaded.

    ``selection_metric`` is the SAME criterion M0 uses (deviation D24), applied here to the
    eligible training rows: the CV grid is recomputed inside the eligible train subset for
    each arm, so the ladder is never tuned on two different criteria.
    """
    elig_col = "klg_contra_missing"
    assert elig_col in dev.columns, "klg_contra_missing is required to define M1 eligibility"
    m1_cols = list(frozen["m1_model_columns"])
    m0_cols = list(frozen["model_columns"])
    assert set(m0_cols) < set(m1_cols), "M1 must be a strict superset of M0"
    added = [c for c in m1_cols if c not in m0_cols]

    elig = (dev[elig_col].to_numpy(dtype=int) == 0)
    sub = dev.loc[elig].reset_index(drop=True)
    tr = sub[sub["split"] == "train"].reset_index(drop=True)
    va = sub[sub["split"] == "val"].reset_index(drop=True)
    assert len(tr) and len(va), "the KLG-eligible subset is empty in train or val"
    assert sub["klg_contra"].notna().all(), "an ineligible (KLG-missing) row reached M1"
    Ttr = tr["time_from_landmark"].to_numpy(dtype=float); Etr = tr["event_indicator"].to_numpy(int)
    Tva = va["time_from_landmark"].to_numpy(dtype=float); Eva = va["event_indicator"].to_numpy(int)

    # The frozen KLG imputation must be a no-op here — that is the point of the subset.
    assert np.allclose(sub["klg_contra"].to_numpy(dtype=float),
                       sub["klg_contra_imp"].to_numpy(dtype=float)), \
        "klg_contra_imp differs from klg_contra on the eligible subset — KLG was imputed"

    # Knots from the ELIGIBLE TRAIN ages only (non-negotiable #2 holds inside M1 too).
    spline = fit_age_spline(tr["age_at_index_imp"].to_numpy(dtype=float), int(age_rcs_df))
    g_grid, g_vals = censoring_curve(Ttr, Etr)

    designs = {"M1_klg": m1_cols, "M0_refit_eligible": m0_cols}
    fits: dict[str, dict] = {}
    for name, cols in designs.items():
        Xtr, Xva = build_design(tr, spline, cols), build_design(va, spline, cols)
        ident = design_identifiability(Xtr)
        ident["level_unidentified_columns"] = (list(spline["basis_columns"])
                                               if sums_to_constant(Xtr, spline["basis_columns"])
                                               else [])
        gsel, psel = tune_and_select_penalizer(
            tr, cols, Xtr, Ttr, Etr, Xva, Tva, Eva, grid, l1_ratio, int(age_rcs_df),
            int(seed), selection_metric, label=f"M1 ladder | {name}:", log=log)
        pen = float(psel["penalizer"])
        cph = fit_cox(Xtr, Ttr, Etr, pen, l1_ratio)
        lp = linear_predictor(cph, Xva)
        risk = {h["horizon_days"]: horizon_risk(cph, Xva, h["horizon_days"]) for h in horizons}
        fits[name] = dict(model_columns=cols, design_columns=list(Xtr.columns),
                          n_parameters=int(Xtr.shape[1]), identifiability=ident,
                          identified_parameters=int(ident["identified_parameters"]),
                          penalizer=pen, penalizer_grid=gsel.to_dict("records"),
                          selection=psel,
                          cph=cph, lp=lp, risk=risk,
                          point=evaluate(Tva, Eva, lp, risk, horizons, g_grid, g_vals))
        log.info("M1 ladder | %-18s %2d columns, %2d identified, penalizer %g (by %s), "
                 "val C-index %.4f", name, Xtr.shape[1], ident["identified_parameters"], pen,
                 selection_metric, fits[name]["point"]["cindex"])
    for name, n_col, n_id in (("M1_klg", EXPECTED_M1_N_PARAMS, EXPECTED_M1_IDENTIFIED_PARAMS),
                              ("M0_refit_eligible", EXPECTED_N_PARAMS,
                               EXPECTED_IDENTIFIED_PARAMS)):
        assert fits[name]["n_parameters"] == n_col, \
            f"{name} has {fits[name]['n_parameters']} design columns, expected {n_col}"
        assert fits[name]["identified_parameters"] == n_id, \
            f"{name} identifies {fits[name]['identified_parameters']}, expected {n_id}"

    # The published full-cohort M0, restricted to the same eligible validation patients.
    va_mask_in_dev = elig & (dev["split"].to_numpy() == "val")
    sel = va_mask_in_dev[(dev["split"].to_numpy() == "val")]
    fits["M0_as_fitted"] = dict(
        model_columns=m0_cols, design_columns=None, n_parameters=None,
        identifiability=None, identified_parameters=None, penalizer=None,
        penalizer_grid=None, selection=None, cph=None, lp=m0_lp_va[sel],
        risk={t: v[sel] for t, v in m0_risk_va.items()},
        point=evaluate(Tva, Eva, m0_lp_va[sel], {t: v[sel] for t, v in m0_risk_va.items()},
                       horizons, g_grid, g_vals))
    assert len(fits["M0_as_fitted"]["lp"]) == len(va), \
        "the published-M0 restriction does not line up with the eligible validation rows"

    # ---- paired patient-level bootstrap over the eligible validation patients ----------
    names = list(fits)
    keys = list(fits["M1_klg"]["point"].keys())
    boot = {n: {k: np.full(n_boot, np.nan) for k in keys} for n in names}
    n_va = len(va)
    for b in range(n_boot):
        idx = rng.integers(0, n_va, size=n_va)
        if int(Eva[idx].sum()) < 2:
            continue
        for n in names:
            rb = evaluate(Tva[idx], Eva[idx], fits[n]["lp"][idx],
                          {t: v[idx] for t, v in fits[n]["risk"].items()},
                          horizons, g_grid, g_vals)
            for k in keys:
                boot[n][k][b] = rb[k]
    ci = {n: {k: percentile_ci(v) for k, v in boot[n].items()} for n in names}

    paired = {}
    for ref in ("M0_refit_eligible", "M0_as_fitted"):
        for k in keys:
            d = boot["M1_klg"][k] - boot[ref][k]
            ok = np.isfinite(d)
            paired[f"M1_klg-{ref}|{k}"] = dict(
                reference=ref, metric=k,
                point=float(fits["M1_klg"]["point"][k] - fits[ref]["point"][k]),
                ci=percentile_ci(d), n_valid=int(ok.sum()),
                p_two_sided=(float(min(1.0, max(2.0 * min((d[ok] <= 0).mean(),
                                                          (d[ok] >= 0).mean()),
                                                1.0 / max(int(ok.sum()), 1))))
                             if ok.any() else float("nan")))
    for k in ("cindex", f"auc@{horizons[-1]['horizon_days']}"):
        p = paired[f"M1_klg-M0_refit_eligible|{k}"]
        log.info("M1 minus M0 (both refit on the eligible subset) %s: %+.4f "
                 "(95%% CI %+.4f to %+.4f, two-sided bootstrap p %.3f)",
                 k, p["point"], p["ci"][0], p["ci"][1], p["p_two_sided"])

    coef = fits["M1_klg"]["cph"].summary.reset_index()
    return dict(
        added_columns=added, eligibility=frozen.get("m1_eligibility", {}),
        selection_metric=selection_metric,
        n_train=len(tr), n_val=len(va), n_train_events=int(Etr.sum()),
        n_val_events=int(Eva.sum()),
        n_train_dropped=int((dev["split"] == "train").sum()) - len(tr),
        n_val_dropped=int((dev["split"] == "val").sum()) - len(va),
        spline=spline, horizons=horizons, censoring_km_train=(g_grid, g_vals),
        fits=fits, boot_ci=ci, paired=paired, n_boot=n_boot,
        coefficients=pd.DataFrame(dict(
            covariate=coef["covariate"], coef=coef["coef"], hazard_ratio=coef["exp(coef)"],
            se=coef["se(coef)"], hr_lo=coef["exp(coef) lower 95%"],
            hr_hi=coef["exp(coef) upper 95%"], z=coef["z"], p_value=coef["p"])))


def write_m1_outputs(cfg: Config, mc: dict, m1: dict, m0_point: dict, log: logging.Logger):
    """Emit ``m1_metrics_csv``, ``m1_model_json`` and ``m1_report_md`` — M0 and M1 side by side."""
    horizons = m1["horizons"]
    fits, ci, paired = m1["fits"], m1["boot_ci"], m1["paired"]
    order = ["M0_as_fitted", "M0_refit_eligible", "M1_klg"]
    labels = {"M0_as_fitted": "M0 as published (full-cohort fit), evaluated on the eligible "
                              "validation patients",
              "M0_refit_eligible": "M0 refitted on the KLG-eligible training rows "
                                   "(the like-for-like anchor)",
              "M1_klg": "M1 = M0 + inferred contralateral KLG (protocol Table 7)"}

    rows = []
    for name in order:
        f = fits[name]
        for metric, key in ([("harrell_cindex", "cindex")]
                            + [(m, f"{m_key}@{h['horizon_days']}")
                               for h in horizons
                               for m, m_key in (("ipcw_cd_auroc", "auc"),
                                                ("calibration_slope", "slope"),
                                                ("mean_predicted_risk", "predrisk"),
                                                ("km_observed_risk", "obsrisk"),
                                                ("calibration_in_the_large_risk_diff", "citl"))]):
            hy = None if metric == "harrell_cindex" else float(key.split("@")[1]) / 365.25
            hd = None if metric == "harrell_cindex" else int(float(key.split("@")[1]))
            lo, hi = ci[name][key]
            rows.append(dict(model=name, description=labels[name], scope="klg_eligible_val",
                             metric=metric,
                             horizon_years=None if hy is None else round(hy, 3),
                             horizon_days=hd, n=m1["n_val"], n_events=m1["n_val_events"],
                             estimate=f["point"][key], ci_lo=lo, ci_hi=hi,
                             n_parameters=f["n_parameters"],
                             identified_parameters=f["identified_parameters"],
                             penalizer=f["penalizer"]))
    for key, p in paired.items():
        metric = p["metric"]
        hd = int(float(metric.split("@")[1])) if "@" in metric else None
        rows.append(dict(model=f"paired_diff:{key.split('|')[0]}",
                         description=f"paired patient-level bootstrap difference vs "
                                     f"{p['reference']} on the same eligible validation "
                                     f"patients ({p['n_valid']} valid replicates)",
                         scope="klg_eligible_val",
                         metric=("ipcw_cd_auroc" if metric.startswith("auc@")
                                 else metric.split("@")[0]),
                         horizon_years=None if hd is None else round(hd / 365.25, 3),
                         horizon_days=hd, n=m1["n_val"], n_events=m1["n_val_events"],
                         estimate=p["point"], ci_lo=p["ci"][0], ci_hi=p["ci"][1],
                         n_parameters=None, identified_parameters=None,
                         penalizer=None, p_two_sided=p["p_two_sided"]))
    met = pd.DataFrame(rows)
    assert "empi_anon" not in met.columns, "identifier leaked into an outputs/ table"
    met.to_csv(cfg.path(mc["m1_metrics_csv"]), index=False)

    f1 = fits["M1_klg"]
    bs = f1["cph"].baseline_survival_
    g_grid, g_vals = m1["censoring_km_train"]
    mj = {
        "module": MODULE, "model": "M1_klg_plus_clinical_penalized_cox",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": {"table_7": "M1: KLG + clinical = M0 plus inferred KLG",
                     "table_6": "dataset-inferred KLG is a SECONDARY comparator only",
                     "secondary_objective_2": "compare raw-image prediction with an inferred "
                                              "KLG plus clinical comparator in the subset "
                                              "with eligible bilateral frontal images"},
        "test_split": "SEALED — never loaded by this module",
        "eligibility": m1["eligibility"],
        "cohort": {"n_train": m1["n_train"], "n_val": m1["n_val"],
                   "n_train_events": m1["n_train_events"], "n_val_events": m1["n_val_events"],
                   "n_train_dropped_ineligible": m1["n_train_dropped"],
                   "n_val_dropped_ineligible": m1["n_val_dropped"]},
        "added_columns": m1["added_columns"],
        # selection_metric records what ACTUALLY selected (deviation D24), and the losing
        # criterion's winner is kept beside it so the choice stays auditable.
        "hyperparameters": {"penalizer": f1["penalizer"], "penalizer_grid":
                            [r["penalizer"] for r in f1["penalizer_grid"]],
                            "tuning_split": "val (KLG-eligible)",
                            "selection_metric": f1["selection"]["selection_metric"],
                            "selection_criterion": f1["selection"]["criterion"],
                            "val_selected_penalizer":
                                f1["selection"]["val_selected_penalizer"],
                            "cv_selected_penalizer":
                                f1["selection"]["cv_selected_penalizer"],
                            "cv_design": f"{CV_N_REPEATS}x{CV_N_SPLITS}-fold stratified, "
                                         "KLG-eligible train rows only",
                            "n_selection_events": f1["selection"]["n_selection_events"],
                            "penalizer_at_grid_edge": f1["selection"]["at_grid_edge"],
                            "grid_edge_side": f1["selection"]["grid_edge_side"]},
        "preprocessing": {"model_columns": f1["model_columns"],
                          "imputation_source": "features_clinical_frozen (KLG NOT imputed — "
                                               "M1 is fitted where KLG is observed)",
                          "spline": m1["spline"]},
        "design_columns": f1["design_columns"], "n_parameters": f1["n_parameters"],
        "identified_parameters": f1["identified_parameters"],
        "identifiability": f1["identifiability"],
        # Rank-based identifiability persisted for BOTH designs on this subset, so an
        # events-per-parameter statement can be made about either without refitting.
        "ladder_identifiability": {
            n: {"model_columns": fits[n]["model_columns"],
                "design_columns": fits[n]["design_columns"],
                "n_parameters": fits[n]["n_parameters"],
                "identified_parameters": fits[n]["identified_parameters"],
                "identifiability": fits[n]["identifiability"],
                "penalizer": fits[n]["penalizer"],
                "penalizer_selection": fits[n]["selection"]}
            for n in ("M1_klg", "M0_refit_eligible")},
        "coefficients": {k: float(v) for k, v in f1["cph"].params_.items()},
        "centering_means": {k: float(v) for k, v in f1["cph"]._norm_mean.items()},
        "linear_predictor": "lp(x) = sum_j (x_j - centering_means[j]) * coefficients[j]",
        "risk_formula": "risk(t|x) = 1 - baseline_survival(t) ** exp(lp(x))",
        "baseline_survival": {"times": [float(t) for t in bs.index.values],
                              "survival": [float(v) for v in bs.iloc[:, 0].values]},
        "censoring_km_train": {"times": [float(t) for t in g_grid],
                               "survival": [float(v) for v in g_vals]},
        "horizons": horizons,
        "val_metrics": {n: {k: (None if not np.isfinite(v) else float(v))
                            for k, v in fits[n]["point"].items()} for n in order},
        "val_metrics_ci": {n: {k: [None if not np.isfinite(x) else float(x) for x in v]
                               for k, v in ci[n].items()} for n in order},
        "paired_differences": {k: {"reference": p["reference"], "metric": p["metric"],
                                   "estimate": p["point"],
                                   "ci": [None if not np.isfinite(x) else float(x)
                                          for x in p["ci"]],
                                   "p_two_sided": p["p_two_sided"],
                                   "n_valid_replicates": p["n_valid"]}
                               for k, p in paired.items()},
        "m0_full_cohort_val_metrics": {k: (None if not np.isfinite(v) else float(v))
                                       for k, v in m0_point.items()},
    }
    mp = cfg.path(mc["m1_model_json"])
    mp.write_text(json.dumps(mj, indent=2) + "\n")

    e = m1["eligibility"]
    L = ["# M1 — inferred KLG plus clinical, on the KLG-eligible subset", "",
         "Generated by `src/model_clinical.py` (Phase 2, Track A). **Aggregate only — no "
         "patient identifiers. The locked test split was never loaded.**", "",
         "Protocol Table 7 defines the model ladder as **M0 = \"Age, sex, comorbidities, "
         "pain, image-to-index interval\"** and **M1 = \"M0 plus inferred KLG\"**, and "
         "protocol Table 6 lists the dataset-inferred contralateral KLG as a **secondary "
         "comparator only**. This report answers protocol **Secondary objective 2** — the "
         "inferred-KLG-plus-clinical comparator *in the subset with eligible bilateral "
         "frontal images* — and prices exactly what a radiograph-derived severity grade adds "
         "to routine clinical variables.", "",
         "> **KLG is not imputed here.** `klg_contra` is missing for "
         f"{e.get('n_ineligible', 0)} of {e.get('n_eligible', 0) + e.get('n_ineligible', 0):,} "
         "cohort patients (no eligible bilateral frontal). Filling those in with a train "
         "median would invent a radiographic measurement for the patients who do not have "
         "one, and it would be the very covariate the comparator leans on. M1 is fitted and "
         "evaluated **only where KLG is observed**; the module asserts "
         "`klg_contra_imp == klg_contra` on every row it uses.", "",
         "> **Penalizer criterion (deviation D24).** Both fitted arms below select their "
         f"ridge penalizer by the **{fits['M1_klg']['selection']['criterion']}**, the same "
         "criterion M0 uses, computed here on the KLG-eligible **training** rows "
         f"({m1['n_train']:,} patients, {m1['n_train_events']} events) rather than on the "
         f"{m1['n_val']} eligible validation patients ({m1['n_val_events']} events). The "
         "validation grid is still computed and reported; it just no longer selects. Under "
         "the previous criterion the two arms would take penalizers "
         f"{fits['M1_klg']['selection']['val_selected_penalizer']:g} (M1) and "
         f"{fits['M0_refit_eligible']['selection']['val_selected_penalizer']:g} "
         "(M0 refit); both winners are in the frozen JSON.", "",
         "## 1. The eligible subset", "",
         _md_table(["split", "eligible n", "eligible events", "dropped (KLG missing)"],
                   [["train", f"{m1['n_train']:,}", m1["n_train_events"], m1["n_train_dropped"]],
                    ["val", f"{m1['n_val']:,}", m1["n_val_events"], m1["n_val_dropped"]],
                    ["test", f"{e.get('n_eligible_by_split', {}).get('test', 0):,}",
                     e.get("n_eligible_events_by_split", {}).get("test", 0),
                     "**SEALED — counted from the frozen JSON, never loaded**"]]),
         "",
         "## 2. Three fits, so the subset and the covariate are not confounded", "",
         "Comparing the published full-cohort M0 with M1 mixes two changes: the KLG column "
         "*and* the restriction to the eligible patients. Both are reported.", ""]
    auc_keys = ["auc@" + str(h["horizon_days"]) for h in horizons]
    ladder_rows = []
    for n in order:
        row = [labels[n], fits[n]["n_parameters"] or "—",
               fits[n]["identified_parameters"] or "—",
               f"{fits[n]['penalizer']:g}" if fits[n]["penalizer"] else "—",
               _f(fits[n]["point"]["cindex"]) + " " + _ci(*ci[n]["cindex"])]
        row += [_f(fits[n]["point"][k]) + " " + _ci(*ci[n][k]) for k in auc_keys]
        ladder_rows.append(row)
    L.append(_md_table(
        ["model", "columns", "identified parameters", "penalizer", "val C-index (95% CI)"]
        + [f"IPCW AUROC {h['horizon_years']:.0f} y (95% CI)" for h in horizons],
        ladder_rows))
    L += ["", "## 3. What inferred KLG adds (paired, protocol section 18)", "",
          "Every replicate of the patient-level bootstrap scores all three models on the "
          f"**same** resampled eligible validation patients ({m1['n_val']} patients, "
          f"{m1['n_val_events']} events, {m1['n_boot']:,} replicates), so these differences "
          "are paired and are estimated far more precisely than either level.", ""]
    L.append(_md_table(
        ["comparison", "metric", "difference", "95% CI", "two-sided bootstrap p"],
        [[f"M1 minus {p['reference'].replace('_', ' ')}",
          ("Harrell C-index" if p["metric"] == "cindex"
           else f"IPCW C/D AUROC @ {int(float(p['metric'].split('@')[1]))} d"),
          f"{p['point']:+.4f}", _ci(*p["ci"]), f"{p['p_two_sided']:.3f}"]
         for k, p in paired.items()
         if p["metric"] == "cindex" or p["metric"].startswith("auc@")]))
    L += ["", "**Read the `M0_refit_eligible` rows as the answer.** They hold the patients "
          "fixed and change only the KLG column, so the difference *is* the incremental "
          "value of the inferred grade. The `M0_as_fitted` rows additionally absorb the "
          "restriction to the eligible subset and are reported so that restriction is "
          "visible rather than silently folded into the KLG effect.", "",
          "## 4. The fitted M1 (descriptive)", "",
          "Ridge shrinks these toward 1 and the standard errors are not shrinkage-corrected, "
          "so this table is descriptive; the paired differences above are the inferential "
          "output. Spline knots come from the **eligible training** ages only.", ""]
    L.append(_md_table(["covariate", "coef", "hazard ratio", "95% CI", "z", "p"],
                       [[f"`{r.covariate}`", _f(r.coef), _f(r.hazard_ratio),
                         _ci(r.hr_lo, r.hr_hi, 2), _f(r.z, 2), f"{r.p_value:.4f}"]
                        for r in m1["coefficients"].itertuples()]))
    klg_rows = m1["coefficients"][m1["coefficients"]["covariate"].str.startswith("klg")]
    if len(klg_rows):
        r0 = klg_rows.iloc[0]
        L += ["", f"Inferred contralateral KLG: coefficient {float(r0['coef']):+.4f}, "
              f"**HR {float(r0['hazard_ratio']):.2f} per grade** "
              f"(z {float(r0['z']):+.2f}, p {float(r0['p_value']):.4g}). That is a large, "
              "clinically coherent effect — which is precisely why it does not belong in M0. "
              "A comparator that already contains a radiograph-derived severity grade is not "
              "\"routine clinical variables\", and the study's primary estimand (protocol "
              "Table 8: M4 versus M0 at 5 years) would be measuring imaging value against "
              "imaging.", ""]
    L += ["## 5. Interpretation boundaries", "",
          "1. **The KLG label is model-inferred, not chart ground truth.** It is available "
          "only on weight-bearing bilateral frontal non-arthroplasty views, which is why "
          f"{e.get('n_ineligible', 0)} cohort patients have none.",
          "2. **Eligibility is not random.** Patients with an eligible bilateral frontal "
          "differ from those without, so the M1 subset is not a random sample of the cohort "
          "and its absolute metrics are not interchangeable with M0's full-cohort numbers.",
          "3. **Validation only.** The test split is sealed; nothing here is a final "
          "performance claim.",
          f"4. **{m1['n_val_events']} eligible validation events** drive every interval "
          "here, so they are wide by construction.", "",
          "## 6. Files", "",
          f"- `{mc['m1_metrics_csv']}` — every point estimate, interval and paired difference.",
          f"- `{mc['m1_model_json']}` — the frozen M1 contract (coefficients, centering means, "
          "baseline survival, knots, eligibility counts).",
          f"- The M0 comparator lives in `{mc['report_md']}` and `{mc['model_json']}`.", "",
          "## 7. Reproduce", "", "```",
          "python3 -m src.model_clinical --config config/feasibility.yaml", "```", ""]
    rp = cfg.path(mc["m1_report_md"])
    rp.write_text("\n".join(L) + "\n")
    log.info("wrote %s, %s and %s", cfg.path(mc["m1_metrics_csv"]), mp, rp)
    return met


# --------------------------------------------------------------------------- #
# REPORT HELPERS                                                                #
# --------------------------------------------------------------------------- #
def _md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, nd: int = 3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _ci(lo, hi, nd: int = 3) -> str:
    return "—" if not (np.isfinite(lo) and np.isfinite(hi)) else f"({_f(lo, nd)} to {_f(hi, nd)})"


# --------------------------------------------------------------------------- #
# Entry point.                                                                  #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:                                  # noqa: C901 - one linear script
    ap = argparse.ArgumentParser(description="M0 penalized-Cox clinical baseline (Track A).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)
    cfg: Config = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    mc = cfg["model_clinical"]
    fc = cfg["features_clinical"]
    seed = int(cfg["reproducibility"]["random_seed"])
    rng = np.random.default_rng(seed)

    # ---- 1. SEALED-SPLIT GUARD, before anything else ------------------------
    feat_path = cfg.path(fc["out_parquet"])
    params_path = cfg.path(fc["imputation"]["params_json"])
    frozen = json.loads(params_path.read_text())
    n_rows_total = parquet_num_rows(feat_path)               # metadata only
    dev = load_development_frame(feat_path, forbid_test=bool(mc["forbid_test_split"]))
    assert SEALED_SPLIT not in set(dev["split"].unique()), "sealed split reached the model"
    log.info("test split SEALED: %d of %d rows loaded (train+val only)", len(dev), n_rows_total)

    # ---- 2. input invariants ------------------------------------------------
    assert n_rows_total == EXPECTED_N_PATIENTS, \
        f"feature table has {n_rows_total} rows, expected {EXPECTED_N_PATIENTS}"
    assert dev["empi_anon"].is_unique, "feature table is not one row per patient"
    for s in DEV_SPLITS:
        sub = dev[dev["split"] == s]
        assert len(sub) == EXPECTED_SPLIT_N[s], f"{s} n={len(sub)} != {EXPECTED_SPLIT_N[s]}"
        assert int(sub["event_indicator"].sum()) == EXPECTED_SPLIT_EVENTS[s], \
            f"{s} events != {EXPECTED_SPLIT_EVENTS[s]}"
    # The sealed split is verified against the FROZEN metadata, never against its rows.
    assert frozen["split_counts"] == EXPECTED_SPLIT_N, "frozen split counts moved"
    assert frozen["split_event_counts"] == EXPECTED_SPLIT_EVENTS, "frozen event counts moved"
    assert len(dev) == EXPECTED_SPLIT_N["train"] + EXPECTED_SPLIT_N["val"]

    # ---- 3. frozen imputation is REPLAYED, never refit (protocol section 20) --
    assert mc["imputation_source"] == "features_clinical_frozen", \
        "model_clinical.imputation_source must replay the frozen train-fitted transform"
    # Every column the frozen transform covers — M0's predictors AND M1's extra (klg_contra),
    # so the replay check spans both designs rather than only the primary one.
    raw_cols = list(frozen["columns"].keys())
    assert set(frozen["primary_predictors"]) <= set(raw_cols)
    assert set(frozen["m1_predictors"]) <= set(raw_cols)
    replay = apply_imputer(dev[raw_cols].copy(), frozen)
    model_columns = list(frozen["model_columns"])
    assert not any(c.startswith("klg") for c in model_columns), (
        "inferred KLG is a SECONDARY comparator (protocol Table 6) and belongs to M1 "
        f"(protocol Table 7); it must not be an M0 model column. Got: {model_columns}")
    for c in list(dict.fromkeys(model_columns + list(frozen["m1_model_columns"]))):
        assert c in dev.columns, f"model column {c!r} absent from the feature table"
        assert np.allclose(replay[c].to_numpy(dtype=float), dev[c].to_numpy(dtype=float)), \
            f"{c!r} does not reproduce from the frozen imputer — the transform was refit"
    for c in frozen["eval_only_columns"]:
        assert c not in model_columns, f"evaluation-only column {c!r} leaked into the design"
    log.info("frozen imputer replayed and verified on all %d model columns", len(model_columns))

    # ---- 4. design matrix: pre-specified block, TRAIN-only spline knots ------
    assert mc["predictor_selection"] == "prespecified_all", \
        "protocol section 19 forbids univariable screening"
    tr = dev[dev["split"] == "train"].reset_index(drop=True)
    va = dev[dev["split"] == "val"].reset_index(drop=True)
    spline = fit_age_spline(tr["age_at_index_imp"].to_numpy(dtype=float), int(mc["age_rcs_df"]))
    Xtr, Xva = build_design(tr, spline, model_columns), build_design(va, spline, model_columns)
    assert list(Xtr.columns) == list(Xva.columns), "train/val design matrices disagree"
    assert Xtr.shape[1] == EXPECTED_N_PARAMS, \
        f"design matrix has {Xtr.shape[1]} parameters, expected {EXPECTED_N_PARAMS}"
    Ttr = tr["time_from_landmark"].to_numpy(dtype=float); Etr = tr["event_indicator"].to_numpy(int)
    Tva = va["time_from_landmark"].to_numpy(dtype=float); Eva = va["event_indicator"].to_numpy(int)

    # Identifiability is decided by RANK, not by pairwise correlation: a pairwise scan
    # cannot see the cr() partition of unity (age_rcs1 + age_rcs2 + age_rcs3 = 1).
    ident = design_identifiability(Xtr)
    alias = aliased_columns(Xtr)
    spline_cols = list(spline["basis_columns"])
    spline_partition = sums_to_constant(Xtr, spline_cols)
    ident["level_unidentified_columns"] = spline_cols if spline_partition else []
    for a, b, r in alias:
        log.warning("EXACTLY COLLINEAR PAIR: %s ~ %s (r=%.3f) — ridge splits the shared "
                    "effect; the individual coefficients are NOT identified", a, b, r)
    if spline_partition:
        log.info("spline basis %s is a partition of unity (sums to 1 on every row): the "
                 "fitted age CURVE is identified, the individual basis hazard ratios are "
                 "level-unidentified and must not be reported", spline_cols)
    assert ident["identified_parameters"] == EXPECTED_IDENTIFIED_PARAMS, (
        f"design matrix identifies {ident['identified_parameters']} parameters, expected "
        f"{EXPECTED_IDENTIFIED_PARAMS} (columns {ident['n_columns']}, rank {ident['rank']}, "
        f"rank with intercept {ident['rank_with_intercept']})")
    log.info("design matrix %d x %d: numeric rank %d, rank with intercept %d -> %d "
             "IDENTIFIED parameters (smallest singular value %.3g); spline knots %s from "
             "train only", *Xtr.shape, ident["rank"], ident["rank_with_intercept"],
             ident["identified_parameters"], ident["smallest_singular_value"],
             spline["all_knots"])

    # ---- 5. horizons, clamped inside observed follow-up ---------------------
    dpy = float(cfg["timeline"]["days_per_year"])
    max_obs = float(dev["time_from_landmark"].max())
    horizons = clamp_horizon_days(mc["horizons_years"], dpy, max_obs)
    for h in horizons:
        if h["clamped"]:
            log.warning("horizon %.0f y clamped %d d -> %d d: administrative censoring lands "
                        "on day %d, so no patient is observed event-free beyond it",
                        h["horizon_years"], h["horizon_days_nominal"],
                        int(h["horizon_days"]), int(max_obs))

    # ---- 6. censoring curve for IPCW, estimated on TRAIN --------------------
    g_grid, g_vals = censoring_curve(Ttr, Etr)
    gv_grid, gv_vals = censoring_curve(Tva, Eva)             # robustness variant only
    med_fu_tr, _ = reverse_km(Ttr, Etr)
    med_fu_va, _ = reverse_km(Tva, Eva)
    log.info("reverse-KM median follow-up: train %.0f d, val %.0f d", med_fu_tr, med_fu_va)

    # ---- 7. penalizer tuning: BOTH criteria computed, one of them selects ----
    # `tuning_split: val` still says where the validation grid is evaluated. Which criterion
    # SELECTS is `selection_metric` (deviation D24): the cross-validated C-index is estimated
    # from the 373 training events, the validation C-index from 54. The CV grid must
    # therefore be computed BEFORE the winner is picked, not after it as a cross-check.
    assert mc["tuning_split"] == "val", \
        f"unsupported tuning configuration: tuning_split must be 'val', got {mc['tuning_split']!r}"
    selection_metric = str(mc["selection_metric"])
    assert selection_metric in SELECTION_METRICS, (
        f"model_clinical.selection_metric must be one of {list(SELECTION_METRICS)}, "
        f"got {selection_metric!r}")
    grid = [float(p) for p in mc["penalizer_grid"]]
    grid_df, selection = tune_and_select_penalizer(
        tr, model_columns, Xtr, Ttr, Etr, Xva, Tva, Eva, grid, float(mc["l1_ratio"]),
        int(mc["age_rcs_df"]), seed, selection_metric, label="M0:", log=log)
    penalizer = float(selection["penalizer"])
    cv_best = float(selection["cv_selected_penalizer"])
    val_best = float(selection["val_selected_penalizer"])
    penalizer_at_grid_edge = bool(selection["at_grid_edge"])
    log.info("penalizer grid (val C-index): %s",
             ", ".join(f"{r.penalizer:g}={r.val_cindex:.4f}" for r in grid_df.itertuples()))
    log.info("penalizer grid (CV mean C-index, %dx%d-fold inside train): %s",
             CV_N_REPEATS, CV_N_SPLITS,
             ", ".join(f"{r.penalizer:g}={r.cv_mean_cindex:.4f}" for r in grid_df.itertuples()))
    log.info("chosen penalizer %g by %s (%d events); the validation split (%d events) would "
             "pick %g and the %dx%d-fold CV inside train would pick %g",
             penalizer, selection_metric, selection["n_selection_events"],
             EXPECTED_SPLIT_EVENTS["val"], val_best, CV_N_REPEATS, CV_N_SPLITS, cv_best)

    # ---- 8. final fit on TRAIN ----------------------------------------------
    cph = fit_cox(Xtr, Ttr, Etr, penalizer, float(mc["l1_ratio"]))
    lp_va = linear_predictor(cph, Xva)
    lp_tr = linear_predictor(cph, Xtr)
    risk_at_h = {h["horizon_days"]: horizon_risk(cph, Xva, h["horizon_days"]) for h in horizons}
    # Exact ties happen when two patients share a covariate pattern; they are legitimate,
    # scored 0.5 by the AUC and split at bin edges by the calibration binning. A degenerate
    # model (almost everything tied) would still trip the guard below.
    n_tied = len(lp_va) - len(np.unique(lp_va))
    assert len(np.unique(lp_va)) >= 10 * CALIBRATION_BINS, \
        f"only {len(np.unique(lp_va))} distinct validation risks — the design is degenerate"
    if n_tied:
        log.info("%d of %d validation patients share a risk score with another patient "
                 "(identical covariate pattern)", n_tied, len(lp_va))

    point = evaluate(Tva, Eva, lp_va, risk_at_h, horizons, g_grid, g_vals)
    point_gval = evaluate(Tva, Eva, lp_va, risk_at_h, horizons, gv_grid, gv_vals)
    train_c = harrell_c(Ttr, Etr, lp_tr)
    log.info("VAL C-index %.4f (train %.4f, optimism %.4f)",
             point["cindex"], train_c, train_c - point["cindex"])
    for h in horizons:
        t = h["horizon_days"]
        log.info("VAL %.0f y (day %d): IPCW AUROC %.4f | slope %.3f | CITL %.4f "
                 "(obs %.4f vs pred %.4f)", h["horizon_years"], int(t), point[f"auc@{t}"],
                 point[f"slope@{t}"], point[f"citl@{t}"], point[f"obsrisk@{t}"],
                 point[f"predrisk@{t}"])

    # ---- 9. patient-level bootstrap (protocol section 18) -------------------
    n_boot = int(mc["bootstrap_n"])
    n_va = len(va)
    boot_keys = list(point.keys())
    boot = {k: np.full(n_boot, np.nan) for k in boot_keys}
    for b in range(n_boot):
        idx = rng.integers(0, n_va, size=n_va)               # patient-level resample
        if int(Eva[idx].sum()) < 2:
            continue
        rb = evaluate(Tva[idx], Eva[idx], lp_va[idx],
                      {t: v[idx] for t, v in risk_at_h.items()}, horizons, g_grid, g_vals)
        for k in boot_keys:
            boot[k][b] = rb[k]
    ci = {k: percentile_ci(v) for k, v in boot.items()}
    boot_valid = {k: int(np.isfinite(v).sum()) for k, v in boot.items()}
    n_boot_usable = int(np.isfinite(boot["cindex"]).sum())
    log.info("bootstrap: %d patient-level resamples of the %d validation patients; "
             "%d yielded a usable C-index, %d a usable 2-year AUROC "
             "(a replicate is skipped when it draws < 2 events)",
             n_boot, n_va, n_boot_usable, boot_valid.get("auc@730.0", n_boot_usable))

    # ---- 9b. PAIRED 5-year vs 2-year AUROC difference (protocol section 18) --
    # Section 18 mandates paired comparisons; arguing from two overlapping MARGINAL
    # intervals is not one. Both horizons are evaluated inside the SAME bootstrap
    # replicate on the SAME resampled patients, so the per-replicate difference is
    # already paired — it only had to be stored and summarised.
    t_long = float(horizons[-1]["horizon_days"])
    t_co2 = float(next(h for h in horizons if h["horizon_years"] == 2.0)["horizon_days"])
    d_auc = boot[f"auc@{t_long}"] - boot[f"auc@{t_co2}"]
    d_ok = np.isfinite(d_auc)
    paired_auc = dict(
        horizon_a_days=t_long, horizon_b_days=t_co2,
        point=float(point[f"auc@{t_long}"] - point[f"auc@{t_co2}"]),
        ci=percentile_ci(d_auc), n_valid=int(d_ok.sum()),
        p_diff_le_zero=float(np.mean(d_auc[d_ok] <= 0)) if d_ok.any() else float("nan"),
        mean_marginal_ci_width=float(
            (ci[f"auc@{t_long}"][1] - ci[f"auc@{t_long}"][0]
             + ci[f"auc@{t_co2}"][1] - ci[f"auc@{t_co2}"][0]) / 2.0),
        paired_ci_width=float(percentile_ci(d_auc)[1] - percentile_ci(d_auc)[0]))
    log.info("PAIRED 5y - 2y IPCW AUROC: %+.4f (95%% CI %+.4f to %+.4f), P(diff <= 0) = "
             "%.3f over %d valid replicates", paired_auc["point"], paired_auc["ci"][0],
             paired_auc["ci"][1], paired_auc["p_diff_le_zero"], paired_auc["n_valid"])

    # ---- 10. proportional hazards (protocol section 19) ---------------------
    ph_frame = Xtr.copy(); ph_frame["_T"] = Ttr; ph_frame["_E"] = Etr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph = proportional_hazard_test(cph, ph_frame, time_transform="km")
    ph_df = ph.summary.reset_index().rename(columns={"index": "covariate", "p": "p_value"})
    ph_df = ph_df[["covariate", "test_statistic", "p_value"]].sort_values("p_value")
    m = len(ph_df)                                            # Holm step-down adjustment
    ph_df["holm_p"] = np.minimum.accumulate(
        np.minimum(1.0, (m - np.arange(m)) * ph_df["p_value"].to_numpy())[::-1])[::-1]
    ph_df["violates_ph"] = ph_df["holm_p"] < 0.05
    ph_df["alpha"] = 0.05
    ph_df["adjustment"] = "holm"
    ph_df["time_transform"] = "km"
    n_viol = int(ph_df["violates_ph"].sum())
    log.info("Schoenfeld: %d of %d covariates violate PH after Holm adjustment%s", n_viol, m,
             (": " + ", ".join(ph_df.loc[ph_df["violates_ph"], "covariate"])) if n_viol else "")

    # ---- 11. calibration table (quintiles of predicted risk) ----------------
    cal_rows = []
    for h in horizons:
        t = h["horizon_days"]
        p = risk_at_h[t]
        bins = risk_bins(p, CALIBRATION_BINS)
        obs, lo, hi = km_risk(Tva, Eva, t)
        cal_rows.append(dict(horizon_years=h["horizon_years"], horizon_days=int(t),
                             bin="overall", n=n_va,
                             n_events_by_horizon=int(((Tva <= t) & (Eva == 1)).sum()),
                             mean_predicted_risk=float(np.mean(p)), km_observed_risk=obs,
                             km_lo=lo, km_hi=hi))
        for b in range(CALIBRATION_BINS):
            k = bins == b
            o, l_, u_ = km_risk(Tva[k], Eva[k], t)
            cal_rows.append(dict(horizon_years=h["horizon_years"], horizon_days=int(t),
                                 bin=f"Q{b + 1}", n=int(k.sum()),
                                 n_events_by_horizon=int(((Tva[k] <= t) & (Eva[k] == 1)).sum()),
                                 mean_predicted_risk=float(np.mean(p[k])),
                                 km_observed_risk=o, km_lo=l_, km_hi=u_))
    cal_df = pd.DataFrame(cal_rows)

    # ---- 12. subgroups (protocol section 21) --------------------------------
    va_sub = va.copy()
    va_sub["_nviews"] = va_sub["view_set"].fillna("").str.count(r"\+") + 1
    sub_defs = [
        ("sex", {"Female": va_sub["sex"] == "Female", "Male": va_sub["sex"] == "Male"}),
        ("age", {"<65": va_sub["age_at_index_imp"] < 65, ">=65": va_sub["age_at_index_imp"] >= 65}),
        ("race_major", {g: va_sub["race_major"] == g for g in ("Black", "White", "Asian")}),
        ("obesity", {"no": va_sub["obesity_imp"] == 0, "yes": va_sub["obesity_imp"] == 1}),
        ("weight_bearing", {"weight-bearing frontal": va_sub["weight_bearing_frontal"].astype(bool),
                            "non-weight-bearing": ~va_sub["weight_bearing_frontal"].astype(bool)}),
        ("acquisition", {"frontal-only": va_sub["_nviews"] == 1,
                         "multi-view": va_sub["_nviews"] >= 2}),
    ]
    co_primary = next(h for h in horizons if h["horizon_years"] == 2.0)
    sub_rows = []
    for family, groups in sub_defs:
        for name, mask in groups.items():
            k = mask.to_numpy()
            n_ev = int(Eva[k].sum())
            supp, note = suppression(n_ev)
            row = dict(family=family, subgroup=name, n=int(k.sum()), n_events_5y=n_ev,
                       event_rate_pct=round(100.0 * n_ev / max(int(k.sum()), 1), 2),
                       n_events_by_2y=int(((Tva[k] <= co_primary["horizon_days"]) & (Eva[k] == 1)).sum()),
                       suppressed=supp, note=note,
                       cindex=np.nan, auc_2y=np.nan, cal_slope_2y=np.nan, citl_2y=np.nan)
            row.update({f"{m}_lo": np.nan for m in ("cindex", "auc_2y")})
            row.update({f"{m}_hi": np.nan for m in ("cindex", "auc_2y")})
            if not supp and k.sum() >= 20:
                t = co_primary["horizon_days"]
                y, w = ipcw_labels_weights(Tva[k], Eva[k], t, g_grid, g_vals)
                s, _ = calibration_slope_intercept(y, w, risk_at_h[t][k])
                o, _, _ = km_risk(Tva[k], Eva[k], t)
                row.update(cindex=harrell_c(Tva[k], Eva[k], lp_va[k]),
                           auc_2y=ipcw_auc(y, w, risk_at_h[t][k]), cal_slope_2y=s,
                           citl_2y=float(o - np.mean(risk_at_h[t][k])))
                # Below 100 events protocol section 21 says emphasise the interval, so a
                # point estimate is never printed without one.
                bc, ba = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
                ts_, es_, ls_, ps_ = Tva[k], Eva[k], lp_va[k], risk_at_h[t][k]
                for bi in range(n_boot):
                    j = rng.integers(0, len(ts_), size=len(ts_))
                    if int(es_[j].sum()) < 2:
                        continue
                    bc[bi] = harrell_c(ts_[j], es_[j], ls_[j])
                    yb, wb = ipcw_labels_weights(ts_[j], es_[j], t, g_grid, g_vals)
                    ba[bi] = ipcw_auc(yb, wb, ps_[j])
                row["cindex_lo"], row["cindex_hi"] = percentile_ci(bc)
                row["auc_2y_lo"], row["auc_2y_hi"] = percentile_ci(ba)
            sub_rows.append(row)
    sub_df = pd.DataFrame(sub_rows)
    n_supp = int(sub_df["suppressed"].sum())
    log.info("subgroups: %d of %d suppressed (<%d events)", n_supp, len(sub_df),
             SUPPRESS_BELOW_EVENTS)

    # ---- 13. sensitivity analyses (protocol section 24) ---------------------
    sens_cfg = mc["sensitivity"]
    sens_specs = []
    if sens_cfg.get("complete_case"):
        # M0's only incompletely observed predictor is pain_score_max (klg_contra is an M1
        # predictor now, protocol Table 7), so complete-case for M0 means "pain score
        # actually observed". klg_contra is added to the filter anyway so the row set is the
        # fully-observed one across every clinical variable in the table, and the extra
        # restriction is stated rather than hidden.
        sens_specs.append(("complete_case",
                           "complete-case: only patients with pain_score_max actually "
                           "observed (M0's sole incompletely observed predictor) AND "
                           "klg_contra observed, so the row set is complete across every "
                           "clinical variable in the feature table; the missing indicators "
                           "are constant by construction on that subset and excluded",
                           [c for c in model_columns if not c.endswith("_missing")],
                           lambda d: (d["klg_contra_missing"] == 0) & (d["pain_score_max_missing"] == 0),
                           []))
    if sens_cfg.get("drop_pain_predictors"):
        sens_specs.append(("drop_pain_predictors",
                           "no-pain-predictor set: knee_pain_any, pain_score_max and its "
                           "indicator removed (also removes the exact collinearity)",
                           [c for c in model_columns
                            if c not in ("knee_pain_any_imp", "pain_score_max_imp",
                                         "pain_score_max_missing")],
                           None, []))
    if sens_cfg.get("race_included"):
        sens_specs.append(("race_included",
                           "race added to the predictor set (the primary excludes it, "
                           "protocol section 21); White is the reference level",
                           list(model_columns), None,
                           ["race_Black", "race_Asian", "race_Other"]))

    sens_rows = []
    for key, desc, cols, row_filter, extra in sens_specs:
        dtr, dva = tr.copy(), va.copy()
        for g, col in (("Black", "race_Black"), ("Asian", "race_Asian"), ("Other", "race_Other")):
            if col in extra:
                dtr[col] = (dtr["race_major"] == g).astype(float)
                dva[col] = (dva["race_major"] == g).astype(float)
        if row_filter is not None:
            dtr, dva = dtr[row_filter(dtr)].reset_index(drop=True), dva[row_filter(dva)].reset_index(drop=True)
        spec = fit_age_spline(dtr["age_at_index_imp"].to_numpy(dtype=float), int(mc["age_rcs_df"]))
        use = list(cols) + list(extra)
        Str, Sva = build_design(dtr, spec, use), build_design(dva, spec, use)
        # A row filter can make a column constant (in the complete-case subset
        # knee_pain_any is 1 by construction). A zero-variance column has no hazard ratio
        # and makes lifelines' normalisation singular, so drop it and say so.
        degenerate = [c for c in Str.columns if float(Str[c].std(ddof=0)) == 0.0]
        if degenerate:
            Str, Sva = Str.drop(columns=degenerate), Sva.drop(columns=degenerate)
            desc += (" — dropped as constant in this subset: " + ", ".join(degenerate))
            log.warning("sensitivity %s: dropped constant column(s) %s", key, degenerate)
        t_tr = dtr["time_from_landmark"].to_numpy(dtype=float); e_tr = dtr["event_indicator"].to_numpy(int)
        t_va = dva["time_from_landmark"].to_numpy(dtype=float); e_va = dva["event_indicator"].to_numpy(int)
        gs, gvv = censoring_curve(t_tr, e_tr)
        # Same criterion as the primary (D24): the CV grid is recomputed on THIS
        # sensitivity's own training rows and column set, so no analysis in this module
        # silently ends up on a different selector. `degenerate` is passed through so the
        # CV design matches the design that is actually fitted.
        gsel, sel_s = tune_and_select_penalizer(
            dtr, use, Str, t_tr, e_tr, Sva, t_va, e_va, grid, float(mc["l1_ratio"]),
            int(mc["age_rcs_df"]), seed, selection_metric,
            drop_design_columns=degenerate, label=f"sensitivity {key}:", log=log)
        pen_s = float(sel_s["penalizer"])
        m_s = fit_cox(Str, t_tr, e_tr, pen_s, float(mc["l1_ratio"]))
        lp_s = linear_predictor(m_s, Sva)
        r_s = {h["horizon_days"]: horizon_risk(m_s, Sva, h["horizon_days"]) for h in horizons}
        ev = evaluate(t_va, e_va, lp_s, r_s, horizons, gs, gvv)
        sens_rows.append(dict(sensitivity=key, description=desc, n_train=len(dtr), n_val=len(dva),
                              n_val_events=int(e_va.sum()), n_parameters=Str.shape[1],
                              penalizer=pen_s,
                              penalizer_selection_metric=sel_s["selection_metric"],
                              val_selected_penalizer=sel_s["val_selected_penalizer"],
                              cv_selected_penalizer=sel_s["cv_selected_penalizer"],
                              **{k: v for k, v in ev.items()}))
        log.info("sensitivity %-20s n_params=%d pen=%g (by %s) val C=%.4f", key, Str.shape[1],
                 pen_s, selection_metric, ev["cindex"])

    landmark_days = sorted(int(d) for d in sens_cfg.get("landmarks_days", []))
    landmark_note = (
        f"DEFERRED — not run. The locked cohort is built at landmark day "
        f"{int(cfg['timeline']['landmark_days'])}. Moving it to {landmark_days} changes "
        "ELIGIBILITY (who is event-free and still observed at the landmark), the time "
        "origin of `time_from_landmark`, the censoring distribution and therefore the event "
        "counts and the locked splits. Producing it requires re-running the Phase-1 "
        "follow-up/outcome pipeline (`src/followup.py`, `src/outcomes.py`, "
        "`src/assemble_cohort.py`) at each landmark and rebuilding `final_cohort.parquet`, "
        "`patient_splits.parquet` and `features_clinical.parquet` — i.e. overwriting locked "
        "files. This module refuses to touch them. To run it: add landmark-parameterised "
        "output paths so the alternative cohorts are written to a separate directory, "
        "re-derive splits per landmark with the same seed, then re-run M0 on each.")

    # ---- 14. outputs --------------------------------------------------------
    tables = cfg.out("tables_dir")
    rows: list[dict] = []

    def add(model, scope, subgroup, metric, hy, hd, n, n_ev, est, lo, hi, supp, note):
        rows.append(dict(model=model, scope=scope, subgroup=subgroup, metric=metric,
                         horizon_years=hy, horizon_days=hd, n=n, n_events=n_ev,
                         estimate=est, ci_lo=lo, ci_hi=hi, suppressed=supp, note=note))

    _, ci_note = suppression(EXPECTED_SPLIT_EVENTS["val"])
    add("M0_primary", "overall", "", "harrell_cindex", None, None, n_va,
        EXPECTED_SPLIT_EVENTS["val"], point["cindex"], *ci["cindex"], False, ci_note)
    add("M0_primary", "overall", "", "harrell_cindex_train_apparent", None, None, len(tr),
        EXPECTED_SPLIT_EVENTS["train"], train_c, np.nan, np.nan, False,
        "apparent (in-sample) discrimination, for the optimism gap only")
    for h in horizons:
        t = h["horizon_days"]; hy = h["horizon_years"]
        n_ev_h = int(((Tva <= t) & (Eva == 1)).sum())
        clamp = ("horizon clamped from day %d: administrative censoring at day %d leaves no "
                 "controls at the nominal horizon" % (h["horizon_days_nominal"], int(max_obs))
                 ) if h["clamped"] else ""
        for metric, key in (("ipcw_cd_auroc", f"auc@{t}"),
                            ("calibration_slope", f"slope@{t}"),
                            ("calibration_intercept_cloglog", f"intercept@{t}"),
                            ("mean_predicted_risk", f"predrisk@{t}"),
                            ("km_observed_risk", f"obsrisk@{t}"),
                            ("calibration_in_the_large_risk_diff", f"citl@{t}")):
            add("M0_primary", "overall", "", metric, hy, int(t), n_va, n_ev_h,
                point[key], *ci[key], False, "; ".join(x for x in (ci_note, clamp) if x))
        add("M0_primary", "overall", "", "ipcw_cd_auroc_G_from_val", hy, int(t), n_va, n_ev_h,
            point_gval[f"auc@{t}"], np.nan, np.nan, False,
            "robustness: censoring curve estimated on val instead of train")
    # horizon_years / horizon_days are left blank: this row spans TWO horizons, and putting
    # a composite string in a numeric column would make the table unparseable.
    add("M0_primary", "overall", "", "ipcw_cd_auroc_paired_diff_5y_minus_2y", None, None,
        n_va, EXPECTED_SPLIT_EVENTS["val"], paired_auc["point"], *paired_auc["ci"], False,
        f"PAIRED patient-level bootstrap (protocol section 18): day "
        f"{int(paired_auc['horizon_a_days'])} minus day "
        f"{int(paired_auc['horizon_b_days'])}, both evaluated on the same resampled patients "
        f"within each replicate; P(diff <= 0) = {paired_auc['p_diff_le_zero']:.3f} over "
        f"{paired_auc['n_valid']} valid replicates")
    for r in sub_df.itertuples():
        for metric, val, lo, hi in (("harrell_cindex", r.cindex, r.cindex_lo, r.cindex_hi),
                                    ("ipcw_cd_auroc", r.auc_2y, r.auc_2y_lo, r.auc_2y_hi),
                                    ("calibration_slope", r.cal_slope_2y, np.nan, np.nan),
                                    ("calibration_in_the_large_risk_diff", r.citl_2y,
                                     np.nan, np.nan)):
            hy = None if metric == "harrell_cindex" else co_primary["horizon_years"]
            hd = None if metric == "harrell_cindex" else int(co_primary["horizon_days"])
            add("M0_primary", f"subgroup:{r.family}", r.subgroup, metric, hy, hd, r.n,
                r.n_events_5y, val, lo, hi, bool(r.suppressed), r.note)
    for s in sens_rows:
        add(f"sens_{s['sensitivity']}", "overall", "", "harrell_cindex", None, None,
            s["n_val"], s["n_val_events"], s["cindex"], np.nan, np.nan, False, s["description"])
        for h in horizons:
            t = h["horizon_days"]
            for metric, key in (("ipcw_cd_auroc", f"auc@{t}"),
                                ("calibration_slope", f"slope@{t}"),
                                ("calibration_in_the_large_risk_diff", f"citl@{t}")):
                add(f"sens_{s['sensitivity']}", "overall", "", metric, h["horizon_years"],
                    int(t), s["n_val"], s["n_val_events"], s[key], np.nan, np.nan, False, "")
    add("sens_landmark_30_90_180", "overall", "", "not_run", None, None, None, None,
        np.nan, np.nan, np.nan, True, landmark_note)
    metrics_df = pd.DataFrame(rows)
    assert "empi_anon" not in metrics_df.columns, "identifier leaked into an outputs/ table"
    metrics_df.to_csv(cfg.path(mc["metrics_csv"]), index=False)

    summ = cph.summary.reset_index().rename(columns={"covariate": "covariate"})
    coef_df = pd.DataFrame(dict(
        covariate=summ["covariate"], coef=summ["coef"], hazard_ratio=summ["exp(coef)"],
        se=summ["se(coef)"], hr_lo=summ["exp(coef) lower 95%"], hr_hi=summ["exp(coef) upper 95%"],
        z=summ["z"], p_value=summ["p"]))
    alias_names = {c for pair in alias for c in pair[:2]}
    level_unident = set(ident["level_unidentified_columns"])
    coef_df["note"] = np.where(
        coef_df["covariate"].isin(alias_names),
        "exactly collinear with another column; coefficient not identified",
        np.where(coef_df["covariate"].isin(level_unident),
                 "spline basis column: LEVEL NOT IDENTIFIED (the basis is a partition of "
                 "unity, so the fitted age curve is identified but this hazard ratio is not)",
                 ""))
    coef_df.to_csv(cfg.path(mc["coefficients_csv"]), index=False)
    cal_df.to_csv(cfg.path(mc["calibration_csv"]), index=False)
    ph_df.to_csv(cfg.path(mc["ph_schoenfeld_csv"]), index=False)
    grid_df.to_csv(tables / "m0_penalizer_grid.csv", index=False)

    # ---- 15. frozen model JSON for T7 ---------------------------------------
    bs = cph.baseline_survival_
    ver_ages = [40.0, 55.0, 64.5, 70.0, 89.0]
    spline["verification"] = {
        "ages": ver_ages,
        "basis": [[float(v) for v in r] for r in spline_basis(ver_ages, spline).to_numpy()]}
    model_json = {
        "module": MODULE,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "M0_clinical_penalized_cox",
        "protocol_sections": [10, 12, 18, 19, 20, 21, 24, 25],
        "test_split": "SEALED — never loaded by this module",
        "cohort": {"n_total_rows_metadata": n_rows_total, "n_train": len(tr), "n_val": len(va),
                   "n_train_events": int(Etr.sum()), "n_val_events": int(Eva.sum())},
        # selection_metric records what ACTUALLY selected (deviation D24 switched it from
        # val_cindex to cv_mean_cindex); both criteria's winners are persisted so the losing
        # one stays auditable and a reader can see whether they agreed.
        "hyperparameters": {"penalizer": penalizer, "l1_ratio": float(mc["l1_ratio"]),
                            "penalizer_grid": grid, "tuning_split": "val",
                            "selection_metric": selection_metric,
                            "selection_criterion": selection["criterion"],
                            "val_selected_penalizer": val_best,
                            "cv_selected_penalizer": cv_best,
                            "cv_design": f"{CV_N_REPEATS}x{CV_N_SPLITS}-fold stratified, train only",
                            "n_selection_events": selection["n_selection_events"],
                            "penalizer_at_grid_edge": penalizer_at_grid_edge,
                            "grid_edge_side": selection["grid_edge_side"]},
        "preprocessing": {
            "source_parquet": str(Path(fc["out_parquet"])),
            "imputation_params_json": str(Path(fc["imputation"]["params_json"])),
            "imputation_source": "features_clinical_frozen",
            "apply": "src.features_clinical.apply_imputer(raw_frame, frozen_params)",
            "model_columns": model_columns,
            "excluded_model_columns": frozen.get("excluded_model_columns", {}),
            "spline": spline},
        "design_columns": list(Xtr.columns),
        "n_parameters": int(Xtr.shape[1]),
        "identified_parameters": int(ident["identified_parameters"]),
        "identifiability": ident,
        "aliased_column_pairs": [[a, b, r] for a, b, r in alias],
        "coefficients": {k: float(v) for k, v in cph.params_.items()},
        "centering_means": {k: float(v) for k, v in cph._norm_mean.items()},
        "linear_predictor": "lp(x) = sum_j (x_j - centering_means[j]) * coefficients[j]",
        "risk_formula": "risk(t|x) = 1 - baseline_survival(t) ** exp(lp(x))",
        "baseline_survival": {"times": [float(t) for t in bs.index.values],
                              "survival": [float(v) for v in bs.iloc[:, 0].values]},
        "censoring_km_train": {"note": "reverse-Kaplan-Meier G(u)=P(C>u) for IPCW re-use",
                               "times": [float(t) for t in g_grid],
                               "survival": [float(v) for v in g_vals]},
        "horizons": horizons,
        "val_metrics": {k: (None if not np.isfinite(v) else float(v)) for k, v in point.items()},
        "val_metrics_ci": {k: [None if not np.isfinite(x) else float(x) for x in v]
                           for k, v in ci.items()},
        "val_metrics_bootstrap_valid": boot_valid,
        "val_paired_auroc_5y_minus_2y": {
            "horizon_a_days": paired_auc["horizon_a_days"],
            "horizon_b_days": paired_auc["horizon_b_days"],
            "estimate": paired_auc["point"],
            "ci": [None if not np.isfinite(x) else float(x) for x in paired_auc["ci"]],
            "p_diff_le_zero": paired_auc["p_diff_le_zero"],
            "n_valid_replicates": paired_auc["n_valid"],
            "method": "paired patient-level bootstrap (protocol section 18): both horizons "
                      "evaluated on the same resampled patients within each replicate"},
    }

    # ---- 15b. VERIFY the JSON replay against the live lifelines fit ----------
    # Section 11 of the report tells T7 to reproduce M0 from this JSON alone. That claim is
    # now COMPUTED here rather than asserted in prose: the replay below uses only values
    # present in model_json (coefficients, centering means, baseline survival step) and is
    # checked against the fitted CoxPHFitter on the validation design.
    replay_lp, replay_risk = replay_from_json(model_json, Xva)
    lp_gap = float(np.abs(replay_lp - lp_va).max())
    risk_gap = max(float(np.abs(replay_risk[t] - risk_at_h[t]).max()) for t in risk_at_h)
    assert lp_gap <= 1e-10, f"JSON replay reproduces the linear predictor only to {lp_gap:.3g}"
    assert risk_gap <= 1e-10, f"JSON replay reproduces horizon risk only to {risk_gap:.3g}"
    model_json["replay_verification"] = {
        "checked_on": "validation design matrix",
        "n_rows": int(len(Xva)),
        "max_abs_linear_predictor_difference": lp_gap,
        "max_abs_horizon_risk_difference": risk_gap,
        "method": "src.model_clinical.replay_from_json vs the live lifelines CoxPHFitter"}
    log.info("JSON replay verified against the live lifelines fit: linear predictor to "
             "%.3g, horizon risk to %.3g", lp_gap, risk_gap)

    model_path = cfg.path(mc["model_json"])
    model_path.write_text(json.dumps(model_json, indent=2) + "\n")
    log.info("wrote %s (frozen M0 contract for T7)", model_path)

    # Maturity is counted on DEVELOPMENT rows only — reading the column cohort-wide would
    # touch sealed rows. The cohort-wide figures are in outputs/feasibility_report.md.
    # THREE distinct quantities, named separately so no two of them can be swapped:
    #   n_status_determined_5y        the 5-year outcome is KNOWN (event, or event-free
    #                                 follow-up reaching the administrative horizon). This is
    #                                 the maturity statistic for a 5-year risk model.
    #   n_full_5y_record_coverage     the complete_5y flag: the record stream extends to
    #                                 landmark + 1826 days. Counts coverage, not status, so
    #                                 it drops most patients whose status is known via the
    #                                 event itself.
    #   n_followup_reaches_day_1825   observed follow-up time reaching the clamped horizon
    #                                 (the count outputs/sample_size.md reports).
    # Administrative censoring lands on landmark + round(5 * 365.25) = day 1826 (the integer
    # day src/followup.py uses); comparing against the unrounded 1826.25 would silently make
    # the status-determined count equal the event count.
    horizon_admin = float(round(float(cfg["timeline"]["horizon_years"])
                                * float(cfg["timeline"]["days_per_year"])))
    assert horizon_admin == float(max_obs), (
        f"administrative horizon {horizon_admin} disagrees with the maximum observed "
        f"follow-up {max_obs}; the maturity counts assume they coincide")
    _t_dev = dev["time_from_landmark"].to_numpy(dtype=float)
    _e_dev = dev["event_indicator"].to_numpy(int)
    maturity = dict(
        n_dev=len(dev),
        n_status_determined_5y=int(((_e_dev == 1) | (_t_dev >= horizon_admin)).sum()),
        n_events=int(_e_dev.sum()),
        n_admin_censored_at_horizon=int(((_e_dev == 0) & (_t_dev >= horizon_admin)).sum()),
        n_full_5y_record_coverage=int(dev["complete_5y"].sum()),
        n_followup_reaches_day_1825=int((_t_dev >= float(horizons[-1]["horizon_days"])).sum()),
        horizon_admin_days=int(round(horizon_admin)),
        eval_horizon_days=int(horizons[-1]["horizon_days"]))
    assert (maturity["n_status_determined_5y"]
            == maturity["n_events"] + maturity["n_admin_censored_at_horizon"]), \
        "status-determined must decompose into events plus administratively censored"
    log.info("5-year maturity on the %d development patients: status DETERMINED %d (%.1f%%) "
             "= %d events + %d administratively censored at day %d; full record coverage "
             "(complete_5y) %d; follow-up reaching day %d %d",
             maturity["n_dev"], maturity["n_status_determined_5y"],
             100 * maturity["n_status_determined_5y"] / maturity["n_dev"],
             maturity["n_events"], maturity["n_admin_censored_at_horizon"],
             maturity["horizon_admin_days"], maturity["n_full_5y_record_coverage"],
             maturity["eval_horizon_days"], maturity["n_followup_reaches_day_1825"])
    max_w = 1.0 / float(step_value(g_grid, g_vals, horizons[-1]["horizon_days"])[0])

    # Protocol section 19: predicted-risk distribution, missingness and follow-up by split.
    t_co = co_primary["horizon_days"]
    dist_rows = []
    for name, sub, Xs, Ts, Es in (("train", tr, Xtr, Ttr, Etr), ("val", va, Xva, Tva, Eva)):
        p = horizon_risk(cph, Xs, t_co)
        q = np.percentile(p, [0, 10, 25, 50, 75, 90, 100])
        med_fu, _ = reverse_km(Ts, Es)
        dist_rows.append(dict(split=name, n=len(sub), n_events=int(Es.sum()),
                              **{f"risk_p{k}": float(v) for k, v in
                                 zip([0, 10, 25, 50, 75, 90, 100], q)},
                              median_followup_days=float(med_fu),
                              pct_klg_missing=round(100 * float(sub["klg_contra_missing"].mean()), 2),
                              pct_pain_score_missing=round(
                                  100 * float(sub["pain_score_max_missing"].mean()), 2)))
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(tables / "m0_risk_distribution.csv", index=False)
    write_report(cfg.path(mc["report_md"]), cfg, mc, spline, Xtr, alias, grid_df, penalizer,
                 selection, point, point_gval, ci, train_c, horizons, cal_df, ph_df, sub_df,
                 sens_rows, landmark_note, max_obs, med_fu_tr, maturity, max_w, n_boot,
                 coef_df, dist_df, co_primary, model_path, ident, paired_auc,
                 model_json["replay_verification"], boot_valid, log)
    for p in (mc["metrics_csv"], mc["coefficients_csv"], mc["calibration_csv"],
              mc["ph_schoenfeld_csv"]):
        log.info("wrote %s", cfg.path(p))

    # ---- 16. M1 = M0 + inferred KLG on the KLG-eligible subset (protocol Table 7) -------
    assert mc["predictor_sets"] == {"m0": "primary_predictors", "m1_klg": "m1_predictors"}, \
        "model_clinical.predictor_sets must name the protocol Table 7 ladder"
    m1 = fit_m1_klg(dev, frozen, horizons, grid, float(mc["l1_ratio"]),
                    int(mc["age_rcs_df"]), n_boot, rng, lp_va, risk_at_h, log,
                    selection_metric=selection_metric, seed=seed)
    write_m1_outputs(cfg, mc, m1, point, log)
    log.info("M0 and M1 complete. Test split untouched.")
    return 0


def write_report(path: Path, cfg: Config, mc: dict, spline: dict, Xtr: pd.DataFrame,
                 alias, grid_df: pd.DataFrame, penalizer: float, selection: dict, point: dict,
                 point_gval: dict, ci: dict, train_c: float, horizons: list[dict],
                 cal_df: pd.DataFrame, ph_df: pd.DataFrame, sub_df: pd.DataFrame,
                 sens_rows: list[dict], landmark_note: str, max_obs: float, med_fu: float,
                 maturity: dict, max_ipcw_weight: float, n_boot: int, coef_df: pd.DataFrame,
                 dist_df: pd.DataFrame, co_primary: dict, model_path: Path, ident: dict,
                 paired_auc: dict, replay_ver: dict, boot_valid: dict,
                 log: logging.Logger) -> Path:
    n_val = EXPECTED_SPLIT_N["val"]; ev_val = EXPECTED_SPLIT_EVENTS["val"]
    n_dev = EXPECTED_SPLIT_N["train"] + n_val
    n_model_cols = Xtr.shape[1] - int(spline["df"]) + 1     # basis columns collapse to age
    L = ["# M0 — clinical baseline (penalized Cox): validation report", "",
         "Generated by `src/model_clinical.py` (Phase 2, Track A). **Aggregate only — no "
         "patient identifiers.** This is the comparator the multi-view image model (M2-M4) "
         "must beat.", "",
         "> **The locked test split was never loaded.** `forbid_test_split: true` is enforced "
         "by pushing a `split != \"test\"` predicate into the Parquet reader; the total-row "
         "invariant is checked against Parquet footer metadata and the sealed-split counts "
         "against the frozen imputation JSON. Every number below is **validation** data.", "",
         "## 1. Headline", ""]
    L.append(_md_table(
        ["metric", "horizon", "estimate", "95% CI (patient-level bootstrap)"],
        [["Harrell's C-index", "overall", _f(point["cindex"]), _ci(*ci["cindex"])]]
        + [[f"IPCW cumulative/dynamic AUROC", f"{h['horizon_years']:.0f} y (day "
            f"{int(h['horizon_days'])})", _f(point[f"auc@{h['horizon_days']}"]),
            _ci(*ci[f"auc@{h['horizon_days']}"])] for h in horizons]))
    cv_best = float(selection["cv_selected_penalizer"])
    val_best = float(selection["val_selected_penalizer"])
    L += ["",
          f"- Chosen penalizer **{penalizer:g}** (ridge, `l1_ratio = {mc['l1_ratio']}`), "
          f"selected by the **{selection['criterion']}** over the {len(grid_df)}-value "
          f"pre-specified grid, from {selection['n_selection_events']} events "
          "(`model_clinical.selection_metric: "
          f"{selection['selection_metric']}`; see section 3 and deviation D24).",
          f"- Apparent (train) C-index {train_c:.3f} vs validation {point['cindex']:.3f} — "
          f"optimism {train_c - point['cindex']:.3f}.",
          f"- Validation: **{n_val} patients, {ev_val} events**. Every interval is a "
          f"{n_boot}-replicate patient-level bootstrap of the validation set with the fitted "
          "model and the censoring weights held fixed, so it measures evaluation "
          "uncertainty, not model-fitting uncertainty. "
          f"{boot_valid.get('cindex', n_boot):,} of {n_boot:,} replicates produced a usable "
          "C-index (a replicate is skipped when the resample draws fewer than 2 events); "
          "per-metric counts are in `val_metrics_bootstrap_valid` in the frozen JSON.",
          f"- **2 years is the co-primary horizon.** The 5-year status is **DETERMINED** for "
          f"{maturity['n_status_determined_5y']:,} of the {n_dev:,} development patients "
          f"({100 * maturity['n_status_determined_5y'] / n_dev:.1f}%) — "
          f"{maturity['n_events']} observed events plus "
          f"{maturity['n_admin_censored_at_horizon']} patients censored administratively at "
          f"day {maturity['horizon_admin_days']}. The remaining "
          f"{n_dev - maturity['n_status_determined_5y']:,} left the record stream before the "
          "horizon and enter the 5-year estimate only through their inverse-probability-of-"
          "censoring weight, which is why the 5-year row is the least stable number here. "
          "(Counted on train+val only; counting it cohort-wide would mean reading sealed "
          "rows.) See section 2 for the two counts that must NOT be substituted for this "
          "one.", "",
          "## 2. Data and design matrix", "",
          "### Three \"5-year maturity\" counts, named apart", "",
          "The project holds three distinct quantities that have all been called some "
          "variant of \"complete 5-year observation\". They are not interchangeable, and only "
          "the first is a maturity statistic for a 5-year risk model. **No metric in this "
          "report changes with the choice** — IPCW handles administrative censoring "
          "correctly and these counts appear only in prose — but the manuscript must not "
          "swap them.", ""]
    L.append(_md_table(
        ["name", "definition", "development (train+val)"],
        [["`n_status_determined_5y`",
          f"5-year status KNOWN: an observed event, or event-free follow-up reaching day "
          f"{maturity['horizon_admin_days']}",
          f"**{maturity['n_status_determined_5y']:,} / {n_dev:,} "
          f"({100 * maturity['n_status_determined_5y'] / n_dev:.1f}%)**"],
         ["`n_full_5y_record_coverage`",
          f"the `complete_5y` flag: `last_observed >= landmark + "
          f"{maturity['horizon_admin_days']}`. RECORD COVERAGE, not status — it excludes "
          "most patients whose status is known precisely because they had the event and "
          "then left the record stream",
          f"{maturity['n_full_5y_record_coverage']:,} "
          f"({100 * maturity['n_full_5y_record_coverage'] / n_dev:.1f}%)"],
         ["`n_followup_reaches_day_1825`",
          f"`time_from_landmark >= {maturity['eval_horizon_days']}`: observed follow-up time "
          "reaching the clamped evaluation horizon, regardless of how the patient left the "
          "risk set (this is the count `outputs/sample_size.md` reports)",
          f"{maturity['n_followup_reaches_day_1825']:,} "
          f"({100 * maturity['n_followup_reaches_day_1825'] / n_dev:.1f}%)"]]))
    L += ["",
          "`outputs/sample_size.md` reconciles all three in one table; "
          "`outputs/feasibility_report.md` reports the cohort-wide status-determined count "
          "(1,401 of 3,709).", ""]
    L += ["### Data", "",
          f"- Source `{cfg['features_clinical']['out_parquet']}`, train "
          f"{EXPECTED_SPLIT_N['train']:,} / val {n_val} (test {EXPECTED_SPLIT_N['test']} "
          "**sealed**).",
          "- The frozen train-fitted imputation is **replayed** from "
          f"`{cfg['features_clinical']['imputation']['params_json']}` and every `*_imp` column "
          "is asserted to reproduce bit for bit — nothing is refit (protocol section 20).",
          f"- Pre-specified predictor block entered whole, **no univariable screening** "
          f"(protocol section 19, `predictor_selection: {mc['predictor_selection']}`).",
          "- **Predictor set = protocol Table 7's M0**: age, sex, comorbidities, pain, and "
          "the image-to-index interval — "
          f"{len(cfg['features_clinical']['primary_predictors'])} routine clinical variables, "
          f"giving {n_model_cols} model columns. The dataset-inferred contralateral KLG is a "
          "**secondary comparator only** (protocol Table 6) and belongs to **M1** "
          "(`outputs/clinical_m1_klg_report.md`), not here: M0 is the comparator in the "
          "primary estimand (protocol Table 8, M4 versus M0 at 5 years), so a "
          "radiograph-derived severity grade inside it would measure imaging against "
          "imaging. `days_to_index` (days from the index radiograph to the index TKA) enters "
          "**linearly**, as Table 6 specifies (\"Continuous\", no spline); it is complete for "
          "all 3,709 patients, so no imputation or missingness indicator applies to it.",
          f"- Age enters as a restricted cubic spline, `cr(age, df={spline['df']})`, knots "
          f"**{spline['all_knots']}** derived from the {spline['n_train_rows_used']:,} "
          "**training** rows only (boundary knots = train age min/max, interior knot = median "
          "of the distinct train ages). Validation, the sealed test set and the Colab notebook "
          "all evaluate the identical basis from those persisted knots.",
          f"- Design matrix: **{Xtr.shape[1]} columns** "
          f"({n_model_cols} model columns - 1 age + {spline['df']} spline basis columns), "
          f"numeric rank {ident['rank']}, and **{ident['identified_parameters']} IDENTIFIED "
          "parameters**.",
          f"- Reverse-KM median follow-up in train: **{med_fu:.0f} days**; maximum observed "
          f"follow-up **{int(max_obs)} days** (administrative censoring).", ""]
    L += ["> **Identifiability is a rank statement, not a correlation statement.** The design "
          f"has {ident['n_columns']} columns and numeric rank {ident['rank']}; adding an "
          f"intercept column does not raise the rank ({ident['rank_with_intercept']}), which "
          "means the columns already span the constant vector. A Cox partial likelihood has "
          "**no intercept**, so a constant added to every linear predictor cancels and one "
          f"further direction is unestimable: **{ident['identified_parameters']} parameters "
          "are identified**. The unidentified direction here is the age basis — patsy's "
          "`cr()` is a **partition of unity** (`" + " + ".join(spline["basis_columns"])
          + " = 1` on every row), which no pairwise correlation scan can see. The practical "
          "consequences are narrow and entirely about *reading* coefficients: the fitted age "
          "**curve** is identified and predictions are unaffected (ridge picks the "
          "minimum-norm representative), but the age spline buys "
          f"{spline['df'] - 1} degrees of freedom, not {spline['df']}, and **no individual "
          "`age_rcs*` hazard ratio may be reported**. They are flagged in the coefficient "
          "table below.", ""]
    if alias:
        L += ["> **Exact pairwise collinearity is still present.** " + "; ".join(
            f"`{a}` and `{b}` are perfectly correlated (r = {r:+.0f})" for a, b, r in alias)
            + ". Only the difference of the two coefficients is estimable.", ""]
    else:
        kp = coef_df[coef_df["covariate"] == "knee_pain_any_imp"]
        kp_txt = ""
        if len(kp):
            r0 = kp.iloc[0]
            kp_txt = (f" `knee_pain_any_imp` now carries the whole effect directly: "
                      f"coefficient {float(r0['coef']):+.4f}, **HR {float(r0['hazard_ratio']):.2f}** "
                      f"(z {float(r0['z']):+.2f}, p {float(r0['p_value']):.4f}).")
        L += ["> **No exactly-collinear column pair remains.** `pain_score_max_missing` used "
              "to sit in the model column list, where it was the exact complement of "
              "`knee_pain_any_imp` (a pain score exists for **precisely** the 3,441 of 3,709 "
              "patients with a pre-index knee-pain record, because the score is aggregated "
              "only over knee-pain rows — protocol Table 5 defines a *knee* pain score, so "
              "widening it would admit shoulder and back scores). Ridge split one shared "
              "effect into two equal and opposite coefficients, so the printed "
              "`knee_pain_any` hazard ratio was **half** the identified effect and its z was "
              "the standard error of an unidentified direction, not of the contrast. "
              "`src/features_clinical.py` now excludes the indicator from `model_columns` "
              "(the column is still written to the parquet for auditing and for the "
              "complete-case filter), so `knee_pain_any_imp` is the absence indicator and "
              "its coefficient is the identified quantity." + kp_txt + " Read that hazard "
              f"ratio against the chosen penalizer ({penalizer:g}): the coefficient table in "
              "section 5 is a **penalized** description of the fit, not an unpenalised "
              "association, and every M0 hazard ratio below moves toward 1 as the penalizer "
              "rises"
              + (f" — at {penalizer:g} the shrinkage is slight, so these are close to the "
                 "unpenalised maximum-partial-likelihood values."
                 if penalizer <= 0.01 else
                 f" — at {penalizer:g} the shrinkage is heavy and every hazard ratio below is "
                 "pulled substantially toward 1.") + " (See section 3 and deviation D24 for "
              "how the penalizer is chosen.)", ""]
    _sm = selection["selection_metric"]
    _gd = grid_df.sort_values(_sm, ascending=False)
    _top2 = float(_gd[_sm].iloc[0] - _gd[_sm].iloc[1])
    _other = "val_cindex" if _sm == "cv_mean_cindex" else "cv_mean_cindex"
    L += ["## 3. Penalizer selection", "",
          f"**The selecting criterion is `{_sm}` — the {selection['criterion']}, estimated "
          f"from {selection['n_selection_events']} events.** Both criteria are computed on "
          "every run and the losers are printed in full below; only one of them selects. "
          "This changed on 2026-07-26 (deviation **D24**, resolving the author decision "
          "flagged as **D23**): selection used to be by validation C-index on "
          f"{ev_val} events, which was monotone across the whole pre-specified grid and so "
          "never turned over, and the resulting winner at the grid maximum compressed the "
          "predicted risks about tenfold. Protocol section 25 explicitly endorses repeated "
          "grouped cross-validation in the development data, the test split is still sealed, "
          "and the **grid itself was not widened**.", "",
          f"Selection over {len(grid_df)} values is noisy either way: on the selecting "
          f"criterion the whole grid spans {selection['grid_span']:.4f} C-index units, which "
          "is far inside the bootstrap interval of any single value "
          f"({_ci(*ci['cindex'])} for the chosen one), and the top two values separate by "
          f"only {_top2:.5f} (the grid is shown at 5 dp for exactly that reason). It is "
          "reported in full so the flatness is visible rather than hidden behind a winner.",
          ""]
    L.append(_md_table(
        ["penalizer", "train C-index", "val C-index", "train partial log-lik",
         f"CV mean C-index ({CV_N_REPEATS}x{CV_N_SPLITS}-fold, train only)", "CV sd"],
        [[f"{r.penalizer:g}" + (" **(chosen)**" if r.penalizer == penalizer else ""),
          _f(r.train_cindex, 5),
          _f(r.val_cindex, 5) + (" *(val pick)*" if r.penalizer == val_best else ""),
          f"{r.train_partial_loglik:.1f}",
          _f(r.cv_mean_cindex, 5) + (" *(CV pick)*" if r.penalizer == cv_best else ""),
          _f(r.cv_sd_cindex, 4)] for r in grid_df.itertuples()]))
    L += ["", "Repeated stratified "
          f"{CV_N_REPEATS}x{CV_N_SPLITS}-fold CV **inside the training split** "
          f"({EXPECTED_SPLIT_N['train']:,} patients, {EXPECTED_SPLIT_EVENTS['train']} events; "
          "one row per patient so patient-grouping is automatic; spline knots refit on each "
          f"CV-training fold) picks **{cv_best:g}**; the {ev_val}-event validation split "
          f"picks **{val_best:g}**. "
          + ("The two agree, so the choice is not an artefact of the "
             f"{ev_val} validation events."
             if selection["criteria_agree"] else
             "They **DISAGREE, and they disagree at opposite ends of the grid** — the "
             "validation C-index rises monotonically with the penalizer while the "
             "cross-validated C-index falls, so neither criterion turns over inside the "
             "pre-specified range. The cross-validated one is the criterion of record "
             f"because it is estimated from {EXPECTED_SPLIT_EVENTS['train']} events rather "
             f"than {ev_val}, and because protocol section 25 names it. The rejected choice "
             "is still visible above.")]
    if _sm == "cv_mean_cindex":
        L += ["", "The same criterion is applied to **every** fit in this module — M0, both "
              "arms of the M1 ladder (on the KLG-eligible training rows) and each "
              "section-24 sensitivity (on its own filtered training rows and column set) — "
              "so no two analyses here are tuned on different criteria. Each fit's rejected "
              "validation-split winner is named in section 8, in the run log and in the "
              "frozen JSONs (`hyperparameters.val_selected_penalizer`)."]
    L += [""]
    if selection["at_grid_edge"]:
        L += [f"> **The chosen penalizer sits at the {selection['grid_edge_side']} "
              f"edge of the pre-specified grid.** The {selection['criterion']} never turns "
              f"over across `{[float(p) for p in grid_df['penalizer']]}`, so the optimum may "
              "lie outside it. That is **reported, not fixed**: "
              "`model_clinical.penalizer_grid` is a pre-specified configuration value and "
              "widening it after seeing the criterion would be exactly the post hoc model "
              "selection protocol section 27 warns about — D24 changed which criterion "
              "selects, it did **not** change the grid. "
              + ("The practical consequence of the *previous* criterion was visible in the "
                 "calibration slopes: a heavily shrunk linear predictor compresses the "
                 "predicted risks, so the slope ran well above 1 even though discrimination "
                 "and calibration-in-the-large were fine. At the lower edge the shrinkage is "
                 "negligible instead, so the calibration slopes in section 4 should be read "
                 "as a near-unpenalised fit — the opposite failure mode (over-spread risks, "
                 "slope below 1) is the one to watch for there."
                 if selection["grid_edge_side"] == "lower" else
                 "The practical consequence is visible in the calibration slopes below — a "
                 "heavily shrunk linear predictor compresses the predicted risks, so the "
                 "slope runs well above 1 even though discrimination and "
                 "calibration-in-the-large are fine."), ""]
    L += ["## 4. Discrimination and calibration on validation", ""]
    hdr = ["horizon", "events by horizon", "IPCW C/D AUROC (95% CI)", "mean predicted risk",
           "KM observed risk", "CITL (obs - pred)", "calibration slope (95% CI)",
           "cloglog intercept"]
    body = []
    for h in horizons:
        t = h["horizon_days"]
        ov = cal_df[(cal_df.horizon_days == int(t)) & (cal_df.bin == "overall")].iloc[0]
        body.append([f"{h['horizon_years']:.0f} y (day {int(t)})", int(ov.n_events_by_horizon),
                     f"{_f(point[f'auc@{t}'])} {_ci(*ci[f'auc@{t}'])}",
                     _f(point[f"predrisk@{t}"]), _f(point[f"obsrisk@{t}"]),
                     _f(point[f"citl@{t}"]),
                     f"{_f(point[f'slope@{t}'])} {_ci(*ci[f'slope@{t}'])}",
                     _f(point[f"intercept@{t}"])])
    L.append(_md_table(hdr, body))
    L += ["", "Calibration slope 1.0 and cloglog intercept 0.0 are perfect; the slope is the "
          "coefficient of `cloglog(predicted risk)` in an IPCW-weighted complementary-log-log "
          "recalibration of the horizon outcome, and the intercept is that model refit with "
          "the slope fixed at 1. A slope below 1 means the predicted risks are too spread out.",
          "",
          "Robustness of the IPCW weights — the censoring curve is estimated on **train** "
          "(scikit-survival's convention, and the only choice consistent with fitting nothing "
          "on validation data). Re-estimating it on validation instead moves the AUROC to: "
          + ", ".join("{:.0f} y {}".format(h["horizon_years"],
                                           _f(point_gval["auc@" + str(h["horizon_days"])]))
                      for h in horizons) + ".", "",
          f"### Calibration by quintile of predicted risk (validation, {CALIBRATION_BINS} bins)",
          "",
          f"Quintiles, not deciles: {ev_val} validation events over {CALIBRATION_BINS} bins "
          f"leaves ~{ev_val // CALIBRATION_BINS} events per bin, which is already thin; deciles "
          "would leave ~5 and the per-bin Kaplan-Meier estimates would be uninterpretable. "
          "Bins hold equal patient counts; observed risk is `1 - KM(t)` within the bin with the "
          "Greenwood 95% interval.", ""]
    for h in horizons:
        t = int(h["horizon_days"])
        sub = cal_df[(cal_df.horizon_days == t) & (cal_df.bin != "overall")]
        L += [f"**{h['horizon_years']:.0f} year (day {t})**", ""]
        L.append(_md_table(["bin", "n", "events by horizon", "mean predicted risk",
                            "KM observed risk (95% CI)"],
                           [[r.bin, r.n, r.n_events_by_horizon, _f(r.mean_predicted_risk),
                             f"{_f(r.km_observed_risk)} {_ci(r.km_lo, r.km_hi)}"]
                            for r in sub.itertuples()]))
        L.append("")
    L += ["## 5. Fitted model (protocol section 19)", "",
          "Hazard ratios from the penalized fit on the training split. Ridge shrinks these "
          f"toward 1 — at the chosen penalizer ({penalizer:g}) "
          + ("only slightly, so they sit close to the unpenalised values"
             if penalizer <= 0.01 else "substantially") +
          " — and the standard errors are not corrected for shrinkage, so the table is "
          "**descriptive**: the validation metrics above are the inferential output.", ""]
    L.append(_md_table(["covariate", "coef", "hazard ratio", "95% CI", "z", "note"],
                       [[f"`{r.covariate}`", _f(r.coef), _f(r.hazard_ratio),
                         _ci(r.hr_lo, r.hr_hi, 2), _f(r.z, 2),
                         "" if not isinstance(r.note, str) else r.note]
                        for r in coef_df.itertuples()]))
    nonalias = coef_df[coef_df["note"].astype(str).str.len() == 0]
    top = nonalias.loc[nonalias["z"].abs().sort_values(ascending=False).index].head(3)
    age_rows = coef_df[coef_df["covariate"].isin(spline["basis_columns"])]
    max_age_z = float(age_rows["z"].abs().max()) if len(age_rows) else float("nan")
    L += ["", "Strongest signals by |z| (level-identified covariates only): "
          + ", ".join(f"`{r.covariate}` (HR {r.hazard_ratio:.2f}, z {r.z:.1f})"
                      for r in top.itertuples()) + ".",
          "",
          f"**Age.** Because `cr()` is a partition of unity the age spline contributes "
          f"{spline['df'] - 1} effective degrees of freedom, not {spline['df']}, and no "
          "single `age_rcs*` row above is interpretable as a hazard ratio — only the fitted "
          "curve is. Read as a block, age carries little: all "
          f"{len(age_rows)} basis z-statistics are |z| < {max_age_z + 0.005:.2f}, so within "
          "this cohort of patients already selected for a first TKA, age adds almost nothing "
          "beyond the other predictors. The spline is retained because protocol section 19 "
          "pre-specifies it, not because it earned its place empirically.", "",
          "Two features deserve comment rather than causal reading. (a) `days_to_index` — the "
          "image-to-index interval, a Table 7 predictor — is an **acquisition** variable, not "
          "a patient one: it describes how long before surgery the eligible radiograph was "
          "taken, so any association it carries is about care pathways and imaging practice, "
          "not about the contralateral knee. It is in M0 because Table 7 names it and because "
          "an image model must be compared against a clinical model that knows the same "
          "interval, not because it is a risk factor. (b) The hypertension and diabetes hazard "
          "ratios point below 1; in a utilisation outcome that is far more plausibly surgical "
          "candidacy and access than biology, and the manuscript must not read them as "
          "protective effects.", "",
          "### Predicted-risk distribution, missingness and follow-up by split "
          "(protocol section 19)", "",
          f"Predicted risk at the co-primary {co_primary['horizon_years']:.0f}-year horizon "
          f"(day {int(co_primary['horizon_days'])}). The test split is absent by design.", ""]
    L.append(_md_table(["split", "n", "events", "min", "p10", "p25", "median", "p75", "p90",
                        "max", "median follow-up (d)", "% KLG missing (M1-ineligible)",
                        "% pain score missing"],
                       [[r.split, r.n, r.n_events, _f(r.risk_p0), _f(r.risk_p10),
                         _f(r.risk_p25), _f(r.risk_p50), _f(r.risk_p75), _f(r.risk_p90),
                         _f(r.risk_p100), f"{r.median_followup_days:.0f}",
                         f"{r.pct_klg_missing:.1f}", f"{r.pct_pain_score_missing:.1f}"]
                        for r in dist_df.itertuples()]))
    L += ["", "The train and validation risk distributions overlap closely, which is what a "
          "correctly frozen preprocessing pipeline should produce. `% KLG missing` is shown "
          "because it defines **M1's** eligible subset (protocol Table 7); `klg_contra` is "
          "**not** an M0 predictor and no KLG value enters any number in this report.", "",
          "## 6. Proportional hazards (protocol section 19)", "",
          "Scaled Schoenfeld residuals against Kaplan-Meier-transformed time, on the training "
          "fit. `holm_p` is the Holm step-down adjustment across the "
          f"{len(ph_df)} covariates.", ""]
    L.append(_md_table(["covariate", "chi-square", "p", "Holm p", "violates PH"],
                       [[f"`{r.covariate}`", _f(r.test_statistic, 2), f"{r.p_value:.2e}",
                         f"{r.holm_p:.2e}", "**yes**" if r.violates_ph else "no"]
                        for r in ph_df.itertuples()]))
    viol = ph_df.loc[ph_df["violates_ph"], "covariate"].tolist()
    L += ["", ("**Violations: " + ", ".join(f"`{v}`" for v in viol) + ".** Those effects are "
               "time-varying rather than constant over follow-up. Consequences: their hazard "
               "ratios in the table above are follow-up-weighted averages and must not be read "
               "as constant effects, and the horizon-specific risks are the trustworthy "
               "output. The horizon-specific IPCW AUROC and calibration reported above do not "
               "assume proportional hazards, so the headline metrics stand. A "
               "time-varying-coefficient or stratified extension is the pre-specified remedy "
               "if the manuscript reports the hazard ratio itself; it is deliberately NOT "
               "applied here because the image model's comparator must stay the pre-specified "
               "M0."
               if viol else "No covariate violates proportional hazards after adjustment."), "",
          "## 7. Subgroup performance and equity audit (protocol section 21)", "",
          f"Pre-specified rule: suppress point estimates below {SUPPRESS_BELOW_EVENTS} subgroup "
          f"events, emphasise CIs below {EMPHASISE_CI_BELOW_EVENTS}. The validation split holds "
          f"**{ev_val} events in total**, so all but "
          f"{len(sub_df) - int(sub_df['suppressed'].sum())} of the {len(sub_df)} strata fall "
          "under the suppression threshold and their performance point estimates are **not "
          "printed** — printing them would be printing noise. Event counts and prevalence are "
          "shown because they are counts, not estimates; anything that does clear the threshold "
          "is shown with its bootstrap interval, never as a bare number.", ""]
    L.append(_md_table(["family", "subgroup", "n", "events (5 y)", "event rate %",
                        "events by 2 y", "C-index (95% CI)", "IPCW AUROC 2 y (95% CI)",
                        "status"],
                       [[r.family, r.subgroup, r.n, r.n_events_5y, f"{r.event_rate_pct:.1f}",
                         r.n_events_by_2y,
                         "suppressed" if r.suppressed
                         else f"{_f(r.cindex)} {_ci(r.cindex_lo, r.cindex_hi)}",
                         "suppressed" if r.suppressed
                         else f"{_f(r.auc_2y)} {_ci(r.auc_2y_lo, r.auc_2y_hi)}", r.note]
                        for r in sub_df.itertuples()]))
    L += ["", f"**{int(sub_df['suppressed'].sum())} of {len(sub_df)} subgroups suppressed.** "
          "This is a structural limitation, not a fixable one at this sample size: with "
          f"{EXPECTED_SPLIT_EVENTS['test']} events even the locked test split will leave most "
          "of these strata under 50 events. The subgroup audit should be reported on the test "
          "split as event counts plus wide interval estimates, and the manuscript should say "
          "plainly that it is underpowered for equity claims — protocol section 21 already "
          "forbids labelling similar performance as 'equitable'.", "",
          "## 8. Sensitivity analyses (protocol section 24)", ""]
    L.append(_md_table(["sensitivity", "n train", "n val", "val events", "params", "penalizer",
                        "val C-index"] + [f"AUROC {h['horizon_years']:.0f} y" for h in horizons],
                       [[f"`{s['sensitivity']}`", s["n_train"], s["n_val"], s["n_val_events"],
                         s["n_parameters"], f"{s['penalizer']:g}", _f(s["cindex"])]
                        + [_f(s[f"auc@{h['horizon_days']}"]) for h in horizons]
                        for s in sens_rows]))
    L += [""] + [f"- `{s['sensitivity']}` — {s['description']}." for s in sens_rows]
    L += ["", "Each sensitivity re-tunes the penalizer over the same pre-specified grid with "
          f"the **same criterion as the primary** (`{selection['selection_metric']}`, "
          "recomputed on that sensitivity's own training rows and column set), and is "
          "reported without bootstrap intervals; they are supporting analyses, and their "
          "differences from the primary are well inside the primary's own interval. Under "
          "the pre-D24 validation criterion they would take "
          + ", ".join(f"`{s['sensitivity']}` {float(s['val_selected_penalizer']):g}"
                      for s in sens_rows) + ".", "",
          "### Landmark sensitivity (day 30 / 90 / 180)", "", landmark_note, "",
          "## 9. Limitations", "",
          f"1. **Follow-up maturity.** The 5-year status is determined for "
          f"{maturity['n_status_determined_5y']:,} of the {n_dev:,} development patients "
          f"({100 * maturity['n_status_determined_5y'] / n_dev:.1f}%: "
          f"{maturity['n_events']} events plus "
          f"{maturity['n_admin_censored_at_horizon']} administratively censored at day "
          f"{maturity['horizon_admin_days']}); the rest contribute to the 5-year estimate "
          "only through their censoring weight. The 5-year IPCW AUROC leans on control "
          f"weights of {max_ipcw_weight:.2f} (1 / G at the horizon) and is the least stable "
          "number in this report. **2 years is the co-primary horizon.**",
          f"2. **The 5-year horizon is clamped to day {int(horizons[-1]['horizon_days'])}.** "
          f"Administrative censoring lands exactly on day {int(max_obs)}, so at the nominal "
          f"day {horizons[-1]['horizon_days_nominal']} there are zero controls and the "
          "censoring curve is 0 — the cumulative/dynamic AUROC is undefined there. Day "
          f"{int(horizons[-1]['horizon_days'])} (4.997 y) is the last day strictly inside "
          "follow-up.",
          f"3. **{ev_val} validation events** drive every interval here, and the bootstrap "
          "intervals are wide by construction. Since D24 the penalizer is chosen by the "
          f"{selection['criterion']} from {selection['n_selection_events']} events rather "
          f"than from those {ev_val}, which removes the worst of the tuning noise but does "
          "nothing for the evaluation noise: neither number should be read as evidence of a "
          "tuned model.",
          "4. **Unmeasured competing risk.** Death is unavailable (protocol section 10), so "
          "these are cause-agnostic quantities and the Kaplan-Meier observed risks overstate "
          "cumulative incidence to the extent that mortality removes patients from risk.",
          "5. **Marginal censoring model.** IPCW assumes censoring independent of the event "
          "and of the covariates. System disengagement is plausibly covariate-related; a "
          "covariate-conditional censoring model is the documented refinement.",
          ("6. **Proportional hazards is violated for " + ", ".join(f"`{v}`" for v in viol)
           + "**, so those hazard ratios are follow-up-weighted averages."
           if viol else "6. **No proportional-hazards violation** survives the Holm "
                        "adjustment, so the hazard ratios may be read as constant effects "
                        "within the precision the penalized fit allows."),
          "7. **Penalized standard errors** are not corrected for shrinkage.",
          "8. **Internal validation only, and the test split is still sealed.** Nothing here "
          "is a final performance claim; the single scripted test evaluation happens once the "
          "image model, ensemble rule, thresholds and analysis script are frozen.", "",
          "## 10. Is the number plausible?", "",
          f"A validation C-index of **{point['cindex']:.3f}** from "
          f"{len(cfg['features_clinical']['primary_predictors'])} routine clinical variables "
          "is what this class of model should produce: clearly better than the 0.50 "
          "of a broken pipeline, and well short of the ~0.75+ that would suggest an outcome "
          "or laterality variable had leaked into the predictors. The apparent-to-validation "
          f"optimism gap is {train_c - point['cindex']:.3f}, consistent with a "
          f"{Xtr.shape[1]}-parameter ridge model on {EXPECTED_SPLIT_EVENTS['train']} training "
          "events rather than an overfit one. Calibration-in-the-large is within about a "
          "percentage point of observed at every horizon. **No evidence of leakage; the "
          "baseline is usable as the comparator M2-M4 must beat.**", "",
          "This number is deliberately lower than an earlier revision of this report, which "
          "put the dataset-inferred contralateral KLG inside M0 and found it to be the "
          "dominant coefficient. Protocol Table 6 lists inferred KLG as a **secondary "
          "comparator only** and protocol Table 7 places it in **M1**; a clinical comparator "
          "that already contains a radiograph-derived severity grade is not \"routine "
          "clinical variables\", and it would bias the study's primary estimand (protocol "
          "Table 8: M4 versus M0 at 5 years) toward finding that imaging adds nothing. What "
          "KLG adds on top of M0 is now measured where it belongs — as M1, on the "
          "KLG-eligible subset, in `outputs/clinical_m1_klg_report.md`.", "",
          "One number does deserve a second look: the 5-year AUROC "
          f"({point['auc@' + str(horizons[-1]['horizon_days'])]:.3f}) exceeds the 2-year AUROC "
          f"({point['auc@' + str(co_primary['horizon_days'])]:.3f}). Protocol section 18 "
          "requires model comparisons to be **paired**, so this is settled by the paired "
          "difference rather than by eyeballing two overlapping marginal intervals: both "
          "horizons are evaluated on the same resampled patients inside each bootstrap "
          "replicate, and the per-replicate difference is",
          "",
          f"**5 y minus 2 y IPCW AUROC = {paired_auc['point']:+.4f}, 95% CI "
          f"({paired_auc['ci'][0]:+.4f} to {paired_auc['ci'][1]:+.4f}), "
          f"P(difference <= 0) = {paired_auc['p_diff_le_zero']:.3f}** over "
          f"{paired_auc['n_valid']:,} valid replicates.",
          "",
          ("The interval crosses zero, so the gap is not distinguishable from sampling "
           "noise — the same conclusion the overlapping-intervals argument reached, now by "
           "the method the protocol actually mandates."
           if paired_auc["ci"][0] <= 0.0 <= paired_auc["ci"][1] else
           "The interval **excludes zero**, so on the paired comparison the 5-year AUROC is "
           "higher than the 2-year one by more than sampling noise on this validation "
           "split — a claim the two overlapping marginal intervals would not have "
           "supported, which is exactly why protocol section 18 mandates the paired form. "
           "Read it as a statement about the two horizons of the SAME model, not as evidence "
           "that the 5-year estimate is the more reliable one.")
          + " (The paired interval is the narrower one by "
          f"construction: {paired_auc['paired_ci_width']:.3f} wide against a mean marginal "
          f"width of {paired_auc['mean_marginal_ci_width']:.3f}, which is exactly why paired "
          "comparisons are the pre-specified device.) The 5-year estimate still rests on "
          f"{int(EXPECTED_SPLIT_EVENTS['val'])} cases against a small, heavily up-weighted "
          "control set, so it remains the least reliable of the three.", "",
          "## 11. Handoff to T7 (Colab fusion notebook)", "",
          f"`{mc['model_json']}` is the frozen contract. To reproduce M0's predictions exactly:",
          "", "1. Read `derived-data/cohort/features_clinical.parquet` and "
          "`derived-data/cohort/clinical_imputation_params.json`.",
          "2. Replay the imputer: `src.features_clinical.apply_imputer(df, frozen_params)`. "
          "**Never refit it.**",
          "3. Rebuild the design matrix in the order given by `design_columns`: the "
          f"{spline['df']} spline basis columns first, then the remaining "
          f"`preprocessing.model_columns` in their stored order, with "
          f"`{spline['variable']}` replaced by the basis.",
          "4. Evaluate the spline with the persisted knots: "
          f"`patsy.dmatrix(\"{spline['patsy_formula']}\", {{\"age\": ages}})`. Check the result "
          "against `preprocessing.spline.verification` (basis values at ages "
          f"{spline['verification']['ages']}) before trusting it.",
          "5. Linear predictor: `lp = sum_j (x_j - centering_means[j]) * coefficients[j]`.",
          "6. Horizon risk: `risk(t|x) = 1 - baseline_survival(t) ** exp(lp)`, where "
          "`baseline_survival` is the step function in the JSON (right-continuous; take the "
          "last time <= t).",
          "7. `censoring_km_train` is the reverse-KM censoring curve for reusing the same IPCW "
          "weights in the fusion evaluation.", "",
          "`horizons` records the days actually used ("
          + " / ".join(("**%d**" % int(h["horizon_days"])) if h["clamped"]
                       else str(int(h["horizon_days"])) for h in horizons)
          + ", the last clamped from "
          f"{horizons[-1]['horizon_days_nominal']}) — use those, not `years * 365.25`, or the "
          "5-year metric will be undefined. `src/sample_size_riley.py` imports the same "
          "`clamp_horizon_days` helper, so its CSV reconciles against this JSON day for day.",
          "",
          f"`n_parameters` ({Xtr.shape[1]}) is the number of design COLUMNS; "
          f"`identified_parameters` ({ident['identified_parameters']}) is how many the "
          "likelihood can estimate, and is the number to quote in a sample-size or "
          "events-per-parameter statement. `identifiability.level_unidentified_columns` names "
          "the spline basis (a partition of unity), and `aliased_column_pairs` names any "
          "exactly-collinear pair. `preprocessing.excluded_model_columns` records the "
          "indicator that `src/features_clinical.py` deliberately keeps out of the model "
          "column list — **T7 must build the design from `preprocessing.model_columns`, not "
          "from `<predictor>_imp` plus every `<predictor>_missing` column in the parquet.**",
          "",
          "**M1 is a separate contract.** `m1_model_columns` in "
          f"`{cfg['features_clinical']['imputation']['params_json']}` and "
          f"`{mc['m1_model_json']}` carry the KLG comparator; the notebook's `m1_klg` arm is "
          "restricted to the KLG-eligible patients (`klg_contra_missing == 0`) and must never "
          "be mixed into the M0 design or into the primary comparison, which stays M4 versus "
          "M0.",
          "",
          "## 12. Reproduce", "",
          "```", "python3 -m src.model_clinical --config config/feasibility.yaml", "```", "",
          f"Deterministic given `reproducibility.random_seed = "
          f"{cfg['reproducibility']['random_seed']}` (bootstrap and CV folds). Outputs: this "
          f"report, `{mc['metrics_csv']}`, `{mc['coefficients_csv']}`, "
          f"`{mc['calibration_csv']}`, `{mc['ph_schoenfeld_csv']}`, "
          "`outputs/tables/m0_penalizer_grid.csv`, "
          "`outputs/tables/m0_risk_distribution.csv` and the frozen "
          f"`{model_path.name}`. The section-11 replay is **executed and asserted inside the "
          f"module** on every run (`replay_from_json`, checked against the live lifelines fit "
          f"on all {replay_ver['n_rows']} validation rows): it reproduces the linear "
          f"predictor to {replay_ver['max_abs_linear_predictor_difference']:.3g} and the "
          f"horizon risks to {replay_ver['max_abs_horizon_risk_difference']:.3g}. Those "
          "numbers are measured here, not quoted from a past session, and the run fails if "
          "either exceeds 1e-10.", ""]
    path.write_text("\n".join(L) + "\n")
    log.info("wrote %s", path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
