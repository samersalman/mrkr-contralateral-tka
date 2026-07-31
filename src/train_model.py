"""train_model.py - the image / multimodal survival ladder, trained LOCALLY (protocol Table 7).

Phase 2 / Track B, step 1. This module is the **single source of truth for training** the
M2 / M3 / M4 image arms and their discrete-time clinical counterparts. It is a
device-agnostic port of ``notebooks/train_colab.ipynb``, which stays untouched as the
record of the Colab/GPU path: that notebook resolves ``DEVICE`` as cuda-or-cpu, gates AMP
and ``empty_cache`` on cuda and has no MPS branch, so it cannot run on this machine.
Everything here resolves ``cuda -> mps -> cpu`` from ``model_image.local.device_preference``
and enables AMP only for a device listed in ``model_image.local.amp_devices``.

Why so much of this file is a port rather than a rewrite
--------------------------------------------------------
The numpy arithmetic below (discrete-time labels, the censoring-aware likelihood, the
hazard-to-risk conversion, the IPCW estimators) was hand-checked interval by interval in
notebook cells 19, 21 and 23 against numbers worked out on paper. Re-deriving it would
throw that away, so it is ported **verbatim**, and ``tests/test_train_model.py`` re-asserts
the same hand-computed expectations. Where ``src/model_clinical.py`` already carries the
real implementation (``step_value``, ``ipcw_labels_weights``, ``ipcw_auc``,
``calibration_slope_intercept``, ``cloglog``, ``harrell_c``) it is **re-exported, not
copied**, so a single estimator serves both the Cox comparator and the image ladder and no
drift between them is possible. The one deliberate duplicate is
:func:`harrell_c_numpy`, the notebook's lifelines-free reimplementation: it is kept as an
independent check and the unit tests prove it agrees with the canonical lifelines-backed
:func:`harrell_c` on censored data with ties. The canonical one is lifelines, because that
is the estimator that produced ``m0_clinical_model.json["val_metrics"]["cindex"]`` and
every model in the ladder must be scored by the same one.

Import cost and the two interpreters
------------------------------------
The numpy / evaluation half of this module imports cleanly under system Python 3.14, which
has no torch. ``import torch`` is attempted once, guarded; if it fails, ``TORCH_AVAILABLE``
is False, the nn classes are still *defined* (on a placeholder base) and any attempt to
build or train one raises a named ImportError pointing at
``~/.venvs/mrkr-torch/bin/python``. So ``src/eval_models.py`` may import the estimators
from here under either interpreter, and ``tests/test_train_model.py`` guards its
torch-dependent tests with ``pytest.importorskip("torch")``.

Protocol sections implemented
-----------------------------
* **12** - every normalization parameter is fitted in development data only. The clinical
  design vector is standardized with the frozen Cox centering means and the TRAIN standard
  deviation; the imputer and the M0/M1 coefficient sets are **replayed from JSON, never
  refitted**. The test split is refused outright: the ``split != "test"`` predicate is
  pushed into the Parquet reader and into the shard reader, and the sealed path is simply
  not implemented here.
* **13** - augmentation is read from ``model_image.augmentation`` and capped: rotation
  <= 5 deg, translation and scale <= 5%, mild intensity only. RandomResizedCrop, any flip,
  elastic, perspective, shear and erasing are asserted absent. The 31 px masked border is
  re-zeroed after the affine, because rotation drags pixels into it.
* **14** - shared ConvNeXt-Tiny encoder with ImageNet init across all views, learned view
  embeddings, mask-aware attention pooling, concatenation with the clinical design vector,
  10 six-month discrete hazards.
* **15** - AdamW, cosine schedule with warm-up, gradient clipping, early stopping on
  validation NLL, five pre-specified seeds, across-seed variability reported.
* **16 / 17** - the ensemble rule is ``average_hazard``: hazards are averaged across seeds
  and converted to risk **once**. Averaging risks instead is a different (and wrong)
  estimator, and the unit tests pin the difference.
* **20** - a missing view is masked, never imputed. A patient with no crop in an arm's view
  set is dropped from that arm and counted.
* **24 / 25** - the frontal-only and DenseNet121 robustness arms come from the same code
  path via ``model_image.local.arms``.

What this module does NOT do
----------------------------
It does not evaluate. Discrimination, calibration, bootstrap intervals, subgroups and the
paired contrasts against M0 are ``src/eval_models.py``. This module trains, ensembles,
fits and freezes the horizon-specific recalibration on validation, and hands over
``derived-data/cohort/train_arms.json`` plus one
``derived-data/cohort/val_hazards_{arm}.npz`` per arm.

The hand-over contract src/eval_models.py reads
-----------------------------------------------
``derived-data/cohort/train_arms.json`` (aggregate; no patient identifier)::

    module, generated_utc, interpreter{python,platform,torch,timm,numpy,pandas,device,amp},
    test_split (a statement that it was not read),
    grid{n_intervals, interval_days, edges[11]}, horizons_days[3], seeds[5],
    ensemble{rule, note}, recalibration{form, fitted_on, frozen},
    cohort{development_rows, patients_with_crops, crops{train,val}, max_crops_per_patient},
    clinical_standardization{m0|m1: {columns, mean, std, n_train, degenerate, source}},
    training_contract{...}, training_contract_hash,
    arms{<arm>: {arm, label, mode, arch, views[], design, stage, subset_arm,
                 seeds[], best_epochs[], best_val_nlls[], n_epochs_run[],
                 val_nll_mean, val_nll_sd, val_nll_min, val_nll_max,
                 seed_val_nlls_ensemble_members[], ensemble_val_nll, ensemble_rule,
                 n_patients, n_events, n_train_patients, n_train_events,
                 n_no_crop_dropped, n_clinical,
                 recalibration{"365.0"|"730.0"|"1825.0":
                     {intercept, slope, note, n_cases, n_controls}},
                 hazards_npz, contract_hash, complete}}

``derived-data/cohort/val_hazards_{arm}.npz`` (git-ignored, so per-patient rows are allowed)::

    hazards          float64 (n_patients, 10)     ensembled, averaged across seeds
    hazards_per_seed float64 (n_seeds, n_patients, 10)
    seeds            int64   (n_seeds,)           aligned with axis 0 of hazards_per_seed
    empi_anon        <U      (n_patients,)        row order; join key for the M0 pairing
    time             float64 (n_patients,)        time_from_landmark
    event            int64   (n_patients,)
    at_risk, target  float64 (n_patients, 10)     the discrete-time likelihood masks
    n_scored         int64   (n_patients,)
    edges            float64 (11,)
    arm, mode, arch, design                       0-d unicode (use ``arr["arm"].item()``)
    views            <U      (n_views_allowed,)

Apply the frozen recalibration with :func:`apply_recalibration` on
``risk_at_horizon(hazards, h)``; rank with :func:`risk_score` for Harrell / Uno C.

Assumptions bought
------------------
* The five seeds, the interval grid, the arm ladder and the augmentation caps are
  pre-specified. Nothing here is tuned on validation except the early-stopping epoch and
  the frozen recalibration, both of which are declared.
* ``~/mrkr-shards`` holds exactly the development crops (4,254 train + 601 val over 2,966
  patients). Both counts are asserted; a change means the paired comparison against M0 is
  no longer on the same patients.
* The masked border band is exactly zero on every edge. That is a contract from
  ``src/preprocess_images.py``, re-checked here after augmentation.

Run:
  ~/.venvs/mrkr-torch/bin/python -m src.train_model --config config/feasibility.yaml \\
      [--stage stage1|stage2] [--arms m4_fusion,m3_image] [--seeds 20250720,...] \\
      [--smoke] [--time-steps N] [--max-epochs N] [--force-retrain] \\
      [--grad-accum N] [--num-workers N]

On this 24 GB M4 Pro ``--grad-accum`` is not optional for the image arms: the
pre-specified 32-patient batch carries ~52 real crops, and ConvNeXt-Tiny holds roughly
0.5 GB of fp32 backward activations per 512x512 crop, so a micro-batch of 32 patients
asks for ~26 GB of unified memory and the machine falls into swap. Splitting the SAME
32-patient optimisation batch into smaller forward passes is the pre-specified valve
(protocol section 13); downscaling the 512 px crop is not.

Writes outputs/tables/train_history.csv and outputs/tables/seed_variability.csv (aggregate
only), plus derived-data/cohort/train_arms.json and derived-data/cohort/val_hazards_*.npz
(git-ignored, per-patient arrays allowed there).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import platform
import random
import sys
import tarfile
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.features_clinical import apply_imputer
from src.model_clinical import (  # the REAL estimators, re-exported not copied
    SEALED_SPLIT,
    calibration_slope_intercept,
    cloglog,
    harrell_c,
    ipcw_auc,
    ipcw_labels_weights,
    load_development_frame,
    replay_from_json,
    spline_basis,
    step_value,
)

MODULE = "train_model"

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS - deliberately NOT in config.                               #
# A config edit must not be able to weaken a guard, so the discrete-time grid   #
# and the cohort anchors live here and are ASSERTED against config at run time. #
# --------------------------------------------------------------------------- #
N_INTERVALS = 10                       # 10 x six-month intervals = 5 years
GRID_MAX_DAYS = 1826.0                 # administrative censoring lands ON day 1826
EDGES = np.linspace(0.0, GRID_MAX_DAYS, N_INTERVALS + 1)
INTERVAL_DAYS = GRID_MAX_DAYS / N_INTERVALS                      # 182.6

DEV_SPLITS = ("train", "val")          # the only splits this module may ever hold
EXPECTED_DEV_ROWS = 2968               # feature table, train 2,597 + val 371
EXPECTED_DEV_PATIENTS_WITH_CROPS = 2966
EXPECTED_SPLIT_CROPS = {"train": 4254, "val": 601}
EXPECTED_HORIZONS = [365.0, 730.0, 1825.0]

# Protocol section 13 augmentation caps.
FORBIDDEN_AUGMENTATIONS = ("randomresizedcrop", "flip", "elastic", "perspective", "shear",
                           "erasing")
MAX_ROTATION_DEG = 5.0
MAX_TRANSLATE_FRAC = 0.05
MAX_SCALE_FRAC = 0.05

RECALIBRATION_FORM = "cloglog(P) = intercept + slope * cloglog(p_hat)"

# Protocol section 14 pre-specified convnext_tiny. Departing from it is allowed
# (D28) but warned about on every run; an unknown name is still refused outright.
PRESPECIFIED_ARCHITECTURE = "convnext_tiny"
KNOWN_ARCHITECTURES = {"convnext_tiny", "densenet121", "resnet50"}


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
# SEALED-SPLIT GUARD - the first thing that touches any patient data.           #
# --------------------------------------------------------------------------- #
def assert_development_splits(splits, allow_sealed: bool = False) -> list[str]:
    """Protocol section 12: refuse the sealed test split unless it is explicitly unlocked.

    Every reader in this module routes through this function, so there is exactly one
    place that decides which splits may be materialised. ``allow_sealed`` defaults to
    False, so nothing reaches the test split by accident or by omission: the caller has to
    name it. The only caller that passes True is ``src/score_test.py``, which exists to
    perform the single sealed read after the model, ensemble rule, thresholds and analysis
    script are frozen (protocol sections 12 and 17).

    Training NEVER passes it. A model that has seen the test split is not a model this
    study can report.
    """
    out = [str(s) for s in splits]
    if allow_sealed:
        return out
    assert SEALED_SPLIT not in out, (
        f"REFUSED: {MODULE} was asked for split {SEALED_SPLIT!r}. The locked test set stays "
        f"unread until the model, ensemble rule, thresholds and analysis script are frozen "
        f"(protocol section 12); use src/score_test.py for the single sealed read.")
    unknown = sorted(set(out) - set(DEV_SPLITS))
    assert not unknown, f"unknown split(s) {unknown}; this module knows only {list(DEV_SPLITS)}"
    return out


def read_json_retrying(path: Path, attempts: int = 6, delay: float = 0.75) -> dict:
    """Read a JSON file that another process may be rewriting; retry a truncated parse."""
    last = None
    for i in range(attempts):
        try:
            return json.loads(Path(path).read_text())
        except (json.JSONDecodeError, FileNotFoundError) as exc:   # noqa: PERF203
            last = exc
            time.sleep(delay * (i + 1))
    raise AssertionError(f"could not parse {path} after {attempts} attempts: {last}")


# =========================================================================== #
# 1. DISCRETE-TIME GRID AND LABELS (pure numpy, ported from notebook cell 18)  #
# =========================================================================== #
def discretize_survival(time_days, event, n_intervals=None, edges=None):
    """Discrete-time survival targets on the frozen interval grid.

    Returns (at_risk, target, k_event, n_scored):
      at_risk[i, k]  1 while patient i is under observation in interval k
      target[i, k]   1 in the single interval where the event occurred, else 0
      k_event[i]     interval index of the event, -1 for censored
      n_scored[i]    number of intervals that contribute to the likelihood
    """
    n_intervals = N_INTERVALS if n_intervals is None else int(n_intervals)
    edges = EDGES if edges is None else np.asarray(edges, dtype=float)
    t = np.asarray(time_days, dtype=float)
    e = np.asarray(event, dtype=int)
    assert t.shape == e.shape and t.ndim == 1, "time and event must be 1-D and aligned"
    assert np.all(t >= 0), "negative time from landmark"
    assert np.all(t <= edges[-1] + 1e-9), (
        f"time beyond the {edges[-1]:.0f}-day grid; administrative censoring should have clamped it")
    assert set(np.unique(e)).issubset({0, 1}), "event must be 0/1"

    k_raw    = np.searchsorted(edges[1:], t, side="right")      # interval containing t
    k_ev     = np.minimum(k_raw, n_intervals - 1)               # event on the last edge -> interval 9
    n_scored = np.where(e == 1, k_ev + 1, np.minimum(k_raw, n_intervals)).astype(int)
    k_event  = np.where(e == 1, k_ev, -1).astype(int)

    kk = np.arange(n_intervals)[None, :]
    at_risk = (kk < n_scored[:, None]).astype(np.float64)
    target  = ((kk == k_event[:, None]) & (e[:, None] == 1)).astype(np.float64)
    assert (target.sum(axis=1) == (e == 1)).all(), "exactly one target interval per event"
    assert (target * (1.0 - at_risk)).sum() == 0.0, "an event interval must be at risk"
    return at_risk, target, k_event, n_scored


def dt_nll_numpy(hazards, at_risk, target, sample_weight=None, eps=1e-7):
    """Censoring-aware discrete-time NLL (numpy reference). Returns (mean, per_patient)."""
    h = np.clip(np.asarray(hazards, dtype=float), eps, 1.0 - eps)
    a = np.asarray(at_risk, dtype=float)
    y = np.asarray(target, dtype=float)
    per_patient = -(a * (y * np.log(h) + (1.0 - y) * np.log(1.0 - h))).sum(axis=1)
    w = np.ones(h.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return float((w * per_patient).sum() / w.sum()), per_patient


# =========================================================================== #
# 2. HAZARDS -> SURVIVAL -> HORIZON RISK; THE ENSEMBLE RULE (cell 21)          #
# =========================================================================== #
def hazards_to_survival(hazards):
    """S[:, k] = prod_{j<k} (1 - h_j); S[:, 0] == 1. Shape (n, n_intervals + 1)."""
    h = np.clip(np.asarray(hazards, dtype=float), 0.0, 1.0)
    return np.concatenate([np.ones((h.shape[0], 1)), np.cumprod(1.0 - h, axis=1)], axis=1)


def risk_at_horizon(hazards, horizon_days, edges=None):
    """1 - S(t), piecewise-constant hazard inside the interval containing t."""
    edges = EDGES if edges is None else np.asarray(edges, dtype=float)
    h = np.clip(np.asarray(hazards, dtype=float), 0.0, 1.0)
    t = float(horizon_days)
    assert 0.0 <= t <= edges[-1], f"horizon {t} outside the discrete grid [0, {edges[-1]}]"
    S = hazards_to_survival(h)
    k = int(np.searchsorted(edges[1:], t, side="right"))
    if k >= h.shape[1]:
        return 1.0 - S[:, -1]
    frac = (t - edges[k]) / (edges[k + 1] - edges[k])
    return 1.0 - S[:, k] * np.power(1.0 - h[:, k], frac)


def average_hazard(hazard_list):
    """config model_image.ensemble == 'average_hazard': average HAZARDS across seeds, convert once."""
    stack = np.stack([np.asarray(h, dtype=float) for h in hazard_list], axis=0)
    assert stack.ndim == 3, "expected a list of (n_patients, n_intervals) hazard arrays"
    return stack.mean(axis=0)


def risk_score(hazards) -> np.ndarray:
    """Horizon-free ranking score for Harrell C: total cumulative hazard -log S(1826)."""
    return -np.log(np.clip(1.0 - np.asarray(hazards, float), 1e-12, 1.0)).sum(axis=1)


# =========================================================================== #
# 3. IPCW EVALUATION (cell 22). step_value / ipcw_labels_weights / ipcw_auc /  #
#    cloglog / calibration_slope_intercept / harrell_c are RE-EXPORTED from    #
#    src.model_clinical above; only the estimators it does not carry are ported #
#    here, and harrell_c_numpy is kept purely as an independent cross-check.    #
#    km_cif_numpy is the FORWARD twin of reverse_km and a fast stand-in for     #
#    model_clinical.km_risk, which the decision curve calls ~1e5 times.         #
# =========================================================================== #
def reverse_km(times, events):
    """Reverse Kaplan-Meier G(u) = P(C > u), lifelines-free.

    ``src.model_clinical.censoring_curve`` is the lifelines-backed equivalent that produced
    the frozen ``censoring_km_train`` curve every arm shares; this one exists so the shape
    of that curve can be re-derived without lifelines, and the unit tests prove the two
    agree.
    """
    t = np.asarray(times, float); e = 1 - np.asarray(events, int)     # censoring is the "event"
    order = np.argsort(t, kind="mergesort"); t, e = t[order], e[order]
    uniq = np.unique(t)
    n, g, grid, vals = t.size, 1.0, [0.0], [1.0]
    for u in uniq:
        at_risk = int((t >= u).sum())
        d = int(e[t == u].sum())
        if at_risk > 0 and d > 0:
            g *= (1.0 - d / at_risk)
        grid.append(float(u)); vals.append(g)
    return np.asarray(grid), np.asarray(vals)


def km_cif_numpy(times, events, horizon: float) -> tuple[float, float]:
    """Forward Kaplan-Meier cumulative incidence ``F(h) = 1 - S(h)``, lifelines-free.

    The forward twin of :func:`reverse_km`. ``src.model_clinical.km_risk`` is the
    lifelines-backed equivalent (it also returns the Greenwood interval, which this one does
    not); this exists because the decision-curve analysis refits Kaplan-Meier ~1e5 times and
    lifelines costs ~2.2 ms a call. The unit tests prove the two agree to ~1e-15 on censored
    data with ties, so it is a drop-in for the point estimate and nothing else.

    Returns ``(cif, last_obs_day)``. ``last_obs_day`` is the largest follow-up time in the
    sample, so ``last_obs_day < horizon`` is exactly the condition "nobody was followed as
    far as the horizon". In that case ``cif`` is the last observed value of the curve
    CARRIED FORWARD - never NaN - and the caller is expected to record ``last_obs_day``
    (the ``km_last_obs_day`` column) so those rows stay identifiable.

    Conventions, all shared with ``km_risk`` / lifelines and asserted in the tests:

    * ``F`` is right-continuous, so an event **on** day ``horizon`` is INCLUDED;
    * a censoring tied with an event on the same day stays in that day's risk set, i.e.
      censoring is treated as occurring after the events it ties with - the same
      ``at_risk = (t >= u).sum()`` rule :func:`reverse_km` uses;
    * ``cif`` is finite for every input, including ones ``km_risk`` cannot take: empty
      input returns ``(0.0, 0.0)``, matching :func:`reverse_km`'s empty curve
      ``([0.0], [1.0])``, and an all-censored sample returns ``0.0`` rather than NaN.
    """
    t = np.asarray(times, float); e = np.asarray(events, int)
    if t.size == 0:
        return 0.0, 0.0                                   # no curve at all; S(0) = 1
    order = np.argsort(t, kind="mergesort"); t, e = t[order], e[order]
    first = np.flatnonzero(np.concatenate(([True], t[1:] != t[:-1])))    # start of each tie block
    at_risk = t.size - first                              # #{T >= u}: censored ties stay in
    d = np.add.reduceat(e, first)                         # events at each unique time
    step = np.where(at_risk > 0, d / np.maximum(at_risk, 1), 0.0)        # never divide by zero
    s = float(np.prod(1.0 - step[t[first] <= float(horizon)]))           # empty product = 1.0
    return 1.0 - s, float(t[-1])


def harrell_c_numpy(times, events, risk):
    """Harrell's C without lifelines. Higher risk must mean shorter survival.

    Kept as an INDEPENDENT check on the canonical lifelines-backed ``harrell_c`` that this
    module re-exports from ``src.model_clinical``; ``tests/test_train_model.py`` asserts
    the two agree exactly on censored data with time and risk ties.
    """
    t = np.asarray(times, float); e = np.asarray(events, int); r = np.asarray(risk, float)
    ti, tj = t[:, None], t[None, :]
    comparable = ((ti < tj) & (e[:, None] == 1)) | ((ti == tj) & (e[:, None] == 1) & (e[None, :] == 0))
    conc = (r[:, None] > r[None, :]).astype(float) + 0.5 * (r[:, None] == r[None, :])
    n = comparable.sum()
    return float((conc * comparable).sum() / n) if n else float("nan")


def uno_c(times, events, risk, g_grid, g_vals):
    """Uno's censoring-robust C (protocol Table 7). Case pairs weighted by 1 / G(T_i-)^2.

    ``g_grid`` / ``g_vals`` are required rather than defaulted to a module global, so no
    caller can silently score against a censoring curve it did not choose.
    """
    t = np.asarray(times, float); e = np.asarray(events, int); r = np.asarray(risk, float)
    g = step_value(g_grid, g_vals, t, left=True)
    wt = np.where(e == 1, 1.0 / np.maximum(g, 1e-12) ** 2, 0.0)
    comparable = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    conc = (r[:, None] > r[None, :]).astype(float) + 0.5 * (r[:, None] == r[None, :])
    W = wt[:, None] * comparable
    den = W.sum()
    return float((conc * W).sum() / den) if den > 0 else float("nan")


def ipcw_brier(y, w, risk):
    """IPCW Brier score at one horizon (unusable patients carry weight 0)."""
    y = np.asarray(y, int); w = np.asarray(w, float); p = np.asarray(risk, float)
    m = (y >= 0) & (w > 0)
    return float((w[m] * (y[m] - p[m]) ** 2).sum() / w[m].sum()) if m.any() else float("nan")


def inv_cloglog(x):
    return 1.0 - np.exp(-np.exp(np.clip(np.asarray(x, float), -30.0, 30.0)))


# --------------------------------------------------------------------------- #
# Horizon-specific recalibration, fitted on VALIDATION only (cell 42).          #
# --------------------------------------------------------------------------- #
def fit_recalibration(hazards, times, events, horizons, g_grid, g_vals) -> dict:
    """One (intercept, slope) per horizon from the validation set. Applied unchanged later."""
    out = {}
    for h in horizons:
        y, w = ipcw_labels_weights(times, events, float(h), g_grid, g_vals)
        p = risk_at_horizon(hazards, float(h))
        b, a = calibration_slope_intercept(y, w, p)
        if not (np.isfinite(a) and np.isfinite(b)):
            a, b = 0.0, 1.0                       # identity; recorded, never silent
            note = ("GLM did not converge on the 54 validation events - identity "
                    "recalibration frozen")
        else:
            note = ""
        out[str(float(h))] = {"intercept": float(a), "slope": float(b), "note": note,
                              "n_cases": int((y == 1).sum()), "n_controls": int((y == 0).sum())}
    return out


def apply_recalibration(p, recal_h: dict) -> np.ndarray:
    """P_recal = inv_cloglog(a + b * cloglog(p_hat)); the frozen horizon-specific transform."""
    return inv_cloglog(float(recal_h["intercept"]) + float(recal_h["slope"]) * cloglog(p))


# =========================================================================== #
# 4. THE FROZEN CLINICAL CONTRACTS - replayed, never refitted (cells 14-16)    #
# =========================================================================== #
class FrozenContracts:
    """The frozen M0 / M1 Cox contracts, the frozen imputer, and the shared IPCW curve."""

    def __init__(self, cohort_dir: Path):
        self.cohort_dir = Path(cohort_dir)
        self.m0 = read_json_retrying(self.cohort_dir / "m0_clinical_model.json")
        self.m1 = read_json_retrying(self.cohort_dir / "m1_klg_model.json")
        self.imputer = read_json_retrying(self.cohort_dir / "clinical_imputation_params.json")
        self.features_pq = self.cohort_dir / "features_clinical.parquet"
        assert self.features_pq.exists(), f"missing {self.features_pq}"

        self.model_columns = list(self.m0["preprocessing"]["model_columns"])
        self.design_columns = list(self.m0["design_columns"])
        self.spline = self.m0["preprocessing"]["spline"]
        self.m1_model_columns = list(self.m1["preprocessing"]["model_columns"])
        self.m1_design_columns = list(self.m1["design_columns"])
        self.m1_spline = self.m1["preprocessing"]["spline"]
        self.m1_eligibility = self.m1["eligibility"]
        self.horizons = [float(h["horizon_days"]) for h in self.m0["horizons"]]

        assert len(self.model_columns) == 11 and len(self.design_columns) == 13, \
            "protocol Table 7 M0 is 11 model columns / 13 design columns"
        assert len(self.m1_design_columns) == 14, "protocol Table 7 M1 adds exactly klg_contra_imp"
        assert not [c for c in self.model_columns if c.startswith("klg")], (
            "a radiograph-derived severity grade inside M0 would make the primary estimand "
            "measure imaging against imaging (protocol Table 6/8)")
        assert self.horizons == EXPECTED_HORIZONS, (self.horizons, EXPECTED_HORIZONS)
        assert [float(h["horizon_days"]) for h in self.m1["horizons"]] == EXPECTED_HORIZONS
        # The 5-year horizon is 1825, clamped from the nominal 1826: administrative censoring
        # lands ON 1826, so the cumulative/dynamic control set (T > t) is empty there.
        assert float(self.m0["horizons"][-1]["horizon_days_nominal"]) == GRID_MAX_DAYS

        km = self.m0["censoring_km_train"]
        self.g_grid = np.asarray(km["times"], dtype=float)
        self.g_vals = np.asarray(km["survival"], dtype=float)
        assert self.g_grid[0] == 0.0 and abs(self.g_vals[0] - 1.0) < 1e-12, \
            "the frozen reverse-KM censoring curve must start at G(0) = 1"

    def design_spec(self, design: str) -> dict:
        assert design in ("m0", "m1"), f"unknown clinical design {design!r}"
        if design == "m0":
            return dict(model_columns=self.model_columns, design_columns=self.design_columns,
                        spline=self.spline, json=self.m0, eligible=None)
        return dict(model_columns=self.m1_model_columns, design_columns=self.m1_design_columns,
                    spline=self.m1_spline, json=self.m1,
                    eligible=lambda f: (f["klg_contra_missing"].to_numpy(int) == 0))


def build_clinical_design(contracts: FrozenContracts, splits=DEV_SPLITS, design: str = "m0",
                          allow_sealed: bool = False):
    """(patient frame, design matrix) for the requested splits, in a fixed row order.

    The ``split != "test"`` predicate is pushed into the Parquet reader by
    :func:`src.model_clinical.load_development_frame`, so no sealed row is ever
    materialised. ``design`` selects a FROZEN contract; "m1" additionally drops patients
    with no inferred contralateral KLG, because protocol Secondary objective 2 restricts
    that comparator to the subset with eligible bilateral frontal images and nothing here
    may impute a radiographic severity grade.
    """
    wanted = assert_development_splits(splits, allow_sealed=allow_sealed)
    frame = load_development_frame(contracts.features_pq, forbid_test=not allow_sealed)
    assert allow_sealed or len(frame) == EXPECTED_DEV_ROWS, (
        f"development feature table has {len(frame)} rows, not {EXPECTED_DEV_ROWS}; the locked "
        f"cohort moved and every paired comparison against M0 is invalidated (protocol "
        f"section 17)")
    frame = (frame[frame["split"].isin(wanted)].sort_values("empi_anon").reset_index(drop=True))
    spec = contracts.design_spec(design)
    if spec["eligible"] is not None:
        frame = frame[spec["eligible"](frame)].reset_index(drop=True)
    raw = list(contracts.imputer["columns"].keys())
    imp = apply_imputer(frame[raw].copy(), contracts.imputer)
    sp = spec["spline"]
    basis = spline_basis(imp[sp["variable"]].to_numpy(dtype=float), sp)
    rest = (imp[[c for c in spec["model_columns"] if c != sp["variable"]]]
            .astype("float64").reset_index(drop=True))
    X = pd.concat([basis, rest], axis=1)
    assert list(X.columns) == spec["design_columns"], (list(X.columns), spec["design_columns"])
    assert X.notna().all().all(), "design matrix has missing values (imputation was skipped?)"
    if design == "m1":
        assert frame["klg_contra"].notna().all(), "a KLG-missing patient reached the M1 design"
    assert allow_sealed or (frame["split"] == SEALED_SPLIT).sum() == 0, "SEALED SPLIT VIOLATION"
    return frame, X


def fit_clin_stats(frame: pd.DataFrame, X: pd.DataFrame, cox_json: dict,
                   design_columns: list[str]) -> dict:
    """Frozen TRAIN-ONLY standardization of one clinical design (protocol section 12).

    Means come from the frozen Cox JSON so a fusion model and its Cox counterpart are
    centred identically; the scale is the TRAIN standard deviation. M1 has its own entry
    because it is a different design on a different (KLG-eligible) patient set, so its
    train statistics are NOT M0's.
    """
    tr = (frame["split"] == "train").to_numpy()
    assert tr.any(), "no train rows to fit the clinical standardization on"
    mean = np.array([float(cox_json["centering_means"][c]) for c in design_columns])
    std = X.to_numpy(float)[tr].std(axis=0, ddof=0)
    return dict(mean=mean, std=np.where(std < 1e-8, 1.0, std), columns=list(design_columns),
                n_train=int(tr.sum()),
                degenerate=[c for c, s in zip(design_columns, std) if s < 1e-8])


def standardize_clinical(X: pd.DataFrame, stats: dict) -> np.ndarray:
    assert list(X.columns) == stats["columns"], (list(X.columns), stats["columns"])
    return (X.to_numpy(dtype=float) - stats["mean"]) / stats["std"]


def replay_cox(contracts: FrozenContracts, X: pd.DataFrame, design: str = "m0"):
    """Linear predictor and horizon risks for the frozen Cox comparator, from JSON alone."""
    return replay_from_json(contracts.design_spec(design)["json"], X)


# =========================================================================== #
# 5. SHARDS -> ONE uint8 MEMMAP PER SPLIT (cell 25, tarfile reader only)       #
# =========================================================================== #
def shard_urls(shard_dir: Path, split: str, allow_sealed: bool = False) -> list[str]:
    """Shards for ONE split. Never glob '*.tar' - that mixes splits."""
    assert_development_splits([split], allow_sealed=allow_sealed)
    return sorted(glob(str(Path(shard_dir) / f"{split}-*.tar")))


def iter_tar_samples(urls, image_ext: str = "png"):
    """(key, HxW uint8 array, meta dict) straight from tarfile - no webdataset involved.

    ``webdataset`` is not installed locally and is not needed: shard members are written
    contiguously as ``{key}.png`` then ``{key}.json``, which stdlib ``tarfile`` streams
    just as well.
    """
    from PIL import Image                       # local import: keeps module import cheap

    for u in urls:
        with tarfile.open(u, "r") as tf:
            cur_key, cur_img = None, None
            for m in tf:
                if not m.isfile():
                    continue
                k, _, ext = m.name.rpartition(".")
                data = tf.extractfile(m).read()
                if ext == image_ext:
                    cur_key, cur_img = k, np.array(Image.open(io.BytesIO(data)).convert("L"))
                elif ext == "json":
                    assert k == cur_key, f"json {k} does not follow its png in {Path(u).name}"
                    yield k, cur_img, json.loads(data.decode())
                    cur_key, cur_img = None, None


def load_sidecar(shard_dir: Path, *, splits=None, expected=None,
                 expected_patients=None) -> pd.DataFrame:
    """labels.csv restricted to ``splits``, with the crop contract asserted.

    Defaults are the development contract. ``src/score_test.py`` passes the sealed split
    and its own frozen counts; nothing else may.
    """
    labels = pd.read_csv(Path(shard_dir) / "labels.csv",
                         dtype={"empi_anon": str, "sop_uid": str, "key": str})
    labels["empi_anon"] = labels["empi_anon"].astype(str)
    want = list(DEV_SPLITS if splits is None else splits)
    exp_crops = EXPECTED_SPLIT_CROPS if expected is None else dict(expected)
    exp_pat = (EXPECTED_DEV_PATIENTS_WITH_CROPS if expected_patients is None
               else int(expected_patients))
    if SEALED_SPLIT not in want:
        assert SEALED_SPLIT not in set(labels["split"]), (
            f"REFUSED: the shard sidecar carries {SEALED_SPLIT!r} rows but the sealed split "
            f"was not requested (protocol section 12)")
    labels = labels[labels["split"].isin(want)].reset_index(drop=True)
    got = labels["split"].value_counts().to_dict()
    assert {k: int(v) for k, v in got.items()} == exp_crops, (
        f"crop counts moved: {got} vs the frozen {exp_crops}; the image set changed "
        f"and the paired comparison against M0 is no longer on the same patients")
    n_pat = labels["empi_anon"].nunique()
    assert n_pat == exp_pat, f"{n_pat} patients carry a crop, not {exp_pat}"
    return labels


def materialize_split(split: str, *, shard_dir: Path, cache_dir: Path, labels: pd.DataFrame,
                      out_size: int, image_ext: str = "png", force: bool = False,
                      log: logging.Logger | None = None, allow_sealed: bool = False):
    """Read every shard of ONE split once; return (memmap path, index DataFrame).

    The memmap and its ``{split}_index.parquet`` sidecar live under
    ``model_image.local.cache_dir``. A cache whose shape or length disagrees with the
    sidecar is rebuilt rather than trusted.
    """
    assert_development_splits([split], allow_sealed=allow_sealed)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{split}_images.npy"
    idx = cache_dir / f"{split}_index.parquet"
    side = labels[labels["split"] == split].reset_index(drop=True)
    n = len(side)
    assert n > 0, f"no sidecar rows for split {split!r}"
    if npy.exists() and idx.exists() and not force:
        arr = np.load(npy, mmap_mode="r")
        index = pd.read_parquet(idx)
        if arr.shape == (n, out_size, out_size) and len(index) == n:
            if log:
                log.info("%s: cache hit (%d images, %.2f GB)", split, n, arr.nbytes / 1e9)
            del arr
            return npy, index
        if log:
            log.info("%s: cache stale, rebuilding", split)
        del arr

    urls = shard_urls(shard_dir, split, allow_sealed=allow_sealed)
    assert urls, f"no {split}-*.tar under {shard_dir}"
    out = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(n, out_size, out_size))
    rows, seen, t0 = [], set(), time.time()
    for i, (key, img, meta) in enumerate(iter_tar_samples(urls, image_ext)):
        assert key not in seen, f"duplicate sample key in the {split} shards"
        seen.add(key)
        assert img.shape == (out_size, out_size) and img.dtype == np.uint8, \
            f"{key}: expected {out_size}x{out_size} uint8, got {img.shape} {img.dtype}"
        assert meta["key"] == key and meta["split"] == split, f"{key}: sidecar/member mismatch"
        out[i] = img
        rows.append({"row": i, "key": key, "empi_anon": str(meta["empi_anon"]),
                     "view": meta["view"], "contra_side": meta["contra_side"]})
    out.flush(); del out
    assert len(rows) == n, f"{split}: read {len(rows)} samples, sidecar has {n}"
    index = pd.DataFrame(rows)
    assert set(index["key"]) == set(side["key"]), f"{split}: shard keys differ from the sidecar"
    index.to_parquet(idx, index=False)
    if log:
        log.info("%s: %d images -> %s (%.2f GB) in %.0fs", split, n, npy.name,
                 npy.stat().st_size / 1e9, time.time() - t0)
    return npy, index


def assert_border_is_zero(img: np.ndarray, band: int) -> None:
    """The masked border band is EXACTLY zero on every edge; that is a contract, not a bug."""
    a = np.asarray(img)
    assert (a[..., :band, :] == 0).all() and (a[..., -band:, :] == 0).all() \
        and (a[..., :, :band] == 0).all() and (a[..., :, -band:] == 0).all(), (
        f"the {band} px masked border (protocol section 13) is not exactly zero")


# =========================================================================== #
# 6. TORCH - imported once, guarded. Everything below needs it.               #
# =========================================================================== #
try:                                        # torch is OPTIONAL at import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    import timm
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: Exception | None = None
    _NNModule, _DatasetBase = nn.Module, Dataset
except ImportError as _exc:                 # system Python 3.14 carries no torch
    torch = nn = F = TF = timm = None                              # type: ignore[assignment]
    DataLoader = WeightedRandomSampler = None                      # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = _exc

    class _TorchRequired:
        """Placeholder base so the nn classes below still DEFINE without torch installed."""

        def __init__(self, *args, **kwargs):
            require_torch()

    _NNModule = _DatasetBase = _TorchRequired                      # type: ignore[assignment]


def require_torch() -> None:
    """Fail loudly and by name when the trainer half is reached under the wrong interpreter."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"{MODULE}: the trainer needs torch, torchvision and timm, which system Python "
            f"{platform.python_version()} does not carry. Run this module as\n"
            f"  cd <project root> && ~/.venvs/mrkr-torch/bin/python -m src.train_model ...\n"
            f"The numpy / evaluation half of this module imports fine without torch.\n"
            f"Original import error: {TORCH_IMPORT_ERROR}")


def resolve_device(preference, amp_devices):
    """Device resolution cuda -> mps -> cpu, from model_image.local.device_preference.

    Returns ``(device, amp_enabled)``. AMP is enabled only for a device named in
    ``model_image.local.amp_devices`` (cuda); mps and cpu run float32 with the scaler off,
    which is the whole reason this module exists separately from the Colab notebook.
    """
    require_torch()
    for name in [str(p) for p in preference]:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda"), ("cuda" in amp_devices)
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps"), ("mps" in amp_devices)
        if name == "cpu":
            return torch.device("cpu"), ("cpu" in amp_devices)
    return torch.device("cpu"), ("cpu" in amp_devices)


def empty_device_cache(device) -> None:
    """torch.cuda.empty_cache() on cuda, torch.mps.empty_cache() on mps, nothing on cpu."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def seed_everything(seed: int) -> None:
    """Seed every generator this module can reach."""
    require_torch()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seeding so augmentation is reproducible for a given (seed, epoch)."""
    s = torch.initial_seed() % (2 ** 31)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def dt_nll_torch(logits, at_risk, target, sample_weight=None):
    """Same estimator as dt_nll_numpy, computed from logits for numerical stability."""
    require_torch()
    ll = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    per_patient = (ll * at_risk.float()).sum(dim=1)
    if sample_weight is None:
        return per_patient.mean(), per_patient
    w = sample_weight.float()
    return (w * per_patient).sum() / w.sum().clamp_min(1e-12), per_patient


# =========================================================================== #
# 7. AUGMENTATION, capped by protocol section 13 (cell 27)                    #
# =========================================================================== #
def parse_augmentation(aug_spec) -> dict:
    """Parse model_image.augmentation and enforce the protocol section 13 caps."""
    spec = [str(a) for a in aug_spec]
    for a in spec:
        assert not any(f in a.lower().replace("_", "") for f in FORBIDDEN_AUGMENTATIONS), \
            (f"augmentation {a!r} is forbidden by protocol section 13: it deforms joint-space "
             f"geometry or changes laterality")

    def _num(prefix, default):
        for a in spec:
            if a.startswith(prefix):
                tail = a[len(prefix):].strip("_").replace("deg", "")
                try:
                    return float(tail)
                except ValueError:
                    return default
        return default

    aug = {"rotation_deg": _num("rotation", 0.0), "translate": _num("translate", 0.0),
           "scale": _num("scale", 0.0), "intensity": _num("brightness_contrast", 0.0),
           "spec": spec}
    assert aug["rotation_deg"] <= MAX_ROTATION_DEG + 1e-9, \
        "protocol section 13 caps rotation at 5 degrees"
    assert aug["translate"] <= MAX_TRANSLATE_FRAC + 1e-9 and aug["scale"] <= MAX_SCALE_FRAC + 1e-9, \
        "protocol section 13 caps translation and scale at 5%"
    return aug


_IMAGENET_STATS: dict = {}


def _imagenet_stats(device):
    key = str(device)
    if key not in _IMAGENET_STATS:
        _IMAGENET_STATS[key] = (
            torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1))
    return _IMAGENET_STATS[key]


def to_model_input(x):
    """(..., H, W) grayscale -> (..., 3, H, W) float, ImageNet-normalized.

    Deliberately out-of-place: integrated gradients feeds a float tensor that carries a
    gradient, and an in-place div_ on it would break the graph.
    """
    require_torch()
    x = (x.to(torch.float32) / 255.0).unsqueeze(-3)           # (..., 1, H, W)
    x = x.expand(*x.shape[:-3], 3, *x.shape[-2:]).contiguous()
    mean, std = _imagenet_stats(x.device)
    return (x - mean) / std


def zero_border_(x, band: int):
    """Re-blank the masked border band in place (rotation can drag pixels into it)."""
    b = int(band)
    x[..., :b, :] = 0; x[..., -b:, :] = 0; x[..., :, :b] = 0; x[..., :, -b:] = 0
    return x


def augment_train(x_u8, rng: np.random.Generator, aug: dict, border_px: int):
    """Section-13-capped augmentation on a single (H, W) uint8 tensor. Returns uint8."""
    require_torch()
    ang   = float(rng.uniform(-aug["rotation_deg"], aug["rotation_deg"]))
    tx    = int(round(float(rng.uniform(-aug["translate"], aug["translate"])) * x_u8.shape[-1]))
    ty    = int(round(float(rng.uniform(-aug["translate"], aug["translate"])) * x_u8.shape[-2]))
    scl   = 1.0 + float(rng.uniform(-aug["scale"], aug["scale"]))
    x = TF.affine(x_u8.unsqueeze(0), angle=ang, translate=[tx, ty], scale=scl, shear=[0.0, 0.0],
                  interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    if aug["intensity"] > 0:
        b = 1.0 + float(rng.uniform(-aug["intensity"], aug["intensity"]))
        c = 1.0 + float(rng.uniform(-aug["intensity"], aug["intensity"]))
        x = TF.adjust_contrast(TF.adjust_brightness(x, b), c)
    return zero_border_(x.squeeze(0), border_px)


def probe_augmentation(image_u8: np.ndarray, aug: dict, border_px: int, seed: int = 0):
    """Augment one real crop and assert the masked border is STILL exactly zero."""
    require_torch()
    probe = torch.from_numpy(np.asarray(image_u8).copy())
    out = augment_train(probe, np.random.default_rng(seed), aug, border_px)
    assert out.dtype == torch.uint8 and out.shape == probe.shape
    assert_border_is_zero(out.numpy(), border_px)
    return out


# =========================================================================== #
# 8. PATIENT-GROUPED DATASET (cell 28)                                        #
# =========================================================================== #
class PatientViewDataset(_DatasetBase):
    """One item = one patient: a padded set of views, a presence mask, clinical vector, labels.

    The image memmap is opened LAZILY per process. macOS spawns DataLoader workers rather
    than forking them, and a ``np.memmap`` attribute would be pickled by value (a 1.1 GB
    copy per worker); holding only the path and reopening on first use avoids that.
    """

    def __init__(self, split: str, frame: pd.DataFrame, X: pd.DataFrame, *, train: bool,
                 npy_path: Path, index: pd.DataFrame, clin_stats: dict, views: list[str],
                 views_allowed=None, max_elems: int = 5, design: str = "m0",
                 out_size: int = 512, aug: dict | None = None, border_px: int = 31,
                 allow_sealed: bool = False):
        assert_development_splits([split], allow_sealed=allow_sealed)
        assert not (train and allow_sealed), (
            "REFUSED: a TRAINING dataset may never be built on the sealed split")
        self.split = str(split)
        self.npy_path = Path(npy_path)
        self._images = None
        self.train = bool(train)
        self.design = str(design)               # which frozen clinical contract X follows
        self.views = list(views)
        self.view_id = {v: i for i, v in enumerate(self.views)}
        self.views_allowed = list(views if views_allowed is None else views_allowed)
        self.max_elems = int(max_elems)
        self.out_size = int(out_size)
        self.aug = aug
        self.border_px = int(border_px)
        self.epoch = 0
        if self.train:
            assert aug is not None, "a training dataset needs a parsed augmentation spec"

        keep = index[index["view"].isin(set(self.views_allowed))]
        by_pat = defaultdict(list)
        for r, p, v in zip(keep["row"].to_numpy(), keep["empi_anon"].to_numpy(),
                           keep["view"].to_numpy()):
            by_pat[str(p)].append((int(r), self.view_id[v]))
        in_split = (frame["split"].to_numpy() == split)
        sub = frame[in_split].reset_index(drop=True)
        Xs = standardize_clinical(X, clin_stats)[in_split]
        self.pids, self.elems, clin, rows_kept = [], [], [], []
        self.n_no_image, self.n_truncated = 0, 0
        for i, p in enumerate(sub["empi_anon"].to_numpy()):
            e = sorted(by_pat.get(str(p), []))
            if not e:
                self.n_no_image += 1
                continue                                  # cannot score a patient with no crop
            if len(e) > self.max_elems:
                self.n_truncated += 1
                e = e[:self.max_elems]                    # deterministic, and counted
            self.pids.append(str(p)); self.elems.append(e); clin.append(Xs[i])
            rows_kept.append(i)
        self.frame = sub.iloc[rows_kept].reset_index(drop=True)
        self.clin = np.asarray(clin, dtype=np.float32)
        t = self.frame["time_from_landmark"].to_numpy(float)
        e = self.frame["event_indicator"].to_numpy(int)
        self.at_risk, self.target, self.k_event, self.n_scored = discretize_survival(t, e)
        self.time, self.event = t, e
        self.loss_weight = np.ones(len(self.pids))

    # -- lazy, spawn-safe memmap ------------------------------------------------------- #
    @property
    def images(self):
        if self._images is None:
            self._images = np.load(self.npy_path, mmap_mode="r")
        return self._images

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_images"] = None
        return state

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        # Deterministic per (patient, seed, epoch): hashlib, never the salted builtin hash().
        # The epoch is mixed in as well as the notebook's torch seed, so augmentation still
        # varies across epochs when num_workers is 0 (a DataLoader with workers reseeds every
        # epoch by itself; without workers torch.initial_seed() is constant, which would have
        # frozen the augmentation). Documented divergence from cell 28.
        _h = int(hashlib.sha1(self.pids[i].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(
            (_h ^ (int(torch.initial_seed()) & 0x7FFFFFFF) ^ (int(self.epoch) * 0x9E3779B1)) % 2**32)
        e = self.elems[i]
        imgs = torch.zeros(self.max_elems, self.out_size, self.out_size, dtype=torch.uint8)
        vids = torch.zeros(self.max_elems, dtype=torch.long)
        mask = torch.zeros(self.max_elems, dtype=torch.bool)
        avail = torch.zeros(len(self.views), dtype=torch.float32)
        for j, (row, vid) in enumerate(e):
            x = torch.from_numpy(np.asarray(self.images[row]).copy())
            imgs[j] = augment_train(x, rng, self.aug, self.border_px) if self.train else x
            vids[j] = vid; mask[j] = True; avail[vid] = 1.0
        return {
            "images": imgs, "view_id": vids, "mask": mask, "view_available": avail,
            "clinical": torch.from_numpy(self.clin[i]),
            "at_risk": torch.from_numpy(self.at_risk[i].astype(np.float32)),
            "target": torch.from_numpy(self.target[i].astype(np.float32)),
            "event": torch.tensor(float(self.event[i])),
            "time": torch.tensor(float(self.time[i])),
            "idx": torch.tensor(i),
        }


def make_loader(ds, batch_size: int, *, shuffle: bool, seed: int = 0, event_aware: bool = False,
                num_workers: int = 0, pin_memory: bool = False):
    """DataLoader. Event-aware sampling attaches the exact inverse-probability correction."""
    require_torch()
    g = torch.Generator(); g.manual_seed(int(seed))
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                  drop_last=False, generator=g)
    if num_workers > 0:
        common["worker_init_fn"] = worker_init_fn
    if event_aware:
        p_nat = np.ones(len(ds))
        w = np.where(ds.event == 1, 1.0 / max(ds.event.mean(), 1e-9),
                     1.0 / max(1.0 - ds.event.mean(), 1e-9))
        w = w / w.sum()
        ds.loss_weight = (p_nat / len(ds)) / w              # undoes the oversampling exactly
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(ds), replacement=True, generator=g)
        return DataLoader(ds, sampler=sampler, **common)
    ds.loss_weight = np.ones(len(ds))
    return DataLoader(ds, shuffle=shuffle, **common)


# =========================================================================== #
# 9. MODEL (cell 30)                                                          #
# =========================================================================== #
class MaskedAttentionPool(_NNModule):
    """Gated attention over a padded set. Padded slots are -inf BEFORE the softmax."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x, mask):                      # x (B, E, D), mask (B, E) bool
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)   # (B, E)
        a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)
        a = torch.nan_to_num(a, nan=0.0)             # a patient with an empty set cannot occur
        return torch.einsum("be,bed->bd", a, x), a


class SurvivalFusionNet(_NNModule):
    """Protocol section 14: shared encoder -> view embeddings + mask -> attention -> hazards."""

    def __init__(self, *, n_intervals: int, n_clinical: int, n_views: int,
                 mode: str = "fusion", arch: str = "convnext_tiny", pretrained: bool = True,
                 view_emb_dim: int = 32, head_hidden: int = 256, dropout: float = 0.2,
                 base_hazard=None, encode_masked_only: bool = True,
                 head_init_scale: float = 0.01):
        super().__init__()
        assert mode in ("fusion", "image_only", "clinical_only")
        self.mode, self.n_intervals, self.arch, self.n_views = mode, n_intervals, arch, n_views
        self.encode_masked_only = bool(encode_masked_only)
        self.use_image = mode in ("fusion", "image_only")
        self.use_clinical = mode in ("fusion", "clinical_only")
        if self.use_image:
            self.encoder = timm.create_model(arch, pretrained=pretrained, num_classes=0)
            self.feat_dim = int(self.encoder.num_features)
            self.view_emb = nn.Embedding(n_views, view_emb_dim)
            self.proj = nn.Linear(self.feat_dim + view_emb_dim, self.feat_dim)
            self.pool = MaskedAttentionPool(self.feat_dim)
            # The image embedding is normalised ON ITS OWN, before the concatenation. A single
            # LayerNorm over the concatenated vector would compute its statistics over 768 image
            # dimensions and 13 clinical ones, so the image block would set the scale and the
            # clinical block would be squashed - which would quietly weaken the very contrast
            # this model exists to measure. The clinical vector arrives already standardised on
            # train, and view_available is 0/1, so neither needs one.
            self.img_norm = nn.LayerNorm(self.feat_dim)
        else:
            self.feat_dim = 0
        in_dim = (self.feat_dim + n_views if self.use_image else 0) + \
                 (n_clinical if self.use_clinical else 0)
        assert in_dim > 0, "a model with neither branch has no inputs"
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, n_intervals))
        if base_hazard is not None:
            # The bias starts the model at the marginal hazard, which is right. Shrinking the
            # final WEIGHT also scales every gradient reaching the encoder by the same factor,
            # so at 0.01 with a single learning rate the backbone barely moves: the primary
            # fusion arm fell only 3.7% below the base-rate training loss before early
            # stopping. The factor is configurable so that damping can be traded against
            # learning rate rather than silently fixed.
            b = torch.as_tensor(base_hazard, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
            with torch.no_grad():
                self.head[-1].bias.copy_(torch.log(b / (1 - b)))
                self.head[-1].weight.mul_(float(head_init_scale))

    def embed_images(self, images, view_id, mask):
        """Encode the view set. With ``encode_masked_only`` the padded slots never reach
        the encoder at all.

        The padded slots are all-zero images whose features are multiplied by the mask two
        lines later and then given -inf attention, so they cannot influence the output.
        Running them through the encoder anyway costs 3x the compute for a multi-view arm
        (5 padded slots against 1.64 real crops per patient on average) and, at the
        pre-specified batch of 32 patients, exceeds the 30 GiB MPS allocation limit
        outright. Gathering the real slots is exactly equivalent for an encoder whose
        normalisation is per-sample, which ConvNeXt-Tiny's LayerNorm is; the unit tests
        assert that equivalence to float tolerance. It is NOT bitwise equivalent for
        DenseNet121, whose BatchNorm would otherwise compute its training statistics over
        a batch that is two thirds all-zero padding. That is a documented deviation, and
        the more defensible of the two behaviours.
        """
        B, E = images.shape[:2]
        flat = images.reshape(B * E, *images.shape[2:])
        if self.encode_masked_only:
            sel = mask.reshape(B * E).nonzero(as_tuple=True)[0]
            feats = self.encoder(to_model_input(flat.index_select(0, sel)))
            f = torch.zeros(B * E, self.feat_dim, dtype=feats.dtype, device=feats.device)
            f = f.index_copy(0, sel, feats).reshape(B, E, self.feat_dim)
        else:
            f = self.encoder(to_model_input(flat)).reshape(B, E, self.feat_dim)
        f = self.proj(torch.cat([f, self.view_emb(view_id)], dim=-1))
        f = f * mask.unsqueeze(-1)                    # padded rows carry no gradient
        return self.pool(f, mask)

    def forward(self, batch, return_attention: bool = False):
        parts, attn = [], None
        if self.use_image:
            emb, attn = self.embed_images(batch["images"], batch["view_id"], batch["mask"])
            parts += [self.img_norm(emb), batch["view_available"]]
        if self.use_clinical:
            parts.append(batch["clinical"])
        logits = self.head(torch.cat(parts, dim=-1))
        return (logits, attn) if return_attention else logits


def base_hazard_from(ds) -> np.ndarray:
    """Marginal discrete hazards under the observed TRAIN prevalence: events_k / at_risk_k."""
    at_risk = ds.at_risk.sum(axis=0)
    events = ds.target.sum(axis=0)
    return np.clip(events / np.maximum(at_risk, 1.0), 1e-4, 1 - 1e-4)


# =========================================================================== #
# 10. OPTIMIZER, SCHEDULE, TRAIN / EVAL STEPS (cells 35, 36)                  #
# =========================================================================== #
class TrainSettings:
    """Every optimizer and schedule knob, read from config.model_image. None hard-coded."""

    def __init__(self, cfg: Config, *, max_epochs_override: int | None = None,
                 batch_size_override: int | None = None,
                 grad_accum_override: int | None = None,
                 num_workers_override: int | None = None):
        mi = cfg["model_image"]
        loc = mi["local"]
        assert str(mi["optimizer"]) == "adamw", "protocol section 15 pre-specifies AdamW"
        assert str(mi["lr_schedule"]) == "cosine", "protocol section 15 pre-specifies cosine decay"
        assert str(mi["early_stopping"]["monitor"]) == "val_nll", \
            "protocol section 15 monitors validation NLL"
        assert str(mi["ensemble"]) == "average_hazard", \
            "protocol section 16 pre-specifies averaging HAZARDS across seeds"
        assert str(mi["recalibration"]) == "horizon_specific_on_val"
        assert str(mi["survival_head"]["kind"]) == "discrete_time"
        assert str(mi["survival_head"]["loss"]) == "censoring_aware_nll"
        assert int(mi["survival_head"]["n_intervals"]) == N_INTERVALS
        assert int(mi["survival_head"]["interval_months"]) == 6
        assert str(mi["pretrained"]) == "imagenet" and bool(mi["shared_encoder_across_views"])
        assert str(mi["aggregator"]) == "attention"
        # Protocol section 14 pre-specifies convnext_tiny. That is no longer a hard stop,
        # because on this cohort it never fitted the training data (D28), but departing
        # from it costs the "no architecture tournament" guarantee and must be visible in
        # the log of every run rather than discoverable only by diffing the config.
        self.architecture = str(mi["architecture"])
        assert self.architecture in KNOWN_ARCHITECTURES, (
            f"unknown architecture {self.architecture!r}; known: {sorted(KNOWN_ARCHITECTURES)}")
        if self.architecture != PRESPECIFIED_ARCHITECTURE:
            warnings.warn(
                f"architecture is {self.architecture!r}, not the pre-specified "
                f"{PRESPECIFIED_ARCHITECTURE!r}. The protocol section 14 'no architecture "
                f"tournament' guarantee no longer holds and any comparison across "
                f"architectures is exploratory. Record this in outputs/protocol_deviations.md.",
                stacklevel=2)

        self.lr = float(mi["lr"])
        self.head_lr_mult = float(mi.get("head_lr_mult", 1.0))
        self.head_init_scale = float(mi.get("head_init_scale", 0.01))
        self.freeze_encoder_epochs = int(mi.get("freeze_encoder_epochs", 0))
        self.dropout = float(mi.get("dropout", 0.2))
        self.weight_decay = float(mi["weight_decay"])
        self.grad_clip_norm = float(mi["grad_clip_norm"])
        self.warmup_epochs = int(mi["warmup_epochs"])
        self.batch_size = int(batch_size_override or mi["batch_size"])
        self.max_epochs = int(max_epochs_override or mi["max_epochs"])
        self.patience = int(mi["early_stopping"]["patience"])
        self.view_emb_dim = int(mi["view_embedding_dim"])
        self.default_arch = str(mi["architecture"])
        self.ensemble_rule = str(mi["ensemble"])
        self.n_seeds = int(mi["n_seeds"])
        self.augmentation = parse_augmentation(mi["augmentation"])

        self.seeds = [int(s) for s in loc["seeds"]]
        assert len(self.seeds) == self.n_seeds, \
            f"model_image.n_seeds is {self.n_seeds} but local.seeds has {len(self.seeds)}"
        self.grad_accum_steps = int(grad_accum_override or loc["grad_accum_steps"])
        assert self.grad_accum_steps >= 1, "grad_accum_steps must be at least 1"
        # Gradient accumulation is the memory valve protocol section 13 leaves open: the
        # optimisation batch stays the pre-specified 32 patients, it is just assembled from
        # `grad_accum_steps` smaller forward passes. Downscaling the 512x512 crop is NOT an
        # option (protocol section 13 fixes the crop size).
        self.micro_batch_size = max(1, math.ceil(self.batch_size / self.grad_accum_steps))
        # The augmentation RNG stream is derived from the DataLoader worker seed, so the
        # worker count is part of what makes a run reproducible and is hashed into the
        # training contract below.
        self.num_workers = int(loc["num_workers"] if num_workers_override is None
                               else num_workers_override)
        assert self.num_workers >= 0, "num_workers must be non-negative"
        self.event_aware_minibatch = bool(loc["event_aware_minibatch"])
        self.device_preference = list(loc["device_preference"])
        self.amp_devices = [str(d) for d in loc["amp_devices"]]
        self.shard_dir = cfg.path(loc["shard_dir"])
        self.cache_dir = cfg.path(loc["cache_dir"])
        self.ckpt_dir = cfg.path(loc["checkpoint_dir"])
        self.history_csv = cfg.path(loc["history_csv"])
        self.seed_variability_csv = cfg.path(loc["seed_variability_csv"])
        self.stages = {k: list(v) for k, v in loc["stages"].items()}
        self.arms = {k: dict(v) for k, v in loc["arms"].items()}
        assert bool(loc["forbid_test_split"]), \
            "model_image.local.forbid_test_split must stay true; the sealed path is not here"

        pp = cfg["preprocess"]
        self.views = list(pp["views_kept"])
        self.out_size = int(pp["out_size"])
        self.image_ext = str(pp["shards"]["image_ext"])
        self.border_px = int(round(float(pp["mask_border_frac"]) * self.out_size))
        assert self.border_px == 31, \
            f"the masked border band moved to {self.border_px} px; protocol section 13 fixed 31"

    def training_contract(self) -> dict:
        """The knobs a checkpoint is only allowed to resume under (cell 35)."""
        return {
            "pretrained": "imagenet", "shared_encoder": True,
            "view_embedding_dim": self.view_emb_dim, "aggregator": "attention",
            "n_intervals": N_INTERVALS, "interval_days": INTERVAL_DAYS,
            "loss": "censoring_aware_nll", "optimizer": "adamw", "lr": self.lr,
            "head_lr_mult": self.head_lr_mult, "head_init_scale": self.head_init_scale,
            "freeze_encoder_epochs": self.freeze_encoder_epochs,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay, "grad_clip_norm": self.grad_clip_norm,
            "lr_schedule": "cosine", "warmup_epochs": self.warmup_epochs,
            "batch_size": self.batch_size, "max_epochs": self.max_epochs,
            "patience": self.patience, "augmentation": list(self.augmentation["spec"]),
            "grad_accum_steps": self.grad_accum_steps,
            "micro_batch_size": self.micro_batch_size, "seeds": list(self.seeds),
            "views": list(self.views), "out_size": self.out_size,
            "event_aware_minibatch": self.event_aware_minibatch,
            "encode_masked_only": True, "num_workers": self.num_workers,
        }

    def contract_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.training_contract(), sort_keys=True).encode()).hexdigest()[:16]


def param_groups(model, settings: TrainSettings) -> list[dict]:
    """Discriminative learning rates: a fresh head needs a far larger step than a
    pretrained backbone.

    ``head_lr_mult`` multiplies the base rate for everything that is randomly initialised
    (head, projection, view embedding, attention pool) and leaves the ImageNet encoder at
    the base rate. One rate for both is why the encoder crawled: the head starts near zero
    (``head_init_scale``), so it contributes almost nothing until it grows, and it cannot
    grow quickly at a rate chosen for a pretrained backbone.

    ``LambdaLR`` scales each group by its own ``initial_lr``, so the cosine schedule and
    the warm-up apply to both groups unchanged.
    """
    # Every parameter is registered even when the encoder is currently frozen: gradual
    # unfreezing toggles requires_grad per epoch, and a parameter left out of the optimizer
    # at construction would never train once it thaws. AdamW skips a parameter whose grad
    # is None, so a frozen group costs nothing.
    enc, fresh = [], []
    for name, p in model.named_parameters():
        (enc if name.startswith("encoder.") else fresh).append(p)
    groups = [{"params": fresh, "lr": settings.lr * settings.head_lr_mult, "name": "head"}]
    if enc:
        groups.append({"params": enc, "lr": settings.lr, "name": "encoder"})
    return groups


def set_encoder_trainable(model, flag: bool) -> None:
    """Freeze or thaw the pretrained backbone. The head always trains.

    Linear probing first, then fine-tuning: with 373 events a randomly initialised head
    attached to a 28M-parameter backbone drags the whole network around before it has
    learned anything useful. Training the head alone for a few epochs gives the backbone a
    sensible error signal when it does thaw, and limits how far it can memorise.
    """
    enc = getattr(model, "encoder", None)
    if enc is None:
        return
    for p in enc.parameters():
        p.requires_grad_(bool(flag))


def make_scheduler(opt, steps_per_epoch: int, settings: TrainSettings):
    """Cosine decay with a linear warm-up over model_image.warmup_epochs epochs."""
    require_torch()
    total = max(1, settings.max_epochs * steps_per_epoch)
    warm = max(1, settings.warmup_epochs * steps_per_epoch)

    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def predict_hazards(model, ds, *, device, settings: TrainSettings, amp: bool,
                    batch_size: int | None = None) -> np.ndarray:
    """(n_patients, n_intervals) hazards in dataset order. No augmentation, no shuffling."""
    require_torch()
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size or settings.micro_batch_size, shuffle=False,
                        num_workers=settings.num_workers,
                        pin_memory=(device.type == "cuda"),
                        worker_init_fn=(worker_init_fn if settings.num_workers > 0 else None))
    out = np.zeros((len(ds), N_INTERVALS), dtype=np.float64)
    with torch.no_grad():
        for batch in loader:
            b = to_device(batch, device)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(b)
            out[b["idx"].cpu().numpy()] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def val_nll_of(hazards, ds) -> float:
    return dt_nll_numpy(hazards, ds.at_risk, ds.target)[0]


def train_one_epoch(model, loader, opt, sched, scaler, ds, *, device, settings: TrainSettings,
                    amp: bool) -> float:
    require_torch()
    model.train()
    total, n = 0.0, 0
    accum = max(1, settings.grad_accum_steps)
    opt.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        b = to_device(batch, device)
        w = torch.as_tensor(ds.loss_weight[b["idx"].cpu().numpy()], dtype=torch.float32,
                            device=device)
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(b)
        loss, _ = dt_nll_torch(logits.float(), b["at_risk"], b["target"], w)
        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip_norm)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()
        total += float(loss.detach()) * b["at_risk"].shape[0]; n += b["at_risk"].shape[0]
    return total / max(n, 1)


def ckpt_path(ckpt_dir: Path, arm: str, seed: int) -> Path:
    return Path(ckpt_dir) / f"{arm}_seed{seed}.pt"


def train_one_seed(arm: str, seed: int, *, spec: dict, train_ds, val_ds, settings: TrainSettings,
                   device, amp: bool, contract_hash: str, force_retrain: bool = False,
                   log: logging.Logger | None = None) -> dict:
    """Train ONE seed. Skips a complete checkpoint; resumes an incomplete one.

    A checkpoint written under a different training-contract hash is REFUSED, not silently
    reused: resuming across a changed optimizer, schedule, loss or augmentation would make
    the recorded epoch history meaningless.
    """
    require_torch()
    path = ckpt_path(settings.ckpt_dir, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode, arch, design = spec["mode"], spec["arch"], spec["design"]
    ck = None
    if path.exists() and not force_retrain:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("contract_hash") != contract_hash:
            raise AssertionError(
                f"{path.name} was written under a different training contract "
                f"({ck.get('contract_hash')} != {contract_hash}). Delete it deliberately or "
                f"pass --force-retrain, and record the change as a protocol deviation.")
        assert ck.get("mode") == mode and ck.get("arch") == arch and ck.get("design") == design, \
            f"{path.name} was trained as {ck.get('mode')}/{ck.get('arch')}/{ck.get('design')}"
        assert list(ck.get("views_allowed", [])) == list(spec["views"]), \
            f"{path.name} was trained on views {ck.get('views_allowed')}, not {spec['views']}"
        if ck.get("complete"):
            if log:
                log.info("  [%s seed %d] complete - best val NLL %.6f at epoch %d (not retrained)",
                         arm, seed, ck["best_val_nll"], ck["best_epoch"])
            return ck

    seed_everything(seed)
    n_clinical = train_ds.clin.shape[1]
    assert train_ds.design == val_ds.design == design, \
        "train and val datasets use different clinical designs"
    model = SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=n_clinical,
                              n_views=len(settings.views), mode=mode, arch=arch, pretrained=True,
                              view_emb_dim=settings.view_emb_dim,
                              base_hazard=base_hazard_from(train_ds),
                              head_init_scale=settings.head_init_scale,
                              dropout=settings.dropout).to(device)
    opt = torch.optim.AdamW(param_groups(model, settings), lr=settings.lr,
                            weight_decay=settings.weight_decay)
    loader = make_loader(train_ds, settings.micro_batch_size, shuffle=True, seed=seed,
                         event_aware=settings.event_aware_minibatch,
                         num_workers=settings.num_workers,
                         pin_memory=(device.type == "cuda"))
    sched = make_scheduler(opt, max(1, math.ceil(len(loader) / max(1, settings.grad_accum_steps))),
                           settings)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch, best, best_epoch, bad, history = 0, float("inf"), -1, 0, []
    best_state = ck["best_model"] if (ck is not None and "best_model" in ck) else None
    if ck is not None:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["rng_torch"]); np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        start_epoch, best = ck["epoch"] + 1, ck["best_val_nll"]
        best_epoch, bad = ck["best_epoch"], ck["bad"]
        history = list(ck["history"])
        if log:
            log.info("  [%s seed %d] resuming at epoch %d (best val NLL so far %.6f)",
                     arm, seed, start_epoch, best)

    for epoch in range(start_epoch, settings.max_epochs):
        t0 = time.time()
        thawed = epoch >= settings.freeze_encoder_epochs
        set_encoder_trainable(model, thawed)
        if log and epoch == settings.freeze_encoder_epochs and settings.freeze_encoder_epochs:
            log.info("    encoder unfrozen at epoch %d (head-only for the first %d)",
                     epoch, settings.freeze_encoder_epochs)
        train_ds.epoch = epoch
        tr_loss = train_one_epoch(model, loader, opt, sched, scaler, train_ds, device=device,
                                  settings=settings, amp=amp)
        vh = predict_hazards(model, val_ds, device=device, settings=settings, amp=amp)
        v_nll = val_nll_of(vh, val_ds)
        improved = v_nll < best - 1e-6
        if improved:
            best, best_epoch, bad = v_nll, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if best_state is None:              # only possible if the very first epoch is NaN
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_nll": float(tr_loss), "val_nll": float(v_nll),
                        "lr": float(sched.get_last_lr()[0]), "secs": time.time() - t0,
                        "improved": bool(improved)})
        if log:
            log.info("    epoch %2d  train NLL %.6f  val NLL %.6f%s  lr %.2e  %.0fs  bad=%d",
                     epoch, tr_loss, v_nll, "  *" if improved else "", sched.get_last_lr()[0],
                     time.time() - t0, bad)
        ck = {"model": model.state_dict(), "best_model": best_state,
              "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": epoch, "best_val_nll": best,
              "best_epoch": best_epoch, "bad": bad, "history": history,
              "complete": bool(bad >= settings.patience or epoch == settings.max_epochs - 1),
              "contract_hash": contract_hash, "arm": arm, "seed": seed, "mode": mode,
              "arch": arch, "views_allowed": list(spec["views"]), "design": design,
              "n_clinical": n_clinical,
              "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
              "rng_python": random.getstate()}
        torch.save(ck, path)
        if bad >= settings.patience:
            if log:
                log.info("    early stop: val NLL has not improved for %d epochs (best %.6f at "
                         "epoch %d)", settings.patience, best, best_epoch)
            break
    del model, opt, loader
    empty_device_cache(device)
    return ck


def load_seed_model(arm: str, seed: int, *, spec: dict, n_clinical: int, settings: TrainSettings,
                    device, contract_hash: str):
    """Rebuild the BEST epoch of one finished seed. Refuses an unfinished or foreign checkpoint."""
    require_torch()
    ck = torch.load(ckpt_path(settings.ckpt_dir, arm, seed), map_location="cpu",
                    weights_only=False)
    assert ck.get("complete"), f"{arm} seed {seed} is not finished"
    assert ck["contract_hash"] == contract_hash, f"{arm} seed {seed} has a foreign contract hash"
    assert ck.get("design") == spec["design"], f"{arm} seed {seed} used a different design"
    net = SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=n_clinical,
                            n_views=len(settings.views), mode=spec["mode"], arch=spec["arch"],
                            pretrained=False, view_emb_dim=settings.view_emb_dim)
    net.load_state_dict(ck["best_model"])
    return net.to(device).eval(), ck


# =========================================================================== #
# 11. THE ARM LADDER: build data once, train five seeds, ensemble, recalibrate #
# =========================================================================== #
class TrainingData:
    """Everything the arms share: frozen contracts, designs, memmaps, clinical statistics."""

    def __init__(self, cfg: Config, settings: TrainSettings, log: logging.Logger | None = None):
        self.cfg, self.settings, self.log = cfg, settings, log
        self.cohort_dir = cfg.path(cfg["paths"]["cohort_dir"])
        self.contracts = FrozenContracts(self.cohort_dir)
        self.labels = load_sidecar(settings.shard_dir)
        self.max_elems = int(self.labels.groupby("empi_anon").size().max())
        self.frames, self.designs, self.clin_stats = {}, {}, {}
        for design in ("m0", "m1"):
            frame, X = build_clinical_design(self.contracts, DEV_SPLITS, design=design)
            self.frames[design], self.designs[design] = frame, X
            self.clin_stats[design] = fit_clin_stats(
                frame, X, self.contracts.design_spec(design)["json"],
                self.contracts.design_spec(design)["design_columns"])
            # The frozen Cox contract is REPLAYED, never refitted. Its centering means ARE
            # the train means, so the mean linear predictor on train must be ~0. If the JSON
            # were regenerated against a different cohort this fails here, loudly, before a
            # single epoch runs.
            lp, _ = replay_cox(self.contracts, X, design)
            is_train = (frame["split"] == "train").to_numpy()
            assert abs(float(lp[is_train].mean())) < 1e-6, (
                f"{design} replay is inconsistent with its frozen JSON: mean train linear "
                f"predictor is {float(lp[is_train].mean()):.3e}, not 0")
            if log:
                log.info("%s clinical design %s replayed from JSON; standardization fitted on "
                         "%d TRAIN rows (protocol section 12)", design, tuple(X.shape),
                         self.clin_stats[design]["n_train"])
        self.npy, self.index = {}, {}
        for split in DEV_SPLITS:
            npy, index = materialize_split(split, shard_dir=settings.shard_dir,
                                           cache_dir=settings.cache_dir, labels=self.labels,
                                           out_size=settings.out_size,
                                           image_ext=settings.image_ext, log=log)
            self.npy[split], self.index[split] = npy, index
        probe = np.asarray(np.load(self.npy["train"], mmap_mode="r")[0])
        assert_border_is_zero(probe, settings.border_px)
        probe_augmentation(probe, settings.augmentation, settings.border_px)
        if log:
            log.info("border band %d px is exactly zero before AND after augmentation "
                     "(protocol section 13)", settings.border_px)

    def max_elems_for(self, views_allowed) -> int:
        """Padding width for ONE arm: the most crops any development patient has in that
        arm's view set.

        The notebook pads every arm to the global maximum of 5, which for a frontal-only
        arm means four empty slots per patient against 1.15 real crops. Nothing is dropped
        either way (the per-arm maximum is by construction never exceeded), so this only
        removes padding that would have been masked out anyway.
        """
        allowed = set(views_allowed)
        widest = 1
        for split in DEV_SPLITS:
            idx = self.index[split]
            sub = idx[idx["view"].isin(allowed)]
            if len(sub):
                widest = max(widest, int(sub.groupby("empi_anon").size().max()))
        return min(widest, self.max_elems)

    def dataset(self, split: str, spec: dict, *, train: bool) -> PatientViewDataset:
        design = spec["design"]
        return PatientViewDataset(
            split, self.frames[design], self.designs[design], train=train,
            npy_path=self.npy[split], index=self.index[split],
            clin_stats=self.clin_stats[design], views=self.settings.views,
            views_allowed=spec["views"], max_elems=self.max_elems_for(spec["views"]),
            design=design, out_size=self.settings.out_size,
            aug=(self.settings.augmentation if train else None),
            border_px=self.settings.border_px)


def arm_spec(settings: TrainSettings, arm: str) -> dict:
    raw = settings.arms[arm]
    spec = {"arm": arm, "mode": str(raw["mode"]), "views": list(raw["views"]),
            "design": str(raw.get("design", "m0")),
            "arch": str(raw.get("arch", settings.default_arch)),
            "label": str(raw.get("label", arm))}
    assert spec["mode"] in ("fusion", "image_only", "clinical_only"), spec["mode"]
    assert set(spec["views"]) <= set(settings.views), (spec["views"], settings.views)
    return spec


def stage_of(settings: TrainSettings, arm: str) -> str:
    for stage, arms in settings.stages.items():
        if arm in arms:
            return stage
    return "unstaged"


def run_arm(arm: str, *, data: TrainingData, settings: TrainSettings, device, amp: bool,
            seeds: list[int], force_retrain: bool, log: logging.Logger) -> dict:
    """Train every seed of one arm, ensemble by averaged hazard, freeze the recalibration."""
    require_torch()
    spec = arm_spec(settings, arm)
    contract_hash = settings.contract_hash()
    tr_ds = data.dataset("train", spec, train=True)
    va_ds = data.dataset("val", spec, train=False)
    log.info("=== %s: %s (mode=%s, arch=%s, views=%s, design=%s) ===", arm, spec["label"],
             spec["mode"], spec["arch"], spec["views"], spec["design"])
    log.info("  patients: train %d (%d events), val %d (%d events); %d dropped with no crop in "
             "this arm's view set (protocol section 20: masked, never imputed)",
             len(tr_ds), int(tr_ds.event.sum()), len(va_ds), int(va_ds.event.sum()),
             tr_ds.n_no_image + va_ds.n_no_image)

    history_rows, per_seed_meta = [], []
    for seed in seeds:
        ck = train_one_seed(arm, seed, spec=spec, train_ds=tr_ds, val_ds=va_ds,
                            settings=settings, device=device, amp=amp,
                            contract_hash=contract_hash, force_retrain=force_retrain, log=log)
        for h in ck["history"]:
            history_rows.append({"arm": arm, "seed": seed, "epoch": int(h["epoch"]),
                                 "train_nll": float(h["train_nll"]), "val_nll": float(h["val_nll"]),
                                 "lr": float(h["lr"]), "secs": round(float(h["secs"]), 2),
                                 "improved": bool(h.get("improved", False))})
        per_seed_meta.append({"seed": int(seed), "best_val_nll": float(ck["best_val_nll"]),
                              "best_epoch": int(ck["best_epoch"]),
                              "n_epochs": len(ck["history"]), "complete": bool(ck["complete"])})
    assert all(m["complete"] for m in per_seed_meta), \
        f"{arm}: not every seed finished; rerun before ensembling"

    per_seed_haz = []
    n_clinical = tr_ds.clin.shape[1]
    for seed in seeds:
        net, _ = load_seed_model(arm, seed, spec=spec, n_clinical=n_clinical, settings=settings,
                                 device=device, contract_hash=contract_hash)
        per_seed_haz.append(predict_hazards(net, va_ds, device=device, settings=settings, amp=amp))
        del net
        empty_device_cache(device)
    ens = average_hazard(per_seed_haz)
    seed_nlls = [dt_nll_numpy(h, va_ds.at_risk, va_ds.target)[0] for h in per_seed_haz]
    ens_nll = dt_nll_numpy(ens, va_ds.at_risk, va_ds.target)[0]
    log.info("  val NLL per seed %s -> ensemble (%s) %.6f",
             np.round(seed_nlls, 4).tolist(), settings.ensemble_rule, ens_nll)

    recal = fit_recalibration(ens, va_ds.time, va_ds.event, data.contracts.horizons,
                              data.contracts.g_grid, data.contracts.g_vals)
    for h in data.contracts.horizons:
        r = recal[str(float(h))]
        log.info("  recalibration @%4dd  intercept %+.4f  slope %+.4f  (cases %d, controls %d) %s",
                 int(h), r["intercept"], r["slope"], r["n_cases"], r["n_controls"], r["note"])

    npz_name = f"val_hazards_{arm}.npz"
    np.savez(data.cohort_dir / npz_name,
             hazards=ens.astype(np.float64),
             hazards_per_seed=np.stack(per_seed_haz).astype(np.float64),
             seeds=np.asarray(seeds, dtype=np.int64),
             empi_anon=np.asarray(va_ds.pids, dtype=object).astype("U"),
             time=va_ds.time.astype(np.float64), event=va_ds.event.astype(np.int64),
             at_risk=va_ds.at_risk.astype(np.float64), target=va_ds.target.astype(np.float64),
             n_scored=va_ds.n_scored.astype(np.int64), edges=EDGES.astype(np.float64),
             arm=np.asarray(arm), mode=np.asarray(spec["mode"]), arch=np.asarray(spec["arch"]),
             design=np.asarray(spec["design"]),
             views=np.asarray(spec["views"], dtype=object).astype("U"))

    v = np.asarray([m["best_val_nll"] for m in per_seed_meta], dtype=float)
    summary = {
        "arm": arm, "label": spec["label"], "mode": spec["mode"], "arch": spec["arch"],
        "views": spec["views"], "design": spec["design"], "stage": stage_of(settings, arm),
        "subset_arm": spec["design"] == "m1",
        "seeds": [int(s) for s in seeds],
        "best_epochs": [m["best_epoch"] for m in per_seed_meta],
        "best_val_nlls": [m["best_val_nll"] for m in per_seed_meta],
        "n_epochs_run": [m["n_epochs"] for m in per_seed_meta],
        "val_nll_mean": float(v.mean()),
        "val_nll_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "val_nll_min": float(v.min()), "val_nll_max": float(v.max()),
        "seed_val_nlls_ensemble_members": [float(x) for x in seed_nlls],
        "ensemble_val_nll": float(ens_nll),
        "ensemble_rule": settings.ensemble_rule,
        "n_patients": int(len(va_ds)), "n_events": int(va_ds.event.sum()),
        "n_train_patients": int(len(tr_ds)), "n_train_events": int(tr_ds.event.sum()),
        "n_no_crop_dropped": int(tr_ds.n_no_image + va_ds.n_no_image),
        "n_clinical": int(n_clinical),
        "recalibration": recal,
        "hazards_npz": npz_name,
        "contract_hash": contract_hash,
        "complete": True,
    }
    return {"summary": summary, "history_rows": history_rows}


# --------------------------------------------------------------------------- #
# Output writers. outputs/ is AGGREGATE ONLY: no empi_anon, no UID, no row.     #
# --------------------------------------------------------------------------- #
HISTORY_COLUMNS = ["arm", "seed", "epoch", "train_nll", "val_nll", "lr", "secs", "improved"]
SEED_VARIABILITY_COLUMNS = ["arm", "label", "n_seeds", "val_nll_mean", "val_nll_sd",
                            "val_nll_min", "val_nll_max", "best_epochs", "n_epochs_run",
                            "ensemble_val_nll"]


def _merge_csv(path: Path, new: pd.DataFrame, columns: list[str]) -> None:
    """Replace the rows of the arms just trained, keep every other arm's rows.

    A partial run (``--stage stage2``) must not erase stage 1's history, and rerunning an
    arm must not leave two copies of it.
    """
    if new.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) == columns:
            new = pd.concat([old[~old["arm"].isin(set(new["arm"]))], new], ignore_index=True)
    new = new[columns]
    assert "empi_anon" not in new.columns, "outputs/ is aggregate only (protocol section 28)"
    new.to_csv(path, index=False)


def write_history(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=HISTORY_COLUMNS) if rows else pd.DataFrame(columns=HISTORY_COLUMNS)
    _merge_csv(Path(path), df, HISTORY_COLUMNS)


def write_seed_variability(path: Path, summaries: list[dict]) -> None:
    rows = [{"arm": s["arm"], "label": s["label"], "n_seeds": len(s["seeds"]),
             "val_nll_mean": s["val_nll_mean"], "val_nll_sd": s["val_nll_sd"],
             "val_nll_min": s["val_nll_min"], "val_nll_max": s["val_nll_max"],
             "best_epochs": str(s["best_epochs"]), "n_epochs_run": str(s["n_epochs_run"]),
             "ensemble_val_nll": s["ensemble_val_nll"]} for s in summaries]
    df = pd.DataFrame(rows, columns=SEED_VARIABILITY_COLUMNS) if rows \
        else pd.DataFrame(columns=SEED_VARIABILITY_COLUMNS)
    _merge_csv(Path(path), df, SEED_VARIABILITY_COLUMNS)


def write_train_arms(path: Path, *, summaries: list[dict], data: TrainingData,
                     settings: TrainSettings, device, amp: bool) -> dict:
    """The hand-over index src/eval_models.py reads. Per-patient arrays stay in the npz files."""
    path = Path(path)
    doc = read_json_retrying(path) if path.exists() else {}
    arms = dict(doc.get("arms", {}))
    for s in summaries:
        arms[s["arm"]] = s
    doc = {
        "module": MODULE,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interpreter": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": (torch.__version__ if TORCH_AVAILABLE else None),
            "timm": (timm.__version__ if TORCH_AVAILABLE else None),
            "numpy": np.__version__, "pandas": pd.__version__,
            "device": str(device), "amp": bool(amp),
        },
        "test_split": ("not read; the split != 'test' predicate is pushed into the Parquet "
                       "reader and the shard sidecar, and the sealed read path is not "
                       "implemented in src/train_model.py"),
        "grid": {"n_intervals": N_INTERVALS, "interval_days": INTERVAL_DAYS,
                 "edges": EDGES.tolist()},
        "horizons_days": data.contracts.horizons,
        "seeds": list(settings.seeds),
        "ensemble": {"rule": settings.ensemble_rule,
                     "note": "per-interval hazards averaged across seeds, converted to risk once"},
        "recalibration": {"form": RECALIBRATION_FORM,
                          "fitted_on": "validation only", "frozen": True},
        "cohort": {"development_rows": EXPECTED_DEV_ROWS,
                   "patients_with_crops": EXPECTED_DEV_PATIENTS_WITH_CROPS,
                   "crops": dict(EXPECTED_SPLIT_CROPS),
                   "max_crops_per_patient": data.max_elems},
        "clinical_standardization": {
            d: {"columns": st["columns"], "mean": [float(x) for x in st["mean"]],
                "std": [float(x) for x in st["std"]], "n_train": st["n_train"],
                "degenerate": st["degenerate"],
                "source": ("centering means from the frozen Cox JSON, scale from the TRAIN "
                           "standard deviation (protocol section 12)")}
            for d, st in data.clin_stats.items()},
        "training_contract": settings.training_contract(),
        "training_contract_hash": settings.contract_hash(),
        "arms": arms,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str))
    return doc


# =========================================================================== #
# 12. SMOKE TEST AND TIMING (plan verification step 2 / A3)                    #
# =========================================================================== #
def numpy_torch_nll_agreement(n: int = 64, seed: int = 7) -> tuple[float, float]:
    """Cell 33: the torch loss must match the hand-checked numpy reference to 1e-6."""
    require_torch()
    rng = np.random.default_rng(seed)
    lg = rng.normal(size=(n, N_INTERVALS)) * 2.0
    t = rng.uniform(0, GRID_MAX_DAYS, size=n)
    e = (rng.uniform(size=n) < 0.3).astype(int)
    ar, tg, _, _ = discretize_survival(t, e)
    h = 1.0 / (1.0 + np.exp(-lg))
    np_mean, np_per = dt_nll_numpy(h, ar, tg)
    pt_mean, pt_per = dt_nll_torch(torch.tensor(lg), torch.tensor(ar), torch.tensor(tg))
    return (abs(np_mean - float(pt_mean)),
            float(np.abs(np_per - pt_per.detach().numpy()).max()))


def run_smoke(data: TrainingData, settings: TrainSettings, device, amp: bool,
              log: logging.Logger, arm: str = "m4_fusion") -> int:
    """One batch forward/backward on the resolved device, plus the numpy/torch NLL check."""
    require_torch()
    spec = arm_spec(settings, arm)
    seed_everything(settings.seeds[0])
    tr = data.dataset("train", spec, train=True)
    net = SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=tr.clin.shape[1],
                            n_views=len(settings.views), mode=spec["mode"], arch=spec["arch"],
                            pretrained=True, view_emb_dim=settings.view_emb_dim,
                            base_hazard=base_hazard_from(tr),
                            head_init_scale=settings.head_init_scale,
                            dropout=settings.dropout).to(device)
    loader = make_loader(tr, min(settings.batch_size, 8), shuffle=True, seed=settings.seeds[0],
                         num_workers=0)
    batch = to_device(next(iter(loader)), device)
    log.info("batch shapes: %s", {k: tuple(v.shape) for k, v in batch.items()
                                  if torch.is_tensor(v)})
    log.info("views per patient in this batch: %s", batch["mask"].sum(1).tolist())

    t0 = time.time()
    with torch.amp.autocast(device.type, enabled=amp):
        logits, attn = net(batch, return_attention=True)
    loss, per = dt_nll_torch(logits.float(), batch["at_risk"], batch["target"])
    loss.backward()
    if device.type != "cpu":
        getattr(torch, device.type).synchronize()
    fwd = time.time() - t0
    log.info("logits %s | attention %s | loss %.6f | forward+backward %.2fs (first batch on a "
             "cold device also pays kernel compilation; see --time-steps for the real rate)",
             tuple(logits.shape), (tuple(attn.shape) if attn is not None else None),
             float(loss.detach()), fwd)
    assert logits.shape == (batch["at_risk"].shape[0], N_INTERVALS)
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    ref = dt_nll_numpy(np.tile(base_hazard_from(tr), (len(per), 1)),
                       batch["at_risk"].cpu().numpy(), batch["target"].cpu().numpy())[0]
    log.info("loss under the marginal-hazard model on the same batch = %.6f "
             "(the initialised head starts near this)", ref)

    watched = ("head.3.weight",) + (("view_emb.weight", "pool.w.weight")
                                    if net.use_image else ())
    grads = {n_: float(p.grad.abs().sum()) for n_, p in net.named_parameters()
             if p.grad is not None and n_.endswith(watched)}
    log.info("gradient reaches: %s", {k: f"{v:.3e}" for k, v in grads.items()})
    assert len(grads) == len(watched) and all(v > 0 for v in grads.values()), \
        "some branch is disconnected from the loss"
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    if net.use_image:
        a = attn.detach().float().cpu().numpy()
        m = batch["mask"].cpu().numpy()
        assert np.allclose(a.sum(1), 1.0, atol=1e-5) and float(np.abs(a * ~m).sum()) < 1e-8, \
            "attention leaked mass onto a padded slot"
        log.info("attention sums to 1 on every row; mass on padded slots %.3e",
                 float(np.abs(a * ~m).sum()))

    d_mean, d_per = numpy_torch_nll_agreement()
    log.info("torch NLL vs the hand-checked numpy reference (64 random patients): "
             "mean |diff| %.2e, per-patient max |diff| %.2e", d_mean, d_per)
    assert d_mean < 1e-6 and d_per < 1e-6, "the torch loss disagrees with the numpy reference"

    probe_augmentation(np.asarray(np.load(data.npy["train"], mmap_mode="r")[0]),
                       settings.augmentation, settings.border_px)
    log.info("augmentation caps %s; %d px border still exactly zero after the affine",
             {k: v for k, v in settings.augmentation.items() if k != "spec"}, settings.border_px)
    del net, batch, logits, attn, loss
    empty_device_cache(device)
    log.info("SMOKE TEST PASSED on device %s (amp=%s)", device, amp)
    return 0


def run_timing(data: TrainingData, settings: TrainSettings, device, amp: bool,
               log: logging.Logger, n_steps: int, arms: list[str]) -> int:
    """Time N real training steps on the real data pipeline and project the whole run."""
    require_torch()
    rows = []
    for arm in arms:
        spec = arm_spec(settings, arm)
        seed_everything(settings.seeds[0])
        tr = data.dataset("train", spec, train=True)
        va = data.dataset("val", spec, train=False)
        net = SurvivalFusionNet(n_intervals=N_INTERVALS, n_clinical=tr.clin.shape[1],
                                n_views=len(settings.views), mode=spec["mode"], arch=spec["arch"],
                                pretrained=True, view_emb_dim=settings.view_emb_dim,
                                base_hazard=base_hazard_from(tr),
                            head_init_scale=settings.head_init_scale,
                            dropout=settings.dropout).to(device)
        opt = torch.optim.AdamW(param_groups(net, settings), lr=settings.lr,
                                weight_decay=settings.weight_decay)
        loader = make_loader(tr, settings.micro_batch_size, shuffle=True, seed=settings.seeds[0],
                             num_workers=settings.num_workers,
                             pin_memory=(device.type == "cuda"))
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        sched = make_scheduler(opt, max(1, len(loader)), settings)
        net.train()
        # Three warm-up steps: the first forward on MPS pays kernel compilation, which does
        # not recur. The DataLoader's worker start-up DOES recur - torch keeps workers alive
        # across epochs only with persistent_workers=True, and macOS *spawns* rather than
        # forks, so every epoch pays a fresh import of torch in every worker. It is measured
        # separately below and added to the per-epoch estimate rather than warmed away.
        crops, slots, done, t_start, warm = 0, 0, 0, None, 3
        it = iter(loader)
        for step in range(n_steps + warm):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader); batch = next(it)
            b = to_device(batch, device)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = net(b)
            loss, _ = dt_nll_torch(logits.float(), b["at_risk"], b["target"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip_norm)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if device.type != "cpu":
                getattr(torch, device.type).synchronize()
            if step == warm - 1:
                t_start = time.time()
            elif step >= warm:
                done += 1
                crops += int(b["mask"].sum())
                slots += int(b["mask"].numel())
        train_secs = time.time() - t_start
        ms_step = 1000.0 * train_secs / max(done, 1)
        img_s = crops / max(train_secs, 1e-9)
        slot_s = slots / max(train_secs, 1e-9)

        # What a SECOND pass over the same loader costs before its first batch arrives:
        # this is the per-epoch worker start-up every epoch after the first actually pays.
        # (It overstates by one steady-state batch, which is the conservative direction.)
        del it
        t0 = time.time()
        it = iter(loader)
        next(it)
        loader_start_secs = time.time() - t0
        del it

        t0 = time.time()
        _ = predict_hazards(net, va, device=device, settings=settings, amp=amp)
        val_secs = time.time() - t0

        steps_per_epoch = len(loader)
        train_crops = int(sum(len(e) for e in tr.elems))
        epoch_secs = steps_per_epoch * (ms_step / 1000.0) + loader_start_secs + val_secs
        early_epochs = settings.patience + 8          # a plausible best_epoch + patience run
        rows.append({"arm": arm, "mode": spec["mode"], "arch": spec["arch"],
                     "views": ",".join(spec["views"]), "train_patients": len(tr),
                     "train_crops": train_crops, "steps_per_epoch": steps_per_epoch,
                     "ms_per_step": ms_step, "crops_per_s": img_s, "encoder_slots_per_s": slot_s,
                     "loader_start_secs": loader_start_secs,
                     "val_secs": val_secs, "epoch_secs": epoch_secs,
                     "hours_per_seed_max": settings.max_epochs * epoch_secs / 3600.0,
                     "hours_per_seed_early": early_epochs * epoch_secs / 3600.0,
                     "hours_per_arm_max": len(settings.seeds) * settings.max_epochs
                                          * epoch_secs / 3600.0,
                     "hours_per_arm_early": len(settings.seeds) * early_epochs
                                            * epoch_secs / 3600.0})
        log.info("%-20s %8.1f ms/step  %6.2f real crops/s (%6.2f encoder slots/s)  "
                 "%6.0f s/epoch (%d steps + %.0f s loader start-up + %.0f s val)  "
                 "%.2f h/seed at %d epochs  %.2f h/seed at %d epochs  %.2f h/arm worst case",
                 arm, ms_step, img_s, slot_s, epoch_secs, steps_per_epoch, loader_start_secs,
                 val_secs, rows[-1]["hours_per_seed_max"], settings.max_epochs,
                 rows[-1]["hours_per_seed_early"], early_epochs,
                 rows[-1]["hours_per_arm_max"])
        del net, opt, loader
        empty_device_cache(device)

    by_arm = {r["arm"]: r for r in rows}
    for stage, stage_arms in settings.stages.items():
        known = [a for a in stage_arms if a in by_arm]
        if not known:
            continue
        total = sum(by_arm[a]["hours_per_arm_max"] for a in known)
        log.info("%s projection from %d measured arm(s) %s: %.1f h worst case (all %d epochs, "
                 "%d seeds)", stage, len(known), known, total, settings.max_epochs,
                 len(settings.seeds))
    return 0


# =========================================================================== #
# 13. CLI                                                                     #
# =========================================================================== #
def resolve_arms(settings: TrainSettings, stage: str | None, arms: str | None) -> list[str]:
    if arms:
        want = [a.strip() for a in str(arms).split(",") if a.strip()]
    elif stage:
        assert stage in settings.stages, f"unknown stage {stage!r}; have {list(settings.stages)}"
        want = list(settings.stages[stage])
    else:
        want = [a for s in settings.stages.values() for a in s]
    unknown = [a for a in want if a not in settings.arms]
    assert not unknown, f"unknown arm(s) {unknown}; have {sorted(settings.arms)}"
    return want


def main(argv=None) -> int:                                  # noqa: C901 - one linear script
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--stage", default=None, help="stage1 or stage2 (model_image.local.stages)")
    ap.add_argument("--arms", default=None, help="comma-separated arm keys")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: all five)")
    ap.add_argument("--smoke", action="store_true",
                    help="one batch forward/backward plus the numpy/torch NLL check, then stop")
    ap.add_argument("--time-steps", type=int, default=0,
                    help="time N real training steps and project the run, then stop")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="split the 32-patient batch into this many forward passes; the "
                         "optimisation batch is unchanged. Never downscale the 512px crop.")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader workers; each one is a spawned process that imports "
                         "torch, so on a 24 GB machine fewer can be faster")
    ap.add_argument("--force-retrain", action="store_true",
                    help="ignore existing checkpoints (records a protocol deviation)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    require_torch()
    settings = TrainSettings(cfg, max_epochs_override=args.max_epochs,
                             batch_size_override=args.batch_size,
                             grad_accum_override=args.grad_accum,
                             num_workers_override=args.num_workers)
    device, amp = resolve_device(settings.device_preference, settings.amp_devices)
    log.info("device %s (preference %s) | AMP %s (amp_devices %s) | torch %s | workers %d",
             device, settings.device_preference, amp, settings.amp_devices, torch.__version__,
             settings.num_workers)
    log.info("batch %d patients = %d micro-batch x %d accumulation steps | max_epochs %d | "
             "patience %d | lr %g | wd %g | clip %g | warmup %d",
             settings.batch_size, settings.micro_batch_size, settings.grad_accum_steps,
             settings.max_epochs, settings.patience, settings.lr, settings.weight_decay,
             settings.grad_clip_norm, settings.warmup_epochs)
    log.info("training contract hash %s", settings.contract_hash())

    seeds = ([int(s) for s in str(args.seeds).split(",")] if args.seeds else list(settings.seeds))
    unknown = [s for s in seeds if s not in settings.seeds]
    assert not unknown, (f"seed(s) {unknown} are not in the five pre-specified "
                         f"model_image.local.seeds {settings.seeds}")

    seed_everything(int(cfg["reproducibility"]["random_seed"]))
    data = TrainingData(cfg, settings, log=log)
    log.info("development design %s; max crops per patient %d",
             data.designs["m0"].shape, data.max_elems)

    arms = resolve_arms(settings, args.stage, args.arms)

    if args.smoke:
        # `resolve_arms` falls back to the WHOLE ladder, so `arms` is never empty and
        # `arms[0]` is m0d_clinical - a clinical-only arm with no encoder, no view
        # embedding and no attention pool. That is exactly the half of run_smoke that
        # lives under `if net.use_image`, so a bare `--smoke` would have reported PASS
        # without ever touching the image path it exists to check. Only an EXPLICIT
        # --arms/--stage moves the smoke test off the primary fusion arm.
        return run_smoke(data, settings, device, amp, log,
                         arm=(arms[0] if (args.arms or args.stage) else "m4_fusion"))
    if args.time_steps:
        return run_timing(data, settings, device, amp, log, int(args.time_steps), arms)

    if args.force_retrain:
        log.warning("--force-retrain is set: existing checkpoints are ignored. Record this in "
                    "outputs/protocol_deviations.md.")

    summaries, history_rows = [], []
    for arm in arms:
        res = run_arm(arm, data=data, settings=settings, device=device, amp=amp, seeds=seeds,
                      force_retrain=args.force_retrain, log=log)
        summaries.append(res["summary"]); history_rows += res["history_rows"]
        write_history(settings.history_csv, history_rows)
        write_seed_variability(settings.seed_variability_csv, summaries)
        write_train_arms(data.cohort_dir / "train_arms.json", summaries=summaries, data=data,
                         settings=settings, device=device, amp=amp)

    L = []; A = L.append
    A("")
    A("=" * 88)
    A(f"{MODULE}: {len(summaries)} arm(s) trained, {len(seeds)} seeds each".center(88))
    A("=" * 88)
    for s in summaries:
        A(f"  {s['arm']:22s} val NLL {s['val_nll_mean']:.6f} (SD {s['val_nll_sd']:.6f}) "
          f"-> ensemble {s['ensemble_val_nll']:.6f}  best epochs {s['best_epochs']}")
    A(f"  history            {settings.history_csv}")
    A(f"  seed variability   {settings.seed_variability_csv}")
    A(f"  hand-over index    {data.cohort_dir / 'train_arms.json'}")
    A("  the test split was not read")
    A("=" * 88)
    for line in L:
        log.info("%s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
