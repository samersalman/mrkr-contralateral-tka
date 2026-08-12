"""eval_models.py - evaluation of the model ladder on one split (protocol sections 18/21).

Phase 2 / Track B, step 2. ``src/train_model.py`` trains, ensembles and freezes a
horizon-specific recalibration; it deliberately does not evaluate. This module is the
single place where the ladder is *scored*: discrimination, calibration, the paired
patient-level bootstrap, the pre-specified contrasts and the subgroup audit.

Which split, and what that costs
--------------------------------
``--split val`` (the default) is the development path and is guarded exactly as it always
was: ``model_eval.forbid_test_split: true`` is re-asserted, and every reader routes through
:func:`assert_validation_only`, which reuses ``src.train_model.assert_development_splits``
(so there is exactly one predicate in the repository that decides what "not the test set"
means) and then narrows further: on this path the module may hold the VALIDATION split and
nothing else, not even train. The ``split != "test"`` predicate itself is pushed down into
the Parquet reader by ``src.model_clinical.load_development_frame``, so a sealed row is
never materialised in memory.

``--split test`` is THE SEALED READ, and it **has now been performed**: it consumes the
``test_hazards_*.npz`` written by ``src/score_test.py`` (whose ``test_scoring.json`` records
``sealed_read: PERFORMED`` under the frozen training-contract hash) and writes the
``test_*`` siblings of every output below, beside the validation ones and never over them.
Those estimates took no part in early stopping, checkpoint selection or recalibration and
are the only unbiased numbers in the study. :func:`assert_sealed_read_is_recorded` is the
guard a downstream renderer calls before reporting them.

What is computed, and on whom
-----------------------------
* Horizons **365 / 730 / 1825 days**, from ``model_eval.horizons_days``. The nominal
  5-year horizon is 1826, which is exactly where administrative censoring lands, so the
  cumulative/dynamic control set ``T > t`` is empty there; 1825 is the clamped value the
  frozen Cox contracts already use.
* Per arm: IPCW cumulative/dynamic AUROC at each horizon, Harrell's C, Uno's C,
  calibration slope, calibration-in-the-large (observed Kaplan-Meier risk minus mean
  predicted risk, the same definition the frozen M0 JSON carries) and the IPCW Brier
  score.
* **Every arm shares M0's frozen ``censoring_km_train``.** The IPCW weights are therefore
  identical across the ladder and a difference between two arms is a difference in the
  models, never in the weighting. (M1's own JSON carries a censoring curve fitted on the
  KLG-eligible training rows; it is deliberately NOT used, so the ``m1`` row here can
  differ in the last decimals from ``m1_klg_model.json``'s published numbers.)
* Arms differ in **who they score**. ``m1``/``m1_klg`` run on the KLG-eligible subset
  (``model_eval.subset_arms``), and protocol section 20 drops any patient with no crop in
  an arm's view set. A paired contrast therefore intersects the two arms' patient sets and
  reports ``n_paired`` honestly; both point estimates in a contrast row are recomputed on
  that intersection, so the difference is never a difference of two different cohorts.
* **Decision-curve analysis** (protocol section 18, D29). Section 7b holds the
  censoring-aware net-benefit estimator: at each threshold the flagged set's event risk is
  estimated by Kaplan-Meier at 1825 days, so the true- and false-positive terms come from
  one fit and no censored patient is dropped or double-counted. It matters here because
  63.8% of the test split is censored before the horizon and the naive event rate (14.3%)
  understates the 5-year risk (20.0%) by 5.7 percentage points - which is exactly where
  treat-all crosses zero. Section 7c drives those pure functions from the SAME shared draw
  through :class:`NetBenefitEngine` - a second engine, never a metric folded into
  :class:`BootstrapEngine` - and writes ``{split}_net_benefit.csv``. Its intervals are
  POINTWISE and unadjusted across thresholds: the 35 flagged sets are nested, so the grid
  is 35 views of one curve rather than 35 hypotheses.

The bootstrap (protocol section 18)
-----------------------------------
``model_eval.bootstrap_n`` = 2,000 replicates, seeded from ``model_eval.bootstrap_seed``.
This deliberately does NOT inherit ``model_clinical.bootstrap_n: 500``, which is a
development-report value. **One shared draw**: a single ``(n_boot, n_val)`` matrix of
patient positions is drawn once and every arm and every contrast is evaluated on it, so
each replicate scores all arms on the same patients and the per-replicate difference is
already paired. An arm that does not score a drawn patient simply drops it from that
replicate; the draw itself is never re-rolled.

Reporting rules honoured
------------------------
* **Section 21**: a subgroup level with fewer than ``model_eval.suppress_below_events``
  (50) events has its point estimate suppressed (NaN) with ``suppressed=True`` and a
  reason string. The predicate is ``n_events < threshold``, applied level by level to the
  patients the scored arm actually covers, BEFORE anything is estimated. How many levels
  that suppresses is decided by the data and is deliberately not asserted anywhere in this
  module or in its docstrings: the validation split carries only 54 events in total, so a
  level can clear the floor only by holding almost the entire split, which makes it the
  all-patient estimate under another name rather than a group contrast. A suppressed level
  is the correct output, not a failure, and it is reported as such rather than as noise.
* Secondary contrasts carry Benjamini-Hochberg adjusted p values computed **within each
  declared family separately**; families are never pooled. The pre-specified primary
  contrast is unadjusted (``p_adjusted`` NaN, ``fdr_method`` empty, ``is_primary`` True).
* **outputs/ is aggregate only** (protocol section 28): no ``empi_anon``, no DICOM UID, no
  per-patient row. Asserted on every frame before it is written. Per-patient arrays stay in
  ``derived-data/``.

Inputs
------
``derived-data/cohort/train_arms.json`` - the hand-over index written by
``src/train_model.py``; absent means training has not run and this module fails with the
command to run. Per arm it carries ``recalibration`` (frozen, fitted on validation),
``hazards_npz``, ``seeds``, ``label`` and the arm's cohort counts.
``derived-data/cohort/val_hazards_{arm}.npz`` - the per-patient ensembled hazards. An arm
listed in the index whose npz is absent is **skipped and logged**, so this module produces
partial, honest output after stage 1 and before stage 2 exists.
``derived-data/cohort/m0_clinical_model.json`` / ``m1_klg_model.json`` - the frozen Cox
comparators, replayed from JSON (never refitted) onto the validation rows.

Outputs (schemas pinned; ``src/make_manuscript.py`` consumes the first three)
----------------------------------------------------------------------------
``outputs/tables/val_metrics.csv``     one row per arm (``m0``, ``m1``, then the ladder)
``outputs/tables/val_comparisons.csv`` the primary contrast then each declared family
``outputs/tables/val_subgroups.csv``   the section-21 equity audit
``outputs/tables/val_convergence.csv`` the D28 gate's per-arm verdict
``outputs/tables/val_net_benefit.csv`` the decision curve: one row per declared arm per
threshold, read by ``src/manuscript_figures.py`` to draw Figure 4
``derived-data/cohort/val_results.json`` the run header (cohort, horizons, bootstrap,
recalibration). It is the only output allowed a timestamp: the CSVs carry none, so two
consecutive runs are byte-identical.

On ``--split test`` each of those paths is rewritten to its ``test_`` sibling by
:func:`split_path`, which is the ONE implementation of that rewrite and is imported by
``src/manuscript_figures.py`` and ``src/make_manuscript.py`` so the writer and both readers
cannot disagree about a filename.

Run::

    ~/.venvs/mrkr-torch/bin/python -m src.eval_models --config config/feasibility.yaml
        [--train-arms PATH] [--bootstrap-n N] [--arms m4_fusion,m3_image]

torch is not imported here and is not needed; the estimators are numpy/lifelines. The
venv interpreter is named above only because ``src.train_model`` (from which the numeric
primitives are imported, not copied) attempts a guarded ``import torch``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.model_clinical import (  # the REAL estimators, re-exported by train_model too
    SEALED_SPLIT,
    SUPPRESS_BELOW_EVENTS,
    _ci,
    _f,
    _md_table,
    calibration_slope_intercept,
    ipcw_auc,
    ipcw_labels_weights,
    km_risk,
    load_development_frame,
    percentile_ci,
)
from src.subgroups import (  # the ONE subgroup-family declaration and its parser
    Family,
    family_mask,
    load_families,
)
from src.train_model import (  # numeric primitives: imported, never reimplemented
    EDGES,
    EXPECTED_DEV_PATIENTS_WITH_CROPS,
    EXPECTED_DEV_ROWS,
    EXPECTED_HORIZONS,
    FrozenContracts,
    apply_recalibration,
    assert_development_splits,
    build_clinical_design,
    harrell_c,
    ipcw_brier,
    km_cif_numpy,
    load_sidecar,
    read_json_retrying,
    replay_cox,
    risk_at_horizon,
    risk_score,
    uno_c,
)

MODULE = "eval_models"

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS - deliberately NOT in config.                               #
# A config edit must not be able to weaken a guard or move a cohort anchor.    #
# --------------------------------------------------------------------------- #
VAL_SPLIT = "val"                      # the ONLY split this module may ever hold
EXPECTED_VAL_PATIENTS = 371            # src/splits.py, outputs/tables/split_summary.csv
EXPECTED_VAL_EVENTS = 54
COX_ARMS = ("m0", "m1")                # the frozen penalized-Cox comparators
COX_DESIGN = {"m0": "m0", "m1": "m1"}
COX_LABELS = {"m0": "M0 clinical only (frozen penalized Cox)",
              "m1": "M1 clinical plus inferred KLG (frozen penalized Cox)"}

# A replicate that draws fewer than two events cannot support a C-index or an AUROC; it is
# skipped (recorded as NaN and excluded from the percentile interval), exactly as
# src/model_clinical.py does, rather than silently contributing a degenerate value.
MIN_EVENTS_PER_REPLICATE = 2
MIN_PATIENTS_PER_REPLICATE = 2

# Protocol Table 7 pre-specifies 2,000 replicates for the primary paired patient-level
# bootstrap. It is anchored here, not read as "whatever config says", so the module cannot
# silently inherit ``model_clinical.bootstrap_n: 500`` - which is a development-report value
# the trainer and this evaluator are both told not to take. ``--bootstrap-n`` overrides the
# value actually used for a fast smoke run without editing (or weakening) the anchor.
PROTOCOL_BOOTSTRAP_N = 2000

PRIMARY_METRIC = "auc"                 # the primary estimand is IPCW cumulative/dynamic AUROC
FDR_METHODS = {"bh": "Benjamini-Hochberg"}
ROUND_DECIMALS = 6                     # fixed rounding -> byte-identical CSVs across runs

# Column names that would carry an identifier into outputs/ (protocol section 28).
FORBIDDEN_OUTPUT_COLUMN_TOKENS = ("empi", "uid", "mrn", "accession", "patient_id", "pid")

# What the run header may truthfully say about the sealed split, keyed BY THE SPLIT THAT
# WAS EVALUATED. A caller selects its text with ``TEST_SPLIT_STATEMENT[split]``; there is no
# split-free default, because a single string is exactly how test_results.json came to
# announce "SEALED, never loaded" about a read that had just been performed.
# The ``val`` text is frozen byte-for-byte: on the development path nothing about the sealed
# split changed, so the validation artefacts must stay reproducible to the byte.
TEST_SPLIT_STATEMENT = {
    VAL_SPLIT: (
        "SEALED, never loaded. The split != 'test' predicate is pushed into the Parquet reader "
        "and every arm's patient set is asserted to be a subset of the validation roster; the "
        "sealed read path is not implemented in src/eval_models.py."),
    SEALED_SPLIT: (
        "READ. This file reports the single permitted read of the locked test split; "
        "derived-data/cohort/test_scoring.json records it as sealed_read: PERFORMED under the "
        "same training-contract hash that train_arms.json still carries. The test split took no "
        "part in early stopping, checkpoint selection or recalibration, so these estimates are "
        "unbiased out of sample; src.eval_models.assert_sealed_read_is_recorded makes any later "
        "change to those models fail the render rather than be reported as an out-of-sample "
        "result."),
}


def setup_logging(log_path: Path) -> logging.Logger:
    """House logging idiom (src/splits.py, src/subgroups.py, src/model_clinical.py).

    Deliberately defined here rather than imported from ``src.train_model``: that one hard
    codes its own ``MODULE`` prefix, so every line this module wrote would be attributed to
    the trainer in ``outputs/logs/run.log``. Every module in this repository owns its
    prefix; the numeric primitives are what is shared, not the log header.
    """
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


# =========================================================================== #
# 1. SEALED-SPLIT GUARD AND THE COHORT ANCHORS                                #
# =========================================================================== #
def assert_validation_only(splits) -> list[str]:
    """Protocol sections 12/17: refuse the sealed test split, and refuse train as well.

    ``src.train_model.assert_development_splits`` is REUSED rather than re-implemented, so
    the repository has exactly one definition of "not the sealed split". This function then
    narrows it: evaluation happens on validation, and a request for train rows here is a
    programming error worth failing on rather than a harmless superset.
    """
    out = assert_development_splits(splits)          # refuses "test" with the protocol message
    unexpected = sorted(set(out) - {VAL_SPLIT})
    assert not unexpected, (
        f"REFUSED: {MODULE} was asked for split(s) {unexpected}. This module evaluates the "
        f"{VAL_SPLIT!r} split only; training-split metrics belong to src/train_model.py and "
        f"the sealed {SEALED_SPLIT!r} split is reachable ONLY through --split {SEALED_SPLIT}, "
        f"which is a separate, logged, one-shot path and never something a caller of this "
        f"function may reach into.")
    return out


def assert_forbid_test_split_is_on(cfg: Config) -> None:
    """``model_eval.forbid_test_split`` is a declaration, and it must still be true."""
    flag = cfg["model_eval"].get("forbid_test_split")
    assert bool(flag) is True, (
        "config model_eval.forbid_test_split is not true; this module refuses to run the "
        f"development path with the sealed-split guard switched off, and switching it off "
        f"would not unseal anything anyway: the sealed read is reached only by asking for it "
        f"with --split {SEALED_SPLIT}, which does not consult this flag")


SEALED_READ_RECORD = "test_scoring.json"   # written by src/score_test.py, read never written here
SEALED_READ_PERFORMED = "PERFORMED"        # the leading token score_test.py records, verbatim


def assert_sealed_read_is_recorded(cfg: Config) -> str:
    """Refuse to report a sealed-split result unless the sealed read is on the record.

    Called by the downstream render modules (``src/manuscript_figures.py``,
    ``src/make_manuscript.py``) and ONLY when the split being rendered is the sealed one.
    Two facts are checked against the artefacts themselves rather than against any constant
    in this file:

    * ``derived-data/cohort/test_scoring.json`` records ``sealed_read`` beginning
      ``PERFORMED``, so the numbers about to be rendered came from a read that actually
      happened, not from a stale, partial or hand-edited file;
    * the ``training_contract_hash`` frozen into that record is STILL equal to the one in
      ``train_arms.json``, so the models scored on the sealed split are the models the
      document is about to describe.

    **This is stronger than** ``model_eval.forbid_test_split``, and deliberately so. That
    flag only stops the sealed split being READ; it says nothing once the read has happened.
    Retraining, re-tuning or re-freezing an arm after the sealed read and then re-rendering
    the manuscript passes every other check in this repository, and would publish an
    out-of-sample claim for a model that was never scored out of sample. That is exactly the
    failure a sealed protocol exists to prevent, and nothing currently catches it. A
    post-hoc model change moves ``train_arms.json``'s contract hash away from the one frozen
    into ``test_scoring.json``, and this function turns that into a failed render.

    Returns the verified contract hash so a caller can print it as provenance.
    """
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    rec_path, arms_path = coh / SEALED_READ_RECORD, coh / "train_arms.json"
    if not rec_path.exists():
        raise FileNotFoundError(
            f"{MODULE}: {rec_path} does not exist, so the single permitted read of the "
            f"{SEALED_SPLIT!r} split is not on the record and nothing may be rendered from it. "
            f"Either render the {VAL_SPLIT!r} split, or perform the sealed read with\n"
            f"    ~/.venvs/mrkr-torch/bin/python -m src.score_test "
            f"--config config/feasibility.yaml --confirm-sealed-read\n"
            f"which is a signed-off, one-shot operation and not something to repeat casually.")
    if not arms_path.exists():
        raise FileNotFoundError(
            f"{MODULE}: {arms_path} does not exist, so there is nothing to check "
            f"{rec_path.name}'s frozen training contract against; a sealed-split result cannot "
            f"be shown to describe the models that are still on disk.")

    record = read_json_retrying(rec_path)
    train_arms = read_json_retrying(arms_path)

    performed = str(record.get("sealed_read") or "")
    assert performed.startswith(SEALED_READ_PERFORMED), (
        f"REFUSED: {rec_path} does not record sealed_read as {SEALED_READ_PERFORMED} (it says "
        f"{performed!r}), so there is no evidence the locked {SEALED_SPLIT!r} split was ever "
        f"actually read. Nothing may be rendered as an out-of-sample result on that basis.")

    scored_hash = str(record.get("training_contract_hash") or "")
    frozen_hash = str(train_arms.get("training_contract_hash") or "")
    blank = [p.name for p, h in ((rec_path, scored_hash), (arms_path, frozen_hash)) if not h]
    assert not blank, (
        f"REFUSED: {' and '.join(blank)} carries no training_contract_hash, so it cannot be "
        f"shown that the models scored on the {SEALED_SPLIT!r} split are the models this "
        f"render describes.")
    assert scored_hash == frozen_hash, (
        f"REFUSED: {rec_path.name} records that the {SEALED_SPLIT!r} split was scored on models "
        f"frozen under training contract {scored_hash}, but {arms_path.name} now carries "
        f"{frozen_hash}. The models CHANGED AFTER the single permitted sealed read, so "
        f"rendering these numbers would publish an out-of-sample result for a model that was "
        f"never scored out of sample. This guard is stronger than "
        f"model_eval.forbid_test_split on purpose: that flag only stops the sealed split being "
        f"read and is silent about everything that happens afterwards, so a post-hoc model "
        f"change is the one failure a sealed protocol exists to prevent that nothing else here "
        f"catches. Restore the configuration and checkpoints that produced {scored_hash}, or "
        f"obtain sign-off for a second sealed read; do not re-render against a moved contract.")
    return scored_hash


def measured_development_rows(contracts: FrozenContracts) -> int:
    """MEASURE the development rows in ``features_clinical.parquet``. Never a constant.

    :func:`assert_development_anchors` is only worth calling on a number that came from the
    data, so the count is taken here from the Parquet itself, with ``split != 'test'`` pushed
    into the reader (so no sealed row is materialised) by the same
    :func:`src.model_clinical.load_development_frame` the trainer uses.
    """
    return int(len(load_development_frame(contracts.features_pq, forbid_test=True)))


def measured_patients_with_crops(cfg: Config, log: logging.Logger) -> int | None:
    """MEASURE the development patients that carry a crop, from the shard sidecar itself.

    ``src.train_model.load_sidecar`` is REUSED rather than re-implemented, so ``labels.csv``
    is checked by exactly the predicate that gated training: the per-split crop counts and
    the 2,966 distinct patients. Nothing this repository wrote is consulted.

    Returns ``None`` when the shard directory is not mounted on this machine. Evaluation
    reads per-patient hazard npz files, not shards, so a missing sidecar must not stop it,
    and the anchor is not silently dropped either: no hazard file can exist unless
    ``src/train_model.py`` passed the same assertion when it produced them. That case is
    logged as not re-measured rather than reported as checked.
    """
    sidecar = cfg.path(cfg["model_image"]["local"]["shard_dir"]) / "labels.csv"
    if not sidecar.exists():
        log.warning("shard sidecar %s is absent, so the %d patients-with-crops anchor is NOT "
                    "re-measured here; it was enforced against labels.csv by "
                    "src/train_model.py::load_sidecar before any hazard file could be written",
                    sidecar, EXPECTED_DEV_PATIENTS_WITH_CROPS)
        return None
    return int(load_sidecar(sidecar.parent)["empi_anon"].nunique())


def assert_development_anchors(n_dev_rows: int, train_arms: dict | None,
                               log: logging.Logger,
                               measured_with_crops: int | None = None) -> None:
    """Plan verification step 5: the development cohort has not moved under us.

    Two anchors, both frozen in ``src/train_model.py`` so a config edit cannot weaken them:
    ``EXPECTED_DEV_ROWS`` = 2,968 rows in the clinical feature table (train 2,597 + val 371),
    and ``EXPECTED_DEV_PATIENTS_WITH_CROPS`` = 2,966 patients that actually have a crop and
    are therefore scoreable by an image arm. The two differ by the two patients with no
    usable crop; both matter because a change in either invalidates every paired comparison
    against M0 (protocol section 17).

    **What each argument is worth.** ``n_dev_rows`` must be a MEASUREMENT of the feature
    table, from :func:`measured_development_rows`; handing this function
    ``EXPECTED_DEV_ROWS`` would compare the constant with itself and certify nothing.
    ``measured_with_crops`` is likewise a measurement, from :func:`measured_patients_with_crops`
    reading the shard sidecar, and is the only argument that can genuinely falsify the 2,966
    anchor here; it is ``None`` when the shards are not mounted, in which case that anchor is
    reported as not re-measured. ``train_arms`` is weaker on purpose: ``train_model.py``
    writes ``cohort.patients_with_crops`` FROM the same constant, so checking it is a
    provenance test on the hand-over index (it catches an index written by a build carrying a
    different anchor) and is described that way in the log rather than as a cohort check.
    """
    assert n_dev_rows == EXPECTED_DEV_ROWS, (
        f"the development feature table holds {n_dev_rows} rows, not {EXPECTED_DEV_ROWS}; the "
        f"locked cohort moved and every paired comparison against M0 is invalidated "
        f"(protocol section 17)")
    if measured_with_crops is not None:
        assert int(measured_with_crops) == EXPECTED_DEV_PATIENTS_WITH_CROPS, (
            f"the shard sidecar carries crops for {measured_with_crops} development patients, "
            f"not {EXPECTED_DEV_PATIENTS_WITH_CROPS}; the image cohort moved after the splits "
            f"were locked and no paired comparison against M0 is on the same patients")
    stated = None
    if train_arms:
        stated = (train_arms.get("cohort") or {}).get("patients_with_crops")
    if stated is not None:
        assert int(stated) == EXPECTED_DEV_PATIENTS_WITH_CROPS, (
            f"train_arms.json reports {stated} development patients with crops, not "
            f"{EXPECTED_DEV_PATIENTS_WITH_CROPS}; the hand-over index was written by a build "
            f"whose image cohort moved after the splits were locked")
    log.info("cohort anchors: %d development feature rows MEASURED against %d; patients with "
             "crops %s; hand-over index %s (protocol section 17)",
             n_dev_rows, EXPECTED_DEV_ROWS,
             (f"{measured_with_crops} MEASURED from the shard sidecar against "
              f"{EXPECTED_DEV_PATIENTS_WITH_CROPS}") if measured_with_crops is not None
             else "NOT re-measured here (enforced in src/train_model.py::load_sidecar)",
             f"states {stated}" if stated is not None else "states nothing")


def horizons_from_config(cfg: Config) -> list[int]:
    """The evaluation horizons, from config, checked against the frozen Cox contracts."""
    hz = [int(h) for h in cfg["model_eval"]["horizons_days"]]
    assert [float(h) for h in hz] == EXPECTED_HORIZONS, (
        f"model_eval.horizons_days is {hz}, but the frozen M0/M1 contracts were built at "
        f"{EXPECTED_HORIZONS}. 1826 is where administrative censoring lands, so the T > t "
        f"control set is empty there and 1825 is the clamped 5-year horizon.")
    return hz


# =========================================================================== #
# 2. THE VALIDATION ROSTER - one fixed row order every arm is aligned to       #
# =========================================================================== #
@dataclass
class Roster:
    """The 371 validation patients in a fixed order, with the shared IPCW curve.

    Every arm's per-patient array is mapped onto these positions, so "the same patients"
    is a fact about integer indices rather than a hope about join order. The bootstrap
    draws positions into this roster once and every arm reuses the draw.
    """

    pids: np.ndarray                 # (n,) unicode, ascending - the join key
    time: np.ndarray                 # (n,) time_from_landmark
    event: np.ndarray                # (n,) event_indicator
    frame: pd.DataFrame              # the split's rows of the feature table, same order
    g_grid: np.ndarray               # M0's FROZEN censoring curve, shared by every arm
    g_vals: np.ndarray
    label: str = "validation"        # names the split in assertion messages

    def __len__(self) -> int:
        return int(self.pids.size)

    def positions_of(self, pids) -> np.ndarray:
        """Roster positions for an arm's patient ids. Any id outside the roster raises."""
        want = np.asarray(pids).astype(str)
        pos = np.searchsorted(self.pids, want)
        assert pos.max(initial=-1) < self.pids.size and np.all(self.pids[np.clip(pos, 0, self.pids.size - 1)] == want), (
            f"an arm scored a patient that is not in the {self.label} roster; that is either a "
            "join-order bug or a SEALED SPLIT VIOLATION")
        return pos.astype(np.int64)


def load_roster(contracts: FrozenContracts, log: logging.Logger,
                split: str = VAL_SPLIT) -> tuple[Roster, pd.DataFrame]:
    """The roster for ONE split plus M0's design matrix on it.

    ``build_clinical_design`` is reused verbatim: it replays the FROZEN imputer and returns
    the frame in ``empi_anon`` order, which is exactly the row order ``src/train_model.py``
    and ``src/score_test.py`` wrote into every ``*_hazards_{arm}.npz``.

    ``split`` defaults to validation. Passing the sealed split is the one-shot read and is
    only reached through ``--split test``, which the module logs loudly; the frozen
    censoring curve still comes from M0's TRAIN-estimated ``censoring_km_train``, so IPCW
    weights are identical to the ones every validation number used.
    """
    sealed = (split == SEALED_SPLIT)
    if not sealed:
        assert_validation_only([split])
    frame, X = build_clinical_design(contracts, (split,), design="m0", allow_sealed=sealed)
    assert (frame["split"] == split).all(), f"a non-{split} row reached the roster"
    if not sealed:
        assert len(frame) == EXPECTED_VAL_PATIENTS, (
            f"validation split holds {len(frame)} patients, not {EXPECTED_VAL_PATIENTS}")
        n_ev = int(frame["event_indicator"].sum())
        assert n_ev == EXPECTED_VAL_EVENTS, (
            f"validation split holds {n_ev} events, not {EXPECTED_VAL_EVENTS}")
    n_ev = int(frame["event_indicator"].sum())
    pids = frame["empi_anon"].astype(str).to_numpy()
    assert np.all(pids[:-1] < pids[1:]), "the roster must be strictly ascending in empi_anon"
    roster = Roster(pids=pids,
                    time=frame["time_from_landmark"].to_numpy(float),
                    event=frame["event_indicator"].to_numpy(int),
                    frame=frame.reset_index(drop=True),
                    g_grid=contracts.g_grid, g_vals=contracts.g_vals,
                    label=(SEALED_SPLIT if sealed else "validation"))
    log.info("%s roster: %d patients, %d events; IPCW weights come from M0's frozen "
             "censoring_km_train (%d steps), shared by every arm",
             split, len(roster), n_ev, roster.g_grid.size)
    return roster, X


# =========================================================================== #
# 3. ONE ARM'S PER-PATIENT SCORES, ALIGNED TO THE ROSTER                       #
# =========================================================================== #
@dataclass
class ArmScores:
    """Everything needed to score one arm, on roster positions.

    ``present`` is the protocol section 20 / subset-arm reality: False for a patient this
    arm cannot score (no crop in its view set, or not KLG-eligible). ``rank`` and ``risk``
    hold NaN there and are never read, because every consumer filters by ``present`` first.
    """

    arm: str
    label: str
    source: str                       # "frozen_cox" | "discrete_time"
    present: np.ndarray               # (n_roster,) bool
    rank: np.ndarray                  # (n_roster,) horizon-free ranking score
    risk: dict[int, np.ndarray]       # horizon days -> (n_roster,) predicted risk
    val_nll: float = float("nan")
    recalibration: dict | None = None
    subset_arm: bool = False
    n_no_crop_dropped: int = 0
    seeds: list[int] = field(default_factory=list)

    @property
    def n_patients(self) -> int:
        return int(self.present.sum())


def _blank(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def cox_arm_scores(arm: str, contracts: FrozenContracts, roster: Roster, horizons: list[int],
                   log: logging.Logger, split: str = VAL_SPLIT) -> ArmScores:
    """A frozen penalized-Cox comparator, REPLAYED from its JSON onto ``split``'s rows.

    Nothing is refitted. ``m1`` additionally restricts to the KLG-eligible subset, because
    protocol Secondary objective 2 defines it there and nothing here may impute a
    radiographic severity grade.
    """
    design = COX_DESIGN[arm]
    frame, X = build_clinical_design(contracts, (split,), design=design,
                                     allow_sealed=(split == SEALED_SPLIT))
    lp, risk = replay_cox(contracts, X, design)
    pos = roster.positions_of(frame["empi_anon"].astype(str).to_numpy())
    n = len(roster)
    present = np.zeros(n, dtype=bool); present[pos] = True
    rank = _blank(n); rank[pos] = np.asarray(lp, dtype=float)
    out_risk: dict[int, np.ndarray] = {}
    for h in horizons:
        col = _blank(n); col[pos] = np.asarray(risk[float(h)], dtype=float)
        out_risk[h] = col
    # The frozen Cox risks are published as fitted; no recalibration is applied to them.
    log.info("arm %-20s frozen Cox replay: %d/%d %s patients, %d events%s",
             arm, int(present.sum()), n, split, int(roster.event[present].sum()),
             " (KLG-eligible subset)" if design == "m1" else "")
    return ArmScores(arm=arm, label=COX_LABELS[arm], source="frozen_cox", present=present,
                     rank=rank, risk=out_risk, subset_arm=(design == "m1"))


def trained_arm_scores(arm: str, summary: dict, npz_dir: Path, roster: Roster,
                       horizons: list[int], log: logging.Logger,
                       split: str = VAL_SPLIT) -> ArmScores | None:
    """One trained arm from its ``val_hazards_{arm}.npz``. Returns None if it is not there.

    A missing npz means that arm has not finished training. It is logged and skipped, so
    this module runs and produces honest partial output after stage 1, before stage 2
    exists. Everything else about the npz is a contract and is asserted.
    """
    # On the sealed read the hand-over index still names the VALIDATION npz, because it
    # was written at training time. The split being scored decides the file, not the index.
    name = (f"{split}_hazards_{arm}.npz" if split != VAL_SPLIT
            else str(summary.get("hazards_npz") or f"val_hazards_{arm}.npz"))
    path = Path(npz_dir) / name
    if not path.exists():
        log.warning("arm %-20s SKIPPED: %s is absent, so this arm has not been trained yet; "
                    "its rows are omitted from every output", arm, path)
        return None
    with np.load(path, allow_pickle=False) as z:
        stored_arm = str(z["arm"].item()) if "arm" in z else arm
        assert stored_arm == arm, (
            f"{path} carries arm {stored_arm!r} but is indexed under {arm!r} in train_arms.json")
        hazards = np.asarray(z["hazards"], dtype=float)
        pids = np.asarray(z["empi_anon"]).astype(str)
        t_npz = np.asarray(z["time"], dtype=float)
        e_npz = np.asarray(z["event"], dtype=int)
        edges = np.asarray(z["edges"], dtype=float)
    assert hazards.ndim == 2 and hazards.shape[0] == pids.size, (
        f"{path}: hazards {hazards.shape} do not line up with {pids.size} patient ids")
    assert np.allclose(edges, EDGES), (
        f"{path} was written on a different interval grid ({edges[-1]} vs {EDGES[-1]} days); "
        f"the discrete-time labels and every horizon risk would be on the wrong scale")
    assert pd.Index(pids).is_unique, f"{path} carries a duplicated empi_anon"

    pos = roster.positions_of(pids)                    # raises if an arm scored a non-val patient
    assert np.allclose(t_npz, roster.time[pos], atol=1e-6), (
        f"{path}: follow-up times disagree with the locked cohort; the npz was written "
        f"against a different feature table")
    assert np.array_equal(e_npz, roster.event[pos]), (
        f"{path}: event indicators disagree with the locked cohort")

    recal = summary.get("recalibration") or None
    n = len(roster)
    present = np.zeros(n, dtype=bool); present[pos] = True
    rank = _blank(n); rank[pos] = risk_score(hazards)
    out_risk: dict[int, np.ndarray] = {}
    for h in horizons:
        p = risk_at_horizon(hazards, float(h), edges=edges)
        if recal is not None:
            key = str(float(h))
            assert key in recal, (
                f"train_arms.json arm {arm!r} has no frozen recalibration at horizon {key}; "
                f"src/train_model.py writes one entry per horizon")
            slope = float(recal[key]["slope"])
            if not (slope > 0):
                log.warning("arm %-20s frozen recalibration slope at %d d is %+.4f (not > 0); "
                            "the transform is not monotone increasing and will REVERSE the risk "
                            "ordering at that horizon", arm, h, slope)
            p = apply_recalibration(p, recal[key])
        col = _blank(n); col[pos] = np.asarray(p, dtype=float)
        out_risk[h] = col

    log.info("arm %-20s %-12s %d/%d validation patients, %d events, ensemble val NLL %s%s",
             arm, str(summary.get("mode", "?")), int(present.sum()), n,
             int(roster.event[present].sum()), _f(summary.get("ensemble_val_nll"), 4),
             " (frozen recalibration applied)" if recal is not None else "")
    return ArmScores(arm=arm, label=str(summary.get("label", arm)), source="discrete_time",
                     present=present, rank=rank, risk=out_risk,
                     val_nll=float(summary.get("ensemble_val_nll", float("nan"))),
                     recalibration=recal, subset_arm=bool(summary.get("subset_arm", False)),
                     n_no_crop_dropped=int(summary.get("n_no_crop_dropped", 0)),
                     seeds=[int(s) for s in summary.get("seeds", [])])


def load_train_arms(path: Path) -> dict:
    """The hand-over index. Absent means training has not run; say so, with the command."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{MODULE}: the hand-over index {path} does not exist, so no arm of the ladder has "
            f"been trained yet and there is nothing for this module to score. Produce it with\n"
            f"    ~/.venvs/mrkr-torch/bin/python -m src.train_model "
            f"--config config/feasibility.yaml --stage stage1\n"
            f"and re-run this module. Partial output is supported: arms whose "
            f"val_hazards_{{arm}}.npz is still missing are skipped and logged, so stage 1 alone "
            f"is enough to get a first val_metrics.csv. The sealed test split is not an "
            f"alternative source and is not readable from here.")
    doc = read_json_retrying(path)
    assert isinstance(doc.get("arms"), dict), (
        f"{path} has no 'arms' object; it was not written by src/train_model.py")
    return doc


# =========================================================================== #
# 4. POINT METRICS FOR ONE ARM ON ONE SAMPLE OF PATIENTS                       #
# =========================================================================== #
def boot_metric_keys(horizons: list[int]) -> list[str]:
    """The metrics that carry a bootstrap interval (and therefore run 2,000 times)."""
    return ["harrell_c", "uno_c"] + [f"auc@{h}" for h in horizons]


def point_metric_keys(horizons: list[int]) -> list[str]:
    """Every reported metric. Calibration is a point estimate only: the schema carries no
    interval for slope, CITL or Brier, and a weighted GLM per replicate would cost hours."""
    keys = ["harrell_c", "uno_c"]
    for h in horizons:
        keys += [f"auc@{h}", f"slope@{h}", f"citl@{h}", f"brier@{h}"]
    return keys


def arm_metrics(scores: ArmScores, roster: Roster, take: np.ndarray, horizons: list[int],
                *, full: bool) -> dict[str, float]:
    """All metrics for one arm on one sample of roster positions.

    ``take`` is already filtered to patients this arm scores, and may contain repeats (a
    bootstrap replicate). The same function serves the point estimate and every replicate,
    so the two can never drift apart - the idiom ``src/model_clinical.evaluate`` uses.
    """
    keys = point_metric_keys(horizons) if full else boot_metric_keys(horizons)
    out: dict[str, float] = {k: float("nan") for k in keys}
    out["n_patients"] = float(take.size)
    t, e = roster.time[take], roster.event[take]
    out["n_events"] = float(int(e.sum()))
    if take.size < MIN_PATIENTS_PER_REPLICATE or int(e.sum()) < MIN_EVENTS_PER_REPLICATE:
        return out
    r = scores.rank[take]
    out["harrell_c"] = harrell_c(t, e, r)
    out["uno_c"] = uno_c(t, e, r, roster.g_grid, roster.g_vals)
    for h in horizons:
        p = scores.risk[h][take]
        y, w = ipcw_labels_weights(t, e, float(h), roster.g_grid, roster.g_vals)
        out[f"auc@{h}"] = ipcw_auc(y, w, p)
        if full:
            slope, _ = calibration_slope_intercept(y, w, p)
            out[f"slope@{h}"] = float(slope)
            obs, _, _ = km_risk(t, e, float(h))
            out[f"citl@{h}"] = float(obs - float(np.mean(p)))
            out[f"brier@{h}"] = ipcw_brier(y, w, p)
    return out


# =========================================================================== #
# 5. THE ONE SHARED PATIENT-LEVEL BOOTSTRAP DRAW (protocol section 18)         #
# =========================================================================== #
def bootstrap_draw(n_patients: int, n_boot: int, seed: int) -> np.ndarray:
    """ONE draw, ``(n_boot, n_patients)`` roster positions, reused by every arm.

    Drawn once from a seeded generator and never re-rolled, so replicate ``b`` is the same
    set of patients for every arm and every contrast; the per-replicate difference between
    two arms is therefore already paired (protocol section 18). Two runs with the same seed
    produce the identical matrix, which is what makes the CSVs byte-identical.
    """
    assert n_boot > 0 and n_patients > 0, "a bootstrap needs patients and replicates"
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_patients), size=(int(n_boot), int(n_patients)), dtype=np.int64)


class BootstrapEngine:
    """Point estimates and bootstrap distributions for (arm, patient-set) combinations.

    A paired contrast is evaluated on the INTERSECTION of the two arms' patient sets, so
    the same arm is scored on more than one mask. Results are cached per (arm, mask), which
    keeps the cost at the number of distinct combinations rather than the number of
    contrasts, and guarantees the marginal row and the contrast row quote the same numbers
    whenever the masks coincide.
    """

    def __init__(self, roster: Roster, draw: np.ndarray, horizons: list[int],
                 log: logging.Logger):
        self.roster, self.draw, self.horizons, self.log = roster, draw, horizons, log
        self._point: dict[tuple[str, bytes], dict[str, float]] = {}
        self._boot: dict[tuple[str, bytes], dict[str, np.ndarray]] = {}

    @staticmethod
    def _key(scores: ArmScores, mask: np.ndarray) -> tuple[str, bytes]:
        return (scores.arm, np.packbits(mask).tobytes())

    def point(self, scores: ArmScores, mask: np.ndarray) -> dict[str, float]:
        key = self._key(scores, mask)
        if key not in self._point:
            take = np.flatnonzero(mask & scores.present)
            self._point[key] = arm_metrics(scores, self.roster, take, self.horizons, full=True)
        return self._point[key]

    def boot(self, scores: ArmScores, mask: np.ndarray) -> dict[str, np.ndarray]:
        key = self._key(scores, mask)
        if key not in self._boot:
            usable = mask & scores.present
            keys = boot_metric_keys(self.horizons)
            vals = {k: np.full(len(self.draw), np.nan) for k in keys}
            for b in range(len(self.draw)):
                idx = self.draw[b]
                take = idx[usable[idx]]
                m = arm_metrics(scores, self.roster, take, self.horizons, full=False)
                for k in keys:
                    vals[k][b] = m[k]
            self._boot[key] = vals
            self.log.info("  bootstrapped %-20s on %d patients over %d replicates "
                          "(%d yielded a usable C-index)", scores.arm, int(usable.sum()),
                          len(self.draw), int(np.isfinite(vals["harrell_c"]).sum()))
        return self._boot[key]


def two_sided_bootstrap_p(diff: np.ndarray) -> tuple[float, int]:
    """Two-sided percentile-bootstrap p for a paired difference, and the valid replicates.

    Identical to the estimator ``src/model_clinical.py`` already uses for M1 minus M0, so
    the clinical report and the ladder report quote the same kind of number. The floor at
    ``1 / n_valid`` is the resolution of the bootstrap: no finite resample can support a
    smaller p, and reporting one would be a fabrication.
    """
    d = np.asarray(diff, dtype=float)
    ok = np.isfinite(d)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    d = d[ok]
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return float(min(1.0, max(p, 1.0 / n))), n


# =========================================================================== #
# 6. CONTRASTS AND BENJAMINI-HOCHBERG **WITHIN** EACH FAMILY                   #
# =========================================================================== #
COMPARISON_COLUMNS = ["family", "model", "reference", "metric", "horizon_days", "n_paired",
                      "estimate_model", "estimate_reference", "difference", "ci_lo", "ci_hi",
                      "p_two_sided", "p_adjusted", "fdr_method", "is_primary", "note"]

# --------------------------------------------------------------------------- #
# 6b. Convergence gate. An arm that never fitted a model is not a comparator.  #
# --------------------------------------------------------------------------- #
CONVERGENCE_COLUMNS = ["arm", "n_seeds", "train_nll_drop", "val_overfit_gap", "status", "reason"]

STATUS_OK = "ok"
STATUS_NO_CONVERGE = "did_not_converge"
STATUS_OVERFIT = "severe_overfit"


def convergence_diagnostics(cfg: Config, log: logging.Logger) -> pd.DataFrame:
    """Classify each trained arm by whether it actually fitted a model.

    Two failure modes, both observed in this cohort and both invisible in a table of
    validation discrimination:

    * an arm whose TRAINING loss never falls has not fitted anything. Its predictions are
      constant, so its IPCW AUROC is exactly 0.500 with a zero-width interval, which reads
      as "no better than chance" when the truth is "did not train". Every contrast against
      such an arm measures the training failure rather than the comparison it names.
    * an arm whose training loss falls steeply while its validation loss RISES has
      memorised the training split. Early stopping then hands it whichever epoch best fits
      the 54 validation events, so its validation metrics are the most optimistic in the
      ladder rather than the best model in it.

    Both are read off ``train_history.csv``, which the trainer writes per epoch per seed,
    so this is a diagnostic on the fitting process and never touches an outcome.
    """
    me = cfg["model_eval"]
    min_drop = float(me.get("min_train_nll_drop", 0.001))
    max_gap = float(me.get("max_val_overfit_gap", 0.10))
    path = cfg.path(cfg["model_image"]["local"]["history_csv"])
    if not path.exists():
        log.warning("no %s: the convergence gate is NOT applied and no contrast is suppressed "
                    "on convergence grounds", path.name)
        return pd.DataFrame(columns=CONVERGENCE_COLUMNS)

    hist = pd.read_csv(path)
    rows: list[dict] = []
    for arm, g in hist.groupby("arm", sort=False):
        drops, gaps = [], []
        for _seed, s in g.groupby("seed"):
            s = s.sort_values("epoch")
            drops.append(float(s["train_nll"].iloc[0] - s["train_nll"].iloc[-1]))
            gaps.append(float(s["val_nll"].iloc[-1] - s["val_nll"].min()))
        drop, gap = float(np.mean(drops)), float(np.mean(gaps))
        if drop < min_drop:
            status = STATUS_NO_CONVERGE
            reason = (f"training NLL fell by only {drop:.2e} over {len(drops)} seeds "
                      f"(floor {min_drop:g}); the fitted model is effectively constant, so its "
                      f"metrics describe a training failure rather than predictive performance")
        elif gap > max_gap:
            status = STATUS_OVERFIT
            reason = (f"validation NLL ends {gap:.3f} above its own minimum over {len(gaps)} "
                      f"seeds (ceiling {max_gap:g}); the checkpoint is selected on the same "
                      f"validation split it is then scored on, so its metrics are optimistic")
        else:
            status, reason = STATUS_OK, ""
        rows.append(dict(arm=str(arm), n_seeds=len(drops), train_nll_drop=drop,
                         val_overfit_gap=gap, status=status, reason=reason))

    df = pd.DataFrame(rows, columns=CONVERGENCE_COLUMNS).sort_values("arm").reset_index(drop=True)
    for _, r in df[df["status"] != STATUS_OK].iterrows():
        log.warning("convergence gate: arm %s is %s -- %s", r["arm"], r["status"], r["reason"])
    n_bad = int((df["status"] != STATUS_OK).sum())
    log.info("convergence gate: %d of %d trained arm(s) flagged (train-NLL floor %g, "
             "val-overfit ceiling %g)", n_bad, len(df), min_drop, max_gap)
    return df


def suppress_unfit_contrasts(rows: list[dict], convergence: pd.DataFrame,
                             log: logging.Logger, sealed: bool = False, *,
                             arm_keys: tuple[str, ...] = ("model", "reference"),
                             blank_keys: tuple[str, ...] = ("difference", "ci_lo", "ci_hi",
                                                            "p_two_sided", "p_adjusted"),
                             flag_key: str | None = None,
                             what: str = "contrast") -> list[dict]:
    """Blank the estimate of any row that involves an arm which did not fit a model.

    Follows the protocol section 21 precedent exactly: the row stays so the reader can see
    the comparison was specified, the point estimate and interval are suppressed, and the
    reason is stated in ``note``. Suppressed rows carry a NaN p value, so
    :func:`benjamini_hochberg` drops them from the family's multiplicity rather than
    letting a training failure inflate every other contrast's adjusted p.

    THE GATE IS PARAMETERISED, NOT COPIED. Decision-curve rows carry the same semantics -
    ``did_not_converge`` suppresses on every split, ``severe_overfit`` on validation only -
    but a different schema: they name their arms in ``arm``/``reference`` rather than
    ``model``/``reference``, they have no ``family``, and the estimates to blank are the
    net-benefit columns rather than ``difference``/``ci_lo``/.../``p_adjusted``. Assigning
    this function's default key set to such a row would ``KeyError`` on the way in and then
    invent five columns that :func:`write_table`'s exact column-order assert would reject.
    So the keys are arguments with today's contrast behaviour as their defaults: there is
    one implementation of "which statuses suppress on which split", and adding a second
    consumer cannot make it drift.

    * ``arm_keys`` - the row fields naming the arms whose status is checked. Duplicates are
      collapsed, so a decision-curve row whose ``arm`` IS the ``reference`` states its
      reason once rather than twice.
    * ``blank_keys`` - the fields set to NaN. Every one must already exist in the row; a
      typo would otherwise add a column and break the pinned schema at write time.
    * ``flag_key`` - an optional boolean field set to True (``suppressed`` in the
      decision-curve schema). ``val_comparisons.csv`` has no such column and passes None.
    * ``what`` - what the log line calls the row.
    """
    if convergence.empty:
        return rows
    # On the SEALED split, `severe_overfit` is no longer a reason to suppress. That flag
    # exists because a validation metric computed at a validation-selected epoch is
    # optimistic - circular. The test split took no part in that selection, so its estimate
    # is unbiased no matter how hard the arm overfitted, and it is precisely the number
    # that reveals whether the overfitting cost anything. `did_not_converge` still
    # suppresses everywhere: a constant predictor is constant on every split.
    statuses = ({STATUS_NO_CONVERGE} if sealed else {STATUS_NO_CONVERGE, STATUS_OVERFIT})
    bad = {r["arm"]: (r["status"], r["reason"])
           for _, r in convergence.iterrows() if r["status"] in statuses}
    if not bad:
        return rows
    for r in rows:
        hit: list[str] = []
        for k in arm_keys:
            a = r[k]
            if a in bad and a not in hit:
                hit.append(a)
        if not hit:
            continue
        why = "; ".join(f"{a} {bad[a][0]}: {bad[a][1]}" for a in hit)
        r["note"] = _append_note(r["note"], "SUPPRESSED -- " + why)
        for k in blank_keys:
            assert k in r, (
                f"suppress_unfit_contrasts was asked to blank {k!r}, which this row does not "
                f"carry; assigning it would invent a column and break the pinned schema")
            r[k] = float("nan")
        if flag_key is not None:
            r[flag_key] = True
        log.warning("%s %s%s SUPPRESSED: %s", what,
                    " vs ".join(str(r[k]) for k in arm_keys),
                    f" ({r['family']})" if "family" in r else "", why)
    return rows


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p values. NaNs stay NaN and do not count in m.

    Applied to ONE family at a time by :func:`build_comparisons`. Pooling the families
    would silently change the multiplicity of every contrast in every other family, so it
    is never done here.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]       # enforce monotonicity
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def contrast_row(family: str, model: str, reference: str, *, metric: str, horizon: int,
                 arms: dict[str, ArmScores], engine: BootstrapEngine,
                 is_primary: bool) -> dict | None:
    """One paired contrast on the INTERSECTION of the two arms' patient sets.

    ``m1``/``m1_klg`` run on the KLG-eligible subset and any arm drops patients with no crop
    in its view set, so two arms rarely score exactly the same people. Both point estimates
    are recomputed on the intersection and ``n_paired`` reports its size, so the difference
    is a difference between two models rather than between two cohorts.
    """
    if model not in arms or reference not in arms:
        return None
    a, b = arms[model], arms[reference]
    mask = a.present & b.present
    key = f"{metric}@{horizon}" if metric == PRIMARY_METRIC else metric
    n_paired = int(mask.sum())
    note = ""
    if n_paired < a.n_patients or n_paired < b.n_patients:
        note = (f"paired on the {n_paired} patients both arms score "
                f"({model} scores {a.n_patients}, {reference} scores {b.n_patients})")
    em = engine.point(a, mask)[key]
    er = engine.point(b, mask)[key]
    d = engine.boot(a, mask)[key] - engine.boot(b, mask)[key]
    lo, hi = percentile_ci(d)
    p, n_valid = two_sided_bootstrap_p(d)
    if n_valid < len(engine.draw):
        note = (note + "; " if note else "") + (
            f"{len(engine.draw) - n_valid} of {len(engine.draw)} replicates were not "
            f"estimable and are excluded")
    return dict(family=family, model=model, reference=reference, metric=metric,
                horizon_days=int(horizon), n_paired=n_paired,
                estimate_model=float(em), estimate_reference=float(er),
                difference=float(em - er), ci_lo=float(lo), ci_hi=float(hi),
                p_two_sided=float(p), p_adjusted=float("nan"), fdr_method="",
                is_primary=bool(is_primary), note=note)


def build_comparisons(cfg: Config, arms: dict[str, ArmScores], engine: BootstrapEngine,
                      log: logging.Logger,
                      convergence: pd.DataFrame | None = None,
                      sealed: bool = False) -> pd.DataFrame:
    """The primary contrast, then each declared family with BH applied inside it alone.

    Contrasts involving an arm that failed the convergence gate are suppressed BEFORE the
    family is adjusted, so a training failure does not consume multiplicity and inflate
    the adjusted p of every genuine contrast beside it.
    """
    conv = convergence if convergence is not None else pd.DataFrame(columns=CONVERGENCE_COLUMNS)
    me = cfg["model_eval"]
    pc = me["primary_contrast"]
    fdr = str(me.get("fdr_method", "bh")).lower()
    assert fdr in FDR_METHODS, f"unknown model_eval.fdr_method {fdr!r}; known: {sorted(FDR_METHODS)}"
    horizon = int(pc["horizon_days"])
    rows: list[dict] = []

    primary = contrast_row("primary", str(pc["model"]), str(pc["reference"]),
                           metric=PRIMARY_METRIC, horizon=horizon, arms=arms, engine=engine,
                           is_primary=True)
    if primary is None:
        log.warning("PRIMARY CONTRAST %s vs %s at %d d is NOT estimable: one of the arms has no "
                    "scores yet. val_comparisons.csv will carry no is_primary row until it is "
                    "trained.", pc["model"], pc["reference"], horizon)
    else:
        suppress_unfit_contrasts([primary], conv, log, sealed=sealed)
        rows.append(primary)
        if math.isnan(primary["difference"]):
            log.warning("PRIMARY %s minus %s at %d d is SUPPRESSED on convergence grounds; "
                        "see the note column and val_convergence.csv",
                        primary["model"], primary["reference"], horizon)
        else:
            log.info("PRIMARY %s minus %s, IPCW AUROC @%d d: %+.4f (95%% CI %+.4f to %+.4f), "
                     "two-sided bootstrap p %.4f, on %d paired patients",
                     primary["model"], primary["reference"], horizon, primary["difference"],
                     primary["ci_lo"], primary["ci_hi"], primary["p_two_sided"],
                     primary["n_paired"])

    # Families are adjusted SEPARATELY. Sorted by family name so the file is deterministic;
    # the declared pair order is preserved inside each family.
    for family in sorted(me["comparison_families"]):
        fam_rows: list[dict] = []
        for pair in me["comparison_families"][family]:
            model, reference = str(pair[0]), str(pair[1])
            r = contrast_row(family, model, reference, metric=PRIMARY_METRIC, horizon=horizon,
                             arms=arms, engine=engine, is_primary=False)
            if r is None:
                log.info("family %-12s %s vs %s SKIPPED: an arm is not scored yet",
                         family, model, reference)
                continue
            fam_rows.append(r)
        if not fam_rows:
            continue
        suppress_unfit_contrasts(fam_rows, conv, log, sealed=sealed)
        adj = benjamini_hochberg([r["p_two_sided"] for r in fam_rows])
        for r, q in zip(fam_rows, adj):
            r["p_adjusted"] = float(q)
            r["fdr_method"] = fdr
        log.info("family %-12s %d contrast(s), %s-adjusted within the family only",
                 family, len(fam_rows), FDR_METHODS[fdr])
        rows.extend(fam_rows)

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


# =========================================================================== #
# 7. SUBGROUPS - the protocol section 21 suppression rule                      #
# =========================================================================== #
SUBGROUP_COLUMNS = ["subgroup", "level", "n_patients", "n_events", "metric", "horizon_days",
                    "estimate", "ci_lo", "ci_hi", "suppressed", "suppression_reason"]


def subgroup_levels(cfg: Config, frame: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(subgroup, level, boolean mask over the roster) for every pre-specified stratum.

    The families are exactly ``src/subgroups.py``'s - sex, age at index, race, obesity,
    weight-bearing frontal radiograph and view set - because BOTH modules now read the one
    declaration at ``config/feasibility.yaml -> subgroups.families``, so the cohort
    description and the equity audit cannot drift apart. Before 2026-08-11 the two lists
    were written out twice, here and at ``subgroups.build_rows``, and agreement was a
    convention rather than a fact.

    ``src.subgroups.load_families(cfg, "equity")`` refuses to return anything but the six
    original families, in the original order, because this list is the row set of the
    PUBLISHED ``{val,test}_subgroups.csv``. A seventh family belongs in scope
    ``robustness`` (:func:`build_imaging_robustness`), which writes new ``v6_`` files.
    Names are the reviewer-facing ones; no underscores reach a table.
    """
    return [(fam.report_label, lv.report_label, family_mask(lv.rule, frame, cfg))
            for fam in load_families(cfg, "equity") for lv in fam.levels]


def build_subgroups(cfg: Config, roster: Roster, scores: ArmScores | None,
                    engine: BootstrapEngine, horizon: int,
                    log: logging.Logger) -> pd.DataFrame:
    """The equity audit, with the section-21 floor applied BEFORE anything is estimated.

    The rule is ``n_events < model_eval.suppress_below_events``, evaluated per level on the
    intersection of the level mask with the patients the arm actually scored. A level that
    fails it gets a NaN estimate, ``suppressed=True`` and a reason string.

    **How many levels that suppresses is a property of the data, not of this function**, so
    no count is claimed here or in the module docstring; the caller reads it off the
    returned frame. Two consequences of the 54-event validation split are worth stating
    because they are true whatever the counts turn out to be. First, most levels will be
    suppressed, and a fully suppressed table is a legitimate result rather than a failure.
    Second, a level that DOES clear the floor at this event count must hold nearly the whole
    split, so it repeats the all-patient estimate under a subgroup label instead of
    contrasting one group with another; ``src/make_manuscript.py`` computes that caveat from
    this frame and prints it beside any surviving estimate. Either way the table is reported
    plainly rather than filled with numbers nobody may interpret.
    """
    thresh = int(cfg["model_eval"]["suppress_below_events"])
    assert thresh == SUPPRESS_BELOW_EVENTS, (
        f"model_eval.suppress_below_events is {thresh}, but protocol section 21 (and "
        f"src/model_clinical.SUPPRESS_BELOW_EVENTS) fixes it at {SUPPRESS_BELOW_EVENTS}")
    rows: list[dict] = []
    scored = scores.present if scores is not None else np.zeros(len(roster), dtype=bool)
    for subgroup, level, mask in subgroup_levels(cfg, roster.frame):
        m = mask & scored
        n_pat, n_ev = int(m.sum()), int(roster.event[m].sum())
        estimate = lo = hi = float("nan")
        if scores is None:
            suppressed, reason = True, ("no arm is scored yet, so no subgroup estimate exists "
                                        "(train the ladder first)")
        elif n_ev < thresh:
            suppressed = True
            reason = (f"protocol section 21: fewer than {thresh} events "
                      f"({n_ev} in this level)")
        else:
            suppressed, reason = False, ""
            estimate = engine.point(scores, m)[f"{PRIMARY_METRIC}@{horizon}"]
            lo, hi = percentile_ci(engine.boot(scores, m)[f"{PRIMARY_METRIC}@{horizon}"])
        rows.append(dict(subgroup=subgroup, level=level, n_patients=n_pat, n_events=n_ev,
                         metric=PRIMARY_METRIC, horizon_days=int(horizon),
                         estimate=estimate, ci_lo=lo, ci_hi=hi,
                         suppressed=bool(suppressed), suppression_reason=reason))
    df = pd.DataFrame(rows, columns=SUBGROUP_COLUMNS)
    n_supp, n_scored = int(df["suppressed"].sum()), int(scored.sum())
    log.info("subgroups: %d level(s), %d suppressed below the %d-event floor "
             "(largest level holds %d events; the validation split has %d in total)",
             len(df), n_supp, thresh, int(df["n_events"].max()) if len(df) else 0,
             EXPECTED_VAL_EVENTS)
    if n_supp == len(df) and len(df):
        log.info("every subgroup estimate is suppressed. That is the correct, honest result "
                 "at %d validation events; the equity audit is deferred to the sealed test "
                 "split (protocol section 21).", EXPECTED_VAL_EVENTS)
    for _, r in df[~df["suppressed"].astype(bool)].iterrows():
        log.info("subgroup level %r / %r CLEARED the %d-event floor: %d events on %d patients, "
                 "%.1f%% of the %d this arm scored. At this event count that is the all-patient "
                 "estimate under a subgroup label rather than a group contrast, and "
                 "src/make_manuscript.py prints that caveat beside it (protocol section 21).",
                 str(r["subgroup"]), str(r["level"]), thresh, int(r["n_events"]),
                 int(r["n_patients"]), 100.0 * int(r["n_patients"]) / max(n_scored, 1), n_scored)
    return df


# =========================================================================== #
# 7a2. V6 REVISION - IMAGING ROBUSTNESS STRATA (A3)                            #
#                                                                              #
# The academic editor asked for performance "by weight-bearing status,         #
# acquisition year, view availability, image quality, laterality source, and   #
# equipment or site when metadata permit". Weight-bearing and view availability #
# are already in the equity table above. This section adds acquisition era,     #
# image quality (masking, localization confidence, localization method,         #
# photometric inversion, bilateral half-selection) and laterality source, and   #
# records in a table of its own that equipment, manufacturer and site are NOT   #
# in these data at all - MRKR released no such DICOM tag and the source DICOMs  #
# are gone. A request that cannot be met is answered, not dropped.             #
#                                                                              #
# THREE things this section does NOT do:                                        #
#   * it does not touch the published {val,test}_subgroups.csv. Every output    #
#     here is namespaced v6_ and the write path asserts it.                     #
#   * it does not weaken the protocol section-21 50-event floor. The same       #
#     constant, the same assertion and the same suppression semantics as        #
#     build_subgroups above. Most of these strata suppress, several of them     #
#     provably-unavoidably, and that is reported as the result.                 #
#   * it does not add a family to the equity scope. load_families refuses it.   #
# =========================================================================== #
V6_POSTHOC_NOTE = ("POST HOC EXPLORATORY (deviation D35): specified after the single "
                   "permitted read of the sealed test split, not before it")

#: The arms the robustness table is built for. ``m2_frontal`` first because the revision's
#: central claim is about the single frontal view; ``m4_fusion`` second because it is the
#: arm the published equity table used, which makes the weight-bearing and view rows here
#: directly checkable against ``outputs/tables/test_subgroups.csv``.
V6_ROBUSTNESS_ARMS = ("m2_frontal", "m4_fusion")

#: A per-patient image attribute can be read off every crop the pipeline wrote for that
#: patient, or off the frontal crop alone. Neither is "the" right choice, so both are
#: reported and the table carries the scope on every row.
#:
#: ``all_crops``    - the union the multi-view arms actually see; a patient is exposed if
#:                    ANY of their crops carries the flag. Partitions cleanly (740 of the
#:                    741 test roster patients have at least one crop).
#: ``frontal_only`` - the single image ``m2_frontal``, ``m4_frontal`` and
#:                    ``r1_densenet_frontal`` read. Unconfounded by view availability,
#:                    which matters most for the masking family; costs the 7 test patients
#:                    who have no frontal crop.
IMAGE_SCOPES = ("all_crops", "frontal_only")

#: Pre-declared width at which a 95% interval is flagged wide, carried over unchanged from
#: the A1/A2 module so one revision does not use two definitions: 0.15 is the top of the
#: range spanned by the six published test-split subgroup estimates that cleared the same
#: floor (0.101 to 0.160 in outputs/tables/test_subgroups.csv). A flagged cell is no more
#: precise than the least precise subgroup the paper already reports.
WIDE_INTERVAL_WIDTH = 0.15

#: Both metrics are reported for every level, and the second one is not decoration.
#: The IPCW cumulative/dynamic AUROC at 1,825 days compares patients with an event by the
#: horizon against patients observed EVENT-FREE BEYOND it, and only 162 of the 741 test
#: patients reach 1,826 days. Acquisition date and laterality source are almost perfectly
#: collinear with follow-up length in this cohort, so several of the strata the editor
#: asked for hold zero such controls and have NO 5-year AUROC at all - not a wide one, none.
#: Harrell C is horizon free and is estimable there, so reporting it is the difference
#: between answering the question and reporting a blank cell.
ROBUSTNESS_METRICS = (PRIMARY_METRIC, "harrell_c")

HARRELL_C_NOTE = ("Harrell C is horizon free; horizon_days carries the primary horizon "
                  "only because the schema has one column for it, not as a claim")

ROBUSTNESS_COLUMNS = ["arm", "family", "subgroup", "level", "image_scope", "metric",
                      "horizon_days", "n_patients", "n_events", "n_cases_by_horizon",
                      "n_controls_beyond_horizon", "estimate", "ci_lo", "ci_hi", "ci_width",
                      "wide_interval", "suppressed", "suppression_reason", "note"]

METADATA_AVAILABILITY_COLUMNS = ["stratum", "requested_by", "available", "reason"]

V6_ROBUSTNESS_BASENAMES = {
    "strata": "v6_robustness_strata.csv",
    "availability": "v6_robustness_metadata_availability.csv",
}


def patient_image_attributes(cfg: Config, scope: str, log: logging.Logger) -> pd.DataFrame:
    """One row per patient of per-image quality flags, aggregated under ``scope``.

    ``derived-data/cohort/preprocess_labels.csv`` is PER IMAGE (6,071 rows over 3,706
    patients; 3 final-cohort patients have no row because their preprocessing failed).
    Every aggregation here is the worst case over the patient's crops, because these are
    contamination-style flags and a patient is exposed if any image the model reads is:

    ``image_masked_pct_max``                     max masked fraction  (worst)
    ``image_crop_confidence_min``                min localization confidence (worst)
    ``image_crop_method_any_intensity_profile``  any fallback localization
    ``image_inverted_any``                       any photometric inversion applied
    ``image_half_selected_any``                  any bilateral film cut at the midline

    Read-only. The index is ``empi_anon`` as a string, which is the roster's join key.
    """
    assert scope in IMAGE_SCOPES, f"unknown image scope {scope!r}; known {IMAGE_SCOPES}"
    path = cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_labels.csv"
    assert path.exists(), f"{path} is missing; the image-quality strata need the per-image flags"
    pl = pd.read_csv(path)
    pl["empi_anon"] = pl["empi_anon"].astype(str)
    n_all = len(pl)
    if scope == "frontal_only":
        pl = pl[pl["view"] == "frontal"]
    g = pl.groupby("empi_anon")
    out = pd.DataFrame({
        "image_masked_pct_max": g["masked_pct"].max(),
        "image_crop_confidence_min": g["crop_confidence"].min(),
        "image_crop_method_any_intensity_profile":
            g["crop_method"].apply(lambda s: bool((s == "intensity_profile").any())),
        "image_inverted_any": g["inverted"].max().astype(bool),
        "image_half_selected_any":
            g["half_selected"].apply(lambda s: bool((s.astype(str) != "none").any())),
        "n_crops_in_scope": g.size(),
    })
    log.info("image attributes | scope %s: %d of %d preprocessed images over %d patients",
             scope, len(pl), n_all, len(out))
    return out


def robustness_frame(cfg: Config, roster: Roster, scope: str,
                     log: logging.Logger) -> pd.DataFrame:
    """The roster frame plus every column a ``robustness`` family rule can name.

    Two joins, both one-to-one on ``empi_anon`` and both left joins onto the roster so the
    row order is the roster's and stays the row order every mask indexes into:

    * ``acquisition_year`` from ``final_cohort.study_date``. **That column derives from
      image ``StudyDate_anon``, which carries a PER-PATIENT RANDOM SHIFT** (see
      ``src/inventory.py`` and ``outputs/data_quality_report.md``: "Absolute calendar dates
      are meaningless, but within-patient day intervals are valid"), and protocol section
      17's written confirmation that de-identified dates are comparable across patients has
      never been obtained (deviation D17). The caveat rides into the output on the era
      families' ``note``; it is repeated here so it cannot be lost by a caller.
    * the per-image quality flags, aggregated by :func:`patient_image_attributes`.

    ``side_source`` is already on the roster frame, so laterality source needs no join.
    """
    frame = roster.frame.reset_index(drop=True).copy()
    frame["empi_anon"] = frame["empi_anon"].astype(str)
    coh = cfg.path(cfg["paths"]["cohort_dir"])

    fcoh = pd.read_parquet(coh / "final_cohort.parquet", columns=["empi_anon", "study_date",
                                                                  "side_source"])
    fcoh["empi_anon"] = fcoh["empi_anon"].astype(str)
    assert fcoh["empi_anon"].is_unique, "final_cohort.parquet is not one row per patient"
    year = pd.to_datetime(fcoh["study_date"]).dt.year
    ymap = dict(zip(fcoh["empi_anon"], year))
    frame["acquisition_year"] = frame["empi_anon"].map(ymap).astype("Float64")
    assert frame["acquisition_year"].notna().all(), (
        "a roster patient has no study_date in final_cohort.parquet; the era stratum would "
        "silently drop them into no level at all")
    # side_source travels on the clinical feature table AND on final_cohort. They must agree,
    # or the laterality-source stratum means two different things depending on the source.
    smap = dict(zip(fcoh["empi_anon"], fcoh["side_source"]))
    assert (frame["empi_anon"].map(smap) == frame["side_source"]).all(), (
        "side_source disagrees between features_clinical.parquet and final_cohort.parquet")

    attrs = patient_image_attributes(cfg, scope, log)
    for col in attrs.columns:
        frame[col] = frame["empi_anon"].map(attrs[col])
    n_missing = int(frame["n_crops_in_scope"].isna().sum())
    log.info("robustness frame | scope %s: %d roster patients, %d with no crop in scope "
             "(they satisfy no image-quality level and are reported as such)",
             scope, len(frame), n_missing)
    return frame


def build_imaging_robustness(cfg: Config, roster: Roster, arms: dict[str, ArmScores],
                             engine: BootstrapEngine, horizon: int,
                             log: logging.Logger) -> pd.DataFrame:
    """The A3 table: every ``robustness`` family, every declared arm, both image scopes.

    The estimator, the floor and the suppression semantics are exactly
    :func:`build_subgroups`'s - same constant, same assertion, same reason string - because
    a robustness stratum that used a different rule from the equity audit would not be
    comparable with it. ``source: clinical`` families do not depend on the image scope, so
    they are emitted once with ``image_scope='not applicable'`` instead of twice.

    **How many strata suppress is a property of the data.** The test split carries 106
    events, so at most two levels of any partition can clear a 50-event floor and a
    three-level scheme can never leave three. Where a level's suppression is arithmetically
    forced rather than a shortfall of effort, the family's config ``note`` says so.
    """
    thresh = int(cfg["model_eval"]["suppress_below_events"])
    assert thresh == SUPPRESS_BELOW_EVENTS, (
        f"model_eval.suppress_below_events is {thresh}, but protocol section 21 (and "
        f"src/model_clinical.SUPPRESS_BELOW_EVENTS) fixes it at {SUPPRESS_BELOW_EVENTS}")
    families = load_families(cfg, "robustness")
    frames = {scope: robustness_frame(cfg, roster, scope, log) for scope in IMAGE_SCOPES}

    rows: list[dict] = []
    for arm in V6_ROBUSTNESS_ARMS:
        scores = arms.get(arm)
        assert scores is not None, f"{arm} is not scored; the robustness table needs it"
        for fam in families:
            scopes = IMAGE_SCOPES if fam.source == "imaging" else ("not applicable",)
            for scope in scopes:
                frame = frames[scope] if fam.source == "imaging" else frames[IMAGE_SCOPES[0]]
                for lv in fam.levels:
                    m = family_mask(lv.rule, frame, cfg) & scores.present
                    for metric in ROBUSTNESS_METRICS:
                        rows.append(_robustness_row(fam, lv, arm, scope, m, roster, scores,
                                                    engine, horizon, thresh, metric))
    df = pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)
    n_supp = int(df["suppressed"].sum())
    log.info("v6 robustness: %d row(s) over %d famil(ies) x %d arm(s) x %d metric(s); "
             "%d suppressed, %d estimated, %d of those flagged wide (>= %.2f)",
             len(df), len(families), len(V6_ROBUSTNESS_ARMS), len(ROBUSTNESS_METRICS),
             n_supp, len(df) - n_supp, int(df["wide_interval"].sum()), WIDE_INTERVAL_WIDTH)
    n_floor = int(df["suppression_reason"].str.startswith("protocol section 21").sum())
    n_nocontrol = int(df["suppression_reason"].str.startswith("not estimable").sum())
    log.info("  of the %d suppressed rows, %d fail the %d-event floor and %d clear it but "
             "have no estimable metric (no control observed beyond the horizon)",
             n_supp, n_floor, thresh, n_nocontrol)
    for _, r in df[~df["suppressed"].astype(bool)].iterrows():
        log.info("  CLEARED %-11s %-9s %-46s %-15s n=%-4d ev=%-4d %.3f (%.3f to %.3f)%s",
                 r["arm"], r["metric"], f"{r['subgroup']} / {r['level']}", r["image_scope"],
                 int(r["n_patients"]), int(r["n_events"]), r["estimate"], r["ci_lo"],
                 r["ci_hi"], "  WIDE" if r["wide_interval"] else "")
    return df


def _robustness_row(fam: Family, lv, arm: str, scope: str, m: np.ndarray, roster: Roster,
                    scores: ArmScores, engine: BootstrapEngine, horizon: int,
                    thresh: int, metric: str) -> dict:
    """One robustness cell. Split out so the floor is applied in exactly one place.

    Two ways a cell fails to produce a number, and they are NOT the same thing:

    1. **The protocol section-21 floor.** Fewer than ``thresh`` events in the level. Same
       constant, same wording as :func:`build_subgroups`.
    2. **The estimator has nothing to work with.** The IPCW AUROC at the horizon needs at
       least one case (event by the horizon) AND at least one control (observed event-free
       beyond it). The test split holds 162 controls in total, and acquisition date and
       laterality source are nearly collinear with follow-up length here, so several levels
       clear the event floor and still have no 5-year AUROC. That is reported with its own
       reason string and its own counts, never as a blank cell or as a wide interval.
    """
    key = f"{PRIMARY_METRIC}@{horizon}" if metric == PRIMARY_METRIC else metric
    n_pat, n_ev = int(m.sum()), int(roster.event[m].sum())
    t, e = roster.time[m], roster.event[m]
    n_case = int(((t <= horizon) & (e == 1)).sum())
    n_ctrl = int((t > horizon).sum())
    estimate = lo = hi = width = float("nan")
    suppressed, reason = True, ""
    if n_ev < thresh:
        reason = f"protocol section 21: fewer than {thresh} events ({n_ev} in this level)"
    elif metric == PRIMARY_METRIC and (n_case == 0 or n_ctrl == 0):
        reason = (f"not estimable: the IPCW AUROC at {horizon} d compares patients with an "
                  f"event by {horizon} days against patients observed event-free beyond it, "
                  f"and this level holds {n_case} of the former and {n_ctrl} of the latter. "
                  f"This is undefined, not imprecise; read the horizon-free Harrell C row")
    else:
        estimate = engine.point(scores, m)[key]
        lo, hi = percentile_ci(engine.boot(scores, m)[key])
        if not np.isfinite(estimate):
            reason = (f"not estimable: {key} returned no value on this level's "
                      f"{n_pat} patients ({n_case} case(s), {n_ctrl} control(s))")
        else:
            suppressed = False
            width = float(hi - lo)
    wide = bool(np.isfinite(width) and width >= WIDE_INTERVAL_WIDTH)
    note = V6_POSTHOC_NOTE if not fam.note else f"{fam.note} | {V6_POSTHOC_NOTE}"
    if metric == "harrell_c":
        note = f"{note} | {HARRELL_C_NOTE}"
    elif not suppressed:
        # Stated on every estimated AUROC row rather than behind a threshold nobody
        # declared: a 5-year AUROC resting on 94 cases and 3 controls is arithmetically
        # defined and substantively empty, and only these two counts show that.
        note = (f"{note} | estimated on {n_case} case(s) with an event by {horizon} d "
                f"against {n_ctrl} control(s) observed event-free beyond it")
    if wide:
        note = (f"{note} | WIDE INTERVAL: the 95% interval spans {width:.3f}, at or above "
                f"the pre-declared {WIDE_INTERVAL_WIDTH:.2f} flag")
    return dict(arm=arm, family=fam.key, subgroup=fam.report_label, level=lv.report_label,
                image_scope=scope, metric=metric, horizon_days=int(horizon),
                n_patients=n_pat, n_events=n_ev, n_cases_by_horizon=n_case,
                n_controls_beyond_horizon=n_ctrl, estimate=estimate, ci_lo=lo, ci_hi=hi,
                ci_width=width, wide_interval=wide, suppressed=bool(suppressed),
                suppression_reason=reason, note=note)


def build_metadata_availability(cfg: Config) -> pd.DataFrame:
    """The editor's requested strata that these data cannot supply, stated rather than dropped.

    Equipment, manufacturer and site are not "not run": MRKR never released those DICOM
    tags, the 34.85 GB source DICOM set is no longer held, and the retained artefacts are
    512x512 preprocessed crops with the border band zeroed, so they are not recoverable by
    any amount of further work. ``horizontal_flip`` is a fourth case again: the flag exists
    and is described, but on the test split it marks 63 images, one per patient, carrying 7
    events in total, so no stratum-level estimate can exist under the 50-event floor.
    """
    rows = [dict(stratum=str(u["report_label"]),
                 requested_by="academic editor, imaging robustness",
                 available=False, reason=str(u["reason"]))
            for u in cfg["subgroups"]["unavailable_strata"]]
    return pd.DataFrame(rows, columns=METADATA_AVAILABILITY_COLUMNS)


# =========================================================================== #
# 7a3. V6 REVISION - LEARNING CURVES (A4, supplementary figure S1)             #
#                                                                              #
# THE TRAP THIS SECTION EXISTS TO DEFUSE. ``val_overfit_gap`` in               #
# {val,test}_convergence.csv is NOT a train-validation gap. It is              #
# val_nll[last] - min(val_nll) (convergence_diagnostics, above), and since     #
# early stopping runs at patience=8 the last epoch is ALWAYS exactly 8 epochs  #
# past the checkpoint that was kept. So it measures how far training ran after #
# the retained model, and a figure that plots it unmarked reads as "the model  #
# we kept was diverging". Every row here carries is_retained_epoch, and the    #
# per-seed summary reports the gap AT the retained checkpoint separately.      #
# =========================================================================== #
LEARNING_CURVE_COLUMNS = ["arm", "seed", "epoch", "train_nll", "val_nll", "train_val_gap",
                          "val_nll_above_own_min", "lr", "secs", "improved",
                          "is_retained_epoch", "epochs_from_retained", "n_epochs_run", "note"]

LEARNING_CURVE_SEED_COLUMNS = ["arm", "seed", "n_epochs_run", "retained_epoch", "last_epoch",
                               "epochs_after_retained", "train_nll_at_retained",
                               "val_nll_at_retained", "gap_at_retained", "val_nll_last",
                               "val_overfit_gap_last_minus_min", "note"]

LEARNING_CURVE_ARM_COLUMNS = ["arm", "n_seeds", "mean_retained_epoch", "mean_epochs_run",
                              "mean_train_nll_at_retained", "mean_val_nll_at_retained",
                              "mean_gap_at_retained", "mean_val_overfit_gap",
                              "convergence_status", "note"]

V6_LEARNING_CURVE_BASENAMES = {
    "curves": "v6_learning_curves.csv",
    "per_seed": "v6_learning_curves_by_seed.csv",
    "per_arm": "v6_learning_curves_by_arm.csv",
}

#: Why the gap in these tables is a bound and not a measurement. Verified line by line
#: against src/train_model.py: train_nll is accumulated in ``model.train()`` mode with
#: dropout 0.3 and augmentation active and averaged across the epoch WHILE the weights
#: update (``train_one_epoch``); val_nll is one clean ``model.eval()`` pass on the
#: epoch-end weights (``predict_hazards`` / ``val_nll_of``). All three differences inflate
#: train_nll, and the gap is val_nll - train_nll, so an inflated train_nll makes the gap
#: too small.
TRAIN_VAL_GAP_NOTE = (
    "LOWER BOUND, not a measured train-validation gap: train_nll is accumulated in "
    "model.train() mode with dropout 0.3 and augmentation active and averaged across the "
    "epoch while the weights update, while val_nll is a single clean eval-mode pass on the "
    "epoch-end weights. All three differences inflate train_nll and the gap is "
    "val_nll - train_nll, so the true gap is at least this large and by an unknown margin")

#: Why the retained epoch has to be marked on the face of the table.
RETAINED_EPOCH_NOTE = (
    "val_overfit_gap in {val,test}_convergence.csv is val_nll[last] - min(val_nll), NOT a "
    "train-validation gap; at patience=8 the last epoch is always exactly 8 past the "
    "retained checkpoint, so it measures how far training ran AFTER the model that was "
    "kept. Use is_retained_epoch / gap_at_retained for the model that was actually scored")


def build_learning_curves(cfg: Config, log: logging.Logger
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-epoch, per-seed and per-arm views of ``outputs/tables/train_history.csv``.

    The retained epoch is taken from the trainer's own ``improved`` column - the LAST epoch
    it marked True, which is exactly the epoch whose weights ``load_seed_model`` restores -
    and cross-checked against ``argmin(val_nll)``. Deriving it from ``argmin`` alone would
    be wrong at a tie, because the trainer improves only on ``v_nll < best - 1e-6``.
    """
    path = cfg.path(cfg["model_image"]["local"]["history_csv"])
    assert path.exists(), f"{path} is missing; A4 has no input"
    patience = int(cfg["model_image"]["early_stopping"]["patience"])
    max_epochs = int(cfg["model_image"]["max_epochs"])
    hist = pd.read_csv(path)
    need = {"arm", "seed", "epoch", "train_nll", "val_nll", "improved"}
    assert need <= set(hist.columns), f"{path.name} is missing {sorted(need - set(hist.columns))}"

    curves, per_seed = [], []
    for (arm, seed), g in hist.groupby(["arm", "seed"], sort=False):
        g = g.sort_values("epoch").reset_index(drop=True)
        improved = g["improved"].astype(str).str.lower().isin(("true", "1"))
        assert improved.any(), f"{arm} seed {seed}: no epoch is marked improved"
        retained = int(g.loc[improved, "epoch"].iloc[-1])
        vmin = float(g["val_nll"].min())
        at = g[g["epoch"] == retained].iloc[0]
        assert abs(float(at["val_nll"]) - vmin) <= 1e-6, (
            f"{arm} seed {seed}: the last improved epoch ({retained}) is not the validation "
            f"minimum ({float(at['val_nll'])} vs {vmin}); the retained-epoch rule is wrong")
        last = int(g["epoch"].iloc[-1])
        after = last - retained
        # Training stops either because patience expired or because it hit max_epochs
        # (train_model.py: complete = bad >= patience or epoch == max_epochs - 1). Every
        # series in this study stopped on patience, so `after` is always exactly 8 - which
        # is the whole reason val_overfit_gap is not a train-validation gap.
        assert after == patience or last == max_epochs - 1, (
            f"{arm} seed {seed}: training ran {after} epochs past the retained checkpoint, "
            f"which is neither the configured patience of {patience} nor a max_epochs "
            f"({max_epochs}) stop")
        for _, r in g.iterrows():
            curves.append(dict(
                arm=str(arm), seed=int(seed), epoch=int(r["epoch"]),
                train_nll=float(r["train_nll"]), val_nll=float(r["val_nll"]),
                train_val_gap=float(r["val_nll"]) - float(r["train_nll"]),
                val_nll_above_own_min=float(r["val_nll"]) - vmin,
                lr=float(r.get("lr", float("nan"))), secs=float(r.get("secs", float("nan"))),
                improved=bool(str(r["improved"]).lower() in ("true", "1")),
                is_retained_epoch=bool(int(r["epoch"]) == retained),
                epochs_from_retained=int(r["epoch"]) - retained, n_epochs_run=len(g),
                note=TRAIN_VAL_GAP_NOTE))
        per_seed.append(dict(
            arm=str(arm), seed=int(seed), n_epochs_run=len(g), retained_epoch=retained,
            last_epoch=last, epochs_after_retained=after,
            train_nll_at_retained=float(at["train_nll"]),
            val_nll_at_retained=float(at["val_nll"]),
            gap_at_retained=float(at["val_nll"]) - float(at["train_nll"]),
            val_nll_last=float(g["val_nll"].iloc[-1]),
            val_overfit_gap_last_minus_min=float(g["val_nll"].iloc[-1]) - vmin,
            note=RETAINED_EPOCH_NOTE))

    curves_df = pd.DataFrame(curves, columns=LEARNING_CURVE_COLUMNS)
    seed_df = pd.DataFrame(per_seed, columns=LEARNING_CURVE_SEED_COLUMNS)

    conv = convergence_diagnostics(cfg, log)
    status = dict(zip(conv["arm"], conv["status"])) if len(conv) else {}
    published_gap = dict(zip(conv["arm"], conv["val_overfit_gap"])) if len(conv) else {}
    arm_rows = []
    for arm, g in seed_df.groupby("arm", sort=False):
        mean_gap = float(g["val_overfit_gap_last_minus_min"].mean())
        if arm in published_gap:
            assert abs(mean_gap - float(published_gap[arm])) <= 1e-9, (
                f"{arm}: the mean last-minus-min gap here ({mean_gap}) is not the one "
                f"convergence_diagnostics computes ({published_gap[arm]}); one of the two "
                f"is reading train_history.csv wrongly")
        arm_rows.append(dict(
            arm=str(arm), n_seeds=int(len(g)),
            mean_retained_epoch=float(g["retained_epoch"].mean()),
            mean_epochs_run=float(g["n_epochs_run"].mean()),
            mean_train_nll_at_retained=float(g["train_nll_at_retained"].mean()),
            mean_val_nll_at_retained=float(g["val_nll_at_retained"].mean()),
            mean_gap_at_retained=float(g["gap_at_retained"].mean()),
            mean_val_overfit_gap=mean_gap,
            convergence_status=str(status.get(arm, "")),
            note=f"{TRAIN_VAL_GAP_NOTE} | {RETAINED_EPOCH_NOTE}"))
    arm_df = pd.DataFrame(arm_rows, columns=LEARNING_CURVE_ARM_COLUMNS)

    log.info("learning curves: %d epoch rows over %d arm x seed series, %d arms",
             len(curves_df), len(seed_df), len(arm_df))
    for _, r in arm_df.iterrows():
        log.info("  %-20s retained epoch %.1f of %.1f run | gap at retained %.4f | "
                 "last-minus-min %.4f (%s)", r["arm"], r["mean_retained_epoch"],
                 r["mean_epochs_run"] - 1, r["mean_gap_at_retained"],
                 r["mean_val_overfit_gap"], r["convergence_status"] or "ok")
    return curves_df, seed_df, arm_df


# =========================================================================== #
# 7b. DECISION-CURVE ANALYSIS - the censoring-aware net-benefit estimator      #
#                                                                              #
# Protocol section 18 pre-specifies exploratory decision-curve analysis and    #
# cites Vickers, so the analysis is pre-specified IN KIND; the numeric         #
# thresholds appear nowhere in the protocol and D18 recorded DCA as not run,   #
# so adding it after the sealed read is post-hoc IN TIMING (recorded as D29).  #
#                                                                              #
# THE ESTIMATOR. At threshold p_t the rule flags A = {i : risk_i >= p_t} and   #
# the event risk WITHIN A is estimated by Kaplan-Meier at the horizon:         #
#                                                                              #
#     w(p_t)  = p_t / (1 - p_t)                    exchange rate, TPs per FP   #
#     NB(p_t) = F_A(h) * (n_A/n) - (1 - F_A(h)) * (n_A/n) * w(p_t)             #
#     treat-all:  the same expression with A = everyone                        #
#     treat-none: 0 exactly, at every threshold, with no estimation at all     #
#                                                                              #
# Units: NET TRUE POSITIVES PER PATIENT SCREENED at the horizon. The true- and #
# false-positive terms come from the SAME Kaplan-Meier fit, so they partition  #
# the flagged set - TP + FP = n_A/n - and no censored patient is dropped from  #
# the numerator or counted in both halves. That is the whole reason the naive  #
# events/n version is wrong here: 63.8% of the test split is censored before   #
# day 1825, the naive 5-year event rate is 14.3% against a Kaplan-Meier 20.0%, #
# and treat-all crosses zero exactly at the prevalence - so a naive curve puts #
# that crossing 5.7 percentage points too low and roughly halves net benefit   #
# across the range.                                                            #
#                                                                              #
# Treat-all is not a separate formula. ``risk=None`` means "flag everyone" and #
# runs through :func:`net_benefit_at` unchanged, so the reference curve cannot #
# drift away from the model curves it is subtracted from.                      #
# =========================================================================== #

# The odds weight p/(1-p) amplifies any error in F_A: it is 19x larger at 0.50
# than at 0.05, and above an arm's maximum predicted risk the curve is
# identically zero (m0 tops out at 0.4715). Beyond ~0.30 the flagged sets also
# get small enough that a large share of bootstrap replicates have nobody in
# them followed as far as the horizon (35.6% for m4_fusion at p_t = 0.30, 100%
# for m0 at 0.40). The ceiling is frozen HERE, in the module, and checked
# against config, so a config edit alone cannot push the grid into that region.
NB_THRESHOLD_CEILING_PCT = 50

# The pinned ``{split}_net_benefit.csv`` schema. ORDER IS PART OF THE CONTRACT:
# write_table asserts the frame's columns match this list exactly.
#
# Filled by the estimator in this section:
#     split..threshold_pct        identity of the row
#     n_scored..km_last_obs_day   who was flagged and what the KM saw.
#                                 km_risk_above is the one ESTIMATE in that
#                                 run - F_A, the Kaplan-Meier risk inside the
#                                 set the model chose to flag - and the
#                                 convergence gate blanks it with the rest;
#                                 see NET_BENEFIT_SUPPRESSED_KEYS
#     net_benefit                 the point estimate
#     nb_treat_all_same_set       treat-all on the SAME patients (never a
#                                 different denominator)
#     diff_vs_treat_all           net_benefit minus the line above
#     net_reduction_per_100       100 * diff_vs_treat_all / w(p_t): avoidable
#                                 interventions per 100 patients screened
#     sparse                      events_above < sparse_events_min. FLAGGED,
#                                 NOT SUPPRESSED - the row keeps its estimate
#                                 and only the figure truncates a curve where
#                                 its flag first trips. Blanking exactly the
#                                 thresholds at which a model flags few events
#                                 would delete the evidence of imprecision a
#                                 reader needs.
#     note                        empty flagged set / carried-forward KM /
#                                 sparse, in words
# Filled by the bootstrap engine (a later stage), from the ONE shared draw:
#     net_benefit_lo/_hi, diff_vs_treat_all_lo/_hi/_p, diff_vs_reference*,
#     n_paired, n_replicates_valid
# Filled by the convergence gate, which also blanks every column in
# NET_BENEFIT_SUPPRESSED_KEYS on the rows it flags:
#     suppressed
# Filled by the sensitivity estimator :func:`net_benefit_ipcw_curve`:
#     net_benefit_ipcw
NET_BENEFIT_COLUMNS = [
    "split", "arm", "label", "threshold", "threshold_pct", "horizon_days",
    "n_scored", "n_above", "events_above", "km_risk_above", "km_last_obs_day",
    "net_benefit", "net_benefit_lo", "net_benefit_hi",
    "net_benefit_ipcw",
    "nb_treat_all_same_set",
    "diff_vs_treat_all", "diff_vs_treat_all_lo", "diff_vs_treat_all_hi", "diff_vs_treat_all_p",
    "net_reduction_per_100",
    "reference", "diff_vs_reference", "diff_vs_reference_lo", "diff_vs_reference_hi",
    "diff_vs_reference_p",
    "n_paired", "n_replicates_valid", "sparse", "suppressed", "note",
]


def _append_note(existing: str, addition: str) -> str:
    """The house ``"; "`` note idiom (``contrast_row``, ``suppress_unfit_contrasts``)."""
    return (existing + "; " + addition) if existing else addition


def threshold_grid(min_pct: int, max_pct: int) -> np.ndarray:
    """The threshold probabilities, built on an INTEGER-PERCENT basis.

    ``np.arange(min_pct, max_pct + 1, dtype=int) / 100.0``, never a float
    ``np.arange(0.01, 0.36, 0.01)``. A float arange accumulates its step, so its entries are
    not the nearest doubles to ``k/100``: on this numpy build the two grids already differ at
    0.35, and which entries differ is a property of the build. Since the flagged set is
    ``risk >= p_t``, a p_t that is one ulp off flags a different set of patients, so the
    whole curve would stop being reproducible across numpy versions. On the integer basis
    every entry is exactly the double ``k / 100`` and ``grid[6] == 0.07`` is exact.

    The ceiling is enforced here rather than in config: ``max_pct`` may not exceed the frozen
    :data:`NB_THRESHOLD_CEILING_PCT`.
    """
    assert isinstance(min_pct, (int, np.integer)) and isinstance(max_pct, (int, np.integer)), (
        f"the threshold grid is built on an integer-percent basis, so min_pct/max_pct must be "
        f"ints, not {type(min_pct).__name__}/{type(max_pct).__name__}; a float bound would "
        f"reintroduce exactly the accumulated-step problem the integer basis exists to avoid")
    min_pct, max_pct = int(min_pct), int(max_pct)
    assert 1 <= min_pct <= max_pct, (
        f"the threshold grid must run from at least 1% upwards, got {min_pct}..{max_pct}; "
        f"p_t = 0 is 'treat everyone at no cost', which is not a decision threshold")
    assert max_pct <= NB_THRESHOLD_CEILING_PCT, (
        f"model_eval.net_benefit.threshold_max_pct is {max_pct}%, above the frozen ceiling "
        f"NB_THRESHOLD_CEILING_PCT = {NB_THRESHOLD_CEILING_PCT}%. The odds weight p/(1-p) "
        f"amplifies error 19x more at 0.50 than at 0.05, m0's curve is identically zero above "
        f"its 0.4715 maximum predicted risk, and up there most bootstrap replicates have "
        f"nobody in the flagged set followed to the horizon. The ceiling is frozen in "
        f"src/eval_models.py so a config edit alone cannot move it")
    return np.arange(min_pct, max_pct + 1, dtype=int) / 100.0


def net_benefit_settings(cfg: Config) -> dict:
    """Read and check ``cfg["model_eval"]["net_benefit"]``, the decision-curve block.

    Read DIRECTLY, never through :func:`split_path`: ``net_benefit`` is a settings dict
    sitting one token away from the path key ``net_benefit_csv``, and ``split_path`` now
    refuses it precisely so that mistake fails loudly instead of resolving to a filename
    made out of a stringified dict.

    Returns the resolved settings plus the grid itself, so every caller (the engine, the
    writer, the figure) gets the same thresholds from one place.
    """
    me = cfg["model_eval"]
    assert "net_benefit" in me, (
        "model_eval has no net_benefit block; config/feasibility.yaml declares the "
        "decision-curve grid, arms, reference and horizon there")
    nb = me["net_benefit"]
    assert isinstance(nb, dict), (
        f"model_eval.net_benefit must be the settings block, not {type(nb).__name__}; the "
        f"OUTPUT PATH is model_eval.net_benefit_csv and is resolved by split_path")
    thresholds = threshold_grid(nb["threshold_min_pct"], nb["threshold_max_pct"])
    plot_min, plot_max = int(nb["plot_min_pct"]), int(nb["plot_max_pct"])
    tmin, tmax = int(nb["threshold_min_pct"]), int(nb["threshold_max_pct"])
    assert tmin <= plot_min <= plot_max <= tmax, (
        f"the plotted range {plot_min}-{plot_max}% is not inside the estimated range "
        f"{tmin}-{tmax}%; the figure can only draw thresholds the CSV carries")
    sparse_events_min = int(nb["sparse_events_min"])
    assert sparse_events_min >= 1, (
        f"sparse_events_min is {sparse_events_min}; the flag exists to mark thin flagged "
        f"sets, and a floor below one event marks nothing")
    arms = [str(a) for a in nb["arms"]]
    reference = str(nb["reference"])
    assert arms, "model_eval.net_benefit.arms is empty; there is no curve to draw"
    assert reference in arms, (
        f"model_eval.net_benefit.reference {reference!r} is not among the arms {arms}; every "
        f"diff_vs_reference would be against a curve the table does not carry")
    horizon_days = int(nb["horizon_days"])
    assert horizon_days > 0, f"net_benefit.horizon_days must be positive, got {horizon_days}"
    return {"thresholds": thresholds,
            "threshold_pcts": np.rint(thresholds * 100.0).astype(int),
            "threshold_min_pct": tmin, "threshold_max_pct": tmax,
            "plot_min_pct": plot_min, "plot_max_pct": plot_max,
            "sparse_events_min": sparse_events_min,
            "arms": arms, "reference": reference, "horizon_days": horizon_days}


def net_benefit_at(time, event, risk, threshold: float, *, horizon: float,
                   n_scored: int | None = None) -> dict:
    """Net benefit at ONE threshold, on ONE set of patients. The whole estimator.

    ``time``/``event``/``risk`` are parallel arrays over the patients being screened - the
    ones an arm actually scores, or one bootstrap replicate's draw of them (repeats are fine
    and are what makes the replicate a replicate). ``risk=None`` means FLAG EVERYONE, which
    is how treat-all is computed: it is the same code path, the same Kaplan-Meier call and
    the same arithmetic as every model curve, so it cannot drift away from the curves it is
    subtracted from.

    ``n_scored`` is the denominator ``n`` - "per patient screened". It defaults to the size
    of the arrays, which is the right answer whenever the caller has already restricted to
    the arm's own patients (the arms rest on deliberately different populations, 741 / 740 /
    734 / 707, and are never harmonised).

    EVERY case is defined and none is NaN:

    * **empty flagged set** -> ``net_benefit`` is ``0.0`` exactly. Nobody is treated, so
      TP = FP = 0 and the rule has BECOME treat-none. NaN would truncate the curve and hide
      that a model whose predicted risks top out early has degenerated into treat-none;
    * **flagged set with no observed event** -> the negative floor ``-(n_A/n) * w(p_t)``.
      Blanking exactly the thresholds at which a model flags non-events would delete the
      evidence of harm;
    * **nobody in the flagged set followed to the horizon** -> ``km_cif_numpy`` carries the
      last observed value of the curve forward, ``km_last_obs_day`` records how far the
      follow-up actually reached, and ``note`` says so. Carrying forward can only understate
      F_A, so it biases net benefit DOWN and is conservative for the arm under evaluation.

    ``model_clinical.km_risk`` is deliberately not used: it is lifelines-backed at ~2.2 ms
    against 20 us here (the curve needs ~1e5 fits) and it RAISES on an empty flagged set,
    which is a case this estimator must answer rather than fail on.

    Returned keys. CSV columns: ``threshold``, ``horizon_days``, ``n_scored``, ``n_above``,
    ``events_above``, ``km_risk_above``, ``km_last_obs_day``, ``net_benefit``, ``note``.
    Diagnostics, not columns: ``weight`` (the exchange rate w) and ``tp`` / ``fp`` (the
    true- and false-positive terms per patient screened, which sum to ``n_above/n_scored``;
    both are recoverable from ``km_risk_above`` and ``n_above``).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    assert t.ndim == 1 and t.shape == e.shape, (
        f"time {t.shape} and event {e.shape} must be parallel 1-D arrays over the patients "
        f"being screened")
    n = int(t.size) if n_scored is None else int(n_scored)
    assert n >= t.size and n > 0, (
        f"n_scored = {n} is not a usable denominator for {t.size} screened patients; net "
        f"benefit is reported per patient screened")
    p = float(threshold)
    assert 0.0 <= p < 1.0, (
        f"the threshold probability must lie in [0, 1), got {p}; w = p/(1-p) is undefined "
        f"at 1")

    if risk is None:                       # TREAT-ALL: flag everyone, same code path
        flag = np.ones(t.size, dtype=bool)
    else:
        r = np.asarray(risk, dtype=float)
        assert r.shape == t.shape, f"risk {r.shape} does not line up with time {t.shape}"
        assert bool(np.isfinite(r).all()), (
            "a patient with a non-finite predicted risk reached net_benefit_at. "
            "ArmScores.risk holds NaN for every patient the arm cannot score, and NaN >= p_t "
            "is False, so such a patient would silently inflate the denominator without ever "
            "being flagged. Restrict to ArmScores.present BEFORE calling")
        flag = r >= p

    n_above = int(flag.sum())
    events_above = int(e[flag].sum())
    cif, last_obs = km_cif_numpy(t[flag], e[flag], float(horizon))
    weight = p / (1.0 - p)

    # TP and FP in PATIENT units, from the one Kaplan-Meier fit. Written as a residual so
    # the flagged set is partitioned exactly: ``tp_patients + fp_patients == n_above`` holds
    # to the last bit because n_above is an exact integer, which is the arithmetic statement
    # of "no censored patient is dropped or double-counted". Dividing by n gives TP and FP
    # per patient screened, whose sum reproduces n_A/n to within one ulp.
    tp_patients = cif * n_above
    fp_patients = n_above - tp_patients
    net_benefit = (tp_patients - fp_patients * weight) / n

    note = ""
    if n_above == 0:
        note = ("no patient reaches this threshold, so the rule has become treat-none; the "
                "net benefit is 0 by construction, not missing")
    elif last_obs < float(horizon):
        note = (f"nobody in the flagged set was followed to day {float(horizon):g}; the "
                f"Kaplan-Meier estimate is carried forward from day {last_obs:g}, which can "
                f"only understate the event risk and so biases net benefit down")
    return {"threshold": p,
            "horizon_days": int(horizon),
            "weight": float(weight),
            "n_scored": n,
            "n_above": n_above,
            "events_above": events_above,
            "km_risk_above": float(cif),
            "km_last_obs_day": float(last_obs),
            "tp": tp_patients / n,
            "fp": fp_patients / n,
            "net_benefit": float(net_benefit),
            "note": note}


def net_benefit_curve(time, event, risk, thresholds, *, horizon: float,
                      sparse_events_min: int, n_scored: int | None = None,
                      treat_all: bool = True) -> list[dict]:
    """One arm's whole curve: :func:`net_benefit_at` at every threshold, plus treat-all.

    ``thresholds`` must be strictly ascending and on the integer-percent basis
    :func:`threshold_grid` produces; that is asserted rather than assumed, because a float
    ``np.arange`` grid is the one input that would silently change which patients are
    flagged.

    ``treat_all=True`` also evaluates ``risk=None`` on the SAME patients at the SAME
    threshold, so ``nb_treat_all_same_set`` and ``diff_vs_treat_all`` are never a comparison
    across two different denominators. ``net_reduction_per_100`` is the standard reading of
    that difference: ``100 * (NB - NB_all) / w(p_t)``, interventions avoided per 100
    patients screened at no cost in missed events. Pass ``treat_all=False`` for the passes
    that do not need it (the reference arm re-scored on a contrast's intersection), which
    halves the Kaplan-Meier fits.

    ``sparse_events_min`` comes from config (15) and is required rather than defaulted, so
    the floor has exactly one source of truth. A sparse row is FLAGGED, NOT SUPPRESSED: it
    keeps its estimate, and truncating the drawn curve is the figure's job.

    Treat-none needs no entry here at all - its net benefit is 0.0 at every threshold by
    definition, with nothing estimated.

    Returns one dict per threshold, carrying :func:`net_benefit_at`'s keys plus
    ``threshold_pct``, ``sparse`` and, when ``treat_all``, ``nb_treat_all_same_set`` /
    ``diff_vs_treat_all`` / ``net_reduction_per_100``. The bootstrap engine fills the
    interval, p-value and reference columns; nothing here touches them.
    """
    th = np.asarray(thresholds, dtype=float)
    assert th.ndim == 1 and th.size > 0, "the threshold grid must be a non-empty 1-D array"
    assert bool(np.all(np.diff(th) > 0)), (
        "the threshold grid must be strictly ascending; the flagged sets are then nested, "
        "which is what makes a curve a curve")
    pcts = np.rint(th * 100.0).astype(int)
    assert np.array_equal(pcts / 100.0, th), (
        "the thresholds are not on the integer-percent basis. They were almost certainly "
        "built with a float np.arange, whose accumulated step leaves an entry one ulp off "
        "the nearest double to k/100 - and `risk >= p_t` then flags a different set of "
        "patients. Build them with threshold_grid()")
    floor = int(sparse_events_min)

    rows: list[dict] = []
    for p, pct in zip(th, pcts):
        row = net_benefit_at(time, event, risk, float(p), horizon=horizon, n_scored=n_scored)
        row["threshold_pct"] = int(pct)
        row["sparse"] = bool(row["events_above"] < floor)
        if row["sparse"]:
            row["note"] = _append_note(
                row["note"],
                f"sparse: {row['events_above']} observed event(s) in the flagged set, below "
                f"the {floor}-event floor; the estimate stands but is imprecise")
        if treat_all:
            ref = net_benefit_at(time, event, None, float(p), horizon=horizon,
                                 n_scored=n_scored)
            row["nb_treat_all_same_set"] = ref["net_benefit"]
            row["diff_vs_treat_all"] = row["net_benefit"] - ref["net_benefit"]
            row["net_reduction_per_100"] = (
                100.0 * row["diff_vs_treat_all"] / row["weight"] if row["weight"] > 0
                else float("nan"))
        rows.append(row)
    return rows


def net_benefit_ipcw_curve(time, event, risk, thresholds, *, horizon: float, g_grid, g_vals,
                           n_scored: int | None = None) -> np.ndarray:
    """The IPCW net-benefit curve - a POINT-ESTIMATE-ONLY sensitivity column.

    Per-patient inverse-probability-of-censoring weighting, using the shared censoring curve
    ``(g_grid, g_vals)``:

        TP(p_t) = (1/n) * sum_{i in A} 1{event by h} * w_i
        FP(p_t) = (1/n) * sum_{i in A} 1{event-free past h} * w_i
        NB      = TP - FP * w(p_t)

    with the weights ``ipcw_labels_weights`` already builds for every other estimand in this
    module (cases carry 1/G(T_i-), controls 1/G(h), and a patient censored before the horizon
    carries weight 0). It is emitted so the question "would IPCW have said something else?"
    is answered inside the artefact rather than by a reviewer's imagination.

    WHY KAPLAN-MEIER WITHIN THE FLAGGED SET IS PRIMARY AND THIS IS THE SENSITIVITY.

    1. **The assumption, not convenience.** IPCW with the marginal reverse-Kaplan-Meier
       censoring curve requires censoring to be independent of the PREDICTORS. Here follow-up
       is administrative EHR contact, which plausibly tracks comorbidity and radiographic
       severity - exactly what the model reads off the film - so that assumption is not one
       this cohort earns. Kaplan-Meier within the flagged set needs only censoring
       independent of outcome CONDITIONAL ON MEMBERSHIP OF A, which is strictly weaker: A is
       itself a function of the predictors, so conditioning on it absorbs precisely the
       dependence IPCW has to assume away.
    2. **The weights are a TRAINING quantity.** Every arm shares M0's frozen
       ``censoring_km_train``, which is what makes an AUROC difference a difference between
       models rather than between weightings. But ``G(1825) = 0.2962`` frozen against 0.2735
       re-estimated on the test split - a 7.7% relative gap - and net benefit is an ABSOLUTE
       estimate, not a contrast in which a common nuisance parameter cancels. Importing a
       train-population nuisance parameter into an absolute number buys nothing and costs
       that gap.
    3. **Exactness.** The Kaplan-Meier form partitions the flagged set: TP + FP = n_A/n by
       construction. The IPCW form's weighted case and control masses only sum to n_A/n in
       expectation, so its true- and false-positive terms are not two halves of one quantity.
    4. **The two DISAGREE at this sample size, and this column exists to show that.** An
       earlier version of this docstring claimed the two "agree to within 0.005 for
       p_t <= 0.30". That claim is FALSE on this study's own data; it was carried over from a
       simulation at n = 6,000. Measured on the artefacts, across the four decision-curve
       arms and p_t <= 0.30, in net true positives per patient screened:

       * SEALED TEST SPLIT (741 patients, 106 events). The largest absolute disagreement is
         **0.0161** (m0, at p_t = 0.20), then 0.0098 (m4_fusion, at 0.05), 0.0078
         (m2_frontal, at 0.24) and 0.0077 (m1, at 0.17) - three times the bound that used to
         be claimed. At the headline p_t = 0.20 the m0 curve reads +0.0396 under the primary
         estimator against +0.0557 under IPCW: a gap of **41% of the estimate itself**. The
         image arms hold together better at that threshold (0.003 to 0.005, which is 4.5% to
         6.9% of their estimates), so the worst case is the CLINICAL comparator - and that is
         the arm every contrast is taken against.
       * VALIDATION (371 patients, 54 events). The largest disagreement is **0.0283**
         (m2_frontal, at p_t = 0.26), with 0.0216 (m4_fusion), 0.0200 (m0) and 0.0129 (m1);
         at p_t = 0.20 the gap is 27% of m2_frontal's estimate and 40% of m0's.
       * The FROZEN G is where it comes from, exactly as point 2 predicts. Weighting the
         sealed split with M0's ``censoring_km_train`` implies a 5-year risk of 0.2125
         against the Kaplan-Meier 0.2004, so even TREAT-ALL moves: up to 0.0252 over
         p_t <= 0.30, +0.0005 against +0.0147 at p_t = 0.20, and its zero crossing shifts
         from 21% to 22%. That is diagnostic rather than incidental, because treat-all
         involves no subsetting at all: under a G estimated from the sample ITSELF the IPCW
         estimator of the marginal risk IS the Kaplan-Meier estimator. Re-estimate G on the
         test split and the two treat-all curves come back together to 4e-5 - about 500
         times closer - with that residual being tie handling rather than estimation, since
         follow-up is in whole days and 33 of the 106 test events fall on a day that also
         carries a censoring. The disagreement is imported, not discovered.
       * It is an **n effect**, not a defect in either estimator. On simulated cohorts whose
         censoring is drawn INDEPENDENTLY of risk - the assumption IPCW needs and this cohort
         cannot promise - 11 of 20 seeded replications at n = 741 exceed 0.005, 17 of 20 do
         at n = 400, and at n = 6,000 none do.

    5. **What survives the choice of estimator is the CONTRAST, in a precise sense.** The
       frozen-G bias is largely common to two arms scored on the same patients, so it partly
       cancels in a difference: arm minus m0, taken on the intersection both arms score,
       moves by at most 0.0132 on test (m2_frontal, at p_t = 0.14) and 0.0209 on validation,
       against 0.0161 and 0.0283 for the curves themselves. PARTLY, not wholly - at
       p_t = 0.20 on test m2_frontal minus m0 is +0.0495 under the primary estimator and
       +0.0390 under IPCW, still 21% of the estimate. What is genuinely invariant is every
       comparison the figure is read for. Against the 2,000-replicate paired bootstrap on the
       sealed split, over the three contrasts and the 30 plotted thresholds:

       * the IPCW contrast lies INSIDE the primary estimator's own 95% interval at 89 of
         those 90 points (the exception is m2_frontal at p_t = 0.01: +0.00028 against
         [+0.00036, +0.00227], a 0.0001 miss at a threshold two orders of magnitude below the
         headline effect);
       * the gap is at most 0.93 of that interval's half-width (at p_t = 0.01) and 0.33 to
         0.43 of it at p_t = 0.20, so it is never larger than the sampling uncertainty the
         figure already draws;
       * wherever the primary contrast excludes zero, the IPCW contrast carries the same
         sign, for all three arms.

       SO: report the arm-versus-arm contrast confidently, and read ABSOLUTE net benefit as
       estimator-dependent at this sample size. The height of a curve at a given threshold,
       and the threshold from which a curve first clears treat-all, both move with the
       estimator; which arm is better, at the thresholds where the paired interval says so,
       does not.

    Degenerate cases follow the primary estimator: an empty flagged set gives 0.0, and a
    flagged set whose weighted case and control masses are both zero gives 0.0. Nothing here
    calls ``model_clinical.km_risk``, which raises on empty input.

    Returns a ``(len(thresholds),)`` array of net benefits, aligned with ``thresholds``.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    assert t.ndim == 1 and t.shape == e.shape, (
        f"time {t.shape} and event {e.shape} must be parallel 1-D arrays")
    n = int(t.size) if n_scored is None else int(n_scored)
    assert n >= t.size and n > 0, (
        f"n_scored = {n} is not a usable denominator for {t.size} screened patients")
    th = np.asarray(thresholds, dtype=float)
    assert th.ndim == 1 and th.size > 0, "the threshold grid must be a non-empty 1-D array"
    assert bool(np.all((th >= 0.0) & (th < 1.0))), "every threshold must lie in [0, 1)"

    # The weights do not depend on the threshold, so they are built ONCE for the whole
    # screened set and only the membership mask moves.
    y, w = ipcw_labels_weights(t, e, float(horizon), g_grid, g_vals)
    case, ctrl = (y == 1), (y == 0)

    if risk is None:
        flags = [np.ones(t.size, dtype=bool)] * th.size
    else:
        r = np.asarray(risk, dtype=float)
        assert r.shape == t.shape, f"risk {r.shape} does not line up with time {t.shape}"
        assert bool(np.isfinite(r).all()), (
            "a patient with a non-finite predicted risk reached net_benefit_ipcw_curve; "
            "restrict to ArmScores.present before calling")
        flags = [r >= float(p) for p in th]

    out = np.empty(th.size, dtype=float)
    for i, (p, flag) in enumerate(zip(th, flags)):
        tp = float(w[flag & case].sum())
        fp = float(w[flag & ctrl].sum())
        out[i] = (tp - fp * (float(p) / (1.0 - float(p)))) / n
    return out


# --------------------------------------------------------------------------- #
# 7c. THE ENGINE. The curves above are pure; this drives them from the ONE      #
#     shared draw and assembles {split}_net_benefit.csv.                        #
# --------------------------------------------------------------------------- #

# What suppression blanks on a decision-curve row. Every model-derived ESTIMATE in the
# schema AND every column the estimate can be reconstructed from, and nothing else: an arm
# that never fitted a model has no net benefit to report, and leaving one of its estimates -
# or the last term of the identity that produces one - in a neighbouring column would let a
# reader recover exactly the number the gate exists to withhold. ``net_benefit_ipcw`` is on
# this list because it estimates the SAME quantity as ``net_benefit`` by a second route: a
# suppressed row whose sensitivity column still reads +0.09 has not been suppressed.
#
# ``km_risk_above`` is on this list for the stronger form of that same argument, and was
# added after a reviewer closed the arithmetic on a published table. The estimator is
#
#     NB = F_A * (n_A/n) - (1 - F_A) * (n_A/n) * w,        w = p_t / (1 - p_t)
#
# so ``threshold`` with ``km_risk_above``, ``n_above`` and ``n_scored`` reproduces a blanked
# ``net_benefit`` in one line. Measured on all 140 sealed rows, that identity returns
# ``net_benefit`` to 7.5e-7 and ``diff_vs_treat_all`` to 9.0e-7 - and the table is written
# rounded to six decimals, so the reconstruction is exact to the last digit anyone can read.
# F_A is a Kaplan-Meier fit on the set the MODEL chose to flag: it is as much an output of
# the model as the net benefit it multiplies. A validation render, where m2_frontal and
# m4_fusion are severe_overfit and therefore suppressed, would otherwise have shipped a
# val_net_benefit.csv publishing precisely the estimates the gate had just withheld.
#
# Deliberately NOT blanked. Each survives the same test - none of them recovers the estimate:
#   n_scored / n_above / n_paired
#       how many patients a threshold flags, and out of how many. A property of the risk
#       DISTRIBUTION with no outcome in it, and the descriptive a reader needs to see that
#       the row was specified and why it is not reported, exactly as a suppressed subgroup
#       keeps its n. With F_A gone they do not close the arithmetic: n_A/n and w bound NB
#       only to [-(n_A/n) * w, n_A/n], a median width of 0.41 against a median |NB| of 0.085.
#   events_above
#       an OBSERVED count, and what makes the ``sparse`` flag beside it readable. Its naive
#       rate events_above/n_above is NOT F_A and does not stand in for one: 63.8% of the
#       split is censored before day 1825, so across all 140 sealed rows it sits 0.045 or
#       more BELOW the Kaplan-Meier risk, never once above it, and the net benefit it
#       reconstructs misses by a median 0.047 - 55% of the median |NB| itself, up to 33x |NB|
#       at the top of the grid - and by no less than 0.0024 on any row. That is a different,
#       biased estimator any reader could compute from any censored cohort, not a recovery
#       of this one. Both floors are re-measured by a test against the written table, so a
#       cohort with light censoring cannot inherit this reasoning silently.
#   km_last_obs_day
#       how far follow-up inside the flagged set actually reached. It does not enter the
#       estimator and cannot be inverted to F_A; it is what makes the carried-forward
#       Kaplan-Meier sentence in ``note`` mean anything.
#   nb_treat_all_same_set
#       treat-all flags EVERYONE, so it depends only on which patients the arm scores and
#       carries no output of the model at all. It is the reference line the suppressed curve
#       would have been drawn against, and it stays readable. ONE residual is accepted
#       knowingly: below an arm's minimum predicted risk the arm flags everyone, n_above ==
#       n_scored, and the row's own net benefit IS the treat-all value (5 of the 140 sealed
#       rows, all at p_t <= 4%). At such a threshold the rule has degenerated into treat-all
#       and the number is a property of the cohort's Kaplan-Meier risk rather than of the
#       fitted model - which is the same reason the column is published at all.
NET_BENEFIT_SUPPRESSED_KEYS = (
    "km_risk_above",
    "net_benefit", "net_benefit_lo", "net_benefit_hi", "net_benefit_ipcw",
    "diff_vs_treat_all", "diff_vs_treat_all_lo", "diff_vs_treat_all_hi",
    "diff_vs_treat_all_p", "net_reduction_per_100",
    "diff_vs_reference", "diff_vs_reference_lo", "diff_vs_reference_hi",
    "diff_vs_reference_p",
)

# Pointwise, and said so in the artefact. 35 thresholds are 35 views of ONE curve, not 35
# hypotheses: the flagged sets are nested by construction, so the estimates are almost
# perfectly dependent and a family-wise correction across them would be a correction for a
# multiplicity that does not exist. Every interval and every p below is therefore pointwise
# and unadjusted, and the figure caption says so.
NET_BENEFIT_MULTIPLICITY_NOTE = (
    "pointwise 95% percentile-bootstrap intervals and unadjusted two-sided bootstrap p "
    "values, from the one shared patient-level draw. NO multiplicity adjustment is applied "
    "across thresholds: the flagged sets are nested, so the grid is 35 views of one curve "
    "rather than 35 hypotheses")


class NetBenefitEngine:
    """Point and bootstrap decision curves for (arm, patient-set) pairs, on the SHARED draw.

    Deliberately a SECOND engine beside :class:`BootstrapEngine` rather than more metrics
    inside it, for three reasons that are all properties of that class rather than matters
    of taste:

    * its cache key is ``(arm, packbits(mask))`` with no horizon and no metric-set
      component, and it is exercised for every arm, every contrast intersection and every
      subgroup level - so a net-benefit entry would collide with a metrics entry;
    * ``boot_metric_keys`` seeds ``{k: nan for k in keys}`` and fills only the names
      ``arm_metrics`` knows, so an added key returns silently all-NaN, and it feeds the
      ``val_metrics.csv`` schema that three tests pin;
    * it would multiply the 2,000-replicate loop for every one of those combinations, to
      compute a curve that only four arms and two masks each ever need.

    What the two engines DO share is the thing that has to be shared: ``draw`` is the same
    object, never a copy and never re-rolled, so replicate *b* is the same set of patients
    here as in every AUROC contrast and the paired difference is already paired.

    THE DENOMINATOR RULE, which is the whole reason this class exists rather than a loop in
    ``main``. Every arm is evaluated on ITS OWN patients with ``n_scored`` equal to that set
    - the arms rest on deliberately different populations (741 / 740 / 734 / 707 with 106 /
    106 / 98 events) and harmonising them onto the 707 intersection would throw away 8 of
    106 events to buy nothing. Every paired CONTRAST is evaluated on the intersection, both
    arms recomputed there, and the row's ``note`` says so. And the risks are
    ``ArmScores.risk[horizon]``, which carries the frozen cloglog recalibration on the image
    arms: measuring on unrecalibrated risks moves the flagged set at a given threshold and
    therefore the whole curve.

    ``BootstrapEngine.boot``'s ``take = idx[usable[idx]]`` is mirrored EXACTLY: a replicate
    drops the patients an arm cannot score, so replicate sizes differ per arm, and any other
    rule here would silently unpair the difference the shared draw exists to provide.
    """

    def __init__(self, roster: Roster, draw: np.ndarray, thresholds, horizon_days: int,
                 sparse_events_min: int, log: logging.Logger):
        self.roster, self.draw, self.log = roster, draw, log
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.horizon_days = int(horizon_days)
        self.sparse_events_min = int(sparse_events_min)
        assert self.thresholds.ndim == 1 and self.thresholds.size > 0, (
            "the decision-curve engine needs a non-empty threshold grid")
        self._point: dict[tuple[str, bytes], list[dict]] = {}
        self._ipcw: dict[tuple[str, bytes], np.ndarray] = {}
        self._boot: dict[tuple[str, bytes], dict[str, np.ndarray]] = {}

    @staticmethod
    def _key(scores: ArmScores, mask: np.ndarray) -> tuple[str, bytes]:
        return (scores.arm, np.packbits(mask).tobytes())

    def _restrict(self, scores: ArmScores, mask: np.ndarray):
        """``(time, event, risk, n)`` for the patients this arm scores inside ``mask``."""
        take = np.flatnonzero(mask & scores.present)
        return (self.roster.time[take], self.roster.event[take],
                scores.risk[self.horizon_days][take], int(take.size))

    def point(self, scores: ArmScores, mask: np.ndarray) -> list[dict]:
        """One arm's point curve on ``mask``, treat-all included and always cached.

        Treat-all costs one extra Kaplan-Meier fit per threshold, which is 35 fits once -
        nothing beside the 2,000-replicate loop - so it is never skipped here and the
        cached entry is always complete. Skipping it is worth doing only inside ``boot``.
        """
        key = self._key(scores, mask)
        if key not in self._point:
            t, e, r, n = self._restrict(scores, mask)
            self._point[key] = net_benefit_curve(
                t, e, r, self.thresholds, horizon=float(self.horizon_days),
                sparse_events_min=self.sparse_events_min, n_scored=n, treat_all=True)
        return self._point[key]

    def ipcw(self, scores: ArmScores, mask: np.ndarray) -> np.ndarray:
        """The IPCW sensitivity curve, point estimate only - no interval is claimed."""
        key = self._key(scores, mask)
        if key not in self._ipcw:
            t, e, r, n = self._restrict(scores, mask)
            self._ipcw[key] = net_benefit_ipcw_curve(
                t, e, r, self.thresholds, horizon=float(self.horizon_days),
                g_grid=self.roster.g_grid, g_vals=self.roster.g_vals, n_scored=n)
        return self._ipcw[key]

    def boot(self, scores: ArmScores, mask: np.ndarray, *,
             treat_all: bool = True) -> dict[str, np.ndarray]:
        """``(n_boot, n_thresholds)`` net benefit, and its difference from treat-all.

        ``treat_all=False`` halves the Kaplan-Meier fits and is what the reference arm's
        pass on a contrast intersection wants: that pass exists only to be subtracted, and
        the treat-all line it would compute belongs to the arm's own row, not to this one.
        A cached entry that omitted treat-all is recomputed if a later caller needs it, so
        the saving can never turn into a missing column.

        A replicate that draws fewer than :data:`MIN_PATIENTS_PER_REPLICATE` patients this
        arm scores is left NaN and drops out of the percentile interval, exactly as an
        under-powered replicate does in ``arm_metrics``. There is deliberately no minimum
        EVENT count: unlike a C-index, net benefit is defined at zero events - it is the
        negative floor - and discarding those replicates would trim the harmful tail of the
        distribution and narrow every interval in the direction of the model.
        """
        key = self._key(scores, mask)
        hit = self._boot.get(key)
        if hit is not None and (not treat_all or "diff_vs_treat_all" in hit):
            return hit
        usable = mask & scores.present
        risk = scores.risk[self.horizon_days]
        n_boot, n_th = len(self.draw), int(self.thresholds.size)
        nb = np.full((n_boot, n_th), np.nan)
        da = np.full((n_boot, n_th), np.nan) if treat_all else None
        for b in range(n_boot):
            idx = self.draw[b]
            take = idx[usable[idx]]                 # BootstrapEngine.boot, verbatim
            if take.size < MIN_PATIENTS_PER_REPLICATE:
                continue
            rows = net_benefit_curve(
                self.roster.time[take], self.roster.event[take], risk[take], self.thresholds,
                horizon=float(self.horizon_days), sparse_events_min=self.sparse_events_min,
                n_scored=int(take.size), treat_all=treat_all)
            nb[b] = [r["net_benefit"] for r in rows]
            if treat_all:
                da[b] = [r["diff_vs_treat_all"] for r in rows]
        out = {"net_benefit": nb}
        if treat_all:
            out["diff_vs_treat_all"] = da
        self._boot[key] = out
        self.log.info("  net benefit: bootstrapped %-20s on %d patients over %d replicates "
                      "(%d estimable%s)", scores.arm, int(usable.sum()), n_boot,
                      int(np.isfinite(nb[:, 0]).sum()),
                      "" if treat_all else ", treat-all skipped")
        return out


def _pointwise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise percentile interval over replicates: ``(lo, hi)``, one per threshold.

    :func:`percentile_ci` is reused rather than reimplemented so the decision curve's bands
    and every AUROC interval in the study are the same estimator - including its rule that
    fewer than two finite replicates give NaN rather than a degenerate point.
    """
    pairs = [percentile_ci(values[:, j]) for j in range(values.shape[1])]
    return (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


def build_net_benefit(cfg: Config, arms: dict[str, ArmScores], engine: NetBenefitEngine,
                      log: logging.Logger, *, split: str,
                      convergence: pd.DataFrame | None = None,
                      sealed: bool = False) -> pd.DataFrame:
    """``{split}_net_benefit.csv``: one row per declared arm per threshold.

    Arm order is the configured order (``m2_frontal``, ``m1``, ``m0``, ``m4_fusion``) and
    thresholds ascend inside it, so the file is deterministic. ``m4_fusion`` is in that list
    although ``m2_frontal`` is the protagonist: protocol section 18 pins M4 vs M0 as the
    primary contrast, and M2's curve lying on M4's IS the visual argument that the extra
    views and the clinical data buy nothing.

    Three quantities per row, each from the one shared draw:

    * ``net_benefit`` with a pointwise interval, on the arm's OWN patients;
    * ``diff_vs_treat_all`` - treat-all evaluated on those same patients at that same
      threshold, never a different denominator - with a pointwise interval and p;
    * ``diff_vs_reference`` - the arm minus ``net_benefit.reference``, both recomputed on
      the INTERSECTION the two arms score, with a pointwise interval and p.

    The reference's own row runs through the identical code path against itself, so its
    ``diff_vs_reference`` is exactly 0 with a zero-width interval and p = 1, and the figure
    can draw it as the flat zero line it is rather than special-casing a NaN.

    Every interval is POINTWISE and no multiplicity adjustment is applied across thresholds;
    see :data:`NET_BENEFIT_MULTIPLICITY_NOTE`.

    An arm the ladder has not trained is skipped and logged, as everywhere else in this
    module. If the REFERENCE is missing there is no contrast column to fill, so the schema
    is written empty rather than half-filled.
    """
    nb = net_benefit_settings(cfg)
    assert np.array_equal(engine.thresholds, nb["thresholds"]), (
        "the engine's threshold grid is not the configured one; the writer and the engine "
        "must read model_eval.net_benefit through net_benefit_settings exactly once")
    assert engine.horizon_days == nb["horizon_days"], (
        f"the engine evaluates day {engine.horizon_days} and config declares "
        f"{nb['horizon_days']}")
    assert engine.sparse_events_min == nb["sparse_events_min"], (
        "the sparse-event floor has one source of truth, model_eval.net_benefit")
    conv = convergence if convergence is not None else pd.DataFrame(columns=CONVERGENCE_COLUMNS)
    n_boot = len(engine.draw)
    reference = nb["reference"]
    ref = arms.get(reference)
    if ref is None:
        log.warning("net benefit: the reference arm %r is not scored, so no decision curve "
                    "is estimable; %s_net_benefit.csv is written with its schema and no rows",
                    reference, split)
        return pd.DataFrame(columns=NET_BENEFIT_COLUMNS)

    rows: list[dict] = []
    for arm in nb["arms"]:
        sc = arms.get(arm)
        if sc is None:
            log.info("net benefit: arm %-20s SKIPPED, it is not scored yet", arm)
            continue
        own = sc.present
        n_own = int(own.sum())
        pair = own & ref.present
        n_paired = int(pair.sum())
        assert n_paired > 0, (
            f"{arm} and the decision-curve reference {reference} score no patient in common, "
            f"so no paired difference exists")

        curve = engine.point(sc, own)
        ipcw = engine.ipcw(sc, own)
        bt = engine.boot(sc, own, treat_all=True)
        nb_lo, nb_hi = _pointwise(bt["net_benefit"])
        da_lo, da_hi = _pointwise(bt["diff_vs_treat_all"])

        # The paired difference. For the reference itself ``pair`` IS ``own``, so both
        # lookups hit the same cache entry and the difference is exactly zero replicate by
        # replicate - the identity, not a special case bolted on beside it.
        d_pt = (np.array([r["net_benefit"] for r in engine.point(sc, pair)])
                - np.array([r["net_benefit"] for r in engine.point(ref, pair)]))
        d_boot = (engine.boot(sc, pair, treat_all=False)["net_benefit"]
                  - engine.boot(ref, pair, treat_all=False)["net_benefit"])
        dr_lo, dr_hi = _pointwise(d_boot)

        # How many replicates actually support THIS row. Reported rather than assumed: a
        # replicate is estimable when both the arm's own pass and the paired pass produced a
        # finite value, which is the intersection every column in the row rests on.
        n_valid = (np.isfinite(bt["net_benefit"]) & np.isfinite(d_boot)).sum(axis=0)

        pair_note = ""
        if n_paired < n_own or n_paired < ref.n_patients:
            pair_note = (f"diff_vs_reference is paired on the {n_paired} patients both arms "
                         f"score ({arm} scores {n_own}, {reference} scores "
                         f"{ref.n_patients}); net_benefit is on this arm's own {n_own}")

        for j, base in enumerate(curve):
            note = base["note"]
            if pair_note:
                note = _append_note(note, pair_note)
            if int(n_valid[j]) < n_boot:
                note = _append_note(note, f"{n_boot - int(n_valid[j])} of {n_boot} replicates "
                                          f"were not estimable and are excluded")
            da = bt["diff_vs_treat_all"][:, j]
            rows.append({
                "split": split, "arm": arm, "label": sc.label,
                "threshold": base["threshold"], "threshold_pct": base["threshold_pct"],
                "horizon_days": base["horizon_days"],
                "n_scored": base["n_scored"], "n_above": base["n_above"],
                "events_above": base["events_above"],
                "km_risk_above": base["km_risk_above"],
                "km_last_obs_day": base["km_last_obs_day"],
                "net_benefit": base["net_benefit"],
                "net_benefit_lo": float(nb_lo[j]), "net_benefit_hi": float(nb_hi[j]),
                "net_benefit_ipcw": float(ipcw[j]),
                "nb_treat_all_same_set": base["nb_treat_all_same_set"],
                "diff_vs_treat_all": base["diff_vs_treat_all"],
                "diff_vs_treat_all_lo": float(da_lo[j]),
                "diff_vs_treat_all_hi": float(da_hi[j]),
                "diff_vs_treat_all_p": two_sided_bootstrap_p(da)[0],
                "net_reduction_per_100": base["net_reduction_per_100"],
                "reference": reference,
                "diff_vs_reference": float(d_pt[j]),
                "diff_vs_reference_lo": float(dr_lo[j]),
                "diff_vs_reference_hi": float(dr_hi[j]),
                "diff_vs_reference_p": two_sided_bootstrap_p(d_boot[:, j])[0],
                "n_paired": n_paired, "n_replicates_valid": int(n_valid[j]),
                "sparse": bool(base["sparse"]), "suppressed": False, "note": note,
            })

        first_sparse = next((r["threshold_pct"] for r in curve if r["sparse"]), None)
        log.info("net benefit %-20s n=%-4d paired=%-4d vs %-4s | %s | pointwise, unadjusted "
                 "across thresholds", arm, n_own, n_paired, reference,
                 f"sparse from p_t = {first_sparse}%" if first_sparse is not None
                 else "no threshold trips the sparse floor")

    # The SAME gate as every other contrast, with this schema's key names. did_not_converge
    # suppresses on both splits; severe_overfit on validation only, because a test-split
    # estimate took no part in the checkpoint selection that made the val one optimistic.
    suppress_unfit_contrasts(rows, conv, log, sealed=sealed,
                             arm_keys=("arm", "reference"),
                             blank_keys=NET_BENEFIT_SUPPRESSED_KEYS,
                             flag_key="suppressed", what="net-benefit curve")
    n_supp = sum(1 for r in rows if r["suppressed"])
    if n_supp:
        log.warning("net benefit: %d of %d rows SUPPRESSED on convergence grounds; the rows "
                    "keep their counts and their reason and lose every estimate, "
                    "km_risk_above included, so no surviving column reconstructs the "
                    "withheld net benefit", n_supp, len(rows))
    return pd.DataFrame(rows, columns=NET_BENEFIT_COLUMNS)


# =========================================================================== #
# 8. OUTPUT WRITERS - outputs/ IS AGGREGATE ONLY (protocol section 28)         #
# =========================================================================== #
def is_path_value(value: object) -> bool:
    """Does this configured value name a file this module writes?

    A path is a string with a file extension: ``"outputs/tables/val_metrics.csv"`` yes,
    ``"bh"`` (``fdr_method``) no, and an int, a list or a dict never. Deliberately a rule
    about the VALUE and not about the key spelling, so the ``_csv`` / ``_json`` naming
    convention stays a convention rather than becoming a second, silent schema.
    """
    return isinstance(value, str) and bool(Path(value).suffix)


def split_path(cfg: Config, key: str, split: str) -> Path:
    """The ``model_eval`` output path named by ``key``, resolved for ``split``.

    THE one implementation of the sealed-read path rewrite. ``main`` derives every file it
    writes from this function, and ``src/manuscript_figures.py`` and
    ``src/make_manuscript.py`` import it to derive every file they read, so a writer and a
    reader cannot disagree about a filename.

    Sealed-read outputs are written BESIDE the validation ones, never over them: the
    configured ``val_*`` basename has its first ``val_`` rewritten to ``test_`` and nothing
    else about the path changes. The rewrite is a rewrite rather than a parallel set of
    ``test_*`` config keys on purpose. Duplicating the keys would give five paths (six once
    decision-curve analysis lands) two sources of truth, and a reader could then resolve,
    with no error, a file the writer never wrote.

    ``str.replace(..., 1)`` is bounded to ONE substitution and is applied to ``p.name``
    only, so a directory component called ``val_...`` and a second ``val_`` later in the
    basename are both left exactly as configured. That bound is the behaviour the writer has
    always had and it is reproduced here rather than reasoned about again.

    ``key`` must name a FILE, not a setting. ``model_eval`` mixes the six output paths with
    a dozen numeric and structural settings, several of them one token away from a path key
    (``net_benefit`` beside ``net_benefit_csv``), and ``Config.path`` will happily
    ``str()`` a dict or an int into a filename: ``split_path(cfg, "net_benefit", "test")``
    used to return ``<repo>/{'threshold_min_pct': 1, ...}``. Since this function exists so
    that a reader and a writer cannot disagree about a filename, a key whose value is not a
    path is rejected as loudly as an unknown split rather than resolved into nonsense. The
    test is on the VALUE, not on the key spelling, so a path declared under any key name
    still resolves.
    """
    me = cfg["model_eval"]
    assert split in (VAL_SPLIT, SEALED_SPLIT), (
        f"split must be {VAL_SPLIT!r} or {SEALED_SPLIT!r}, not {split!r}; a mistyped split "
        f"must fail here rather than silently resolve a validation path for a sealed render")
    assert key in me, (
        f"model_eval has no path key {key!r}; it declares {sorted(me)}")
    value = me[key]
    assert is_path_value(value), (
        f"model_eval.{key} is not a path; it is a {type(value).__name__} setting "
        f"({value!r}). split_path resolves the output FILENAME a writer and a reader must "
        f"agree on, so a settings key must fail here rather than be stringified into one. "
        f"The path keys are {sorted(k for k, v in me.items() if is_path_value(v))}")
    p = cfg.path(value)
    sealed = (split == SEALED_SPLIT)
    return p if not sealed else p.with_name(p.name.replace("val_", "test_", 1))


def metrics_columns(horizons: list[int]) -> list[str]:
    """The pinned ``val_metrics.csv`` schema. ``src/make_manuscript.py`` reads these names
    and ``src/manuscript_figures.py`` requires the ``arm`` column and ``auc_{h}[_lo|_hi]``."""
    cols = ["arm", "label", "n_patients", "n_events", "val_nll",
            "harrell_c", "harrell_c_lo", "harrell_c_hi", "uno_c", "uno_c_lo", "uno_c_hi"]
    for h in horizons:
        cols += [f"auc_{h}", f"auc_{h}_lo", f"auc_{h}_hi", f"slope_{h}", f"citl_{h}",
                 f"brier_{h}"]
    return cols


def build_metrics(arms: dict[str, ArmScores], engine: BootstrapEngine, horizons: list[int],
                  order: list[str], log: logging.Logger) -> pd.DataFrame:
    """One row per scored arm, in the declared ladder order. Absent arms are simply absent."""
    rows = []
    for arm in order:
        sc = arms.get(arm)
        if sc is None:
            continue
        pt = engine.point(sc, sc.present)
        bt = engine.boot(sc, sc.present)
        c_lo, c_hi = percentile_ci(bt["harrell_c"])
        u_lo, u_hi = percentile_ci(bt["uno_c"])
        r: dict[str, object] = {"arm": arm, "label": sc.label,
                                "n_patients": int(pt["n_patients"]),
                                "n_events": int(pt["n_events"]),
                                "val_nll": float(sc.val_nll),
                                "harrell_c": pt["harrell_c"], "harrell_c_lo": c_lo,
                                "harrell_c_hi": c_hi, "uno_c": pt["uno_c"],
                                "uno_c_lo": u_lo, "uno_c_hi": u_hi}
        for h in horizons:
            a_lo, a_hi = percentile_ci(bt[f"auc@{h}"])
            r[f"auc_{h}"] = pt[f"auc@{h}"]; r[f"auc_{h}_lo"] = a_lo; r[f"auc_{h}_hi"] = a_hi
            r[f"slope_{h}"] = pt[f"slope@{h}"]
            r[f"citl_{h}"] = pt[f"citl@{h}"]
            r[f"brier_{h}"] = pt[f"brier@{h}"]
        rows.append(r)
        log.info("%-20s n=%-4d ev=%-3d  C %s %s | Uno %s | AUROC@%dd %s %s",
                 arm, r["n_patients"], r["n_events"], _f(r["harrell_c"]),
                 _ci(c_lo, c_hi), _f(r["uno_c"]), horizons[-1], _f(r[f"auc_{horizons[-1]}"]),
                 _ci(r[f"auc_{horizons[-1]}_lo"], r[f"auc_{horizons[-1]}_hi"]))
    return pd.DataFrame(rows, columns=metrics_columns(horizons))


def assert_aggregate_only(df: pd.DataFrame, forbidden_values, what: str) -> None:
    """No identifier may reach ``outputs/`` - not as a column name, not as a cell value."""
    bad_cols = [c for c in df.columns
                if any(tok in str(c).lower() for tok in FORBIDDEN_OUTPUT_COLUMN_TOKENS)]
    assert not bad_cols, (
        f"{what}: column(s) {bad_cols} would carry an identifier into outputs/; that directory "
        f"is aggregate only and per-patient arrays stay in derived-data/ (protocol section 28)")
    forbidden = set(str(v) for v in forbidden_values)
    for col in df.columns:
        if df[col].dtype != object:
            continue
        hits = sorted(set(df[col].dropna().astype(str)) & forbidden)
        if hits:
            raise AssertionError(
                f"{what}: column {col!r} contains {len(hits)} patient identifier(s); outputs/ is "
                f"aggregate only (protocol section 28)")


def round_floats(df: pd.DataFrame, nd: int = ROUND_DECIMALS) -> pd.DataFrame:
    """Fixed rounding on every float column, so two runs write byte-identical CSVs."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(nd)
    return out


def write_table(path: Path, df: pd.DataFrame, columns: list[str], forbidden_values,
                what: str) -> None:
    """Write one aggregate CSV: pinned column order, fixed rounding, no timestamp."""
    assert list(df.columns) == list(columns), (
        f"{what}: column order drifted from the pinned schema\n  got      {list(df.columns)}\n"
        f"  expected {list(columns)}")
    assert_aggregate_only(df, forbidden_values, what)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    round_floats(df).to_csv(path, index=False)


def build_results_json(cfg: Config, roster: Roster, arms: dict[str, ArmScores],
                       missing: list[str], horizons: list[int], n_boot: int,
                       seed: int, subgroup_arm: str | None, train_arms: dict,
                       split: str = VAL_SPLIT) -> dict:
    """The run header. The only output allowed a timestamp - the CSVs carry none.

    ``split`` names the split that was actually evaluated and decides two things that used
    to be hard-coded to validation and were therefore false in ``test_results.json``:

    * ``test_split`` is :data:`TEST_SPLIT_STATEMENT` looked up by split, so the sealed run
      no longer announces "SEALED, never loaded" about a read it had just performed;
    * the cohort counts are keyed ``n_{split}`` / ``n_{split}_events``, so 741 test patients
      are reported as ``n_test`` rather than labelled ``n_val``.

    On ``val`` both are byte-for-byte what they always were, so the validation artefacts and
    every consumer reading ``n_val`` from them keep working unchanged.
    """
    assert split in TEST_SPLIT_STATEMENT, (
        f"no run-header statement is defined for split {split!r}; known: "
        f"{sorted(TEST_SPLIT_STATEMENT)}")
    me = cfg["model_eval"]
    recal: dict[str, dict] = {}
    for arm, sc in arms.items():
        if sc.recalibration is None:
            recal[arm] = {"fitted_on": "not applicable",
                          "per_horizon": False,
                          "note": "frozen penalized Cox comparator replayed from JSON; the "
                                  "published risks are used unchanged"}
        else:
            recal[arm] = {"fitted_on": "val", "per_horizon": True,
                          "form": (train_arms.get("recalibration") or {}).get("form", ""),
                          "frozen_by": "src/train_model.py",
                          "params": sc.recalibration}
    return {
        "module": MODULE,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_split": TEST_SPLIT_STATEMENT[split],
        "cohort": {f"n_{split}": len(roster),
                   f"n_{split}_events": int(roster.event.sum())},
        "horizons_days": list(horizons),
        "bootstrap": {"n": int(n_boot), "seed": int(seed),
                      "note": "one patient-level resample per replicate, shared across arms"},
        "recalibration": recal,
        # ---- additive keys; every consumer above reads only the seven declared ones ----
        "arms_scored": sorted(arms),
        "arms_not_scored": sorted(missing),
        "primary_contrast": dict(me["primary_contrast"]),
        "comparison_families": {k: [list(p) for p in v]
                                for k, v in me["comparison_families"].items()},
        "fdr": {"method": str(me.get("fdr_method", "bh")),
                "applied": "within each declared family separately, never pooled"},
        "censoring": {"source": "m0_clinical_model.json censoring_km_train",
                      "note": "shared by every arm so the IPCW weights are identical"},
        "subgroups": {"arm": subgroup_arm, "metric": PRIMARY_METRIC,
                      "horizon_days": int(me["primary_contrast"]["horizon_days"]),
                      "suppress_below_events": int(me["suppress_below_events"])},
    }


# =========================================================================== #
# 9. REPORT - markdown into the run log, using the model_clinical formatters   #
# =========================================================================== #
REPORT_HEADING = {
    VAL_SPLIT: "Validation evaluation ({n} shared bootstrap replicates, test split sealed)",
    SEALED_SPLIT: ("Test evaluation ({n} shared bootstrap replicates, THE single permitted "
                   "sealed read)"),
}


def build_report(metrics: pd.DataFrame, comparisons: pd.DataFrame, subgroups: pd.DataFrame,
                 horizons: list[int], missing: list[str], n_boot: int, *,
                 split: str = VAL_SPLIT) -> list[str]:
    """A human-readable summary. Emitted into run.log, not to a new tracked file.

    ``split`` picks the heading from :data:`REPORT_HEADING`. It defaults to validation, so
    every existing caller and test is unaffected, and ``main`` passes the split it actually
    evaluated: a sealed run used to announce "test split sealed" into ``outputs/logs/run.log``
    on the very line summarising the test-split numbers, which is the same falsehood this
    module already stopped writing into ``test_results.json``. The log is not an artefact a
    reader cites, but a module that contradicts itself teaches a reader to distrust the
    statements that do matter.
    """
    L: list[str] = []
    A = L.append
    h = horizons[-1]
    assert split in REPORT_HEADING, (
        f"no report heading is defined for split {split!r}; known: {sorted(REPORT_HEADING)}")
    A("## " + REPORT_HEADING[split].format(n=n_boot))
    A("")
    A(_md_table(["arm", "n", "events", "Harrell C (95% CI)", "Uno C",
                 f"IPCW AUROC @{h} d (95% CI)", f"slope @{h} d"],
                [[r["arm"], int(r["n_patients"]), int(r["n_events"]),
                  f"{_f(r['harrell_c'])} {_ci(r['harrell_c_lo'], r['harrell_c_hi'])}",
                  _f(r["uno_c"]),
                  f"{_f(r[f'auc_{h}'])} {_ci(r[f'auc_{h}_lo'], r[f'auc_{h}_hi'])}",
                  _f(r[f"slope_{h}"])] for _, r in metrics.iterrows()]))
    A("")
    if missing:
        A(f"Not scored (no val_hazards npz yet): {', '.join(missing)}.")
        A("")
    if len(comparisons):
        A("### Paired contrasts (IPCW AUROC, one shared patient-level draw)")
        A("")
        A(_md_table(["family", "contrast", "n paired", "difference (95% CI)", "p", "BH p"],
                    [[r["family"], f"{r['model']} vs {r['reference']}", int(r["n_paired"]),
                      f"{_f(r['difference'])} {_ci(r['ci_lo'], r['ci_hi'])}",
                      _f(r["p_two_sided"], 4), _f(r["p_adjusted"], 4)]
                     for _, r in comparisons.iterrows()]))
        A("")
    n_supp = int(subgroups["suppressed"].sum()) if len(subgroups) else 0
    A(f"### Subgroups: {n_supp} of {len(subgroups)} levels suppressed (protocol section 21)")
    return L


# =========================================================================== #
# 10. ENTRY POINT                                                             #
# =========================================================================== #
def main(argv=None) -> int:                                  # noqa: C901 - one linear script
    ap = argparse.ArgumentParser(
        description="Validation-only evaluation of the model ladder (the test split is sealed).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--train-arms", default=None,
                    help="override the hand-over index path (default: "
                         "<cohort_dir>/train_arms.json). npz files resolve next to it.")
    ap.add_argument("--bootstrap-n", type=int, default=None,
                    help="override model_eval.bootstrap_n for a fast smoke run; the reported "
                         "value is whatever was actually used")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of the trained ladder to score")
    ap.add_argument("--split", default=VAL_SPLIT, choices=[VAL_SPLIT, SEALED_SPLIT],
                    help="which split to evaluate. 'test' is THE SEALED READ: it requires "
                         "test_hazards_*.npz from src/score_test.py, writes test_*.csv, and "
                         "is unbiased because the test split took no part in early stopping "
                         "or recalibration.")
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    me = cfg["model_eval"]
    sealed = (args.split == SEALED_SPLIT)
    if sealed:
        log.warning("*** EVALUATING THE SEALED %s SPLIT. These estimates took no part in "
                    "early stopping or recalibration and are the only unbiased numbers in "
                    "this study. ***", SEALED_SPLIT.upper())
    else:
        assert_forbid_test_split_is_on(cfg)
    horizons = horizons_from_config(cfg)
    n_boot = int(args.bootstrap_n if args.bootstrap_n is not None else me["bootstrap_n"])
    seed = int(me["bootstrap_seed"])
    assert int(me["bootstrap_n"]) == PROTOCOL_BOOTSTRAP_N, (
        f"model_eval.bootstrap_n is {me['bootstrap_n']}, but protocol Table 7 pre-specifies "
        f"{PROTOCOL_BOOTSTRAP_N} replicates for the primary paired patient-level bootstrap. It "
        f"must not inherit model_clinical.bootstrap_n "
        f"({cfg['model_clinical']['bootstrap_n']}), which is a development-report value. Pass "
        f"--bootstrap-n for a fast smoke run instead of editing the pre-specified value.")
    log.info("START %s | split %s | horizons %s d | %d bootstrap replicates (seed %d) | "
             "test split %s", MODULE, args.split, horizons, n_boot, seed,
             "READ (sealed)" if sealed else SEALED_SPLIT.upper() + ", NEVER LOADED")

    # ---- 1. frozen contracts, roster, and the hand-over index ---------------
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    contracts = FrozenContracts(coh)
    train_arms_path = Path(args.train_arms) if args.train_arms else (coh / "train_arms.json")
    train_arms = load_train_arms(train_arms_path)
    # Both anchors are MEASURED before the roster is built: the feature-table row count from
    # the Parquet, the patients-with-crops count from the shard sidecar. Passing
    # EXPECTED_DEV_ROWS here instead would compare the constant with itself.
    if not sealed:
        assert_development_anchors(measured_development_rows(contracts), train_arms, log,
                                   measured_with_crops=measured_patients_with_crops(cfg, log))
    roster, _ = load_roster(contracts, log, split=args.split)

    # ---- 2. the arms: frozen Cox comparators, then the trained ladder -------
    arms: dict[str, ArmScores] = {}
    for arm in COX_ARMS:
        arms[arm] = cox_arm_scores(arm, contracts, roster, horizons, log,
                                   split=args.split)

    declared = list(cfg["model_image"]["local"]["arms"].keys())
    wanted = [a.strip() for a in args.arms.split(",")] if args.arms else declared
    unknown = [a for a in wanted if a not in declared]
    assert not unknown, f"--arms names {unknown}, which are not in model_image.local.arms"
    subset_arms = set(me.get("subset_arms") or [])
    missing: list[str] = []
    for arm in wanted:
        summary = (train_arms["arms"] or {}).get(arm)
        if not summary:
            log.warning("arm %-20s SKIPPED: not present in %s, so it has not been trained yet",
                        arm, train_arms_path.name)
            missing.append(arm)
            continue
        sc = trained_arm_scores(arm, summary, train_arms_path.parent, roster, horizons,
                                log, split=args.split)
        if sc is None:
            missing.append(arm)
            continue
        if arm in subset_arms:
            assert sc.subset_arm or sc.n_patients < len(roster), (
                f"{arm} is declared in model_eval.subset_arms but scores every validation "
                f"patient; the KLG-eligible restriction was not applied")
        arms[arm] = sc
    log.info("scored %d arm(s): %s | skipped %d: %s", len(arms), ", ".join(sorted(arms)),
             len(missing), ", ".join(missing) if missing else "none")

    # ---- 3. ONE shared draw, then every arm and every contrast on it --------
    draw = bootstrap_draw(len(roster), n_boot, seed)
    log.info("one shared patient-level draw: %d replicates x %d patients, seed %d "
             "(protocol section 18; every arm and every contrast reuses it)",
             draw.shape[0], draw.shape[1], seed)
    engine = BootstrapEngine(roster, draw, horizons, log)

    order = list(COX_ARMS) + declared
    metrics = build_metrics(arms, engine, horizons, order, log)
    convergence = convergence_diagnostics(cfg, log)
    comparisons = build_comparisons(cfg, arms, engine, log, convergence=convergence,
                                    sealed=sealed)

    primary_arm = str(me["primary_contrast"]["model"])
    subgroup_arm = primary_arm if primary_arm in arms else ("m0" if "m0" in arms else None)
    if subgroup_arm != primary_arm:
        log.warning("subgroups are computed on %r because the primary arm %r is not scored yet",
                    subgroup_arm, primary_arm)
    subgroups = build_subgroups(cfg, roster, arms.get(subgroup_arm), engine,
                                int(me["primary_contrast"]["horizon_days"]), log)

    # ---- 3b. decision-curve analysis, on THE SAME DRAW ----------------------
    # A parallel engine, not more metrics inside BootstrapEngine: see NetBenefitEngine's
    # docstring for why that cache may not be reused. What IS reused is the draw itself, so
    # replicate b flags the same patients here as in every AUROC contrast above.
    nb_set = net_benefit_settings(cfg)
    assert nb_set["horizon_days"] in horizons, (
        f"model_eval.net_benefit.horizon_days is {nb_set['horizon_days']}, which is not among "
        f"the evaluated horizons {horizons}; ArmScores.risk carries a column per horizon and "
        f"the decision curve can only be drawn on one this module actually predicted")
    nb_engine = NetBenefitEngine(roster, draw, nb_set["thresholds"], nb_set["horizon_days"],
                                 nb_set["sparse_events_min"], log)
    assert nb_engine.draw is engine.draw, (
        "the decision curve must run on the SAME draw object as every other contrast, not on "
        "a copy: a second draw would silently unpair every difference in the figure")
    log.info("decision curve: %d thresholds %.2f-%.2f at %d d over %s (reference %s); %s",
             nb_set["thresholds"].size, nb_set["thresholds"][0], nb_set["thresholds"][-1],
             nb_set["horizon_days"], ", ".join(nb_set["arms"]), nb_set["reference"],
             NET_BENEFIT_MULTIPLICITY_NOTE)
    net_benefit = build_net_benefit(cfg, arms, nb_engine, log, split=args.split,
                                    convergence=convergence, sealed=sealed)

    # ---- 4. write. outputs/ is aggregate only; per-patient arrays stay put --
    forbidden = roster.pids

    def _out(key: str) -> Path:
        """Sealed-read outputs are written beside the validation ones, never over them.

        Delegates to :func:`split_path` and holds no rule of its own: the two downstream
        render modules import that function to find these same files, so the rewrite must
        exist exactly once.
        """
        return split_path(cfg, key, args.split)

    write_table(_out("metrics_csv"), metrics, metrics_columns(horizons), forbidden,
                "val_metrics.csv")
    write_table(_out("comparisons_csv"), comparisons, COMPARISON_COLUMNS, forbidden,
                "val_comparisons.csv")
    write_table(_out("subgroups_csv"), subgroups, SUBGROUP_COLUMNS, forbidden,
                "val_subgroups.csv")
    write_table(_out("convergence_csv"), convergence, CONVERGENCE_COLUMNS, forbidden,
                "val_convergence.csv")
    write_table(_out("net_benefit_csv"), net_benefit, NET_BENEFIT_COLUMNS, forbidden,
                "val_net_benefit.csv")
    results = build_results_json(cfg, roster, arms, missing, horizons, n_boot, seed,
                                 subgroup_arm, train_arms, split=args.split)
    results["convergence"] = {
        "min_train_nll_drop": float(me.get("min_train_nll_drop", 0.001)),
        "max_val_overfit_gap": float(me.get("max_val_overfit_gap", 0.10)),
        "flagged": {r["arm"]: {"status": r["status"], "train_nll_drop": r["train_nll_drop"],
                               "val_overfit_gap": r["val_overfit_gap"], "reason": r["reason"]}
                    for _, r in convergence.iterrows() if r["status"] != STATUS_OK},
        "note": ("contrasts involving a flagged arm carry a suppressed estimate and a NaN p "
                 "value, and are excluded from their family's Benjamini-Hochberg multiplicity"),
    }
    results["net_benefit"] = {
        "arms": list(nb_set["arms"]),
        "reference": nb_set["reference"],
        "horizon_days": int(nb_set["horizon_days"]),
        "threshold_min_pct": int(nb_set["threshold_min_pct"]),
        "threshold_max_pct": int(nb_set["threshold_max_pct"]),
        "plot_min_pct": int(nb_set["plot_min_pct"]),
        "plot_max_pct": int(nb_set["plot_max_pct"]),
        "sparse_events_min": int(nb_set["sparse_events_min"]),
        "estimator": ("Kaplan-Meier cumulative incidence WITHIN the flagged set at the "
                      "horizon, so the true- and false-positive terms come from one fit and "
                      "no censored patient is dropped or double-counted; treat-all is the "
                      "same code path with risk=None"),
        "sensitivity": ("net_benefit_ipcw is a point-estimate-only per-patient IPCW "
                        "sensitivity column using M0's frozen censoring_km_train; it is not "
                        "the primary estimator and carries no interval"),
        "denominators": ("each arm is estimated on its own patients and every paired "
                         "difference on the intersection, which the note column states; the "
                         "arms are deliberately not harmonised onto one population"),
        "multiplicity": NET_BENEFIT_MULTIPLICITY_NOTE,
        "n_rows": int(len(net_benefit)),
        "n_suppressed": int(net_benefit["suppressed"].sum()) if len(net_benefit) else 0,
    }
    res_path = _out("results_json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(results, indent=2, default=str))

    for line in build_report(metrics, comparisons, subgroups, horizons, missing, n_boot,
                             split=args.split):
        log.info("%s", line)
    log.info("DONE - wrote %s (%d rows), %s (%d rows), %s (%d rows), %s (%d rows) and %s. %s",
             _out("metrics_csv").name, len(metrics),
             _out("comparisons_csv").name, len(comparisons),
             _out("subgroups_csv").name, len(subgroups),
             _out("net_benefit_csv").name, len(net_benefit), res_path.name,
             (f"The {SEALED_SPLIT} split WAS read: this is the single permitted sealed read "
              f"and these estimates are unbiased." if sealed
              else f"The {SEALED_SPLIT} split was never loaded."))
    return 0


# =========================================================================== #
# 10. THE V6 REVISION ENTRY POINT - a SEPARATE main, on purpose                #
#                                                                              #
# main() above writes the published metrics, comparisons, subgroups,           #
# convergence and net-benefit tables. A6 and A4 must never be able to touch    #
# any of those, so they do not share its argument parser, its writers or its   #
# code path: main_v6 writes only the basenames declared above and asserts it.  #
#                                                                              #
#     python -m src.eval_models v6 --config config/feasibility.yaml \          #
#                                  --out-dir outputs/tables                    #
# =========================================================================== #
V6_SUBCOMMAND = "v6"


def run_v6(cfg: Config, log: logging.Logger, out_dir: Path,
           n_boot: int | None = None) -> dict[str, pd.DataFrame]:
    """Build and write the A3 robustness tables and the A4 learning-curve tables.

    Reads the sealed split. That is not a new read: ``src/score_test.py`` performed the
    single permitted one and recorded it, and :func:`assert_sealed_read_is_recorded` makes
    this function refuse to run if that record is absent or the training contract has
    moved. The bootstrap is the protocol's replicate count on the protocol's seed, i.e.
    THE SAME DRAW the published tables used, so replicate b here is the same set of
    patients as replicate b there. Returns the frames so tests can assert on them.
    """
    contract = assert_sealed_read_is_recorded(cfg)
    me = cfg["model_eval"]
    horizons = horizons_from_config(cfg)
    horizon = int(me["primary_contrast"]["horizon_days"])
    assert horizon in horizons
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    contracts = FrozenContracts(coh)
    train_arms = load_train_arms(coh / "train_arms.json")
    roster, _ = load_roster(contracts, log, split=SEALED_SPLIT)

    arms: dict[str, ArmScores] = {}
    for arm in V6_ROBUSTNESS_ARMS:
        summary = (train_arms["arms"] or {}).get(arm)
        assert summary, f"{arm} is not in train_arms.json; it was never trained"
        sc = trained_arm_scores(arm, summary, coh, roster, horizons, log, split=SEALED_SPLIT)
        assert sc is not None, f"{SEALED_SPLIT}_hazards_{arm}.npz is missing"
        arms[arm] = sc

    n_boot = int(n_boot if n_boot is not None else me["bootstrap_n"])
    seed = int(me["bootstrap_seed"])
    draw = bootstrap_draw(len(roster), n_boot, seed)
    engine = BootstrapEngine(roster, draw, horizons, log)
    log.info("v6 context: %d test patients, %d events, contract %s, %d replicates (seed %d)",
             len(roster), int(roster.event.sum()), contract, n_boot, seed)

    curves, per_seed, per_arm = build_learning_curves(cfg, log)
    tables = {
        V6_ROBUSTNESS_BASENAMES["strata"]:
            (build_imaging_robustness(cfg, roster, arms, engine, horizon, log),
             ROBUSTNESS_COLUMNS),
        V6_ROBUSTNESS_BASENAMES["availability"]:
            (build_metadata_availability(cfg), METADATA_AVAILABILITY_COLUMNS),
        V6_LEARNING_CURVE_BASENAMES["curves"]: (curves, LEARNING_CURVE_COLUMNS),
        V6_LEARNING_CURVE_BASENAMES["per_seed"]: (per_seed, LEARNING_CURVE_SEED_COLUMNS),
        V6_LEARNING_CURVE_BASENAMES["per_arm"]: (per_arm, LEARNING_CURVE_ARM_COLUMNS),
    }
    allowed = set(V6_ROBUSTNESS_BASENAMES.values()) | set(V6_LEARNING_CURVE_BASENAMES.values())
    out_dir = Path(out_dir)
    out: dict[str, pd.DataFrame] = {}
    for name, (df, cols) in tables.items():
        assert name in allowed and name.startswith("v6_"), (
            f"{name} is not a declared v6 output; this entry point may not write over a "
            f"published table")
        write_table(out_dir / name, df, cols, roster.pids, name)
        log.info("wrote %s (%d rows)", out_dir / name, len(df))
        out[name] = df
    return out


def main_v6(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.eval_models v6",
        description="v6 Phase-2 analyses A3 (imaging robustness strata) and A4 (learning "
                    "curves). Writes only v6_* tables; the published tables are untouched.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--out-dir", default="outputs/tables")
    ap.add_argument("--bootstrap-n", type=int, default=None,
                    help="override model_eval.bootstrap_n for a fast smoke run; the reported "
                         "tables always use the protocol value unless this is set")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    log.warning("*** A3 and A4 are POST HOC on the already-read sealed test split "
                "(deviation D35). The acquisition-era strata carry a SECOND caveat: "
                "StudyDate_anon has a per-patient random shift and protocol section 17's "
                "written confirmation of cross-patient date comparability has never been "
                "obtained (D17). Both caveats are in the note column of every affected "
                "row. ***")
    run_v6(cfg, log, Path(args.out_dir), n_boot=args.bootstrap_n)
    return 0


if __name__ == "__main__":
    # A first positional token of "v6" selects the revision entry point. Anything else -
    # including no argument at all - runs main() exactly as before, so the published
    # rendering path is unchanged and cannot be reached by a v6 flag.
    if len(sys.argv) > 1 and sys.argv[1] == V6_SUBCOMMAND:
        raise SystemExit(main_v6(sys.argv[2:]))
    raise SystemExit(main())
