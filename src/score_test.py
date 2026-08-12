"""THE SEALED READ. Score the frozen model ladder on the locked test split, exactly once.

This module exists so that reading the test set is a deliberate act in one auditable
place, rather than a flag on the trainer. ``src/train_model.py`` refuses the sealed split
everywhere by default. Two callers lift that refusal, and only for the sealed split:
this module, and ``src/eval_models.py`` when it renders that split
(``eval_models.py:504`` and ``:570``, both gated on the split actually being the sealed
one). No other caller passes ``allow_sealed=True``.

What it does, and equally what it must never do
-----------------------------------------------
It **loads** frozen artefacts and **applies** the two that produce a hazard:

* the per-seed checkpoints written by ``src/train_model.py`` under one training-contract
  hash, which is asserted to match the hand-over index;
* the ensemble rule (per-interval hazards averaged across the five pre-specified seeds).

It **records, and does not apply**, the horizon-specific recalibration **fitted on
validation** and frozen in ``train_arms.json``. The parameters are copied verbatim into
``test_scoring.json`` so the transform is pinned at the moment of the read, but the
``hazards`` array written below is the **raw** seed-averaged ensemble.
``src/eval_models.py`` applies the transform downstream, at render time
(``apply_recalibration`` at ``eval_models.py:646``), which is where every recalibrated
number in ``outputs/tables/test_metrics.csv`` comes from; the two frozen Cox arms are
published as fitted and are never recalibrated at all (``eval_models.py:580``). Anything
reading ``test_hazards_{arm}.npz`` directly is therefore reading pre-recalibration
hazards and must apply the transform itself.

It **never** trains, refits, re-tunes, re-selects an epoch, or fits a recalibration on
test rows. Protocol sections 12 and 17 permit exactly one scripted read once the model,
ensemble rule, thresholds and analysis script are frozen; a second read, or any change to
a model after a read, invalidates the estimate this produces.

Why the test estimate is the one that matters
---------------------------------------------
Every number reported from the validation split carries selection optimism: validation is
the early-stopping split AND the recalibration split, so the retained checkpoint is the
epoch that best fitted the very 54 events the metric is then computed on. The test split
took part in none of that. It is 741 patients and 106 events, roughly double the
validation event count, and it is the only estimate in this study free of that circularity.

Outputs
-------
``derived-data/cohort/test_hazards_{arm}.npz`` per arm, in the same schema
``src/train_model.py`` writes for validation, so ``src/eval_models.py`` reads them through
the same code path. Per-patient arrays stay in ``derived-data/``; nothing patient-level is
written under ``outputs/``.

Run::

    python -m src.score_test --shard-dir ~/mrkr-shards-test --confirm-sealed-read
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.train_model import (  # frozen primitives: imported, never reimplemented
    EDGES,
    N_INTERVALS,
    SEALED_SPLIT,
    FrozenContracts,
    PatientViewDataset,
    TrainSettings,
    apply_recalibration,
    average_hazard,
    build_clinical_design,
    discretize_survival,
    dt_nll_numpy,
    fit_clin_stats,
    load_seed_model,
    load_sidecar,
    materialize_split,
    predict_hazards,
    read_json_retrying,
    replay_cox,
    require_torch,
    resolve_device,
    seed_everything,
)

MODULE = "score_test"

# Frozen by the preprocessing run of 2026-07-29 (`--splits test --include-test`).
# 1,218 scheduled, 3 excluded by the protocol section-13 masking rule, 1,216 written.
EXPECTED_TEST_CROPS = {"test": 1216}
EXPECTED_TEST_PATIENTS = 740


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(MODULE)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in lg.handlers):
        fh = logging.FileHandler(log_path, mode="a")
        fh._mrkr = True                                    # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"))
        lg.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh._mrkr = True                                    # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s"))
        lg.addHandler(sh)
    return lg


def assert_frozen(train_arms: dict, settings: TrainSettings, log: logging.Logger) -> str:
    """The models must be frozen under ONE contract, and it must be the live one."""
    frozen = str(train_arms.get("training_contract_hash", ""))
    live = settings.contract_hash()
    assert frozen, "train_arms.json carries no training_contract_hash; refusing to score"
    assert frozen == live, (
        f"REFUSED: the frozen models were trained under contract {frozen} but the current "
        f"config resolves to {live}. Scoring the sealed split with a changed contract would "
        f"measure a model that was never trained. Restore the config that produced "
        f"{frozen} before reading the test split.")
    arms = train_arms.get("arms") or {}
    assert arms, "train_arms.json lists no arms"
    incomplete = [a for a, s in arms.items() if not s.get("complete")]
    assert not incomplete, f"arms not marked complete: {incomplete}"
    for a, s in arms.items():
        assert s.get("recalibration"), f"{a} carries no frozen recalibration"
    log.info("frozen contract %s | %d arm(s) | ensemble rule %r | recalibration %s",
             frozen, len(arms), train_arms.get("ensemble", {}).get("rule"),
             train_arms.get("recalibration", {}).get("fitted_on"))
    return frozen


def build_test_frames(contracts: FrozenContracts, log: logging.Logger):
    """Clinical designs on train+val+test, standardised on TRAIN rows only.

    The standardisation statistics come from TRAIN, exactly as at fit time. Recomputing
    them on test rows would leak the test distribution into the features.
    """
    frames, designs, stats = {}, {}, {}
    for design in ("m0", "m1"):
        frame, X = build_clinical_design(contracts, ["train", "val", SEALED_SPLIT],
                                         design=design, allow_sealed=True)
        spec = contracts.design_spec(design)
        st = fit_clin_stats(frame, X, spec["json"], spec["design_columns"])
        assert st["n_train"] > 0, "standardisation must be fitted on TRAIN rows"
        lp, _ = replay_cox(contracts, X, design)
        is_train = (frame["split"] == "train").to_numpy()
        assert abs(float(lp[is_train].mean())) < 1e-6, (
            f"{design} replay drifted from its frozen JSON (mean train linear predictor "
            f"{float(lp[is_train].mean()):.3e})")
        n_test = int((frame["split"] == SEALED_SPLIT).sum())
        frames[design], designs[design], stats[design] = frame, X, st
        log.info("%s design %s | %d test rows | standardisation on %d TRAIN rows",
                 design, tuple(X.shape), n_test, st["n_train"])
    return frames, designs, stats


def score_arm(arm: str, summary: dict, *, frames, designs, stats, labels, npy, index,
              settings: TrainSettings, device, amp: bool, cohort_dir: Path,
              log: logging.Logger) -> dict:
    """Apply one frozen arm to the test split and write its npz. Nothing is fitted."""
    design = str(summary["design"])
    spec_views = list(summary["views"])
    ds = PatientViewDataset(
        SEALED_SPLIT, frames[design], designs[design], train=False,
        npy_path=npy, index=index, clin_stats=stats[design],
        views=list(settings.views), views_allowed=spec_views,
        max_elems=int(labels.groupby("empi_anon").size().max()),
        design=design, out_size=settings.out_size, aug=None,
        border_px=settings.border_px, allow_sealed=True)

    seeds = [int(s) for s in summary["seeds"]]
    per_seed = []
    for seed in seeds:
        net, _ = load_seed_model(arm, seed, spec={"mode": summary["mode"],
                                                  "arch": summary["arch"],
                                                  "views": spec_views,
                                                  "design": design},
                                 n_clinical=ds.clin.shape[1], settings=settings,
                                 device=device, contract_hash=settings.contract_hash())
        per_seed.append(predict_hazards(net, ds, device=device, settings=settings, amp=amp))
        del net
    ens = average_hazard(per_seed)
    nll = dt_nll_numpy(ens, ds.at_risk, ds.target)[0]

    recal = dict(summary["recalibration"])
    log.info("  %-22s %d/%d test patients, %d events | test NLL %.4f | hazards written RAW "
             "(pre-recalibration); frozen validation recalibration recorded, not applied, "
             "at %s",
             arm, len(ds), EXPECTED_TEST_PATIENTS, int(ds.event.sum()), nll,
             ", ".join(sorted(recal)))

    np.savez(cohort_dir / f"test_hazards_{arm}.npz",
             hazards=ens.astype(np.float64),
             hazards_per_seed=np.stack(per_seed).astype(np.float64),
             seeds=np.asarray(seeds, dtype=np.int64),
             empi_anon=np.asarray(ds.pids, dtype=object).astype("U"),
             time=ds.time.astype(np.float64), event=ds.event.astype(np.int64),
             at_risk=ds.at_risk.astype(np.float64), target=ds.target.astype(np.float64),
             n_scored=ds.n_scored.astype(np.int64), edges=EDGES.astype(np.float64),
             arm=np.asarray(arm), mode=np.asarray(summary["mode"]),
             arch=np.asarray(summary["arch"]), design=np.asarray(design),
             views=np.asarray(spec_views, dtype=object).astype("U"),
             split=np.asarray(SEALED_SPLIT))
    return {"arm": arm, "n_patients": int(len(ds)), "n_events": int(ds.event.sum()),
            "test_nll": float(nll), "seeds": seeds, "recalibration": recal}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--shard-dir", required=True,
                    help="directory holding test-00000.tar and its labels.csv")
    ap.add_argument("--cache-dir", default=None, help="defaults to <shard-dir>/../mrkr-cache-test")
    ap.add_argument("--arms", default=None, help="comma-separated subset (default: all frozen)")
    ap.add_argument("--confirm-sealed-read", action="store_true",
                    help="REQUIRED. Acknowledges that this is the single permitted read of "
                         "the locked test split and that the models are frozen.")
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    if not args.confirm_sealed_read:
        log.error("REFUSED: --confirm-sealed-read is required. Protocol sections 12 and 17 "
                  "permit exactly ONE scripted read of the locked test split, after the "
                  "model, ensemble rule, thresholds and analysis script are frozen.")
        return 2

    log.warning("*** SEALED TEST READ. This is the ONE permitted read. Nothing is fitted "
                "here: frozen checkpoints and the frozen ensemble rule are applied as-is, "
                "and the validation-fitted recalibration is recorded but NOT applied — "
                "src/eval_models.py applies it at render time. Any later change to a model "
                "invalidates this estimate. ***")

    require_torch()
    cohort_dir = cfg.path(cfg["paths"]["cohort_dir"])
    train_arms = read_json_retrying(cohort_dir / "train_arms.json")

    # Reproduce the TRAINING contract rather than asking the operator to re-type the flags.
    # grad_accum_steps, micro_batch_size and num_workers are all inside the hash, so a
    # mismatch here would refuse a perfectly valid set of frozen checkpoints for a reason
    # that has nothing to do with the model. Derived from the frozen record, then asserted.
    tc = train_arms.get("training_contract") or {}
    settings = TrainSettings(cfg,
                             grad_accum_override=tc.get("grad_accum_steps"),
                             num_workers_override=tc.get("num_workers"))
    device, amp = resolve_device(settings.device_preference, settings.amp_devices)
    seed_everything(int(cfg["reproducibility"]["random_seed"]))
    frozen = assert_frozen(train_arms, settings, log)

    shard_dir = Path(args.shard_dir).expanduser()
    labels = load_sidecar(shard_dir, splits=[SEALED_SPLIT], expected=EXPECTED_TEST_CROPS,
                          expected_patients=EXPECTED_TEST_PATIENTS)
    cache_dir = (Path(args.cache_dir).expanduser() if args.cache_dir
                 else shard_dir.parent / "mrkr-cache-test")
    npy, index = materialize_split(SEALED_SPLIT, shard_dir=shard_dir, cache_dir=cache_dir,
                                   labels=labels, out_size=settings.out_size, log=log,
                                   allow_sealed=True)
    contracts = FrozenContracts(cohort_dir)
    frames, designs, stats = build_test_frames(contracts, log)

    arms = train_arms["arms"]
    wanted = [a.strip() for a in args.arms.split(",")] if args.arms else list(arms)
    unknown = [a for a in wanted if a not in arms]
    assert not unknown, f"--arms names {unknown}, which are not frozen in train_arms.json"

    log.info("scoring %d frozen arm(s) on the sealed split | device %s | AMP %s",
             len(wanted), device, amp)
    results = [score_arm(a, arms[a], frames=frames, designs=designs, stats=stats,
                         labels=labels, npy=npy, index=index, settings=settings,
                         device=device, amp=amp, cohort_dir=cohort_dir, log=log)
               for a in wanted]

    doc = {
        "module": MODULE,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sealed_read": "PERFORMED. This is the single permitted read of the locked test split.",
        "training_contract_hash": frozen,
        "cohort": {"n_test_patients_with_crops": EXPECTED_TEST_PATIENTS,
                   "n_test_crops": int(EXPECTED_TEST_CROPS[SEALED_SPLIT])},
        "grid": {"n_intervals": N_INTERVALS, "edges": EDGES.tolist()},
        "ensemble": train_arms.get("ensemble"),
        "recalibration": {"fitted_on": "validation", "refitted_on_test": False},
        "arms": {r["arm"]: r for r in results},
    }
    (cohort_dir / "test_scoring.json").write_text(json.dumps(doc, indent=2, default=str))

    L = []
    A = L.append
    A("")
    A("=" * 88)
    A(f"{MODULE}: {len(results)} frozen arm(s) scored on the SEALED test split".center(88))
    A("=" * 88)
    for r in results:
        A(f"  {r['arm']:22s} n {r['n_patients']:4d}  events {r['n_events']:4d}  "
          f"test NLL {r['test_nll']:.6f}")
    A(f"  hazards            {cohort_dir}/test_hazards_*.npz")
    A(f"  run record         {cohort_dir / 'test_scoring.json'}")
    A("  nothing was fitted on test rows")
    A("=" * 88)
    for line in L:
        log.info("%s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
