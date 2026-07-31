"""splits.py — LOCKED patient-level train / validation / test splits (protocol section 17).

70% / 10% / 20%, grouped by patient (final_cohort has one row per patient, so grouping
is automatic), stratified approximately by 5-year event status x sex x major race group.
Deterministic (config reproducibility.random_seed). The 20% test set is the LOCKED test
set: it must remain unread until the model, ensemble rule, thresholds, and analysis script
are frozen. This module only ASSIGNS splits; it never trains or reads outcomes for tuning.

Run:  python3 -m src.splits --config config/feasibility.yaml
Writes derived-data/cohort/patient_splits.parquet and outputs/tables/split_summary.csv
(aggregate only, no empi_anon).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

MODULE = "splits"
FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
RACE_MAJOR = {"African American or Black": "Black", "Caucasian or White": "White", "Asian": "Asian"}


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


def assign_splits(strata: pd.Series, seed: int) -> pd.Series:
    """Deterministic stratified 70/10/20 assignment. Within each stratum, shuffle the
    patients with a seeded RNG and slice by cumulative fraction so every stratum keeps
    ~the same train/val/test proportions (small strata handled by rounding)."""
    rng = np.random.default_rng(seed)
    out = pd.Series(index=strata.index, dtype=object)
    for key, idx in strata.groupby(strata).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * FRACTIONS["train"]))
        n_va = int(round(n * FRACTIONS["val"]))
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        out.iloc[[strata.index.get_loc(i) for i in idx[:n_tr]]] = "train"
        out.iloc[[strata.index.get_loc(i) for i in idx[n_tr:n_tr + n_va]]] = "val"
        out.iloc[[strata.index.get_loc(i) for i in idx[n_tr + n_va:]]] = "test"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    seed = int(cfg["reproducibility"]["random_seed"])

    fc = pd.read_parquet(coh / "final_cohort.parquet")[["empi_anon", "event_indicator", "index_side"]]
    demo = pd.read_parquet(cfg.parquet_path("demographics"))[["empi_anon", "sex", "race"]]
    df = fc.merge(demo, on="empi_anon", how="left")
    assert df["empi_anon"].is_unique, "final_cohort not one row per patient"

    df["race_major"] = df["race"].map(RACE_MAJOR).fillna("Other")
    df["stratum"] = (df["event_indicator"].astype(int).astype(str) + "|"
                     + df["sex"].fillna("NA") + "|" + df["race_major"])
    df = df.reset_index(drop=True)
    df["split"] = assign_splits(df["stratum"], seed)

    # ---- validation ----
    assert df["split"].isin(["train", "val", "test"]).all(), "unassigned patients"
    assert df.groupby("empi_anon")["split"].nunique().max() == 1, "a patient in >1 split"
    n = len(df)
    summary_rows = []
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        summary_rows.append(dict(split=s, n_patients=len(sub), pct=round(100 * len(sub) / n, 1),
                                 n_events=int(sub["event_indicator"].sum()),
                                 event_pct=round(100 * sub["event_indicator"].mean(), 2)))
    summ = pd.DataFrame(summary_rows)
    # test set within a reasonable band of 20%, event rate preserved (within ~3 pts)
    test_pct = summ.loc[summ.split == "test", "pct"].iloc[0]
    assert 17.0 <= test_pct <= 23.0, f"test split {test_pct}% off target 20%"
    overall_ev = 100 * df["event_indicator"].mean()
    assert (summ["event_pct"] - overall_ev).abs().max() <= 3.0, "event rate not preserved across splits"

    df[["empi_anon", "split", "stratum"]].to_parquet(coh / "patient_splits.parquet", index=False)
    summ.to_csv(cfg.path(cfg["paths"]["tables_dir"]) / "split_summary.csv", index=False)
    for _, r in summ.iterrows():
        log.info("split %-5s n=%d (%.1f%%) events=%d (%.2f%%)",
                 r["split"], r["n_patients"], r["pct"], r["n_events"], r["event_pct"])
    log.info("LOCKED splits written (seed=%d); overall event rate %.2f%%. Test set is locked.",
             seed, overall_ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
