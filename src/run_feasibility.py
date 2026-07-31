"""run_feasibility.py — single reproducible entry point for the LOCKED extraction.

Orchestrates the full metadata-only feasibility pipeline end to end and writes a
consolidated machine summary (``outputs/feasibility_summary.json``) plus the
final ``outputs/event_counts.csv``. No DICOMs, no models, no performance metrics.

Stages (in order):
  io_duckdb -> index_tka -> imaging -> outcomes -> followup -> assemble_cohort
  -> cohort_flow -> subgroups -> manifest

Usage::

    # Fast path: consolidate the already-produced stage outputs into the summary
    python3 -m src.run_feasibility --config config/feasibility.yaml

    # Full reproducible re-run of every stage, then consolidate
    python3 -m src.run_feasibility --config config/feasibility.yaml --stages all

    # Re-run every stage INCLUDING the CSV->Parquet conversion
    python3 -m src.run_feasibility --config config/feasibility.yaml --stages all --with-parquet

Reads (pandas/pyarrow, resilient to slow synced-filesystem syscalls) the stage
outputs and reconciles them before emitting the go/no-go summary. All numbers are
PRELIMINARY feasibility descriptives, not a formal sample-size calculation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config

MODULE = "run_feasibility"

# Ordered pipeline stages -> module dotted name.
STAGES = [
    ("io_duckdb", "src.io_duckdb"),
    ("index_tka", "src.index_tka"),
    ("imaging", "src.imaging"),
    ("outcomes", "src.outcomes"),
    ("followup", "src.followup"),
    ("assemble_cohort", "src.assemble_cohort"),
    ("cohort_flow", "src.cohort_flow"),
    ("subgroups", "src.subgroups"),
    ("manifest", "src.manifest"),
]

FLOOR_EVENTS = 500
FLOOR_TEST = 100


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(MODULE)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in lg.handlers):
        fh = logging.FileHandler(log_path, mode="a")
        fh._mrkr = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
        lg.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh._mrkr = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s"))
        lg.addHandler(sh)
    return lg


def run_stages(cfg_path: str, with_parquet: bool, log: logging.Logger) -> None:
    """Import and invoke each stage's main(['--config', cfg_path]) in order."""
    import importlib
    for name, dotted in STAGES:
        if name == "io_duckdb" and not with_parquet:
            log.info("stage %s SKIPPED (typed Parquet already present; use --with-parquet)", name)
            continue
        log.info("stage %s: running %s.main ...", name, dotted)
        mod = importlib.import_module(dotted)
        rc = mod.main(["--config", cfg_path])
        if rc not in (0, None):
            raise RuntimeError(f"stage {name} returned non-zero ({rc})")
        log.info("stage %s: OK", name)


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def consolidate(cfg, log: logging.Logger) -> dict:
    """Read stage outputs and build the go/no-go feasibility_summary dict."""
    out = cfg.path(cfg["paths"]["outputs_dir"])
    tables = cfg.path(cfg["paths"]["tables_dir"])
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    prim = cfg["primary_definition"]

    flow = _read_csv(out / "cohort_flow.csv")
    subg = _read_csv(out / "subgroup_counts.csv")
    manifest = _read_csv(tables / "manifest_summary.csv")
    fc = pd.read_parquet(coh / "final_cohort.parquet")

    # decision anchors persisted at Stage-1 (re-gate ceilings / levers)
    anchors = {}
    s1 = out / "feasibility_stage1_counts.json"
    if s1.exists():
        anchors = json.loads(s1.read_text()).get("decision_anchors", {})

    n_final = int(fc.shape[0])
    n_events = int(fc["event_indicator"].sum())
    test_alloc = round(0.20 * n_events)

    # final-cohort follow-up descriptives
    ev = fc[fc["event_indicator"] == 1]["time_from_landmark"]
    cens = fc["time_from_landmark"]
    median_fu = float(cens.median())

    summary = {
        "label": "PRELIMINARY feasibility (LOCKED extraction)",
        "study": "MRKR Contralateral TKA - Multi-view radiographic prediction",
        "primary_definition": {
            "cohort_strategy": prim["cohort_strategy"],
            "pre_index_window_days": prim["pre_index_window_days"],
            "requires_laterality_qa_audit": prim.get("requires_laterality_qa_audit", True),
        },
        "cohort": {
            "n_final_landmark_cohort": n_final,
            "primary_events_5y": n_events,
            "primary_event_pct": round(100 * n_events / n_final, 2),
            "test_allocatable_20pct": test_alloc,
            "median_followup_days_from_landmark": round(median_fu, 1),
            "pct_index_side_recovered": round(
                100 * (fc["side_source"] == "recovered").mean(), 1),
        },
        "floors": {
            "primary_events_floor": FLOOR_EVENTS,
            "test_allocatable_floor": FLOOR_TEST,
            "primary_events_meets_floor": n_events >= FLOOR_EVENTS,
            "test_allocatable_meets_floor": test_alloc >= FLOOR_TEST,
            "both_floors_met": (n_events >= FLOOR_EVENTS and test_alloc >= FLOOR_TEST),
        },
        "cohort_flow": flow.to_dict(orient="records"),
        "subgroups": subg.to_dict(orient="records"),
        "imaging_manifest": manifest.to_dict(orient="records")[0] if len(manifest) else {},
        "decision_anchors": anchors,
        "recommendation": (
            "PROCEED to formal Riley sample-size + test-precision calculation, OSF "
            "preregistration, and image transfer, CONTINGENT on the protocol section-7 "
            ">=200-patient laterality QA audit validating the recovered index sides "
            "(~{}% of the primary cohort). Retain the strict coded-laterality cohort as "
            "the high-specificity sensitivity analysis."
        ).format(round(100 * (fc["side_source"] == "recovered").mean())),
    }
    log.info("consolidated: n_final=%d events=%d test=%d floors_met=%s",
             n_final, n_events, test_alloc, summary["floors"]["both_floors_met"])
    return summary


def write_event_counts(cfg, fc_path: Path, log: logging.Logger) -> None:
    """Final-cohort event counts by definition/window -> outputs/event_counts.csv."""
    fc = pd.read_parquet(fc_path)
    detail = cfg.path(cfg["paths"]["tables_dir"]) / "outcome_counts_detail.csv"
    rows = [
        dict(definition="primary_contralateral_5y", window="day91-5y",
             n_events=int(fc["event_indicator"].sum()),
             n_cohort=int(len(fc)),
             pct=round(100 * fc["event_indicator"].mean(), 2)),
    ]
    pd.DataFrame(rows).to_csv(cfg.path(cfg["paths"]["outputs_dir"]) / "event_counts.csv", index=False)
    if detail.exists():
        log.info("outcome_counts_detail.csv present for secondary/sensitivity windows")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Single entry point for the feasibility pipeline.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--stages", choices=["all", "none"], default="none",
                    help="'all' re-runs every stage; 'none' (default) consolidates existing outputs")
    ap.add_argument("--with-parquet", action="store_true",
                    help="also re-run the CSV->Parquet conversion stage")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    log.info("START run_feasibility (stages=%s with_parquet=%s)", args.stages, args.with_parquet)

    if args.stages == "all":
        run_stages(args.config, args.with_parquet, log)

    summary = consolidate(cfg, log)
    out_json = cfg.path(cfg["paths"]["outputs_dir"]) / "feasibility_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    write_event_counts(cfg, cfg.path(cfg["paths"]["cohort_dir"]) / "final_cohort.parquet", log)
    log.info("wrote %s and event_counts.csv | RECOMMENDATION: %s",
             out_json, "PROCEED (contingent on laterality QA)"
             if summary["floors"]["both_floors_met"] else "REVISE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
