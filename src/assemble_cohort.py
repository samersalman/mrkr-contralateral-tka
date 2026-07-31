"""assemble_cohort.py — intersect the index / imaging / outcome / follow-up gates
into the LOCKED final landmark cohort, and reconcile every (strategy x window)
cell against the verified re-gate numbers.

Gate order (protocol; sequential): index_final already encodes valid index +
age>=40 + no prior contralateral arthroplasty + no infection. This module adds:
  S8  eligible pre-index contralateral study in the window   (from candidate_studies)
  S9  NOT has_contra_27447_day_0_90                            (from outcomes)
  S10 observed_through_90                                      (from followup)
The PRIMARY cohort is recovery_any / 730-day (config primary_definition); the
selected study details come from selected_studies. Sensitivity windows/strategies
are derived from candidate_studies presence flags + index_final.side_source.

Run:  python3 -m src.assemble_cohort --config config/feasibility.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.config import load_config
from src.followup import resolve_followup

MODULE = "assemble_cohort"

# Re-gate regression anchors (src/regate.py -> outputs/tables/regate_grid.csv).
ANCHORS = {
    ("strict", 365): (1664, 357),
    ("strict", 730): (1881, 394),
    ("strict", 1095): (1967, 406),
    ("recovery_any", 365): (3372, 489),
    ("recovery_any", 730): (3709, 533),      # <-- LOCKED PRIMARY
    ("recovery_any", 1095): (3842, 547),
    ("recovery_confirmed", 730): (3541, 517),
    ("recovery_confirmed", 1095): (3670, 531),
}


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    con = duckdb.connect()

    def load(name):
        return con.execute(f"SELECT * FROM read_parquet('{coh / name}')").df()

    idx = load("index_final.parquet")                 # 6381
    outc = load("outcomes.parquet")                   # 6381
    fu = load("followup.parquet")                     # 6381
    cand = load("candidate_studies.parquet")          # eligible candidates @<=1095
    sel = load("selected_studies.parquet")            # 3981 primary-selected studies
    log.info("loaded index=%d outcomes=%d followup=%d candidates=%d selected=%d",
             len(idx), len(outc), len(fu), len(cand), len(sel))

    # ---- per-window eligibility sets (S8) from candidate_studies -------------
    def elig(window):
        if window == 365:
            m = cand["in_window_365"]
        elif window == 730:
            m = cand["in_window_730"]
        else:  # 1095 = all candidate rows (built at the widest window)
            m = pd.Series(True, index=cand.index)
        return set(cand.loc[m, "empi_anon"])

    elig_sets = {w: elig(w) for w in (365, 730, 1095)}
    # S9 keep-set and S10 keep-set
    keep_s9 = set(outc.loc[~outc["has_contra_27447_day_0_90"].astype(bool), "empi_anon"])
    keep_s10 = set(fu.loc[fu["observed_through_90"].astype(bool), "empi_anon"])
    ev = dict(zip(outc["empi_anon"], outc["primary_event"].astype(bool)))

    def strat_mask(strategy):
        if strategy == "strict":
            return idx["side_source"] == "coded"
        if strategy == "recovery_confirmed":
            return (idx["side_source"] == "coded") | (idx["n_concordant_signals"] >= 2)
        return pd.Series(True, index=idx.index)   # recovery_any

    # ---- grid reconciliation -------------------------------------------------
    rows, ok = [], True
    for (strategy, window), (exp_n, exp_e) in ANCHORS.items():
        pool = set(idx.loc[strat_mask(strategy), "empi_anon"])
        final = pool & elig_sets[window] & keep_s9 & keep_s10
        n = len(final)
        e = sum(1 for p in final if ev.get(p, False))
        match = (n == exp_n and e == exp_e)
        ok = ok and match
        (log.info if match else log.error)(
            "CELL %-18s w=%-4d n_final=%d (exp %d) events=%d (exp %d) %s",
            strategy, window, n, exp_n, e, exp_e, "OK" if match else "MISMATCH")
        rows.append(dict(strategy=strategy, window_days=window, n_final=n,
                         primary_events=e, exp_n=exp_n, exp_events=exp_e, match=match))
    pd.DataFrame(rows).to_csv(
        cfg.path(cfg["paths"]["tables_dir"]) / "cohort_assembly_summary.csv", index=False)

    # ---- build & persist the LOCKED PRIMARY final cohort (recovery_any/730) --
    prim = cfg["primary_definition"]
    W = prim["pre_index_window_days"][1]
    strategy = prim["cohort_strategy"]
    pool = set(idx.loc[strat_mask(strategy), "empi_anon"])
    final_ids = pool & elig_sets[W] & keep_s9 & keep_s10

    fc = idx[idx["empi_anon"].isin(final_ids)].merge(
        sel.drop(columns=["side_source"], errors="ignore"), on="empi_anon", how="left").merge(
        outc[["empi_anon", "primary_event", "event_date", "days_index_to_event"]],
        on="empi_anon", how="left").merge(
        fu[["empi_anon", "last_observed", "landmark_date", "censor_date_if_no_event",
            "complete_5y"]], on="empi_anon", how="left")

    # resolve landmark-anchored event/censor time via the tested followup helper
    res = fc.apply(lambda r: resolve_followup(
        r["landmark_date"], r["last_observed"],
        r["event_date"] if r["primary_event"] else None), axis=1, result_type="expand")
    fc["event_indicator"] = res[0].astype(int)
    fc["time_from_landmark"] = res[1]
    fc["censor_reason"] = res[2]

    assert fc["empi_anon"].is_unique, "final cohort not one-row-per-patient"
    fc.to_parquet(coh / "final_cohort.parquet", index=False)
    log.info("PRIMARY final_cohort (%s/%dd): n=%d events=%d written -> final_cohort.parquet",
             strategy, W, len(fc), int(fc["event_indicator"].sum()))

    if not ok:
        log.error("GRID RECONCILIATION FAILED — see MISMATCH lines above")
        return 1
    log.info("ALL %d grid cells reconcile to the re-gate. Primary=recovery_any/730.", len(ANCHORS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
