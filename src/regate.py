"""regate.py — Stage-1 feasibility RE-GATE grid (PRELIMINARY, read-only).

MRKR Contralateral TKA — Phase-1 metadata-only feasibility. The strict cohort
with a 1-365-day pre-index imaging window yields only 357 primary events, below
the >=500 protocol floor. This module computes the primary-event GRID over
(imaging window x cohort strategy) so the analyst can pick the most clinically
grounded configuration that clears the floor.

REUSE, DO NOT FORK. The strict-cohort construction and every gate definition are
imported directly from ``src.preliminary_counts`` and ``src.laterality``:

  * ``build_index_frames``    -> the EXACT strict index (S4 = 4,222) + ``per``/
    ``priorarth`` frames and the blank-index (earliest-blank) population.
  * ``sql_infection_flags``   -> S7a high-specificity osteomyelitis flag.
  * ``compute_cpt_flags``     -> primary_event, contra_0_90, prior_contra_cpt.
  * side-recovery signal SQL  -> reproduced from the preliminary_counts OUTPUT-3
    block (same-day image laterality / same-day StudyDescription text / ICD-M17
    laterality on-or-before index) so recovery_any reproduces the verified
    ``feasibility_stage1_counts.json`` side_recovery.concordant_single_side.

Only TWO things are added, both required by the grid spec and neither of which
changes a gate definition:
  1. A windowed contralateral-image eligibility SQL (the pre-index imaging
     window is the grid axis; the eligibility predicate itself — contra
     laterality OR B-frontal — is byte-for-byte the verified one).
  2. Recovered-index frames (recovery_any / recovery_confirmed) built from the
     earliest-blank population using ONLY same-day-or-earlier signals.

Boundary / guardrails honoured here (identical to preliminary_counts):
  * Read-only on the typed Parquet inputs; no DICOMs, no models, no metrics.
  * NO patient identifiers (``empi_anon``) are ever written. All persisted
    outputs are AGGREGATE COUNTS only; patient-level intermediates stay in
    memory / ephemeral DuckDB temp tables.
  * NO post-index leakage: side recovery uses only same-day-or-earlier signals;
    the predictor imaging window is strictly pre-index; the outcome is never
    used to define the index or its side.
  * Deterministic. Params come from ``config/feasibility.yaml``.
  * Logging APPENDS to ``outputs/logs/run.log`` with the ``regate`` prefix.

Run from the project root::

    python3 -m src.regate --config config/feasibility.yaml

Writes ONLY: ``outputs/tables/regate_grid.csv``, ``outputs/feasibility_regate.json``,
and appends ``outputs/logs/run.log``.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src.config import ensure_dirs, load_config
from src.laterality import (
    contralateral_side,
    days_between,
    horizon_date,
    landmark_date,
    last_observation,
)
# REUSE the verified strict-cohort construction + gate helpers verbatim.
from src.preliminary_counts import (
    _no_empi,
    _pct,
    _register_idx,
    _to_pydate,
    build_index_frames,
    compute_cpt_flags,
    create_views,
    sql_infection_flags,
)

MODULE = "regate"
PRELIM = "PRELIMINARY"

# Grid axes.
WINDOWS = [365, 730, 1095, "lifetime"]           # pre-index imaging window (gate 4)
STRATEGIES = ["strict", "recovery_any", "recovery_confirmed"]
_WIN_COL = {365: "elig_365", 730: "elig_730", 1095: "elig_1095", "lifetime": "elig_lifetime"}


# --------------------------------------------------------------------------- #
# Logging: append to run.log (module | ISO-timestamp | level | message).      #
# --------------------------------------------------------------------------- #
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(MODULE)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a")  # APPEND, never truncate
        fh._mrkr = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh._mrkr = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s"))
        logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# Windowed contralateral-image eligibility (gate 4 axis).                     #
# The eligibility predicate is byte-for-byte the verified one used in          #
# preliminary_counts.sql_image_flags (elig_img) / the decision-anchor scan;    #
# only the pre-index window boundary varies. prior_contra_img is unchanged.    #
# --------------------------------------------------------------------------- #
def sql_image_flags_windowed(con, idx_tbl: str) -> pd.DataFrame:
    elig = ("(g.laterality = t.contra_side "
            "OR (g.laterality = 'B' AND g.view_position = 'F'))")

    def pre(days: int) -> str:
        return (f"g.StudyDate_anon BETWEEN t.index_date - INTERVAL {days} DAY "
                f"AND t.index_date - INTERVAL 1 DAY")

    return con.execute(f"""
        SELECT t.empi_anon,
          MAX(CASE WHEN g.StudyDate_anon < t.index_date
                    AND g.arthroplasty IN (t.contra_side, 'B') THEN 1 ELSE 0 END) AS prior_contra_img,
          MAX(CASE WHEN {pre(365)}  AND {elig} THEN 1 ELSE 0 END) AS elig_365,
          MAX(CASE WHEN {pre(730)}  AND {elig} THEN 1 ELSE 0 END) AS elig_730,
          MAX(CASE WHEN {pre(1095)} AND {elig} THEN 1 ELSE 0 END) AS elig_1095,
          MAX(CASE WHEN g.StudyDate_anon < t.index_date AND {elig} THEN 1 ELSE 0 END) AS elig_lifetime
        FROM {idx_tbl} t JOIN image g ON g.empi_anon = t.empi_anon
        GROUP BY 1
    """).df()


# --------------------------------------------------------------------------- #
# Observation flag (S10) — max-date dict built ONCE, then reused per strategy. #
# Mirrors preliminary_counts.compute_observation's per-patient logic exactly   #
# (same last_observation + landmark_date helpers, same > comparison).          #
# --------------------------------------------------------------------------- #
def build_maxes(con) -> dict:
    frames = [
        con.execute("SELECT empi_anon, MAX(date_anon) m FROM cpt GROUP BY 1").df(),
        con.execute("SELECT empi_anon, MAX(date_anon) m FROM icd GROUP BY 1").df(),
        con.execute("SELECT empi_anon, MAX(date_anon) m FROM pain GROUP BY 1").df(),
        con.execute("SELECT empi_anon, MAX(StudyDate_anon) m FROM image GROUP BY 1").df(),
    ]
    maxes: dict = {}
    for df in frames:
        for r in df.itertuples():
            maxes.setdefault(r.empi_anon, []).append(_to_pydate(r.m))
    return maxes


def obs_flags(index_frame: pd.DataFrame, maxes: dict, landmark_days: int) -> pd.DataFrame:
    recs = []
    for r in index_frame.itertuples():
        idate = _to_pydate(r.index_date)
        maxobs = last_observation(maxes.get(r.empi_anon, []))
        lm = landmark_date(idate, landmark_days)
        ok = int(maxobs is not None and lm is not None and maxobs > lm)
        recs.append((r.empi_anon, ok))
    return pd.DataFrame(recs, columns=["empi_anon", "obs_ok"])


# --------------------------------------------------------------------------- #
# Recovered-index frames from the earliest-blank population.                   #
# Signals reproduced EXACTLY from preliminary_counts OUTPUT-3 (side recovery). #
# --------------------------------------------------------------------------- #
def build_recovered_signals(con, cfg, per: pd.DataFrame) -> pd.DataFrame:
    """Return one row per earliest-blank patient with the three side signals,
    the recovered side (concordant single side), and the number of signals that
    agree on that side. NO empi is persisted downstream — in-memory only."""
    blank = per[per["earliest_blank"]].copy()
    blank_idx = pd.DataFrame({
        "empi_anon": blank.index,
        "index_date": blank["first_date"].values,
        "index_age": blank["idx_age"].values,
    })
    con.register("blank_df", blank_idx[["empi_anon", "index_date"]])
    con.execute("""CREATE OR REPLACE TEMP TABLE blank_pop AS
                   SELECT CAST(empi_anon AS VARCHAR) empi_anon, CAST(index_date AS DATE) index_date
                   FROM blank_df""")
    con.unregister("blank_df")

    # (a) same-day image laterality; (c) same-day StudyDescription text.
    img_sig = con.execute("""
        SELECT b.empi_anon,
          MAX(CASE WHEN g.StudyDate_anon = b.index_date AND g.laterality='R' THEN 1 ELSE 0 END) img_R,
          MAX(CASE WHEN g.StudyDate_anon = b.index_date AND g.laterality='L' THEN 1 ELSE 0 END) img_L,
          MAX(CASE WHEN g.StudyDate_anon = b.index_date AND lower(g.StudyDescription) LIKE '%right%' THEN 1 ELSE 0 END) desc_R,
          MAX(CASE WHEN g.StudyDate_anon = b.index_date AND lower(g.StudyDescription) LIKE '%left%'  THEN 1 ELSE 0 END) desc_L
        FROM blank_pop b JOIN image g ON g.empi_anon = b.empi_anon
        GROUP BY 1""").df()
    # (b) ICD-M17 laterality on-or-before index (config codes; M17 IN-list already
    # excludes ICD10='--' and unspecified codes).
    rc = cfg["icd_side_recovery"]
    rcodes = "','".join(rc["right_codes"])
    lcodes = "','".join(rc["left_codes"])
    icd_sig = con.execute(f"""
        SELECT b.empi_anon,
          MAX(CASE WHEN i.ICD10 IN ('{rcodes}') AND i.date_anon <= b.index_date THEN 1 ELSE 0 END) icd_R,
          MAX(CASE WHEN i.ICD10 IN ('{lcodes}') AND i.date_anon <= b.index_date THEN 1 ELSE 0 END) icd_L
        FROM blank_pop b JOIN icd i ON i.empi_anon = b.empi_anon AND i.ICD10 LIKE 'M17%'
        GROUP BY 1""").df()

    sig = blank_idx.merge(img_sig, on="empi_anon", how="left") \
                   .merge(icd_sig, on="empi_anon", how="left")
    for c in ["img_R", "img_L", "desc_R", "desc_L", "icd_R", "icd_L"]:
        sig[c] = sig[c].fillna(0).astype(int)

    # per-side aggregate presence (identical to preliminary_counts tot_R / tot_L).
    tot_R = (sig.img_R + sig.icd_R + sig.desc_R) > 0
    tot_L = (sig.img_L + sig.icd_L + sig.desc_L) > 0
    sig["concordant_single"] = (tot_R ^ tot_L).astype(int)            # XOR: exactly one side, no conflict
    sig["recovered_side"] = pd.Series(
        ["R" if r else ("L" if l else None) for r, l in zip(tot_R, tot_L)], index=sig.index)
    # signals agreeing on the (single) concordant side; within concordant_single
    # every present signal points to that side, so agree = count of side signals.
    agree_R = sig.img_R + sig.icd_R + sig.desc_R
    agree_L = sig.img_L + sig.icd_L + sig.desc_L
    sig["n_agree"] = [int(ar if r else (al if l else 0))
                      for r, l, ar, al in zip(tot_R, tot_L, agree_R, agree_L)]
    # number of distinct signal-types present on the concordant side (0..3).
    sig["a_on"] = [int((ir if r else il)) for r, ir, il in zip(tot_R, sig.img_R, sig.img_L)]
    sig["b_on"] = [int((ir if r else il)) for r, ir, il in zip(tot_R, sig.icd_R, sig.icd_L)]
    sig["c_on"] = [int((ir if r else il)) for r, ir, il in zip(tot_R, sig.desc_R, sig.desc_L)]
    return sig


def recovered_index_frame(sig: pd.DataFrame, confirmed: bool) -> pd.DataFrame:
    """Recovered index rows. recovery_any = concordant single side (>=1 signal, no
    conflict). recovery_confirmed = additionally >=2 of the 3 signals agree."""
    m = sig["concordant_single"] == 1
    if confirmed:
        m = m & (sig["n_agree"] >= 2)
    r = sig[m].copy()
    out = pd.DataFrame({
        "empi_anon": r["empi_anon"].values,
        "index_date": r["index_date"].values,
        "index_side": r["recovered_side"].values,
        "contra_side": r["recovered_side"].map(contralateral_side).values,
        "index_age": r["index_age"].values,
    })
    out["recovered"] = True
    return out


# --------------------------------------------------------------------------- #
# Per-strategy: build the union index frame, compute all gate inputs once.     #
# --------------------------------------------------------------------------- #
def build_strategy_frame(con, cfg, log, strategy, strict_idx, sig,
                         df447, priorarth, maxes, landmark_days, event_start,
                         horizon_days, sec_windows):
    if strategy == "strict":
        idxf = strict_idx.copy()
    elif strategy == "recovery_any":
        idxf = pd.concat([strict_idx, recovered_index_frame(sig, confirmed=False)],
                         ignore_index=True)
    elif strategy == "recovery_confirmed":
        idxf = pd.concat([strict_idx, recovered_index_frame(sig, confirmed=True)],
                         ignore_index=True)
    else:
        raise ValueError(strategy)

    assert idxf["empi_anon"].is_unique, f"{strategy}: duplicate index patient"
    assert idxf["index_side"].isin(["R", "L"]).all(), f"{strategy}: non-single index side"
    assert idxf["contra_side"].isin(["R", "L"]).all(), f"{strategy}: non-single contra side"

    tbl = f"idx_{strategy}"
    _register_idx(con, tbl, idxf[["empi_anon", "index_date", "contra_side"]])

    inf = sql_infection_flags(con, tbl)                       # osteo, jinf
    img = sql_image_flags_windowed(con, tbl)                  # prior_contra_img + elig_{window}
    cptf = compute_cpt_flags(idxf, df447, priorarth,
                             "index_side", "contra_side", "index_date",
                             landmark_days, event_start, horizon_days, sec_windows)
    obs = obs_flags(idxf, maxes, landmark_days)

    F = idxf.merge(inf, on="empi_anon", how="left") \
            .merge(img, on="empi_anon", how="left") \
            .merge(cptf, on="empi_anon", how="left") \
            .merge(obs, on="empi_anon", how="left")
    flag_cols = [c for c in F.columns if c not in
                 ("empi_anon", "index_date", "contra_side", "index_side", "index_age", "recovered")]
    F[flag_cols] = F[flag_cols].fillna(0)
    for c in flag_cols:
        F[c] = F[c].astype(int)
    log.info("%s: strategy=%s n_index=%d (recovered=%d)",
             PRELIM, strategy, len(F), int(F["recovered"].sum()))
    return F


def run_flow_window(F: pd.DataFrame, elig_col: str, age_min: float):
    """Apply gates S5..S10 in the exact preliminary_counts order (high-specificity
    infection S7a carried forward). Only the eligibility column varies by window."""
    m = pd.Series(True, index=F.index)
    steps = {"S4": int(m.sum())}
    m = m & (F["index_age"] >= age_min);                         steps["S5"] = int(m.sum())
    m = m & (F["prior_contra_cpt"] == 0) & (F["prior_contra_img"] == 0); steps["S6"] = int(m.sum())
    m = m & (F["osteo"] == 0);                                   steps["S7a"] = int(m.sum())
    m = m & (F[elig_col] == 1);                                  steps["S8"] = int(m.sum())
    m = m & (F["contra_0_90"] == 0);                             steps["S9"] = int(m.sum())
    m = m & (F["obs_ok"] == 1);                                  steps["S10"] = int(m.sum())
    steps["S11"] = steps["S10"]
    return steps, m


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage-1 feasibility RE-GATE grid (read-only).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    tables_dir = cfg.out("tables_dir")
    outputs_dir = cfg.out("outputs_dir")

    tl = cfg["timeline"]
    landmark_days = int(tl["landmark_days"])
    event_start = int(tl["event_start_day"])
    horizon_years = float(tl["horizon_years"])
    days_per_year = float(tl["days_per_year"])
    _ref = date(2000, 1, 1)
    horizon_days = days_between(_ref, horizon_date(_ref, horizon_years, days_per_year))
    age_min = float(cfg["index"]["age_min"])
    age_cut = int(cfg["subgroups"]["age_cutoff"])
    sec_windows = cfg["secondary_event_windows"]
    floors = cfg["floors"]
    ev_min = int(floors["primary_events_min"])
    ta_min = int(floors["test_allocatable_min"])
    rgr = cfg["subgroups"]["race_groups"]
    race_black, race_white, race_asian = rgr["black"], rgr["white"], rgr["asian"]

    log.info("START %s RE-GATE grid (landmark=%dd event_start=%dd horizon=%dd age_min=%.0f "
             "floors: events>=%d test>=%d)", PRELIM, landmark_days, event_start, horizon_days,
             age_min, ev_min, ta_min)

    tmpdir = tempfile.mkdtemp(prefix="mrkr_regate_")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmpdir}'")
    con.execute("SET preserve_insertion_order=false")

    try:
        create_views(con, cfg)
        # EXACT strict construction (S4=4222) + per/priorarth + blank population.
        df447, per, priorarth = build_index_frames(con, cfg, log)

        strict = per[per["strict"]].copy()
        strict_idx = pd.DataFrame({
            "empi_anon": strict.index,
            "index_date": strict["index_date_strict"].values,
            "contra_side": strict["contra_strict"].values,
            "index_side": strict["index_side_strict"].values,
            "index_age": strict["index_age_strict"].values,
        })
        strict_idx["recovered"] = False
        n_strict = int(len(strict_idx))
        assert n_strict == int(per["strict"].sum())

        # Side-recovery signals over the earliest-blank population.
        sig = build_recovered_signals(con, cfg, per)
        n_blank = int(len(sig))
        n_concordant = int((sig["concordant_single"] == 1).sum())
        n_confirmed = int(((sig["concordant_single"] == 1) & (sig["n_agree"] >= 2)).sum())
        log.info("%s: blank_pop=%d concordant_single=%d confirmed(>=2 agree)=%d",
                 PRELIM, n_blank, n_concordant, n_confirmed)

        # Global last-observation dict (built once, reused across strategies).
        maxes = build_maxes(con)

        # demographics for subgroup EVENT tallies.
        demo = con.execute("SELECT empi_anon, sex, race FROM demographics").df()

        # ---- compute the grid ------------------------------------------- #
        grid_rows = []
        json_cells = []
        strat_meta = {}
        frames = {}
        for strategy in STRATEGIES:
            F = build_strategy_frame(con, cfg, log, strategy, strict_idx, sig,
                                     df447, priorarth, maxes, landmark_days, event_start,
                                     horizon_days, sec_windows)
            F = F.merge(demo, on="empi_anon", how="left")
            frames[strategy] = F
            n_index = int(len(F))
            n_recovered_index = int(F["recovered"].sum())

            for window in WINDOWS:
                elig_col = _WIN_COL[window]
                steps, mask = run_flow_window(F, elig_col, age_min)
                final = F[mask]
                n_final = int(len(final))
                is_ev = final["primary_event"] == 1
                primary = int(is_ev.sum())
                test_alloc = int(round(0.20 * primary))
                n_final_recovered = int(final["recovered"].sum())

                def ev(mask_series):
                    return int((is_ev & mask_series).sum())

                row = {
                    "strategy": strategy,
                    "window_days": window,
                    "n_index": n_index,
                    "n_final_cohort": n_final,
                    "primary_events": primary,
                    "test_allocatable_20pct": test_alloc,
                    "clears_500": bool(primary >= ev_min),
                    "clears_100": bool(test_alloc >= ta_min),
                    "pct_index_recovered": round(100.0 * n_final_recovered / n_final, 3) if n_final else 0.0,
                    "black_events": ev(final["race"] == race_black),
                    "white_events": ev(final["race"] == race_white),
                    "asian_events": ev(final["race"] == race_asian),
                    "female_events": ev(final["sex"] == "Female"),
                    "male_events": ev(final["sex"] == "Male"),
                    "age_lt65_events": ev(final["index_age"] < age_cut),
                    "age_ge65_events": ev(final["index_age"] >= age_cut),
                }
                grid_rows.append(row)
                cell = dict(row)
                cell["flow"] = steps
                cell["n_final_recovered"] = n_final_recovered
                json_cells.append(cell)
                log.info("%s: [%s | win=%s] flow=%s primary=%d test20=%d clears(500/100)=%s/%s recov%%=%.1f",
                         PRELIM, strategy, window,
                         [steps[k] for k in ("S4", "S5", "S6", "S7a", "S8", "S9", "S10", "S11")],
                         primary, test_alloc, row["clears_500"], row["clears_100"],
                         row["pct_index_recovered"])

            # per-strategy recovered counts + signal breakdown (aggregates only).
            if strategy == "strict":
                strat_meta[strategy] = {"n_index": n_index, "n_recovered_index": 0,
                                        "recovered_note": "no recovery; index side always coded"}
            else:
                confirmed = strategy == "recovery_confirmed"
                rmask = sig["concordant_single"] == 1
                if confirmed:
                    rmask = rmask & (sig["n_agree"] >= 2)
                rsig = sig[rmask]
                strat_meta[strategy] = {
                    "n_index": n_index,
                    "n_recovered_index": n_recovered_index,
                    "recovered_side_R": int((rsig["recovered_side"] == "R").sum()),
                    "recovered_side_L": int((rsig["recovered_side"] == "L").sum()),
                    "agreeing_signal_count_hist": {
                        "1_signal": int((rsig["n_agree"] == 1).sum()),
                        "2_signals": int((rsig["n_agree"] == 2).sum()),
                        "3_signals": int((rsig["n_agree"] == 3).sum()),
                    },
                    "signal_present_on_recovered_side": {
                        "same_day_image_laterality": int(rsig["a_on"].sum()),
                        "icd_m17_on_or_before": int(rsig["b_on"].sum()),
                        "same_day_studydesc_text": int(rsig["c_on"].sum()),
                    },
                }

        grid = pd.DataFrame(grid_rows)

        # ---- VALIDATION ------------------------------------------------- #
        # (1) regression: strict/365 must reproduce the known values.
        def cell(strat, win):
            return grid[(grid.strategy == strat) & (grid.window_days == win)].iloc[0]
        s365 = cell("strict", 365)
        assert int(s365["n_index"]) == 4222, f"strict n_index {int(s365['n_index'])} != 4222"
        assert int(s365["n_final_cohort"]) == 1664, \
            f"REGRESSION FAIL: strict/365 n_final {int(s365['n_final_cohort'])} != 1664"
        assert int(s365["primary_events"]) == 357, \
            f"REGRESSION FAIL: strict/365 primary {int(s365['primary_events'])} != 357"
        # full strict/365 flow must match preliminary_counts run.log.
        strict365_flow = [json_cells[0]["flow"][k]
                          for k in ("S4", "S5", "S6", "S7a", "S8", "S9", "S10", "S11")]
        assert strict365_flow == [4222, 4203, 3756, 3752, 1807, 1775, 1664, 1664], \
            f"strict/365 flow diverged: {strict365_flow}"
        # (2) recovery anchor: concordant_single must reproduce the verified 2983.
        assert n_concordant == 2983, f"concordant_single {n_concordant} != 2983 (side-recovery diverged)"
        # (3) monotonicity: wider window => n_final & primary non-decreasing (fixed strategy).
        for strat in STRATEGIES:
            sub = grid[grid.strategy == strat].set_index("window_days")
            seq_f = [int(sub.loc[w, "n_final_cohort"]) for w in WINDOWS]
            seq_e = [int(sub.loc[w, "primary_events"]) for w in WINDOWS]
            assert all(seq_f[i] <= seq_f[i + 1] for i in range(len(seq_f) - 1)), \
                f"{strat}: n_final not monotone over window: {seq_f}"
            assert all(seq_e[i] <= seq_e[i + 1] for i in range(len(seq_e) - 1)), \
                f"{strat}: primary not monotone over window: {seq_e}"
        # (4) monotonicity across strategies at fixed window: any >= confirmed >= strict.
        for window in WINDOWS:
            a = cell("recovery_any", window)
            c = cell("recovery_confirmed", window)
            s = cell("strict", window)
            for col in ("n_final_cohort", "primary_events"):
                assert int(a[col]) >= int(c[col]) >= int(s[col]), \
                    f"window {window}: {col} not nested any>=confirmed>=strict: " \
                    f"{int(a[col])},{int(c[col])},{int(s[col])}"
        # (5) no identifier leakage in the persisted grid.
        _no_empi(grid, "regate_grid")

        # ---- WRITE grid CSV --------------------------------------------- #
        col_order = ["strategy", "window_days", "n_index", "n_final_cohort", "primary_events",
                     "test_allocatable_20pct", "clears_500", "clears_100", "pct_index_recovered",
                     "black_events", "white_events", "asian_events", "female_events", "male_events",
                     "age_lt65_events", "age_ge65_events"]
        grid = grid[col_order]
        grid_out = grid.copy()
        grid_out.insert(0, "label", PRELIM)
        grid_out.to_csv(tables_dir / "regate_grid.csv", index=False)

        # ---- WRITE JSON ------------------------------------------------- #
        cells_clearing = [{"strategy": r["strategy"], "window_days": r["window_days"],
                           "primary_events": r["primary_events"],
                           "test_allocatable_20pct": r["test_allocatable_20pct"],
                           "pct_index_recovered": r["pct_index_recovered"]}
                          for r in json_cells if r["clears_500"] and r["clears_100"]]

        headline = {
            "label": PRELIM,
            "study": "MRKR Contralateral TKA — Phase-1 metadata-only feasibility RE-GATE",
            "note": ("Primary-event grid over (imaging window x cohort strategy). Strict-cohort "
                     "construction and every gate are reused verbatim from preliminary_counts; "
                     "only the pre-index imaging window (gate 4) and the index-side recovery vary. "
                     "Aggregates only; no patient identifiers; no post-index leakage."),
            "timeline": {"landmark_days": landmark_days, "event_start_day": event_start,
                         "horizon_days": horizon_days, "age_min": age_min},
            "floors": {"primary_events_min": ev_min, "test_allocatable_min": ta_min},
            "axes": {"strategies": STRATEGIES, "windows": [str(w) for w in WINDOWS]},
            "base_populations": {
                "n_strict_index": n_strict,
                "blank_index_population": n_blank,
                "concordant_single_side_recovered": n_concordant,
                "confirmed_recovered_ge2_signals": n_confirmed,
            },
            "strategy_recovery": strat_meta,
            "grid": json_cells,
            "cells_clearing_floor": cells_clearing,
            "validation": {
                "strict_365_n_final": int(s365["n_final_cohort"]),
                "strict_365_primary_events": int(s365["primary_events"]),
                "strict_365_regression_pass": True,
                "strict_365_flow": strict365_flow,
                "concordant_single_side_matches_2983": n_concordant == 2983,
            },
        }
        # belt-and-braces: no 'empi' anywhere in the serialized JSON.
        blob = json.dumps(headline, default=str)
        assert "empi" not in blob.lower(), "identifier token found in JSON output"

        json_path = outputs_dir / "feasibility_regate.json"
        with open(json_path, "w") as fh:
            json.dump(headline, fh, indent=2, default=str)

        log.info("%s: regression OK — strict/365 n_final=%d primary=%d; concordant_single=%d(==2983)",
                 PRELIM, int(s365["n_final_cohort"]), int(s365["primary_events"]), n_concordant)
        log.info("%s: cells clearing floor (>=%d events AND >=%d test): %s",
                 PRELIM, ev_min, ta_min,
                 [f"{c['strategy']}/{c['window_days']}" for c in cells_clearing] or "NONE")
        log.info("%s: DONE — wrote regate_grid.csv (%d rows) + feasibility_regate.json",
                 PRELIM, len(grid))
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
