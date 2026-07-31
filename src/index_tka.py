"""index_tka.py — LOCKED cohort extraction FOUNDATION module.

MRKR Contralateral TKA — Phase-1 feasibility, LOCKED EXTRACTION phase (author
sign-off 2026-07-21, Decision E). This module builds the per-patient INDEX for
every patient with >=1 CPT 27447 and applies the three INDEX-LEVEL gates. It is
the shared foundation the downstream imaging / outcomes / follow-up modules read.

Scope boundary (unchanged from the feasibility work):
  * Read-only on the typed Parquet inputs; NO DICOMs, NO models, NO metrics.
  * NO post-index leakage: side recovery uses only same-day-or-earlier signals;
    every index-level gate uses only at-or-before-index data. The imaging-window
    and event/outcome gates are NOT applied here (they belong to later modules).
  * Deterministic; params come from ``config/feasibility.yaml``.

REUSE, DO NOT FORK. The verified index / recovery / gate logic is imported
directly from ``src.preliminary_counts``, ``src.regate`` and ``src.laterality``:
  * ``build_index_frames``       -> df447 / per / priorarth; the coded (strict)
    earliest-27447 single-side determination (Decision A) and the earliest-blank
    population definition.
  * ``build_recovered_signals`` / ``recovered_index_frame`` (regate) -> the
    same-day image laterality + ICD-M17 + StudyDescription side recovery over the
    earliest-blank population (concordant single side, no conflict).
  * ``sql_infection_flags``      -> high-specificity osteomyelitis gate (S7a).
  * ``sql_image_flags``          -> prior contralateral prosthesis on image (S6b).
  * ``compute_cpt_flags``        -> prior contralateral arthroplasty CPT (S6a).

Unlike preliminary_counts / regate, this module DOES persist patient-level cohort
tables (they legitimately carry ``empi_anon`` — they are git-ignored derived
linkage tables for downstream joins). The only aggregate, id-free output is
``outputs/tables/index_flow.csv``.

Persists (to ``derived-data/cohort/``):
  * ``index_candidates.parquet`` — one row per patient with >=1 27447.
  * ``index_final.parquet``      — recovery_any rows passing ALL index-level gates
    (the SHARED CONTRACT downstream modules read).
Emits:
  * ``outputs/tables/index_flow.csv`` — per-strategy index-level funnel (counts).
Appends ``outputs/logs/run.log`` (prefix ``index_tka``).

Run from the project root::

    python3 -m src.index_tka --config config/feasibility.yaml
"""
from __future__ import annotations

import argparse
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
    DEFAULT_BILATERAL_TOKENS,
    DEFAULT_LEFT_TOKENS,
    DEFAULT_RIGHT_TOKENS,
    contralateral_side,
    days_between,
    horizon_date,
    parse_modifier,
)
# REUSE the verified index-frame construction + gate SQL verbatim.
from src.preliminary_counts import (
    _no_empi,
    _register_idx,
    build_index_frames,
    compute_cpt_flags,
    create_views,
    sql_image_flags,
    sql_infection_flags,
)
# REUSE the verified same-day-or-earlier side recovery verbatim.
from src.regate import build_recovered_signals, recovered_index_frame

MODULE = "index_tka"
LABEL = "LOCKED"

# Reason codes for patients with no valid index (side_source == 'none').
REASON_UNSIDED_NO_BLANK = "unsided_earliest_no_blank_modifier"
REASON_CONFLICT = "conflicting_recovery_signal"
REASON_NO_SIGNAL = "no_recovery_signal"

# Regression anchors (index-level funnel per strategy):
#   (n_valid_index, n_age40, n_no_prior_contra_arth, n_no_infection)
# strict reproduces preliminary_counts S4..S7a; recovery_* reproduce regate.
EXPECTED_FUNNEL = {
    "strict":             (4222, 4203, 3756, 3752),
    "recovery_any":       (7205, 7112, 6393, 6381),
    "recovery_confirmed": (6903, 6815, 6141, 6131),
}
EXPECTED_N_27447 = 8525          # distinct patients with >=1 CPT 27447
EXPECTED_N_RECOVERED = 2983      # concordant-single-side recovered (before gates)


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
# PURE helpers (no I/O, no side effects) — unit-tested in tests/test_index_tka #
# --------------------------------------------------------------------------- #
def _normalize_signal_sides(val) -> set[str]:
    """Coerce one recovery signal's value to a set of sides in {'R','L'}.

    Accepts ``None`` (no signal), a single side string (``'R'``/``'L'``), or any
    iterable of side strings (e.g. ``{'R'}``, ``['R','L']``). Anything not R/L is
    dropped. A signal that carries BOTH ``R`` and ``L`` yields ``{'R','L'}`` and
    therefore contributes a conflict.
    """
    if val is None:
        return set()
    if isinstance(val, str):
        v = val.strip().upper()
        return {v} & {"R", "L"}
    try:
        it = iter(val)
    except TypeError:
        return set()
    out: set[str] = set()
    for x in it:
        if x is None:
            continue
        xs = str(x).strip().upper()
        if xs in ("R", "L"):
            out.add(xs)
    return out


def resolve_index_side(
    same_day_modifiers,
    recovery_signals=None,
    *,
    right_tokens=DEFAULT_RIGHT_TOKENS,
    left_tokens=DEFAULT_LEFT_TOKENS,
    bilateral_tokens=DEFAULT_BILATERAL_TOKENS,
) -> dict:
    """Resolve one patient's index side + provenance from their earliest-27447 date.

    This is the pure, canonical statement of the module's index-side decision
    (Decision A coded rule, then blank-earliest recovery). It mirrors the verified
    vectorized path in :func:`main` exactly and is cross-checked against it at run
    time for every patient.

    Args:
        same_day_modifiers: the raw ``cpt_group_modifier`` values on the earliest
            27447 DATE (the same-day companion lines, any of which may be blank/None).
        recovery_signals: optional mapping ``{signal_name: sides}`` describing the
            same-day-or-earlier recovery signals for the blank case, where ``sides``
            is coerced by :func:`_normalize_signal_sides` to a subset of {'R','L'}.
            Only consulted when the earliest date is blank/unsided.
        right_tokens / left_tokens / bilateral_tokens: laterality tokens (defaults
            mirror ``config/feasibility.yaml``); passed through to ``parse_modifier``.

    Returns:
        dict with keys ``index_side`` ('R'|'L'|None), ``contra_side`` ('R'|'L'|None),
        ``side_source`` ('coded'|'recovered'|'none'), ``n_concordant_signals``
        (0 for coded/none, 1..3 for recovered), and ``exclude_reason`` (None unless
        side_source == 'none').

    Decision order:
      1. Coded (Decision A): among the earliest-date companion lines, if exactly one
         of R/L is present (the other absent) -> coded to that side, regardless of
         any blank companions. (No same-date R+L conflicts exist in the data.)
      2. Otherwise the earliest date is unsided. If NO blank/NULL companion line is
         present -> no valid index (``unsided_earliest_no_blank_modifier``); recovery
         is attempted ONLY on a genuinely blank earliest date.
      3. Recovery: aggregate the signals. Both sides present -> conflict (excluded).
         No side present -> no signal (excluded). Exactly one side across signals ->
         recovered to that side, ``n_concordant_signals`` = number of signals that
         carry that side.
    """
    n_R = n_L = n_missing = 0
    for raw in (same_day_modifiers or []):
        side, flag = parse_modifier(raw, right_tokens, left_tokens, bilateral_tokens)
        if side == "R":
            n_R += 1
        elif side == "L":
            n_L += 1
        if flag == "missing":
            n_missing += 1

    # 1. Coded single side (Decision A) — dominates, blank companions allowed.
    coded_side = None
    if n_R > 0 and n_L == 0:
        coded_side = "R"
    elif n_L > 0 and n_R == 0:
        coded_side = "L"
    if coded_side is not None:
        return {"index_side": coded_side, "contra_side": contralateral_side(coded_side),
                "side_source": "coded", "n_concordant_signals": 0, "exclude_reason": None}

    # 2. Unsided earliest with no blank companion -> not eligible for recovery.
    if n_missing <= 0:
        return {"index_side": None, "contra_side": None, "side_source": "none",
                "n_concordant_signals": 0, "exclude_reason": REASON_UNSIDED_NO_BLANK}

    # 3. Recovery from concordant same-day-or-earlier signals (no conflict).
    sides_by_signal = {name: _normalize_signal_sides(val)
                       for name, val in (recovery_signals or {}).items()}
    tot_R = any("R" in s for s in sides_by_signal.values())
    tot_L = any("L" in s for s in sides_by_signal.values())
    if tot_R and tot_L:
        return {"index_side": None, "contra_side": None, "side_source": "none",
                "n_concordant_signals": 0, "exclude_reason": REASON_CONFLICT}
    if not tot_R and not tot_L:
        return {"index_side": None, "contra_side": None, "side_source": "none",
                "n_concordant_signals": 0, "exclude_reason": REASON_NO_SIGNAL}
    rec_side = "R" if tot_R else "L"
    n_sig = sum(1 for s in sides_by_signal.values() if rec_side in s)
    return {"index_side": rec_side, "contra_side": contralateral_side(rec_side),
            "side_source": "recovered", "n_concordant_signals": n_sig, "exclude_reason": None}


# --------------------------------------------------------------------------- #
# Small write helper.                                                          #
# --------------------------------------------------------------------------- #
def _q(p) -> str:
    """Single-quote-escape a path for inline SQL."""
    return str(p).replace("'", "''")


def _copy_parquet(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame,
                  select_sql: str, path: Path) -> None:
    """Register ``frame`` as ``out_df`` and COPY ``select_sql`` to a Parquet file
    with explicit column types (so the persisted schema is the locked contract)."""
    con.register("out_df", frame)
    con.execute(f"COPY ({select_sql}) TO '{_q(path)}' (FORMAT PARQUET)")
    con.unregister("out_df")


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LOCKED index-TKA foundation extraction.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    tables_dir = cfg.out("tables_dir")
    cohort_dir = cfg.path(cfg["paths"]["cohort_dir"])
    cohort_dir.mkdir(parents=True, exist_ok=True)

    # LOCKED primary definition (Decision E) — assert we are building what was signed off.
    pdf = cfg["primary_definition"]
    assert pdf["cohort_strategy"] == "recovery_any", \
        f"primary cohort_strategy must be recovery_any, got {pdf['cohort_strategy']!r}"
    assert list(pdf["pre_index_window_days"]) == [1, 730], \
        f"primary pre_index_window must be [1, 730], got {pdf['pre_index_window_days']}"

    # laterality tokens (from config).
    lat = cfg["laterality"]
    rt = list(lat.get("right_tokens", ["RT"]))
    lt = list(lat.get("left_tokens", ["LT"]))
    bl = list(lat.get("bilateral_tokens", ["50"]))

    # timeline constants (only needed to satisfy compute_cpt_flags; NOT applied as
    # gates here — the outcome/event windows belong to downstream modules).
    tl = cfg["timeline"]
    landmark_days = int(tl["landmark_days"])
    event_start = int(tl["event_start_day"])
    horizon_years = float(tl["horizon_years"])
    days_per_year = float(tl["days_per_year"])
    _ref = date(2000, 1, 1)
    horizon_days = days_between(_ref, horizon_date(_ref, horizon_years, days_per_year))
    sec_windows = cfg["secondary_event_windows"]
    age_min = float(cfg["index"]["age_min"])

    log.info("START %s index extraction (primary=recovery_any/730; age_min=%.0f; "
             "index-level gates: age>=40, no prior contra arthroplasty, no high-spec infection)",
             LABEL, age_min)

    tmpdir = tempfile.mkdtemp(prefix="mrkr_index_")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmpdir}'")
    con.execute("SET preserve_insertion_order=false")

    try:
        create_views(con, cfg)
        # -------------------------------------------------------------- #
        # 1. Verified index frames (coded / earliest-blank / priorarth). #
        # -------------------------------------------------------------- #
        df447, per, priorarth = build_index_frames(con, cfg, log)
        n_with_27447 = int(len(per))
        assert n_with_27447 == EXPECTED_N_27447, \
            f"n_with_27447 {n_with_27447} != {EXPECTED_N_27447}"

        # coded index (side_source='coded') = the strict earliest-single-side cohort.
        strict = per[per["strict"]].copy()
        coded = pd.DataFrame({
            "empi_anon": strict.index,
            "index_date": strict["index_date_strict"].values,
            "index_side": strict["index_side_strict"].values,
            "contra_side": strict["contra_strict"].values,
            "age_at_index": strict["index_age_strict"].values,
        })
        coded["side_source"] = "coded"
        coded["n_concordant_signals"] = 0

        # -------------------------------------------------------------- #
        # 2. Recovered index (side_source='recovered') over blank-earliest#
        #    via same-day-or-earlier concordant signals (no conflict).    #
        # -------------------------------------------------------------- #
        sig = build_recovered_signals(con, cfg, per)
        n_recovered = int((sig["concordant_single"] == 1).sum())
        assert n_recovered == EXPECTED_N_RECOVERED, \
            f"recovered concordant-single {n_recovered} != {EXPECTED_N_RECOVERED}"

        rec = recovered_index_frame(sig, confirmed=False)  # recovery_any candidates
        rec = rec.merge(sig[["empi_anon", "n_agree"]], on="empi_anon", how="left")
        rec = rec.rename(columns={"index_age": "age_at_index", "n_agree": "n_concordant_signals"})
        rec["side_source"] = "recovered"
        assert len(rec) == n_recovered

        cols = ["empi_anon", "index_date", "index_side", "contra_side",
                "age_at_index", "side_source", "n_concordant_signals"]
        # recovery_any valid-index universe = coded + all recovered.
        U = pd.concat([coded[cols], rec[cols]], ignore_index=True)
        assert U["empi_anon"].is_unique, "duplicate empi across coded + recovered"
        assert U["index_side"].isin(["R", "L"]).all()
        assert U["contra_side"].isin(["R", "L"]).all()

        # -------------------------------------------------------------- #
        # 3. INDEX-LEVEL gates (reuse verified SQL / cpt-flag helpers).   #
        #    S5 age>=40 ; S6 no prior contralateral arthroplasty         #
        #    (contra-side CPT before index OR contra/B image prosthesis) ;#
        #    S7a no high-specificity osteomyelitis in [index-365, index). #
        # -------------------------------------------------------------- #
        _register_idx(con, "idx_union", U[["empi_anon", "index_date", "contra_side"]])
        inf = sql_infection_flags(con, "idx_union")          # osteo, jinf
        img = sql_image_flags(con, "idx_union")              # prior_contra_img (+ unused)
        cptf = compute_cpt_flags(U, df447, priorarth,
                                 "index_side", "contra_side", "index_date",
                                 landmark_days, event_start, horizon_days, sec_windows)
        U = U.merge(inf[["empi_anon", "osteo"]], on="empi_anon", how="left") \
             .merge(img[["empi_anon", "prior_contra_img"]], on="empi_anon", how="left") \
             .merge(cptf[["empi_anon", "prior_contra_cpt"]], on="empi_anon", how="left")
        for c in ("osteo", "prior_contra_img", "prior_contra_cpt"):
            U[c] = U[c].fillna(0).astype(int)
        U["age_ok"] = (U["age_at_index"] >= age_min)
        U["no_prior_contra_arthroplasty"] = (U["prior_contra_cpt"] == 0) & (U["prior_contra_img"] == 0)
        U["no_infection_highspec"] = (U["osteo"] == 0)

        n_coded = int((U["side_source"] == "coded").sum())
        n_rec = int((U["side_source"] == "recovered").sum())
        log.info("%s: index universe (recovery_any) = %d (coded=%d, recovered=%d)",
                 LABEL, len(U), n_coded, n_rec)

        # -------------------------------------------------------------- #
        # 4. Per-strategy index-level funnel (counts only).              #
        # -------------------------------------------------------------- #
        def funnel(mask: pd.Series) -> tuple[int, int, int, int]:
            sub = U[mask]
            a = sub[sub["age_ok"]]                                     # S5
            p = a[a["no_prior_contra_arthroplasty"]]                   # S6
            i = p[p["no_infection_highspec"]]                         # S7a
            return (len(sub), len(a), len(p), len(i))

        strat_masks = {
            "strict": U["side_source"] == "coded",
            "recovery_any": pd.Series(True, index=U.index),
            "recovery_confirmed": (U["side_source"] == "coded") | (U["n_concordant_signals"] >= 2),
        }
        funnels = {s: funnel(m) for s, m in strat_masks.items()}
        for s, f in funnels.items():
            assert f == EXPECTED_FUNNEL[s], \
                f"REGRESSION FAIL: {s} funnel {f} != {EXPECTED_FUNNEL[s]}"
            log.info("%s: funnel[%s] n_valid=%d age40=%d no_prior_contra=%d no_infection=%d",
                     LABEL, s, *f)

        # -------------------------------------------------------------- #
        # 5. index_final = recovery_any rows passing ALL index-level gates#
        # -------------------------------------------------------------- #
        final_mask = (U["age_ok"] & U["no_prior_contra_arthroplasty"] & U["no_infection_highspec"])
        index_final = U[final_mask][cols].copy().sort_values("empi_anon").reset_index(drop=True)
        assert len(index_final) == EXPECTED_FUNNEL["recovery_any"][3] == 6381, len(index_final)
        assert index_final["empi_anon"].is_unique, "index_final has duplicate empi_anon"
        # index_date is strictly the earliest 27447 date.
        _chk = index_final.merge(per["first_date"].rename("fd"),
                                 left_on="empi_anon", right_index=True, how="left")
        assert (pd.to_datetime(_chk["index_date"]) == pd.to_datetime(_chk["fd"])).all(), \
            "index_final.index_date is not the earliest 27447 date"
        # documented subsets of the shared contract.
        n_final_coded = int((index_final["side_source"] == "coded").sum())
        n_final_rec = int((index_final["side_source"] == "recovered").sum())
        assert n_final_coded == EXPECTED_FUNNEL["strict"][3] == 3752, n_final_coded
        rec_final = index_final[index_final["side_source"] == "recovered"]
        rec_sig_hist = {k: int((rec_final["n_concordant_signals"] == k).sum()) for k in (1, 2, 3)}
        log.info("%s: index_final=%d (coded=%d, recovered=%d; recovered n_signals 1/2/3=%d/%d/%d)",
                 LABEL, len(index_final), n_final_coded, n_final_rec,
                 rec_sig_hist[1], rec_sig_hist[2], rec_sig_hist[3])

        # -------------------------------------------------------------- #
        # 6. index_candidates — one row per 27447 patient (incl. 'none').#
        # -------------------------------------------------------------- #
        # exclude_reason for the 'none' patients.
        reason_map: dict = {}
        sig_r = sig.copy()
        sig_r["tot_R"] = (sig_r["img_R"] + sig_r["icd_R"] + sig_r["desc_R"]) > 0
        sig_r["tot_L"] = (sig_r["img_L"] + sig_r["icd_L"] + sig_r["desc_L"]) > 0
        for r in sig_r.itertuples():
            if r.concordant_single == 1:
                continue                                              # recovered, not 'none'
            reason_map[r.empi_anon] = REASON_CONFLICT if (r.tot_R and r.tot_L) else REASON_NO_SIGNAL
        for e in per[(~per["strict"]) & (~per["earliest_blank"])].index:
            reason_map[e] = REASON_UNSIDED_NO_BLANK

        cand = pd.DataFrame(index=per.index)
        cand["index_date"] = per["first_date"]
        cand["age_at_index"] = per["idx_age"]
        attr = U.set_index("empi_anon")[
            ["index_side", "contra_side", "side_source", "n_concordant_signals",
             "age_ok", "no_prior_contra_arthroplasty", "no_infection_highspec"]]
        cand = cand.join(attr)
        none_mask = cand["side_source"].isna()
        cand["side_source"] = cand["side_source"].where(~none_mask, "none")
        cand["n_concordant_signals"] = cand["n_concordant_signals"].fillna(0).astype(int)
        cand["exclude_reason"] = [reason_map.get(e) for e in cand.index]
        cand = cand.reset_index()  # -> empi_anon column (per.index.name == 'empi_anon')

        # nullable-friendly dtypes: 'none' rows have NULL side/contra/gate flags.
        def _nastr(v):
            return None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)

        def _nabool(v):
            return None if (v is None or (isinstance(v, float) and pd.isna(v))) else bool(v)

        for c in ("index_side", "contra_side"):
            cand[c] = cand[c].map(_nastr).astype(object)
        cand["exclude_reason"] = cand["exclude_reason"].map(_nastr).astype(object)
        for c in ("age_ok", "no_prior_contra_arthroplasty", "no_infection_highspec"):
            cand[c] = pd.array([_nabool(v) for v in cand[c]], dtype="boolean")
        cand = cand.sort_values("empi_anon").reset_index(drop=True)

        n_none = int((cand["side_source"] == "none").sum())
        assert n_coded + n_rec + n_none == n_with_27447, \
            f"candidate partition {n_coded}+{n_rec}+{n_none} != {n_with_27447}"
        reason_counts = (cand.loc[cand["side_source"] == "none", "exclude_reason"]
                         .value_counts().to_dict())
        log.info("%s: candidates=%d (coded=%d, recovered=%d, none=%d; reasons=%s)",
                 LABEL, len(cand), n_coded, n_rec, n_none, reason_counts)

        # -------------------------------------------------------------- #
        # 7. Cross-check the pure resolve_index_side against the verified #
        #    vectorized classification for EVERY patient (single source   #
        #    of truth guard). Aggregate-only work; nothing is written.    #
        # -------------------------------------------------------------- #
        _d = df447.merge(per["first_date"].rename("first_date"),
                         left_on="empi_anon", right_index=True)
        _earl = _d[_d["date_anon"] == _d["first_date"]]
        mod_lists = _earl.groupby("empi_anon")["cpt_group_modifier"].apply(list).to_dict()
        sig_map: dict = {}
        for r in sig.itertuples():
            sig_map[r.empi_anon] = {
                "same_day_image_laterality":
                    (({"R"} if r.img_R else set()) | ({"L"} if r.img_L else set())),
                "icd_m17_laterality":
                    (({"R"} if r.icd_R else set()) | ({"L"} if r.icd_L else set())),
                "studydesc_text":
                    (({"R"} if r.desc_R else set()) | ({"L"} if r.desc_L else set())),
            }
        auth = {r.empi_anon: (r.side_source, r.index_side, int(r.n_concordant_signals))
                for r in U.itertuples()}
        mism = 0
        for e in per.index:
            res = resolve_index_side(mod_lists.get(e, []), sig_map.get(e),
                                     right_tokens=rt, left_tokens=lt, bilateral_tokens=bl)
            exp = auth.get(e, ("none", None, 0))
            exp_side = None if (isinstance(exp[1], float) and pd.isna(exp[1])) else exp[1]
            if (res["side_source"], res["index_side"], res["n_concordant_signals"]) \
                    != (exp[0], exp_side, exp[2]):
                mism += 1
                if mism <= 5:
                    log.warning("%s: resolve_index_side mismatch empi=%s got=%s exp=%s", LABEL, e,
                                (res["side_source"], res["index_side"], res["n_concordant_signals"]),
                                (exp[0], exp_side, exp[2]))
        assert mism == 0, f"resolve_index_side diverged from vectorized path for {mism} patients"
        log.info("%s: resolve_index_side matches vectorized classification for all %d patients",
                 LABEL, n_with_27447)

        # -------------------------------------------------------------- #
        # 8. WRITE outputs.                                              #
        # -------------------------------------------------------------- #
        flow_rows = [(s, n_with_27447, *funnels[s])
                     for s in ("strict", "recovery_any", "recovery_confirmed")]
        flow = pd.DataFrame(flow_rows, columns=[
            "strategy", "n_with_27447", "n_valid_index", "n_age40",
            "n_no_prior_contra_arth", "n_no_infection"])
        flow.insert(0, "label", LABEL)
        _no_empi(flow, "index_flow")
        flow.to_csv(tables_dir / "index_flow.csv", index=False)

        _copy_parquet(con, cand, """
            SELECT
              CAST(empi_anon AS VARCHAR)  AS empi_anon,
              CAST(index_date AS DATE)    AS index_date,
              CAST(index_side AS VARCHAR) AS index_side,
              CAST(contra_side AS VARCHAR) AS contra_side,
              CAST(side_source AS VARCHAR) AS side_source,
              CAST(n_concordant_signals AS INTEGER) AS n_concordant_signals,
              CAST(age_at_index AS DOUBLE) AS age_at_index,
              CAST(age_ok AS BOOLEAN) AS age_ok,
              CAST(no_prior_contra_arthroplasty AS BOOLEAN) AS no_prior_contra_arthroplasty,
              CAST(no_infection_highspec AS BOOLEAN) AS no_infection_highspec,
              CAST(exclude_reason AS VARCHAR) AS exclude_reason
            FROM out_df
        """, cohort_dir / "index_candidates.parquet")

        _copy_parquet(con, index_final, """
            SELECT
              CAST(empi_anon AS VARCHAR)  AS empi_anon,
              CAST(index_date AS DATE)    AS index_date,
              CAST(index_side AS VARCHAR) AS index_side,
              CAST(contra_side AS VARCHAR) AS contra_side,
              CAST(side_source AS VARCHAR) AS side_source,
              CAST(n_concordant_signals AS INTEGER) AS n_concordant_signals,
              CAST(age_at_index AS DOUBLE) AS age_at_index
            FROM out_df
        """, cohort_dir / "index_final.parquet")

        # verify persisted schema/rowcounts round-trip.
        cpath = _q(cohort_dir / "index_candidates.parquet")
        fpath = _q(cohort_dir / "index_final.parquet")
        n_cand_pq = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT empi_anon) FROM read_parquet('{cpath}')").fetchone()
        n_final_pq = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT empi_anon) FROM read_parquet('{fpath}')").fetchone()
        assert n_cand_pq == (n_with_27447, n_with_27447), n_cand_pq
        assert n_final_pq == (len(index_final), len(index_final)), n_final_pq

        log.info("%s: DONE — wrote index_candidates.parquet (%d rows), index_final.parquet "
                 "(%d rows), outputs/tables/index_flow.csv", LABEL, n_cand_pq[0], n_final_pq[0])
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
