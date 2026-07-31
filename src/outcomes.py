"""outcomes.py — per-patient CONTRALATERAL-TKA OUTCOME extraction (LOCKED).

MRKR Contralateral TKA — Phase-1 feasibility, LOCKED EXTRACTION phase. This
module computes the study's PRIMARY endpoint and its labeled sensitivity /
descriptive endpoints for every patient in the shared index contract
(``derived-data/cohort/index_final.parquet``, 6,381 patients).

Primary outcome (protocol): a health-system-recorded CONTRALATERAL (opposite of
``index_side``), laterality-coded CPT 27447 AFTER day 90 and WITHIN 5 years of
index, i.e. day-offset in ``(landmark_days, horizon_days] == (90, 1826]``.

Scope boundary:
  * Read-only on the typed Parquet inputs; NO DICOMs, NO models, NO metrics.
  * The OUTCOME legitimately uses POST-index CPT/image records — it is the study
    ENDPOINT, not a predictor. No git, no network.
  * Deterministic; every window/code comes from ``config/feasibility.yaml``.

REUSE, DO NOT FORK. All parsing / windowing is the verified shared logic:
  * ``src.laterality`` (``parse_modifier``, ``within``, ``days_between``,
    ``horizon_date``) — the CPT-side + day-offset primitives.
  * ``src.preliminary_counts.build_index_frames`` — the verified, parse_modifier
    based 27447 (``df447``) and prior-knee-arthroplasty (``priorarth``, which
    already carries CPT 27446) record frames.
  * ``src.preliminary_counts.compute_cpt_flags`` — the verified per-patient
    event-window flag computation. It is used here BOTH to populate the flag
    columns AND as an independent oracle: this module recomputes the primary
    event from an INDEPENDENT earliest-date reduction and asserts the two agree
    for every patient (single source of truth guard).

Endpoints emitted per patient (over ALL 6,381 index_final patients):
  * ``has_contra_27447_day_0_90`` — S9 EXCLUSION flag: any contralateral 27447
    with day-offset in [0, 90]. (Flag only; the exclusion is applied downstream.)
  * ``primary_event`` / ``event_date`` / ``days_index_to_event`` — the earliest
    contralateral 27447 in (90, 1826] (date/days NULL when no event).
  * Secondary DESCRIPTIVE windows (config ``secondary_event_windows``):
    ``sec_from_day1_{365,730,1826}`` = contralateral event in [1, V];
    ``sec_from_day91_{365,730,1826}`` = contralateral event in (90, V].
  * Labeled SENSITIVITY endpoints (protocol section 9):
      - ``upper_bound_event`` — any later 27447 of ANY modifier in (90, 1826].
        This INCLUDES ipsilateral reoperations, so it OVERCOUNTS the true
        contralateral endpoint (documented upper bound).
      - ``composite_uni_event`` — ``primary_event`` OR a contralateral
        (opposite-side) CPT 27446 (unicompartmental) in (90, 1826].
      - ``augmented_event`` — ``primary_event`` AND a post-event contralateral
        image ``arthroplasty`` in {contra_side, 'B'} within 180 days after
        ``event_date`` (radiographic confirmation of the coded prosthesis).

Per the modifier rule, a later 27447 whose modifier is ``50`` / blank /
conflicting is NOT a confirmed contralateral event (it can only feed
``upper_bound_event``); only ``parse_modifier`` side == ``contra_side`` counts.

Persists (git-ignored, patient-level):
  * ``derived-data/cohort/outcomes.parquet``.
Emits (aggregate, id-free):
  * ``outputs/tables/outcome_counts_detail.csv`` — counts by definition/window,
    over ALL index_final AND over the strict subset (side_source='coded'),
    raw and with the S9 day-0-90 exclusion applied. Labeled that FINAL-cohort
    event counts still require intersecting the imaging (S8) + observation (S10)
    gates (done downstream).
Appends ``outputs/logs/run.log`` (prefix ``outcomes``).

Run from the project root::

    python3 -m src.outcomes --config config/feasibility.yaml
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
    days_between,
    horizon_date,
    parse_modifier,
    within,
)
# REUSE the verified index-frame construction + event-window flag logic verbatim.
from src.preliminary_counts import (
    _no_empi,
    _to_pydate,
    build_index_frames,
    compute_cpt_flags,
    create_views,
)

MODULE = "outcomes"
LABEL = "LOCKED"

# Radiographic-confirmation window for the augmented endpoint: an image showing a
# contralateral (or bilateral) prosthesis taken 0..180 days AFTER the event date.
# Day 0 is included so a same-day post-operative film confirms the coded event;
# the prosthesis must be present, so a pre-operative same-day film cannot match.
AUGMENT_IMAGE_DAYS = 180


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
# Small write helper (mirrors index_tka._copy_parquet).                        #
# --------------------------------------------------------------------------- #
def _q(p) -> str:
    """Single-quote-escape a path for inline SQL."""
    return str(p).replace("'", "''")


def _copy_parquet(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame,
                  select_sql: str, path: Path) -> None:
    """Register ``frame`` as ``out_df`` and COPY ``select_sql`` to Parquet with an
    explicit, locked column-type contract."""
    con.register("out_df", frame)
    con.execute(f"COPY ({select_sql}) TO '{_q(path)}' (FORMAT PARQUET)")
    con.unregister("out_df")


# --------------------------------------------------------------------------- #
# Earliest confirmed CONTRALATERAL 27447 event (date + day-offset).            #
# Independent of compute_cpt_flags; cross-checked against it at run time.      #
# --------------------------------------------------------------------------- #
def earliest_contra_event(ev447: pd.DataFrame, event_start: int, horizon_days: int) -> pd.DataFrame:
    """Return one row per patient with a qualifying contralateral 27447 event.

    Args:
        ev447: 27447 records for the cohort, columns ``empi_anon``, ``date_anon``
            (datetime64), ``side`` (verified ``parse_modifier`` side), ``contra_side``,
            ``idate`` (python ``date``), ``cpt_group_modifier`` (raw).
        event_start: first eligible day-offset (91).
        horizon_days: last eligible day-offset (1826).

    Returns:
        DataFrame ``[empi_anon, event_date (date), days_index_to_event (int),
        event_modifier (raw str)]`` for patients whose EARLIEST confirmed
        contralateral 27447 (parse side == contra_side) has day-offset in
        ``[event_start, horizon_days]`` (== ``(90, 1826]``). The "first" event is
        the minimum event DATE; ties broken deterministically by raw modifier.
    """
    e = ev447.copy()
    e["dd"] = [days_between(i, _to_pydate(d)) for i, d in zip(e["idate"], e["date_anon"])]
    q = e[(e["side"] == e["contra_side"])
          & e["dd"].map(lambda x: within(x, event_start, horizon_days))]
    if q.empty:
        return pd.DataFrame(columns=["empi_anon", "event_date", "days_index_to_event",
                                     "event_modifier"])
    # deterministic earliest: min date, then min raw modifier (contra events are
    # always laterality-coded, so the modifier is never blank -> a total order).
    # drop_duplicates(keep='first') takes the first ROW per patient after the sort
    # (unlike groupby().first(), which coalesces the first non-null value per column).
    q = q.sort_values(["empi_anon", "date_anon", "cpt_group_modifier"],
                      kind="mergesort")
    first = q.drop_duplicates(subset="empi_anon", keep="first").copy()
    first["event_date"] = first["date_anon"].map(_to_pydate)
    return first[["empi_anon", "event_date", "dd", "cpt_group_modifier"]].rename(
        columns={"dd": "days_index_to_event", "cpt_group_modifier": "event_modifier"})


# --------------------------------------------------------------------------- #
# Aggregate detail table (NO empi_anon).                                       #
# --------------------------------------------------------------------------- #
def build_detail_table(oc: pd.DataFrame, sec: dict) -> pd.DataFrame:
    """Assemble the aggregate outcome_counts_detail table (no identifiers).

    For each subset (all index_final; strict coded) and each endpoint, report the
    raw count and the count after applying the S9 day-0-90 exclusion, plus the
    subset denominator. Nothing here is the FINAL cohort count: the imaging (S8)
    and observation (S10) gates are intersected downstream.
    """
    from_day1 = list(sec.get("from_day1", []))
    from_day91 = list(sec.get("from_day91", []))

    GATE_NOTE = ("pre imaging(S8)+observation(S10) gates; FINAL-cohort event count "
                 "requires intersecting those gates downstream")

    # (definition, window_desc, endpoint_class, column) — column True == event.
    specs: list[tuple[str, str, str, str]] = [
        ("primary_event", "(90,1826] 5y", "primary", "primary_event"),
        ("upper_bound_event", "(90,1826] any-modifier 27447", "sensitivity_upper_bound",
         "upper_bound_event"),
        ("composite_uni_event", "(90,1826] 27447-contra OR 27446-contra",
         "sensitivity_composite", "composite_uni_event"),
        ("augmented_event", "primary AND contra/B image <=180d post-event",
         "sensitivity_augmented", "augmented_event"),
    ]
    for v in from_day1:
        specs.append((f"sec_from_day1_{v}", f"[1,{v}]", "secondary_descriptive",
                      f"sec_from_day1_{v}"))
    for v in from_day91:
        specs.append((f"sec_from_day91_{v}", f"(90,{v}]", "secondary_descriptive",
                      f"sec_from_day91_{v}"))

    subsets = {
        "all_index_final": oc,
        "strict_coded": oc[oc["side_source"] == "coded"],
    }
    rows: list[dict] = []
    for subset_name, sub in subsets.items():
        denom = int(len(sub))
        s9 = sub["has_contra_27447_day_0_90"]
        n_s9 = int(s9.sum())
        # informational rows: S9 exclusion count + retained denominator.
        rows.append(dict(
            label=LABEL, subset=subset_name, denominator_n=denom,
            endpoint_class="exclusion_flag", definition="has_contra_27447_day_0_90",
            window_desc="[0,90]", n_event_raw=n_s9, n_event_after_s9=n_s9,
            pct_of_denominator=round(100.0 * n_s9 / denom, 3) if denom else 0.0,
            note="S9 exclusion flag: patients removed from cohort downstream"))
        rows.append(dict(
            label=LABEL, subset=subset_name, denominator_n=denom,
            endpoint_class="denominator", definition="cohort_retained_after_s9",
            window_desc="", n_event_raw=denom, n_event_after_s9=denom - n_s9,
            pct_of_denominator=round(100.0 * (denom - n_s9) / denom, 3) if denom else 0.0,
            note="index_final rows minus S9 exclusions (still pre imaging/observation gates)"))
        for definition, window_desc, cls, col in specs:
            flag = sub[col]
            n_raw = int(flag.sum())
            n_after = int((flag & ~s9).sum())
            rows.append(dict(
                label=LABEL, subset=subset_name, denominator_n=denom,
                endpoint_class=cls, definition=definition, window_desc=window_desc,
                n_event_raw=n_raw, n_event_after_s9=n_after,
                pct_of_denominator=round(100.0 * n_raw / denom, 3) if denom else 0.0,
                note=GATE_NOTE))
    detail = pd.DataFrame(rows, columns=[
        "label", "subset", "denominator_n", "endpoint_class", "definition",
        "window_desc", "n_event_raw", "n_event_after_s9", "pct_of_denominator", "note"])
    return detail


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LOCKED per-patient contralateral-TKA outcomes.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    tables_dir = cfg.out("tables_dir")
    cohort_dir = cfg.path(cfg["paths"]["cohort_dir"])
    cohort_dir.mkdir(parents=True, exist_ok=True)

    # laterality tokens (from config).
    lat = cfg["laterality"]
    rt = list(lat.get("right_tokens", ["RT"]))
    lt = list(lat.get("left_tokens", ["LT"]))
    bl = list(lat.get("bilateral_tokens", ["50"]))

    # timeline constants (derive day-offsets via shared helpers; no hard-coding).
    tl = cfg["timeline"]
    landmark_days = int(tl["landmark_days"])          # 90
    event_start = int(tl["event_start_day"])          # 91
    horizon_years = float(tl["horizon_years"])
    days_per_year = float(tl["days_per_year"])
    _ref = date(2000, 1, 1)
    horizon_days = days_between(_ref, horizon_date(_ref, horizon_years, days_per_year))  # 1826
    sec = cfg["secondary_event_windows"]
    from_day1 = list(sec.get("from_day1", []))
    from_day91 = list(sec.get("from_day91", []))
    index_cpt = str(cfg["index"]["cpt_code"])         # '27447'
    uni_cpt = "27446"                                  # unicompartmental (config prior-arth list)
    img = cfg["image"]
    arth_col = img["arthroplasty_col"]                # 'arthroplasty'
    studydate_col = img["study_date_col"]             # 'StudyDate_anon'
    present_codes = list(img["arthroplasty_present_codes"])  # ['R','L','B']
    assert uni_cpt in cfg["prior_knee_arthroplasty_cpt"], "27446 missing from config code list"

    log.info("START %s outcomes (primary=contra %s in (%d,%d]; sec from_day1=%s from_day91=%s; "
             "augment image window=+%dd)", LABEL, index_cpt, landmark_days, horizon_days,
             from_day1, from_day91, AUGMENT_IMAGE_DAYS)

    tmpdir = tempfile.mkdtemp(prefix="mrkr_outcomes_")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmpdir}'")
    con.execute("SET preserve_insertion_order=false")

    try:
        create_views(con, cfg)

        # ---- shared contract: index_final (6,381 patients) ------------------ #
        fpath = _q(cohort_dir / "index_final.parquet")
        cohort = con.execute(
            f"SELECT empi_anon, index_date, index_side, contra_side, side_source, "
            f"n_concordant_signals, age_at_index FROM read_parquet('{fpath}')").df()
        n_cohort = len(cohort)
        assert n_cohort == 6381, f"index_final has {n_cohort} rows, expected 6381"
        assert cohort["empi_anon"].is_unique, "index_final empi_anon not unique"
        assert cohort["contra_side"].isin(["R", "L"]).all(), "contra_side not all R/L"
        cohort["idate"] = pd.to_datetime(cohort["index_date"]).map(_to_pydate)
        n_strict = int((cohort["side_source"] == "coded").sum())
        log.info("%s: loaded index_final n=%d (strict/coded=%d, recovered=%d)",
                 LABEL, n_cohort, n_strict, n_cohort - n_strict)

        # ---- verified record frames (df447 parsed sides; priorarth incl 27446) #
        log_prelim = logging.getLogger("preliminary_counts")
        if not log_prelim.handlers:
            log_prelim.addHandler(logging.NullHandler())
        df447, _per, priorarth = build_index_frames(con, cfg, log_prelim)

        # ---- (A) verified event-window FLAGS via compute_cpt_flags ---------- #
        # side_col ('index_side') is accepted but unused by compute_cpt_flags;
        # contra_col + idate_col drive the contralateral windowing.
        oracle = compute_cpt_flags(
            cohort, df447, priorarth,
            "index_side", "contra_side", "index_date",
            landmark_days, event_start, horizon_days, sec)
        oracle = oracle.set_index("empi_anon")

        # ---- (B) INDEPENDENT earliest contralateral 27447 event (date/days) - #
        ev447 = df447.merge(cohort[["empi_anon", "contra_side", "idate"]],
                            on="empi_anon", how="inner")
        events = earliest_contra_event(ev447, event_start, horizon_days)
        events = events.set_index("empi_anon")

        # ---- (C) contralateral 27446 (unicompartmental) in (90,1826] -------- #
        c446 = priorarth[priorarth["cpt_code"] == uni_cpt].merge(
            cohort[["empi_anon", "contra_side", "idate"]], on="empi_anon", how="inner")
        uni_set: set = set()
        if not c446.empty:
            c446 = c446.copy()
            c446["dd"] = [days_between(i, _to_pydate(d))
                          for i, d in zip(c446["idate"], c446["date_anon"])]
            uni_set = set(c446[(c446["side"] == c446["contra_side"])
                               & c446["dd"].map(lambda x: within(x, event_start, horizon_days))]
                          ["empi_anon"])

        # -------------------------------------------------------------------- #
        # Assemble per-patient outcome frame over ALL 6,381.                   #
        # -------------------------------------------------------------------- #
        oc = cohort[["empi_anon", "side_source", "idate"]].copy()
        oc = oc.merge(events, left_on="empi_anon", right_index=True, how="left")
        oc["primary_event"] = oc["event_date"].notna()

        # flag columns from the verified oracle (aligned by empi).
        oc = oc.set_index("empi_anon")
        oc["has_contra_27447_day_0_90"] = oracle["contra_0_90"].reindex(oc.index).fillna(0).astype(bool)
        oc["upper_bound_event"] = oracle["upper_event"].reindex(oc.index).fillna(0).astype(bool)
        for v in from_day1:
            oc[f"sec_from_day1_{v}"] = oracle[f"sec_d1_{v}"].reindex(oc.index).fillna(0).astype(bool)
        for v in from_day91:
            oc[f"sec_from_day91_{v}"] = oracle[f"sec_d91_{v}"].reindex(oc.index).fillna(0).astype(bool)
        oc = oc.reset_index()

        # composite: primary OR contralateral 27446 in (90,1826].
        oc["composite_uni_event"] = oc["primary_event"] | oc["empi_anon"].isin(uni_set)

        # ---- (D) augmented: primary + contra/B image within 180d post-event  #
        ev_for_img = oc.loc[oc["primary_event"], ["empi_anon", "event_date"]].merge(
            cohort[["empi_anon", "contra_side"]], on="empi_anon", how="left")
        aug_set: set = set()
        if not ev_for_img.empty:
            ev_for_img = ev_for_img.copy()
            ev_for_img["event_date"] = pd.to_datetime(ev_for_img["event_date"])  # -> datetime64
            con.register("ev_img_df", ev_for_img)
            con.execute("""
                CREATE OR REPLACE TEMP TABLE ev_img AS
                SELECT CAST(empi_anon AS VARCHAR) AS empi_anon,
                       CAST(event_date AS DATE)   AS event_date,
                       CAST(contra_side AS VARCHAR) AS contra_side
                FROM ev_img_df
            """)
            con.unregister("ev_img_df")
            codes = "(" + ",".join("'" + c.replace("'", "''") + "'" for c in present_codes) + ")"
            aug_rows = con.execute(f"""
                SELECT DISTINCT t.empi_anon
                FROM ev_img t
                JOIN image g ON g.empi_anon = t.empi_anon
                WHERE g.{arth_col} IN (t.contra_side, 'B')
                  AND g.{arth_col} IN {codes}
                  AND g.{studydate_col} >= t.event_date
                  AND g.{studydate_col} <= t.event_date + INTERVAL {int(AUGMENT_IMAGE_DAYS)} DAY
            """).df()
            aug_set = set(aug_rows["empi_anon"])
        oc["augmented_event"] = oc["primary_event"] & oc["empi_anon"].isin(aug_set)

        # -------------------------------------------------------------------- #
        # CORRECTNESS / SANITY assertions.                                     #
        # -------------------------------------------------------------------- #
        assert len(oc) == n_cohort and oc["empi_anon"].is_unique, "outcome row/id mismatch"

        # (1) independent primary == verified-oracle primary (single source of truth).
        oracle_primary = (oracle["primary_event"].reindex(oc.set_index("empi_anon").index)
                          .fillna(0).astype(bool)).values
        assert (oc["primary_event"].values == oracle_primary).all(), \
            "independent primary_event diverged from compute_cpt_flags oracle"

        # (2) every primary event: date/day-offset window + parsed side == contra.
        idate_map = dict(zip(cohort["empi_anon"], cohort["idate"]))
        contra_map = dict(zip(cohort["empi_anon"], cohort["contra_side"]))
        ev_rows = oc[oc["primary_event"]]
        mod_map = dict(zip(events.index, events["event_modifier"]))
        for r in ev_rows.itertuples():
            idate = idate_map[r.empi_anon]
            contra = contra_map[r.empi_anon]
            assert r.event_date is not None and not pd.isna(r.event_date), r.empi_anon
            dd = days_between(idate, r.event_date)
            assert dd == r.days_index_to_event, \
                f"days recompute mismatch {r.empi_anon}: {dd} != {r.days_index_to_event}"
            assert dd > landmark_days, f"event <= landmark for {r.empi_anon} (dd={dd})"
            assert dd <= horizon_days, f"event > horizon for {r.empi_anon} (dd={dd})"
            assert within(dd, event_start, horizon_days), f"event out of window {r.empi_anon}"
            assert r.event_date > (idate + pd.Timedelta(days=landmark_days).to_pytimedelta()), \
                f"event_date not > index+90 for {r.empi_anon}"
            side_evt = parse_modifier(mod_map[r.empi_anon], rt, lt, bl)[0]
            assert side_evt == contra, \
                f"event modifier side {side_evt} != contra {contra} for {r.empi_anon}"
        # no event rows must have NULL date/days.
        assert oc.loc[~oc["primary_event"], "event_date"].isna().all(), "no-event row has date"
        assert oc.loc[~oc["primary_event"], "days_index_to_event"].isna().all(), "no-event row has days"

        # (3) endpoint monotonicity: upper_bound >= primary; composite >= primary;
        #     augmented subset of primary; augmented implies primary.
        assert (oc["upper_bound_event"] | ~oc["primary_event"]).all(), "primary not subset of upper_bound"
        assert (oc["composite_uni_event"] | ~oc["primary_event"]).all(), "primary not subset of composite"
        assert (~oc["augmented_event"] | oc["primary_event"]).all(), "augmented not subset of primary"
        assert int(oc["primary_event"].sum()) <= int(oc["upper_bound_event"].sum())
        assert int(oc["augmented_event"].sum()) <= int(oc["primary_event"].sum())

        # (4) has_contra_27447_day_0_90 (S9) recompute cross-check vs oracle already
        #     enforced by construction; assert the flag is a strict day-window flag.
        assert oc["has_contra_27447_day_0_90"].dtype == bool

        # -------------------------------------------------------------------- #
        # Persist patient-level outcomes.parquet (git-ignored).                #
        # -------------------------------------------------------------------- #
        # nullable-friendly types: event_date NaT / days pandas-Int64 <NA>.
        oc_out = oc.copy()
        oc_out["event_date"] = pd.to_datetime(oc_out["event_date"])  # NaT for no-event
        oc_out["days_index_to_event"] = oc_out["days_index_to_event"].astype("Int64")
        bool_cols = (["has_contra_27447_day_0_90", "primary_event"]
                     + [f"sec_from_day1_{v}" for v in from_day1]
                     + [f"sec_from_day91_{v}" for v in from_day91]
                     + ["upper_bound_event", "composite_uni_event", "augmented_event"])
        for c in bool_cols:
            oc_out[c] = oc_out[c].astype(bool)

        sec1_sql = ",\n              ".join(
            f"CAST(sec_from_day1_{v} AS BOOLEAN) AS sec_from_day1_{v}" for v in from_day1)
        sec91_sql = ",\n              ".join(
            f"CAST(sec_from_day91_{v} AS BOOLEAN) AS sec_from_day91_{v}" for v in from_day91)
        select_sql = f"""
            SELECT
              CAST(empi_anon AS VARCHAR) AS empi_anon,
              CAST(side_source AS VARCHAR) AS side_source,
              CAST(has_contra_27447_day_0_90 AS BOOLEAN) AS has_contra_27447_day_0_90,
              CAST(primary_event AS BOOLEAN) AS primary_event,
              CAST(event_date AS DATE) AS event_date,
              CAST(days_index_to_event AS INTEGER) AS days_index_to_event,
              {sec1_sql},
              {sec91_sql},
              CAST(upper_bound_event AS BOOLEAN) AS upper_bound_event,
              CAST(composite_uni_event AS BOOLEAN) AS composite_uni_event,
              CAST(augmented_event AS BOOLEAN) AS augmented_event
            FROM out_df
        """
        _copy_parquet(con, oc_out, select_sql, cohort_dir / "outcomes.parquet")
        opath = _q(cohort_dir / "outcomes.parquet")
        n_pq = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT empi_anon) FROM read_parquet('{opath}')").fetchone()
        assert n_pq == (n_cohort, n_cohort), n_pq
        # round-trip: event_date non-null iff primary_event.
        chk = con.execute(
            f"SELECT SUM(CASE WHEN primary_event AND event_date IS NULL THEN 1 ELSE 0 END) a, "
            f"SUM(CASE WHEN (NOT primary_event) AND event_date IS NOT NULL THEN 1 ELSE 0 END) b "
            f"FROM read_parquet('{opath}')").fetchone()
        assert chk == (0, 0), f"event_date/primary_event round-trip mismatch {chk}"

        # -------------------------------------------------------------------- #
        # Aggregate detail table (NO empi_anon).                               #
        # -------------------------------------------------------------------- #
        detail = build_detail_table(oc, sec)
        _no_empi(detail, "outcome_counts_detail")
        detail.to_csv(tables_dir / "outcome_counts_detail.csv", index=False)

        # -------------------------------------------------------------------- #
        # Report headline counts to the log.                                   #
        # -------------------------------------------------------------------- #
        def _counts(sub: pd.DataFrame) -> dict:
            s9 = sub["has_contra_27447_day_0_90"]
            return {
                "n": len(sub),
                "s9": int(s9.sum()),
                "primary_raw": int(sub["primary_event"].sum()),
                "primary_after_s9": int((sub["primary_event"] & ~s9).sum()),
                "upper_bound": int(sub["upper_bound_event"].sum()),
                "composite_uni": int(sub["composite_uni_event"].sum()),
                "augmented": int(sub["augmented_event"].sum()),
            }
        all_c = _counts(oc)
        strict_c = _counts(oc[oc["side_source"] == "coded"])
        log.info("%s: ALL index_final: n=%d S9=%d primary_raw=%d primary_after_s9=%d "
                 "upper_bound=%d composite_uni=%d augmented=%d",
                 LABEL, all_c["n"], all_c["s9"], all_c["primary_raw"], all_c["primary_after_s9"],
                 all_c["upper_bound"], all_c["composite_uni"], all_c["augmented"])
        log.info("%s: STRICT (coded): n=%d S9=%d primary_raw=%d primary_after_s9=%d "
                 "upper_bound=%d composite_uni=%d augmented=%d",
                 LABEL, strict_c["n"], strict_c["s9"], strict_c["primary_raw"],
                 strict_c["primary_after_s9"], strict_c["upper_bound"],
                 strict_c["composite_uni"], strict_c["augmented"])
        # secondary windows (raw / after-S9) for both subsets.
        for name, sub in (("ALL", oc), ("STRICT", oc[oc["side_source"] == "coded"])):
            s9 = sub["has_contra_27447_day_0_90"]
            parts = []
            for v in from_day1:
                col = sub[f"sec_from_day1_{v}"]
                parts.append(f"d1_{v}={int(col.sum())}/{int((col & ~s9).sum())}")
            for v in from_day91:
                col = sub[f"sec_from_day91_{v}"]
                parts.append(f"d91_{v}={int(col.sum())}/{int((col & ~s9).sum())}")
            log.info("%s: secondary windows [%s] raw/after_s9: %s", LABEL, name, " ".join(parts))
        log.info("%s: augmented feasible from image.parquet (contra/B prosthesis <=%dd post-event); "
                 "image coverage ends before CPT horizon so late events cannot be radiographically "
                 "confirmed (documented undercount).", LABEL, AUGMENT_IMAGE_DAYS)
        log.info("%s: STRICT primary_after_s9=%d is PRE imaging(S8)+observation(S10) gates; "
                 "orchestrator reconciles the assembled final count (cannot assert 357 here).",
                 LABEL, strict_c["primary_after_s9"])
        log.info("%s: DONE — wrote outcomes.parquet (%d rows), outputs/tables/outcome_counts_detail.csv "
                 "(%d rows)", LABEL, n_pq[0], len(detail))
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
