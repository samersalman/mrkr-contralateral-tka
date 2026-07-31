"""followup.py — LOCKED cohort extraction FOLLOW-UP / CENSORING scaffold module.

MRKR Contralateral TKA — Phase-1 feasibility, LOCKED EXTRACTION phase. This
module reads the SHARED CONTRACT ``derived-data/cohort/index_final.parquet``
(6,381 patients) and computes, per patient, the day-90-landmark follow-up
scaffold used by downstream event/censor resolution and Kaplan-Meier reporting.

Scope boundary (unchanged from the feasibility work):
  * Read-only on the typed Parquet inputs; NO DICOMs, NO models, NO metrics.
  * Observation / censoring legitimately use POST-index dates (last-observed
    record across cpt/icd/pain/image). These are administrative-follow-up dates,
    NOT predictors — no leakage into any at-index feature.
  * Deterministic; params come from ``config/feasibility.yaml``.

REUSE, DO NOT FORK. The verified last-observation / timeline logic is imported:
  * ``src.laterality.last_observation`` — MAX date across the four data domains
    (exactly the rule used by ``preliminary_counts.compute_observation``).
  * ``src.laterality.landmark_date`` / ``add_days`` / ``days_between`` — timeline.
  * ``src.preliminary_counts.create_views`` / ``_no_empi`` / ``_to_pydate`` — the
    verified Parquet views, the id-free-output guard, and the date coercion.

SECTION-9 vs SECTION-10 HORIZON NUANCE (documented, faithful reading):
  The protocol defines the EVENT horizon (section 9) as 5 years from the INDEX
  (index_date + 1826), and the study FOLLOWS patients FROM THE LANDMARK (section
  10), i.e. a censoring horizon of 5 years AFTER the landmark
  (landmark_date + 1826 = index_date + 1916). This module records BOTH: the
  scaffold's censoring horizon is landmark_date + 1826 (section 10), while the
  index-anchored event horizon (index_date + 1826, section 9) is documented and
  logged. The exposed :func:`resolve_followup` helper is landmark-anchored and
  uses a single landmark+1826 horizon for internal consistency of the survival
  time it emits; the downstream caller supplies resolved event dates.

Persists (to ``derived-data/cohort/``, git-ignored, may carry ``empi_anon``):
  * ``followup.parquet`` — one row per index_final patient with the scaffold.
Emits (id-free aggregate):
  * ``outputs/tables/followup_scaffold_summary.csv``.
Appends ``outputs/logs/run.log`` (prefix ``followup``).

The final-cohort median follow-up and the reverse-KM follow-up estimate are NOT
computed here (there are no resolved events yet); :func:`resolve_followup` and
:func:`reverse_km` are exposed for the downstream cohort-assembly step to call.

Run from the project root::

    python3 -m src.followup --config config/feasibility.yaml
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
    add_days,
    days_between,
    horizon_date,
    landmark_date,
    last_observation,
)
# REUSE the verified Parquet views, id-free guard, and date coercion verbatim.
from src.preliminary_counts import _no_empi, _to_pydate, create_views

MODULE = "followup"
LABEL = "LOCKED"

# Study defaults for the PURE helpers (mirror config/feasibility.yaml -> timeline).
# 5 years * 365.25 d/y = round(1826.25) = 1826 days (the study 5-year horizon).
DEFAULT_LANDMARK_DAYS = 90
DEFAULT_HORIZON_DAYS = 1826

# Censor-reason codes emitted by resolve_followup for the no-event branch.
REASON_EVENT = "event"
REASON_ADMIN_HORIZON = "admin_horizon"      # censored at landmark + horizon (5 y)
REASON_LAST_OBSERVED = "last_observed"      # censored at the last observed record

# Regression / plausibility anchors (see prompt CORRECTNESS block):
#   strict/365 FINAL was 1,664 after imaging + day0-90 + observation gates, so
#   observation-only among the coded (strict) index_final subset must be a
#   superset of that -> observed_through_90 among coded >= 1,664.
STRICT_365_FINAL_FLOOR = 1664
EXPECTED_N_INDEX_FINAL = 6381


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
# PURE helpers (no I/O, no side effects) — unit-tested in tests/test_followup. #
# --------------------------------------------------------------------------- #
def followup_scaffold(
    index_date: date | None,
    last_observed: date | None,
    landmark_days: int = DEFAULT_LANDMARK_DAYS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict:
    """Per-patient day-90-landmark follow-up scaffold (no event data required).

    All fields are functions ONLY of the index date and the last observed record
    date, so this is fully deterministic and side-effect free.

    Args:
        index_date: the earliest-27447 index date (time reference).
        last_observed: MAX record date across cpt/icd/pain/image for the patient
            (``None`` only if the patient has no records at all — impossible for
            index_final, handled defensively as zero follow-up).
        landmark_days: landmark offset from index (default 90 = day-90 origin).
        horizon_days: 5-year horizon in days (default 1826).

    Returns:
        dict with ``landmark_date``, ``observed_through_90`` (bool),
        ``censor_date_if_no_event`` (date), ``followup_days_from_landmark_if_no_event``
        (int >= 0), and ``complete_5y`` (bool).

    Definitions (protocol sections 9-10):
      * landmark_date               = index_date + landmark_days.
      * observed_through_90         = last_observed > landmark_date            (S10 gate).
      * censoring horizon           = landmark_date + horizon_days             (section 10).
      * censor_date_if_no_event     = min(landmark_date + horizon_days, last_observed).
      * followup_days_..._no_event  = days_between(landmark_date, censor_date)  (clamped >= 0;
                                      0 when last_observed <= landmark_date).
      * complete_5y                 = last_observed >= landmark_date + horizon_days.
    """
    lm = landmark_date(index_date, landmark_days)               # index + 90
    censor_horizon = add_days(lm, horizon_days)                 # landmark + 1826 (section 10)

    observed_through_90 = last_observed is not None and lm is not None and last_observed > lm
    complete_5y = (last_observed is not None and censor_horizon is not None
                   and last_observed >= censor_horizon)

    # censor_date_if_no_event = min(landmark + horizon, last_observed).
    if last_observed is None or lm is None:
        censor = lm                                             # degenerate: zero follow-up
    else:
        censor = min(censor_horizon, last_observed)

    days = days_between(lm, censor)
    if days is None or days < 0:                               # 0 when last_observed <= landmark
        days = 0

    return {
        "landmark_date": lm,
        "observed_through_90": bool(observed_through_90),
        "censor_date_if_no_event": censor,
        "followup_days_from_landmark_if_no_event": int(days),
        "complete_5y": bool(complete_5y),
    }


def resolve_followup(
    landmark_date_val: date | None,
    last_observed: date | None,
    event_date: date | None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[int, int, str]:
    """Resolve one patient's event/censor status from the landmark time origin.

    Landmark-anchored (see the SECTION-9/10 note in the module docstring): a
    single ``landmark_date + horizon_days`` horizon governs BOTH the event
    window and the administrative censor, so the emitted survival time is
    internally consistent and non-negative. The downstream caller is responsible
    for supplying a resolved contralateral-TKA ``event_date`` (only genuine
    post-landmark events; events inside the day-0..90 blanking window are handled
    upstream and must not be passed here).

    Args:
        landmark_date_val: the day-90 landmark (time origin).
        last_observed: MAX record date across cpt/icd/pain/image.
        event_date: the resolved event date, or ``None`` if no event.
        horizon_days: 5-year horizon in days (default 1826).

    Returns:
        ``(event_indicator, time_from_landmark, censor_reason)``:
          * event within [landmark, landmark+horizon]  -> ``(1, event_date - landmark, 'event')``.
          * otherwise censored at ``min(landmark+horizon, last_observed)`` ->
            ``(0, that - landmark, 'admin_horizon' | 'last_observed')`` where the
            reason is ``'admin_horizon'`` when the horizon binds
            (landmark+horizon <= last_observed) and ``'last_observed'`` when the
            patient's records run out first. Time is clamped to >= 0.

    An event strictly AFTER the horizon (or before the landmark) is NOT counted;
    the patient is administratively censored at the horizon / last observation.
    """
    admin_horizon = add_days(landmark_date_val, horizon_days)   # landmark + 1826

    # Event branch: a genuine event at/after the landmark and at/before the horizon.
    if (event_date is not None and landmark_date_val is not None
            and admin_horizon is not None
            and landmark_date_val <= event_date <= admin_horizon):
        t = days_between(landmark_date_val, event_date)
        if t is None or t < 0:
            t = 0
        return (1, int(t), REASON_EVENT)

    # Censored branch: min(landmark + horizon, last_observed).
    if last_observed is None or landmark_date_val is None:
        censor = landmark_date_val
        reason = REASON_LAST_OBSERVED
    else:
        if admin_horizon is not None and admin_horizon <= last_observed:
            censor = admin_horizon
            reason = REASON_ADMIN_HORIZON
        else:
            censor = last_observed
            reason = REASON_LAST_OBSERVED

    t = days_between(landmark_date_val, censor)
    if t is None or t < 0:                                      # 0 when last_observed <= landmark
        t = 0
    return (0, int(t), reason)


def reverse_km(times, event_indicators):
    """Reverse-Kaplan-Meier median follow-up (censoring KM) via lifelines.

    Flips the event indicator (censored observations are treated as the
    "events") so the fitted curve describes time-to-censoring; its median is the
    reverse-KM estimate of median follow-up on the observed cohort. When the
    curve does not reach 0.5 (e.g. no censored observations) the median is
    ``numpy.inf`` (lifelines convention).

    Args:
        times: per-patient follow-up times (from the landmark).
        event_indicators: per-patient event indicators (1 = event, 0 = censored).

    Returns:
        ``(median_followup, fitter)`` where ``fitter`` is the fitted
        ``KaplanMeierFitter`` on the FLIPPED indicators.
    """
    from lifelines import KaplanMeierFitter  # lazy: no import cost unless called

    t = [float(x) for x in times]
    flipped = [1 - int(e) for e in event_indicators]           # censored -> "event"
    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=flipped, label="reverse_km_followup")
    return kmf.median_survival_time_, kmf


# --------------------------------------------------------------------------- #
# Small write helpers (mirror src.index_tka; kept local to avoid pulling in    #
# the heavier index_tka import chain).                                         #
# --------------------------------------------------------------------------- #
def _q(p) -> str:
    """Single-quote-escape a path for inline SQL."""
    return str(p).replace("'", "''")


def _copy_parquet(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame,
                  select_sql: str, path: Path) -> None:
    """Register ``frame`` as ``out_df`` and COPY ``select_sql`` to a typed Parquet."""
    con.register("out_df", frame)
    con.execute(f"COPY ({select_sql}) TO '{_q(path)}' (FORMAT PARQUET)")
    con.unregister("out_df")


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LOCKED follow-up / censoring scaffold.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    tables_dir = cfg.out("tables_dir")
    cohort_dir = cfg.path(cfg["paths"]["cohort_dir"])
    cohort_dir.mkdir(parents=True, exist_ok=True)

    # ---- timeline constants (from config) ---------------------------------
    tl = cfg["timeline"]
    landmark_days = int(tl["landmark_days"])
    horizon_years = float(tl["horizon_years"])
    days_per_year = float(tl["days_per_year"])
    _ref = date(2000, 1, 1)
    horizon_days = days_between(_ref, horizon_date(_ref, horizon_years, days_per_year))
    assert landmark_days == DEFAULT_LANDMARK_DAYS == 90, landmark_days
    assert horizon_days == DEFAULT_HORIZON_DAYS == 1826, horizon_days

    log.info("START %s follow-up scaffold (landmark=day %d; horizon=%d d = 5 y; "
             "event horizon=index+%d [S9]; censor horizon=landmark+%d=index+%d [S10])",
             LABEL, landmark_days, horizon_days, horizon_days, horizon_days,
             landmark_days + horizon_days)

    tmpdir = tempfile.mkdtemp(prefix="mrkr_followup_")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmpdir}'")
    con.execute("SET preserve_insertion_order=false")

    try:
        create_views(con, cfg)

        # ---- shared contract: index_final (6,381 patients) ----------------
        fpath = _q(cohort_dir / "index_final.parquet")
        index_final = con.execute(
            f"SELECT empi_anon, index_date, side_source FROM read_parquet('{fpath}')"
        ).df()
        index_final["index_date"] = pd.to_datetime(index_final["index_date"])
        n_final = len(index_final)
        assert n_final == EXPECTED_N_INDEX_FINAL, \
            f"index_final has {n_final} rows, expected {EXPECTED_N_INDEX_FINAL}"
        assert index_final["empi_anon"].is_unique, "index_final has duplicate empi_anon"
        cohort_set = set(index_final["empi_anon"])

        # ---- last-observed date per patient = MAX across the four domains --
        # Verified pattern from preliminary_counts.compute_observation: per-table
        # MAX (a MAX is unaffected by duplicate rows -> no global de-dup needed),
        # combined via the shared last_observation() helper.
        cpt_m = con.execute("SELECT empi_anon, MAX(date_anon) m FROM cpt GROUP BY 1").df()
        icd_m = con.execute("SELECT empi_anon, MAX(date_anon) m FROM icd GROUP BY 1").df()
        pain_m = con.execute("SELECT empi_anon, MAX(date_anon) m FROM pain GROUP BY 1").df()
        img_m = con.execute("SELECT empi_anon, MAX(StudyDate_anon) m FROM image GROUP BY 1").df()
        maxes: dict[str, list] = {}
        for mdf in (cpt_m, icd_m, pain_m, img_m):
            for r in mdf.itertuples(index=False):
                e = r.empi_anon
                if e in cohort_set:
                    maxes.setdefault(e, []).append(_to_pydate(r.m))

        # ---- per-patient scaffold -----------------------------------------
        rows = []
        for r in index_final.itertuples(index=False):
            e = r.empi_anon
            idate = _to_pydate(r.index_date)
            last_obs = last_observation(maxes.get(e, []))
            # Every index_final patient has the index 27447 CPT -> a record exists.
            assert last_obs is not None, f"empi {e} has no observed record (impossible)"
            assert idate is not None and last_obs >= idate, \
                f"empi {e}: last_observed {last_obs} precedes index {idate}"

            sc = followup_scaffold(idate, last_obs, landmark_days, horizon_days)

            # Cross-check: resolve_followup with NO event reproduces the scaffold.
            ev_ind, t_res, _reason = resolve_followup(
                sc["landmark_date"], last_obs, None, horizon_days=horizon_days)
            assert ev_ind == 0
            assert t_res == sc["followup_days_from_landmark_if_no_event"], \
                f"empi {e}: resolve_followup {t_res} != scaffold {sc['followup_days_from_landmark_if_no_event']}"
            # Sanity asserts (prompt CORRECTNESS block).
            assert sc["observed_through_90"] == (last_obs > sc["landmark_date"])
            assert sc["followup_days_from_landmark_if_no_event"] >= 0

            rows.append({
                "empi_anon": e,
                "side_source": r.side_source,
                "last_observed": last_obs,
                "landmark_date": sc["landmark_date"],
                "observed_through_90": sc["observed_through_90"],
                "censor_date_if_no_event": sc["censor_date_if_no_event"],
                "followup_days_from_landmark_if_no_event":
                    sc["followup_days_from_landmark_if_no_event"],
                "complete_5y": sc["complete_5y"],
            })
        fu = pd.DataFrame(rows)
        assert len(fu) == n_final and fu["empi_anon"].is_unique

        # ---- aggregate summary over index_final and the strict (coded) subset
        strict_mask = fu["side_source"] == "coded"     # coded == strict earliest-single-side
        n_obs_all = int(fu["observed_through_90"].sum())
        n_obs_strict = int(fu.loc[strict_mask, "observed_through_90"].sum())
        n_strict = int(strict_mask.sum())

        # Plausibility gate: observation-only among coded must exceed the strict/365
        # FINAL count (1,664), which additionally applied imaging + day0-90 gates.
        assert n_obs_strict >= STRICT_365_FINAL_FLOOR, \
            f"observed_through_90 among coded {n_obs_strict} < strict/365 floor {STRICT_365_FINAL_FLOOR}"
        assert n_obs_all >= n_obs_strict, "all-cohort observed < strict subset observed"

        def _summ(sub: pd.DataFrame, cohort_name: str) -> dict:
            n = len(sub)
            fd = sub["followup_days_from_landmark_if_no_event"]
            return {
                "cohort": cohort_name,
                "n_patients": n,
                "n_observed_through_90": int(sub["observed_through_90"].sum()),
                "pct_observed_through_90": round(100.0 * int(sub["observed_through_90"].sum()) / n, 2) if n else 0.0,
                "followup_days_median": round(float(fd.median()), 1),
                "followup_days_q1": round(float(fd.quantile(0.25)), 1),
                "followup_days_q3": round(float(fd.quantile(0.75)), 1),
                "n_complete_5y": int(sub["complete_5y"].sum()),
                "pct_complete_5y": round(100.0 * int(sub["complete_5y"].sum()) / n, 2) if n else 0.0,
            }

        summary = pd.DataFrame([
            _summ(fu, "index_final"),
            _summ(fu[strict_mask], "strict_coded_subset"),
        ])
        # NOTE column: final-cohort median follow-up + reverse-KM are downstream.
        summary["note"] = ("scaffold no-event follow-up over all rows in the cohort; "
                           "final-cohort median follow-up + reverse-KM computed downstream "
                           "on the assembled cohort")
        _no_empi(summary, "followup_scaffold_summary")

        med_all = summary.loc[0, "followup_days_median"]
        q1_all = summary.loc[0, "followup_days_q1"]
        q3_all = summary.loc[0, "followup_days_q3"]
        n_c5_all = int(fu["complete_5y"].sum())
        log.info("%s: index_final n=%d observed_through_90=%d (%.1f%%); "
                 "no-event follow-up median=%.0f d [IQR %.0f-%.0f]; complete_5y=%d",
                 LABEL, n_final, n_obs_all, 100.0 * n_obs_all / n_final,
                 med_all, q1_all, q3_all, n_c5_all)
        log.info("%s: strict(coded) subset n=%d observed_through_90=%d (floor %d, PASS); "
                 "complete_5y=%d", LABEL, n_strict, n_obs_strict, STRICT_365_FINAL_FLOOR,
                 int(fu.loc[strict_mask, "complete_5y"].sum()))
        log.info("%s: reverse_km + final-cohort median follow-up are exposed helpers, "
                 "computed downstream on the assembled cohort (events not resolved here)", LABEL)

        # ---- WRITE outputs -------------------------------------------------
        summary.to_csv(tables_dir / "followup_scaffold_summary.csv", index=False)

        # Persist scaffold (drops side_source; that lives in index_final).
        fu_out = fu.drop(columns=["side_source"]).copy()
        for c in ("last_observed", "landmark_date", "censor_date_if_no_event"):
            fu_out[c] = pd.to_datetime(fu_out[c])
        _copy_parquet(con, fu_out, """
            SELECT
              CAST(empi_anon AS VARCHAR) AS empi_anon,
              CAST(last_observed AS DATE) AS last_observed,
              CAST(landmark_date AS DATE) AS landmark_date,
              CAST(observed_through_90 AS BOOLEAN) AS observed_through_90,
              CAST(censor_date_if_no_event AS DATE) AS censor_date_if_no_event,
              CAST(followup_days_from_landmark_if_no_event AS INTEGER)
                AS followup_days_from_landmark_if_no_event,
              CAST(complete_5y AS BOOLEAN) AS complete_5y
            FROM out_df
        """, cohort_dir / "followup.parquet")

        # verify persisted schema / rowcounts round-trip.
        ppath = _q(cohort_dir / "followup.parquet")
        n_pq = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT empi_anon), "
            f"       SUM(CASE WHEN followup_days_from_landmark_if_no_event < 0 THEN 1 ELSE 0 END) "
            f"FROM read_parquet('{ppath}')").fetchone()
        assert n_pq[0] == n_final and n_pq[1] == n_final, n_pq
        assert n_pq[2] == 0, f"{n_pq[2]} negative follow-up days persisted"

        log.info("%s: DONE — wrote followup.parquet (%d rows), "
                 "outputs/tables/followup_scaffold_summary.csv", LABEL, n_pq[0])
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
