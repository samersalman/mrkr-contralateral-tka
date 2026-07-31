"""io_duckdb.py — faithful, typed CSV -> Parquet conversion (MRKR Phase-1 feasibility).

One pass over each of the 7 source CSVs named in config/feasibility.yaml, writing
ONE typed Parquet per table to ``derived-data/source-parquet/<key>.parquet`` using
DuckDB (the ~1.79 GB ICD file is streamed, never loaded into pandas).

This module performs FAITHFUL typed conversion only — it does NOT derive cohort
logic, normalise codes, or trim strings. Raw string values are preserved exactly
for every VARCHAR column (notably ``cpt_group_modifier``, which a later module
parses). Numeric / date / flag columns are cast per the brief's typing rules with
``TRY_CAST`` / ``try_strptime`` so a bad value becomes NULL rather than dropping a
row; row counts are then reconciled against the CSV and the config ``ref_rows``.

Run from the project root::

    python3 -m src.io_duckdb --config config/feasibility.yaml [--force]

Exit code is non-zero if any table fails to reconcile (CSV rows == Parquet rows ==
ref_rows). Progress and a final summary are appended to outputs/logs/run.log.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from src.config import Config, ensure_dirs, load_config

MODULE = "io_duckdb"

# Canonical output order (matches config / ground-truth listing).
TABLE_ORDER = [
    "demographics", "cpt", "icd", "image", "pain",
    "cpt_dictionary", "icd_dictionary",
]
# Processing order = smallest first, so a coding error fails fast on a tiny file
# instead of after streaming the 1.79 GB ICD table.
PROCESS_ORDER = [
    "cpt_dictionary", "icd_dictionary", "demographics", "pain", "image", "cpt", "icd",
]

# The 9 curated 0/1 comorbidity flags carried on the ICD table (verified present).
ICD_FLAGS = [
    "autoimmune", "diabetes", "hypertension", "joint_infection",
    "knee_osteoarthritis", "knee_osteomyelitis", "obesity",
    "nicotine_use", "trauma_lower_extremity",
]

CHECKSUM_COLUMNS = [
    "table", "csv_rows", "parquet_rows", "ref_rows", "match",
    "n_columns", "distinct_patients", "n_unparseable_dates", "parquet_bytes",
]


# --------------------------------------------------------------------------- #
# Logging: append to run.log (module | timestamp | level | message) + stdout.  #
# --------------------------------------------------------------------------- #
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(MODULE)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a")          # APPEND (never truncate)
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
# Column-expression helpers.  Every expression reads an all-VARCHAR source     #
# column and casts explicitly; TRY_CAST / try_strptime yield NULL (not a       #
# dropped row) on failure, preserving row counts.                              #
# --------------------------------------------------------------------------- #
def _q(col: str) -> str:
    """Double-quote a SQL identifier."""
    return '"' + col.replace('"', '""') + '"'


def keep(col: str):
    """VARCHAR preserved EXACTLY as the raw string (no trim / no normalisation)."""
    return (col, _q(col))


def vc(col: str):
    """empi_anon -> explicit VARCHAR (consistent key type across all tables)."""
    return (col, f"CAST({_q(col)} AS VARCHAR)")


def dbl(col: str):
    """Ages / KLG inference -> nullable DOUBLE."""
    return (col, f"TRY_CAST({_q(col)} AS DOUBLE)")


def tiny(col: str):
    """Curated 0/1 ICD flag -> nullable TINYINT."""
    return (col, f"TRY_CAST({_q(col)} AS TINYINT)")


def int01(col: str):
    """'0.0'/'1.0'/'0'/'1' -> nullable INTEGER 0/1 (cast via DOUBLE)."""
    return (col, f"CAST(TRY_CAST({_q(col)} AS DOUBLE) AS INTEGER)")


def dt(col: str, fmt: str):
    """Date string -> DATE via try_strptime(fmt); NULL on parse failure."""
    return (col, f"try_strptime({_q(col)}, '{fmt}')::DATE")


def table_spec(key: str, fmt: str):
    """Ordered list of (output_column, sql_expression) for one source table."""
    if key == "demographics":
        return [vc("empi_anon"), keep("sex"), keep("race"), keep("ethnicity")]
    if key == "cpt":
        return [vc("empi_anon"), keep("cpt_code"), keep("cpt_group_modifier"),
                dt("date_anon", fmt), dbl("age_at_procedure")]
    if key == "icd":
        return ([vc("empi_anon"), keep("ICD9"), keep("ICD10"), dt("date_anon", fmt),
                 dbl("age_at_dx"), keep("DX_LINE"), keep("DX_ICD_SCOPE")]
                + [tiny(f) for f in ICD_FLAGS])
    if key == "image":
        return [vc("empi_anon"),
                keep("StudyInstanceUID_anon"), keep("SeriesInstanceUID_anon"),
                keep("SOPInstanceUID_anon"),
                keep("img_height"), keep("img_width"),          # raw float-strings
                keep("laterality"), keep("view_position"),
                int01("horizontal_flip"), int01("weight_bearing"), int01("inverted"),
                keep("arthroplasty"),
                dbl("L_KLG_inference"), dbl("R_KLG_inference"),
                keep("SeriesDescription"), keep("StudyDescription"),
                dt("StudyDate_anon", fmt), dbl("age_at_exam"), keep("dicom_path")]
    if key == "pain":
        # Per brief, pain fields (incl. knee_pain / pain_score) stay VARCHAR (raw);
        # date_anon is the one date column and is parsed to DATE.
        return [vc("empi_anon"), keep("pain_location"), keep("knee_pain"),
                keep("pain_score"), keep("laterality"), dt("date_anon", fmt)]
    if key == "cpt_dictionary":
        return [keep("cpt_code"), keep("cpt_description")]
    if key == "icd_dictionary":
        return [keep("ICD9"), keep("ICD10"), keep("DX_NAME")]
    raise KeyError(f"no column spec for source key {key!r}")


def read_expr(path: Path) -> str:
    """DuckDB read_csv over the raw CSV as all-VARCHAR (no row-dropping type sniff)."""
    p = str(path).replace("'", "''")
    return f"read_csv('{p}', all_varchar=true, header=true, auto_detect=true)"


# --------------------------------------------------------------------------- #
# Per-table conversion + reconciliation.                                       #
# --------------------------------------------------------------------------- #
def convert_table(con: duckdb.DuckDBPyConnection, cfg: Config, key: str,
                  fmt: str, force: bool, log: logging.Logger) -> dict:
    src = cfg.source_path(key)
    pq = cfg.parquet_path(key)
    ref_rows = int(cfg["source_files"][key]["ref_rows"])
    date_cols = list(cfg["source_files"][key].get("date_cols") or [])
    spec = table_spec(key, fmt)
    n_columns = len(spec)
    has_empi = any(name == "empi_anon" for name, _ in spec)

    pq_lit = str(pq).replace("'", "''")

    # ---- restartable skip: existing Parquet whose row count already == ref ---
    if pq.exists() and not force:
        try:
            existing = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{pq_lit}')").fetchone()[0]
        except Exception:                       # unreadable / partial -> reconvert
            existing = None
        if existing == ref_rows:
            distinct = None
            if has_empi:
                distinct = con.execute(
                    f"SELECT COUNT(DISTINCT empi_anon) FROM read_parquet('{pq_lit}')"
                ).fetchone()[0]
            log.info("%s: SKIP (parquet exists, rows=%s == ref_rows; CSV not "
                     "re-scanned). Use --force to rebuild.", key, existing)
            return dict(table=key, csv_rows=existing, parquet_rows=existing,
                        ref_rows=ref_rows, match=(existing == ref_rows),
                        n_columns=n_columns, distinct_patients=distinct,
                        n_unparseable_dates=None,
                        parquet_bytes=pq.stat().st_size)

    # ---- write typed Parquet (streamed straight from CSV) --------------------
    select_clause = ",\n  ".join(f"{expr} AS {_q(name)}" for name, expr in spec)
    copy_sql = (f"COPY (\n  SELECT\n  {select_clause}\n  FROM {read_expr(src)}\n) "
                f"TO '{pq_lit}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    log.info("%s: converting %s -> %s (%d cols)...", key, src.name, pq.name, n_columns)
    con.execute(copy_sql)

    # ---- CSV-side reconciliation: true row count + unparseable dates ---------
    if date_cols:
        unp = " + ".join(
            f"CASE WHEN {_q(d)} IS NOT NULL AND {_q(d)} <> '' "
            f"AND try_strptime({_q(d)}, '{fmt}') IS NULL THEN 1 ELSE 0 END"
            for d in date_cols)
    else:
        unp = "0"
    csv_rows, n_unparseable = con.execute(
        f"SELECT COUNT(*), COALESCE(SUM({unp}), 0) FROM {read_expr(src)}"
    ).fetchone()
    csv_rows = int(csv_rows)
    n_unparseable = int(n_unparseable)

    # per-date-column detail (helps the downstream report)
    if date_cols:
        for d in date_cols:
            c = con.execute(
                f"SELECT COALESCE(SUM(CASE WHEN {_q(d)} IS NOT NULL AND {_q(d)} <> '' "
                f"AND try_strptime({_q(d)}, '{fmt}') IS NULL THEN 1 ELSE 0 END), 0) "
                f"FROM {read_expr(src)}").fetchone()[0]
            level = log.warning if c else log.info
            level("%s: date column %s unparseable=%d", key, d, int(c))

    # ---- Parquet-side reconciliation ----------------------------------------
    if has_empi:
        parquet_rows, distinct = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT empi_anon) FROM read_parquet('{pq_lit}')"
        ).fetchone()
    else:
        parquet_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{pq_lit}')").fetchone()[0]
        distinct = None
    parquet_rows = int(parquet_rows)

    # optional sanity vs configured reference patient/study counts
    ref_patients = cfg["source_files"][key].get("ref_patients")
    if distinct is not None and ref_patients is not None:
        lvl = log.info if distinct == int(ref_patients) else log.warning
        lvl("%s: distinct_patients=%s (ref_patients=%s)", key, distinct, ref_patients)
    if key == "image":
        studies = con.execute(
            f"SELECT COUNT(DISTINCT StudyInstanceUID_anon) FROM read_parquet('{pq_lit}')"
        ).fetchone()[0]
        ref_studies = cfg["source_files"][key].get("ref_studies")
        lvl = log.info if (ref_studies is None or studies == int(ref_studies)) else log.warning
        lvl("image: distinct_studies=%s (ref_studies=%s)", studies, ref_studies)

    match = (csv_rows == ref_rows == parquet_rows)
    (log.info if match else log.error)(
        "%s: csv_rows=%d parquet_rows=%d ref_rows=%d match=%s unparseable_dates=%d "
        "distinct_patients=%s bytes=%d",
        key, csv_rows, parquet_rows, ref_rows, match, n_unparseable,
        distinct, pq.stat().st_size)

    return dict(table=key, csv_rows=csv_rows, parquet_rows=parquet_rows,
                ref_rows=ref_rows, match=match, n_columns=n_columns,
                distinct_patients=distinct, n_unparseable_dates=n_unparseable,
                parquet_bytes=pq.stat().st_size)


# --------------------------------------------------------------------------- #
# Entry point.                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Faithful typed CSV->Parquet conversion.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--force", action="store_true",
                    help="rebuild every Parquet even if it already matches ref_rows")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    fmt = cfg["date_format"]

    log.info("START typed CSV->Parquet conversion (force=%s, date_format=%s)",
             args.force, fmt)

    tmpdir = tempfile.mkdtemp(prefix="mrkr_duckdb_")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmpdir}'")
    con.execute("SET preserve_insertion_order=false")   # memory-safe on 21M-row ICD

    results: dict[str, dict] = {}
    try:
        for key in PROCESS_ORDER:
            try:
                results[key] = convert_table(con, cfg, key, fmt, args.force, log)
            except Exception as exc:                    # collect, don't abort
                log.error("%s: CONVERSION FAILED: %s", key, exc, exc_info=True)
                ref_rows = int(cfg["source_files"][key]["ref_rows"])
                results[key] = dict(
                    table=key, csv_rows=None, parquet_rows=None, ref_rows=ref_rows,
                    match=False, n_columns=len(table_spec(key, fmt)),
                    distinct_patients=None, n_unparseable_dates=None,
                    parquet_bytes=(cfg.parquet_path(key).stat().st_size
                                   if cfg.parquet_path(key).exists() else None))
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- write conversion_checksums.csv (canonical table order) -------------
    rows = [results[k] for k in TABLE_ORDER if k in results]
    checks = cfg.path(cfg["paths"]["logs_dir"]) / "conversion_checksums.csv"
    pd.DataFrame(rows, columns=CHECKSUM_COLUMNS).to_csv(checks, index=False)
    log.info("wrote %s", checks)

    n_ok = sum(1 for r in rows if r["match"])
    total_bytes = sum(r["parquet_bytes"] or 0 for r in rows)
    all_ok = all(r["match"] for r in rows) and len(rows) == len(TABLE_ORDER)
    log.info("SUMMARY: %d/%d tables reconciled; parquet total %.1f MB; overall=%s",
             n_ok, len(TABLE_ORDER), total_bytes / 1e6, "OK" if all_ok else "FAIL")

    if not all_ok:
        failed = [r["table"] for r in rows if not r["match"]]
        log.error("RECONCILIATION FAILURES: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
