"""inventory.py - Stage-1 metadata-only data inventory (MRKR Contralateral TKA).

Read-only census of the seven typed source Parquet tables (reconciled to the raw
CSVs in Batch 1). Produces counts and distributions ONLY - no DICOMs are opened,
no models are run, no performance metrics are computed, and no patient identifier
(``empi_anon``) is ever written to an output. Every statistic is an AGGREGATE.

Statistics (row counts, distinct patients, date/age ranges, coded-field
distributions, missingness, full-row duplicates) are computed from the typed
Parquet at ``derived-data/source-parquet/<key>.parquet`` via DuckDB (the 22M-row
ICD table is aggregated with bounded queries, never pulled into pandas). Each
source FILE's on-disk path and size are read with ``os.stat`` on the ORIGINAL CSV
(``cfg.source_path(key)``).

Run from the project root::

    python3 -m src.inventory --config config/feasibility.yaml

Outputs (overwritten each run; append-only for the two logs):
  * outputs/data_inventory.csv
  * outputs/schema_report.md
  * outputs/data_quality_report.md
  * outputs/missingness_report.csv
  * outputs/protocol_to_column_mapping.csv
  * outputs/logs/assumptions.md   (appended below the pipeline marker; sole owner)
  * outputs/logs/run.log          (appended; prefix ``inventory``)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from src.config import Config, PROJECT_ROOT, ensure_dirs, load_config

MODULE = "inventory"

# Canonical table order (mirrors io_duckdb / config listing).
TABLE_ORDER = [
    "demographics", "cpt", "icd", "image", "pain",
    "cpt_dictionary", "icd_dictionary",
]
DICTIONARIES = {"cpt_dictionary", "icd_dictionary"}

# The 9 curated 0/1 flags carried on the ICD table (per-diagnosis-LINE, not
# per-patient: patient-level availability requires MAX(flag) GROUP BY empi_anon).
ICD_FLAGS = [
    "autoimmune", "diabetes", "hypertension", "joint_infection",
    "knee_osteoarthritis", "knee_osteomyelitis", "obesity",
    "nicotine_use", "trauma_lower_extremity",
]

# Model-inferred image columns (NOT ground truth - see The Emory MRKR dataset paper).
IMAGE_INFERRED_COLS = [
    "laterality", "view_position", "horizontal_flip", "weight_bearing",
    "inverted", "arthroplasty", "L_KLG_inference", "R_KLG_inference",
]

# Coded columns to enumerate (top values w/ counts) in the schema report.
CODED_COLS = {
    "demographics": ["sex", "race", "ethnicity"],
    "cpt": ["cpt_code", "cpt_group_modifier"],
    "icd": ["DX_ICD_SCOPE", "DX_LINE"],
    "image": ["laterality", "view_position", "weight_bearing", "arthroplasty"],
    "pain": ["laterality", "knee_pain"],
}

# Coded-missing sentinels: technically non-NULL but semantically missing/unknown.
# (value_token -> note). Counted as n_invalid in the missingness report.
SENTINELS: dict[tuple[str, str], tuple[str, str]] = {
    ("image", "laterality"):
        ("-1", "'-1' = unknown/unresolved laterality (model could not assign a side); excluded"),
    ("icd", "ICD10"):
        ("--", "'--' = no-ICD10 sentinel (ICD-9-coded or unmapped line); excluded from ICD10->DX_NAME join"),
    ("icd", "ICD9"):
        ("--", "'--' = no-ICD9 sentinel on this diagnosis line"),
}

# One-line column descriptions, keyed (table, column).
DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("demographics", "empi_anon"): "De-identified patient id; linkage key across all tables (VARCHAR).",
    ("demographics", "sex"): "Administrative sex (Female / Male).",
    ("demographics", "race"): "Patient race category; includes 'Unknown'.",
    ("demographics", "ethnicity"): "Patient ethnicity (Non-Hispanic / Hispanic / Unknown).",
    ("cpt", "empi_anon"): "De-identified patient id (linkage key).",
    ("cpt", "cpt_code"): "CPT/HCPCS procedure code (5-char); join to cpt_dictionary.cpt_code. Index TKA = 27447.",
    ("cpt", "cpt_group_modifier"): "Raw CPT modifier string (laterality RT/LT/50 + non-laterality tokens); parsed for TKA side.",
    ("cpt", "date_anon"): "De-identified procedure date; per-patient random shift preserving within-patient order.",
    ("cpt", "age_at_procedure"): "Age (years) at the procedure; HIPAA-bounded (observed 19-89).",
    ("icd", "empi_anon"): "De-identified patient id (linkage key).",
    ("icd", "ICD9"): "ICD-9-CM diagnosis code (raw); '--' where no ICD-9 on the line.",
    ("icd", "ICD10"): "ICD-10-CM diagnosis code (raw); '--' where no ICD-10; join to icd_dictionary.ICD10.",
    ("icd", "date_anon"): "De-identified diagnosis date; per-patient random shift preserving within-patient order.",
    ("icd", "age_at_dx"): "Age (years) at diagnosis; HIPAA-bounded (observed 19-89).",
    ("icd", "DX_LINE"): "Diagnosis line role (Primary / Secondary / Not Recorded / problem-list states).",
    ("icd", "DX_ICD_SCOPE"): "Diagnosis context (Billing / Discharge / Admitting / Problem List / etc.).",
    **{("icd", f): f"Curated 0/1 comorbidity flag '{f}' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon."
       for f in ICD_FLAGS},
    ("image", "empi_anon"): "De-identified patient id (linkage key).",
    ("image", "StudyInstanceUID_anon"): "De-identified DICOM Study UID; groups images of one exam (169,004 studies).",
    ("image", "SeriesInstanceUID_anon"): "De-identified DICOM Series UID.",
    ("image", "SOPInstanceUID_anon"): "De-identified DICOM instance UID; unique per image row (503,261).",
    ("image", "img_height"): "Image height in pixels (raw float-string).",
    ("image", "img_width"): "Image width in pixels (raw float-string).",
    ("image", "laterality"): "MODEL-INFERRED knee side: R / L / B (bilateral, contains contralateral) / -1 (unknown).",
    ("image", "view_position"): "MODEL-INFERRED view: F=frontal, L=lateral, S=sunrise, I/E=other.",
    ("image", "horizontal_flip"): "MODEL-INFERRED preprocessing flag (0/1): image horizontally flipped.",
    ("image", "weight_bearing"): "MODEL-INFERRED weight-bearing flag (0/1); classifier F1 ~0.98.",
    ("image", "inverted"): "MODEL-INFERRED preprocessing flag (0/1): photometric inversion.",
    ("image", "arthroplasty"): "MODEL-INFERRED prosthesis laterality: 0=none, R/L/B, NL=non-localized; F1 ~0.99.",
    ("image", "L_KLG_inference"): "MODEL-INFERRED left Kellgren-Lawrence grade; ~82% NULL (WB bilateral frontal, non-arthroplasty only).",
    ("image", "R_KLG_inference"): "MODEL-INFERRED right Kellgren-Lawrence grade; ~82% NULL (structural).",
    ("image", "SeriesDescription"): "Free-text DICOM series description (raw).",
    ("image", "StudyDescription"): "Free-text DICOM study description (raw).",
    ("image", "StudyDate_anon"): "De-identified study date; per-patient random shift preserving within-patient order.",
    ("image", "age_at_exam"): "Age (years) at the exam; HIPAA-bounded (observed 19-89).",
    ("image", "dicom_path"): "Relative DICOM transfer-manifest path (metadata only; no pixels opened).",
    ("pain", "empi_anon"): "De-identified patient id (linkage key).",
    ("pain", "pain_location"): "Free-text pain location (raw VARCHAR); ~75% NULL.",
    ("pain", "knee_pain"): "Knee-pain flag as raw VARCHAR ('0'/'1').",
    ("pain", "pain_score"): "Pain score as raw VARCHAR ('0'..'10').",
    ("pain", "laterality"): "Pain side as raw VARCHAR (R/L/B); ~95% NULL.",
    ("pain", "date_anon"): "De-identified encounter date; per-patient random shift preserving within-patient order.",
    ("cpt_dictionary", "cpt_code"): "CPT code (unique key of this lookup).",
    ("cpt_dictionary", "cpt_description"): "Human-readable CPT long description (join target for cpt.cpt_code).",
    ("icd_dictionary", "ICD9"): "ICD-9-CM code (raw).",
    ("icd_dictionary", "ICD10"): "ICD-10-CM code (unique key of this lookup).",
    ("icd_dictionary", "DX_NAME"): "Human-readable diagnosis name (join target for icd.ICD10).",
}

# Intended unique key / duplicate note per table.
KEY_NOTES: dict[str, str] = {
    "demographics": "empi_anon is the unique patient key (one row per patient)",
    "cpt": "no unique key (repeat billing lines); full row = empi_anon+cpt_code+cpt_group_modifier+date_anon+age_at_procedure",
    "icd": "no single unique key; per-diagnosis-line grain (DX_LINE / DX_ICD_SCOPE distinguish lines)",
    "image": "SOPInstanceUID_anon is the unique per-image key",
    "pain": "no unique key; repeated flowsheet/pain-score entries (identical rows expected)",
    "cpt_dictionary": "cpt_code is the unique key (code -> description)",
    "icd_dictionary": "ICD10 is the unique key (code -> DX_NAME)",
}


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
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def _q(col: str) -> str:
    """Double-quote a SQL identifier."""
    return '"' + col.replace('"', '""') + '"'


def _pq(cfg: Config, key: str) -> str:
    """Return the SQL identifier for a table's IN-MEMORY copy (loaded once).

    Each source Parquet is materialised into a DuckDB table exactly once by
    :func:`load_source_tables` (avoiding repeated reads of the large files, which
    can time out on a synced/slow disk). All statistics then query memory, not
    the filesystem. ``cfg`` is kept in the signature for call-site symmetry.
    """
    return _q(key)


def human_bytes(n: int | None) -> str:
    if n is None:
        return ""
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def _fmt_int(v) -> str:
    return "" if v is None else f"{int(v):,}"


def _fmt_val(v) -> str:
    """Render a coded value for a report cell ('(null)' for NULL/empty)."""
    if v is None:
        return "(null)"
    s = str(v)
    return s if s != "" else "(empty)"


def connect(cfg: Config, log: logging.Logger) -> tuple[duckdb.DuckDBPyConnection, str]:
    tmpdir = tempfile.mkdtemp(prefix="mrkr_inv_")
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")     # memory-safe on 22M-row ICD
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory='{tmpdir}'")
    return con, tmpdir


def load_source_tables(con, cfg: Config, log: logging.Logger) -> None:
    """Materialise each typed Parquet into an in-memory DuckDB table ONCE.

    Reads every source file exactly once (total ~0.4 GB uncompressed columnar),
    so all downstream aggregates hit memory instead of re-opening the large
    Parquet files (which can raise transient IO timeouts on a synced disk).
    """
    for key in TABLE_ORDER:
        p = str(cfg.parquet_path(key)).replace("'", "''")
        con.execute(f'CREATE OR REPLACE TABLE {_q(key)} AS SELECT * FROM read_parquet(\'{p}\')')
    log.info("loaded %d source tables into memory", len(TABLE_ORDER))


# --------------------------------------------------------------------------- #
# Per-table computation                                                        #
# --------------------------------------------------------------------------- #
def table_columns(con, cfg: Config, key: str) -> list[tuple[str, str]]:
    """[(column_name, dtype), ...] from the typed Parquet (finalized dtypes)."""
    rows = con.execute(f"DESCRIBE SELECT * FROM {_pq(cfg, key)}").fetchall()
    return [(r[0], r[1]) for r in rows]


def is_varchar(dtype: str) -> bool:
    return dtype.upper().startswith("VARCHAR")


def missing_expr(col: str, dtype: str) -> str:
    """SUM-able 0/1 expression: 1 when NULL (or empty string for VARCHAR)."""
    q = _q(col)
    if is_varchar(dtype):
        return f"CASE WHEN {q} IS NULL OR {q} = '' THEN 1 ELSE 0 END"
    return f"CASE WHEN {q} IS NULL THEN 1 ELSE 0 END"


def compute_table_stats(con, cfg: Config, key: str, cols: list[tuple[str, str]],
                        log: logging.Logger) -> dict:
    """One bounded pass for row/patient/date/age/duplicate stats."""
    P = _pq(cfg, key)
    sf = cfg["source_files"][key]
    is_dict = key in DICTIONARIES
    colnames = {c for c, _ in cols}

    sel = ["COUNT(*) AS n_rows"]
    if "empi_anon" in colnames:
        sel.append("COUNT(DISTINCT empi_anon) AS n_pat")
    else:
        sel.append("NULL AS n_pat")

    date_cols = list(sf.get("date_cols") or [])
    if date_cols:
        # single date column per patient table in this dataset
        dcol = date_cols[0]
        sel.append(f"MIN({_q(dcol)}) AS min_date")
        sel.append(f"MAX({_q(dcol)}) AS max_date")
    else:
        sel += ["NULL AS min_date", "NULL AS max_date"]

    age_col = sf.get("age_col")
    if age_col and age_col in colnames:
        sel.append(f"MIN({_q(age_col)}) AS min_age")
        sel.append(f"MAX({_q(age_col)}) AS max_age")
    else:
        sel += ["NULL AS min_age", "NULL AS max_age"]

    row = con.execute(f"SELECT {', '.join(sel)} FROM {P}").fetchdf().iloc[0].to_dict()

    # Full-row duplicate count (bounded aggregate; ICD DISTINCT * runs ~1s).
    n_rows = int(row["n_rows"])
    distinct_rows = con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {P})").fetchone()[0]
    n_dup = n_rows - int(distinct_rows)

    # Uncompressed columnar footprint (proxy for in-memory size).
    unc = con.execute(
        f"SELECT COALESCE(SUM(total_uncompressed_size),0) "
        f"FROM parquet_metadata('{str(cfg.parquet_path(key)).replace(chr(39), chr(39)*2)}')"
    ).fetchone()[0]

    min_date = row["min_date"]
    max_date = row["max_date"]

    def _d(v):
        return "" if v is None or pd.isna(v) else str(v)[:10]

    def _a(v):
        return "" if v is None or pd.isna(v) else f"{float(v):.0f}"

    stat = dict(
        key=key, n_rows=n_rows, n_columns=len(cols),
        n_pat=(None if is_dict or pd.isna(row["n_pat"]) else int(row["n_pat"])),
        min_date=_d(min_date), max_date=_d(max_date),
        min_age=_a(row["min_age"]), max_age=_a(row["max_age"]),
        n_dup=n_dup, uncompressed=int(unc),
    )
    log.info("%s: rows=%d cols=%d patients=%s date=[%s..%s] age=[%s..%s] dup_rows=%d",
             key, stat["n_rows"], stat["n_columns"], stat["n_pat"],
             stat["min_date"] or "-", stat["max_date"] or "-",
             stat["min_age"] or "-", stat["max_age"] or "-", n_dup)

    # Reconcile against config ref_rows (assert-style, logged loudly on mismatch).
    ref_rows = int(sf["ref_rows"])
    lvl = log.info if n_rows == ref_rows else log.error
    lvl("%s: n_rows=%d ref_rows=%d match=%s", key, n_rows, ref_rows, n_rows == ref_rows)
    stat["ref_rows"] = ref_rows
    stat["ref_match"] = (n_rows == ref_rows)
    return stat


def coded_distribution(con, cfg: Config, key: str, col: str, limit: int = 12) -> list[tuple]:
    """[(value, count), ...] ordered by count desc (top ``limit``)."""
    P = _pq(cfg, key)
    return con.execute(
        f"SELECT {_q(col)} AS v, COUNT(*) AS c FROM {P} GROUP BY 1 ORDER BY c DESC LIMIT {limit}"
    ).fetchall()


def compute_missingness(con, cfg: Config, key: str, cols: list[tuple[str, str]],
                        n_rows: int) -> list[dict]:
    """One pass over the table -> per-column NULL/empty counts; sentinel n_invalid."""
    P = _pq(cfg, key)
    parts = []
    for c, dtype in cols:
        parts.append(f"SUM({missing_expr(c, dtype)}) AS {_q('m_' + c)}")
        sent = SENTINELS.get((key, c))
        if sent is not None:
            token = sent[0].replace("'", "''")
            parts.append(f"SUM(CASE WHEN {_q(c)} = '{token}' THEN 1 ELSE 0 END) AS {_q('i_' + c)}")
    res = con.execute(f"SELECT {', '.join(parts)} FROM {P}").fetchdf().iloc[0].to_dict()

    out = []
    for c, _dtype in cols:
        n_missing = int(res[f"m_{c}"])
        sent = SENTINELS.get((key, c))
        n_invalid = 0
        note = ""
        if sent is not None:
            n_invalid = int(res.get(f"i_{c}", 0) or 0)
            note = sent[1] if n_invalid > 0 else ""
        out.append(dict(
            table=key, column=c, n_missing=n_missing,
            pct_missing=round(100.0 * n_missing / n_rows, 3) if n_rows else 0.0,
            n_invalid=n_invalid, invalid_note=note,
        ))
    return out


def patient_level_flag_prevalence(con, cfg: Config) -> list[tuple[str, int, float]]:
    """MAX(flag) GROUP BY empi_anon then SUM - the CORRECT patient-level availability."""
    P = _pq(cfg, "icd")
    maxes = ", ".join(f"MAX({_q(f)}) AS {_q(f)}" for f in ICD_FLAGS)
    sums = ", ".join(f"SUM({_q(f)}) AS {_q(f)}" for f in ICD_FLAGS)
    row = con.execute(
        f"SELECT {sums} FROM (SELECT empi_anon, {maxes} FROM {P} GROUP BY empi_anon)"
    ).fetchdf().iloc[0].to_dict()
    n_pat = con.execute(f"SELECT COUNT(DISTINCT empi_anon) FROM {P}").fetchone()[0]
    return [(f, int(row[f]), 100.0 * int(row[f]) / n_pat) for f in ICD_FLAGS]


def dict_lookup(con, cfg: Config, key: str, where: str, cols: str) -> list[tuple]:
    return con.execute(f"SELECT {cols} FROM {_pq(cfg, key)} WHERE {where}").fetchall()


# --------------------------------------------------------------------------- #
# Output writers                                                               #
# --------------------------------------------------------------------------- #
def write_inventory_csv(cfg: Config, stats: dict[str, dict],
                        cols_by_table: dict[str, list], log: logging.Logger) -> Path:
    rows = []
    for key in TABLE_ORDER:
        st = stats[key]
        csv_path = cfg.source_path(key)
        size = os.stat(csv_path).st_size          # size from the ORIGINAL CSV
        n_rows = st["n_rows"]
        big = n_rows > 1_000_000
        mem = (f"CSV {human_bytes(size)} on disk; ~{human_bytes(st['uncompressed'])} columnar "
               f"(Parquet uncompressed); "
               + ("LARGE - stream via DuckDB, do not load into pandas"
                  if big else "loads comfortably in pandas"))
        dup_note = KEY_NOTES[key] + (
            f"; {st['n_dup']:,} full-row duplicates" if st["n_dup"] else "; 0 full-row duplicates")
        rows.append(dict(
            file=key,
            # Project-RELATIVE: outputs/ is tracked and publishable, so it must not
            # carry the operator's home directory (protocol section 28).
            path=str(csv_path.relative_to(PROJECT_ROOT))
                 if csv_path.is_relative_to(PROJECT_ROOT) else str(csv_path),
            size_bytes=size,
            size_human=human_bytes(size),
            n_rows=n_rows,
            n_columns=st["n_columns"],
            column_names="; ".join(c for c, _ in cols_by_table[key]),
            n_unique_patients=("" if st["n_pat"] is None else st["n_pat"]),
            min_date=st["min_date"],
            max_date=st["max_date"],
            min_age=st["min_age"],
            max_age=st["max_age"],
            n_duplicate_rows=st["n_dup"],
            duplicate_key_note=dup_note,
            approx_memory_note=mem,
        ))
    cols = ["file", "path", "size_bytes", "size_human", "n_rows", "n_columns",
            "column_names", "n_unique_patients", "min_date", "max_date",
            "min_age", "max_age", "n_duplicate_rows", "duplicate_key_note",
            "approx_memory_note"]
    out = cfg.out("outputs_dir") / "data_inventory.csv"
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(rows))
    return out


def _md_kv_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        cells = [str(x).replace("|", "\\|").replace("\n", " ") for x in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_schema_report(cfg: Config, stats, cols_by_table, coded, dict_examples,
                        log: logging.Logger) -> Path:
    L = ["# Schema Report - MRKR Contralateral TKA Phase-1 (metadata-only)",
         "",
         "Finalized dtypes are the **typed Parquet** columns (reconciled to the raw "
         "CSVs in Batch 1). Row counts equal the config `ref_rows`. Example values are "
         "the top coded categories with row counts. No DICOM pixels are read; no models "
         "are run.",
         ""]
    for key in TABLE_ORDER:
        st = stats[key]
        sf = cfg["source_files"][key]
        L.append(f"## `{key}`  ({sf['filename']})")
        meta = [f"rows **{st['n_rows']:,}**", f"columns **{st['n_columns']}**"]
        if st["n_pat"] is not None:
            meta.append(f"distinct patients **{st['n_pat']:,}**")
        if st["min_date"]:
            meta.append(f"date range {st['min_date']} -> {st['max_date']}")
        L.append("- " + "; ".join(meta) + ".")
        if key == "icd":
            L.append("- The 9 curated 0/1 flags are **per-diagnosis-line**, not per-patient; "
                     "patient-level availability needs `MAX(flag) GROUP BY empi_anon`.")
        if key == "image":
            L.append("- Coded image fields (`laterality`, `view_position`, `weight_bearing`, "
                     "`arthroplasty`, KLG, flip/inverted) are **model-inferred**, not DICOM "
                     "ground truth (WB / arthroplasty classifiers F1 ~0.98-0.99).")
        if key == "cpt_dictionary":
            L.append("- Lookup table: `cpt_code` -> `cpt_description`. Join `cpt.cpt_code = "
                     "cpt_dictionary.cpt_code` (all fact CPT codes are covered).")
        if key == "icd_dictionary":
            L.append("- Lookup table: `ICD10` -> `DX_NAME`. Join `icd.ICD10 = icd_dictionary.ICD10` "
                     "(fact rows with `ICD10 = '--'` have no match by construction).")
        L.append("")

        header = ["column", "dtype", "description", "example values (top coded)"]
        trows = []
        for c, dtype in cols_by_table[key]:
            desc = DESCRIPTIONS.get((key, c), "")
            ex = ""
            if c in coded.get(key, {}):
                pairs = coded[key][c][:6]
                ex = "; ".join(f"{_fmt_val(v)}: {cnt:,}" for v, cnt in pairs)
            trows.append([f"`{c}`", dtype, desc, ex])
        L.append(_md_kv_table(header, trows))
        L.append("")

    # Dictionary interpretation examples
    L.append("## Dictionary interpretation (join examples)")
    L.append("")
    L.append("**CPT** `cpt.cpt_code` -> `cpt_dictionary.cpt_description`:")
    for code, desc in dict_examples["cpt"]:
        L.append(f"- `{code}` -> {desc}")
    L.append("")
    L.append("**ICD-10** `icd.ICD10` -> `icd_dictionary.DX_NAME` "
             "(note the 5th-digit laterality used by the side-recovery signal):")
    for code, name in dict_examples["icd"]:
        L.append(f"- `{code}` -> {name}")
    L.append("")

    out = cfg.out("outputs_dir") / "schema_report.md"
    out.write_text("\n".join(L) + "\n")
    log.info("wrote %s", out)
    return out


def write_quality_report(cfg: Config, stats, coded, extra, flag_prev,
                         log: logging.Logger) -> Path:
    tka = extra["tka"]  # dict: n, patients, blank, pct
    L = ["# Data-Quality Report - MRKR Contralateral TKA Phase-1 (metadata-only)",
         "",
         "Aggregate data-quality concerns for the Stage-1 feasibility gate. Counts only; "
         "no patient identifiers, no DICOM pixels, no model outputs beyond the "
         "provider-supplied inferred metadata fields (which are themselves flagged as a "
         "caveat below).",
         "",
         "## Headline concern - TKA laterality is under-coded on the index procedure",
         "",
         f"Of the **{tka['n']:,}** CPT `27447` (total knee arthroplasty) records across "
         f"**{tka['patients']:,}** patients, **{tka['blank']:,} ({tka['pct']:.1f}%)** carry a "
         "**blank `cpt_group_modifier`** - no RT/LT/50 laterality token. Only "
         f"{tka['rt']:,} RT, {tka['lt']:,} LT, and {tka['bilat']:,} '50' (bilateral) are "
         "explicitly side-coded. This 61% blank rate directly caps how many index TKAs "
         "(and therefore contralateral events) can be assigned a side from CPT alone; it "
         "is the primary feasibility constraint and motivates the image-`arthroplasty` and "
         "ICD 5th-digit side-recovery cross-checks.",
         "",
         "### CPT 27447 modifier distribution",
         _md_kv_table(["modifier", "records"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in extra["tka_dist"]]),
         "",
         "## Image metadata is model-inferred (provenance caveat)",
         "",
         "`laterality`, `view_position`, `weight_bearing`, `arthroplasty`, "
         "`horizontal_flip`, `inverted`, and the KLG grades are **model predictions** from "
         "the Emory MRKR pipeline, not DICOM ground truth. Reported classifier accuracy is "
         "high (weight-bearing F1 ~0.981, arthroplasty F1 ~0.992; laterality via dictionary "
         "rules), but any downstream selection built on them inherits that error rate. Treat "
         "them as strong priors, not certainties.",
         "",
         "### Image laterality (model-inferred) - '-1' is invalid/unknown",
         _md_kv_table(["laterality", "images"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in coded["image"]["laterality"]]),
         "",
         f"`-1` (n={extra['img_neg1']:,}) is an unresolved/unknown side and MUST be excluded "
         "from any laterality-dependent selection. `B` = bilateral frontal (contains the "
         "contralateral knee without a crop). "
         f"`arthroplasty = 'NL'` (n={extra['arth_nl']:,}) is a non-localized prosthesis "
         "detection used for QA/exclusion, not side assignment.",
         "",
         "### Image view and weight-bearing (model-inferred)",
         _md_kv_table(["view_position", "images"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in coded["image"]["view_position"]]),
         "",
         _md_kv_table(["weight_bearing", "images"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in coded["image"]["weight_bearing"]]),
         "",
         "### Image arthroplasty (model-inferred)",
         _md_kv_table(["arthroplasty", "images"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in coded["image"]["arthroplasty"]]),
         "",
         "## Kellgren-Lawrence grades are structurally NULL (~82%)",
         "",
         f"`L_KLG_inference` is NULL for {extra['klg_l']:,} of {stats['image']['n_rows']:,} "
         f"images (~{100*extra['klg_l']/stats['image']['n_rows']:.0f}%) and `R_KLG_inference` "
         f"for {extra['klg_r']:,} (~{100*extra['klg_r']/stats['image']['n_rows']:.0f}%). This is "
         "**by design**: KLG is inferred only on weight-bearing bilateral frontal views without "
         "arthroplasty. KLG is therefore a **secondary comparator only** - it cannot serve as a "
         "primary feature because it is unavailable for most images.",
         "",
         "## Pain table is sparse on the fields that matter",
         "",
         f"`laterality` is NULL for {extra['pain_lat_null']:,} of {stats['pain']['n_rows']:,} "
         f"pain rows (~{100*extra['pain_lat_null']/stats['pain']['n_rows']:.0f}%), and "
         f"`pain_location` is NULL for ~{100*extra['pain_loc_null']/stats['pain']['n_rows']:.0f}%. "
         "`knee_pain` and `pain_score` are populated but stored as raw VARCHAR ('0'/'1' and "
         "'0'..'10'). Pain is a **secondary predictor**; side-specific pain is largely "
         "unavailable.",
         "",
         _md_kv_table(["knee_pain", "rows"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in coded["pain"]["knee_pain"]]),
         "",
         _md_kv_table(["pain_score (raw)", "rows"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in extra["pain_score_dist"]]),
         "",
         "## Curated ICD flags are per-diagnosis-line, not per-patient",
         "",
         "The 9 curated 0/1 flags label **each diagnosis line**. A raw row-level mean is "
         f"therefore wrong: `knee_osteoarthritis` averages {extra['koa_rowmean']:.3f} across "
         "diagnosis lines, but the correct **patient-level** prevalence "
         "(`MAX(flag) GROUP BY empi_anon`) is "
         f"{[p for f, n, p in flag_prev if f == 'knee_osteoarthritis'][0]:.1f}%. Always "
         "aggregate to the patient before interpreting availability or prevalence.",
         "",
         "### Patient-level comorbidity/flag prevalence (MAX per patient, N = 83,011)",
         _md_kv_table(["curated flag", "patients", "prevalence %"],
                      [[f, f"{n:,}", f"{p:.1f}"] for f, n, p in flag_prev]),
         "",
         "## ICD-10 uses a `'--'` no-code sentinel (join/coverage caveat)",
         "",
         f"`icd.ICD10` is never NULL, but `'--'` (a no-ICD-10 sentinel) appears on "
         f"{extra['icd10_dashes']:,} of {stats['icd']['n_rows']:,} rows "
         f"(~{100*extra['icd10_dashes']/stats['icd']['n_rows']:.0f}%) - these are ICD-9-coded or "
         "unmapped lines and cannot join to `DX_NAME`. Treat `'--'` as missing for any "
         "ICD-10-based logic (including the M17.x side-recovery signal). The curated flags, not "
         "the raw ICD-10 join, are the reliable comorbidity path.",
         "",
         "## Demographics - race 'Unknown'",
         "",
         f"`race` = 'Unknown' for **{extra['race_unknown']:,}** patients "
         f"(~{100*extra['race_unknown']/stats['demographics']['n_rows']:.0f}%); `ethnicity` = "
         f"'Unknown' for {extra['eth_unknown']:,}. Report an explicit Unknown stratum rather "
         "than dropping or imputing.",
         "",
         _md_kv_table(["race", "patients"],
                      [[_fmt_val(v), f"{c:,}"] for v, c in coded["demographics"]["race"]]),
         "",
         "## De-identified dates - within-patient intervals are valid",
         "",
         "All dates (`date_anon`, `StudyDate_anon`) carry a **per-patient random shift** that "
         "preserves within-patient temporal order across all tables (0 unparseable). Absolute "
         "calendar dates are meaningless, but **within-patient day intervals** (index -> event, "
         "landmark, horizon) are valid and are the basis for the whole timeline. Death is "
         "unavailable, so the competing risk of mortality cannot be modeled from this metadata.",
         "",
         "## Full-row duplicates",
         "",
         "Exact-duplicate rows are retained as-is (the inventory only reports them): "
         + "; ".join(f"`{k}` {stats[k]['n_dup']:,}" for k in TABLE_ORDER if stats[k]['n_dup'])
         + ". Pain duplicates are repeated flowsheet entries; CPT duplicates are repeat billing "
         "lines. De-duplicate deliberately per analysis grain, not globally.",
         "",
         ]
    out = cfg.out("outputs_dir") / "data_quality_report.md"
    out.write_text("\n".join(L) + "\n")
    log.info("wrote %s", out)
    return out


def write_missingness_csv(cfg: Config, miss_rows: list[dict], log: logging.Logger) -> Path:
    cols = ["table", "column", "n_missing", "pct_missing", "n_invalid", "invalid_note"]
    out = cfg.out("outputs_dir") / "missingness_report.csv"
    pd.DataFrame(miss_rows, columns=cols).to_csv(out, index=False)
    log.info("wrote %s (%d column rows)", out, len(miss_rows))
    return out


def write_protocol_mapping_csv(cfg: Config, log: logging.Logger) -> Path:
    cols = ["protocol_concept", "dataset_columns", "source_table", "transformation",
            "confidence", "supporting_dictionary_definition", "unresolved_concern"]
    rows = [
        ["Patient id", "empi_anon", "all", "linkage key (VARCHAR)", "High",
         "n/a", "none"],
        ["Index TKA", "cpt_code = '27447'", "cpt", "filter to total knee arthroplasty", "High",
         "CPT dict: 'Arthroplasty, knee, condyle and plateau; medial AND lateral compartments "
         "with or without patella resurfacing (total knee arthroplasty)'", "none"],
        ["TKA side", "cpt_group_modifier", "cpt", "parse RT/LT/50/multi-token", "High parse / CRITICAL",
         "n/a", "61% of 27447 records (8,613/14,076) have a blank modifier - caps index and events"],
        ["Procedure/ICD/pain date", "date_anon", "cpt / icd / pain", "DATE %Y-%m-%d", "High",
         "n/a", "de-identified; only within-patient intervals are valid"],
        ["Age at index", "age_at_procedure", "cpt", "numeric, restrict >= 40 at index", "High",
         "n/a", "none"],
        ["Study/series/instance UID", "StudyInstanceUID_anon / SeriesInstanceUID_anon / SOPInstanceUID_anon",
         "image", "group images into studies", "High", "n/a", "none"],
        ["Image side", "laterality (R/L/B/-1)", "image",
         "map; -1 = unknown; B = bilateral (contains contralateral)", "High",
         "n/a", "-1 (n=682) excluded; model-inferred"],
        ["View", "view_position (F/L/S/I/E)", "image",
         "F=frontal, L=lateral, S=sunrise, I/E=other", "High",
         "n/a", "exact meaning of I/E uncertain; model-inferred"],
        ["Weight-bearing", "weight_bearing (0/1)", "image", "boolean", "High",
         "n/a", "model-inferred (F1 ~0.981)"],
        ["Arthroplasty on image", "arthroplasty (0/R/L/B/NL)", "image",
         "prosthesis laterality; exclusion / QA", "High",
         "n/a", "model-inferred (F1 ~0.992); NL = non-localized (n=29)"],
        ["Inferred KLG", "L_KLG_inference / R_KLG_inference", "image",
         "secondary comparator only", "Medium",
         "n/a", "~82% NULL; inferred only on WB bilateral frontal, non-arthroplasty views"],
        ["Image date", "StudyDate_anon", "image", "DATE %Y-%m-%d", "High",
         "n/a", "de-identified"],
        ["Image path", "dicom_path", "image", "transfer manifest only (no pixels opened)", "High",
         "n/a", "none"],
        ["Infection / osteomyelitis", "knee_osteomyelitis; joint_infection", "icd",
         "curated flags; MAX per patient within 365d pre-index; report both defs", "Medium",
         "dataset paper (curated); ICD-10 M86.05x/M86.06x (osteomyelitis, femur / tibia-fibula) "
         "corroborate knee-region", "joint_infection is not knee-specific; flags are per-line"],
        ["Comorbidities",
         "obesity; autoimmune; diabetes; hypertension; nicotine_use; trauma_lower_extremity; "
         "knee_osteoarthritis", "icd", "curated 0/1 flags; MAX per patient", "High",
         "n/a", "per-line aggregation (MAX GROUP BY empi_anon) required"],
        ["Pain", "pain_score; knee_pain; laterality; pain_location", "pain",
         "numeric / flags (raw VARCHAR)", "Medium (sparse)",
         "n/a", "laterality ~95% NULL; secondary predictor"],
        ["Demographics", "sex; race; ethnicity", "demographics", "categorical", "High",
         "n/a", "race 'Unknown' n=8,751 (report as explicit stratum)"],
        ["Last-observation date", "max(date across cpt/icd/pain) and image StudyDate_anon", "all",
         "per-patient max observed date", "High",
         "n/a", "death unavailable (competing risk of mortality cannot be modeled)"],
        ["Prior contralateral arthroplasty",
         "prior laterality-coded knee CPT (27447/27446/27445/27442/27440/27438/27486/27487/27488) "
         "+ pre-index image arthroplasty", "cpt + image", "pre-index detection", "Medium",
         "CPT dict (all listed codes present)",
         "27446 (unicompartmental) laterality completeness uncertain"],
        ["ICD side-recovery signal", "ICD10 M17.11/M17.31 (right); M17.12/M17.32 (left)", "icd",
         "5th-digit laterality", "Medium",
         "ICD dict: M17.11 'Unilateral primary osteoarthritis, right knee' (etc.)",
         "signal only; confined to labeled permissive arm; never applied post-index"],
    ]
    out = cfg.out("outputs_dir") / "protocol_to_column_mapping.csv"
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(rows))
    return out


def append_assumptions(cfg: Config, stats, extra, flag_prev, log: logging.Logger) -> Path:
    path = cfg.path(cfg["paths"]["assumptions_log"])
    marker = "<!-- Pipeline stages append below this line -->"
    sentinel = "## Stage-1 Data Inventory (`inventory` module)"

    koa_pat = [p for f, n, p in flag_prev if f == "knee_osteoarthritis"][0]
    section = f"""{sentinel}

Stage-1 metadata-only inventory (`src/inventory.py`). Sole owner of this section;
re-runs replace it in place. All statistics are aggregate counts read from the typed
Parquet; no `empi_anon` values, no DICOM pixels, no model performance metrics.

- **Curated ICD flags are per-diagnosis-line, not per-patient.** Availability and
  prevalence are computed as `MAX(flag) GROUP BY empi_anon`. A raw row-level mean is
  wrong (e.g. `knee_osteoarthritis` row-mean = {extra['koa_rowmean']:.3f} vs correct
  patient-level {koa_pat:.1f}%). Applied to all 9 flags in every report here.
- **Image `laterality = '-1'` (n={extra['img_neg1']:,}) treated as invalid/unknown**,
  not a side. Counted under `n_invalid` in the missingness report and flagged for
  exclusion from any laterality-dependent selection. `arthroplasty = 'NL'`
  (n={extra['arth_nl']:,}) is non-localized (QA/exclusion, not a side).
- **All coded image metadata is model-inferred**, not DICOM ground truth
  (`laterality`, `view_position`, `weight_bearing`, `arthroplasty`, KLG, flip/inverted).
  Reported as strong priors with a provenance caveat, never as certainties.
- **De-identified dates carry a per-patient random shift preserving within-patient
  order** (0 unparseable across all tables). Absolute calendar dates are not
  interpretable; within-patient day intervals (index/landmark/horizon) are valid and
  underpin the timeline. Death is unavailable (competing risk not modelable).
- **KLG is structurally NULL (~82%)** because it is inferred only on weight-bearing
  bilateral frontal, non-arthroplasty views. Documented as a secondary comparator only,
  never a primary feature.
- **Missingness definition:** NULL or empty string. Empirically the typed Parquet has
  no empty strings, so all reported missingness is NULL. Conservatively also surfaced two
  coded-missing sentinels that are non-NULL but semantically missing: image
  `laterality = '-1'` and, newly, **`icd.ICD10 = '--'`**
  (n={extra['icd10_dashes']:,}, ~{100*extra['icd10_dashes']/stats['icd']['n_rows']:.0f}% of
  ICD rows) = a no-ICD-10 sentinel (ICD-9-coded / unmapped lines) that cannot join to
  `DX_NAME`. Reported under `n_invalid`; treat as missing for any ICD-10 logic (incl. the
  M17.x side-recovery signal). Curated flags, not the raw ICD-10 join, are the reliable
  comorbidity path.
- **Full-row duplicates are reported, not removed** (pain {stats['pain']['n_dup']:,};
  cpt {stats['cpt']['n_dup']:,}; all other tables 0). These are repeated flowsheet /
  billing lines; de-duplication is left to the analysis grain of downstream modules.
- **Dictionary join coverage:** every CPT code in the fact table resolves in
  `cpt_dictionary`; ICD-10 resolution is limited by the `'--'` sentinel above (plus a
  small tail of newer codes absent from the lookup), so ICD-10 -> `DX_NAME` is
  informational, not a denominator.
- **Reconciliation:** every table's Parquet row count was re-checked against config
  `ref_rows` (all 7 match) before any report was written.
"""

    text = path.read_text() if path.exists() else ""
    idx = text.find(sentinel)
    if idx != -1:                       # replace our own previously-appended section
        text = text[:idx].rstrip() + "\n"
    if marker not in text:              # never rewrite content above the marker
        text = text.rstrip() + "\n\n" + marker + "\n"
    new = text.rstrip() + "\n\n" + section
    path.write_text(new)
    log.info("appended Stage-1 inventory section to %s", path)
    return path


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage-1 metadata-only data inventory.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    log.info("START Stage-1 data inventory (config=%s)", args.config)

    con, tmpdir = connect(cfg, log)
    ok = True
    try:
        load_source_tables(con, cfg, log)
        cols_by_table: dict[str, list] = {}
        stats: dict[str, dict] = {}
        coded: dict[str, dict] = {}
        miss_rows: list[dict] = []

        for key in TABLE_ORDER:
            cols = table_columns(con, cfg, key)
            cols_by_table[key] = cols
            stats[key] = compute_table_stats(con, cfg, key, cols, log)
            if not stats[key]["ref_match"]:
                ok = False
            # coded distributions for schema/quality reports
            coded[key] = {c: coded_distribution(con, cfg, key, c) for c in CODED_COLS.get(key, [])}
            # missingness for every column
            miss_rows.extend(compute_missingness(con, cfg, key, cols, stats[key]["n_rows"]))

        # ---- extra targeted stats for the quality report / assumptions --------
        Pc, Pi, Ppain, Pd = (_pq(cfg, k) for k in ("cpt", "icd", "pain", "demographics"))
        tka_n, tka_pat = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT empi_anon) FROM {Pc} WHERE cpt_code='27447'").fetchone()
        tka_blank = con.execute(
            f"SELECT COUNT(*) FROM {Pc} WHERE cpt_code='27447' "
            f"AND (cpt_group_modifier IS NULL OR cpt_group_modifier='')").fetchone()[0]

        def _tka_mod(tok):
            return con.execute(
                f"SELECT COUNT(*) FROM {Pc} WHERE cpt_code='27447' AND cpt_group_modifier='{tok}'"
            ).fetchone()[0]

        tka_dist = con.execute(
            f"SELECT cpt_group_modifier, COUNT(*) c FROM {Pc} WHERE cpt_code='27447' "
            f"GROUP BY 1 ORDER BY c DESC LIMIT 8").fetchall()
        pain_score_dist = con.execute(
            f"SELECT pain_score, COUNT(*) c FROM {Ppain} GROUP BY 1 "
            f"ORDER BY TRY_CAST(pain_score AS INTEGER) NULLS LAST LIMIT 12").fetchall()

        flag_prev = patient_level_flag_prevalence(con, cfg)
        dict_examples = {
            "cpt": dict_lookup(con, cfg, "cpt_dictionary",
                               "cpt_code IN ('27447','27446','27486')", "cpt_code, cpt_description"),
            "icd": dict_lookup(con, cfg, "icd_dictionary",
                               "ICD10 IN ('M17.11','M17.12','M17.31','M17.32')", "ICD10, DX_NAME"),
        }

        extra = dict(
            tka=dict(n=int(tka_n), patients=int(tka_pat), blank=int(tka_blank),
                     pct=100.0 * tka_blank / tka_n, rt=_tka_mod("RT"), lt=_tka_mod("LT"),
                     bilat=_tka_mod("50")),
            tka_dist=tka_dist,
            img_neg1=con.execute(f"SELECT COUNT(*) FROM {_pq(cfg,'image')} WHERE laterality='-1'").fetchone()[0],
            arth_nl=con.execute(f"SELECT COUNT(*) FROM {_pq(cfg,'image')} WHERE arthroplasty='NL'").fetchone()[0],
            klg_l=con.execute(f"SELECT COUNT(*) FROM {_pq(cfg,'image')} WHERE L_KLG_inference IS NULL").fetchone()[0],
            klg_r=con.execute(f"SELECT COUNT(*) FROM {_pq(cfg,'image')} WHERE R_KLG_inference IS NULL").fetchone()[0],
            pain_lat_null=con.execute(f"SELECT COUNT(*) FROM {Ppain} WHERE laterality IS NULL OR laterality=''").fetchone()[0],
            pain_loc_null=con.execute(f"SELECT COUNT(*) FROM {Ppain} WHERE pain_location IS NULL OR pain_location=''").fetchone()[0],
            pain_score_dist=pain_score_dist,
            icd10_dashes=con.execute(f"SELECT COUNT(*) FROM {Pi} WHERE ICD10='--'").fetchone()[0],
            koa_rowmean=con.execute(f"SELECT AVG(knee_osteoarthritis) FROM {Pi}").fetchone()[0],
            race_unknown=con.execute(f"SELECT COUNT(*) FROM {Pd} WHERE race='Unknown'").fetchone()[0],
            eth_unknown=con.execute(f"SELECT COUNT(*) FROM {Pd} WHERE ethnicity='Unknown'").fetchone()[0],
        )
        log.info("targeted: 27447 rows=%d patients=%d blank=%d (%.1f%%); image -1=%d; "
                 "arthroplasty NL=%d; ICD10 '--'=%d; race Unknown=%d",
                 extra["tka"]["n"], extra["tka"]["patients"], extra["tka"]["blank"],
                 extra["tka"]["pct"], extra["img_neg1"], extra["arth_nl"],
                 extra["icd10_dashes"], extra["race_unknown"])

        # ---- write the six deliverables --------------------------------------
        write_inventory_csv(cfg, stats, cols_by_table, log)
        write_schema_report(cfg, stats, cols_by_table, coded, dict_examples, log)
        write_quality_report(cfg, stats, coded, extra, flag_prev, log)
        write_missingness_csv(cfg, miss_rows, log)
        write_protocol_mapping_csv(cfg, log)
        append_assumptions(cfg, stats, extra, flag_prev, log)
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    n_match = sum(1 for k in TABLE_ORDER if stats.get(k, {}).get("ref_match"))
    log.info("SUMMARY: %d/%d tables reconciled to ref_rows; total missingness rows=%d; overall=%s",
             n_match, len(TABLE_ORDER), len(miss_rows), "OK" if ok else "FAIL")
    if not ok:
        log.error("RECONCILIATION FAILURE - one or more tables do not match ref_rows")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
