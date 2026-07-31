"""features_clinical.py — development CLINICAL feature table for the LOCKED cohort.

Phase 2 / Track A, step 1. Assembles ONE row per patient for the 3,709-patient
locked cohort, joined to the LOCKED splits and to the locked outcome labels
(``event_indicator`` / ``time_from_landmark``). This table is the input to the M0
penalized-Cox clinical baseline (``src/model_clinical.py``) and the clinical arm of
the multimodal fusion model. It reads only PRE-INDEX information; every ICD, pain
and image contribution is asserted to come from a date STRICTLY BEFORE index_date.

Protocol section 21 (predictor set) and non-negotiable #2 (every normalisation /
imputation fit on TRAIN ONLY) govern this module.

The M0 / M1 split (protocol Table 7, and Table 6's "secondary comparator only")
-------------------------------------------------------------------------------
Protocol Table 7 defines the model ladder: **M0 = "Age, sex, comorbidities, pain,
image-to-index interval"** and **M1 = "M0 plus inferred KLG"**, and Table 6 lists the
dataset-inferred contralateral KLG as a **secondary comparator only**. This module
therefore builds TWO model-column lists from config:

* ``model_columns``     — M0, from ``features_clinical.primary_predictors`` (11 predictors,
                          KLG absent, ``days_to_index`` present). This is the clinical
                          comparator the image models must beat, so it must not contain a
                          radiograph-derived severity grade.
* ``m1_model_columns``  — M1, from ``features_clinical.m1_predictors`` (M0 + ``klg_contra``),
                          fitted downstream on the **KLG-eligible subset only**
                          (``m1_requires_klg_eligible_subset: true``); KLG is never
                          median-imputed across the whole cohort for M1.

``klg_contra``, ``klg_contra_imp`` and ``klg_contra_missing`` all stay in the parquet: they
are M1's predictor and the evaluation/secondary columns, and `klg_contra_missing` still
drives the complete-case sensitivity filter. They are simply not M0 columns.

Feature sources
---------------
* ``final_cohort.parquet``      keys, index_date, sides, labels, imaging tier, and the
                                ``features_clinical.carry_columns`` (weight-bearing status)
* ``patient_splits.parquet``    LOCKED split (train 2,597 / val 371 / test 741)
* ``demographics.parquet``      sex -> ``sex_female``; race / race_major EVALUATION ONLY
* ``icd.parquet``               per-patient MAX(flag) over rows date_anon < index_date
                                (21.9M rows — streamed with DuckDB, never in pandas)
* ``pain.parquet``              rows in [index_date - pain_lookback_days, index_date)
                                (4.97M rows — streamed with DuckDB)
* ``image.parquet`` x ``selected_study_images.parquet``
                                contralateral inferred KLG of the SELECTED study's frontal

Contralateral-KLG selection rule (deterministic, documented for the manuscript)
------------------------------------------------------------------------------
Only the frontal (``view_position == klg_source_view``) images of the patient's
SELECTED study are eligible. 4,269 frontals cover 3,690 of the 3,709 patients;
569 patients have more than one frontal (560 have 2, 8 have 3, 1 has 4) and the
19 ``no_frontal``-tier patients have none. Rule:

  1. keep the frontal(s) with the SMALLEST ``days_to_index`` (closest to index);
  2. read ``L_KLG_inference`` when ``contra_side == 'L'``, else ``R_KLG_inference``;
  3. if step 1 leaves a tie, take the MEAN of the tied values (order-independent,
     therefore deterministic); NaN values are skipped, and the result is NaN only
     if every eligible frontal is NaN.

In this cohort every multi-frontal patient's frontals come from the same study and
therefore share one ``days_to_index``, so step 3 (the mean) is what actually
resolves all 569 multi-frontal patients; 173 patients get a non-integer KLG.

Carried EVALUATION-ONLY columns (protocol sections 21 and 24)
------------------------------------------------------------
``features_clinical.carry_columns`` names columns that are copied verbatim from
``final_cohort.parquet`` into the feature table so the audits that need them can run
downstream, WITHOUT them ever becoming predictors. Currently
``weight_bearing_frontal`` (bool), required by the section-21 equity/performance audit
(weight-bearing-status subgroup) and by the section-24 "weight-bearing frontal
radiographs only" sensitivity analysis. ``final_cohort`` is the source of record — it
already carries the column and it agrees with ``selected_studies.parquet`` on all 3,709
patients, so no re-join is performed. ``src/subgroups.py`` reads the same column from the
same file. Every carried column is asserted out of ``primary_predictors`` and out of
``params['model_columns']``.

Column naming convention (relied on by the downstream tasks)
-----------------------------------------------------------
``<name>``          raw as observed; NaN where genuinely unobserved.
``<name>_missing``  0/1 indicator, emitted for ``imputation.missing_indicator_cols``.
``<name>_imp``      MODEL-READY: raw with the TRAIN-fitted fill value applied, float64.
                    Emitted for EVERY column in ``primary_predictors`` and every extra
                    ``m1_predictors`` column (a straight copy when nothing was missing) so
                    downstream code has one uniform rule: fit on ``<predictor>_imp`` plus
                    the ``_missing`` indicators named in the model column list.
Both the raw and the imputed columns are kept so the imputation is auditable.

Aliased missing indicators are written but EXCLUDED from ``model_columns``
-------------------------------------------------------------------------
``pain_score_max`` is aggregated only over rows with ``knee_pain == '1'`` (see
:func:`load_pain`), because protocol Table 5 defines the predictor as a *knee* pain
score — widening it to every pre-index pain row would admit shoulder and back scores.
The arithmetic consequence in this cohort is exact: a pain score exists for precisely
the patients with a pre-index knee-pain record, 3,441 of 3,709, so "no pre-index
knee-pain record" and "no pain score" are the SAME EVENT and

    knee_pain_any_imp + pain_score_max_missing == 1   on every row.

A Cox partial likelihood has no intercept, so a constant added to the linear predictor
cancels: that pair spans an unidentified direction, and only the DIFFERENCE of the two
coefficients is estimable. :func:`aliased_missing_indicators` detects the relation from
the data (it is not hard-coded) and the indicator is dropped from ``model_columns``;
``knee_pain_any_imp`` remains as the explicit absence indicator and its coefficient is
then the identified quantity. The ``pain_score_max_missing`` COLUMN is still written to
the parquet for auditing — it is only its presence in the MODEL column list that creates
the alias. Every excluded column and its reason are recorded in
``params['excluded_model_columns']``, and the identity is re-verified on every run, so a
future cohort in which the two stop coinciding will silently re-admit the indicator.

Run:  python3 -m src.features_clinical --config config/feasibility.yaml
Writes derived-data/cohort/features_clinical.parquet (patient-level, git-ignored),
derived-data/cohort/clinical_imputation_params.json (the frozen transform),
outputs/tables/features_clinical_completeness.csv and
outputs/features_clinical_completeness.md (both AGGREGATE ONLY — no empi_anon).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import Config, ensure_dirs, load_config

MODULE = "features_clinical"

# Same mapping as src/splits.py — the split strata were built with it.
RACE_MAJOR = {"African American or Black": "Black", "Caucasian or White": "White", "Asian": "Asian"}

# LOCKED regression anchors (src/splits.py output; outputs/tables/split_summary.csv).
# Deliberately NOT in config: these are locked facts, and a config edit must not be able
# to weaken the guard that detects a silently-changed cohort.
EXPECTED_SPLIT_N = {"train": 2597, "val": 371, "test": 741}
EXPECTED_SPLIT_EVENTS = {"train": 373, "val": 54, "test": 106}
EXPECTED_N_PATIENTS = 3709
EXPECTED_N_EVENTS = 533

# LOCKED pain-domain anchors, valid at LOCKED_PAIN_LOOKBACK_DAYS. The pain domain values
# are now read from config (features_clinical.pain_knee_flag_true / pain_laterality_col /
# pain_bilateral_code), so these anchors exist to make the STRING-vs-int trap impossible
# to introduce silently: knee_pain and pain_score are VARCHAR in pain.parquet, and a
# comparison against a non-string domain value would zero the feature without erroring.
LOCKED_PAIN_LOOKBACK_DAYS = 365
EXPECTED_KNEE_PAIN_ANY = 3441        # patients with any pre-index knee_pain == "1" row
EXPECTED_PAIN_SCORE_OBSERVED = 3441  # patients with a non-null pain_score_max

# Verified value domain not parameterised in config (demographics.sex).
SEX_FEMALE_VALUE = "Female"

KLG_LEFT_COL = "L_KLG_inference"
KLG_RIGHT_COL = "R_KLG_inference"


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


# --------------------------------------------------------------------------- #
# PURE HELPERS — DataFrame in, DataFrame out; unit-tested on synthetic frames.  #
# --------------------------------------------------------------------------- #
def select_pre_index(records: pd.DataFrame, cohort: pd.DataFrame, *,
                     date_col: str = "date_anon",
                     index_date_col: str = "index_date",
                     lookback_days: int | None = None,
                     key: str = "empi_anon") -> pd.DataFrame:
    """Inner-join ``records`` to ``cohort`` and keep only STRICTLY pre-index rows.

    Kept window is ``[index_date - lookback_days, index_date)``: a record dated
    exactly ON index_date is EXCLUDED (that day can already carry the index
    procedure's own documentation), one day before is included, and
    ``lookback_days=None`` means all available history. This is the single
    definition of "pre-index" in this module; the DuckDB streaming queries for
    the ICD and pain tables implement the identical predicate in SQL and their
    joined intermediates are asserted against it.
    """
    df = records.merge(cohort, on=key, how="inner")
    dates = pd.to_datetime(df[date_col])
    index_dates = pd.to_datetime(df[index_date_col])
    keep = dates.notna() & (dates < index_dates)
    if lookback_days is not None:
        keep &= dates >= (index_dates - pd.Timedelta(days=int(lookback_days)))
    return df.loc[keep].reset_index(drop=True)


def select_klg_contra(frontal: pd.DataFrame, cohort: pd.DataFrame, *,
                      key: str = "empi_anon",
                      days_col: str = "days_to_index",
                      contra_col: str = "contra_side",
                      left_col: str = KLG_LEFT_COL,
                      right_col: str = KLG_RIGHT_COL) -> pd.DataFrame:
    """Per-patient contralateral inferred KLG from the selected study's frontal(s).

    ``frontal`` must already carry ``contra_col`` (it comes out of
    :func:`select_pre_index` joined to the cohort). Implements the three-step rule
    documented in the module docstring: nearest-to-index frontal(s), contra-side
    column, mean over ties. Returns one row per patient in ``cohort`` with
    ``klg_contra`` (NaN when unavailable) and ``klg_n_frontal`` (how many frontals
    the value was averaged over; 0 when the patient has none).
    """
    out = cohort[[key]].drop_duplicates().reset_index(drop=True)
    if len(frontal):
        f = frontal.copy()
        f["_klg"] = np.where(f[contra_col].to_numpy() == "L",
                             f[left_col].to_numpy(dtype=float),
                             f[right_col].to_numpy(dtype=float))
        nearest = f[days_col] == f.groupby(key)[days_col].transform("min")
        f = f.loc[nearest]
        agg = f.groupby(key).agg(klg_contra=("_klg", "mean"), klg_n_frontal=("_klg", "size"))
        out = out.merge(agg.reset_index(), on=key, how="left")
    else:
        out["klg_contra"] = np.nan
        out["klg_n_frontal"] = 0
    out["klg_n_frontal"] = out["klg_n_frontal"].fillna(0).astype("int16")
    return out


def fit_imputer(df: pd.DataFrame, columns: list[str], *,
                split_col: str = "split", fit_split: str = "train",
                numeric_strategy: str = "median",
                categorical_strategy: str = "most_frequent",
                missing_indicator_cols: list[str] | None = None) -> dict:
    """Fit fill values on the ``fit_split`` rows ONLY (non-negotiable #2).

    Returns a JSON-serialisable parameter dict that :func:`apply_imputer` replays
    verbatim on val/test, so no downstream module ever refits. Numeric columns get
    ``numeric_strategy`` (median), everything else ``categorical_strategy``
    (most-frequent). Raises if a column is entirely missing in the fit split.
    """
    missing_indicator_cols = list(missing_indicator_cols or [])
    assert split_col in df.columns, f"{split_col!r} column required to fit train-only"
    fit_mask = df[split_col] == fit_split
    n_fit = int(fit_mask.sum())
    assert n_fit > 0, f"no rows with {split_col} == {fit_split!r}"
    fit_df = df.loc[fit_mask]
    # Loud guard: the statistics below MUST see train rows and nothing else.
    assert (fit_df[split_col] == fit_split).all(), "imputer fit frame leaked a non-train row"
    assert len(fit_df) == n_fit, "imputer fit frame size disagrees with the train mask"

    params: dict = {
        "module": MODULE,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fit_split": fit_split,
        "n_fit_rows": n_fit,
        "numeric_strategy": numeric_strategy,
        "categorical_strategy": categorical_strategy,
        "missing_indicator_cols": missing_indicator_cols,
        "columns": {},
    }
    for col in columns:
        assert col in df.columns, f"column {col!r} missing from the feature frame"
        s_fit = fit_df[col]
        assert s_fit.notna().any(), f"{col!r} is entirely missing in the {fit_split} split"
        if pd.api.types.is_numeric_dtype(s_fit):
            strategy = numeric_strategy
            if strategy != "median":
                raise ValueError(f"unsupported numeric_strategy {strategy!r}")
            fill = float(s_fit.median(skipna=True))
        else:
            strategy = categorical_strategy
            if strategy != "most_frequent":
                raise ValueError(f"unsupported categorical_strategy {strategy!r}")
            modes = s_fit.dropna().mode()
            assert len(modes), f"{col!r} is entirely missing in the {fit_split} split"
            fill = str(modes.sort_values().iloc[0])       # sort -> deterministic on ties
        params["columns"][col] = {
            "strategy": strategy,
            "fill_value": fill,
            "dtype": str(df[col].dtype),
            "n_missing_fit_split": int(s_fit.isna().sum()),
            "n_missing_all": int(df[col].isna().sum()),
            "imputed_column": f"{col}_imp",
            "indicator_column": f"{col}_missing" if col in missing_indicator_cols else None,
        }
    return params


def apply_imputer(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply frozen :func:`fit_imputer` parameters — no statistic is recomputed.

    Adds ``<col>_imp`` for every fitted column (float64 when numeric) and
    ``<col>_missing`` for every column named in ``missing_indicator_cols``.
    """
    out = df.copy()
    for col, spec in params["columns"].items():
        assert col in out.columns, f"column {col!r} missing when applying the imputer"
        if spec["indicator_column"]:
            out[spec["indicator_column"]] = out[col].isna().astype("int8")
        imp = out[col].fillna(spec["fill_value"])
        if spec["strategy"] == "median":
            imp = imp.astype("float64")
        out[spec["imputed_column"]] = imp
        assert out[spec["imputed_column"]].notna().all(), f"{col!r} still missing after imputation"
    return out


def check_eval_only(carry_columns: list[str], primary: list[str],
                    model_columns: list[str] | None = None) -> list[str]:
    """Assert the carried columns are EVALUATION ONLY, then return them de-duplicated.

    Protocol sections 21 / 24 need the weight-bearing status inside the feature table,
    but a carried column must never become a predictor: not in ``primary_predictors``,
    not in the ``<name>_imp`` / ``<name>_missing`` model column list, and never named
    such that it collides with a generated model column.
    """
    carry = list(dict.fromkeys(carry_columns or []))
    for c in carry:
        assert c not in primary, \
            f"carry column {c!r} is EVALUATION ONLY and must not be a primary predictor"
        assert not c.endswith(("_imp", "_missing")), \
            f"carry column {c!r} collides with the generated model-column naming convention"
        if model_columns is not None:
            assert c not in model_columns, f"carry column {c!r} leaked into model_columns"
            assert f"{c}_imp" not in model_columns, f"{c}_imp leaked into model_columns"
    return carry


def aliased_missing_indicators(df: pd.DataFrame, candidate_columns: list[str],
                               indicator_columns: list[str]) -> dict[str, str]:
    """Missing indicators that duplicate another model column up to a sign and a constant.

    A Cox partial likelihood has no intercept, so any constant added to the linear
    predictor cancels. If an indicator ``I`` satisfies ``I + C == 1`` (complement) or
    ``I == C`` (duplicate) for some other model column ``C`` on EVERY row, then ``I`` and
    ``C`` span a direction the likelihood cannot see: ridge still fits, but it reports the
    minimum-norm split of one shared effect across two coefficients, and neither is
    interpretable on its own. Dropping ``I`` loses nothing — ``C`` alone carries the
    identified quantity.

    Returns ``{indicator: partner_column}`` for the aliased indicators, in the order given.
    Detection is exact (``np.array_equal``) rather than correlation-based, so it states a
    fact about the data instead of a threshold.
    """
    out: dict[str, str] = {}
    for ind in indicator_columns:
        if ind not in df.columns:
            continue
        a = df[ind].to_numpy(dtype=float)
        for col in candidate_columns:
            if col == ind or col not in df.columns:
                continue
            b = df[col].to_numpy(dtype=float)
            if np.array_equal(a + b, np.ones_like(a)) or np.array_equal(a, b):
                out[ind] = col
                break
    return out


def missingness_table(df: pd.DataFrame, columns: list[str], kinds: dict[str, str],
                      primary: list[str], *, split_col: str = "split") -> pd.DataFrame:
    """Aggregate-only per-column missingness, overall and by split (no ids)."""
    rows = []
    scopes = [("overall", df)] + [(s, df[df[split_col] == s]) for s in ("train", "val", "test")]
    for col in columns:
        for scope, sub in scopes:
            n = len(sub)
            n_missing = int(sub[col].isna().sum())
            rows.append(dict(column=col, kind=kinds.get(col, "feature"),
                             primary_predictor=int(col in primary), scope=scope,
                             n=n, n_missing=n_missing,
                             pct_missing=round(100.0 * n_missing / n, 4) if n else 0.0))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Streamed source aggregations (DuckDB; the 21.9M-row ICD table never enters    #
# pandas).  Both queries implement the same window as select_pre_index and      #
# return per-patient min/max record dates so the caller can assert no leakage.  #
# --------------------------------------------------------------------------- #
def _window_sql(alias: str, date_col: str, lookback_days: int | None) -> str:
    sql = (f"{alias}.{date_col} IS NOT NULL AND {alias}.{date_col} < c.index_date")
    if lookback_days is not None:
        sql += f" AND {alias}.{date_col} >= c.index_date - INTERVAL {int(lookback_days)} DAY"
    return sql


def load_comorbidities(con: duckdb.DuckDBPyConnection, parquet: Path,
                       flags: list[str], lookback_days: int | None) -> pd.DataFrame:
    """Per-patient MAX(flag) over pre-index ICD rows. Absent patient -> no row here."""
    flag_sql = ",\n           ".join(
        f'MAX(COALESCE(i."{f}", 0)) AS "{f}"' for f in flags)
    return con.execute(f"""
        SELECT c.empi_anon,
               COUNT(*) AS n_icd_preindex_rows,
               MIN(i.date_anon) AS icd_min_date,
               MAX(i.date_anon) AS icd_max_date,
               {flag_sql}
        FROM cohort c
        JOIN read_parquet('{str(parquet).replace("'", "''")}') i USING (empi_anon)
        WHERE {_window_sql("i", "date_anon", lookback_days)}
        GROUP BY c.empi_anon
    """).df()


def load_pain(con: duckdb.DuckDBPyConnection, parquet: Path, lookback_days: int,
              knee_col: str, score_col: str, flag_true: str,
              laterality_col: str, bilateral_code: str) -> pd.DataFrame:
    """Pre-index pain summary. ``knee_pain``/``pain_score`` are VARCHAR in the source.

    The value domain (``flag_true``, ``laterality_col``, ``bilateral_code``) comes from
    the caller (config), so it is verified in ONE place. ``flag_true`` and
    ``bilateral_code`` MUST be strings: the source columns are VARCHAR, so a numeric
    domain value would compare false on every row and silently zero the feature.
    """
    assert isinstance(flag_true, str), (
        f"pain_knee_flag_true must be a STRING ({knee_col} is VARCHAR in the source); got "
        f"{type(flag_true).__name__} {flag_true!r} — quote it in the YAML")
    assert isinstance(bilateral_code, str), (
        f"pain_bilateral_code must be a STRING; got {type(bilateral_code).__name__} "
        f"{bilateral_code!r} — quote it in the YAML")
    lit = str(parquet).replace("'", "''")
    flag_lit = flag_true.replace("'", "''")
    bilat_lit = bilateral_code.replace("'", "''")
    lat = f'p."{laterality_col}"'
    return con.execute(f"""
        SELECT c.empi_anon,
               COUNT(*) AS n_pain_preindex_rows,
               MIN(p.date_anon) AS pain_min_date,
               MAX(p.date_anon) AS pain_max_date,
               MAX(CASE WHEN p."{knee_col}" = '{flag_lit}' THEN 1 ELSE 0 END)
                   AS knee_pain_any,
               MAX(CASE WHEN p."{knee_col}" = '{flag_lit}'
                        THEN TRY_CAST(p."{score_col}" AS INTEGER) END) AS pain_score_max,
               MAX(CASE WHEN p."{knee_col}" = '{flag_lit}'
                         AND ({lat} = c.contra_side OR {lat} = '{bilat_lit}')
                        THEN 1 ELSE 0 END) AS knee_pain_contra,
               MAX(CASE WHEN {lat} IS NOT NULL AND {lat} <> '' THEN 1 ELSE 0 END)
                   AS knee_pain_contra_observed
        FROM cohort c
        JOIN read_parquet('{lit}') p USING (empi_anon)
        WHERE {_window_sql("p", "date_anon", lookback_days)}
        GROUP BY c.empi_anon
    """).df()


# --------------------------------------------------------------------------- #
# Report writers (AGGREGATE ONLY — outputs/ is not git-ignored).                #
# --------------------------------------------------------------------------- #
def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_completeness_md(path: Path, cfg_fc: dict, split_stats: list[dict],
                          miss: pd.DataFrame, params: dict, n_zero_icd: int,
                          n_zero_pain: int, klg_stats: dict, carry_stats: dict,
                          log: logging.Logger) -> Path:
    primary = list(cfg_fc["primary_predictors"])
    L = ["# Clinical feature table — completeness and imputation report", "",
         "Generated by `src/features_clinical.py` (Phase 2, Track A). Aggregate only: "
         "this file contains no patient identifiers.", "",
         "## Cohort", "",
         f"- Patients (one row per `empi_anon`): **{EXPECTED_N_PATIENTS:,}**",
         f"- 5-year contralateral-TKA events: **{EXPECTED_N_EVENTS}** "
         f"({100 * EXPECTED_N_EVENTS / EXPECTED_N_PATIENTS:.2f}%)",
         "- Splits are the LOCKED assignment from `src/splits.py`; the **test split is "
         "sealed** and is present here only so the frozen transform can be applied to it.",
         ""]
    L.append(_md_table(["split", "n patients", "n events", "event rate %"],
                       [[s["split"], f"{s['n']:,}", s["n_events"], f"{s['event_pct']:.2f}"]
                        for s in split_stats]))
    m1 = list(cfg_fc["m1_predictors"])
    m1_extra = [c for c in m1 if c not in primary]
    elig = params.get("m1_eligibility", {})
    L += ["", "## Predictor set (protocol Table 7 model ladder, protocol section 21)", "",
          "Protocol Table 7 defines **M0 = \"Age, sex, comorbidities, pain, image-to-index "
          "interval\"** and **M1 = \"M0 plus inferred KLG\"**, and protocol Table 6 lists the "
          "dataset-inferred contralateral KLG as a **secondary comparator only**. The two "
          "column lists are frozen separately in "
          f"`{cfg_fc['imputation']['params_json']}` (`model_columns` and `m1_model_columns`).",
          "",
          f"**M0 — primary clinical comparator ({len(primary)} predictors)**, model-ready as "
          "`<name>_imp`:", ""]
    L += [f"- `{p}`" for p in primary]
    L += ["",
          f"**M1 — secondary severity comparator ({len(m1)} predictors)** = M0 plus "
          + ", ".join(f"`{c}`" for c in m1_extra) + ".", ""]
    if elig:
        L += [f"- Fitted on the **KLG-eligible subset only** ({elig['rule']}): "
              f"**{elig['n_eligible']:,}** of {EXPECTED_N_PATIENTS:,} patients "
              f"({100 * elig['n_eligible'] / EXPECTED_N_PATIENTS:.1f}%); "
              f"{elig['n_ineligible']} patients have no inferred contralateral KLG.",
              "- Eligible by split: " + ", ".join(
                  f"{s} {elig['n_eligible_by_split'][s]:,} "
                  f"({elig['n_eligible_events_by_split'][s]} events)"
                  for s in ("train", "val", "test")) + ".",
              "- `klg_contra` is **never median-imputed for M1** — imputing a "
              "radiograph-derived severity grade for the patients who have no eligible "
              "bilateral frontal image would invent the very measurement M1 exists to test. "
              "The `_imp` column is written for uniformity, but on the eligible subset it "
              "equals the observed value on every row.", ""]
    L += ["Held out of the M0 predictor set on purpose:", "",
          "- `klg_contra` — dataset-inferred contralateral KLG. Protocol Table 6: "
          "**secondary comparator only**; protocol Table 7 puts it in **M1**. Keeping it in "
          "M0 would put a radiograph-derived severity grade inside the very comparator the "
          "image models must beat, which biases the study's primary estimand (does imaging "
          "improve prediction beyond routine clinical variables) toward zero.",
          "- `race`, `race_major` — retained for subgroup / fairness EVALUATION only "
          f"(`race_in_primary_predictors: {cfg_fc['race_in_primary_predictors']}`).",
          "- `knee_pain_contra` — side-specific pain. `pain.laterality` is coded on only "
          "~5% of source rows, so this variable is largely unobserved "
          f"({klg_stats['n_pain_lat_observed']:,} of {EXPECTED_N_PATIENTS:,} patients have any "
          "laterality-coded pre-index pain row, flagged by `knee_pain_contra_observed`); "
          "a 0 mostly means 'not coded', not 'no contralateral pain'. Descriptive use only.",
          ""]
    if carry_stats:
        L += ["## Carried evaluation-only columns (protocol sections 21 and 24)", "",
              "Copied verbatim from `final_cohort.parquet` (the same source "
              "`src/subgroups.py` reads; it agrees with `selected_studies.parquet` on all "
              f"{EXPECTED_N_PATIENTS:,} patients, so no re-join is done). These columns "
              "are **never predictors** — the module asserts each one out of "
              "`primary_predictors` and out of `model_columns`. They are here so the "
              "section-21 equity/performance audit can report a **weight-bearing-status "
              "subgroup** and the section-24 **\"weight-bearing frontal radiographs only\" "
              "sensitivity analysis** can be run downstream.", ""]
        L.append(_md_table(["column", "dtype", "n missing", "distribution"],
                           [[f"`{c}`", s["dtype"], s["n_missing"],
                             ", ".join(f"{k} = {v:,} ({100 * v / EXPECTED_N_PATIENTS:.1f}%)"
                                       for k, v in s["value_counts"].items())]
                            for c, s in carry_stats.items()]))
        L.append("")
    L += ["## Contralateral KLG selection rule", "",
          "Frontal (`view_position == 'F'`) images of the patient's SELECTED pre-index study "
          "only. (1) keep the frontal(s) with the smallest `days_to_index`; (2) read "
          "`L_KLG_inference` when `contra_side == 'L'`, else `R_KLG_inference`; (3) average "
          "over any remaining tie (order-independent, deterministic), skipping NaN.", "",
          f"- eligible frontal images: **{klg_stats['n_frontal_rows']:,}** across "
          f"**{klg_stats['n_patients_with_frontal']:,}** patients",
          f"- frontals per patient: " + ", ".join(
              f"{k}: {v:,}" for k, v in sorted(klg_stats["frontals_per_patient"].items())),
          f"- patients with no frontal at all (tier `no_frontal`): "
          f"**{klg_stats['n_no_frontal']}**",
          f"- patients resolved by averaging a tie (non-integer KLG): "
          f"**{klg_stats['n_averaged']:,}**",
          f"- `klg_contra` missing after the rule: **{klg_stats['n_klg_missing']}** "
          f"({100 * klg_stats['n_klg_missing'] / EXPECTED_N_PATIENTS:.2f}%)", "",
          "## Observation coverage (a 0 is not always an observed 0)", "",
          f"- Patients with **zero** pre-index ICD rows: **{n_zero_icd}**"
          + (" — every patient contributes at least one pre-index diagnosis row, so every "
             "comorbidity 0 in this table is an OBSERVED negative, not an unobserved one."
             if n_zero_icd == 0 else
             " — for those patients a comorbidity 0 means *unobserved*, not *absent*, and "
             "they should be flagged in any sensitivity analysis."),
          f"- Patients with zero pre-index pain rows inside the "
          f"{cfg_fc['pain_lookback_days']}-day lookback: **{n_zero_pain}** — their "
          "`knee_pain_any` 0 is likewise unobserved rather than observed-negative.", "",
          "## Imputation", "",
          f"**Fit on `split == \"{params['fit_split']}\"` rows ONLY "
          f"(n = {params['n_fit_rows']:,}), then applied unchanged to val and test.** No "
          "statistic in this table was computed from validation or test data. The frozen fill "
          f"values live in `{cfg_fc['imputation']['params_json']}` and are replayed verbatim by "
          "the downstream model modules — they are never refit.", ""]
    L.append(_md_table(["column", "strategy", "fill value", "n missing (train)", "n missing (all)",
                        "imputed column", "indicator column"],
                       [[f"`{c}`", s["strategy"], s["fill_value"], s["n_missing_fit_split"],
                         s["n_missing_all"], f"`{s['imputed_column']}`",
                         f"`{s['indicator_column']}`" if s["indicator_column"] else "—"]
                        for c, s in params["columns"].items()]))
    L += ["", "## Model columns, and the ones that are deliberately not model columns", "",
          f"`model_columns` (the **M0** list `src/model_clinical.py` and the Colab notebook "
          f"build the design matrix from) holds **{len(params['model_columns'])}** columns: "
          f"{len(primary)} `<predictor>_imp` columns plus the missing "
          "indicators that are *identified* and whose own predictor is in M0.",
          "",
          f"`m1_model_columns` holds **{len(params.get('m1_model_columns', []))}** columns — "
          "the M0 list plus " + ", ".join(f"`{c}_imp`" for c in m1_extra) + ".", ""]
    excl_m1 = params.get("excluded_m1_model_columns") or {}
    if excl_m1:
        L += [_md_table(["column", "excluded from `m1_model_columns` because"],
                        [[f"`{c}`", r] for c, r in excl_m1.items()]), ""]
    excluded = params.get("excluded_model_columns") or {}
    if excluded:
        L += [_md_table(["column", "excluded from `model_columns` because"],
                        [[f"`{c}`", r] for c, r in excluded.items()]), "",
              "**Why this matters, stated once.** `pain_score_max` is aggregated only over "
              "rows with `knee_pain == '1'`: protocol Table 5 defines the predictor as a "
              "*knee* pain score, and scoring it over every pre-index pain row would admit "
              "shoulder and back scores. The arithmetic consequence in this cohort is exact — "
              "a pain score exists for **precisely** the patients with a pre-index knee-pain "
              f"record ({EXPECTED_KNEE_PAIN_ANY:,} of {EXPECTED_N_PATIENTS:,}) — so *no "
              "pre-index knee-pain record* and *no pain score* are the **same event** and "
              "`knee_pain_any_imp + pain_score_max_missing = 1` on every row. A Cox partial "
              "likelihood has no intercept, so that pair spans a direction the likelihood "
              "cannot see: ridge still fits, but it splits one shared effect into two equal "
              "and opposite coefficients, neither interpretable alone. Dropping the indicator "
              "loses no information — `knee_pain_any_imp` **is** the absence indicator, and "
              "its coefficient is now the identified quantity. The column is still written to "
              "the parquet, and the complete-case sensitivity analysis still filters on it. "
              "The identity is re-verified on every run rather than assumed, so a cohort in "
              "which the two stop coinciding will automatically re-admit the indicator.", ""]
    else:
        L += ["No missing indicator is aliased in this build; every one of them is in "
              "`model_columns`.", ""]
    L += ["## Per-column missingness (raw, before imputation)", ""]
    wide = miss[miss["scope"] != "overall"].pivot(index="column", columns="scope",
                                                  values="n_missing")
    ov = miss[miss["scope"] == "overall"].set_index("column")
    rows = []
    for col in miss["column"].drop_duplicates():
        rows.append([f"`{col}`", ov.loc[col, "kind"],
                     "yes" if ov.loc[col, "primary_predictor"] else "",
                     int(ov.loc[col, "n_missing"]), f"{ov.loc[col, 'pct_missing']:.2f}",
                     int(wide.loc[col, "train"]), int(wide.loc[col, "val"]),
                     int(wide.loc[col, "test"])])
    L.append(_md_table(["column", "kind", "primary", "n missing (all)", "% missing",
                        "train", "val", "test"], rows))
    L += ["", "Every `*_imp` column has zero missingness by construction (asserted in code). "
          "The full long-format table is `outputs/tables/features_clinical_completeness.csv`.", ""]
    path.write_text("\n".join(L) + "\n")
    log.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Entry point.                                                                  #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the clinical feature table (Track A).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)
    cfg: Config = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    fcfg = cfg["features_clinical"]
    coh_dir = cfg.path(cfg["paths"]["cohort_dir"])
    flags = list(fcfg["comorbidity_flags"])
    primary = list(fcfg["primary_predictors"])
    # Protocol Table 7: M1 = M0 + inferred KLG, fitted on the KLG-eligible subset only.
    m1_predictors = list(fcfg["m1_predictors"])
    m1_extra = [c for c in m1_predictors if c not in primary]
    # Every column the frozen transform must cover: M0's predictors plus M1's extras.
    imputer_columns = primary + m1_extra
    imp_cfg = fcfg["imputation"]
    # EVALUATION-ONLY columns carried from final_cohort (protocol sections 21 and 24).
    carry_cols = check_eval_only(list(fcfg["carry_columns"]), primary)

    assert "race" not in primary and "race_major" not in primary, \
        "race must stay out of primary_predictors (protocol section 21)"
    assert not fcfg["race_in_primary_predictors"], "config says race_in_primary_predictors"
    # Protocol Table 6 lists inferred KLG as a SECONDARY COMPARATOR ONLY, and Table 7 puts it
    # in M1, not M0. Guarding this in code (not only in config) is the point: an M0 that
    # already contains a radiograph-derived severity grade biases the incremental value of
    # imaging toward zero, which is the study's primary estimand.
    assert not any(c.startswith("klg") for c in primary), (
        "inferred KLG is a SECONDARY comparator (protocol Table 6) and belongs to M1, not to "
        f"features_clinical.primary_predictors (got {[c for c in primary if c.startswith('klg')]})")
    assert set(primary) <= set(m1_predictors), "m1_predictors must be a superset of M0"
    assert m1_extra == ["klg_contra"], \
        f"protocol Table 7 defines M1 as M0 plus inferred KLG; extras are {m1_extra}"
    assert bool(fcfg["m1_requires_klg_eligible_subset"]), \
        "M1 must be restricted to the KLG-eligible subset (protocol Secondary objective 2)"
    assert "days_to_index" in primary, \
        "protocol Table 7 names the image-to-index interval as an M0 predictor"

    log.info("START clinical feature assembly (M0: %d primary predictors; M1: %d = M0 + %s; "
             "%d comorbidity flags, %d carried eval-only column(s): %s)",
             len(primary), len(m1_predictors), ", ".join(m1_extra), len(flags),
             len(carry_cols), ", ".join(carry_cols) or "none")

    # ---- 1. cohort + LOCKED splits + labels ---------------------------------
    # final_cohort is the source of record for the carried columns: it already holds them
    # (verified identical to selected_studies.parquet on all 3,709 patients) and
    # src/subgroups.py reads weight_bearing_frontal from the same file, so no re-join.
    cohort_all = pd.read_parquet(coh_dir / "final_cohort.parquet")
    for c in carry_cols:
        assert c in cohort_all.columns, \
            f"carry column {c!r} is not in final_cohort.parquet (columns: " \
            f"{sorted(cohort_all.columns)})"
    # days_to_index (protocol Table 6 "Image interval": days from index radiograph to index
    # TKA, continuous) is an M0 PREDICTOR under Table 7 and is complete for all 3,709.
    cohort = cohort_all[[
        "empi_anon", "index_date", "index_side", "contra_side", "side_source",
        "age_at_index", "days_to_index", "StudyInstanceUID_anon", "tier_name", "view_set",
        "event_indicator", "time_from_landmark", "complete_5y", "censor_reason"]
        + carry_cols]
    splits = pd.read_parquet(coh_dir / "patient_splits.parquet")[["empi_anon", "split"]]
    df = cohort.merge(splits, on="empi_anon", how="inner", validate="one_to_one")
    assert len(df) == EXPECTED_N_PATIENTS, f"expected {EXPECTED_N_PATIENTS} patients, got {len(df)}"
    assert df["empi_anon"].is_unique, "features table is not one row per patient"
    assert int(df["event_indicator"].sum()) == EXPECTED_N_EVENTS, "event count moved"
    for s, n in EXPECTED_SPLIT_N.items():
        assert int((df["split"] == s).sum()) == n, f"split {s} n changed"
        assert int(df.loc[df["split"] == s, "event_indicator"].sum()) == EXPECTED_SPLIT_EVENTS[s], \
            f"split {s} event count changed"

    # ---- 2. demographics (sex predictor; race EVALUATION ONLY) --------------
    demo = pd.read_parquet(cfg.parquet_path("demographics"))[["empi_anon", "sex", "race"]]
    df = df.merge(demo, on="empi_anon", how="left", validate="one_to_one")
    df["sex_female"] = (df["sex"] == SEX_FEMALE_VALUE).astype("int8").where(df["sex"].notna())
    df["race_major"] = df["race"].map(RACE_MAJOR).fillna("Other")

    # ---- 3+4. streamed pre-index ICD and pain aggregations ------------------
    tmpdir = tempfile.mkdtemp(prefix="mrkr_features_")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmpdir}'")
    con.execute("SET preserve_insertion_order=false")     # memory-safe on the 21.9M-row ICD
    try:
        con.register("cohort", df[["empi_anon", "index_date", "contra_side"]])
        icd = load_comorbidities(con, cfg.parquet_path("icd"), flags,
                                 fcfg["comorbidity_lookback_days"])
        pain = load_pain(con, cfg.parquet_path("pain"), int(fcfg["pain_lookback_days"]),
                         fcfg["pain_knee_flag_col"], fcfg["pain_score_col"],
                         fcfg["pain_knee_flag_true"], fcfg["pain_laterality_col"],
                         fcfg["pain_bilateral_code"])
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    # leakage guard on the joined intermediates (not just a log line)
    for name, agg, dmin, dmax, lookback in (
            ("icd", icd, "icd_min_date", "icd_max_date", fcfg["comorbidity_lookback_days"]),
            ("pain", pain, "pain_min_date", "pain_max_date", int(fcfg["pain_lookback_days"]))):
        chk = agg[["empi_anon", dmin, dmax]].merge(df[["empi_anon", "index_date"]], on="empi_anon")
        assert (pd.to_datetime(chk[dmax]) < chk["index_date"]).all(), \
            f"{name}: a contributing record is dated on/after index_date"
        if lookback is not None:
            assert (pd.to_datetime(chk[dmin])
                    >= chk["index_date"] - pd.Timedelta(days=lookback)).all(), \
                f"{name}: a contributing record predates the configured lookback"
        log.info("%s pre-index aggregation: %d/%d patients contributed at least one row",
                 name, len(agg), len(df))

    n_zero_icd = len(df) - len(icd)
    n_zero_pain = len(df) - len(pain)
    df = df.merge(icd.drop(columns=["icd_min_date", "icd_max_date"]), on="empi_anon", how="left")
    df = df.merge(pain.drop(columns=["pain_min_date", "pain_max_date"]), on="empi_anon", how="left")
    # No pre-index record -> observed-absent 0 for the binary flags (see the report: for
    # the patients with NO pre-index rows at all this 0 is "unobserved", hence n_zero_*).
    for c in flags + ["knee_pain_any", "knee_pain_contra", "knee_pain_contra_observed"]:
        df[c] = df[c].fillna(0).astype("int8")
    for c in ["n_icd_preindex_rows", "n_pain_preindex_rows"]:
        df[c] = df[c].fillna(0).astype("int32")
    df["pain_score_max"] = df["pain_score_max"].astype("float64")   # NaN preserved

    # Pain-domain guard. The domain values now come from config, so verify the resulting
    # prevalence rather than trusting them: a wrong/non-string pain_knee_flag_true would
    # zero knee_pain_any (and with it pain_score_max) without raising anywhere.
    n_knee_pain = int(df["knee_pain_any"].sum())
    n_pain_score = int(df["pain_score_max"].notna().sum())
    assert n_knee_pain > 0, (
        f"knee_pain_any is 0 for every patient — features_clinical.pain_knee_flag_true "
        f"({fcfg['pain_knee_flag_true']!r}) does not match the {fcfg['pain_knee_flag_col']!r} "
        "value domain in pain.parquet")
    if int(fcfg["pain_lookback_days"]) == LOCKED_PAIN_LOOKBACK_DAYS:
        assert n_knee_pain == EXPECTED_KNEE_PAIN_ANY, \
            f"knee_pain_any moved: {n_knee_pain} != {EXPECTED_KNEE_PAIN_ANY}"
        assert n_pain_score == EXPECTED_PAIN_SCORE_OBSERVED, \
            f"pain_score_max coverage moved: {n_pain_score} != {EXPECTED_PAIN_SCORE_OBSERVED}"
    else:
        log.warning("pain_lookback_days=%s differs from the locked %d — the pain prevalence "
                    "anchors were not checked", fcfg["pain_lookback_days"],
                    LOCKED_PAIN_LOOKBACK_DAYS)
    log.info("pain domain (from config): %s == %r -> knee_pain_any=%d (%.2f%%), "
             "pain_score_max observed=%d; laterality col %r, bilateral code %r",
             fcfg["pain_knee_flag_col"], fcfg["pain_knee_flag_true"], n_knee_pain,
             100 * n_knee_pain / len(df), n_pain_score, fcfg["pain_laterality_col"],
             fcfg["pain_bilateral_code"])

    # ---- 5. contralateral inferred KLG from the selected study's frontal(s) --
    ssi = pd.read_parquet(coh_dir / "selected_study_images.parquet")
    ssi = ssi[["empi_anon", "StudyInstanceUID_anon", "SOPInstanceUID_anon",
               "view_position", "StudyDate_anon", "days_to_index"]]
    sel = ssi.merge(df[["empi_anon", "StudyInstanceUID_anon", "index_date", "contra_side"]],
                    on=["empi_anon", "StudyInstanceUID_anon"], how="inner")
    frontal = sel[sel["view_position"] == fcfg["klg_source_view"]]
    # The selected study is pre-index by construction; re-derive it here so the guarantee
    # is enforced by this module rather than assumed from an upstream one.
    frontal_pre = select_pre_index(frontal.drop(columns=["index_date"]),
                                   df[["empi_anon", "index_date"]],
                                   date_col="StudyDate_anon")
    assert len(frontal_pre) == len(frontal), \
        f"{len(frontal) - len(frontal_pre)} selected frontal(s) are dated on/after index_date"
    assert (frontal_pre["days_to_index"] >= 1).all(), "a frontal has days_to_index < 1"
    img = duckdb.connect().execute(
        f"""SELECT SOPInstanceUID_anon, {KLG_LEFT_COL}, {KLG_RIGHT_COL}
            FROM read_parquet('{str(cfg.parquet_path("image")).replace("'", "''")}')""").df()
    frontal_klg = frontal_pre.merge(img, on="SOPInstanceUID_anon", how="left",
                                    validate="many_to_one")
    klg = select_klg_contra(frontal_klg, df[["empi_anon"]])
    df = df.merge(klg, on="empi_anon", how="left", validate="one_to_one")
    klg_stats = dict(
        n_frontal_rows=len(frontal_klg),
        n_patients_with_frontal=int(frontal_klg["empi_anon"].nunique()),
        frontals_per_patient={int(k): int(v) for k, v in
                              frontal_klg.groupby("empi_anon").size().value_counts().items()},
        n_no_frontal=int(len(df) - frontal_klg["empi_anon"].nunique()),
        n_averaged=int((df["klg_contra"].dropna() % 1 != 0).sum()),
        n_klg_missing=int(df["klg_contra"].isna().sum()),
        n_pain_lat_observed=int(df["knee_pain_contra_observed"].sum()))
    log.info("KLG: %d frontals -> %d patients; missing=%d; tie-averaged=%d; no_frontal=%d",
             klg_stats["n_frontal_rows"], klg_stats["n_patients_with_frontal"],
             klg_stats["n_klg_missing"], klg_stats["n_averaged"], klg_stats["n_no_frontal"])

    # ---- 6. imputation: FIT ON TRAIN ONLY, applied unchanged to val/test -----
    params = fit_imputer(df, imputer_columns, split_col="split", fit_split=imp_cfg["fit_split"],
                         numeric_strategy=imp_cfg["numeric_strategy"],
                         categorical_strategy=imp_cfg["categorical_strategy"],
                         missing_indicator_cols=list(imp_cfg["missing_indicator_cols"]))
    assert params["n_fit_rows"] == EXPECTED_SPLIT_N[imp_cfg["fit_split"]], \
        "imputer was not fit on the full locked train split"
    # Independent re-derivation of every fill value from the train rows alone.
    train_only = df.loc[df["split"] == imp_cfg["fit_split"]]
    for col, spec in params["columns"].items():
        if spec["strategy"] == "median":
            assert spec["fill_value"] == float(train_only[col].median()), \
                f"fill value for {col!r} is not the train-only median"
        else:
            assert spec["fill_value"] == str(train_only[col].dropna().mode().sort_values().iloc[0]), \
                f"fill value for {col!r} is not the train-only mode"
    df = apply_imputer(df, params)
    params["primary_predictors"] = primary
    params["m1_predictors"] = m1_predictors

    # ---- 6b. model columns, minus any missing indicator that is EXACTLY aliased --------
    # See the module docstring: pain_score_max is scored only on knee-pain rows, so in this
    # cohort knee_pain_any_imp + pain_score_max_missing == 1 identically and the pair spans
    # a constant the intercept-free Cox likelihood cannot identify. The relation is detected
    # from the data, never assumed, and the excluded column is still written to the parquet.
    # An indicator only enters a model column list when ITS OWN predictor is in that model:
    # klg_contra is an M1 predictor, so klg_contra_missing is not an M0 column.
    imp_cols_all = [f"{c}_imp" for c in primary]
    ind_cols_written = [f"{c}_missing" for c in imp_cfg["missing_indicator_cols"]]
    ind_cols_all = [f"{c}_missing" for c in imp_cfg["missing_indicator_cols"] if c in primary]
    aliased_ind = aliased_missing_indicators(df, imp_cols_all + ind_cols_all, ind_cols_all)
    params["model_columns"] = [c for c in imp_cols_all + ind_cols_all if c not in aliased_ind]
    params["indicator_columns_written"] = ind_cols_written
    params["excluded_model_columns"] = {
        ind: (f"exactly aliased with {partner!r} on all {len(df):,} rows; a Cox model has no "
              "intercept, so the pair spans an unidentified constant and only their "
              f"difference is estimable. {partner!r} carries it. The column is still written "
              "to the parquet for auditing and for the complete-case sensitivity filter.")
        for ind, partner in aliased_ind.items()}
    for ind, partner in aliased_ind.items():
        log.warning("EXACTLY ALIASED: %s is a deterministic function of %s — dropped from "
                    "model_columns (still written to the parquet)", ind, partner)
    # The pain alias is a construction artefact worth stating as a number, not just a flag.
    n_pain_alias_agree = int(((df["knee_pain_any"] == 1)
                              == df["pain_score_max"].notna()).sum())
    pain_alias_holds = n_pain_alias_agree == len(df)
    log.info("pain alias check: knee_pain_any == 1 and pain_score_max observed agree on "
             "%d/%d patients (%d with both) -> %s", n_pain_alias_agree, len(df), n_pain_score,
             "ALIASED, indicator dropped from model_columns" if pain_alias_holds
             else "NOT aliased, indicator retained")
    if int(fcfg["pain_lookback_days"]) == LOCKED_PAIN_LOOKBACK_DAYS:
        assert pain_alias_holds, (
            "the pain-score/knee-pain alias no longer holds at the locked lookback "
            f"({n_pain_alias_agree}/{len(df)} agree) — re-check load_pain() before trusting "
            "the model column list")
        assert "pain_score_max_missing" in aliased_ind, \
            "pain_score_max_missing should be detected as aliased with knee_pain_any_imp"

    # ---- 6c. M1 = M0 + inferred KLG, on the KLG-eligible subset (protocol Table 7) -----
    # M1 answers protocol Secondary objective 2 ("inferred KLG plus clinical comparator in
    # the subset with eligible bilateral frontal images"). Its design is frozen HERE so the
    # model module and the Colab notebook build it from the JSON rather than reconstructing
    # it. KLG is NEVER median-imputed for M1: the model is fitted where KLG is observed.
    m1_eligible = df["klg_contra"].notna()
    m1_imp_cols = [f"{c}_imp" for c in m1_extra]
    m1_ind_all = [f"{c}_missing" for c in imp_cfg["missing_indicator_cols"] if c in m1_extra]
    # On the eligible subset klg_contra_missing is identically 0 — a zero-variance column has
    # no hazard ratio, so it is excluded and the exclusion is recorded rather than assumed.
    m1_ind_constant = [c for c in m1_ind_all
                       if int(df.loc[m1_eligible, c].nunique(dropna=False)) <= 1]
    params["m1_model_columns"] = (list(params["model_columns"]) + m1_imp_cols
                                  + [c for c in m1_ind_all if c not in m1_ind_constant])
    params["m1_eligibility"] = {
        "column": "klg_contra",
        "rule": "klg_contra observed (not NaN) — i.e. klg_contra_missing == 0",
        "requires_klg_eligible_subset": bool(fcfg["m1_requires_klg_eligible_subset"]),
        "n_eligible": int(m1_eligible.sum()),
        "n_ineligible": int((~m1_eligible).sum()),
        "n_eligible_by_split": {s: int((m1_eligible & (df["split"] == s)).sum())
                                for s in ("train", "val", "test")},
        "n_eligible_events_by_split": {
            s: int(df.loc[m1_eligible & (df["split"] == s), "event_indicator"].sum())
            for s in ("train", "val", "test")},
        "note": "protocol Table 6 lists inferred KLG as a SECONDARY COMPARATOR ONLY and "
                "Table 7 places it in M1, not M0; M1 is fitted only where KLG is observed, "
                "so no radiograph-derived severity grade is imputed into any model.",
    }
    params["excluded_m1_model_columns"] = {
        c: (f"identically constant on the {int(m1_eligible.sum()):,} KLG-eligible patients "
            "(M1 is fitted only where klg_contra is observed), so it carries no information "
            "and has no hazard ratio")
        for c in m1_ind_constant}
    for c in [f"{c}_imp" for c in m1_extra] + m1_ind_all:
        params["excluded_model_columns"].setdefault(
            c, "M1-only column: protocol Table 6 lists dataset-inferred KLG as a SECONDARY "
               "comparator and Table 7 puts it in M1 ('M0 plus inferred KLG'), so it is out "
               "of the M0 design. It is written to the parquet and listed in "
               "`m1_model_columns`.")
    log.info("M1 (M0 + inferred KLG): %d model columns; KLG-eligible %d of %d patients "
             "(train %d / val %d / test %d); excluded as constant on the subset: %s",
             len(params["m1_model_columns"]), int(m1_eligible.sum()), len(df),
             *[params["m1_eligibility"]["n_eligible_by_split"][s]
               for s in ("train", "val", "test")],
             ", ".join(m1_ind_constant) or "none")

    params["eval_only_columns"] = ["race", "race_major", "sex", "knee_pain_contra",
                                   "knee_pain_contra_observed"] + carry_cols
    params["carry_columns"] = carry_cols
    # Sections 21/24 need these columns present and PROVABLY not predictors — in M0 and M1.
    check_eval_only(carry_cols, primary, params["model_columns"])
    check_eval_only(carry_cols, m1_predictors, params["m1_model_columns"])
    params["label_columns"] = ["event_indicator", "time_from_landmark", "complete_5y",
                               "censor_reason"]
    params["expected_rows"] = EXPECTED_N_PATIENTS
    params["split_counts"] = EXPECTED_SPLIT_N
    params["split_event_counts"] = EXPECTED_SPLIT_EVENTS

    # ---- 7. guards ----------------------------------------------------------
    for c in imputer_columns:
        assert c in df.columns, f"predictor {c!r} absent from the feature table"
        assert df[f"{c}_imp"].notna().all(), f"{c}_imp still has missing values"
    for c in imp_cfg["missing_indicator_cols"]:
        assert set(df[f"{c}_missing"].unique()) <= {0, 1}, f"{c}_missing is not 0/1"
    assert len(df) == EXPECTED_N_PATIENTS and df["empi_anon"].is_unique
    assert df["split"].notna().all()
    # Protocol Table 7, enforced on the artefact this module actually writes.
    assert not any(c.startswith("klg") for c in params["model_columns"]), \
        f"inferred KLG leaked into the M0 model columns: {params['model_columns']}"
    assert any(c.startswith("klg") for c in params["m1_model_columns"]), \
        "M1 must contain inferred KLG (protocol Table 7: 'M0 plus inferred KLG')"
    assert set(params["model_columns"]) < set(params["m1_model_columns"]), \
        "M1 must be a strict superset of M0 so the difference is exactly inferred KLG"
    # days_to_index is complete by construction; a silently-missing image interval would be
    # median-imputed and the coefficient would then describe an imputation, not an interval.
    assert int(df["days_to_index"].isna().sum()) == 0, \
        "days_to_index has missing values — protocol Table 7 requires the image-to-index " \
        "interval as an observed M0 predictor"

    # ---- 8. outputs ---------------------------------------------------------
    key_cols = ["empi_anon", "split", "index_date", "index_side", "contra_side", "side_source",
                "StudyInstanceUID_anon", "tier_name", "view_set"]
    label_cols = ["event_indicator", "time_from_landmark", "complete_5y", "censor_reason"]
    raw_cols = list(imputer_columns)
    ind_cols = [f"{c}_missing" for c in imp_cfg["missing_indicator_cols"]]
    imp_cols = [f"{c}_imp" for c in imputer_columns]
    eval_cols = ["sex", "race", "race_major", "knee_pain_contra", "knee_pain_contra_observed"]
    audit_cols = ["n_icd_preindex_rows", "n_pain_preindex_rows", "klg_n_frontal"]
    ordered = (key_cols + label_cols + raw_cols + ind_cols + imp_cols + eval_cols
               + carry_cols + audit_cols)
    assert len(set(ordered)) == len(ordered), "duplicate column in the output ordering"
    assert not set(carry_cols) & set(raw_cols + ind_cols + imp_cols), \
        "a carried evaluation-only column collides with a model column"
    out_df = df[ordered]
    out_parquet = cfg.path(fcfg["out_parquet"])
    out_df.to_parquet(out_parquet, index=False)
    log.info("wrote %s (%d rows x %d cols)", out_parquet, len(out_df), out_df.shape[1])

    params_json = cfg.path(imp_cfg["params_json"])
    params_json.write_text(json.dumps(params, indent=2, default=str) + "\n")
    log.info("wrote %s (frozen transform; downstream must NOT refit)", params_json)

    kinds = ({c: "predictor" for c in raw_cols} | {c: "missing_indicator" for c in ind_cols}
             | {c: "imputed" for c in imp_cols} | {c: "eval_only" for c in eval_cols}
             | {c: "eval_only_carried" for c in carry_cols}
             | {c: "audit" for c in audit_cols} | {c: "label" for c in label_cols})
    miss = missingness_table(out_df, raw_cols + ind_cols + imp_cols + eval_cols + carry_cols
                             + audit_cols + label_cols, kinds, primary)
    assert "empi_anon" not in miss.columns and not miss["column"].eq("empi_anon").any(), \
        "identifier leaked into an outputs/ table"
    miss_csv = cfg.path(fcfg["completeness_csv"])
    miss.to_csv(miss_csv, index=False)
    log.info("wrote %s", miss_csv)

    split_stats = [dict(split=s, n=int((df["split"] == s).sum()),
                        n_events=int(df.loc[df["split"] == s, "event_indicator"].sum()),
                        event_pct=100 * float(df.loc[df["split"] == s, "event_indicator"].mean()))
                   for s in ("train", "val", "test")]
    carry_stats = {c: dict(dtype=str(out_df[c].dtype),
                           n_missing=int(out_df[c].isna().sum()),
                           # sorted on the string key: deterministic for any dtype, and
                           # safe when a carried column mixes NaN with its values.
                           value_counts=dict(sorted((str(k), int(v)) for k, v in
                                                    out_df[c].value_counts(dropna=False).items())))
                   for c in carry_cols}
    write_completeness_md(cfg.path(fcfg["completeness_md"]), fcfg, split_stats, miss,
                          params, n_zero_icd, n_zero_pain, klg_stats, carry_stats, log)
    for c, s in carry_stats.items():
        log.info("carried eval-only %s (%s): %s; missing=%d; NOT a predictor",
                 c, s["dtype"], ", ".join(f"{k}={v}" for k, v in s["value_counts"].items()),
                 s["n_missing"])

    for c in imputer_columns:
        log.info("predictor %-24s [%s] missing_raw=%4d (%.2f%%)  fill=%s",
                 c, "M0" if c in primary else "M1 only",
                 int(df[c].isna().sum()), 100 * float(df[c].isna().mean()),
                 params["columns"][c]["fill_value"])
    log.info("zero pre-index ICD rows: %d patients; zero pre-index pain rows (%dd): %d patients",
             n_zero_icd, int(fcfg["pain_lookback_days"]), n_zero_pain)
    log.info("DONE: %d patients x %d columns (%d M0 model columns, %d M1 model columns, "
             "%d carried eval-only); imputation fit on %s only (n=%d); test split untouched "
             "except for applying the frozen transform.",
             len(out_df), out_df.shape[1], len(params["model_columns"]),
             len(params["m1_model_columns"]), len(carry_cols),
             imp_cfg["fit_split"], params["n_fit_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
