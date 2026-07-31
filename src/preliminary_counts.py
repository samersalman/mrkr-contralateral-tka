"""preliminary_counts.py — Stage-1 feasibility CHECKPOINT (PRELIMINARY, read-only).

MRKR Contralateral TKA — Phase-1 metadata-only feasibility. This module produces
BOUNDED, READ-ONLY *preliminary* count tables to support a go/no-go memo. It is
NOT the locked cohort extraction (that happens only after human sign-off).

Boundary / guardrails honoured here:
  * Read-only on the typed Parquet inputs; no DICOMs, no models, no metrics.
  * NO patient identifiers (``empi_anon``) are ever written. All persisted
    outputs are AGGREGATE COUNTS only. Patient-level intermediates live purely
    in-memory (pandas) / in ephemeral DuckDB temp tables and are discarded.
  * Reuses the shared parser + timeline helpers from ``src.laterality`` — the
    laterality logic is NOT reimplemented here.
  * Params come from ``config/feasibility.yaml`` via ``src.config.load_config``.
  * Logging APPENDS to ``outputs/logs/run.log`` with the ``preliminary_counts``
    prefix.

Run from the project root::

    python3 -m src.preliminary_counts --config config/feasibility.yaml

Everything written is labelled PRELIMINARY. See the module's final-message report
for the assumptions taken (this module does NOT write assumptions.md).
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

from src.config import Config, ensure_dirs, load_config
from src.laterality import (
    add_days,
    contralateral_side,
    days_between,
    horizon_date,
    landmark_date,
    last_observation,
    normalize_cpt,
    parse_modifier,
    within,
)

MODULE = "preliminary_counts"
PRELIM = "PRELIMINARY"


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
# Small helpers.                                                              #
# --------------------------------------------------------------------------- #
def _to_pydate(v) -> date | None:
    """pandas Timestamp / NaT / date -> python ``date`` or ``None``."""
    if v is None or pd.isna(v):
        return None
    if isinstance(v, date) and not isinstance(v, pd.Timestamp):
        return v
    return pd.Timestamp(v).date()


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 3) if den else 0.0


def _no_empi(df: pd.DataFrame, name: str) -> None:
    """Guardrail: assert no patient-identifier column leaked into an output."""
    bad = [c for c in df.columns if "empi" in c.lower() or c.lower() in {"patient_id", "mrn"}]
    if bad:
        raise AssertionError(f"output {name!r} contains identifier column(s): {bad}")


# --------------------------------------------------------------------------- #
# Parquet views (read-only).                                                  #
# --------------------------------------------------------------------------- #
def create_views(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    for key in ("cpt", "icd", "image", "pain", "demographics"):
        p = str(cfg.parquet_path(key)).replace("'", "''")
        con.execute(f"CREATE OR REPLACE VIEW {key} AS SELECT * FROM read_parquet('{p}')")


# --------------------------------------------------------------------------- #
# Build per-patient PROVISIONAL index tables (in-memory only).               #
# --------------------------------------------------------------------------- #
def build_index_frames(con: duckdb.DuckDBPyConnection, cfg: Config, log: logging.Logger):
    """Return (df447, per_patient, priorarth) pandas frames — never written out.

    * df447        : all CPT-27447 records with parsed (side, quality_flag).
    * per_patient  : one row per 27447 patient with strict / permissive index.
    * priorarth    : all prior-knee-arthroplasty-CPT records with parsed side.
    """
    lat = cfg["laterality"]
    rt = list(lat.get("right_tokens", ["RT"]))
    lt = list(lat.get("left_tokens", ["LT"]))
    bl = list(lat.get("bilateral_tokens", ["50"]))
    index_cpt = normalize_cpt(cfg["index"]["cpt_code"])            # '27447'

    def pm(raw):
        return parse_modifier(raw, rt, lt, bl)

    # ---- 27447 subset -> pandas (tiny) ------------------------------------
    df447 = con.execute(
        "SELECT empi_anon, date_anon, cpt_group_modifier, age_at_procedure, cpt_code "
        "FROM cpt WHERE cpt_code = ?", [cfg["index"]["cpt_code"]]).df()
    # Faithful normalisation via the shared parser (validation).
    df447["cpt_norm"] = df447["cpt_code"].map(normalize_cpt)
    assert (df447["cpt_norm"] == index_cpt).all(), "unexpected non-27447 code in subset"
    parsed = df447["cpt_group_modifier"].map(pm)
    df447["side"] = parsed.map(lambda t: t[0])
    df447["flag"] = parsed.map(lambda t: t[1])
    df447["is_single"] = df447["side"].isin(["R", "L"])
    df447["date_anon"] = pd.to_datetime(df447["date_anon"])

    # ---- prior knee-arthroplasty CPT records -> pandas --------------------
    codes = list(cfg["prior_knee_arthroplasty_cpt"].keys())
    inlist = ",".join("?" for _ in codes)
    priorarth = con.execute(
        f"SELECT empi_anon, date_anon, cpt_group_modifier, cpt_code "
        f"FROM cpt WHERE cpt_code IN ({inlist})", codes).df()
    pparsed = priorarth["cpt_group_modifier"].map(pm)
    priorarth["side"] = pparsed.map(lambda t: t[0])
    priorarth["is_single"] = priorarth["side"].isin(["R", "L"])
    priorarth["date_anon"] = pd.to_datetime(priorarth["date_anon"])

    # ---- per-patient earliest-27447 classification ------------------------
    fd = df447.groupby("empi_anon")["date_anon"].min().rename("first_date")
    d = df447.merge(fd, on="empi_anon")
    earl = d[d["date_anon"] == d["first_date"]]
    er = earl.groupby("empi_anon").agg(
        n_R=("side", lambda s: int((s == "R").sum())),
        n_L=("side", lambda s: int((s == "L").sum())),
        n_missing=("flag", lambda s: int((s == "missing").sum())),
        n_rec=("side", "size"),
        idx_age=("age_at_procedure", "max"),
    )

    def earliest_side(row):
        if row.n_R > 0 and row.n_L == 0:
            return "R"
        if row.n_L > 0 and row.n_R == 0:
            return "L"
        return None  # unsided / conflicting / bilateral / uninterpretable earliest

    per = er.copy()
    per["first_date"] = fd
    per["earliest_single_side"] = per.apply(earliest_side, axis=1)
    per["has_any_single"] = df447.groupby("empi_anon")["is_single"].max()
    per["strict"] = per["earliest_single_side"].notna()
    # earliest is blank when not strict and the earliest date carries a NULL/missing modifier
    per["earliest_blank"] = (~per["strict"]) & (per["n_missing"] > 0)

    # strict index attributes
    per["index_side_strict"] = per["earliest_single_side"]
    per["contra_strict"] = per["index_side_strict"].map(contralateral_side)
    per["index_date_strict"] = per["first_date"]
    per["index_age_strict"] = per["idx_age"]

    # ---- permissive index: earliest laterality-coded (single-side) 27447 --
    sng = df447[df447["is_single"]].copy()
    psd = sng.groupby("empi_anon")["date_anon"].min().rename("perm_date")
    s2 = sng.merge(psd, on="empi_anon")
    permrows = s2[s2["date_anon"] == s2["perm_date"]]
    perm_side = permrows.groupby("empi_anon")["side"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else None).rename("perm_side")
    perm_age = permrows.groupby("empi_anon")["age_at_procedure"].max().rename("perm_age")
    per = per.join(psd).join(perm_side).join(perm_age)
    per["contra_perm"] = per["perm_side"].map(
        lambda s: contralateral_side(s) if s is not None else None)

    log.info("%s: built index frames — 27447 patients=%d strict=%d has_single=%d",
             PRELIM, len(per), int(per["strict"].sum()), int(per["has_any_single"].sum()))
    return df447, per, priorarth


# --------------------------------------------------------------------------- #
# Register a cohort (empi, index_date, contra_side) as a DuckDB temp table    #
# and compute windowed ICD / IMAGE per-patient flags via SQL.                 #
# --------------------------------------------------------------------------- #
def _register_idx(con, name: str, frame: pd.DataFrame) -> None:
    con.register(f"{name}_df", frame)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {name} AS
        SELECT CAST(empi_anon AS VARCHAR) AS empi_anon,
               CAST(index_date AS DATE)   AS index_date,
               CAST(contra_side AS VARCHAR) AS contra_side
        FROM {name}_df
    """)
    con.unregister(f"{name}_df")


def sql_infection_flags(con, idx_tbl: str) -> pd.DataFrame:
    """MAX(flag) over ICD within [index-365, index) per patient."""
    w = "i.date_anon >= t.index_date - INTERVAL 365 DAY AND i.date_anon < t.index_date"
    return con.execute(f"""
        SELECT t.empi_anon,
               MAX(CASE WHEN {w} AND i.knee_osteomyelitis = 1 THEN 1 ELSE 0 END) AS osteo,
               MAX(CASE WHEN {w} AND i.joint_infection   = 1 THEN 1 ELSE 0 END) AS jinf
        FROM {idx_tbl} t
        JOIN icd i ON i.empi_anon = t.empi_anon
        GROUP BY 1
    """).df()


def sql_image_flags(con, idx_tbl: str) -> pd.DataFrame:
    """Per-patient pre-index image flags (prosthesis, eligible contra image, WB, views)."""
    pre = ("g.StudyDate_anon BETWEEN t.index_date - INTERVAL 365 DAY "
           "AND t.index_date - INTERVAL 1 DAY")
    return con.execute(f"""
        SELECT t.empi_anon,
          MAX(CASE WHEN g.StudyDate_anon < t.index_date
                    AND g.arthroplasty IN (t.contra_side, 'B') THEN 1 ELSE 0 END) AS prior_contra_img,
          MAX(CASE WHEN {pre}
                    AND (g.laterality = t.contra_side
                         OR (g.laterality = 'B' AND g.view_position = 'F'))
                   THEN 1 ELSE 0 END) AS elig_img,
          MAX(CASE WHEN {pre} AND g.weight_bearing = 1 THEN 1 ELSE 0 END) AS wb_pre,
          COUNT(DISTINCT CASE WHEN {pre} THEN g.view_position END) AS ndistview_pre,
          MAX(CASE WHEN {pre} AND g.view_position = 'F' THEN 1 ELSE 0 END) AS has_frontal_pre
        FROM {idx_tbl} t
        JOIN image g ON g.empi_anon = t.empi_anon
        GROUP BY 1
    """).df()


# --------------------------------------------------------------------------- #
# Pandas-side event / prior-CPT / observation flags (reuse timeline helpers). #
# --------------------------------------------------------------------------- #
def compute_cpt_flags(per_cohort: pd.DataFrame, df447: pd.DataFrame, priorarth: pd.DataFrame,
                      side_col: str, contra_col: str, idate_col: str,
                      landmark_days: int, event_start: int, horizon_days: int,
                      sec_windows: dict) -> pd.DataFrame:
    """Return per-patient CPT-based flags for the given cohort/index columns.

    Uses ``days_between`` + ``within`` (shared helpers) on each patient's 27447
    day-offsets so the outcome windows are defined exactly as the timeline spec.
    """
    cohort = per_cohort[["empi_anon", contra_col, idate_col]].rename(
        columns={contra_col: "contra", idate_col: "idate"})
    cohort = cohort[cohort["idate"].notna()]

    # contralateral prior-arthroplasty CPT (any of the 9 codes, single side == contra, before index)
    pa = priorarth[priorarth["is_single"]].merge(cohort, on="empi_anon", how="inner")
    pa_prior = (pa[(pa["side"] == pa["contra"]) & (pa["date_anon"] < pa["idate"])]
                .groupby("empi_anon").size().rename("prior_contra_cpt"))

    # 27447 records for cohort patients only
    ev = df447.merge(cohort, on="empi_anon", how="inner")
    idx = {r.empi_anon: (_to_pydate(r.idate), r.contra) for r in cohort.itertuples()}

    from_day1 = list(sec_windows.get("from_day1", []))
    from_day91 = list(sec_windows.get("from_day91", []))

    rows = {}
    for empi, g in ev.groupby("empi_anon"):
        idate, contra = idx[empi]
        contra_days, any_days = [], []
        for r in g.itertuples():
            dd = days_between(idate, _to_pydate(r.date_anon))
            if dd is None:
                continue
            any_days.append(dd)
            if r.side == contra:
                contra_days.append(dd)
        rec = {
            "primary_event": int(any(within(dd, event_start, horizon_days) for dd in contra_days)),
            "upper_event": int(any(within(dd, event_start, horizon_days) for dd in any_days)),
            "contra_0_90": int(any(within(dd, 0, landmark_days) for dd in contra_days)),
        }
        for v in from_day1:
            rec[f"sec_d1_{v}"] = int(any(within(dd, 1, v) for dd in contra_days))
        for v in from_day91:
            rec[f"sec_d91_{v}"] = int(any(within(dd, event_start, v) for dd in contra_days))
        rows[empi] = rec

    flags = pd.DataFrame.from_dict(rows, orient="index")
    flags.index.name = "empi_anon"
    flags = flags.reset_index()
    out = cohort[["empi_anon"]].merge(flags, on="empi_anon", how="left")
    out = out.merge(pa_prior, on="empi_anon", how="left")
    fill_cols = [c for c in out.columns if c != "empi_anon"]
    out[fill_cols] = out[fill_cols].fillna(0).astype(int)
    out["prior_contra_cpt"] = (out["prior_contra_cpt"] > 0).astype(int)
    return out


def compute_observation(con, per_cohort: pd.DataFrame, idate_col: str,
                        landmark_days: int) -> pd.DataFrame:
    """obs_ok = last_observation across cpt/icd/pain/image > index + landmark_days."""
    # per-patient max date per table (global; cheap group-bys)
    cpt_m = con.execute("SELECT empi_anon, MAX(date_anon) m FROM cpt GROUP BY 1").df()
    icd_m = con.execute("SELECT empi_anon, MAX(date_anon) m FROM icd GROUP BY 1").df()
    pain_m = con.execute("SELECT empi_anon, MAX(date_anon) m FROM pain GROUP BY 1").df()
    img_m = con.execute("SELECT empi_anon, MAX(StudyDate_anon) m FROM image GROUP BY 1").df()
    maxes = {}
    for df in (cpt_m, icd_m, pain_m, img_m):
        for r in df.itertuples():
            maxes.setdefault(r.empi_anon, []).append(_to_pydate(r.m))

    recs = []
    for r in per_cohort.itertuples():
        empi = r.empi_anon
        idate = _to_pydate(getattr(r, idate_col))
        if idate is None:
            recs.append((empi, 0))
            continue
        maxobs = last_observation(maxes.get(empi, []))
        lm = landmark_date(idate, landmark_days)
        ok = int(maxobs is not None and maxobs > lm)
        recs.append((empi, ok))
    return pd.DataFrame(recs, columns=["empi_anon", "obs_ok"])


# --------------------------------------------------------------------------- #
# OUTPUT 1 — modifier distribution.                                           #
# --------------------------------------------------------------------------- #
def out_modifier_distribution(df447: pd.DataFrame) -> pd.DataFrame:
    total = len(df447)
    raw = df447.copy()
    raw["raw_modifier"] = raw["cpt_group_modifier"].fillna("<NULL>")
    g = raw.groupby("raw_modifier").agg(
        n_records=("empi_anon", "size"),
        n_patients=("empi_anon", "nunique")).reset_index()
    g["view"] = "raw"
    g["quality_flag"] = ""
    g["side"] = ""
    g["n_tokens"] = g["raw_modifier"].map(
        lambda s: 0 if s == "<NULL>" else len(str(s).split()))
    g["is_multitoken"] = (g["n_tokens"] > 1).astype(int)
    g["is_blank"] = (g["raw_modifier"] == "<NULL>").astype(int)
    g["pct_records"] = g["n_records"].map(lambda n: _pct(n, total))
    g = g.rename(columns={"raw_modifier": "key"})

    p = df447.groupby(["side", "flag"]).agg(
        n_records=("empi_anon", "size"),
        n_patients=("empi_anon", "nunique")).reset_index()
    p["view"] = "parsed"
    p["key"] = ""
    p["quality_flag"] = p["flag"]
    p["n_tokens"] = -1
    p["is_multitoken"] = -1
    p["is_blank"] = -1
    p["pct_records"] = p["n_records"].map(lambda n: _pct(n, total))
    p = p.drop(columns=["flag"])

    cols = ["view", "key", "side", "quality_flag", "n_records", "n_patients",
            "pct_records", "n_tokens", "is_multitoken", "is_blank"]
    out = pd.concat([g[cols], p[cols]], ignore_index=True)
    out.insert(0, "label", PRELIM)
    out = out.sort_values(["view", "n_records"], ascending=[True, False]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Driver.                                                                     #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage-1 PRELIMINARY feasibility counts (read-only).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    tables_dir = cfg.out("tables_dir")
    outputs_dir = cfg.out("outputs_dir")

    # timeline constants (derive day-offsets via the shared helpers, no hard-coding)
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

    log.info("START %s checkpoint (landmark=%dd event_start=%dd horizon=%dd age_min=%.0f)",
             PRELIM, landmark_days, event_start, horizon_days, age_min)

    tmpdir = tempfile.mkdtemp(prefix="mrkr_prelim_")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmpdir}'")
    con.execute("SET preserve_insertion_order=false")

    headline: dict = {
        "label": PRELIM,
        "study": "MRKR Contralateral TKA — Phase-1 metadata-only feasibility",
        "note": "Preliminary read-only counts; NOT the locked cohort. Aggregates only.",
        "timeline": {"landmark_days": landmark_days, "event_start_day": event_start,
                     "horizon_days": horizon_days, "age_min": age_min},
    }

    try:
        create_views(con, cfg)
        df447, per, priorarth = build_index_frames(con, cfg, log)

        # base populations
        n_demo = con.execute("SELECT COUNT(*) FROM demographics").fetchone()[0]
        n_img_pat = con.execute("SELECT COUNT(DISTINCT empi_anon) FROM image").fetchone()[0]
        n_27447 = int(len(per))
        n_any_single = int(per["has_any_single"].sum())
        n_strict = int(per["strict"].sum())
        n_earlier_unsided = int((per["has_any_single"] & ~per["strict"]).sum())

        # -------------------------------------------------------------- #
        # OUTPUT 1 — modifier distribution                               #
        # -------------------------------------------------------------- #
        mod = out_modifier_distribution(df447)
        _no_empi(mod, "modifier_distribution")
        mod.to_csv(tables_dir / "stage1_modifier_distribution.csv", index=False)
        # quality_flag breakdown for reporting (single_rt/single_lt vs multi_single_side)
        flagct = df447.groupby("flag")["empi_anon"].nunique().to_dict()
        headline["modifier"] = {
            "n_records_27447": int(len(df447)),
            "n_patients_27447": n_27447,
            "n_blank_records": int(df447["cpt_group_modifier"].isna().sum()),
            "pct_blank_records": _pct(int(df447["cpt_group_modifier"].isna().sum()), len(df447)),
            "patients_by_quality_flag": {k: int(v) for k, v in flagct.items()},
        }

        # -------------------------------------------------------------- #
        # Strict / permissive index frames (in-memory)                   #
        # -------------------------------------------------------------- #
        strict = per[per["strict"]].copy()
        strict_idx = pd.DataFrame({
            "empi_anon": strict.index,
            "index_date": strict["index_date_strict"].values,
            "contra_side": strict["contra_strict"].values,
            "index_side": strict["index_side_strict"].values,
            "index_age": strict["index_age_strict"].values,
        })
        _register_idx(con, "idx_strict", strict_idx)

        # permissive: earliest single-side 27447 as index, unambiguous side,
        # AND no evidence of earlier arthroplasty (contra OR ipsi):
        #   (a) no earlier laterality-coded knee-arthroplasty CPT (any of 9 codes, single side) < index
        #   (b) no pre-index image prosthesis (arthroplasty in R/L/B) with StudyDate < index
        permcand = per[per["has_any_single"] & per["perm_side"].notna()].copy()
        permcand_idx = pd.DataFrame({
            "empi_anon": permcand.index,
            "index_date": permcand["perm_date"].values,
            "contra_side": permcand["contra_perm"].values,
            "index_side": permcand["perm_side"].values,
            "index_age": permcand["perm_age"].values,
        })
        # (a) prior single-side arthroplasty CPT before permissive index
        pa = priorarth[priorarth["is_single"]].merge(
            permcand_idx[["empi_anon", "index_date"]], on="empi_anon", how="inner")
        prior_cpt_any = set(pa[pa["date_anon"] < pa["index_date"]]["empi_anon"].unique())
        # (b) prior prosthesis on any pre-index image
        con.register("permcand_df", permcand_idx[["empi_anon", "index_date"]])
        con.execute("""CREATE OR REPLACE TEMP TABLE permcand AS
                       SELECT CAST(empi_anon AS VARCHAR) empi_anon, CAST(index_date AS DATE) index_date
                       FROM permcand_df""")
        con.unregister("permcand_df")
        prior_img = con.execute("""
            SELECT p.empi_anon
            FROM permcand p JOIN image g ON g.empi_anon = p.empi_anon
            WHERE g.StudyDate_anon < p.index_date AND g.arthroplasty IN ('R','L','B')
            GROUP BY 1""").df()
        prior_img_any = set(prior_img["empi_anon"].unique())
        permcand_idx["prior_evidence"] = permcand_idx["empi_anon"].map(
            lambda e: e in prior_cpt_any or e in prior_img_any)
        perm_idx = permcand_idx[~permcand_idx["prior_evidence"]].drop(columns=["prior_evidence"]).copy()
        _register_idx(con, "idx_perm", perm_idx)
        n_permissive = int(len(perm_idx))

        # -------------------------------------------------------------- #
        # OUTPUT 2 — cohort strategies                                   #
        # -------------------------------------------------------------- #
        def age40(mask_df, age_series):
            return int((age_series >= age_min).sum()) if len(mask_df) else 0

        strat_rows = [
            ("n_patients_with_27447", n_27447,
             int((per["idx_age"] >= age_min).sum())),
            ("n_with_any_single_side_27447", n_any_single,
             int((per.loc[per["has_any_single"], "perm_age"] >= age_min).sum())),
            ("n_strict", n_strict,
             int((strict["index_age_strict"] >= age_min).sum())),
            ("n_with_earlier_unsided_before_first_sided", n_earlier_unsided,
             int((per.loc[per["has_any_single"] & ~per["strict"], "perm_age"] >= age_min).sum())),
            ("n_permissive", n_permissive,
             int((perm_idx["index_age"] >= age_min).sum())),
            ("delta_permissive_minus_strict", n_permissive - n_strict,
             int((perm_idx["index_age"] >= age_min).sum()) -
             int((strict["index_age_strict"] >= age_min).sum())),
        ]
        strat = pd.DataFrame(strat_rows, columns=["metric", "n_patients", "n_patients_age40plus"])
        strat.insert(0, "label", PRELIM)
        _no_empi(strat, "cohort_strategies")
        strat.to_csv(tables_dir / "stage1_cohort_strategies.csv", index=False)
        headline["strategies"] = {r[0]: r[1] for r in strat_rows}

        # -------------------------------------------------------------- #
        # OUTPUT 3 — side recovery (blank-modifier index; CHARACTERIZE)  #
        # -------------------------------------------------------------- #
        blank = per[per["earliest_blank"]].copy()
        blank_idx = pd.DataFrame({
            "empi_anon": blank.index,
            "index_date": blank["first_date"].values,
        })
        n_blank = int(len(blank_idx))
        con.register("blank_df", blank_idx)
        con.execute("""CREATE OR REPLACE TEMP TABLE blank_pop AS
                       SELECT CAST(empi_anon AS VARCHAR) empi_anon, CAST(index_date AS DATE) index_date
                       FROM blank_df""")
        con.unregister("blank_df")
        img_sig = con.execute("""
            SELECT b.empi_anon,
              MAX(CASE WHEN g.StudyDate_anon = b.index_date AND g.laterality='R' THEN 1 ELSE 0 END) img_R,
              MAX(CASE WHEN g.StudyDate_anon = b.index_date AND g.laterality='L' THEN 1 ELSE 0 END) img_L,
              MAX(CASE WHEN g.StudyDate_anon = b.index_date AND lower(g.StudyDescription) LIKE '%right%' THEN 1 ELSE 0 END) desc_R,
              MAX(CASE WHEN g.StudyDate_anon = b.index_date AND lower(g.StudyDescription) LIKE '%left%'  THEN 1 ELSE 0 END) desc_L
            FROM blank_pop b JOIN image g ON g.empi_anon = b.empi_anon
            GROUP BY 1""").df()
        rc = cfg["icd_side_recovery"]
        rcodes = "','".join(rc["right_codes"]); lcodes = "','".join(rc["left_codes"])
        icd_sig = con.execute(f"""
            SELECT b.empi_anon,
              MAX(CASE WHEN i.ICD10 IN ('{rcodes}') AND i.date_anon <= b.index_date THEN 1 ELSE 0 END) icd_R,
              MAX(CASE WHEN i.ICD10 IN ('{lcodes}') AND i.date_anon <= b.index_date THEN 1 ELSE 0 END) icd_L
            FROM blank_pop b JOIN icd i ON i.empi_anon = b.empi_anon AND i.ICD10 LIKE 'M17%'
            GROUP BY 1""").df()
        sig = blank_idx[["empi_anon"]].merge(img_sig, on="empi_anon", how="left") \
                                      .merge(icd_sig, on="empi_anon", how="left").fillna(0)
        for c in ["img_R", "img_L", "desc_R", "desc_L", "icd_R", "icd_L"]:
            sig[c] = sig[c].astype(int)
        # per-signal presence & single-side determination
        sig["a_present"] = ((sig.img_R + sig.img_L) > 0).astype(int)
        sig["a_single"] = ((sig.img_R == 1) ^ (sig.img_L == 1)).astype(int)
        sig["b_present"] = ((sig.icd_R + sig.icd_L) > 0).astype(int)
        sig["b_single"] = ((sig.icd_R == 1) ^ (sig.icd_L == 1)).astype(int)
        sig["c_present"] = ((sig.desc_R + sig.desc_L) > 0).astype(int)
        sig["c_single"] = ((sig.desc_R == 1) ^ (sig.desc_L == 1)).astype(int)
        tot_R = ((sig.img_R + sig.icd_R + sig.desc_R) > 0)
        tot_L = ((sig.img_L + sig.icd_L + sig.desc_L) > 0)
        sig["any_present"] = (tot_R | tot_L).astype(int)
        sig["concordant_single"] = (tot_R ^ tot_L).astype(int)
        sr_rows = [
            ("blank_index_population", n_blank, n_blank, 100.0),
            ("same_day_image_laterality", int(sig.a_present.sum()), int(sig.a_single.sum()),
             _pct(int(sig.a_present.sum()), n_blank)),
            ("icd_m17_laterality_on_or_before", int(sig.b_present.sum()), int(sig.b_single.sum()),
             _pct(int(sig.b_present.sum()), n_blank)),
            ("same_day_studydesc_text", int(sig.c_present.sum()), int(sig.c_single.sum()),
             _pct(int(sig.c_present.sum()), n_blank)),
            ("any_signal_present", int(sig.any_present.sum()), int(sig.concordant_single.sum()),
             _pct(int(sig.any_present.sum()), n_blank)),
        ]
        sr = pd.DataFrame(sr_rows, columns=["signal", "n_present", "n_points_to_single_side", "pct_of_blank_pop"])
        sr.insert(0, "label", PRELIM)
        _no_empi(sr, "side_recovery")
        sr.to_csv(tables_dir / "stage1_side_recovery.csv", index=False)
        headline["side_recovery"] = {
            "blank_index_population": n_blank,
            "same_day_image_laterality_present": int(sig.a_present.sum()),
            "icd_m17_present": int(sig.b_present.sum()),
            "studydesc_present": int(sig.c_present.sum()),
            "any_signal_present": int(sig.any_present.sum()),
            "concordant_single_side": int(sig.concordant_single.sum()),
            "pct_any_signal": _pct(int(sig.any_present.sum()), n_blank),
            "pct_concordant_single": _pct(int(sig.concordant_single.sum()), n_blank),
        }

        # -------------------------------------------------------------- #
        # ICD / IMAGE / CPT flags for strict & permissive cohorts        #
        # -------------------------------------------------------------- #
        infS = sql_infection_flags(con, "idx_strict")
        imgS = sql_image_flags(con, "idx_strict")
        cptS = compute_cpt_flags(strict_idx, df447, priorarth,
                                 "index_side", "contra_side", "index_date",
                                 landmark_days, event_start, horizon_days, sec_windows)
        obsS = compute_observation(con, strict_idx, "index_date", landmark_days)
        S = strict_idx.merge(infS, on="empi_anon", how="left") \
                      .merge(imgS, on="empi_anon", how="left") \
                      .merge(cptS, on="empi_anon", how="left") \
                      .merge(obsS, on="empi_anon", how="left")
        num_cols = [c for c in S.columns if c not in
                    ("empi_anon", "index_date", "contra_side", "index_side", "index_age")]
        S[num_cols] = S[num_cols].fillna(0)
        for c in num_cols:
            S[c] = S[c].astype(int)

        infP = sql_infection_flags(con, "idx_perm")
        imgP = sql_image_flags(con, "idx_perm")
        cptP = compute_cpt_flags(perm_idx, df447, priorarth,
                                 "index_side", "contra_side", "index_date",
                                 landmark_days, event_start, horizon_days, sec_windows)
        obsP = compute_observation(con, perm_idx, "index_date", landmark_days)
        P = perm_idx.merge(infP, on="empi_anon", how="left") \
                    .merge(imgP, on="empi_anon", how="left") \
                    .merge(cptP, on="empi_anon", how="left") \
                    .merge(obsP, on="empi_anon", how="left")
        numP = [c for c in P.columns if c not in
                ("empi_anon", "index_date", "contra_side", "index_side", "index_age")]
        P[numP] = P[numP].fillna(0)
        for c in numP:
            P[c] = P[c].astype(int)

        # -------------------------------------------------------------- #
        # OUTPUT 4 — infection definitions (strict index candidates)     #
        # -------------------------------------------------------------- #
        hi_excl = int((S["osteo"] == 1).sum())
        se_excl = int(((S["osteo"] == 1) | (S["jinf"] == 1)).sum())
        life = con.execute("""
            SELECT COUNT(DISTINCT CASE WHEN knee_osteomyelitis=1 THEN empi_anon END) osteo,
                   COUNT(DISTINCT CASE WHEN joint_infection=1   THEN empi_anon END) jinf,
                   COUNT(DISTINCT CASE WHEN knee_osteomyelitis=1 OR joint_infection=1 THEN empi_anon END) either
            FROM icd""").df().iloc[0]
        inf_rows = [
            ("high_specificity_knee_osteomyelitis", hi_excl, _pct(hi_excl, n_strict),
             int(life["osteo"])),
            ("sensitivity_osteo_or_jointinf", se_excl, _pct(se_excl, n_strict),
             int(life["either"])),
        ]
        inf = pd.DataFrame(inf_rows, columns=["definition", "n_excluded_strict",
                                              "pct_of_strict", "lifetime_distinct_patient_ceiling"])
        inf.insert(0, "label", PRELIM)
        inf.insert(2, "n_strict_candidates", n_strict)
        _no_empi(inf, "infection_defs")
        inf.to_csv(tables_dir / "stage1_infection_defs.csv", index=False)
        headline["infection"] = {
            "n_strict_candidates": n_strict,
            "high_specificity_excluded": hi_excl, "high_specificity_pct": _pct(hi_excl, n_strict),
            "sensitivity_excluded": se_excl, "sensitivity_pct": _pct(se_excl, n_strict),
            "lifetime_knee_osteomyelitis": int(life["osteo"]),
            "lifetime_joint_infection": int(life["jinf"]),
            "lifetime_either": int(life["either"]),
        }

        # -------------------------------------------------------------- #
        # OUTPUT 5 — PRELIMINARY flow (the money table)                  #
        # -------------------------------------------------------------- #
        def run_flow(F: pd.DataFrame, n_index: int):
            """Apply S5..S11 filters sequentially; return dict of step -> (n, mask)."""
            steps = {}
            m = pd.Series(True, index=F.index)
            steps["S4"] = int(m.sum())
            m = m & (F["index_age"] >= age_min); steps["S5"] = int(m.sum())
            m = m & (F["prior_contra_cpt"] == 0) & (F["prior_contra_img"] == 0); steps["S6"] = int(m.sum())
            m7a = m & (F["osteo"] == 0); steps["S7a"] = int(m7a.sum())
            m7b = m & (F["osteo"] == 0) & (F["jinf"] == 0); steps["S7b"] = int(m7b.sum())
            m = m7a  # carry HIGH-SPECIFICITY infection exclusion forward
            m = m & (F["elig_img"] == 1); steps["S8"] = int(m.sum())
            m = m & (F["contra_0_90"] == 0); steps["S9"] = int(m.sum())
            m = m & (F["obs_ok"] == 1); steps["S10"] = int(m.sum())
            steps["S11"] = int(m.sum())
            return steps, m

        stepsS, maskS = run_flow(S, n_strict)
        stepsP, maskP = run_flow(P, n_permissive)
        final_strict = S[maskS].copy()
        final_perm = P[maskP].copy()

        flow_defs = [
            ("S0", "total demographics patients", n_demo, n_demo),
            ("S1", "with >=1 knee radiograph (image rows -> patients)", n_img_pat, n_img_pat),
            ("S2", "with >=1 CPT 27447", n_27447, n_27447),
            ("S3", "with >=1 single-side (laterality-coded) 27447", n_any_single, n_any_single),
            ("S4", "provisional index (strict: earliest 27447 single-side; permissive: earliest single-side 27447, no prior-arthroplasty evidence)",
             stepsS["S4"], stepsP["S4"]),
            ("S5", "+ age >= %.0f at index" % age_min, stepsS["S5"], stepsP["S5"]),
            ("S6", "- prior contralateral arthroplasty (pre-index contra CPT or contra/B image prosthesis)",
             stepsS["S6"], stepsP["S6"]),
            ("S7a", "- infection high_specificity (MAX knee_osteomyelitis in 365d pre-index) [carried forward]",
             stepsS["S7a"], stepsP["S7a"]),
            ("S7b", "- infection sensitivity (MAX osteo OR joint_infection in 365d pre-index) [alternative, NOT carried]",
             stepsS["S7b"], stepsP["S7b"]),
            ("S8", "with >=1 eligible pre-index contralateral image in 1-365d (contra laterality OR B-frontal)",
             stepsS["S8"], stepsP["S8"]),
            ("S9", "- contralateral 27447 within day 0-90", stepsS["S9"], stepsP["S9"]),
            ("S10", "observed through day 90 (last_observation across cpt/icd/pain/image > index+90)",
             stepsS["S10"], stepsP["S10"]),
            ("S11", "provisional final landmark cohort", stepsS["S11"], stepsP["S11"]),
        ]
        flow = pd.DataFrame(flow_defs, columns=["step", "description", "n_strict", "n_permissive"])
        flow.insert(0, "label", PRELIM)
        _no_empi(flow, "prelim_flow")
        flow.to_csv(tables_dir / "stage1_prelim_flow.csv", index=False)
        headline["flow_strict"] = {r[0]: r[2] for r in flow_defs}
        headline["flow_permissive"] = {r[0]: r[3] for r in flow_defs}

        n_cohort = int(len(final_strict))
        n_cohort_perm = int(len(final_perm))

        # -------------------------------------------------------------- #
        # OUTPUT 6 — event counts (provisional strict landmark cohort)   #
        # -------------------------------------------------------------- #
        n_primary = int((final_strict["primary_event"] == 1).sum())
        n_upper = int((final_strict["upper_event"] == 1).sum())
        ev_rows = [
            ("primary_strict_contralateral", "day91_to_5y", n_primary, n_cohort,
             _pct(n_primary, n_cohort)),
            ("upper_bound_any_modifier", "day91_to_5y", n_upper, n_cohort,
             _pct(n_upper, n_cohort)),
            ("blank_modifier_capture_gap", "day91_to_5y", n_upper - n_primary, n_cohort,
             _pct(n_upper - n_primary, n_cohort)),
        ]
        for v in list(sec_windows.get("from_day1", [])):
            c = int((final_strict[f"sec_d1_{v}"] == 1).sum())
            ev_rows.append((f"secondary_strict_from_day1", f"1..{v}d", c, n_cohort, _pct(c, n_cohort)))
        for v in list(sec_windows.get("from_day91", [])):
            c = int((final_strict[f"sec_d91_{v}"] == 1).sum())
            ev_rows.append((f"secondary_strict_from_day91", f"91..{v}d", c, n_cohort, _pct(c, n_cohort)))
        ev = pd.DataFrame(ev_rows, columns=["definition", "window", "n_events", "n_cohort", "pct"])
        ev.insert(0, "label", PRELIM)
        _no_empi(ev, "event_counts")
        ev.to_csv(tables_dir / "stage1_event_counts.csv", index=False)
        headline["events"] = {
            "n_cohort_strict": n_cohort,
            "primary_5y": n_primary, "primary_pct": _pct(n_primary, n_cohort),
            "upper_bound_5y": n_upper, "capture_gap": n_upper - n_primary,
            "secondary_from_day1": {str(v): int((final_strict[f"sec_d1_{v}"] == 1).sum())
                                    for v in list(sec_windows.get("from_day1", []))},
            "secondary_from_day91": {str(v): int((final_strict[f"sec_d91_{v}"] == 1).sum())
                                     for v in list(sec_windows.get("from_day91", []))},
            "n_cohort_permissive": n_cohort_perm,
            "primary_5y_permissive": int((final_perm["primary_event"] == 1).sum()),
        }

        # -------------------------------------------------------------- #
        # OUTPUT 7 — subgroup preview (provisional strict cohort)        #
        # -------------------------------------------------------------- #
        demo = con.execute("SELECT empi_anon, sex, race FROM demographics").df()
        obes = con.execute("SELECT empi_anon, MAX(obesity) obesity_ever FROM icd GROUP BY 1").df()
        cohort = final_strict.merge(demo, on="empi_anon", how="left") \
                             .merge(obes, on="empi_anon", how="left")
        cohort["obesity_ever"] = cohort["obesity_ever"].fillna(0).astype(int)
        sg = cfg["subgroups"]
        rgr = sg["race_groups"]
        bins = sg["event_flag_bins"]

        def flag_bin(n):
            if n < bins["sparse"]:
                return "<50"
            if n < bins["moderate"]:
                return "50-99"
            return ">=100"

        def sub_row(name, mask):
            sub = cohort[mask]
            n_p = int(len(sub))
            n_e = int((sub["primary_event"] == 1).sum())
            return (name, n_p, n_e, flag_bin(n_e))

        sg_rows = [
            sub_row("sex_Female", cohort["sex"] == "Female"),
            sub_row("sex_Male", cohort["sex"] == "Male"),
            sub_row("age_lt_%d" % age_cut, cohort["index_age"] < age_cut),
            sub_row("age_ge_%d" % age_cut, cohort["index_age"] >= age_cut),
            sub_row("race_Black", cohort["race"] == rgr["black"]),
            sub_row("race_White", cohort["race"] == rgr["white"]),
            sub_row("race_Asian", cohort["race"] == rgr["asian"]),
            sub_row("obesity_yes", cohort["obesity_ever"] == 1),
            sub_row("obesity_no", cohort["obesity_ever"] == 0),
            sub_row("weight_bearing_preindex", cohort["wb_pre"] == 1),
            sub_row("non_weight_bearing_preindex", cohort["wb_pre"] == 0),
            sub_row("frontal_only_preindex",
                    (cohort["ndistview_pre"] == 1) & (cohort["has_frontal_pre"] == 1)),
            sub_row("multiview_preindex", cohort["ndistview_pre"] >= 2),
        ]
        sgt = pd.DataFrame(sg_rows, columns=["subgroup", "n_patients", "n_primary_events", "event_flag"])
        sgt.insert(0, "label", PRELIM)
        _no_empi(sgt, "subgroup_preview")
        sgt.to_csv(tables_dir / "stage1_subgroup_preview.csv", index=False)
        headline["subgroups"] = {r[0]: {"n": r[1], "events": r[2], "flag": r[3]} for r in sg_rows}

        # -------------------------------------------------------------- #
        # OUTPUT 8 — machine-readable JSON (all headline numbers)        #
        # -------------------------------------------------------------- #
        test_alloc = int(round(0.20 * n_primary))
        headline["floors"] = {
            "primary_events": n_primary,
            "primary_events_min": int(floors["primary_events_min"]),
            "primary_events_meets_floor": bool(n_primary >= int(floors["primary_events_min"])),
            "test_allocatable_est_20pct": test_alloc,
            "test_allocatable_min": int(floors["test_allocatable_min"]),
            "test_allocatable_meets_floor": bool(test_alloc >= int(floors["test_allocatable_min"])),
            "upper_bound_primary_events": n_upper,
            "upper_bound_meets_floor": bool(n_upper >= int(floors["primary_events_min"])),
        }

        # -------------------------------------------------------------- #
        # OUTPUT 9 — DECISION ANCHORS (five auditability quantities)     #
        # Re-derived from the SAME in-memory strict-cohort structures    #
        # built above (df447 / per / strict / strict_idx / S / stepsS /  #
        # idx_strict). NO new cohort logic, NO changed existing value —  #
        # aggregates only, nothing patient-level is written out.         #
        # -------------------------------------------------------------- #
        # Q1 — absolute laterality-confirmed contralateral-TKA ceiling:
        # distinct 27447 patients with BOTH an R-side and an L-side single
        # record (any interval). Persist BOTH the exact-token and the
        # parse_modifier definitions to remove ambiguity.
        _da = df447[["empi_anon", "side", "flag"]].copy()
        _da["rt_tok"] = _da["flag"] == "single_rt"
        _da["lt_tok"] = _da["flag"] == "single_lt"
        _da["r_side"] = _da["side"] == "R"
        _da["l_side"] = _da["side"] == "L"
        _dag = _da.groupby("empi_anon").agg(rt=("rt_tok", "max"), lt=("lt_tok", "max"),
                                            r=("r_side", "max"), l=("l_side", "max"))
        ceiling_exact_token = int((_dag["rt"] & _dag["lt"]).sum())
        ceiling_parse_modifier = int((_dag["r"] & _dag["l"]).sum())

        # Q3 — same-day companion-line population (Decision A): STRICT index
        # patients whose earliest-27447 DATE carries exactly one single-side
        # record PLUS >=1 same-day blank/NULL-modifier 27447 companion line.
        # parse_modifier single-side (module-native) is the definitive count;
        # exact-token-only is reported for reconciliation of the memo gap.
        _ev = df447.merge(per["first_date"].rename("first_date"),
                          left_on="empi_anon", right_index=True)
        _earl = _ev[_ev["date_anon"] == _ev["first_date"]]
        _eg = _earl.groupby("empi_anon").agg(
            n_single_parse=("is_single", "sum"),
            n_single_exact=("flag", lambda s: int(s.isin(["single_rt", "single_lt"]).sum())),
            n_blank=("flag", lambda s: int((s == "missing").sum())),
        ).reindex(strict.index).fillna(0)
        same_day_companion_strict = int(((_eg["n_single_parse"] == 1) & (_eg["n_blank"] >= 1)).sum())
        same_day_companion_exact = int(((_eg["n_single_exact"] == 1) & (_eg["n_blank"] >= 1)).sum())

        # Q2 — pre-gate contralateral 5-year event ceiling among STRICT
        # age>=40 index patients (S5), i.e. a contralateral laterality-coded
        # 27447 in (90, 1826] days from index BEFORE applying gates S6-S10.
        # Reuses the per-patient primary_event flag already on S (all strict).
        _S5 = S[S["index_age"] >= age_min]
        pregate_contra_5y = int((_S5["primary_event"] == 1).sum())

        # Q4/Q5 — pre-index imaging windows among STRICT age>=40 (S5), one
        # image scan over the same idx_strict temp table. Q4 eligible-contra
        # image (contra laterality OR B-frontal) over widening pre-index
        # windows; Q5 ANY image (any laterality/view) in [index-365, index-1].
        _elig = "(g.laterality = t.contra_side OR (g.laterality = 'B' AND g.view_position = 'F'))"
        _win = con.execute(f"""
            SELECT t.empi_anon,
              MAX(CASE WHEN g.StudyDate_anon BETWEEN t.index_date - INTERVAL 365 DAY
                        AND t.index_date - INTERVAL 1 DAY AND {_elig} THEN 1 ELSE 0 END) e365,
              MAX(CASE WHEN g.StudyDate_anon BETWEEN t.index_date - INTERVAL 730 DAY
                        AND t.index_date - INTERVAL 1 DAY AND {_elig} THEN 1 ELSE 0 END) e730,
              MAX(CASE WHEN g.StudyDate_anon BETWEEN t.index_date - INTERVAL 1095 DAY
                        AND t.index_date - INTERVAL 1 DAY AND {_elig} THEN 1 ELSE 0 END) e1095,
              MAX(CASE WHEN g.StudyDate_anon < t.index_date AND {_elig} THEN 1 ELSE 0 END) elife,
              MAX(CASE WHEN g.StudyDate_anon BETWEEN t.index_date - INTERVAL 365 DAY
                        AND t.index_date - INTERVAL 1 DAY THEN 1 ELSE 0 END) any365
            FROM idx_strict t JOIN image g ON g.empi_anon = t.empi_anon
            GROUP BY 1""").df()
        _W = strict_idx.merge(_win, on="empi_anon", how="left")
        for _c in ("e365", "e730", "e1095", "elife", "any365"):
            _W[_c] = _W[_c].fillna(0).astype(int)
        _W5 = _W[_W["index_age"] >= age_min]
        n_s5 = int(len(_W5))
        n365 = int((_W5["e365"] == 1).sum())
        n730 = int((_W5["e730"] == 1).sum())
        n1095 = int((_W5["e1095"] == 1).sum())
        nlife = int((_W5["elife"] == 1).sum())
        q5n = int((_W5["any365"] == 1).sum())
        q5pct = _pct(q5n, n_s5)
        contra_elig_pct = _pct(stepsS["S8"], stepsS["S5"])  # S8/S5 contralateral-eligible rate

        def _incr(n):
            return round(100.0 * (n - n365) / n365, 3) if n365 else 0.0

        iw_rows = [
            ("365", n365, 0.0),
            ("730", n730, _incr(n730)),
            ("1095", n1095, _incr(n1095)),
            ("lifetime", nlife, _incr(nlife)),
        ]
        iw = pd.DataFrame(iw_rows,
                          columns=["window_days", "n_with_contralateral_image", "pct_increase_over_365"])
        iw.insert(0, "label", PRELIM)
        _no_empi(iw, "imaging_window_sensitivity")
        iw.to_csv(tables_dir / "stage1_imaging_window_sensitivity.csv", index=False)

        # self-consistency (additive asserts; do NOT affect existing outputs)
        assert ceiling_parse_modifier >= ceiling_exact_token, "parse ceiling < exact ceiling"
        assert ceiling_parse_modifier >= pregate_contra_5y >= n_primary, (
            f"anchor monotonicity broken: ceiling {ceiling_parse_modifier} >= "
            f"pregate {pregate_contra_5y} >= primary {n_primary}")
        assert n365 <= n730 <= n1095 <= nlife, \
            f"imaging windows not monotone: {[n365, n730, n1095, nlife]}"
        _elig_s5 = int((S.loc[S["index_age"] >= age_min, "elig_img"] == 1).sum())
        assert n365 == _elig_s5, f"window[365] {n365} != existing elig_img|S5 {_elig_s5}"

        headline["decision_anchors"] = {
            "label": PRELIM,
            "note": ("Five decision-critical quantities re-derived from the same strict-cohort "
                     "in-memory structures (df447/per/S/stepsS). Aggregates only; no cohort "
                     "computation or existing value was changed."),
            "ceiling_exact_token": ceiling_exact_token,
            "ceiling_parse_modifier": ceiling_parse_modifier,
            "pregate_contra_5y_events_strict_age40": pregate_contra_5y,
            "same_day_companion_strict": same_day_companion_strict,
            "same_day_companion_strict_exact_token": same_day_companion_exact,
            "any_preindex_radiograph_1yr_n": q5n,
            "any_preindex_radiograph_1yr_pct": q5pct,
            "contralateral_eligible_1yr_pct": contra_elig_pct,
            "imaging_window_sensitivity_lever1": [
                {"window_days": w, "n_with_contralateral_image": n, "pct_increase_over_365": p}
                for (w, n, p) in iw_rows
            ],
            "_definitions": {
                "ceiling_exact_token":
                    "distinct 27447 patients with >=1 exact-token RT (single_rt) AND >=1 "
                    "exact-token LT (single_lt) record, any interval",
                "ceiling_parse_modifier":
                    "distinct 27447 patients with >=1 parse_modifier single-side R AND >=1 "
                    "single-side L record (includes multi_single_side e.g. 'RT XP'), any interval",
                "pregate_contra_5y_events_strict_age40":
                    "STRICT age>=40 index patients (S5=%d) with a contralateral laterality-coded "
                    "27447 in (90,1826] days from index, BEFORE the S6-S10 gates" % stepsS["S5"],
                "same_day_companion_strict":
                    "STRICT index patients (S4=%d) whose earliest 27447 date has exactly one "
                    "parse_modifier single-side (RT xor LT) record PLUS >=1 same-day blank/NULL-"
                    "modifier 27447 companion line (module-native single-side definition; definitive)"
                    % stepsS["S4"],
                "same_day_companion_strict_exact_token":
                    "as same_day_companion_strict but counting only exact-token single_rt/single_lt "
                    "as the single-side record; the %d-vs-%d gap is multi_single_side ('RT XP'-type) "
                    "earliest lines (resolves the 2604-vs-2597 memo discrepancy)"
                    % (same_day_companion_strict, same_day_companion_exact),
                "any_preindex_radiograph_1yr_n":
                    "STRICT age>=40 index patients (S5=%d) with ANY image (any laterality/view) with "
                    "StudyDate in [index-365, index-1]" % stepsS["S5"],
                "any_preindex_radiograph_1yr_pct": "any_preindex_radiograph_1yr_n / S5",
                "contralateral_eligible_1yr_pct":
                    "S8/S5 = %d/%d: contralateral-specific eligible pre-index image rate (contra "
                    "laterality OR B-frontal, 1-365d), for contrast with the any-radiograph rate"
                    % (stepsS["S8"], stepsS["S5"]),
                "imaging_window_sensitivity_lever1":
                    "STRICT age>=40 (S5=%d) with >=1 eligible contralateral pre-index image (contra "
                    "laterality OR B-frontal) per widening pre-index window; pct_increase_over_365 "
                    "vs the 1-365d count; also written to stage1_imaging_window_sensitivity.csv"
                    % stepsS["S5"],
            },
        }
        log.info("%s: decision anchors — ceiling exact=%d parse=%d; pregate-5y(S5)=%d; "
                 "same_day_companion=%d (exact=%d); any-radiograph-1yr=%d/%d (%.1f%%); "
                 "contra-eligible-1yr=%.1f%%; img-windows 365/730/1095/life=%d/%d/%d/%d",
                 PRELIM, ceiling_exact_token, ceiling_parse_modifier, pregate_contra_5y,
                 same_day_companion_strict, same_day_companion_exact, q5n, n_s5, q5pct,
                 contra_elig_pct, n365, n730, n1095, nlife)

        json_path = outputs_dir / "feasibility_stage1_counts.json"
        with open(json_path, "w") as fh:
            json.dump(headline, fh, indent=2, default=str)
        log.info("%s: wrote %s", PRELIM, json_path)

        # -------------------------------------------------------------- #
        # Reconciliation asserts (fail loud).                            #
        # -------------------------------------------------------------- #
        assert n_demo >= n_img_pat >= n_27447 >= n_any_single >= n_strict, "base populations not monotone"
        assert n_strict + n_earlier_unsided == n_any_single, "strict/earlier-unsided partition broken"
        seqS = [stepsS[k] for k in ("S4", "S5", "S6", "S7a", "S8", "S9", "S10", "S11")]
        assert all(seqS[i] >= seqS[i + 1] for i in range(len(seqS) - 1)), f"strict flow not monotone: {seqS}"
        seqP = [stepsP[k] for k in ("S4", "S5", "S6", "S7a", "S8", "S9", "S10", "S11")]
        assert all(seqP[i] >= seqP[i + 1] for i in range(len(seqP) - 1)), f"perm flow not monotone: {seqP}"
        assert n_upper >= n_primary, "upper-bound events < strict primary events"
        # from_day91 5y window must equal the primary strict event count
        big91 = max(list(sec_windows.get("from_day91", [horizon_days])))
        assert int((final_strict[f"sec_d91_{big91}"] == 1).sum()) == n_primary, \
            "from_day91 5y window != primary strict events"
        log.info("%s: reconciliation OK (strict flow %s; primary=%d upper=%d cohort=%d)",
                 PRELIM, seqS, n_primary, n_upper, n_cohort)
        log.info("%s: floors — primary %d/%d (%s); test-allocatable %d/%d (%s)",
                 PRELIM, n_primary, int(floors["primary_events_min"]),
                 "PASS" if n_primary >= int(floors["primary_events_min"]) else "FAIL",
                 test_alloc, int(floors["test_allocatable_min"]),
                 "PASS" if test_alloc >= int(floors["test_allocatable_min"]) else "FAIL")
        log.info("%s: DONE — wrote 8 tables + JSON to %s", PRELIM, tables_dir)
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
