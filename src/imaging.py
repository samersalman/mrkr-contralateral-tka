"""imaging.py — LOCKED cohort extraction: eligible pre-index contralateral STUDY selection.

MRKR Contralateral TKA — Phase-1 feasibility, LOCKED EXTRACTION. For every patient in
``derived-data/cohort/index_final.parquet`` (6,381 patients), this module selects ONE
eligible pre-index contralateral radiographic STUDY, applying the protocol section-8
preference order. It selects STUDY METADATA ONLY: no DICOM pixels are read, no models
are trained, no metrics are computed.

REUSE, DO NOT FORK. The contralateral-eligibility predicate is byte-for-byte the verified
one used in ``src.preliminary_counts.sql_image_flags`` (``elig_img``) and
``src.regate.sql_image_flags_windowed``: a pre-index image "contains the contralateral
knee" iff

    laterality = contra_side   OR   (laterality = 'B' AND view_position = 'F')

(a B-frontal is uncropped and contains both knees). This reproduces the S8 regression
anchor: among the STRICT subset (``side_source = 'coded'``, 3,752 patients), the count
with an eligible pre-index contralateral image at WINDOW = 365 days == 1,807.

Selection (protocol section 8 / config ``imaging_selection.preference_order``). Each
candidate study is assigned its BEST satisfied tier:

    tier1  wb_frontal_lateral_sunrise : a weight-bearing frontal AND a lateral AND a sunrise
    tier2  wb_frontal_plus_any        : a weight-bearing frontal AND >=1 additional (non-frontal) view
    tier3  frontal_only               : a weight-bearing frontal, no additional views
    tier4  nonwb_frontal              : a frontal is present but NONE is weight-bearing
    tier5  no_frontal                 : eligible (contralateral) but with NO frontal view at all

tiers 1-4 mirror the four config-declared preference names; tier5 is a documented residual
that is REQUIRED for internal consistency: the S8 eligibility rule counts a patient as
eligible on the strength of ANY contralateral image (a contralateral lateral/sunrise with
no frontal still passes S8), so those patients must be classifiable and selectable or the
selected-study count would fall below the S8 anchor. tier3/tier4 are split on whether the
frontal is weight-bearing (reading that makes tier4 reachable under first-satisfied-tier
evaluation and honours "weight-bearing frontal is the key predictor view").

Within the best available tier the MOST RECENT study is selected (largest study_date =
smallest days_to_index); exact same-tier same-date ties are broken deterministically by
(more images, then StudyInstanceUID_anon) so the selection is fully reproducible.

Guardrails honoured here:
  * Read-only on the typed Parquet inputs (image.parquet, index_final.parquet). No DICOMs,
    no models, no metrics; STUDY metadata only.
  * NO post-index leakage: eligible images are strictly PRE-index, StudyDate in
    [index_date - WINDOW, index_date - 1].
  * ``derived-data/cohort/*.parquet`` MAY carry empi_anon and dicom_path (git-ignored
    linkage tables). ``outputs/tables/*.csv`` is AGGREGATE only — no empi_anon, no paths.
  * Deterministic. Every parameter comes from ``config/feasibility.yaml``.
  * Logging APPENDS to ``outputs/logs/run.log`` with the ``imaging`` prefix.

Run from the project root::

    python3 -m src.imaging --config config/feasibility.yaml

Writes ONLY: ``derived-data/cohort/selected_studies.parquet``,
``derived-data/cohort/selected_study_images.parquet``,
``derived-data/cohort/candidate_studies.parquet``,
``outputs/tables/imaging_availability.csv``, and appends ``outputs/logs/run.log``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.config import Config, ensure_dirs, load_config
from src.preliminary_counts import _no_empi, _pct

MODULE = "imaging"

# Canonical view ordering for the human-readable view_set label (F/L/S then other).
VIEW_ORDER = ["frontal", "lateral", "sunrise", "other"]


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
# Helpers.                                                                     #
# --------------------------------------------------------------------------- #
def _view_set_label(view_codes: set[str], view_map: dict) -> str:
    """Map raw view_position codes to a canonical 'frontal+lateral+sunrise'-style label."""
    names = {view_map.get(code, "other") for code in view_codes}
    return "+".join(n for n in VIEW_ORDER if n in names)


def _write_parquet(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, path: Path) -> None:
    """Write a pandas frame to Parquet via DuckDB (stable type inference, incl. date32)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con.register("_writer_df", df)
    p = str(path).replace("'", "''")
    con.execute(f"COPY (SELECT * FROM _writer_df) TO '{p}' (FORMAT PARQUET)")
    con.unregister("_writer_df")


# --------------------------------------------------------------------------- #
# Candidate pull: eligible pre-index contralateral images (widest window).     #
# --------------------------------------------------------------------------- #
def pull_candidate_images(con: duckdb.DuckDBPyConnection, cfg: Config,
                          widest_window: int) -> pd.DataFrame:
    """One row per eligible pre-index contralateral IMAGE, out to the widest window.

    Reproduces the verified S8 eligibility predicate (contra laterality OR B-frontal),
    plus the explicit spec exclusions (unresolved laterality, contralateral prosthesis).
    The arthroplasty exclusion is a verified no-op for index_final patients — they have
    already passed the patient-level prior-contralateral-prosthesis gate (S6) — but it is
    applied at image grain exactly as the spec states.
    """
    im = cfg["image"]
    frontal = im["frontal_code"]                 # 'F'
    bilateral = im["bilateral_code"]             # 'B'
    unknown = im["unknown_codes"][0]             # '-1'

    elig = (f"(g.laterality = t.contra_side "
            f"OR (g.laterality = '{bilateral}' AND g.view_position = '{frontal}'))")
    where = (
        f"g.StudyDate_anon BETWEEN t.index_date - INTERVAL {int(widest_window)} DAY "
        f"AND t.index_date - INTERVAL 1 DAY "
        f"AND {elig} "
        f"AND g.laterality <> '{unknown}' "
        f"AND g.arthroplasty NOT IN (t.contra_side, '{bilateral}')"
    )
    df = con.execute(f"""
        SELECT t.empi_anon, t.index_date, t.contra_side, t.side_source,
               g.StudyInstanceUID_anon AS study_uid,
               g.SOPInstanceUID_anon   AS sop_uid,
               g.StudyDate_anon        AS study_date,
               g.view_position, g.laterality, g.weight_bearing, g.arthroplasty,
               g.dicom_path
        FROM idx t
        JOIN image g ON g.empi_anon = t.empi_anon
        WHERE {where}
    """).df()
    df["index_date"] = pd.to_datetime(df["index_date"])
    df["study_date"] = pd.to_datetime(df["study_date"])
    df["days_to_index"] = (df["index_date"] - df["study_date"]).dt.days.astype(int)
    return df


# --------------------------------------------------------------------------- #
# Study grouping + tier classification.                                        #
# --------------------------------------------------------------------------- #
def build_study_table(cand: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Group eligible candidate images into STUDIES and classify each study's best tier."""
    im = cfg["image"]
    frontal = im["frontal_code"]
    bilateral = im["bilateral_code"]
    view_map = im["view_map"]
    wb_true = int(im["weight_bearing_true"])
    pref_names = [p["name"] for p in cfg["imaging_selection"]["preference_order"]]
    tier_name = {1: pref_names[0], 2: pref_names[1], 3: pref_names[2], 4: pref_names[3],
                 5: "no_frontal"}

    rows = []
    for (empi, suid), g in cand.groupby(["empi_anon", "study_uid"], sort=False):
        views = set(g["view_position"])
        is_frontal = g["view_position"] == frontal
        is_wb = g["weight_bearing"] == wb_true
        has_frontal = frontal in views
        has_lateral = "L" in views
        has_sunrise = "S" in views
        has_extra_view = len(views - {frontal}) > 0
        wb_frontal = bool((is_frontal & is_wb).any())
        any_wb = bool(is_wb.any())
        laterality_kind = ("bilateral_B" if (g["laterality"] == bilateral).any()
                           else "unilateral_contra")
        study_date = g["study_date"].min()               # a study has one date (rare 2-date -> earliest)
        index_date = g["index_date"].iloc[0]
        days_to_index = int((index_date - study_date).days)

        if wb_frontal and has_lateral and has_sunrise:
            tier = 1
        elif wb_frontal and has_extra_view:
            tier = 2
        elif wb_frontal:
            tier = 3
        elif has_frontal:
            tier = 4
        else:
            tier = 5

        rows.append({
            "empi_anon": empi,
            "study_uid": suid,
            "side_source": g["side_source"].iloc[0],
            "study_date": study_date.date(),
            "days_to_index": days_to_index,
            "tier": tier,
            "tier_name": tier_name[tier],
            "view_set": _view_set_label(views, view_map),
            "weight_bearing_frontal": wb_frontal,
            "any_weight_bearing": any_wb,
            "has_frontal": has_frontal,
            "has_lateral": has_lateral,
            "has_sunrise": has_sunrise,
            "laterality_kind": laterality_kind,
            "n_images": int(len(g)),
        })

    studies = pd.DataFrame(rows)
    studies["in_window_365"] = studies["days_to_index"] <= 365
    studies["in_window_730"] = studies["days_to_index"] <= 730
    return studies


def select_primary_study(studies: pd.DataFrame, primary_hi: int) -> pd.DataFrame:
    """Best-tier, most-recent study per patient within the primary pre-index window."""
    pool = studies[studies["days_to_index"] <= primary_hi].copy()
    # tier asc (best first), days_to_index asc (most recent first), n_images desc (richer),
    # study_uid asc (final deterministic tiebreak).
    pool = pool.sort_values(
        ["empi_anon", "tier", "days_to_index", "n_images", "study_uid"],
        ascending=[True, True, True, False, True], kind="mergesort")
    return pool.groupby("empi_anon", sort=False).head(1).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main.                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="LOCKED: select one eligible pre-index contralateral study per patient.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    cohort_dir = cfg.out("cohort_dir")
    tables_dir = cfg.out("tables_dir")

    # ---- windows (data-driven) --------------------------------------------- #
    primary_lo, primary_hi = cfg["primary_definition"]["pre_index_window_days"]  # [1, 730]
    sens_windows = cfg["primary_definition"]["sensitivities"]["windows"]          # [[1,365],[1,1095]]
    window_his = sorted({int(primary_hi)} | {int(w[1]) for w in sens_windows})    # [365, 730, 1095]
    widest = max(window_his)
    assert int(primary_lo) == 1, f"pre-index window must start at day 1, got {primary_lo}"

    log.info("START LOCKED contralateral study selection (primary window=[1,%d]; "
             "sensitivity windows=%s; widest=%d)", primary_hi, window_his, widest)

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")

    # ---- read-only views on the typed Parquet ------------------------------ #
    img_path = str(cfg.parquet_path("image")).replace("'", "''")
    idx_path = str((cohort_dir / "index_final.parquet")).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW image AS SELECT * FROM read_parquet('{img_path}')")
    con.execute(f"CREATE OR REPLACE VIEW idx AS SELECT * FROM read_parquet('{idx_path}')")

    index_final = con.execute(
        "SELECT empi_anon, side_source FROM idx").df()
    n_index = int(len(index_final))
    log.info("index_final roster: %d patients (coded=%d, recovered=%d)",
             n_index,
             int((index_final["side_source"] == "coded").sum()),
             int((index_final["side_source"] == "recovered").sum()))

    # ---- candidate eligible images -> study table -------------------------- #
    cand = pull_candidate_images(con, cfg, widest)
    log.info("eligible candidate images (window<=%d): %d images | %d patients | %d studies",
             widest, len(cand), cand["empi_anon"].nunique(), cand["study_uid"].nunique())
    studies = build_study_table(cand, cfg)

    # ---- per-patient presence flags at each window ------------------------- #
    min_days = studies.groupby("empi_anon")["days_to_index"].min()
    pres = {hi: set(min_days[min_days <= hi].index) for hi in window_his}
    coded_empis = set(index_final.loc[index_final["side_source"] == "coded", "empi_anon"])
    reco_empis = set(index_final.loc[index_final["side_source"] == "recovered", "empi_anon"])

    def _n(hi: int, subset: set | None) -> int:
        s = pres[hi] if subset is None else (pres[hi] & subset)
        return len(s)

    presence = {
        "all": {hi: _n(hi, None) for hi in window_his},
        "coded": {hi: _n(hi, coded_empis) for hi in window_his},
        "recovered": {hi: _n(hi, reco_empis) for hi in window_his},
    }

    # ---- REGRESSION ANCHOR + monotonicity asserts -------------------------- #
    anchor = presence["coded"][365]
    assert anchor == 1807, (
        f"S8 anchor broken: coded-subset eligible@365 = {anchor} != 1807 "
        "(eligibility logic diverged from preliminary_counts)")
    for strat, d in presence.items():
        seq = [d[hi] for hi in window_his]
        assert seq == sorted(seq), f"imaging windows not monotone for {strat}: {seq}"
    log.info("ANCHOR OK: coded eligible@365 = %d (== 1807). Presence all 365/730/1095 = %d/%d/%d",
             anchor, presence["all"][365], presence["all"][730], presence["all"][1095])

    # ---- select ONE primary study per patient (window [1,730]) ------------- #
    selected = select_primary_study(studies, int(primary_hi))
    for hi in window_his:
        selected[f"has_eligible_study_{hi}"] = selected["empi_anon"].isin(pres[hi])
    assert selected["empi_anon"].is_unique, "selected_studies has duplicate patients"
    n_selected = int(len(selected))
    assert n_selected == presence["all"][primary_hi], (
        f"selected patients {n_selected} != eligible@{primary_hi} "
        f"{presence['all'][primary_hi]} (selection dropped eligible patients)")
    assert bool(selected[f"has_eligible_study_{primary_hi}"].all()), \
        "a selected patient is not flagged eligible at the primary window"
    log.info("selected primary studies: %d patients (== eligible@%d)", n_selected, primary_hi)

    # ---- image grain for the selected studies ------------------------------ #
    sel_img = cand.merge(selected[["empi_anon", "study_uid"]],
                         on=["empi_anon", "study_uid"], how="inner")
    sel_img_out = pd.DataFrame({
        "empi_anon": sel_img["empi_anon"],
        "StudyInstanceUID_anon": sel_img["study_uid"],
        "SOPInstanceUID_anon": sel_img["sop_uid"],
        "view_position": sel_img["view_position"],
        "laterality": sel_img["laterality"],
        "weight_bearing": sel_img["weight_bearing"].astype(int),
        "arthroplasty": sel_img["arthroplasty"],
        "dicom_path": sel_img["dicom_path"],
        "StudyDate_anon": sel_img["study_date"].dt.date,
        "days_to_index": sel_img["days_to_index"].astype(int),
    })

    # ---- assemble output frames (study grain) ------------------------------ #
    study_cols = ["empi_anon", "StudyInstanceUID_anon", "study_date", "days_to_index",
                  "tier", "tier_name", "view_set", "weight_bearing_frontal",
                  "any_weight_bearing", "has_frontal", "has_lateral", "has_sunrise",
                  "laterality_kind", "n_images", "side_source"]

    candidate_out = studies.rename(columns={"study_uid": "StudyInstanceUID_anon"})
    candidate_out = candidate_out[study_cols + ["in_window_365", "in_window_730"]].copy()
    candidate_out = candidate_out.sort_values(
        ["empi_anon", "days_to_index", "tier", "StudyInstanceUID_anon"]).reset_index(drop=True)

    selected_out = selected.rename(columns={"study_uid": "StudyInstanceUID_anon"})
    selected_out = selected_out[
        study_cols + [f"has_eligible_study_{hi}" for hi in window_his]].copy()
    selected_out = selected_out.sort_values("empi_anon").reset_index(drop=True)

    # ---- persist linkage parquets (empi_anon / dicom_path allowed here) ----- #
    _write_parquet(con, selected_out, cohort_dir / "selected_studies.parquet")
    _write_parquet(con, sel_img_out, cohort_dir / "selected_study_images.parquet")
    _write_parquet(con, candidate_out, cohort_dir / "candidate_studies.parquet")
    log.info("wrote selected_studies (%d rows), selected_study_images (%d rows), "
             "candidate_studies (%d rows)", len(selected_out), len(sel_img_out),
             len(candidate_out))

    # ---- AGGREGATE availability table (NO empi_anon, NO paths) -------------- #
    rows: list[dict] = []

    def add(section: str, metric: str, stratum: str, value) -> None:
        rows.append({"section": section, "metric": metric, "stratum": stratum, "value": value})

    add("cohort", "n_patients_index_final", "all", n_index)
    for strat in ("all", "coded", "recovered"):
        for hi in window_his:
            add("eligibility_presence", f"n_eligible_study_{hi}d", strat, presence[strat][hi])
    add("eligibility_presence", "primary_window_days", "all", int(primary_hi))
    add("eligibility_presence", "n_selected_primary", "all", n_selected)

    # tier distribution of the SELECTED primary studies
    tname = {1: "tier1", 2: "tier2", 3: "tier3", 4: "tier4", 5: "tier5"}
    for t in range(1, 6):
        add("selected_tier", f"{tname[t]}_{_tier_label(cfg, t)}", "all",
            int((selected["tier"] == t).sum()))

    # view-set distribution of SELECTED studies
    for vs, n in selected["view_set"].value_counts().sort_index().items():
        add("selected_view_set", vs, "all", int(n))

    # WB vs non-WB (weight-bearing frontal is the tier-relevant flag) + any-WB
    add("selected_weight_bearing", "weight_bearing_frontal_true", "all",
        int(selected["weight_bearing_frontal"].sum()))
    add("selected_weight_bearing", "weight_bearing_frontal_false", "all",
        int((~selected["weight_bearing_frontal"]).sum()))
    add("selected_weight_bearing", "any_weight_bearing_true", "all",
        int(selected["any_weight_bearing"].sum()))
    add("selected_weight_bearing", "any_weight_bearing_false", "all",
        int((~selected["any_weight_bearing"]).sum()))

    # frontal-only vs multi-view composition of SELECTED studies
    add("selected_composition", "multi_view_wb_frontal(tier1_2)", "all",
        int(selected["tier"].isin([1, 2]).sum()))
    add("selected_composition", "frontal_only_wb(tier3)", "all",
        int((selected["tier"] == 3).sum()))
    add("selected_composition", "nonwb_frontal(tier4)", "all",
        int((selected["tier"] == 4).sum()))
    add("selected_composition", "no_frontal(tier5)", "all",
        int((selected["tier"] == 5).sum()))

    # laterality kind of SELECTED studies
    for lk, n in selected["laterality_kind"].value_counts().sort_index().items():
        add("selected_laterality_kind", lk, "all", int(n))

    # days-to-index of SELECTED studies (median + IQR)
    dti = selected["days_to_index"]
    add("selected_days_to_index", "median", "all", float(dti.median()))
    add("selected_days_to_index", "q1", "all", float(dti.quantile(0.25)))
    add("selected_days_to_index", "q3", "all", float(dti.quantile(0.75)))
    add("selected_days_to_index", "min", "all", int(dti.min()))
    add("selected_days_to_index", "max", "all", int(dti.max()))

    # path availability over selected-study images
    n_img = int(len(sel_img_out))
    nonempty = int((sel_img_out["dicom_path"].notna()
                    & (sel_img_out["dicom_path"].astype(str).str.strip() != "")).sum())
    add("path_availability", "n_selected_study_images", "all", n_img)
    add("path_availability", "pct_images_nonempty_dicom_path", "all",
        _pct(nonempty, n_img))

    avail = pd.DataFrame(rows, columns=["section", "metric", "stratum", "value"])
    _no_empi(avail, "imaging_availability")
    avail_path = tables_dir / "imaging_availability.csv"
    avail.to_csv(avail_path, index=False)
    log.info("wrote %s (%d aggregate rows)", avail_path.name, len(avail))

    # ---- headline log ------------------------------------------------------ #
    tier_counts = {tname[t]: int((selected["tier"] == t).sum()) for t in range(1, 6)}
    log.info("PRIMARY(730d) selected=%d | tiers %s | wb_frontal=%d (%.1f%%) | "
             "days_to_index median=%.0f IQR[%.0f-%.0f] | dicom_path nonempty=%.1f%%",
             n_selected, tier_counts, int(selected["weight_bearing_frontal"].sum()),
             _pct(int(selected["weight_bearing_frontal"].sum()), n_selected),
             float(dti.median()), float(dti.quantile(0.25)), float(dti.quantile(0.75)),
             _pct(nonempty, n_img))
    log.info("DONE imaging module.")
    con.close()
    return 0


def _tier_label(cfg: Config, t: int) -> str:
    names = [p["name"] for p in cfg["imaging_selection"]["preference_order"]]
    return names[t - 1] if t <= 4 else "no_frontal"


if __name__ == "__main__":
    raise SystemExit(main())
