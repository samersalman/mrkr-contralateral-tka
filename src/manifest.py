"""manifest.py — image-transfer REVIEW manifest for the LOCKED final cohort.

Read-only on inputs. This module opens NO DICOMs, initiates NO transfer, and
touches NO network/git. It only filters the already-selected image metadata down
to the 3,709 final-cohort patients (their SELECTED StudyInstanceUID_anon) and
writes review tables for a human to approve BEFORE any transfer is arranged.

Inputs (Parquet, read-only):
  derived-data/cohort/final_cohort.parquet          (3,709 patients; the SELECTED study/patient)
  derived-data/cohort/selected_study_images.parquet (6,578 image rows across 3,981 selected patients)

Outputs (outputs/tables/):
  selected_studies.csv             STUDY grain  — one row per final-cohort patient
  image_transfer_manifest.csv      IMAGE grain  — one row per image of those selected studies
  image_transfer_manifest_paths.txt             — deduplicated dicom_path list, sorted
  manifest_summary.csv             AGGREGATE     — no ids/paths

NOTE (by design): selected_studies.csv and image_transfer_manifest.csv carry the
de-identified empi_anon + dicom_path (that IS their purpose per prompt-1 Stage 5);
they are git-ignored. Only manifest_summary.csv is aggregate. Do not aggregate the
per-image/per-study manifests away.

Run:  python3 -m src.manifest --config config/feasibility.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.config import load_config

MODULE = "manifest"


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


def _fmt_date(s: pd.Series, date_format: str) -> pd.Series:
    """Format a datetime series to YYYY-MM-DD strings; NaT -> '' (empty)."""
    out = s.dt.strftime(date_format)
    return out.where(s.notna(), "")


def _malformed_path_mask(paths: pd.Series) -> pd.Series:
    """A dicom_path is missing/malformed if it is null, blank, or not '*.dcm'."""
    p = paths.astype("string")
    blank = p.isna() | (p.str.strip() == "")
    not_dcm = ~p.str.endswith(".dcm").fillna(False)
    return blank | not_dcm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/feasibility.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    tables_dir = cfg.path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    date_format = cfg["date_format"]

    # Regression anchors sourced from config (LOCKED primary definition).
    exp_patients = int(cfg["primary_definition"]["verified_regate"]["n_final_cohort"])  # 3709
    lo, hi = cfg["primary_definition"]["pre_index_window_days"]                          # [1, 730]

    con = duckdb.connect()

    def load(name: str) -> pd.DataFrame:
        return con.execute(f"SELECT * FROM read_parquet('{coh / name}')").df()

    fc = load("final_cohort.parquet")                # 3,709 — one row per patient, SELECTED study
    img = load("selected_study_images.parquet")      # 6,578 image rows across 3,981 selected patients
    log.info("loaded final_cohort=%d (patients=%d) selected_study_images=%d (patients=%d studies=%d)",
             len(fc), fc["empi_anon"].nunique(), len(img),
             img["empi_anon"].nunique(), img["StudyInstanceUID_anon"].nunique())

    # ---- STUDY-level attributes to attach to the image grain (from final_cohort) --
    # Attach on BOTH keys so we keep ONLY images belonging to each patient's SELECTED study.
    fc_study_attrs = fc[["empi_anon", "StudyInstanceUID_anon", "index_side", "contra_side",
                         "index_date", "study_date", "view_set"]].copy()

    # ---- IMAGE-grain manifest: images of the final-cohort SELECTED studies ------
    man = img.merge(fc_study_attrs, on=["empi_anon", "StudyInstanceUID_anon"], how="inner")
    log.info("filtered image manifest: rows=%d patients=%d studies=%d images(SOP)=%d",
             len(man), man["empi_anon"].nunique(), man["StudyInstanceUID_anon"].nunique(),
             man["SOPInstanceUID_anon"].nunique())

    # ---- Coverage: every final-cohort patient must contribute >=1 selected image -
    fc_pairs = set(zip(fc["empi_anon"], fc["StudyInstanceUID_anon"]))
    man_pairs = set(zip(man["empi_anon"], man["StudyInstanceUID_anon"]))
    zero_image = fc_pairs - man_pairs
    if zero_image:
        for e, s in sorted(zero_image)[:20]:
            log.error("FINAL-COHORT PATIENT WITH 0 SELECTED IMAGES: empi=%s study=%s", e, s)
    assert not zero_image, f"{len(zero_image)} final-cohort patients have 0 selected images"

    # ---- REGRESSION / SANITY assertions ----------------------------------------
    n_patients = man["empi_anon"].nunique()
    assert n_patients == exp_patients, f"n_unique_patients={n_patients} != {exp_patients}"
    assert set(man["empi_anon"]).issubset(set(fc["empi_anon"])), "manifest patients NOT a subset of final_cohort"
    assert set(man["StudyInstanceUID_anon"]) == set(fc["StudyInstanceUID_anon"]), \
        "manifest studies != final_cohort selected StudyInstanceUID_anon set"
    assert man["days_to_index"].between(lo, hi).all(), \
        f"some image days_to_index outside [{lo},{hi}] (strictly pre-index)"
    lat_ok = (man["laterality"] == man["contra_side"]) | (man["laterality"] == "B")
    assert lat_ok.all(), "some image laterality is neither contra_side nor 'B'"

    # ---- Path missing/malformed rate (report; empty/null OR not ending in .dcm) -
    malformed = _malformed_path_mask(man["dicom_path"])
    pct_malformed = round(100.0 * float(malformed.mean()), 4)
    log.info("path check: %d/%d image rows missing/malformed (%.4f%%) [rule: null/blank OR not '*.dcm']",
             int(malformed.sum()), len(man), pct_malformed)

    # ---- Build IMAGE-grain output (exact column order per spec) -----------------
    man["final_cohort_member"] = True
    image_cols = ["empi_anon", "index_side", "contra_side", "index_date", "StudyInstanceUID_anon",
                  "SOPInstanceUID_anon", "study_date", "view_position", "laterality", "weight_bearing",
                  "arthroplasty", "dicom_path", "days_to_index", "view_set", "final_cohort_member"]
    image_out = man[image_cols].copy()
    image_out["index_date"] = _fmt_date(image_out["index_date"], date_format)
    image_out["study_date"] = _fmt_date(image_out["study_date"], date_format)
    image_out = image_out.sort_values(
        ["empi_anon", "StudyInstanceUID_anon", "SOPInstanceUID_anon"], kind="mergesort"
    ).reset_index(drop=True)

    # ---- Build STUDY-grain output (n_images derived from THIS manifest) ---------
    n_images = (man.groupby(["empi_anon", "StudyInstanceUID_anon"], sort=False)
                   .size().rename("n_images").reset_index())
    study_out = fc[["empi_anon", "index_side", "contra_side", "index_date", "StudyInstanceUID_anon",
                    "study_date", "days_to_index", "view_set", "weight_bearing_frontal",
                    "laterality_kind", "tier"]].merge(
        n_images, on=["empi_anon", "StudyInstanceUID_anon"], how="left")
    # Cross-check the manifest-derived count against the cohort's own n_images.
    if "n_images" in fc.columns:
        mism = fc[["empi_anon", "StudyInstanceUID_anon", "n_images"]].rename(
            columns={"n_images": "n_images_fc"}).merge(
            n_images, on=["empi_anon", "StudyInstanceUID_anon"])
        n_mismatch = int((mism["n_images_fc"] != mism["n_images"]).sum())
        log.info("n_images vs final_cohort.n_images mismatches: %d", n_mismatch)
    study_out["index_date"] = _fmt_date(study_out["index_date"], date_format)
    study_out["study_date"] = _fmt_date(study_out["study_date"], date_format)
    study_out = study_out.sort_values("empi_anon", kind="mergesort").reset_index(drop=True)
    assert len(study_out) == exp_patients, f"selected_studies rows={len(study_out)} != {exp_patients}"
    assert study_out["empi_anon"].is_unique, "selected_studies not one-row-per-patient"
    assert int(study_out["n_images"].sum()) == len(image_out), "study n_images sum != image manifest rows"

    # ---- Deduplicated, sorted dicom_path list (real paths only) -----------------
    paths = man["dicom_path"].astype("string")
    valid_paths = sorted(p for p in paths.dropna().unique() if str(p).strip() != "")
    n_unique_paths = len(valid_paths)

    # ---- AGGREGATE summary (no ids/paths) --------------------------------------
    summary = pd.DataFrame([{
        "n_unique_patients": int(n_patients),
        "n_unique_studies": int(man["StudyInstanceUID_anon"].nunique()),
        "n_unique_image_files": int(man["SOPInstanceUID_anon"].nunique()),
        "n_unique_paths": int(n_unique_paths),
        "pct_paths_missing_or_malformed": pct_malformed,
        "est_transfer_size": "N/A: file sizes unavailable in metadata",
    }])

    # ---- Write outputs ---------------------------------------------------------
    p_study = tables_dir / "selected_studies.csv"
    p_image = tables_dir / "image_transfer_manifest.csv"
    p_paths = tables_dir / "image_transfer_manifest_paths.txt"
    p_summary = tables_dir / "manifest_summary.csv"

    study_out.to_csv(p_study, index=False)
    image_out.to_csv(p_image, index=False)
    p_paths.write_text("\n".join(valid_paths) + ("\n" if valid_paths else ""))
    summary.to_csv(p_summary, index=False)

    log.info("wrote selected_studies.csv rows=%d -> %s", len(study_out), p_study)
    log.info("wrote image_transfer_manifest.csv rows=%d -> %s", len(image_out), p_image)
    log.info("wrote image_transfer_manifest_paths.txt lines=%d -> %s", n_unique_paths, p_paths)
    log.info("wrote manifest_summary.csv -> %s", p_summary)
    log.info("SUMMARY patients=%d studies=%d image_files=%d paths=%d pct_malformed=%.4f est_size=%s",
             int(n_patients), int(man["StudyInstanceUID_anon"].nunique()),
             int(man["SOPInstanceUID_anon"].nunique()), n_unique_paths, pct_malformed,
             "N/A(sizes unavailable)")
    log.info("view_position dist: %s",
             man["view_position"].value_counts().sort_index().to_dict())
    log.info("weight_bearing dist: %s",
             man["weight_bearing"].value_counts().sort_index().to_dict())
    log.info("laterality dist: %s",
             man["laterality"].value_counts().sort_index().to_dict())
    log.info("ALL manifest assertions passed. REVIEW-ONLY: no DICOMs opened, no transfer initiated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
