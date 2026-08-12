"""crop_qa.py — the crop QA gate that BLOCKS training (protocol sections 7 / 13 / 23).

Success non-negotiable #1 is that every crop written by src.preprocess_images contains
the CONTRALATERAL knee, ZERO index-knee pixels, and ZERO burned-in laterality markers.
4,075 of 4,269 frontal films are bilateral, so that claim rests on a metadata-driven
half-select which no automated test can fully validate — a human has to look.

WHY A CROP-ONLY CONTACT SHEET CANNOT CLOSE THAT GAP (this is the design constraint):
every image in this cohort is PRE-index, so BOTH knees on the film are still native —
there is no prosthesis to tell them apart. `standardize_to_left` then mirrors every
right knee, destroying the left/right cue, and `mask_border_frac` blanks the margins
where a burned-in L/R marker would sit. A finished crop is therefore ANATOMICALLY
IDENTICAL whether the correct or the wrong half was taken. Asking a reviewer to confirm
"this is the contralateral knee" from a crop alone asks an undecidable question, and a
reviewer would sign it in good faith while `horizontal_flip` semantics or
`bilateral_display_convention` were inverted across all 4,075 bilateral frontals.

So each QA sample is rendered as a TWO-PANEL row:
  panel A  the full decoded film, flip-CORRECTED but NOT mirrored, downsampled, with the
           half the pipeline actually sliced drawn as a rectangle and the discarded half
           labelled INDEX — annotated with index_side / contra_side / horizontal_flip /
           half_selected / laterality
  panel B  the final crop BEFORE standardize_orientation mirrors it
Panel A makes half-select decidable; panel B shows what the model will actually see.
Rendering panel A requires re-reading the DICOM for the sampled rows only (--dicom-root).
Without --dicom-root the sheet degrades to crop-only tiles and says so, loudly, in both
the contact sheet and the checklist: half-select was NOT visually verifiable.

This module produces FOUR artifacts:
  1. the two-panel contact sheet + the reviewer checklist (config signoff_criteria,
     verbatim), sampled from TRAIN + VAL only (the test split stays sealed for the
     contact sheet);
  2. protocol section 23 (i) — the IMAGE-LEVEL audit: a stratified random sample of
     >= image_audit_min_images index images spanning config audit_splits, scored by
     n_reviewers independent reviewers on the six config score_items. Emitted as a blank
     adjudication workbook; `--score <filled workbook>` computes raw agreement and
     Cohen's kappa per item and flags any item over critical_error_threshold;
  3. protocol section 23 (ii) — the OUTCOME-RECORD audit: a SEPARATE sample of
     >= outcome_audit_min_records records with the CPT chronology a reviewer needs to
     adjudicate the endpoint;
  4. protocol section 7 — the laterality audit, over-sampling side_source == "recovered".

DATA HYGIENE: outputs/ is not git-ignored, so nothing written there carries empi_anon.
Contact-sheet tiles are labelled with an opaque per-run index (P0001, ...) whose key
lives in derived-data/. Every audit workbook — which a clinical reviewer needs ids for —
is written under derived-data/ and only AGGREGATE summaries go to outputs/.

Run:  python3 -m src.crop_qa --dicom-root <DICOM root>   (full, decidable gate)
      python3 -m src.crop_qa                             (degraded: crop-only tiles)
      python3 -m src.crop_qa --rebuild-image-audit --sidecar A/labels.csv --sidecar B/labels.csv
      python3 -m src.crop_qa --score <filled image-audit workbook>

Reviewers do not edit the workbook by hand: `python3 -m src.qa_review_app` serves the
panels one at a time and writes one CSV per reviewer, which `--merge` folds back into the
`<item>_r<k>` columns that `--score` reads.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches      # noqa: E402  (backend must be set first)
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
from scipy import ndimage as ndi           # noqa: E402
import pandas as pd                        # noqa: E402
from PIL import Image                      # noqa: E402

from src.config import PROJECT_ROOT, load_config          # noqa: E402
from src.preprocess_images import (                        # noqa: E402
    PreprocessParams, build_manifest, crop_stages, read_dicom,
)

MODULE = "crop_qa"

# --- derived-data/ (git-ignored) filenames. CONFIG ADDITION NEEDED if these should be
# --- configurable; they are patient-level files so config only names their AGGREGATES.
INDEX_KEY_CSV = "crop_qa_index_key.csv"
AUDIT_PATIENT_CSV = "laterality_audit_sample.csv"
AUDIT_CROP_DIR = "laterality_audit_crops"
IMAGE_AUDIT_WORKBOOK = "image_audit_workbook.csv"
OUTCOME_AUDIT_WORKBOOK = "outcome_audit_workbook.csv"
OUTCOME_AUDIT_CPT_CSV = "outcome_audit_cpt_rows.csv"
QA_PANEL_DIR = "qa_panels"

LATERALITY_FAIL_REASONS = ("laterality_mismatch", "laterality_unresolved",
                           "contra_side_unresolved")

# A run whose DICOM root or shard directory looks like a scratch/fixture location did not
# read real patient images, so its artifacts must never present themselves as a signable
# gate (batch-1 review finding: a 30-image synthetic smoke run produced a checklist that
# opened "TRAINING IS BLOCKED UNTIL THIS FILE IS SIGNED" over `crude_knee()` fixtures).
SYNTHETIC_PATH_MARKERS = ("scratchpad", "/tmp/", "/private/tmp/", "synthetic", "synth",
                          "fixture", "smoke", "claude-", "pytest")
SYNTHETIC_BANNER = "**SYNTHETIC SMOKE TEST — NOT A GATE**"
DEGRADED_BANNER = ("**DEGRADED MODE — HALF-SELECT WAS NOT VISUALLY VERIFIABLE**")

# Reviewer scoring vocabulary for the protocol section 23 image audit.
SCORE_OK = "OK"
SCORE_ERROR = "ERROR"
# A THIRD, distinct verdict: the reviewer looked and the item is undecidable from the
# evidence in front of them. It is NOT "OK" (nothing was verified), NOT "ERROR" (nothing
# was found wrong), and NOT blank (blank means not yet reviewed, so a blank workbook and
# a finished-but-partly-undecidable one would look identical). It exists because the
# source DICOMs are gone: rows whose reviewer panel is crop-only cannot answer
# `laterality`, and the protocol's 2% critical-error rate must be computed over the rows
# that were actually decidable, with the undecidable count reported alongside it.
SCORE_NA = "NOT_ASSESSABLE"
_OK_ALIASES = {"OK", "PASS", "Y", "YES", "CORRECT", "GOOD", "1", "TRUE", "T"}
_ERROR_ALIASES = {"ERROR", "FAIL", "N", "NO", "WRONG", "BAD", "0", "FALSE", "F"}
# NOTE: bare "NA" is deliberately NOT here — it has always parsed as "not yet reviewed"
# and workbooks in the wild use it that way. The tokens below are unambiguous, and
# src/qa_review_app.py only ever writes the canonical SCORE_NA.
_NA_ALIASES = {"NOT_ASSESSABLE", "NOT ASSESSABLE", "NOT-ASSESSABLE", "UNASSESSABLE",
               "N/A", "CANNOT_ASSESS", "CANNOT ASSESS", "UNDECIDABLE"}


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(MODULE)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not any(getattr(h, "_mrkr", False) for h in lg.handlers):
        fh = logging.FileHandler(log_path, mode="a")   # run.log is shared + APPEND-ONLY
        fh._mrkr = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            f"{MODULE} | %(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
        lg.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh._mrkr = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(f"{MODULE} | %(levelname)s | %(message)s"))
        lg.addHandler(sh)
    return lg


def _rel(p) -> str:
    """Display path for a TRACKED artifact: project-relative, or redacted if outside.

    outputs/ is committed, so a full machine path (a scratchpad, a home directory, a
    Drive mount) must never be written into it verbatim — the shard directory and the
    DICOM root both live outside the project. Anything outside PROJECT_ROOT is reduced
    to `.../<parent>/<name>`, which still identifies which directory was inspected
    without recording where on the reviewer's disk it sat.
    """
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        parts = Path(str(p)).parts
        return ".../" + "/".join(parts[-2:]) if len(parts) > 2 else str(p)


# =============================================================================
# Locating the preprocess run
# =============================================================================
def resolve_shard_dir(cfg, explicit: str | None) -> tuple[Path | None, Path | None, str]:
    """Locate the shards + sidecar. Returns (shard_dir, sidecar_csv, message)."""
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    sidecar_name = str(cfg["preprocess"]["shards"]["sidecar_csv"])
    if explicit:
        d = Path(explicit).expanduser()
        return d, d / sidecar_name, f"--shard-dir {d}"
    run_json = coh / "preprocess_run.json"
    if run_json.exists():
        rec = json.loads(run_json.read_text())
        d = Path(rec["out_dir"])
        return d, Path(rec.get("sidecar_csv", d / sidecar_name)), f"preprocess_run.json -> {d}"
    return None, None, "no --shard-dir and no derived-data/cohort/preprocess_run.json"


def load_run_record(cfg) -> dict:
    run_json = cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_run.json"
    if run_json.exists():
        try:
            return json.loads(run_json.read_text())
        except Exception:
            return {}
    return {}


def load_crops(shard_dir: Path, rows: pd.DataFrame, image_format: str = "png") -> dict[str, np.ndarray]:
    """Extract the crops for `rows` from their shards. Opens each tar exactly once."""
    out: dict[str, np.ndarray] = {}
    if rows.empty or "shard" not in rows.columns:
        return out
    for shard, grp in rows.groupby("shard"):
        path = shard_dir / str(shard)
        if not path.exists():
            continue
        wanted = {f"{k}.{image_format}": k for k in grp["key"]}
        with tarfile.open(path, "r") as tf:
            for member in tf:
                key = wanted.get(member.name)
                if key is None:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                out[key] = np.asarray(Image.open(io.BytesIO(fh.read())).convert("L"))
    return out


# =============================================================================
# BD-1: re-deriving the FULL FILM for the sampled rows
# =============================================================================
def load_path_table(cfg, log: logging.Logger | None = None) -> pd.DataFrame | None:
    """sop_uid -> dicom_path (+ the flags crop_stages needs), from the same manifest
    src.preprocess_images processed. Returns None when the cohort parquet is unavailable."""
    try:
        man = build_manifest(cfg)
    except Exception as exc:                       # parquet missing / regression tripped
        if log is not None:
            log.warning("cannot rebuild the manifest for full-film QA (%s: %s)",
                        type(exc).__name__, exc)
        return None
    keep = ["SOPInstanceUID_anon", "dicom_path", "view", "laterality", "contra_side",
            "index_side", "horizontal_flip", "inverted", "split"]
    out = man[keep].rename(columns={"SOPInstanceUID_anon": "sop_uid"})
    return out.set_index("sop_uid")


def film_stages_for_rows(rows: pd.DataFrame, paths: pd.DataFrame | None, dicom_root: Path | None,
                         params: PreprocessParams,
                         log: logging.Logger | None = None) -> tuple[dict[str, dict], dict]:
    """Re-run the crop pipeline from the DICOM for `rows` and return the stage dicts.

    Cheap by design: this touches only the sampled rows, not the 6,090-image cohort. The
    stages come from src.preprocess_images.crop_stages — the SAME code path that wrote
    the shard — so the rectangle drawn on panel A is the slice that was actually taken,
    not a re-derivation that could disagree with it.

    Returns (stages_by_key, diagnostics).
    """
    stages: dict[str, dict] = {}
    diag = {"attempted": 0, "ok": 0, "no_path_table": paths is None, "missing_file": 0,
            "no_dicom_root": dicom_root is None, "decode_error": 0, "errors": []}
    if dicom_root is None or paths is None or rows.empty:
        return stages, diag

    for r in rows.itertuples(index=False):
        sop = str(getattr(r, "sop_uid", ""))
        key = str(getattr(r, "key", ""))
        if sop not in paths.index:
            diag["missing_file"] += 1
            continue
        meta = paths.loc[sop]
        path = Path(dicom_root) / str(meta["dicom_path"])
        diag["attempted"] += 1
        if not path.exists():
            diag["missing_file"] += 1
            continue
        try:
            arr, _ = read_dicom(path, params)
            st = crop_stages(arr, view=str(meta["view"]), laterality=str(meta["laterality"]),
                             contra_side=str(meta["contra_side"]),
                             horizontal_flip=int(meta["horizontal_flip"]), params=params)
            st["index_side"] = str(meta["index_side"])
            st["laterality"] = str(meta["laterality"])
            st["horizontal_flip"] = int(meta["horizontal_flip"])
            stages[key] = st
            diag["ok"] += 1
        except Exception as exc:            # a bad QA tile must never kill the whole gate
            diag["decode_error"] += 1
            if len(diag["errors"]) < 5:
                diag["errors"].append(f"{type(exc).__name__}: {exc}")
    if log is not None and diag["attempted"]:
        log.info("full-film QA: re-read %d/%d sampled DICOMs (missing %d, errors %d)",
                 diag["ok"], diag["attempted"], diag["missing_file"], diag["decode_error"])
    return stages, diag


def _downsample(img: np.ndarray, max_px: int = 900) -> np.ndarray:
    step = max(1, int(np.ceil(max(img.shape) / float(max_px))))
    return img[::step, ::step]


def draw_film_panel(ax, stage: dict, label: str) -> None:
    """Panel A: the flip-corrected full film with the SELECTED half outlined."""
    film = _downsample(np.asarray(stage["film"], dtype=np.float32))
    h, w = film.shape
    ax.imshow(film, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")

    bounds = stage.get("half_bounds")
    scale = np.asarray(stage["film"]).shape[1] / float(max(1, w))
    if bounds is None:
        c0, c1 = 0, w
        kept_txt, other = "CONTRA (whole film)", None
    else:
        c0, c1 = int(bounds[0] / scale), int(bounds[1] / scale)
        kept_txt = "CONTRA (kept)"
        other = (c1, w) if c0 == 0 else (0, c0)
    ax.add_patch(mpatches.Rectangle((c0 - 0.5, -0.5), max(1, c1 - c0), h,
                                    fill=False, edgecolor="#00ff66", linewidth=1.6))
    ax.text((c0 + c1) / 2.0, h * 0.045, kept_txt, color="#00ff66", fontsize=4.0,
            ha="center", va="top", fontweight="bold")
    if other is not None:
        ax.text((other[0] + other[1]) / 2.0, h * 0.045, "INDEX\n(discarded)", color="#ff5555",
                fontsize=4.0, ha="center", va="top", fontweight="bold")
    ax.set_title(label, fontsize=4.2, pad=1.6)
    ax.set_xticks([]); ax.set_yticks([])


def draw_missing_film_panel(ax, why: str) -> None:
    ax.set_facecolor("0.15")
    ax.text(0.5, 0.5, "NO FULL FILM\n\nhalf-select NOT\nvisually verifiable\n\n" + why,
            ha="center", va="center", fontsize=4.2, color="#ff5555", fontweight="bold",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def render_contact_sheet(sampled: pd.DataFrame, crops: dict[str, np.ndarray],
                         stages: dict[str, dict], index_of: dict[str, str],
                         views: list[str], n_per_cell: int, out_png: Path,
                         premirror: bool, degraded: bool, gate_reasons: list[str]) -> tuple[int, int, list[str]]:
    """Two-panel-per-sample contact sheet. Tiles carry an OPAQUE patient index only.

    Returns (n_crop_panels_drawn, n_film_panels_drawn, cells).
    """
    cells = [f"{v}|{cs}" for v in views for cs in ("L", "R")]
    cells = [c for c in cells if (sampled["cell"] == c).any()]
    n_rows = max(1, len(cells))
    n_cols = max(2, 2 * n_per_cell)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 2.6 * n_rows), dpi=150,
                             squeeze=False)

    n_crop, n_film = 0, 0
    for i, cell in enumerate(cells):
        sub = sampled[sampled["cell"] == cell].reset_index(drop=True)
        for j in range(n_per_cell):
            ax_a, ax_b = axes[i, 2 * j], axes[i, 2 * j + 1]
            for ax in (ax_a, ax_b):
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
            if j >= len(sub):
                ax_a.set_facecolor("0.92"); ax_b.set_facecolor("0.92")
                continue
            r = sub.iloc[j]
            pidx = index_of.get(r["empi_anon"], "?")
            st = stages.get(r["key"])

            # ---- panel A: the full film with the selected half outlined ----
            if st is not None:
                lbl = (f"{pidx} lat={st['laterality']} idx={st['index_side']} "
                       f"con={r['contra_side']} hf={st['horizontal_flip']}\n"
                       f"half={st['half_selected']}  ({r['view'][:4]})")
                draw_film_panel(ax_a, st, lbl)
                n_film += 1
            else:
                draw_missing_film_panel(ax_a, f"{pidx} {r['view'][:4]} con={r['contra_side']}")

            # ---- panel B: the crop, PRE-mirror when we could recompute it ----
            img = st["premirror"] if (st is not None and premirror) else crops.get(r["key"])
            tag = "PRE-mirror" if (st is not None and premirror) else "final (mirrored)"
            if img is None:
                ax_b.set_facecolor("0.92")
                ax_b.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=5)
                continue
            ax_b.imshow(img, cmap="gray", vmin=0, vmax=255)
            n_crop += 1
            method = "ip" if r["crop_method"] == "intensity_profile" else "FB"
            msk = float(r["masked_pct"]) if "masked_pct" in r.index and pd.notna(r["masked_pct"]) else float("nan")
            ax_b.set_title(f"crop {tag}\n{method} q={float(r['crop_confidence']):.2f} "
                           f"msk={100.0 * msk:.0f}%", fontsize=4.2, pad=1.6)
        axes[i, 0].set_ylabel(cell.replace("|", "\ncontra="), fontsize=6)

    banner = []
    if gate_reasons:
        banner.append(SYNTHETIC_BANNER.replace("**", ""))
    if degraded:
        banner.append(DEGRADED_BANNER.replace("**", ""))
    title = ("Contralateral-knee crop QA — LEFT panel: the full film, flip-corrected, with the "
             "half the pipeline KEPT outlined in green (the discarded INDEX half is marked red). "
             "RIGHT panel: that crop before the left/right mirror.\n"
             "Both knees are native on every pre-index film, so the crop ALONE cannot tell you "
             "which half was taken — judge half-select on the LEFT panel.")
    if banner:
        title = "  ///  ".join(banner) + "\n" + title
    fig.suptitle(title, fontsize=7, y=0.997,
                 color=("#b00000" if banner else "black"),
                 fontweight=("bold" if banner else "normal"))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return n_crop, n_film, cells


def save_qa_panel(stage: dict, crop: np.ndarray | None, out_path: Path, label: str) -> None:
    """One reviewer-facing panel (film + crop) for an audit workbook row."""
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 4.2), dpi=130, squeeze=False)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    if stage is not None:
        draw_film_panel(ax_a, stage, label)
        img = stage["premirror"]
        tag = "crop (PRE-mirror)"
    else:
        draw_missing_film_panel(ax_a, label)
        img, tag = crop, "crop (final, mirrored)"
    ax_b.set_xticks([]); ax_b.set_yticks([])
    if img is not None:
        ax_b.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax_b.set_title(tag, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Sampling
# =============================================================================
def sample_cells(side: pd.DataFrame, views: list[str], n_per_cell: int, seed: int) -> pd.DataFrame:
    """n_per_cell rows per (view x contra_side) cell, deterministic given the seed."""
    picks = []
    for view in views:
        for cs in ("L", "R"):
            cell = side[(side["view"] == view) & (side["contra_side"] == cs)]
            if cell.empty:
                continue
            take = cell.sample(n=min(n_per_cell, len(cell)), random_state=seed)
            picks.append(take.assign(cell=f"{view}|{cs}"))
    return pd.concat(picks).reset_index(drop=True) if picks else side.head(0).assign(cell=[])


def stratified_sample(df: pd.DataFrame, strata_cols: list[str], n_target: int,
                      seed: int, prefer: pd.Series | None = None) -> pd.DataFrame:
    """Proportional stratified sample of size ~n_target (largest-remainder allocation).

    Every non-empty stratum contributes at least one row so no cell of the design is
    invisible to the reviewers; the remainder is allocated proportionally and any
    shortfall is topped up at random from the unsampled pool.

    `prefer` is an optional boolean Series aligned to `df.index` naming rows to fill a
    stratum's slots FIRST. It changes neither the strata nor their quotas — only WHICH
    member of a stratum takes a slot — and it exists for one reason: a re-sample must
    not silently throw away rows whose reviewer panel can still be rendered. The
    full-film panel (the only thing that makes the `laterality` item answerable) is
    re-read from the source DICOM, and once that DICOM is gone a row that leaves the
    sample can never come back with a decidable panel. Preferring already-panelled rows
    introduces no bias, because those rows were themselves drawn at random from the same
    strata: "was in the previous random sample" carries no information about the image.
    """
    if df.empty:
        return df.copy()
    n_target = int(min(n_target, len(df)))
    groups = list(df.groupby(strata_cols, dropna=False, sort=True))
    sizes = np.array([len(g) for _, g in groups], dtype=float)
    quota = np.ones(len(groups), dtype=int)                      # at least 1 per stratum
    remaining = max(0, n_target - int(quota.sum()))
    if remaining > 0 and sizes.sum() > 0:
        share = sizes / sizes.sum() * remaining
        base = np.floor(share).astype(int)
        rem = remaining - int(base.sum())
        order = np.argsort(-(share - base))
        for k in order[:max(0, rem)]:
            base[k] += 1
        quota = quota + base
    quota = np.minimum(quota, sizes.astype(int))

    picks = []
    for (_, g), q in zip(groups, quota):
        q = int(q)
        if q <= 0:
            continue
        if prefer is None:
            picks.append(g.sample(n=q, random_state=seed))
            continue
        mask = prefer.reindex(g.index).fillna(False).astype(bool)
        head, tail = g[mask], g[~mask]
        take = head if len(head) <= q else head.sample(n=q, random_state=seed)
        short = q - len(take)
        if short > 0 and len(tail):
            take = pd.concat([take, tail.sample(n=min(short, len(tail)), random_state=seed)])
        picks.append(take)
    out = pd.concat(picks) if picks else df.head(0)
    if len(out) < n_target:
        rest = df.loc[~df.index.isin(out.index)]
        extra = min(len(rest), n_target - len(out))
        if extra:
            if prefer is None:
                out = pd.concat([out, rest.sample(n=extra, random_state=seed)])
            else:
                # Shuffle first, then float the preferred rows to the front, so the
                # top-up is still random within "panelled" and "not panelled".
                shuffled = rest.sample(frac=1.0, random_state=seed)
                pref_first = prefer.reindex(shuffled.index).fillna(False).astype(bool)
                out = pd.concat([out, shuffled[pref_first].head(extra),
                                 shuffled[~pref_first].head(max(0, extra - int(pref_first.sum())))])
    return out.sort_index()


# =============================================================================
# Protocol section 23 (i) — IMAGE-LEVEL audit
# =============================================================================
def build_image_audit(cfg, side_all: pd.DataFrame, audit_splits: list[str], min_images: int,
                      items: list[str], n_reviewers: int, seed: int,
                      index_of: dict[str, str],
                      full_film_keys: set[str] | None = None,
                      prefer_keys: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Blank adjudication workbook: one row per sampled image, one column per (item x reviewer).

    NOT OUTCOME-UNBLINDING. This is LABEL/CROP quality assurance and it precedes model
    registration: the reviewer sees pixels and acquisition metadata, and the workbook
    deliberately carries NO outcome column (no event_indicator, no time_from_landmark,
    no event date). Nothing here reveals a test-set label or informs model selection, so
    the sample spans config `audit_splits` — including test — exactly as protocol
    section 23 requires ("a stratified random sample of at least 400 index images").

    Stratified on split x view x contra_side x bilateral-vs-unilateral, because those are
    the strata whose failure modes differ: only bilateral frontals undergo the half-select.

    `full_film_keys` names the rows for which a FULL-FILM reviewer panel already exists on
    disk; they are flagged `panel_has_full_film`. This matters because the source DICOMs are
    no longer available: a row drawn today that has no panel yet can only ever get a
    CROP-ONLY panel, on which the `laterality` item is undecidable by construction (both
    knees are native pre-index, the mirror removes the left/right cue, the border mask
    removes the marker). A reviewer must score that item NOT_ASSESSABLE rather than guess,
    and the workbook has to say which rows those are.

    `prefer_keys` (default: `full_film_keys`) names rows to fill a stratum's slots first.
    Pass the PREVIOUS sample's keys to re-draw over an enlarged frame without discarding
    the rows whose panel is already decidable — see `stratified_sample` for why that is
    not a biased draw.
    """
    pool = side_all[side_all["split"].isin(audit_splits)].copy()
    info = {"splits_requested": list(audit_splits),
            "splits_available": sorted(pool["split"].unique().tolist()),
            "splits_missing": sorted(set(audit_splits) - set(pool["split"].unique().tolist())),
            "pool_size": int(len(pool)), "min_images": int(min_images)}
    if pool.empty:
        return pool, info
    pool["laterality_kind"] = np.where(pool["laterality"].astype(str) == "B",
                                       "bilateral_B", "unilateral")
    anchor = prefer_keys if prefer_keys is not None else full_film_keys
    prefer = pool["key"].astype(str).isin(set(anchor)) if anchor else None
    take = stratified_sample(pool, ["split", "view", "contra_side", "laterality_kind"],
                             min_images, seed, prefer=prefer).reset_index(drop=True)

    take["qa_index"] = take["empi_anon"].map(index_of).fillna("")
    take["crop_png"] = take["key"].astype(str) + ".png"
    take["qa_panel_png"] = take["key"].astype(str) + ".png"
    take["panel_has_full_film"] = take["key"].astype(str).isin(set(full_film_keys or ()))
    for k in range(1, int(n_reviewers) + 1):
        for item in items:
            take[f"{item}_r{k}"] = ""            # reviewer k scores OK / ERROR
        take[f"notes_r{k}"] = ""
    for item in items:
        take[f"{item}_adjudicated"] = ""         # consensus / third reviewer
    take["adjudicator_notes"] = ""

    cols = ["qa_index", "empi_anon", "split", "view", "laterality", "laterality_kind",
            "contra_side", "index_side", "horizontal_flip", "inverted", "half_selected",
            "orientation", "mirrored", "crop_method", "crop_confidence", "masked_pct",
            "out_size", "shard", "key", "crop_png", "qa_panel_png", "panel_has_full_film"]
    cols = [c for c in cols if c in take.columns]
    review_cols = [c for c in take.columns if c.endswith(tuple(
        [f"_r{k}" for k in range(1, int(n_reviewers) + 1)])) or c.endswith("_adjudicated")]
    take = take[cols + review_cols + ["adjudicator_notes"]]
    info["n_sampled"] = int(len(take))
    info["by_split"] = take["split"].value_counts().to_dict()
    info["by_view"] = take["view"].value_counts().to_dict()
    info["by_laterality_kind"] = take["laterality_kind"].value_counts().to_dict()
    info["n_with_full_film_panel"] = int(take["panel_has_full_film"].sum())
    info["n_crop_only_panel"] = int((~take["panel_has_full_film"]).sum())
    info["n_strata"] = int(take.groupby(
        ["split", "view", "contra_side", "laterality_kind"], dropna=False).ngroups)
    return take, info


def stable_index_map(existing_csv: Path, patients) -> dict[str, str]:
    """empi_anon -> opaque qa_index, PRESERVING any assignment already on disk.

    The index is burned into every reviewer panel PNG as its title (`P0026 fron ...`), so
    renumbering would silently re-label 584 images a reviewer is about to adjudicate.
    Patients already in the key keep their index; new ones continue from the current max.
    """
    prev: dict[str, str] = {}
    if existing_csv.exists():
        old = pd.read_csv(existing_csv, dtype=str)
        if {"empi_anon", "qa_index"}.issubset(old.columns):
            prev = dict(zip(old["empi_anon"].astype(str), old["qa_index"].astype(str)))
    nxt = 0
    for v in prev.values():
        try:
            nxt = max(nxt, int(str(v).lstrip("P")))
        except ValueError:
            continue
    out = dict(prev)
    for p in sorted(str(x) for x in patients):
        if p not in out:
            nxt += 1
            out[p] = f"P{nxt:04d}"
    return out


def load_sidecars(paths: list[Path], log: logging.Logger | None = None) -> pd.DataFrame:
    """Concatenate several preprocess sidecars into ONE frame, tagged with their shard dir.

    src.preprocess_images writes one sidecar per run and `preprocess_run.json` records only
    the LAST one, so the train+val run (2026-07-26) and the test run (2026-07-29) live in
    two separate files. The protocol section 23 audit spans config `audit_splits`, which
    includes test, so the audit frame is the UNION of the runs, not whichever ran last.
    """
    frames = []
    for p in paths:
        p = Path(p).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"sidecar not found: {p}")
        df = pd.read_csv(p, dtype={"empi_anon": str, "sop_uid": str})
        df["_shard_dir"] = str(p.parent)
        frames.append(df)
        if log is not None:
            log.info("sidecar %s: %d rows, splits %s", _rel(p), len(df),
                     df["split"].value_counts().to_dict())
    if not frames:
        raise ValueError("no sidecars given")
    out = pd.concat(frames, ignore_index=True)
    n_dup = int(out["key"].duplicated().sum())
    if n_dup and log is not None:
        log.warning("%d duplicate crop keys across sidecars — keeping the first", n_dup)
    return out.drop_duplicates("key").reset_index(drop=True)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items. NaN when it is undefined.

    kappa = (po - pe) / (1 - pe). It is UNDEFINED when pe == 1, which happens whenever
    both reviewers used exactly one category — i.e. perfect agreement with no variation,
    the most likely result of a clean audit. Returning NaN and saying so is correct;
    reporting kappa = 0 there would read as chance-level agreement and be badly wrong.
    """
    a = list(a); b = list(b)
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)), dtype=float)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1.0
    po = float(np.trace(m)) / n
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum()) / float(n * n)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def _normalize_score(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().upper()
    if s == "" or s in {"NAN", "NA", "NONE"}:
        return None
    if s in _OK_ALIASES:
        return SCORE_OK
    if s in _ERROR_ALIASES:
        return SCORE_ERROR
    if s in _NA_ALIASES:
        return SCORE_NA
    return "UNPARSED"


def score_image_audit(cfg, workbook: Path, log: logging.Logger) -> int:
    """Read a FILLED image-audit workbook; report agreement, kappa and critical errors.

    Writes the aggregate (no ids) to config crop_qa.image_audit_csv. A workbook that has
    not been filled in yet is reported as "awaiting reviewer input", not an error.
    """
    qa = cfg["crop_qa"]
    items = list(qa["score_items"])
    n_rev = int(qa["n_reviewers"])
    thresh = float(qa["critical_error_threshold"])
    out_csv = cfg.path(qa["image_audit_csv"])

    if not workbook.exists():
        log.error("image-audit workbook not found: %s", workbook)
        return 2
    wb = pd.read_csv(workbook, dtype=str)
    log.info("scoring %d workbook rows from %s (%d reviewers, %d items, threshold %.1f%%)",
             len(wb), workbook, n_rev, len(items), 100.0 * thresh)

    rows: list[dict] = []
    expanded: list[str] = []
    total_scored = 0
    for item in items:
        cols = [f"{item}_r{k}" for k in range(1, n_rev + 1)]
        missing = [c for c in cols if c not in wb.columns]
        if missing:
            rows.append({"item": item, "status": f"missing columns {missing}",
                         "n_scored": 0, "raw_agreement": "", "cohens_kappa": "",
                         "critical_error_rate": "", "n_critical": "", "exceeds_threshold": "",
                         "n_unparsed": 0, "n_not_assessable": 0})
            continue
        scored = wb[cols].map(_normalize_score)
        n_unparsed = int((scored == "UNPARSED").sum().sum())
        # A row is NOT_ASSESSABLE as soon as either reviewer says so: agreement, kappa and
        # the critical-error rate are all defined over the rows that were DECIDABLE, and a
        # row one reviewer could not decide is not one of them.
        not_assessable = (scored == SCORE_NA).any(axis=1)
        n_na = int(not_assessable.sum())
        complete = (scored.notna().all(axis=1) & (scored != "UNPARSED").all(axis=1)
                    & ~not_assessable)
        sub = scored[complete]
        total_scored += int(len(sub))
        if sub.empty:
            rows.append({"item": item, "status": "awaiting reviewer input", "n_scored": 0,
                         "raw_agreement": "", "cohens_kappa": "", "critical_error_rate": "",
                         "n_critical": "", "exceeds_threshold": "", "n_unparsed": n_unparsed,
                         "n_not_assessable": n_na})
            continue

        # Raw agreement / kappa are defined pairwise; with n_reviewers == 2 this is the
        # single pair protocol section 23 asks for.
        a = sub[cols[0]].tolist()
        b = sub[cols[1]].tolist() if n_rev >= 2 else a
        agree = float(np.mean([x == y for x, y in zip(a, b)]))
        kappa = cohens_kappa(a, b) if n_rev >= 2 else float("nan")
        # A critical error is an image ANY reviewer flagged: conservative, because the
        # protocol expands the review when a critical error exceeds 2%.
        n_crit = int(sum(1 for x, y in zip(a, b) if SCORE_ERROR in (x, y)))
        rate = n_crit / float(len(sub))
        over = rate > thresh
        if over:
            expanded.append(f"{item} ({100.0 * rate:.2f}% > {100.0 * thresh:.2f}%)")
        rows.append({
            "item": item,
            "status": "scored" if n_rev >= 2 else "single reviewer (kappa undefined)",
            "n_scored": int(len(sub)),
            "raw_agreement": round(agree, 4),
            "cohens_kappa": ("undefined (no variation)" if np.isnan(kappa) else round(kappa, 4)),
            "critical_error_rate": round(rate, 4),
            "n_critical": n_crit,
            "exceeds_threshold": bool(over),
            "n_unparsed": n_unparsed,
            "n_not_assessable": n_na,
        })

    summary = pd.DataFrame(rows)
    summary["n_workbook_rows"] = int(len(wb))
    summary["critical_error_threshold"] = thresh
    summary["n_reviewers"] = n_rev
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    log.info("image-audit aggregate -> %s", out_csv)
    for r in rows:
        log.info("  %-16s %-28s n=%s agree=%s kappa=%s crit=%s n/a=%s", r["item"], r["status"],
                 r.get("n_scored"), r.get("raw_agreement"), r.get("cohens_kappa"),
                 r.get("critical_error_rate"), r.get("n_not_assessable"))
    if total_scored == 0:
        log.warning("AWAITING REVIEWER INPUT: no item has two completed reviewer columns yet. "
                    "Fill %s (each cell OK, ERROR or %s) and re-run --score.", workbook, SCORE_NA)
        return 0
    if expanded:
        log.error("*** EXPAND THE REVIEW *** critical-error rate exceeds "
                  "critical_error_threshold=%.2f%% for: %s. Protocol section 23 requires the "
                  "review to be EXPANDED (draw a further stratified sample of the affected "
                  "stratum, re-adjudicate, and do not proceed to training on the current "
                  "sample).", 100.0 * thresh, "; ".join(expanded))
        # The 2% rule is a GATE, so a failing gate must not exit 0. Anything scripting this
        # module (a Makefile, CI, `&&` in a shell) reads the status code, not the log.
        return 2
    log.info("no item exceeds critical_error_threshold=%.2f%%", 100.0 * thresh)
    return 0


def rebuild_image_audit(cfg, sidecar_paths: list[Path], log: logging.Logger,
                        write_panels: bool = True, backup_name: str | None = None) -> int:
    """Rebuild ONLY the protocol section 23 (i) workbook + its reviewer panels.

    A targeted mode, not a re-run of the gate. The full `main()` re-renders the contact
    sheet, the checklist and both other audits; re-running it today would rewrite a
    signed-off-pending checklist from a DICOM tree that no longer exists and would sample
    the contact sheet from whichever run wrote `preprocess_run.json` last. This entry
    point touches four things and nothing else: the index key, the image-audit workbook,
    its backup, and the missing panel PNGs.

    Why it exists: the workbook on disk is train-344 / val-56 / test-0, because it was
    built on 2026-07-26 and the 1,216 test crops were not written until 2026-07-29.
    `crop_qa.audit_splits` has always asked for all three.
    """
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    qa = cfg["crop_qa"]
    seed = int(cfg["reproducibility"]["random_seed"])
    image_format = str(cfg["preprocess"]["shards"]["image_format"])
    panel_dir = coh / QA_PANEL_DIR

    side_all = load_sidecars(sidecar_paths, log)
    log.info("audit frame: %d crops / %d patients, splits %s", len(side_all),
             side_all["empi_anon"].nunique(), side_all["split"].value_counts().to_dict())

    # Panels already on disk were rendered WITH the full film (run.log, 584/584). The
    # DICOMs are gone, so this set can never grow — it is the decidability frontier.
    full_film_keys = {p.stem for p in panel_dir.glob(f"*.{image_format}")} if panel_dir.exists() else set()
    log.info("%d existing reviewer panels carry a full film (source DICOMs are gone, so no "
             "new full-film panel can be rendered)", len(full_film_keys))

    index_of = stable_index_map(coh / INDEX_KEY_CSV, side_all["empi_anon"].unique())
    pd.DataFrame({"empi_anon": list(index_of), "qa_index": list(index_of.values())}
                 ).sort_values("qa_index").to_csv(coh / INDEX_KEY_CSV, index=False)

    # Anchor the re-draw on the PREVIOUS sample, not on every panel on disk. Both keep the
    # same number of decidable rows, but anchoring on the previous sample makes the train
    # and val portion a verifiable SUBSET of it, whereas anchoring on all panels would also
    # recruit the protocol section 7 panels — and that sample deliberately over-samples
    # side_source == "recovered", which would leak an enrichment into this one.
    wb_path = coh / IMAGE_AUDIT_WORKBOOK
    prefer_keys = full_film_keys
    if wb_path.exists():
        prev = pd.read_csv(wb_path, dtype=str)
        if "key" in prev.columns:
            prefer_keys = set(prev["key"].astype(str)) & full_film_keys
            log.info("anchoring the re-draw on %d rows of the previous sample", len(prefer_keys))

    take, info = build_image_audit(
        cfg, side_all, list(qa["audit_splits"]), int(qa["image_audit_min_images"]),
        list(qa["score_items"]), int(qa["n_reviewers"]), seed, index_of,
        full_film_keys=full_film_keys, prefer_keys=prefer_keys)
    info["n_carried_over"] = int(take["key"].astype(str).isin(prefer_keys).sum())
    if backup_name and wb_path.exists():
        backup = coh / backup_name
        if backup.exists():
            log.info("backup already present, not overwriting: %s", backup)
        else:
            backup.write_bytes(wb_path.read_bytes())
            log.info("previous workbook preserved -> %s", backup)
    take.to_csv(wb_path, index=False)
    log.info("image audit workbook -> %s (%d rows; by split %s; by view %s; %d strata; "
             "%d carried over from the previous sample)", wb_path, len(take), info["by_split"],
             info["by_view"], info["n_strata"], info["n_carried_over"])
    log.info("panels: %d rows keep a FULL-FILM panel, %d rows are CROP-ONLY (the `laterality` "
             "item is undecidable on those and must be scored %s)",
             info["n_with_full_film_panel"], info["n_crop_only_panel"], SCORE_NA)
    if info.get("splits_missing"):
        log.warning("audit_splits %s are still absent from the frame", info["splits_missing"])

    n_new = 0
    if write_panels:
        need = take[~take["panel_has_full_film"]].merge(
            side_all[["key", "_shard_dir"]], on="key", how="left")
        panel_dir.mkdir(parents=True, exist_ok=True)
        for shard_dir, grp in need.groupby("_shard_dir"):
            crops = load_crops(Path(str(shard_dir)), grp, image_format)
            for r in grp.itertuples(index=False):
                key = str(r.key)
                img = crops.get(key)
                if img is None:
                    log.warning("no crop in %s for %s — panel not written", _rel(shard_dir), key)
                    continue
                label = (f"{r.qa_index} {str(r.view)[:4]} lat={r.laterality} "
                         f"idx={r.index_side} con={r.contra_side} hf={r.horizontal_flip}\n"
                         f"source DICOM unavailable — half-select NOT verifiable")
                save_qa_panel(None, img, panel_dir / f"{key}.{image_format}", label)
                n_new += 1
        log.info("wrote %d NEW crop-only reviewer panels -> %s", n_new, panel_dir)

    log.info("score it with: python3 -m src.qa_review_app --mode image --reviewer <name>")
    return 0


# =============================================================================
# Protocol section 23 (ii) — OUTCOME-RECORD audit
# =============================================================================
def build_outcome_audit(cfg, min_records: int, seed: int,
                        log: logging.Logger | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """>= min_records outcome records with the CPT chronology needed to adjudicate them.

    A SEPARATE sample from the image audit (protocol section 23: "A second sample of at
    least 200 outcome records will be reviewed using CPT chronology and available
    post-event images"). Events are deliberately over-sampled: the endpoint being
    validated is the EVENT, and a sample drawn at the natural 14% prevalence would give
    a reviewer ~28 events to check.

    Returns (workbook, cpt_rows_long, info).
    """
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    fc = pd.read_parquet(coh / "final_cohort.parquet")[
        ["empi_anon", "index_date", "index_side", "contra_side", "side_source",
         "n_concordant_signals", "primary_event", "event_date", "days_index_to_event",
         "event_indicator", "time_from_landmark", "landmark_date", "last_observed",
         "censor_reason"]]
    out = pd.read_parquet(coh / "outcomes.parquet")[
        ["empi_anon", "primary_event", "event_date", "days_index_to_event",
         "has_contra_27447_day_0_90", "upper_bound_event", "composite_uni_event",
         "augmented_event", "side_source"]]
    sp = pd.read_parquet(coh / "patient_splits.parquet")[["empi_anon", "split"]]

    # outcomes.parquet is a superset (6,381 rows); the audit universe is the LOCKED cohort.
    pool = fc.merge(out, on="empi_anon", how="inner", suffixes=("", "_outcomes"))
    pool = pool.merge(sp, on="empi_anon", how="left")
    n_events = int(pool["primary_event"].astype(bool).sum())

    n_target = int(min(min_records, len(pool)))
    n_ev_take = int(min(n_events, max(1, n_target // 2)))
    ev = pool[pool["primary_event"].astype(bool)]
    nonev = pool[~pool["primary_event"].astype(bool)]
    take_ev = stratified_sample(ev, ["side_source"], n_ev_take, seed) if n_ev_take else ev.head(0)
    take_ne = stratified_sample(nonev, ["side_source"], n_target - len(take_ev), seed)
    take = pd.concat([take_ev, take_ne])
    if len(take) < n_target:
        rest = pool.loc[~pool.index.isin(take.index)]
        extra = min(len(rest), n_target - len(take))
        if extra:
            take = pd.concat([take, rest.sample(n=extra, random_state=seed)])
    take = take.reset_index(drop=True)

    # ---- the CPT chronology the reviewer adjudicates against ----
    codes = [str(c) for c in cfg["prior_knee_arthroplasty_cpt"].keys()]
    cpt = pd.read_parquet(cfg.parquet_path("cpt"),
                          columns=["empi_anon", "cpt_code", "cpt_group_modifier", "date_anon"])
    cpt["cpt_code"] = cpt["cpt_code"].astype(str).str.strip().str.zfill(5)
    cpt = cpt[cpt["empi_anon"].isin(set(take["empi_anon"])) & cpt["cpt_code"].isin(codes)]
    cpt = cpt.sort_values(["empi_anon", "date_anon", "cpt_code"]).reset_index(drop=True)
    cpt = cpt.merge(take[["empi_anon", "index_date", "index_side", "contra_side"]],
                    on="empi_anon", how="left")
    cpt["days_from_index"] = (pd.to_datetime(cpt["date_anon"]) -
                              pd.to_datetime(cpt["index_date"])).dt.days
    cpt["modifier"] = cpt["cpt_group_modifier"].fillna("(none)")
    cpt_long = cpt[["empi_anon", "date_anon", "days_from_index", "cpt_code", "modifier",
                    "index_date", "index_side", "contra_side"]]

    chrono = (cpt_long.assign(_s=cpt_long["date_anon"].astype(str) + " d"
                              + cpt_long["days_from_index"].astype("Int64").astype(str)
                              + " " + cpt_long["cpt_code"] + " [" + cpt_long["modifier"] + "]")
              .groupby("empi_anon")["_s"].agg(" ; ".join).rename("cpt_chronology").reset_index())
    n_rows = cpt_long.groupby("empi_anon").size().rename("n_knee_arthroplasty_cpt_rows").reset_index()
    take = take.merge(chrono, on="empi_anon", how="left").merge(n_rows, on="empi_anon", how="left")
    take["cpt_chronology"] = take["cpt_chronology"].fillna("(no knee-arthroplasty CPT rows)")
    take["n_knee_arthroplasty_cpt_rows"] = take["n_knee_arthroplasty_cpt_rows"].fillna(0).astype(int)

    # blank reviewer columns
    take["reviewer_event_confirmed_Y_N"] = ""
    take["reviewer_event_date"] = ""
    take["reviewer_event_side"] = ""
    take["reviewer_agrees_with_primary_event_Y_N"] = ""
    take["reviewer_notes"] = ""

    cols = ["empi_anon", "split", "side_source", "n_concordant_signals", "index_date",
            "index_side", "contra_side", "primary_event", "event_date",
            "days_index_to_event", "event_indicator", "time_from_landmark",
            "landmark_date", "last_observed", "censor_reason",
            "has_contra_27447_day_0_90", "upper_bound_event", "composite_uni_event",
            "augmented_event", "n_knee_arthroplasty_cpt_rows", "cpt_chronology",
            "reviewer_event_confirmed_Y_N", "reviewer_event_date", "reviewer_event_side",
            "reviewer_agrees_with_primary_event_Y_N", "reviewer_notes"]
    cols = [c for c in cols if c in take.columns]
    take = take[cols].sort_values(["primary_event", "side_source", "empi_anon"],
                                  ascending=[False, True, True]).reset_index(drop=True)

    info = {
        "n_sampled": int(len(take)),
        "min_required": int(min_records),
        "cohort_pool": int(len(pool)),
        "cohort_events": n_events,
        "n_events_sampled": int(take["primary_event"].astype(bool).sum()),
        "n_nonevents_sampled": int((~take["primary_event"].astype(bool)).sum()),
        "by_side_source": take["side_source"].value_counts().to_dict(),
        "by_split": take["split"].value_counts().to_dict(),
        "n_with_cpt_rows": int((take["n_knee_arthroplasty_cpt_rows"] > 0).sum()),
        "n_cpt_rows_total": int(len(cpt_long)),
    }
    if log is not None:
        log.info("outcome-record audit: %d records (%d events / %d non-events) from a %d-patient "
                 "cohort pool; %d knee-arthroplasty CPT rows attached",
                 info["n_sampled"], info["n_events_sampled"], info["n_nonevents_sampled"],
                 info["cohort_pool"], info["n_cpt_rows_total"])
    return take, cpt_long, info


# =============================================================================
# Protocol section 7 — laterality audit
# =============================================================================
def build_laterality_audit(cfg, side: pd.DataFrame, splits: list[str], min_patients: int,
                           seed: int) -> pd.DataFrame:
    """>=min_patients patient sample for the protocol section 7 laterality audit,
    deliberately over-sampling side_source == "recovered" (inferred, not coded).

    The reviewer is given the CONTRALATERAL CROP, from which `index_side` is by
    construction unjudgeable — so the row also carries `qa_panel_png`, the full-film
    panel from the contact-sheet machinery, which is what actually makes the laterality
    question answerable.
    """
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    fc = pd.read_parquet(coh / "final_cohort.parquet")[
        ["empi_anon", "index_side", "contra_side", "side_source", "n_concordant_signals",
         "view_set", "tier_name", "n_images"]]
    pats = side[side["split"].isin(splits)][["empi_anon", "split"]].drop_duplicates()
    pool = pats.merge(fc, on="empi_anon", how="left")

    rec = pool[pool["side_source"] == "recovered"]
    cod = pool[pool["side_source"] != "recovered"]
    # 2:1 recovered:coded — recovered laterality is the inferred half and is what the
    # audit exists to validate.
    n_rec = min(len(rec), max(1, int(round(min_patients * 2 / 3))))
    n_cod = min(len(cod), max(0, min_patients - n_rec))
    take = pd.concat([rec.sample(n=n_rec, random_state=seed) if n_rec else rec.head(0),
                      cod.sample(n=n_cod, random_state=seed) if n_cod else cod.head(0)])
    if len(take) < min_patients:                      # top up from whatever is left
        rest = pool[~pool["empi_anon"].isin(take["empi_anon"])]
        extra = min(len(rest), min_patients - len(take))
        if extra:
            take = pd.concat([take, rest.sample(n=extra, random_state=seed)])

    # Reference the frontal crop where one exists (the view a reviewer adjudicates on).
    pref = {"frontal": 0, "lateral": 1, "sunrise": 2}
    s = side.copy()
    s["_rank"] = s["view"].map(pref).fillna(9)
    first = s.sort_values(["empi_anon", "_rank", "key"]).groupby("empi_anon").first().reset_index()
    take = take.merge(first[["empi_anon", "view", "shard", "key", "crop_method",
                             "crop_confidence", "orientation"]], on="empi_anon", how="left")
    views_by_pat = (side.groupby("empi_anon")["view"].agg(lambda v: "+".join(sorted(set(v))))
                    .rename("views_in_shards").reset_index())
    take = take.merge(views_by_pat, on="empi_anon", how="left")

    take = take.rename(columns={"view": "crop_view", "key": "crop_key", "shard": "crop_shard"})
    take["crop_png"] = take["crop_key"].astype(str) + ".png"
    take["qa_panel_png"] = take["crop_key"].astype(str) + ".png"
    # Blank columns the clinical reviewer fills in.
    take["reviewer_index_side"] = ""
    take["reviewer_agrees_Y_N"] = ""
    take["reviewer_notes"] = ""
    cols = ["empi_anon", "split", "side_source", "n_concordant_signals", "index_side",
            "contra_side", "tier_name", "view_set", "views_in_shards", "n_images",
            "crop_view", "orientation", "crop_method", "crop_confidence",
            "crop_shard", "crop_key", "crop_png", "qa_panel_png",
            "reviewer_index_side", "reviewer_agrees_Y_N", "reviewer_notes"]
    return take[cols].sort_values(["side_source", "empi_anon"]).reset_index(drop=True)


# =============================================================================
# BD-6 — is this artifact actually a gate, or a smoke test?
# =============================================================================
def assess_gate_validity(cfg, side: pd.DataFrame, run_record: dict, sampled: pd.DataFrame,
                         n_crop_panels: int) -> list[str]:
    """Reasons this run must NOT be presented as a signable gate (empty list == it is one).

    A reviewer must not be able to sign a checklist built from synthetic fixtures or from
    a 30-image smoke run. Any reason here replaces the signature block with the
    SYNTHETIC banner.
    """
    qa = cfg["crop_qa"]
    reasons: list[str] = []
    n_images, n_pats = len(side), int(side["empi_anon"].nunique())
    min_images = int(qa["image_audit_min_images"])
    min_pats = int(qa["laterality_audit_min_patients"])
    if n_images < min_images:
        reasons.append(f"only {n_images} crops in the reviewed splits, below the configured "
                       f"image_audit_min_images = {min_images}")
    if n_pats < min_pats:
        reasons.append(f"only {n_pats} patients present, below the configured "
                       f"laterality_audit_min_patients = {min_pats}")
    for label in ("dicom_root", "out_dir"):
        p = str(run_record.get(label, "") or "")
        hit = next((m for m in SYNTHETIC_PATH_MARKERS if m in p.lower()), None)
        if hit:
            reasons.append(f"the preprocess run record's {label} points at a scratch/synthetic "
                           f"location (matched {hit!r}): {_rel(p)}")
    sched = int(run_record.get("n_images_scheduled", 0) or 0)
    nfail = int(run_record.get("n_failures", 0) or 0)
    if sched and nfail / sched > 0.5:
        reasons.append(f"the source run failed on {nfail}/{sched} scheduled images "
                       f"({100.0 * nfail / sched:.1f}%) — this is not a completed preprocessing run")
    if len(sampled) and n_crop_panels < len(sampled):
        reasons.append(f"{len(sampled) - n_crop_panels} of {len(sampled)} sampled crops could "
                       f"NOT be read from the shards")
    return reasons


# =============================================================================
# Checklist
# =============================================================================
def residual_marker_scan(images: list[np.ndarray], views: list[str], params) -> dict:
    """Measure burned-in markers that SURVIVED into the finished crops.

    This deliberately measures the artifact rather than trusting a producer-side counter:
    `src.preprocess_images.mask_burned_in_markers` reports what it removed, but the number
    the gate needs is what is still THERE. Non-negotiable #1 is "zero laterality markers",
    and protocol section 22 treats surviving text as a shortcut the model can exploit.

    A residual is a saturated, small, background-isolated blob — the same signature the
    masker uses, so anything it reports is something the masker declined to remove. Note
    that saturated BONE edges also match, so this is an upper bound: the reviewer, not the
    number, decides whether a given crop carries real text.
    """
    if not images:
        return {"n": 0, "pct": 0.0, "by_view": {}, "mean_blobs": 0.0}
    sat = int(getattr(params, "marker_sat_level", 250))
    lo = int(getattr(params, "marker_min_px", 20))
    hi = float(getattr(params, "marker_max_area_frac", 0.01))
    per_view: dict[str, list[int]] = {}
    counts: list[int] = []
    for img, vw in zip(images, views):
        hot = img >= sat
        n_blob = 0
        if hot.any():
            lab, n = ndi.label(hot)
            if n:
                sizes = ndi.sum(hot, lab, range(1, n + 1))
                biggest = int(np.argmax(sizes)) + 1
                area = float(img.size)
                for k in range(1, n + 1):
                    if k == biggest:
                        continue
                    if lo <= float(sizes[k - 1]) <= hi * area:
                        n_blob += 1
        counts.append(n_blob)
        per_view.setdefault(str(vw), []).append(n_blob)
    n_with = sum(1 for c in counts if c > 0)
    return {
        "n": n_with,
        "total": len(counts),
        "pct": 100.0 * n_with / len(counts),
        "mean_blobs": float(np.mean(counts)),
        "by_view": {k: 100.0 * sum(1 for c in v if c > 0) / len(v) for k, v in sorted(per_view.items())},
    }


def write_checklist(path: Path, *, criteria: list[str], sampled: pd.DataFrame, cells: list[str],
                    n_crop_panels: int, n_film_panels: int, side: pd.DataFrame,
                    splits: list[str], n_per_cell: int, fallback: dict, masked: dict,
                    lat_violations: dict, audit: pd.DataFrame, audit_min: int,
                    contact_sheet: Path, audit_patient_path: Path, audit_summary_path: Path,
                    shard_dir: Path, gate_reasons: list[str], degraded: bool, film_diag: dict,
                    image_audit_info: dict, image_audit_path: Path, image_audit_summary: Path,
                    outcome_info: dict, outcome_path: Path, outcome_cpt_path: Path,
                    outcome_summary: Path, panel_dir: Path | None, qa: dict,
                    residual: dict | None = None) -> None:
    L: list[str] = []
    A = L.append
    A("# Crop QA sign-off — contralateral-knee crops")
    A("")
    if gate_reasons:
        A(f"> {SYNTHETIC_BANNER}")
        A(">")
        A("> This file was generated from a run that is NOT a valid quality gate, so the "
          "sign-off block has been REMOVED. Do not treat anything below as reviewed evidence.")
        A(">")
        for r in gate_reasons:
            A(f"> - {r}")
        A("")
    if degraded:
        A(f"> {DEGRADED_BANNER}")
        A(">")
        A("> The contact sheet was rendered WITHOUT `--dicom-root`, so it shows finished crops "
          "only. Every pre-index film in this cohort shows TWO NATIVE knees; `standardize_to_left` "
          "then mirrors right knees and the border mask removes any burned-in L/R marker. A "
          "finished crop is therefore anatomically identical whether the CORRECT or the WRONG "
          "half was taken, and criterion 1 below (\"the outlined half ... is the CONTRALATERAL "
          "knee\") CANNOT BE ANSWERED from this sheet.")
        A(">")
        A("> Re-run as `python3 -m src.crop_qa --dicom-root <DICOM root>` to render the full "
          "film with the selected half outlined. Until then the half-select is UNVERIFIED.")
        if film_diag.get("errors"):
            A(f">\n> Diagnostics: attempted {film_diag.get('attempted', 0)}, "
              f"missing {film_diag.get('missing_file', 0)}, errors {film_diag.get('decode_error', 0)} "
              f"({'; '.join(film_diag['errors'])})")
        A("")

    A("**TRAINING IS BLOCKED UNTIL THIS FILE IS SIGNED.** Success non-negotiable #1 of the "
      "approved plan is contralateral crop fidelity. 4,075 of the 4,269 frontal films in this "
      "cohort are bilateral (BOTH knees on one image), so the correct knee is selected from "
      "metadata (`laterality == 'B'`, `contra_side`, `horizontal_flip`, and the radiological "
      "display convention). An inverted sign there would train the model on the knee that has "
      "ALREADY been replaced, and every downstream metric would be meaningless. No automated "
      "test can close that gap; a human must look at the images below.")
    A("")
    A("`horizontal_flip` is a MODEL-INFERRED MRKR annotation with an unquantified error rate "
      "and no DICOM tag to check it against. It drives the half-select on 287 images. That is "
      "the specific thing panel A of the contact sheet exists to let you verify.")
    A("")
    A("## 1. Reviewer criteria — tick every box")
    A("")
    A("How to read a contact-sheet row: the **left panel** is the full film, flip-corrected and "
      "NOT mirrored, with the half the pipeline KEPT outlined in green and the discarded half "
      "marked INDEX in red. The **right panel** is that crop before the left/right mirror. "
      "Judge criterion 1 on the LEFT panel; criteria 2 and 3 on the RIGHT panel.")
    A("")
    for c in criteria:
        A(f"- [ ] {c}")
    A("- [ ] the sample above is representative (all view x side cells populated)")
    A("")
    A("## 2. Evidence")
    A("")
    A(f"- Contact sheet: `{_rel(contact_sheet)}`")
    A(f"- Shards inspected: `{_rel(shard_dir)}`")
    A(f"- Splits sampled for the contact sheet: **{', '.join(splits)}** — the LOCKED test split "
      f"is NOT sampled here and stays sealed.")
    A(f"- Target per cell: **{n_per_cell}**; crop panels rendered: **{n_crop_panels}**, "
      f"full-film panels rendered: **{n_film_panels}**, across **{len(cells)}** "
      f"(view x contra_side) cells.")
    if not degraded:
        A(f"- Full films were re-read from the DICOMs for {film_diag.get('ok', 0)} of "
          f"{film_diag.get('attempted', 0)} sampled rows, so half-select is visually decidable.")
    A("")
    A("### Sample composition")
    A("")
    A("| view | contra_side | tiles | fallback crops | mean crop_confidence | mean masked_pct |")
    A("|---|---|---|---|---|---|")
    for cell in cells:
        sub = sampled[sampled["cell"] == cell]
        v, cs = cell.split("|")
        nfb = int((sub["crop_method"] == "fallback_center").sum())
        mp = (f"{100.0 * sub['masked_pct'].astype(float).mean():.1f}%"
              if "masked_pct" in sub.columns else "n/a")
        A(f"| {v} | {cs} | {len(sub)} | {nfb} | "
          f"{sub['crop_confidence'].astype(float).mean():.3f} | {mp} |")
    A("")
    A("### Whole-run crop statistics (all sampled splits, not just the tiles)")
    A("")
    A(f"- Crops written: **{len(side)}** across **{side['empi_anon'].nunique()}** patients.")
    A(f"- **Crop centre: `localizer_mode` = `{qa.get('_localizer_mode', 'center_default')}`.** "
      f"Fallback-localization rate among WRITTEN crops: {fallback['pct']:.2f}% "
      f"({fallback['n']}/{fallback['total']}); mean crop_confidence {fallback['mean_conf']:.3f}. "
      f"Under `center_default` the deterministic CENTRED box is the primary estimate and the "
      f"intensity-profile localizer may only move it when it clears "
      f"`localizer_refine_min_confidence`. On 800 real TRAIN films that localizer fell back on "
      f"26.0% of images (41.4% of single-knee views) and its *successes* were frequently worse "
      f"than the centre — it locks onto a bright shaft or collimator edge — so a centred crop is "
      f"a deliberate choice here, NOT a failure, and nothing is excluded for using one.")
    A("- Fallback rate by view: " + ", ".join(f"{k} {v:.1f}%" for k, v in fallback["by_view"].items()))
    A(f"- **Masked pixels (protocol section 13): mean {100.0 * masked['mean']:.2f}%, "
      f"p90 {100.0 * masked['p90']:.2f}%, max {100.0 * masked['max']:.2f}%** "
      f"(fixed border band {100.0 * masked['band']:.2f}% + out-of-bounds padding; cap "
      f"`max_masked_pct` = {100.0 * masked['cap']:.0f}%). "
      f"{masked['n_above_band']} crops carry padding beyond the border band.")
    if residual:
        A(f"- **Burned-in markers surviving into the finished crops: {residual['pct']:.1f}% of "
          f"the {residual['total']} sampled crops** (mean {residual['mean_blobs']:.2f} blob(s) each; "
          + "by view: " + ", ".join(f"{k} {v:.1f}%" for k, v in residual["by_view"].items()) + "). "
          "Measured on the FINISHED crops, so it is what SURVIVED, not what the masker believes "
          "it removed. This is an UPPER BOUND — saturated bone edges share the signature — so read "
          "it as a list of crops to LOOK AT. Criterion 3 above is the finding of record.")
    A(f"- **Protocol section 13 exclusions (never written to a shard): "
      f"excessive_masking {masked['n_excluded_masking']}, "
      f"localization_failed {masked['n_excluded_localization']}.**")
    A(f"- **Laterality-assertion violations: {lat_violations['n']}** "
      f"(images whose manifest `laterality` did not equal `contra_side` on a single-knee view, "
      f"or whose side could not be resolved). These were routed to the failure report and NOT "
      f"processed.")
    A(f"- All preprocessing failure reasons (counts): {lat_violations['detail']}")
    A(f"- Half-select applied to **{int((side['half_selected'] != 'none').sum())}** bilateral "
      f"frontals; **{int((side['half_selected'] == 'none').sum())}** single-knee films needed none.")
    A(f"- Orientation after standardization: {side['orientation'].value_counts().to_dict()} "
      "(every crop should read as a LEFT knee).")
    A("")
    A("## 3. Protocol section 23 (i) — image-level manual QA")
    A("")
    A(f"- Required: **{image_audit_info.get('min_images')}** index images, "
      f"**{qa['n_reviewers']}** independent reviewers with orthopedic or MSK imaging experience, "
      f"scored on **{', '.join(qa['score_items'])}**.")
    A(f"- Sampled: **{image_audit_info.get('n_sampled', 0)}** images "
      f"(stratified on split x view x contra_side x bilateral/unilateral) from a pool of "
      f"**{image_audit_info.get('pool_size', 0)}**.")
    A(f"- Splits: requested {image_audit_info.get('splits_requested')}, present "
      f"{image_audit_info.get('splits_available')}"
      + (f", MISSING {image_audit_info.get('splits_missing')} (those crops have not been "
         f"generated yet)" if image_audit_info.get("splits_missing") else "") + ".")
    A("- This audit spans **all** splits on purpose. It is LABEL/CROP quality assurance that "
      "precedes model registration: the workbook carries pixels and acquisition metadata and "
      "**no outcome column at all**, so it is NOT outcome-unblinding and does not touch the "
      "sealed-test guarantee.")
    A(f"- By view: {image_audit_info.get('by_view', {})}; by split: "
      f"{image_audit_info.get('by_split', {})}; by laterality kind: "
      f"{image_audit_info.get('by_laterality_kind', {})}.")
    A(f"- Blank adjudication workbook (contains empi_anon — git-ignored): "
      f"`{_rel(image_audit_path)}`")
    if panel_dir is not None:
        A(f"- Reviewer panels (full film + pre-mirror crop, one PNG per row): "
          f"`{_rel(panel_dir)}/`")
    else:
        A("- **No reviewer panels were exported** (no `--dicom-root`). The workbook's "
          "`laterality` item is NOT answerable from a crop alone — re-run with `--dicom-root`.")
    A("")
    A(f"**Scoring convention.** Every `<item>_r<k>` cell takes exactly one of `{SCORE_OK}` or "
      f"`{SCORE_ERROR}` (aliases PASS/FAIL, Y/N, 1/0 are accepted). Leave a cell blank if not "
      f"yet reviewed. Reviewers must not see each other's columns while scoring. Resolve "
      f"disagreements by consensus or a third reviewer and record the result in "
      f"`<item>_adjudicated`.")
    A("")
    A(f"Then run `python3 -m src.crop_qa --score {_rel(image_audit_path)}`. It reports raw "
      f"agreement and Cohen's kappa per item and writes the aggregate to "
      f"`{_rel(image_audit_summary)}`. If any item's critical-error rate exceeds "
      f"**{100.0 * float(qa['critical_error_threshold']):.0f}%** the review MUST be expanded "
      f"before training.")
    A("")
    A("## 4. Protocol section 23 (ii) — outcome-record audit")
    A("")
    A(f"- Required: **{outcome_info.get('min_required')}** outcome records reviewed via CPT "
      f"chronology and available post-event images. Sampled: "
      f"**{outcome_info.get('n_sampled', 0)}**.")
    A(f"- Composition: **{outcome_info.get('n_events_sampled', 0)} events** and "
      f"**{outcome_info.get('n_nonevents_sampled', 0)} non-events** drawn from "
      f"{outcome_info.get('cohort_pool', 0)} cohort patients carrying "
      f"{outcome_info.get('cohort_events', 0)} events. Events are deliberately over-sampled — "
      f"the endpoint being validated is the event, and a sample at the natural prevalence "
      f"would give the reviewer too few to check.")
    A(f"- side_source: {outcome_info.get('by_side_source', {})}; split: "
      f"{outcome_info.get('by_split', {})}.")
    A(f"- {outcome_info.get('n_with_cpt_rows', 0)} of the sampled records carry at least one "
      f"knee-arthroplasty CPT row ({outcome_info.get('n_cpt_rows_total', 0)} rows in total).")
    A(f"- Workbook (contains empi_anon — git-ignored): `{_rel(outcome_path)}`")
    A(f"- Full CPT chronology, long format: `{_rel(outcome_cpt_path)}`")
    A(f"- Aggregate-only summary in outputs: `{_rel(outcome_summary)}`")
    A("- Reviewer fills `reviewer_event_confirmed_Y_N`, `reviewer_event_date`, "
      "`reviewer_event_side`, `reviewer_agrees_with_primary_event_Y_N` and `reviewer_notes`.")
    A("- The sample spans all splits because it validates the LABEL-EXTRACTION algorithm, not "
      "the model. Adjudication results must not be used to select, tune or threshold any "
      "model; a discrepancy re-opens the cohort lock and the labels are rebuilt for every "
      "patient, which is the only legitimate response.")
    A("")
    A("## 5. Laterality QA audit (protocol section 7)")
    A("")
    A(f"- Patient sample: **{len(audit)}** (minimum required {audit_min}).")
    if len(audit) < audit_min:
        A(f"- **AUDIT NOT SATISFIED: only {len(audit)} patients are present in the processed "
          f"shards, below the required {audit_min}.** Re-run src.preprocess_images over the full "
          "train+val manifest and re-run this gate before the audit can be adjudicated.")
    A(f"- side_source composition: {audit['side_source'].value_counts().to_dict()}.")
    A("- The sample **deliberately over-samples `side_source == \"recovered\"`** patients "
      "(1,828 of 3,709 in the cohort). Those patients' index laterality was INFERRED from "
      "concordant signals rather than coded on the CPT modifier, so they carry all of the "
      "residual side-assignment risk; the coded patients are included only as a control.")
    A("- `index_side` **cannot be judged from a contralateral crop** — the crop is by "
      "construction the other knee. Each row therefore also carries `qa_panel_png`, the "
      "full-film panel, which is what makes the question answerable.")
    A(f"- Patient-level adjudication file (contains empi_anon — git-ignored): "
      f"`{_rel(audit_patient_path)}` (crops alongside it in `{_rel(audit_patient_path.parent)}/"
      f"{AUDIT_CROP_DIR}/`)")
    A(f"- Aggregate-only summary in outputs: `{_rel(audit_summary_path)}`")
    A("- Reviewer fills `reviewer_index_side`, `reviewer_agrees_Y_N` and `reviewer_notes` per row. "
      "The audit PASSES only if the reviewed index side matches the coded/recovered `index_side` "
      "for every row; any disagreement re-opens the cohort lock.")
    A("")
    A("## 6. Sign-off")
    A("")
    if gate_reasons:
        A(f"{SYNTHETIC_BANNER} — **the sign-off block is intentionally absent.** This run does "
          "not qualify as a gate for the reasons listed at the top of this file, so there is "
          "nothing here to sign. Re-run `src.preprocess_images` over the real DICOM tree and "
          "then `src.crop_qa --dicom-root <DICOM root>` to produce a signable checklist.")
        A("")
        A("`notebooks/train_colab.ipynb` must not be run.")
    elif degraded:
        A(f"{DEGRADED_BANNER} — **the sign-off block is intentionally absent.** Criterion 1 "
          "cannot be answered from a crop-only contact sheet, so a signature here would "
          "certify something no one could see. Re-run with `--dicom-root <DICOM root>`.")
        A("")
        A("`notebooks/train_colab.ipynb` must not be run.")
    else:
        A("Signing means: I inspected the contact sheet, the outlined half of each full film is "
          "the contralateral knee, no index-knee pixels or laterality markers survive in the "
          "crops, and the protocol section 23 audits above are adjudicated and passed.")
        A("")
        A("| field | value |")
        A("|---|---|")
        A("| Reviewer name | |")
        A("| Role | |")
        A("| Date (YYYY-MM-DD) | |")
        A("| Signature | |")
        A("| Result (PASS / FAIL) | |")
        A("| If FAIL, defect and required fix | |")
        A("")
        A("Until `Result` reads PASS, `notebooks/train_colab.ipynb` must not be run.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


# =============================================================================
# Driver
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Crop QA gate: two-panel contact sheet + reviewer "
                                             "checklist + protocol section 23 audits.")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--shard-dir", default=None,
                    help="directory holding the .tar shards (default: last src.preprocess_images run)")
    ap.add_argument("--dicom-root", default=None,
                    help="root of the DICOM tree. REQUIRED for a valid gate: without it the "
                         "contact sheet cannot show the full film, and half-select is not "
                         "visually verifiable.")
    ap.add_argument("--no-audit-crops", action="store_true",
                    help="skip exporting the audit-sample PNGs to derived-data/")
    ap.add_argument("--no-audit-panels", action="store_true",
                    help="skip exporting the per-row film+crop reviewer panels")
    ap.add_argument("--score", default=None,
                    help="score a COMPLETED protocol section 23 image-audit workbook and exit")
    ap.add_argument("--rebuild-image-audit", action="store_true",
                    help="rebuild ONLY the protocol section 23 (i) workbook and its missing "
                         "reviewer panels, then exit. Does not touch the contact sheet, the "
                         "checklist, or the other two audits.")
    ap.add_argument("--sidecar", action="append", default=None, metavar="LABELS_CSV",
                    help="preprocess sidecar to fold into the audit frame; repeatable. Its "
                         "parent directory is used as the shard dir. Defaults to the sidecar "
                         "named by preprocess_run.json (and preprocess_run_test.json if "
                         "present), because one run record cannot describe two runs.")
    ap.add_argument("--backup-workbook", default=None, metavar="FILENAME",
                    help="with --rebuild-image-audit, copy the existing workbook to this name "
                         "under the cohort dir before overwriting it")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    qa = cfg["crop_qa"]

    # ---- scorer mode: reads a filled workbook, writes the aggregate, exits ----
    if args.score:
        return score_image_audit(cfg, Path(args.score).expanduser(), log)

    # ---- targeted rebuild of the protocol section 23 (i) sample ----
    if args.rebuild_image_audit:
        sidecars = [Path(s).expanduser() for s in (args.sidecar or [])]
        if not sidecars:
            coh = cfg.path(cfg["paths"]["cohort_dir"])
            for name in ("preprocess_run.json", "preprocess_run_test.json"):
                rec = coh / name
                if rec.exists():
                    try:
                        sidecars.append(Path(json.loads(rec.read_text())["sidecar_csv"]))
                    except Exception as exc:
                        log.warning("could not read %s (%s)", name, exc)
            sidecars = list(dict.fromkeys(sidecars))
        if not sidecars:
            log.error("no --sidecar given and no preprocess run record names one")
            return 2
        return rebuild_image_audit(cfg, sidecars, log,
                                   write_panels=not args.no_audit_panels,
                                   backup_name=args.backup_workbook)

    coh = cfg.path(cfg["paths"]["cohort_dir"])
    seed = int(cfg["reproducibility"]["random_seed"])
    views = list(cfg["preprocess"]["views_kept"])
    image_format = str(cfg["preprocess"]["shards"]["image_format"])
    splits = list(qa["splits_sampled"])
    assert "test" not in splits, "the crop-QA contact sheet must never sample the LOCKED test split"
    params = PreprocessParams.from_config(cfg)

    shard_dir, sidecar, how = resolve_shard_dir(cfg, args.shard_dir)
    if shard_dir is None or sidecar is None or not Path(sidecar).exists():
        log.error("No crops to review (%s).", how)
        log.error("Run src.preprocess_images first, e.g.: python3 -m src.preprocess_images "
                  "--dicom-root <DICOM root> --out-dir <shard dir>")
        return 2
    side_all = pd.read_csv(sidecar, dtype={"empi_anon": str, "sop_uid": str})
    if "masked_pct" not in side_all.columns:
        log.warning("sidecar %s predates the protocol-section-13 masked_pct column; re-run "
                    "src.preprocess_images to regenerate it", sidecar)
        side_all["masked_pct"] = np.nan
    side = side_all[side_all["split"].isin(splits)].reset_index(drop=True)
    if side.empty:
        log.error("sidecar %s has no rows for splits=%s — nothing to review", sidecar, splits)
        return 2
    log.info("loaded %d crops / %d patients from %s (%s)", len(side), side["empi_anon"].nunique(),
             sidecar, how)
    run_record = load_run_record(cfg)

    dicom_root = Path(args.dicom_root).expanduser() if args.dicom_root else None
    if dicom_root is not None and not dicom_root.is_dir():
        log.warning("--dicom-root %s does not exist; falling back to crop-only tiles", dicom_root)
        dicom_root = None
    if dicom_root is None:
        log.warning("*** NO --dicom-root: the contact sheet will show finished crops only. Both "
                    "knees are native on every pre-index film and standardize_to_left removes the "
                    "left/right cue, so HALF-SELECT IS NOT VISUALLY VERIFIABLE from this sheet. ***")

    # Opaque per-run patient index; the id->index key stays in git-ignored derived-data/.
    pats = sorted(side_all["empi_anon"].unique())
    index_of = {p: f"P{i + 1:04d}" for i, p in enumerate(pats)}
    pd.DataFrame({"empi_anon": pats, "qa_index": [index_of[p] for p in pats]}).to_csv(
        coh / INDEX_KEY_CSV, index=False)

    paths = load_path_table(cfg, log) if dicom_root is not None else None

    # ---- contact sheet -------------------------------------------------------------
    sampled = sample_cells(side, views, int(qa["n_per_cell"]), seed)
    crops = load_crops(Path(shard_dir), sampled, image_format)
    stages, film_diag = film_stages_for_rows(sampled, paths, dicom_root, params, log)
    degraded = len(stages) < len(sampled)
    contact_sheet = cfg.path(qa["contact_sheet"])

    # Assessed BEFORE rendering so the banner is burned into the image a reviewer opens,
    # in a single pass. Shard-readability is judged on the crops the tars actually yield,
    # since the shards are what training will read.
    n_loadable = int(sum(1 for k in sampled["key"] if k in crops))
    gate_reasons = assess_gate_validity(cfg, side, run_record, sampled, n_loadable)
    for r in gate_reasons:
        log.warning("NOT A GATE: %s", r)
    n_crop_panels, n_film_panels, cells = render_contact_sheet(
        sampled, crops, stages, index_of, views, int(qa["n_per_cell"]), contact_sheet,
        premirror=bool(qa["contact_sheet_premirror"]), degraded=degraded,
        gate_reasons=gate_reasons)
    log.info("contact sheet -> %s (%d crop panels, %d full-film panels, %d cells: %s)",
             contact_sheet, n_crop_panels, n_film_panels, len(cells), cells)

    # ---- whole-run statistics ------------------------------------------------------
    n_fb = int((side["crop_method"] == "fallback_center").sum())
    fallback = {
        "n": n_fb, "total": len(side), "pct": 100.0 * n_fb / max(1, len(side)),
        "mean_conf": float(side["crop_confidence"].astype(float).mean()),
        "by_view": {v: 100.0 * float((side[side["view"] == v]["crop_method"] == "fallback_center").mean())
                    for v in views if (side["view"] == v).any()},
    }
    mp = side["masked_pct"].astype(float)
    masked = {
        "mean": float(np.nanmean(mp)) if len(mp) else float("nan"),
        "p90": float(np.nanpercentile(mp, 90)) if len(mp) else float("nan"),
        "max": float(np.nanmax(mp)) if len(mp) else float("nan"),
        "band": float(run_record.get("border_band_fraction", float("nan"))),
        "cap": float(params.max_masked_pct),
        "n_above_band": int((mp > float(run_record.get("border_band_fraction", 0.0)) + 1e-9).sum()),
        "n_excluded_masking": int(run_record.get("n_excluded_excessive_masking", 0) or 0),
        "n_excluded_localization": int(run_record.get("n_excluded_localization_failed", 0) or 0),
    }

    # src.preprocess_images writes fail_report_csv as COUNTS BY REASON (aggregate-safe).
    fail_path = cfg.path(cfg["preprocess"]["fail_report_csv"])
    lat = {"n": 0, "detail": "no failure report found"}
    if fail_path.exists():
        fdf = pd.read_csv(fail_path)
        if len(fdf):
            hits = fdf[fdf["reason"].isin(LATERALITY_FAIL_REASONS)]
            n_lat = int(hits["n_images"].sum()) if "n_images" in hits.columns else int(len(hits))
            lat = {"n": n_lat,
                   "detail": str(fdf.groupby("reason")["n_images"].sum().to_dict())
                             if "n_images" in fdf.columns else str(fdf["reason"].value_counts().to_dict())}
        else:
            lat = {"n": 0, "detail": "failure report is empty (0 failures of any kind)"}

    # ---- protocol section 23 (i): image-level audit ---------------------------------
    _panelled = {p.stem for p in (coh / QA_PANEL_DIR).glob(f"*.{image_format}")} \
        if (coh / QA_PANEL_DIR).exists() else set()
    image_audit, image_audit_info = build_image_audit(
        cfg, side_all, list(qa["audit_splits"]), int(qa["image_audit_min_images"]),
        list(qa["score_items"]), int(qa["n_reviewers"]), seed, index_of,
        full_film_keys=_panelled)
    image_audit_path = coh / IMAGE_AUDIT_WORKBOOK
    image_audit.to_csv(image_audit_path, index=False)
    if len(image_audit) < int(qa["image_audit_min_images"]):
        log.warning("image audit sample is %d images, BELOW the required %d (pool has %d rows "
                    "for splits %s)", len(image_audit), int(qa["image_audit_min_images"]),
                    image_audit_info.get("pool_size", 0), image_audit_info.get("splits_available"))
    if image_audit_info.get("splits_missing"):
        log.warning("image audit could not cover audit_splits %s: those crops do not exist yet "
                    "(run src.preprocess_images for them; the test split additionally needs "
                    "--include-test)", image_audit_info["splits_missing"])
    log.info("image audit workbook -> %s (%d rows, %d reviewers x %d items)", image_audit_path,
             len(image_audit), int(qa["n_reviewers"]), len(list(qa["score_items"])))

    # ---- protocol section 23 (ii): outcome-record audit -----------------------------
    outcome_audit, outcome_cpt, outcome_info = build_outcome_audit(
        cfg, int(qa["outcome_audit_min_records"]), seed, log)
    outcome_path = coh / OUTCOME_AUDIT_WORKBOOK
    outcome_cpt_path = coh / OUTCOME_AUDIT_CPT_CSV
    outcome_audit.to_csv(outcome_path, index=False)
    outcome_cpt.to_csv(outcome_cpt_path, index=False)
    outcome_summary_path = cfg.path(qa["outcome_audit_csv"])
    outcome_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"metric": k, "value": str(v)} for k, v in outcome_info.items()]
                 + [{"metric": "note",
                     "value": f"patient-level workbook is git-ignored at {_rel(outcome_path)}"}]
                 ).to_csv(outcome_summary_path, index=False)
    if len(outcome_audit) < int(qa["outcome_audit_min_records"]):
        log.warning("outcome-record audit sample is %d, BELOW the required %d",
                    len(outcome_audit), int(qa["outcome_audit_min_records"]))

    # ---- protocol section 7: laterality audit ---------------------------------------
    audit = build_laterality_audit(cfg, side, splits, int(qa["laterality_audit_min_patients"]), seed)
    audit_patient_path = coh / AUDIT_PATIENT_CSV
    audit.to_csv(audit_patient_path, index=False)
    if len(audit) < int(qa["laterality_audit_min_patients"]):
        log.warning("laterality audit sample is %d patients, BELOW the required minimum %d "
                    "(only %d patients are present in the processed shards)",
                    len(audit), int(qa["laterality_audit_min_patients"]), side["empi_anon"].nunique())

    # outputs/ is NOT git-ignored: only aggregates go there, never a patient row.
    audit_summary_path = cfg.path(qa["laterality_audit_csv"])
    src_counts = audit["side_source"].value_counts().to_dict()
    summary = pd.DataFrame([
        {"metric": "n_patients_sampled", "value": len(audit)},
        {"metric": "min_patients_required", "value": int(qa["laterality_audit_min_patients"])},
        {"metric": "n_side_source_recovered", "value": int(src_counts.get("recovered", 0))},
        {"metric": "n_side_source_coded", "value": int(src_counts.get("coded", 0))},
        {"metric": "pct_recovered", "value": round(100.0 * src_counts.get("recovered", 0) / max(1, len(audit)), 2)},
        {"metric": "splits_sampled", "value": "+".join(splits)},
        {"metric": "n_with_frontal_crop", "value": int((audit["crop_view"] == "frontal").sum())},
        {"metric": "mean_n_concordant_signals", "value": round(float(audit["n_concordant_signals"].mean()), 3)},
        {"metric": "contra_side_L", "value": int((audit["contra_side"] == "L").sum())},
        {"metric": "contra_side_R", "value": int((audit["contra_side"] == "R").sum())},
        {"metric": "note", "value": f"patient-level adjudication file is git-ignored at {_rel(audit_patient_path)}"},
    ])
    audit_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(audit_summary_path, index=False)

    # ---- reviewer assets under derived-data/ ----------------------------------------
    if not args.no_audit_crops:
        crop_dir = coh / AUDIT_CROP_DIR
        crop_dir.mkdir(parents=True, exist_ok=True)
        imgs = load_crops(Path(shard_dir), audit.rename(columns={"crop_shard": "shard",
                                                                "crop_key": "key"}), image_format)
        for k, im in imgs.items():
            Image.fromarray(im, mode="L").save(crop_dir / f"{k}.{image_format}")
        log.info("audit crops -> %s (%d images, git-ignored)", crop_dir, len(imgs))

    panel_dir: Path | None = None
    if not args.no_audit_panels and dicom_root is not None and paths is not None:
        # One film+crop panel per audit row: the laterality item is unanswerable without it.
        keys = pd.concat([
            image_audit[["key", "shard"]],
            audit.rename(columns={"crop_key": "key", "crop_shard": "shard"})[["key", "shard"]],
        ]).dropna(subset=["key"]).drop_duplicates("key")
        want = keys.merge(side_all[["key", "sop_uid"]], on="key", how="left")
        panel_stages, panel_diag = film_stages_for_rows(want, paths, dicom_root, params, log)
        panel_crops = load_crops(Path(shard_dir), want, image_format)
        panel_dir = coh / QA_PANEL_DIR
        panel_dir.mkdir(parents=True, exist_ok=True)
        meta = side_all.set_index("key")
        for k in want["key"]:
            m = meta.loc[k] if k in meta.index else None
            label = (f"{index_of.get(str(m['empi_anon']), '?')} {str(m['view'])[:4]} "
                     f"lat={m['laterality']} idx={m['index_side']} con={m['contra_side']} "
                     f"hf={m['horizontal_flip']}" if m is not None else str(k))
            save_qa_panel(panel_stages.get(k), panel_crops.get(k), panel_dir / f"{k}.png", label)
        log.info("reviewer panels -> %s (%d panels, %d with a full film; git-ignored)",
                 panel_dir, len(want), panel_diag["ok"])
        # The workbook is written before the panels exist, so tell it which rows ended up
        # with a decidable (full-film) panel — the `laterality` item depends on it.
        image_audit["panel_has_full_film"] = (image_audit["panel_has_full_film"].astype(bool)
                                              | image_audit["key"].isin(set(panel_stages)))
        image_audit.to_csv(image_audit_path, index=False)

    # ---- residual burned-in markers (protocol section 13 / non-negotiable #1) --------
    # Measured on the FINISHED crops, not from a producer-side counter: the gate needs to
    # know what SURVIVED, not what the masker believes it removed.
    _scan_imgs = [crops[k] for k in sampled["key"] if k in crops]
    _scan_views = [str(v) for k, v in zip(sampled["key"], sampled["view"]) if k in crops]
    residual = residual_marker_scan(_scan_imgs, _scan_views, params)
    qa = {**qa, "_localizer_mode": getattr(params, "localizer_mode", "center_default")}

    # ---- checklist ------------------------------------------------------------------
    checklist = cfg.path(qa["checklist_md"])
    write_checklist(
        checklist, criteria=list(qa["signoff_criteria"]), sampled=sampled, cells=cells,
        n_crop_panels=n_crop_panels, n_film_panels=n_film_panels, side=side, splits=splits,
        n_per_cell=int(qa["n_per_cell"]), fallback=fallback, masked=masked, lat_violations=lat,
        audit=audit, audit_min=int(qa["laterality_audit_min_patients"]),
        contact_sheet=contact_sheet, audit_patient_path=audit_patient_path,
        audit_summary_path=audit_summary_path, shard_dir=Path(shard_dir),
        gate_reasons=gate_reasons, degraded=degraded, film_diag=film_diag,
        image_audit_info=image_audit_info, image_audit_path=image_audit_path,
        image_audit_summary=cfg.path(qa["image_audit_csv"]), outcome_info=outcome_info,
        outcome_path=outcome_path, outcome_cpt_path=outcome_cpt_path,
        outcome_summary=outcome_summary_path, panel_dir=panel_dir, qa=qa,
        residual=residual)

    log.info("fallback-localization rate %.2f%% (%d/%d); mean crop_confidence %.3f",
             fallback["pct"], fallback["n"], fallback["total"], fallback["mean_conf"])
    log.info("masked_pct: mean %.4f p90 %.4f max %.4f (band %.4f, cap %.2f); excluded: "
             "excessive_masking=%d localization_failed=%d", masked["mean"], masked["p90"],
             masked["max"], masked["band"], masked["cap"], masked["n_excluded_masking"],
             masked["n_excluded_localization"])
    log.info("laterality-assertion violations: %d (%s)", lat["n"], lat["detail"])
    log.info("laterality audit sample: %d patients (%s) -> %s ; aggregate -> %s",
             len(audit), src_counts, audit_patient_path, audit_summary_path)
    log.info("outcome-record audit: %d records -> %s ; aggregate -> %s",
             len(outcome_audit), outcome_path, outcome_summary_path)
    log.info("checklist -> %s", checklist)
    if gate_reasons:
        log.error("%s — the checklist carries NO signature block. Reasons: %s",
                  SYNTHETIC_BANNER.replace("**", ""), "; ".join(gate_reasons))
        return 0
    if degraded:
        log.error("%s — the checklist carries NO signature block: half-select could not be shown. "
                  "Re-run with --dicom-root <DICOM root>.", DEGRADED_BANNER.replace("**", ""))
        return 0
    log.info("TRAINING IS BLOCKED until %s is signed with Result = PASS.", checklist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
