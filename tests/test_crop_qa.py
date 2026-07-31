"""test_crop_qa.py — the QA gate's own logic, tested without shards or DICOMs.

The gate exists to stop a reviewer signing off on something they could not actually see.
Three things must therefore never regress:
  1. `assess_gate_validity` refuses to present a synthetic / undersized run as a gate;
  2. `cohens_kappa` and the score parser implement protocol section 23 correctly,
     including the case where kappa is UNDEFINED (both reviewers used one category);
  3. `_rel` keeps machine paths out of the tracked checklist.

Run:  python3 -m pytest tests/test_crop_qa.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.crop_qa import (
    SCORE_ERROR, SCORE_OK, _normalize_score, _rel, assess_gate_validity, cohens_kappa,
    sample_cells, stratified_sample,
)


def fake_sidecar(n_patients: int = 300, per_patient: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        for j in range(per_patient):
            rows.append({
                "empi_anon": f"9{i:07d}",
                "key": f"9{i:07d}_v{j}_abc{i:04d}{j}",
                "shard": "train-00000.tar",
                "split": "train" if i % 5 else "val",
                "view": ["frontal", "lateral", "sunrise"][j % 3],
                "contra_side": "L" if i % 2 else "R",
                "laterality": "B" if j == 0 else ("L" if i % 2 else "R"),
                "crop_method": "intensity_profile",
                "crop_confidence": 0.5,
                "masked_pct": 0.2275,
                "orientation": "left",
                "half_selected": "left" if j == 0 else "none",
            })
    return pd.DataFrame(rows)


# =============================================================================
# 1. The synthetic / not-a-gate guard (BD-6)
# =============================================================================
def test_a_real_looking_run_is_accepted_as_a_gate():
    cfg = load_config()
    side = fake_sidecar()
    run = {"dicom_root": "/Volumes/mrkr/DICOMs", "out_dir": "/Volumes/mrkr/shards",
           "n_images_scheduled": 600, "n_failures": 3}
    sampled = side.head(12).assign(cell="frontal|L")
    assert assess_gate_validity(cfg, side, run, sampled, n_crop_panels=12) == []


@pytest.mark.parametrize("field", ["dicom_root", "out_dir"])
def test_a_scratchpad_path_disqualifies_the_run(field):
    cfg = load_config()
    side = fake_sidecar()
    run = {"dicom_root": "/Volumes/mrkr/DICOMs", "out_dir": "/Volumes/mrkr/shards",
           "n_images_scheduled": 600, "n_failures": 3}
    run[field] = "/private/tmp/claude-501/xyz/scratchpad/dicom_root"
    reasons = assess_gate_validity(cfg, side, run, side.head(12).assign(cell="frontal|L"), 12)
    assert any(field in r and "scratch/synthetic" in r for r in reasons)


def test_an_undersized_or_mostly_failed_run_disqualifies_the_gate():
    cfg = load_config()
    small = fake_sidecar(n_patients=15)                     # 30 crops / 15 patients
    run = {"dicom_root": "/Volumes/mrkr/DICOMs", "out_dir": "/Volumes/mrkr/shards",
           "n_images_scheduled": 4872, "n_failures": 4843}
    reasons = assess_gate_validity(cfg, small, run, small.head(12).assign(cell="frontal|L"), 12)
    joined = " | ".join(reasons)
    assert "image_audit_min_images" in joined
    assert "laterality_audit_min_patients" in joined
    assert "not a completed preprocessing run" in joined


def test_unreadable_tiles_disqualify_the_gate():
    cfg = load_config()
    side = fake_sidecar()
    run = {"dicom_root": "/Volumes/mrkr/DICOMs", "out_dir": "/Volumes/mrkr/shards",
           "n_images_scheduled": 600, "n_failures": 0}
    sampled = side.head(12).assign(cell="frontal|L")
    reasons = assess_gate_validity(cfg, side, run, sampled, n_crop_panels=7)
    assert any("could NOT be read" in r for r in reasons)


# =============================================================================
# 2. Protocol section 23 scoring
# =============================================================================
def test_cohens_kappa_matches_the_hand_computed_value():
    """4 both-ERROR, 2 r1-only, 4 r2-only, 90 both-OK  ->  po .94, pe .8696, kappa .5399."""
    r1 = ["ERROR"] * 4 + ["ERROR"] * 2 + ["OK"] * 4 + ["OK"] * 90
    r2 = ["ERROR"] * 4 + ["OK"] * 2 + ["ERROR"] * 4 + ["OK"] * 90
    assert len(r1) == len(r2) == 100
    assert cohens_kappa(r1, r2) == pytest.approx((0.94 - 0.8696) / (1 - 0.8696), abs=1e-9)
    assert cohens_kappa(r1, r2) == pytest.approx(0.5398773006, abs=1e-9)


def test_cohens_kappa_is_undefined_when_neither_reviewer_varies():
    """The MOST LIKELY clean-audit result. Reporting 0.0 here would read as chance-level."""
    assert np.isnan(cohens_kappa(["OK"] * 50, ["OK"] * 50))
    assert cohens_kappa(["OK"] * 49 + ["ERROR"], ["OK"] * 49 + ["ERROR"]) == pytest.approx(1.0)
    assert np.isnan(cohens_kappa([], []))
    assert np.isnan(cohens_kappa(["OK"], ["OK", "OK"]))


def test_cohens_kappa_is_zero_for_chance_agreement():
    r1 = ["OK", "OK", "ERROR", "ERROR"]
    r2 = ["OK", "ERROR", "OK", "ERROR"]
    assert cohens_kappa(r1, r2) == pytest.approx(0.0)


def test_score_parser_accepts_the_documented_aliases_and_flags_the_rest():
    for v in ("OK", "ok", " Pass ", "Y", "yes", "1", "TRUE", "correct"):
        assert _normalize_score(v) == SCORE_OK
    for v in ("ERROR", "fail", "N", "no", "0", "False", "WRONG"):
        assert _normalize_score(v) == SCORE_ERROR
    for v in ("", "  ", None, float("nan"), "NA"):
        assert _normalize_score(v) is None
    assert _normalize_score("probably fine") == "UNPARSED"


def test_score_items_and_thresholds_come_from_config():
    qa = load_config()["crop_qa"]
    assert qa["score_items"] == ["laterality", "view", "native_knee", "crop_adequacy",
                                 "burned_in_text", "non_knee_content"]
    assert int(qa["n_reviewers"]) == 2
    assert float(qa["critical_error_threshold"]) == 0.02
    assert int(qa["image_audit_min_images"]) >= 400
    assert int(qa["outcome_audit_min_records"]) >= 200
    assert set(qa["audit_splits"]) == {"train", "val", "test"}
    # the contact sheet itself must never sample the sealed test split
    assert "test" not in qa["splits_sampled"]
    assert qa["show_full_film_in_contact_sheet"] is True
    assert qa["contact_sheet_premirror"] is True


# =============================================================================
# 3. Sampling
# =============================================================================
def test_stratified_sample_covers_every_stratum_and_hits_the_target():
    df = fake_sidecar(n_patients=400)
    take = stratified_sample(df, ["view", "contra_side"], 400, seed=7)
    assert len(take) == 400
    assert set(map(tuple, take[["view", "contra_side"]].drop_duplicates().to_numpy())) == \
        set(map(tuple, df[["view", "contra_side"]].drop_duplicates().to_numpy()))
    # deterministic given the seed
    assert take.index.tolist() == stratified_sample(df, ["view", "contra_side"], 400, 7).index.tolist()


def test_stratified_sample_cannot_exceed_the_pool():
    df = fake_sidecar(n_patients=5)
    assert len(stratified_sample(df, ["view"], 400, seed=7)) == len(df)
    assert stratified_sample(df.head(0), ["view"], 10, seed=7).empty


def test_sample_cells_is_capped_per_cell_and_deterministic():
    df = fake_sidecar(n_patients=300)
    out = sample_cells(df, ["frontal", "lateral", "sunrise"], 12, seed=7)
    assert (out.groupby("cell").size() <= 12).all()
    assert out["cell"].nunique() == 6
    assert out["key"].tolist() == sample_cells(df, ["frontal", "lateral", "sunrise"], 12, 7)["key"].tolist()


# =============================================================================
# 4. Data hygiene in the tracked checklist
# =============================================================================
def test_rel_is_project_relative_inside_and_redacted_outside():
    assert _rel(PROJECT_ROOT / "outputs" / "figures" / "x.png") == "outputs/figures/x.png"
    out = _rel("/private/tmp/claude-501/some-session/scratchpad/shards")
    assert out == ".../scratchpad/shards"
    assert "/private/tmp" not in out and "claude-501" not in out
    assert _rel("/Users/someone/Library/CloudStorage/GoogleDrive-x/My Drive/DICOMs") == \
        ".../My Drive/DICOMs"
