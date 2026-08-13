"""test_crop_qa.py — the QA gate's own logic, tested without shards or DICOMs.

The gate exists to stop a reviewer signing off on something they could not actually see.
Six things must therefore never regress:
  1. `assess_gate_validity` refuses to present a synthetic / undersized run as a gate;
  2. `cohens_kappa` and the score parser implement protocol section 23 correctly,
     including the case where kappa is UNDEFINED (both reviewers used one category);
  3. `_rel` keeps machine paths out of the tracked checklist;
  4. NOT_ASSESSABLE is a verdict, not a blank: it leaves the agreement denominator and is
     counted separately, because the test-split rows have no full film to judge
     `laterality` on and pretending otherwise would fabricate a reviewed result;
  5. a critical-error rate over the 2% threshold makes `--score` EXIT NON-ZERO — a gate
     that logs failure and exits 0 is not a gate;
  6. re-drawing the sample over an enlarged frame keeps the rows whose reviewer panel is
     still decidable (`stratified_sample(prefer=...)`), and the opaque patient index is
     stable, so a re-draw cannot renumber a patient between two versions of a sample.

The protocol section-23(i) review these facilities were built for was DECLINED on
2026-08-12 (deviation D40), and the scoring application, the workbooks and the reviewer
panels were removed with it; the five tests that covered `src/qa_review_app.py` came out
here in the same change. Everything above is still tested because `src/crop_qa.py` stays
in the project: `src/interpretability.py` imports `residual_marker_scan` from it, and that
function produced the residual-marker rate the paper reports as its leakage evidence. Do
not rebuild the scoring app in order to "complete" a declined review; reversing that
decision is a new register entry, not a restoration.

Run:  python3 -m pytest tests/test_crop_qa.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import PROJECT_ROOT, load_config
from src.crop_qa import (
    SCORE_ERROR, SCORE_NA, SCORE_OK, _normalize_score, _rel, assess_gate_validity,
    build_image_audit, cohens_kappa, load_sidecars, sample_cells, score_image_audit,
    stable_index_map, stratified_sample,
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


# =============================================================================
# 5. NOT_ASSESSABLE — the verdict that is neither OK, ERROR, nor blank
# =============================================================================
def test_not_assessable_parses_and_bare_NA_still_means_not_reviewed():
    for v in ("NOT_ASSESSABLE", "not assessable", " N/A ", "unassessable", "CANNOT_ASSESS",
              "undecidable"):
        assert _normalize_score(v) == SCORE_NA
    # "NA" without a slash has always meant "no answer yet" and must keep meaning that,
    # or every historic workbook silently changes meaning.
    assert _normalize_score("NA") is None
    assert SCORE_NA not in (SCORE_OK, SCORE_ERROR)


def _scoring_config(tmp_path, workbook: pd.DataFrame):
    """A config whose cohort/summary paths point into tmp_path, plus the workbook on disk."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg_dict = yaml.safe_load((PROJECT_ROOT / "config" / "feasibility.yaml").read_text())
    cfg_dict["paths"]["run_log"] = str(tmp_path / "run.log")
    cfg_dict["crop_qa"]["image_audit_csv"] = str(tmp_path / "image_audit_summary.csv")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    wb = tmp_path / "workbook.csv"
    workbook.to_csv(wb, index=False)
    return load_config(path), wb


def _filled_workbook(n=100, **overrides) -> pd.DataFrame:
    items = ["laterality", "view", "native_knee", "crop_adequacy", "burned_in_text",
             "non_knee_content"]
    wb = pd.DataFrame({"key": [f"k{i}" for i in range(n)]})
    for it in items:
        wb[f"{it}_r1"] = SCORE_OK
        wb[f"{it}_r2"] = SCORE_OK
    for col, vals in overrides.items():
        wb[col] = vals
    return wb


def test_not_assessable_rows_leave_the_denominator_and_are_counted(tmp_path, caplog):
    """20 of 100 rows undecidable -> n_scored 80, n_not_assessable 20, rate over 80."""
    wb = _filled_workbook(100)
    wb.loc[wb.index[:20], "laterality_r1"] = SCORE_NA
    wb.loc[wb.index[:20], "laterality_r2"] = SCORE_NA
    wb.loc[wb.index[20:21], "laterality_r1"] = SCORE_ERROR      # 1 defect among the 80
    wb.loc[wb.index[20:21], "laterality_r2"] = SCORE_ERROR
    cfg, path = _scoring_config(tmp_path, wb)
    assert score_image_audit(cfg, path, __import__("logging").getLogger("t")) == 0
    out = pd.read_csv(cfg.path(cfg["crop_qa"]["image_audit_csv"]))
    row = out[out["item"] == "laterality"].iloc[0]
    assert int(row["n_scored"]) == 80
    assert int(row["n_not_assessable"]) == 20
    assert float(row["critical_error_rate"]) == pytest.approx(1 / 80)
    assert bool(row["exceeds_threshold"]) is False
    assert int(out[out["item"] == "view"].iloc[0]["n_not_assessable"]) == 0


def test_a_failing_two_percent_gate_exits_non_zero(tmp_path):
    """A gate that logs 'EXPAND THE REVIEW' and exits 0 is not a gate."""
    log = __import__("logging").getLogger("t")
    clean = _filled_workbook(100)
    clean.loc[clean.index[:2], "native_knee_r1"] = SCORE_ERROR
    clean.loc[clean.index[:2], "native_knee_r2"] = SCORE_ERROR   # exactly 2% — not OVER
    cfg, path = _scoring_config(tmp_path, clean)
    assert score_image_audit(cfg, path, log) == 0

    bad = _filled_workbook(100)
    bad.loc[bad.index[:3], "native_knee_r1"] = SCORE_ERROR       # 3% > 2%
    bad.loc[bad.index[:3], "native_knee_r2"] = SCORE_ERROR
    cfg2, path2 = _scoring_config(tmp_path / "bad", bad)
    assert score_image_audit(cfg2, path2, log) == 2


def test_an_unfilled_workbook_is_awaiting_input_not_a_failure(tmp_path):
    wb = _filled_workbook(50)
    for c in [c for c in wb.columns if c.endswith(("_r1", "_r2"))]:
        wb[c] = ""
    cfg, path = _scoring_config(tmp_path, wb)
    assert score_image_audit(cfg, path, __import__("logging").getLogger("t")) == 0
    out = pd.read_csv(cfg.path(cfg["crop_qa"]["image_audit_csv"]))
    assert (out["status"] == "awaiting reviewer input").all()


# =============================================================================
# 6. Re-drawing the sample over an enlarged frame
# =============================================================================
def test_prefer_fills_slots_first_without_changing_strata_or_quotas():
    df = fake_sidecar(n_patients=400)
    plain = stratified_sample(df, ["view", "contra_side"], 200, seed=7)
    keep = set(df["key"].sample(n=150, random_state=1))
    pref = stratified_sample(df, ["view", "contra_side"], 200, seed=7,
                             prefer=df["key"].isin(keep))
    assert len(pref) == len(plain) == 200
    # same design: same strata, same per-stratum counts
    g = ["view", "contra_side"]
    assert pref.groupby(g).size().to_dict() == plain.groupby(g).size().to_dict()
    # and it actually preferred them
    assert pref["key"].isin(keep).sum() > plain["key"].isin(keep).sum()
    assert pref["key"].tolist() == stratified_sample(
        df, g, 200, 7, prefer=df["key"].isin(keep))["key"].tolist()


def test_prefer_cannot_take_more_than_a_stratum_holds():
    df = fake_sidecar(n_patients=20)
    everything = stratified_sample(df, ["view"], 999, seed=7, prefer=df["key"].isin(set()))
    assert len(everything) == len(df)


def test_the_audit_sample_flags_which_rows_have_a_decidable_panel():
    cfg = load_config()
    side = fake_sidecar(n_patients=300)
    side["split"] = np.where(side.index < 400, "train", np.where(side.index < 700, "val", "test"))
    side["index_side"] = "L"
    side["horizontal_flip"] = 0
    with_film = set(side[side["split"] != "test"]["key"])
    take, info = build_image_audit(cfg, side, ["train", "val", "test"], 200,
                                   list(cfg["crop_qa"]["score_items"]), 2, 7, {},
                                   full_film_keys=with_film)
    assert set(take["split"]) == {"train", "val", "test"}
    assert take["panel_has_full_film"].sum() == take["key"].isin(with_film).sum()
    assert info["n_with_full_film_panel"] + info["n_crop_only_panel"] == len(take)
    # every test row is crop-only, so `laterality` is undecidable on exactly those
    assert not take.loc[take["split"] == "test", "panel_has_full_film"].any()


def test_the_opaque_patient_index_is_never_renumbered(tmp_path):
    """It is burned into the panel PNGs; renumbering silently relabels the evidence."""
    key = tmp_path / "key.csv"
    pd.DataFrame({"empi_anon": ["b", "a"], "qa_index": ["P0002", "P0001"]}).to_csv(key, index=False)
    out = stable_index_map(key, ["a", "b", "c", "d"])
    assert out["a"] == "P0001" and out["b"] == "P0002"
    assert out["c"] == "P0003" and out["d"] == "P0004"
    assert stable_index_map(tmp_path / "missing.csv", ["z", "y"]) == {"y": "P0001", "z": "P0002"}


def test_sidecars_from_separate_preprocess_runs_are_unioned(tmp_path):
    """preprocess_run.json records only the LAST run; the audit frame is every run."""
    a, b = fake_sidecar(n_patients=10), fake_sidecar(n_patients=10)
    b["split"] = "test"
    b["key"] = b["key"] + "_t"
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a.to_csv(tmp_path / "a" / "labels.csv", index=False)
    b.to_csv(tmp_path / "b" / "labels.csv", index=False)
    out = load_sidecars([tmp_path / "a" / "labels.csv", tmp_path / "b" / "labels.csv"])
    assert len(out) == len(a) + len(b)
    assert set(out["split"]) == {"train", "val", "test"}
    assert out["_shard_dir"].nunique() == 2
    # a key present in both runs is kept once
    dup = load_sidecars([tmp_path / "a" / "labels.csv", tmp_path / "a" / "labels.csv"])
    assert len(dup) == len(a)
