"""Unit tests for src/interpretability.py.

Two things are worth testing here and one is not.

WORTH TESTING: the geometry (a region partition that does not partition, or a medial mask
on the wrong side, would invert the paper's anatomic claim silently), and the
``encode_masked_only`` slot reindexing (a CAM attributed to the wrong patient is invisible
in every downstream number). The reindexing test builds a padded batch where the padding
pattern is deliberately ragged, which is the only situation in which the bug can appear.

NOT WORTH TESTING here: whether the frozen checkpoints reproduce the published hazards.
That is the Phase-0 reproduction gate and it needs 2 GB of staged artefacts; these tests
run without any of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import interpretability as I

torch = pytest.importorskip("torch")

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"


def _table(name: str) -> pd.DataFrame:
    p = TABLES / name
    if not p.exists():                                    # a fresh clone has no outputs/
        pytest.skip(f"{p} not present")
    return pd.read_csv(p)


# --------------------------------------------------------------------------- #
# Region geometry                                                              #
# --------------------------------------------------------------------------- #
def test_joint_border_peripheral_partition_the_image_exactly():
    m = I.region_masks()
    assert not (m["joint"] & m["border"]).any()
    assert not (m["joint"] & m["peripheral"]).any()
    assert not (m["border"] & m["peripheral"]).any()
    assert (m["joint"] | m["border"] | m["peripheral"]).all()


def test_region_area_fractions_match_the_protocol_numbers():
    a = I.region_area_fractions()
    # 1 - (450/512)^2, the band src/preprocess_images.border_band_fraction blanks
    assert a["border"] == pytest.approx(0.227524, abs=5e-7)
    assert round(a["border"] * 100, 2) == 22.75
    assert a["joint"] == pytest.approx(0.25, abs=1e-12)
    assert a["medial"] == pytest.approx(0.125, abs=1e-12)
    assert a["lateral"] == pytest.approx(0.125, abs=1e-12)
    assert a["joint"] + a["border"] + a["peripheral"] == pytest.approx(1.0, abs=1e-12)


def test_medial_is_the_image_left_half_of_the_joint_box():
    """Crops are mirrored to read as LEFT knees under the radiological convention, so the
    medial compartment is on the image LEFT. Inverting this inverts the paper's claim."""
    m = I.region_masks()
    n = I.OUT_SIZE
    assert m["medial"][:, n // 2:].sum() == 0
    assert m["lateral"][:, :n // 2].sum() == 0
    assert np.array_equal(m["medial"] | m["lateral"], m["joint"])
    cols = np.flatnonzero(m["medial"].any(axis=0))
    assert cols.min() == I.JOINT_BOX[2] and cols.max() == n // 2 - 1


def test_joint_box_never_touches_the_masked_border_band():
    r0, r1, c0, c1 = I.JOINT_BOX
    assert r0 > I.BORDER_PX and c0 > I.BORDER_PX
    assert r1 < I.OUT_SIZE - I.BORDER_PX and c1 < I.OUT_SIZE - I.BORDER_PX


def test_detect_joint_row_finds_a_synthetic_dark_line_between_bright_bone():
    img = np.full((512, 512), 40, dtype=np.uint8)
    img[100:300, 150:360] = 200          # femur
    img[320:470, 150:360] = 200          # tibia
    img[300:320, 150:360] = 20           # the joint space
    assert abs(I.detect_joint_row(img) - 310) <= 12


# --------------------------------------------------------------------------- #
# Pixel operations                                                             #
# --------------------------------------------------------------------------- #
def _params():
    return {"sat": 250, "lo": 20, "hi_frac": 0.01}


def test_occlusion_blanks_exactly_the_named_region_and_nothing_else():
    m = I.region_masks()
    img = np.full((512, 512), 137, dtype=np.uint8)
    out = I.apply_pixel_op(img, I.PixelOp("x", zero=("medial",)), m, view="frontal",
                           marker_params=_params())
    assert (out[m["medial"]] == 0).all()
    assert (out[~m["medial"]] == 137).all()


def test_keep_only_blanks_the_complement():
    m = I.region_masks()
    img = np.full((512, 512), 200, dtype=np.uint8)
    out = I.apply_pixel_op(img, I.PixelOp("x", keep=("joint",)), m, view="frontal",
                           marker_params=_params())
    assert (out[m["joint"]] == 200).all()
    assert (out[~m["joint"]] == 0).all()


def test_mean_fill_uses_the_mean_of_the_retained_pixels():
    m = I.region_masks()
    img = np.zeros((512, 512), dtype=np.uint8)
    img[~m["joint"]] = 100
    out = I.apply_pixel_op(img, I.PixelOp("x", zero=("joint",), mean_fill=True), m,
                           view="frontal", marker_params=_params())
    assert (out[m["joint"]] == 100).all()


def test_anatomic_ops_are_skipped_off_the_frontal_view():
    """Medial and lateral are meaningless on a sagittal or axial projection, so the op
    must be a no-op there rather than blanking an arbitrary strip."""
    m = I.region_masks()
    img = np.full((512, 512), 90, dtype=np.uint8)
    op = I.PixelOp("x", zero=("medial",), frontal_only=True)
    assert np.array_equal(I.apply_pixel_op(img, op, m, view="lateral",
                                           marker_params=_params()), img)
    assert not np.array_equal(I.apply_pixel_op(img, op, m, view="frontal",
                                               marker_params=_params()), img)


def test_border_widening_is_a_superset_of_the_protocol_band():
    m = I.region_masks()
    img = np.full((512, 512), 55, dtype=np.uint8)
    out = I.apply_pixel_op(img, I.PixelOp("x", border_px=62), m, view="frontal",
                           marker_params=_params())
    assert (out[m["border"]] == 0).all()
    assert (out[:62, :] == 0).all() and (out[:, -62:] == 0).all()
    assert out[256, 256] == 55


def test_residual_marker_masking_removes_a_saturated_isolated_blob_not_the_bone():
    m = I.region_masks()
    img = np.full((512, 512), 30, dtype=np.uint8)
    img[200:320, 200:320] = 255                 # the largest saturated object = "bone"
    img[60:75, 430:448] = 255                   # a small saturated marker
    out = I.apply_pixel_op(img, I.PixelOp("x", mask_residual_markers=True), m,
                           view="frontal", marker_params=_params())
    assert (out[60:75, 430:448] == 0).all()
    assert (out[200:320, 200:320] == 255).all()


# --------------------------------------------------------------------------- #
# Attribution mass accounting                                                  #
# --------------------------------------------------------------------------- #
def test_attribution_fractions_sum_to_one_over_the_partition():
    m = I.region_masks()
    rng = np.random.default_rng(0)
    a = rng.random((512, 512))
    f = I.attribution_fractions(a, m)
    assert f["joint"] + f["border"] + f["peripheral"] == pytest.approx(1.0, abs=1e-9)
    assert f["medial"] + f["lateral"] == pytest.approx(f["joint"], abs=1e-9)


def test_attribution_fractions_use_absolute_mass():
    """Integrated gradients are signed; a region of strong negative attribution is still
    attribution and must not cancel a positive region."""
    m = I.region_masks()
    a = np.zeros((512, 512))
    a[m["joint"]] = -1.0
    assert I.attribution_fractions(a, m)["joint"] == pytest.approx(1.0)


def test_uniform_attribution_recovers_the_area_fractions():
    m = I.region_masks()
    f = I.attribution_fractions(np.ones((512, 512)), m)
    area = I.region_area_fractions()
    for k in f:
        assert f[k] == pytest.approx(area[k], abs=1e-9)


# --------------------------------------------------------------------------- #
# The encode_masked_only reindexing trap                                       #
# --------------------------------------------------------------------------- #
def test_real_slot_index_maps_gathered_rows_back_to_the_right_patient():
    """A ragged padding pattern is the only case where positional mapping breaks.

    Patient 0 has one crop, patient 1 has three, patient 2 has two. ``embed_images``
    flattens to B*E and gathers the masked positions, so the encoder sees six feature maps
    whose row order is [ (0,0), (1,0), (1,1), (1,2), (2,0), (2,1) ]. Mapping row r to slot
    r would put patient 1's second crop on patient 0.
    """
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    sel, b, e = I._real_slot_index(mask)
    assert sel.numel() == 6
    assert b.tolist() == [0, 1, 1, 1, 2, 2]
    assert e.tolist() == [0, 0, 1, 2, 0, 1]
    # and it agrees with what the model itself computes
    B, E = mask.shape
    assert torch.equal(sel, mask.reshape(B * E).nonzero(as_tuple=True)[0])


def test_real_slot_index_is_dense_when_nothing_is_padded():
    mask = torch.ones((3, 2), dtype=torch.bool)
    sel, b, e = I._real_slot_index(mask)
    assert b.tolist() == [0, 0, 1, 1, 2, 2]
    assert e.tolist() == [0, 1, 0, 1, 0, 1]


def test_gradcam_returns_one_map_per_real_slot_on_a_ragged_batch():
    """End-to-end on a tiny randomly initialised network: the number of CAMs must equal
    the number of REAL slots, not B*E, and each must carry its own patient's index."""
    from src.train_model import SurvivalFusionNet

    torch.manual_seed(0)
    net = SurvivalFusionNet(n_intervals=10, n_clinical=13, n_views=3, mode="image_only",
                            arch="densenet121", pretrained=False, view_emb_dim=32).eval()
    B, E, S = 3, 4, 64
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    batch = {"images": torch.randint(0, 255, (B, E, S, S), dtype=torch.uint8),
             "view_id": torch.zeros((B, E), dtype=torch.long),
             "mask": mask,
             "view_available": torch.zeros((B, 3)),
             "clinical": torch.zeros((B, 13)),
             "idx": torch.arange(B)}
    sb, se, cam = I.gradcam(net, batch, torch.device("cpu"))
    assert cam.shape[0] == int(mask.sum()) == 6
    assert sb.tolist() == [0, 1, 1, 1, 2, 2]
    assert se.tolist() == [0, 0, 1, 2, 0, 1]
    assert (cam >= 0).all(), "Grad-CAM is rectified by construction"


def test_integrated_gradients_shape_and_zero_baseline_completeness():
    """A zero input must produce exactly zero attribution: IG is (x - baseline) * grad and
    the baseline is the zero image, which is what the pipeline writes into the border."""
    from src.train_model import SurvivalFusionNet

    torch.manual_seed(0)
    net = SurvivalFusionNet(n_intervals=10, n_clinical=13, n_views=3, mode="image_only",
                            arch="densenet121", pretrained=False, view_emb_dim=32).eval()
    B, E, S = 2, 2, 64
    mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    imgs = torch.randint(0, 255, (B, E, S, S), dtype=torch.uint8)
    imgs[1, 1] = 0                                   # a genuinely blank real slot
    batch = {"images": imgs, "view_id": torch.zeros((B, E), dtype=torch.long), "mask": mask,
             "view_available": torch.zeros((B, 3)), "clinical": torch.zeros((B, 13)),
             "idx": torch.arange(B)}
    sb, se, ig = I.integrated_gradients(net, batch, torch.device("cpu"), steps=4)
    assert ig.shape == (3, S, S)
    assert sb.tolist() == [0, 1, 1] and se.tolist() == [0, 0, 1]
    assert np.abs(ig[2]).max() == 0.0


# --------------------------------------------------------------------------- #
# The risk target                                                              #
# --------------------------------------------------------------------------- #
def test_risk_target_matches_the_frozen_numpy_risk_at_horizon():
    """Attribution must explain the quantity the paper reports, so the differentiable
    target has to be the same function ``train_model.risk_at_horizon`` computes."""
    from src.train_model import risk_at_horizon

    rng = np.random.default_rng(1)
    logits = rng.normal(-2.0, 1.0, size=(7, 10))
    h = 1.0 / (1.0 + np.exp(-logits))
    for horizon in (365, 730, 1825):
        want = risk_at_horizon(h, float(horizon))
        got = I.risk_target(torch.tensor(logits, dtype=torch.float64), horizon).numpy()
        assert np.allclose(got, want, atol=1e-12)


def test_risk_target_is_monotone_in_every_interval_hazard():
    logits = torch.full((1, 10), -2.0, dtype=torch.float64)
    lo = float(I.risk_target(logits, 1825))
    for k in range(10):
        bumped = logits.clone()
        bumped[0, k] += 0.5
        assert float(I.risk_target(bumped, 1825)) > lo


def test_five_year_risk_applies_the_frozen_recalibration_not_the_identity():
    from src.train_model import apply_recalibration, risk_at_horizon

    rng = np.random.default_rng(2)
    hz = rng.uniform(0.001, 0.05, size=(5, 10))
    recal = {"1825.0": {"intercept": 0.1643375335875856, "slope": 1.5621890711659725}}
    raw = risk_at_horizon(hz, 1825.0)
    got = I.five_year_risk(hz, recal)
    assert np.allclose(got, apply_recalibration(raw, recal["1825.0"]))
    assert not np.allclose(got, raw)
    assert np.allclose(I.five_year_risk(hz, None), raw)


# --------------------------------------------------------------------------- #
# Attention tables: every column holds what its name says                      #
# --------------------------------------------------------------------------- #
def _fake_attention():
    """Four patients: frontal+lateral, frontal+lateral, all three, two frontals only."""
    n_slots = np.array([[1., 1., 0.], [1., 1., 0.], [1., 1., 1.], [2., 0., 0.]])
    share = np.array([[0.30, 0.70, 0.00], [0.50, 0.50, 0.00],
                      [0.20, 0.50, 0.30], [1.00, 0.00, 0.00]])
    return share, n_slots


def test_three_view_row_sd_columns_are_standard_deviations_not_the_next_mean():
    """The defect this test exists for: ``weight_a_sd`` used to hold the LATERAL MEAN on
    the three-view row (0.4580 / 0.4920), which a figure would draw as an error bar."""
    share, n_slots = _fake_attention()
    rows = I.paired_attention_rows("arm", ["frontal", "lateral", "sunrise"], share, n_slots,
                                   min_patients=1)
    r = [x for x in rows if x["comparison"] == "all_three_views_present"][0]
    assert r["n_patients"] == 1
    assert r["weight_a_mean"] == pytest.approx(0.20)
    assert r["weight_b_mean"] == pytest.approx(0.50)
    assert r["weight_c_mean"] == pytest.approx(0.30)
    # one patient -> no SD is defined; the old code would have put 0.50 in weight_a_sd
    assert np.isnan(r["weight_a_sd"]) and r["weight_a_sd"] != r["weight_b_mean"]


def test_three_view_sd_is_the_sd_of_its_own_column():
    n_slots = np.ones((3, 3))
    share = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
    r = [x for x in I.paired_attention_rows("a", ["f", "l", "s"], share, n_slots,
                                            min_patients=1)
         if x["comparison"] == "all_three_views_present"][0]
    for col, k in (("weight_a_sd", 0), ("weight_b_sd", 1), ("weight_c_sd", 2)):
        assert r[col] == pytest.approx(float(np.std(share[:, k], ddof=1)))
    assert (r["weight_a_mean"] + r["weight_b_mean"] + r["weight_c_mean"]) == pytest.approx(1.0)


def test_pairwise_rows_use_only_patients_with_exactly_one_crop_of_each_view():
    share, n_slots = _fake_attention()
    r = [x for x in I.paired_attention_rows("arm", ["frontal", "lateral", "sunrise"], share,
                                            n_slots, min_patients=1)
         if x["comparison"] == "pairwise_one_crop_each"][0]
    assert r["n_patients"] == 2                      # the all-three and two-frontal are out
    assert r["view_c"] == "" and np.isnan(r["weight_c_mean"])
    assert r["weight_a_mean"] == pytest.approx(0.40)
    assert r["weight_a_sd"] == pytest.approx(float(np.std([0.30, 0.50], ddof=1)))


def test_by_view_multiview_denominator_counts_distinct_views_not_crops():
    """A patient with two frontal crops and nothing else has ONE view, not 'more than one'.
    Conflating those two counts is how '315 patients who have more than one view' happened."""
    share, n_slots = _fake_attention()
    rows = I.by_view_attention_rows("arm", ["frontal", "lateral", "sunrise"], share, n_slots)
    front = rows[0]
    assert front["n_patients_with_view"] == 4
    assert front["n_patients_with_2plus_distinct_views"] == 3      # not 4
    assert "distinct" in front["note"]


# --------------------------------------------------------------------------- #
# Occlusion notes: the degenerate control must announce that it is degenerate  #
# --------------------------------------------------------------------------- #
def test_risk_degeneracy_reports_the_modal_risk_and_the_distinct_count():
    n, v, d = I.risk_degeneracy(np.array([0.1, 0.1, 0.1, 0.2, 0.3]))
    assert (n, v, d) == (3, 0.1, 3)


def test_keep_border_only_note_refuses_to_be_read_as_a_leakage_test():
    note = I.occlusion_note("keep_border_only", n_patients=734, n_identical=0,
                            n_max_tied=676, n_distinct=59, tied_risk=0.0377898019855466,
                            delta_auroc=-0.3393, delta_lo=-0.3956, delta_hi=-0.2799)
    assert "NOT A TEXT-LEAKAGE TEST" in note
    assert "676 of 734" in note and "0.0377898020" in note
    assert "mask_residual_markers" in note


def test_occlude_border_note_says_pipeline_validation():
    note = I.occlusion_note("occlude_border", n_patients=734, n_identical=677, n_max_tied=1,
                            n_distinct=734, tied_risk=0.1, delta_auroc=-0.0002,
                            delta_lo=-0.0009, delta_hi=0.0)
    assert "PIPELINE VALIDATION" in note and "677 of 734" in note


def test_every_anatomic_note_warns_when_the_interval_crosses_zero():
    for cond in I.ANATOMIC_CONDITIONS:
        note = I.occlusion_note(cond, n_patients=734, n_identical=0, n_max_tied=1,
                                n_distinct=734, tied_risk=0.1, delta_auroc=-0.010,
                                delta_lo=-0.039, delta_hi=0.019)
        assert "includes zero" in note and "compartment ranking" in note


def test_view_ablation_note_calls_the_common_count_a_paired_denominator():
    assert "PAIRED-ANALYSIS DENOMINATOR" in I.VIEW_ABLATION_NOTE
    for n in ("315", "316", "321", "322"):
        assert n in I.VIEW_ABLATION_NOTE


# --------------------------------------------------------------------------- #
# The published tables carry those disclosures                                 #
# --------------------------------------------------------------------------- #
def test_published_occlusion_table_discloses_the_degenerate_control():
    df = _table("interp_occlusion.csv")
    assert "note" in df.columns
    r = df[df.condition == "keep_border_only"].iloc[0]
    # a near-constant predictor, not a weak one
    assert int(r.n_max_tied_risk) == 676 and int(r.n_patients) == 734
    assert int(r.n_distinct_risk) < 0.1 * int(r.n_patients)
    assert "NOT A TEXT-LEAKAGE TEST" in r.note
    assert int(df[df.condition == "occlude_border"].iloc[0].n_identical_to_baseline) == 677
    assert df.note.map(lambda s: isinstance(s, str) and len(s) > 0).all()


def test_published_attention_paired_table_has_no_mean_in_an_sd_column():
    df = _table("interp_attention_paired.csv")
    three = df[df.comparison == "all_three_views_present"]
    assert len(three) == 2
    for _, r in three.iterrows():
        assert r.weight_a_sd != pytest.approx(r.weight_b_mean, abs=1e-9)
        assert r.weight_a_sd < r.weight_a_mean          # an SD, not a competing weight
        assert (r.weight_a_mean + r.weight_b_mean + r.weight_c_mean) == pytest.approx(1.0,
                                                                                      abs=1e-6)


def test_published_sanity_table_names_both_denominators():
    df = _table("interp_sanity_checks.csv")
    assert "n_images" not in df.columns
    assert set(df.n_images_cam) == {60}
    assert set(df.n_patients_auroc) == {734}


def test_published_ig_border_mass_is_essentially_zero_not_exactly_zero():
    """The claim in the hand-off was 'exactly zero'. It is not: the band is not exactly
    zero on 11.9% of crops, so a minority carry a real, tiny IG border mass."""
    df = _table("interp_attribution_summary.csv")
    r = df[(df.method == "integrated_gradients") & (df.region == "border")].iloc[0]
    assert 0.0 < float(r["mean"]) < 1e-5
    assert int(r["n_images_nonzero"]) == 18 and int(r["n_images"]) == 230
    assert float(r["max"]) > 0.0


# --------------------------------------------------------------------------- #
# Sanity-check model surgery                                                   #
# --------------------------------------------------------------------------- #
def test_cascading_randomisation_scopes_are_nested():
    """Randomising ``denseblock2`` must also randomise everything above it plus the head,
    which is what makes the Adebayo cascade a cascade rather than five spot checks."""
    top = I.ENCODER_STAGES[I.ENCODER_STAGES.index("denseblock4"):]
    mid = I.ENCODER_STAGES[I.ENCODER_STAGES.index("denseblock2"):]
    assert set(top) < set(mid)
    assert I.ENCODER_STAGES[-1] == "norm5"
    assert I.CAM_LAYER.endswith("norm5")
