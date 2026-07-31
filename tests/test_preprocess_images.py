"""test_preprocess_images.py — proves the contralateral crop logic WITHOUT real data.

Success non-negotiable #1 is that the crop holds the CONTRALATERAL knee and zero
index-knee pixels. The half-select truth table below is the test that protects it: an
inverted sign convention in `contralateral_image_half` would flip all four rows and
silently train the model on the already-replaced knee.

No cohort parquet, no real DICOM and no network is touched here; every fixture is
synthesized (pydicom writes the test DICOMs into tmp_path).

Run:  python3 -m pytest tests/test_preprocess_images.py -q
"""
from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.preprocess_images import (
    KEY_TEMPLATE, MANIFEST_REGRESSION, REASON_EXCESSIVE_MASKING, REASON_LOCALIZATION_FAILED,
    Localization, PreprocessError, PreprocessParams, ShardWriter,
    assert_out_dir_is_outside_repo, border_band_fraction, contralateral_image_half,
    crop_contralateral_knee, crop_stages, encode_image, finalize_image, half_column_bounds,
    localize_joint, mask_borders, otsu_threshold, read_dicom, robust_scale, sample_key,
    select_contralateral_half, square_crop, standardize_orientation, uid_short,
)


def make_params(**over) -> PreprocessParams:
    """Config defaults, overridable per test."""
    base = PreprocessParams.from_config(load_config())
    return PreprocessParams(**{**base.__dict__, **over})


# =============================================================================
# 1. Half-select truth table — the guard on non-negotiable #1
# =============================================================================
def bilateral_fixture() -> np.ndarray:
    """200-wide bilateral film: left half 0.2, an 8-px midline strip 0.5, right half 0.8.

    The midline strip stands in for the pixels that could belong to EITHER knee; a
    correct inset must discard all of it.
    """
    arr = np.zeros((100, 200), dtype=np.float32)
    arr[:, :96] = 0.2
    arr[:, 96:104] = 0.5      # midline / ambiguous strip
    arr[:, 104:] = 0.8
    return arr


@pytest.mark.parametrize("contra_side,horizontal_flip,expected_half,expected_value", [
    ("R", 0, "left", 0.2),    # radiological: patient RIGHT is on the IMAGE LEFT
    ("L", 0, "right", 0.8),
    ("R", 1, "right", 0.8),   # stored pixels mirrored -> the answer swaps
    ("L", 1, "left", 0.2),
])
def test_half_select_truth_table(contra_side, horizontal_flip, expected_half, expected_value):
    arr = bilateral_fixture()
    assert contralateral_image_half(contra_side, horizontal_flip, "radiological",
                                    flip_corrected=False) == expected_half
    half, name = select_contralateral_half(arr, contra_side, horizontal_flip,
                                           convention="radiological", flip_corrected=False,
                                           inset_frac=0.02)
    assert name == expected_half
    assert np.allclose(half, expected_value), "half-select returned the WRONG knee"


@pytest.mark.parametrize("contra_side,horizontal_flip", [("L", 0), ("L", 1), ("R", 0), ("R", 1)])
def test_half_select_inset_excludes_the_opposite_half_and_the_midline(contra_side, horizontal_flip):
    arr = bilateral_fixture()
    half, name = select_contralateral_half(arr, contra_side, horizontal_flip,
                                           convention="radiological", flip_corrected=False,
                                           inset_frac=0.02)
    vals = set(np.unique(half).tolist())
    assert 0.5 not in vals, "midline strip survived the inset"
    other = 0.8 if name == "left" else 0.2
    assert other not in vals, "the opposite (index-knee) half leaked into the crop"
    assert half.shape[1] == 96, f"inset should drop mid+/-4 columns, got width {half.shape[1]}"


def test_half_select_flip_corrected_ignores_the_flag():
    """Once apply_flip_correction has un-mirrored the array the flag must NOT re-apply."""
    for hf in (0, 1):
        assert contralateral_image_half("R", hf, "radiological", flip_corrected=True) == "left"
        assert contralateral_image_half("L", hf, "radiological", flip_corrected=True) == "right"


def test_half_select_anatomical_convention_is_the_mirror_image():
    for hf in (0, 1):
        assert contralateral_image_half("R", hf, "anatomical", flip_corrected=True) == "right"
        assert contralateral_image_half("L", hf, "anatomical", flip_corrected=True) == "left"


def test_half_select_rejects_bad_inputs():
    with pytest.raises(AssertionError):
        contralateral_image_half("B", 0)
    with pytest.raises(AssertionError):
        contralateral_image_half("L", 0, convention="sagittal")


def test_half_column_bounds_match_the_slice_the_pipeline_takes():
    """src.crop_qa draws its overlay from half_column_bounds; it must be the SAME slice."""
    arr = bilateral_fixture()
    for contra_side in ("L", "R"):
        half, name = select_contralateral_half(arr, contra_side, 0, convention="radiological",
                                               flip_corrected=False, inset_frac=0.02)
        c0, c1 = half_column_bounds(arr.shape[1], name, 0.02)
        assert np.array_equal(half, arr[:, c0:c1]), "the QA overlay would outline the wrong columns"
    assert half_column_bounds(200, "left", 0.02) == (0, 96)
    assert half_column_bounds(200, "right", 0.02) == (104, 200)
    with pytest.raises(AssertionError):
        half_column_bounds(200, "middle", 0.02)


# =============================================================================
# 2. DICOM decode round trip
# =============================================================================
def write_test_dicom(path, pixels: np.ndarray, photometric: str,
                     slope: float = 2.0, intercept: float = -100.0) -> None:
    """Minimal but valid uncompressed CR DICOM (no VOI window, so the VOI step no-ops)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    cr_image_storage = "1.2.840.10008.5.1.4.1.1.1"   # Computed Radiography Image Storage
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = cr_image_storage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.SOPClassUID = cr_image_storage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CR"
    ds.PatientID = "SYNTHETIC"
    ds.PatientName = "SYNTHETIC^TEST"
    ds.Rows, ds.Columns = int(pixels.shape[0]), int(pixels.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)      # pydicom >= 3
    except TypeError:                                         # pragma: no cover
        ds.save_as(str(path), write_like_original=False)
    assert pydicom.dcmread(str(path)) is not None


def ramp_pixels(h: int = 64, w: int = 80) -> np.ndarray:
    """Left-to-right intensity ramp; the ordering makes inversion unmistakable."""
    return np.tile(np.linspace(100, 4000, w, dtype=np.uint16), (h, 1))


def test_read_dicom_monochrome1_is_inverted_and_rescaled(tmp_path):
    params = make_params()
    px = ramp_pixels()
    path = tmp_path / "mono1.dcm"
    write_test_dicom(path, px, "MONOCHROME1", slope=2.0, intercept=-100.0)

    arr, meta = read_dicom(path, params)
    assert meta["photometric"] == "MONOCHROME1"
    assert meta["dicom_inverted"] is True
    assert arr.dtype == np.float32 and arr.shape == px.shape
    assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0
    # a rising raw ramp must come back FALLING once MONOCHROME1 is inverted
    row = arr[0]
    assert row[0] > row[-1]
    assert np.all(np.diff(row) <= 1e-6)

    rescaled = px.astype(np.float32) * 2.0 - 100.0
    expected = robust_scale(float(rescaled.max()) - rescaled, params.clip_percentiles)
    assert np.allclose(arr, expected, atol=1e-5)


def test_read_dicom_monochrome2_is_not_inverted(tmp_path):
    params = make_params()
    px = ramp_pixels()
    path = tmp_path / "mono2.dcm"
    write_test_dicom(path, px, "MONOCHROME2", slope=2.0, intercept=-100.0)

    arr, meta = read_dicom(path, params)
    assert meta["dicom_inverted"] is False
    row = arr[0]
    assert row[0] < row[-1], "MONOCHROME2 must NOT be inverted"
    assert np.all(np.diff(row) >= -1e-6)

    rescaled = px.astype(np.float32) * 2.0 - 100.0
    assert np.allclose(arr, robust_scale(rescaled, params.clip_percentiles), atol=1e-5)


def test_read_dicom_missing_file_raises_preprocess_error(tmp_path):
    with pytest.raises(PreprocessError) as exc:
        read_dicom(tmp_path / "nope.dcm", make_params())
    assert exc.value.reason == "decode_failed"


def write_voi_lut_dicom(path, pixels, lut_data, first_mapped=0):
    """A DICOM carrying a VOI LUT *Sequence* — an INTEGER-INDEXED lookup table.

    The original synthetic fixtures used windowing only, which is exactly why the
    float-input defect below went unnoticed until real MRKR data arrived.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    cr = "1.2.840.10008.5.1.4.1.1.1"
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = cr
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.SOPClassUID = cr
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CR"
    ds.PatientID = "SYNTHETIC"
    ds.Rows, ds.Columns = int(pixels.shape[0]), int(pixels.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    # identity modality transform — the real MRKR case (verified on 300 files)
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    lut = Dataset()
    lut.LUTDescriptor = [len(lut_data), int(first_mapped), 16]
    lut.LUTExplanation = "test"
    # VR is OW — LUTData must be raw bytes, not a list of ints
    lut.LUTData = np.asarray(lut_data, dtype=np.uint16).tobytes()
    ds.VOILUTSequence = [lut]
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:                                             # pragma: no cover
        ds.save_as(str(path), write_like_original=False)


def test_voi_lut_sequence_is_applied_on_integer_pixels_not_floats(tmp_path):
    """Regression: a VOI LUT Sequence is indexed by pixel VALUE.

    Casting to float before the lookup makes the index undefined (pydicom warns
    "Applying a VOI LUT on a float input array may give incorrect results"). Measured on
    real MRKR data, 8% of images carry a VOILUTSequence, so this silently corrupted the
    contrast curve on ~1 image in 12. read_dicom must keep the array integer through the
    VOI step whenever the modality transform is the identity.
    """
    import warnings

    params = make_params()
    # LUT reverses the input: value v -> 4095 - v, over 0..15
    lut_data = [4095 - i for i in range(16)]
    px = np.tile(np.arange(16, dtype=np.uint16), (8, 1))
    path = tmp_path / "voilut.dcm"
    write_voi_lut_dicom(path, px, lut_data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arr, meta = read_dicom(path, params)

    floatwarn = [w for w in caught if "VOI LUT on a float" in str(w.message)]
    assert not floatwarn, "VOI LUT was applied to a float array — the lookup is undefined"
    assert meta["voi_lut_on_float"] is False

    # the LUT reverses intensity, so a rising raw ramp must come back falling
    row = arr[0]
    assert row[0] > row[-1]
    assert np.all(np.diff(row) <= 1e-6)


def test_marker_masking_removes_multi_glyph_text_but_spares_bone():
    """Protocol section 13 / non-negotiable #1: burned-in markers must not survive.

    Two properties are load-bearing, and both were established on real MRKR films:
      1. multi-glyph text is removed AS A UNIT — masking one letter at a time leaves a
         half-erased marker, which is still a learnable artifact;
      2. saturated BONE is never masked, because bone sits on mid-grey soft tissue while a
         lead marker sits on unexposed background.
    """
    from src.preprocess_images import mask_burned_in_markers

    params = make_params()
    img = np.zeros((512, 512), dtype=np.uint8)
    # saturated "bone": a large bright mass surrounded by mid-grey tissue
    img[150:400, 150:360] = 90                      # soft-tissue halo
    img[200:350, 200:300] = 255                     # the bone itself
    # a three-glyph marker on black background, glyphs separated by a few px
    for x0 in (430, 452, 474):
        img[40:70, x0:x0 + 14] = 255

    out, n_masked, frac = mask_burned_in_markers(img, params)

    # the whole marker cluster is gone, not just one glyph
    assert out[40:70, 430:490].max() < params.marker_sat_level, "marker text survived"
    assert n_masked >= 1 and frac > 0.0
    # bone is untouched
    assert np.array_equal(out[200:350, 200:300], img[200:350, 200:300]), "bone was masked"

    # and the switch genuinely disables it
    off = make_params(mask_markers=False)
    same, n_off, frac_off = mask_burned_in_markers(img, off)
    assert n_off == 0 and frac_off == 0.0 and np.array_equal(same, img)


def test_marker_masking_is_equivariant_under_the_left_right_mirror():
    """crop_stages must satisfy image == premirror[:, ::-1] exactly for a right knee.

    The QA contact sheet shows the pre-mirror crop next to the final one and the reviewer
    is told they are the same image; a component-size tie resolving differently on the
    mirrored copy would quietly break that.
    """
    params = make_params()
    arr = np.concatenate([crude_knee(600, 300), crude_knee(600, 300)], axis=1)
    arr[20:50, 40:120] = 1.0                        # a marker in the discarded half
    arr[20:50, 460:540] = 1.0                       # and one in the kept half
    st = crop_stages(arr, view="frontal", laterality="B", contra_side="R",
                     horizontal_flip=0, params=params)
    assert st["spec"].mirrored is True
    assert np.array_equal(st["image"], st["premirror"][:, ::-1])


def test_modality_identity_detection():
    """_modality_is_identity gates whether the array may stay integer."""
    from src.preprocess_images import _modality_is_identity

    class FakeDS(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    assert _modality_is_identity(FakeDS()) is True                     # no tags at all
    assert _modality_is_identity(FakeDS(RescaleSlope=1, RescaleIntercept=0)) is True
    assert _modality_is_identity(FakeDS(RescaleSlope=2, RescaleIntercept=0)) is False
    assert _modality_is_identity(FakeDS(RescaleSlope=1, RescaleIntercept=-100)) is False
    assert _modality_is_identity(FakeDS(ModalityLUTSequence=[object()])) is False


# =============================================================================
# 3. Localizer
# =============================================================================
def crude_knee(h: int = 400, w: int = 300) -> np.ndarray:
    """Two bright bone masses joined by a narrow bridge across a dark joint band."""
    arr = np.full((h, w), 0.05, dtype=np.float32)
    arr[40:170, 80:220] = 0.9        # distal femur
    arr[230:360, 80:220] = 0.9       # proximal tibia
    arr[170:230, 130:170] = 0.9      # narrow bridge -> one connected component
    return arr


def test_otsu_separates_two_modes():
    img = np.concatenate([np.full(5000, 0.1), np.full(5000, 0.9)]).reshape(100, 100)
    t = otsu_threshold(img)
    assert 0.1 < t < 0.9


def test_localizer_finds_the_joint_band_and_respects_the_crop_clamps():
    params = make_params()
    img = crude_knee()
    loc = localize_joint(img, params)
    assert loc.method == "intensity_profile"
    assert 170 <= loc.row <= 230, f"joint centre {loc.row} is outside the dark band"
    assert 80 <= loc.col <= 220
    assert 0.0 < loc.confidence <= 1.0
    short = min(img.shape)
    assert params.min_crop_frac * short <= loc.side <= params.max_crop_frac * short


@pytest.mark.parametrize("img", [
    np.zeros((256, 256), dtype=np.float32),                  # blank
    np.ones((256, 256), dtype=np.float32),                   # saturated
    np.full((10, 10), 0.5, dtype=np.float32),                # too small
    np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256),  # no joint structure
])
def test_localizer_falls_back_and_labels_the_fallback(img):
    params = make_params()
    loc = localize_joint(img, params)
    assert loc.method == "fallback_center", "degenerate image must take the labelled fallback"
    assert loc.confidence == 0.0
    assert loc.row == img.shape[0] / 2 and loc.col == img.shape[1] / 2
    assert isinstance(loc, Localization)


def test_square_crop_pads_rather_than_wraps():
    arr = np.zeros((50, 50), dtype=np.float32)
    arr[:, :] = 0.4
    arr[0, 0] = 1.0
    out, padded_frac = square_crop(arr, (0.0, 0.0), 40, pad_value=0.0)
    assert out.shape == (40, 40)
    assert out[0, 0] == 0.0 and out[19, 19] == 0.0          # padded region
    assert out[20, 20] == 1.0                                # arr[0,0] lands at the centre
    assert not np.any(out[:20, :20] == 0.4), "crop wrapped instead of padding"
    # 40x40 crop anchored at (-20,-20): only the 20x20 lower-right quadrant is real pixels.
    assert padded_frac == pytest.approx(1.0 - (20 * 20) / (40 * 40))


def test_square_crop_reports_zero_padding_when_fully_inside():
    arr = np.full((100, 100), 0.4, dtype=np.float32)
    out, padded_frac = square_crop(arr, (50.0, 50.0), 40)
    assert out.shape == (40, 40) and padded_frac == 0.0


# =============================================================================
# 4. Orientation standardization + marker masking
# =============================================================================
def test_right_knee_is_mirrored_to_read_as_left():
    crop = np.zeros((20, 20), dtype=np.float32)
    crop[:, :5] = 1.0
    out, orientation, mirrored = standardize_orientation(crop, "R", True)
    assert mirrored is True and orientation == "left"
    assert np.array_equal(out, crop[:, ::-1])

    out_l, orient_l, mirrored_l = standardize_orientation(crop, "L", True)
    assert mirrored_l is False and orient_l == "left"
    assert np.array_equal(out_l, crop)

    out_off, orient_off, mirrored_off = standardize_orientation(crop, "R", False)
    assert mirrored_off is False and orient_off == "right"
    assert np.array_equal(out_off, crop)


def test_masking_blanks_a_constant_border_on_every_edge():
    params = make_params()
    rng = np.random.default_rng(0)
    img = (rng.integers(40, 255, size=(512, 512))).astype(np.uint8)
    out = mask_borders(img, params.mask_border_frac, fill=0)
    t = int(round(params.mask_border_frac * 512))
    assert t > 0
    for band in (out[:t, :], out[-t:, :], out[:, :t], out[:, -t:]):
        assert band.size and np.all(band == 0), "border band is not constant"
    assert np.array_equal(out[t:-t, t:-t], img[t:-t, t:-t]), "masking altered the interior"


def test_finalize_image_resizes_then_masks():
    params = make_params()
    crop = np.full((900, 900), 0.9, dtype=np.float32)
    out, masked_pct = finalize_image(crop, params.out_size, params.mask_border_frac)
    assert out.shape == (params.out_size, params.out_size) and out.dtype == np.uint8
    t = int(round(params.mask_border_frac * params.out_size))
    assert np.all(out[:t, :] == 0) and np.all(out[:, -t:] == 0)
    assert out[params.out_size // 2, params.out_size // 2] > 200
    # With no padding, masked_pct IS the border band and must equal the blanked area.
    assert masked_pct == pytest.approx(border_band_fraction(params.out_size,
                                                            params.mask_border_frac))
    assert masked_pct == pytest.approx(float((out == 0).mean()))


# =============================================================================
# 4b. Protocol section 13 — masked-pixel accounting and exclusion
# =============================================================================
def test_border_band_fraction_matches_the_pixels_mask_borders_blanks():
    for out_size, frac in ((512, 0.06), (256, 0.10), (128, 0.0)):
        img = np.full((out_size, out_size), 200, dtype=np.uint8)
        blanked = mask_borders(img, frac, fill=0)
        assert border_band_fraction(out_size, frac) == pytest.approx(float((blanked == 0).mean()))
    # the documented constant for the configured geometry
    assert border_band_fraction(512, 0.06) == pytest.approx(0.22752, abs=1e-5)


def test_masked_pct_is_border_band_plus_padding():
    params = make_params()
    arr = np.full((100, 100), 0.5, dtype=np.float32)
    crop, padded_frac = square_crop(arr, (0.0, 0.0), 40, pad_value=0.0)
    assert padded_frac == pytest.approx(0.75)
    _, masked_pct = finalize_image(crop, params.out_size, params.mask_border_frac, padded_frac)
    assert masked_pct == pytest.approx(
        min(1.0, border_band_fraction(params.out_size, params.mask_border_frac) + 0.75))


def test_excessive_masking_is_excluded_not_written():
    """A crop that is mostly invented padding must raise, not reach a shard."""
    params = make_params()
    img = crude_knee()
    # cap just below the fixed border band -> every crop is "excessively masked"
    strict = make_params(max_masked_pct=0.10)
    with pytest.raises(PreprocessError) as exc:
        crop_contralateral_knee(img, view="lateral", laterality="L", contra_side="L",
                                horizontal_flip=0, params=strict)
    assert exc.value.reason == REASON_EXCESSIVE_MASKING
    assert "masked_pct" in exc.value.detail
    # the same image passes under the configured cap
    out, spec = crop_contralateral_knee(img, view="lateral", laterality="L", contra_side="L",
                                        horizontal_flip=0, params=params)
    assert out.shape == (params.out_size, params.out_size)
    assert 0.0 < spec.masked_pct <= params.max_masked_pct


def test_failed_localization_is_excluded_when_configured():
    """The exclusion MECHANISM still works when a deployment turns it back on.

    The shipped config no longer uses it (`localizer_mode: center_default`,
    `exclude_failed_localization: false`) because on real films a centred box beats the
    localizer — so these params are set explicitly rather than read from config.
    """
    params = make_params(localizer_mode="localizer_primary", exclude_failed_localization=True)
    blank = np.full((400, 300), 0.5, dtype=np.float32)      # no joint structure -> fallback
    assert localize_joint(blank, params).method == "fallback_center"

    with pytest.raises(PreprocessError) as exc:
        crop_contralateral_knee(blank, view="frontal", laterality="L", contra_side="L",
                                horizontal_flip=0, params=params)
    assert exc.value.reason == REASON_LOCALIZATION_FAILED

    # with the exclusion switched off the same image is written, flagged fallback_center
    permissive = make_params(localizer_mode="localizer_primary", exclude_failed_localization=False)
    out, spec = crop_contralateral_knee(blank, view="frontal", laterality="L", contra_side="L",
                                        horizontal_flip=0, params=permissive)
    assert spec.crop_method == "fallback_center" and spec.crop_confidence == 0.0
    assert out.shape == (permissive.out_size, permissive.out_size)


def test_center_default_mode_writes_a_centred_crop_instead_of_excluding():
    """Under the shipped config a weak localization yields a CENTRED crop, not an exclusion.

    Measured on 800 real TRAIN films the intensity-profile localizer fell back on 26.0%
    of images and its "successes" were frequently worse than the centred box, so the
    centred box is the primary estimate and nothing is dropped for using it.
    """
    params = make_params()                                   # shipped config
    assert params.localizer_mode == "center_default"
    assert params.exclude_failed_localization is False

    blank = np.full((400, 300), 0.5, dtype=np.float32)
    assert localize_joint(blank, params).method == "fallback_center"

    out, spec = crop_contralateral_knee(blank, view="frontal", laterality="L", contra_side="L",
                                        horizontal_flip=0, params=params)
    assert spec.crop_method == "center_default"
    assert spec.crop_confidence == 1.0
    assert out.shape == (params.out_size, params.out_size)


def test_localizer_may_only_override_the_centre_when_highly_confident():
    """The refinement gate: a low-confidence localization must NOT move the box."""
    from src.preprocess_images import center_localization, choose_localization

    img = crude_knee(600, 400)
    strict = make_params(localizer_refine_min_confidence=1.01)   # nothing can clear this
    used, opinion = choose_localization(img, strict)
    centre = center_localization(img.shape, strict)
    assert used.method == "center_default"
    assert (used.row, used.col, used.side) == (centre.row, centre.col, centre.side)

    loose = make_params(localizer_refine_min_confidence=0.0)     # anything non-fallback wins
    used2, opinion2 = choose_localization(img, loose)
    if opinion2.method != "fallback_center":
        assert used2.method == opinion2.method
        assert used2.row == opinion2.row and used2.col == opinion2.col
    # the localizer's opinion is always returned for auditing, whether or not it was used
    assert opinion.method in {"intensity_profile", "fallback_center"}


def test_crop_stages_returns_the_evidence_the_qa_gate_renders():
    """src.crop_qa panel A/B come from here; the stages must agree with the written crop."""
    params = make_params()
    contra = crude_knee(600, 300)
    index_knee = crude_knee(600, 300)
    arr = np.concatenate([contra, index_knee], axis=1)      # contra knee on the IMAGE LEFT

    st = crop_stages(arr, view="frontal", laterality="B", contra_side="R",
                     horizontal_flip=0, params=params)
    assert st["reject_reason"] is None
    assert st["half_selected"] == "left"
    c0, c1 = st["half_bounds"]
    assert (c0, c1) == half_column_bounds(arr.shape[1], "left", params.half_inset_frac)
    assert st["film"].shape == arr.shape                     # flip-corrected, NOT mirrored
    assert np.array_equal(st["film"], arr)                   # horizontal_flip == 0 -> untouched
    # panel B is the PRE-mirror crop; the written image is its mirror for a right knee
    assert st["spec"].mirrored is True
    assert np.array_equal(st["image"], st["premirror"][:, ::-1])

    out, spec = crop_contralateral_knee(arr, view="frontal", laterality="B", contra_side="R",
                                        horizontal_flip=0, params=params)
    assert np.array_equal(out, st["image"]) and spec == st["spec"]


def test_crop_stages_flip_correction_is_visible_in_the_film_panel():
    """horizontal_flip == 1 must un-mirror the film BEFORE the overlay is drawn."""
    params = make_params()
    arr = np.concatenate([crude_knee(600, 300), crude_knee(600, 300)], axis=1)
    st = crop_stages(arr, view="frontal", laterality="B", contra_side="R",
                     horizontal_flip=1, params=params)
    assert np.array_equal(st["film"], arr[:, ::-1]), "panel A must show the flip-CORRECTED film"
    assert st["half_selected"] == "left", "after correction the radiological rule applies as usual"


# =============================================================================
# End-to-end crop contract
# =============================================================================
def test_crop_contralateral_knee_reports_its_decisions():
    params = make_params()
    # BOTH halves carry a knee — every pre-index film in this cohort shows two NATIVE
    # knees, which is exactly why the finished crop cannot reveal which half was taken.
    arr = np.concatenate([crude_knee(600, 300), crude_knee(600, 300)], axis=1)

    out, spec = crop_contralateral_knee(arr, view="frontal", laterality="B", contra_side="R",
                                        horizontal_flip=0, params=params)
    assert out.shape == (params.out_size, params.out_size) and out.dtype == np.uint8
    assert spec.half_selected == "left"                  # radiological: patient R -> image left
    assert spec.orientation == "left" and spec.mirrored is True
    assert spec.crop_method == "intensity_profile"       # a fallback would now be EXCLUDED
    assert 0.0 <= spec.crop_confidence <= 1.0
    assert 0.0 < spec.masked_pct <= params.max_masked_pct

    _, spec_l = crop_contralateral_knee(arr, view="frontal", laterality="B", contra_side="L",
                                        horizontal_flip=0, params=params)
    assert spec_l.half_selected == "right" and spec_l.mirrored is False


def test_a_blank_selected_half_fails_localization_instead_of_being_written():
    """If the half-select lands on empty film, the exclusion mechanism catches it.

    Explicit `localizer_primary` + exclusion, since the shipped config deliberately
    prefers a centred crop over dropping the image (see
    test_center_default_mode_writes_a_centred_crop_instead_of_excluding).
    """
    params = make_params(localizer_mode="localizer_primary", exclude_failed_localization=True)
    arr = np.concatenate([crude_knee(600, 300),
                          np.full((600, 300), 0.02, dtype=np.float32)], axis=1)
    with pytest.raises(PreprocessError) as exc:
        crop_contralateral_knee(arr, view="frontal", laterality="B", contra_side="L",
                                horizontal_flip=0, params=params)
    assert exc.value.reason == REASON_LOCALIZATION_FAILED


def test_single_knee_views_skip_half_select_and_enforce_laterality():
    params = make_params()
    img = crude_knee()
    _, spec = crop_contralateral_knee(img, view="lateral", laterality="L", contra_side="L",
                                      horizontal_flip=0, params=params)
    assert spec.half_selected == "none"

    with pytest.raises(PreprocessError) as exc:
        crop_contralateral_knee(img, view="lateral", laterality="R", contra_side="L",
                                horizontal_flip=0, params=params)
    assert exc.value.reason == "laterality_mismatch"

    with pytest.raises(PreprocessError) as exc2:
        crop_contralateral_knee(img, view="sunrise", laterality="unknown", contra_side="L",
                                horizontal_flip=0, params=params)
    assert exc2.value.reason == "laterality_unresolved"


# =============================================================================
# 5. Shard integrity
# =============================================================================
def test_sample_key_is_webdataset_safe():
    key = sample_key("12345678", "frontal", "1.2.826.0.1.3680043.8.498.9999")
    assert "." not in key
    assert key.startswith("12345678_frontal_")
    assert len(uid_short("1.2.3")) == 12 and all(c in "0123456789abcdef" for c in uid_short("1.2.3"))
    assert KEY_TEMPLATE.format(empi_anon="a", view="b", uid_short="c") == "a_b_c"


def synthetic_sample(i: int) -> tuple[str, list[tuple[str, bytes]]]:
    rng = np.random.default_rng(i)
    u8 = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    key = sample_key(f"1000{i:03d}", "frontal", f"1.2.3.{i}")
    payload = json.dumps({"key": key, "i": i}, sort_keys=True).encode("utf-8")
    return key, [("png", encode_image(u8)), ("json", payload)]


def test_shard_members_are_contiguous_and_paired(tmp_path):
    writer = ShardWriter(tmp_path, "train", "{split}-{index:05d}.tar", max_shard_mb=0.02)
    written = {}
    for i in range(24):
        key, members = synthetic_sample(i)
        written[key] = writer.write(key, members)
    writer.close()
    assert writer.n_samples == 24
    assert len(writer.shards) >= 2, "max_shard_mb did not trigger a rotation"

    seen_keys = set()
    for shard in writer.shards:
        with tarfile.open(tmp_path / shard, "r") as tf:
            names = tf.getnames()
        assert names, f"{shard} is empty"
        keys = [n.rsplit(".", 1)[0] for n in names]
        exts = [n.rsplit(".", 1)[1] for n in names]
        # contiguity: each key occupies one unbroken run of members
        runs = [k for j, k in enumerate(keys) if j == 0 or keys[j - 1] != k]
        assert len(runs) == len(set(runs)), f"{shard} interleaves samples"
        for k in set(keys):
            assert sorted(e for j, e in enumerate(exts) if keys[j] == k) == ["json", "png"]
        assert not (set(keys) & seen_keys), "a key appears in two shards"
        seen_keys |= set(keys)
    assert seen_keys == set(written)
    assert all(s.startswith("train-") and s.endswith(".tar") for s in writer.shards)


def test_shards_never_mix_two_splits(tmp_path):
    keys = {}
    for split in ("train", "val"):
        writer = ShardWriter(tmp_path, split, "{split}-{index:05d}.tar", max_shard_mb=100)
        ks = []
        for i in range(4):
            key, members = synthetic_sample(i + (0 if split == "train" else 100))
            writer.write(key, members)
            ks.append(key)
        writer.close()
        keys[split] = set(ks)
        assert all(s.startswith(f"{split}-") for s in writer.shards)

    tars = sorted(p.name for p in tmp_path.glob("*.tar"))
    assert tars == ["train-00000.tar", "val-00000.tar"]
    for split, other in (("train", "val"), ("val", "train")):
        with tarfile.open(tmp_path / f"{split}-00000.tar", "r") as tf:
            names = {n.rsplit(".", 1)[0] for n in tf.getnames()}
        assert names == keys[split]
        assert not (names & keys[other]), "a tar mixes two splits"


# =============================================================================
# Config contract
# =============================================================================
def test_params_come_from_config_not_from_hard_coded_values():
    cfg = load_config()
    p = PreprocessParams.from_config(cfg)
    c = cfg["preprocess"]
    assert p.out_size == c["out_size"]
    assert p.half_inset_frac == c["half_inset_frac"]
    assert p.mask_border_frac == c["mask_border_frac"]
    assert p.min_crop_frac == c["min_crop_frac"] and p.max_crop_frac == c["max_crop_frac"]
    assert p.bilateral_display_convention == c["bilateral_display_convention"]
    assert p.standardize_to_left == c["standardize_to_left"]
    assert p.localizer == c["localizer"]
    assert tuple(p.clip_percentiles) == tuple(c["clip_percentiles"])
    # protocol section 13 exclusion parameters
    assert p.min_crop_confidence == c["min_crop_confidence"]
    assert p.max_masked_pct == c["max_masked_pct"]
    assert p.exclude_failed_localization == c["exclude_failed_localization"]


def test_masked_pct_is_a_declared_sidecar_column():
    """The sidecar contract is frozen in config and mirrored by the Colab notebook (T5)."""
    cols = list(load_config()["preprocess"]["sidecar_columns"])
    assert "masked_pct" in cols, "protocol section 13 requires masked_pct in the sidecar"
    assert "crop_confidence" in cols and "crop_method" in cols


def test_max_masked_pct_leaves_headroom_above_the_fixed_border_band():
    """The cap must exclude PADDING, not reject every crop the moment masking is on."""
    cfg = load_config()
    c = cfg["preprocess"]
    band = border_band_fraction(int(c["out_size"]), float(c["mask_border_frac"]))
    assert band < float(c["max_masked_pct"]), \
        (f"max_masked_pct={c['max_masked_pct']} is at or below the constant border band "
         f"{band:.4f}; every crop would be excluded")
    assert float(c["max_masked_pct"]) - band > 0.05, "less than 5 points of padding headroom"


# =============================================================================
# Data hygiene + manifest regression
# =============================================================================
def test_out_dir_inside_the_repo_is_refused(tmp_path):
    """The sidecar carries empi_anon and full SOP UIDs — it must never land in the repo."""
    for bad in (PROJECT_ROOT / "outputs" / "shards", PROJECT_ROOT / "derived-data" / "shards",
                PROJECT_ROOT):
        with pytest.raises(ValueError) as exc:
            assert_out_dir_is_outside_repo(bad)
        assert "INSIDE the repository" in str(exc.value)
    assert_out_dir_is_outside_repo(tmp_path)            # outside the repo: allowed


@pytest.mark.skipif(not (PROJECT_ROOT / "derived-data" / "cohort" / "final_cohort.parquet").exists(),
                    reason="cohort parquet not present (git-ignored derived data)")
def test_build_manifest_regression_anchors_hold():
    from src.preprocess_images import build_manifest
    man = build_manifest(load_config())                 # asserts internally; re-check here
    R = MANIFEST_REGRESSION
    assert len(man) == R["n_rows_kept"]
    assert man["empi_anon"].nunique() == R["n_patients"]
    assert man["view"].value_counts().to_dict() == {
        "frontal": R["n_frontal"], "lateral": R["n_lateral"], "sunrise": R["n_sunrise"]}
    assert man["split"].value_counts().to_dict() == R["by_split"]
    frontal = man[man["view"] == "frontal"]
    assert int((frontal["laterality"] == "B").sum()) == R["n_frontal_bilateral"]
    assert int((frontal["laterality"] != "B").sum()) == R["n_frontal_unilateral"]
    assert int((man["horizontal_flip"] == 1).sum()) == R["n_horizontal_flip"]
    skip_half_select = man[(man["view"] != "frontal") | (man["laterality"] != "B")]
    assert (skip_half_select["laterality"] == skip_half_select["contra_side"]).all()
