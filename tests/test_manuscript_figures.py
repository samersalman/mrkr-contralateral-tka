"""Unit tests for src/manuscript_figures.py - the split-aware figure module.

The module had no tests at all until it had to render two splits instead of one. Four
contracts are pinned here rather than merely exercised:

* :data:`src.manuscript_figures.SPLIT_ANCHORS` is checked against the REAL artefacts, split
  by split. ``eval_models.load_roster`` skips its own anchor check on the sealed split, so
  this module is now the only place 741 / 106 / 740 / 1216 is verified at all; a test that
  compared the table against a copy of itself would verify nothing.
* the figure registry's invariants are run on deliberately broken registries, not only on
  the good one, so "numbers are 1..N and filenames are unique" is a property and not a
  coincidence of today's three figures.
* the caption CLAUSES compose to the frozen validation text byte for byte. Figure 1 and
  Figure 3 must read exactly as the v1 document read; only the claims that genuinely flip
  between splits are allowed two versions.
* the 740-vs-741 rule. On the sealed split one patient lost every crop to the protocol
  section 13 border mask, so "every patient has a finite risk" is false and the assertion
  that replaced it has to fire on a real drift and stay silent on a legitimate subset arm.

Real artefacts are used wherever one exists, so these tests fail when the pipeline's own
output changes rather than when a mock goes stale. Tests needing an artefact that lives
outside the repository (the image shards, the per-arm hazard npz) skip when it is absent.

Run::

    ~/.venvs/mrkr-torch/bin/python -m pytest tests/test_manuscript_figures.py -q
"""
from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.eval_models as em                                            # noqa: E402
import src.manuscript_figures as mf                                     # noqa: E402
from src.config import Config, load_config                              # noqa: E402

VAL, TEST = mf.VAL_SPLIT, mf.SEALED_SPLIT

QUIET = logging.getLogger("test_manuscript_figures")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False


# --------------------------------------------------------------------------- #
# The frozen validation prose. Transcribed from the v1 document, NOT imported   #
# from the module, so a change to the module cannot move the target with it.    #
# --------------------------------------------------------------------------- #
# Keyed by the SUPPLEMENTARY key each figure now carries. The cohort flow and the tertile
# curves moved out of the main registry at v6, when the four main figures became the imaging
# set the editor asked for; the renderers, the clause constants and therefore the text are
# unchanged, which is exactly what these frozen strings are here to prove.
FROZEN_VAL_TITLES = {
    "figureS2": ("Cohort assembly from the source registry to the development patients with "
                 "a usable contralateral radiograph"),
    "figureS4": ("Cumulative incidence of contralateral knee arthroplasty by predicted 5-year "
                 "risk tertile"),
}
# Re-frozen on 2026-07-30. A published caption may not carry a repository path, and
# "outputs/cohort_flow.csv" was one; nor may it speak the pipeline's private vocabulary at
# a reader who has no access to it, and "image shard label index" and "the recalibrated
# crop pipeline" were two. Only the WORDS changed: no count, no denominator and no claim
# about what the sealed split contributed moved, and no figure was re-rendered.
FROZEN_VAL_FIG1_CAPTION = (
    "Patient flow for the primary landmark cohort. Boxes in the left column give the number "
    "of patients retained after each eligibility criterion, and the branches to the right "
    "give the number removed at that step. Counts down to the 3,709-patient landmark cohort "
    "are taken from the cohort assembly ledger; the final two steps are recomputed at figure "
    "time from the locked patient-level split assignment and the record of which radiographs "
    "were successfully prepared for the model. The 741 patients assigned to the test split, "
    "carrying 106 events, were set aside unread and contributed to no model fitting, "
    "hyperparameter choice or evaluation reported in this manuscript, so every number "
    "downstream of that step describes development patients only. Two development patients "
    "had no usable image of the contralateral knee; both were event free, so all 427 "
    "development events were retained."
)
FROZEN_VAL_FIG3_CAPTION = (
    "Cumulative incidence of contralateral knee arthroplasty by predicted risk tertile. "
    "Validation patients were split into tertiles of predicted 5-year risk from the "
    "multimodal fusion model, and each curve is the Kaplan-Meier cumulative incidence, one "
    "minus the survival function, from the landmark through day 1826. The number of patients "
    "remaining at risk in each tertile is given beneath the horizontal axis. Death is not "
    "ascertainable in this data source, so mortality acts as an unmeasured competing event; "
    "these are therefore cause-agnostic cumulative incidences rather than cause-specific "
    "ones, and where competing mortality is appreciable they will overstate what a "
    "competing-risk estimator would give. The three curves rest on 371 patients and 54 "
    "events in total, so the separation between tertiles should be read as exploratory "
    "rather than as a precise estimate of absolute risk."
)

# Figure 4's frozen counterpart, transcribed from the SHIPPED v2 document
# (outputs/manuscript/v2/MRKR_contralateral_TKA_v2.docx), not from the module. Figure 1
# and figure 3 are frozen against v1 and figure 4 had nothing, so the assembly test below
# was re-typing the function's own body and comparing it with the function: a clause
# edited in both places at once passed. v1 declined figure 4 on validation, so the sealed
# document is the only rendered ground truth there is, and it is the right one: it is what
# a reader holds.
#
# ONE WORD OF THIS IS NO LONGER THE SHIPPED TEXT, and it is the only licensed divergence.
# v2 read "Table 2 gives them per arm"; v6 split v5's Table 2 into Table 2 (contrasts) and
# Table 3 (per-arm discrimination and calibration), so the per-arm denominators this
# sentence points at are Table 3's rows, and the frozen string was re-pinned to "Table 3"
# on 2026-08-11 together with src.manuscript_figures.FIG4_DENOMINATORS. A transcription is
# a guard against a clause drifting unnoticed, not a licence to keep printing a dangling
# cross-reference, so the pin MOVED rather than being deleted or exempted: every other word
# is still the shipped v2 text, and a second divergence would have to be argued for here in
# the same way. The layout it is now pinned against is
# outputs/manuscript/"v6- resubmission"/sections/tables.md.
FROZEN_TEST_FIG4_TITLE = (
    "Decision-curve analysis of net benefit against threshold probability in the test split"
)
FROZEN_TEST_FIG4_CAPTION = (
    "Decision-curve analysis in the test split. Panel A gives net benefit, in net true "
    "positives per patient screened at 5 years, against threshold probability, for the M0 "
    "clinical Cox, M1 clinical plus KLG grade, M2 frontal radiograph and M4 multimodal fusion "
    "arms, with treat all and treat none in grey; the marker on treat all is its zero "
    "crossing, at the 5-year cumulative incidence of 0.2004. At each threshold the rule flags "
    "every patient whose predicted 5-year risk reaches it. Panel B gives the paired "
    "differences for the M2 frontal radiograph arm, against treat all and against M0 clinical "
    "Cox, with pointwise 2,000-replicate percentile bootstrap 95 percent intervals, "
    "unadjusted across thresholds, and a zero reference line. Panel A carries no intervals: "
    "marginal intervals over the same patients overlap where the paired difference between "
    "them does not. Thresholds 0.02 to 0.30 are drawn of the 0.01 to 0.35 the underlying "
    "table reports, and each curve stops where its flagged set falls below the 15-event "
    "floor. The arms score different populations: 741 patients and 106 events for M0 clinical "
    "Cox, 734 and 106 for M2 frontal radiograph, and 707 and 98 for the set every arm scores; "
    "Table 3 gives them per arm. Panel A draws M0 clinical Cox's treat all while Panel B's "
    "difference is taken against M2 frontal radiograph's own, so the two do not line up "
    "exactly. M2 frontal radiograph under-predicts 5-year risk by 5.6 percentage points in "
    "the large, and net benefit is sensitive to calibration where discrimination is not, so "
    "the horizontal axis is a decision-rule parameter rather than a true risk. The estimator, "
    "the threshold grid, the plotted window and the sensitivity curve beside this one are set "
    "out in the deviation register. These are out-of-sample estimates: the test split was "
    "read once, after every model and every hyperparameter choice had been frozen."
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cohort_dir(cfg):
    return cfg.path(cfg["paths"]["cohort_dir"])


def _skip_unless(path: Path, what: str) -> Path:
    if not Path(path).exists():
        pytest.skip(f"{what} is not present at {path}")
    return Path(path)


def _register_text(cfg) -> str:
    """The deviation register, read the way ``src.make_manuscript`` reads it.

    Figure 4's caption was cut from 557 words to about 300 on 2026-07-30 by moving four
    justifications into this file, which the Methods already cite by their D-range as
    accompanying material. A test that only checked the words were GONE would be happy with
    them having been deleted, so the ones that had to survive are asserted here, in the
    place they were moved to.
    """
    path = cfg.path(cfg["paths"]["outputs_dir"]) / "protocol_deviations.md"
    return _skip_unless(path, "the deviation register").read_text()


# =========================================================================== #
# 1. THE ANCHOR TABLE AGAINST THE REAL ARTEFACTS                              #
# =========================================================================== #
def test_anchor_table_is_the_only_source_of_the_legacy_names():
    """The EXPECTED_* names are DERIVED, so there is one place a count is written down."""
    assert mf.EXPECTED_SPLIT_N == {"train": 2597, "val": 371, "test": 741}
    assert mf.EXPECTED_SPLIT_EVENTS == {"train": 373, "val": 54, "test": 106}
    assert mf.EXPECTED_DEV_N == 2968
    assert mf.EXPECTED_DEV_EVENTS == 427
    assert mf.EXPECTED_CROP_N == 2966
    assert mf.EXPECTED_CROP_SPLIT_N == {"train": 2595, "val": 371}
    assert mf.EXPECTED_CROP_SPLIT_EVENTS == {"train": 373, "val": 54}
    assert mf.EXPECTED_N_CROPS == 4855
    assert mf.EXPECTED_CROPS_BY_SPLIT == {"train": 4254, "val": 601}
    assert mf.EXPECTED_N_LANDMARK == 3709
    assert sum(mf.EXPECTED_SPLIT_N.values()) == mf.EXPECTED_N_LANDMARK


def test_sealed_anchors_are_the_plan_numbers():
    """741 patients, 106 events, 740 of them carrying a crop, 1,216 crops."""
    assert mf.SPLIT_ANCHORS[TEST] == {"n": 741, "events": 106, "crop_n": 740,
                                      "crop_events": 106, "crops": 1216,
                                      "panel_b_n": 707, "panel_b_events": 98}


def test_split_sizes_match_the_locked_split_table(cohort_dir):
    path = _skip_unless(cohort_dir / "patient_splits.parquet", "the locked split table")
    counts = pd.read_parquet(path, columns=["split"])["split"].value_counts().to_dict()
    assert {k: int(v) for k, v in counts.items()} == mf.EXPECTED_SPLIT_N


def test_split_events_match_the_frozen_imputation_metadata(cohort_dir):
    path = _skip_unless(cohort_dir / "clinical_imputation_params.json",
                        "the frozen imputation metadata")
    ev = json.loads(path.read_text())["split_event_counts"]
    assert {k: int(v) for k, v in ev.items()} == mf.EXPECTED_SPLIT_EVENTS


@pytest.mark.parametrize("split", [VAL, TEST])
def test_crop_anchors_match_the_crop_label_index(cfg, split):
    """crops / crop-bearing patients / crop-bearing events, per split.

    Read through :func:`mf._crop_label_index`, which prefers the shard directory's own
    ``labels.csv`` and falls back to ``derived-data/cohort/preprocess_labels.csv`` when the
    shards are not staged on this machine. Both hold the same rows; the fallback is why the
    cohort-flow figure can still be rendered from a clone, and this test therefore runs
    everywhere instead of skipping wherever the shards are absent.
    """
    lab, src = mf._crop_label_index(cfg, split)
    assert "labels.csv" in src or "preprocess_labels.csv" in src
    pat = lab.drop_duplicates("empi_anon")
    for s in mf.RENDERED_SPLITS[split]:
        anchors = mf.SPLIT_ANCHORS[s]
        rows = lab[lab["split"] == s]
        pats = pat[pat["split"] == s]
        assert len(rows) == anchors["crops"], f"{s}: crops moved"
        assert len(pats) == anchors["crop_n"], f"{s}: crop-bearing patients moved"
        assert int(pats["event_indicator"].sum()) == anchors["crop_events"], \
            f"{s}: crop-bearing events moved"
    assert len(lab) == sum(mf.SPLIT_ANCHORS[s]["crops"] for s in mf.RENDERED_SPLITS[split])


@pytest.mark.parametrize("split", [VAL, TEST])
def test_crop_attrition_is_zero_events_and_the_expected_patients(cfg, split):
    """The 740-vs-741 fact itself: 2 development patients and 1 test patient lost every crop,
    and none of the three carried an event."""
    rendered = mf.RENDERED_SPLITS[split]
    n = sum(mf.SPLIT_ANCHORS[s]["n"] for s in rendered)
    crop_n = sum(mf.SPLIT_ANCHORS[s]["crop_n"] for s in rendered)
    ev = sum(mf.SPLIT_ANCHORS[s]["events"] for s in rendered)
    crop_ev = sum(mf.SPLIT_ANCHORS[s]["crop_events"] for s in rendered)
    assert n - crop_n == (2 if split == VAL else 1)
    assert ev - crop_ev == 0


def test_anchor_table_is_a_literal_and_never_read_from_config():
    """Deliberately NOT in config: a config edit must not be able to weaken the guard.

    Checked on the parse tree rather than by grepping the config, because the config does
    carry a REPORTED cohort size in its Phase 1 summary block. What matters is that nothing
    in this module reads it: SPLIT_ANCHORS has to be a literal whose only names are the two
    canonical split names imported from src.eval_models.
    """
    import ast
    import inspect as inspect_module

    tree = ast.parse(inspect_module.getsource(mf))
    assigns = [n for n in tree.body
               if isinstance(n, ast.AnnAssign)
               and isinstance(n.target, ast.Name) and n.target.id == "SPLIT_ANCHORS"]
    assert len(assigns) == 1, "SPLIT_ANCHORS must be assigned exactly once, at module level"
    for node in ast.walk(assigns[0].value):
        assert isinstance(node, (ast.Dict, ast.Constant, ast.Name, ast.Load)), \
            f"SPLIT_ANCHORS holds a computed value ({type(node).__name__}); it must be literal"
        if isinstance(node, ast.Name):
            assert node.id in ("VAL_SPLIT", "SEALED_SPLIT"), \
                f"SPLIT_ANCHORS refers to {node.id!r}; only the canonical split names allowed"


def test_anchored_counts_survive_a_config_that_disagrees(cfg):
    """Editing the config's reported cohort size must not move a caption's count."""
    edited = Config(cfg)
    edited["cohort"] = {**dict(cfg.get("cohort", {})), "final_cohort": 1}
    spec = mf.supplement_figures(edited, VAL)
    assert "3,709-patient landmark cohort" in spec["figureS2"]["caption"]
    assert "The 741 patients assigned to the test split" in spec["figureS2"]["caption"]


def test_split_path_and_the_sealed_guard_are_imported_not_reimplemented():
    """One implementation of the val_ -> test_ rewrite, and one sealed-read guard."""
    assert mf.split_path is em.split_path
    assert mf.assert_sealed_read_is_recorded is em.assert_sealed_read_is_recorded
    assert (mf.VAL_SPLIT, mf.SEALED_SPLIT) == (em.VAL_SPLIT, em.SEALED_SPLIT)


# =========================================================================== #
# 2. THE FIGURE REGISTRY                                                      #
# =========================================================================== #
def test_registry_is_ordered_numbered_and_unique():
    assert [d.number for d in mf.FIGURE_DEFS] == list(range(1, len(mf.FIGURE_DEFS) + 1))
    assert mf.FIGURE_KEYS == tuple(d.key for d in mf.FIGURE_DEFS)
    assert len({d.filename for d in mf.FIGURE_DEFS}) == len(mf.FIGURE_DEFS)
    assert all(callable(d.renderer) for d in mf.FIGURE_DEFS)
    assert set(mf.RENDERERS) == set(mf.FIGURE_KEYS)


@pytest.mark.parametrize("break_it, needle", [
    (lambda ds: (ds[0], replace(ds[1], number=4), ds[2]), "numbered 1.."),
    (lambda ds: (ds[0], ds[1], replace(ds[2], number=2)), "numbered 1.."),
    (lambda ds: (ds[1], ds[0], ds[2]), "numbered 1.."),
    (lambda ds: (ds[0], ds[2]), "numbered 1.."),
    (lambda ds: (ds[0], replace(ds[1], filename=ds[0].filename), ds[2]),
     "duplicate figure filename"),
    (lambda ds: (ds[0], replace(ds[1], key=ds[0].key, number=2), ds[2]),
     "duplicate figure key"),
    (lambda ds: (ds[0], ds[1], replace(ds[2], renderer=None)), "needs a renderer"),
    (lambda ds: (ds[0], ds[1], replace(ds[2], width_key="triple_column_in")), "width_key"),
])
def test_registry_invariants_fire_on_a_broken_registry(break_it, needle):
    with pytest.raises(AssertionError, match=needle):
        mf.assert_registry(break_it(mf.FIGURE_DEFS))


def test_registry_accepts_one_more_figure():
    """Registering the next figure must be one appended entry and nothing else.

    The number and key are derived from the registry's current length, not written as
    literals: figure 4 landed while this test was being written, and a hardcoded 4 turned
    a passing test into ``numbered 1..5, got [1, 2, 3, 4, 4]``. Whatever N is today, N+1
    must register with no other edit.
    """
    n = len(mf.FIGURE_DEFS) + 1
    nxt = mf.FigureDef(key=f"figure{n}", number=n, filename=f"figure{n}_next.png",
                       width_key="single_column_in",
                       title=lambda ctx: "The next figure",
                       caption=lambda ctx: f"Something in the {ctx['split_word']} split.",
                       renderer=lambda cfg, out_dir, split: out_dir / f"figure{n}.png")
    grown = (*mf.FIGURE_DEFS, nxt)
    assert mf.assert_registry(grown) is grown


def test_figures_returns_exactly_the_five_spec_keys_in_number_order(cfg):
    spec = mf.figures(cfg, VAL)
    assert list(spec) == list(mf.FIGURE_KEYS)
    assert [spec[k]["number"] for k in spec] == sorted(spec[k]["number"] for k in spec)
    for k, v in spec.items():
        assert set(v) == {"number", "filename", "width_in", "title", "caption"}
        assert isinstance(v["width_in"], float) and v["width_in"] > 0
        assert v["title"] and v["caption"]


def test_figures_widths_come_from_config(cfg):
    spec = mf.figures(cfg, VAL)
    supp = mf.supplement_figures(cfg, VAL)
    single = float(cfg["manuscript"]["single_column_in"])
    double = float(cfg["manuscript"]["double_column_in"])
    # Every v6 main figure is double column: three of the four are image grids or forests
    # that a single column cannot hold at 300 dpi without the labels colliding.
    for key in mf.FIGURE_KEYS:
        assert spec[key]["width_in"] == double, key
    assert supp["figureS1"]["width_in"] == double
    assert supp["figureS2"]["width_in"] == double
    assert supp["figureS3"]["width_in"] == single
    assert supp["figureS4"]["width_in"] == single
    assert supp["figureS5"]["width_in"] == double
    assert supp["figureS6"]["width_in"] == double


@pytest.mark.parametrize("bad", ["train", "Val", "TEST", "", "validation"])
def test_figures_rejects_an_unknown_split(cfg, bad):
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.figures(cfg, bad)


def test_module_level_FIGURES_is_pinned_to_validation(cfg):
    """It carries no split argument, so it must not return sealed-split prose."""
    assert mf.FIGURES == mf.figures(cfg, VAL)


# =========================================================================== #
# 3. CAPTIONS: CLAUSES THAT COMPOSE TO THE FROZEN VALIDATION TEXT             #
# =========================================================================== #
def test_val_figure1_and_figure3_reproduce_the_frozen_document_text(cfg):
    spec = mf.supplement_figures(cfg, VAL)
    assert spec["figureS2"]["title"] == FROZEN_VAL_TITLES["figureS2"]
    assert spec["figureS2"]["caption"] == FROZEN_VAL_FIG1_CAPTION
    assert spec["figureS4"]["title"] == FROZEN_VAL_TITLES["figureS4"]
    assert spec["figureS4"]["caption"] == FROZEN_VAL_FIG3_CAPTION


def test_figure1_caption_is_assembled_from_its_clause_constants(cfg):
    """The frozen text is the CLAUSES joined, not a second copy of the paragraph."""
    ctx = mf.caption_context(cfg, VAL)
    rebuilt = mf._sentences(
        mf.FIG1_LEDGER,
        f"{mf.FIG1_SEALED_LEAD} {mf.FIG1_SEALED_NEVER_READ}",
        mf.FIG1_CROP_DEV,
    ).format(**ctx)
    assert rebuilt == FROZEN_VAL_FIG1_CAPTION


def test_figure3_caption_is_assembled_from_its_clause_constants(cfg):
    ctx = mf.caption_context(cfg, VAL)
    rebuilt = mf._sentences(mf.FIG3_LEAD, mf.FIG3_AT_RISK, mf.FIG3_COMPETING,
                            mf.FIG3_DENOMINATOR).format(**ctx)
    assert rebuilt == FROZEN_VAL_FIG3_CAPTION


def test_split_invariant_clauses_are_shared_objects_not_copies(cfg):
    """Text true on both splits appears once. Only the flipping claims have two versions."""
    val = {**mf.figures(cfg, VAL), **mf.supplement_figures(cfg, VAL)}
    test = {**mf.figures(cfg, TEST), **mf.supplement_figures(cfg, TEST)}
    for key, clause in (("figureS2", mf.FIG1_LEDGER),
                        ("figureS4", mf.FIG3_AT_RISK),
                        ("figureS4", mf.FIG3_COMPETING),
                        ("figureS3", mf.FIG4_ESTIMATOR),
                        ("figureS3", mf.FIG4_WHY_NO_BANDS),
                        ("figureS3", mf.FIG4_REGISTER),
                        ("figure1", mf.WF_NETWORK),
                        ("figure2", mf.FIND_ROWS),
                        ("figure3", mf.FOREST_OCCLUSION),
                        ("figure4", mf.CAL_COMPETING)):
        rendered = clause.format(**mf.caption_context(cfg, VAL))
        assert rendered in val[key]["caption"], key
        assert clause.format(**mf.caption_context(cfg, TEST)) in test[key]["caption"], key


def test_figure1_keeps_the_two_true_clauses_and_flips_only_the_third(cfg):
    """"No model fitting" and "no hyperparameter choice" hold on BOTH splits; only whether
    the sealed patients contributed to the reported evaluation changes."""
    val = mf.supplement_figures(cfg, VAL)["figureS2"]["caption"]
    test = mf.supplement_figures(cfg, TEST)["figureS2"]["caption"]
    for cap in (val, test):
        assert "no model fitting" in cap
        assert "hyperparameter choice" in cap
        assert "were set aside unread" in cap
    assert "no evaluation" not in val and "evaluation reported in this manuscript" in val
    assert "read once, after every model was frozen" in test


def test_test_split_cohort_flow_caption_stops_saying_never_read(cfg):
    test = mf.supplement_figures(cfg, TEST)
    main = mf.figures(cfg, TEST)
    assert "never read" not in test["figureS2"]["caption"]
    assert "development patients only" not in test["figureS2"]["caption"]
    assert "single read" in test["figureS2"]["caption"]
    # The sealed-read clause is shared by every figure that carries one, so the check that
    # it flipped is made on all of them rather than on whichever one used to hold it.
    for key, spec in {**main, **test}.items():
        if mf.SEALED_READ_VAL in spec["caption"] or mf.SEALED_READ_TEST in spec["caption"]:
            assert mf.SEALED_READ_VAL not in spec["caption"], key
            assert "out-of-sample" in spec["caption"], key


def test_captions_name_the_split_they_describe(cfg):
    val = {**mf.figures(cfg, VAL), **mf.supplement_figures(cfg, VAL)}
    test = {**mf.figures(cfg, TEST), **mf.supplement_figures(cfg, TEST)}
    assert "validation split" in val["figure4"]["caption"]
    assert "Validation patients" in val["figureS4"]["caption"]
    assert "test split" in test["figure4"]["caption"]
    assert "Test patients" in test["figureS4"]["caption"]
    assert "Validation" not in test["figureS4"]["caption"]
    assert "test-split patients" in test["figure2"]["caption"]


def test_captions_carry_the_split_specific_denominators(cfg):
    """Every denominator in a caption is an anchor, so it cannot drift from the image."""
    val = {**mf.figures(cfg, VAL), **mf.supplement_figures(cfg, VAL)}
    test = {**mf.figures(cfg, TEST), **mf.supplement_figures(cfg, TEST)}
    assert "371 patients and 54 events" in val["figureS4"]["caption"]
    assert "740 patients and 106 events" in test["figureS4"]["caption"]
    assert "359 patients, carrying 52 events" in val["figure4"]["caption"]
    assert "707 patients, carrying 98 events" in test["figure4"]["caption"]


# --------------------------------------------------------------------------- #
# The crop-attrition count was the ONE cohort number written twice: spelled out #
# in the caption clause and subtracted from SPLIT_ANCHORS everywhere else. It   #
# is now interpolated, and these tests hold it that way.                        #
# --------------------------------------------------------------------------- #
def test_crop_loss_is_the_anchor_subtraction_over_the_rendered_splits():
    assert mf.crop_loss(VAL) == mf.EXPECTED_DEV_N - mf.EXPECTED_CROP_N == 2
    assert mf.crop_loss(TEST) == (mf.SPLIT_ANCHORS[TEST]["n"]
                                  - mf.SPLIT_ANCHORS[TEST]["crop_n"]) == 1
    # On the sealed split the render describes one split, so this is the same subtraction
    # expected_unscored makes for an all-view arm. On validation it is deliberately wider:
    # the frame is val alone, the flow chart and its caption are train plus val.
    assert mf.crop_loss(TEST) == mf.expected_unscored("m4_fusion", TEST)
    assert mf.crop_loss(VAL) == 2 and mf.expected_unscored("m4_fusion", VAL) == 0
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.crop_loss("train")


def test_figure1_crop_clauses_spell_no_count_of_their_own():
    """The clause constants carry placeholders. A spelled-out numeral here is the defect."""
    for clause in (mf.FIG1_CROP_DEV, mf.FIG1_CROP_TEST):
        assert "{Crop_lost_word}" in clause
        assert "{crop_lost_noun}" in clause
        assert "{crop_lost_subject}" in clause
        low = clause.lower()
        for spelled in ("one ", "two ", "both ", "that patient", "patients "):
            assert spelled not in low, f"{spelled!r} is typed into {clause!r}"


def test_figure1_crop_count_follows_a_moved_test_anchor(cfg, monkeypatch):
    """Re-anchor the sealed split's crop-bearing patients and the caption must move with it.

    This is the drift the spelled-out literal allowed: expected_unscored and figure 3's
    render-time assertion both pick up a changed crop_n, while the caption went on saying
    "One test patient".
    """
    before = mf.supplement_figures(cfg, TEST)["figureS2"]["caption"]
    assert "One test patient had no usable image of the contralateral knee" in before
    assert "that patient was event free" in before

    moved = {**mf.SPLIT_ANCHORS, TEST: {**mf.SPLIT_ANCHORS[TEST], "crop_n": 739}}
    monkeypatch.setattr(mf, "SPLIT_ANCHORS", moved)
    after = mf.supplement_figures(cfg, TEST)["figureS2"]["caption"]
    assert "Two test patients had no usable image of the contralateral knee" in after
    assert "both were event free" in after
    assert "One test patient" not in after
    assert "all 106 test events were retained" in after      # events did not move


def test_figure1_crop_count_follows_a_moved_development_anchor(cfg, monkeypatch):
    """The development side is derived too, and pluralises past two."""
    moved = {**mf.SPLIT_ANCHORS, "train": {**mf.SPLIT_ANCHORS["train"], "crop_n": 2594}}
    monkeypatch.setattr(mf, "SPLIT_ANCHORS", moved)
    cap = mf.supplement_figures(cfg, VAL)["figureS2"]["caption"]
    assert "Three development patients had no usable image of the contralateral knee" in cap
    assert "all three were event free" in cap
    assert "Two development patients" not in cap
    assert "all 427 development events were retained" in cap


def test_no_caption_cites_a_table_by_number(cfg):
    """Per-arm denominators genuinely differ (741 / 740 / 734 / 707) and the forest prints
    each one beside its own row. What no caption may do is cite a table by NUMBER: the table
    set is restructured by the manuscript task, so a number typed here is either a dangling
    citation the document's own reverse scan rejects or a pointer at the wrong table."""
    for split in (VAL, TEST):
        for key, spec in {**mf.figures(cfg, split),
                          **mf.supplement_figures(cfg, split)}.items():
            if key in ("figureS2", "figureS3", "figureS4"):
                continue          # the v5 main figures, frozen prose and all; see below
            for field in ("title", "caption"):
                assert not re.search(r"\bTables?\s+\d", spec[field]), f"{split}/{key}/{field}"
    assert "printed beside each row" in mf.figures(cfg, TEST)["figure3"]["caption"]
    # S2, S3 and S4 are the v5 main figures and keep their reviewed prose, and S3 carries the
    # one sanctioned table number in the whole figure set: "Table 3 gives them per arm".
    # It said "Table 2" until 2026-08-11. v6 split v5's Table 2 into Table 2 (primary and
    # head-to-head contrasts) and Table 3 (per-arm discrimination and calibration), and the
    # denominators the clause promises are per ARM, so they are Table 3's rows; Table 2 now
    # prints paired counts per contrast and does not give them per arm at all. The exemption
    # is therefore a POINTER that has to be maintained, not a licence to leave the number
    # wherever it was: the string moved here, in FIG4_DENOMINATORS and in
    # FROZEN_TEST_FIG4_CAPTION in one edit, and the authoritative layout it points at is
    # outputs/manuscript/"v6- resubmission"/sections/tables.md.
    assert "Table 3" in mf.supplement_figures(cfg, TEST)["figureS3"]["caption"]
    # and nothing points at the table that no longer holds these counts
    assert "Table 2" not in mf.supplement_figures(cfg, TEST)["figureS3"]["caption"]


def test_no_em_dashes_or_underscores_in_any_caption(cfg):
    """Both halves of the name. The underscore half was never checked, and it is the one
    that catches a caption printing an ARM KEY: "m2_frontal" and "auc_1825" are the
    pipeline's vocabulary, not a reader's, and they reach a caption by interpolation rather
    than by being typed, so no reviewer would see them coming."""
    for split in (VAL, TEST):
        for key, spec in {**mf.figures(cfg, split),
                          **mf.supplement_figures(cfg, split)}.items():
            for field in ("title", "caption"):
                assert "—" not in spec[field], f"{split}/{key}/{field} has an em-dash"
                offenders = [w for w in spec[field].split() if "_" in w]
                assert not offenders, f"{split}/{key}/{field} has underscores: {offenders}"


def test_caption_context_rejects_a_bad_split(cfg):
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.caption_context(cfg, "trian")


# =========================================================================== #
# 4. SPLIT PLUMBING                                                           #
# =========================================================================== #
def test_default_split_comes_from_manuscript_report_split(cfg):
    assert mf.default_split(cfg) == str(cfg["manuscript"]["report_split"])


@pytest.mark.parametrize("bad", ["train", "TEST", "sealed", ""])
def test_default_split_rejects_anything_but_val_or_test(cfg, bad):
    edited = Config(cfg)
    edited["manuscript"] = {**dict(cfg["manuscript"]), "report_split": bad}
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.default_split(edited)


def test_metrics_table_resolves_through_split_path(cfg):
    for split in (VAL, TEST):
        want = em.split_path(cfg, "metrics_csv", split)
        assert want.name == f"{split}_metrics.csv"
        _skip_unless(want, f"{split}_metrics.csv")
        df = mf._metrics_table(cfg, split)
        for arm in mf.FIG2_MODELS:
            assert arm in df.index, f"{split}_metrics.csv has no row for {arm}"


def test_load_hazards_path_is_split_prefixed(cohort_dir):
    for split in (VAL, TEST):
        path = cohort_dir / f"{split}_hazards_{mf.FIG3_MODEL}.npz"
        _skip_unless(path, f"{split} hazards for {mf.FIG3_MODEL}")
        hz = mf.load_hazards(cohort_dir, mf.FIG3_MODEL, split)
        assert hz["path"] == path
        assert hz["hazards"].shape[0] == hz["patient_ids"].size


def test_load_hazards_rejects_a_bad_split(cohort_dir):
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.load_hazards(cohort_dir, mf.FIG3_MODEL, "train")


def test_load_hazards_names_the_producer_when_the_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.score_test"):
        mf.load_hazards(tmp_path, "m4_fusion", TEST)
    with pytest.raises(FileNotFoundError, match="src.train_model"):
        mf.load_hazards(tmp_path, "m4_fusion", VAL)


def _stub_renderer(key: str):
    """A renderer that writes the file it returns, as every real renderer does.

    The bytes are the KEY, so two stubbed renders of the same split produce byte-identical
    files and therefore byte-identical provenance manifests, and a stub standing in for
    figure 1 can never be confused with one standing in for figure 2.
    """
    def _r(cfg, out_dir, split):
        p = Path(out_dir) / f"{key}.png"
        p.write_bytes(key.encode())
        return p
    return _r


def test_render_all_checks_the_sealed_read_only_on_the_sealed_split(cfg, tmp_path,
                                                                    monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(mf, "assert_sealed_read_is_recorded",
                        lambda c: calls.append("checked") or "deadbeef")
    monkeypatch.setattr(mf, "RENDERERS",
                        {k: _stub_renderer(k) for k in mf.FIGURE_KEYS})
    mf.render_all(cfg, out_dir=tmp_path, split=VAL)
    assert calls == []
    mf.render_all(cfg, out_dir=tmp_path, split=TEST)
    assert calls == ["checked"]


def test_render_all_rejects_an_unknown_split(cfg, tmp_path):
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.render_all(cfg, out_dir=tmp_path, split="train")


def test_cli_rejects_a_split_argparse_does_not_know(capsys):
    """--split {val,test}: a third value must not reach the renderers."""
    with pytest.raises(SystemExit):
        mf.main(["--split", "train"])
    assert "invalid choice" in capsys.readouterr().err


# =========================================================================== #
# 5. THE 740-vs-741 RULE                                                      #
# =========================================================================== #
def test_expected_unscored_is_zero_on_val_and_one_on_test():
    assert mf.expected_unscored("m4_fusion", VAL) == 0
    assert mf.expected_unscored("m4_fusion", TEST) == 1
    assert mf.expected_unscored("m3_image", TEST) == 1
    assert mf.expected_unscored("m0", VAL) == 0
    assert mf.expected_unscored("m0", TEST) == 0


def test_expected_unscored_does_not_pin_a_legitimate_subset_arm():
    """A frontal-only arm needs a frontal crop and m1 needs an observed grade: their
    denominators are cross-checked against the metrics table, not against an anchor."""
    for arm in ("m1", "m2_frontal", "m4_frontal", "m1_klg", "r1_densenet_frontal"):
        assert mf.expected_unscored(arm, TEST) is None
        assert mf.expected_unscored(arm, VAL) is None


@pytest.fixture(scope="module")
def frames(cfg):
    """The real split frames, loaded once. Skips if the feature table is absent."""
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    _skip_unless(coh / "features_clinical.parquet", "the clinical feature table")
    mc = mf._import_model_clinical()
    return mc, {s: mf._split_frame(cfg, mc, s) for s in (VAL, TEST)}


def test_split_frame_holds_the_anchored_rows_and_events(frames):
    _, fr = frames
    for split, f in fr.items():
        assert len(f) == mf.SPLIT_ANCHORS[split]["n"]
        assert int(f["event_indicator"].sum()) == mf.SPLIT_ANCHORS[split]["events"]
        assert set(f["split"].unique()) == {split}


def test_arm_risks_leaves_exactly_one_test_patient_unscored(cfg, frames):
    mc, fr = frames
    _skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) / f"test_hazards_{mf.FIG3_MODEL}.npz",
                 "the sealed hazards")
    r = mf._arm_risks(cfg, mc, fr[TEST], 1825.0, (mf.FIG3_MODEL,), TEST, QUIET)
    assert int((~np.isfinite(r[mf.FIG3_MODEL])).sum()) == 1
    assert int(np.isfinite(r[mf.FIG3_MODEL]).sum()) == mf.SPLIT_ANCHORS[TEST]["crop_n"]


def test_arm_risks_scores_every_validation_patient(cfg, frames):
    mc, fr = frames
    _skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) / f"val_hazards_{mf.FIG3_MODEL}.npz",
                 "the validation hazards")
    r = mf._arm_risks(cfg, mc, fr[VAL], 1825.0, (mf.FIG3_MODEL,), VAL, QUIET)
    assert np.isfinite(r[mf.FIG3_MODEL]).all()


def test_arm_risks_stays_silent_for_a_subset_arm_that_scores_far_fewer(cfg, frames):
    """m2_frontal misses 7 of 741 on test and m1 misses 34; neither is a drift."""
    mc, fr = frames
    _skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) / "test_hazards_m2_frontal.npz",
                 "the sealed frontal hazards")
    r = mf._arm_risks(cfg, mc, fr[TEST], 1825.0, ("m1", "m2_frontal"), TEST, QUIET)
    assert int(np.isfinite(r["m2_frontal"]).sum()) == 734
    assert int(np.isfinite(r["m1"]).sum()) == 707


def test_arm_risks_fires_when_an_all_view_arm_loses_another_patient(cfg, frames,
                                                                    monkeypatch):
    """The replacement for "every patient has a finite risk" has to catch a real drift."""
    mc, fr = frames
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    _skip_unless(coh / f"test_hazards_{mf.FIG3_MODEL}.npz", "the sealed hazards")
    real = mf.load_hazards

    def one_fewer(cohort_dir, arm, split):
        hz = real(cohort_dir, arm, split)
        hz["patient_ids"] = hz["patient_ids"][:-1]
        hz["hazards"] = hz["hazards"][:-1]
        return hz

    monkeypatch.setattr(mf, "load_hazards", one_fewer)
    with pytest.raises(AssertionError, match="have no predicted risk, expected 1"):
        mf._arm_risks(cfg, mc, fr[TEST], 1825.0, (mf.FIG3_MODEL,), TEST, QUIET)


def test_arm_risks_fires_when_the_metrics_table_disagrees(cfg, frames):
    mc, fr = frames
    _skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) / f"test_hazards_{mf.FIG3_MODEL}.npz",
                 "the sealed hazards")
    with pytest.raises(AssertionError, match="disagree about which patients"):
        mf._arm_risks(cfg, mc, fr[TEST], 1825.0, (mf.FIG3_MODEL,), TEST, QUIET,
                      expected_n={mf.FIG3_MODEL: 999})


def test_arm_risks_agrees_with_the_metrics_table_on_every_figure2_arm(cfg, frames):
    """The cross-artefact check that replaces the anchor for the subset arms."""
    mc, fr = frames
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    for split in (VAL, TEST):
        _skip_unless(coh / f"{split}_hazards_m2_frontal.npz", f"{split} frontal hazards")
        metrics = mf._metrics_table(cfg, split)
        expected = {a: int(metrics.loc[a, "n_patients"]) for a in mf.FIG2_MODELS}
        risks = mf._arm_risks(cfg, mc, fr[split], 1825.0, mf.FIG2_MODELS, split, QUIET,
                              expected_n=expected)
        for arm in mf.FIG2_MODELS:
            assert int(np.isfinite(risks[arm]).sum()) == expected[arm]


def test_panel_b_common_set_matches_the_caption_anchor(cfg, frames):
    """The number the figure 2 caption states is the intersection the image actually uses."""
    mc, fr = frames
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    for split in (VAL, TEST):
        _skip_unless(coh / f"{split}_hazards_m2_frontal.npz", f"{split} frontal hazards")
        risks = mf._arm_risks(cfg, mc, fr[split], 1825.0, mf.FIG2_MODELS, split, QUIET)
        common = np.ones(len(fr[split]), dtype=bool)
        for arm in mf.FIG2_MODELS:
            common &= np.isfinite(risks[arm])
        E = fr[split]["event_indicator"].to_numpy(dtype=int)
        assert int(common.sum()) == mf.SPLIT_ANCHORS[split]["panel_b_n"]
        assert int(E[common].sum()) == mf.SPLIT_ANCHORS[split]["panel_b_events"]


# =========================================================================== #
# 6. THE FROZEN COX REPLAYS                                                   #
# =========================================================================== #
@pytest.mark.parametrize("arm, expect", [("m0", {VAL: 371, TEST: 741}),
                                         ("m1", {VAL: 359, TEST: 707})])
def test_cox_replay_covers_the_arms_own_eligible_rows(cfg, frames, arm, expect):
    mc, fr = frames
    for split, n in expect.items():
        _, mj, mask = mf.cox_replay_risk(cfg, mc, fr[split], arm)
        assert int(mask.sum()) == n
        assert list(mj["design_columns"])


def test_cox_replay_reproduces_the_metrics_tables_discrimination(cfg, frames):
    """A replay that did not reproduce eval_models' own number would be a different model."""
    from src.train_model import harrell_c

    mc, fr = frames
    for split in (VAL, TEST):
        _skip_unless(em.split_path(cfg, "metrics_csv", split), f"{split}_metrics.csv")
        metrics = mf._metrics_table(cfg, split)
        f = fr[split]
        T = f["time_from_landmark"].to_numpy(dtype=float)
        E = f["event_indicator"].to_numpy(dtype=int)
        for arm in mf.COX_ARMS:
            risk, _, mask = mf.cox_replay_risk(cfg, mc, f, arm)
            key = min(risk, key=lambda t: abs(t - 1825.0))
            got = harrell_c(T[mask], E[mask], np.asarray(risk[key], dtype=float))
            assert got == pytest.approx(float(metrics.loc[arm, "harrell_c"]), abs=1e-6)


def test_m0_replay_risk_is_a_rename_not_a_new_contract(cfg, frames):
    mc, fr = frames
    risk, m0 = mf.m0_replay_risk(cfg, mc, fr[VAL])
    assert set(m0["design_columns"]) and 1825.0 in risk
    assert np.asarray(risk[1825.0]).size == mf.SPLIT_ANCHORS[VAL]["n"]


def test_cox_replay_rejects_an_arm_that_is_not_a_frozen_comparator(cfg, frames):
    mc, fr = frames
    with pytest.raises(AssertionError, match="not a frozen Cox comparator"):
        mf.cox_replay_risk(cfg, mc, fr[VAL], "m4_fusion")


# =========================================================================== #
# 7. FIGURE 1 BOX TEXT, PER SPLIT                                             #
# =========================================================================== #
@pytest.fixture(scope="module")
def flow_boxes(cfg):
    out = {}
    for split in (VAL, TEST):
        out[split] = mf._flow_boxes(cfg, QUIET, split)
    return out


def test_flow_ends_on_the_rendered_splits_crop_bearing_patients(flow_boxes):
    for split, boxes in flow_boxes.items():
        anchors = mf.SPLIT_ANCHORS
        rendered = mf.RENDERED_SPLITS[split]
        assert boxes[-1]["n"] == sum(anchors[s]["crop_n"] for s in rendered)
        assert boxes[-1]["events"] == sum(anchors[s]["crop_events"] for s in rendered)
        assert boxes[-2]["n"] == sum(anchors[s]["n"] for s in rendered)
        assert boxes[-2]["events"] == sum(anchors[s]["events"] for s in rendered)


def test_val_flow_still_says_the_sealed_split_was_never_read(flow_boxes):
    excl = flow_boxes[VAL][-2]["excl"]["lines"]
    assert "Sealed test split, never read" in excl
    assert "n = 741 (106 events)" in excl


def test_test_flow_branch_stops_saying_never_read(flow_boxes):
    boxes = flow_boxes[TEST]
    text = " ".join(ln for b in boxes for ln, _ in b["lines"]) + " " + " ".join(
        ln for b in boxes if b["excl"] for ln in b["excl"]["lines"])
    assert "never read" not in text
    assert "n = 741 (106 events)" in [ln for ln, _ in boxes[-2]["lines"]]
    assert any("Development cohort" in ln for ln in boxes[-2]["excl"]["lines"])
    assert "n = 1 (0 events)" in boxes[-1]["excl"]["lines"]
    assert "1,216 crops" in [ln for ln, _ in boxes[-1]["lines"]]


def test_flow_box_lines_are_wrapped_to_their_column(flow_boxes):
    """Text is drawn centred and is not clipped, so an unwrapped line breaks the width lock."""
    for boxes in flow_boxes.values():
        for b in boxes:
            for line, _bold in b["lines"]:
                assert len(line) <= mf.FLOW_WRAP_CHARS + 1, line
            if b["excl"]:
                for line in b["excl"]["lines"]:
                    assert len(line) <= mf.FLOW_EXCL_WRAP_CHARS + 1, line


def test_flow_fires_when_the_crop_index_moves(cfg, monkeypatch):
    """The crop anchors are restricted to the rendered splits, and still bite.

    Whichever of the two crop label indexes this machine has, dropping one row from it must
    fail the render rather than redraw a smaller cohort.
    """
    real = mf._crop_label_index

    def short(c, split):
        df, src = real(c, split)
        return df.iloc[:-1], src

    monkeypatch.setattr(mf, "_crop_label_index", short)
    with pytest.raises(AssertionError, match="crops"):
        mf._flow_boxes(cfg, QUIET, TEST)


# =========================================================================== #
# 8. FIGURE 4: THE DECISION CURVE                                             #
#                                                                              #
# The renderer reads ONE artefact and recomputes nothing, so these tests are    #
# mostly about that boundary: the schema it accepts, the truncation it owns,    #
# the prevalence it recovers from the table rather than from a literal, and     #
# the anchors its caption states, each checked against the real artefact that   #
# produced it rather than against a copy of itself.                             #
# =========================================================================== #
NB_LABEL = {"m0": "M0 clinical only (frozen penalized Cox)",
            "m1": "M1 clinical plus inferred KLG (frozen penalized Cox)",
            "m2_frontal": "M2 frontal radiograph", "m4_fusion": "M4 multimodal fusion"}


def _nb_settings(cfg):
    return em.net_benefit_settings(cfg)


def _nb_scored(split: str, arm: str, settings: dict) -> int:
    """The denominator render_figure4 asserts for each arm, from the anchor tables."""
    if arm == settings["reference"]:
        return mf.SPLIT_ANCHORS[split]["n"]
    if arm == next(a for a in settings["arms"] if a != settings["reference"]):
        return int(mf.NB_ANCHORS[split]["arm_n"])
    return mf.SPLIT_ANCHORS[split]["panel_b_n"]


def _nb_frame(cfg, split: str, *, sparse_from_pct: dict | None = None,
              suppress: tuple[str, ...] = ()) -> pd.DataFrame:
    """A synthetic net-benefit table in the pinned schema, consistent with the anchors.

    Built rather than read so the truncation, suppression and drift branches can be
    exercised without an artefact that a concurrent run is regenerating. The treat-all
    column is the estimator's own algebra, ``F - (1 - F) * w``, at the anchored prevalence,
    so :func:`mf.nb_prevalence_from_treat_all` has something real to invert.
    """
    s = _nb_settings(cfg)
    prev = float(mf.NB_ANCHORS[split]["prevalence"])
    sparse_from_pct = sparse_from_pct or {}
    rows = []
    for i, arm in enumerate(s["arms"]):
        n = _nb_scored(split, arm, s)
        for p, pct in zip(s["thresholds"], s["threshold_pcts"]):
            w = float(p) / (1.0 - float(p))
            # each arm sits on its own slightly different population, so its treat-all
            # differs a little; only the REFERENCE arm's has to hit the anchor exactly
            f = prev if arm == s["reference"] else prev + 0.0003 * (i + 1)
            nb_all = f - (1.0 - f) * w
            nb = nb_all + 0.02 * (i + 1) * float(p)
            rows.append({
                "split": split, "arm": arm, "label": NB_LABEL[arm],
                "threshold": float(p), "threshold_pct": int(pct),
                "horizon_days": int(s["horizon_days"]),
                "n_scored": n, "n_above": max(n - 20 * int(pct), 1),
                "events_above": max(int(mf.SPLIT_ANCHORS[split]["events"]) - 2 * int(pct), 0),
                "km_risk_above": 0.25, "km_last_obs_day": 1825.0,
                "net_benefit": nb, "net_benefit_lo": nb - 0.03, "net_benefit_hi": nb + 0.03,
                "net_benefit_ipcw": nb + 0.001,
                "nb_treat_all_same_set": nb_all,
                "diff_vs_treat_all": nb - nb_all,
                "diff_vs_treat_all_lo": nb - nb_all - 0.02,
                "diff_vs_treat_all_hi": nb - nb_all + 0.02,
                "diff_vs_treat_all_p": 0.0005,
                "net_reduction_per_100": 100.0 * (nb - nb_all) / w,
                "reference": s["reference"],
                "diff_vs_reference": 0.01 * (i + 1),
                "diff_vs_reference_lo": 0.01 * (i + 1) - 0.02,
                "diff_vs_reference_hi": 0.01 * (i + 1) + 0.02,
                "diff_vs_reference_p": 0.0005,
                "n_paired": min(n, mf.SPLIT_ANCHORS[split]["n"]),
                "n_replicates_valid": 2000,
                "sparse": int(pct) >= int(sparse_from_pct.get(arm, 10**6)),
                "suppressed": arm in suppress,
                "note": "",
            })
    return pd.DataFrame(rows, columns=list(em.NET_BENEFIT_COLUMNS))


def _write_convergence(cfg, split: str, arms, suppressed=()) -> Path:
    """A ``{split}_convergence.csv`` in the pinned schema, consistent with ``suppressed``.

    ``render_figure4`` settles whether figure 4 has an honest render from the CONVERGENCE
    GATE and not from the net-benefit table, so a fixture that writes one without the other
    is half a pipeline state and its consistency is accidental. This writes the other half.

    The verdict chosen for a suppressed arm is the one that actually disqualifies it on that
    split: ``severe_overfit`` is a validation-only disqualification, so on the sealed split
    it has to be ``did_not_converge`` instead. A fixture that wrote ``severe_overfit`` for a
    suppressed sealed-split arm would be building the contradiction, not the state.
    """
    bad = em.STATUS_OVERFIT if split == VAL else em.STATUS_NO_CONVERGE
    rows = [{"arm": str(a), "n_seeds": 5, "train_nll_drop": 0.2,
             "val_overfit_gap": 0.18 if a in suppressed else 0.02,
             "status": bad if a in suppressed else em.STATUS_OK,
             "reason": f"fixture: {bad}" if a in suppressed else ""}
            for a in arms]
    path = em.split_path(cfg, "convergence_csv", split)
    pd.DataFrame(rows, columns=list(em.CONVERGENCE_COLUMNS)).to_csv(path, index=False)
    return path


@pytest.fixture
def nb_cfg(cfg, tmp_path):
    """A config whose net-benefit AND convergence tables point into tmp_path, plus a writer.

    The basenames keep their ``val_`` prefix so ``split_path``'s one rewrite still produces
    ``test_net_benefit.csv``; nothing here may write into the repository's outputs/.

    ``write`` emits BOTH artefacts, and derives the convergence verdicts from the frame's own
    ``suppressed`` column, so every fixture below rehearses a state the pipeline could
    actually produce. Leaving ``convergence_csv`` pointed at the repository would make each
    of these a half-real hybrid: a synthetic validation table that keeps m2_frontal, read
    beside the real val_convergence.csv that disqualifies it, is exactly the disagreement
    ``decision_curve_decline_reason`` exists to catch.
    """
    edited = Config(cfg)
    edited["model_eval"] = {**dict(cfg["model_eval"]),
                            "net_benefit_csv": str(tmp_path / "val_net_benefit.csv"),
                            "convergence_csv": str(tmp_path / "val_convergence.csv")}

    def write(split: str, frame: pd.DataFrame) -> Path:
        p = em.split_path(edited, "net_benefit_csv", split)
        frame.to_csv(p, index=False)
        arms = list(dict.fromkeys(frame["arm"].astype(str)))
        _write_convergence(edited, split, arms,
                           tuple(a for a in arms if mf.nb_arm_is_suppressed(frame, a)))
        return p

    return edited, write


# --------------------------------------------------------------------------- #
# 8a. Registration                                                             #
# --------------------------------------------------------------------------- #
def test_the_decision_curve_is_registered_as_supplementary_figure_three(cfg):
    """It moved out of the main registry at v6 and took its renderer and prose with it."""
    d = {x.key: x for x in mf.SUPPLEMENT_DEFS}["figureS3"]
    assert (d.number, d.width_key) == (3, "single_column_in")
    assert d.filename == "figureS3_decision_curve.png"
    assert d.renderer is mf.render_decision_curve
    assert mf.SUPPLEMENT_RENDERERS["figureS3"] is mf.render_decision_curve
    assert mf.supplement_figures(cfg, TEST)["figureS3"]["number"] == 3
    # and it is NOT reachable through the main registry, which is what make_manuscript
    # numbers and embeds
    assert "figureS3" not in mf.FIGURE_KEYS and "figureS3" not in mf.figures(cfg, TEST)


# --------------------------------------------------------------------------- #
# 8b. The caption's required disclosures                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", [VAL, TEST])
def test_figure4_caption_states_units_pairing_and_the_pointwise_rule(cfg, split):
    """Re-pinned 2026-07-30, when the caption was cut from 557 words to about 300.

    What a caption must carry stays: the units, the two grey references, the zero crossing,
    what Panel B plots, and that its intervals are pointwise and unadjusted across
    thresholds - all four change how the panels are read. The ARGUMENT for the last of them,
    that the nested flagged sets are views of one curve rather than a family of hypotheses,
    is deviation D29's third mitigation and is asserted against the register itself below.
    """
    cap = mf.supplement_figures(cfg, split)["figureS3"]["caption"]
    assert "net true positives per patient screened at 5 years" in cap
    assert "treat all and treat none" in cap
    assert "zero crossing" in cap
    assert "paired differences" in cap
    assert "Panel A carries no intervals" in cap
    assert "pointwise" in cap and "unadjusted across thresholds" in cap
    moved = "35 views of one curve rather than 35 hypotheses"
    assert moved not in cap, "the multiplicity argument was moved, not kept"
    assert moved in _register_text(cfg), "it was moved OUT of the caption and lost"


@pytest.mark.parametrize("split", [VAL, TEST])
def test_figure4_caption_states_the_plotted_range_and_the_truncation_rule(cfg, split):
    cap = mf.supplement_figures(cfg, split)["figureS3"]["caption"]
    s = _nb_settings(cfg)
    lo, hi = s["plot_min_pct"] / 100, s["plot_max_pct"] / 100
    assert f"Thresholds {lo:.2f} to {hi:.2f}" in cap
    assert f"{s['threshold_min_pct'] / 100:.2f} to {s['threshold_max_pct'] / 100:.2f}" in cap
    assert f"below the {s['sparse_events_min']}-event floor" in cap
    # WHAT a reader is looking at stays; WHY the window ends where it does went to the
    # register on 2026-07-30. The odds-weight ceiling and the reference arm's largest
    # predicted risk justified a choice that a reader can check against the full
    # 35-threshold table, and neither changes how the drawn curves are read. The anchor is
    # still verified against the frozen Cox replay by
    # test_nb_reference_max_risk_anchor_matches_the_frozen_cox_replay; it is simply no
    # longer printed.
    assert "odds weight" not in cap
    assert f"{float(mf.NB_ANCHORS[split]['reference_max_risk']):.4f}" not in cap
    assert "set out in the deviation register" in cap
    assert f"Figure 4 plots {lo:.2f} to {hi:.2f}" in _register_text(cfg)


@pytest.mark.parametrize("split", [VAL, TEST])
def test_figure4_caption_discloses_the_three_differing_denominators(cfg, split):
    """741 / 734 / 707 and 106 / 106 / 98 on the sealed split: stated, never harmonised.

    The three denominators are the one thing the 2026-07-30 caption cut did not touch: they
    are what a reader needs to know the four curves are not drawn over one population. Only
    the wording tightened. The PAIRING rule that used to follow them moved to the contrast
    table's own note, which is where a reader comparing two printed levels is already
    looking, and it is asserted there below rather than assumed to have survived.

    The per-arm pointer is Table 3 from 2026-08-11, not Table 2: v6 split v5's Table 2 into
    the contrast table (Table 2, which prints PAIRED counts per contrast) and the per-arm
    table (Table 3, which prints exactly the 741/106, 734/106 and 707/98 this clause
    states). The pairing rule stayed with the contrasts and the denominators went with the
    arms, so the two assertions below now name two different tables on purpose.
    """
    cap = mf.supplement_figures(cfg, split)["figureS3"]["caption"]
    a, nb = mf.SPLIT_ANCHORS[split], mf.NB_ANCHORS[split]
    assert f"{a['n']} patients and {a['events']} events" in cap
    assert f"{int(nb['arm_n'])} and {int(nb['arm_events'])}" in cap
    assert f"{a['panel_b_n']} and {a['panel_b_events']} for the set every arm scores" in cap
    assert "Table 3 gives them per arm" in cap
    mm = pytest.importorskip("src.make_manuscript")
    assert "computed on the intersection of the two arms it compares" in \
        inspect.getsource(mm.build_table2)


@pytest.mark.parametrize("split", [VAL, TEST])
def test_figure4_caption_says_whose_patients_panel_as_treat_all_rests_on(cfg, split):
    """Panel A draws ONE grey treat-all, the reference arm's; panel B differences against the
    protagonist's. Every other denominator in this caption is disclosed, and until this
    clause landed that one was not, so a reader who measured panel A's vertical gap by eye
    got a number that disagreed with the panel B curve underneath it.
    """
    cap = mf.supplement_figures(cfg, split)["figureS3"]["caption"]
    s = _nb_settings(cfg)
    ref = mf.MODEL_DISPLAY[s["reference"]]
    prot = mf.MODEL_DISPLAY[next(a for a in s["arms"] if a != s["reference"])]
    # Re-pinned to the shorter wording on 2026-07-30. The CAVEAT is kept, because a reader
    # who measures Panel A's vertical gap by eye gets a number Panel B does not report; only
    # the explanation of why each arm carries its own treat-all was dropped.
    assert f"Panel A draws {ref}'s treat all" in cap
    assert f"Panel B's difference is taken against {prot}'s own" in cap
    assert "do not line up exactly" in cap


def test_the_real_table_really_does_give_each_arm_its_own_treat_all(cfg):
    """The clause is a disclosure, so it has to stay true of the artefact.

    Panel A plots ``nb_treat_all_same_set`` from the REFERENCE arm's rows while panel B's
    grey curve is the protagonist's ``diff_vs_treat_all``, which the estimator took against
    the protagonist's own treat-all. If those two columns ever coincided the sentence would
    be noise; while they differ it is the difference between what a reader measures and what
    the figure reports.
    """
    split = TEST
    _skip_unless(em.split_path(cfg, "net_benefit_csv", split), f"{split}_net_benefit.csv")
    s = _nb_settings(cfg)
    df = mf._net_benefit_table(cfg, split)
    prot = next(a for a in s["arms"] if a != s["reference"])
    a = mf._nb_arm_rows(df, prot, split, s, QUIET).set_index("threshold_pct")
    ref = mf.nb_treat_all_window(df, s).set_index("threshold_pct")
    pcts = a.index.intersection(ref.index)
    assert len(pcts) > 1

    # what panel A shows: the protagonist's curve minus the ONE grey curve that is drawn
    eye = a.loc[pcts, "net_benefit"] - ref.loc[pcts, "nb_treat_all_same_set"]
    # what panel B plots, from the estimator, against the protagonist's own treat-all
    plotted = a.loc[pcts, "diff_vs_treat_all"]
    gap = (eye - plotted).abs()
    assert gap.max() > mf.NB_PREVALENCE_TOL, (
        "panel A's treat-all and the protagonist's own now agree to within the printing "
        "tolerance, so FIG4_TREAT_ALL_SET warns about a discrepancy that no longer exists")
    # and the direction the clause implies: the drawn grey curve sits on a different, larger
    # population than the protagonist's, so the eyeballed gap is not the reported one
    assert int(ref["n_scored"].iloc[0]) != int(a["n_scored"].iloc[0])


@pytest.mark.parametrize("split", [VAL, TEST])
def test_figure4_caption_states_the_calibration_caveat_that_moves_the_axis(cfg, split):
    """ONE caveat now, not two, and the difference is deliberate (2026-07-30).

    KEPT, because it changes how the horizontal axis is read: the protagonist under-predicts
    in the large, net benefit is calibration-sensitive where discrimination is not, and the
    axis is therefore a decision-rule parameter rather than a true risk.

    DROPPED: the recalibration asymmetry, that the unrecalibrated reference arm is the
    better calibrated in the large. It defended the asymmetry rather than telling a reader
    how to read a panel, and unlike the estimator and the threshold grid it is NOT recorded
    in the register, so this test says plainly that the document no longer states it.
    ``reference_citl`` is still checked against {split}_metrics.csv by
    test_nb_calibration_anchors_match_the_metrics_table, and the monotonicity of the
    recalibration is still stated in figure 2's sealed caption.
    """
    cap = mf.supplement_figures(cfg, split)["figureS3"]["caption"]
    a = mf.NB_ANCHORS[split]
    assert "sensitive to calibration where discrimination is not" in cap
    assert f"{100.0 * float(a['arm_citl']):.1f} percentage points" in cap
    assert "decision-rule parameter rather than a true risk" in cap
    assert f"calibration in the large {float(a['arm_citl']):.4f}" not in cap
    assert f"strictly increasing (slope {mf.NB_RECAL_SLOPE:.3f})" not in cap
    assert "receive no recalibration while the image arms do" not in cap
    assert f"calibration in the large {float(a['reference_citl']):.4f}" not in cap
    if split == TEST:
        # v5 stated the monotonicity of the recalibration in figure 2's caption, because
        # that figure plotted UNRECALIBRATED predictions and the monotonicity was the reason
        # a threshold still flagged the same patients. The v6 calibration figure applies the
        # transform, so the caption states that instead; there is nothing left for
        # monotonicity to excuse.
        cal = mf.figures(cfg, split)["figure4"]["caption"]
        assert "drawn AFTER the horizon-specific recalibration" in cal
        assert "fitted on the validation split and applied unchanged here" in cal
        assert "before the horizon-specific recalibration" not in cal


def test_test_figure4_reproduces_the_shipped_document_text(cfg):
    """The frozen counterpart figure 4 never had. Independent ground truth, not the module.

    Figures 1 and 3 are pinned against v1's rendered text; figure 4 was pinned only against
    a re-typed copy of its own assembly, which cannot catch a clause edited in both places.
    v1 declined figure 4 on validation, so the sealed document is the only rendered version
    that exists, and it is the version a reader holds.
    """
    spec = mf.supplement_figures(cfg, TEST)["figureS3"]
    assert spec["title"] == FROZEN_TEST_FIG4_TITLE
    assert spec["caption"] == FROZEN_TEST_FIG4_CAPTION


def test_figure4_caption_is_assembled_from_its_clause_constants(cfg):
    """The paragraph is the CLAUSES joined, not a second copy of it.

    Together with the frozen test above this says both halves: the sealed caption is the
    shipped text, and the shipped text is these clauses in this order, on BOTH splits.
    """
    for split in (VAL, TEST):
        ctx = mf.caption_context(cfg, split)
        sealed = mf.SEALED_READ_VAL if split == VAL else mf.SEALED_READ_TEST
        rebuilt = mf._sentences(
            mf.FIG4_LEAD, mf.FIG4_ESTIMATOR, mf.FIG4_PANEL_B, mf.FIG4_WHY_NO_BANDS,
            mf.FIG4_RANGE, mf.FIG4_DENOMINATORS, mf.FIG4_TREAT_ALL_SET, mf.FIG4_CALIBRATION,
            mf.FIG4_REGISTER, sealed).format(**ctx)
        assert rebuilt == mf.supplement_figures(cfg, split)["figureS3"]["caption"]
    assert mf.supplement_figures(cfg, TEST)["figureS3"]["caption"] == FROZEN_TEST_FIG4_CAPTION


def test_figure4_caption_flips_only_the_sealed_read_clause(cfg):
    val = mf.supplement_figures(cfg, VAL)["figureS3"]["caption"]
    test = mf.supplement_figures(cfg, TEST)["figureS3"]["caption"]
    assert val.endswith(mf.SEALED_READ_VAL) and test.endswith(mf.SEALED_READ_TEST)
    assert "out-of-sample" in test and "out-of-sample" not in val


def test_figure4_caption_spells_no_cohort_or_calibration_number_of_its_own():
    """Every count, threshold and calibration value is a placeholder, as in figure 1's
    crop clause. A literal here is the drift the anchor tables exist to prevent.

    Two exemptions, both for numbers that are not measurements: "95 percent", the
    confidence level, which protocol section 18 fixes and no artefact can move; and a
    "Table N" cross-reference, which the manuscript's own numbering owns.
    """
    for clause in (mf.FIG4_LEAD, mf.FIG4_ESTIMATOR, mf.FIG4_PANEL_B, mf.FIG4_WHY_NO_BANDS,
                   mf.FIG4_RANGE, mf.FIG4_DENOMINATORS, mf.FIG4_TREAT_ALL_SET,
                   mf.FIG4_CALIBRATION, mf.FIG4_REGISTER):
        stripped = re.sub(r"95 percent|Table \d+", "", clause)
        assert not re.search(r"(?<![{\w.])\d", stripped), f"a literal number in {clause!r}"


def test_figure4_direction_word_is_asserted_not_assumed(cfg, monkeypatch):
    """"Under-predicts" is a CLAIM, not formatting: if it flips, the clause has to be
    rewritten rather than silently print its opposite.

    Its companion guard, over "the unrecalibrated reference arm is the BETTER calibrated in
    the large", was removed on 2026-07-30 together with FIG4_ASYMMETRY, the clause it
    protected. A guard over a sentence the document no longer prints can only fail a future
    render for a claim nobody is making, so the clause and its assertion went together, and
    that is asserted here rather than left to be noticed.
    """
    real = mf.NB_ANCHORS
    monkeypatch.setattr(mf, "NB_ANCHORS",
                        {**real, TEST: {**real[TEST], "arm_citl": -0.02}})
    with pytest.raises(AssertionError, match="UNDER-predicts"):
        mf.figures(cfg, TEST)

    monkeypatch.setattr(mf, "NB_ANCHORS",
                        {**real, TEST: {**real[TEST], "reference_citl": 0.30}})
    mf.figures(cfg, TEST)          # no longer a claim, so no longer a guard
    assert not hasattr(mf, "FIG4_ASYMMETRY") and not hasattr(mf, "FIG4_MONOTONE")


def test_figure4_caption_follows_the_configured_plot_range(cfg):
    """The plotted range is config, not prose: move it and the sentence moves."""
    edited = Config(cfg)
    nb = {**dict(cfg["model_eval"]["net_benefit"]), "plot_min_pct": 3, "plot_max_pct": 25}
    edited["model_eval"] = {**dict(cfg["model_eval"]), "net_benefit": nb}
    assert "Thresholds 0.03 to 0.25 are drawn" in mf.supplement_figures(edited, TEST)["figureS3"]["caption"]


# --------------------------------------------------------------------------- #
# 8c. The anchors, against the artefacts that produced them                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", [VAL, TEST])
def test_nb_calibration_anchors_match_the_metrics_table(cfg, split):
    """The caption and Table 2 quote one number, so they are read from one file."""
    _skip_unless(em.split_path(cfg, "metrics_csv", split), f"{split}_metrics.csv")
    met = mf._metrics_table(cfg, split)
    s = _nb_settings(cfg)
    prot = next(a for a in s["arms"] if a != s["reference"])
    horizon = int(s["horizon_days"])
    for arm, key in ((prot, "arm_citl"), (s["reference"], "reference_citl")):
        got = float(met.loc[arm, f"citl_{horizon}"])
        assert got == pytest.approx(float(mf.NB_ANCHORS[split][key]), abs=5e-7), (
            f"{split}_metrics.csv records citl_{horizon} = {got} for {arm}, but "
            f"NB_ANCHORS[{split!r}][{key!r}] states {mf.NB_ANCHORS[split][key]}. The split "
            "was re-scored; move the anchor with it so figure 4's caption and Table 2 do "
            "not print different numbers.")


@pytest.mark.parametrize("split", [VAL, TEST])
def test_nb_arm_anchor_matches_the_metrics_table(cfg, split):
    _skip_unless(em.split_path(cfg, "metrics_csv", split), f"{split}_metrics.csv")
    met = mf._metrics_table(cfg, split)
    s = _nb_settings(cfg)
    prot = next(a for a in s["arms"] if a != s["reference"])
    assert int(met.loc[prot, "n_patients"]) == int(mf.NB_ANCHORS[split]["arm_n"])
    assert int(met.loc[prot, "n_events"]) == int(mf.NB_ANCHORS[split]["arm_events"])
    # the two extremes the caption states come from SPLIT_ANCHORS, and must bracket it
    assert (mf.SPLIT_ANCHORS[split]["panel_b_n"] <= int(mf.NB_ANCHORS[split]["arm_n"])
            <= mf.SPLIT_ANCHORS[split]["n"])


@pytest.mark.parametrize("split", [VAL, TEST])
def test_nb_prevalence_anchor_is_the_splits_kaplan_meier_incidence(cfg, frames, split):
    """Treat-all crosses zero exactly at the prevalence, so this is THE number on the plot."""
    mc, fr = frames
    s = _nb_settings(cfg)
    f = fr[split]
    obs, _, _ = mc.km_risk(f["time_from_landmark"].to_numpy(dtype=float),
                           f["event_indicator"].to_numpy(dtype=int),
                           float(s["horizon_days"]))
    assert obs == pytest.approx(float(mf.NB_ANCHORS[split]["prevalence"]),
                                abs=mf.NB_PREVALENCE_TOL)


@pytest.mark.parametrize("split", [VAL, TEST])
def test_nb_reference_max_risk_anchor_matches_the_frozen_cox_replay(cfg, frames, split):
    """Above it the reference curve is identically zero, which is why the range stops."""
    mc, fr = frames
    s = _nb_settings(cfg)
    risk, _, mask = mf.cox_replay_risk(cfg, mc, fr[split], s["reference"])
    key = min(risk, key=lambda t: abs(t - float(s["horizon_days"])))
    got = float(np.max(np.asarray(risk[key], dtype=float)))
    assert got == pytest.approx(float(mf.NB_ANCHORS[split]["reference_max_risk"]), abs=5e-5)
    assert mask.all()


def test_nb_recal_slope_matches_the_frozen_train_arms_index(cfg, cohort_dir):
    """The monotonicity claim rests on this slope being positive, and on it being THIS one."""
    path = _skip_unless(cohort_dir / "train_arms.json", "the training hand-over index")
    s = _nb_settings(cfg)
    prot = next(a for a in s["arms"] if a != s["reference"])
    arms = json.loads(path.read_text())["arms"]
    key = prot if prot in arms else next(k for k in arms if k.startswith(prot))
    slope = float(arms[key]["recalibration"][str(float(s["horizon_days"]))]["slope"])
    assert slope > 0, "a non-positive slope would REVERSE the risk ordering"
    assert round(slope, 3) == pytest.approx(mf.NB_RECAL_SLOPE, abs=1e-9)


def test_nb_anchors_cover_exactly_the_renderable_splits():
    assert set(mf.NB_ANCHORS) == set(mf.SPLITS)
    for split, a in mf.NB_ANCHORS.items():
        assert set(a) == {"prevalence", "arm_n", "arm_events", "arm_citl",
                          "reference_citl", "reference_max_risk"}, split


# --------------------------------------------------------------------------- #
# 8d. Reading the one artefact                                                 #
# --------------------------------------------------------------------------- #
def test_net_benefit_table_names_its_producer_when_it_is_absent(cfg, tmp_path):
    edited = Config(cfg)
    edited["model_eval"] = {**dict(cfg["model_eval"]),
                            "net_benefit_csv": str(tmp_path / "val_net_benefit.csv")}
    for split in (VAL, TEST):
        with pytest.raises(FileNotFoundError, match=f"src.eval_models --split {split}"):
            mf._net_benefit_table(edited, split)


def test_net_benefit_table_resolves_through_split_path(nb_cfg):
    """The function under test is CALLED. The previous version asserted split_path alone
    and would have passed with _net_benefit_table deleted, which is the one outcome a test
    named for it must not have.

    Two files are written, differing only in a value, and the sealed read must come back
    with the sealed one. That fails if the reader takes the configured val_ path as given.
    """
    edited, write = nb_cfg
    frames = {}
    for split in (VAL, TEST):
        frame = _nb_frame(edited, split)
        frame.loc[:, "n_scored"] = 111 if split == VAL else 999
        frames[split] = frame
        path = write(split, frame)
        assert path.name == f"{split}_net_benefit.csv"
    for split in (VAL, TEST):
        got = mf._net_benefit_table(edited, split)
        assert set(got["split"].astype(str)) == {split}
        assert set(got["n_scored"]) == set(frames[split]["n_scored"]), (
            f"the {split!r} read returned the other split's file, so the val_ to test_ "
            f"rewrite is not happening on the way in")


def test_net_benefit_table_rejects_a_frame_that_is_not_the_pinned_schema(nb_cfg):
    edited, write = nb_cfg
    frame = _nb_frame(edited, TEST).drop(columns=["net_reduction_per_100"])
    write(TEST, frame)
    with pytest.raises(AssertionError, match="pinned net-benefit schema"):
        mf._net_benefit_table(edited, TEST)


def test_net_benefit_table_rejects_rows_from_another_split(nb_cfg):
    edited, write = nb_cfg
    frame = _nb_frame(edited, TEST)
    frame.loc[0, "split"] = VAL
    write(TEST, frame)
    with pytest.raises(AssertionError, match="must not disagree about which patients"):
        mf._net_benefit_table(edited, TEST)


def test_net_benefit_table_accepts_the_schema_it_is_given(nb_cfg):
    edited, write = nb_cfg
    write(TEST, _nb_frame(edited, TEST))
    df = mf._net_benefit_table(edited, TEST)
    assert list(df.columns) == list(em.NET_BENEFIT_COLUMNS)
    assert set(df["arm"]) == set(_nb_settings(edited)["arms"])


# --------------------------------------------------------------------------- #
# 8e. Truncation: the one thing the FIGURE decides                             #
# --------------------------------------------------------------------------- #
def test_nb_arm_rows_truncates_at_the_first_sparse_threshold(cfg):
    """The estimator FLAGS, the figure truncates. The flagged row itself is not drawn."""
    s = _nb_settings(cfg)
    df = _nb_frame(cfg, TEST, sparse_from_pct={"m0": 18})
    rows = mf._nb_arm_rows(df, "m0", TEST, s, QUIET)
    assert int(rows["threshold_pct"].max()) == 17
    assert not bool(rows["sparse"].any())
    # every other arm keeps the whole plotted window
    other = mf._nb_arm_rows(df, "m4_fusion", TEST, s, QUIET)
    assert int(other["threshold_pct"].max()) == s["plot_max_pct"]


def test_nb_arm_rows_keeps_a_sparse_flag_outside_the_window_from_shortening_the_curve(cfg):
    """m0 first goes sparse at 0.31 on the sealed split, past the 0.30 ceiling, so nothing
    is truncated inside the drawn range; the rule still has to be applied, not skipped."""
    s = _nb_settings(cfg)
    df = _nb_frame(cfg, TEST, sparse_from_pct={"m0": s["plot_max_pct"] + 1})
    rows = mf._nb_arm_rows(df, "m0", TEST, s, QUIET)
    assert int(rows["threshold_pct"].max()) == s["plot_max_pct"]
    assert int(rows["threshold_pct"].min()) == s["plot_min_pct"]


def test_nb_arm_rows_is_clipped_to_the_configured_plot_window(cfg):
    s = _nb_settings(cfg)
    rows = mf._nb_arm_rows(_nb_frame(cfg, TEST), "m2_frontal", TEST, s, QUIET)
    assert list(rows["threshold_pct"]) == list(range(s["plot_min_pct"], s["plot_max_pct"] + 1))
    assert len(rows) < int(s["thresholds"].size), "the CSV carries more than the figure draws"


def test_nb_arm_rows_drops_rows_the_convergence_gate_suppressed(cfg):
    s = _nb_settings(cfg)
    df = _nb_frame(cfg, TEST, suppress=("m1",))
    assert mf._nb_arm_rows(df, "m1", TEST, s, QUIET).empty
    assert not mf._nb_arm_rows(df, "m0", TEST, s, QUIET).empty


def test_nb_flag_reads_a_boolean_column_however_pandas_typed_it():
    """One NaN makes the column object dtype, and Series.astype(bool) is then True for the
    string "False" - which would draw exactly the rows the flag exists to withhold."""
    assert list(mf.nb_flag(pd.Series([True, False]))) == [True, False]
    assert list(mf.nb_flag(pd.Series(["True", "False"]))) == [True, False]
    assert list(mf.nb_flag(pd.Series(["False", None], dtype=object))) == [False, False]
    assert list(mf.nb_flag(pd.Series([1, 0]))) == [True, False]
    assert list(mf.nb_flag(pd.Series([1.0, np.nan]))) == [True, False]


def test_nb_arm_rows_names_the_producer_for_an_arm_the_table_does_not_carry(cfg):
    s = _nb_settings(cfg)
    df = _nb_frame(cfg, TEST)
    with pytest.raises(KeyError, match="src.eval_models"):
        mf._nb_arm_rows(df, "m3_image", TEST, s, QUIET)


# --------------------------------------------------------------------------- #
# 8f. The prevalence is INVERTED from the table, never read from a literal     #
# --------------------------------------------------------------------------- #
def test_nb_prevalence_inverts_the_treat_all_formula_by_hand():
    """NB_all = F - (1 - F) w with w = p / (1 - p); at F = 0.25 and p = 0.10, w = 1/9 and
    NB_all = 0.25 - 0.75/9 = 0.1666..., and the inversion has to return 0.25 exactly."""
    p = np.array([0.10, 0.20, 0.30])
    w = p / (1.0 - p)
    f = 0.25
    rows = pd.DataFrame({"threshold": p, "nb_treat_all_same_set": f - (1.0 - f) * w})
    assert mf.nb_prevalence_from_treat_all(rows) == pytest.approx(f, abs=1e-12)


def test_nb_prevalence_tolerance_is_the_half_ulp_of_the_printed_precision():
    """NB_PREVALENCE_TOL is a rounding rule, not an error budget.

    Both the caption and the annotation drawn on the image print four decimals, so the
    assertion's whole claim is "these two print the same string". That is exactly a half unit
    in the last place, which is why the sealed split's 2.5e-5 gap is the expected state and
    not a near miss: tightening the constant would fire on figures a reader cannot tell
    apart, loosening it would let the two disagree on the page.
    """
    anchor = 0.2004
    for delta in (0.0, 1e-5, 2.5e-5, 4.9e-5):
        for signed in (delta, -delta):
            got = anchor + signed
            assert (abs(got - anchor) < mf.NB_PREVALENCE_TOL) == (
                f"{got:.4f}" == f"{anchor:.4f}")
    assert f"{anchor + 6e-5:.4f}" != f"{anchor:.4f}"
    assert abs(6e-5) > mf.NB_PREVALENCE_TOL


@pytest.mark.parametrize("split", [VAL, TEST])
def test_the_drawn_prevalence_and_the_captions_print_the_same_string(cfg, split):
    """The invariant the tolerance stands in for, checked on the artefact itself."""
    _skip_unless(em.split_path(cfg, "net_benefit_csv", split), f"{split}_net_benefit.csv")
    s = _nb_settings(cfg)
    df = mf._net_benefit_table(cfg, split)
    drawn = mf.nb_prevalence_from_treat_all(mf.nb_treat_all_window(df, s))
    stated = float(mf.NB_ANCHORS[split]["prevalence"])
    assert f"{drawn:.4f}" == f"{stated:.4f}"
    assert f"{stated:.4f}" in mf.supplement_figures(cfg, split)["figureS3"]["caption"]


def test_nb_prevalence_fires_when_the_treat_all_column_is_inconsistent():
    """Treat-all flags everyone at every threshold, so ONE Kaplan-Meier estimate serves the
    whole column. Two different ones is a bug in the writer, not a wobble."""
    p = np.array([0.10, 0.20, 0.30])
    w = p / (1.0 - p)
    f = np.array([0.20, 0.25, 0.30])
    rows = pd.DataFrame({"threshold": p, "nb_treat_all_same_set": f - (1.0 - f) * w})
    with pytest.raises(AssertionError, match="spread in the cumulative incidence"):
        mf.nb_prevalence_from_treat_all(rows)


# --------------------------------------------------------------------------- #
# 8g. The render itself                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", [VAL, TEST])
def test_render_figure4_writes_the_registered_file_at_the_column_width(nb_cfg, tmp_path,
                                                                       split):
    from PIL import Image

    edited, write = nb_cfg
    write(split, _nb_frame(edited, split))
    out = mf.render_decision_curve(edited, tmp_path, split)
    assert out.name == "figureS3_decision_curve.png" and out.exists()
    with Image.open(out) as im:
        want = round(float(edited["manuscript"]["single_column_in"])
                     * int(edited["manuscript"]["figure_dpi"]))
        assert abs(im.size[0] - want) <= mf.WIDTH_LOCK_TOL_PX
        assert im.mode == "RGB", "an alpha channel is a routine cause of upload failure"


def test_render_figure4_reads_the_csv_and_nothing_else(nb_cfg, tmp_path, monkeypatch):
    """The whole reason this renderer lives here and the estimator lives in eval_models.

    Hazards, the clinical feature table and the frozen Cox replays are all booby-trapped;
    a render that still succeeds touched none of them. The two small CSVs it may read -
    ``{split}_net_benefit.csv``, which it draws, and ``{split}_convergence.csv``, which
    decides whether there is anything to draw - are deliberately not among them.
    """
    edited, write = nb_cfg
    write(TEST, _nb_frame(edited, TEST))

    def forbidden(*a, **k):
        raise AssertionError("figure 4 must read only {split}_net_benefit.csv")

    for name in ("load_hazards", "_split_frame", "_arm_risks", "cox_replay_risk",
                 "m0_replay_risk", "_import_model_clinical", "_metrics_table"):
        monkeypatch.setattr(mf, name, forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    assert mf.render_decision_curve(edited, tmp_path, TEST).exists()


def test_render_figure4_fires_when_the_prevalence_moves(nb_cfg, tmp_path):
    """The annotation drawn on the image and the number in the caption are one number.

    It fires BEFORE the figure is written, so a drift never leaves a wrong PNG on disk for
    someone to embed; the empty output directory is part of the contract.
    """
    edited, write = nb_cfg
    frame = _nb_frame(edited, TEST)
    p = frame["threshold"].to_numpy(dtype=float)
    w = p / (1.0 - p)
    moved = float(mf.NB_ANCHORS[TEST]["prevalence"]) + 0.02
    ref = frame["arm"].astype(str) == _nb_settings(edited)["reference"]
    frame.loc[ref, "nb_treat_all_same_set"] = (moved - (1.0 - moved) * w)[ref.to_numpy()]
    write(TEST, frame)
    out = tmp_path / "figs"
    out.mkdir()
    with pytest.raises(AssertionError, match="would draw treat-all crossing zero at"):
        mf.render_decision_curve(edited, out, TEST)
    assert list(out.iterdir()) == []


def test_render_figure4_fires_when_a_stated_denominator_moves(nb_cfg, tmp_path):
    edited, write = nb_cfg
    frame = _nb_frame(edited, TEST)
    prot = next(a for a in _nb_settings(edited)["arms"]
                if a != _nb_settings(edited)["reference"])
    frame.loc[frame["arm"].astype(str) == prot, "n_scored"] = 700
    write(TEST, frame)
    with pytest.raises(AssertionError, match="but the caption states"):
        mf.render_decision_curve(edited, tmp_path, TEST)


def test_render_figure4_fires_when_one_arm_changes_denominator_mid_curve(nb_cfg, tmp_path):
    edited, write = nb_cfg
    frame = _nb_frame(edited, TEST)
    frame.loc[frame.index[-1], "n_scored"] = 1
    write(TEST, frame)
    with pytest.raises(AssertionError, match="one arm has one denominator"):
        mf.render_decision_curve(edited, tmp_path, TEST)


def test_render_figure4_fires_when_the_table_is_at_another_horizon(nb_cfg, tmp_path):
    edited, write = nb_cfg
    frame = _nb_frame(edited, TEST)
    frame["horizon_days"] = 730
    write(TEST, frame)
    with pytest.raises(AssertionError, match="estimated at horizon"):
        mf.render_decision_curve(edited, tmp_path, TEST)


def test_render_figure4_names_the_producer_for_a_missing_arm(nb_cfg, tmp_path):
    edited, write = nb_cfg
    frame = _nb_frame(edited, TEST)
    write(TEST, frame[frame["arm"].astype(str) != "m4_fusion"])
    with pytest.raises(KeyError, match="src.eval_models --split test"):
        mf.render_decision_curve(edited, tmp_path, TEST)


def test_render_figure4_survives_a_truncated_curve(nb_cfg, tmp_path):
    """On validation m0 goes sparse at 0.27, inside the drawn range; the render must draw
    the shortened curve rather than fall over or silently drop the arm."""
    edited, write = nb_cfg
    write(VAL, _nb_frame(edited, VAL, sparse_from_pct={"m0": 27}))
    assert mf.render_decision_curve(edited, tmp_path, VAL).exists()


# --------------------------------------------------------------------------- #
# 8g-ii. The validation split, whose protagonist the convergence gate blanks   #
#                                                                              #
# outputs/tables/val_convergence.csv records m2_frontal, m3_image, m4_frontal  #
# and m4_fusion as severe_overfit, which suppresses on VALIDATION ONLY. So the #
# day val_net_benefit.csv exists, every image arm's rows arrive suppressed and #
# the decision curve's protagonist has nothing to draw. These tests drive that #
# state from a synthetic table, so it is covered BEFORE the file exists rather #
# than after, when it would turn a skip into a red suite.                      #
# --------------------------------------------------------------------------- #
def _val_suppressed_arms(cfg) -> tuple[str, ...]:
    """The configured decision-curve arms the REAL val convergence table suppresses.

    Read from the artefact rather than typed, so the fixture below rehearses the state the
    pipeline will actually produce and not a guess about it.
    """
    s = _nb_settings(cfg)
    path = em.split_path(cfg, "convergence_csv", VAL)
    if not Path(path).exists():                    # fall back to today's recorded pattern
        return tuple(a for a in s["arms"] if a in ("m2_frontal", "m4_fusion"))
    conv = pd.read_csv(path)
    bad = set(conv.loc[conv["status"].isin({em.STATUS_NO_CONVERGE, em.STATUS_OVERFIT}), "arm"])
    return tuple(a for a in s["arms"] if a in bad)


def test_the_val_convergence_table_really_does_suppress_the_protagonist(cfg):
    """The premise of the three tests below, checked against the artefact."""
    _skip_unless(em.split_path(cfg, "convergence_csv", VAL), "val_convergence.csv")
    s = _nb_settings(cfg)
    prot = next(a for a in s["arms"] if a != s["reference"])
    assert prot in _val_suppressed_arms(cfg), (
        "the validation convergence gate no longer blanks the decision curve's protagonist, "
        "so render_figure4's skip branch is describing a state that no longer exists")
    assert s["reference"] not in _val_suppressed_arms(cfg), (
        "the reference arm is suppressed too; panel A has no comparator and the skip "
        "branch's reasoning has to be revisited")


def test_nb_suppressed_arms_reports_the_estimators_own_reason(cfg):
    s = _nb_settings(cfg)
    supp = _val_suppressed_arms(cfg)
    df = _nb_frame(cfg, VAL, suppress=supp)
    df.loc[df["arm"].isin(supp), "note"] = "SUPPRESSED -- m2_frontal severe_overfit: because"
    got = mf.nb_suppressed_arms(df, s)
    assert set(got) == set(supp)
    assert all("severe_overfit" in v for v in got.values())
    # an arm with even one live row is not "suppressed": the row filter handles that
    partial = _nb_frame(cfg, VAL)
    partial.loc[partial.index[0], "suppressed"] = True
    assert mf.nb_suppressed_arms(partial, s) == {}


def test_render_figure4_declines_the_split_whose_protagonist_is_suppressed(cfg, nb_cfg,
                                                                           tmp_path, caplog):
    """The legitimate state. No PNG, no exception, and the reason in the log.

    Before this branch existed the render raised out of _nb_panel_b's assertion, which under
    render_all's dict comprehension took figures 1 to 3 down with it.
    """
    edited, write = nb_cfg
    supp = _val_suppressed_arms(cfg)
    write(VAL, _nb_frame(edited, VAL, suppress=supp))
    out = tmp_path / "figs"
    out.mkdir()
    with caplog.at_level(logging.WARNING, logger="manuscript_figures"):
        assert mf.render_decision_curve(edited, out, VAL) is None
    assert list(out.iterdir()) == [], "a declined figure must not leave a PNG to be embedded"
    text = caplog.text
    assert "is NOT drawn for the validation split" in text
    assert "the gate working, not a render failure" in text
    assert "val_net_benefit.csv" in text
    # the sealed split is untouched by the branch: severe_overfit stops suppressing there
    write(TEST, _nb_frame(edited, TEST))
    assert mf.render_decision_curve(edited, tmp_path, TEST).exists()


def test_render_all_drops_a_declined_figure_and_still_renders_the_rest(cfg, nb_cfg, tmp_path,
                                                                       monkeypatch):
    """One figure with nothing honest to show must not cost the others.

    The decline lives in the SUPPLEMENTARY set now, because that is where the decision curve
    went; the mechanism is the shared one in ``_render_set`` and this exercises it there.
    """
    edited, write = nb_cfg
    write(VAL, _nb_frame(edited, VAL, suppress=_val_suppressed_arms(cfg)))
    drawn: list[str] = []

    def _stub(key):
        def _r(c, o, s):
            drawn.append(key)
            p = o / f"{key}.png"
            p.write_bytes(b"")
            return p
        return _r

    others = [k for k in mf.SUPPLEMENT_KEYS if k != "figureS3"]
    monkeypatch.setattr(mf, "SUPPLEMENT_RENDERERS",
                        {**mf.SUPPLEMENT_RENDERERS, **{k: _stub(k) for k in others}})
    written = mf.render_supplement(edited, out_dir=tmp_path, split=VAL)
    assert drawn == others
    assert set(written) == set(others)
    assert "figureS3" not in written
    assert not (tmp_path / mf.SUPPLEMENT_DIRNAME / "figureS3_decision_curve.png").exists()


def test_panel_b_still_fires_when_an_unsuppressed_arm_has_no_drawable_row(nb_cfg, tmp_path):
    """The assertion is not deleted, only narrowed: an arm that survives the gate and still
    draws nothing is a real defect and must still stop the render."""
    edited, write = nb_cfg
    s = _nb_settings(edited)
    prot = next(a for a in s["arms"] if a != s["reference"])
    write(TEST, _nb_frame(edited, TEST, sparse_from_pct={prot: s["plot_min_pct"]}))
    with pytest.raises(AssertionError, match="is NOT suppressed"):
        mf.render_decision_curve(edited, tmp_path, TEST)


# --------------------------------------------------------------------------- #
# 8g-iii. WHICH QUESTION IS ASKED FIRST                                        #
#                                                                              #
# The decline is a property of the SPLIT and the convergence gate, so it is     #
# settled before the net-benefit table is required. It used to be read off      #
# that table's own suppressed column, which meant requiring a file that has     #
# never been produced in order to conclude that the figure it holds is not      #
# drawn - and a whole validation render died on it. These tests pin the order   #
# from both sides: a declined figure must not need the table, and a figure that #
# should draw must still fail loudly when the table is missing.                 #
# --------------------------------------------------------------------------- #
def test_render_all_completes_on_validation_with_no_net_benefit_table(cfg, tmp_path,
                                                                      monkeypatch, caplog):
    """THE REGRESSION. A whole-module validation render must not die on a table it declines.

    ``outputs/tables/val_net_benefit.csv`` has never been produced and does not need to be:
    the validation convergence gate disqualifies the decision curve's protagonist, so that
    figure has nothing honest to draw there whatever the file would have said. Requiring it
    first turned a legitimately declined figure into a FileNotFoundError that took the whole
    render down with it. The decision curve is supplementary figure S3 at v6, so the render
    exercised here is the supplementary one; the mechanism is the shared ``_render_set``.

    The other supplementary figures are stubbed deliberately. This is about the ORDER of the
    decision curve's two questions, and the real S4 writes a table into the repository.
    """
    if em.split_path(cfg, "net_benefit_csv", VAL).exists():
        pytest.skip("val_net_benefit.csv now exists; this regression needs it absent")
    _skip_unless(em.split_path(cfg, "convergence_csv", VAL), "val_convergence.csv")

    drawn: list[str] = []

    def _stub(key):
        def _r(c, o, s):
            drawn.append(key)
            p = o / f"{key}.png"
            p.write_bytes(b"")
            return p
        return _r

    others = [k for k in mf.SUPPLEMENT_KEYS if k != "figureS3"]
    monkeypatch.setattr(mf, "SUPPLEMENT_RENDERERS",
                        {**mf.SUPPLEMENT_RENDERERS, **{k: _stub(k) for k in others}})
    with caplog.at_level(logging.WARNING, logger="manuscript_figures"):
        written = mf.render_supplement(cfg, out_dir=tmp_path, split=VAL)
    assert drawn == others
    assert set(written) == set(others)
    assert not (tmp_path / mf.SUPPLEMENT_DIRNAME / "figureS3_decision_curve.png").exists()
    assert "is NOT drawn for the validation split" in caplog.text
    assert "val_convergence.csv" in caplog.text


def test_the_decline_reason_is_the_gates_verdict_not_the_state_of_the_disk(cfg):
    """The reason names an arm, a verdict and the table that recorded it - never a filename
    that happens to be absent. Absence is not a reason; it is what the missing-artefact error
    exists to report."""
    _skip_unless(em.split_path(cfg, "convergence_csv", VAL), "val_convergence.csv")
    s = _nb_settings(cfg)
    prot = next(a for a in s["arms"] if a != s["reference"])
    why = mf.decision_curve_decline_reason(cfg, VAL)
    assert why, "the validation gate disqualifies the protagonist, so there is a reason"
    assert prot in why and em.STATUS_OVERFIT in why and "val_convergence.csv" in why
    assert "net_benefit" not in why and "not found" not in why
    assert mf.nb_convergence_status(cfg, VAL, prot) == em.STATUS_OVERFIT


def test_a_missing_table_still_raises_where_the_protagonist_is_not_disqualified(nb_cfg,
                                                                                tmp_path):
    """Declining requires a positive reason, so an absent table is a failure and not a skip."""
    edited, _write = nb_cfg
    s = _nb_settings(edited)
    for split in (VAL, TEST):
        _write_convergence(edited, split, s["arms"])          # every arm ok on both splits
        assert mf.decision_curve_decline_reason(edited, split) == ""
        assert not em.split_path(edited, "net_benefit_csv", split).exists()
        with pytest.raises(FileNotFoundError, match=f"src.eval_models --split {split}"):
            mf.render_decision_curve(edited, tmp_path, split)


def test_a_missing_convergence_table_is_not_a_verdict_either(nb_cfg):
    """No gate, no positive reason, so no decline: the render goes on to need its artefact."""
    edited, _write = nb_cfg
    s = _nb_settings(edited)
    prot = next(a for a in s["arms"] if a != s["reference"])
    assert not em.split_path(edited, "convergence_csv", VAL).exists()
    assert mf.nb_convergence_status(edited, VAL, prot) == em.STATUS_OK
    assert mf.decision_curve_decline_reason(edited, VAL) == ""


def test_the_table_and_the_gate_cannot_disagree_about_the_protagonist(nb_cfg, tmp_path):
    """Two readings of one gate. Either direction of disagreement means one of them was
    computed against a different split, and that has to stop the render rather than be
    resolved by whichever was consulted first."""
    edited, _write = nb_cfg
    s = _nb_settings(edited)
    prot = next(a for a in s["arms"] if a != s["reference"])
    p = em.split_path(edited, "net_benefit_csv", VAL)
    # the gate disqualifies the protagonist; the table kept every one of its rows
    _nb_frame(edited, VAL).to_csv(p, index=False)
    _write_convergence(edited, VAL, s["arms"], suppressed=(prot,))
    with pytest.raises(AssertionError, match="two readings of one gate"):
        mf.render_decision_curve(edited, tmp_path, VAL)
    # and the other way round: the table blanked it while the gate says it is interpretable
    _nb_frame(edited, VAL, suppress=(prot,)).to_csv(p, index=False)
    _write_convergence(edited, VAL, s["arms"])
    with pytest.raises(AssertionError, match="two readings of one gate"):
        mf.render_decision_curve(edited, tmp_path, VAL)


# =========================================================================== #
# 9. ONE READ RULE, SHARED WITH src/make_manuscript.py                        #
#                                                                              #
# render_all gates the sealed read only when the RENDERED split is the sealed  #
# one. _split_frame's predicate used to collapse to False on validation as     #
# soon as model_eval.forbid_test_split was turned off, which would have        #
# materialised all 3,709 rows with that gate never called - and the row-count  #
# assertion followed the same flag, so it stayed silent on exactly that path.  #
# =========================================================================== #
def test_readable_splits_is_the_whole_rule_and_only_test_gets_the_sealed_rows():
    mm = pytest.importorskip("src.make_manuscript")
    assert mf.readable_splits(TEST) == mf.ALL_SPLITS
    assert TEST not in mf.readable_splits(VAL)
    for split in (VAL, TEST):
        assert mf.readable_splits(split) == mm.readable_splits(split), (
            "the two render modules describe one split of one study in one document set")
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.readable_splits("train")


def test_the_two_render_modules_disqualify_the_same_statuses():
    """One convergence rule, mirrored rather than imported, pinned so it cannot drift.

    src.eval_models.suppress_unfit_contrasts APPLIES the rule when it writes the suppressed
    column; make_manuscript reads it to decide whether a figure belongs in the document, and
    manuscript_figures reads it to decide whether that figure may be drawn at all. Two
    answers to one question about one split is a contradiction, not a configuration.
    """
    mm = pytest.importorskip("src.make_manuscript")
    assert set(mf.DISQUALIFYING) == set(mm.DISQUALIFYING) == set(mf.SPLITS)
    for split in (VAL, TEST):
        assert set(mf.disqualifying_statuses(split)) == set(mm.disqualifying_statuses(split))
    # the asymmetry itself, stated once so the mirror above cannot be trivially satisfied
    assert em.STATUS_NO_CONVERGE in mf.disqualifying_statuses(VAL)
    assert em.STATUS_NO_CONVERGE in mf.disqualifying_statuses(TEST)
    assert em.STATUS_OVERFIT in mf.disqualifying_statuses(VAL)
    assert em.STATUS_OVERFIT not in mf.disqualifying_statuses(TEST)
    assert em.STATUS_OK not in mf.disqualifying_statuses(VAL)
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.disqualifying_statuses("train")


def test_both_render_modules_decline_figure4_on_the_same_splits(cfg):
    """The verdict, not just the status table: a document that omits figure 4 while the
    renderer draws it (or the reverse) is the drift the mirror exists to prevent."""
    mm = pytest.importorskip("src.make_manuscript")
    for split in (VAL, TEST):
        path = em.split_path(cfg, "convergence_csv", split)
        conv = pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
        inp = mm.Inputs(cfg=cfg, dry_run=False, split=split, convergence=conv)
        assert bool(mm.decision_curve_decline_reason(inp)) == bool(
            mf.decision_curve_decline_reason(cfg, split)), split
    if em.split_path(cfg, "convergence_csv", VAL).exists():
        assert mf.decision_curve_decline_reason(cfg, VAL) != ""
    if em.split_path(cfg, "convergence_csv", TEST).exists():
        assert mf.decision_curve_decline_reason(cfg, TEST) == ""


def test_forbid_test_rows_is_the_same_expression_in_both_modules(cfg):
    mm = pytest.importorskip("src.make_manuscript")
    for split in (VAL, TEST):
        assert mf.forbid_test_rows(cfg, split) == mm.forbid_test_rows(cfg, split)
    assert mf.forbid_test_rows(cfg, VAL) is True
    assert mf.forbid_test_rows(cfg, TEST) is False


def test_forbid_test_rows_is_only_a_hint_and_readable_splits_is_the_rule(cfg):
    """Turning the config flag off must not widen what a validation render may hold.

    The previous version built ``edited`` and never used it: it asserted
    ``readable_splits(VAL) == readable_splits(cfg and VAL)``, and ``cfg and VAL`` is just
    ``VAL``, so it compared the function with itself. The claim is STRUCTURAL, and that is
    what is pinned: no configuration can reach the rule, because the rule takes only a
    split. A ``readable_splits(cfg, split)`` refactor is exactly the regression this must
    catch, and it now fails on the signature rather than passing on a tautology.
    """
    import inspect as inspect_module

    assert list(inspect_module.signature(mf.readable_splits).parameters) == ["split"], (
        "readable_splits takes something other than a split, so a config value can now "
        "widen what a render may materialise")
    src = inspect_module.getsource(mf.readable_splits)
    assert "forbid_test_split" not in src and "cfg" not in src

    edited = Config(cfg)
    edited["model_eval"] = {**dict(cfg["model_eval"]), "forbid_test_split": False}
    # The HINT moves with the flag...
    assert mf.forbid_test_rows(cfg, VAL) is True
    assert mf.forbid_test_rows(edited, VAL) is False
    # ...and the RULE does not, on either split.
    assert TEST not in mf.readable_splits(VAL)
    assert TEST in mf.readable_splits(TEST)
    # so the two are not the same predicate: on the sealed split the hint is already False
    # and the rule still admits the sealed rows, and on validation the hint is now False
    # too and the rule still does not.
    assert mf.forbid_test_rows(edited, TEST) is False
    assert mf.readable_splits(VAL) != mf.readable_splits(TEST)
    # the mirror in the other render module answers identically, flag off included
    mm = pytest.importorskip("src.make_manuscript")
    for split in (VAL, TEST):
        assert mf.readable_splits(split) == mm.readable_splits(split)
        assert mf.forbid_test_rows(edited, split) == mm.forbid_test_rows(edited, split)


def _spy_mc(seen: dict, frame: pd.DataFrame):
    class _SpyMC:
        @staticmethod
        def load_development_frame(path, *, forbid_test=True):
            seen["forbid_test"] = forbid_test
            out = frame if not forbid_test else frame[frame["split"] != TEST]
            seen["materialised"] = set(out["split"].unique())
            return out.reset_index(drop=True)
    return _SpyMC


def _landmark_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    split = np.array(["train"] * mf.SPLIT_ANCHORS["train"]["n"]
                     + [VAL] * mf.SPLIT_ANCHORS[VAL]["n"]
                     + [TEST] * mf.SPLIT_ANCHORS[TEST]["n"])
    ev = np.zeros(split.size, dtype=int)
    for s in ("train", VAL, TEST):
        ev[rng.choice(np.flatnonzero(split == s), mf.SPLIT_ANCHORS[s]["events"],
                      replace=False)] = 1
    return pd.DataFrame({"split": split, "event_indicator": ev,
                         "empi_anon": [f"p{i:06d}" for i in range(split.size)],
                         "time_from_landmark": rng.integers(90, 1826, split.size)})


def test_split_frame_reads_with_the_shared_predicate(cfg):
    for split in (VAL, TEST):
        seen: dict = {}
        fr = mf._split_frame(cfg, _spy_mc(seen, _landmark_frame()), split, QUIET)
        assert seen["forbid_test"] == mf.forbid_test_rows(cfg, split)
        assert seen["materialised"] == set(mf.readable_splits(split))
        assert set(fr["split"].unique()) == {split}
        assert len(fr) == mf.SPLIT_ANCHORS[split]["n"]


def test_split_frame_gates_the_sealed_read_before_the_flag_can_leak_a_row(cfg,
                                                                          monkeypatch):
    """With forbid_test_split off, a VALIDATION render would materialise the sealed rows.

    The invariant is "a sealed row is in memory implies the gate passed", so the gate runs
    first, and the frame is then cut back to readable_splits regardless.
    """
    edited = Config(cfg)
    edited["model_eval"] = {**dict(cfg["model_eval"]), "forbid_test_split": False}
    calls: list[str] = []
    monkeypatch.setattr(mf, "assert_sealed_read_is_recorded",
                        lambda c: calls.append("checked") or "deadbeef")
    seen: dict = {}
    fr = mf._split_frame(edited, _spy_mc(seen, _landmark_frame()), VAL, QUIET)
    assert calls == ["checked"], "the sealed rows were materialised with no gate"
    assert seen["forbid_test"] is False
    assert seen["materialised"] == set(mf.ALL_SPLITS), "the leak this test describes"
    assert set(fr["split"].unique()) == {VAL}
    assert len(fr) == mf.SPLIT_ANCHORS[VAL]["n"]


def test_split_frame_row_count_assertion_does_not_follow_the_config_flag(cfg, monkeypatch):
    """The assertion used to be n_expected = DEV if forbid else LANDMARK, so on the leaking
    path it expected the 3,709 rows it had just been handed and passed."""
    edited = Config(cfg)
    edited["model_eval"] = {**dict(cfg["model_eval"]), "forbid_test_split": False}
    monkeypatch.setattr(mf, "assert_sealed_read_is_recorded", lambda c: "deadbeef")
    short = _landmark_frame()
    short = short[~((short["split"] == VAL) & (short.index % 37 == 0))]
    with pytest.raises(AssertionError, match="development frame has"):
        mf._split_frame(edited, _spy_mc({}, short), VAL, QUIET)


def test_split_frame_gates_the_sealed_read_however_it_is_reached(cfg, monkeypatch):
    """REVERSED on 2026-07-30, and this test used to pin the hole rather than the rule.

    It previously asserted that ``_split_frame`` does NOT call the gate on the sealed split,
    on the reasoning that ``render_all`` had already called it. That is true of
    ``render_all`` and of nothing else: ``render_figure2`` and ``render_figure3`` are
    ordinary functions, and calling either directly with the sealed split - which several
    tests in this file do deliberately, and which nothing prevented a caller from doing -
    materialised all 741 sealed rows with the gate never invoked. The invariant the design
    rests on is "a sealed row is in memory only if the gate passed", so it is enforced where
    the rows appear rather than at whichever entry point happened to be used.

    The cost is that a ``render_all`` pass now calls the gate more than once. That is two
    small JSON reads of ``test_scoring.json`` and ``train_arms.json``, with no state and no
    side effect, and the alternative - dropping the entry-point call - would move the
    failure of a moved training contract from before the first pixel to somewhere in the
    middle of figure 2. Defence in depth is the deliberate choice; see
    test_a_sealed_render_holds_the_sealed_rows_and_the_gate_ran_first, which pins the same
    rule from the manuscript module's side.
    """
    calls: list[str] = []
    monkeypatch.setattr(mf, "assert_sealed_read_is_recorded",
                        lambda c: calls.append("checked") or "deadbeef")
    mf._split_frame(cfg, _spy_mc({}, _landmark_frame()), TEST, QUIET)
    assert calls == ["checked"], "a sealed row reached memory with the gate uncalled"
    # ...and a validation render still needs no sealed read to have happened at all.
    calls.clear()
    mf._split_frame(cfg, _spy_mc({}, _landmark_frame()), VAL, QUIET)
    assert calls == []


# --------------------------------------------------------------------------- #
# 8h. Against the REAL net-benefit table, wherever one exists                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", [VAL, TEST])
def test_render_figure4_from_the_real_artefact(cfg, tmp_path, split):
    """The synthetic frames above exercise the branches; this proves the file the pipeline
    actually writes goes through the same reader, schema check and anchor assertions.

    On a split whose protagonist the convergence gate suppressed the honest result is no
    figure, so the assertion is on the DECISION and not on a file: the val leg of this test
    skips only while val_net_benefit.csv is absent, and must not turn red the day it lands.
    """
    _skip_unless(em.split_path(cfg, "net_benefit_csv", split), f"{split}_net_benefit.csv")
    out = mf.render_decision_curve(cfg, tmp_path, split)
    s = _nb_settings(cfg)
    prot = next(a for a in s["arms"] if a != s["reference"])
    df = mf._net_benefit_table(cfg, split)
    if prot in mf.nb_suppressed_arms(df, s):
        assert out is None and list(tmp_path.iterdir()) == []
        return
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("split", [VAL, TEST])
def test_the_real_net_benefit_table_agrees_with_the_caption_anchors(cfg, split):
    _skip_unless(em.split_path(cfg, "net_benefit_csv", split), f"{split}_net_benefit.csv")
    s = _nb_settings(cfg)
    df = mf._net_benefit_table(cfg, split)
    prot = next(a for a in s["arms"] if a != s["reference"])
    scored = {a: int(df.loc[df["arm"].astype(str) == a, "n_scored"].iloc[0])
              for a in s["arms"]}
    assert scored[s["reference"]] == mf.SPLIT_ANCHORS[split]["n"]
    assert scored[prot] == int(mf.NB_ANCHORS[split]["arm_n"])
    assert min(scored.values()) == mf.SPLIT_ANCHORS[split]["panel_b_n"]
    ref_rows = df[df["arm"].astype(str) == s["reference"]]
    assert mf.nb_prevalence_from_treat_all(ref_rows) == pytest.approx(
        float(mf.NB_ANCHORS[split]["prevalence"]), abs=mf.NB_PREVALENCE_TOL)


def test_the_marginal_bands_really_would_have_read_as_no_difference(cfg):
    """The caption's reason for keeping the intervals off Panel A, checked on the artefact.

    The claim is that a marginal interval discards the pairing: somewhere in the drawn
    range the protagonist's own interval overlaps the reference arm's, which reads as no
    difference, while the PAIRED difference at that same threshold excludes zero. If that
    ever stops being true of the data, the sentence stops being an argument.
    """
    split = mf.SEALED_SPLIT
    _skip_unless(em.split_path(cfg, "net_benefit_csv", split), f"{split}_net_benefit.csv")
    s = _nb_settings(cfg)
    df = mf._net_benefit_table(cfg, split)
    prot = next(a for a in s["arms"] if a != s["reference"])
    a = mf._nb_arm_rows(df, prot, split, s, QUIET).set_index("threshold_pct")
    b = mf._nb_arm_rows(df, s["reference"], split, s, QUIET).set_index("threshold_pct")
    hits = []
    for pct in a.index.intersection(b.index):
        overlap = (a.loc[pct, "net_benefit_lo"] <= b.loc[pct, "net_benefit_hi"]
                   and b.loc[pct, "net_benefit_lo"] <= a.loc[pct, "net_benefit_hi"])
        clear = a.loc[pct, "diff_vs_reference_lo"] > 0.0
        if overlap and clear:
            hits.append(int(pct))
    assert hits, (
        f"nowhere in {s['plot_min_pct']}-{s['plot_max_pct']}% do {prot}'s and "
        f"{s['reference']}'s marginal intervals overlap while the paired difference "
        "excludes zero, so FIG4_WHY_NO_BANDS no longer describes this data")


# =========================================================================== #
# 10. THE RISK-TERTILE TABLE figure 3 EMITS                                   #
#                                                                             #
# The 5-year incidence in the lowest against the highest tertile is the one   #
# number in this paper a surgeon acts on, so it has to be quotable in prose   #
# and not only readable off a picture. render_figure3 therefore writes the    #
# summary it drew. Two things are pinned here:                                #
#                                                                             #
#  * the table and the image come off ONE Kaplan-Meier fit, so a sentence     #
#    cannot disagree with the figure printed beside it. That is checked as a  #
#    property (the reported value is a point on the drawn step curve, and it  #
#    equals the house km_risk helper) rather than asserted in a docstring;    #
#  * the emitted values themselves, transcribed as literals from a real       #
#    render. If the sealed-split gradient ever moves, the manuscript's        #
#    headline sentence has to move with it, and that must fail loudly.        #
# =========================================================================== #
# The 5-year cumulative incidence by tertile, per split, MEASURED off the real
# artefacts and written down here so a drift fails. Transcribed, not imported.
FROZEN_TERTILE_INCIDENCE = {
    VAL: (0.071246, 0.171248, 0.366220),
    TEST: (0.035912, 0.198303, 0.406247),          # the manuscript's 3.6% / 19.8% / 40.6%
}


@pytest.fixture
def tert_cfg(cfg, tmp_path):
    """A config whose risk_tertiles_csv points into tmp_path.

    Keeps the ``val_`` prefix so ``split_path``'s single rewrite still yields
    ``test_risk_tertiles.csv``, and keeps a figure render in these tests from writing into
    the repository's outputs/.
    """
    edited = Config(cfg)
    edited["model_eval"] = {**dict(cfg["model_eval"]),
                            "risk_tertiles_csv": str(tmp_path / "val_risk_tertiles.csv")}
    return edited


def _tertile_inputs(n: int = 300, seed: int = 7):
    """A cohort with a genuine risk gradient, so the three tertiles differ."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.02, 0.75, n)
    # Higher predicted risk -> shorter time and likelier event, so the curves separate.
    t = rng.uniform(120.0, 2400.0, n) * (1.0 - 0.6 * pred)
    e = (rng.uniform(size=n) < pred).astype(int)
    return pred, np.round(t, 0), e


# --------------------------------------------------------------------------- #
# 10a. Where the table goes                                                     #
# --------------------------------------------------------------------------- #
def test_risk_tertiles_path_uses_the_config_key_when_one_is_declared(tert_cfg, tmp_path):
    assert mf.risk_tertiles_path(tert_cfg, VAL) == tmp_path / "val_risk_tertiles.csv"
    assert mf.risk_tertiles_path(tert_cfg, TEST) == tmp_path / "test_risk_tertiles.csv"


@pytest.mark.parametrize("configured", ["outputs/tables/val_risk_tertiles.csv",
                                        "outputs/tables/val_odd_val_name.csv"])
def test_risk_tertiles_path_is_split_paths_rewrite_and_not_a_second_copy(cfg, configured):
    """Whatever split_path does with a declared key, this must do - including the bound of
    ONE substitution, which a hand-rolled ``replace("val_", "test_")`` here would lose."""
    edited = Config(cfg)
    edited["model_eval"] = {**dict(cfg["model_eval"]), "risk_tertiles_csv": configured}
    for split in (VAL, TEST):
        assert mf.risk_tertiles_path(edited, split) == \
            em.split_path(edited, "risk_tertiles_csv", split)


def test_risk_tertiles_path_falls_back_beside_the_net_benefit_table(cfg):
    """With no config key the table still lands beside net_benefit_csv, split-rewritten."""
    assert mf.RISK_TERTILES_CSV_KEY not in dict(cfg["model_eval"]) or pytest.skip(
        "config now declares the key, so the fallback branch is unreachable from here")
    for split, stem in ((VAL, "val_risk_tertiles.csv"), (TEST, "test_risk_tertiles.csv")):
        p = mf.risk_tertiles_path(cfg, split)
        assert p.name == stem
        assert p.parent == em.split_path(cfg, "net_benefit_csv", split).parent


def test_risk_tertiles_path_does_not_mutate_the_config_it_is_given(cfg):
    before = dict(cfg["model_eval"])
    mf.risk_tertiles_path(cfg, TEST)
    assert dict(cfg["model_eval"]) == before
    assert mf.RISK_TERTILES_CSV_KEY not in cfg["model_eval"] or \
        cfg["model_eval"][mf.RISK_TERTILES_CSV_KEY] == before[mf.RISK_TERTILES_CSV_KEY]


def test_risk_tertiles_path_rejects_an_unknown_split(cfg):
    with pytest.raises(AssertionError, match="split must be one of"):
        mf.risk_tertiles_path(cfg, "train")


# --------------------------------------------------------------------------- #
# 10b. The schema                                                               #
# --------------------------------------------------------------------------- #
def test_risk_tertile_schema_is_pinned():
    assert mf.RISK_TERTILE_COLUMNS == [
        "split", "arm", "tertile", "tertile_label", "horizon_days", "curve_max_day",
        "n_patients", "n_events", "n_at_risk_horizon",
        "min_predicted_risk", "max_predicted_risk", "mean_predicted_risk",
        "km_cumulative_incidence", "km_ci_lo", "km_ci_hi",
        "n_scored", "n_events_scored", "note",
    ]


def test_no_risk_tertile_column_name_could_carry_an_identifier():
    """The same rule eval_models.assert_aggregate_only applies, checked on the schema."""
    for col in mf.RISK_TERTILE_COLUMNS:
        bad = [tok for tok in em.FORBIDDEN_OUTPUT_COLUMN_TOKENS if tok in col.lower()]
        assert not bad, f"column {col!r} carries identifier token(s) {bad}"


def test_the_note_carries_the_competing_risk_caveat_with_the_number():
    """The caveat travels in the artefact, not only in a caption a quoter may not read."""
    assert "cause-agnostic" in mf.RISK_TERTILE_NOTE
    assert "competing mortality is unmeasured" in mf.RISK_TERTILE_NOTE


# --------------------------------------------------------------------------- #
# 10c. One fit, two consumers                                                   #
# --------------------------------------------------------------------------- #
def test_tertile_curves_partition_the_patients_they_were_given():
    mc = mf._import_model_clinical()
    pred, t, e = _tertile_inputs()
    curves = mf.tertile_curves(mc, pred, t, e, horizon=1825.0, t_max=1826.0,
                               ticks=[0, 913, 1826])
    assert [c.index for c in curves] == [1, 2, 3]
    assert sum(c.n for c in curves) == pred.size
    assert sum(c.events for c in curves) == int(e.sum())
    # Ascending, non-overlapping predicted-risk bands: the tertiles ARE ordered.
    assert [c.risk_hi for c in curves] == sorted(c.risk_hi for c in curves)
    assert curves[0].risk_hi < curves[1].risk_lo < curves[1].risk_hi < curves[2].risk_lo


def test_tertile_curves_report_the_value_the_drawn_curve_carries_at_the_horizon():
    """Not a re-derivation: the number is a point ON the step curve the figure draws."""
    mc = mf._import_model_clinical()
    pred, t, e = _tertile_inputs()
    for c in mf.tertile_curves(mc, pred, t, e, horizon=1825.0, t_max=1826.0,
                               ticks=[0, 1826]):
        # step(where="post") holds cif[i] over [t[i], t[i+1]), so this is what a reader
        # measuring the plotted line at x = 1825 would read off the page.
        i = max(int(np.searchsorted(c.t, 1825.0, side="right")) - 1, 0)
        assert c.cif[i] == pytest.approx(c.cif_horizon, abs=1e-12)


def test_tertile_curves_agree_with_the_house_kaplan_meier_helper():
    """The same estimate src.model_clinical.km_risk gives - the helper figure 2 panel B
    uses - so the two figures cannot report different Kaplan-Meier conventions."""
    mc = mf._import_model_clinical()
    pred, t, e = _tertile_inputs()
    curves = mf.tertile_curves(mc, pred, t, e, horizon=1825.0, t_max=1826.0, ticks=[0, 1826])
    groups = mc.risk_bins(pred, mf.RISK_TERTILES)
    for g, c in enumerate(curves):
        k = groups == g
        obs, lo, hi = mc.km_risk(t[k], e[k], 1825.0)
        assert (c.cif_horizon, c.cif_ci_lo, c.cif_ci_hi) == pytest.approx((obs, lo, hi),
                                                                          abs=1e-12)


def test_tertile_curves_bound_the_estimate_they_report():
    mc = mf._import_model_clinical()
    pred, t, e = _tertile_inputs()
    for c in mf.tertile_curves(mc, pred, t, e, horizon=1825.0, t_max=1826.0, ticks=[0, 1826]):
        assert 0.0 <= c.cif_ci_lo <= c.cif_horizon <= c.cif_ci_hi <= 1.0
        assert 0 <= c.n_at_risk_horizon <= c.n
        assert c.t[-1] == 1826.0 and c.t.size == c.cif.size


def test_tertile_curves_label_the_groups_the_legend_labels_them():
    mc = mf._import_model_clinical()
    pred, t, e = _tertile_inputs()
    curves = mf.tertile_curves(mc, pred, t, e, horizon=1825.0, t_max=1826.0, ticks=[0, 1826])
    assert [c.label for c in curves] == [s["label"] for s in mf.TERTILE_STYLE]
    assert [c.label for c in curves] == ["Lowest tertile", "Middle tertile", "Highest tertile"]


# --------------------------------------------------------------------------- #
# 10d. The frame, and the leak guard on the way out                             #
# --------------------------------------------------------------------------- #
def test_build_risk_tertiles_writes_the_pinned_order_and_the_stated_totals():
    mc = mf._import_model_clinical()
    pred, t, e = _tertile_inputs()
    curves = mf.tertile_curves(mc, pred, t, e, horizon=1825.0, t_max=1826.0, ticks=[0, 1826])
    df = mf.build_risk_tertiles(curves, split=TEST, arm=mf.FIG3_MODEL, horizon_days=1825,
                                curve_max_day=1826, n_scored=pred.size,
                                n_events_scored=int(e.sum()))
    assert list(df.columns) == mf.RISK_TERTILE_COLUMNS
    assert len(df) == mf.RISK_TERTILES
    assert list(df["tertile"]) == [1, 2, 3]
    assert set(df["split"]) == {TEST} and set(df["arm"]) == {mf.FIG3_MODEL}
    assert set(df["horizon_days"]) == {1825} and set(df["curve_max_day"]) == {1826}
    assert int(df["n_patients"].sum()) == int(df["n_scored"].iloc[0]) == pred.size
    assert int(df["n_events"].sum()) == int(df["n_events_scored"].iloc[0]) == int(e.sum())


def test_the_tertile_table_goes_out_through_write_table(tmp_path):
    """The aggregate-only guarantee is eval_models', not a local imitation of it."""
    mc = mf._import_model_clinical()
    pred, t, e = _tertile_inputs()
    curves = mf.tertile_curves(mc, pred, t, e, horizon=1825.0, t_max=1826.0, ticks=[0, 1826])
    df = mf.build_risk_tertiles(curves, split=VAL, arm=mf.FIG3_MODEL, horizon_days=1825,
                                curve_max_day=1826, n_scored=pred.size,
                                n_events_scored=int(e.sum()))
    p = tmp_path / "val_risk_tertiles.csv"
    em.write_table(p, df, mf.RISK_TERTILE_COLUMNS, ["p000001"], p.name)
    back = pd.read_csv(p)
    assert list(back.columns) == mf.RISK_TERTILE_COLUMNS
    # And a frame that DID carry an identifier is refused, so the guard is not decorative.
    leaky = df.copy()
    leaky["note"] = "p000001"
    with pytest.raises(AssertionError, match="identifier"):
        em.write_table(p, leaky, mf.RISK_TERTILE_COLUMNS, ["p000001"], p.name)


# --------------------------------------------------------------------------- #
# 10e. Against the REAL artefacts: the numbers the manuscript will quote         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", [VAL, TEST])
def test_render_figure3_emits_the_tertile_table_beside_the_figure(tert_cfg, tmp_path, split):
    _skip_unless(tert_cfg.path(tert_cfg["paths"]["cohort_dir"]) /
                 f"{split}_hazards_{mf.FIG3_MODEL}.npz", f"the {split} hazards")
    png = mf.render_cumulative_incidence(tert_cfg, tmp_path, split)
    csv = mf.risk_tertiles_path(tert_cfg, split)
    assert png.exists() and csv.exists()
    df = pd.read_csv(csv)
    assert list(df.columns) == mf.RISK_TERTILE_COLUMNS
    a = mf.SPLIT_ANCHORS[split]
    # The denominators the caption states, in the artefact rather than only in prose.
    assert int(df["n_scored"].iloc[0]) == a["crop_n"]
    assert int(df["n_events_scored"].iloc[0]) == a["crop_events"]
    assert int(df["n_patients"].sum()) == a["crop_n"]
    assert int(df["n_events"].sum()) == a["crop_events"]
    assert set(df["split"]) == {split} and set(df["arm"]) == {mf.FIG3_MODEL}


@pytest.mark.parametrize("split", [VAL, TEST])
def test_the_emitted_tertile_incidences_are_the_ones_the_manuscript_quotes(tert_cfg,
                                                                           tmp_path, split):
    """The sealed row is the paper's headline: 3.6% / 19.8% / 40.6%, an 11-fold gradient.

    Frozen as literals so that re-scoring the split fails here, in a test that names the
    sentence, rather than silently rewriting a number a surgeon was going to act on.
    """
    _skip_unless(tert_cfg.path(tert_cfg["paths"]["cohort_dir"]) /
                 f"{split}_hazards_{mf.FIG3_MODEL}.npz", f"the {split} hazards")
    mf.render_cumulative_incidence(tert_cfg, tmp_path, split)
    df = pd.read_csv(mf.risk_tertiles_path(tert_cfg, split)).sort_values("tertile")
    got = tuple(float(v) for v in df["km_cumulative_incidence"])
    assert got == pytest.approx(FROZEN_TERTILE_INCIDENCE[split], abs=5e-7)
    assert list(got) == sorted(got), "the tertiles no longer order by observed incidence"
    lo, hi = df["km_ci_lo"].to_numpy(), df["km_ci_hi"].to_numpy()
    assert ((lo <= np.array(got)) & (np.array(got) <= hi)).all()


def test_the_tertile_table_is_deterministic_across_two_renders(tert_cfg, tmp_path, split=TEST):
    """Two renders into two directories, and each table read from ITS OWN directory.

    The table used to be written through the config path whatever ``out_dir`` said, so this
    read one file twice; it now follows the images, which is the point of the fix below.
    """
    _skip_unless(tert_cfg.path(tert_cfg["paths"]["cohort_dir"]) /
                 f"{split}_hazards_{mf.FIG3_MODEL}.npz", f"the {split} hazards")
    name = mf.risk_tertiles_path(tert_cfg, split).name
    mf.render_cumulative_incidence(tert_cfg, tmp_path / "a", split)
    mf.render_cumulative_incidence(tert_cfg, tmp_path / "b", split)
    first = (tmp_path / "a" / name).read_bytes()
    assert (tmp_path / "b" / name).read_bytes() == first


# --------------------------------------------------------------------------- #
# 10f. --out-dir has to reach the TABLE, not only the images                    #
# --------------------------------------------------------------------------- #
def test_risk_tertiles_output_path_follows_out_dir_and_only_out_dir(cfg, tmp_path):
    """The configured path when the images go where they are configured to go, and beside
    the images otherwise. Figure 3 is the only renderer that writes a repository TABLE, so
    it was the only one for which --out-dir was not in fact an isolation."""
    configured = cfg.path(cfg["manuscript"]["figures_dir"])
    for split in (VAL, TEST):
        want = mf.risk_tertiles_path(cfg, split)
        assert mf.risk_tertiles_output_path(cfg, split, None) == want
        assert mf.risk_tertiles_output_path(cfg, split, configured) == want
        # the same directory spelled differently is still the same directory
        assert mf.risk_tertiles_output_path(cfg, split, str(configured)) == want
        assert mf.risk_tertiles_output_path(cfg, split, configured / ".." /
                                            configured.name) == want
        assert mf.risk_tertiles_output_path(cfg, split, tmp_path) == tmp_path / want.name


@pytest.mark.parametrize("split", [VAL, TEST])
def test_render_figure3_into_a_scratch_directory_writes_no_repository_table(cfg, tmp_path,
                                                                            split):
    """The real config, a scratch --out-dir, and outputs/tables/ untouched.

    This is the guarantee that was missing: render_figure3 resolved the CSV through the
    CONFIG path, so a scratch render of figure 3 silently rewrote
    outputs/tables/{split}_risk_tertiles.csv, which is the file the manuscript's tertile
    sentence reads. A --split val render would therefore have left validation incidences
    in the file the sealed document quotes.
    """
    _skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) /
                 f"{split}_hazards_{mf.FIG3_MODEL}.npz", f"the {split} hazards")
    repo = mf.risk_tertiles_path(cfg, split)
    before = repo.read_bytes() if repo.exists() else None
    mtime = repo.stat().st_mtime_ns if repo.exists() else None

    png = mf.render_cumulative_incidence(cfg, tmp_path, split)
    assert png.parent == tmp_path
    scratch = tmp_path / repo.name
    assert scratch.exists(), "the table did not follow the images into --out-dir"
    assert list(pd.read_csv(scratch).columns) == mf.RISK_TERTILE_COLUMNS

    assert (repo.read_bytes() if repo.exists() else None) == before, \
        f"a scratch render rewrote {repo}"
    assert (repo.stat().st_mtime_ns if repo.exists() else None) == mtime, \
        f"a scratch render touched {repo}"


@pytest.mark.parametrize("split", [VAL, TEST])
def test_the_emitted_table_matches_the_curve_that_was_drawn(tert_cfg, tmp_path, split):
    """The renderer's own inputs, re-fitted through the SAME function, reproduce the file.

    This is the check that the CSV is a transcript of the figure and not a parallel
    estimate: it goes through tertile_curves, which is the only Kaplan-Meier the renderer
    has.
    """
    _skip_unless(tert_cfg.path(tert_cfg["paths"]["cohort_dir"]) /
                 f"{split}_hazards_{mf.FIG3_MODEL}.npz", f"the {split} hazards")
    mf.render_cumulative_incidence(tert_cfg, tmp_path, split)
    df = pd.read_csv(mf.risk_tertiles_path(tert_cfg, split)).sort_values("tertile")
    mc = mf._import_model_clinical()
    fr = mf._split_frame(tert_cfg, mc, split)
    horizon = float([int(h) for h in tert_cfg["model_eval"]["horizons_days"]][-1])
    t_max = float(round(float(tert_cfg["timeline"]["horizon_years"])
                        * float(tert_cfg["timeline"]["days_per_year"])))
    pred = mf._arm_risks(tert_cfg, mc, fr, horizon, (mf.FIG3_MODEL,), split,
                         QUIET)[mf.FIG3_MODEL]
    ok = np.isfinite(pred)
    ticks = [0, int(t_max)]
    curves = mf.tertile_curves(mc, pred[ok],
                               fr["time_from_landmark"].to_numpy(dtype=float)[ok],
                               fr["event_indicator"].to_numpy(dtype=int)[ok],
                               horizon=horizon, t_max=t_max, ticks=ticks)
    assert [c.n for c in curves] == list(df["n_patients"])
    assert [c.events for c in curves] == list(df["n_events"])
    assert [c.n_at_risk_horizon for c in curves] == list(df["n_at_risk_horizon"])
    assert [round(c.cif_horizon, em.ROUND_DECIMALS) for c in curves] == \
        list(df["km_cumulative_incidence"])


# =========================================================================== #
# 11. THE PROVENANCE MANIFEST                                                  #
#                                                                              #
# A PNG carries no record of the split it was drawn for, and figure 1 is       #
# split-aware prose: the validation render says "Sealed test split, never      #
# read, n = 741" in an exclusion box. The v2 build embedded exactly that image #
# in a document reporting those 741 patients, and every self-check passed,     #
# because make_manuscript embeds bytes it did not draw and had nothing to      #
# compare them against. render_all now writes down what it drew, beside what   #
# it drew. These tests pin the record; tests/test_make_manuscript.py pins the  #
# refusal to build on images the record does not describe.                     #
# =========================================================================== #
def _stub_all(monkeypatch) -> None:
    """Every renderer stubbed to write its own key. No artefact is read."""
    monkeypatch.setattr(mf, "RENDERERS", {k: _stub_renderer(k) for k in mf.FIGURE_KEYS})


def _manifest(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / mf.FIGURES_MANIFEST).read_text())


def test_a_whole_render_writes_a_manifest_beside_the_images_it_wrote(cfg, tmp_path,
                                                                     monkeypatch):
    _stub_all(monkeypatch)
    written = mf.render_all(cfg, out_dir=tmp_path, split=VAL)
    m = _manifest(tmp_path)
    assert m["schema"] == mf.MANIFEST_SCHEMA and m["schema_version"] == 1
    assert m["written_by"] == "manuscript_figures" and m["split"] == VAL
    assert m["complete"] is True and m["not_attempted"] == []
    assert [e["key"] for e in m["rendered"]] == list(mf.FIGURE_KEYS)
    assert [e["number"] for e in m["rendered"]] == [1, 2, 3, 4]
    for entry in m["rendered"]:
        path = tmp_path / entry["filename"]
        assert path == written[entry["key"]]
        assert entry["sha256"] == mf.sha256_file(path)
        assert entry["size_bytes"] == path.stat().st_size


def test_two_identical_renders_produce_byte_identical_manifests(cfg, tmp_path, monkeypatch):
    """No timestamp, no absolute path, no run identifier. Two renders are comparable or
    the record cannot be used to tell one render's output from another's."""
    _stub_all(monkeypatch)
    a, b = tmp_path / "a", tmp_path / "b"
    mf.render_all(cfg, out_dir=a, split=TEST)
    mf.render_all(cfg, out_dir=b, split=TEST)
    assert (a / mf.FIGURES_MANIFEST).read_bytes() == (b / mf.FIGURES_MANIFEST).read_bytes()
    # and re-rendering into the SAME directory reproduces the file it replaced
    first = (a / mf.FIGURES_MANIFEST).read_bytes()
    mf.render_all(cfg, out_dir=a, split=TEST)
    assert (a / mf.FIGURES_MANIFEST).read_bytes() == first


def test_a_partial_render_is_marked_partial_rather_than_claiming_the_whole_set(cfg, tmp_path,
                                                                               monkeypatch):
    """--only figure1 must not leave a record that reads as a description of all four."""
    _stub_all(monkeypatch)
    mf.render_all(cfg, out_dir=tmp_path, split=VAL, only="figure1")
    m = _manifest(tmp_path)
    assert m["complete"] is False
    assert m["attempted"] == ["figure1"]
    assert [e["key"] for e in m["rendered"]] == ["figure1"]
    assert [e["key"] for e in m["not_attempted"]] == ["figure2", "figure3", "figure4"]
    for entry in m["not_attempted"]:
        assert entry["reason"] == mf.MANIFEST_NOT_ATTEMPTED_REASON
        assert "--only" in entry["reason"]


def test_a_partial_render_replaces_a_complete_manifest_rather_than_leaving_it(cfg, tmp_path,
                                                                              monkeypatch):
    """The dangerous direction. A complete record plus one figure redrawn underneath it is
    a record that is true of three files and false of the fourth, and nothing in the file
    says which. The render that began invalidated it, so it goes."""
    _stub_all(monkeypatch)
    mf.render_all(cfg, out_dir=tmp_path, split=TEST)
    assert _manifest(tmp_path)["complete"] is True
    mf.render_all(cfg, out_dir=tmp_path, split=VAL, only="figure1")
    m = _manifest(tmp_path)
    assert m["complete"] is False and m["split"] == VAL
    assert [e["key"] for e in m["rendered"]] == ["figure1"]


def test_a_render_that_raises_leaves_no_manifest_at_all(cfg, tmp_path, monkeypatch):
    """Fail closed. A half-finished directory must report "no provenance", never a stale
    record whose surviving entries still happen to match."""
    _stub_all(monkeypatch)
    mf.render_all(cfg, out_dir=tmp_path, split=TEST)
    assert (tmp_path / mf.FIGURES_MANIFEST).exists()

    def _boom(c, o, s):
        raise RuntimeError("the renderer fell over")

    monkeypatch.setattr(mf, "RENDERERS", {**mf.RENDERERS, "figure2": _boom})
    with pytest.raises(RuntimeError, match="fell over"):
        mf.render_all(cfg, out_dir=tmp_path, split=TEST)
    assert not (tmp_path / mf.FIGURES_MANIFEST).exists()


def test_the_manifest_records_the_sealed_contract_only_where_there_is_one(cfg, tmp_path,
                                                                          monkeypatch):
    """The hash render_all already verified on a sealed render, carried into the record so
    a reader can see WHICH frozen models the images describe. A development split has no
    sealed read to be on the record and therefore no contract to pin."""
    _stub_all(monkeypatch)
    monkeypatch.setattr(mf, "assert_sealed_read_is_recorded", lambda c: "4b862b5ecb947314")
    mf.render_all(cfg, out_dir=tmp_path / "t", split=TEST)
    assert _manifest(tmp_path / "t")["training_contract_hash"] == "4b862b5ecb947314"
    mf.render_all(cfg, out_dir=tmp_path / "v", split=VAL)
    assert _manifest(tmp_path / "v")["training_contract_hash"] is None


def test_the_real_sealed_contract_reaches_the_manifest(cfg, tmp_path, monkeypatch, cohort_dir):
    """Unmocked, against the recorded sealed read itself."""
    record = _skip_unless(cohort_dir / em.SEALED_READ_RECORD, "the sealed read record")
    _stub_all(monkeypatch)
    mf.render_all(cfg, out_dir=tmp_path, split=TEST)
    recorded = json.loads(record.read_text())
    assert _manifest(tmp_path)["training_contract_hash"] == \
        recorded["training_contract_hash"]


def test_a_declined_figure_is_recorded_as_declined_and_not_merely_absent(cfg, tmp_path,
                                                                         monkeypatch):
    """THE STATE THAT MUST NOT BE INFERRED. The decision curve declines on validation, so no
    image is written for it; a reader of the directory sees the same absence a deleted file
    leaves. The record has to say which, in words, or the consumer is guessing."""
    _skip_unless(em.split_path(cfg, "convergence_csv", VAL), "val_convergence.csv")
    others = [k for k in mf.SUPPLEMENT_KEYS if k != "figureS3"]
    monkeypatch.setattr(mf, "SUPPLEMENT_RENDERERS",
                        {**mf.SUPPLEMENT_RENDERERS,
                         **{k: _stub_renderer(k) for k in others}})
    written = mf.render_supplement(cfg, out_dir=tmp_path, split=VAL)
    assert "figureS3" not in written
    m = _manifest(tmp_path / mf.SUPPLEMENT_DIRNAME)
    assert m["schema"] == mf.SUPPLEMENT_MANIFEST_SCHEMA
    assert m["complete"] is True, "every figure was offered to a renderer"
    assert [e["key"] for e in m["rendered"]] == others
    assert m["not_attempted"] == [], "the decision curve was attempted; it declined"
    assert [e["key"] for e in m["declined"]] == ["figureS3"]
    d = m["declined"][0]
    assert d["number"] == 3 and d["filename"] == "figureS3_decision_curve.png"
    assert d["reason"] == mf.MANIFEST_DECLINED_REASON and "declined" in d["reason"]


def test_a_renderer_that_returns_a_path_it_never_wrote_is_a_failure(cfg, tmp_path,
                                                                    monkeypatch):
    """Returning None is how a renderer declines. Returning a path is a claim about bytes."""
    monkeypatch.setattr(mf, "RENDERERS",
                        {**{k: _stub_renderer(k) for k in mf.FIGURE_KEYS},
                         "figure3": lambda c, o, s: o / "never_written.png"})
    with pytest.raises(FileNotFoundError, match="which does not exist"):
        mf.render_all(cfg, out_dir=tmp_path, split=TEST)


def test_a_renderer_that_writes_outside_the_render_directory_is_a_failure(cfg, tmp_path,
                                                                          monkeypatch):
    """The manifest records filenames and is read beside the images it describes, so a
    figure written somewhere else could never be found again from it."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def _astray(c, o, s):
        p = elsewhere / "figure3_cumulative_incidence.png"
        p.write_bytes(b"astray")
        return p

    monkeypatch.setattr(mf, "RENDERERS",
                        {**{k: _stub_renderer(k) for k in mf.FIGURE_KEYS},
                         "figure3": _astray})
    with pytest.raises(ValueError, match="not in the render directory"):
        mf.render_all(cfg, out_dir=tmp_path / "into", split=TEST)


def test_a_scratch_render_writes_its_own_manifest_and_never_the_real_one(cfg, tmp_path,
                                                                         monkeypatch):
    """--out-dir is the whole isolation mechanism: the manifest follows the images."""
    real = cfg.path(cfg["manuscript"]["figures_dir"]) / mf.FIGURES_MANIFEST
    before = real.read_bytes() if real.exists() else None
    _stub_all(monkeypatch)
    mf.render_all(cfg, out_dir=tmp_path, split=VAL)
    assert (tmp_path / mf.FIGURES_MANIFEST).exists()
    after = real.read_bytes() if real.exists() else None
    assert after == before, "a scratch render moved the real figures directory's manifest"


def test_the_manifest_carries_no_patient_identifier(cfg, tmp_path, monkeypatch):
    """Protocol section 28. The payload is a split name, a hash, filenames and digests, and
    the hex digests are the only long tokens in it."""
    _stub_all(monkeypatch)
    mf.render_all(cfg, out_dir=tmp_path, split=TEST)
    text = (tmp_path / mf.FIGURES_MANIFEST).read_text()
    digests = {e["sha256"] for e in _manifest(tmp_path)["rendered"]}
    scannable = text
    for d in digests:
        scannable = scannable.replace(d, " ")
    assert not re.findall(r"(?<!\d)\d{8}(?!\d)", scannable), "an empi_anon-shaped token"
    assert not re.findall(r"\b1\.2\.\d[\d.]*", scannable), "a DICOM UID-shaped token"
    assert "empi" not in text.lower()


def test_the_cli_writes_the_manifest_into_the_directory_it_was_pointed_at(tmp_path,
                                                                          monkeypatch):
    _stub_all(monkeypatch)
    assert mf.main(["--split", "val", "--only", "all", "--out-dir", str(tmp_path)]) == 0
    assert _manifest(tmp_path)["split"] == VAL


def test_figure_one_really_does_differ_between_the_two_splits(cfg, tmp_path):
    """THE DEFECT, stated as a fact about the bytes. Figure 1 is split-aware prose, so the
    two renders are different files under the same name, and the only thing that ever
    distinguished them on disk was the modification time."""
    for name in ("cohort_flow.csv",):
        _skip_unless(cfg.path(cfg["paths"]["outputs_dir"]) / name, name)
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    for name in ("patient_splits.parquet", "clinical_imputation_params.json"):
        _skip_unless(coh / name, name)
    out = {s: tmp_path / s for s in (VAL, TEST)}
    for d in out.values():
        d.mkdir()
    a = mf.render_cohort_flow(cfg, out[VAL], VAL)
    b = mf.render_cohort_flow(cfg, out[TEST], TEST)
    assert a.name == b.name, "same filename, which is why nothing downstream could tell"
    assert mf.sha256_file(a) != mf.sha256_file(b)


# =========================================================================== #
# 12. THE v6 IMAGING FIGURE SET                                                #
#                                                                              #
# Journal of Imaging rejected v5 on scope: none of its four figures carried a  #
# radiograph. The four main figures are now the imaging set, and the three v5  #
# figures that survive in substance are supplementary. Everything below is     #
# about the new four and the new supplementary registry; the tests above are   #
# unchanged in intent and pinned at their new keys.                            #
# =========================================================================== #
TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(_skip_unless(TABLES / name, name))


# --------------------------------------------------------------------------- #
# 12a. The two registries                                                      #
# --------------------------------------------------------------------------- #
def test_the_main_registry_is_the_v6_imaging_set(cfg):
    d = {x.key: x for x in mf.FIGURE_DEFS}
    assert mf.FIGURE_KEYS == ("figure1", "figure2", "figure3", "figure4")
    assert [x.number for x in mf.FIGURE_DEFS] == [1, 2, 3, 4]
    assert d["figure1"].renderer is mf.render_figure1
    assert d["figure2"].renderer is mf.render_figure2
    assert d["figure3"].renderer is mf.render_figure3
    assert d["figure4"].renderer is mf.render_figure4
    assert all(x.width_key == "double_column_in" for x in mf.FIGURE_DEFS)
    titles = {k: v["title"] for k, v in mf.figures(cfg, TEST).items()}
    assert "workflow" in titles["figure1"]
    assert "Kellgren-Lawrence grade" in titles["figure2"]
    assert "withholding views" in titles["figure3"]
    assert "Calibration" in titles["figure4"]


def test_the_supplementary_registry_holds_six_figures_and_shares_nothing_with_the_main_one():
    assert mf.SUPPLEMENT_KEYS == ("figureS1", "figureS2", "figureS3", "figureS4",
                                  "figureS5", "figureS6")
    assert [x.number for x in mf.SUPPLEMENT_DEFS] == [1, 2, 3, 4, 5, 6]
    assert not set(mf.FIGURE_KEYS) & set(mf.SUPPLEMENT_KEYS)
    assert not ({x.filename for x in mf.FIGURE_DEFS}
                & {x.filename for x in mf.SUPPLEMENT_DEFS})
    # every supplementary filename says which supplementary figure it is
    for x in mf.SUPPLEMENT_DEFS:
        assert x.filename.startswith(f"figureS{x.number}_")


def test_the_v5_main_figures_kept_their_renderers_when_they_became_supplementary():
    """S2, S3 and S4 are the same functions, not reimplementations of them."""
    d = {x.key: x for x in mf.SUPPLEMENT_DEFS}
    assert d["figureS2"].renderer is mf.render_cohort_flow
    assert d["figureS3"].renderer is mf.render_decision_curve
    assert d["figureS4"].renderer is mf.render_cumulative_incidence
    assert d["figureS2"].caption is mf._fig1_caption
    assert d["figureS3"].caption is mf._fig4_caption
    assert d["figureS4"].caption is mf._fig3_caption


def test_supplement_figures_returns_the_same_five_fields_in_number_order(cfg):
    spec = mf.supplement_figures(cfg, TEST)
    assert list(spec) == list(mf.SUPPLEMENT_KEYS)
    assert [v["number"] for v in spec.values()] == [1, 2, 3, 4, 5, 6]
    for v in spec.values():
        assert set(v) == {"number", "filename", "width_in", "title", "caption"}
        assert isinstance(v["width_in"], float) and v["width_in"] > 0
        assert v["title"] and v["caption"]


def test_assert_registry_polices_the_supplementary_registry_too():
    good = mf.SUPPLEMENT_DEFS
    broken = (*good[:-1], replace(good[-1], number=99))
    with pytest.raises(AssertionError, match="SUPPLEMENT_DEFS must be ordered"):
        mf.assert_registry(broken, "SUPPLEMENT_DEFS")


def test_the_document_generator_can_only_reach_the_main_registry(cfg):
    """A supplementary figure that leaked into figures() would be numbered Figure 5."""
    assert set(mf.figures(cfg, TEST)) == set(mf.FIGURE_KEYS)
    assert set(mf.FIGURES) == set(mf.FIGURE_KEYS)


# --------------------------------------------------------------------------- #
# 12b. Figure 1: the workflow, over a real radiograph                          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fig1(cfg):
    _skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) / "qa_panels" /
                 f"{mf.FIG1_CASE_KEY}.png", "the Figure 1 reviewer QA panel")
    _skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_labels.csv",
                 "the crop label table")
    _skip_unless(cfg.path(cfg["paths"]["source_parquet_dir"]) / "image.parquet",
                 "the delivered image inventory")
    return mf.figure1_assets(cfg)


def test_figure_ones_case_is_a_bilateral_frontal_the_pipeline_half_selected(cfg, fig1):
    """Everything the figure illustrates has to be true of the case it illustrates it on."""
    lab = pd.read_csv(cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_labels.csv")
    row = lab[lab["key"].astype(str) == mf.FIG1_CASE_KEY].iloc[0]
    assert str(row["view"]) == "frontal"
    assert str(row["laterality"]).upper() == "B", "the half-select needs a bilateral film"
    assert str(row["half_selected"]) in ("left", "right")
    assert bool(row["mirrored"]), "the mirror step is only visible where one was applied"
    assert str(row["crop_method"]) == "center_default"
    assert float(row["crop_confidence"]) == 1.0
    assert str(row["split"]) != TEST, (
        "the sealed split's reviewer panels are crop-only, so no test-split case can show a "
        "pre-crop film; Figure 1's radiograph is a development-split film and says so")


def test_the_recovered_film_is_restored_to_the_acquired_aspect_ratio(cfg, fig1):
    lab = pd.read_csv(cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_labels.csv")
    row = lab[lab["key"].astype(str) == mf.FIG1_CASE_KEY].iloc[0]
    true_h, true_w = mf._source_image_dims(cfg, str(row["sop_uid"]))
    h, w = fig1["film"].shape
    assert abs(h / w - true_h / true_w) < 0.01, (
        "the QA panel draws the film with aspect='auto', so a recovered film that is not "
        "resampled back is stretched")


def test_the_half_bounds_are_the_pipelines_own_arithmetic(cfg, fig1):
    from src.preprocess_images import half_column_bounds
    lab = pd.read_csv(cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_labels.csv")
    row = lab[lab["key"].astype(str) == mf.FIG1_CASE_KEY].iloc[0]
    want = half_column_bounds(fig1["film"].shape[1], str(row["half_selected"]),
                              float(cfg["preprocess"]["half_inset_frac"]))
    assert fig1["half_cols"] == want
    assert fig1["half"].shape[1] == want[1] - want[0]


def test_the_localization_box_is_the_box_the_pipeline_used(cfg, fig1):
    """The claim the caption makes, checked: replaying the geometry rebuilds the crop.

    D5. This used to open ``if fig1["crop_source"] != "shard": pytest.skip(...)``, which
    made the ONE check that licenses Figure 1's recovered film optional on exactly the
    state in which it mattered: with the crop asset absent, ``figure1_assets`` fell back to
    the QA panel, set ``pearson`` to NaN, skipped both of its own assertions with a log
    warning, and the caption went on printing 0.99 from a module constant. There is no
    fallback any more - a missing asset raises - so there is no branch here either.
    """
    assert fig1["pearson"] >= mf.FIG1_CROP_PEARSON_MIN
    assert round(fig1["pearson"], 2) == mf.FIG1_CROP_PEARSON
    r0, c0, side = fig1["box"]
    hh, hw = fig1["half"].shape
    assert side == int(round(float(cfg["preprocess"]["max_crop_frac"]) * min(hh, hw)))
    assert r0 == int(round(hh / 2.0 - side / 2.0))
    assert c0 == int(round(hw / 2.0 - side / 2.0))


def test_a_missing_crop_asset_is_a_hard_failure_and_not_a_quiet_fallback(cfg, tmp_path,
                                                                        monkeypatch):
    """D5, from the other side. The caption's Pearson value may not outlive its check."""
    monkeypatch.setattr(mf, "figure_assets_dir", lambda _cfg: tmp_path / "nothing-here")
    with pytest.raises(FileNotFoundError, match="never run"):
        mf.figure1_assets(cfg)


def test_the_masked_border_band_really_is_blank_in_the_crop_the_figure_draws(cfg, fig1):
    b = fig1["band"]
    final = fig1["final"]
    assert b == int(round(float(cfg["preprocess"]["mask_border_frac"])
                          * int(cfg["preprocess"]["out_size"])))
    for edge in (final[:b], final[-b:], final[:, :b], final[:, -b:]):
        assert float(edge.max()) == 0.0


def test_the_workflow_figure_discloses_the_marker_its_own_crop_still_carries(cfg, fig1):
    """D1. The caption's two claims about this crop, measured on this crop.

    Panel 3 used to be titled "border and markers masked" and the caption said any marker
    the band did not reach "is masked". The shipped model input disproves both: the zero
    fraction is EXACTLY the integer band, so the marker step accepted nothing here, and a
    saturated burned-in character survives inside the retained region and is visible in
    every rendered panel. The figure now says so, and this holds it to saying so.
    """
    assert fig1["zero_frac"] == pytest.approx(fig1["band_only_frac"], abs=1e-9), (
        "the marker step blanked pixels on this crop after all, so the caption's "
        "'the marker detector accepted nothing on this crop' is no longer true")
    assert fig1["zero_frac"] == pytest.approx(1.0 - (450 / 512) ** 2, abs=1e-9)
    assert fig1["residual_saturated_px"] >= int(cfg["preprocess"]["marker_min_px"])
    cap = mf.figures(cfg, TEST)["figure1"]["caption"]
    assert "The marker detector accepted nothing on this crop" in cap
    assert "saturated burned-in marker survives inside the retained region" in cap
    assert "and markers masked" not in cap, "the disproved claim is gone from the prose"
    for n in (f"{mf.CROP_AUDIT_N:,}", f"{mf.CROP_RESIDUAL_MARKER_PCT:g} percent",
              str(mf.CROP_NONZERO_BAND_N), f"{mf.CROP_NONZERO_BAND_PCT:g} percent"):
        assert n in cap, f"the caption drops the audit anchor {n!r}"


def test_the_workflow_panel_title_claims_only_the_step_that_ran(cfg):
    """D1. The title over the tile that shows a surviving marker may not deny it."""
    titles = mf.FIG1_PANEL_TITLES
    assert len(titles) == 4
    assert "markers masked" not in titles[2], (
        "panel 3's title asserts a step the picture it sits over disproves")
    assert titles[2] == "3  Square crop, {out_size} px,\nborder masked"
    inset = f"{100.0 * float(cfg['preprocess']['half_inset_frac']):g}"
    assert titles[1].format(half_inset_pct=inset).endswith(f"{inset}% midline inset")


def test_the_marker_audit_anchors_are_the_audit_tables_own_numbers(cfg):
    """D1 and D2. Every residual-marker number both captions print, against its table."""
    mf.assert_marker_audit_anchors(cfg)          # raises if any anchor moved
    reg = _table("interp_regions.csv").set_index("item")
    assert float(reg.loc["residual_marker_crops_pct", "value"]) == mf.CROP_RESIDUAL_MARKER_PCT
    assert int(float(reg.loc["n_crops_with_nonzero_border_band", "value"])) == \
        mf.CROP_NONZERO_BAND_N
    assert str(mf.CROP_AUDIT_N) in str(reg.loc["residual_marker_crops_pct", "note"])
    occl = _table("interp_occlusion.csv")
    row = occl[(occl["arm"].astype(str) == mf.INTERP_ARM)
               & (occl["condition"].astype(str) == mf.MARKER_DELTA_CONDITION)].iloc[0]
    assert -float(row["delta_auroc"]) == pytest.approx(mf.MARKER_DELTA_AUROC, abs=5e-6)
    assert -float(row["delta_auroc_hi"]) == pytest.approx(mf.MARKER_DELTA_LO, abs=5e-6)
    assert -float(row["delta_auroc_lo"]) == pytest.approx(mf.MARKER_DELTA_HI, abs=5e-6)
    # THE SCOPE, which is the part a reviewer would break. THREE conditions in this table
    # have an interval that excludes zero, so "the only perturbation whose interval
    # excludes zero" is false unqualified; the caption says "the only row of Figure 3's
    # masked band and burned-in marker block", and that is what is true.
    excl = {str(r["condition"]) for _, r in occl.iterrows()
            if str(r["arm"]) == mf.INTERP_ARM and str(r["condition"]) != "baseline"
            and not (float(r["delta_auroc_lo"]) <= 0.0 <= float(r["delta_auroc_hi"]))}
    assert excl == {mf.MARKER_DELTA_CONDITION, "keep_border_only",
                    mf.FOREST_MEANFILL_CONDITION}, sorted(excl)
    block = {c for c, _ in mf.FOREST_MASKING_ROWS}
    assert excl & block == {mf.MARKER_DELTA_CONDITION}
    cap = mf.figures(cfg, TEST)["figure2"]["caption"]
    assert "the only row of Figure 3's masked band and burned-in marker block" in cap
    assert "the only perturbation" not in cap, (
        "an unqualified 'only perturbation' is false of this table: the degenerate "
        "band-only control and the mean-filled joint-only occlusion also exclude zero")


def test_the_workflow_caption_states_the_preprocessing_constants_from_config(cfg):
    cap = mf.figures(cfg, TEST)["figure1"]["caption"]
    pp = cfg["preprocess"]
    band = float(pp["mask_border_frac"])
    assert f"{100.0 * float(pp['half_inset_frac']):g} percent of the film width" in cap
    assert f"{100.0 * float(pp['max_crop_frac']):g} percent of its short side" in cap
    assert f"{int(pp['out_size'])} by {int(pp['out_size'])} pixels" in cap
    assert f"outer {int(round(band * int(pp['out_size'])))} pixels" in cap
    # 22.75 percent, and the arithmetic that gives it. The band is an INTEGER 31 px per
    # edge, so the blanked area is 1 - (450/512)^2 = 22.752%. The two near misses are
    # 22.76 (a rounding of the right quantity that the checklist rejects) and 22.56 (the
    # continuous expression 1 - (1 - 2f)^2, which ignores that 31 > 0.06 * 512).
    size = int(pp["out_size"])
    px = int(round(band * size))
    assert px == 31 and size == 512
    assert f"{100.0 * (1.0 - ((size - 2 * px) / size) ** 2):.2f} percent of the output" in cap
    assert "22.75 percent of the output" in cap
    assert "22.76" not in cap and "22.56" not in cap
    assert f"{mf.FIG1_CROP_PEARSON:.2f}" in cap
    assert f"{int(cfg['model_image']['survival_head']['n_intervals'])} interval hazards" in cap
    assert "no longer held" in cap, "the caption must say where the film came from"


def test_the_workflow_render_writes_the_registered_file_at_the_column_width(cfg, tmp_path,
                                                                           fig1):
    from PIL import Image
    out = mf.render_figure1(cfg, tmp_path, TEST)
    assert out.name == "figure1_imaging_workflow.png" and out.exists()
    with Image.open(out) as im:
        want = round(float(cfg["manuscript"]["double_column_in"])
                     * int(cfg["manuscript"]["figure_dpi"]))
        assert abs(im.size[0] - want) <= mf.WIDTH_LOCK_TOL_PX
        assert im.mode == "RGB"


def test_a_crop_only_qa_panel_is_refused_rather_than_drawn_without_its_film(cfg):
    """The sealed split's panels have no full film; asking for one must say so."""
    lab = pd.read_csv(_skip_unless(cfg.path(cfg["paths"]["cohort_dir"]) /
                                   "preprocess_labels.csv", "the crop label table"))
    panels = cfg.path(cfg["paths"]["cohort_dir"]) / "qa_panels"
    _skip_unless(panels, "the reviewer QA panels")
    crop_only = [p.stem for p in panels.glob("*.png")
                 if str(lab[lab["key"].astype(str) == p.stem]["split"].iloc[0]) == TEST] \
        if len(lab) else []
    if not crop_only:
        pytest.skip("no crop-only panel on this machine")
    with pytest.raises(AssertionError, match="crop-only panel"):
        mf.read_qa_panel(cfg, crop_only[0])


# --------------------------------------------------------------------------- #
# 12c. Figure 2: representative image findings at one KL grade                 #
# --------------------------------------------------------------------------- #
def test_figure_twos_four_cases_are_one_per_cell_at_one_grade():
    assert [c for c, _, _ in mf.FIND_CASES] == ["TN", "FN", "FP", "TP"]
    risks = [r for _, _, r in mf.FIND_CASES]
    assert risks == sorted(risks), "columns are ordered by predicted risk"
    assert len({p for _, p, _ in mf.FIND_CASES}) == 4


def test_figure_twos_anchored_risks_are_the_published_ones(cfg):
    man = _table("interp_panel_manifest.csv")
    man = man[man["arm"].astype(str) == mf.INTERP_ARM]
    by_id = {str(r["empi_anon"]): r for _, r in man.iterrows()}
    for cell, pid, risk in mf.FIND_CASES:
        row = by_id[pid]
        assert str(row["cell"]) == cell
        assert float(row["klg_contra"]) == mf.FIND_KLG
        assert abs(float(row["risk_published"]) - risk) < mf.FIND_RISK_TOL
        assert (float(row["risk_published"]) >= mf.FIND_THRESHOLD) == (cell in ("TP", "FP"))


def test_figure_twos_caption_prints_the_spread_the_cases_carry(cfg):
    cap = mf.figures(cfg, TEST)["figure2"]["caption"]
    risks = [r for _, _, r in mf.FIND_CASES]
    for r in risks:
        assert f"{r:.3f}" in cap
    assert f"{max(risks) / min(risks):.1f}-fold spread" in cap
    assert f"{mf.FIND_THRESHOLD:.3f}" in cap
    assert "smoothed for display only" in cap, "the IG tiles are smoothed and must say so"


# --------------------------------------------------------------------------- #
# D4. The operating point, and the scale the spread is a property of.          #
#                                                                              #
# The caption said the threshold "makes the number of predicted positives      #
# equal the number of observed events". That is true on the 263 patients       #
# classifiable at the horizon and FALSE on the 734 the rest of the figure set  #
# reports, where the same threshold flags 281; _stratify drops the 471         #
# censored before the horizon before it takes the quantile. All four risks are #
# also RECALIBRATED, which the caption never said, and "10.8-fold" is a        #
# property of that scale: the same two patients are 4.24-fold apart raw.       #
# --------------------------------------------------------------------------- #
def test_the_operating_point_is_the_quantile_over_the_classifiable_patients(cfg):
    """Recomputed from the published hazards, the way interpretability computed it."""
    import json

    coh = cfg.path(cfg["paths"]["cohort_dir"])
    _skip_unless(coh / f"test_hazards_{mf.INTERP_ARM}.npz", "the sealed frontal hazards")
    _skip_unless(coh / "train_arms.json", "the training hand-over index")
    from src.interpretability import EDGES, HORIZON_DAYS, risk_at_horizon
    from src.train_model import apply_recalibration

    z = np.load(coh / f"test_hazards_{mf.INTERP_ARM}.npz", allow_pickle=False)
    pids = np.asarray(z["empi_anon"]).astype(str)
    t, e = np.asarray(z["time"], float), np.asarray(z["event"], int)
    recal = json.loads((coh / "train_arms.json").read_text())["arms"][mf.INTERP_ARM][
        "recalibration"][str(float(HORIZON_DAYS))]
    raw = risk_at_horizon(np.asarray(z["hazards"], float), float(HORIZON_DAYS), edges=EDGES)
    pub = np.asarray(apply_recalibration(raw, recal), dtype=float)

    H = float(HORIZON_DAYS)
    y = np.where((t <= H) & (e == 1), 1, np.where(t > H, 0, -1))
    ok = y >= 0
    assert pids.size == mf.FOREST_OCCLUSION_N
    assert int(ok.sum()) == mf.FIND_CLASSIFIABLE
    assert int((y == 1).sum()) == mf.FIND_CASE_N
    assert int((y == 0).sum()) == mf.FIND_CONTROL_N
    assert int((y == -1).sum()) == mf.FIND_CENSORED
    thr = float(np.quantile(pub[ok], 1.0 - float((y[ok] == 1).mean())))
    assert round(thr, 4) == mf.FIND_THRESHOLD
    # The claim the old caption made, true only on the classifiable set...
    assert int((pub[ok] >= thr).sum()) == mf.FIND_CASE_N
    # ...and false on the roster the figure set reports, which is why it is now scoped.
    assert int((pub >= thr).sum()) == mf.FIND_FLAGGED_ALL != mf.FIND_CASE_N

    by_id = dict(zip(pids, zip(raw, pub)))
    for (_, pid, published), want_raw in zip(mf.FIND_CASES, mf.FIND_RAW_RISKS):
        got_raw, got_pub = by_id[pid]
        assert got_raw == pytest.approx(want_raw, abs=5e-6)
        assert got_pub == pytest.approx(published, abs=mf.FIND_RISK_TOL)


def test_figure_twos_caption_scopes_the_operating_point_and_names_the_scale(cfg):
    cap = mf.figures(cfg, TEST)["figure2"]["caption"]
    assert f"among the {mf.FIND_CLASSIFIABLE} patients who are classifiable" in cap
    assert f"the remaining {mf.FIND_CENSORED} of {mf.FOREST_OCCLUSION_N} are censored" in cap
    assert f"flags {mf.FIND_FLAGGED_ALL}" in cap
    assert "makes the number of predicted positives equal the number of observed events" \
        not in cap, "that sentence is false on the 734-patient roster the figures report"
    assert "after the frozen horizon-specific recalibration" in cap
    raw_fold = max(mf.FIND_RAW_RISKS) / min(mf.FIND_RAW_RISKS)
    assert f"{raw_fold:.2f}-fold apart on the raw model output" in cap
    assert "on that recalibrated scale" in cap


def test_figure_twos_caption_discloses_the_marker_and_claims_nothing_about_attribution(cfg):
    """D2. The marker is named, its rate and its bound are given, and no more than that.

    The one thing this clause may NOT say is that the attribution maps ignore the marker.
    The persisted Grad-CAM is 16 by 16 native, one cell of which is 32 by 32 crop pixels,
    so it cannot resolve a 40 by 17 px blob, and no integrated-gradient array is persisted
    at all; there is nothing in the repository that could support the claim.
    """
    cap = mf.figures(cfg, TEST)["figure2"]["caption"]
    assert "saturated burned-in character at the upper right of the crop" in cap
    assert "visible in all three rows" in cap
    assert "the one drawn here is the one carrying the largest residual blob" in cap
    assert f"{mf.CROP_RESIDUAL_MARKER_PCT:g} percent of the {mf.CROP_AUDIT_N:,}" in cap
    assert f"{mf.MARKER_DELTA_AUROC:.5f}" in cap
    assert f"{mf.MARKER_DELTA_LO:.5f} to {mf.MARKER_DELTA_HI:.5f}" in cap
    assert "No claim is made here about where either attribution method placed its mass" in cap
    for forbidden in ("ignores the marker", "ignore the marker", "does not attend",
                      "no attribution mass", "attribution falls on the joint"):
        assert forbidden not in cap, f"unsupportable attribution claim: {forbidden!r}"


def test_the_disclosed_marker_is_measured_on_the_tile_the_figure_draws(cfg):
    """D2. If the FN case is ever reselected onto a clean crop, the caption must fail."""
    cases = mf.figure2_cases(cfg)
    n = mf.assert_disclosed_marker(cfg, cases)
    assert n >= int(cfg["preprocess"]["marker_min_px"])
    assert mf.FIND_MARKER_CELL == "FN"
    clean = [dict(c, strip=next(x["strip"] for x in cases if x["cell"] == "TN"))
             if c["cell"] == mf.FIND_MARKER_CELL else c for c in cases]
    with pytest.raises(AssertionError, match="saturated burned-in character"):
        mf.assert_disclosed_marker(cfg, clean)


def test_the_drawn_false_negative_is_the_marked_candidate_not_the_clean_one(cfg):
    """D2. The caption's anti-cherry-picking claim, measured over its own candidate pool.

    Three false negatives were available at this grade. The one drawn carries by far the
    largest residual blob (298 saturated px inside the band; the others 30 and 0), so the
    quartet was not assembled to avoid markers. That is what makes the disclosure credible,
    and it is checked rather than asserted because a future reselection could quietly
    reverse it.
    """
    man = _table("interp_panel_manifest.csv")
    man = man[(man["arm"].astype(str) == mf.INTERP_ARM)
              & (man["klg_contra"].astype(float) == mf.FIND_KLG)]
    assert len(man) == mf.FIND_GRADE_CANDIDATES
    fn = man[man["cell"].astype(str) == mf.FIND_MARKER_CELL]
    assert len(fn) == mf.FIND_FN_CANDIDATES
    blobs = {}
    for _, r in fn.iterrows():
        p = Path(str(r["file"]))
        if not p.exists():
            p = mf.interpretability_dir(cfg) / mf.INTERP_PANEL_DIRNAME / p.name
        blobs[str(r["empi_anon"])] = mf._saturated_inside_band(cfg, _skip_unless(p, p.name))
    drawn = next(pid for cell, pid, _ in mf.FIND_CASES if cell == mf.FIND_MARKER_CELL)
    assert max(blobs, key=lambda k: blobs[k]) == drawn, blobs
    assert blobs[drawn] > 0 and sorted(blobs.values())[-2] < blobs[drawn]


def test_the_ten_point_eight_fold_spread_is_bounded_as_the_maximum_available(cfg):
    """D4. The headline separation is the widest quartet the candidate pool allows.

    The four cases were chosen one per cell out of nine KL-2 panels, and they were chosen
    for the spread. Printing 10.8-fold without saying so invites a reader to take it as
    typical of the grade; it is the maximum. Every one-per-cell combination is enumerated,
    so this is a proof and not a recollection.
    """
    drawn = mf.assert_spread_is_the_widest_available(cfg)
    risks = [r for _, _, r in mf.FIND_CASES]
    assert drawn == pytest.approx(max(risks) / min(risks))
    cap = mf.figures(cfg, TEST)["figure2"]["caption"]
    assert "WIDEST-SPREAD quartet available at this grade" in cap
    assert "no other choice of one per cell spans a larger ratio" in cap
    assert "not a typical one" in cap


def test_a_wider_quartet_than_the_one_drawn_stops_the_render(cfg, monkeypatch):
    """The bound is a real check: shrink the drawn top risk and the claim must fail."""
    shrunk = tuple((c, p, r if c != "TP" else 0.30) for c, p, r in mf.FIND_CASES)
    monkeypatch.setattr(mf, "FIND_CASES", shrunk)
    with pytest.raises(AssertionError, match="widest-spread one available"):
        mf.assert_spread_is_the_widest_available(cfg)


def test_figure_two_selects_the_strips_the_interpretability_run_rendered(cfg):
    cases = mf.figure2_cases(cfg)
    assert [c["cell"] for c in cases] == ["TN", "FN", "FP", "TP"]
    for c in cases:
        assert c["strip"].exists()
        tiles = mf._strip_tiles(c["strip"])
        assert len(tiles) == len(mf.FIND_ROW_LABELS)
        assert all(t.shape[0] == t.shape[1] == tiles[0].shape[0] for t in tiles)


def test_figure_two_refuses_a_case_whose_published_risk_moved(cfg, monkeypatch):
    bad = tuple((c, p, r + 0.05) for c, p, r in mf.FIND_CASES)
    monkeypatch.setattr(mf, "FIND_CASES", bad)
    with pytest.raises(AssertionError, match="published risk"):
        mf.figure2_cases(cfg)


def test_figure_two_refuses_a_case_from_another_grade(cfg, monkeypatch):
    monkeypatch.setattr(mf, "FIND_KLG", 3.0)
    with pytest.raises(AssertionError, match="Kellgren-Lawrence"):
        mf.figure2_cases(cfg)


# --------------------------------------------------------------------------- #
# 12d. Figure 3: the comparison forest                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def forest(cfg):
    for name in ("test_metrics.csv", "test_comparisons.csv",
                 "v6_test_comparisons_posthoc.csv", "interp_view_ablation.csv",
                 "interp_occlusion.csv"):
        _skip_unless(TABLES / name, name)
    return mf.forest_rows(cfg, TEST)


def test_the_forest_panel_a_is_the_metrics_table(cfg, forest):
    a_rows, _ = forest
    met = _table("test_metrics.csv").set_index("arm")
    assert [r["label"] for r in a_rows] == [mf.MODEL_DISPLAY[a] for a in mf.FOREST_ARMS]
    for arm, row in zip(mf.FOREST_ARMS, a_rows):
        assert row["est"] == pytest.approx(float(met.loc[arm, "auc_1825"]))
        assert row["lo"] == pytest.approx(float(met.loc[arm, "auc_1825_lo"]))
        assert row["hi"] == pytest.approx(float(met.loc[arm, "auc_1825_hi"]))
        assert row["n"] == int(met.loc[arm, "n_patients"])


def test_the_forest_caption_does_not_claim_to_draw_every_model_arm(cfg, forest):
    """Panel A draws five arms; the ladder has nine. The caption must say which.

    It read "for each model arm", which is a claim about all nine and is false of the
    picture: the two discrete-time clinical controls, the frontal-only fusion arm and the
    ConvNeXt-Tiny robustness arm are scored, are in Table 3, and are not drawn.
    """
    a_rows, _ = forest
    met = _table("test_metrics.csv")
    assert len(a_rows) == len(mf.FOREST_ARMS) == 5
    assert len(met) == mf.FOREST_ARMS_TOTAL == 9
    cap = mf._forest_caption(mf.caption_context(cfg, TEST))
    assert "for five of the nine model arms" in cap
    assert "for each model arm" not in cap, "the unqualified claim came back"


def test_every_forest_row_carries_the_denominator_it_rests_on(forest):
    a_rows, b_rows = forest
    for row in a_rows + [r for r in b_rows if "group" in ()]:
        assert row["n"] > 0
    for row in b_rows:
        if "group" in row:
            continue
        assert row["n"] > 0
    # the two populations that differ, stated
    ablation = [r for r in b_rows if r.get("colour") == "ablation"]
    assert ablation and all(r["n"] == mf.FOREST_ABLATION_N for r in ablation)
    perturb = [r for r in b_rows if r.get("colour") in ("region", "masking", "control")]
    assert perturb and all(r["n"] == mf.FOREST_OCCLUSION_N for r in perturb)


def test_the_forest_marks_the_degenerate_control_and_nothing_else(forest):
    _, b_rows = forest
    deg = [r for r in b_rows if r.get("degenerate")]
    assert len(deg) == 1
    assert deg[0]["est"] == pytest.approx(-0.339, abs=5e-4)
    assert deg[0]["shade"] is True
    assert not (mf.FOREST_XLIM[0] <= deg[0]["est"] <= mf.FOREST_XLIM[1]), (
        "the degenerate control is drawn off scale on purpose; inside the range it reads as "
        "a leakage result forty times larger than any of them")


def test_the_pipeline_check_block_is_shaded_and_named_as_one(forest, cfg):
    _, b_rows = forest
    heads = [r["group"] for r in b_rows if "group" in r]
    assert "Pipeline checks, NOT leakage tests" in heads
    shaded = [r for r in b_rows if r.get("shade")]
    assert len(shaded) == 1 + len(mf.FOREST_CONTROL_ROWS)   # the heading and its rows
    cap = mf.figures(cfg, TEST)["figure3"]["caption"]
    assert "are pipeline checks and not leakage tests" in cap
    assert "says nothing about burned-in text" in cap
    assert "The evidence about text is the two widened bands and the" in cap


def test_the_forest_caption_forbids_a_compartment_ranking(cfg, forest):
    _, b_rows = forest
    for row in [r for r in b_rows if r.get("colour") == "region"]:
        assert row["lo"] <= 0.0 <= row["hi"], row["label"]
    assert "must not be ranked against one another" in \
        mf.figures(cfg, TEST)["figure3"]["caption"]


def test_the_zero_crossing_claim_is_checked_against_the_whole_table_not_the_drawn_rows(cfg):
    """D3. The claim used to be universal and was verified only over the drawn subset.

    ``Every anatomic occlusion interval crosses zero`` was false of the table it describes:
    the occlusion table holds SEVEN anatomic conditions and the figure draws five, and one
    of the two undrawn ones, the mean-filled joint-only occlusion, has an interval that
    excludes zero and says so in its own note column. The guard iterated over
    ``forest_rows``' drawn rows, so it could only ever re-derive the reason those five were
    the ones drawn. It now reads the table.

    The undrawn exception is the condition most FAVOURABLE to the model, so scoping the
    sentence without naming it would have been selection in the other direction. The
    caption does both: it scopes the claim to the rows on the page and states the exception
    with its numbers.
    """
    occl = _table("interp_occlusion.csv")
    occl = occl[occl["arm"].astype(str) == mf.INTERP_ARM]
    fam = occl[occl["condition"].astype(str).isin(mf.FOREST_ANATOMIC_CONDITIONS)]
    assert len(fam) == len(mf.FOREST_ANATOMIC_CONDITIONS) == 7
    drawn = {c for c, _ in mf.FOREST_REGION_ROWS}
    assert len(drawn) == 5 and drawn < set(mf.FOREST_ANATOMIC_CONDITIONS)
    excludes = {str(r["condition"]) for _, r in fam.iterrows()
                if not (float(r["delta_auroc_lo"]) <= 0.0 <= float(r["delta_auroc_hi"]))}
    assert excludes == {mf.FOREST_MEANFILL_CONDITION}, sorted(excludes)
    assert not (excludes & drawn), "a drawn anatomic row stopped crossing zero"
    # The table's own note says the interval excludes zero; the caption must not disagree.
    note = str(fam[fam["condition"].astype(str)
                   == mf.FOREST_MEANFILL_CONDITION]["note"].iloc[0])
    assert "interval excludes zero" in note

    cap = mf.figures(cfg, TEST)["figure3"]["caption"]
    assert "Every anatomic occlusion interval drawn here crosses zero" in cap
    assert "Every anatomic occlusion interval crosses zero" not in cap, (
        "the unqualified sentence is false of the table this figure reads")
    assert "the only anatomic condition in the table whose interval excludes zero" in cap
    assert f"lowers 5-year discrimination by {mf.FOREST_MEANFILL_DELTA:.3f}" in cap
    assert f"{mf.FOREST_MEANFILL_LO:.3f} to {mf.FOREST_MEANFILL_HI:.3f}" in cap
    assert "most favourable to the model" in cap
    row = fam[fam["condition"].astype(str) == mf.FOREST_MEANFILL_CONDITION].iloc[0]
    assert -float(row["delta_auroc"]) == pytest.approx(mf.FOREST_MEANFILL_DELTA, abs=5e-4)
    assert -float(row["delta_auroc_hi"]) == pytest.approx(mf.FOREST_MEANFILL_LO, abs=5e-4)
    assert -float(row["delta_auroc_lo"]) == pytest.approx(mf.FOREST_MEANFILL_HI, abs=5e-4)


def test_a_second_anatomic_exception_stops_the_render_rather_than_the_caption(cfg):
    """The scoped sentence names ONE exception; a second one must raise, not be printed."""
    occl = _table("interp_occlusion.csv")
    bad = occl.copy()
    sel = ((bad["arm"].astype(str) == mf.INTERP_ARM)
           & (bad["condition"].astype(str) == "occlude_joint_meanfill"))
    bad.loc[sel, "delta_auroc_hi"] = -0.001
    with pytest.raises(AssertionError, match="exactly one anatomic condition"):
        mf._assert_occlusion_exceptions(bad)


def test_a_drawn_anatomic_row_that_stops_crossing_zero_stops_the_render(cfg):
    occl = _table("interp_occlusion.csv")
    bad = occl.copy()
    sel = ((bad["arm"].astype(str) == mf.INTERP_ARM)
           & (bad["condition"].astype(str) == "occlude_medial"))
    bad.loc[sel, "delta_auroc_hi"] = -0.001
    with pytest.raises(AssertionError, match="drawn anatomic condition"):
        mf._assert_occlusion_exceptions(bad)


def test_the_forest_caption_states_the_degeneracy_counts_the_table_carries(cfg):
    occl = _table("interp_occlusion.csv")
    occl = occl[occl["arm"].astype(str) == mf.INTERP_ARM].set_index("condition")
    assert int(occl.loc["keep_border_only", "n_max_tied_risk"]) == mf.FOREST_DEGENERATE_TIED
    assert int(occl.loc["keep_border_only", "n_distinct_risk"]) == \
        mf.FOREST_DEGENERATE_DISTINCT
    assert int(occl.loc["occlude_border", "n_identical_to_baseline"]) == \
        mf.FOREST_BORDER_IDENTICAL
    cap = mf.figures(cfg, TEST)["figure3"]["caption"]
    for n in (mf.FOREST_DEGENERATE_TIED, mf.FOREST_DEGENERATE_DISTINCT,
              mf.FOREST_BORDER_IDENTICAL, mf.FOREST_OCCLUSION_N, mf.FOREST_ABLATION_N):
        assert str(n) in cap


def test_the_forest_ablation_denominator_is_not_the_multi_view_stratum(cfg):
    """315, 316, 321 and 322 are four different counts and only one of them is right."""
    ab = _table("interp_view_ablation.csv")
    assert set(ab["n_patients_common"].astype(int)) == {mf.FOREST_ABLATION_N}
    sub = _table("test_subgroups.csv")
    multi = sub[sub["level"].astype(str) == "Multiple views"]
    if len(multi):
        assert int(multi["n_patients"].iloc[0]) != mf.FOREST_ABLATION_N
    cap = mf.figures(cfg, TEST)["figure3"]["caption"]
    assert "contributed both a frontal and at least one non-frontal radiograph" in cap
    assert "322" not in cap


def test_forest_rows_refuse_a_selector_that_names_no_row(cfg, monkeypatch):
    monkeypatch.setattr(mf, "FOREST_REGION_ROWS", (("no_such_condition", "nowhere"),))
    with pytest.raises(AssertionError, match="selects 0 rows"):
        mf.forest_rows(cfg, TEST)


# --------------------------------------------------------------------------- #
# 12e. Figure 4: calibration and risk separation                               #
# --------------------------------------------------------------------------- #
def test_the_frozen_recalibration_is_read_from_train_arms_not_reimplemented(cfg, cohort_dir):
    _skip_unless(cohort_dir / "train_arms.json", "train_arms.json")
    arms = json.loads((cohort_dir / "train_arms.json").read_text())["arms"]
    for arm in ("m2_frontal", "m4_fusion"):
        got = mf.frozen_recalibration(cfg, arm, 1825.0)
        want = arms[arm]["recalibration"]["1825.0"]
        assert got == dict(want)
        assert float(got["slope"]) > 0
    # the frozen Cox comparators were published as fitted and carry no transform
    for arm in mf.COX_ARMS:
        assert mf.frozen_recalibration(cfg, arm, 1825.0) is None


def test_apply_frozen_recalibration_is_the_training_modules_own_function(cfg, cohort_dir):
    _skip_unless(cohort_dir / "train_arms.json", "train_arms.json")
    from src.train_model import apply_recalibration
    recal = mf.frozen_recalibration(cfg, "m4_fusion", 1825.0)
    p = np.array([0.01, 0.05, 0.2, 0.5, 0.9])
    np.testing.assert_allclose(mf.apply_frozen_recalibration(p, recal),
                               apply_recalibration(p, recal))
    # and it MOVES the predictions: the transform is substantial, not cosmetic
    assert np.max(np.abs(mf.apply_frozen_recalibration(p, recal) - p)) > 0.02


def test_a_recalibration_that_would_reverse_the_ranking_is_refused():
    with pytest.raises(AssertionError, match="not monotone increasing"):
        mf.apply_frozen_recalibration(np.array([0.1, 0.2]),
                                      {"intercept": 0.0, "slope": -1.0})


def test_figure_fours_caption_says_the_recalibration_is_applied(cfg):
    for split in (VAL, TEST):
        cap = mf.figures(cfg, split)["figure4"]["caption"]
        assert "drawn AFTER the horizon-specific recalibration" in cap
        assert "frozen Cox comparators carry no such transform" in cap
        assert "before the horizon-specific recalibration" not in cap


def test_figure_fours_second_panel_is_the_within_stratum_tertile_table(cfg):
    rows = mf.klg_tertile_rows(cfg)
    tab = _table("v6_klg_risk_tertiles.csv")
    tab = tab[(tab["arm"].astype(str) == mf.CAL_ARM)
              & (tab["scheme"].astype(str) == mf.CAL_SCHEME)]
    assert [r["stratum"] for r in rows] == list(mf.CAL_STRATA)
    assert "All KL grades" not in [r["stratum"] for r in rows], (
        "the pooled row double-counts the strata beside it")
    for r in rows:
        sub = tab[tab["stratum"].astype(str) == r["stratum"]].sort_values("tertile")
        np.testing.assert_allclose(r["cif"], sub["km_cumulative_incidence"].to_numpy())
        assert r["n"] == int(sub["n_stratum_patients"].iloc[0])
        assert r["events"] == int(sub["n_stratum_events"].iloc[0])


def test_the_uninformative_stratum_carries_its_event_count_on_the_caption(cfg):
    rows = {r["stratum"]: r for r in mf.klg_tertile_rows(cfg)}
    low = rows[mf.CAL_LOW_STRATUM]
    assert (low["n"], low["events"]) == (mf.CAL_LOW_N, mf.CAL_LOW_EVENTS)
    cap = mf.figures(cfg, TEST)["figure4"]["caption"]
    assert f"{mf.CAL_LOW_N} patients" in cap
    assert "uninformative cell rather than a demonstrated null" in cap
    assert "post hoc" in cap and "D35" in cap


def test_figure_four_refuses_a_moved_low_stratum(cfg, monkeypatch):
    monkeypatch.setattr(mf, "CAL_LOW_EVENTS", 99)
    with pytest.raises(AssertionError, match="the caption states"):
        mf.klg_tertile_rows(cfg)


# --------------------------------------------------------------------------- #
# 12f. Supplementary S1, S5 and S6                                             #
# --------------------------------------------------------------------------- #
def test_s1_marks_the_retained_epoch_on_every_series():
    curves = _table("v6_learning_curves.csv")
    per_seed = _table("v6_learning_curves_by_seed.csv")
    n = int(curves.groupby(["arm", "seed"]).ngroups)
    assert n == mf.S1_SERIES
    assert int(curves["is_retained_epoch"].astype(bool).sum()) == n
    assert set(per_seed["epochs_after_retained"].astype(int)) == {mf.S1_PATIENCE}


def test_s1_caption_separates_the_two_gaps_and_calls_them_lower_bounds(cfg):
    cap = mf.supplement_figures(cfg, TEST)["figureS1"]["caption"]
    assert f"{mf.S1_PATIENCE} epochs past it" in cap
    assert f"{mf.S1_SERIES} series" in cap
    assert "which is a different measurement" in cap
    assert "LOWER BOUND on the train-validation gap" in cap
    assert "the model we kept was diverging" not in cap


def test_s5_draws_three_states_and_leaves_out_acquisition_era(cfg):
    df = _table("v6_robustness_strata.csv")
    families = set(df["family"].astype(str))
    for fam in mf.S5_FAMILIES:
        assert fam in families, fam
    assert not any(f.startswith("acquisition_era") for f in mf.S5_FAMILIES), (
        "era is confounded with follow-up and rests on a shifted date; see deviation D17")
    assert "image_crop_method" not in mf.S5_FAMILIES, (
        "crop method and crop confidence are the same partition")
    drawn = df[(df["arm"].astype(str) == mf.INTERP_ARM)
               & (df["metric"].astype(str) == "auc")
               & (df["family"].astype(str).isin(mf.S5_FAMILIES))]
    reasons = drawn["suppression_reason"].fillna("").astype(str)
    assert (~drawn["suppressed"]).any(), "no estimable level"
    assert reasons.str.startswith("protocol section 21").any(), "no floor-suppressed level"
    assert reasons.str.startswith("not estimable").any(), (
        "S5 must include at least one level whose estimator is undefined, or its caption "
        "describes a state the figure never shows")
    cap = mf.supplement_figures(cfg, TEST)["figureS5"]["caption"]
    assert "undefined, not imprecise" in cap
    assert str(mf.S5_CONTROLS_BEYOND_HORIZON) in cap
    assert "Equipment, manufacturer and acquisition site" in cap


def test_s5_control_anchor_is_the_splits_own_follow_up(cfg, frames):
    """S5's caption says how many patients in the SPLIT reach the horizon, not how many the
    arm scores; the two differ, so the anchor is checked against the follow-up itself."""
    _, fr = frames
    horizon = float(int(cfg["model_eval"]["horizons_days"][-1]))
    beyond = int((fr[TEST]["time_from_landmark"].to_numpy(dtype=float) > horizon).sum())
    assert beyond == mf.S5_CONTROLS_BEYOND_HORIZON
    assert beyond < mf.SPLIT_ANCHORS[TEST]["n"]


def test_s6_denominators_and_percentages_match_the_attention_table(cfg):
    paired = _table("interp_attention_paired.csv")
    pair = paired[paired["comparison"].astype(str) == "pairwise_one_crop_each"]
    triple = paired[paired["comparison"].astype(str) == "all_three_views_present"]
    assert set(pair["n_patients"].astype(int)) == {mf.S6_PAIR_N}
    assert set(triple["n_patients"].astype(int)) == {mf.S6_TRIPLE_N}
    by_arm = {str(r["arm"]): r for _, r in pair.iterrows()}
    assert float(by_arm["m3_image"]["pct_b_outweighs_a"]) == pytest.approx(mf.S6_M3_PCT,
                                                                          abs=0.05)
    assert float(by_arm["m4_fusion"]["pct_b_outweighs_a"]) == pytest.approx(mf.S6_M4_PCT,
                                                                            abs=0.05)
    # every three-view row sums to one, which is the first thing a reviewer checks
    for _, r in triple.iterrows():
        total = sum(float(r[f"weight_{c}_mean"]) for c in "abc")
        assert abs(total - 1.0) < 5e-4, (r["arm"], total)


def test_s6_caption_does_not_claim_the_extra_views_are_ignored(cfg):
    cap = mf.supplement_figures(cfg, TEST)["figureS6"]["caption"]
    assert "The aggregator is not ignoring the additional views" in cap
    assert "not a statement that the second view is uninformative" in cap
    assert f"{mf.S6_PAIR_N} patients" in cap and f"{mf.S6_TRIPLE_N} patients" in cap


# --------------------------------------------------------------------------- #
# 12g. The supplementary render, its manifest and the submission bundle        #
# --------------------------------------------------------------------------- #
def test_a_supplementary_render_writes_its_own_manifest_under_its_own_schema(cfg, tmp_path,
                                                                             monkeypatch):
    monkeypatch.setattr(mf, "SUPPLEMENT_RENDERERS",
                        {k: _stub_renderer(k) for k in mf.SUPPLEMENT_KEYS})
    written = mf.render_supplement(cfg, out_dir=tmp_path, split=TEST)
    assert set(written) == set(mf.SUPPLEMENT_KEYS)
    m = _manifest(tmp_path / mf.SUPPLEMENT_DIRNAME)
    assert m["schema"] == mf.SUPPLEMENT_MANIFEST_SCHEMA
    assert m["split"] == TEST and m["complete"] is True
    assert [e["key"] for e in m["rendered"]] == list(mf.SUPPLEMENT_KEYS)
    # and the MAIN manifest, which src/make_manuscript.py reads, is untouched by it
    assert not (tmp_path / mf.FIGURES_MANIFEST).exists()


def test_the_supplementary_render_never_writes_beside_the_main_figures(cfg, tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr(mf, "SUPPLEMENT_RENDERERS",
                        {k: _stub_renderer(k) for k in mf.SUPPLEMENT_KEYS})
    mf.render_supplement(cfg, out_dir=tmp_path, split=TEST)
    assert sorted(p.name for p in tmp_path.iterdir()) == [mf.SUPPLEMENT_DIRNAME]


def test_the_tertile_table_still_reaches_outputs_tables_from_the_supplement_directory(cfg):
    """The one repository table a renderer writes, after that renderer moved to S4."""
    figures_dir = cfg.path(cfg["manuscript"]["figures_dir"])
    for out in (None, figures_dir, figures_dir / mf.SUPPLEMENT_DIRNAME):
        assert mf.risk_tertiles_output_path(cfg, TEST, out) == \
            mf.risk_tertiles_path(cfg, TEST)


def test_a_scratch_supplement_render_still_keeps_the_table_beside_its_images(cfg, tmp_path):
    got = mf.risk_tertiles_output_path(cfg, TEST, tmp_path / mf.SUPPLEMENT_DIRNAME)
    assert got.parent == tmp_path / mf.SUPPLEMENT_DIRNAME
    assert got.name == mf.risk_tertiles_path(cfg, TEST).name


def _named_stub(spec: dict):
    """A stub that writes the REGISTRY filename, which is what the bundle copies by."""
    def _r(cfg, out_dir, split):
        p = Path(out_dir) / str(spec["filename"])
        p.write_bytes(str(spec["filename"]).encode())
        return p
    return _r


def test_the_submission_bundle_names_every_copy_by_its_figure_number(cfg, tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(mf, "RENDERERS",
                        {k: _named_stub(v) for k, v in mf.figures(cfg, TEST).items()})
    monkeypatch.setattr(mf, "SUPPLEMENT_RENDERERS",
                        {k: _named_stub(v)
                         for k, v in mf.supplement_figures(cfg, TEST).items()})
    edited = Config(cfg)
    edited["paths"] = {**dict(cfg["paths"]), "figures_dir": str(tmp_path / "figures")}
    mf.render_all(edited, out_dir=tmp_path / "main", split=TEST)
    mf.render_supplement(edited, out_dir=tmp_path / "main", split=TEST)
    dest = mf.write_submission_bundle(edited, TEST, tmp_path / "main")
    names = sorted(p.name for p in dest.glob("*.png"))
    assert names == [f"Figure{i}.png" for i in (1, 2, 3, 4)] + \
        [f"Supplementary-Figure-S{i}.png" for i in (1, 2, 3, 4, 5, 6)]
    readme = (dest / mf.SUBMISSION_README).read_text()
    for spec in mf.figures(edited, TEST).values():
        assert str(spec["filename"]) in readme, "the README names the repository file"
    for p in dest.glob("*.png"):
        assert mf.sha256_file(p) in readme


def test_no_registry_filename_describes_a_figure_it_is_not(cfg):
    """D9. The legacy name is gone, and no other filename may drift from its content.

    ``figure2`` shipped as ``figure2_discrimination_calibration.png`` at v6 while carrying
    the representative image findings, because
    tests/test_make_manuscript.py::test_verify_fires_when_an_image_does_not_hash_to_what_was_
    recorded typed that literal into an assertion and the renaming task could not edit that
    file. Both halves moved together on 2026-08-11: the registry entry is renamed and that
    assertion now reads the name off the registry. This test replaces the one that recorded
    the workaround, and it is a stronger rule than the one it replaces - every main and
    supplementary filename must start with its own key, so no future move can leave a name
    behind again.
    """
    d = {x.key: x for x in mf.FIGURE_DEFS}
    assert d["figure2"].filename == "figure2_representative_findings.png"
    assert d["figure2"].caption is mf._find_caption
    src = Path(__file__).resolve().parents[1] / "tests" / "test_make_manuscript.py"
    code = [ln for ln in src.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    typed = [ln for ln in code if "figure2_discrimination_calibration.png" in ln]
    assert not typed, (
        "the v5 filename is typed into the make_manuscript suite again; it and the "
        f"registry entry move together or not at all: {typed}")
    for defs in (mf.FIGURE_DEFS, mf.SUPPLEMENT_DEFS):
        for x in defs:
            assert x.filename.startswith(f"{x.key}_") and x.filename.endswith(".png"), (
                f"{x.key} is filed as {x.filename!r}, which does not name its own key")
