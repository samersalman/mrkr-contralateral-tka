"""manuscript_figures.py: the manuscript figures, rendered from frozen artefacts.

MRKR Contralateral TKA. This module renders exactly the figures the manuscript embeds and
nothing else. It is the only module allowed to write into ``manuscript.figures_dir``;
``src/make_manuscript.py`` consumes :func:`figures` for the numbering, titles and captions
and never re-derives them. It writes exactly one table, and only because a figure computed
it: ``{split}_risk_tertiles.csv`` (see figure 3 below). Everything else under
``outputs/tables/`` belongs to ``src/eval_models.py``.

Every figure is a function of the SPLIT being reported. ``--split val`` renders the
development-split figures; ``--split test`` renders the sealed split, and does so only after
:func:`src.eval_models.assert_sealed_read_is_recorded` has confirmed that the single
permitted sealed read is on the record and that the models scored on it are still the models
on disk. The default comes from ``manuscript.report_split``, so the config and the two render
modules cannot disagree about which split the paper is about. Artefact FILENAMES are resolved
through :func:`src.eval_models.split_path`, never by a second copy of the ``val_`` to
``test_`` rewrite.

Figures implemented:
  * Figure 1 (protocol section 16, cohort assembly): patient flow for the PRIMARY landmark
    arm, from the 83,011-patient source registry down to the rendered split's patients that
    actually carry a contralateral crop. The first nine boxes replay
    ``outputs/cohort_flow.csv`` (the Phase 1 assembly ledger, which stops at the 3,709-patient
    landmark cohort); the last two are recomputed at render time from the LOCKED split table
    and the shard label index so that a silent cohort change fails the render instead of
    quietly redrawing itself.
  * Figure 2 (protocol sections 18 and 20, discrimination and calibration): IPCW
    cumulative/dynamic time-dependent AUROC at the three pre-specified horizons, and
    5-year calibration by quintile of predicted risk, for the four arms in
    :data:`FIG2_MODELS` - clinical, clinical plus inferred radiographic grade, one frontal
    radiograph, and the combined model.
  * Figure 3 (protocol section 19, absolute risk): Kaplan-Meier cumulative incidence by
    tertile of predicted 5-year M4 risk, with a number-at-risk row. It also EMITS the
    tertile summary it draws, as ``outputs/tables/{split}_risk_tertiles.csv``, because the
    5-year incidence in the lowest and highest tertile is the number a surgeon acts on and
    it therefore has to be quotable in the Results prose. The manuscript reads that CSV and
    recomputes nothing - the same rule figure 4 follows in the other direction.
  * Figure 4 (protocol section 18, exploratory decision-curve analysis): net benefit
    against threshold probability for the arms in ``model_eval.net_benefit.arms``, and the
    protagonist arm's paired differences against treat-all and against the reference arm.
    It draws from ONE artefact, ``outputs/tables/{split}_net_benefit.csv``, and recomputes
    nothing; ``src/eval_models.py`` owns that estimator and the shared bootstrap draw that
    makes the differences paired. It is also the one figure that may DECLINE to draw, on a
    split whose protagonist arm the convergence gate disqualifies, and it settles that
    question from ``{split}_convergence.csv`` BEFORE it requires the table it would draw -
    so a split with no honest figure 4 does not take the other three down with it.

Assumptions bought, stated so a reader can price them:
  * NO title is baked into any image. Titles and captions live in :func:`figures` and are
    placed by the manuscript generator, which is what a journal expects. Axis labels, tick
    labels, legends, panel letters and the number-at-risk annotation ARE drawn, because they
    are part of the plot rather than part of the caption.
  * Figure 2 panel B plots the ensembled predictions BEFORE the horizon-specific
    recalibration. On validation that recalibration is estimated on the same patients and
    plotting after it would be circular; on test it is a validation-fitted, strictly monotone
    transform, so it moves the markers along the x axis without changing the ranking.
  * Figure 2 panel B and Figure 3 are drawn on the patients the plotted arms all score. The
    arms rest on different populations (on the sealed split: 741 clinical, 740 all-view
    image, 734 frontal-only, 707 KLG-eligible), and those per-arm denominators are reported
    in Table 2 rather than harmonised away; the one denominator a shared panel needs is
    stated in the caption and asserted against :data:`SPLIT_ANCHORS` at render time.
  * Figure 3 is cause-agnostic. Death is not ascertainable in this data source (protocol
    section 10), so mortality is an UNMEASURED competing event. That caveat is caption text,
    never image text.

Provenance: a whole-module render also writes ``figures_manifest.json`` into the directory
it rendered into, recording the split, the sealed read's training contract hash and the
sha256 of every image it wrote. ``src/make_manuscript.py`` embeds bytes it did not draw and
has no other way to tell a validation figure 1 from a sealed-split one of the same
filename; its ``verify()`` reads this file and refuses to build on images it does not
describe. See the section comment above :func:`build_manifest`.

Data hygiene: every artefact written here is aggregate. No ``empi_anon`` and no DICOM UID
reaches ``outputs/``; patient identifiers are used in memory only, to align the per-patient
hazard arrays (which live under the git-ignored ``derived-data/cohort/``) with the frozen Cox
replays.

Interpreter: the torch venv (Python 3.12), which carries matplotlib, lifelines and pandas::

    cd <project root>
    ~/.venvs/mrkr-torch/bin/python -m src.manuscript_figures --config config/feasibility.yaml
    ~/.venvs/mrkr-torch/bin/python -m src.manuscript_figures --split val --only figure1

Figure 1 depends only on artefacts that already exist, so ``--only figure1`` renders today.
Figures 2, 3 and 4 need ``src/train_model.py`` and ``src/eval_models.py`` (or
``src/score_test.py`` on the sealed split) to have run first and raise a named error until
they have.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import textwrap
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402  (backend must be set first)
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle   # noqa: E402

from src.config import Config, load_config         # noqa: E402

# ``src.eval_models`` pulls in ``src.model_clinical``, which pulls in ``src.features_clinical``
# and ``src.followup``; both import duckdb at module level for the Phase 1 ingestion path.
# Nothing on the figure path calls duckdb, so a namespace stub keeps this module importable in
# an interpreter that does not carry it. See :func:`_import_model_clinical`, which installs the
# same stub for the lazy import.
try:                                               # noqa: E402
    import duckdb                                  # noqa: F401
except ModuleNotFoundError:                        # pragma: no cover - duckdb is installed here
    sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))

from src.eval_models import (                      # noqa: E402
    NET_BENEFIT_COLUMNS,
    SEALED_SPLIT,
    STATUS_NO_CONVERGE,
    STATUS_OK,
    STATUS_OVERFIT,
    VAL_SPLIT,
    assert_sealed_read_is_recorded,
    net_benefit_settings,
    split_path,
    write_table,
)

MODULE = "manuscript_figures"

# --------------------------------------------------------------------------- #
# LOCKED regression anchors. Deliberately NOT in config, exactly as in                        #
# src/model_clinical.py: a config edit must not be able to weaken the guard that              #
# would catch a silently-changed cohort. Sealed-split values come from FROZEN                 #
# METADATA (clinical_imputation_params.json) and from the shard label index,                  #
# never from sealed outcome rows.                                                             #
#                                                                                             #
# This is now the ONLY place the sealed cohort's 741 / 106 / 740 / 1216 is                    #
# verified: eval_models.load_roster skips its own anchor check on the sealed                  #
# split, so if these move nothing else in the repository notices.                             #
# --------------------------------------------------------------------------- #
SPLITS = (VAL_SPLIT, SEALED_SPLIT)                # the splits a figure may be rendered for

# Per split: patients, events, patients carrying at least one usable contralateral crop,
# events among those, and crops written into that split's shard directory.
# ``panel_b_*`` is the FIG2_MODELS intersection - the patients every arm in figure 2 scores -
# and is asserted against the rendered set so the caption cannot drift from the image.
SPLIT_ANCHORS: dict[str, dict[str, int]] = {
    "train":      {"n": 2597, "events": 373, "crop_n": 2595, "crop_events": 373, "crops": 4254},
    VAL_SPLIT:    {"n": 371, "events": 54, "crop_n": 371, "crop_events": 54, "crops": 601,
                   "panel_b_n": 359, "panel_b_events": 52},
    SEALED_SPLIT: {"n": 741, "events": 106, "crop_n": 740, "crop_events": 106, "crops": 1216,
                   "panel_b_n": 707, "panel_b_events": 98},
}
# Which splits' patients a render describes. A validation render draws the development
# cohort (the patients every model was fitted and selected on); a sealed render draws the
# test cohort. Every crop-side anchor below is the table above RESTRICTED to these.
RENDERED_SPLITS: dict[str, tuple[str, ...]] = {
    VAL_SPLIT: ("train", VAL_SPLIT),
    SEALED_SPLIT: (SEALED_SPLIT,),
}
# The word each split goes by in prose. "val" never appears in a caption or a box.
SPLIT_WORD = {"train": "training", VAL_SPLIT: "validation", SEALED_SPLIT: "test"}
SPLIT_WORD_CAP = {k: v.capitalize() for k, v in SPLIT_WORD.items()}
# ``model_image.local`` key holding the shard directory whose labels.csv describes a split.
SHARD_DIR_KEY = {VAL_SPLIT: "shard_dir", SEALED_SPLIT: "test_shard_dir"}

EXPECTED_N_LANDMARK = 3709                       # final primary landmark cohort
_DEV_SPLITS = RENDERED_SPLITS[VAL_SPLIT]
# Every split whose rows exist at all, in cohort order. ``readable_splits`` decides which of
# them a given render may hold; this is only the universe it chooses from.
ALL_SPLITS = (*_DEV_SPLITS, SEALED_SPLIT)

# --------------------------------------------------------------------------- #
# DECISION-CURVE ANCHORS (figure 4). Same argument as SPLIT_ANCHORS above and   #
# the same rule: a number a caption states is written down ONCE, in code, and   #
# checked against the artefact at render time or in the test suite.             #
#                                                                               #
#   prevalence          the split's Kaplan-Meier 5-year cumulative incidence.   #
#                       Treat-all crosses zero EXACTLY here, so render_figure4  #
#                       recovers it from the net-benefit table's own            #
#                       nb_treat_all_same_set column and asserts this value.    #
#   arm_n / arm_events  the decision-curve protagonist's own scored set. The    #
#                       four arms rest on deliberately different populations;   #
#                       the two extremes are already anchored (SPLIT_ANCHORS    #
#                       "n"/"events" for the full-cohort reference and          #
#                       "panel_b_n"/"panel_b_events" for the set every arm      #
#                       scores) and this is the one in between.                 #
#   arm_citl            the protagonist's calibration in the large at the long  #
#   reference_citl      horizon, and the reference arm's, from                  #
#                       {split}_metrics.csv. Net benefit is calibration         #
#                       sensitive in a way AUROC is not, so both belong in the  #
#                       caption rather than in a reviewer's letter.             #
#   reference_max_risk  the reference arm's largest predicted risk. Its curve   #
#                       is identically zero above this, which is one of the     #
#                       reasons the plotted range stops where it does.          #
# --------------------------------------------------------------------------- #
#
# The two calibration entries are transcribed from {split}_metrics.csv at the precision that
# file carries, so the caption and Table 2 quote ONE number. Re-scoring a split moves them,
# and the test suite compares them against that file rather than against a copy of itself.
#
# PROVENANCE OF THE VALIDATION arm_citl (read this before "fixing" it). 0.026129 is what
# outputs/tables/val_metrics.csv states for the protagonist at the long horizon, and it is
# deliberately kept in step with that FILE rather than recomputed here, because Table 2 reads
# the same file and the two must print one number. That file is STALE with respect to the
# current hazard artefacts: replaying val_hazards_m2_frontal.npz through
# train_model.risk_at_horizon and the frozen cloglog recalibration in train_arms.json gives a
# calibration in the large of +0.041266 against the same 0.200080 Kaplan-Meier risk, a 58%
# relative move. Nothing reaches a reader today, because figure 4 is not drawn on validation
# at all: the convergence gate disqualifies its protagonist there, which is the reason, and
# val_net_benefit.csv not existing is merely the state of the disk (see
# render_figure4 and decision_curve_decline_reason). WHEN VALIDATION IS
# RE-SCORED, test_nb_calibration_anchors_match_the_metrics_table will start failing: that is
# the guard working, NOT a regression. Move this entry to whatever the rewritten
# val_metrics.csv then states, in the same commit as the re-score, and do not relax the test.
# The sealed entries carry no such gap: test_metrics.csv and test_net_benefit.csv were
# written by one run.
NB_ANCHORS: dict[str, dict[str, float]] = {
    VAL_SPLIT:    {"prevalence": 0.1996, "arm_n": 370, "arm_events": 54,
                   "arm_citl": 0.026129, "reference_citl": -0.010424,
                   "reference_max_risk": 0.5546},
    SEALED_SPLIT: {"prevalence": 0.2004, "arm_n": 734, "arm_events": 106,
                   "arm_citl": 0.055538, "reference_citl": -0.000561,
                   "reference_max_risk": 0.4715},
}
# The frozen complementary log-log recalibration the IMAGE arms carry at the long horizon,
# fitted on validation and applied unchanged to both splits (derived-data/cohort/
# train_arms.json, arm m2_frontal, day 1825). It is strictly increasing, which is the whole
# reason recalibration cannot change which patients a threshold flags.
NB_RECAL_SLOPE = 1.562
# The prevalence anchor is quoted to four decimals, which is the precision BOTH the caption
# and the annotation drawn on the image print. Half a unit in that last place is therefore
# the tolerance at which the two can still never print different strings.
#
# It is a ROUNDING RULE, not an error budget, and the difference matters when someone reads
# the margin. The sealed table's treat-all column inverts to 0.200375 against the stated
# 0.2004, a gap of 2.5e-5, which is half this tolerance and reads like a near miss; it is
# not. Both values print "0.2004", which is the entire claim the assertion makes. Any value
# that prints the same string is within 5e-5 by construction and any value outside 5e-5
# prints a different one, so tightening this number would make the guard fire on figures a
# reader cannot tell apart, and loosening it would let the caption and the annotation
# disagree on the page. test_nb_prevalence_tolerance_is_the_half_ulp_of_the_printed_precision
# pins that equivalence, so the constant cannot be "tuned" without the reason failing.
NB_PREVALENCE_TOL = 5e-5

# Legacy names, DERIVED from the table above so the assertion messages that quote them stay
# verbatim and there is still exactly one place a number is written down.
EXPECTED_SPLIT_N = {s: a["n"] for s, a in SPLIT_ANCHORS.items()}
EXPECTED_SPLIT_EVENTS = {s: a["events"] for s, a in SPLIT_ANCHORS.items()}
EXPECTED_DEV_N = sum(SPLIT_ANCHORS[s]["n"] for s in _DEV_SPLITS)            # train + val
EXPECTED_DEV_EVENTS = sum(SPLIT_ANCHORS[s]["events"] for s in _DEV_SPLITS)
EXPECTED_CROP_N = sum(SPLIT_ANCHORS[s]["crop_n"] for s in _DEV_SPLITS)
EXPECTED_CROP_SPLIT_N = {s: SPLIT_ANCHORS[s]["crop_n"] for s in _DEV_SPLITS}
EXPECTED_CROP_SPLIT_EVENTS = {s: SPLIT_ANCHORS[s]["crop_events"] for s in _DEV_SPLITS}
EXPECTED_N_CROPS = sum(SPLIT_ANCHORS[s]["crops"] for s in _DEV_SPLITS)
EXPECTED_CROPS_BY_SPLIT = {s: SPLIT_ANCHORS[s]["crops"] for s in _DEV_SPLITS}

CALIBRATION_BINS = 5                             # quintiles, matching src/model_clinical.py
RISK_TERTILES = 3                                # protocol section 19

# The four models figure 2 compares: clinical, clinical plus a structured radiographic
# severity grade, one frontal radiograph, and radiograph plus clinical. That ladder IS the
# paper's argument, so it belongs in one panel. Display strings only: the fitted quantities
# all come from config-driven artefacts.
FIG2_MODELS = ("m0", "m1", "m2_frontal", "m4_fusion")
FIG3_MODEL = "m4_fusion"
# The frozen penalized-Cox comparators, replayed here rather than re-fitted. m1 is defined on
# the KLG-eligible subset (protocol Secondary objective 2); nothing here imputes a grade.
COX_ARMS = ("m0", "m1")
COX_MODEL_JSON = {"m0": "m0_clinical_model.json", "m1": "m1_klg_model.json"}
COX_ELIGIBLE_COLUMN = {"m0": None, "m1": "klg_contra_missing"}
# Arms whose scored set is every patient in the frame, and arms whose scored set is every
# patient carrying at least one usable crop. Anything else is legitimately a subset arm
# (a frontal-only arm needs a frontal crop; m1 needs an observed grade) and its denominator
# is cross-checked against {split}_metrics.csv instead of against an anchor.
FULL_COHORT_ARMS = ("m0",)
ALL_VIEW_ARMS = ("m0d_clinical", "m3_image", "m4_fusion")

MODEL_DISPLAY = {
    "m0": "M0 clinical Cox",
    "m1": "M1 clinical plus KLG grade",
    "m2_frontal": "M2 frontal radiograph",
    "m3_image": "M3 multi-view image",
    "m4_fusion": "M4 multimodal fusion",
}
# Okabe-Ito qualitative colours for the NOMINAL model arms, each paired with a distinct
# marker and dash pattern so the series survive greyscale printing and colour blindness.
MODEL_STYLE = {
    "m0": {"color": "#000000", "marker": "o", "linestyle": "-"},
    "m1": {"color": "#009E73", "marker": "D", "linestyle": ":"},
    "m2_frontal": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "m3_image": {"color": "#56B4E9", "marker": "v", "linestyle": (0, (4, 1, 1, 1))},
    "m4_fusion": {"color": "#D55E00", "marker": "^", "linestyle": "-."},
}
# Tertiles are ORDINAL, so a single-hue sequential ramp is the correct encoding: it is
# colour-blind safe by construction and monotone in greyscale luminance.
TERTILE_STYLE = [
    {"color": "#6BAED6", "linestyle": "-", "label": "Lowest tertile"},
    {"color": "#2171B5", "linestyle": "--", "label": "Middle tertile"},
    {"color": "#08306B", "linestyle": "-.", "label": "Highest tertile"},
]

# --------------------------------------------------------------------------- #
# v6 IMAGING ANCHORS                                                            #
#                                                                              #
# Same rule as SPLIT_ANCHORS and NB_ANCHORS above: a number a CAPTION states is #
# written down once, here, and checked against the artefact that produced it at #
# render time or in the suite. Nothing below is read from a table at caption    #
# time, because a caption assembled from whatever a CSV happens to say cannot   #
# fail when the CSV moves - it just prints something else.                      #
# --------------------------------------------------------------------------- #
# Figure 1's radiograph. A DEVELOPMENT-split film, deliberately: the sealed split's
# reviewer panels are crop-only (the full-film panels were rendered before the test crops
# existed and the source DICOMs are gone), so no test-split case can show a pre-crop film
# at all. This one is a bilateral weight-bearing frontal whose contralateral knee is the
# RIGHT one, so the half-select, the marker on the discarded half and the mirror step are
# all visible in a single case.
FIG1_CASE_KEY = "20001294_frontal_f003764b5803"
# Pearson correlation between the crop this module reconstructs from the recovered film by
# replaying the pipeline's own geometry and the crop actually shipped in the shard. It is
# the evidence that the localization box drawn on the second tile is the real one, and it
# is asserted at render time, unconditionally: the crop asset is required, not preferred.
FIG1_CROP_PEARSON = 0.99
FIG1_CROP_PEARSON_MIN = 0.98
FIG1_ENCODER = "DenseNet-121"
# Panel A's four tile titles, named here rather than typed at the draw site so the suite
# can hold them to what the pictures under them actually show.
#
# D1. The third read "border and markers masked" until 2026-08-11, and the tile it sits
# over disproves the second half: the marker detector accepted no blob on this crop - its
# zero fraction is exactly the integer band - and a burned-in character is plainly visible
# inside the retained region. A title may only name a step the picture beneath it shows, so
# it names the band alone; the caption discloses the survivor and its split-wide rate.
FIG1_PANEL_TITLES = (
    "1  Acquired film,\nflip corrected",
    "2  Contralateral half,\n{half_inset_pct}% midline inset",
    "3  Square crop, {out_size} px,\nborder masked",
    "4  Mirrored to a\ncommon laterality",
)

# Figures 2 and 3 rest on the interpretability package (src/interpretability.py), which is
# a frontal-view analysis on the sealed split alone.
INTERP_ARM = "m2_frontal"
INTERP_TABLES = ("interp_panel_manifest.csv", "interp_occlusion.csv",
                 "interp_view_ablation.csv")
INTERP_PANEL_DIRNAME = "panels"
# Figure 2's four cases: one per cell of the classification table, ALL at Kellgren-Lawrence
# 2, ordered by predicted risk. Holding the grade fixed is the whole point - a difference
# between the columns cannot then be a difference in radiographic severity.
# (cell, empi_anon, published five-year risk), in the column order the figure draws.
FIND_KLG = 2.0
FIND_CASES = (("TN", "65765984", 0.036776), ("FN", "79677280", 0.052575),
              ("FP", "92527124", 0.207177), ("TP", "93136233", 0.397715))
FIND_RISK_TOL = 5e-4             # the precision the caption prints, so a match is a match
# The operating point interpretability used: the risk quantile at which predicted positives
# equal observed cases. Cells are assigned against the PUBLISHED risk, so the figure's cells
# are the ones every published table implies.
#
# WHAT THE QUANTILE IS TAKEN OVER, and why the caption has to say so. `_stratify` in
# src/interpretability.py builds y = 1 for an event by the horizon, 0 for observed event
# free beyond it, and DROPS everyone censored earlier; the quantile is then taken over the
# survivors alone. So "predicted positives equal observed events" is true on 263 patients
# and false on the 734 the rest of the figure set reports, where the same threshold flags
# 281. Every one of these is recomputed from the published hazards in the suite.
FIND_THRESHOLD = 0.1340
FIND_CLASSIFIABLE = 263          # cases + controls at the horizon; the quantile's population
FIND_CASE_N = 106                # of those, event by day 1825
FIND_CONTROL_N = 157             # of those, observed event free beyond day 1825
FIND_CENSORED = 471              # censored before the horizon, so not classifiable at all
FIND_FLAGGED_ALL = 281           # patients the same threshold flags on the whole 734
# The recalibrated spread the caption prints is 10.8-fold; the SAME two patients are
# 4.24-fold apart on the raw model output, so the caption names the scale beside the number.
# Raw = risk_at_horizon of the published ensemble hazards before apply_recalibration;
# published = after it, which is FIND_CASES and every risk in every published table.
FIND_RAW_RISKS = (0.104141, 0.129747, 0.297741, 0.441648)
# How the four were chosen out of the panels the sampler produced at this grade, stated on
# the page because both claims the caption makes about the choice are checkable and one of
# them bounds the headline number. `_stratify` filtered only on cell and grade; picking one
# per cell out of what it produced was an editorial choice, and it was made for the SPREAD.
FIND_GRADE_CANDIDATES = 9        # KL-2 panels in interp_panel_manifest.csv (3 FN, 2 each other)
FIND_FN_CANDIDATES = 3           # of which false negatives; the one drawn has the largest blob
FIND_ROW_LABELS = ("Radiograph", "Grad-CAM", "Integrated gradients")
FIND_CELL_WORD = {"TP": "true positive", "FP": "false positive",
                  "TN": "true negative", "FN": "false negative"}

# Figure 3's denominators, from the tables it draws.
FOREST_ABLATION_N = 315          # patients with a frontal AND at least one non-frontal crop
FOREST_OCCLUSION_N = 734         # patients the frontal arm scores on the sealed split
FOREST_BORDER_IDENTICAL = 677    # of those, bit-identical after the band is re-zeroed
FOREST_DEGENERATE_TIED = 676     # largest group sharing one identical risk, border-band only
FOREST_DEGENERATE_DISTINCT = 59  # distinct risks the border-band-only control produces
FOREST_DEGENERATE_AUROC = 0.497
# The one anatomic condition in interp_occlusion.csv whose interval excludes zero. It is
# NOT drawn (the fill value is a nuisance choice, not a finding), which is exactly why the
# caption has to name it: a sentence about "every anatomic occlusion interval" that is
# checked only against the drawn rows is a sentence checked against its own conclusion.
# Magnitudes of a reduction; the table carries -0.050642 (-0.091646, -0.010173).
FOREST_MEANFILL_CONDITION = "keep_joint_only_meanfill"
FOREST_MEANFILL_DELTA = 0.051
FOREST_MEANFILL_LO = 0.010
FOREST_MEANFILL_HI = 0.092

# --------------------------------------------------------------------------- #
# THE RESIDUAL-MARKER AUDIT, from outputs/tables/interp_regions.csv.            #
#                                                                              #
# Figures 1 and 2 both DRAW a finished crop that still carries a burned-in      #
# character, so both disclose it rather than letting a reader find it. Two      #
# facts about the pipeline make that expected rather than exceptional:          #
#                                                                              #
#  * the marker step accepts a blob only when it is saturated AND small AND     #
#    sitting on dark background, and it never masks the largest saturated       #
#    component, so a marker touching bone or a collimator edge survives;        #
#  * it runs AFTER the border band is zeroed and refills each accepted blob     #
#    with that blob's own ring median, so a blob at the edge of the retained    #
#    region writes a non-zero value back into pixels the band had blanked.      #
#                                                                              #
# Both constants are asserted against the audit table before either figure      #
# draws (see :func:`assert_marker_audit_anchors`) and pinned in the suite.      #
# --------------------------------------------------------------------------- #
CROP_AUDIT_N = 1216              # finished test crops scanned by crop_qa.residual_marker_scan
CROP_RESIDUAL_MARKER_PCT = 18.5  # UPPER BOUND: saturated bone edges share the signature
CROP_NONZERO_BAND_N = 145        # of those crops, border band not exactly zero
CROP_NONZERO_BAND_PCT = 11.9
# The measured cost of blanking every residual blob in the split, from
# interp_occlusion.csv's `mask_residual_markers` row - the only perturbation drawn in
# figure 3 whose interval excludes zero. The captions say "lowers ... by", so these are
# MAGNITUDES; the table carries them signed, as -0.00096 (-0.00230, -0.00008).
MARKER_DELTA_AUROC = 0.00096
MARKER_DELTA_LO = 0.00008
MARKER_DELTA_HI = 0.00230
MARKER_DELTA_CONDITION = "mask_residual_markers"

# Figure 4 panel B's uninformative stratum, stated on the face of the caption so a reader
# never has to infer it from an interval that spans half the range.
CAL_LOW_STRATUM = "KL 0-1"
CAL_LOW_N = 144
CAL_LOW_EVENTS = 3
CAL_SCHEME = "kl3"
CAL_ARM = "m2_frontal"
# Panel B's only source. Written for the SEALED SPLIT ALONE and carrying no split column,
# which is why figure 4 declines on every other split (calibration_decline_reason).
KLG_TERTILE_TABLE = "v6_klg_risk_tertiles.csv"

# Supplementary anchors.
S1_PATIENCE = 8                  # early-stopping patience: the last epoch is always +8
S1_SERIES = 35                   # 7 arms x 5 seeds
S5_CONTROLS_BEYOND_HORIZON = 162  # test patients observed beyond day 1825
S6_PAIR_N = 218                  # patients holding exactly one frontal and one lateral crop
S6_TRIPLE_N = 32                 # patients holding all three views
S6_M3_PCT = 95.4
S6_M4_PCT = 96.8

# Print-size typography. 8 pt body, 7 pt inside the flow boxes and the at-risk row.
BASE_FONT_PT = 8.0
SMALL_FONT_PT = 7.0
PANEL_LETTER_PT = 9.0
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],           # always present, so rendering is portable
    "font.size": BASE_FONT_PT,
    "axes.labelsize": BASE_FONT_PT,
    "axes.titlesize": BASE_FONT_PT,               # unused: no figure carries a title
    "xtick.labelsize": SMALL_FONT_PT,
    "ytick.labelsize": SMALL_FONT_PT,
    "legend.fontsize": SMALL_FONT_PT,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
    "legend.frameon": False,
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
# matplotlib stamps a "Software: Matplotlib vX.Y.Z" chunk into the PNG. Dropping it keeps the
# bytes identical across matplotlib upgrades, which is what "deterministic" has to mean here.
PNG_METADATA = {"Software": None}
PAD_INCHES = 0.01
# Fixed-point passes onto the column width. The iteration approaches the target from below
# and stops as soon as it is within a pixel, so a figure that already converged is unchanged
# by raising this; what needs the extra passes is a figure carrying text OUTSIDE the axes -
# figure 2's shared legend - because that text keeps its size in points while the canvas is
# rescaled, which slows the approach without breaking it.
WIDTH_LOCK_PASSES = 8
WIDTH_LOCK_TOL_PX = 3                             # slack for the tight-bbox rounding

# Figure 1 geometry, in "line units": one unit is one line of 7 pt text.
FLOW_WRAP_CHARS = 64
# The excluded-count column is about two thirds the width of the retained-count column, so it
# needs its own wrap. Text is drawn centred and is NOT clipped to the box, so a line that
# overruns the column pushes the tight bounding box outward and the width-lock iteration in
# _save stops converging: an unwrapped branch label fails the render rather than looking
# slightly wide. Every branch label the validation render draws is shorter than this, so the
# wrap is a guard rather than a change.
FLOW_EXCL_WRAP_CHARS = 42
FLOW_LINE_IN = 0.105                              # inches per line unit
FLOW_BOX_PAD_UNITS = 0.70                         # total vertical padding inside a box
FLOW_GAP_UNITS = 3.00                             # vertical gap between consecutive boxes
FLOW_MAIN_X = (0.000, 0.545)                      # retained-count column, axes x fraction
FLOW_EXCL_X = (0.620, 0.980)                      # excluded-count column
# The ledger stores ASCII comparison operators; typeset them properly in the figure.
FLOW_GLYPHS = {">=": "≥", "<=": "≤"}


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


def assert_split(split: str) -> str:
    assert split in SPLITS, (
        f"{MODULE}: split must be one of {list(SPLITS)}, not {split!r}; a mistyped split must "
        "fail here rather than silently render the wrong cohort")
    return split


def default_split(cfg: Config) -> str:
    """The split the manuscript reports, from ``manuscript.report_split``.

    Mirrors ``src.make_manuscript.report_split``. The two render modules read the same key so
    a document and its figures cannot describe different splits; ``--split`` overrides it for
    a one-off render without editing the config back and forth.
    """
    return assert_split(str(cfg["manuscript"].get("report_split", VAL_SPLIT)))


def readable_splits(split: str) -> tuple[str, ...]:
    """The splits whose patient rows a render of ``split`` is allowed to materialise.

    Mirrors :func:`src.make_manuscript.readable_splits` and is tested against it. A
    validation render gets the development splits and nothing else; only a sealed render
    gets the sealed rows, and only after
    :func:`src.eval_models.assert_sealed_read_is_recorded` has passed.

    NOT :data:`RENDERED_SPLITS`, which is a narrower thing: that says which splits a figure
    DESCRIBES (a validation figure 1 draws train plus val boxes), while this says which rows
    may be held in memory at all. The two agree on the sealed split and differ on validation
    only in that both name the development splits for different reasons.
    """
    return ALL_SPLITS if assert_split(split) == SEALED_SPLIT else _DEV_SPLITS


def forbid_test_rows(cfg: Config, split: str) -> bool:
    """Whether the feature-table read must push the sealed rows out, for ``split``.

    Deliberately the same expression as ``src.make_manuscript.forbid_test_rows``, and tested
    against it: the two render modules describe the same split of the same study in the same
    document set, so a split one of them may read and the other may not is a contradiction,
    not a configuration. ``model_eval.forbid_test_split`` is read, never written.

    It is a HINT to the reader, not the rule. The rule is :func:`readable_splits`, which no
    config flag can move; this only decides whether the sealed rows are dropped in the
    Parquet predicate (cheap, and they never reach memory) or dropped afterwards (in which
    case the sealed-read gate has to have passed first).
    """
    return assert_split(split) != SEALED_SPLIT and bool(cfg["model_eval"]["forbid_test_split"])


# --------------------------------------------------------------------------- #
# WHICH CONVERGENCE VERDICTS DISQUALIFY AN ARM, PER RENDERED SPLIT              #
#                                                                              #
# One rule, three readings, and they cannot be allowed to disagree:             #
#   * src.eval_models.suppress_unfit_contrasts APPLIES it, blanking a row and   #
#     setting the ``suppressed`` column figure 4 reads back;                    #
#   * src.make_manuscript.DISQUALIFYING mirrors it, to decide whether a figure  #
#     belongs in the document and whether an arm may be interpreted in prose;   #
#   * this table mirrors it, to decide whether a figure may be DRAWN at all.    #
# test_the_two_render_modules_disqualify_the_same_statuses pins the second and  #
# third against each other for every split, so the mirror cannot quietly become #
# a second opinion about the same arm on the same split.                        #
#                                                                              #
# It is a mirror rather than an import for the reason :func:`_word` already     #
# gives: ``src.make_manuscript`` imports THIS module for the figure registry    #
# and itself imports python-docx, so importing it back would make rendering a   #
# PNG depend on a document library and close an import cycle. The STATUS_*      #
# strings are imported from ``src.eval_models``, so only the per-split GROUPING #
# is written twice and neither copy can invent a verdict.                       #
#                                                                              #
# The sealed entry is deliberately the SMALLER set. STATUS_OVERFIT says the     #
# retained checkpoint was selected on the validation split, so a validation     #
# metric read at that checkpoint is circular and optimistic; the sealed split   #
# took part in neither the fitting nor the selection, so its estimate is        #
# precisely the number that reveals whether the overfitting cost anything.      #
# STATUS_NO_CONVERGE disqualifies on every split: a constant predictor is       #
# constant on every split.                                                      #
# --------------------------------------------------------------------------- #
DISQUALIFYING: dict[str, frozenset[str]] = {
    VAL_SPLIT: frozenset({STATUS_NO_CONVERGE, STATUS_OVERFIT}),
    SEALED_SPLIT: frozenset({STATUS_NO_CONVERGE}),
}
assert set(DISQUALIFYING) == set(SPLITS), (
    f"{MODULE}: every renderable split needs a convergence rule; DISQUALIFYING covers "
    f"{sorted(DISQUALIFYING)} and SPLITS is {list(SPLITS)}")


def disqualifying_statuses(split: str) -> frozenset[str]:
    """The convergence verdicts that disqualify an arm on ``split``. Unknown split RAISES.

    Mirrors :func:`src.make_manuscript.disqualifying_statuses`, including its refusal to
    resolve a split it has no rule for: a split with no entry must fail here rather than
    borrow another split's, which is how a validation gate came to be applied to
    sealed-split numbers. A guard is only as strict as its most forgiving twin.
    """
    return DISQUALIFYING[assert_split(split)]


def crop_loss(split: str) -> int:
    """Patients the crop pipeline left with no usable contralateral crop.

    Counted over the splits a ``split`` render DESCRIBES, so a validation render reports the
    development loss (train plus val, 2) and a sealed render the test loss (1). That is the
    number figure 1's caption states and the number its final exclusion branch draws, so the
    two are one computation instead of two.

    NOT interchangeable with :func:`expected_unscored`, which counts the loss inside the
    rendered split's OWN frame: the two agree on the sealed split, where the render describes
    one split, and differ on validation, where the frame is val alone but the flow chart and
    its caption describe train plus val.
    """
    return sum(SPLIT_ANCHORS[s]["n"] - SPLIT_ANCHORS[s]["crop_n"]
               for s in RENDERED_SPLITS[assert_split(split)])


# --------------------------------------------------------------------------- #
# CAPTION CLAUSES                                                               #
# Captions are assembled from named clauses rather than written twice, so the   #
# split-invariant text is literally shared and only the claims that FLIP have   #
# two versions. Figure 1 is the instructive case: the sealed split contributed  #
# to no model fitting and no hyperparameter choice on either split, and only    #
# the third clause - whether it contributed to the evaluation - changes.        #
# --------------------------------------------------------------------------- #
FIG1_TITLE = ("Cohort assembly from the source registry to the {cohort_word} patients with a "
              "usable contralateral radiograph")
FIG1_LEDGER = (
    "Patient flow for the primary landmark cohort. Boxes in the left column give the number "
    "of patients retained after each eligibility criterion, and the branches to the right "
    "give the number removed at that step. Counts down to the {landmark:,}-patient landmark "
    "cohort are taken from the cohort assembly ledger; the final two steps are recomputed "
    "at figure time from the locked patient-level split assignment and the record of which "
    "radiographs were successfully prepared for the model.")
FIG1_SEALED_LEAD = ("The {test_n} patients assigned to the test split, carrying {test_ev} "
                    "events, were set aside unread and")
FIG1_SEALED_NEVER_READ = (
    "contributed to no model fitting, hyperparameter choice or evaluation reported in this "
    "manuscript, so every number downstream of that step describes development patients only.")
FIG1_SEALED_READ_ONCE = (
    "contributed to no model fitting and no hyperparameter choice; they were read once, after "
    "every model was frozen, and the evaluation reported in this manuscript is that single "
    "read.")
# The crop-attrition counts are INTERPOLATED, not typed. They are the one cohort number a
# caption states that is not already an anchor lookup, and spelling them out here would put
# "One test patient" in a document whose figure 3 assertion had just been re-anchored to a
# different count. The three grammar slots come from :func:`_crop_loss_context`.
FIG1_CROP_DEV = ("{Crop_lost_word} development {crop_lost_noun} had no usable image of the "
                 "contralateral knee; {crop_lost_subject} event free, so "
                 "all {dev_ev} development events were retained.")
FIG1_CROP_TEST = ("{Crop_lost_word} test {crop_lost_noun} had no usable image of the "
                  "contralateral knee; {crop_lost_subject} event free, so all "
                  "{test_ev} test events were retained.")

# The acronym is TIED to its expansion here, and this is the only place in the document
# where it can be. "IPCW" is Panel A's own y-axis label, it heads a Table 2 column, and it
# names the primary contrast in that table's final row; the expansion appeared once, in
# this caption, without the acronym beside it, so a reader who met "IPCW" first had nowhere
# to look it up. This clause precedes Table 2 in document order, and Table 2's note repeats
# the definition so the table still stands alone.
SEALED_READ_VAL = "The test split was not read."
SEALED_READ_TEST = ("These are out-of-sample estimates: the test split was read once, after "
                    "every model and every hyperparameter choice had been frozen.")

FIG3_TITLE = ("Cumulative incidence of contralateral knee arthroplasty by predicted "
              "{long_adj} risk tertile")
FIG3_LEAD = (
    "Cumulative incidence of contralateral knee arthroplasty by predicted risk tertile. "
    "{Split_word} patients were split into tertiles of predicted {long_adj} risk from the "
    "multimodal fusion model, and each curve is the Kaplan-Meier cumulative incidence, one "
    "minus the survival function, from the landmark through day {grid_max}.")
FIG3_AT_RISK = ("The number of patients remaining at risk in each tertile is given beneath "
                "the horizontal axis.")
FIG3_COMPETING = (
    "Death is not ascertainable in this data source, so mortality acts as an unmeasured "
    "competing event; these are therefore cause-agnostic cumulative incidences rather than "
    "cause-specific ones, and where competing mortality is appreciable they will overstate "
    "what a competing-risk estimator would give.")
FIG3_DENOMINATOR = (
    "The three curves rest on {fig3_n} patients and {fig3_ev} events in total, so the "
    "separation between tertiles should be read as exploratory rather than as a precise "
    "estimate of absolute risk.")

# Figure 4 shares SEALED_READ_VAL / SEALED_READ_TEST above: whether the split was read once
# after freezing is a property of the SPLIT, not of the figure, so it is written once.
FIG4_TITLE = ("Decision-curve analysis of net benefit against threshold probability in the "
              "{split_word} split")
FIG4_LEAD = (
    "Decision-curve analysis in the {split_word} split. Panel A gives net benefit, in net "
    "true positives per patient screened at {long_lab}, against threshold probability, for "
    "{nb_arms_prose}, with treat all and treat none in grey; the marker on treat "
    "all is its zero crossing, at the {long_adj} cumulative incidence of {nb_prev}.")
FIG4_ESTIMATOR = (
    "At each threshold the rule flags every patient whose predicted {long_adj} risk reaches "
    "it.")
FIG4_PANEL_B = (
    "Panel B gives the paired differences for the {nb_protagonist} arm, against treat all "
    "and against {nb_reference}, with pointwise {n_boot:,}-replicate percentile bootstrap 95 "
    "percent intervals, unadjusted across thresholds, and a zero reference line.")
FIG4_WHY_NO_BANDS = (
    "Panel A carries no intervals: marginal intervals over the same patients overlap where "
    "the paired difference between them does not.")
FIG4_RANGE = (
    "Thresholds {nb_lo} to {nb_hi} are drawn of the {nb_full_lo} to {nb_full_hi} the "
    "underlying table reports, and each curve stops where its flagged set falls below the "
    "{nb_sparse_min}-event floor.")
# THE ONE TABLE NUMBER IN THE WHOLE FIGURE SET, and it MOVED on 2026-08-11. Every other
# caption is barred from citing a table by number (test_no_caption_cites_a_table_by_number),
# because the table set is restructured by the manuscript task and a number typed into a
# caption becomes a pointer at the wrong table. This clause is the sanctioned exception: the
# per-arm denominators it states are worth pointing at, and the pointer is exempted rather
# than deleted. v5's Table 2 was the per-arm performance ladder; v6 SPLIT it into Table 2
# (primary and head-to-head contrasts) and Table 3 (per-arm discrimination and calibration),
# so the per-arm denominators this sentence promises - 741/106, 734/106 and 707/98 on the
# sealed split - are Table 3's rows now. Table 2 prints PAIRED counts per contrast, which is
# a different number, so leaving "Table 2" here would have sent a reader who wanted 741 to
# the one table that does not print it. The authoritative layout is
# outputs/manuscript/"v6- resubmission"/sections/tables.md.
FIG4_DENOMINATORS = (
    "The arms score different populations: {split_n} patients and {split_ev} events for "
    "{nb_reference}, {nb_arm_n} and {nb_arm_ev} for {nb_protagonist}, and {panel_b_n} and "
    "{panel_b_ev} for the set every arm scores; Table 3 gives them per arm.")
# Panel A draws ONE treat-all curve, the reference arm's, because four nearly coincident grey
# lines would be unreadable at a single column - but treat-all is a property of the POPULATION
# it is estimated on, so each arm has its own and panel B differences against the
# protagonist's. On the sealed split those two treat-alls sit about 0.003 to 0.004 apart
# across the drawn window, so a reader who measures the vertical gap off panel A overstates
# panel B's grey curve by roughly that much. Disclosing which population panel A's grey curve
# rests on is the sanctioned fix; redrawing four grey lines is not.
FIG4_TREAT_ALL_SET = (
    "Panel A draws {nb_reference}'s treat all while Panel B's difference is taken against "
    "{nb_protagonist}'s own, so the two do not line up exactly.")
FIG4_CALIBRATION = (
    "{nb_protagonist} under-predicts {long_adj} risk by {nb_citl_pp} percentage points in "
    "the large, and net benefit is sensitive to calibration where discrimination is not, so "
    "the horizontal axis is a decision-rule parameter rather than a true risk.")
# WHERE THE ARGUMENT WENT, and why it is a pointer rather than a paragraph. Four things this
# caption used to carry are recorded in full in the deviation register the Methods already
# cite as accompanying material, and a caption is not the place to make an estimator's case:
# the choice of Kaplan-Meier within the flagged set and the inverse-probability-of-censoring
# sensitivity curve beside it are D30; the threshold grid, the plotted window and the fact
# that every threshold is tabulated are D29. The odds-weight ceiling above the plotted
# window, the reference arm's largest predicted risk, the sparse-set truncation floor and
# the strict monotonicity of the image arms' recalibration were justifications for choices a
# reader can check in those entries and in the table behind the figure; figure 2's caption
# already states that the recalibration is monotone and what that implies. What stays here
# is what changes how the panels are read.
FIG4_REGISTER = (
    "The estimator, the threshold grid, the plotted window and the sensitivity curve "
    "beside this one are set out in the deviation register.")


# --------------------------------------------------------------------------- #
# THE v6 IMAGING FIGURE SET                                                     #
#                                                                              #
# Journal of Imaging desk-rejected v5 on scope: none of its four figures        #
# carried a radiograph. The four clause blocks below are the replacements. The  #
# four clause blocks ABOVE are not deleted - they still write the captions of   #
# the supplementary figures the old main figures became (cohort flow -> S2,     #
# decision curve -> S3, cumulative incidence by tertile -> S4), so the prose is #
# shared rather than retyped and the numbers in it keep their anchors.          #
# --------------------------------------------------------------------------- #
WF_TITLE = ("Imaging and modelling workflow, from the acquired radiograph to the {long_adj} "
            "hazard")
WF_PIPELINE = (
    "Panel A follows one bilateral weight-bearing frontal radiograph through the "
    "preprocessing pipeline. The decoded film is corrected for the acquisition system's "
    "horizontal-flip flag; the half holding the contralateral knee is selected at a midline "
    "inset of {half_inset_pct} percent of the film width, which discards the central strip "
    "in which index-knee pixels could survive an off-centre patient; a square region "
    "centred on the collimated field is cropped from that half at {crop_side_pct} percent "
    "of its short side and resampled to {out_size} by {out_size} pixels with Lanczos "
    "interpolation; the outer {border_px} pixels are set to zero, which blanks "
    "{border_pct} percent of the output; a detector then searches the retained region for "
    "any burned-in marker the band did not reach and blanks the saturated, small, "
    "background-bordered blobs it accepts; and the finished crop is mirrored where "
    "necessary so that every knee the network sees reads as a left knee.")
# D1. The picture disproves the old caption, which said the marker step had run on this
# crop: exactly 22.75 percent of it is zero, which is the integer band and nothing else, so
# the detector accepted no blob here, and a burned-in character is plainly visible in the
# retained region of panels 3 and 4. Concealing that by reselecting the case would be worse
# than disclosing it, so the caption discloses it and gives the split-wide rates.
WF_RESIDUAL = (
    "The marker detector accepted nothing on this crop. Exactly {border_pct} percent of it "
    "is zero, which is the band and nothing more, and a small saturated burned-in marker "
    "survives inside the retained region, clipped by the band at the edge of the crop: it "
    "sits near the top left of the third tile and, the mirror having been applied, near "
    "the top right of the fourth, which is the array the network reads. It is visible on "
    "the contralateral half of the first two tiles as well. This is a true property of the "
    "image the network was given rather than a fault of the example. Across the {crop_audit_n} "
    "finished crops of the {crop_audit_split} split, {crop_marker_pct} percent carry a residual "
    "marker-like blob, an upper bound because saturated bone edges share the detector's "
    "signature, and {crop_band_n} of them, {crop_band_pct} percent, carry a border band "
    "that is not exactly zero, because the marker step runs after the band is zeroed and "
    "refills each blob it does accept with that blob's own surrounding median. What the "
    "residual markers are worth to the model is measured on the whole split and reported "
    "in Figure 3.")
WF_NETWORK = (
    "Panel B gives the model. Every crop a patient contributes passes through one shared "
    "{encoder} encoder, a learned view embedding is added to the pooled feature vector, an "
    "attention pool over that patient's available views returns a single patient "
    "representation, and a discrete-time head emits {n_intervals} interval hazards from "
    "which the {long_adj} risk is computed. Patients contribute different numbers of views, "
    "so the pool is masked and the padded slots are never encoded.")
WF_PROVENANCE = (
    "The radiograph is a development-split film. The source archive of full DICOM images is "
    "no longer held, so the reviewer quality-assurance panels are the only surviving record "
    "of a film before cropping; the film shown here was recovered from one of those panels, "
    "which are rendered images, so it is a downsampled copy with the panel's own overlay "
    "annotation repaired away and its aspect ratio restored from the stored image "
    "dimensions. The crop tiles are the shipped {out_size}-pixel crop itself. The "
    "localization box drawn on the second tile is the box the pipeline's own arithmetic "
    "specifies rather than one placed by eye, and resampling that box reproduces the "
    "shipped crop with a Pearson correlation of {wf_crop_r}.")

FIND_TITLE = ("Attribution and occlusion maps for four {split_word}-split patients at the same "
              "Kellgren-Lawrence grade")
# D4. The operating point is a QUANTILE OVER THE CLASSIFIABLE PATIENTS ONLY, and the old
# caption's "makes the number of predicted positives equal the number of observed events"
# was false on the reported cohort: `_stratify` (src/interpretability.py) drops the 471 of
# 734 patients censored before the horizon, so the equality holds on the 263 that remain
# and the same threshold flags 281 of the full 734. The denominator now sits on the page.
FIND_LEAD = (
    "Representative image findings at a single radiographic grade. Four {split_word}-split "
    "patients whose contralateral knee carried an inferred Kellgren-Lawrence grade of "
    "{find_grade}, one from each cell of the classification table at a fixed operating "
    "point of {find_threshold} predicted {long_adj} risk. That threshold is the quantile "
    "at which the number of predicted positives equals the number of observed events among "
    "the {find_classifiable} patients who are classifiable at {long_lab}, which is the "
    "{find_cases} with the event by then plus the {find_controls} observed event free "
    "beyond it; the remaining {find_censored} of {forest_occl_n} are censored earlier and "
    "cannot be placed in the table at all, and applying the same threshold to the whole "
    "{split_word} split flags {find_flagged_all}. Columns are ordered by predicted risk.")
FIND_ROWS = (
    "The top row is the image the network reads, after the preprocessing in Figure 1. The "
    "middle row is the Grad-CAM map at the encoder's final normalisation layer and the "
    "bottom row is integrated gradients, both averaged over the five pre-specified seeds "
    "and both taken with respect to the predicted {long_adj} risk.")
# D2. The false-negative crop carries a burned-in character inside the retained region and
# it is visible in all three rows. What this clause may NOT say is that the attribution
# maps ignore it: the persisted Grad-CAM is 16 by 16 native, one cell of which is 32 by 32
# crop pixels, so it cannot resolve a 40 by 17 px marker, and no integrated-gradient array
# is persisted at all. The defensible statement is the measured one, so that is what it
# makes.
FIND_MARKER = (
    "The false-negative column carries a saturated burned-in character at the upper right "
    "of the crop, clipped by the masked band and visible in all three rows. It was not "
    "selected around: the candidate panels come from a seeded sampler that filters only on "
    "classification cell and Kellgren-Lawrence grade, and of the {find_fn_candidates} "
    "false-negative candidates at this grade the one drawn here is the one carrying the "
    "largest residual blob. {crop_marker_pct} percent of the {crop_audit_n} finished crops "
    "of this split carry such a blob, an upper bound because saturated bone edges share "
    "the detector's signature. Blanking every such blob across the whole split lowers {long_adj} "
    "discrimination by {marker_delta} of area under the curve, with a 95 percent interval "
    "of {marker_delta_lo} to {marker_delta_hi}; that is the only row of Figure 3's masked "
    "band and burned-in marker block whose interval excludes zero, and it is the bound on "
    "what the residual markers can contribute. No claim is made here about where either "
    "attribution method placed its mass relative to that character.")
# D4. "10.8-fold" is a property of the RECALIBRATED scale. On the raw model output the
# same two patients are 4.24-fold apart, so the scale is now named beside the number.
FIND_SPREAD = (
    "All four risks are shown after the frozen horizon-specific recalibration described in "
    "the Methods, which was fitted on the validation split and applied unchanged here, so "
    "they are on the scale of every predicted risk this manuscript reports. Predicted "
    "{long_adj} risks across the four columns are {find_risks}, a {find_fold}-fold spread "
    "on that recalibrated scale within one radiographic grade, which is the separation the "
    "grade itself does not provide; the transform is monotone, so it moves the spread but "
    "not the ordering, and the same two patients are {find_fold_raw}-fold apart on the raw "
    "model output. These four are the WIDEST-SPREAD quartet available at this grade: the "
    "sampler produced {find_grade_candidates} candidates at it, and no other choice of one "
    "per cell spans a larger ratio, so {find_fold}-fold is the largest separation these "
    "candidates could have been made to show and not a typical one.")
FIND_SCALE = (
    "Each map is scaled to its own 99.5th percentile and drawn with opacity proportional to "
    "magnitude, so colour is comparable within a tile and not across tiles; the "
    "integrated-gradient tiles are smoothed for display only and every quantity reported in "
    "the text is computed on the unsmoothed maps.")

FOREST_TITLE = ("Discrimination of the clinical and imaging models, and the effect of "
                "withholding views, occluding regions and widening the masked band")
FOREST_PANEL_A = (
    "Panel A gives the {long_adj} IPCW cumulative dynamic time-dependent area under the "
    "receiver operating characteristic curve for {forest_arms_drawn} of the "
    "{forest_arms_total} model arms on the {split_word} split, with {n_boot:,}-replicate "
    "percentile bootstrap 95 percent intervals. Each arm is estimated on the patients it "
    "can score and those denominators differ; they are printed beside each row.")
FOREST_PANEL_B = (
    "Panel B gives differences on the same scale, each from a paired bootstrap on the "
    "patients both conditions score, with the number of paired patients printed beside each "
    "row. The model contrasts compare separately trained arms across the whole split. The "
    "view-withholding rows take one frozen multi-view network and remove views at input, so "
    "they are restricted to the {forest_ablation_n} patients who contributed both a frontal "
    "and at least one non-frontal radiograph; they answer what a view CONTAINS, which is a "
    "different question from what a model that was trained without it achieves.")
# D3. The sentence used to read "Every anatomic occlusion interval crosses zero", full
# stop. That is false of the table: it holds seven anatomic conditions and the figure draws
# five, and one of the two it does not draw, the mean-filled joint-only condition, has an
# interval that excludes zero and says so in its own note column. The claim is now scoped
# to the rows on the page AND the exception is stated, because the undrawn row is the one
# most FAVOURABLE to the model and suppressing it would be selection in the other
# direction. Both halves are asserted against the whole table before the figure draws.
FOREST_OCCLUSION = (
    "The region and masking rows re-score the frozen frontal-view network on perturbed "
    "images. Every anatomic occlusion interval drawn here crosses zero, so the compartments "
    "must not be ranked against one another. The underlying table also holds a mean-fill "
    "variant of each of the two tibiofemoral conditions, which are not drawn because the "
    "value the occluded pixels are filled with is a nuisance choice rather than a finding. "
    "One of them is the only anatomic condition in the table whose interval excludes zero: "
    "keeping the tibiofemoral joint alone and filling the rest with the image mean lowers "
    "{long_adj} discrimination by {forest_meanfill_delta}, with a 95 percent interval of "
    "{forest_meanfill_lo} to {forest_meanfill_hi}. It is reported here rather than left in "
    "the table because it is the condition most favourable to the model, and it does not "
    "license a ranking either: the anatomic conditions were not adjusted for multiplicity, "
    "and the zero-fill version of the same condition crosses zero.")
FOREST_CONTROLS = (
    "The two rows in the shaded block are pipeline checks and not leakage tests. The "
    "preprocessing pipeline already zeroes the outer {border_px} pixels, so re-zeroing that "
    "band leaves {forest_border_identical} of {forest_occl_n} patients bit-identical, and an "
    "input consisting of the band alone is a degenerate image on which "
    "{forest_degenerate_tied} patients receive one identical risk and only "
    "{forest_degenerate_distinct} distinct risks exist; its value of "
    "{forest_degenerate_auroc} is the discrimination of a near-constant predictor and says "
    "nothing about burned-in text. The evidence about text is the two widened bands and the "
    "residual-marker row, all of which perturb pixels that do carry signal and all of which "
    "are upper bounds.")
FOREST_POSTHOC = (
    "The single-view contrasts, the view-withholding rows and the occlusion rows are post "
    "hoc analyses of an already-read split and are reported as exploratory (deviations D35 "
    "and D36).")

CAL_TITLE = ("Calibration at {long_lab} and observed incidence by predicted-risk tertile within "
             "Kellgren-Lawrence stratum")
CAL_PANEL_A = (
    "Panel A gives calibration at the {long_adj} horizon in the {split_word} split: each "
    "marker is one quintile of predicted risk, plotted as the mean predicted risk against "
    "the Kaplan-Meier observed risk within that quintile, vertical bars are Greenwood 95 "
    "percent intervals, and the diagonal is the line of perfect calibration. It rests on the "
    "{panel_b_n} patients, carrying {panel_b_ev} events, that all four arms score, so the "
    "curves share one population.")
CAL_RECAL = (
    "The two image arms are drawn AFTER the horizon-specific recalibration described in the "
    "Methods, which was fitted on the validation split and applied unchanged here, so this "
    "panel and the reported calibration slope and calibration in the large describe the same "
    "predictions. The two frozen Cox comparators carry no such transform and are drawn as "
    "fitted.")
CAL_PANEL_B = (
    "Panel B gives the observed {long_adj} Kaplan-Meier incidence of contralateral "
    "arthroplasty in each tertile of predicted risk from the frontal-radiograph model, "
    "formed WITHIN each Kellgren-Lawrence stratum, with Greenwood 95 percent intervals. "
    "Stratum sizes and event counts are printed on each group. The lowest stratum carries "
    "{cal_low_events} events in {cal_low_n} patients, so its intervals span almost the whole "
    "range and no separation is observable there; that is an uninformative cell rather than "
    "a demonstrated null.")
CAL_COMPETING = (
    "Death is not ascertainable in this data source, so mortality acts as an unmeasured "
    "competing event and both panels report cause-agnostic risks.")
CAL_POSTHOC = (
    "The stratified analysis in Panel B is post hoc on an already-read split and is reported "
    "as exploratory (deviation D35). The Kellgren-Lawrence grade is itself model inferred, "
    "so 'beyond the grade' means beyond an automated grade rather than beyond a "
    "radiologist's reading.")

# --------------------------------------------------------------------------- #
# SUPPLEMENTARY FIGURE CLAUSES                                                  #
# --------------------------------------------------------------------------- #
S1_TITLE = "Training and validation loss for every arm and every seed"
S1_LEAD = (
    "Learning curves for all seven arms and all five pre-specified seeds. The horizontal "
    "axis is epochs from the RETAINED checkpoint, so zero is the epoch whose weights were "
    "kept, and the filled marker on each pair of curves is that epoch. Training ran a "
    "further {s1_patience} epochs past it in every one of the {s1_series} series, because "
    "that is the early-stopping patience.")
S1_GAP = (
    "The number under each panel is the validation-minus-training negative log likelihood at "
    "the retained checkpoint, beside the quantity reported as the overfitting gap in the "
    "convergence table, which is a different measurement: that one is the validation loss at "
    "the LAST epoch minus its own minimum, so it measures how far training ran after the "
    "model that was kept and not how far the kept model was from its training data.")
S1_LOWER_BOUND = (
    "The training loss is a running average taken in training mode with dropout and "
    "augmentation active while the weights update, and the validation loss is a clean "
    "evaluation pass, so every gap shown here is a LOWER BOUND on the train-validation gap "
    "rather than a measurement of it.")

S5_TITLE = "Discrimination within strata of image acquisition and image quality"
S5_LEAD = (
    "Imaging robustness of the frontal-radiograph model on the {split_word} split. Each row "
    "is one level of one stratum. A level is drawn as an estimate with its "
    "{n_boot:,}-replicate percentile bootstrap 95 percent interval only where it clears the "
    "pre-specified 50-event floor and the estimator is defined; the other two states are "
    "marked rather than left blank.")
S5_STATES = (
    "'Suppressed' means the level carries fewer than the 50 events the analysis plan "
    "requires. 'Not estimable' means something different and stronger: the {long_adj} "
    "IPCW area under the curve compares patients with an event by the horizon against "
    "patients observed event free beyond it, and only {s5_controls} of {split_n} patients "
    "in this split reach that day, so several levels clear the event floor and still hold no "
    "such control at all. Those cells are undefined, not imprecise.")
S5_SCOPE = (
    "The image-quality strata are reported twice, over ALL of a patient's crops and over "
    "the frontal crop alone; the two are different denominators and, for masking, different "
    "conclusions, so they are labelled and never averaged. Joint-localization method is not "
    "drawn separately: a crop confidence below one holds if and only if the intensity-profile "
    "localizer was used, so the two are one partition and not two findings.")
S5_CAVEAT = (
    "Only one stratum family has two estimable levels, so most rows compare one stratum "
    "against the whole split rather than one stratum against another. Acquisition era is "
    "omitted from this figure: the de-identified study date carries a per-patient random "
    "shift whose cross-patient comparability has never been confirmed in writing (deviation "
    "D17), and era is almost perfectly confounded with follow-up length in this cohort. "
    "Equipment, manufacturer and acquisition site were never released with the data and "
    "cannot be evaluated at all.")

S6_TITLE = "Attention weight assigned to each radiographic view by the multi-view models"
S6_LEAD = (
    "Per-view attention weights read out of the frozen aggregators on the {split_word} "
    "split. Weights sum to one over the views a patient actually contributed, so a patient "
    "holding only a frontal radiograph contributes a weight of one to it trivially; the "
    "comparison that means anything is therefore WITHIN patients holding one crop of each "
    "view.")
S6_PAIRED = (
    "Panel A gives the mean weight in the {s6_pair_n} patients holding exactly one frontal "
    "and one lateral crop, with standard deviations; the lateral view outweighs the frontal "
    "in {s6_m3_pct} percent of them under the multi-view image model and {s6_m4_pct} percent "
    "under the fusion model. Panel B gives the three-view mean in the {s6_triple_n} patients "
    "holding all three views. Panel C gives the marginal share over every patient with the "
    "view, which is the number the trivial single-view weights inflate and is shown so the "
    "two can be told apart.")
S6_READING = (
    "The aggregator is not ignoring the additional views; where a lateral radiograph exists "
    "it takes the majority of the weight. The frontal-only result in the main text is a "
    "statement about a population in which most patients have no second view, not a "
    "statement that the second view is uninformative.")


def _sentences(*clauses: str) -> str:
    """Join caption clauses with a single space. Each clause is a whole sentence."""
    return " ".join(c.strip() for c in clauses if c)


# --------------------------------------------------------------------------- #
# CONFIG-DRIVEN FIGURE REGISTRY                                                 #
# --------------------------------------------------------------------------- #
_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve")


def _word(n: int, cap: bool = False) -> str:
    """Spell small integers so no sentence opens with a numeral.

    Mirrors ``src.make_manuscript._word``, deliberately as a local copy: that module imports
    THIS one for the figure registry, so importing it back would make the figures depend on
    python-docx and close an import cycle for the sake of thirteen words.
    """
    v = int(n)
    s = _WORDS[v] if 0 <= v < len(_WORDS) else f"{v:,}"
    return s[0].upper() + s[1:] if cap else s


def _crop_loss_context(n_lost: int) -> dict[str, str]:
    """The grammar figure 1's crop-attrition clause needs, as a function of the count alone.

    One template covers both splits: rendered at 1 it reads "One test patient ... that patient
    was event free", at 2 "Two development patients ... both were event free". Because the
    count is the only input, the sentence agrees with :data:`SPLIT_ANCHORS` by construction
    and cannot be left behind by a re-anchored cohort.
    """
    return {
        "Crop_lost_word": _word(n_lost, cap=True),
        "crop_lost_noun": "patient" if n_lost == 1 else "patients",
        "crop_lost_subject": ("that patient was" if n_lost == 1 else
                              "both were" if n_lost == 2 else
                              f"all {_word(n_lost)} were"),
    }


def _horizon_label(days: int, days_per_year: float) -> str:
    """'5 years' for 1825 days. Rounded to the nearest whole year for prose."""
    y = int(round(days / days_per_year))
    return f"{y} year" if y == 1 else f"{y} years"


def _horizon_adj(days: int, days_per_year: float) -> str:
    """'5-year' for 1825 days: the attributive form, for use in front of a noun."""
    return f"{int(round(days / days_per_year))}-year"


def _nb_caption_context(cfg: Config, split: str) -> dict:
    """The figure 4 half of :func:`caption_context`.

    The decision-curve grid, arms and reference come from ``model_eval.net_benefit`` through
    :func:`src.eval_models.net_benefit_settings`, which is the same reader the estimator and
    the writer use, so the caption cannot state a plotted range or a comparator the figure
    does not draw. The cohort and calibration numbers come from :data:`NB_ANCHORS`.

    The direction word is ASSERTED rather than assumed. "Under-predicts" is a claim, not
    formatting, and if it flips on some future split the clause has to be rewritten instead
    of silently printing its opposite. The companion assertion, that the unrecalibrated
    reference arm is the better calibrated in the large, went with the clause it guarded:
    that clause was a DEFENCE of the recalibration asymmetry rather than a caveat a reader
    needs in order to read the panels, and it is not a claim the caption makes any more.
    ``reference_citl`` is still checked against ``{split}_metrics.csv`` in the suite.
    """
    s = net_benefit_settings(cfg)
    a = NB_ANCHORS[split]
    ref, arms = s["reference"], list(s["arms"])
    prot = nb_protagonist(s)              # the caption's subject IS panel B's subject
    citl, ref_citl = float(a["arm_citl"]), float(a["reference_citl"])
    assert citl > 0, (
        f"{MODULE}: FIG4_CALIBRATION says {prot} UNDER-predicts, but NB_ANCHORS[{split!r}] "
        f"records a calibration in the large of {citl:+.4f}; rewrite the clause rather than "
        "printing its opposite")
    return {
        "nb_arms_prose": _display_series([m for m in MODEL_DISPLAY if m in arms]),
        "nb_protagonist": MODEL_DISPLAY.get(prot, prot),
        "nb_reference": MODEL_DISPLAY.get(ref, ref),
        "nb_lo": f"{s['plot_min_pct'] / 100.0:.2f}",
        "nb_hi": f"{s['plot_max_pct'] / 100.0:.2f}",
        "nb_full_lo": f"{s['threshold_min_pct'] / 100.0:.2f}",
        "nb_full_hi": f"{s['threshold_max_pct'] / 100.0:.2f}",
        "nb_n_thresholds": int(s["thresholds"].size),
        "nb_sparse_min": int(s["sparse_events_min"]),
        "nb_horizon": int(s["horizon_days"]),
        "nb_prev": f"{float(a['prevalence']):.4f}",
        "nb_max_risk": f"{float(a['reference_max_risk']):.4f}",
        "nb_arm_n": int(a["arm_n"]),
        "nb_arm_ev": int(a["arm_events"]),
        "nb_citl": f"{citl:.4f}",
        "nb_citl_pp": f"{100.0 * citl:.1f}",
        "nb_ref_citl": f"{ref_citl:.4f}",
        "nb_slope": f"{NB_RECAL_SLOPE:.3f}",
    }


def _display_series(arms: list[str]) -> str:
    """'the A, B and C arms', in the ladder's own order. Display strings only."""
    names = [MODEL_DISPLAY.get(a, a) for a in arms]
    if len(names) == 1:
        return f"the {names[0]} arm"
    return "the " + ", ".join(names[:-1]) + f" and {names[-1]} arms"


def _find_risk_context() -> dict:
    """Figure 2's four predicted risks and their spread, from :data:`FIND_CASES`.

    Anchored rather than read, for the reason the anchor tables above give: a caption
    assembled from whatever ``interp_panel_manifest.csv`` happens to say cannot FAIL when
    that file moves, it just prints something else. ``render_figure2`` reads the manifest
    and asserts every one of these against it, and the suite does the same, so the caption
    and the tiles beside it are checked to agree rather than assumed to.
    """
    risks = [r for _, _, r in FIND_CASES]
    fold = max(risks) / min(risks)
    raw_fold = max(FIND_RAW_RISKS) / min(FIND_RAW_RISKS)
    return {
        "find_grade": f"{FIND_KLG:.0f}",
        "find_threshold": f"{FIND_THRESHOLD:.3f}",
        "find_classifiable": FIND_CLASSIFIABLE,
        "find_cases": _word(FIND_CASE_N),
        "find_controls": _word(FIND_CONTROL_N),
        "find_censored": FIND_CENSORED,
        "find_flagged_all": FIND_FLAGGED_ALL,
        "find_fold_raw": f"{raw_fold:.2f}",
        "find_grade_candidates": _word(FIND_GRADE_CANDIDATES),
        "find_fn_candidates": _word(FIND_FN_CANDIDATES),
        "find_risks": ", ".join(f"{r:.3f}" for r in risks[:-1]) + f" and {risks[-1]:.3f}",
        "find_fold": f"{fold:.1f}",
    }


def _border_pct(cfg: Config) -> float:
    """Percent of a finished crop the masked band blanks, from the INTEGER band.

    ``round(0.06 * 512) = 31`` px per edge, so ``1 - (450/512)^2 = 22.752%``. The
    continuous expression ``1 - (1 - 2f)^2`` gives 22.56% and is the number to avoid: it
    ignores that 31 > 0.06 x 512. The crop QA checklist, ``interp_regions.csv`` and the
    pipeline's own recorded ``masked_pct`` all carry the integer form.
    """
    pp = cfg["preprocess"]
    out_size = int(pp["out_size"])
    border_px = int(round(float(pp["mask_border_frac"]) * out_size))
    return 100.0 * (1.0 - ((out_size - 2 * border_px) / out_size) ** 2)


def _v6_caption_context(cfg: Config, split: str) -> dict:
    """Everything the four v6 imaging clauses and the supplementary clauses interpolate.

    The preprocessing numbers come from ``config.preprocess`` rather than from a literal,
    so a config edit moves the caption with the pipeline; the border percentage comes from
    :func:`_border_pct`, which uses the INTEGER band the pipeline actually zeroes, which is
    why it prints 22.75 and not 22.76 and never the 22.56 the continuous expression gives.
    """
    pp = cfg["preprocess"]
    out_size = int(pp["out_size"])
    border_frac = float(pp["mask_border_frac"])
    border_px = int(round(border_frac * out_size))
    border_pct = _border_pct(cfg)
    return {
        "half_inset_pct": f"{100.0 * float(pp['half_inset_frac']):g}",
        "crop_side_pct": f"{100.0 * float(pp['max_crop_frac']):g}",
        "out_size": out_size,
        "border_px": border_px,
        "border_pct": f"{border_pct:.2f}",
        "encoder": FIG1_ENCODER,
        "n_intervals": int(cfg["model_image"]["survival_head"]["n_intervals"]),
        "wf_crop_r": f"{FIG1_CROP_PEARSON:.2f}",
        # The residual-marker audit is a fixed measurement on the SEALED split's crops, so
        # it names that split by name rather than by the split being rendered: on a
        # validation render 1,216 would still be a test-split count and saying otherwise
        # would be the same class of error this clause exists to correct.
        "crop_audit_split": SPLIT_WORD[SEALED_SPLIT],
        "crop_audit_n": _word(CROP_AUDIT_N),
        "crop_marker_pct": f"{CROP_RESIDUAL_MARKER_PCT:g}",
        "crop_band_n": CROP_NONZERO_BAND_N,
        "crop_band_pct": f"{CROP_NONZERO_BAND_PCT:g}",
        "marker_delta": f"{MARKER_DELTA_AUROC:.5f}",
        "marker_delta_lo": f"{MARKER_DELTA_LO:.5f}",
        "marker_delta_hi": f"{MARKER_DELTA_HI:.5f}",
        "forest_ablation_n": FOREST_ABLATION_N,
        "forest_arms_drawn": _word(len(FOREST_ARMS)),
        "forest_arms_total": _word(FOREST_ARMS_TOTAL),
        "forest_occl_n": FOREST_OCCLUSION_N,
        "forest_border_identical": FOREST_BORDER_IDENTICAL,
        "forest_degenerate_tied": FOREST_DEGENERATE_TIED,
        "forest_degenerate_distinct": FOREST_DEGENERATE_DISTINCT,
        "forest_degenerate_auroc": f"{FOREST_DEGENERATE_AUROC:.3f}",
        "forest_meanfill_delta": f"{FOREST_MEANFILL_DELTA:.3f}",
        "forest_meanfill_lo": f"{FOREST_MEANFILL_LO:.3f}",
        "forest_meanfill_hi": f"{FOREST_MEANFILL_HI:.3f}",
        "cal_low_n": CAL_LOW_N,
        "cal_low_events": _word(CAL_LOW_EVENTS),
        "s1_patience": S1_PATIENCE,
        "s1_series": S1_SERIES,
        "s5_controls": S5_CONTROLS_BEYOND_HORIZON,
        "s6_pair_n": S6_PAIR_N,
        "s6_triple_n": S6_TRIPLE_N,
        "s6_m3_pct": f"{S6_M3_PCT:g}",
        "s6_m4_pct": f"{S6_M4_PCT:g}",
        **_find_risk_context(),
    }


def caption_context(cfg: Config, split: str) -> dict:
    """Everything a caption clause can interpolate, for one config and one split.

    Every width, horizon and bootstrap count comes from ``config/feasibility.yaml`` and every
    cohort count from :data:`SPLIT_ANCHORS`, so a config edit moves the caption with the
    analysis and a cohort change fails the render rather than leaving prose that quietly
    disagrees with the numbers.
    """
    assert_split(split)
    man = cfg["manuscript"]
    dpy = float(cfg["timeline"]["days_per_year"])
    hz = [int(h) for h in cfg["model_eval"]["horizons_days"]]
    long_h = hz[-1]
    anchors = SPLIT_ANCHORS[split]
    return {
        "split": split,
        "split_word": SPLIT_WORD[split],
        "Split_word": SPLIT_WORD_CAP[split],
        "cohort_word": "development" if split == VAL_SPLIT else SPLIT_WORD[split],
        "single_in": float(man["single_column_in"]),
        "double_in": float(man["double_column_in"]),
        "n_boot": int(cfg["model_eval"]["bootstrap_n"]),
        "long_lab": _horizon_label(long_h, dpy),
        "long_adj": _horizon_adj(long_h, dpy),
        "hz_prose": (", ".join(_horizon_label(h, dpy) for h in hz[:-1])
                     + f" and {_horizon_label(long_h, dpy)}"),
        "grid_max": int(round(float(cfg["timeline"]["horizon_years"]) * dpy)),
        "landmark": EXPECTED_N_LANDMARK,
        "dev_n": EXPECTED_DEV_N,
        "dev_ev": EXPECTED_DEV_EVENTS,
        "test_n": SPLIT_ANCHORS[SEALED_SPLIT]["n"],
        "test_ev": SPLIT_ANCHORS[SEALED_SPLIT]["events"],
        "split_n": anchors["n"],
        "split_ev": anchors["events"],
        "panel_b_n": anchors["panel_b_n"],
        "panel_b_ev": anchors["panel_b_events"],
        "fig3_n": anchors["crop_n"],
        "fig3_ev": anchors["crop_events"],
        **_crop_loss_context(crop_loss(split)),
        **_nb_caption_context(cfg, split),
        **_v6_caption_context(cfg, split),
    }


def _fig1_title(ctx: dict) -> str:
    return FIG1_TITLE.format(**ctx)


def _fig1_caption(ctx: dict) -> str:
    sealed = (FIG1_SEALED_NEVER_READ if ctx["split"] == VAL_SPLIT else FIG1_SEALED_READ_ONCE)
    crop = (FIG1_CROP_DEV if ctx["split"] == VAL_SPLIT else FIG1_CROP_TEST)
    return _sentences(FIG1_LEDGER, f"{FIG1_SEALED_LEAD} {sealed}", crop).format(**ctx)


def _fig3_title(ctx: dict) -> str:
    return FIG3_TITLE.format(**ctx)


def _fig3_caption(ctx: dict) -> str:
    return _sentences(FIG3_LEAD, FIG3_AT_RISK, FIG3_COMPETING,
                      FIG3_DENOMINATOR).format(**ctx)


def _fig4_title(ctx: dict) -> str:
    return FIG4_TITLE.format(**ctx)


def _fig4_caption(ctx: dict) -> str:
    val = ctx["split"] == VAL_SPLIT
    return _sentences(
        FIG4_LEAD, FIG4_ESTIMATOR, FIG4_PANEL_B, FIG4_WHY_NO_BANDS, FIG4_RANGE,
        FIG4_DENOMINATORS, FIG4_TREAT_ALL_SET, FIG4_CALIBRATION, FIG4_REGISTER,
        SEALED_READ_VAL if val else SEALED_READ_TEST,
    ).format(**ctx)


# --------------------------------------------------------------------------- #
# THE v6 CAPTION BUILDERS                                                       #
#                                                                              #
# The four main figures are new; the six supplementary ones REUSE the clause    #
# constants above wherever the content moved rather than changed, so the cohort #
# flow, the decision curve and the tertile curves keep the prose (and the       #
# anchors inside it) they were reviewed with.                                   #
# --------------------------------------------------------------------------- #
def _wf_title(ctx: dict) -> str:
    return WF_TITLE.format(**ctx)


def _wf_caption(ctx: dict) -> str:
    return _sentences(WF_PIPELINE, WF_NETWORK, WF_PROVENANCE, WF_RESIDUAL).format(**ctx)


def _find_title(ctx: dict) -> str:
    return FIND_TITLE.format(**ctx)


def _find_caption(ctx: dict) -> str:
    return _sentences(FIND_LEAD, FIND_ROWS, FIND_SPREAD, FIND_SCALE, FIND_MARKER,
                      SEALED_READ_VAL if ctx["split"] == VAL_SPLIT else SEALED_READ_TEST
                      ).format(**ctx)


def _forest_title(ctx: dict) -> str:
    return FOREST_TITLE.format(**ctx)


def _forest_caption(ctx: dict) -> str:
    return _sentences(FOREST_PANEL_A, FOREST_PANEL_B, FOREST_OCCLUSION, FOREST_CONTROLS,
                      FOREST_POSTHOC,
                      SEALED_READ_VAL if ctx["split"] == VAL_SPLIT else SEALED_READ_TEST
                      ).format(**ctx)


def _cal_title(ctx: dict) -> str:
    return CAL_TITLE.format(**ctx)


def _cal_caption(ctx: dict) -> str:
    return _sentences(CAL_PANEL_A, CAL_RECAL, CAL_PANEL_B, CAL_COMPETING, CAL_POSTHOC,
                      SEALED_READ_VAL if ctx["split"] == VAL_SPLIT else SEALED_READ_TEST
                      ).format(**ctx)


def _s1_title(ctx: dict) -> str:
    return S1_TITLE.format(**ctx)


def _s1_caption(ctx: dict) -> str:
    return _sentences(S1_LEAD, S1_GAP, S1_LOWER_BOUND).format(**ctx)


def _s5_title(ctx: dict) -> str:
    return S5_TITLE.format(**ctx)


def _s5_caption(ctx: dict) -> str:
    return _sentences(S5_LEAD, S5_STATES, S5_SCOPE, S5_CAVEAT).format(**ctx)


def _s6_title(ctx: dict) -> str:
    return S6_TITLE.format(**ctx)


def _s6_caption(ctx: dict) -> str:
    return _sentences(S6_LEAD, S6_PAIRED, S6_READING).format(**ctx)


@dataclass(frozen=True)
class FigureDef:
    """One manuscript figure: its number, its file, its width, its prose and its renderer.

    ONE ordered registry, so a figure cannot be renderable but uncaptioned (or captioned but
    never drawn), and so ``make_manuscript`` orders by ``number`` instead of by a tuple of
    keys it maintains separately. Registering a new figure is one entry appended here.
    """
    key: str
    number: int
    filename: str
    width_key: str                                # "single_column_in" | "double_column_in"
    title: Callable[[dict], str]
    caption: Callable[[dict], str]
    # ``None`` means the renderer declined to draw on this split and said why in the log; see
    # render_figure4 and render_all. It is not an error channel: every real failure raises.
    renderer: Callable[[Config, Path, str], Path | None]


def figures(cfg: Config, split: str) -> dict[str, dict]:
    """The figure registry for one config and one split, ordered by figure number.

    ``src/make_manuscript.py`` calls this for the numbering, titles and captions. The
    returned mapping carries exactly ``number``, ``filename``, ``width_in``, ``title`` and
    ``caption`` per key, and iterates in figure-number order.
    """
    ctx = caption_context(cfg, split)
    return {
        d.key: {
            "number": d.number,
            "filename": d.filename,
            "width_in": float(cfg["manuscript"][d.width_key]),
            "title": d.title(ctx),
            "caption": d.caption(ctx),
        }
        for d in FIGURE_DEFS
    }


def _spec(cfg: Config, split: str, key: str) -> dict:
    return figures(cfg, split)[key]


# --------------------------------------------------------------------------- #
# INPUT GUARDS                                                                  #
# --------------------------------------------------------------------------- #
def _require_file(path: Path, producer: str, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{MODULE}: {what} not found at {path}. Run `{producer}` first; this module only "
            "renders artefacts, it never recomputes them.")
    return path


def _import_model_clinical():
    """Import the frozen clinical helpers under the torch venv.

    ``src.model_clinical`` pulls in ``src.features_clinical`` and ``src.followup``, both of
    which import ``duckdb`` at module level for the Phase 1 ingestion path. An interpreter
    that deliberately does not carry duckdb can still render figures: nothing on the figure
    path calls it, because duckdb only appears inside function bodies and inside annotations,
    which ``from __future__ import annotations`` leaves unevaluated. A namespace stub
    therefore makes the pure numpy and pandas survival helpers importable without weakening
    anything. If duckdb is installed the real module is used and this branch never runs.
    """
    try:
        import duckdb                             # noqa: F401
    except ModuleNotFoundError:
        sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))
    from src import model_clinical                 # noqa: PLC0415 - deliberately lazy
    return model_clinical


# --------------------------------------------------------------------------- #
# FIGURE 1. COHORT FLOW                                                        #
# --------------------------------------------------------------------------- #
_PRIMARY_STRICT_RE = re.compile(r"\(primary[:\s]*([^;)]*);\s*strict[^)]*\)", re.IGNORECASE)


def _primary_only_description(desc: str) -> str:
    """Drop the strict-arm half of a two-arm parenthetical.

    ``outputs/cohort_flow.csv`` documents both the primary and the strict arm in one string.
    The figure renders the PRIMARY arm, so carrying "strict: coded only" into the box would
    describe a cohort that is not being drawn.
    """
    out = _PRIMARY_STRICT_RE.sub(lambda m: f"({m.group(1).strip()})", desc).strip()
    for ascii_op, glyph in FLOW_GLYPHS.items():
        out = out.replace(ascii_op, glyph)
    return out


def _wrap(text: str, width: int = FLOW_WRAP_CHARS) -> list[str]:
    """Wrap to the box width. Hyphens are never break points: "high-specificity" split across
    two lines reads as two words."""
    return textwrap.wrap(text, width=width, break_on_hyphens=False) or [text]


def _flow_boxes(cfg: Config, log: logging.Logger, split: str) -> list[dict]:
    """Assemble the Figure 1 box list: the ledger steps, then the two Phase 2 steps.

    Consecutive ledger rows that exclude nobody AND leave the count unchanged are merged into
    one box, because two identical counts stacked on top of each other read as an error.

    The ledger and the locked split table are split-independent and their five anchors are
    checked in full on every render. The six crop-side anchors are the same table RESTRICTED
    to the splits this render describes, because a validation render reads the development
    shard directory and a sealed render reads the test one.
    """
    assert_split(split)
    rendered = RENDERED_SPLITS[split]
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    flow_csv = _require_file(cfg.path(cfg["paths"]["outputs_dir"]) / "cohort_flow.csv",
                             "python3 -m src.cohort_flow", "the Phase 1 cohort assembly ledger")
    flow = pd.read_csv(flow_csv)
    for col in ("step", "description", "n_primary", "n_excluded_primary"):
        assert col in flow.columns, f"cohort_flow.csv is missing the {col!r} column"
    assert int(flow["n_primary"].iloc[-1]) == EXPECTED_N_LANDMARK, (
        f"protocol section 16: the ledger ends at {int(flow['n_primary'].iloc[-1]):,} primary "
        f"patients, expected {EXPECTED_N_LANDMARK:,}")

    boxes: list[dict] = []
    n_merged = 0
    for _, r in flow.iterrows():
        desc = _primary_only_description(str(r["description"]))
        n = int(r["n_primary"])
        excl = int(r["n_excluded_primary"])
        if boxes and excl == 0 and n == boxes[-1]["n"]:
            boxes[-1]["desc"].append(desc)
            n_merged += 1
            continue
        boxes.append({"desc": [desc], "n": n, "events": None, "sub": [],
                      "excl": {"lines": [f"Excluded (n = {excl:,})"]} if excl else None})
    log.info("figure 1: %d ledger rows collapsed to %d boxes (%d step(s) merged because they "
             "excluded nobody and left the count unchanged)", len(flow), len(boxes), n_merged)

    # ---- the two Phase 2 steps the Phase 1 ledger does not hold ----
    splits = pd.read_parquet(coh / "patient_splits.parquet", columns=["split"])
    counts = splits["split"].value_counts().to_dict()
    counts = {k: int(v) for k, v in counts.items()}
    assert counts == EXPECTED_SPLIT_N, (
        f"protocol section 17: split sizes moved, got {counts}, expected {EXPECTED_SPLIT_N}")
    assert sum(counts.values()) == EXPECTED_N_LANDMARK, "split table does not cover the cohort"

    frozen = json.loads((_require_file(coh / "clinical_imputation_params.json",
                                       "python3 -m src.features_clinical",
                                       "the frozen imputation metadata")).read_text())
    ev = {k: int(v) for k, v in frozen["split_event_counts"].items()}
    assert ev == EXPECTED_SPLIT_EVENTS, (
        f"frozen split event counts moved, got {ev}, expected {EXPECTED_SPLIT_EVENTS}")

    dev_n = counts["train"] + counts[VAL_SPLIT]
    dev_ev = ev["train"] + ev[VAL_SPLIT]
    assert dev_n == EXPECTED_DEV_N and dev_ev == EXPECTED_DEV_EVENTS, (
        f"development cohort moved: {dev_n} patients / {dev_ev} events, expected "
        f"{EXPECTED_DEV_N} / {EXPECTED_DEV_EVENTS}")

    # ---- crop-side anchors, restricted to the splits this render describes ----
    exp_crops_by_split = {s: SPLIT_ANCHORS[s]["crops"] for s in rendered}
    exp_crop_n = {s: SPLIT_ANCHORS[s]["crop_n"] for s in rendered}
    exp_crop_ev = {s: SPLIT_ANCHORS[s]["crop_events"] for s in rendered}
    exp_n_crops = sum(exp_crops_by_split.values())
    exp_crop_total = sum(exp_crop_n.values())
    cohort_n = sum(counts[s] for s in rendered)
    cohort_ev = sum(ev[s] for s in rendered)
    exp_lost_pat = crop_loss(split)               # == cohort_n - exp_crop_total, asserted above

    lab, lab_src = _crop_label_index(cfg, split)
    log.info("figure S2: crop counts read from %s", lab_src)
    assert len(lab) == exp_n_crops, \
        f"{lab_src} holds {len(lab):,} crops, expected {exp_n_crops:,}"
    crops_by_split = {k: int(v) for k, v in lab["split"].value_counts().items()}
    assert crops_by_split == exp_crops_by_split, \
        f"crops per split moved: {crops_by_split}, expected {exp_crops_by_split}"
    pat = lab.drop_duplicates("empi_anon")
    crop_n = {k: int(v) for k, v in pat["split"].value_counts().items()}
    crop_ev = {k: int(v) for k, v in pat.groupby("split")["event_indicator"].sum().items()}
    assert crop_n == exp_crop_n, \
        f"crop-bearing patients moved: {crop_n}, expected {exp_crop_n}"
    assert crop_ev == exp_crop_ev, \
        f"crop-bearing events moved: {crop_ev}, expected {exp_crop_ev}"
    crop_total = sum(crop_n.values())
    assert crop_total == exp_crop_total == int(pat["empi_anon"].nunique()), \
        f"crop-bearing {SPLIT_WORD[split]} patients {crop_total}, expected {exp_crop_total}"
    lost_pat = cohort_n - crop_total
    lost_ev = cohort_ev - sum(crop_ev.values())
    assert lost_pat == exp_lost_pat and lost_ev == 0, (
        f"crop attrition moved: {lost_pat} patients and {lost_ev} events lost, expected "
        f"{exp_lost_pat} and 0")

    multi = len(rendered) > 1
    if split == VAL_SPLIT:
        cohort_desc = ("Development cohort, after the locked 20% test split was set aside "
                       "unread")
        cohort_excl = ["Sealed test split, never read",
                       f"n = {counts[SEALED_SPLIT]:,} ({ev[SEALED_SPLIT]} events)"]
        crop_desc = "Development patients with at least one usable contralateral crop"
    else:
        cohort_desc = ("Test cohort, held sealed through development and read once after "
                       "every model was frozen")
        cohort_excl = ["Development cohort, used for fitting and model selection",
                       f"n = {dev_n:,} ({dev_ev} events)"]
        crop_desc = "Test patients with at least one usable contralateral crop"

    cohort_sub = (_wrap(", ".join(f"{SPLIT_WORD[s]} {counts[s]:,} ({ev[s]} events)"
                                  for s in rendered)) if multi else [])
    if multi:
        crop_sub = [", ".join(f"{SPLIT_WORD[s]} {crop_n[s]:,} ({crop_ev[s]} events)"
                              for s in rendered),
                    f"{len(lab):,} crops ("
                    + ", ".join(f"{SPLIT_WORD[s]} {crops_by_split[s]:,}" for s in rendered)
                    + ")"]
    else:
        crop_sub = [f"{len(lab):,} crops"]

    boxes.append({
        "desc": _wrap(cohort_desc),
        "n": cohort_n, "events": cohort_ev,
        "sub": cohort_sub,
        "excl": {"lines": cohort_excl},
    })
    boxes.append({
        "desc": _wrap(crop_desc),
        "n": crop_total, "events": sum(crop_ev.values()),
        "sub": crop_sub,
        "excl": {"lines": ["No crop survived the crop pipeline",
                           f"n = {lost_pat} ({lost_ev} events)"]},
    })

    for b in boxes:
        b["desc"] = [ln for d in b["desc"] for ln in _wrap(d)]
        b["sub"] = [ln for s in b["sub"] for ln in _wrap(s)]
        lines: list[tuple[str, bool]] = [(ln, False) for ln in b["desc"]]
        count = f"n = {b['n']:,}" + (f" ({b['events']} events)" if b["events"] is not None else "")
        lines.append((count, True))
        lines.extend((ln, False) for ln in b["sub"])
        b["lines"] = lines
        b["height"] = len(lines) + FLOW_BOX_PAD_UNITS
        if b["excl"]:
            b["excl"]["lines"] = [ln for s in b["excl"]["lines"]
                                  for ln in _wrap(s, FLOW_EXCL_WRAP_CHARS)]
            b["excl"]["height"] = len(b["excl"]["lines"]) + FLOW_BOX_PAD_UNITS
    return boxes


# --------------------------------------------------------------------------- #
# WHERE THE CROP COUNTS COME FROM                                              #
#                                                                              #
# The cohort-flow figure's last two boxes are recomputed at render time from   #
# the record of which radiographs were successfully prepared. That record has  #
# TWO copies and they are the same rows: the shard directory's ``labels.csv``, #
# written beside the images, and ``derived-data/cohort/preprocess_labels.csv``,#
# the cohort-level table for all 6,071 crops. The shard directory is machine-  #
# local staging (``model_image.local``) and is not part of the repository, so  #
# on a clone it is simply absent - and this figure then had no way to draw at  #
# all, even though every row it needs is checked in. The shard index is still  #
# PREFERRED, because it is the file written beside the bytes the model read;   #
# the cohort table is the fallback, restricted to the splits this render        #
# describes, and which one was used is logged and asserted against the same    #
# anchors either way.                                                          #
# --------------------------------------------------------------------------- #
CROP_LABEL_COLUMNS = ["empi_anon", "split", "event_indicator"]


def _crop_label_index(cfg: Config, split: str) -> tuple[pd.DataFrame, str]:
    """The finished-crop index for the splits ``split`` describes, and where it came from."""
    shard = cfg.path(cfg["model_image"]["local"][SHARD_DIR_KEY[assert_split(split)]]) / "labels.csv"
    if shard.exists():
        return pd.read_csv(shard, usecols=CROP_LABEL_COLUMNS), str(shard)
    cohort = _require_file(
        cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_labels.csv",
        "python3 -m src.preprocess_images",
        f"the shard label index at {shard}, or the cohort-level crop label table")
    df = pd.read_csv(cohort, usecols=CROP_LABEL_COLUMNS)
    keep = list(RENDERED_SPLITS[split])
    return df[df["split"].isin(keep)].reset_index(drop=True), f"{cohort} restricted to {keep}"


def _draw_box(ax, x0: float, x1: float, y_top: float, height: float,
              lines: list[tuple[str, bool]], *, fontsize: float) -> None:
    ax.add_patch(Rectangle((x0, y_top - height), x1 - x0, height, facecolor="white",
                           edgecolor="black", linewidth=0.7, zorder=2))
    xc = 0.5 * (x0 + x1)
    y = y_top - FLOW_BOX_PAD_UNITS / 2.0 - 0.5
    for text, bold in lines:
        ax.text(xc, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=("bold" if bold else "normal"), zorder=3)
        y -= 1.0


def render_cohort_flow(cfg: Config, out_dir: Path, split: str) -> Path:
    """Cohort flow, primary arm (protocol section 16). Double column."""
    log = logging.getLogger(MODULE)
    spec = _supp_spec(cfg, split, "figureS2")
    boxes = _flow_boxes(cfg, log, split)

    total_units = sum(b["height"] for b in boxes) + FLOW_GAP_UNITS * (len(boxes) - 1)
    width_in = float(spec["width_in"])
    height_in = total_units * FLOW_LINE_IN
    x0, x1 = FLOW_MAIN_X
    ex0, ex1 = FLOW_EXCL_X
    xc = 0.5 * (x0 + x1)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(width_in, height_in))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, total_units)
        ax.set_axis_off()

        y_top = total_units
        for i, b in enumerate(boxes):
            _draw_box(ax, x0, x1, y_top, b["height"], b["lines"], fontsize=SMALL_FONT_PT)
            y_bottom = y_top - b["height"]
            if i + 1 < len(boxes):
                nxt = boxes[i + 1]
                y_next_top = y_bottom - FLOW_GAP_UNITS
                ax.add_patch(FancyArrowPatch((xc, y_bottom), (xc, y_next_top),
                                             arrowstyle="-|>", mutation_scale=7,
                                             linewidth=0.7, color="black",
                                             shrinkA=0, shrinkB=0, zorder=1))
                if nxt["excl"]:
                    y_branch = y_bottom - FLOW_GAP_UNITS / 2.0
                    ax.add_patch(FancyArrowPatch((xc, y_branch), (ex0, y_branch),
                                                 arrowstyle="-|>", mutation_scale=7,
                                                 linewidth=0.7, color="black",
                                                 shrinkA=0, shrinkB=0, zorder=1))
                    eh = nxt["excl"]["height"]
                    _draw_box(ax, ex0, ex1, y_branch + eh / 2.0, eh,
                              [(ln, False) for ln in nxt["excl"]["lines"]],
                              fontsize=SMALL_FONT_PT)
                y_top = y_next_top

        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("figure 1 written: %s (%d boxes, %.2f x %.2f in, %s split)", out, len(boxes),
             width_in, height_in, split)
    return out


# --------------------------------------------------------------------------- #
# PREDICTED-RISK LOADERS (figures 2 and 3)                                      #
# --------------------------------------------------------------------------- #
# EXPECTED npz SCHEMA for derived-data/cohort/{split}_hazards_{arm}.npz, written by
# src/train_model.py (validation) and src/score_test.py (sealed). Alternative spellings are
# accepted because the writer was being finalised in parallel with this module; if none
# match, the error prints the keys found.
_NPZ_IDS = ("patient_ids", "pids", "empi_anon", "patient_id", "ids", "empi")
_NPZ_ENSEMBLE = ("hazards", "hazard", "hazards_ensemble", "hazard_ensemble",
                 "ensemble_hazards", "ensembled_hazards", "h_ensemble", "ensemble")
_NPZ_PER_SEED = ("hazards_per_seed", "per_seed_hazards", "seed_hazards", "hazards_seeds",
                 "hazard_seeds", "per_seed")
_NPZ_EDGES = ("edges", "interval_edges", "grid_edges")


def _first_key(npz, candidates: tuple[str, ...]) -> str | None:
    return next((k for k in candidates if k in npz.files), None)


def _hazards_producer(arm: str, split: str) -> str:
    return ("~/.venvs/mrkr-torch/bin/python -m src.train_model --arm " + arm
            if split == VAL_SPLIT
            else "~/.venvs/mrkr-torch/bin/python -m src.score_test --confirm-sealed-read")


def load_hazards(cohort_dir: Path, arm: str, split: str) -> dict:
    """Load one arm's per-interval hazards for ``split``.

    Expected contents of ``{split}_hazards_{arm}.npz`` (written by ``src/train_model.py`` on
    validation and by ``src/score_test.py`` on the sealed split):

    ==================  ====================  ==================================================
    key                 shape                 meaning
    ==================  ====================  ==================================================
    ``patient_ids``     (n_patients,)         patient id, giving the ROW ORDER of every array
    ``hazards``         (n_patients, n_int)   ensembled per-interval discrete-time hazards
    ``hazards_per_seed``(n_seeds, n, n_int)   optional; per-seed hazards before the ensemble
    ``edges``           (n_int + 1,)          optional; interval edges in days
    ==================  ====================  ==================================================

    If only the per-seed array is present the ensemble is formed here by averaging HAZARDS
    (config ``model_image.ensemble == "average_hazard"``), never by averaging risks.
    """
    assert_split(split)
    path = _require_file(Path(cohort_dir) / f"{split}_hazards_{arm}.npz",
                         _hazards_producer(arm, split),
                         f"{SPLIT_WORD[split]} hazards for arm {arm!r}")
    with np.load(path, allow_pickle=True) as npz:
        k_ids = _first_key(npz, _NPZ_IDS)
        k_ens = _first_key(npz, _NPZ_ENSEMBLE)
        k_seed = _first_key(npz, _NPZ_PER_SEED)
        k_edges = _first_key(npz, _NPZ_EDGES)
        if k_ids is None or (k_ens is None and k_seed is None):
            raise KeyError(
                f"{MODULE}: {path.name} does not match the expected schema. Found keys "
                f"{sorted(npz.files)}; need a patient-id key (one of {list(_NPZ_IDS)}) and a "
                f"hazard key (one of {list(_NPZ_ENSEMBLE)} or {list(_NPZ_PER_SEED)}). "
                "Reconcile with src/train_model.py.")
        ids = np.asarray(npz[k_ids]).astype(str).ravel()
        per_seed = np.asarray(npz[k_seed], dtype=float) if k_seed else None
        if k_ens is not None:
            haz = np.asarray(npz[k_ens], dtype=float)
        else:
            assert per_seed is not None and per_seed.ndim == 3, \
                f"{path.name}: {k_seed!r} must be (n_seeds, n_patients, n_intervals)"
            haz = per_seed.mean(axis=0)
        edges = np.asarray(npz[k_edges], dtype=float) if k_edges else None
    assert haz.ndim == 2, f"{path.name}: ensembled hazards must be 2-D, got shape {haz.shape}"
    assert haz.shape[0] == ids.size, \
        f"{path.name}: {ids.size} patient ids but {haz.shape[0]} hazard rows"
    return {"arm": arm, "patient_ids": ids, "hazards": haz, "per_seed": per_seed,
            "edges": edges, "path": path}


def interval_edges(cfg: Config, m0_json: dict) -> np.ndarray:
    """The discrete-time interval grid, ``linspace(0, 1826, n_intervals + 1)``.

    Transcribed from notebooks/train_colab.ipynb cell 18 so that the horizon risk computed
    here is the same quantity src/train_model.py optimised.
    """
    n_int = int(cfg["model_image"]["survival_head"]["n_intervals"])
    grid_max = float(m0_json["horizons"][-1]["horizon_days_nominal"])
    return np.linspace(0.0, grid_max, n_int + 1)


def risk_at_horizon(hazards, horizon_days: float, edges: np.ndarray) -> np.ndarray:
    """``1 - S(t)`` with a piecewise-constant hazard inside the interval containing ``t``.

    Transcribed from notebooks/train_colab.ipynb cell 21.
    """
    h = np.clip(np.asarray(hazards, dtype=float), 0.0, 1.0)
    t = float(horizon_days)
    assert 0.0 <= t <= edges[-1], f"horizon {t} outside the discrete grid [0, {edges[-1]}]"
    S = np.concatenate([np.ones((h.shape[0], 1)), np.cumprod(1.0 - h, axis=1)], axis=1)
    k = int(np.searchsorted(edges[1:], t, side="right"))
    if k >= h.shape[1]:
        return 1.0 - S[:, -1]
    frac = (t - edges[k]) / (edges[k + 1] - edges[k])
    return 1.0 - S[:, k] * np.power(1.0 - h[:, k], frac)


def _split_frame(cfg: Config, mc, split: str, log: logging.Logger | None = None
                 ) -> pd.DataFrame:
    """The rendered split's rows of the feature table.

    On validation the sealed rows are pushed out at read. On the sealed split they have to be
    materialised - that is the whole point of the render - and the caller has already passed
    :func:`src.eval_models.assert_sealed_read_is_recorded`, so the read is on the record and
    the models it describes are still the models on disk.

    Three things are enforced here rather than trusted, exactly as in
    ``src.make_manuscript.load_split_features``:

    * the reader is given :func:`forbid_test_rows`, so on a validation render the sealed rows
      are dropped in the Parquet predicate and never reach memory;
    * THE GATE RUNS HERE, at the point the sealed rows are materialised, and not only at the
      module's entry points. ``render_all`` calls it once for the whole run, but
      ``render_figure2`` and ``render_figure3`` are ordinary functions and a caller that
      invokes one of them directly with the sealed split - which the suite does deliberately
      - would otherwise materialise all 741 sealed rows with the gate never invoked. The
      branch used to fire only on the OTHER leak, a non-sealed render with
      ``model_eval.forbid_test_split`` turned off, so the sealed path itself was unguarded
      below ``render_all``. The invariant is "a sealed row is in memory only if the gate
      passed", and it is an invariant of this function because this function is where the
      rows appear. Calling it twice on a ``render_all`` pass costs two small JSON reads;
      not calling it on a direct call costs the guarantee;
    * the frame is then restricted to :func:`readable_splits` regardless, and the row-count
      assertion is taken from that same rule, so neither the rows the figure rests on nor the
      guard over them depends on a configuration flag. The assertion used to follow
      ``forbid``, which is why it stayed silent on precisely the path that leaked.
    """
    assert_split(split)
    log = log if log is not None else logging.getLogger(MODULE)
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    feats = _require_file(coh / "features_clinical.parquet",
                          "python3 -m src.features_clinical", "the clinical feature table")
    forbid = forbid_test_rows(cfg, split)
    allowed = readable_splits(split)
    sealed_allowed = SEALED_SPLIT in allowed
    if sealed_allowed or not forbid:
        # Either this render legitimately holds the sealed rows, or the flag is off and it
        # would hold them by accident. Both materialise them, so the single permitted sealed
        # read has to be on the record first, whichever entry point got here.
        contract = assert_sealed_read_is_recorded(cfg)
        if sealed_allowed:
            log.info("the %s render materialises the sealed rows of %s; the sealed read is "
                     "on the record (training contract %s)", split, feats.name, contract)
        else:
            log.warning("model_eval.forbid_test_split is off, so the %s render's read of %s "
                        "would materialise sealed rows; the sealed read is on the record "
                        "(training contract %s) and the rows are dropped again below",
                        split, feats.name, contract)
    dev = mc.load_development_frame(feats, forbid_test=forbid)
    if not set(dev["split"].unique()) <= set(allowed):
        dev = dev[dev["split"].isin(list(allowed))].reset_index(drop=True)
    n_expected = sum(SPLIT_ANCHORS[s]["n"] for s in allowed)
    frame_word = "landmark" if SEALED_SPLIT in allowed else "development"
    assert len(dev) == n_expected, \
        f"{frame_word} frame has {len(dev)} rows, expected {n_expected}"
    fr = dev[dev["split"] == split].reset_index(drop=True)
    assert len(fr) == EXPECTED_SPLIT_N[split], \
        f"{SPLIT_WORD[split]} frame has {len(fr)} rows, expected {EXPECTED_SPLIT_N[split]}"
    assert int(fr["event_indicator"].sum()) == EXPECTED_SPLIT_EVENTS[split], \
        f"{SPLIT_WORD[split]} event count moved"
    return fr


def cox_replay_risk(cfg: Config, mc, fr: pd.DataFrame,
                    arm: str) -> tuple[dict[float, np.ndarray], dict, np.ndarray]:
    """Replay a frozen Cox comparator on ``fr``'s rows. Nothing is refit.

    Returns ``(risk by horizon, the frozen model JSON, the eligible-row mask)``. ``m1`` is
    defined only where an inferred contralateral KLG was observed (protocol Secondary
    objective 2), so its mask is a strict subset and the rows it does not cover carry no
    risk rather than an imputed grade.
    """
    assert arm in COX_ARMS, f"{arm!r} is not a frozen Cox comparator; expected {list(COX_ARMS)}"
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    mj = json.loads((_require_file(coh / COX_MODEL_JSON[arm], "python3 -m src.model_clinical",
                                   f"the frozen {arm.upper()} clinical model")).read_text())
    col = COX_ELIGIBLE_COLUMN[arm]
    if col is None:
        mask = np.ones(len(fr), dtype=bool)
    else:
        assert col in fr.columns, (
            f"{arm}: the feature table has no {col!r} column, so the frozen eligibility rule "
            "cannot be replayed")
        mask = fr[col].to_numpy(dtype=int) == 0
    sub = fr[mask].reset_index(drop=True)
    X = mc.build_design(sub, mj["preprocessing"]["spline"],
                        list(mj["preprocessing"]["model_columns"]))
    assert list(X.columns) == list(mj["design_columns"]), \
        f"{arm.upper()} design column order differs from the frozen contract"
    _, risk = mc.replay_from_json(mj, X)
    return risk, mj, mask


def m0_replay_risk(cfg: Config, mc, fr: pd.DataFrame) -> tuple[dict[float, np.ndarray], dict]:
    """Replay the frozen M0 Cox model on ``fr``'s rows. Nothing is refit.

    M0 has no eligibility restriction, so it covers every row; the mask is asserted rather
    than returned. The name says ``replay`` and not ``validation`` because this was never
    split-dependent - the frozen JSON is the same object whichever rows are scored.
    """
    risk, m0, mask = cox_replay_risk(cfg, mc, fr, "m0")
    assert mask.all(), "the frozen M0 model must cover every row of the frame"
    return risk, m0


# --------------------------------------------------------------------------- #
# FIGURE 2. DISCRIMINATION AND CALIBRATION                                     #
# --------------------------------------------------------------------------- #
def _metrics_table(cfg: Config, split: str) -> pd.DataFrame:
    """``{split}_metrics.csv``, indexed by arm. The filename comes from ``split_path``."""
    assert_split(split)
    path = _require_file(split_path(cfg, "metrics_csv", split),
                         f"~/.venvs/mrkr-torch/bin/python -m src.eval_models --split {split}",
                         f"the {SPLIT_WORD[split]} metrics table")
    df = pd.read_csv(path)
    assert "arm" in df.columns, f"{path.name} is missing the 'arm' column"
    missing = [a for a in FIG2_MODELS if a not in set(df["arm"].astype(str))]
    if missing:
        raise KeyError(
            f"{MODULE}: {path} has no row for arm(s) {missing}. Train them with "
            "`~/.venvs/mrkr-torch/bin/python -m src.train_model` and score them with "
            f"`~/.venvs/mrkr-torch/bin/python -m src.eval_models --split {split}` before "
            "rendering figure 2.")
    return df.set_index(df["arm"].astype(str))


def _panel_b(ax, mc, risks: dict[str, np.ndarray], T: np.ndarray, E: np.ndarray,
             horizon: float) -> None:
    """Calibration at the long horizon: mean predicted risk against Kaplan-Meier risk.

    ``risks``, ``T`` and ``E`` have already been restricted to the patients every plotted arm
    scores, so the quintile boundaries and the observed risks are comparable across arms.
    """
    lim_hi = 0.0
    for arm in FIG2_MODELS:
        p = risks.get(arm)
        if p is None:
            continue
        bins = mc.risk_bins(p, CALIBRATION_BINS)
        xs, ys, los, his = [], [], [], []
        for b in range(CALIBRATION_BINS):
            k = bins == b
            obs, lo, hi = mc.km_risk(T[k], E[k], horizon)
            xs.append(float(np.mean(p[k]))); ys.append(obs); los.append(lo); his.append(hi)
        xs = np.array(xs); ys = np.array(ys)
        err = np.vstack([ys - np.array(los), np.array(his) - ys])
        st = MODEL_STYLE[arm]
        # Thinner, capless whiskers here than in panel A: five quintiles times four arms is
        # twenty overlapping Greenwood intervals, and caps turn that into a hedge.
        ax.errorbar(xs, ys, yerr=err, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], linewidth=1.0, markersize=4.0, capsize=0.0,
                    elinewidth=0.7, markeredgewidth=0.8, markerfacecolor="white",
                    label=MODEL_DISPLAY[arm], zorder=3)
        lim_hi = max(lim_hi, float(np.max(his)), float(np.max(xs)))
    lim = min(1.0, 1.05 * lim_hi) if lim_hi > 0 else 1.0
    ax.plot([0.0, lim], [0.0, lim], color="0.45", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xlim(0.0, lim); ax.set_ylim(0.0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted risk within quintile")
    ax.set_ylabel("Observed risk (Kaplan-Meier)")
    # Keyed by the shared legend below the panels, not in here; see :func:`_fig2_legend`.


def _model_legend(fig, axes) -> int:
    """One shared key for both panels, laid out in a single row beneath them.

    Both panels draw the same four arms, so an in-axes legend in each was the same key
    printed twice, each copy competing with the data for the corner it sat in. A
    figure-level legend gives the panels their whole plot area back and states the key
    once. Handles are collected in panel order and deduplicated by label, so an arm that
    only one panel could draw is still keyed, and the row order still follows
    :data:`FIG2_MODELS`. Returns the number of entries drawn.
    """
    seen: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    if not seen:
        return 0
    fig.legend(list(seen.values()), list(seen), loc="outside lower center", ncols=len(seen),
               handlelength=2.4, columnspacing=1.5, handletextpad=0.6, borderaxespad=0.0)
    return len(seen)


def expected_unscored(arm: str, split: str) -> int | None:
    """How many rows of the split frame an arm is EXPECTED to leave without a risk.

    ``None`` means "legitimately a subset arm, and the anchor table does not pin it": a
    frontal-only arm needs a frontal crop and ``m1`` needs an observed grade, so their
    denominators are cross-checked against ``{split}_metrics.csv`` instead. The arms that
    ARE pinned are the ones whose scored set is definitional: a frozen Cox comparator with no
    eligibility rule covers everyone, and an all-view image arm covers exactly the patients
    carrying at least one usable crop. That difference is 0 on validation and 1 on the sealed
    split, where one patient lost every crop to the protocol section 13 border mask - which is
    why "every patient has a finite risk" is the wrong assertion to make here.
    """
    anchors = SPLIT_ANCHORS[assert_split(split)]
    if arm in FULL_COHORT_ARMS:
        return 0
    if arm in ALL_VIEW_ARMS:
        return anchors["n"] - anchors["crop_n"]
    return None


def _arm_risks(cfg: Config, mc, fr: pd.DataFrame, horizon: float, arms: tuple[str, ...],
               split: str, log: logging.Logger,
               expected_n: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """Per-patient predicted risk at ``horizon`` for each arm, aligned to ``fr`` row order.

    A patient an arm cannot score carries ``nan``, and every ``nan`` has to be accounted for:
    against :data:`SPLIT_ANCHORS` where the arm's scored set is definitional, and against
    ``{split}_metrics.csv`` (via ``expected_n``) otherwise, so the figure and the metrics
    table cannot disagree about which patients an arm scored.
    """
    assert_split(split)
    coh = cfg.path(cfg["paths"]["cohort_dir"])
    _, m0_json = m0_replay_risk(cfg, mc, fr)
    edges = interval_edges(cfg, m0_json)
    pos = {str(p): i for i, p in enumerate(fr["empi_anon"].astype(str))}
    out: dict[str, np.ndarray] = {}
    for arm in arms:
        r = np.full(len(fr), np.nan)
        if arm in COX_ARMS:
            risk, _, mask = cox_replay_risk(cfg, mc, fr, arm)
            key = min(risk, key=lambda t: abs(t - horizon))
            assert abs(key - horizon) < 1.0, \
                f"the frozen {arm.upper()} model carries no horizon near day {horizon}"
            r[mask] = np.asarray(risk[key], dtype=float)
            log.info("figure 2: %s risk at day %d replayed from %s (%d of %d patients)",
                     arm, int(horizon), COX_MODEL_JSON[arm], int(mask.sum()), len(fr))
        else:
            hz = load_hazards(coh, arm, split)
            idx = np.array([pos[p] for p in hz["patient_ids"] if p in pos], dtype=int)
            assert idx.size == hz["patient_ids"].size, (
                f"{hz['path'].name}: {hz['patient_ids'].size - idx.size} patient id(s) are not "
                f"in the {SPLIT_WORD[split]} feature frame; the arm was scored on different "
                "patients")
            r[idx] = risk_at_horizon(hz["hazards"], horizon, hz["edges"]
                                     if hz["edges"] is not None else edges)
            log.info("figure 2: %s risk at day %d loaded from %s (%d patients, %d intervals)",
                     arm, int(horizon), hz["path"].name, *hz["hazards"].shape)
        n_unscored = int((~np.isfinite(r)).sum())
        exp = expected_unscored(arm, split)
        assert exp is None or n_unscored == exp, (
            f"{arm}: {n_unscored} of {len(fr)} {SPLIT_WORD[split]} patients have no predicted "
            f"risk, expected {exp} (SPLIT_ANCHORS[{split!r}]: {SPLIT_ANCHORS[split]['n']} "
            f"patients, {SPLIT_ANCHORS[split]['crop_n']} of them carrying a usable crop)")
        if expected_n is not None and arm in expected_n:
            assert len(fr) - n_unscored == expected_n[arm], (
                f"{arm}: the figure scores {len(fr) - n_unscored} patients but "
                f"{split}_metrics.csv records {expected_n[arm]}; the figure and the metrics "
                "table disagree about which patients the arm scored")
        out[arm] = r
    return out


def frozen_recalibration(cfg: Config, arm: str, horizon: float) -> dict | None:
    """The frozen cloglog recalibration for ``arm`` at ``horizon``, or None if it has none.

    Read from ``train_arms.json``, which is the same object ``src/eval_models.py`` reads at
    ``:646`` when it computes the calibration slope, calibration in the large and Brier
    score that Table 3 prints. The frozen Cox comparators are not in that file and carry no
    transform, which is correct: they were fitted as absolute-risk models and published as
    fitted.
    """
    path = cfg.path(cfg["paths"]["cohort_dir"]) / "train_arms.json"
    if not path.exists():
        return None
    arms = json.loads(path.read_text()).get("arms", {})
    recal = (arms.get(arm) or {}).get("recalibration")
    if not recal:
        return None
    key = str(float(horizon))
    assert key in recal, (
        f"{MODULE}: train_arms.json arm {arm!r} has no frozen recalibration at horizon "
        f"{key}; src/train_model.py writes one entry per horizon")
    return dict(recal[key])


def apply_frozen_recalibration(p: np.ndarray, recal: dict) -> np.ndarray:
    """``inv_cloglog(a + b cloglog(p))``, imported from the module that froze it.

    Imported rather than reimplemented, for the reason ``split_path`` exists: a second copy
    of the transform is a second thing that can drift from the numbers in Table 3.
    """
    from src.train_model import apply_recalibration        # noqa: PLC0415

    slope = float(recal["slope"])
    assert slope > 0, (
        f"{MODULE}: the frozen recalibration slope is {slope:+.4f}, which is not monotone "
        "increasing, so applying it would reverse the risk ordering")
    return np.asarray(apply_recalibration(np.asarray(p, dtype=float), recal), dtype=float)


# --------------------------------------------------------------------------- #
# FIGURE 3. CUMULATIVE INCIDENCE BY PREDICTED-RISK TERTILE                     #
#                                                                              #
# This renderer both DRAWS the curves and WRITES the tertile summary it drew,  #
# to outputs/tables/{split}_risk_tertiles.csv. The 5-year incidence in the     #
# lowest against the highest tertile is the clinically legible number - the    #
# one a surgeon answers a patient with - so it has to appear in the Results    #
# prose and not only in a picture. Letting src/make_manuscript.py recover it   #
# would mean a second patient-level scoring path in the document writer, with  #
# its own sealed read and its own copy of risk_at_horizon, which could         #
# silently disagree with the figure printed beside the sentence. That is the   #
# same failure mode figure 4 avoids from the other direction, so it gets the   #
# same fix: the computation writes a table, the consumer reads only the table. #
#                                                                              #
# ONE Kaplan-Meier fit per tertile feeds both outputs (:func:`tertile_curves`),#
# so "same code path" is structural rather than a promise.                     #
# --------------------------------------------------------------------------- #
# Where the summary is written. ``model_eval.risk_tertiles_csv`` is honoured if the config
# declares it, and :func:`split_path` then applies the house val_ -> test_ rewrite; if it
# does not, the path is derived from the SIBLING key ``net_benefit_csv`` - the other
# figure-facing table - and resolved through that same :func:`split_path`, so the rewrite
# still has exactly one implementation and the two tables cannot land in different
# directories. Declaring the key in config/feasibility.yaml is the preferred end state and
# needs no change here: the branch below simply stops being taken.
RISK_TERTILES_CSV_KEY = "risk_tertiles_csv"
RISK_TERTILES_BASENAME = "val_risk_tertiles.csv"

# The pinned schema, in the order the CSV carries. Everything figure 3's caption states is
# here, so the table is the single source for the sentence AND for the caption: the arm and
# split the numbers rest on, the horizon they are read at, the plotted range, the per-tertile
# denominators and predicted-risk range, the Kaplan-Meier incidence with its Greenwood
# interval, and the totals the caption quotes as "{fig3_n} patients and {fig3_ev} events".
RISK_TERTILE_COLUMNS = [
    "split", "arm", "tertile", "tertile_label", "horizon_days", "curve_max_day",
    "n_patients", "n_events", "n_at_risk_horizon",
    "min_predicted_risk", "max_predicted_risk", "mean_predicted_risk",
    "km_cumulative_incidence", "km_ci_lo", "km_ci_hi",
    "n_scored", "n_events_scored", "note",
]
# The caveat that has to travel WITH the number, not merely beside it in a caption: these
# are cause-agnostic incidences, because death is unascertainable in this data source
# (protocol section 10) and mortality is therefore an unmeasured competing event.
RISK_TERTILE_NOTE = ("cause-agnostic cumulative incidence; death is not ascertainable in "
                     "this data source, so competing mortality is unmeasured")


def risk_tertiles_path(cfg: Config, split: str) -> Path:
    """Where ``{split}_risk_tertiles.csv`` goes, resolved exactly as every other table is.

    Prefers ``model_eval.risk_tertiles_csv``. With that key absent the value is SYNTHESISED
    beside ``model_eval.net_benefit_csv`` and resolved through the same
    :func:`src.eval_models.split_path`, on a shallow copy of the config so the caller's
    object is never mutated. The point of the copy is that the ``val_`` to ``test_`` rewrite
    stays in one function: a local ``if split == "test"`` here would be the second
    implementation that ``split_path``'s docstring exists to forbid.
    """
    assert_split(split)
    me = cfg["model_eval"]
    if RISK_TERTILES_CSV_KEY in me:
        return split_path(cfg, RISK_TERTILES_CSV_KEY, split)
    sibling = str(me["net_benefit_csv"])
    assert Path(sibling).name.startswith("val_"), (
        f"{MODULE}: model_eval.net_benefit_csv is {sibling!r}, whose basename does not begin "
        f"'val_', so the sealed-split rewrite split_path performs would not fire on a name "
        f"derived from it. Declare model_eval.{RISK_TERTILES_CSV_KEY} explicitly instead")
    declared = str(Path(sibling).with_name(RISK_TERTILES_BASENAME))
    overlay = Config({**cfg, "model_eval": {**me, RISK_TERTILES_CSV_KEY: declared}})
    return split_path(overlay, RISK_TERTILES_CSV_KEY, split)


def risk_tertiles_output_path(cfg: Config, split: str, out_dir: Path | str | None) -> Path:
    """Where a render into ``out_dir`` writes ``{split}_risk_tertiles.csv``.

    :func:`risk_tertiles_path` when the images are going to the configured figures
    directory, and BESIDE THE IMAGES otherwise.

    Figure 3 is the only renderer that writes a repository TABLE as well as a PNG, and it
    wrote that table through the config path whatever ``--out-dir`` said. So ``--out-dir``
    was not in fact an isolation: a scratch render into a temporary directory still
    overwrote ``outputs/tables/{split}_risk_tertiles.csv``, which is the file
    ``src.make_manuscript.risk_tertile_sentence`` reads for the paper's most legible
    number. A render told to put its output somewhere else must put ALL of its output
    there, or "somewhere else" is not true.
    """
    configured = risk_tertiles_path(cfg, split)
    if out_dir is None:
        return configured
    figures_dir = cfg.path(cfg["manuscript"]["figures_dir"])
    # The supplementary subdirectory of the configured figures directory counts as the
    # configured location. This renderer moved into the supplementary set at v6 and its
    # out_dir became figures_dir/supplementary; without this branch the repository table
    # would land beside the supplementary PNGs and
    # src.make_manuscript.risk_tertile_sentence, which reads outputs/tables/, would silently
    # go on quoting whatever the last real render left behind.
    if Path(out_dir).resolve() in (figures_dir.resolve(),
                                   (figures_dir / SUPPLEMENT_DIRNAME).resolve()):
        return configured
    return Path(out_dir) / configured.name


@dataclass(frozen=True, eq=False)
class TertileCurve:
    """One tertile of predicted risk: the curve figure 3 draws AND the row the CSV carries.

    Both come off ONE :class:`lifelines.KaplanMeierFitter` fit, which is the whole reason
    this type exists. ``eq=False`` because two of the fields are arrays and a generated
    ``__eq__`` would return an array rather than a bool.
    """
    index: int                       # 1-based, ascending predicted risk
    label: str                       # the legend string the image draws
    n: int
    events: int
    risk_lo: float                   # the tertile's predicted-risk range, i.e. its cutpoints
    risk_hi: float
    risk_mean: float
    t: np.ndarray                    # step times, with the plot's right edge appended
    cif: np.ndarray                  # 1 - S(t), aligned to ``t``
    at_risk: list[int]               # number at risk at each drawn tick
    cif_horizon: float               # 1 - S(horizon), read off the SAME fit
    cif_ci_lo: float                 # Greenwood bounds, inverted onto the incidence scale
    cif_ci_hi: float
    n_at_risk_horizon: int


def tertile_curves(mc, pred: np.ndarray, T: np.ndarray, E: np.ndarray, *, horizon: float,
                   t_max: float, ticks: list[int]) -> list[TertileCurve]:
    """Fit each tertile's Kaplan-Meier ONCE and return everything both consumers need.

    ``horizon`` is ``model_eval.horizons_days[-1]``, the same day the predicted risk that
    defines the tertiles is taken at, and the same day the AUROC, the decision curve and the
    cohort's 5-year incidence are all reported at - so the paper quotes one horizon
    everywhere. ``t_max`` is the plot's right edge (5 x 365.25 = 1826 days), which is one day
    later; the curve is DRAWN out to it, and the number REPORTED is read at ``horizon``,
    which is a point on that same drawn curve.

    The interval is lifelines' exponential-Greenwood band on the survival scale, indexed with
    the same ``searchsorted`` rule ``src.model_clinical.km_risk`` uses (the helper figure 2
    panel B calls), and inverted onto the cumulative-incidence scale as
    ``1 - upper, 1 - lower``. It is not recomputed from a second fit, so the point estimate
    and its bounds are guaranteed to come from the same estimator as the line.
    """
    from lifelines import KaplanMeierFitter        # noqa: PLC0415 - keeps figure 1 import-light

    groups = mc.risk_bins(pred, RISK_TERTILES)
    out: list[TertileCurve] = []
    for g in range(RISK_TERTILES):
        k = groups == g
        kmf = KaplanMeierFitter()
        kmf.fit(T[k], event_observed=E[k])
        sf = kmf.survival_function_
        t = np.asarray(sf.index.values, dtype=float)
        cif = 1.0 - np.asarray(sf.iloc[:, 0].values, dtype=float)
        t = np.append(t, t_max); cif = np.append(cif, cif[-1])
        surv = float(np.asarray(kmf.predict(horizon)).ravel()[0])
        band = kmf.confidence_interval_
        idx = max(int(np.searchsorted(band.index.values, horizon, side="right")) - 1, 0)
        lo_s, hi_s = float(band.iloc[idx, 0]), float(band.iloc[idx, 1])
        out.append(TertileCurve(
            index=g + 1, label=TERTILE_STYLE[g]["label"],
            n=int(k.sum()), events=int(E[k].sum()),
            risk_lo=float(pred[k].min()), risk_hi=float(pred[k].max()),
            risk_mean=float(pred[k].mean()),
            t=t, cif=cif,
            at_risk=[int((T[k] >= x).sum()) for x in ticks],
            cif_horizon=1.0 - surv, cif_ci_lo=1.0 - hi_s, cif_ci_hi=1.0 - lo_s,
            n_at_risk_horizon=int((T[k] >= horizon).sum())))
    assert sum(c.n for c in out) == int(pred.size) and \
        sum(c.events for c in out) == int(E.sum()), (
        f"{MODULE}: the {RISK_TERTILES} tertiles hold {sum(c.n for c in out)} patients and "
        f"{sum(c.events for c in out)} events but were formed from {pred.size} and "
        f"{int(E.sum())}; a patient in no tertile would be drawn in no curve and counted in "
        "no row")
    return out


def build_risk_tertiles(curves: list[TertileCurve], *, split: str, arm: str,
                        horizon_days: int, curve_max_day: int, n_scored: int,
                        n_events_scored: int) -> pd.DataFrame:
    """The tertile summary as a frame in :data:`RISK_TERTILE_COLUMNS` order.

    Aggregate by construction: every value is a count, a bound or a group-level statistic
    over roughly a third of a split, and nothing here carries a row index or an identifier.
    :func:`src.eval_models.write_table` re-checks that on the way out anyway.
    """
    rows = [{
        "split": split, "arm": arm, "tertile": c.index, "tertile_label": c.label,
        "horizon_days": int(horizon_days), "curve_max_day": int(curve_max_day),
        "n_patients": c.n, "n_events": c.events, "n_at_risk_horizon": c.n_at_risk_horizon,
        "min_predicted_risk": c.risk_lo, "max_predicted_risk": c.risk_hi,
        "mean_predicted_risk": c.risk_mean,
        "km_cumulative_incidence": c.cif_horizon, "km_ci_lo": c.cif_ci_lo,
        "km_ci_hi": c.cif_ci_hi,
        "n_scored": int(n_scored), "n_events_scored": int(n_events_scored),
        "note": RISK_TERTILE_NOTE,
    } for c in curves]
    return pd.DataFrame(rows, columns=RISK_TERTILE_COLUMNS)


def render_cumulative_incidence(cfg: Config, out_dir: Path, split: str) -> Path:
    """Kaplan-Meier cumulative incidence by M4 risk tertile (protocol section 19).

    Single column: a Kaplan-Meier panel with one number-at-risk row per group is the
    canonical single-column figure, and the manuscript already carries two double-column
    figures, so this one is set at ``manuscript.single_column_in``.

    Writes ``{split}_risk_tertiles.csv`` after the image is on disk, so the table can never
    exist describing a figure that failed to render, and writes it where the images go: see
    :func:`risk_tertiles_output_path` for why ``--out-dir`` has to reach the table too.
    """
    log = logging.getLogger(MODULE)
    spec = _supp_spec(cfg, split, "figureS4")
    anchors = SPLIT_ANCHORS[assert_split(split)]
    mc = _import_model_clinical()
    horizons = [int(h) for h in cfg["model_eval"]["horizons_days"]]
    dpy = float(cfg["timeline"]["days_per_year"])
    t_max = float(round(float(cfg["timeline"]["horizon_years"]) * dpy))

    fr = _split_frame(cfg, mc, split)
    T = fr["time_from_landmark"].to_numpy(dtype=float)
    E = fr["event_indicator"].to_numpy(dtype=int)
    risks = _arm_risks(cfg, mc, fr, float(horizons[-1]), (FIG3_MODEL,), split, log)
    pred = risks[FIG3_MODEL]
    # The tertiles rest on the patients this arm scores, which on the sealed split is one
    # fewer than the split holds. The caption states that denominator, so assert it.
    scored = np.isfinite(pred)
    n_scored, ev_scored = int(scored.sum()), int(E[scored].sum())
    assert (n_scored, ev_scored) == (anchors["crop_n"], anchors["crop_events"]), (
        f"figure 3 rests on {n_scored} patients and {ev_scored} events scored by "
        f"{FIG3_MODEL}, but its caption states {anchors['crop_n']} and "
        f"{anchors['crop_events']}; the image and the caption must not disagree")
    pids = fr["empi_anon"].astype(str).tolist()
    pred, T, E = pred[scored], T[scored], E[scored]

    ticks = [int(round(f * t_max)) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    ticks[-1] = int(t_max)
    curves = tertile_curves(mc, pred, T, E, horizon=float(horizons[-1]), t_max=t_max,
                            ticks=ticks)
    at_risk = [c.at_risk for c in curves]
    width_in = float(spec["width_in"])
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(width_in, 1.18 * width_in), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 0.80], hspace=0.02)
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1])

        y_hi = 0.0
        for c in curves:
            st = TERTILE_STYLE[c.index - 1]
            ax.step(c.t, c.cif, where="post", color=st["color"], linestyle=st["linestyle"],
                    linewidth=1.2,
                    label=f"{st['label']} (n = {c.n}, {c.events} events)")
            y_hi = max(y_hi, float(c.cif.max()))

        ax.set_xlim(0.0, t_max)
        ax.set_ylim(0.0, min(1.0, max(0.40, 1.18 * y_hi)))
        ax.set_xticks(ticks)
        ax.set_ylabel("Cumulative incidence")
        ax.set_xlabel("Days from the landmark")
        ax.legend(loc="upper left", handlelength=2.4, borderaxespad=0.2)

        axr.set_xlim(0.0, t_max)
        axr.set_ylim(-0.5, RISK_TERTILES - 0.5)
        axr.set_xticks(ticks)
        axr.set_yticks(list(range(RISK_TERTILES)))
        axr.set_yticklabels([f"T{RISK_TERTILES - i}" for i in range(RISK_TERTILES)])
        axr.tick_params(axis="both", length=0, labelbottom=False)
        for side in ("top", "right", "bottom", "left"):
            axr.spines[side].set_visible(False)
        # The first and last columns are anchored inward: a centred count at day 0 would run
        # under the tertile tick label, and one at the last day would run off the panel.
        aligns = ["left"] + ["center"] * (len(ticks) - 2) + ["right"]
        for g in range(RISK_TERTILES):
            y = RISK_TERTILES - 1 - g
            axr.get_yticklabels()[y].set_color(TERTILE_STYLE[g]["color"])
            for x, n, ha in zip(ticks, at_risk[g], aligns):
                axr.text(x, y, f"{n}", ha=ha, va="center", fontsize=SMALL_FONT_PT,
                         color=TERTILE_STYLE[g]["color"])
        axr.text(0.0, 1.02, "Number at risk", transform=axr.transAxes,
                 fontsize=SMALL_FONT_PT, ha="left", va="bottom")

        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("figure 3 written: %s (%s tertiles of %s risk at day %d, %d %s patients, %d "
             "events total)", out, RISK_TERTILES, FIG3_MODEL, horizons[-1], n_scored,
             SPLIT_WORD[split], int(E.sum()))

    # The summary, written AFTER the image: a table describing a figure that failed to
    # render would be worse than no table. ``pids`` is the rendered split's own identifier
    # list, handed to write_table as the forbidden-value set, so the aggregate-only check
    # is made against the actual patients rather than against a generic pattern.
    tbl = build_risk_tertiles(curves, split=split, arm=FIG3_MODEL,
                              horizon_days=int(horizons[-1]), curve_max_day=int(t_max),
                              n_scored=n_scored, n_events_scored=ev_scored)
    csv = risk_tertiles_output_path(cfg, split, out_dir)
    write_table(csv, tbl, RISK_TERTILE_COLUMNS, pids, csv.name)
    log.info("figure 3 summary written: %s (%s incidence by tertile: %s)", csv,
             _horizon_label(int(horizons[-1]), dpy),
             " / ".join(f"{c.cif_horizon:.1%}" for c in curves))
    return out


# --------------------------------------------------------------------------- #
# FIGURE 4. DECISION-CURVE ANALYSIS                                            #
#                                                                              #
# This renderer DRAWS from one artefact: outputs/tables/{split}_net_benefit.csv #
# and reads one number besides it, the protagonist arm's verdict in             #
# {split}_convergence.csv, which decides whether there is a figure to draw at   #
# all (see decision_curve_decline_reason). It                                   #
# does not load hazards, does not touch derived-data/ and does not recompute a  #
# single net benefit. src/eval_models.py owns the estimator, holds the sealed   #
# read and owns the one shared bootstrap draw that makes the paired differences #
# paired; a second implementation here would need its own sealed-read path with #
# no gate and its own copy of risk_at_horizon that could silently disagree with #
# {split}_metrics.csv. The figure's job is what a figure can uniquely do:       #
# choose the plotted range and truncate a curve where its own sparse flag first #
# trips. The estimator flags rather than suppresses, so the truncation happens  #
# HERE and the table keeps every threshold it estimated.                        #
# --------------------------------------------------------------------------- #
# Treat-all and treat-none are references, not arms: thin, grey, behind the
# curves, and never in the colour vocabulary MODEL_STYLE reserves for models.
NB_TREAT_ALL_STYLE = {"color": "0.45", "linestyle": "-", "linewidth": 0.8}
NB_TREAT_NONE_STYLE = {"color": "0.45", "linestyle": (0, (1, 2)), "linewidth": 0.8}
# Panel B is coloured by the COMPARATOR being subtracted, not by the arm: both curves are
# the same arm's. Grey against black separates them in greyscale, which is what a printed
# difference panel with two shaded bands has to survive.
NB_DIFF_STYLE = {
    "treat_all": {"color": "0.40", "linestyle": "-", "alpha": 0.14},
    "reference": {"color": "#000000", "linestyle": "--", "alpha": 0.22},
}
NB_ZERO_LINE = {"color": "0.45", "linestyle": ":", "linewidth": 0.8}
NB_PREVALENCE_LINE = {"color": "0.80", "linestyle": (0, (1, 3)), "linewidth": 0.7}
NB_X_TICK_STEP_PCT = 5
# Panel A's vertical extent is set by the MODEL curves, not by treat-all. Treat-all falls
# without bound as the threshold rises (it is net-harmful by construction above the
# prevalence), so letting it set the axis would compress every curve the figure is about
# into the top third and would dramatise the one part of the gap that is mechanical rather
# than a finding. It is drawn and allowed to leave the frame; this is how far below zero the
# frame goes, as a fraction of its height above zero.
NB_PANEL_A_FLOOR_FRAC = 0.45


def _net_benefit_table(cfg: Config, split: str) -> pd.DataFrame:
    """``{split}_net_benefit.csv``. The only artefact figure 4 reads.

    The schema is asserted against the IMPORTED :data:`src.eval_models.NET_BENEFIT_COLUMNS`
    rather than against a second copy of the column list, so the writer extending its schema
    extends this check with it and a file written by anything else fails immediately.
    """
    assert_split(split)
    path = _require_file(split_path(cfg, "net_benefit_csv", split),
                         f"~/.venvs/mrkr-torch/bin/python -m src.eval_models --split {split}",
                         f"the {SPLIT_WORD[split]} decision-curve (net benefit) table")
    df = pd.read_csv(path)
    assert list(df.columns) == list(NET_BENEFIT_COLUMNS), (
        f"{path.name} does not carry the pinned net-benefit schema. Got {list(df.columns)}, "
        f"expected {list(NET_BENEFIT_COLUMNS)}; src.eval_models.write_table asserts this "
        "order on the way out, so a mismatch means the file was written by something else")
    held = set(df["split"].astype(str))
    assert held == {split}, (
        f"{path.name} describes split(s) {sorted(held)}, not {split!r}; the filename and "
        "the rows must not disagree about which patients the curve rests on")
    return df


def _net_benefit_table_if_present(cfg: Config, split: str) -> pd.DataFrame | None:
    """The same table, or ``None`` when it has not been produced for ``split``.

    The ONLY caller is :func:`render_figure4`, and only for the window between "is this
    figure declined?" and "this figure must draw, so its artefact has to exist". A table
    that IS there is read through :func:`_net_benefit_table` and gets every schema and
    split check it always got; a table that is not is not an answer to anything yet.

    Absence is deliberately not laundered into a decision here. Whether the figure declines
    is decided by :func:`decision_curve_decline_reason` from the convergence gate, and if it
    does not decline this returning ``None`` costs nothing: ``render_figure4`` then calls
    :func:`_net_benefit_table` and the missing artefact raises, naming its producer, exactly
    as it always has.
    """
    return (_net_benefit_table(cfg, split)
            if split_path(cfg, "net_benefit_csv", split).exists() else None)


def nb_flag(s: pd.Series) -> np.ndarray:
    """A CSV boolean column as a boolean array, whatever pandas inferred it to be.

    ``sparse`` and ``suppressed`` decide which thresholds are drawn, so reading them cannot
    depend on a dtype inference: one NaN anywhere in the column makes it ``object``, and
    ``Series.astype(bool)`` on object dtype is TRUE for every non-empty string, including
    ``"False"``. That failure mode draws exactly the rows the flags exist to withhold.
    """
    if s.dtype == bool:
        return s.to_numpy(dtype=bool)
    if s.dtype.kind in "iuf":
        return np.nan_to_num(s.to_numpy(dtype=float), nan=0.0) != 0.0
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "t"}).to_numpy()


def _nb_arm_rows(df: pd.DataFrame, arm: str, split: str, settings: dict,
                 log: logging.Logger) -> pd.DataFrame:
    """One arm's plotted rows: not suppressed, not past its sparse flag, inside the window.

    Truncation is the FIGURE's job. ``src/eval_models.py`` flags a threshold whose flagged
    set holds fewer than ``sparse_events_min`` observed events and keeps the estimate,
    because blanking exactly the thresholds at which a model flags few events would delete
    the evidence of imprecision a reader needs. The flagged sets are nested and shrink as the
    threshold rises, so the flag trips once; the drawn curve stops there.
    """
    d = df[df["arm"].astype(str) == arm].sort_values("threshold_pct")
    if d.empty:
        raise KeyError(
            f"{MODULE}: the {SPLIT_WORD[split]} net-benefit table has no row for arm "
            f"{arm!r}, which model_eval.net_benefit.arms names. Re-run "
            f"`~/.venvs/mrkr-torch/bin/python -m src.eval_models --split {split}` before "
            "rendering figure 4.")
    live = d[~nb_flag(d["suppressed"])]
    if len(live) < len(d):
        log.info("figure 4: %s has %d suppressed threshold(s) dropped by the convergence "
                 "gate", arm, len(d) - len(live))
    sparse = live[nb_flag(live["sparse"])]
    if not sparse.empty:
        cut = int(sparse["threshold_pct"].min())
        live = live[live["threshold_pct"] < cut]
        log.info("figure 4: %s truncated at p_t = %.2f, the first threshold whose flagged "
                 "set falls below the %d-event floor", arm, cut / 100.0,
                 int(settings["sparse_events_min"]))
    return live[(live["threshold_pct"] >= settings["plot_min_pct"])
                & (live["threshold_pct"] <= settings["plot_max_pct"])]


def nb_prevalence_from_treat_all(rows: pd.DataFrame) -> float:
    """Recover the cumulative incidence the treat-all curve crosses zero at.

    Treat-all is ``NB(p) = F - (1 - F) * w(p)`` with ``w = p / (1 - p)``, so it is zero
    exactly at ``p = F`` and ``F = (NB + w) / (1 + w)`` at EVERY threshold. Inverting it
    rather than reading a prevalence from somewhere else means the number annotated on the
    image is the number the drawn line actually crosses zero at, and the spread across
    thresholds is a free check that the treat-all column was computed consistently.
    """
    p = rows["threshold"].to_numpy(dtype=float)
    nb = rows["nb_treat_all_same_set"].to_numpy(dtype=float)
    w = p / (1.0 - p)
    f = (nb + w) / (1.0 + w)
    spread = float(np.max(f) - np.min(f))
    assert spread < 1e-4, (
        f"the treat-all column implies a {spread:.2e} spread in the cumulative incidence "
        "across thresholds; treat-all flags everyone at every threshold, so one Kaplan-Meier "
        "estimate has to serve them all and this means the column was not computed from it")
    return float(np.mean(f))


def nb_protagonist(settings: dict) -> str:
    """The arm panel B is about: the first configured arm that is not the reference.

    ``model_eval.net_benefit.arms`` is written protagonist first, but the definition is by
    ROLE rather than by position, so reordering the list cannot quietly move the caption's
    subject away from the arm whose differences are drawn.
    """
    ref = settings["reference"]
    return next((a for a in settings["arms"] if a != ref), ref)


def nb_treat_all_window(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """The reference arm's rows inside the plotted window: the treat-all curve panel A draws.

    Treat-all flags everyone, so it never goes sparse and is never truncated; the sparse rule
    belongs to the model curve, not to the reference the model curve is compared against.
    """
    ref = settings["reference"]
    lo_pct, hi_pct = settings["plot_min_pct"], settings["plot_max_pct"]
    window = df[(df["threshold_pct"] >= lo_pct) & (df["threshold_pct"] <= hi_pct)]
    out = window[window["arm"].astype(str) == ref].sort_values("threshold_pct")
    assert not out.empty, (
        f"the net-benefit table carries no rows for the reference arm {ref!r} inside the "
        f"plotted {lo_pct}-{hi_pct}% window, so there is no treat-all curve to draw")
    return out


def nb_arm_is_suppressed(df: pd.DataFrame, arm: str) -> bool | None:
    """Whether the net-benefit table blanked EVERY row of ``arm``; ``None`` if it holds none.

    The one-arm form of :func:`nb_suppressed_arms`, and the reading
    :func:`src.make_manuscript.net_benefit_suppressed` performs on the same file, so the two
    modules cannot answer differently about the same arm on the same split.

    The three-valued return is the point. ``False`` is a POSITIVE statement - this table
    kept the arm - and it is compared against the convergence gate's verdict; ``None`` says
    the table has nothing to say, which is what a table with no row for the arm means and is
    not evidence of anything. Collapsing the two would turn "no rows" into "not suppressed"
    and let a table written for another split silently agree with this one.
    """
    rows = df[df["arm"].astype(str) == str(arm)]
    if rows.empty:
        return None
    return bool(nb_flag(rows["suppressed"]).all())


def nb_convergence_status(cfg: Config, split: str, arm: str) -> str:
    """The convergence verdict recorded for ``arm`` on ``split``; :data:`STATUS_OK` if none.

    Reads ``outputs/tables/{split}_convergence.csv`` through
    :func:`src.eval_models.split_path` - the same file, resolved the same way, that
    ``src.make_manuscript`` reads for the same question - and mirrors
    ``src.make_manuscript.Inputs.arm_status``: the gate writes a row only for an arm it
    classified, so no row means no verdict rather than a bad one.

    A MISSING table is also no verdict, and that is deliberate rather than lenient.
    Declining to draw a figure needs a positive, checkable statement about this split, and
    the absence of an artefact is never one. A render with no convergence table therefore
    proceeds and fails, if it fails at all, on the artefact the figure actually draws from.
    """
    assert_split(split)
    path = split_path(cfg, "convergence_csv", split)
    if not path.exists():
        return STATUS_OK
    conv = pd.read_csv(path)
    if conv.empty or not {"arm", "status"} <= set(conv.columns):
        return STATUS_OK
    hit = conv[conv["arm"].astype(str) == str(arm)]
    return STATUS_OK if hit.empty else str(hit.iloc[0]["status"])


def decision_curve_decline_reason(cfg: Config, split: str,
                                  df: pd.DataFrame | None = None) -> str:
    """Why figure 4 has no honest render on ``split``, or ``""`` when it has one.

    Mirrors :func:`src.make_manuscript.decision_curve_decline_reason` clause for clause,
    because the two modules answer the same question about the same split and a document
    that omits figure 4 while the renderer draws it (or the reverse) is a contradiction
    rather than a configuration.

    The rule is a POSITIVE statement about the reported split: the decision curve has no
    honest figure when its protagonist arm's convergence verdict disqualifies it HERE. It is
    emphatically not "the net-benefit table is missing, so the figure must be declined" -
    absence is not a reason, and reading it as one would swallow exactly the failure the
    missing-artefact error exists to make loud.

    Because the rule is the convergence gate and not the table, it is answerable BEFORE the
    table is required, which is the whole point: ``val_net_benefit.csv`` has never been
    produced, and a validation render used to die on that file while asking a question whose
    answer was already "this figure declines". When the table IS present it is passed in and
    cross-checked, and a disagreement raises: its ``suppressed`` column and this rule are two
    readings of one gate (``src.eval_models.suppress_unfit_contrasts``), so if they differ,
    one of them was computed against a different split.
    """
    assert_split(split)
    settings = net_benefit_settings(cfg)
    arm = nb_protagonist(settings)
    status = nb_convergence_status(cfg, split, arm)
    gate_says = status in disqualifying_statuses(split)
    table_says = None if df is None else nb_arm_is_suppressed(df, arm)
    if table_says is not None and table_says != gate_says:
        raise AssertionError(
            f"{MODULE}: the {SPLIT_WORD[split]} net-benefit table "
            f"{'suppressed' if table_says else 'kept'} every row of {arm}, but its "
            f"convergence verdict {status!r} in "
            f"{split_path(cfg, 'convergence_csv', split).name} says it is "
            f"{'disqualified' if gate_says else 'interpretable'} on this split. The table's "
            f"suppressed column and DISQUALIFYING are two readings of one gate and cannot "
            f"disagree; one of them was computed against a different split")
    if not gate_says:
        return ""
    return (f"{arm} is the decision curve's protagonist and its convergence verdict in "
            f"{split_path(cfg, 'convergence_csv', split).name} is {status!r}, which "
            f"disqualifies it on the {SPLIT_WORD[split]} split, so panel B has no estimate "
            f"to draw and the caption's calibration clauses describe an arm with no curve")


def nb_suppressed_arms(df: pd.DataFrame, settings: dict) -> dict[str, str]:
    """The configured arms the convergence gate blanked on this split, and its own reason.

    ``src.eval_models.suppress_unfit_contrasts`` keeps every row of a failing arm, NaNs each
    estimate, sets ``suppressed`` and states why in ``note``. ``did_not_converge`` suppresses
    on both splits; ``severe_overfit`` on VALIDATION ONLY, because a checkpoint selected on
    validation is then scored on validation and that estimate is circular, while the sealed
    split took no part in the selection and its estimate is the one that reveals whether the
    overfitting cost anything.

    So an arm with nothing to draw is a LEGITIMATE state on validation, not a broken table:
    today every image arm is ``severe_overfit`` there, the decision curve's protagonist among
    them. Returned per arm, with the estimator's own sentence, so the render logs the reason
    the artefact gives rather than inventing one.

    An arm is counted only when EVERY row it carries is suppressed. The gate works per arm,
    so a partly suppressed arm would mean the table was written by something else, and the
    row-level filter in :func:`_nb_arm_rows` handles that case on its own.
    """
    out: dict[str, str] = {}
    for arm in settings["arms"]:
        if not nb_arm_is_suppressed(df, arm):      # None (no rows) and False both mean no
            continue
        d = df[df["arm"].astype(str) == arm]
        why = [n for n in d["note"].astype(str) if "SUPPRESSED" in n]
        out[arm] = why[0] if why else "every row carries suppressed = True"
    return out


def _nb_panel_a(ax, df: pd.DataFrame, split: str, settings: dict, prevalence: float,
                ref_window: pd.DataFrame, log: logging.Logger) -> None:
    """The curves, plus treat-all and treat-none. No intervals, deliberately."""
    # The two references are labelled ON the plot rather than in the legend. Six legend
    # entries do not fit inside a single-column panel without covering the curves, and these
    # two are the entries a reader least needs spelled out: treat-none IS the zero line and
    # treat-all is named by the crossing annotation below.
    ax.axvline(prevalence, zorder=0, **NB_PREVALENCE_LINE)
    ax.axhline(0.0, zorder=1, **NB_TREAT_NONE_STYLE)
    ax.plot(ref_window["threshold"].to_numpy(dtype=float),
            ref_window["nb_treat_all_same_set"].to_numpy(dtype=float),
            zorder=1, **NB_TREAT_ALL_STYLE)

    lo_seen, hi_seen = [0.0], [0.0, float(ref_window["nb_treat_all_same_set"].max())]
    for arm in settings["arms"]:
        d = _nb_arm_rows(df, arm, split, settings, log)
        if d.empty:
            log.warning("figure 4: %s draws no curve; every plotted threshold is suppressed "
                        "or past its sparse flag", arm)
            continue
        y = d["net_benefit"].to_numpy(dtype=float)
        st = MODEL_STYLE[arm]
        ax.plot(d["threshold"].to_numpy(dtype=float), y, color=st["color"],
                linestyle=st["linestyle"], linewidth=1.1, label=MODEL_DISPLAY[arm], zorder=3)
        lo_seen.append(float(np.min(y))); hi_seen.append(float(np.max(y)))

    hi = max(hi_seen)
    lo = min(min(lo_seen) - 0.06 * (hi - min(lo_seen)), -NB_PANEL_A_FLOOR_FRAC * hi)
    top = hi + 0.10 * (hi - lo)
    ax.set_ylim(lo, top)
    ax.plot([prevalence], [0.0], marker="o", markersize=3.2, color=NB_TREAT_ALL_STYLE["color"],
            markeredgewidth=0.0, zorder=4)
    # Above and to the left of the crossing: every curve has fallen well below its starting
    # value by the prevalence, so this corner is the one part of the panel that is reliably
    # empty, and the legend can then keep the bottom left.
    ax.text(prevalence - 0.005, top, f"Treat all crosses\nzero at {prevalence:.4f}",
            ha="right", va="top", fontsize=SMALL_FONT_PT, color="0.30", zorder=4)
    ax.text(settings["plot_min_pct"] / 100.0 + 0.002, 0.004, "Treat none", ha="left",
            va="bottom", fontsize=SMALL_FONT_PT, color="0.30", zorder=4)
    ax.set_ylabel("Net benefit")
    ax.legend(loc="lower left", handlelength=2.4, borderaxespad=0.2, labelspacing=0.30)


def _nb_panel_b(ax, df: pd.DataFrame, split: str, settings: dict, prevalence: float,
                log: logging.Logger) -> None:
    """The PAIRED differences for the protagonist arm, each with a pointwise band.

    The bands live here and nowhere else. At the 5-year prevalence the protagonist's own
    marginal interval overlaps the reference arm's, which reads as "no difference", while the
    paired difference is clear of zero: a marginal interval discards the pairing the one
    shared bootstrap draw exists to provide. Pointwise, and stated as pointwise; no
    adjustment across thresholds, because the thresholds are that many views of one curve
    rather than that many hypotheses.
    """
    ref = settings["reference"]
    arm = nb_protagonist(settings)
    d = _nb_arm_rows(df, arm, split, settings, log)
    # Reaching here empty is now a real defect rather than a legitimate state: render_figure4
    # has already returned None if the convergence gate blanked this arm, so what is left is
    # a table whose whole plotted window is past the sparse floor or outside the grid.
    assert not d.empty, (
        f"the net-benefit table leaves {arm!r} with no drawable threshold inside the plotted "
        f"window, so figure 4 panel B would be empty. {arm!r} is NOT suppressed (render_"
        f"figure4 checks that first and declines to draw), so every threshold from "
        f"{settings['plot_min_pct']}% to {settings['plot_max_pct']}% is past the "
        f"{settings['sparse_events_min']}-event floor or missing from the grid")
    x = d["threshold"].to_numpy(dtype=float)
    ax.axvline(prevalence, zorder=0, **NB_PREVALENCE_LINE)
    ax.axhline(0.0, zorder=1, **NB_ZERO_LINE)
    for key, cols, label in (
            ("treat_all", "diff_vs_treat_all", "minus treat all"),
            ("reference", "diff_vs_reference", f"minus {MODEL_DISPLAY[ref]}")):
        st = NB_DIFF_STYLE[key]
        y = d[cols].to_numpy(dtype=float)
        lo = d[f"{cols}_lo"].to_numpy(dtype=float)
        hi = d[f"{cols}_hi"].to_numpy(dtype=float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=st["color"], alpha=st["alpha"],
                        linewidth=0.0, zorder=2)
        ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], linewidth=1.1,
                label=f"{MODEL_DISPLAY[arm]} {label}", zorder=3)
    ax.set_ylabel("Difference in net benefit")
    ax.legend(loc="upper left", handlelength=2.4, borderaxespad=0.2, labelspacing=0.30)


def render_decision_curve(cfg: Config, out_dir: Path, split: str) -> Path | None:
    """Decision-curve analysis (protocol section 18, exploratory). Single column.

    Two stacked panels sharing one threshold axis: the curves above, the protagonist arm's
    paired differences below. Nothing is estimated here; see the section comment above.

    Returns the path written, or ``None`` on a split where the convergence gate disqualifies
    the protagonist arm and there is therefore no honest figure to draw. That is the only
    renderer in this module that may decline; :func:`render_all` drops the key rather than
    embedding an image the caption does not describe.

    THE DECISION IS TAKEN BEFORE THE NET-BENEFIT TABLE IS REQUIRED, and that ordering is the
    contract rather than an implementation detail; see the block comment below.
    """
    log = logging.getLogger(MODULE)
    spec = _supp_spec(cfg, split, "figureS3")
    anchors = NB_ANCHORS[assert_split(split)]
    settings = net_benefit_settings(cfg)
    ref_arm, prot_arm = settings["reference"], nb_protagonist(settings)

    # A SPLIT WHOSE PROTAGONIST THE CONVERGENCE GATE DISQUALIFIES HAS NO HONEST FIGURE 4, so
    # none is drawn. On validation every image arm is severe_overfit - which disqualifies on
    # validation and not on the sealed split, see DISQUALIFYING - so panel B's subject has no
    # estimate anywhere and panel A would silently lose two of its four curves. Failing here
    # instead would be wrong twice over: the state is legitimate, and the assertion in
    # _nb_panel_b would abort the whole validation render for figures that have nothing to do
    # with the decision curve.
    #
    # THE ORDER MATTERS. This used to be decided from the net-benefit table's own suppressed
    # column, which meant requiring that table first - and on validation the answer was
    # "declined" while the file it was being read from has never been produced, so a whole
    # validation render died on a FileNotFoundError for an artefact the figure was never
    # going to draw. The verdict is a property of the SPLIT and the gate, not of the
    # filesystem, so it is asked of {split}_convergence.csv, which exists for both splits and
    # is the same gate src.eval_models.suppress_unfit_contrasts applies when it writes the
    # suppressed column. Absence of the net-benefit table is NOT a reason to decline: a
    # figure that should draw and whose artefact is missing still raises below, naming its
    # producer, and when the table is present the two readings are cross-checked against each
    # other in decision_curve_decline_reason.
    #
    # Falling panel B back to the first surviving non-reference arm was REJECTED, not
    # overlooked. On this study that arm is m1, a frozen Cox model, and three caption clauses
    # would become false of it: FIG4_CALIBRATION asserts the protagonist UNDER-predicts and
    # m1's validation calibration in the large is negative, FIG4_ASYMMETRY asserts the
    # reference arm is the better calibrated in the large and against m1 it is not, and
    # FIG4_MONOTONE describes a frozen cloglog recalibration that only the image arms carry.
    # Two of those are asserted in _nb_caption_context and would raise. Redirecting the panel
    # therefore needs the caption rewritten rather than re-pointed, and a panel B drawn for
    # an arm the caption does not describe is worse than no panel B.
    df = _net_benefit_table_if_present(cfg, split)
    decline = decision_curve_decline_reason(cfg, split, df)
    if decline:
        nb_name = split_path(cfg, "net_benefit_csv", split).name
        if df is None:
            corroborates = (f"{nb_name} has not been produced, which is NOT the reason: the "
                            f"verdict above is read from the convergence gate")
        elif nb_arm_is_suppressed(df, prot_arm):
            corroborates = f"{nb_name} agrees, and carries no live row for {prot_arm}"
        else:                                      # no rows at all; a disagreement raised
            corroborates = f"{nb_name} carries no row for {prot_arm} either way"
        log.warning("figure 4 is NOT drawn for the %s split: %s. %s. This is the gate "
                    "working, not a render failure.", SPLIT_WORD[split], decline, corroborates)
        return None
    # Not declined, so this figure is meant to be drawn and the one artefact it draws from
    # has to be on disk. A genuinely missing table names its producer here, exactly as it
    # always did; what changed is only that it is no longer asked for on a split that had
    # already decided not to draw.
    if df is None:
        df = _net_benefit_table(cfg, split)

    horizons = {int(h) for h in df["horizon_days"]}
    assert horizons == {int(settings["horizon_days"])}, (
        f"the net-benefit table was estimated at horizon(s) {sorted(horizons)} but "
        f"model_eval.net_benefit.horizon_days is {settings['horizon_days']}; the caption "
        "states the configured one")
    refs = {str(r) for r in df["reference"].dropna().astype(str) if str(r).strip()}
    assert refs <= {settings["reference"]}, (
        f"the net-benefit table takes differences against {sorted(refs)} but "
        f"model_eval.net_benefit.reference is {settings['reference']!r}")

    scored: dict[str, int] = {}
    for a in settings["arms"]:
        col = df.loc[df["arm"].astype(str) == a, "n_scored"]
        if col.empty:
            continue
        assert col.nunique() == 1, (
            f"the net-benefit table gives {a} {sorted(set(col))} as its screened denominator "
            "at different thresholds; net benefit is per patient SCREENED, so one arm has "
            "one denominator across its whole curve")
        scored[a] = int(col.iloc[0])
    missing = [a for a in settings["arms"] if a not in scored]
    if missing:
        raise KeyError(
            f"{MODULE}: {split_path(cfg, 'net_benefit_csv', split)} has no row for arm(s) "
            f"{missing}. Score them with `~/.venvs/mrkr-torch/bin/python -m src.eval_models "
            f"--split {split}` before rendering figure 4.")

    # The image and the caption must not disagree: the prevalence the treat-all line will
    # cross zero at, and the three denominators the caption names, are checked BEFORE the
    # figure is drawn, so a drift never leaves a wrong PNG on disk for someone to embed.
    ref_window = nb_treat_all_window(df, settings)
    prevalence = nb_prevalence_from_treat_all(ref_window)
    assert abs(prevalence - float(anchors["prevalence"])) < NB_PREVALENCE_TOL, (
        f"figure 4 would draw treat-all crossing zero at {prevalence:.4f}, but its caption "
        f"states {float(anchors['prevalence']):.4f} (NB_ANCHORS[{split!r}]['prevalence']); "
        f"the {SPLIT_WORD[split]} split's {int(settings['horizon_days'])}-day cumulative "
        "incidence moved")
    # A list, not a dict: on a one-arm configuration the reference IS the protagonist, and
    # a dict would silently drop one of the two checks by key collision.
    for arm, want in ((ref_arm, SPLIT_ANCHORS[split]["n"]),
                      (prot_arm, int(anchors["arm_n"]))):
        assert scored[arm] == want, (
            f"figure 4: the net-benefit table scores {arm} on {scored[arm]} "
            f"{SPLIT_WORD[split]} patients but the caption states {want}")
    assert min(scored.values()) == SPLIT_ANCHORS[split]["panel_b_n"], (
        f"figure 4: the smallest decision-curve denominator is {min(scored.values())} but "
        f"the caption states {SPLIT_ANCHORS[split]['panel_b_n']}, the patients every arm "
        f"scores; per-arm denominators are {scored}")

    width_in = float(spec["width_in"])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(width_in, 1.34 * width_in), sharex=True,
                                 height_ratios=[1.32, 1.0], layout="constrained")
        _nb_panel_a(axes[0], df, split, settings, prevalence, ref_window, log)
        _nb_panel_b(axes[1], df, split, settings, prevalence, log)
        lo, hi = settings["plot_min_pct"] / 100.0, settings["plot_max_pct"] / 100.0
        ticks = [t / 100.0 for t in range(NB_X_TICK_STEP_PCT,
                                          settings["plot_max_pct"] + 1, NB_X_TICK_STEP_PCT)]
        axes[1].set_xlim(lo, hi)
        axes[1].set_xticks(ticks)
        axes[1].set_xlabel("Threshold probability at which to act")
        for ax, letter in zip(axes, "AB"):
            ax.text(-0.19, 1.02, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
                    fontweight="bold", ha="left", va="bottom")
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("figure 4 written: %s (%d arm(s) over p_t %.2f-%.2f, treat-all crosses zero at "
             "%.4f, panel B on %s, %s split, denominators %s)", out, len(scored),
             settings["plot_min_pct"] / 100.0, settings["plot_max_pct"] / 100.0, prevalence,
             prot_arm, SPLIT_WORD[split], scored)
    return out


# =========================================================================== #
# FIGURE 1. THE IMAGING AND MODELLING WORKFLOW, OVER A REAL RADIOGRAPH          #
#                                                                              #
# The source DICOM archive is gone. crop_stages() still returns every           #
# intermediate by design, but it needs the film, so the pipeline cannot be      #
# re-run to produce a pre-crop image. The ONLY surviving picture of a film      #
# before cropping is the left sub-panel of a reviewer QA panel under            #
# derived-data/cohort/qa_panels/, and those are rendered matplotlib figures: a  #
# grayscale film with a green rectangle round the half that was kept, red and   #
# green overlay text, and a second sub-panel holding the crop.                  #
#                                                                              #
# So the film is RECOVERED rather than re-read, and every step of the recovery  #
# is stated in the caption:                                                     #
#   * the green rectangle locates the film axes exactly (it is drawn from -0.5  #
#     to h-0.5 in data coordinates, so its vertical extent IS the axes');       #
#   * the overlay is repaired by filling each coloured pixel from the nearest   #
#     achromatic pixel in its own column - the overlay is a 2 px outline and    #
#     two blocks of 4 pt text over dark background, so this changes nothing     #
#     outside it and, unlike a blur, cannot invent anatomy;                     #
#   * the panel draws the film with aspect="auto", which STRETCHES it to fill   #
#     the axes, so the recovered film is resampled back to the true aspect      #
#     ratio taken from img_height / img_width in the source image table.        #
#                                                                              #
# The half-select bounds and the localization box are then computed by calling  #
# src/preprocess_images.py's own arithmetic, not measured off the picture, and  #
# the reconstruction is CHECKED: resampling the computed box reproduces the     #
# shipped crop at Pearson r = 0.99. That check is what licenses drawing the box #
# at all.                                                                       #
# =========================================================================== #
QA_PANEL_DIRNAME = "qa_panels"
FIGURE_ASSET_DIRNAME = "assets"
FIG1_ASSET_JSON = "figure1_assets.json"
FIG1_ASSET_CROP = "figure1_crop_final.png"
# Chroma above which a panel pixel is overlay rather than film. The film is strictly
# achromatic (imshow with cmap="gray"), so anything with a channel spread this large was
# drawn on top of it. PNG is lossless, so the threshold has margin either way.
PANEL_CHROMA_THRESHOLD = 28
PANEL_WHITE = 245                                # figure background, for the axes hunt
IMAGE_INTERPOLATION = "lanczos"


def _tables_dir(cfg: Config) -> Path:
    return cfg.path(cfg["paths"]["tables_dir"])


def _read_table(cfg: Config, name: str, producer: str, what: str) -> pd.DataFrame:
    return pd.read_csv(_require_file(_tables_dir(cfg) / name, producer, what))


def _interp_table(cfg: Config, name: str) -> pd.DataFrame:
    return _read_table(cfg, name,
                       "~/.venvs/mrkr-torch/bin/python -m src.interpretability --stage all",
                       f"the interpretability table {name}")


def interpretability_dir(cfg: Config) -> Path:
    """Where ``src/interpretability.py`` writes its panels and arrays."""
    return cfg.path(cfg["paths"]["outputs_dir"]) / "interpretability"


def figure_assets_dir(cfg: Config) -> Path:
    """Where the recovered Figure 1 imaging assets are cached.

    Beside the figures rather than inside ``manuscript.figures_dir``: ``render_all``
    treats that directory as its own output and a cleanup of it must not take the one
    surviving full-resolution crop with it.
    """
    return cfg.path(cfg["paths"]["figures_dir"]) / FIGURE_ASSET_DIRNAME


def _repair_overlay(rgb: np.ndarray) -> np.ndarray:
    """Grayscale copy of ``rgb`` with every coloured overlay pixel filled from its column."""
    hi = rgb.max(axis=2).astype(np.int16)
    lo = rgb.min(axis=2).astype(np.int16)
    bad = (hi - lo) > PANEL_CHROMA_THRESHOLD
    out = rgb.mean(axis=2).astype(np.float32)
    for x in np.nonzero(bad.any(axis=0))[0]:
        m = np.nonzero(bad[:, x])[0]
        good = np.nonzero(~bad[:, x])[0]
        if good.size == 0:
            continue
        nxt = np.clip(np.searchsorted(good, m), 0, good.size - 1)
        prv = np.clip(nxt - 1, 0, good.size - 1)
        take = np.where(np.abs(good[nxt] - m) <= np.abs(good[prv] - m), good[nxt], good[prv])
        out[m, x] = out[take, x]
    return out


def read_qa_panel(cfg: Config, key: str) -> dict:
    """Recover the full film and the pre-mirror crop from one reviewer QA panel.

    Returns ``film`` (float32, overlay repaired, aspect NOT yet corrected), ``crop``
    (the panel's own rendering of the pre-mirror crop) and the pixel bounds of the
    kept-half rectangle the panel drew, which is what proves the panel carries a full film
    at all: a crop-only panel has no rectangle and this raises.
    """
    from PIL import Image                          # noqa: PLC0415 - only needed here

    path = _require_file(cfg.path(cfg["paths"]["cohort_dir"]) / QA_PANEL_DIRNAME / f"{key}.png",
                         "python3 -m src.crop_qa", f"the reviewer QA panel for {key}")
    rgb = np.asarray(Image.open(path).convert("RGB"))
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    green = (g > 150) & (r < 140) & (b < 190) & (chroma > 60)
    ys, xs = np.nonzero(green)
    assert ys.size, (
        f"{MODULE}: {path.name} carries no kept-half rectangle, so it is a crop-only panel "
        "and holds no full film. The sealed split's panels are all crop-only; Figure 1's "
        "case must come from the development split")
    ry0, ry1, rx0, rx1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    nonwhite = rgb.min(axis=2) < PANEL_WHITE
    dense = np.nonzero(nonwhite[ry0:ry1 + 1].mean(axis=0) > 0.6)[0]
    breaks = np.nonzero(np.diff(dense) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [dense.size - 1]])
    runs = [(int(dense[s]), int(dense[e])) for s, e in zip(starts, ends)]
    film_run = next((c for c in runs if c[0] <= rx0 and c[1] >= rx1), None)
    assert film_run is not None, \
        f"{MODULE}: could not bracket {path.name}'s film axes around its kept-half rectangle"
    ax0, ax1 = film_run
    film = _repair_overlay(rgb[ry0:ry1 + 1, ax0:ax1 + 1])

    # The crop sub-panel is drawn with the default equal aspect, so it is NOT stretched and
    # is simply the largest non-white block to the right of the film axes.
    right = nonwhite[:, ax1 + 1:]
    cys, cxs = np.nonzero(right)
    assert cys.size, f"{MODULE}: {path.name} has no crop sub-panel"
    crop = rgb[cys.min():cys.max() + 1, ax1 + 1 + cxs.min():ax1 + 1 + cxs.max() + 1].mean(axis=2)
    return {"path": path, "film": film, "crop": crop.astype(np.float32),
            "rect_cols": (rx0 - ax0, rx1 - ax0 + 1)}


def _source_image_dims(cfg: Config, sop_uid: str) -> tuple[int, int]:
    """(rows, columns) of the acquired film, from the delivered image inventory."""
    src = cfg.path(cfg["paths"]["source_parquet_dir"]) / "image.parquet"
    _require_file(src, "python3 -m src.inventory", "the delivered image inventory")
    df = pd.read_parquet(src, columns=["SOPInstanceUID_anon", "img_height", "img_width"])
    row = df[df["SOPInstanceUID_anon"].astype(str) == str(sop_uid)]
    assert len(row) == 1, \
        f"{MODULE}: the image inventory holds {len(row)} rows for the Figure 1 case, expected 1"
    return int(float(row["img_height"].iloc[0])), int(float(row["img_width"].iloc[0]))


def _resample(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos resample to ``(width, height)``. Same filter the crop pipeline uses."""
    from PIL import Image                          # noqa: PLC0415

    return np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
        size, Image.LANCZOS), dtype=np.float32)


def figure1_assets(cfg: Config, key: str = FIG1_CASE_KEY) -> dict:
    """Every image and every geometric bound Figure 1 draws, for one case.

    The crop tiles are the cached full-resolution asset: the shipped 512 px shard image,
    staged once by ``--build-assets`` into ``outputs/figures/assets/`` and committed, which
    is the only copy of it that survives.

    **D5. Its absence is a HARD FAILURE and no longer a fallback.** This function used to
    fall back to the QA panel's own rendering of the same crop, log a warning, and report
    the reconstruction as unchecked - and the caption went on printing "a Pearson
    correlation of 0.99" from a module constant on a path where the two assertions that
    verify it are skipped. The guarding tests skipped with it, and the asset lives outside
    ``manuscript.figures_dir``, so a cleaned figures directory rendered a figure asserting
    a check nobody had performed. It also drew the wrong picture: the fallback tiles are
    the panel's downsampled rendering, not the array the network was given, which is the
    one thing the third and fourth tiles are for.
    """
    from src.preprocess_images import half_column_bounds   # noqa: PLC0415

    log = logging.getLogger(MODULE)
    pp = cfg["preprocess"]
    lab = pd.read_csv(_require_file(cfg.path(cfg["paths"]["cohort_dir"]) / "preprocess_labels.csv",
                                    "python3 -m src.preprocess_images",
                                    "the finished crop label index"))
    rows = lab[lab["key"].astype(str) == key]
    assert len(rows) == 1, f"{MODULE}: {key!r} names {len(rows)} finished crops, expected 1"
    row = rows.iloc[0]
    assert str(row["half_selected"]) in ("left", "right"), (
        f"{MODULE}: Figure 1's case must be a bilateral film the pipeline half-selected; "
        f"{key!r} has half_selected={row['half_selected']!r}")

    panel = read_qa_panel(cfg, key)
    true_h, true_w = _source_image_dims(cfg, str(row["sop_uid"]))
    h0, w0 = panel["film"].shape
    # Undo the panel's aspect="auto" stretch by growing the axis that was compressed, so
    # nothing is thrown away.
    if true_h / true_w >= h0 / w0:
        new_w, new_h = w0, int(round(w0 * true_h / true_w))
    else:
        new_h, new_w = h0, int(round(h0 * true_w / true_h))
    film = _resample(panel["film"], (new_w, new_h))

    c0, c1 = half_column_bounds(new_w, str(row["half_selected"]), float(pp["half_inset_frac"]))
    half = film[:, c0:c1]
    # The localizer under preprocess.localizer_mode == "center_default" is a deterministic
    # centred box of max_crop_frac x the half's short side. crop_confidence is 1.0 exactly
    # when that estimate was used, which is the case this figure requires.
    assert str(row["crop_method"]) == "center_default" and float(row["crop_confidence"]) == 1.0, (
        f"{MODULE}: Figure 1 draws the centred localization box, so its case must have been "
        f"cropped by it; {key!r} used {row['crop_method']!r} at confidence "
        f"{row['crop_confidence']}")
    hh, hw = half.shape
    side = int(round(float(pp["max_crop_frac"]) * min(hh, hw)))
    r0 = int(round(hh / 2.0 - side / 2.0))
    k0 = int(round(hw / 2.0 - side / 2.0))

    from PIL import Image                          # noqa: PLC0415

    asset = _require_file(
        figure_assets_dir(cfg) / FIG1_ASSET_CROP,
        f"~/.venvs/mrkr-torch/bin/python -m {MODULE} --build-assets <path>/val-00000.tar",
        "the shipped 512 px crop for the Figure 1 case. It is committed, so this is "
        "missing only if it was deleted. The caption states a reconstruction correlation "
        "and the two assertions that verify it need the shipped array; drawing the figure "
        "without them would publish a check that was never run")
    final = np.asarray(Image.open(asset).convert("L"), dtype=np.float32)
    assert final.shape == (int(pp["out_size"]), int(pp["out_size"])), \
        f"{MODULE}: {asset.name} is {final.shape}, expected the shipped crop size"
    mirrored = bool(row["mirrored"])
    pre = final[:, ::-1] if mirrored else final

    # THE CHECK THAT LICENSES THE BOX. Replay the pipeline's own geometry on the recovered
    # film and compare with the crop that shipped. It runs unconditionally now: there is no
    # longer a path on which the caption prints the correlation and this does not measure it.
    box = np.full((side, side), float(np.median(np.concatenate([half[0], half[-1]]))),
                  dtype=np.float32)
    sr0, sc0 = max(0, r0), max(0, k0)
    sr1, sc1 = min(hh, r0 + side), min(hw, k0 + side)
    box[sr0 - r0:sr1 - r0, sc0 - k0:sc1 - k0] = half[sr0:sr1, sc0:sc1]
    out_size = int(pp["out_size"])
    band = int(round(float(pp["mask_border_frac"]) * out_size))
    rebuilt = _resample(box, (out_size, out_size))
    rebuilt[:band] = 0.0; rebuilt[-band:] = 0.0
    rebuilt[:, :band] = 0.0; rebuilt[:, -band:] = 0.0
    inner = np.zeros_like(rebuilt, dtype=bool)
    inner[band:-band, band:-band] = True
    pearson = float(np.corrcoef(rebuilt[inner], pre[inner])[0, 1])
    assert pearson >= FIG1_CROP_PEARSON_MIN, (
        f"{MODULE}: replaying the pipeline geometry on the recovered film reproduces the "
        f"shipped crop at r = {pearson:.4f}, below {FIG1_CROP_PEARSON_MIN}; the "
        f"localization box this figure draws would not be the box the pipeline used")
    assert abs(round(pearson, 2) - FIG1_CROP_PEARSON) < 5e-3, (
        f"{MODULE}: the caption states a reconstruction correlation of "
        f"{FIG1_CROP_PEARSON:.2f} and the render measures {pearson:.4f}")

    # D1. The caption says the marker detector accepted nothing here and that a burned-in
    # character survives, so the render checks both on the array it is about to draw. The
    # blanked fraction is EXACTLY the integer band, which is what "accepted nothing" means
    # arithmetically, and the survivor is a saturated blob inside the retained region.
    zero_frac = float((final == 0).mean())
    band_only = 1.0 - ((out_size - 2 * band) / out_size) ** 2
    assert abs(zero_frac - band_only) < 1e-9, (
        f"{MODULE}: the Figure 1 crop is {100.0 * zero_frac:.4f} percent zero and the band "
        f"alone is {100.0 * band_only:.4f} percent. The caption says the marker detector "
        "accepted nothing on this crop, which is true only while those are equal")
    sat_level = int(pp["marker_sat_level"])
    min_px = int(pp["marker_min_px"])
    keep = final[band:-band, band:-band]
    n_sat = int((keep >= sat_level).sum())
    assert n_sat >= min_px, (
        f"{MODULE}: the caption says a saturated burned-in character survives inside the "
        f"retained region of the Figure 1 crop, and only {n_sat} pixel(s) there reach the "
        f"pipeline's own saturation level of {sat_level}, below its own {min_px} px floor")
    log.info("figure 1: case %s, film recovered at %dx%d and restored to %dx%d (true %dx%d), "
             "half columns [%d, %d), localization box %d px at (%d, %d), reconstruction "
             "r = %.4f, crop %.4f%% zero (band alone %.4f%%), %d saturated px surviving "
             "inside the retained region", key, w0, h0, new_w, new_h, true_w, true_h, c0, c1,
             side, r0, k0, pearson, 100.0 * zero_frac, 100.0 * band_only, n_sat)
    return {"key": key, "film": film, "half": half, "pre": pre, "final": final,
            "half_cols": (c0, c1), "box": (r0, k0, side), "mirrored": mirrored,
            "band": band, "pearson": pearson, "zero_frac": zero_frac,
            "band_only_frac": band_only, "residual_saturated_px": n_sat,
            "contra_side": str(row["contra_side"]), "view": str(row["view"]),
            "split": str(row["split"]), "masked_pct": float(row["masked_pct"])}


def assert_marker_audit_anchors(cfg: Config) -> None:
    """D1 and D2. The residual-marker numbers Figures 1 and 2 print, against their tables.

    Both captions disclose that the crop on the page still carries a burned-in character
    and quote what that class of survivor is worth across the split. Neither number is
    recomputed here - they are measurements this module does not own - so both are checked
    against the artefact that does own them before either figure draws, exactly as Figure
    3's degeneracy counts are.
    """
    audit = _interp_table(cfg, "interp_regions.csv")
    values = {str(k): str(v) for k, v in zip(audit["item"], audit["value"])}
    for item, want, tol, what in (
            ("residual_marker_crops_pct", CROP_RESIDUAL_MARKER_PCT, 5e-2,
             "residual-marker rate"),
            ("n_crops_with_nonzero_border_band", CROP_NONZERO_BAND_N, 0.5,
             "count of crops with a non-zero band"),
            # The same 22.75 percent Figure 1's caption prints, from the table that
            # measured it on the real crops rather than from either derivation of it.
            ("border_band_area_fraction", 0.01 * float(_border_pct(cfg)), 5e-6,
             "blanked band fraction")):
        assert item in values, (
            f"{MODULE}: interp_regions.csv has no {item!r} row, so the captions' "
            f"{what} cannot be checked against it")
        assert abs(float(values[item]) - float(want)) < tol, (
            f"{MODULE}: the crop audit reports a {what} of {values[item]} and the captions "
            f"state {want}")
    # The scan's DENOMINATOR is carried only in the note columns ("n=1216", "of 1216 test
    # crops = 11.9%"), so the notes are parsed rather than trusted: a note that stops
    # carrying the number fails here rather than letting the caption keep printing it.
    note = " ".join(str(v) for v in audit.loc[
        audit["item"].astype(str).isin(
            ["residual_marker_crops_pct", "n_crops_with_nonzero_border_band"]),
        "note"].tolist())
    assert str(CROP_AUDIT_N) in note, (
        f"{MODULE}: the captions state {CROP_AUDIT_N} scanned crops and the audit table's "
        f"notes do not carry that number: {note!r}")
    assert f"{CROP_NONZERO_BAND_PCT:g}%" in note, (
        f"{MODULE}: the captions state {CROP_NONZERO_BAND_PCT:g} percent of crops carry a "
        f"non-zero band and the audit table's notes say: {note!r}")
    occl = _interp_table(cfg, "interp_occlusion.csv")
    r = _one_row(occl, {"arm": INTERP_ARM, "condition": MARKER_DELTA_CONDITION},
                 "the residual-marker leakage row")
    for got, want, what in ((-float(r["delta_auroc"]), MARKER_DELTA_AUROC, "estimate"),
                            (-float(r["delta_auroc_hi"]), MARKER_DELTA_LO, "lower bound"),
                            (-float(r["delta_auroc_lo"]), MARKER_DELTA_HI, "upper bound")):
        assert abs(got - want) < 5e-6, (
            f"{MODULE}: blanking the residual markers has a {what} of {got:.6f} as a "
            f"reduction and the captions state {want}")
    # "The only row of the masked band and burned-in marker block whose interval excludes
    # zero", checked over that block and NOT over the whole table. Three conditions in the
    # table exclude zero - this one, the degenerate band-only control, and the mean-filled
    # joint-only occlusion - so the unqualified claim would be false, and the scope is part
    # of what is asserted here rather than something a reader has to take on trust.
    block = {c for c, _ in FOREST_MASKING_ROWS}
    excludes = {str(x["condition"]) for _, x in occl.iterrows()
                if str(x["arm"]) == INTERP_ARM and str(x["condition"]) in block
                and not (float(x["delta_auroc_lo"]) <= 0.0 <= float(x["delta_auroc_hi"]))}
    assert excludes == {MARKER_DELTA_CONDITION}, (
        f"{MODULE}: Figure 2's caption calls the residual-marker row the only row of the "
        f"masked band and burned-in marker block whose interval excludes zero; the table "
        f"says {sorted(excludes)}")


def build_figure1_assets(cfg: Config, shard_tar: Path, key: str = FIG1_CASE_KEY) -> Path:
    """Stage the shipped full-resolution crop for ``key`` out of a shard tar. Run once.

    The shards live on removable staging, not in the repository, so this is a one-off
    command rather than part of a render: it copies ONE 512 px image and a provenance
    record into ``outputs/figures/assets/`` and everything afterwards reads those.
    """
    import io                                      # noqa: PLC0415
    import tarfile                                 # noqa: PLC0415

    from PIL import Image                          # noqa: PLC0415

    log = logging.getLogger(MODULE)
    out_dir = figure_assets_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(_require_file(Path(shard_tar), "src/preprocess_images.py",
                                    "the shard tar holding the finished crops")) as tf:
        member = next((m for m in tf if m.isfile() and Path(m.name).stem == key
                       and Path(m.name).suffix.lower() in (".png", ".jpg", ".jpeg")), None)
        assert member is not None, f"{MODULE}: {key!r} is not in {shard_tar}"
        img = Image.open(io.BytesIO(tf.extractfile(member).read())).convert("L")
    crop = out_dir / FIG1_ASSET_CROP
    img.save(crop, format="PNG", optimize=True)
    record = {
        "written_by": MODULE,
        "case_key": key,
        "crop_file": FIG1_ASSET_CROP,
        "crop_sha256": sha256_file(crop),
        "crop_size": list(img.size),
        "shard_member": member.name,
        "shard_tar_name": Path(shard_tar).name,
        "why": ("the source DICOM archive is gone and the shard tars are not in the "
                "repository; this is the shipped 512 px crop for the Figure 1 case, staged "
                "once so the render does not depend on removable media"),
    }
    (out_dir / FIG1_ASSET_JSON).write_text(json.dumps(record, indent=2) + "\n")
    log.info("figure 1 assets staged in %s from %s", out_dir, member.name)
    return crop


def _radiograph(ax, arr: np.ndarray, title: str | None = None) -> None:
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation=IMAGE_INTERPOLATION)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=SMALL_FONT_PT, pad=2.0)


def _arrow(ax) -> None:
    ax.axis("off")
    ax.annotate("", xy=(0.92, 0.5), xytext=(0.08, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="0.25", linewidth=0.9,
                                mutation_scale=7))


def render_figure1(cfg: Config, out_dir: Path, split: str) -> Path:
    """The imaging and modelling workflow, drawn on a real radiograph. Double column."""
    log = logging.getLogger(MODULE)
    spec = _spec(cfg, assert_split(split), "figure1")
    assert_marker_audit_anchors(cfg)          # D1: the caption's residual-marker rates
    a = figure1_assets(cfg)
    band = a["band"]
    c0, c1 = a["half_cols"]
    r0, k0, side = a["box"]
    film, half, pre, final = a["film"], a["half"], a["pre"], a["final"]
    width_in = float(spec["width_in"])

    # Widths in proportion to each tile's aspect ratio, so the four tiles share one height.
    aspects = [im.shape[1] / im.shape[0] for im in (film, half, pre, final)]
    gap = 0.16 * max(aspects)
    ratios: list[float] = []
    for i, asp in enumerate(aspects):
        if i:
            ratios.append(gap)
        ratios.append(asp)
    with plt.rc_context(RC):
        # The four tiles share one height by construction (their widths are proportional to
        # their aspect ratios), so the panel A band is that height plus its titles and the
        # figure height follows from the arithmetic rather than from a guessed constant. A
        # taller figure would simply centre the tiles in empty axes.
        tile_h_in = width_in / sum(ratios)
        band_a = tile_h_in + 0.42
        band_b = 0.98
        fig = plt.figure(figsize=(width_in, band_a + band_b), layout="constrained")
        gs = fig.add_gridspec(2, len(ratios), width_ratios=ratios,
                              height_ratios=[band_a, band_b])
        axes = [fig.add_subplot(gs[0, i]) for i in range(0, len(ratios), 2)]
        for i in range(1, len(ratios), 2):
            _arrow(fig.add_subplot(gs[0, i]))

        _radiograph(axes[0], film, FIG1_PANEL_TITLES[0])
        mid = film.shape[1] / 2.0
        axes[0].axvline(mid, color="#D55E00", linestyle=(0, (2, 2)), linewidth=0.8)
        axes[0].add_patch(Rectangle((c0 - 0.5, -0.5), c1 - c0, film.shape[0], fill=False,
                                    edgecolor="#0072B2", linewidth=1.0))
        # Both labels sit over radiograph, which is bright in places, so each carries its own
        # dark plate; a label a reader cannot read is a label that is not there.
        plate = dict(facecolor="black", alpha=0.62, pad=1.4, edgecolor="none")
        axes[0].text(0.5 * (c0 + c1), film.shape[0] * 0.02, "contralateral", color="#56B4E9",
                     fontsize=SMALL_FONT_PT - 1.2, ha="center", va="top", bbox=plate)
        other = 0.5 * (c1 + film.shape[1]) if c0 == 0 else 0.5 * c0
        axes[0].text(other, film.shape[0] * 0.02, "index (discarded)", color="0.85",
                     fontsize=SMALL_FONT_PT - 1.2, ha="center", va="top", bbox=plate)

        _radiograph(axes[1], half, FIG1_PANEL_TITLES[1].format(
            half_inset_pct=f"{100.0 * float(cfg['preprocess']['half_inset_frac']):g}"))
        axes[1].add_patch(Rectangle((k0 - 0.5, r0 - 0.5), side, side, fill=False,
                                    edgecolor="#009E73", linewidth=1.0))

        _radiograph(axes[2], pre, FIG1_PANEL_TITLES[2].format(out_size=final.shape[0]))
        n = pre.shape[0]
        b = band * n / float(final.shape[0])
        axes[2].add_patch(Rectangle((b - 0.5, b - 0.5), n - 2 * b, n - 2 * b, fill=False,
                                    edgecolor="#CC79A7", linestyle=(0, (3, 2)), linewidth=0.9))

        _radiograph(axes[3], final, FIG1_PANEL_TITLES[3])

        ax_net = fig.add_subplot(gs[1, :])
        _draw_network(ax_net, cfg)
        for ax, letter in ((axes[0], "A"), (ax_net, "B")):
            ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
                    fontweight="bold", ha="right", va="bottom")
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("figure 1 written: %s (case %s, crop from the shipped shard image, "
             "reconstruction r = %.4f, %d saturated px surviving inside the retained "
             "region and disclosed in the caption)",
             out, a["key"], a["pearson"], a["residual_saturated_px"])
    return out


def _draw_network(ax, cfg: Config) -> None:
    """Panel B: the architecture, as boxes and arrows. No numbers that are not in config."""
    mi = cfg["model_image"]
    n_intervals = int(mi["survival_head"]["n_intervals"])
    stages = [
        ("Frontal\nLateral\nSunrise", "one crop\nper view"),
        (f"{FIG1_ENCODER}\nencoder", "shared across\nviews"),
        ("+ view\nembedding", "learned, one\nper view type"),
        ("Masked\nattention pool", "over the views\nthe patient has"),
        (f"Discrete-time head\n{n_intervals} interval hazards", f"{_horizon_adj(int(cfg['model_eval']['horizons_days'][-1]), float(cfg['timeline']['days_per_year']))}\nrisk"),
    ]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(stages)
    gap = 0.035
    w = (1.0 - (n - 1) * gap) / n
    for i, (head, sub) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(Rectangle((x, 0.34), w, 0.52, facecolor="#F2F2F2", edgecolor="0.35",
                               linewidth=0.7))
        ax.text(x + w / 2.0, 0.60, head, ha="center", va="center", fontsize=SMALL_FONT_PT)
        ax.text(x + w / 2.0, 0.26, sub, ha="center", va="top", fontsize=SMALL_FONT_PT - 1.2,
                color="0.35")
        if i:
            ax.annotate("", xy=(x - 0.004, 0.60), xytext=(x - gap + 0.004, 0.60),
                        arrowprops=dict(arrowstyle="-|>", color="0.25", linewidth=0.9,
                                        mutation_scale=7))


# =========================================================================== #
# FIGURE 2. REPRESENTATIVE IMAGE FINDINGS AT ONE KELLGREN-LAWRENCE GRADE        #
#                                                                              #
# src/interpretability.py already rendered 32 three-tile strips (radiograph |   #
# Grad-CAM | integrated gradients) at 1536 x 512 and wrote the case-level       #
# manifest beside them. This renderer SELECTS four of those strips - one per    #
# cell of the classification table, all at the same inferred KL grade - and     #
# lays them out as a 3 x 4 grid, so the tiles a reader sees are the same        #
# pixels the interpretability run produced and nothing is recoloured here.      #
# =========================================================================== #
def _interp_panel_rows(cfg: Config) -> pd.DataFrame:
    df = _interp_table(cfg, "interp_panel_manifest.csv")
    for col in ("arm", "cell", "empi_anon", "klg_contra", "risk_published", "event",
                "time_days", "file"):
        assert col in df.columns, f"interp_panel_manifest.csv is missing the {col!r} column"
    return df[df["arm"].astype(str) == INTERP_ARM].copy()


def figure2_cases(cfg: Config) -> list[dict]:
    """The four strips Figure 2 draws, checked against the manifest that produced them.

    Every number the caption states about these cases is asserted here: the cell, the
    grade, and the published risk to the precision the caption prints. A case that moved
    fails the render rather than appearing under a caption describing the old one.
    """
    man = _interp_panel_rows(cfg)
    by_id = {str(r["empi_anon"]): r for _, r in man.iterrows()}
    out: list[dict] = []
    for cell, pid, risk in FIND_CASES:
        assert pid in by_id, (
            f"{MODULE}: interp_panel_manifest.csv has no {INTERP_ARM} panel for patient "
            f"{pid}; Figure 2's cases come from that run and cannot be chosen here")
        row = by_id[pid]
        assert str(row["cell"]) == cell, (
            f"{MODULE}: patient {pid} is in cell {row['cell']!r} and Figure 2 draws it as "
            f"{cell!r}")
        assert float(row["klg_contra"]) == FIND_KLG, (
            f"{MODULE}: patient {pid} carries Kellgren-Lawrence {row['klg_contra']} and the "
            f"caption states {FIND_KLG}; every column must be at one grade")
        got = float(row["risk_published"])
        assert abs(got - risk) < FIND_RISK_TOL, (
            f"{MODULE}: patient {pid} has published risk {got:.6f} and the caption prints "
            f"{risk:.3f}")
        flagged = got >= FIND_THRESHOLD
        assert flagged == (cell in ("TP", "FP")), (
            f"{MODULE}: patient {pid} sits on the wrong side of the {FIND_THRESHOLD} "
            f"operating point for cell {cell}")
        strip = Path(str(row["file"]))
        if not strip.exists():
            strip = interpretability_dir(cfg) / INTERP_PANEL_DIRNAME / strip.name
        _require_file(strip, "~/.venvs/mrkr-torch/bin/python -m src.interpretability "
                             "--stage attribution", f"the attribution strip for patient {pid}")
        out.append({"cell": cell, "pid": pid, "risk": got, "event": int(row["event"]),
                    "time_days": float(row["time_days"]), "strip": strip})
    return out


FIND_MARKER_CELL = "FN"          # the column whose crop carries the disclosed character


def _saturated_inside_band(cfg: Config, strip: Path) -> int:
    """Saturated pixels inside the masked band of a strip's radiograph tile."""
    pp = cfg["preprocess"]
    tile = _strip_tiles(strip)[0].mean(axis=2)
    band = int(round(float(pp["mask_border_frac"]) * tile.shape[0]))
    return int((tile[band:-band, band:-band] >= int(pp["marker_sat_level"])).sum())


def assert_disclosed_marker(cfg: Config, cases: list[dict]) -> int:
    """D2. The burned-in character Figure 2's caption names, checked on the tile it draws.

    Three claims, all about pictures, all measured on the pictures rather than asserted:

    1. the false-negative column carries a saturated burned-in character inside the
       retained region, at the pipeline's own saturation level and above its own minimum
       blob size. If that case is ever reselected onto a clean crop this raises instead of
       leaving the caption describing a marker that is no longer there;
    2. it is the one of its grade's false-negative candidates carrying the LARGEST such
       blob, which is the caption's "it was not selected around" and is the opposite of
       cherry-picking. A quartet chosen to avoid markers would fail here;
    3. the grade holds the number of candidates the caption states.

    Returns the number of saturated pixels found, for the log.
    """
    case = next((c for c in cases if c["cell"] == FIND_MARKER_CELL), None)
    assert case is not None, (
        f"{MODULE}: the caption names the {FIND_MARKER_CELL} column as the one carrying a "
        f"residual marker and no such column is drawn")
    n_sat = _saturated_inside_band(cfg, case["strip"])
    assert n_sat >= int(cfg["preprocess"]["marker_min_px"]), (
        f"{MODULE}: the caption says the {FIND_MARKER_CELL} column carries a saturated "
        f"burned-in character inside the retained region and only {n_sat} pixel(s) there "
        f"reach the pipeline's own saturation level of "
        f"{int(cfg['preprocess']['marker_sat_level'])}")

    man = _interp_panel_rows(cfg)
    grade = man[man["klg_contra"].astype(float) == FIND_KLG]
    assert len(grade) == FIND_GRADE_CANDIDATES, (
        f"{MODULE}: the caption states {FIND_GRADE_CANDIDATES} candidate panels at "
        f"Kellgren-Lawrence {FIND_KLG:.0f} and the manifest holds {len(grade)}")
    fn = grade[grade["cell"].astype(str) == FIND_MARKER_CELL]
    assert len(fn) == FIND_FN_CANDIDATES, (
        f"{MODULE}: the caption states {FIND_FN_CANDIDATES} false-negative candidates at "
        f"that grade and the manifest holds {len(fn)}")
    blobs = {}
    for _, r in fn.iterrows():
        p = Path(str(r["file"]))
        if not p.exists():
            p = interpretability_dir(cfg) / INTERP_PANEL_DIRNAME / p.name
        blobs[str(r["empi_anon"])] = _saturated_inside_band(cfg, p) if p.exists() else -1
    worst = max(blobs, key=lambda k: blobs[k])
    assert worst == case["pid"], (
        f"{MODULE}: the caption says the false-negative drawn is the candidate carrying "
        f"the largest residual blob; patient {worst} carries {blobs[worst]} saturated "
        f"pixels and the figure draws {case['pid']} with {n_sat}")
    return n_sat


def assert_spread_is_the_widest_available(cfg: Config) -> float:
    """D4. "the WIDEST-SPREAD quartet available at this grade", checked exhaustively.

    The four cases were chosen out of the sampler's KL-2 panels one per cell, and they were
    chosen for the spread the caption prints. That makes 10.8-fold the maximum these
    candidates could have been made to show rather than a typical separation, and a caption
    that prints it has to say so. Every one-per-cell combination is enumerated here, so the
    claim is proved rather than remembered.
    """
    from itertools import product                    # noqa: PLC0415

    man = _interp_panel_rows(cfg)
    grade = man[man["klg_contra"].astype(float) == FIND_KLG]
    by_cell = {c: g["risk_published"].astype(float).tolist()
               for c, g in grade.groupby(grade["cell"].astype(str))}
    cells = [c for c, _, _ in FIND_CASES]
    assert set(cells) <= set(by_cell), (
        f"{MODULE}: the manifest has no candidate for cell(s) "
        f"{sorted(set(cells) - set(by_cell))} at Kellgren-Lawrence {FIND_KLG:.0f}")
    best = max(max(q) / min(q) for q in product(*(by_cell[c] for c in cells)))
    drawn = max(r for _, _, r in FIND_CASES) / min(r for _, _, r in FIND_CASES)
    assert abs(best - drawn) < 5e-3, (
        f"{MODULE}: the caption calls the drawn quartet the widest-spread one available at "
        f"this grade, at {drawn:.2f}-fold, and some other one-per-cell choice reaches "
        f"{best:.2f}-fold")
    return drawn


def _strip_tiles(path: Path, n: int = 3) -> list[np.ndarray]:
    """Split a horizontal N-tile attribution strip back into its square tiles."""
    from PIL import Image                          # noqa: PLC0415

    arr = np.asarray(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    assert w == n * h, f"{path.name} is {w}x{h}, expected {n} square tiles side by side"
    return [arr[:, i * h:(i + 1) * h] for i in range(n)]


def render_figure2(cfg: Config, out_dir: Path, split: str) -> Path:
    """Representative image findings at one KL grade, TP / FP / TN / FN. Double column."""
    log = logging.getLogger(MODULE)
    spec = _spec(cfg, assert_split(split), "figure2")
    assert_marker_audit_anchors(cfg)          # D2: the caption's marker rate and its bound
    cases = figure2_cases(cfg)
    n_sat = assert_disclosed_marker(cfg, cases)   # D2: the marker itself, on the tile drawn
    assert_spread_is_the_widest_available(cfg)    # D4: what "10.8-fold" is the maximum of
    width_in = float(spec["width_in"])
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(width_in, 0.86 * width_in), layout="constrained")
        gs = fig.add_gridspec(len(FIND_ROW_LABELS), len(cases))
        for col, case in enumerate(cases):
            tiles = _strip_tiles(case["strip"])
            for row, tile in enumerate(tiles):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(tile, interpolation=IMAGE_INTERPOLATION)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if row == 0:
                    outcome = ("arthroplasty at day "
                               f"{case['time_days']:.0f}" if case["event"] else
                               f"event free to day {case['time_days']:.0f}")
                    ax.set_title(f"{case['cell']}  {FIND_CELL_WORD[case['cell']]}\n"
                                 f"predicted risk {case['risk']:.3f}\n{outcome}",
                                 fontsize=SMALL_FONT_PT, pad=2.5)
                if col == 0:
                    ax.set_ylabel(FIND_ROW_LABELS[row], fontsize=SMALL_FONT_PT)
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("figure 2 written: %s (%d cases at KL %.1f, risks %s; %d saturated px survive "
             "inside the retained region of the %s column, disclosed in the caption)", out,
             len(cases), FIND_KLG, ", ".join(f"{c['risk']:.3f}" for c in cases), n_sat,
             FIND_MARKER_CELL)
    return out


# =========================================================================== #
# FIGURE 3. THE IMAGING-MODEL COMPARISON FOREST                                #
#                                                                              #
# One picture for the whole ladder: what each arm achieves, what the paired    #
# contrasts between arms are worth, what withholding a view from a frozen      #
# multi-view network costs, and what occluding a region or widening the masked #
# band costs. Three things this figure must not do, all of them documented in  #
# the `note` column of the tables it reads and all of them enforced here:      #
#                                                                              #
#  1. It must not present the border-band-only control as a leakage result.    #
#     The preprocessing pipeline already zeroes that band, so the control      #
#     input is uniform for 676 of 734 patients and its chance-level AUROC      #
#     follows by construction. It is drawn in a shaded PIPELINE CHECK block,   #
#     off scale, with its value written out.                                   #
#  2. It must not let the view-withholding rows read as model comparisons.     #
#     They are a different estimand on a different population (315 patients    #
#     with a frontal and at least one non-frontal film), and that denominator  #
#     is printed on the block heading and on every one of those rows.          #
#  3. It must not invite a compartment ranking. Every anatomic occlusion       #
#     interval crosses zero; the block carries that sentence in the caption.   #
# =========================================================================== #
FOREST_ARMS = ("m0", "m1", "m2_frontal", "m3_image", "m4_fusion")
# The ladder holds NINE scored arms and panel A draws FIVE of them, which is the reduction
# the editor asked for. The caption used to say "for each model arm", which is a claim about
# all nine and is false of the picture: the two discrete-time clinical controls, the
# frontal-only fusion arm and the ConvNeXt-Tiny robustness arm are in the metrics table and
# in Table 3 and are not drawn here. Both counts are named in the caption and both are
# asserted against the metrics table in ``forest_rows``, so a change to the ladder fails the
# render instead of printing a stale count.
FOREST_ARMS_TOTAL = 9
# Each row NAMES A TABLE ROW rather than carrying a value, so nothing in this figure is a
# transcribed number and a row that moves fails the render instead of printing something
# else. ``source`` picks the published contrast table or the post-hoc one, which are two
# different Benjamini-Hochberg universes and are deliberately not merged.
FOREST_CONTRAST_ROWS = (
    ("posthoc", dict(family="posthoc_single_view", model="m2_frontal", reference="m0"),
     "M2 frontal minus M0 clinical"),
    ("posthoc", dict(family="posthoc_single_view", model="m2_frontal", reference="m1"),
     "M2 frontal minus M1 clinical plus KLG"),
    ("published", dict(family="views", model="m3_image", reference="m2_frontal"),
     "M3 multi-view minus M2 frontal"),
    ("published", dict(family="modality", model="m4_fusion", reference="m3_image"),
     "M4 fusion minus M3 multi-view"),
)
FOREST_ABLATION_ROWS = (
    ("m3_image", "frontal_only", "M3, lateral and sunrise withheld"),
    ("m3_image", "drop_sunrise", "M3, sunrise withheld"),
    ("m4_fusion", "frontal_only", "M4, lateral and sunrise withheld"),
    ("m4_fusion", "drop_sunrise", "M4, sunrise withheld"),
)
FOREST_REGION_ROWS = (
    ("occlude_medial", "Medial compartment occluded"),
    ("occlude_lateral", "Lateral compartment occluded"),
    ("occlude_patellofemoral", "Patellofemoral region occluded"),
    ("occlude_joint", "Tibiofemoral joint occluded"),
    ("keep_joint_only", "Tibiofemoral joint kept, rest occluded"),
)
FOREST_MASKING_ROWS = (
    ("mask_residual_markers", "Residual marker-like blobs blanked"),
    ("border_62px", "Masked band widened to 62 px"),
    ("border_93px", "Masked band widened to 93 px"),
)
FOREST_CONTROL_ROWS = (
    ("occlude_border", "Already-blank band re-zeroed"),
    ("keep_border_only", "Band only, degenerate input"),
)
FOREST_XLIM = (-0.17, 0.27)
# Vertical inches per forest row. A forest is one row tall per row; the figure height is
# therefore arithmetic, not a fraction of the column width - scaling it by width_in makes a
# double-column forest four times taller than a single-column one for no reason.
FOREST_ROW_IN = 0.155
FOREST_GROUP_COLOUR = {
    "arms": "#000000", "contrasts": "#0072B2", "ablation": "#009E73",
    "region": "#D55E00", "masking": "#CC79A7", "control": "#666666",
}


def _one_row(df: pd.DataFrame, where: dict, what: str) -> pd.Series:
    """The single row of ``df`` matching ``where``. Anything but one row RAISES."""
    sel = np.ones(len(df), dtype=bool)
    for col, val in where.items():
        assert col in df.columns, f"{MODULE}: {what} selects on {col!r}, which is not a column"
        sel &= df[col].astype(str) == str(val)
    hit = df[sel]
    assert len(hit) == 1, (
        f"{MODULE}: {what} selects {len(hit)} rows on {where}, expected exactly 1; this "
        "figure names a table row and never a value")
    return hit.iloc[0]


def forest_rows(cfg: Config, split: str) -> tuple[list[dict], list[dict]]:
    """Every row Figure 3 draws, read from the table that owns each number.

    Returns ``(panel A rows, panel B rows)``. Panel A carries absolute AUROC, panel B
    paired differences; both carry the denominator each estimate rests on, because those
    denominators differ by design and a forest that hides them invites a reader to
    subtract two marginal levels.
    """
    assert_split(split)
    horizon = int(cfg["model_eval"]["horizons_days"][-1])
    metrics = _metrics_table(cfg, split)
    # The caption prints "five of the nine model arms". Pin both halves to the table that
    # owns them, so a ladder that gains or loses an arm fails here instead of shipping a
    # caption that overstates what panel A draws.
    assert len(metrics) == FOREST_ARMS_TOTAL, (
        f"{MODULE}: the {SPLIT_WORD[split]} metrics table holds {len(metrics)} arms and the "
        f"figure 3 caption states {FOREST_ARMS_TOTAL}")
    assert set(FOREST_ARMS) <= set(metrics.index.astype(str)), (
        f"{MODULE}: figure 3 draws an arm the metrics table does not hold")
    a_rows = [{
        "label": MODEL_DISPLAY[arm], "colour": "arms",
        "est": float(metrics.loc[arm, f"auc_{horizon}"]),
        "lo": float(metrics.loc[arm, f"auc_{horizon}_lo"]),
        "hi": float(metrics.loc[arm, f"auc_{horizon}_hi"]),
        "n": int(metrics.loc[arm, "n_patients"]),
    } for arm in FOREST_ARMS]

    published = _read_table(cfg, split_path(cfg, "comparisons_csv", split).name,
                            f"python -m src.eval_models --split {split}",
                            f"the {SPLIT_WORD[split]} contrast table")
    posthoc = _read_table(cfg, "v6_test_comparisons_posthoc.csv",
                          "python -m src.v6_analyses",
                          "the post-hoc single-view contrast table")
    ablation = _interp_table(cfg, "interp_view_ablation.csv")
    occl = _interp_table(cfg, "interp_occlusion.csv")

    b_rows: list[dict] = [{"group": "Model contrasts, whole split"}]
    for source, where, label in FOREST_CONTRAST_ROWS:
        df = posthoc if source == "posthoc" else published
        r = _one_row(df[df["metric"].astype(str) == "auc"], where, label)
        b_rows.append({"label": label, "colour": "contrasts", "est": float(r["difference"]),
                       "lo": float(r["ci_lo"]), "hi": float(r["ci_hi"]),
                       "n": int(r["n_paired"])})
    b_rows.append({"group": "Views withheld from a frozen multi-view model, "
                            f"n = {FOREST_ABLATION_N}"})
    for arm, cond, label in FOREST_ABLATION_ROWS:
        r = _one_row(ablation, {"arm": arm, "condition": cond}, label)
        assert int(r["n_patients_common"]) == FOREST_ABLATION_N, (
            f"{MODULE}: the view-ablation rows rest on {int(r['n_patients_common'])} paired "
            f"patients and the caption states {FOREST_ABLATION_N}")
        b_rows.append({"label": label, "colour": "ablation", "est": float(r["delta_auroc"]),
                       "lo": float(r["delta_auroc_lo"]), "hi": float(r["delta_auroc_hi"]),
                       "n": int(r["n_patients_common"])})
    for heading, rows, colour in (
            (f"Regions occluded, {MODEL_DISPLAY['m2_frontal']}, n = {FOREST_OCCLUSION_N}",
             FOREST_REGION_ROWS, "region"),
            ("Masked band and burned-in markers", FOREST_MASKING_ROWS, "masking"),
            ("Pipeline checks, NOT leakage tests", FOREST_CONTROL_ROWS, "control")):
        b_rows.append({"group": heading, "shade": colour == "control"})
        for cond, label in rows:
            r = _one_row(occl, {"arm": INTERP_ARM, "condition": cond}, label)
            assert int(r["n_patients"]) == FOREST_OCCLUSION_N, (
                f"{MODULE}: occlusion condition {cond!r} rests on {int(r['n_patients'])} "
                f"patients and the caption states {FOREST_OCCLUSION_N}")
            b_rows.append({"label": label, "colour": colour,
                           "est": float(r["delta_auroc"]), "lo": float(r["delta_auroc_lo"]),
                           "hi": float(r["delta_auroc_hi"]), "n": int(r["n_patients"]),
                           "shade": colour == "control",
                           "degenerate": int(r["n_max_tied_risk"]) > 1})

    # The four numbers the pipeline-check clause states, checked against the table that
    # carries them rather than against this module's memory of them.
    ident = _one_row(occl, {"arm": INTERP_ARM, "condition": "occlude_border"}, "border check")
    deg = _one_row(occl, {"arm": INTERP_ARM, "condition": "keep_border_only"}, "band only")
    assert int(ident["n_identical_to_baseline"]) == FOREST_BORDER_IDENTICAL, (
        f"{MODULE}: re-zeroing the band leaves {int(ident['n_identical_to_baseline'])} "
        f"patients bit-identical and the caption states {FOREST_BORDER_IDENTICAL}")
    assert int(deg["n_max_tied_risk"]) == FOREST_DEGENERATE_TIED, (
        f"{MODULE}: the degenerate control ties {int(deg['n_max_tied_risk'])} patients on one "
        f"risk and the caption states {FOREST_DEGENERATE_TIED}")
    assert int(deg["n_distinct_risk"]) == FOREST_DEGENERATE_DISTINCT, (
        f"{MODULE}: the degenerate control produces {int(deg['n_distinct_risk'])} distinct "
        f"risks and the caption states {FOREST_DEGENERATE_DISTINCT}")
    assert abs(float(deg["auroc_5y"]) - FOREST_DEGENERATE_AUROC) < 5e-4, (
        f"{MODULE}: the degenerate control scores {float(deg['auroc_5y']):.4f} and the "
        f"caption states {FOREST_DEGENERATE_AUROC}")
    _assert_occlusion_exceptions(occl)
    return a_rows, b_rows


# The whole anatomic family, not only the five rows the figure draws. The caption's claim
# is about intervals crossing zero, so checking it against the DRAWN subset would be
# checking it against the rows that were kept because they behave that way.
FOREST_ANATOMIC_CONDITIONS = (
    "occlude_medial", "occlude_lateral", "occlude_patellofemoral", "occlude_joint",
    "keep_joint_only", "occlude_joint_meanfill", "keep_joint_only_meanfill",
)


def _assert_occlusion_exceptions(occl: pd.DataFrame) -> None:
    """D3. Every anatomic condition the TABLE holds, sorted into drawn and named.

    Two things have to hold for the occlusion clause to be true:

    * every anatomic condition the figure draws has an interval that crosses zero, which
      is what "every anatomic occlusion interval drawn here crosses zero" asserts; and
    * exactly one anatomic condition anywhere in the table has an interval that excludes
      zero, it is the mean-filled joint-only one, and its numbers are the ones the caption
      prints. If a second exception ever appears, the caption's "the only anatomic
      condition in the table whose interval excludes zero" becomes false and this raises
      instead of printing it.
    """
    rows = occl[(occl["arm"].astype(str) == INTERP_ARM)
                & (occl["condition"].astype(str).isin(FOREST_ANATOMIC_CONDITIONS))]
    missing = set(FOREST_ANATOMIC_CONDITIONS) - set(rows["condition"].astype(str))
    assert not missing, (
        f"{MODULE}: the occlusion table is missing anatomic condition(s) {sorted(missing)}; "
        "the caption's claim about the anatomic family cannot be checked against it")
    drawn = {c for c, _ in FOREST_REGION_ROWS}
    excludes_zero = set()
    for _, r in rows.iterrows():
        cond = str(r["condition"])
        lo, hi = float(r["delta_auroc_lo"]), float(r["delta_auroc_hi"])
        crosses = lo <= 0.0 <= hi
        if not crosses:
            excludes_zero.add(cond)
        assert crosses or cond not in drawn, (
            f"{MODULE}: drawn anatomic condition {cond!r} has an interval ({lo:.6f}, "
            f"{hi:.6f}) that excludes zero, and the caption says every anatomic interval "
            "drawn here crosses zero")
    assert excludes_zero == {FOREST_MEANFILL_CONDITION}, (
        f"{MODULE}: the caption names exactly one anatomic condition whose interval "
        f"excludes zero, {FOREST_MEANFILL_CONDITION!r}; the table says "
        f"{sorted(excludes_zero)}")
    ex = _one_row(rows, {"condition": FOREST_MEANFILL_CONDITION}, "the named exception")
    for got, want, what in ((-float(ex["delta_auroc"]), FOREST_MEANFILL_DELTA, "estimate"),
                            (-float(ex["delta_auroc_hi"]), FOREST_MEANFILL_LO, "lower bound"),
                            (-float(ex["delta_auroc_lo"]), FOREST_MEANFILL_HI, "upper bound")):
        assert abs(got - want) < 5e-4, (
            f"{MODULE}: the named anatomic exception's {what} is {got:.6f} as a reduction "
            f"and the caption states {want}")


def _forest_panel(ax, rows: list[dict], *, xlim: tuple[float, float], reference: float,
                  xlabel: str) -> int:
    """Draw one forest panel top-down. Returns the number of rows drawn off scale.

    An estimate outside ``xlim`` is drawn as an arrow at the axis edge with its value
    still printed in the right-hand column, never clipped to the edge and never silently
    dropped: the one row this happens to is the degenerate control, and a reader has to be
    able to see both that it is off the scale and what it is.
    """
    y, off = 0.0, 0
    ticks: list[float] = []
    labels: list[str] = []
    ax.axvline(reference, color="0.45", linestyle=":", linewidth=0.8, zorder=1)
    for row in rows:
        if "group" in row:
            y -= 0.9
            ax.text(xlim[0], y, row["group"], fontsize=SMALL_FONT_PT, fontweight="bold",
                    ha="left", va="center", clip_on=False)
            y -= 1.0
            continue
        colour = FOREST_GROUP_COLOUR[row.get("colour", "arms")]
        if row.get("shade"):
            ax.axhspan(y - 0.5, y + 0.5, color="0.94", zorder=0)
        est, lo, hi = row["est"], row["lo"], row["hi"]
        if xlim[0] <= est <= xlim[1]:
            ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [y, y], color=colour,
                    linewidth=1.0, solid_capstyle="butt", zorder=3)
            ax.plot([est], [y], marker="s" if row.get("degenerate") else "o", color=colour,
                    markersize=3.6, markerfacecolor="white", markeredgewidth=0.9, zorder=4)
        else:
            off += 1
            ax.annotate("", xy=(xlim[0] + 0.003, y), xytext=(xlim[0] + 0.045, y),
                        arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=0.9,
                                        mutation_scale=6), zorder=4)
        ax.text(xlim[1] + 0.010, y, f"{est:+.3f} ({lo:+.3f} to {hi:+.3f})"
                if reference == 0.0 else f"{est:.3f} ({lo:.3f} to {hi:.3f})",
                fontsize=SMALL_FONT_PT, ha="left", va="center", clip_on=False)
        ticks.append(y)
        labels.append(f"{row['label']}  (n = {row['n']:,})")
        y -= 1.0
    ax.set_ylim(y + 0.5, 1.0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return off


def render_figure3(cfg: Config, out_dir: Path, split: str) -> Path:
    """The imaging-model comparison forest. Double column."""
    log = logging.getLogger(MODULE)
    spec = _spec(cfg, assert_split(split), "figure3")
    a_rows, b_rows = forest_rows(cfg, split)
    dpy = float(cfg["timeline"]["days_per_year"])
    long_adj = _horizon_adj(int(cfg["model_eval"]["horizons_days"][-1]), dpy)
    n_b = sum(1 for r in b_rows if "group" not in r)
    n_head = sum(1 for r in b_rows if "group" in r)
    width_in = float(spec["width_in"])
    xlim_a = (max(0.5, min(r["lo"] for r in a_rows) - 0.03),
              min(1.0, max(r["hi"] for r in a_rows) + 0.03))
    units = len(a_rows) + n_b + 2 * n_head + 9
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(width_in, FOREST_ROW_IN * units), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[len(a_rows) + 3, n_b + 2 * n_head + 4])
        ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        _forest_panel(ax_a, a_rows, xlim=xlim_a, reference=0.5,
                      xlabel=f"IPCW {long_adj} AUROC")
        off = _forest_panel(ax_b, b_rows, xlim=FOREST_XLIM, reference=0.0,
                            xlabel=f"Difference in IPCW {long_adj} AUROC")
        assert off == 1, (
            f"{MODULE}: figure 3 drew {off} rows off the difference scale, expected exactly "
            "one (the degenerate border-band-only control)")
        for ax, letter in ((ax_a, "A"), (ax_b, "B")):
            ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
                    fontweight="bold", ha="right", va="bottom")
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("figure 3 written: %s (%d arms in panel A, %d differences in %d blocks in "
             "panel B, %d drawn off scale)", out, len(a_rows), n_b, n_head, off)
    return out


# =========================================================================== #
# FIGURE 4. CALIBRATION AND RISK SEPARATION                                    #
#                                                                              #
# Panel A is the calibration panel the v5 figure 2 carried, with one change    #
# that is not cosmetic: the image arms are drawn AFTER the frozen              #
# recalibration. v5 drew them before it and said so in the caption, which was  #
# defensible but meant the picture and the calibration statistics in the       #
# metrics table described different predictions - the frozen M4 transform at   #
# 1825 days is cloglog(P) = 0.3469 + 1.6383 cloglog(p), which is a substantial #
# move rather than a cosmetic one. Both are now the recalibrated predictions.  #
#                                                                              #
# Panel B answers "does the image add anything to the grade?" in the form a    #
# clinician reads: observed five-year incidence by tertile of predicted risk,  #
# formed WITHIN each Kellgren-Lawrence stratum. The lowest stratum carries     #
# three events and is drawn with that count on its face, because three bars    #
# over three events without their denominator is a lie of omission.            #
# =========================================================================== #
CAL_STRATA = ("KL 0-1", "KL 2", "KL 3-4")


def calibration_risks(cfg: Config, split: str, log: logging.Logger) -> dict:
    """Per-arm RECALIBRATED risk at the long horizon, on the set every arm scores."""
    mc = _import_model_clinical()
    horizon = float(int(cfg["model_eval"]["horizons_days"][-1]))
    metrics = _metrics_table(cfg, split)
    fr = _split_frame(cfg, mc, split)
    T = fr["time_from_landmark"].to_numpy(dtype=float)
    E = fr["event_indicator"].to_numpy(dtype=int)
    expected_n = {a: int(metrics.loc[a, "n_patients"]) for a in FIG2_MODELS
                  if "n_patients" in metrics.columns}
    risks = _arm_risks(cfg, mc, fr, horizon, FIG2_MODELS, split, log, expected_n=expected_n)
    applied: dict[str, dict | None] = {}
    for arm in FIG2_MODELS:
        recal = frozen_recalibration(cfg, arm, horizon)
        applied[arm] = recal
        if recal is None:
            continue
        finite = np.isfinite(risks[arm])
        risks[arm][finite] = apply_frozen_recalibration(risks[arm][finite], recal)
    common = np.ones(len(fr), dtype=bool)
    for arm in FIG2_MODELS:
        common &= np.isfinite(risks[arm])
    anchors = SPLIT_ANCHORS[split]
    n_common, ev_common = int(common.sum()), int(E[common].sum())
    assert (n_common, ev_common) == (anchors["panel_b_n"], anchors["panel_b_events"]), (
        f"figure 4 panel A rests on {n_common} patients and {ev_common} events scored by all "
        f"of {list(FIG2_MODELS)}, but its caption states {anchors['panel_b_n']} and "
        f"{anchors['panel_b_events']}; the image and the caption must not disagree")
    return {"mc": mc, "risks": {a: risks[a][common] for a in FIG2_MODELS},
            "T": T[common], "E": E[common], "horizon": horizon,
            "n": n_common, "events": ev_common, "recalibrated": applied}


def klg_tertile_rows(cfg: Config) -> list[dict]:
    """Panel B's bars: observed incidence by predicted-risk tertile within KL stratum.

    NO SPLIT ARGUMENT, and that is a property of the table rather than an oversight: it
    holds the sealed split alone and says so nowhere in its own columns. Callers must not
    reach it on another split; :func:`calibration_decline_reason` is the guard.
    """
    df = _read_table(cfg, KLG_TERTILE_TABLE, "python -m src.v6_analyses",
                     "the within-stratum risk-tertile table")
    df = df[(df["arm"].astype(str) == CAL_ARM) & (df["scheme"].astype(str) == CAL_SCHEME)]
    out: list[dict] = []
    for stratum in CAL_STRATA:
        sub = df[df["stratum"].astype(str) == stratum].sort_values("tertile")
        assert len(sub) == RISK_TERTILES, (
            f"{MODULE}: stratum {stratum!r} holds {len(sub)} tertiles, expected "
            f"{RISK_TERTILES}")
        out.append({
            "stratum": stratum,
            "n": int(sub["n_stratum_patients"].iloc[0]),
            "events": int(sub["n_stratum_events"].iloc[0]),
            "cif": sub["km_cumulative_incidence"].to_numpy(dtype=float),
            "lo": sub["km_ci_lo"].to_numpy(dtype=float),
            "hi": sub["km_ci_hi"].to_numpy(dtype=float),
        })
    low = next(r for r in out if r["stratum"] == CAL_LOW_STRATUM)
    assert (low["n"], low["events"]) == (CAL_LOW_N, CAL_LOW_EVENTS), (
        f"{MODULE}: the {CAL_LOW_STRATUM} stratum holds {low['n']} patients and "
        f"{low['events']} events; the caption states {CAL_LOW_N} and {CAL_LOW_EVENTS}")
    return out


def calibration_decline_reason(split: str) -> str:
    """Why Figure 4 has no honest render on ``split``, or "" when it has one.

    D8. Panel B is observed incidence by predicted-risk tertile WITHIN Kellgren-Lawrence
    stratum, and the only artefact that carries it, ``v6_klg_risk_tertiles.csv``, is
    written by ``src/v6_analyses.py`` for the sealed split alone. It names no split of its
    own, so :func:`klg_tertile_rows` cannot tell one from another and would happily draw
    the sealed split's 707 patients and 98 events under a validation caption. That is the
    one thing the split-aware registry exists to prevent, so the figure declines instead.

    This is also what makes ``make_manuscript.FIGURE_DECLINE_RULES`` correct again. That
    mapping keyed the decline on ``decision_curve_decline_reason``, written when figure 4
    WAS the decision curve; at v6 the decision curve is Supplementary Figure S3 and figure
    4 is this one, so the document side was gating the calibration figure on a different
    figure's protagonist. Both sides now name the same condition.
    """
    return "" if assert_split(split) == SEALED_SPLIT else (
        f"panel B rests on {KLG_TERTILE_TABLE}, which is written for the "
        f"{SPLIT_WORD[SEALED_SPLIT]} split alone and carries no split of its own, so there "
        f"is no {SPLIT_WORD[split]}-split incidence by predicted-risk tertile within "
        f"Kellgren-Lawrence stratum to draw")


def render_figure4(cfg: Config, out_dir: Path, split: str) -> Path | None:
    """Calibration at the long horizon, and risk separation within KL stratum.

    Returns ``None``, having drawn nothing, on a split for which panel B has no table.
    See :func:`calibration_decline_reason`; ``_render_set`` records the key as declined.
    """
    log = logging.getLogger(MODULE)
    spec = _spec(cfg, assert_split(split), "figure4")
    why = calibration_decline_reason(split)
    if why:
        log.warning("figure 4 is NOT drawn for the %s split and no image is written: %s. "
                    "This is the split-aware registry working, not a render failure.",
                    SPLIT_WORD[split], why)
        return None
    cal = calibration_risks(cfg, split, log)
    strata = klg_tertile_rows(cfg)
    width_in = float(spec["width_in"])
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(width_in, 0.52 * width_in), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
        ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        _panel_b(ax_a, cal["mc"], cal["risks"], cal["T"], cal["E"], cal["horizon"])
        n_keyed = _model_legend(fig, [ax_a])
        assert n_keyed == len(FIG2_MODELS), (
            f"figure 4: the shared legend keys {n_keyed} arm(s), expected "
            f"{len(FIG2_MODELS)}; a drawn series would go unlabelled")

        w = 0.26
        for i, row in enumerate(strata):
            err = np.vstack([row["cif"] - row["lo"], row["hi"] - row["cif"]])
            for j in range(RISK_TERTILES):
                st = TERTILE_STYLE[j]
                x = i + (j - 1) * w
                ax_b.bar(x, row["cif"][j], width=w * 0.9, color=st["color"],
                         edgecolor="0.25", linewidth=0.5, zorder=2,
                         label=st["label"] if i == 0 else None)
                ax_b.errorbar(x, row["cif"][j], yerr=err[:, j:j + 1], color="0.15",
                              elinewidth=0.7, capsize=1.6, linestyle="none", zorder=3)
            ax_b.text(i, -0.03, f"{row['stratum']}\nn = {row['n']}\n{row['events']} events",
                      ha="center", va="top", fontsize=SMALL_FONT_PT,
                      transform=ax_b.get_xaxis_transform())
        ax_b.set_xlim(-0.55, len(strata) - 0.45)
        ax_b.set_xticks([])
        ax_b.set_ylim(0.0, 1.0)
        ax_b.set_ylabel(f"Observed {_horizon_adj(int(cal['horizon']), float(cfg['timeline']['days_per_year']))} "
                        "cumulative incidence")
        ax_b.legend(loc="upper left", handlelength=1.4, borderaxespad=0.2)
        for side in ("top", "right"):
            ax_b.spines[side].set_visible(False)
        for ax, letter in ((ax_a, "A"), (ax_b, "B")):
            ax.text(-0.22, 1.02, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
                    fontweight="bold", ha="left", va="bottom")
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    recal = [a for a, r in cal["recalibrated"].items() if r is not None]
    log.info("figure 4 written: %s (panel A on %d %s patients with %d events, frozen "
             "recalibration applied to %s; panel B over %d KL strata)", out, cal["n"],
             SPLIT_WORD[split], cal["events"], ", ".join(recal) or "no arm", len(strata))
    return out


# =========================================================================== #
# SUPPLEMENTARY FIGURES                                                       #
#                                                                             #
# S2, S3 and S4 are the v5 main figures, unchanged in substance: the same     #
# renderers and the same caption clauses, pointed at the supplementary        #
# registry. S1, S5 and S6 are new.                                            #
# =========================================================================== #
def render_supp_s1(cfg: Config, out_dir: Path, split: str) -> Path:
    """Learning curves, aligned on the RETAINED checkpoint. Double column.

    The whole point of this figure is the marker at epoch zero. ``val_overfit_gap`` in the
    convergence table is the validation loss at the LAST epoch minus its own minimum, and
    with patience 8 the last epoch is always exactly 8 past the checkpoint that was kept,
    so a curve drawn out to the last epoch with nothing marked reads as "the model we kept
    was diverging", which is the opposite of the truth.
    """
    log = logging.getLogger(MODULE)
    spec = _supp_spec(cfg, assert_split(split), "figureS1")
    curves = _read_table(cfg, "v6_learning_curves.csv", "python -m src.eval_models v6",
                         "the per-epoch learning curves")
    per_arm = _read_table(cfg, "v6_learning_curves_by_arm.csv", "python -m src.eval_models v6",
                          "the per-arm learning-curve summary")
    for col in ("arm", "seed", "epochs_from_retained", "train_nll", "val_nll",
                "is_retained_epoch"):
        assert col in curves.columns, f"v6_learning_curves.csv is missing {col!r}"
    arms = [str(a) for a in per_arm["arm"]]
    n_series = int(curves.groupby(["arm", "seed"]).ngroups)
    assert n_series == S1_SERIES, (
        f"{MODULE}: the learning curves hold {n_series} arm-seed series and the caption "
        f"states {S1_SERIES}")
    marked = int(curves["is_retained_epoch"].astype(bool).sum())
    assert marked == n_series, (
        f"{MODULE}: {marked} rows are marked as the retained epoch, expected one per series "
        f"({n_series}); an unmarked panel is the defect this figure exists to correct")

    ncols = 4
    nrows = int(np.ceil(len(arms) / ncols))
    width_in = float(spec["width_in"])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, 0.34 * width_in * nrows),
                                 layout="constrained", squeeze=False, sharex=True)
        flat = [ax for row in axes for ax in row]
        for ax, arm in zip(flat, arms):
            sub = curves[curves["arm"].astype(str) == arm]
            summary = per_arm[per_arm["arm"].astype(str) == arm].iloc[0]
            first = sub["seed"].min()
            for seed, g in sub.groupby("seed"):
                g = g.sort_values("epochs_from_retained")
                x = g["epochs_from_retained"].to_numpy(dtype=float)
                ax.plot(x, g["train_nll"], color="#0072B2", linewidth=0.7, alpha=0.8,
                        zorder=2, label="Training" if seed == first else None)
                ax.plot(x, g["val_nll"], color="#D55E00", linewidth=0.7, alpha=0.8,
                        linestyle="--", zorder=2,
                        label="Validation" if seed == first else None)
                keep = g[g["is_retained_epoch"].astype(bool)]
                ax.plot(keep["epochs_from_retained"], keep["val_nll"], marker="o",
                        markersize=3.2, color="#D55E00", linestyle="none", zorder=4,
                        label="Retained checkpoint" if seed == first else None)
            ax.axvline(0.0, color="0.6", linestyle=":", linewidth=0.7, zorder=1)
            ax.set_title(f"{arm}\ngap at checkpoint {float(summary['mean_gap_at_retained']):.3f}"
                         f"\nlast minus min {float(summary['mean_val_overfit_gap']):.3f}",
                         fontsize=SMALL_FONT_PT - 0.5, pad=2.0)
            ax.tick_params(labelsize=SMALL_FONT_PT - 1.0)
        for ax in flat[len(arms):]:
            ax.axis("off")
        # One shared axis label rather than one per column: the arms do not fill the grid, so
        # a per-axes label lands under a blank panel and overlaps its neighbours.
        fig.supxlabel("Epochs from the retained checkpoint", fontsize=BASE_FONT_PT)
        fig.supylabel("Negative log likelihood", fontsize=BASE_FONT_PT)
        handles, labels = flat[0].get_legend_handles_labels()
        # The key goes in the grid cell the arms do not fill, not under the figure: a
        # figure-level legend and the shared axis label both want the bottom strip and
        # collide there.
        spare = flat[len(arms):]
        if spare:
            spare[0].legend(handles, labels, loc="center", frameon=False,
                            handlelength=2.2, fontsize=SMALL_FONT_PT)
        else:                                        # pragma: no cover - 7 arms, 4 columns
            fig.legend(handles, labels, loc="outside lower center", ncols=3,
                       handlelength=2.2)
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("supplementary S1 written: %s (%d arms, %d series, retained epoch marked on "
             "every one)", out, len(arms), n_series)
    return out


# The families S5 draws, in reading order. Acquisition era is deliberately ABSENT: the
# de-identified study date carries a per-patient random shift whose cross-patient
# comparability protocol section 17 requires in writing and which has never been obtained
# (deviation D17), and era is almost perfectly confounded with follow-up length here, so an
# era row would be read as an acquisition-technology effect it cannot support. The
# unavailable strata (equipment, manufacturer, site) are answered in a table, not a plot.
# ``image_crop_method`` is deliberately absent: crop_confidence < 1.0 holds if and only if
# crop_method == intensity_profile on all 6,071 preprocessed images, so the two families are
# the same partition with the same masks and the same estimates, and drawing both would
# present one split as two findings. The caption says so.
# ``laterality_source`` is included even though neither of its levels yields an estimate:
# its coded arm holds 76 events and ZERO patients observed beyond the horizon, so it is the
# family that makes the difference between "suppressed" and "not estimable" visible. A
# figure that drew only the families with numbers in them would leave a reader believing the
# only reason a cell is blank is the event floor.
S5_FAMILIES = ("imaging_weight_bearing", "imaging_views", "laterality_source",
               "image_masking", "image_crop_confidence", "image_inverted",
               "image_half_selected")
# Scopes that are a real distinction. ``not applicable`` is the value the table writes for a
# family that has no per-crop scope at all, and printing it in a heading would invent one.
S5_SCOPE_NA = ("", "nan", "none", "None", "not applicable")


def render_supp_s5(cfg: Config, out_dir: Path, split: str) -> Path:
    """Imaging robustness strata, with the three cell states drawn as three states."""
    log = logging.getLogger(MODULE)
    spec = _supp_spec(cfg, assert_split(split), "figureS5")
    df = _read_table(cfg, "v6_robustness_strata.csv", "python -m src.eval_models v6",
                     "the imaging robustness table")
    df = df[(df["arm"].astype(str) == INTERP_ARM) & (df["metric"].astype(str) == "auc")]
    horizon = int(cfg["model_eval"]["horizons_days"][-1])
    rows: list[dict] = []
    for fam in S5_FAMILIES:
        sub = df[df["family"].astype(str) == fam]
        assert not sub.empty, (
            f"{MODULE}: v6_robustness_strata.csv carries no family {fam!r}; S5 names the "
            "families it draws and a missing one must fail rather than shrink the figure")
        for scope, g in sub.groupby("image_scope", sort=False):
            label = str(g["subgroup"].iloc[0])
            if str(scope) not in S5_SCOPE_NA:
                label = f"{label} ({str(scope).replace('_', ' ')})"
            rows.append({"group": label})
            for _, r in g.iterrows():
                reason = str(r["suppression_reason"] or "")
                state = ("estimate" if not bool(r["suppressed"])
                         else "floor" if reason.startswith("protocol section 21")
                         else "undefined")
                rows.append({"label": str(r["level"]), "state": state,
                             "n": int(r["n_patients"]), "events": int(r["n_events"]),
                             "controls": int(r["n_controls_beyond_horizon"]),
                             "est": float(r["estimate"]) if state == "estimate" else np.nan,
                             "lo": float(r["ci_lo"]) if state == "estimate" else np.nan,
                             "hi": float(r["ci_hi"]) if state == "estimate" else np.nan,
                             "wide": bool(r["wide_interval"])})
    marginal = float(_metrics_table(cfg, split).loc[INTERP_ARM, f"auc_{horizon}"])
    n_row = sum(1 for r in rows if "group" not in r)
    n_head = sum(1 for r in rows if "group" in r)
    n_est = sum(1 for r in rows if r.get("state") == "estimate")
    width_in = float(spec["width_in"])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(
            figsize=(width_in, FOREST_ROW_IN * (n_row + 2 * n_head + 5)),
            layout="constrained")
        ax.axvline(marginal, color="0.45", linestyle=":", linewidth=0.8, zorder=1)
        y = 0.0
        ticks, labels = [], []
        for row in rows:
            if "group" in row:
                y -= 0.9
                ax.text(0.50, y, row["group"], fontsize=SMALL_FONT_PT, fontweight="bold",
                        ha="left", va="center", clip_on=False)
                y -= 1.0
                continue
            if row["state"] == "estimate":
                ax.plot([row["lo"], row["hi"]], [y, y], color="#0072B2", linewidth=1.0,
                        solid_capstyle="butt", zorder=3)
                ax.plot([row["est"]], [y], marker="o", color="#0072B2", markersize=3.6,
                        markerfacecolor="white", markeredgewidth=0.9, zorder=4)
                txt = (f"{row['est']:.3f} ({row['lo']:.3f} to {row['hi']:.3f})"
                       + ("  wide" if row["wide"] else ""))
            elif row["state"] == "floor":
                ax.text(0.75, y, "suppressed, fewer than 50 events", fontsize=SMALL_FONT_PT,
                        color="0.35", ha="center", va="center", style="italic", zorder=3)
                txt = "-"
            else:
                ax.text(0.75, y,
                        f"not estimable, {row['controls']} controls beyond the horizon",
                        fontsize=SMALL_FONT_PT, color="#D55E00", ha="center", va="center",
                        style="italic", zorder=3)
                txt = "-"
            ax.text(1.005, y, txt, fontsize=SMALL_FONT_PT, ha="left", va="center",
                    clip_on=False)
            ticks.append(y)
            labels.append(f"{row['label']}  (n = {row['n']:,}, {row['events']} events)")
            y -= 1.0
        ax.set_ylim(y + 0.5, 1.0)
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.50, 1.0)
        ax.set_xlabel(
            f"IPCW {_horizon_adj(horizon, float(cfg['timeline']['days_per_year']))} AUROC "
            f"(dotted line: the whole split, {marginal:.3f})")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("supplementary S5 written: %s (%d levels in %d families, %d estimable)", out,
             n_row, n_head, n_est)
    return out


def render_supp_s6(cfg: Config, out_dir: Path, split: str) -> Path:
    """Per-view attention: within-patient pairs, the three-view case, and the marginals."""
    log = logging.getLogger(MODULE)
    spec = _supp_spec(cfg, assert_split(split), "figureS6")
    paired = _interp_table(cfg, "interp_attention_paired.csv")
    by_view = _interp_table(cfg, "interp_attention_by_view.csv")
    pair = paired[paired["comparison"].astype(str) == "pairwise_one_crop_each"]
    triple = paired[paired["comparison"].astype(str) == "all_three_views_present"]
    assert len(pair) == 2 and len(triple) == 2, (
        f"{MODULE}: interp_attention_paired.csv holds {len(pair)} pairwise and "
        f"{len(triple)} three-view rows, expected two of each, one per multi-view arm")
    assert int(pair["n_patients"].iloc[0]) == S6_PAIR_N, (
        f"{MODULE}: the within-patient pair rests on {int(pair['n_patients'].iloc[0])} "
        f"patients and the caption states {S6_PAIR_N}")
    assert int(triple["n_patients"].iloc[0]) == S6_TRIPLE_N, (
        f"{MODULE}: the three-view row rests on {int(triple['n_patients'].iloc[0])} patients "
        f"and the caption states {S6_TRIPLE_N}")
    views = ("frontal", "lateral", "sunrise")
    view_colour = {"frontal": "#0072B2", "lateral": "#D55E00", "sunrise": "#009E73"}
    arms = [str(a) for a in pair["arm"]]
    short = {a: MODEL_DISPLAY.get(a, a).split(" ")[0] for a in arms}
    width_in = float(spec["width_in"])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(width_in, 0.36 * width_in),
                                 layout="constrained")
        for ax, frame, cols, title in (
                (axes[0], pair, ("a", "b"),
                 f"One frontal and one lateral crop\n(n = {S6_PAIR_N} patients)"),
                (axes[1], triple, ("a", "b", "c"),
                 f"All three views present\n(n = {S6_TRIPLE_N} patients)")):
            w = 0.8 / len(cols)
            for i, arm in enumerate(arms):
                r = frame[frame["arm"].astype(str) == arm].iloc[0]
                for j, c in enumerate(cols):
                    view = str(r[f"view_{c}"])
                    ax.bar(i + (j - (len(cols) - 1) / 2.0) * w, float(r[f"weight_{c}_mean"]),
                           yerr=float(r[f"weight_{c}_sd"]), width=w * 0.9,
                           color=view_colour[view], edgecolor="0.25", linewidth=0.5,
                           error_kw=dict(elinewidth=0.7, capsize=1.6, ecolor="0.15"),
                           label=view.capitalize() if i == 0 else None, zorder=2)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([short[a] for a in arms])
            ax.set_ylim(0.0, 1.0)
            ax.set_title(title, fontsize=SMALL_FONT_PT, pad=2.0)
        marg = axes[2]
        w = 0.8 / len(views)
        for i, arm in enumerate(arms):
            sub = by_view[by_view["arm"].astype(str) == arm]
            for j, view in enumerate(views):
                r = sub[sub["view"].astype(str) == view]
                if r.empty:
                    continue
                marg.bar(i + (j - 1) * w, float(r["attention_share_mean"].iloc[0]),
                         width=w * 0.9, color=view_colour[view], edgecolor="0.25",
                         linewidth=0.5, zorder=2)
        marg.set_xticks(range(len(arms)))
        marg.set_xticklabels([short[a] for a in arms])
        marg.set_ylim(0.0, 1.0)
        marg.set_title("Marginal share over every patient\nwith that view",
                       fontsize=SMALL_FONT_PT, pad=2.0)
        axes[0].set_ylabel("Attention weight")
        for ax, letter in zip(axes, "ABC"):
            ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
                    fontweight="bold", ha="left", va="bottom")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        # Handles from BOTH bar panels, deduplicated by label: panel A draws two views and
        # panel B three, so a key taken from panel A alone omits the sunrise view it colours.
        seen: dict[str, object] = {}
        for ax in axes[:2]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, handle)
        assert len(seen) == len(views), (
            f"{MODULE}: supplementary S6 keys {len(seen)} view(s), expected {len(views)}")
        fig.legend(list(seen.values()), list(seen), loc="outside lower center", ncols=3,
                   handlelength=1.4)
        out = out_dir / spec["filename"]
        _save(fig, out, cfg, width_in)
    log.info("supplementary S6 written: %s (%d views keyed)", out, len(seen))
    return out


# --------------------------------------------------------------------------- #
# SAVE / ORCHESTRATION                                                          #
# --------------------------------------------------------------------------- #
def _save(fig, out: Path, cfg: Config, width_in: float) -> None:
    """Write the PNG at the configured dpi and the configured column width, then flatten it.

    ``bbox_inches="tight"`` trims to the drawn artists, so the saved width is NOT the
    requested ``figsize`` and a figure laid out at 6.7 in lands a few percent off the column
    it was meant to fill. The canvas is therefore rescaled until the trimmed width matches
    ``width_in`` to within a pixel. Font sizes are fixed in points and do not scale, so this
    converges in two or three passes rather than one; it is a fixed-point iteration with no
    random state, so the result is reproducible.

    The canvas dpi is raised to the SAVE dpi before any of that is measured. Text metrics
    are resolved at the renderer's dpi, so measuring a 300 dpi figure on a 100 dpi canvas
    hands the loop a width that is a few pixels off the one ``savefig`` will produce - the
    iteration then converges neatly onto the wrong number. The error grows with how much
    text sits OUTSIDE the axes, which is why figure 2's shared legend is what surfaced it.

    matplotlib emits RGBA. Several submission portals reject an alpha channel outright and
    these figures are opaque white anyway, so compositing onto white loses nothing and
    removes a routine cause of upload failure. The dpi is re-stamped because the re-save
    would otherwise drop the pHYs chunk.
    """
    from PIL import Image                          # noqa: PLC0415 - only needed at save time

    dpi = int(cfg["manuscript"]["figure_dpi"])
    fig.set_dpi(dpi)
    target_px = int(round(float(width_in) * dpi))
    for _ in range(WIDTH_LOCK_PASSES):
        fig.canvas.draw()
        got_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD_INCHES
        if abs(got_in * dpi - target_px) <= 1.0:
            break
        w, h = fig.get_size_inches()
        scale = target_px / (got_in * dpi)
        fig.set_size_inches(w * scale, h * scale)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=PAD_INCHES, format="png",
                metadata=PNG_METADATA)
    plt.close(fig)
    with Image.open(out) as im:
        flat = im
        if im.mode != "RGB":
            flat = Image.new("RGB", im.size, "white")
            flat.paste(im, mask=im.getchannel("A") if "A" in im.getbands() else None)
        flat.save(out, format="PNG", dpi=(dpi, dpi), optimize=True)
        size = flat.size
    assert abs(size[0] - target_px) <= WIDTH_LOCK_TOL_PX, (
        f"{out.name} is {size[0]} px wide, expected {target_px} px "
        f"({width_in} in at {dpi} dpi); the tight bounding box did not converge")


# =========================================================================== #
# THE REGISTRIES. One ordered tuple each: number, file, width, prose and       #
# renderer together, so a figure cannot be drawn without a caption or          #
# captioned without being drawn.                                              #
#                                                                             #
# THE MAIN REGISTRY IS THE v6 IMAGING SET. Journal of Imaging rejected v5 on   #
# scope - none of its four figures carried a radiograph - and the four         #
# entries below are the replacements the editor asked for, in his order:       #
# the workflow over a real film, representative image findings, the model      #
# comparison, and the clinical performance figure. The three v5 figures that   #
# survive in substance moved to the SUPPLEMENTARY registry with their          #
# renderers and their caption clauses unchanged.                              #
#                                                                             #
# D9, CLOSED 2026-08-11. Figure 2's file was still called                      #
# ``figure2_discrimination_calibration.png`` - the v5 name, for a figure of     #
# entirely different content - because                                         #
# tests/test_make_manuscript.py::test_verify_fires_when_an_image_does_not_      #
# hash_to_what_was_recorded typed that literal into an assertion and the task   #
# that renamed the figure could not edit that file. It is now                   #
# ``figure2_representative_findings.png``, and the test reads the name off this #
# registry instead of typing it, so the two can never disagree again. Every     #
# filename in this registry now matches its number and its content. The         #
# journal-facing bundle from :func:`write_submission_bundle` is unchanged in    #
# purpose: it renames each file to Figure1.png .. Supplementary-Figure-S6.png   #
# for upload and records the repository name and sha256 of each beside it.      #
# =========================================================================== #
FIGURE_DEFS: tuple[FigureDef, ...] = (
    FigureDef(key="figure1", number=1, filename="figure1_imaging_workflow.png",
              width_key="double_column_in", title=_wf_title, caption=_wf_caption,
              renderer=render_figure1),
    FigureDef(key="figure2", number=2, filename="figure2_representative_findings.png",
              width_key="double_column_in", title=_find_title, caption=_find_caption,
              renderer=render_figure2),
    FigureDef(key="figure3", number=3, filename="figure3_model_comparison_forest.png",
              width_key="double_column_in", title=_forest_title, caption=_forest_caption,
              renderer=render_figure3),
    FigureDef(key="figure4", number=4, filename="figure4_calibration_risk_separation.png",
              width_key="double_column_in", title=_cal_title, caption=_cal_caption,
              renderer=render_figure4),
)

# The supplementary set. S2, S3 and S4 are the v5 main figures 1, 4 and 3, rendered by the
# same functions with the same captions; S1, S5 and S6 are new. They are a SEPARATE
# registry rather than four more entries above because ``src/make_manuscript.py`` embeds
# and numbers everything :func:`figures` returns, and a supplementary figure in the main
# registry would be numbered Figure 5 in the body of the paper.
SUPPLEMENT_DEFS: tuple[FigureDef, ...] = (
    FigureDef(key="figureS1", number=1, filename="figureS1_learning_curves.png",
              width_key="double_column_in", title=_s1_title, caption=_s1_caption,
              renderer=render_supp_s1),
    FigureDef(key="figureS2", number=2, filename="figureS2_cohort_flow.png",
              width_key="double_column_in", title=_fig1_title, caption=_fig1_caption,
              renderer=render_cohort_flow),
    FigureDef(key="figureS3", number=3, filename="figureS3_decision_curve.png",
              width_key="single_column_in", title=_fig4_title, caption=_fig4_caption,
              renderer=render_decision_curve),
    FigureDef(key="figureS4", number=4, filename="figureS4_cumulative_incidence.png",
              width_key="single_column_in", title=_fig3_title, caption=_fig3_caption,
              renderer=render_cumulative_incidence),
    FigureDef(key="figureS5", number=5, filename="figureS5_imaging_robustness.png",
              width_key="double_column_in", title=_s5_title, caption=_s5_caption,
              renderer=render_supp_s5),
    FigureDef(key="figureS6", number=6, filename="figureS6_view_attention.png",
              width_key="double_column_in", title=_s6_title, caption=_s6_caption,
              renderer=render_supp_s6),
)


def assert_registry(defs: tuple[FigureDef, ...], label: str = "FIGURE_DEFS"
                    ) -> tuple[FigureDef, ...]:
    """The registry invariants, checked AT IMPORT TIME on both registries.

    A gap or a repeat in the numbering would put "Figure 4" in a document with three
    figures; two definitions sharing a filename would have one render silently overwrite the
    other; a definition with no renderer would be captioned and cited but never drawn. All
    of these are cheap to catch here and expensive to find in a submitted PDF.
    """
    numbers = [d.number for d in defs]
    assert numbers == list(range(1, len(defs) + 1)), (
        f"{MODULE}: {label} must be ordered and numbered 1..{len(defs)}, got {numbers}")
    assert len({d.key for d in defs}) == len(defs), \
        f"{MODULE}: duplicate figure key in {label}: {[d.key for d in defs]}"
    assert len({d.filename for d in defs}) == len(defs), \
        f"{MODULE}: duplicate figure filename in {label}: {[d.filename for d in defs]}"
    assert all(callable(d.renderer) for d in defs), (
        f"{MODULE}: every {label} entry needs a renderer; a figure that cannot be drawn "
        "must not be captioned")
    assert all(d.width_key in ("single_column_in", "double_column_in") for d in defs), \
        f"{MODULE}: {label} width_key must name a manuscript column-width setting"
    return defs


assert_registry(FIGURE_DEFS)
assert_registry(SUPPLEMENT_DEFS, "SUPPLEMENT_DEFS")
# The two sets share a render directory tree and a manifest schema, so a filename in both
# would have one render overwrite the other's provenance entry.
assert not ({d.filename for d in FIGURE_DEFS} & {d.filename for d in SUPPLEMENT_DEFS}), (
    f"{MODULE}: a filename appears in both FIGURE_DEFS and SUPPLEMENT_DEFS")
assert not ({d.key for d in FIGURE_DEFS} & {d.key for d in SUPPLEMENT_DEFS}), (
    f"{MODULE}: a figure key appears in both FIGURE_DEFS and SUPPLEMENT_DEFS")

FIGURE_KEYS: tuple[str, ...] = tuple(d.key for d in FIGURE_DEFS)
RENDERERS: dict[str, Callable[[Config, Path, str], Path | None]] = {d.key: d.renderer
                                                                    for d in FIGURE_DEFS}
SUPPLEMENT_KEYS: tuple[str, ...] = tuple(d.key for d in SUPPLEMENT_DEFS)
SUPPLEMENT_RENDERERS: dict[str, Callable[[Config, Path, str], Path | None]] = {
    d.key: d.renderer for d in SUPPLEMENT_DEFS}
# Where the supplementary images go: a subdirectory of manuscript.figures_dir, so the two
# sets travel together and each carries its own provenance manifest without one overwriting
# the other's.
SUPPLEMENT_DIRNAME = "supplementary"
SUPPLEMENT_MANIFEST_SCHEMA = "mrkr-supplementary-figures-manifest"


def supplement_dir(cfg: Config, out_dir: Path | str | None = None) -> Path:
    """Where a render into ``out_dir`` (default: the configured one) puts S1-S6."""
    base = Path(out_dir) if out_dir is not None else cfg.path(cfg["manuscript"]["figures_dir"])
    return base / SUPPLEMENT_DIRNAME


def supplement_figures(cfg: Config, split: str) -> dict[str, dict]:
    """The supplementary registry for one config and one split, in figure-number order.

    Deliberately a SECOND function rather than a flag on :func:`figures`:
    ``src.make_manuscript`` calls that one and numbers everything it returns, so the two
    sets must not be reachable through one call.
    """
    ctx = caption_context(cfg, split)
    return {
        d.key: {
            "number": d.number,
            "filename": d.filename,
            "width_in": float(cfg["manuscript"][d.width_key]),
            "title": d.title(ctx),
            "caption": d.caption(ctx),
        }
        for d in SUPPLEMENT_DEFS
    }


def _supp_spec(cfg: Config, split: str, key: str) -> dict:
    return supplement_figures(cfg, split)[key]


def _default_figures() -> dict[str, dict]:
    """The VALIDATION registry built from the default config, for importers holding no Config.

    :data:`FIGURES` is the legacy module-level registry. It is pinned to the validation split
    on purpose: it carries no split argument, so it must not silently return sealed-split
    prose. Anything that knows which split it is reporting should call ``figures(cfg, split)``
    instead. A missing or unreadable config is reported rather than swallowed, so the failure
    names itself instead of surfacing later as a wrong column width.
    """
    try:
        return figures(load_config(), VAL_SPLIT)
    except (OSError, KeyError) as exc:            # noqa: BLE001 - re-raised with a name
        raise RuntimeError(
            f"{MODULE}: could not build the figure registry from config/feasibility.yaml "
            f"({exc!r}); manuscript.figures_dir / figure_dpi / single_column_in / "
            "double_column_in and model_eval.horizons_days must all be present") from exc


FIGURES: dict[str, dict] = _default_figures()


# --------------------------------------------------------------------------- #
# THE PROVENANCE MANIFEST                                                       #
#                                                                              #
# A PNG on disk carries no record of the split it was drawn for. Figure 1 in    #
# particular is split-aware PROSE - on ``--split val`` it draws "Development    #
# cohort, after the locked 20% test split was set aside unread" beside an       #
# exclusion box reading "Sealed test split, never read" - and a validation      #
# render of it sitting in manuscript.figures_dir is indistinguishable, to       #
# anything downstream, from the sealed-split render of the same filename.       #
# ``src/make_manuscript.py`` does not draw figures; it embeds whatever bytes    #
# are already there. So a val-era PNG left behind by an earlier render is       #
# embedded, unremarked, in a document whose Results report the sealed split,    #
# and every self-check in that module passes. That is not hypothetical: it is   #
# what shipped in the first sealed-split build of v2.                           #
#                                                                              #
# The renderer therefore writes down what it drew, beside what it drew, and the #
# consumer checks the bytes it is about to embed against that record.           #
# --------------------------------------------------------------------------- #
FIGURES_MANIFEST = "figures_manifest.json"
MANIFEST_SCHEMA = "mrkr-manuscript-figures-manifest"
MANIFEST_SCHEMA_VERSION = 1

# Why a figure is in the manifest's ``declined`` list rather than its ``rendered`` one.
# The renderer's own account of the verdict is in the run log; what the manifest has to
# carry is that the ABSENCE was a decision, so a consumer never has to read an empty slot
# as either "declined" or "someone deleted it" and guess which.
MANIFEST_DECLINED_REASON = (
    "the renderer declined to draw this figure on this split and wrote no image; the "
    "verdict it declined on is in the run log")
MANIFEST_NOT_ATTEMPTED_REASON = (
    "this render was restricted with --only, so the figure was never offered to its "
    "renderer; the manifest is marked incomplete and describes nothing about it")


def sha256_file(path: Path | str) -> str:
    """The sha256 of the bytes on disk. Streamed, so a 600 dpi PNG is not held twice."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(out_dir: Path | str) -> Path:
    """Where the manifest for a render into ``out_dir`` lives: beside the PNGs."""
    return Path(out_dir) / FIGURES_MANIFEST


def build_manifest(cfg: Config, split: str, out_dir: Path, written: dict[str, Path],
                   declined: list[str], attempted: list[str],
                   contract: str | None,
                   defs: tuple[FigureDef, ...] = FIGURE_DEFS,
                   schema: str = MANIFEST_SCHEMA) -> dict:
    """The manifest payload for one render. Deterministic: no timestamp, no path.

    Two identical renders must produce byte-identical manifests, so nothing here may vary
    between them. That rules out a render time, and it rules out ``out_dir`` itself: a
    scratch render and a real render draw the same figures and must be comparable by
    hash. Only FILENAMES are recorded, resolved by the consumer against its own figures
    directory, which is also why the manifest can be moved beside the images it describes.

    ``attempted`` is the key list this run offered to the renderers. A run restricted with
    ``--only`` attempts a subset, and the manifest is marked ``complete: false`` and lists
    the rest under ``not_attempted``; a consumer that needs the whole set must reject it
    rather than read the missing entries as declines. That distinction is the whole point
    of the file: DECLINED and NOT ATTEMPTED and DELETED are three different states and an
    absence cannot tell them apart.

    Nothing patient-level is admissible here (protocol section 28). The payload is a split
    name, a contract hash, filenames, sizes and digests.
    """
    by_key = {d.key: d for d in defs}
    order = [d.key for d in defs]

    def _entry(key: str) -> dict:
        d = by_key[key]
        return {"key": key, "number": d.number, "filename": d.filename}

    rendered: list[dict] = []
    for key in order:
        path = written.get(key)
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{MODULE}: the renderer for {key} returned {path}, which does not exist, "
                f"so there are no bytes to record provenance for. A renderer that returns "
                f"a path has written it; returning None is how a renderer declines.")
        if path.parent.resolve() != Path(out_dir).resolve():
            raise ValueError(
                f"{MODULE}: the renderer for {key} wrote {path}, which is not in the render "
                f"directory {out_dir}. The manifest records filenames only and is read "
                f"beside the images it describes, so a figure written elsewhere could not "
                f"be found again from it.")
        rendered.append({**_entry(key), "filename": path.name,
                         "size_bytes": path.stat().st_size,
                         "sha256": sha256_file(path)})

    return {
        "schema": schema,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "written_by": MODULE,
        "split": split,
        "complete": sorted(attempted) == sorted(order),
        "attempted": [k for k in order if k in set(attempted)],
        "figure_dpi": int(cfg["manuscript"]["figure_dpi"]),
        # The contract the sealed read was performed under, as
        # :func:`src.eval_models.assert_sealed_read_is_recorded` verified it at the top of
        # this render. ``None`` on a development split, where there is no sealed read to be
        # on the record and therefore no contract to pin.
        "training_contract_hash": contract,
        "rendered": rendered,
        "declined": [{**_entry(k), "reason": MANIFEST_DECLINED_REASON}
                     for k in order if k in set(declined)],
        "not_attempted": [{**_entry(k), "reason": MANIFEST_NOT_ATTEMPTED_REASON}
                          for k in order if k not in set(attempted)],
    }


def write_manifest(cfg: Config, split: str, out_dir: Path, written: dict[str, Path],
                   declined: list[str], attempted: list[str],
                   contract: str | None,
                   defs: tuple[FigureDef, ...] = FIGURE_DEFS,
                   schema: str = MANIFEST_SCHEMA) -> Path:
    """Write :func:`build_manifest` beside the images, and return the path written."""
    payload = build_manifest(cfg, split, out_dir, written, declined, attempted, contract,
                             defs, schema)
    path = manifest_path(out_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def _render_set(cfg: Config, split: str, out: Path, keys: list[str],
                renderers: dict[str, Callable[[Config, Path, str], Path | None]],
                specs: dict[str, dict], defs: tuple[FigureDef, ...], schema: str,
                contract: str | None, word: str, log: logging.Logger) -> dict[str, Path]:
    """Draw ``keys`` into ``out`` and write the provenance manifest beside them.

    Shared by :func:`render_all` and :func:`render_supplement`, because the two sets must
    not be able to differ in how they record what they drew: the second copy of a
    provenance rule is the copy that goes stale.
    """
    out.mkdir(parents=True, exist_ok=True)
    # THE OLD MANIFEST DIES BEFORE THE FIRST PIXEL MOVES. A render that begins has already
    # invalidated it: from here on the directory is a mixture of what this run wrote and
    # what the last one left, and the previous record describes neither. Deleting it first
    # means a render that raises half way through leaves no manifest at all, which the
    # consumer reports as missing, rather than a stale one whose remaining entries still
    # happen to match. Fail closed, and never on the strength of an entry nobody rewrote.
    stale = manifest_path(out)
    if stale.exists():
        stale.unlink()
        log.info("removed the previous %s; it describes a render this one supersedes",
                 stale.name)
    log.info("rendering %s %s for the %s split into %s at %d dpi", word, ", ".join(keys),
             SPLIT_WORD[split], out, int(cfg["manuscript"]["figure_dpi"]))
    # A LOOP, not a dict comprehension, because a renderer may legitimately decline. Today
    # only the decision curve does, on a split whose protagonist the convergence gate
    # suppressed; it logs its own reason and returns None. The key is dropped from the
    # manifest rather than mapped to a path nobody wrote, and the remaining figures still
    # render: one figure with nothing honest to show must not cost the others. Any other
    # failure still raises, so this is not a swallow-everything guard.
    written: dict[str, Path] = {}
    declined: list[str] = []
    for k in keys:
        drawn = renderers[k](cfg, out, split)
        if drawn is None:
            declined.append(k)
            log.warning("%s was not drawn for the %s split and is absent from this render; "
                        "the renderer logged its reason above", k, SPLIT_WORD[split])
            continue
        written[k] = drawn
    manifest = write_manifest(cfg, split, out, written, declined, keys, contract, defs,
                              schema)
    L = [f"  {word.capitalize()} {specs[k]['number']}: {p.name} "
         f"({specs[k]['width_in']:.2f} in wide)" for k, p in written.items()]
    log.info("wrote %d %s figure(s):\n%s", len(written), word, "\n".join(L))
    log.info("provenance: %s records %d rendered, %d declined and %d not attempted for the "
             "%s split; it is %s", manifest.name, len(written), len(declined),
             len(defs) - len(keys), SPLIT_WORD[split],
             "COMPLETE" if len(keys) == len(defs) else
             "PARTIAL and a manuscript build will refuse it")
    return written


def _render_preamble(cfg: Config | None, split: str | None
                     ) -> tuple[Config, str, str | None, logging.Logger]:
    cfg = cfg if cfg is not None else load_config()
    split = assert_split(split if split is not None else default_split(cfg))
    log = setup_logging(cfg.path(cfg["paths"]["run_log"]))
    contract: str | None = None
    if split == SEALED_SPLIT:
        contract = assert_sealed_read_is_recorded(cfg)
        log.info("sealed read is on the record; training contract %s still matches "
                 "train_arms.json", contract)
    return cfg, split, contract, log


def render_all(cfg: Config | None = None, *, out_dir: Path | str | None = None,
               only: str | None = None, split: str | None = None) -> dict[str, Path]:
    """Render every MAIN manuscript figure for ``split`` and return ``{"figureN": path}``.

    ``only`` restricts the run to one key of :data:`FIGURE_DEFS`. ``split`` defaults to
    ``manuscript.report_split``. On the sealed split the render is refused unless the single
    permitted read is on the record AND the training contract it was performed against still
    matches ``train_arms.json``.

    A key is ABSENT from the returned mapping when its renderer declined to draw on this
    split; the reason is in the log. Callers that embed the figures must treat a missing key
    as a missing figure rather than assume all four.

    A PROVENANCE MANIFEST, :data:`FIGURES_MANIFEST`, is written into the same directory the
    images went into, recording the split, the sealed read's training contract, and the
    sha256 of every image actually written. ``src/make_manuscript.py`` embeds bytes it did
    not draw and cannot otherwise tell one split's figure from another's; see the section
    comment above :func:`build_manifest`. Because it is written into the RESOLVED output
    directory, a scratch ``--out-dir`` render describes its own PNGs and never touches the
    manifest that belongs to the real ones.
    """
    cfg, split, contract, log = _render_preamble(cfg, split)
    out = Path(out_dir) if out_dir is not None else cfg.path(cfg["manuscript"]["figures_dir"])
    keys = list(FIGURE_KEYS) if only in (None, "all") else [only]
    for k in keys:
        assert k in RENDERERS, f"unknown figure key {k!r}; expected one of {list(FIGURE_KEYS)}"
    return _render_set(cfg, split, out, keys, RENDERERS, figures(cfg, split), FIGURE_DEFS,
                       MANIFEST_SCHEMA, contract, "figure", log)


def render_supplement(cfg: Config | None = None, *, out_dir: Path | str | None = None,
                      only: str | None = None, split: str | None = None) -> dict[str, Path]:
    """Render the supplementary figures S1-S6 into ``manuscript.figures_dir/supplementary``.

    A separate entry point with a separate manifest, for the same reason
    :func:`supplement_figures` is a separate function: ``src/make_manuscript.py`` embeds and
    numbers everything the MAIN registry returns, and a supplementary figure that could
    reach it would be numbered Figure 5 in the body of the paper.
    """
    cfg, split, contract, log = _render_preamble(cfg, split)
    out = supplement_dir(cfg, out_dir)
    keys = list(SUPPLEMENT_KEYS) if only in (None, "all") else [only]
    for k in keys:
        assert k in SUPPLEMENT_RENDERERS, \
            f"unknown supplementary key {k!r}; expected one of {list(SUPPLEMENT_KEYS)}"
    return _render_set(cfg, split, out, keys, SUPPLEMENT_RENDERERS,
                       supplement_figures(cfg, split), SUPPLEMENT_DEFS,
                       SUPPLEMENT_MANIFEST_SCHEMA, contract, "supplementary figure", log)


# --------------------------------------------------------------------------- #
# THE JOURNAL-FACING BUNDLE                                                     #
#                                                                              #
# The repository filenames are what the provenance manifest and the document    #
# generator resolve, and one of them is a legacy name kept alive by a test this #
# task could not edit (see the comment above FIGURE_DEFS). A submission upload  #
# should not carry that name, so the bundle below is a COPY, named by figure    #
# number, with a README recording which repository file each copy came from and #
# its sha256. It is a copy and not a rename: renaming would break the manifest  #
# the manuscript generator verifies against.                                    #
# --------------------------------------------------------------------------- #
SUBMISSION_DIRNAME = "v6-submit-figures"
SUBMISSION_README = "README.md"


def write_submission_bundle(cfg: Config, split: str,
                            out_dir: Path | str | None = None) -> Path:
    """Copy the rendered figures into a journal-facing bundle named by figure number."""
    log = logging.getLogger(MODULE)
    src_main = Path(out_dir) if out_dir is not None else \
        cfg.path(cfg["manuscript"]["figures_dir"])
    src_supp = supplement_dir(cfg, out_dir)
    dest = cfg.path(cfg["paths"]["figures_dir"]) / SUBMISSION_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*"):
        old.unlink()
    lines = ["# v6 submission figure bundle", "",
             "Copies of the rendered figures, named by figure number. The authoritative",
             "images are the ones in `outputs/figures/manuscript/`, which the provenance",
             "manifest and `src/make_manuscript.py` resolve; these copies exist so an",
             "upload carries a filename that matches its number and its content.", "",
             "| Bundle file | Repository file | sha256 |", "|---|---|---|"]
    n = 0
    for word, base, specs in (("Figure", src_main, figures(cfg, split)),
                              ("Supplementary Figure S", src_supp,
                               supplement_figures(cfg, split))):
        for key, spec in specs.items():
            src = base / str(spec["filename"])
            if not src.exists():
                log.warning("submission bundle: %s was not rendered and is not copied",
                            spec["filename"])
                continue
            tag = f"{word}{spec['number']}".replace(" ", "-")
            target = dest / f"{tag}.png"
            target.write_bytes(src.read_bytes())
            try:
                shown = src.relative_to(cfg.path("."))
            except ValueError:            # a scratch render outside the project tree
                shown = src
            lines.append(f"| {target.name} | {shown} | {sha256_file(target)} |")
            n += 1
    (dest / SUBMISSION_README).write_text("\n".join(lines) + "\n")
    log.info("submission bundle: %d figure(s) copied into %s", n, dest)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the manuscript figures (no baked-in titles).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--out-dir", default=None,
                    help="override manuscript.figures_dir (default: read from config)")
    ap.add_argument("--split", default=None, choices=list(SPLITS),
                    help="which split to render (default: manuscript.report_split)")
    ap.add_argument("--only", default="all", choices=["all", *FIGURE_KEYS, *SUPPLEMENT_KEYS],
                    help="render a single figure")
    ap.add_argument("--supplement", action="store_true",
                    help="render the supplementary set S1-S6 instead of the main figures")
    ap.add_argument("--both", action="store_true",
                    help="render the main figures and then the supplementary set")
    ap.add_argument("--bundle", action="store_true",
                    help="also write the journal-facing copy named by figure number")
    ap.add_argument("--build-assets", default=None, metavar="SHARD_TAR",
                    help="stage figure 1's full-resolution crop out of a shard tar and exit")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.build_assets:
        setup_logging(cfg.path(cfg["paths"]["run_log"]))
        build_figure1_assets(cfg, Path(args.build_assets))
        return 0
    if args.supplement:
        render_supplement(cfg, out_dir=args.out_dir, only=args.only, split=args.split)
    else:
        render_all(cfg, out_dir=args.out_dir, only=args.only, split=args.split)
        if args.both:
            render_supplement(cfg, out_dir=args.out_dir, split=args.split)
    if args.bundle:
        write_submission_bundle(cfg, assert_split(args.split or default_split(cfg)),
                                args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
