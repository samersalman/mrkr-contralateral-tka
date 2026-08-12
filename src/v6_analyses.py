"""v6 resubmission, Phase 2 analyses A1 and A2 - POST HOC, on the already-read test split.

WHAT THIS MODULE IS
-------------------
The *Journal of Imaging* resubmission moves the paper's central claim from "a multimodal
model beats a clinical model" to an imaging claim: **a single frontal knee radiograph
carries prognostic information about contralateral arthroplasty that Kellgren-Lawrence
grading does not capture.** Two analyses support that claim and neither existed in v5:

* **A1 - single-view headline contrasts.** M2 frontal vs M0 clinical, and M2 frontal vs M1
  clinical-plus-inferred-KLG, paired, at the 5-year horizon, in both the primary estimand
  (IPCW cumulative/dynamic AUROC at 1825 d) and Harrell C.
* **A2 - prediction within Kellgren-Lawrence strata.** Observed Kaplan-Meier 5-year
  incidence by model risk tertile *within* KL stratum; a Cox model carrying KL grade plus
  the model risk score, with a likelihood-ratio test of the score's incremental
  contribution; and exploratory within-stratum AUROC with the protocol section 21
  suppression floor honoured.

**Everything here is post hoc.** The test split was read once, in
``src/score_test.py``/``src/eval_models.py --split test``; these analyses were specified
after that read and are recorded as deviation D35. No row this module writes is
``is_primary``, and every row of every table it writes carries that status in its ``note``
column, not only in the prose around it.

WHAT THIS MODULE IS NOT
-----------------------
It does not re-estimate anything. Every estimator comes from ``src.eval_models`` and
``src.model_clinical`` by import: :class:`~src.eval_models.BootstrapEngine`,
:func:`~src.eval_models.contrast_row`, :func:`~src.eval_models.benjamini_hochberg`,
:func:`~src.eval_models.suppress_unfit_contrasts`, :func:`~src.model_clinical.km_risk`,
:func:`~src.model_clinical.risk_bins`, :func:`~src.model_clinical.cloglog`. The one shared
patient-level bootstrap draw is rebuilt from ``model_eval.bootstrap_seed`` at
``model_eval.bootstrap_n``, so replicate *b* here is the same set of patients as replicate
*b* in the published ``test_comparisons.csv`` and every difference stays paired.

It writes only ``outputs/tables/v6_*.csv``. It never touches a published table, a
``.npz``, or ``derived-data/``. The published ``.npz`` files hold RAW ensemble hazards; the
frozen recalibration is applied inside :func:`~src.eval_models.trained_arm_scores` and is
never applied a second time here.

MULTIPLICITY
------------
``src.eval_models.build_comparisons`` applies Benjamini-Hochberg **within one declared
family at a time**, and its own docstring says pooling families "would silently change the
multiplicity of every contrast in every other family". Adding these post-hoc contrasts to
an existing family (``modality``, ``views``, ``robustness``) would therefore change the
adjusted p of contrasts that are already published in ``outputs/tables/test_comparisons.csv``
and quoted in the v5 manuscript. So they form **their own families**, declared in
:data:`POSTHOC_FAMILIES` and :data:`WITHIN_STRATUM_FAMILY`, each adjusted inside itself
alone. Not one published number moves.

THE 50-EVENT FLOOR
------------------
``model_eval.suppress_below_events`` is 50 and is double-locked against
``src.model_clinical.SUPPRESS_BELOW_EVENTS`` at ``src/eval_models.py:1136``. This module
imports that constant and asserts against it exactly as ``build_subgroups`` does. It is
never weakened, bypassed or patched. The KLG-eligible test population carries 98 events, so
*no* partition of it can leave two strata above the floor (2 x 50 > 98); at most one
stratum can be estimated, and the module reports which and says the rest are suppressed.

Run::

    PYTHONPATH="$PWD" ~/.venvs/mrkr-torch/bin/python -m src.v6_analyses \\
        --config config/feasibility.yaml [--bootstrap-n N] [--out-dir outputs/tables]

Outputs (all under ``--out-dir``, default ``outputs/tables``):

``v6_test_comparisons_posthoc.csv``  A1, plus the within-stratum M4-vs-M2 contrast from A2,
                                     in ``eval_models.COMPARISON_COLUMNS`` order
``v6_klg_strata.csv``                A2, the bin definition and its counts
``v6_klg_risk_tertiles.csv``         A2(i), KM 5-year incidence by risk tertile within stratum
``v6_klg_cox_incremental.csv``       A2(ii), Cox HR per SD and the LR test
``v6_klg_stratum_auroc.csv``         A2(iii), within-stratum AUROC, floor honoured
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.eval_models import (
    COMPARISON_COLUMNS,
    COX_ARMS,
    PRIMARY_METRIC,
    BootstrapEngine,
    _append_note,
    assert_sealed_read_is_recorded,
    benjamini_hochberg,
    bootstrap_draw,
    contrast_row,
    convergence_diagnostics,
    cox_arm_scores,
    horizons_from_config,
    load_roster,
    load_train_arms,
    percentile_ci,
    suppress_unfit_contrasts,
    trained_arm_scores,
    two_sided_bootstrap_p,
    write_table,
)
from src.model_clinical import SEALED_SPLIT, SUPPRESS_BELOW_EVENTS, cloglog, km_risk, risk_bins
from src.train_model import FrozenContracts

MODULE = "v6_analyses"

# --------------------------------------------------------------------------- #
# What every row of every table says about itself.                             #
# --------------------------------------------------------------------------- #
POSTHOC_NOTE = ("POST HOC EXPLORATORY: specified after the single permitted read of the "
                "sealed test split and reported under deviation D35; not pre-specified and "
                "not confirmatory")

# --------------------------------------------------------------------------- #
# A1. The post-hoc contrast families.                                          #
#                                                                              #
# Three families, each Benjamini-Hochberg adjusted inside itself alone, exactly #
# as build_comparisons adjusts the pre-specified families. They are new         #
# families rather than additions to existing ones because BH inside a family is #
# a function of the family's size: appending to `modality` would change the     #
# adjusted p of four already-published rows.                                    #
#                                                                              #
#  posthoc_single_view       the two headline contrasts on the PRIMARY estimand #
#  posthoc_single_view_c     the same two comparisons on Harrell C, a horizon-  #
#                            free secondary metric, adjusted separately so it   #
#                            cannot dilute the primary-metric family            #
#  posthoc_single_view_sens  sensitivity to which M1 implementation is used as  #
#                            the KLG comparator (the discrete-time m1_klg arm   #
#                            in place of the frozen penalized Cox m1). m1 and   #
#                            m1_klg are two implementations of ONE comparator,  #
#                            not two hypotheses, so the sensitivity gets its    #
#                            own family instead of inflating m in the two above #
# --------------------------------------------------------------------------- #
POSTHOC_FAMILIES: dict[str, list[tuple[str, str, str]]] = {
    "posthoc_single_view": [
        ("m2_frontal", "m0", PRIMARY_METRIC),
        ("m2_frontal", "m1", PRIMARY_METRIC),
    ],
    "posthoc_single_view_c": [
        ("m2_frontal", "m0", "harrell_c"),
        ("m2_frontal", "m1", "harrell_c"),
    ],
    "posthoc_single_view_sens": [
        ("m2_frontal", "m1_klg", PRIMARY_METRIC),
        ("m2_frontal", "m1_klg", "harrell_c"),
    ],
}

# A fourth family, from A2 rather than A1, in the same file because it is a paired contrast
# and carries the same schema: does the fusion model beat the frontal-only model among knees
# a radiologist would grade the same? Its rows are subject to the same 50-event floor as
# every other within-stratum estimate.
WITHIN_STRATUM_FAMILY = "posthoc_kl_within"

HARRELL_C_NOTE = ("Harrell C is horizon free; horizon_days carries 1825 only because the "
                  "pinned comparison schema requires the column")
SENS_NOTE = ("sensitivity: m1_klg is the discrete-time KLG-eligible arm, an alternative "
             "implementation of the same clinical-plus-KLG comparator as the frozen "
             "penalized Cox m1")

# --------------------------------------------------------------------------- #
# A2. KELLGREN-LAWRENCE BIN EDGES - DECLARED HERE, NEVER CHOSEN BY PANDAS.      #
#                                                                              #
# klg_contra takes 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 and 4 on the test split. The  #
# half-grades are averages of two DIFFERENT frontal reads of the same knee      #
# (src/features_clinical.py:224), not a finer scale; 31 of 707 KLG-eligible     #
# test patients carry one.                                                      #
#                                                                              #
# PRIMARY SCHEME `kl3`, three strata, left-closed bins on the numeric grade:    #
#     KL 0-1   klg_contra <  1.5     no or doubtful radiographic osteoarthritis #
#     KL 2     1.5 <= klg_contra < 2.5   definite osteophyte, the conventional  #
#                                        threshold for radiographic OA          #
#     KL 3-4   klg_contra >= 2.5    definite to severe joint-space narrowing    #
#                                                                              #
# Why these edges:                                                              #
#  1. They are the standard clinical collapse of an ordinal 5-level scale. KL 0 #
#     and KL 1 are pooled because they carry 3 and 0 events here; KL 3 and KL 4 #
#     are pooled because both are "advanced" and arthroplasty is conventionally #
#     considered in that range.                                                 #
#  2. Edges sit at half-integers so the rule is stated in the grade's own units #
#     and no rounding of klg_contra is required anywhere.                       #
#  3. Left-closed bins send a tied read to the HIGHER stratum: a knee averaging #
#     2.5 had at least one frontal read of grade 3 and is managed as grade 3.   #
#     That rule moves 2 patients at 1.5 and 14 at 2.5. It is a real choice, so  #
#     `kl3_tielow` re-runs everything with the opposite rule as a sensitivity.  #
#  4. The PRIMARY inferential result - the Cox likelihood-ratio test - enters   #
#     KL as the raw numeric grade and therefore does not depend on any binning  #
#     at all. The bins drive only the descriptive and the suppressed-AUROC      #
#     tables.                                                                   #
#                                                                              #
# COARSE SCHEME `kl2`: KL 0-2 vs KL 3-4, a single cut at 2.5. Declared because  #
# 98 events cannot support two strata above the 50-event floor under ANY        #
# partition, so the coarsest clinically meaningful cut is the one with the best #
# chance of leaving an estimable cell, and reporting it beside `kl3` shows the  #
# cut was not shopped.                                                          #
# --------------------------------------------------------------------------- #
KLG_COL = "klg_contra"
KL_EDGES = (1.5, 2.5)                 # THE bin edges. Declared once, used everywhere.

KL_SCHEMES: dict[str, dict] = {
    "kl3": {
        "label": "Kellgren-Lawrence grade, three strata",
        "edges": KL_EDGES,
        "tie_rule": "high",
        "strata": [("KL 0-1", "klg_contra < 1.5"),
                   ("KL 2", "1.5 <= klg_contra < 2.5"),
                   ("KL 3-4", "klg_contra >= 2.5")],
        "primary": True,
    },
    "kl2": {
        "label": "Kellgren-Lawrence grade, advanced versus not",
        "edges": (2.5,),
        "tie_rule": "high",
        "strata": [("KL 0-2", "klg_contra < 2.5"),
                   ("KL 3-4", "klg_contra >= 2.5")],
        "primary": False,
    },
    "kl3_tielow": {
        "label": "Kellgren-Lawrence grade, three strata, tied reads to the lower stratum",
        "edges": KL_EDGES,
        "tie_rule": "low",
        "strata": [("KL 0-1", "klg_contra <= 1.5"),
                   ("KL 2", "1.5 < klg_contra <= 2.5"),
                   ("KL 3-4", "klg_contra > 2.5")],
        "primary": False,
    },
}

# The arm whose risk score is stratified. M2 is the paper's new thesis; M4 runs alongside
# so the Discussion can say whether fusion adds anything once KL grade is held fixed.
PRIMARY_ARM = "m2_frontal"
SECONDARY_ARM = "m4_fusion"
A2_ARMS = (PRIMARY_ARM, SECONDARY_ARM)
KLG_COMPARATOR_ARMS = ("m1", "m1_klg")     # loaded for A1 only
N_TERTILES = 3
TERTILE_LABELS = ("Lowest tertile", "Middle tertile", "Highest tertile")

# A 95% interval at least this wide is flagged. 0.15 is the top of the range spanned by the
# six published test-split subgroup estimates that cleared the same floor (0.101 to 0.160,
# outputs/tables/test_subgroups.csv), so the flag marks any stratum estimate no more precise
# than the least precise subgroup estimate the paper already reports.
WIDE_INTERVAL_WIDTH = 0.15

TERTILE_NOTE = ("cause-agnostic cumulative incidence; death is not ascertainable in this "
                "data source, so competing mortality is unmeasured")

# src.eval_models.write_table rounds every float column to ROUND_DECIMALS = 6 so two runs
# write byte-identical CSVs. That is right for a discrimination estimate and for a
# bootstrap p, whose resolution floor is 1/2000 = 0.0005 anyway. It is WRONG for a
# parametric p: a likelihood-ratio p of 5.1e-16 would be written as 0.0, which reads as an
# impossible exact zero. Every parametric p therefore also gets a text column carrying the
# value to three significant figures, which no rounding can touch.
P_TEXT_PRECISION = 3


def _p_text(p: float) -> str:
    """A parametric p as text, immune to the 6-decimal rounding the CSV writer applies."""
    return "n/a" if not np.isfinite(p) else f"{float(p):.{P_TEXT_PRECISION}g}"

# --------------------------------------------------------------------------- #
# Pinned output schemas. write_table asserts column order against these.       #
# --------------------------------------------------------------------------- #
KLG_STRATA_COLUMNS = ["scheme", "scheme_label", "stratum", "stratum_order", "rule",
                      "klg_values", "n_patients", "n_events", "cleared_event_floor", "note"]

KLG_TERTILE_COLUMNS = ["arm", "scheme", "stratum", "stratum_order", "tertile",
                       "tertile_label", "horizon_days", "n_patients", "n_events",
                       "n_at_risk_horizon", "min_predicted_risk", "max_predicted_risk",
                       "mean_predicted_risk", "km_cumulative_incidence", "km_ci_lo",
                       "km_ci_hi", "n_stratum_patients", "n_stratum_events",
                       "logrank_chi2", "logrank_df", "logrank_p", "logrank_p_text", "note"]

KLG_COX_COLUMNS = ["arm", "specification", "kl_form", "n_patients", "n_events", "term",
                   "coef", "se", "hr", "hr_lo", "hr_hi", "p_wald", "p_wald_text", "ph_p",
                   "is_score_term", "score_sd", "ll_full", "ll_reference", "lr_chi2",
                   "lr_df", "lr_p", "lr_p_text", "note"]

KLG_AUROC_COLUMNS = ["arm", "scheme", "stratum", "stratum_order", "n_patients", "n_events",
                     "metric", "horizon_days", "estimate", "ci_lo", "ci_hi", "ci_width",
                     "wide_interval", "suppressed", "suppression_reason", "note"]

OUTPUT_BASENAMES = {
    "comparisons": "v6_test_comparisons_posthoc.csv",
    "strata": "v6_klg_strata.csv",
    "tertiles": "v6_klg_risk_tertiles.csv",
    "cox": "v6_klg_cox_incremental.csv",
    "auroc": "v6_klg_stratum_auroc.csv",
}


def setup_logging() -> logging.Logger:
    log = logging.getLogger(MODULE)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                         "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


# =========================================================================== #
# 1. CONTEXT - the sealed-split roster, the arms, the ONE shared draw          #
# =========================================================================== #
class Context:
    """Everything both analyses need, built once from the published artefacts.

    The split is ``test``. That read is not a new read: ``src/score_test.py`` performed the
    single permitted sealed read and recorded it in ``derived-data/cohort/test_scoring.json``;
    :func:`src.eval_models.assert_sealed_read_is_recorded` is called here so this module
    refuses to run if that record is absent or the training contract has moved.
    """

    def __init__(self, cfg: Config, log: logging.Logger, n_boot: int | None = None):
        self.cfg, self.log = cfg, log
        self.contract = assert_sealed_read_is_recorded(cfg)
        self.horizons = horizons_from_config(cfg)
        self.horizon = int(cfg["model_eval"]["primary_contrast"]["horizon_days"])
        assert self.horizon in self.horizons
        coh = cfg.path(cfg["paths"]["cohort_dir"])
        contracts = FrozenContracts(coh)
        train_arms = load_train_arms(coh / "train_arms.json")
        self.roster, _ = load_roster(contracts, log, split=SEALED_SPLIT)

        wanted = set(A2_ARMS) | set(KLG_COMPARATOR_ARMS) | {m for f in POSTHOC_FAMILIES.values()
                                                           for pair in f for m in pair[:2]}
        self.arms = {}
        for arm in COX_ARMS:
            if arm in wanted:
                self.arms[arm] = cox_arm_scores(arm, contracts, self.roster, self.horizons,
                                                log, split=SEALED_SPLIT)
        for arm in sorted(wanted - set(COX_ARMS)):
            summary = (train_arms["arms"] or {}).get(arm)
            assert summary, f"{arm} is not in train_arms.json; it was never trained"
            sc = trained_arm_scores(arm, summary, coh, self.roster, self.horizons, log,
                                    split=SEALED_SPLIT)
            assert sc is not None, f"{SEALED_SPLIT}_hazards_{arm}.npz is missing"
            self.arms[arm] = sc

        me = cfg["model_eval"]
        self.n_boot = int(n_boot if n_boot is not None else me["bootstrap_n"])
        self.seed = int(me["bootstrap_seed"])
        self.draw = bootstrap_draw(len(self.roster), self.n_boot, self.seed)
        self.engine = BootstrapEngine(self.roster, self.draw, self.horizons, log)
        self.convergence = convergence_diagnostics(cfg, log)
        log.info("context: %d test patients, %d events, contract %s, %d bootstrap replicates "
                 "(seed %d, the same draw the published tables used)", len(self.roster),
                 int(self.roster.event.sum()), self.contract, self.n_boot, self.seed)

    # -- the KLG-eligible analysis population -------------------------------- #
    @property
    def klg(self) -> np.ndarray:
        return self.roster.frame[KLG_COL].to_numpy(dtype=float)

    @property
    def klg_eligible(self) -> np.ndarray:
        """Patients with an inferred contralateral KL grade. Exactly M1's eligible set."""
        m = np.isfinite(self.klg)
        assert np.array_equal(m, self.arms["m1"].present), (
            "the KLG-eligible mask disagrees with M1's patient set; M1 is DEFINED as the "
            "klg_contra-non-missing subset (src/train_model.build_clinical_design)")
        return m


# =========================================================================== #
# 2. A1 - THE POST-HOC SINGLE-VIEW CONTRASTS                                   #
# =========================================================================== #
def build_posthoc_comparisons(ctx: Context) -> pd.DataFrame:
    """The A1 families, each BH-adjusted inside itself, in the published schema.

    :func:`src.eval_models.contrast_row` does all of the estimation: it intersects the two
    arms' patient sets, recomputes BOTH point estimates on that intersection, and takes the
    paired bootstrap difference on the shared draw. Nothing is subtracted from a marginal
    level here, and no estimator is reimplemented.
    """
    log = ctx.log
    rows: list[dict] = []
    for family in POSTHOC_FAMILIES:                      # declaration order, not sorted
        fam: list[dict] = []
        for model, reference, metric in POSTHOC_FAMILIES[family]:
            r = contrast_row(family, model, reference, metric=metric, horizon=ctx.horizon,
                             arms=ctx.arms, engine=ctx.engine, is_primary=False)
            assert r is not None, f"{model} vs {reference} is not estimable; both arms load"
            r["note"] = _append_note(r["note"], POSTHOC_NOTE)
            if metric != PRIMARY_METRIC:
                r["note"] = _append_note(r["note"], HARRELL_C_NOTE)
            if reference == "m1_klg":
                r["note"] = _append_note(r["note"], SENS_NOTE)
            fam.append(r)
        # The published convergence gate, on the SEALED split: `did_not_converge` still
        # suppresses, `severe_overfit` does not, because the test split took no part in the
        # checkpoint selection that makes a validation metric optimistic.
        suppress_unfit_contrasts(fam, ctx.convergence, log, sealed=True)
        adj = benjamini_hochberg([r["p_two_sided"] for r in fam])
        for r, q in zip(fam, adj):
            r["p_adjusted"] = float(q)
            r["fdr_method"] = str(ctx.cfg["model_eval"].get("fdr_method", "bh")).lower()
            r["note"] = _append_note(
                r["note"],
                f"Benjamini-Hochberg within the new family {family!r} ({len(fam)} contrasts) "
                f"only; no published family's multiplicity is changed")
            log.info("A1 %-24s %s minus %s (%s): %+.4f (95%% CI %+.4f to %+.4f), p %.4f, "
                     "q %.4f, %d paired patients; %s scores %.4f and %s scores %.4f on that set",
                     family, r["model"], r["reference"], r["metric"], r["difference"],
                     r["ci_lo"], r["ci_hi"], r["p_two_sided"], r["p_adjusted"], r["n_paired"],
                     r["model"], r["estimate_model"], r["reference"], r["estimate_reference"])
        rows.extend(fam)
    rows.extend(build_within_stratum_contrasts(ctx))
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def stratum_contrast_row(ctx: Context, family: str, model: str, reference: str, *,
                         metric: str, stratum: str, stratum_rule: str,
                         stratum_mask: np.ndarray) -> dict:
    """One paired contrast restricted to a KL stratum, in the published contrast schema.

    This is :func:`src.eval_models.contrast_row`'s structure with ONE extra term in the
    intersection - the stratum mask - which that function has no parameter for. Every
    estimator is imported and none is reimplemented: ``BootstrapEngine.point``,
    ``BootstrapEngine.boot`` (which intersects the mask with ``ArmScores.present`` itself),
    ``percentile_ci`` and ``two_sided_bootstrap_p``, all on the one shared draw, so the
    difference stays paired replicate by replicate.

    The protocol section 21 floor applies here exactly as it does to a subgroup estimate: a
    stratum below :data:`SUPPRESS_BELOW_EVENTS` events carries no estimate at all, not even
    the two arms' own levels, and its NaN p is dropped from the family's multiplicity.
    """
    a, b = ctx.arms[model], ctx.arms[reference]
    mask = a.present & b.present & stratum_mask
    key = f"{metric}@{ctx.horizon}" if metric == PRIMARY_METRIC else metric
    n_paired = int(mask.sum())
    n_ev = int(ctx.roster.event[mask].sum())
    note = _append_note(POSTHOC_NOTE,
                        f"within the {stratum} stratum ({stratum_rule}); paired on the "
                        f"{n_paired} KLG-eligible patients both arms score, {n_ev} events")
    if metric != PRIMARY_METRIC:
        note = _append_note(note, HARRELL_C_NOTE)
    row = dict(family=family, model=model, reference=reference, metric=metric,
               horizon_days=int(ctx.horizon), n_paired=n_paired,
               estimate_model=float("nan"), estimate_reference=float("nan"),
               difference=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
               p_two_sided=float("nan"), p_adjusted=float("nan"), fdr_method="",
               is_primary=False, note=note)
    if n_ev < SUPPRESS_BELOW_EVENTS:
        row["note"] = _append_note(row["note"], (
            f"SUPPRESSED -- protocol section 21: fewer than {SUPPRESS_BELOW_EVENTS} events "
            f"({n_ev} in this stratum); no estimate, no interval and no p value is reported, "
            f"and this row is excluded from the family's Benjamini-Hochberg multiplicity"))
        return row
    em = float(ctx.engine.point(a, mask)[key])
    er = float(ctx.engine.point(b, mask)[key])
    d = ctx.engine.boot(a, mask)[key] - ctx.engine.boot(b, mask)[key]
    lo, hi = percentile_ci(d)
    p, n_valid = two_sided_bootstrap_p(d)
    if n_valid < len(ctx.engine.draw):
        row["note"] = _append_note(row["note"], (
            f"{len(ctx.engine.draw) - n_valid} of {len(ctx.engine.draw)} replicates were not "
            f"estimable and are excluded"))
    row.update(estimate_model=em, estimate_reference=er, difference=em - er,
               ci_lo=float(lo), ci_hi=float(hi), p_two_sided=float(p))
    return row


def build_within_stratum_contrasts(ctx: Context) -> list[dict]:
    """M4 fusion minus M2 frontal inside each KL stratum: does fusion add anything there?

    The paper's claim is that neither the extra views nor the clinical record buys
    performance the frontal film does not already carry. Two overlapping unpaired
    within-stratum intervals cannot support that; a paired difference can.
    """
    rows: list[dict] = []
    labels: list[str] = []
    for metric in (PRIMARY_METRIC, "harrell_c"):
        for _order, label, rule, mask in kl_stratum_masks(ctx.klg, "kl3"):
            rows.append(stratum_contrast_row(
                ctx, WITHIN_STRATUM_FAMILY, SECONDARY_ARM, PRIMARY_ARM, metric=metric,
                stratum=label, stratum_rule=rule, stratum_mask=mask & ctx.klg_eligible))
            labels.append(label)
    suppress_unfit_contrasts(rows, ctx.convergence, ctx.log, sealed=True)
    adj = benjamini_hochberg([r["p_two_sided"] for r in rows])
    fdr = str(ctx.cfg["model_eval"].get("fdr_method", "bh")).lower()
    n_est = int(np.isfinite([r["p_two_sided"] for r in rows]).sum())
    for r, q, label in zip(rows, adj, labels):
        r["p_adjusted"] = float(q)
        r["fdr_method"] = fdr if np.isfinite(q) else ""
        r["note"] = _append_note(r["note"], (
            f"Benjamini-Hochberg within the new family {WITHIN_STRATUM_FAMILY!r} "
            f"({n_est} estimable of {len(rows)} rows) only; no published family's "
            f"multiplicity is changed"))
        if np.isfinite(r["difference"]):
            ctx.log.info("A2 within-stratum %s minus %s (%s) in %-7s: %+.4f (95%% CI %+.4f to "
                         "%+.4f), p %.4f, q %.4f, %d paired patients", r["model"],
                         r["reference"], r["metric"], label, r["difference"], r["ci_lo"],
                         r["ci_hi"], r["p_two_sided"], r["p_adjusted"], r["n_paired"])
    return rows


# =========================================================================== #
# 3. A2 - KELLGREN-LAWRENCE STRATA                                             #
# =========================================================================== #
def kl_stratum_masks(klg: np.ndarray, scheme: str) -> list[tuple[int, str, str, np.ndarray]]:
    """(order, label, rule, mask) for one declared scheme. Edges come from KL_SCHEMES only.

    ``np.digitize`` is called with the declared edges and an explicit ``right`` flag; no
    quantile, no ``pd.cut``, no automatic binning is involved anywhere in this module.
    """
    spec = KL_SCHEMES[scheme]
    edges = list(spec["edges"])
    right = (spec["tie_rule"] == "low")      # right=True closes each bin on the right
    idx = np.digitize(klg, edges, right=right)
    out = []
    for i, (label, rule) in enumerate(spec["strata"]):
        out.append((i + 1, label, rule, np.isfinite(klg) & (idx == i)))
    covered = np.zeros_like(klg, dtype=bool)
    for _, _, _, m in out:
        assert not (covered & m).any(), "KL strata overlap"
        covered |= m
    assert np.array_equal(covered, np.isfinite(klg)), "KL strata do not cover every graded knee"
    return out


def build_klg_strata(ctx: Context) -> pd.DataFrame:
    """The bin definition itself, with counts, so the edges are auditable from the outputs."""
    klg, ev = ctx.klg, ctx.roster.event
    rows: list[dict] = []
    for scheme, spec in KL_SCHEMES.items():
        for order, label, rule, mask in kl_stratum_masks(klg, scheme):
            n_ev = int(ev[mask].sum())
            rows.append(dict(
                scheme=scheme, scheme_label=str(spec["label"]), stratum=label,
                stratum_order=order, rule=rule,
                klg_values=" ".join(f"{v:g}" for v in sorted(set(klg[mask].tolist()))),
                n_patients=int(mask.sum()), n_events=n_ev,
                cleared_event_floor=bool(n_ev >= SUPPRESS_BELOW_EVENTS),
                note=_append_note(POSTHOC_NOTE, f"bin edges {list(spec['edges'])}, tied "
                                                f"averaged reads to the "
                                                f"{spec['tie_rule']}er stratum")))
    df = pd.DataFrame(rows, columns=KLG_STRATA_COLUMNS)
    for _, r in df.iterrows():
        ctx.log.info("A2 strata %-11s %-8s n=%3d events=%3d %s", r["scheme"], r["stratum"],
                     r["n_patients"], r["n_events"],
                     "CLEARS the 50-event floor" if r["cleared_event_floor"] else "suppressed")
    return df


def model_score(ctx: Context, arm: str) -> np.ndarray:
    """The model risk score on the roster: cloglog of the 5-year predicted risk.

    ``ArmScores.risk`` already carries the frozen recalibration, applied once inside
    :func:`src.eval_models.trained_arm_scores`. The recalibration is affine ON THE CLOGLOG
    SCALE (``cloglog(P) = a + b cloglog(p_hat)``, ``train_arms.json``), so after the
    standardisation this function's consumers apply, the score is IDENTICAL whether or not
    the recalibration was applied. That makes every A2 result invariant to the one thing
    most likely to be got wrong here, and it makes the score a linear-predictor scale, which
    is what a proportional-hazards model wants.
    """
    p = ctx.arms[arm].risk[ctx.horizon]
    return cloglog(np.where(np.isfinite(p), p, 0.5))     # placeholder where not present


def analysis_mask(ctx: Context, arm: str) -> np.ndarray:
    """KLG-eligible AND scored by this arm."""
    return ctx.klg_eligible & ctx.arms[arm].present


def build_klg_tertiles(ctx: Context) -> pd.DataFrame:
    """A2(i). Observed 5-year KM incidence by model risk tertile WITHIN each KL stratum.

    Tertiles are formed inside the stratum, not globally, because the question is whether
    the image separates risk among knees a radiologist would grade the same. Equal-count
    bins come from :func:`src.model_clinical.risk_bins` and the incidence and its Greenwood
    interval from :func:`src.model_clinical.km_risk` - the same two functions the published
    figure 3 tertile table rests on. No AUROC and no bootstrap is involved, so the 50-event
    floor does not apply: these are observed counts, reported with their own intervals.
    """
    from lifelines.statistics import multivariate_logrank_test    # noqa: PLC0415

    log = ctx.log
    T, E = ctx.roster.time, ctx.roster.event
    rows: list[dict] = []
    for arm in A2_ARMS:
        base = analysis_mask(ctx, arm)
        pred = ctx.arms[arm].risk[ctx.horizon]
        strata: list[tuple[int, str, np.ndarray]] = [(0, "All KL grades", base)]
        strata += [(o, lab, m & base) for o, lab, _, m in kl_stratum_masks(ctx.klg, "kl3")]
        for order, label, mask in strata:
            idx = np.flatnonzero(mask)
            n_s, ev_s = idx.size, int(E[idx].sum())
            if n_s < N_TERTILES:
                log.warning("A2 tertiles %s / %s: only %d patients, skipped", arm, label, n_s)
                continue
            groups = risk_bins(pred[idx], N_TERTILES)
            lr_chi2 = lr_p = float("nan")
            if ev_s > 0:
                lr = multivariate_logrank_test(T[idx], groups, E[idx])
                lr_chi2, lr_p = float(lr.test_statistic), float(lr.p_value)
            for g in range(N_TERTILES):
                k = idx[groups == g]
                n_ev = int(E[k].sum())
                inc, lo, hi = (km_risk(T[k], E[k], float(ctx.horizon)) if n_ev > 0
                               else (0.0, 0.0, float("nan")))
                rows.append(dict(
                    arm=arm, scheme="kl3", stratum=label, stratum_order=order,
                    tertile=g + 1, tertile_label=TERTILE_LABELS[g],
                    horizon_days=ctx.horizon, n_patients=int(k.size), n_events=n_ev,
                    n_at_risk_horizon=int((T[k] >= float(ctx.horizon)).sum()),
                    min_predicted_risk=float(pred[k].min()),
                    max_predicted_risk=float(pred[k].max()),
                    mean_predicted_risk=float(pred[k].mean()),
                    km_cumulative_incidence=float(inc), km_ci_lo=float(lo), km_ci_hi=float(hi),
                    n_stratum_patients=n_s, n_stratum_events=ev_s,
                    logrank_chi2=lr_chi2, logrank_df=float(N_TERTILES - 1), logrank_p=lr_p,
                    logrank_p_text=_p_text(lr_p),
                    note=_append_note(POSTHOC_NOTE, TERTILE_NOTE + "; tertiles are formed "
                                      "within the stratum, and the log-rank test across them "
                                      "is exploratory and unadjusted; logrank_p_text carries "
                                      "the p to three significant figures because logrank_p "
                                      "is rounded to six decimals by the house CSV writer")))
            log.info("A2 tertiles %-10s %-14s n=%3d ev=%3d  5-y incidence %s  log-rank p %s",
                     arm, label, n_s, ev_s,
                     " / ".join(f"{r['km_cumulative_incidence']:.1%}" for r in rows[-N_TERTILES:]),
                     f"{lr_p:.4g}" if np.isfinite(lr_p) else "n/a")
    return pd.DataFrame(rows, columns=KLG_TERTILE_COLUMNS)


def _cox_frame(ctx: Context, arm: str) -> pd.DataFrame:
    """The Cox analysis frame: time, event, KL grade, and the standardized model score."""
    mask = analysis_mask(ctx, arm)
    idx = np.flatnonzero(mask)
    z = model_score(ctx, arm)[idx]
    sd = float(z.std(ddof=1))
    assert sd > 0, "the model score has no spread"
    df = pd.DataFrame({
        "T": ctx.roster.time[idx].astype(float),
        "E": ctx.roster.event[idx].astype(int),
        "kl": ctx.klg[idx].astype(float),
        "score_z": (z - z.mean()) / sd,
    })
    assert (df["T"] > 0).all(), "a non-positive follow-up time reached the Cox fit"
    order, labels = [], {}
    for o, lab, _, m in kl_stratum_masks(ctx.klg, "kl3"):
        order.append(lab)
        for i in np.flatnonzero(m[idx]):
            labels[int(i)] = lab
    df["kl_stratum"] = pd.Series([labels[i] for i in range(len(df))], dtype=object)
    for lab in order[1:]:                 # first stratum is the reference level
        df[f"kl_{lab.replace(' ', '').replace('-', '_')}"] = (df["kl_stratum"] == lab).astype(float)
    df.attrs["score_sd"] = sd
    df.attrs["dummies"] = [f"kl_{lab.replace(' ', '').replace('-', '_')}" for lab in order[1:]]
    return df


def _fit(df: pd.DataFrame, covariates: list[str], strata: list[str] | None = None):
    """Fit one Cox model and return it WITH the exact frame it was fitted on.

    The frame is returned rather than rebuilt because
    :func:`lifelines.statistics.proportional_hazard_test` casts every column of whatever it
    is handed to float, so passing the full analysis frame - which carries the string
    stratum label - raises. It must see the model's own columns and nothing else.
    """
    from lifelines import CoxPHFitter                              # noqa: PLC0415
    cols = ["T", "E"] + list(covariates) + (list(strata) if strata else [])
    sub = df[cols].copy()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="T", event_col="E", strata=strata)
    return cph, sub


def _ph_pvalues(cph, fitted_df: pd.DataFrame, log: logging.Logger) -> dict[str, float]:
    """Per-covariate Schoenfeld proportional-hazards test p, rank time transform.

    A failure is logged, never swallowed: a silently empty result would fill the ``ph_p``
    column with NaN and read as "the assumption was not checked" when the truth is "the
    check crashed".
    """
    from lifelines.statistics import proportional_hazard_test      # noqa: PLC0415
    try:
        res = proportional_hazard_test(cph, fitted_df, time_transform="rank")
        s = res.summary["p"]
        return {str(k[0] if isinstance(k, tuple) else k): float(v) for k, v in s.items()}
    except Exception as exc:                                       # noqa: BLE001
        log.warning("the Schoenfeld proportional-hazards test failed for a model on "
                    "%s: %s: %s; ph_p is left NaN for its terms",
                    list(fitted_df.columns), type(exc).__name__, exc)
        return {}


def build_klg_cox(ctx: Context) -> pd.DataFrame:
    """A2(ii). Does the frontal radiograph add information over the KL grade?

    Three specifications, PRE-SPECIFIED HERE in this order and all reported:

    ``kl_linear`` (primary)
        ``kl`` enters as the raw numeric grade, one degree of freedom. KL is an ordinal
        scale whose half-grades are meaningful intermediates (two frontal reads that
        disagreed by one grade), the relation of grade to arthroplasty hazard is expected to
        be monotone, and at 98 events one degree of freedom is where the power is. Because
        the grade enters as recorded, this result does not depend on any bin edge.

    ``kl_categorical`` (sensitivity)
        ``kl`` enters as the three ``kl3`` strata with KL 0-1 as reference, two degrees of
        freedom, relaxing linearity.

    ``kl_stratified`` (sensitivity)
        the baseline hazard is stratified by the three ``kl3`` strata and the score is the
        only covariate, so no functional form for KL is assumed at all. This is the
        assumption-light version of "within stratum".

    ``score_only`` is reported alongside as the unadjusted hazard ratio, so the reader can
    see how much of the score's association the KL grade accounts for.

    In every specification the likelihood-ratio test compares the model WITH the score
    against the identical model WITHOUT it, on the same patients: 2 x (ll_full - ll_reference)
    on 1 degree of freedom. That is the incremental contribution of the radiograph over the
    grade.
    """
    from scipy import stats                                        # noqa: PLC0415

    log = ctx.log
    rows: list[dict] = []
    for arm in A2_ARMS:
        df = _cox_frame(ctx, arm)
        sd = float(df.attrs["score_sd"])
        dummies = list(df.attrs["dummies"])
        n_pat, n_ev = int(len(df)), int(df["E"].sum())
        specs = [
            ("kl_linear", "numeric grade, linear (primary)", ["kl"], None),
            ("kl_categorical", "three kl3 strata, KL 0-1 reference", dummies, None),
            ("kl_stratified", "baseline hazard stratified by the three kl3 strata",
             [], ["kl_stratum"]),
            ("score_only", "no KL term (unadjusted)", [], None),
        ]
        for name, kl_form, base_cov, strata in specs:
            full, full_df = _fit(df, base_cov + ["score_z"], strata=strata)
            if base_cov:
                ref, _ = _fit(df, base_cov, strata=strata)
                ll_ref = float(ref.log_likelihood_)
            else:
                # No covariate left to fit: lifelines' own null log-likelihood, which for a
                # stratified fit is the stratified null.
                ll_ref = float(full._ll_null_ if hasattr(full, "_ll_null_")
                               else full.log_likelihood_ratio_test().test_statistic)
            ll_full = float(full.log_likelihood_)
            chi2 = 2.0 * (ll_full - ll_ref)
            p_lr = float(stats.chi2.sf(max(chi2, 0.0), 1))
            ph = _ph_pvalues(full, full_df, log)
            s = full.summary
            for term in s.index:
                t = str(term)
                rows.append(dict(
                    arm=arm, specification=name, kl_form=kl_form,
                    n_patients=n_pat, n_events=n_ev, term=t,
                    coef=float(s.loc[term, "coef"]), se=float(s.loc[term, "se(coef)"]),
                    hr=float(s.loc[term, "exp(coef)"]),
                    hr_lo=float(s.loc[term, "exp(coef) lower 95%"]),
                    hr_hi=float(s.loc[term, "exp(coef) upper 95%"]),
                    p_wald=float(s.loc[term, "p"]),
                    p_wald_text=_p_text(float(s.loc[term, "p"])),
                    ph_p=float(ph.get(t, float("nan"))),
                    is_score_term=bool(t == "score_z"), score_sd=sd,
                    ll_full=ll_full, ll_reference=ll_ref, lr_chi2=float(chi2),
                    lr_df=1.0, lr_p=p_lr, lr_p_text=_p_text(p_lr),
                    note=_append_note(POSTHOC_NOTE,
                                      "hazard ratio for score_z is per 1 SD of the model risk "
                                      "score (cloglog of the 5-year predicted risk, "
                                      "standardized on this analysis set); the likelihood-ratio "
                                      "test is this model against the same model without "
                                      "score_z, 1 df; the _text p columns carry three "
                                      "significant figures because the numeric ones are "
                                      "rounded to six decimals by the house CSV writer")))
            sc = s.loc["score_z"]
            log.info("A2 cox %-10s %-15s HR/SD %.3f (%.3f to %.3f), LR chi2 %.2f on 1 df, "
                     "p %.3g  [n=%d, %d events]", arm, name, float(sc["exp(coef)"]),
                     float(sc["exp(coef) lower 95%"]), float(sc["exp(coef) upper 95%"]),
                     chi2, p_lr, n_pat, n_ev)
    return pd.DataFrame(rows, columns=KLG_COX_COLUMNS)


def build_klg_auroc(ctx: Context) -> pd.DataFrame:
    """A2(iii). Within-stratum AUROC, with the protocol section 21 floor HONOURED.

    The rule is the one :func:`src.eval_models.build_subgroups` applies, imported from the
    same constant and asserted against config: a stratum with fewer than
    ``SUPPRESS_BELOW_EVENTS`` events gets a NaN estimate, ``suppressed=True`` and a reason.
    It is not weakened here and no bootstrap runs on a suppressed cell.

    98 events cannot leave two strata above a 50-event floor under any partition, so at most
    one cell in each scheme can survive. That is a property of the data, stated plainly
    rather than engineered around.
    """
    thresh = int(ctx.cfg["model_eval"]["suppress_below_events"])
    assert thresh == SUPPRESS_BELOW_EVENTS, (
        f"model_eval.suppress_below_events is {thresh}, but protocol section 21 fixes it at "
        f"{SUPPRESS_BELOW_EVENTS}")
    log = ctx.log
    key = f"{PRIMARY_METRIC}@{ctx.horizon}"
    rows: list[dict] = []
    for arm in A2_ARMS:
        base = analysis_mask(ctx, arm)
        for scheme in KL_SCHEMES:
            for order, label, rule, mask in kl_stratum_masks(ctx.klg, scheme):
                m = mask & base
                n_pat, n_ev = int(m.sum()), int(ctx.roster.event[m].sum())
                est = lo = hi = width = float("nan")
                wide = False
                if n_ev < thresh:
                    suppressed = True
                    reason = (f"protocol section 21: fewer than {thresh} events "
                              f"({n_ev} in this stratum)")
                else:
                    suppressed, reason = False, ""
                    est = float(ctx.engine.point(ctx.arms[arm], m)[key])
                    lo, hi = percentile_ci(ctx.engine.boot(ctx.arms[arm], m)[key])
                    lo, hi = float(lo), float(hi)
                    width = hi - lo
                    wide = bool(width >= WIDE_INTERVAL_WIDTH)
                note = _append_note(POSTHOC_NOTE, f"stratum rule: {rule}")
                if wide:
                    note = _append_note(note, (
                        f"WIDE INTERVAL: the 95% interval spans {width:.3f}, at or above the "
                        f"{WIDE_INTERVAL_WIDTH:g} flag, which is the top of the range spanned "
                        f"by the six published test-split subgroup estimates; this stratum "
                        f"estimate is no more precise than the least precise subgroup already "
                        f"reported and must not be read as a difference from any other cell"))
                rows.append(dict(
                    arm=arm, scheme=scheme, stratum=label, stratum_order=order,
                    n_patients=n_pat, n_events=n_ev, metric=PRIMARY_METRIC,
                    horizon_days=ctx.horizon, estimate=est, ci_lo=lo, ci_hi=hi,
                    ci_width=width, wide_interval=wide, suppressed=bool(suppressed),
                    suppression_reason=reason, note=note))
    df = pd.DataFrame(rows, columns=KLG_AUROC_COLUMNS)
    n_supp = int(df["suppressed"].sum())
    log.info("A2 within-stratum AUROC: %d of %d cells suppressed below the %d-event floor",
             n_supp, len(df), thresh)
    for _, r in df[~df["suppressed"].astype(bool)].iterrows():
        log.info("A2 AUROC %-10s %-11s %-8s n=%3d ev=%3d  %.4f (%.4f to %.4f)%s",
                 r["arm"], r["scheme"], r["stratum"], r["n_patients"], r["n_events"],
                 r["estimate"], r["ci_lo"], r["ci_hi"],
                 "  WIDE" if r["wide_interval"] else "")
    return df


# =========================================================================== #
# 4. ENTRY POINT                                                               #
# =========================================================================== #
def run(cfg: Config, log: logging.Logger, out_dir: Path,
        n_boot: int | None = None) -> dict[str, pd.DataFrame]:
    """Build every table and write it. Returns the frames so tests can assert on them."""
    ctx = Context(cfg, log, n_boot=n_boot)
    tables = {
        "comparisons": (build_posthoc_comparisons(ctx), COMPARISON_COLUMNS),
        "strata": (build_klg_strata(ctx), KLG_STRATA_COLUMNS),
        "tertiles": (build_klg_tertiles(ctx), KLG_TERTILE_COLUMNS),
        "cox": (build_klg_cox(ctx), KLG_COX_COLUMNS),
        "auroc": (build_klg_auroc(ctx), KLG_AUROC_COLUMNS),
    }
    out_dir = Path(out_dir)
    out: dict[str, pd.DataFrame] = {}
    for key, (df, cols) in tables.items():
        name = OUTPUT_BASENAMES[key]
        assert name.startswith("v6_"), "every output of this module is namespaced v6_"
        path = out_dir / name
        write_table(path, df, cols, ctx.roster.pids, name)
        log.info("wrote %s (%d rows)", path, len(df))
        out[key] = df
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="v6 Phase-2 post-hoc analyses A1 (single-view contrasts) and A2 "
                    "(prediction within Kellgren-Lawrence strata).")
    ap.add_argument("--config", default="config/feasibility.yaml")
    ap.add_argument("--out-dir", default="outputs/tables")
    ap.add_argument("--bootstrap-n", type=int, default=None,
                    help="override model_eval.bootstrap_n for a fast smoke run; the "
                         "reported tables always use the protocol value unless this is set")
    args = ap.parse_args(argv)
    log = setup_logging()
    cfg = load_config(args.config)
    log.warning("*** These analyses are POST HOC on the already-read sealed test split "
                "(deviation D35). They are exploratory and are labelled as such in every "
                "row of every table written. ***")
    run(cfg, log, Path(args.out_dir), n_boot=args.bootstrap_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
