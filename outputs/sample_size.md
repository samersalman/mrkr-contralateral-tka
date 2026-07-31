# Sample size: Riley minimum development sample and locked-test-set precision

Generated 2026-07-25 05:00 UTC by `src/sample_size_riley.py` (protocol section 16). Seed 20250720.

Protocol section 16, verbatim: *"For the clinical comparator, calculate the minimum development sample using the Riley time-to-event framework with the final number of candidate parameters, observed event rate, follow-up distribution, and a defensible anticipated Cox-Snell R-squared. Size the locked test set by simulation to obtain acceptable precision for 5-year AUROC and calibration slope... A practical preliminary floor is 500 total primary events and 100 test events, followed by the formal calculation. If precision is inadequate, simplify the model and revise the question before preregistration rather than proceeding with an underpowered deep model."*

## Bottom line

- **Development sample: sufficient.** The Riley requirement at the primary assumption (Nagelkerke R2 = 0.15 of the maximum) is 1,201 patients / 173 events for P = 12 parameters. The development set (train + val) holds 2,968 patients and 427 events, i.e. 2.47x the requirement, 35.6 events per parameter. Even the pessimistic assumption (f = 0.1) needs only 1,832, still met.
- **Protocol section 16's stated numeric floor IS met.** Section 16 names one: "a practical preliminary floor is 500 total primary events and 100 test events". The study has **533 total primary events (>= 500) and 106 test events (>= 100)** — both cleared. Section 16 asks for "acceptable precision" but states no numeric precision target.
- **The locked test set misses the tighter half-width targets adopted for this analysis.** The +/-0.05 AUROC and +/-0.2 calibration-slope half-widths are ANALYST-ADOPTED choices recorded in `config/feasibility.yaml`, not protocol values. At the primary scenario (5-year horizon, true AUROC 0.70) the 741-patient test set gives an AUROC 95% CI half-width of 0.065 and a calibration-slope half-width of 0.291. Meeting both adopted targets would need about 1,570 test patients (42% of the cohort), leaving 2,139 for development, which is still above the pessimistic Riley requirement of 1,832. Because the simulation assumes a normal linear predictor and censoring independent of risk, these half-widths are lower bounds on real-world uncertainty.
- **Widening the imaging window to 3 years is not worth it.** +133 patients and +14 events move the required development sample from 1,201 to 1,207 (already met either way) and shrink the test-set AUROC half-width from 0.065 to only 0.064, at the cost of radiographs up to a year staler relative to the index TKA.
- **Recommendation: PROCEED on model development; SIMPLIFY THE TEST-SET CLAIM (do not widen the imaging window).**

## 1. Observed inputs (nothing here is nominal)

**What was read.** Patient rows come from `derived-data/cohort/features_clinical.parquet` through `src.model_clinical.load_development_frame`, i.e. with the same `split != "test"` predicate pushed into the Parquet reader that the M0 module uses: **2,968 of 3,709 rows are materialised and no sealed row is**. Every quantity in the middle column below is computed from those development rows. The right-hand column is CONTEXT ONLY and is read from the already-published Phase-1 aggregates (`outputs/tables/split_summary.csv + outputs/event_counts.csv`) — it is never recomputed from sealed rows, and only the three counts those files contain are available.

| Quantity | Development set (train + val), computed here | Full cohort (published Phase-1 aggregate) |
|---|---|---|
| Patients | 2,968 | 3,709 |
| 5-year contralateral TKA events | 427 | 533 |
| Event fraction phi = E/n | 0.14387 | 0.14370 |
| Person-years of follow-up | 7,467.2 | not recomputed (test split sealed) |
| Mean follow-up (years) | 2.5159 | not recomputed (test split sealed) |
| Event rate (per person-year) | 0.05718 | not recomputed (test split sealed) |
| Median observed follow-up (days) | 832 | not recomputed (test split sealed) |
| Median follow-up, reverse KM (days) | 1035 | not recomputed (test split sealed) |
| Follow-up IQR (days) | 275 to 1751 | not recomputed (test split sealed) |
| `n_status_determined_5y` — 5-year status DETERMINED (an observed event, or event-free follow-up reaching day 1826) | **1,133** (38.2%) | not recomputed (test split sealed) |
| `n_full_5y_record_coverage` — the `complete_5y` flag: `last_observed >= landmark + 1826` | 746 | not recomputed (test split sealed) |
| `n_followup_reaches_day_1825` — observed follow-up time reaches day 1825 (`time_from_landmark >= horizon`) | 707 | not recomputed (test split sealed) |
| Kaplan-Meier risk at day 1825 | 0.20024 | not recomputed (test split sealed) |
| Greenwood SE of KM survival at day 1825 | 0.00936 | not recomputed (test split sealed) |

### THREE "5-year maturity" counts exist in this project. They are not interchangeable.

All three appear in the table above. Only the first is a maturity statistic for a 5-year risk model; the other two answer different questions and must never be substituted for it. **No computed quantity in this document, or in `outputs/clinical_baseline_report.md`, depends on which one is quoted** — inverse-probability-of-censoring weighting already handles administrative censoring correctly, and these counts appear only in prose.

| name | definition | development (train + val) | full cohort |
|---|---|---|---|
| `n_status_determined_5y` | the 5-year outcome is KNOWN: an observed event (427 patients), or event-free follow-up reaching administrative censoring at day 1826 (706 patients) | **1,133 / 2,968 (38.2%)** | 1,401 / 3,709 (`outputs/feasibility_report.md`) |
| `n_full_5y_record_coverage` | the `complete_5y` flag: `last_observed >= landmark + 1826`. This counts RECORD COVERAGE, not status, so it drops most of the patients whose 5-year status is known precisely because they had the event and then left the record stream | 746 / 2,968 (25.1%) | 916 / 3,709 |
| `n_followup_reaches_day_1825` | `time_from_landmark >= 1825`: observed follow-up time reaching the CLAMPED EVALUATION horizon (one day inside administrative censoring), regardless of how the patient left the risk set | 707 / 2,968 (23.8%) | 869 / 3,709 |

Wherever follow-up maturity is invoked to support or undermine the 5-year horizon — in this report, in `outputs/clinical_baseline_report.md`, in `outputs/feasibility_report.md` and in `notebooks/train_colab.ipynb` — the figure used is `n_status_determined_5y`. An earlier revision argued the 5-year horizon down from 746 / 2,968 (25.1%) in one document and from 1,401 / 3,709 in another, which are two incompatible numbers for one conclusion.

**Using development-only inputs costs about one patient and buys a true statement.** Criterion 1 at the primary assumption needs 1,201 patients from the development event fraction; substituting the published full-cohort event fraction (0.14370 against 0.14387) would make it 1,202. Nothing in this document turns on that difference; what turned on it was whether the module's own claim about its leakage controls was accurate. It previously computed every input over all 3,709 patients, including the 741 sealed ones, while stating in its docstring and in this report that the locked test split is never read.

**Candidate parameters P = 12**, taken (not hard-coded) from `m0_clinical_model.json:identified_parameters`.

The design matrix has **13 columns** — 11 model columns in `derived-data/cohort/clinical_imputation_params.json`, minus the 1 linear age column (`age_at_index_imp`), plus `model_clinical.age_rcs_df` = 3 restricted-cubic-spline basis terms on age — but only **12 of them are IDENTIFIED**, plus `sample_size.extra_image_parameters` = 0. `src/model_clinical.py` computes the identified count by rank (`rank([X | 1]) - 1`, because a Cox partial likelihood has no intercept) and freezes it in `m0_clinical_model.json`; the shortfall is patsy's `cr()` basis, which is a partition of unity, so the three age columns sum to 1 on every row and one direction is a level the likelihood cannot see.

**The identified count is the correct Riley input.** Riley's criteria describe how many parameters the likelihood has to estimate, not how many columns the design matrix happens to have. A previous version of this module asserted P = 15, which was wrong twice over: the design carried an extra aliased indicator (`pain_score_max_missing`, the exact complement of `knee_pain_any_imp`) that has since been removed from the model column list, and the spline partition of unity was never counted at all. Both errors pushed P upward, so **the previous requirement was conservative** — it demanded a larger development sample than the model actually needs, and every 'sufficient' verdict it reached still stands a fortiori.

**P moved again with the M0 correction (2026-07-24).** The dataset-inferred contralateral KLG was removed from M0 and the image-to-index interval was added, restoring protocol Table 7's `M0 = "Age, sex, comorbidities, pain, image-to-index interval"` (Table 6 lists inferred KLG as a **secondary comparator only**, and it now sits in M1). Net effect on the design: `klg_contra_imp` and `klg_contra_missing` out, `days_to_index_imp` in — 13 design columns and **P = 12** where the previous revision had 14 and 13. Fewer parameters lower every Riley requirement, so this direction is again conservative with respect to the earlier 'sufficient' verdicts. The M1 comparator carries one more parameter and is fitted on a smaller (KLG-eligible) subset; it is a secondary comparator and is not what protocol section 16 sizes.

The image model is deliberately not charged extra parameters here: a frozen ConvNeXt-Tiny encoder contributes a learned representation, not free degrees of freedom in the survival head, and the Riley framework has no accepted extension to deep representation learning. That is an assumption, not a result.

**Constant-hazard check.** The exponential approximation implied by the observed rate gives a 5-year risk of 0.24867 versus the observed Kaplan-Meier 0.20024. The hazard is front-loaded (the 1-year KM risk already exceeds what a constant hazard predicts), so criterion 3 is reported both ways below.

## 2. Formulas actually used, with citations

Riley RD, Snell KIE, Ensor J, Burke DL, Harrell FE, Moons KGM, Collins GS. Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes. *Stat Med.* 2019;38(7):1276-1296. Reference implementation: the `pmsampsize` R package, `pmsampsize_surv()`.

```
phi   = E / n                (overall event fraction = rate * mean follow-up)
P     = candidate parameters,  S = target shrinkage,  delta = optimism tolerance

(0)  max R2_CS = 1 - exp( 2 * ( phi*ln(phi) - phi ) )              [Riley 2019 eq. 23]
     R2_CS_adj = f * max R2_CS   ==>   f IS the anticipated Nagelkerke R2

(1)  n1 = P / ( (S  - 1) * ln(1 - R2_CS_adj / S ) )                 criterion 1
(2)  S2 = R2_CS_adj / ( R2_CS_adj + delta * max R2_CS ) = f/(f+delta)
     n2 = P / ( (S2 - 1) * ln(1 - R2_CS_adj / S2) )                 criterion 2
(3)  PT = n * mean_followup;  SE(rate) = sqrt(rate / PT)
     risk(t)       = 1 - exp(-rate * t)
     risk_upper(t) = 1 - exp( -(rate + 1.96*SE(rate)) * t )
     MOE = risk_upper(t) - risk(t) <= MAPE, solved for n:
     n3 = rate * (1.96*t)^2 / ( mean_followup * [ ln(1 - MAPE/exp(-rate*t)) ]^2 )

     n_required = max(n1, n2, n3);  events = n_required * phi;  EPP = events / P
```

Step (3) is where this module goes beyond `pmsampsize_surv()`, which fixes n3 = max(n1, n2) and merely *reports* the resulting risk interval instead of solving for the n that meets the margin. Both are given below; the reported interval at n = max(n1, n2) reproduces the published function exactly.

**Arithmetic verification against a published worked example.** The `pmsampsize` documentation example `pmsampsize(type="s", csrsquared=0.051, parameters=30, rate=0.065, timepoint=2, meanfup=2.07)` gives phi = 0.065*2.07 = 0.13455, max R2_CS = 1 - exp(2*(0.13455*ln(0.13455) - 0.13455)) = 0.5547, Nagelkerke = 0.051/0.5547 = 0.092, n1 = 30/((0.9-1)*ln(1-0.051/0.9)) = 30/0.0058332 = 5143, S2 = 0.051/(0.051+0.05*0.5547) = 0.6477 and n2 = 30/((0.6477-1)*ln(1-0.051/0.6477)) = 30/0.0288856 = 1039. `tests/test_sample_size_riley.py` pins these values.

## 3. Riley minimum development sample (Part 1)

Anticipated Cox-Snell R2 is unknown a priori (no prior model exists for this outcome), so it is expressed as a fraction f of the maximum achievable value **max R2_CS = 0.5707** (computed from the observed event fraction phi = 0.14387). Because Nagelkerke R2 = R2_CS / max R2_CS, f is exactly the anticipated Nagelkerke R2. The headline assumption is f = 0.15.

| f (= Nagelkerke R2) | R2_CS_adj | n1 shrinkage | n2 optimism | n3 risk precision | **n required** | binding | events required | EPP at required n |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.0571 | 1,832 | 403 | 461 | **1,832** | 1_shrinkage | 264 | 22.0 |
| 0.15 **(primary)** | 0.0856 | 1,201 | 397 | 461 | **1,201** | 1_shrinkage | 173 | 14.4 |
| 0.20 | 0.1141 | 885 | 390 | 461 | **885** | 1_shrinkage | 127 | 10.6 |
| 0.30 | 0.1712 | 569 | 377 | 461 | **569** | 1_shrinkage | 82 | 6.8 |

Observed against requirement, at the primary assumption f = 0.15 (required n = 1,201, required events = 173):

| Comparison | n | events | EPP | vs required n |
|---|---|---|---|---|
| Full locked cohort (published Phase-1 aggregate) | 3,709 | 533 | 44.4 | 3.09x |
| Development set actually used to fit (train + val) | 2,968 | 427 | 35.6 | 2.47x |
| Training split alone | 2,597 | 373 | 31.1 | 2.16x |

The development set, not the full cohort, is the honest comparator: the 741 locked test patients are never used to estimate a coefficient.

**Headroom.** With the 2,968 development patients and P = 12, the Riley criteria are satisfied for any anticipated Nagelkerke R2 at or above **f = 0.0625** (0.0357 on the Cox-Snell scale), and the same data would support up to **29 parameters** at the primary f = 0.15 assumption (versus the 12 pre-specified).

### Criterion 3 in detail (the overall-risk margin of error)

- Exponential (pmsampsize) form, solved for n: **n3 = 461**, which is 0.16x the development set.
- At n = max(n1, n2) = 1,201 the published function reports a 5-year risk of 0.2487 (95% CI 0.2160 to 0.2800), margin of error 0.0314 against a target of 0.05.
- Nonparametric alternative, because the hazard is not constant: the observed Kaplan-Meier 5-year risk is 0.2002 with Greenwood SE 0.00936 in 2,968 patients, so the 95% margin of error is already 0.0183. Scaling as 1/sqrt(n), a margin of 0.05 needs only **n = 400**.
- The two routes differ by 61 patients (13% of the larger, roughly 7% on the standard-error scale), which is close agreement for a sample-size calculation and means the constant-hazard idealisation does not change the conclusion: criterion 3 is nowhere near binding. Criterion 1 (shrinkage) drives the requirement at every assumption in the grid.

## 4. Locked test-set precision by simulation (Part 2)

n_test = 741 patients with 106 events (the locked 20% split). 2,000 replicates per scenario, seed 20250720. Horizons are the shared clamped grid (day 365, 730, 1825), identical to `derived-data/cohort/m0_clinical_model.json`.

**Targets adopted for this calculation** (`config/feasibility.yaml`, `sample_size.test_precision_simulation`; protocol section 16 requires "acceptable precision" but specifies **no numeric target**): AUROC 95% CI half-width <= 0.05, calibration slope 95% CI half-width <= 0.2. These are analyst choices and must never be described as protocol values. The **numeric floor section 16 does state — 500 total primary events and 100 test events — IS met (533 and 106).**

The assumed true discrimination grid (0.65, 0.7, 0.75, 0.8) and the primary value 0.7 also come from config (`test_precision_simulation.true_auroc_grid` / `.true_auroc_primary`), so the sensitivity of the headline conclusion to that assumption is visible in the configuration rather than buried in code.

**IPCW weight floor.** Case weights are `1 / max(G(T-), 0.001)`, i.e. capped at 1,000. That cap is a guard against a censoring curve that touches zero inside a replicate, not an operating parameter: the smallest censoring-survival value a case weight can meet at each horizon in these data is G(365-) = 0.769 (max weight 1.3), G(730-) = 0.614 (max weight 1.6), G(1825-) = 0.298 (max weight 3.4), every one of them three orders of magnitude clear of the cap. Raising the floor toward 1 would silently switch IPCW off, so a unit test pins it.

**Assumed data-generating process** (see the module docstring for the full statement):

1. Linear predictor LP ~ N(0, sigma^2); the model is correctly specified, so the true calibration slope is exactly 1.
2. Proportional hazards S(u | LP) = exp(-H0(u) exp(LP)), with the baseline cumulative hazard H0 calibrated at every observed Kaplan-Meier time so the simulated marginal survival reproduces the real, front-loaded event-time curve.
3. sigma is solved numerically so the TRUE cumulative/dynamic AUROC at each horizon equals the assumed value.
4. Censoring times are inverse-sampled from the observed reverse-KM censoring distribution (which carries the administrative atom at day 1826), independent of the event time. The event and censoring distributions are estimated on train + val only; the locked test split is never read.
5. Estimands: IPCW cumulative/dynamic AUROC at the horizon (Uno), and the Cox calibration slope of the outcome on LP with follow-up truncated at the horizon.
6. Reported precision is 1.96 x the Monte-Carlo SD across replicates, computed over the replicates that yielded a finite estimate: at the primary scenario 2,000 of 2,000 for the AUROC and 2,000 of 2,000 for the slope (0 Cox fits failed to converge and are reported as invalid rather than returned as an unconverged last iterate).

> **These assumptions flatter precision, and the direction matters.** The linear predictor is exactly normal and the censoring is independent of risk. Real linear predictors are skewed and heavier-tailed than a Gaussian, which spreads case and control scores less evenly and widens the AUROC interval; and if loss to follow-up is risk-related — plausible here, since disengagement from a health system is not random with respect to arthritis burden — the IPCW weights are misspecified and the true interval is wider still than the marginal-censoring simulation shows. Every half-width below is therefore best read as a **lower bound** on the uncertainty a real validation will face. That makes the shortfall against the adopted AUROC target a conservative statement, not an alarmist one.

Solved LP standard deviations: 1 y / AUROC 0.65: sigma = 0.521, 1 y / AUROC 0.70: sigma = 0.713, 1 y / AUROC 0.75: sigma = 0.927, 1 y / AUROC 0.80: sigma = 1.178, 2 y / AUROC 0.65: sigma = 0.511, 2 y / AUROC 0.70: sigma = 0.700, 2 y / AUROC 0.75: sigma = 0.914, 2 y / AUROC 0.80: sigma = 1.167, 5 y / AUROC 0.65: sigma = 0.494, 5 y / AUROC 0.70: sigma = 0.679, 5 y / AUROC 0.75: sigma = 0.890, 5 y / AUROC 0.80: sigma = 1.144

| Horizon | True AUROC | Cases | Controls | AUROC half-width | target met | Slope half-width | target met | n_test needed for AUROC | n_test needed for slope |
|---|---|---|---|---|---|---|---|---|---|
| 1 y | 0.65 | 65 | 514 | 0.0714 | NO | 0.4846 | NO | 1,511 | 4,351 |
| 1 y | 0.70 | 65 | 514 | 0.0682 | NO | 0.3551 | NO | 1,377 | 2,336 |
| 1 y | 0.75 | 65 | 514 | 0.0632 | NO | 0.2756 | NO | 1,184 | 1,407 |
| 1 y | 0.80 | 65 | 514 | 0.0578 | NO | 0.2263 | NO | 989 | 949 |
| 2 y | 0.65 | 86 | 392 | 0.0642 | NO | 0.4263 | NO | 1,222 | 3,368 |
| 2 y | 0.70 | 86 | 392 | 0.0608 | NO | 0.3133 | NO | 1,096 | 1,819 |
| 2 y | 0.75 | 85 | 392 | 0.0567 | NO | 0.2464 | NO | 954 | 1,125 |
| 2 y | 0.80 | 85 | 392 | 0.0535 | NO | 0.2065 | NO | 848 | 791 |
| 5 y | 0.65 | 107 | 177 | 0.0668 | NO | 0.3810 | NO | 1,323 | 2,689 |
| 5 y | 0.70 | 107 | 176 | 0.0652 | NO | 0.2911 | NO | 1,262 | 1,570 |
| 5 y | 0.75 | 107 | 176 | 0.0614 | NO | 0.2292 | NO | 1,117 | 974 |
| 5 y | 0.80 | 107 | 176 | 0.0551 | NO | 0.1909 | yes | 899 | 741 |

Cross-checks that the simulator is doing what it claims:

- The mean simulated calibration slope is 0.9999 at the primary scenario; the true value is 1 by construction, so the estimator is unbiased.
- The mean simulated AUROC is 0.6990 against a true 0.70, so the IPCW estimator recovers the value the data-generating process was solved for.
- The Monte-Carlo slope half-width (0.2911) and the mean within-replicate Wald half-width (0.2900) agree, so the model-based standard error is trustworthy in a real single validation.
- The Monte-Carlo AUROC half-width (0.0652) is close to the analytic Hanley-McNeil value for the realised 107 cases and 176 controls (0.0647), an independent formula.
- Re-simulating from scratch at the implied n_test = 1,570 (independent seed) returns an AUROC half-width of 0.0451 (target 0.05) and a slope half-width of 0.2031 (target 0.2), both within Monte-Carlo error of the targets. That confirms the 1/sqrt(n) scaling used to derive every 'n_test needed' figure in the table.

What the table says, in words:

- The number of usable cases and controls, not the number of patients, drives precision. At 5 years the test set contributes about 107 cases and 176 controls known event-free; the roughly 458 patients censored before the horizon enter only through the IPCW weights.
- The two horizons trade off in opposite directions, so neither is uniformly better-powered. At true AUROC 0.70 the 2-year AUROC half-width (0.0608) is NARROWER than the 5-year one (0.0652): 2 years gives up 21 cases but gains 215 controls known event-free. The calibration slope goes the other way (0.3133 at 2 years versus 0.2911 at 5 years), because truncating follow-up at 2 years discards the later events that identify the slope. The 2-year co-primary earns its place on clinical relevance and lighter censoring adjustment, not on a precision advantage.
- The ADOPTED AUROC target of +/-0.05 (analyst choice; protocol section 16 sets no numeric target) is missed in EVERY cell of the grid (half-widths 0.0535 to 0.0714 across all horizons and true AUROC 0.65 to 0.80). Discrimination that is genuinely higher helps only weakly: at 5 years the half-width falls from 0.0668 to 0.0551 across the whole AUROC range.
- The calibration-slope target of +/-0.2 depends strongly on how well the model discriminates. At 5 years it is met only from true AUROC 0.80 upward (half-width 0.1909), and missed below it (up to 0.3810 at true AUROC 0.65). Slope precision scales as 1 / (sigma * sqrt(events)), and a weakly discriminating model has a narrow linear predictor, hence a poorly identified slope.
- A useful reframing: with 741 test patients the achievable 5-year AUROC precision is about +/-0.065, i.e. a 95% CI of roughly 0.635 to 0.765 around a true 0.70. That is enough to show the model beats chance and to compare it with the clinical baseline in a paired analysis, but not enough to certify a specific AUROC to two decimal places.

## 5. Widen the pre-index imaging window from 2 to 3 years? (Part 3)

The re-gate grid records that `recovery_any / 3-year pre-index imaging window` would yield 3,842 patients and 547 events, versus 3,709 / 533 now: **+133 patients (+3.6%) and +14 events (+2.6%)**.

| Metric | 2-year window (current) | 3-year window | Change |
|---|---|---|---|
| Patients | 3,709 | 3,842 | +133 |
| 5-year events | 533 | 547 | +14 |
| Event fraction phi | 0.14370 | 0.14237 | -0.00133 |
| max R2_CS | 0.5707 | 0.5682 | -0.0025 |
| Riley required n (f = 0.15) | 1,201 | 1,207 | +6 |
| Development set (same 80/20 split) | 2,968 | 3,074 | +106 |
| Test set (same 20%) | 741 | 768 | +27 |
| Test AUROC half-width at 5 y, true AUROC 0.70 | 0.0652 | 0.0641 | -0.0012 |
| Test calibration-slope half-width | 0.2911 | 0.2859 | -0.0052 |

- The development sample is not the constraint, so extra patients buy nothing there: 2,968 already exceeds the 1,201 required, and the widened cohort would require 1,207.
- The test set is the constraint, and 3.6% more patients shrink the AUROC half-width by 0.0012 (1.8%), from 0.0652 to 0.0641 against a target of 0.05. Reaching that target by growing the cohort while keeping a 20% test fraction would need roughly 6,310 patients (1.7x the current cohort), which no widening of the imaging window can deliver.
- The cost is not neutral: a 3-year pre-index window admits radiographs up to a year staler relative to the index TKA, so the exposure is measured further from the prediction origin and the contralateral knee has had more unobserved time to progress. That directly attacks the study's central claim.
- Verdict: do NOT widen. The +14 events change the required development sample by +6 patients (both already met by a factor of about 2.5) and do not flip a single decision in this document.

## 6. Recommendation

### PROCEED on model development; SIMPLIFY THE TEST-SET CLAIM (do not widen the imaging window)

- **Proceed with model development.** 2,968 development patients and 427 events against a Riley requirement of 1,201 / 173 at P = 12. Do not add parameters casually: the same data supports at most about 29 at the primary R2 assumption, and the pre-specified block of 12 should stay frozen with no univariable screening (protocol section 19).
- **Do not widen the pre-index imaging window.** +133 patients and +14 events change nothing quantitative and cost image recency.
- **Revise what the locked test set is asked to prove.** Protocol section 16's stated preliminary floor — 500 total primary events and 100 test events — IS met (533 and 106); nothing below contradicts that. What is not met is the stricter +/-0.05 AUROC half-width **adopted for this analysis** (`config/feasibility.yaml`), and section 16 states no numeric precision target of its own. With 741 patients the 5-year AUROC is estimable to about +/-0.065 and the calibration slope to about +/-0.291 at a true AUROC of 0.70. Section 16's decision rule still applies to the gap between ambition and precision: simplify the model and revise the question rather than proceed underpowered — which is what (a) and (c) below do.
- **Concretely, one of these three, decided before the test set is unsealed:**
  - (a) Re-specify the primary test-set estimand as a PAIRED comparison against the clinical baseline M0 (difference in AUROC, difference in the index of prediction accuracy) rather than an absolute AUROC. Paired differences share patients and are far more precisely estimated than either absolute value, so a 741-patient test set can support a difference claim it cannot support for a level claim.
  - (b) Increase the test fraction, but note the ceiling. Meeting BOTH targets needs about 1,570 test patients (42% of the cohort), which leaves only 2,139 for development. That still clears the primary Riley requirement (1,201 at f = 0.15) and also clears the pessimistic one (1,832 at f = 0.1). The largest test set that keeps the pessimistic development requirement intact is 1,877 patients (51%), which would give an AUROC half-width of 0.041 (meets the 0.05 target) and a slope half-width of 0.183 (meets the 0.2 target); tolerating only the primary R2 assumption would raise that ceiling to 2,508. So a re-split can buy both adopted targets at once without betting on an optimistic R2 — but it also means re-drawing the LOCKED splits (protocol section 17), which is an investigator decision, not a script's. Nothing in this module changes them, and (a) reaches the same scientific end without touching them.
  - (c) Keep the splits and pre-specify the precision honestly in the protocol and the manuscript: report the CI and state up front that the study is powered to demonstrate discrimination better than chance and better than the clinical baseline, not to certify a point estimate.
- **My recommendation, if only one is chosen: (a) plus (c).** The splits are already locked and used by a sibling module; the scientific question that matters is whether radiographs add anything over clinical variables, which is a paired difference, and a paired difference is exactly the estimand the available test set can support.

## 7. What is solid, and what is interpretation

**High confidence (verified against the reference implementation).**
- Criteria 1 and 2 and the max R2_CS formula. These reproduce `pmsampsize_surv()` line for line; the published worked example (5143 / 1039 / 0.555 / 0.092) is recomputed exactly in the unit tests.
- The observed inputs. Event rate, person-time, follow-up quantiles, Kaplan-Meier risk and the reverse-KM censoring curve are all computed from `derived-data/cohort/features_clinical.parquet`, and the reverse-KM helper is the same `src.followup.reverse_km` used in the feasibility report.
- The univariate Cox solver and the IPCW AUROC. The Cox solver agrees with lifelines to 6 decimal places on untied data (unit test), and the simulated AUROC recovers the value the data-generating process was solved for.
- The conclusion that the development sample is sufficient. It holds by a factor of roughly two under every assumption in the grid, so it is not sensitive to the R2 choice.

**Interpretation, stated so a reviewer can disagree.**
- The anticipated Cox-Snell R2. There is no prior model for this outcome, so the fraction-of-maximum device is a convention, not evidence. It is presented across a grid for exactly that reason.
- Criterion 3 solved for n. `pmsampsize_surv()` does not solve criterion 3 for survival outcomes; it fixes n = max(n1, n2) and reports the interval. The closed form here is my inversion of the same expression. It is cross-checked against a Greenwood-based calculation on the observed Kaplan-Meier curve, and the two agree, but the constant-hazard assumption behind the published form is visibly violated in this cohort (exponential 5-year risk 0.249 vs observed 0.200).
- Charging the image model zero extra parameters. This follows the config (`extra_image_parameters: 0`) and is defensible for a frozen encoder with a small survival head, but no version of the Riley framework covers deep representation learning. The clinical-baseline requirement should not be read as a sample-size justification for the image model.
- The simulation's proportional-hazards, normal-linear-predictor, independent-censoring data-generating process. It is standard and it reproduces the observed marginal survival and censoring curves, but a real model's linear predictor will not be exactly normal and censoring may depend on covariates.
- The discrimination grid (0.65, 0.7, 0.75, 0.8) and the primary value 0.7. These now live in `config/feasibility.yaml` (`sample_size.test_precision_simulation.true_auroc_grid` / `.true_auroc_primary`) rather than in code, so the assumption is visible and auditable — but it is still an assumption: this is the plausible range for a radiographic progression model, not a measurement.
- The +/-0.05 AUROC and +/-0.2 calibration-slope half-widths. **These are analyst-adopted, not protocol values.** Protocol section 16 requires "acceptable precision" without defining it numerically; its only numeric floor is 500 total primary events and 100 test events, and that floor is met (533 and 106). A reviewer who considers +/-0.07 acceptable for a first-in-domain model would read this document's precision verdict differently, and the config now says so in a comment.

**Not verifiable here.**
- Whether the model will in fact reach any particular AUROC. Everything in section 4 is conditional on an assumed true discrimination.
- Whether the laterality QA audit will hold. The phase-1 PROCEED was contingent on it, and a laterality error rate above a few percent would degrade both the labels and the images in ways no sample-size calculation can offset.
- Whether an investigator will accept re-drawing the locked splits. That is a protocol amendment decision.

Machine-readable results: `outputs/tables/sample_size_riley.csv`.
