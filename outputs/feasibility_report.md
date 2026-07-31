# MRKR Contralateral TKA: Phase-1 Metadata Feasibility Report

**Study:** Multi-View Radiographic Prediction of Contralateral Knee Arthroplasty After Unilateral Total Knee Arthroplasty (invited *Journal of Imaging* submission).
**Phase:** 1, metadata-only feasibility (LOCKED extraction under the author-approved definition).
**Date:** 2026-07-21. **Boundary:** no DICOMs opened, no models trained, no performance metrics computed. All figures are PRELIMINARY feasibility descriptives, not a formal sample-size calculation.

---

## 1. Executive summary

The feasibility gate is **passed under the author-approved primary definition**, which the metadata forced: no coded-laterality-only cohort can reach the protocol's 500-event floor (the strict cohort tops out at 420 events even with lifetime imaging), because 61% of CPT 27447 records carry no laterality modifier. The approved primary cohort therefore combines the strict laterality-coded index with a **high-concordance side-recovered index** (recovering the operative side from same-day imaging and diagnosis signals) and widens the pre-index contralateral-imaging window from 1 year to **2 years**.

**Primary landmark cohort: 3,709 patients; 533 primary 5-year contralateral-TKA events (14.4%); ~107 test-allocatable events.** Both preliminary floors (500 events, 100 test events) are met. The cohort has strong subgroup power, including self-reported Black patients (156 events). The single material exposure to manage is that **49.3% of index sides are inferred (side-recovered), not coded**; the protocol's mandated laterality QA audit (>=200 patients) must validate this before the cohort is locked for model training.

**Recommendation: PROCEED** to the formal Riley development-sample and test-precision calculations, OSF preregistration, and image transfer, **contingent on the laterality QA audit**. Retain the strict coded-laterality cohort (357 to 420 events) as the high-specificity sensitivity analysis.

## 2. Data inventory

Seven de-identified source tables (all keyed on `empi_anon`; 83,011 patients), converted once to typed Parquet and reconciled row-for-row:

| Table | Rows | Note |
|---|---|---|
| demographics | 83,011 | 1 row/patient; sex, race, ethnicity |
| CPT | 6,216,190 | includes 14,076 CPT-27447 records / 8,525 patients |
| ICD | 21,956,056 | 9 curated per-line comorbidity/infection flags |
| image metadata | 503,261 | 169,004 studies; model-inferred laterality/view/WB/arthroplasty/KLG |
| pain | 4,970,869 | knee_pain, pain_score, laterality (95% null), pain_location (75% null) |
| CPT dictionary | 7,166 | code to description |
| ICD dictionary | 25,209 | code to name |

Full detail: `outputs/data_inventory.csv`, `schema_report.md`, `missingness_report.csv`.

## 3. Protocol-to-dataset mapping

The verified concept-to-column mapping is in `outputs/protocol_to_column_mapping.csv` (20 rows). Key locks: index = `cpt_code`='27447' with side from `cpt_group_modifier`; image side = `laterality` (R/L/B/-1); view = `view_position` (F/L/S/I/E); study grouping = `StudyInstanceUID_anon`; image date = `StudyDate_anon`; all dates `YYYY-MM-DD`, per-patient-shifted (within-patient intervals valid). All image-derived fields are model-inferred, not chart ground truth.

## 4. Cohort construction

- **Index:** each patient's earliest CPT 27447. Side is taken from the coded RT/LT modifier where present; for a blank-modifier earliest procedure it is **recovered** from concordant, no-conflict, same-day-or-earlier signals (same-day image laterality, same-day StudyDescription text, on-or-before ICD-10 M17 laterality). Contralateral side = opposite of the index side.
- **Primary strategy (approved):** `recovery_any` = strict coded index UNION any single-concordant-signal recovered index; 2-year (730-day) pre-index contralateral-imaging window.
- **Index-level exclusions:** age <40 at index; prior contralateral knee arthroplasty (prior contralateral-side knee-arthroplasty CPT or a pre-index contralateral prosthesis on imaging); knee osteomyelitis in the 365 days before index (high-specificity definition).
- **Landmark:** day 90 after index; events begin day 91; horizon 5 years.
- **No leakage:** independent review confirmed every predictor and eligibility field uses only at-or-before-index data; side recovery is same-day-or-earlier; the outcome, the day-0-to-90 exclusion, observation, and censoring are the only post-index uses and are never predictors.

## 5. Cohort flow

Sequential, non-overlapping exclusions (`outputs/cohort_flow.csv`); primary = recovery_any/730d, strict = coded/365d sensitivity:

| Step | Description | Primary | Strict |
|---|---|---|---|
| 1-2 | Demographics / with a knee radiograph | 83,011 | 83,011 |
| 3 | With CPT 27447 | 8,525 | 8,525 |
| 4 | Aged >=40 at index | 8,417 | 8,417 |
| 5 | Interpretable index side | 7,112 | 4,203 |
| 6 | No prior contralateral arthroplasty | 6,393 | 3,756 |
| 7 | No infection/osteomyelitis (high-spec) | 6,381 | 3,752 |
| 8 | Eligible pre-index contralateral study | 3,981 | 1,807 |
| 9 | No contralateral TKA through day 90 | 3,940 | 1,775 |
| 10 | Observed through day 90 | 3,709 | 1,664 |
| 11 | **Final landmark cohort** | **3,709** | **1,664** |

The pre-index imaging requirement (step 8) is the largest single exclusion (2,400 patients in the primary arm).

## 6. Outcome counts

Primary event = first contralateral (opposite-side), laterality-coded CPT 27447 after day 90 and within 5 years of index (the model-inferred arthroplasty field does not define the endpoint).

| Horizon (from day 91) | Events | % of cohort |
|---|---|---|
| 1 year | 296 | 8.0% |
| 2 years | 398 | 10.7% |
| **5 years (primary)** | **533** | **14.4%** |

Test-allocatable at a 20% split: ~107. Pre-specified labeled sensitivity endpoints are computed (`outputs/tables/outcome_counts_detail.csv`): a contralateral composite adding unicompartmental arthroplasty (27446), and a high-specificity augmented endpoint requiring a post-procedure prosthesis image within 180 days (the augmented endpoint under-counts near the horizon because image coverage ends earlier than CPT coverage; documented).

## 7. Follow-up and censoring

Follow-up is measured from the day-90 landmark. Last-observed date is the maximum date across CPT, ICD, pain, and image records (death is unavailable, so mortality is an unmeasured competing event; patients are censored at last observation).

- Median follow-up from landmark: **831 days** (IQR 274 to 1,740).
- Patients whose **5-year status is DETERMINED** (`n_status_determined_5y`): **1,401** = 533 observed events + 868 patients censored administratively at day 1,826; censored before 1/2/5 years: 826 / 1,321 / 2,308.
- Image-to-index interval of the selected study: median 97 days (IQR 50 to 221), all within the 2-year window.

**Naming note (corrected 2026-07-24, deviation D19).** An earlier revision of this line called the 1,401 figure "patients with complete 5-year observation", which is a different quantity. Three counts exist and they are not interchangeable:

| name | definition | full cohort | development (train + val) |
|---|---|---|---|
| `n_status_determined_5y` | the 5-year outcome is KNOWN: an observed event, or event-free follow-up reaching day 1,826 | **1,401 / 3,709 (37.8%)** | **1,133 / 2,968 (38.2%)** |
| `n_full_5y_record_coverage` | the `complete_5y` flag: `last_observed >= landmark + 1826`. RECORD COVERAGE, not status — it excludes 485 of the 533 event patients, whose status is known precisely because they had the event and then left the record stream | 916 / 3,709 (24.7%) | 746 / 2,968 (25.1%) |
| `n_followup_reaches_day_1825` | `time_from_landmark >= 1825`: observed follow-up reaching the clamped evaluation horizon | 869 / 3,709 (23.4%) | 707 / 2,968 (23.8%) |

Wherever follow-up maturity is invoked to support or undermine the 5-year horizon, the figure used is `n_status_determined_5y`. `outputs/sample_size.md` reconciles all three. No computed metric depends on the choice: inverse-probability-of-censoring weighting already handles administrative censoring correctly.

**Maturity caveat:** RT/LT laterality modifiers were adopted more consistently in later calendar years, so the coded-laterality (strict) patients skew recent and few reach 5-year maturity; the side-recovered patients skew earlier and improve maturity. This is a further reason the recovery-augmented primary is preferable, and it argues for inverse-probability-of-censoring-weighted horizon evaluation and consideration of a 2-year co-primary horizon.

## 8. Imaging availability

The image-transfer manifest (`outputs/tables/image_transfer_manifest.csv`, review only, no transfer initiated) covers the final cohort's selected studies:

- 3,709 patients, 3,709 selected studies, 6,122 images; **100% of image paths present and well formed**.
- Weight-bearing frontal available for ~90% of selected studies; view mix frontal 4,269, lateral 1,659, sunrise 162, other 32.
- Multi-view studies show a much higher event rate than frontal-only (21.9% vs 8.3%), consistent with more imaging being obtained for more symptomatic knees.

## 9. Subgroup feasibility

Primary-event counts by subgroup (`outputs/subgroup_counts.csv`; protocol section 21 stability flags):

| Subgroup | Patients | Events | Flag |
|---|---|---|---|
| Female / Male | 2,323 / 1,386 | 342 / 191 | >=100 |
| Age <65 / >=65 | 1,554 / 2,155 | 228 / 305 | >=100 |
| Black / White | 1,247 / 2,129 | 156 / 328 | >=100 |
| Asian | 121 | 13 | **<50** |
| Obesity yes / no | 2,103 / 1,606 | 351 / 182 | >=100 |
| Weight-bearing / non-WB | 3,337 / 372 | 493 / 40 | >=100 / **<50** |
| Multi-view / frontal-only | 1,661 / 2,048 | 364 / 169 | >=100 |

Every major stratum, including self-reported Black patients, is analyzable. Asian race and non-weight-bearing imaging are below 50 events and unstable; estimates in those strata must be suppressed or shown with wide intervals.

## 10. Data-quality concerns

- **61% blank CPT-27447 modifier** is the defining constraint; it drove the need for side recovery.
- **Model-inferred image metadata** (laterality, view, weight-bearing, arthroplasty, KLG) are strong priors, not ground truth; KLG is structurally missing on ~82% of images.
- **ICD-10 `--` sentinel on 34% of rows** limits the raw ICD-10 join and the M17 side-recovery signal; curated flags (aggregated per patient with MAX) are the reliable comorbidity path.
- **Pain records are 24% exact duplicates** and pain laterality is ~95% missing; pain remains a secondary predictor.

## 11. Unresolved ambiguities

- **Recovered index laterality (49.3% of the primary cohort)** is the central item; it must be validated by the protocol's >=200-patient laterality QA audit before the cohort is locked. Only the index side is recovered; predictor imaging and the coded outcome remain gold-standard.
- The augmented outcome's radiographic confirmation is incomplete near the horizon (image coverage ends before CPT coverage).
- The event horizon (5 years from index) and censoring horizon (5 years from landmark) differ by the 90-day landmark, per protocol; handled explicitly.

## 12. Protocol deviations

Full log in `outputs/protocol_deviations.md`. Summary: pre-index imaging window widened from 1 year to 2 years; a labeled side-recovery arm adopted as the primary cohort with the strict coded arm retained as sensitivity (both author-approved, Decision E); high-specificity infection definition (Decision B); observation-through-day-90 via last-observed date (Decision C); same-day blank companion lines treated as same-side billing artifacts (Decision A); Python 3.14 environment substituted for the preliminary 3.12 plan (package availability).

## 13. Recommendation

**PROCEED to formal sample-size and precision calculations, then registration and image transfer, contingent on resolving the specified item.** Concretely:

1. Run the protocol section-7 laterality QA audit (>=200 patients) to validate the recovered index sides; if agreement is inadequate, fall back to `recovery_confirmed` (>=2 signals) or the strict arm with a revised horizon.
2. Perform the Riley time-to-event development-sample and simulation-based test-precision calculations using 533 primary events and the observed follow-up distribution; if precision is inadequate for a deep multimodal model, adopt the 2-year co-primary horizon or simplify the model before preregistration.
3. Preregister on OSF (protocol + SAP), freeze parsing/landmark/horizon/side-recovery rules, and generate patient-level locked splits.
4. Begin Nightingale/Globus access and institution-approved DICOM storage in parallel (the long pole); transfer only the reviewed manifest.

Retain the strict coded-laterality cohort as the pre-specified high-specificity sensitivity analysis throughout.

---
*Machine summary: `outputs/feasibility_summary.json`. Reproduce end to end: `python3 -m src.run_feasibility --config config/feasibility.yaml --stages all`.*
