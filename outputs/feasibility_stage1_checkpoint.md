# MRKR Contralateral TKA: Stage-1 Feasibility Checkpoint Memo

**Study:** Multi-View Radiographic Prediction of Contralateral Knee Arthroplasty After Unilateral TKA (invited *Journal of Imaging* submission).
**Phase:** 1, metadata-only feasibility. **Status:** hard checkpoint, awaiting sign-off.
**All figures below are PRELIMINARY, bounded, read-only characterizations, not the locked cohort extraction.** No DICOMs were opened; no models trained; no performance metrics computed.

---

## 0. Why you are reading this

The protocol mandates a metadata-only feasibility gate before any image transfer, IRB/OSF registration, or model work. This memo quantifies the cohort, events, imaging availability, and subgroup counts from the seven MRKR CSVs, and surfaces the **exact protocol decisions that require your sign-off before the full cohort extraction runs**. Nothing downstream has been executed. The bottom of this memo lists the decisions (A to E) and a recommendation.

**Headline:** Under the protocol as literally written (strict laterality-defined index, pre-index contralateral imaging within 1 to 365 days), the cohort yields **357 primary 5-year contralateral-TKA events, which is below the protocol's preliminary floor of 500** (and ~71 test-allocatable events, below the floor of 100). However, the events themselves exist in the data; the binding constraint is the pre-index imaging window, and two quantified levers can plausibly close the gap. This is a **revise-and-re-gate** situation, not a hard no-go.

---

## 1. CPT 27447 modifier distribution and raw formatting

CPT 27447 (total knee arthroplasty) appears in **14,076 records across 8,525 patients**. The laterality modifier `cpt_group_modifier` is the single biggest threat to feasibility:

| Modifier (raw) | Records | Parsed side / quality |
|---|---|---|
| blank / NULL | 8,613 (61.2%) | unknown / missing (6,952 patients) |
| `RT` | 2,799 | R / single (2,786 patients) |
| `LT` | 2,641 | L / single (2,637 patients) |
| `50` | 11 | bilateral (11 patients) |
| multi-token (`RT XP`, `74 LT`, `LT XU`, `59 RT`, `22 LT`, `LT XP`, `LT XE`, `73 LT`) | 11 | single side, recoverable |
| `22` | 1 | uninterpretable |

**Raw formatting notes:** modifiers are whitespace-delimited multi-token strings mixing laterality (`RT`/`LT`) with billing modifiers (`XP`, `XU`, `59`, `74`, `22`, `73`); case is as stored; no record carries both `RT` and `LT` (no in-record conflict). The parser normalizes case and whitespace and tokenizes on spaces. **61.2% of index-procedure records carry no usable side.** Distinct patients with at least one single-side 27447: **4,730**. (Table: `outputs/tables/stage1_modifier_distribution.csv`.)

## 2. Is operative side recoverable from documented pre-index fields, and how often?

Among the **4,300 patients whose earliest 27447 is blank**, a concordant pre-index or same-day side signal exists surprisingly often:

- any signal present: **3,744 (87.1%)**
- same-day image `laterality` (R/L): 3,370
- image `StudyDescription` text ("right"/"left"): 3,333
- ICD-10 M17 laterality digit (M17.11/.12/.31/.32): 2,630
- **resolves to a single concordant side: 2,983 (69.4%)**

**Caveats:** image-derived signals are model-inferred (not chart-verified); the ICD-10 signal is unavailable on the 34% of ICD rows carrying the `--` no-code sentinel. Side recovery is therefore a **labeled option confined to a permissive arm, never silently applied, and never from post-index or outcome data.** (Table: `outputs/tables/stage1_side_recovery.csv`.)

## 3. Handling the 61% unsided records: strict vs permissive (with counts)

| Strategy | Definition | Patients (index) |
|---|---|---|
| **Strict (candidate primary)** | earliest-in-time 27447 is itself single-side RT/LT | **4,222** (age >=40: 4,203) |
| **Permissive (cleaner-index sensitivity)** | earliest laterality-coded 27447 as index, only where no evidence of earlier arthroplasty | **3,740** (age >=40: 3,721) |
| Side-recovery (expansion option) | recover index side for blank-modifier patients with a concordant pre-index signal | up to **+2,983** eligible patients |

Two points that matter for interpretation:

1. **Permissive is smaller, not larger** (3,740 vs 4,222; delta -482). Applying the no-prior-arthroplasty screen at index definition correctly removes staged-bilateral and revision patients whose earliest laterality-coded 27447 is actually a second-side procedure. Permissive is a **cleaner-index arm, not a cohort-expansion arm.**
2. **The expansion lever is side recovery, not the permissive strategy.** Recovering side for the ~2,983 concordant blank-index patients is the largest single lever on cohort size and, because it lets these patients contribute contralateral events, it also raises the event ceiling (see section 8). (Table: `outputs/tables/stage1_cohort_strategies.csv`.)

## 4. Can an earliest side-coded 27447 be the index when earlier unsided 27447 records exist?

**No, under the strict rule, and this is deliberate.** If a patient's first-in-time 27447 is unsided/`50`/conflicting, an earlier unsided TKA could be the true first (possibly contralateral) procedure and would invert the index/contralateral assignment. **508 patients** have a single-side 27447 but an *earlier* unsided 27447; strict excludes them (4,730 single-side patients minus 508 = 4,222 strict index).

**Decision A (same-day companion-line assumption) requires your sign-off.** "Earliest single-side" is resolved at the date level, and independent review confirmed this is provably equivalent to a row-level reading for these data (**0 of the 4,222 strict patients change index side or date** under either reading, so date-vs-row is not itself a decision). The residual assumption is different and does need confirming: **2,604 strict patients have an earliest 27447 date consisting of a single RT or LT plus one or more same-day blank companion billing lines** (persisted under `decision_anchors`; 2,597 under an exact-token reading of RT/LT), and those blanks are treated as same-side billing artifacts of the one procedure, not as a separate same-day contralateral TKA. No patient has RT and LT on the same earliest date, and the downstream gates (S6 prior-arthroplasty, S8 imaging, S9 day-0-to-90 contralateral) would catch a genuine same-day bilateral, so this is reasonable, but it should be confirmed rather than assumed.

## 5. Infection / osteomyelitis exclusion: exact fields and estimated pre-index exclusions

Exclusion uses **curated ICD flags aggregated per patient with MAX over the 365 days before index** (the flags are per-diagnosis-line, so a row-level read is wrong). Two clearly labeled candidate definitions:

| Definition | Fields | Excluded (of 4,222 strict) |
|---|---|---|
| **High-specificity** | `knee_osteomyelitis` = 1 (knee-specific) | **6 (0.14%)** |
| **Sensitivity** | `knee_osteomyelitis` OR `joint_infection` = 1 | **10 (0.24%)** |

`joint_infection` is not knee-specific and is therefore the less specific option. Corroborating knee-region osteomyelitis ICD-10 codes verified in the dictionary: **M86.05x (femur), M86.06x (tibia and fibula)**. The dataset paper does not enumerate the exact ICD-to-curated-flag mapping, so the curated flags (not the raw ICD-10 to name join, which is crippled by the `--` sentinel) are the reliable path. **Either definition removes a negligible number of patients; this exclusion is not a feasibility constraint.** The code list is saved. (Table: `outputs/tables/stage1_infection_defs.csv`.)

## 6. Definition of "observation through day 90"

- **Primary rule:** last-observed date (maximum date across CPT, ICD, pain, and image `StudyDate`) is later than index + 90 days. This is valid because MRKR shifts every date field by a per-patient random offset that preserves within-patient temporal order across all tables.
- **Stricter documented variant:** an actual record present in the day 0 to 90 window.

In the strict funnel, requiring observation through day 90 retains **1,664 of 1,775** patients (step S10). Death is unavailable, so mortality is an unmeasured competing event; patients are censored at the last observed date (a stated limitation).

## 7. Exact fields (confirmed and locked for the extraction)

| Concept | Field | Table |
|---|---|---|
| Image laterality | `laterality` (R/L/B/-1; -1 = unknown, B = bilateral holds contralateral knee) | image |
| Arthroplasty on image | `arthroplasty` (0/R/L/B/NL) | image |
| View | `view_position` (F=frontal, L=lateral, S=sunrise, I/E=other) | image |
| Study grouping | `StudyInstanceUID_anon` | image |
| Image date | `StudyDate_anon` | image |
| Index procedure / side | `cpt_code` = 27447, side from `cpt_group_modifier` | CPT |
| Patient id / dates | `empi_anon` / `date_anon` (YYYY-MM-DD) | all |

All image coded fields (laterality, view, weight-bearing, arthroplasty, KLG) are **model-inferred**, not chart ground truth (reported F1 ~0.98 to 0.99 for weight-bearing and arthroplasty; laterality by a dictionary rule). KLG is structurally NULL on ~82% of images (inferred only on weight-bearing bilateral frontal non-arthroplasty views) and is a secondary comparator only.

## 8. Preliminary counts at each major feasibility step

**Sequential cohort flow (patients):**

| Step | Description | Strict | Permissive |
|---|---|---|---|
| S0 | Total patients (demographics) | 83,011 | 83,011 |
| S1 | With any knee radiograph | 83,011 | 83,011 |
| S2 | With CPT 27447 | 8,525 | 8,525 |
| S3 | With any single-side 27447 | 4,730 | 4,730 |
| S4 | Provisional index defined | 4,222 | 3,740 |
| S5 | Aged >= 40 at index | 4,203 | 3,721 |
| S6 | Minus prior contralateral arthroplasty | 3,756 | 3,721 |
| S7a | Minus infection (high-specificity) | 3,752 | 3,717 |
| S7b | Alternative: infection sensitivity definition (reported, not carried) | 3,749 | 3,714 |
| S8 | **With eligible pre-index contralateral image (1 to 365 d)** | **1,807** | 1,806 |
| S9 | Minus contralateral 27447 in day 0 to 90 | 1,775 | 1,775 |
| S10 | Observed through day 90 | 1,664 | 1,662 |
| S11 | **Provisional final landmark cohort** | **1,664** | **1,662** |

**Event counts (strict cohort, n = 1,664):**

| Definition | Events | % |
|---|---|---|
| **Primary (contralateral, laterality-coded, day 91 to 5 y)** | **357** | 21.5% |
| Upper bound (any-modifier later 27447, side unconfirmable) | 381 | 22.9% |
| Blank-modifier event-capture loss (gap) | 24 | 6.7% of events |
| Secondary, 1 y | 247 | |
| Secondary, 2 y | 309 | |
| Secondary, 5 y | 357 | |

*Caveat on the upper bound: it counts any later 27447 of any modifier, which includes index-side reoperations as well as blank-contralateral candidates. The 24-event gap is therefore a loose upper bound on events lost to blank modifiers; the true blank-contralateral loss is smaller. (Also, `from_day1` and `from_day91` secondary counts coincide here because the landmark cohort already excludes every day-0-to-90 contralateral event.)*

**Subgroup event counts (>=100 analyzable; 50 to 99 emphasize CIs; <50 unstable):**

| Subgroup | Patients | Events | Flag |
|---|---|---|---|
| Female / Male | 1,035 / 629 | 227 / 130 | >=100 |
| Age <65 / >=65 | 657 / 1,007 | 153 / 204 | >=100 |
| Black / White | 534 / 955 | 109 / 212 | >=100 |
| Asian | 49 | 8 | <50 |
| Obesity yes / no | 947 / 717 | 243 / 114 | >=100 |
| Weight-bearing / non-WB pre-index | 1,568 / 96 | 337 / 20 | >=100 / <50 |
| Multi-view / frontal-only pre-index | 1,659 / 4 | 357 / 0 | >=100 / <50 |

Major strata, including self-reported Black patients (109 events), are analyzable. Asian race, non-weight-bearing imaging, and frontal-only imaging strata are not. (Tables: `stage1_prelim_flow.csv`, `stage1_event_counts.csv`, `stage1_subgroup_preview.csv`, `feasibility_stage1_counts.json`.)

---

## Feasibility assessment against the protocol floor

The protocol's preliminary practical floor is **500 total primary events and 100 test-allocatable events** (a descriptive benchmark, not the formal Riley calculation).

- **As specified (strict index + 1 to 365 d pre-index imaging): 357 primary events, ~71 test-allocatable. Both fail the floor. Underpowered.**
- **The events exist.** The absolute ceiling of laterality-confirmed contralateral TKAs in the dataset is **699 patients** (patients with both an RT and an LT single-side 27447, exact-token; 701 if the 11 multi-token single-side records are included). Restricting to strict, age >=40, event after day 90 and within 5 years, but **before the imaging and observation gates**, yields **618 events, which clears 500.** These anchor figures are persisted in `outputs/feasibility_stage1_counts.json` under `decision_anchors`.
- **The binding constraint is the pre-index imaging window, not the events.** Step S8 (an eligible pre-index contralateral radiograph within 1 to 365 days) cuts the cohort roughly in half (3,752 to 1,807) and is what drops events below floor. Only **56.4% (2,369 of 4,203)** of strict age >=40 patients have any pre-index radiograph at all within one year, and just **43.0% (1,807 of 4,203)** have an eligible contralateral one.

**Two quantified levers can close the gap:**

1. **Widen the pre-index imaging window.** Contralateral imaging availability rises with the window (strict age >=40 basis: 2,117 at 365 days): 730 days 2,444 (+15.4%), 1,095 days 2,594 (+22.5%), lifetime 2,827 (+33.5%). This moves events from 357 toward the 618 within-5-year ceiling. Trade-off: a baseline radiograph further from index. (Table: `outputs/tables/stage1_imaging_window_sensitivity.csv`.)
2. **Recover index side for blank modifiers (the larger lever).** ~2,983 currently-excluded patients resolve to a concordant single side. This enlarges the eligible population well beyond the 4,203 strict pool and raises the event ceiling itself, because these patients can contribute confirmable contralateral events. Requires a pre-specified, QA-validated recovery rule (protocol section 7 mandates a >=200-patient laterality audit).

Note that blank modifiers cripple cohort **definition** (4,300 patients unsided at index) far more than event **capture** (at most 24 events, itself a loose upper bound that also counts ipsilateral reoperations, so the true loss is smaller), because a contralateral second TKA is usually itself laterality-coded when it occurs.

## Recommendation

**Do not proceed to DICOM transfer under the current strict + 1-to-365-day-imaging definition; it is underpowered at 357 primary events.** Instead, **revise the eligibility definition and re-run this gate before OSF registration**:

1. Widen the pre-index contralateral-imaging window (evaluate ~730 and ~1,095 days) and/or
2. Adopt a pre-specified, QA-validated CPT/image side-recovery step for blank modifiers, then recompute events.

If the revised gate clears roughly 500 primary and 100 test events with acceptable laterality QA, **then** proceed to the formal Riley development-sample and test-precision calculations and OSF preregistration, and only then to image transfer. If no defensible revision clears the floor, invoke the protocol's own section-16 contingency (revise outcome/horizon, or the pre-specified spine-radiograph backup) **before** spending on DICOM transfer and storage.

This maps to the protocol's own graded recommendation: **"Proceed only after resolving specified data issues / revise the eligibility definition before registration."**

## Decisions requiring your sign-off before the full extraction runs

- **A. Same-day companion-line assumption:** for the 2,604 strict patients whose earliest 27447 date is a single RT/LT plus same-day blank companion lines, treat the blanks as same-side billing artifacts (recommended), not a separate contralateral procedure. (Date-level vs row-level resolution is *not* a decision: independent review confirmed 0 patients differ.)
- **B. Infection definition:** high-specificity `knee_osteomyelitis` only (recommended) vs sensitivity adding `joint_infection`. Numerically negligible; choose for defensibility.
- **C. Observation-through-day-90 rule:** last-observed > day 90 (recommended) vs record-in-window (stricter).
- **D. Primary cohort strategy:** strict as candidate primary with permissive as a labeled sensitivity (recommended). Confirm.
- **E. The consequential decision, eligibility revision to reach the floor:** (i) keep the pre-index imaging window at 1 to 365 days as written, or widen it; (ii) whether to adopt the labeled side-recovery arm. These determine whether the study is powered and are the reason for this pause.

## What is built, and what is paused

**Built and validated in Stage 1:** project scaffold and config; typed CSV-to-Parquet conversion (7 tables reconciled); a unit-tested laterality parser (25 tests passing); the data inventory, schema, quality, missingness, and protocol-to-column mapping reports; and these bounded preliminary counts. Independent spec-compliance review passed for both build batches.

**Paused pending your sign-off (Decisions A to E):** the full locked cohort extraction (index, imaging selection, outcomes, follow-up and reverse-KM censoring, cohort-flow, subgroups, and the image-transfer manifest), the final `feasibility_report.md`, and the Times New Roman executive-summary `.docx`. These encode Decisions A to E and will run in a single pass after approval, with no further pause unless a new issue would materially alter the cohort definition.

---
*Supporting artifacts:* `outputs/data_inventory.csv`, `schema_report.md`, `data_quality_report.md`, `missingness_report.csv`, `protocol_to_column_mapping.csv`, `outputs/tables/stage1_*.csv` (including `stage1_imaging_window_sensitivity.csv`), `feasibility_stage1_counts.json` (all go/no-go anchor figures under `decision_anchors`), `outputs/logs/assumptions.md`, `outputs/logs/conversion_checksums.csv`.
