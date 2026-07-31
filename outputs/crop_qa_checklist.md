# Crop QA sign-off — contralateral-knee crops

**TRAINING IS BLOCKED UNTIL THIS FILE IS SIGNED.** Success non-negotiable #1 of the approved plan is contralateral crop fidelity. 4,075 of the 4,269 frontal films in this cohort are bilateral (BOTH knees on one image), so the correct knee is selected from metadata (`laterality == 'B'`, `contra_side`, `horizontal_flip`, and the radiological display convention). An inverted sign there would train the model on the knee that has ALREADY been replaced, and every downstream metric would be meaningless. No automated test can close that gap; a human must look at the images below.

`horizontal_flip` is a MODEL-INFERRED MRKR annotation with an unquantified error rate and no DICOM tag to check it against. It drives the half-select on 287 images. That is the specific thing panel A of the contact sheet exists to let you verify.

## 1. Reviewer criteria — tick every box

How to read a contact-sheet row: the **left panel** is the full film, flip-corrected and NOT mirrored, with the half the pipeline KEPT outlined in green and the discarded half marked INDEX in red. The **right panel** is that crop before the left/right mirror. Judge criterion 1 on the LEFT panel; criteria 2 and 3 on the RIGHT panel.

- [ ] the outlined half of the full film is the CONTRALATERAL knee (opposite index_side)
- [ ] the crop contains no pixels from the index knee or the midline
- [ ] no residual burned-in laterality marker or text
- [ ] the sample above is representative (all view x side cells populated)

## 2. Evidence

- Contact sheet: `outputs/figures/crop_qa_contact_sheet.png`
- Shards inspected: `.../samersalman/mrkr-shards`
- Splits sampled for the contact sheet: **train, val** — the LOCKED test split is NOT sampled here and stays sealed.
- Target per cell: **12**; crop panels rendered: **72**, full-film panels rendered: **72**, across **6** (view x contra_side) cells.
- Full films were re-read from the DICOMs for 72 of 72 sampled rows, so half-select is visually decidable.

### Sample composition

| view | contra_side | tiles | fallback crops | mean crop_confidence | mean masked_pct |
|---|---|---|---|---|---|
| frontal | L | 12 | 0 | 1.000 | 22.8% |
| frontal | R | 12 | 0 | 1.000 | 22.8% |
| lateral | L | 12 | 0 | 1.000 | 23.0% |
| lateral | R | 12 | 0 | 0.988 | 23.4% |
| sunrise | L | 12 | 0 | 1.000 | 23.1% |
| sunrise | R | 12 | 0 | 1.000 | 23.6% |

### Whole-run crop statistics (all sampled splits, not just the tiles)

- Crops written: **4855** across **2966** patients.
- **Crop centre: `localizer_mode` = `center_default`.** Fallback-localization rate among WRITTEN crops: 0.00% (0/4855); mean crop_confidence 0.996. Under `center_default` the deterministic CENTRED box is the primary estimate and the intensity-profile localizer may only move it when it clears `localizer_refine_min_confidence`. On 800 real TRAIN films that localizer fell back on 26.0% of images (41.4% of single-knee views) and its *successes* were frequently worse than the centre — it locks onto a bright shaft or collimator edge — so a centred crop is a deliberate choice here, NOT a failure, and nothing is excluded for using one.
- Fallback rate by view: frontal 0.0%, lateral 0.0%, sunrise 0.0%
- **Masked pixels (protocol section 13): mean 23.10%, p90 23.64%, max 44.75%** (fixed border band 22.75% + out-of-bounds padding; cap `max_masked_pct` = 45%). 1248 crops carry padding beyond the border band.
- **Burned-in markers surviving into the finished crops: 16.7% of the 72 sampled crops** (mean 0.35 blob(s) each; by view: frontal 8.3%, lateral 33.3%, sunrise 8.3%). Measured on the FINISHED crops, so it is what SURVIVED, not what the masker believes it removed. This is an UPPER BOUND — saturated bone edges share the signature — so read it as a list of crops to LOOK AT. Criterion 3 above is the finding of record.
- **Protocol section 13 exclusions (never written to a shard): excessive_masking 17, localization_failed 0.**
- **Laterality-assertion violations: 0** (images whose manifest `laterality` did not equal `contra_side` on a single-knee view, or whose side could not be resolved). These were routed to the failure report and NOT processed.
- All preprocessing failure reasons (counts): {'excessive_masking': 17, 'view_mask_drift': 1}
- Half-select applied to **3255** bilateral frontals; **1600** single-knee films needed none.
- Orientation after standardization: {'left': 4855} (every crop should read as a LEFT knee).

## 3. Protocol section 23 (i) — image-level manual QA

- Required: **400** index images, **2** independent reviewers with orthopedic or MSK imaging experience, scored on **laterality, view, native_knee, crop_adequacy, burned_in_text, non_knee_content**.
- Sampled: **400** images (stratified on split x view x contra_side x bilateral/unilateral) from a pool of **4855**.
- Splits: requested ['train', 'val', 'test'], present ['train', 'val'], MISSING ['test'] (those crops have not been generated yet).
- This audit spans **all** splits on purpose. It is LABEL/CROP quality assurance that precedes model registration: the workbook carries pixels and acquisition metadata and **no outcome column at all**, so it is NOT outcome-unblinding and does not touch the sealed-test guarantee.
- By view: {'frontal': 278, 'lateral': 108, 'sunrise': 14}; by split: {'train': 344, 'val': 56}; by laterality kind: {'bilateral_B': 261, 'unilateral': 139}.
- Blank adjudication workbook (contains empi_anon — git-ignored): `derived-data/cohort/image_audit_workbook.csv`
- Reviewer panels (full film + pre-mirror crop, one PNG per row): `derived-data/cohort/qa_panels/`

**Scoring convention.** Every `<item>_r<k>` cell takes exactly one of `OK` or `ERROR` (aliases PASS/FAIL, Y/N, 1/0 are accepted). Leave a cell blank if not yet reviewed. Reviewers must not see each other's columns while scoring. Resolve disagreements by consensus or a third reviewer and record the result in `<item>_adjudicated`.

Then run `python3 -m src.crop_qa --score derived-data/cohort/image_audit_workbook.csv`. It reports raw agreement and Cohen's kappa per item and writes the aggregate to `outputs/tables/image_audit_summary.csv`. If any item's critical-error rate exceeds **2%** the review MUST be expanded before training.

## 4. Protocol section 23 (ii) — outcome-record audit

- Required: **200** outcome records reviewed via CPT chronology and available post-event images. Sampled: **200**.
- Composition: **100 events** and **100 non-events** drawn from 3709 cohort patients carrying 533 events. Events are deliberately over-sampled — the endpoint being validated is the event, and a sample at the natural prevalence would give the reviewer too few to check.
- side_source: {'coded': 120, 'recovered': 80}; split: {'train': 144, 'test': 41, 'val': 15}.
- 200 of the sampled records carry at least one knee-arthroplasty CPT row (556 rows in total).
- Workbook (contains empi_anon — git-ignored): `derived-data/cohort/outcome_audit_workbook.csv`
- Full CPT chronology, long format: `derived-data/cohort/outcome_audit_cpt_rows.csv`
- Aggregate-only summary in outputs: `outputs/tables/outcome_audit_summary.csv`
- Reviewer fills `reviewer_event_confirmed_Y_N`, `reviewer_event_date`, `reviewer_event_side`, `reviewer_agrees_with_primary_event_Y_N` and `reviewer_notes`.
- The sample spans all splits because it validates the LABEL-EXTRACTION algorithm, not the model. Adjudication results must not be used to select, tune or threshold any model; a discrepancy re-opens the cohort lock and the labels are rebuilt for every patient, which is the only legitimate response.

## 5. Laterality QA audit (protocol section 7)

- Patient sample: **200** (minimum required 200).
- side_source composition: {'recovered': 133, 'coded': 67}.
- The sample **deliberately over-samples `side_source == "recovered"`** patients (1,828 of 3,709 in the cohort). Those patients' index laterality was INFERRED from concordant signals rather than coded on the CPT modifier, so they carry all of the residual side-assignment risk; the coded patients are included only as a control.
- `index_side` **cannot be judged from a contralateral crop** — the crop is by construction the other knee. Each row therefore also carries `qa_panel_png`, the full-film panel, which is what makes the question answerable.
- Patient-level adjudication file (contains empi_anon — git-ignored): `derived-data/cohort/laterality_audit_sample.csv` (crops alongside it in `derived-data/cohort/laterality_audit_crops/`)
- Aggregate-only summary in outputs: `outputs/tables/laterality_audit_summary.csv`
- Reviewer fills `reviewer_index_side`, `reviewer_agrees_Y_N` and `reviewer_notes` per row. The audit PASSES only if the reviewed index side matches the coded/recovered `index_side` for every row; any disagreement re-opens the cohort lock.

## 6. Sign-off

Signing means: I inspected the contact sheet, the outlined half of each full film is the contralateral knee, no index-knee pixels or laterality markers survive in the crops, and the protocol section 23 audits above are adjudicated and passed.

| field | value |
|---|---|
| Reviewer name | |
| Role | |
| Date (YYYY-MM-DD) | |
| Signature | |
| Result (PASS / FAIL) | |
| If FAIL, defect and required fix | |

Until `Result` reads PASS, `notebooks/train_colab.ipynb` must not be run.
