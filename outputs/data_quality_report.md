# Data-Quality Report - MRKR Contralateral TKA Phase-1 (metadata-only)

Aggregate data-quality concerns for the Stage-1 feasibility gate. Counts only; no patient identifiers, no DICOM pixels, no model outputs beyond the provider-supplied inferred metadata fields (which are themselves flagged as a caveat below).

## Headline concern - TKA laterality is under-coded on the index procedure

Of the **14,076** CPT `27447` (total knee arthroplasty) records across **8,525** patients, **8,613 (61.2%)** carry a **blank `cpt_group_modifier`** - no RT/LT/50 laterality token. Only 2,799 RT, 2,641 LT, and 11 '50' (bilateral) are explicitly side-coded. This 61% blank rate directly caps how many index TKAs (and therefore contralateral events) can be assigned a side from CPT alone; it is the primary feasibility constraint and motivates the image-`arthroplasty` and ICD 5th-digit side-recovery cross-checks.

### CPT 27447 modifier distribution
| modifier | records |
| --- | --- |
| (null) | 8,613 |
| RT | 2,799 |
| LT | 2,641 |
| 50 | 11 |
| RT XP | 3 |
| 74 LT | 2 |
| LT XE | 1 |
| 73 LT | 1 |

## Image metadata is model-inferred (provenance caveat)

`laterality`, `view_position`, `weight_bearing`, `arthroplasty`, `horizontal_flip`, `inverted`, and the KLG grades are **model predictions** from the Emory MRKR pipeline, not DICOM ground truth. Reported classifier accuracy is high (weight-bearing F1 ~0.981, arthroplasty F1 ~0.992; laterality via dictionary rules), but any downstream selection built on them inherits that error rate. Treat them as strong priors, not certainties.

### Image laterality (model-inferred) - '-1' is invalid/unknown
| laterality | images |
| --- | --- |
| R | 184,327 |
| L | 176,945 |
| B | 141,307 |
| -1 | 682 |

`-1` (n=682) is an unresolved/unknown side and MUST be excluded from any laterality-dependent selection. `B` = bilateral frontal (contains the contralateral knee without a crop). `arthroplasty = 'NL'` (n=29) is a non-localized prosthesis detection used for QA/exclusion, not side assignment.

### Image view and weight-bearing (model-inferred)
| view_position | images |
| --- | --- |
| F | 205,121 |
| L | 195,802 |
| S | 70,250 |
| I | 27,177 |
| E | 4,911 |

| weight_bearing | images |
| --- | --- |
| 0 | 349,194 |
| 1 | 154,067 |

### Image arthroplasty (model-inferred)
| arthroplasty | images |
| --- | --- |
| 0 | 388,648 |
| R | 55,579 |
| L | 50,023 |
| B | 8,982 |
| NL | 29 |

## Kellgren-Lawrence grades are structurally NULL (~82%)

`L_KLG_inference` is NULL for 413,465 of 503,261 images (~82%) and `R_KLG_inference` for 414,705 (~82%). This is **by design**: KLG is inferred only on weight-bearing bilateral frontal views without arthroplasty. KLG is therefore a **secondary comparator only** - it cannot serve as a primary feature because it is unavailable for most images.

## Pain table is sparse on the fields that matter

`laterality` is NULL for 4,724,556 of 4,970,869 pain rows (~95%), and `pain_location` is NULL for ~75%. `knee_pain` and `pain_score` are populated but stored as raw VARCHAR ('0'/'1' and '0'..'10'). Pain is a **secondary predictor**; side-specific pain is largely unavailable.

| knee_pain | rows |
| --- | --- |
| 0 | 4,705,673 |
| 1 | 265,196 |

| pain_score (raw) | rows |
| --- | --- |
| 0 | 1,456,553 |
| 1 | 101,603 |
| 2 | 264,857 |
| 3 | 335,448 |
| 4 | 386,993 |
| 5 | 467,163 |
| 6 | 414,353 |
| 7 | 455,189 |
| 8 | 510,617 |
| 9 | 237,587 |
| 10 | 340,506 |

## Curated ICD flags are per-diagnosis-line, not per-patient

The 9 curated 0/1 flags label **each diagnosis line**. A raw row-level mean is therefore wrong: `knee_osteoarthritis` averages 0.017 across diagnosis lines, but the correct **patient-level** prevalence (`MAX(flag) GROUP BY empi_anon`) is 62.0%. Always aggregate to the patient before interpreting availability or prevalence.

### Patient-level comorbidity/flag prevalence (MAX per patient, N = 83,011)
| curated flag | patients | prevalence % |
| --- | --- | --- |
| autoimmune | 9,704 | 11.7 |
| diabetes | 18,655 | 22.5 |
| hypertension | 45,300 | 54.6 |
| joint_infection | 551 | 0.7 |
| knee_osteoarthritis | 51,468 | 62.0 |
| knee_osteomyelitis | 1,389 | 1.7 |
| obesity | 27,576 | 33.2 |
| nicotine_use | 20,882 | 25.2 |
| trauma_lower_extremity | 34,230 | 41.2 |

## ICD-10 uses a `'--'` no-code sentinel (join/coverage caveat)

`icd.ICD10` is never NULL, but `'--'` (a no-ICD-10 sentinel) appears on 7,507,093 of 21,956,056 rows (~34%) - these are ICD-9-coded or unmapped lines and cannot join to `DX_NAME`. Treat `'--'` as missing for any ICD-10-based logic (including the M17.x side-recovery signal). The curated flags, not the raw ICD-10 join, are the reliable comorbidity path.

## Demographics - race 'Unknown'

`race` = 'Unknown' for **8,751** patients (~11%); `ethnicity` = 'Unknown' for 14,132. Report an explicit Unknown stratum rather than dropping or imputing.

| race | patients |
| --- | --- |
| Caucasian or White | 36,927 |
| African American or Black | 33,503 |
| Unknown | 8,751 |
| Asian | 2,893 |
| Multiple | 536 |
| American Indian or Alaskan Native | 244 |
| Native Hawaiian or Other Pacific Islander | 157 |

## De-identified dates - within-patient intervals are valid

All dates (`date_anon`, `StudyDate_anon`) carry a **per-patient random shift** that preserves within-patient temporal order across all tables (0 unparseable). Absolute calendar dates are meaningless, but **within-patient day intervals** (index -> event, landmark, horizon) are valid and are the basis for the whole timeline. Death is unavailable, so the competing risk of mortality cannot be modeled from this metadata.

## Full-row duplicates

Exact-duplicate rows are retained as-is (the inventory only reports them): `cpt` 1,793; `pain` 1,184,503. Pain duplicates are repeated flowsheet entries; CPT duplicates are repeat billing lines. De-duplicate deliberately per analysis grain, not globally.

