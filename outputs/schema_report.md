# Schema Report - MRKR Contralateral TKA Phase-1 (metadata-only)

Finalized dtypes are the **typed Parquet** columns (reconciled to the raw CSVs in Batch 1). Row counts equal the config `ref_rows`. Example values are the top coded categories with row counts. No DICOM pixels are read; no models are run.

## `demographics`  (MRKR_demographics.csv)
- rows **83,011**; columns **4**; distinct patients **83,011**.

| column | dtype | description | example values (top coded) |
| --- | --- | --- | --- |
| `empi_anon` | VARCHAR | De-identified patient id; linkage key across all tables (VARCHAR). |  |
| `sex` | VARCHAR | Administrative sex (Female / Male). | Female: 51,175; Male: 31,836 |
| `race` | VARCHAR | Patient race category; includes 'Unknown'. | Caucasian or White: 36,927; African American or Black: 33,503; Unknown: 8,751; Asian: 2,893; Multiple: 536; American Indian or Alaskan Native: 244 |
| `ethnicity` | VARCHAR | Patient ethnicity (Non-Hispanic / Hispanic / Unknown). | Non-Hispanic or Latino: 66,378; Unknown: 14,132; Hispanic or Latino: 2,501 |

## `cpt`  (MRKR_CPT.csv)
- rows **6,216,190**; columns **5**; distinct patients **83,011**; date range 1996-04-03 -> 2023-03-29.

| column | dtype | description | example values (top coded) |
| --- | --- | --- | --- |
| `empi_anon` | VARCHAR | De-identified patient id (linkage key). |  |
| `cpt_code` | VARCHAR | CPT/HCPCS procedure code (5-char); join to cpt_dictionary.cpt_code. Index TKA = 27447. | 99214: 783,155; 99213: 555,257; 99233: 210,404; 99232: 140,745; 99024: 139,391; 93010: 120,234 |
| `cpt_group_modifier` | VARCHAR | Raw CPT modifier string (laterality RT/LT/50 + non-laterality tokens); parsed for TKA side. | (null): 6,169,473; RT: 12,293; LT: 11,990; 25: 10,838; RT XP: 2,431; 50: 2,280 |
| `date_anon` | DATE | De-identified procedure date; per-patient random shift preserving within-patient order. |  |
| `age_at_procedure` | DOUBLE | Age (years) at the procedure; HIPAA-bounded (observed 19-89). |  |

## `icd`  (MRKR_ICD.csv)
- rows **21,956,056**; columns **16**; distinct patients **83,011**; date range 1994-12-27 -> 2023-08-31.
- The 9 curated 0/1 flags are **per-diagnosis-line**, not per-patient; patient-level availability needs `MAX(flag) GROUP BY empi_anon`.

| column | dtype | description | example values (top coded) |
| --- | --- | --- | --- |
| `empi_anon` | VARCHAR | De-identified patient id (linkage key). |  |
| `ICD9` | VARCHAR | ICD-9-CM diagnosis code (raw); '--' where no ICD-9 on the line. |  |
| `ICD10` | VARCHAR | ICD-10-CM diagnosis code (raw); '--' where no ICD-10; join to icd_dictionary.ICD10. |  |
| `date_anon` | DATE | De-identified diagnosis date; per-patient random shift preserving within-patient order. |  |
| `age_at_dx` | DOUBLE | Age (years) at diagnosis; HIPAA-bounded (observed 19-89). |  |
| `DX_LINE` | VARCHAR | Diagnosis line role (Primary / Secondary / Not Recorded / problem-list states). | Secondary: 10,204,274; Not Recorded: 6,479,660; Primary: 5,163,413; Active: 91,895; Resolved: 11,819; Canceled: 3,366 |
| `DX_ICD_SCOPE` | VARCHAR | Diagnosis context (Billing / Discharge / Admitting / Problem List / etc.). | Billing Diagnosis: 15,156,800; Discharge Diagnosis: 4,386,556; Admitting Diagnosis: 1,002,671; Referring Diagnosis: 615,115; Not Recorded: 359,007; Reason For Visit: 238,388 |
| `autoimmune` | TINYINT | Curated 0/1 comorbidity flag 'autoimmune' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `diabetes` | TINYINT | Curated 0/1 comorbidity flag 'diabetes' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `hypertension` | TINYINT | Curated 0/1 comorbidity flag 'hypertension' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `joint_infection` | TINYINT | Curated 0/1 comorbidity flag 'joint_infection' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `knee_osteoarthritis` | TINYINT | Curated 0/1 comorbidity flag 'knee_osteoarthritis' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `knee_osteomyelitis` | TINYINT | Curated 0/1 comorbidity flag 'knee_osteomyelitis' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `obesity` | TINYINT | Curated 0/1 comorbidity flag 'obesity' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `nicotine_use` | TINYINT | Curated 0/1 comorbidity flag 'nicotine_use' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |
| `trauma_lower_extremity` | TINYINT | Curated 0/1 comorbidity flag 'trauma_lower_extremity' - PER-DIAGNOSIS-LINE; patient-level needs MAX(flag) GROUP BY empi_anon. |  |

## `image`  (MRKR_image_metadata.csv)
- rows **503,261**; columns **19**; distinct patients **83,011**; date range 2002-01-31 -> 2021-12-26.
- Coded image fields (`laterality`, `view_position`, `weight_bearing`, `arthroplasty`, KLG, flip/inverted) are **model-inferred**, not DICOM ground truth (WB / arthroplasty classifiers F1 ~0.98-0.99).

| column | dtype | description | example values (top coded) |
| --- | --- | --- | --- |
| `empi_anon` | VARCHAR | De-identified patient id (linkage key). |  |
| `StudyInstanceUID_anon` | VARCHAR | De-identified DICOM Study UID; groups images of one exam (169,004 studies). |  |
| `SeriesInstanceUID_anon` | VARCHAR | De-identified DICOM Series UID. |  |
| `SOPInstanceUID_anon` | VARCHAR | De-identified DICOM instance UID; unique per image row (503,261). |  |
| `img_height` | VARCHAR | Image height in pixels (raw float-string). |  |
| `img_width` | VARCHAR | Image width in pixels (raw float-string). |  |
| `laterality` | VARCHAR | MODEL-INFERRED knee side: R / L / B (bilateral, contains contralateral) / -1 (unknown). | R: 184,327; L: 176,945; B: 141,307; -1: 682 |
| `view_position` | VARCHAR | MODEL-INFERRED view: F=frontal, L=lateral, S=sunrise, I/E=other. | F: 205,121; L: 195,802; S: 70,250; I: 27,177; E: 4,911 |
| `horizontal_flip` | INTEGER | MODEL-INFERRED preprocessing flag (0/1): image horizontally flipped. |  |
| `weight_bearing` | INTEGER | MODEL-INFERRED weight-bearing flag (0/1); classifier F1 ~0.98. | 0: 349,194; 1: 154,067 |
| `inverted` | INTEGER | MODEL-INFERRED preprocessing flag (0/1): photometric inversion. |  |
| `arthroplasty` | VARCHAR | MODEL-INFERRED prosthesis laterality: 0=none, R/L/B, NL=non-localized; F1 ~0.99. | 0: 388,648; R: 55,579; L: 50,023; B: 8,982; NL: 29 |
| `L_KLG_inference` | DOUBLE | MODEL-INFERRED left Kellgren-Lawrence grade; ~82% NULL (WB bilateral frontal, non-arthroplasty only). |  |
| `R_KLG_inference` | DOUBLE | MODEL-INFERRED right Kellgren-Lawrence grade; ~82% NULL (structural). |  |
| `SeriesDescription` | VARCHAR | Free-text DICOM series description (raw). |  |
| `StudyDescription` | VARCHAR | Free-text DICOM study description (raw). |  |
| `StudyDate_anon` | DATE | De-identified study date; per-patient random shift preserving within-patient order. |  |
| `age_at_exam` | DOUBLE | Age (years) at the exam; HIPAA-bounded (observed 19-89). |  |
| `dicom_path` | VARCHAR | Relative DICOM transfer-manifest path (metadata only; no pixels opened). |  |

## `pain`  (MRKR_pain.csv)
- rows **4,970,869**; columns **6**; distinct patients **83,011**; date range 1999-12-05 -> 2023-03-29.

| column | dtype | description | example values (top coded) |
| --- | --- | --- | --- |
| `empi_anon` | VARCHAR | De-identified patient id (linkage key). |  |
| `pain_location` | VARCHAR | Free-text pain location (raw VARCHAR); ~75% NULL. |  |
| `knee_pain` | VARCHAR | Knee-pain flag as raw VARCHAR ('0'/'1'). | 0: 4,705,673; 1: 265,196 |
| `pain_score` | VARCHAR | Pain score as raw VARCHAR ('0'..'10'). |  |
| `laterality` | VARCHAR | Pain side as raw VARCHAR (R/L/B); ~95% NULL. | (null): 4,724,556; R: 96,492; L: 87,153; B: 62,668 |
| `date_anon` | DATE | De-identified encounter date; per-patient random shift preserving within-patient order. |  |

## `cpt_dictionary`  (MRKR_CPT_dictionary.csv)
- rows **7,166**; columns **2**.
- Lookup table: `cpt_code` -> `cpt_description`. Join `cpt.cpt_code = cpt_dictionary.cpt_code` (all fact CPT codes are covered).

| column | dtype | description | example values (top coded) |
| --- | --- | --- | --- |
| `cpt_code` | VARCHAR | CPT code (unique key of this lookup). |  |
| `cpt_description` | VARCHAR | Human-readable CPT long description (join target for cpt.cpt_code). |  |

## `icd_dictionary`  (MRKR_ICD_dictionary.csv)
- rows **25,209**; columns **3**.
- Lookup table: `ICD10` -> `DX_NAME`. Join `icd.ICD10 = icd_dictionary.ICD10` (fact rows with `ICD10 = '--'` have no match by construction).

| column | dtype | description | example values (top coded) |
| --- | --- | --- | --- |
| `ICD9` | VARCHAR | ICD-9-CM code (raw). |  |
| `ICD10` | VARCHAR | ICD-10-CM code (unique key of this lookup). |  |
| `DX_NAME` | VARCHAR | Human-readable diagnosis name (join target for icd.ICD10). |  |

## Dictionary interpretation (join examples)

**CPT** `cpt.cpt_code` -> `cpt_dictionary.cpt_description`:
- `27446` -> Arthroplasty, knee, condyle and plateau; medial OR lateral compartment
- `27447` -> Arthroplasty, knee, condyle and plateau; medial AND lateral compartments with or without patella resurfacing (total knee arthroplasty)
- `27486` -> Revision of total knee arthroplasty, with or without allograft; 1 component

**ICD-10** `icd.ICD10` -> `icd_dictionary.DX_NAME` (note the 5th-digit laterality used by the side-recovery signal):
- `M17.11` -> Unilateral primary osteoarthritis, right knee
- `M17.12` -> Unilateral primary osteoarthritis, left knee
- `M17.31` -> Unilateral post-traumatic osteoarthritis, right knee
- `M17.32` -> Unilateral post-traumatic osteoarthritis, left knee

