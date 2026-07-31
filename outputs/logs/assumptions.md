# Assumptions & Ambiguity Log — MRKR Contralateral TKA Phase-1 Feasibility

This log records every assumption, ambiguity, conservative decision, and
protocol deviation made during the metadata-only feasibility analysis. It is
appended to by pipeline stages. Each entry: what was ambiguous, the options,
the decision, and the justification.

## Environment (Stage-0, orchestrator)

- **Interpreter:** Validated on the active system interpreter **Python 3.14.0**,
  where duckdb 1.4.4, pandas 2.3.3, pyarrow 23.0.0, lifelines 0.30.3,
  python-docx 1.2.0, matplotlib 3.10.8, PyYAML 6.0.3, numpy 2.4.2 are installed.
  pyenv 3.12.8 is present but lacks the scientific stack, so the plan's
  preliminary note to "pin 3.12" was not followed; `.python-version` is set to
  `system` (the interpreter that has the dependencies). Reproducibility is
  carried by pinned `requirements.txt`. **Deviation from plan note, documented.**
- **Git:** The enclosing git repository is rooted at the HOME directory
  (`<HOME>`), not this project, and has no commits. The pipeline
  performs NO git operations. A project-level `.gitignore` excludes
  `metadata-files/`, `derived-data/`, and manuscript source docs as
  defense-in-depth. No PHI is ever staged or committed.

## Verified encodings (Stage-0 inspection, bounded read-only queries)

- All date fields (`date_anon`, `StudyDate_anon`) are `YYYY-MM-DD`.
- CPT 27447: 14,076 records / 8,525 patients; modifier `cpt_group_modifier`
  raw distribution — NULL/blank 8,613 (61%), RT 2,799, LT 2,641, '50' 11,
  plus 11 parseable multi-modifier records (e.g. 'RT XP', '74 LT'). No RT+LT
  conflicts observed.
- Image: `laterality` R/L/B/-1 (-1 = unknown; B = bilateral, contains
  contralateral knee); `view_position` F/L/S/I/E (F=frontal, L=lateral,
  S=sunrise, I/E=other); `weight_bearing` 0/1; `arthroplasty` 0/R/L/B/NL.
- ICD lifetime distinct-patient infection ceilings: knee_osteomyelitis 1,389,
  joint_infection 551, either 1,792 (pre-index-windowed counts will be smaller).

<!-- Pipeline stages append below this line -->

## Author sign-off on Stage-1 checkpoint (2026-07-21)

The author reviewed `outputs/feasibility_stage1_checkpoint.md` and signed off:
- **Decision A (same-day companion-line assumption):** APPROVED — blank same-day 27447 companion lines accompanying a single RT/LT are same-side billing artifacts (2,604 strict patients).
- **Decision B (infection definition):** APPROVED — high-specificity `knee_osteomyelitis` only.
- **Decision C (observation through day 90):** APPROVED — last-observed date > index + 90 days.
- **Decision D (primary cohort strategy):** APPROVED — strict as candidate primary, permissive/side-recovery as labeled sensitivity.
- **Decision E (imaging window + side recovery):** DELEGATED to the analyst with the objective "high impact, well powered, focused, clinically grounded." Resolved empirically by a re-gate grid (`src/regate.py`, `outputs/tables/regate_grid.csv`); the selected configuration and its verified event count are recorded there and in the final report.

## Stage-1 Data Inventory (`inventory` module)

Stage-1 metadata-only inventory (`src/inventory.py`). Sole owner of this section;
re-runs replace it in place. All statistics are aggregate counts read from the typed
Parquet; no `empi_anon` values, no DICOM pixels, no model performance metrics.

- **Curated ICD flags are per-diagnosis-line, not per-patient.** Availability and
  prevalence are computed as `MAX(flag) GROUP BY empi_anon`. A raw row-level mean is
  wrong (e.g. `knee_osteoarthritis` row-mean = 0.017 vs correct
  patient-level 62.0%). Applied to all 9 flags in every report here.
- **Image `laterality = '-1'` (n=682) treated as invalid/unknown**,
  not a side. Counted under `n_invalid` in the missingness report and flagged for
  exclusion from any laterality-dependent selection. `arthroplasty = 'NL'`
  (n=29) is non-localized (QA/exclusion, not a side).
- **All coded image metadata is model-inferred**, not DICOM ground truth
  (`laterality`, `view_position`, `weight_bearing`, `arthroplasty`, KLG, flip/inverted).
  Reported as strong priors with a provenance caveat, never as certainties.
- **De-identified dates carry a per-patient random shift preserving within-patient
  order** (0 unparseable across all tables). Absolute calendar dates are not
  interpretable; within-patient day intervals (index/landmark/horizon) are valid and
  underpin the timeline. Death is unavailable (competing risk not modelable).
- **KLG is structurally NULL (~82%)** because it is inferred only on weight-bearing
  bilateral frontal, non-arthroplasty views. Documented as a secondary comparator only,
  never a primary feature.
- **Missingness definition:** NULL or empty string. Empirically the typed Parquet has
  no empty strings, so all reported missingness is NULL. Conservatively also surfaced two
  coded-missing sentinels that are non-NULL but semantically missing: image
  `laterality = '-1'` and, newly, **`icd.ICD10 = '--'`**
  (n=7,507,093, ~34% of
  ICD rows) = a no-ICD-10 sentinel (ICD-9-coded / unmapped lines) that cannot join to
  `DX_NAME`. Reported under `n_invalid`; treat as missing for any ICD-10 logic (incl. the
  M17.x side-recovery signal). Curated flags, not the raw ICD-10 join, are the reliable
  comorbidity path.
- **Full-row duplicates are reported, not removed** (pain 1,184,503;
  cpt 1,793; all other tables 0). These are repeated flowsheet /
  billing lines; de-duplication is left to the analysis grain of downstream modules.
- **Dictionary join coverage:** every CPT code in the fact table resolves in
  `cpt_dictionary`; ICD-10 resolution is limited by the `'--'` sentinel above (plus a
  small tail of newer codes absent from the lookup), so ICD-10 -> `DX_NAME` is
  informational, not a denominator.
- **Reconciliation:** every table's Parquet row count was re-checked against config
  `ref_rows` (all 7 match) before any report was written.
