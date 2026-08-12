# MRKR Contralateral TKA — Multi-View Radiographic Prediction

Deterministic, auditable, restartable pipeline for *"Multi-View Radiographic
Prediction of Contralateral Knee Arthroplasty After Unilateral TKA"*.

**Phase 1 (COMPLETE, 2026-07-21)** answered the feasibility question from
metadata alone — no DICOM opened, no model trained. Primary cohort =
`recovery_any` / 2-year pre-index imaging window: **3,709 patients, 533 five-year
events**; recommendation **PROCEED**, contingent on the protocol section 7
laterality QA audit.

**Phase 2 (COMPLETE, 2026-07-30)** built the clinical comparators, the multi-view
image model, and the single permitted read of the locked test split. What exists
today:

| track | state |
| --- | --- |
| A — clinical feature table, **M0** and **M1** penalized-Cox models, Riley sample size | **done**; fitted on train, penalizer chosen by cross-validation inside train |
| B — bulk DICOM transfer (Globus batch, transfer, verification) | **done 2026-07-25**; 6,122 of 6,122 manifest images verified, 34.85 GB |
| C — preprocessing, crop QA materials, Colab training | preprocessing and training **done**; the crop QA gate is **still UNSIGNED** (see **D22** and **D31**) |
| D — sealed-split scoring and evaluation | **done 2026-07-29**; `src/score_test.py`, then `src/eval_models.py --split test` |

**Seven arms × five seeds = 35 checkpoints** were trained under training-contract
hash `4b862b5ecb947314`: `m0d_clinical`, `m1_klg`, `m2_frontal`, `m3_image`,
`m4_frontal`, `m4_fusion`, `r1_densenet_frontal`.
`derived-data/cohort/train_arms.json` is the authoritative record of what each arm
is. Read `r1_densenet_frontal` as a historical arm key, not as an architecture
claim: that arm is the protocol section-25 backbone robustness run and its encoder
is **ConvNeXt-Tiny** (`arch: convnext_tiny`, label "R1 ConvNeXt-Tiny frontal-only
robustness"). Every other arm uses DenseNet121. The key survives because it is
baked into the checkpoint filenames and into the stored hazard files. The
checkpoints total about 4.7 GB, live outside this tree, and are never committed.

**The locked test split was read, once.** `src/score_test.py` scored the frozen
ladder on **741 patients and 106 events** (740 of them with crops, 1,216 crops) on
**2026-07-29** and recorded the event in `derived-data/cohort/test_scoring.json` as
`sealed_read: "PERFORMED."` beside the training-contract hash. That is the one read
protocol sections 12 and 17 permit, and it supplies the pre-specified primary
performance estimates. `src.eval_models.assert_sealed_read_is_recorded` enforces the
guarantee at render time: change a model after the read and the hash stops matching,
so the render fails rather than publishing a stale out-of-sample claim. The three
`forbid_test_split: true` flags are untouched and still guard `model_clinical`, the
local trainer and the validation evaluator.

Two caveats belong beside that sentence, not in a footnote. First, the
`model_image.test.unlock_requires` preconditions were **not all met**: the crop QA
sign-off and the section 7 laterality audit were outstanding at the time of the read
and are outstanding now, and the read went ahead on an explicit author instruction.
**D31** records exactly what was and was not met. Second, the v6 revision adds
**post-hoc imaging analyses** (attribution, occlusion, leakage controls, per-view
attention) computed on this same already-read split. Those are exploratory, they are
registered as deviation **D35**, and the single-read guarantee therefore covers the
pre-specified primary estimates only, not them.

**Validation metrics are not performance claims.** Each image arm's checkpoint was
selected on validation negative log-likelihood and its recalibration was fitted on
the same 371 validation patients the slope is then measured on, so a validation
metric is optimistic by construction. The test estimates took no part in fitting,
hyperparameter choice, checkpoint selection or recalibration.

> **PHI note.** `metadata-files/`, `derived-data/` and `DICOMs/` hold
> de-identified patient-level data and are git-ignored. `outputs/` is
> **aggregate only** — no `empi_anon`, no UID. The pipeline performs no git or
> network operations. Never commit patient data, and never commit `CLAUDE.md`.

## Protocol conformance quick reference

* **Model ladder (protocol Table 7).** `M0 = age, sex, comorbidities, pain,
  image-to-index interval` (11 predictors, 13 design columns, **12 identified
  parameters**). `M1 = M0 + dataset-inferred contralateral KLG`, fitted on the
  **KLG-eligible subset only** (3,566 of 3,709 patients). Inferred KLG is a
  **secondary comparator only** (protocol Table 6) and is *not* an M0 predictor —
  see deviation **D14**.
* **Primary estimand (protocol Table 8).** M4 multimodal versus M0 clinical-only,
  IPCW cumulative/dynamic AUROC at 5 years, paired patient-level bootstrap,
  2,000 replicates, one unadjusted comparison.
* **Deviations.** `outputs/protocol_deviations.md` is the canonical register.
  Entries are numbered from `D1` and appended as the study proceeds, and they
  include the items still awaiting an author decision. **No D-range is typed here,
  deliberately.** `src.make_manuscript.deviation_span` derives the range the
  manuscript cites by parsing the register's own `## D<n>.` headings, so a range
  typed into this file would eventually contradict the generated text — it already
  had. Read the current one off the register:
  `grep -o '^## D[0-9]*' outputs/protocol_deviations.md | tail -1`. **D31** records
  the sealed read of the test split and the preconditions that were not met when
  it was performed.

## Project structure

```
metadata-files/     # original CSVs (never modified; git-ignored)
protocol/           # copy of the study protocol + MRKR dataset PDF
config/             # feasibility.yaml — ALL protocol parameters, Phase 1 and 2
src/                # pipeline modules (config-driven, run as python3 -m src.<name>)
notebooks/          # Colab notebooks; train_and_score_colab.ipynb is the one that ran
sql/                # (reserved) SQL is currently inline in the Python modules
tests/              # pytest suite (synthetic inputs only)
derived-data/       # typed Parquet + cohort tables + frozen model JSON + per-patient
                    #   hazards (git-ignored)
outputs/            # reports, tables, figures, logs (AGGREGATE ONLY)
```

## Environment

Phase 1 and Track A were validated on **system Python 3.14.0** (the scientific
packages were installed there; pyenv 3.12.8 lacked them, so `.python-version` is
`system`). Check what `python3` resolves to before a re-run: `system` follows the
machine, and on this one it now resolves to 3.12. Every module is run as
`python3 -m src.<name>`, from the project root, so the package imports resolve.
Pinned dependencies are in `requirements.txt`.

```bash
python3 -m pip install -r requirements.txt   # if using a fresh environment
```

`scikit-survival` is **not** installed locally and is not needed: the survival
estimators are implemented in numpy/lifelines inside `src/`. `torch` is a different
matter. Training ran in Colab, but `src/train_model.py` also holds numeric
primitives that `src/score_test.py` and `src/eval_models.py` import, so those two
modules and five of the test files need a working `torch` and `torchvision`
locally. `requirements-training.txt` pins that environment (`torch==2.13.0`,
`torchvision==0.28.0`, `timm==1.0.28`); the trained models were produced under torch
2.11.0+cu128 and timm 1.0.28, recorded in `derived-data/cohort/train_arms.json`.
Install the pinned pair together. A `torch` and `torchvision` that disagree fail at
import with `operator torchvision::nms does not exist`, which takes those five test
files down as collection errors rather than as failures.

On the cloud-synced Desktop path, DuckDB's Parquet reader can abort on EINTR
under I/O load; pandas/pyarrow reads tolerate it. Prefer running off the synced
location for a clean full re-run.

## Tests

```bash
python3 -m pytest tests/ -q
```

The suite is synthetic-input only — no patient data is read.

**Almost all of it runs with no deep-learning stack at all.** Measured with `torch` and
`torchvision` blocked at import, the great majority of the suite still collects and
passes on a bare `requirements.txt` install. That includes `test_eval_models.py` and
`test_v6_analyses.py`, which reach `src.train_model` for numeric primitives but exercise
paths that never construct a tensor, and `test_train_model.py`, whose torch-dependent
cases guard themselves with `pytest.importorskip` and **skip** rather than take the run
down.

**One file is genuinely torch-gated:** `tests/test_interpretability.py` calls
`pytest.importorskip("torch")` at module scope, so without torch it skips in its
entirety. A small number of render cases in `test_manuscript_figures.py` fail rather
than skip without torch; the rest of that file runs, and it also checks the split
anchors (741 / 106 / 740 / 1216) against the real artefacts.

**The real hazard is a *mismatched* torch/torchvision pair, not an absent one**, and an
earlier version of this section blamed the wrong cause. A `torch` and `torchvision` that
disagree abort at import with `operator torchvision::nms does not exist`, and *that* is
what produces collection errors — in `test_make_manuscript.py`,
`test_manuscript_figures.py` and `test_interpretability.py`, which is a different and
larger set than the files that need torch to pass. Install the pinned pair together from
`requirements-training.txt`.

Neither a file count nor a test count is quoted here, on purpose. Both have drifted as
modules were extended, and a stale count in a README is worse than no count. Get the
current ones from the suite itself, and re-derive the grouping rather than trusting a
prose list:

```bash
python3 -m pytest tests/ --collect-only -q | tail -1     # test count
ls tests/test_*.py | wc -l                               # file count
```

## Run order — Phase 1 (feasibility, already complete)

```bash
python3 -m src.io_duckdb            --config config/feasibility.yaml   # CSV -> typed Parquet
python3 -m src.inventory            --config config/feasibility.yaml   # schema, quality, mapping
python3 -m src.preliminary_counts   --config config/feasibility.yaml   # checkpoint counts

# the locked extraction, end to end (index -> imaging -> outcomes -> followup ->
# assemble -> cohort_flow -> subgroups -> manifest), then consolidate
python3 -m src.run_feasibility      --config config/feasibility.yaml --stages all
```

`--with-parquet` also rebuilds the typed Parquet. Key deliverables:
`outputs/feasibility_report.md`, `feasibility_summary.json`,
`feasibility_executive_summary.docx`, `cohort_flow.csv`, `subgroup_counts.csv`,
`event_counts.csv`, and the review-only
`outputs/tables/image_transfer_manifest.csv`.

## Run order — Phase 2

### Track A: clinical models and sample size (runs locally, in this order)

```bash
python3 -m src.features_clinical  --config config/feasibility.yaml
python3 -m src.model_clinical     --config config/feasibility.yaml
python3 -m src.sample_size_riley  --config config/feasibility.yaml
```

Each step consumes the previous one's frozen artefact, so the order is binding:
`features_clinical` writes `clinical_imputation_params.json` (the train-fitted
transform, plus the `model_columns` / `m1_model_columns` design contracts);
`model_clinical` replays that transform and freezes
`m0_clinical_model.json` / `m1_klg_model.json`; `sample_size_riley` reads the
identified-parameter count and the horizon grid out of the M0 contract.
Outputs: `outputs/features_clinical_completeness.md`,
`outputs/clinical_baseline_report.md`, `outputs/clinical_m1_klg_report.md`,
`outputs/sample_size.md` and `outputs/tables/m0_*.csv`, `m1_metrics.csv`,
`sample_size_riley.csv`.

`model_clinical` and `sample_size_riley` both push a `split != "test"` predicate
into the Parquet reader, so the sealed rows are never materialised.

### Track B: bulk image transfer — COMPLETE (2026-07-25)

```bash
python3 -m src.make_globus_batch --config config/feasibility.yaml   # writes the batch file
# HUMAN STEP: run the Globus transfer (see outputs/globus_transfer_runbook.md)
python3 -m src.verify_transfer   --config config/feasibility.yaml   # verifies what arrived
```

The batch file (`derived-data/cohort/globus_batch.txt`, 6,122 files) exists and the
transfer ran. `outputs/transfer_verification.md` records **PASS, 6,122 of 6,122
manifest images present, 34.85 GB**. The staged DICOMs were consumed by
preprocessing and the staging root (`transfer.dest_root`) no longer exists on this
machine; re-running Track C from source requires re-running the transfer. The
512×512 crops and the WebDataset shards built from them are what training and
scoring consume, and they are held outside this tree.

### Track C: preprocessing, QA gate, training — TRAINED; THE QA GATE IS STILL OPEN

```bash
# 1. Preprocess DICOMs -> 512x512 contralateral crops -> WebDataset shards.
#    Run EITHER locally on the transferred DICOMs:
python3 -m src.preprocess_images --dicom-root <DICOM root> --out-dir <shard dir>
#    OR in Colab with notebooks/preprocess_colab.ipynb, whose pipeline region is a
#    VERBATIM slice of src/preprocess_images.py (the notebook asserts the identity).
#    The test split is EXCLUDED by default and takes a second, deliberate call:
python3 -m src.preprocess_images --dicom-root <DICOM root> --out-dir <test shard dir> \
                                 --splits test --include-test

# 2. Build the crop QA evidence and the reviewer workbooks.
python3 -m src.crop_qa --dicom-root <DICOM root>

# 2b. Rebuild ONLY the section 23(i) image-audit sample, from explicit shard
#     sidecars rather than from the (single, last-run-wins) preprocess_run.json.
python3 -m src.crop_qa --rebuild-image-audit \
    --backup-workbook image_audit_workbook_v1_20260726.csv \
    --sidecar <shard dir>/labels.csv --sidecar <test shard dir>/labels.csv

# 3. HUMAN STEP (protocol sections 7 and 23) — two independent reviewers with
#    orthopedic / MSK imaging experience score >= 400 index images and >= 200
#    outcome records, the >= 200-patient laterality audit is adjudicated, and a
#    reviewer signs outputs/crop_qa_checklist.md with Result = PASS.
python3 -m src.qa_review_app --mode image --reviewer <short name>
python3 -m src.qa_review_app --mode image --merge --reviewers <r1>,<r2>
python3 -m src.crop_qa --score derived-data/cohort/image_audit_workbook.csv

# 4. Train in Colab: notebooks/train_and_score_colab.ipynb
```

**Steps 1, 2, 2b and 4 have run. Step 3, the human review, has not.**
`outputs/crop_qa_checklist.md` still carries an empty sign-off block, with no
reviewer name, no date and a blank `Result`; `outputs/tables/laterality_audit_summary.csv`
and `outcome_audit_summary.csv` report sampling only, with no adjudication, no raw
agreement and no Cohen kappa; the 400-image, 200-patient and 200-record workbooks
are generated and **0% scored** — `outputs/tables/image_audit_summary.csv` does not
exist because nothing real has been scored. Training and the sealed read were
performed anyway, on an explicit author instruction recorded in **D28** and **D31**.
The consequence is that the D2 recovered-index-side cohort, 49.3% of the primary
cohort, is still unvalidated by a human reviewer, and every test-split number rests
on that unvalidated index side. **D22** stays open until the review is done.

#### The Phase 4 rebuild of the section 23(i) sample

The image-audit sample and the tooling around it were rebuilt on 2026-08-10. Nothing
was scored; what changed is what a reviewer will be scoring and how.

* **`derived-data/cohort/image_audit_workbook.csv` is now test-inclusive**: 400 rows ×
  43 columns, **train 273 / val 45 / test 82**, 393 distinct patients. It was previously
  train 344 / val 56 / **test 0** — `config/feasibility.yaml` had always asked for
  `audit_splits: ["train","val","test"]`, but the frame was built from a single
  preprocess sidecar and the 2026-07-29 test run had overwritten the train+val record.
  The frame is now the union of both shard sidecars, 6,071 crops. Stratification is
  unchanged (proportional, largest-remainder, over split × view × contra_side ×
  bilateral/unilateral, seed 20250720, all 24 non-empty strata represented). The 318
  train+val rows are a strict subset of the original 2026-07-26 sample, which is
  preserved verbatim at `image_audit_workbook_v1_20260726.csv`, and every previous
  `P####` assignment in `crop_qa_index_key.csv` is unchanged because the index is burned
  into the titles of the panel PNGs.
* **`src/crop_qa.py` gained `--rebuild-image-audit` and repeatable `--sidecar`.**
  `--rebuild-image-audit` is deliberately narrow: it touches the index key, the workbook,
  the backup and the missing panels and nothing else. A full `python3 -m src.crop_qa`
  would re-render the contact sheet, rewrite `outputs/crop_qa_checklist.md` and redraw
  the other two audits from a DICOM tree that no longer exists. `--sidecar` names the
  shard label files explicitly; without it the rebuild falls back to `preprocess_run.json`
  *and* `preprocess_run_test.json`.
* **`src/qa_review_app.py` (new) is the scoring interface.** A stdlib `http.server` app
  bound to `127.0.0.1`, offline, resumable, keyboard-driven, that writes one CSV per
  reviewer under `derived-data/cohort/scores/` (git-ignored) plus an append-only
  `.jsonl`, flushing before each HTTP response returns. `--mode image|laterality|outcome`,
  `--status` for progress, `--merge --reviewers r1,r2` to fold per-reviewer files back
  into the workbook as `<item>_r1` / `<item>_r2`. It records dwell time per panel, so
  rubber-stamping is auditable.
* **Verdicts are now three, not two.** `NOT_ASSESSABLE` joins `OK` and `ERROR` and is
  excluded from the agreement, kappa and critical-error denominators. It exists because
  **82 of the 400 panels are crop-only**: the 584 pre-existing panels were rendered on
  2026-07-26 with the full film and the source DICOMs are gone, so the test-split panels
  show only the finished 512×512 crop. The workbook carries `panel_has_full_film`
  (318 True / 82 False). Five of the six score items — view, native-knee status, crop
  adequacy, burned-in text, non-knee content — are decidable from a crop; **laterality is
  not**, because `standardize_to_left` mirrors right knees and the border mask removes any
  L/R marker, so a crop is anatomically identical whether the correct or the wrong half was
  taken. **Half-select will therefore be verifiable on 318 of 400 images, not 400** — this
  must reach Methods, Limitations and D22.
* **`--score` now exits non-zero when the gate fails** (2, matching the module's
  "cannot proceed" convention) instead of exiting 0 after logging the failure.

A five-condition hard gate on step 3 used to make step 4 mechanically impossible. It
lived in `notebooks/train_colab.ipynb`, which was **superseded**. Colab wipes `/root` on
teardown and 35 checkpoints are about 5 GB, too much to mirror to Drive inside an
idle timeout, so the two-notebook flow was replaced by
`notebooks/train_and_score_colab.ipynb`, which trains and then reads the sealed
split in one session while the checkpoints are still on local disk. That notebook
runs stage 1 (`m0d_clinical`, `m1_klg`, `m4_fusion`, `m3_image`), then stage 2
(`m2_frontal`, `m4_frontal`, `r1_densenet_frontal`), then a checkpoint audit that
refuses to spend the test split unless all 35 checkpoints exist and every arm is
marked `complete`, then the sealed read. There is no `FROZEN_MANIFEST.json`; the
hand-over index is `derived-data/cohort/train_arms.json` and the read is recorded in
`derived-data/cohort/test_scoring.json`.

### Track D: the sealed read and evaluation — COMPLETE (2026-07-29)

```bash
# THE SEALED READ. Inference only. Refuses to run without the explicit flag.
python3 -m src.score_test  --shard-dir <test shard dir> --confirm-sealed-read

# Metrics, contrasts, calibration, decision curves, subgroups.
python3 -m src.eval_models --split val
python3 -m src.eval_models --split test
```

`score_test` loads the frozen per-seed checkpoints, asserts their training-contract
hash matches `train_arms.json`, refuses any arm not marked `complete`, and averages
per-interval hazards across the five seeds. It never trains, refits, re-tunes,
re-selects an epoch or fits a recalibration on test rows. It writes
`derived-data/cohort/test_hazards_{arm}.npz` plus `test_scoring.json`.

**It does not apply the recalibration.** The `hazards` array in each npz is the raw
seed-averaged ensemble. `score_test` *records* the frozen validation-fitted
horizon-specific recalibration — it copies the parameters verbatim from
`train_arms.json` into `test_scoring.json`, pinning the transform at the moment of the
read — and `eval_models` applies it at render time (`apply_recalibration`,
`src/eval_models.py:641`), which is where the recalibrated slope, calibration-in-the-large
and Brier values in `test_metrics.csv` come from. The two frozen Cox arms (M0, M1) are
published as fitted and are never recalibrated at all (`src/eval_models.py:575`). Anything
that reads `test_hazards_{arm}.npz` directly — including `src/manuscript_figures.py`
panel B — is reading pre-recalibration risks and must apply the transform itself if it
wants the recalibrated scale.

`eval_models --split test` then writes `outputs/tables/test_metrics.csv`,
`test_comparisons.csv`, `test_convergence.csv`, `test_subgroups.csv`,
`test_risk_tertiles.csv` and `test_net_benefit.csv`. Two behaviours differ on the
sealed split and both are deliberate: `assert_sealed_read_is_recorded` refuses to
render unless the read is on record with a matching contract hash, and
`severe_overfit` stops suppressing contrasts, because a test estimate took no part
in the checkpoint selection that flag exists to police. `did_not_converge` still
suppresses on every split. See **D31**.

**This step is spent.** Rerunning `score_test` on this split would be a second
scored read of a split protocol sections 12 and 17 allow to be read once.

## Auditing

- `outputs/protocol_deviations.md` — the canonical deviation register, numbered
  from `D1` and appended to as the study proceeds, including the items flagged for
  an author decision. The range is not typed here; see the note under **Protocol
  conformance quick reference** for why, and for the one-line command that reads it
  off the file.
- `outputs/logs/assumptions.md` — every assumption / ambiguity / deviation.
- `outputs/logs/run.log` — append-only machine run log (never truncated).
- `outputs/protocol_to_column_mapping.csv` — protocol concept → dataset column.
- `outputs/logs/conversion_checksums.csv` — CSV↔Parquet row-count reconciliation.
