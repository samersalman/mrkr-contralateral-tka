# Preoperative radiograph predicts contralateral knee arthroplasty after unilateral total knee arthroplasty

Analysis code for a study that predicts contralateral knee arthroplasty within five years
from the preoperative radiograph of the contralateral knee, benchmarked against a clinical
model and against that clinical model plus a radiographic osteoarthritis grade.

This repository contains **code and aggregate results only**. No patient-level data of any
kind is included here, and none is redistributed. See [Data](#data) below.

## Data

The study used the **Emory Knee Radiograph (MRKR) dataset**, a de-identified archive of
503,261 knee radiographs from 83,011 patients with linked diagnosis, procedure and
patient-reported pain records.

* Data descriptor: Price BJ, Gichoya J, Chavoshi M, et al. The Emory Knee Radiograph (MRKR)
  dataset. *J Imaging Inform Med.* 2026. doi:[10.1007/s10278-026-01902-6](https://doi.org/10.1007/s10278-026-01902-6)
* Hosted on the AWS Open Data platform (~2.3 TB): <https://registry.opendata.aws/mrkr>
* Access registration: <https://data.hitilab.com>
* Dataset structure and metadata schema: <https://github.com/Emory-HITI/MRKR>
* Licensed by its authors under CC BY-SA

MRKR is already public, so this repository does not mirror it. Running the pipeline end to
end requires obtaining the dataset from the sources above.

## What is deliberately not in this repository

| Excluded | Why |
| --- | --- |
| `metadata-files/`, `derived-data/`, `DICOMs/` | Patient-level records, radiographs and per-patient model predictions |
| Image transfer manifests, the selected-studies table | Row-per-patient tables carrying anonymised patient keys and DICOM UIDs |
| The crop QA contact sheet | Rendered radiograph crops |
| Trained-artefact archives | Contain per-patient predicted hazards |
| The manuscript document and `src/make_manuscript.py` | The manuscript itself is not part of the code release |
| Figure rendering code and the rendered figures | Belong to the manuscript rather than to the analysis |
| The full run log | Local filesystem paths only, no scientific content |

One path in `config/feasibility.yaml` (`transfer.dest_root`) is a placeholder, because the
original value was a local cloud-drive location. Set it to wherever you stage DICOMs.

## Layout

```
src/                  pipeline modules, each run as `python -m src.<name>`
  index_tka.py          index arthroplasty identification and laterality
  outcomes.py           contralateral arthroplasty outcome and censoring
  followup.py           landmark follow-up scaffold
  features_clinical.py  clinical feature table
  splits.py             patient-level train/validation/test partition
  model_clinical.py     penalised Cox comparators (M0, M1)
  preprocess_images.py  radiograph crop pipeline
  crop_qa.py            crop quality audit
  train_model.py        image and fusion model training
  score_test.py         sealed-split scoring
  eval_models.py        metrics, comparisons, calibration, decision curves
config/feasibility.yaml every analysis parameter, in one file
notebooks/            Colab notebooks for preprocessing and training
tests/                pytest suite
outputs/              aggregate reports and tables produced by the above
```

`config/feasibility.yaml` is the single source of every analysis parameter. No threshold,
window or model setting is hard-coded in a module.

## Reproducing

```bash
python -m pip install -r requirements.txt          # analysis
python -m pip install -r requirements-training.txt # image model training
```

Modules are run from the repository root in pipeline order, for example:

```bash
python -m src.index_tka
python -m src.outcomes
python -m src.followup
python -m src.features_clinical
python -m src.splits
python -m src.model_clinical
python -m src.eval_models --split val
```

Deep-learning training runs in Colab via `notebooks/`. The test split is sealed by three
independent `forbid_test_split` guards and is only readable after the models, ensemble rule,
thresholds and analysis script are frozen.

## Tests

```bash
python -m pytest tests/
```

On a bare checkout the suite reports **549 passed, 7 failed, 14 skipped**. All seven failures
are in `tests/test_eval_models.py`, and all seven have the same cause: they cross-check the
evaluation module's output schemas against `src/make_manuscript.py`, which is not part of this
release. Nothing else in the suite depends on it.

Excluding that one module, the suite is **411 passed, 1 skipped, 0 failed** with no data
present at all:

```bash
python -m pytest tests/ --ignore=tests/test_eval_models.py
```

## Analysis notes

* `outputs/protocol_deviations.md` is the canonical register of every deviation from the
  pre-specified protocol, including the ones that were declined.
* `outputs/logs/assumptions.md` records the analytical decisions that were delegated rather
  than pre-specified.
* `outputs/tables/` holds the aggregate tables behind the reported results, including
  discrimination, calibration, decision-curve and subgroup outputs for both splits. Every
  table is a summary; the largest has 483 rows, against a cohort of 3,709 patients.

## License

MIT, see [LICENSE](LICENSE). The MRKR dataset itself is licensed separately by its authors
under CC BY-SA and is not covered by this license.
