# Transfer Verification — Contralateral TKA DICOMs

Generated 2026-07-25 21:08 by `python3 -m src.verify_transfer`.

**VERDICT: PASS — 6,122 of 6,122 manifest images present (100.00%).**

## Destination

- Root: `~/mrkr-dicoms`
- Expected images (config `transfer.expected_n_files`): 6,122
- Manifest entries (de-duplicated): 6,122
- DICOM files found under the root: 6,122
- Non-DICOM files found: 0

## Size

- Total transferred: 34.85 GB
- Median file: 5.81 MB
- Smallest / largest file: 577.73 KB / 28.74 MB
- Minimum acceptable size (`transfer.min_file_bytes`): 1,024 B

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| Manifest matches config | PASS | 6,122 manifest paths vs 6,122 expected |
| DICOM file count | PASS | 6,122 `*.dcm` files found vs 6,122 expected |
| Every manifest path present | PASS | 0 missing |
| No unexpected extra DICOMs | PASS | 0 extra |
| No file below the minimum size | PASS | 0 smaller than 1,024 B |
| DICOM read sample | PASS | 40/40 parsed with pixel data, 0 failed |

## DICOM read sample

- Files opened with pydicom: 40
- Parsed with pixel data: 40
- Failures: 0
- Warnings (codec unavailable locally): 0

- example: shape (3050, 2539), MONOCHROME2, Rows x Columns 3050 x 2539, JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])
- example: shape (2517, 3021), MONOCHROME2, Rows x Columns 2517 x 3021, JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])
- example: shape (2517, 3028), MONOCHROME2, Rows x Columns 2517 x 3028, JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])

## Next step

All checks passed. The DICOMs are complete and readable; proceed to the preprocessing step (`notebooks/preprocess_colab.ipynb` in Colab, or `python3 -m src.preprocess_images` locally).
