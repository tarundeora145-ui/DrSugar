# DR-SUGAR — All Datasets Provisioning Report

**Generated:** 2026-09-17T22:51 IST  
**Operator:** Automated extraction + validation pipeline

---

## Summary Table

| Dataset | Task | Images Found | Valid | Labels | Masks / Annotations | Status | Training Readiness |
|---|---|---:|---:|---:|---:|---|---|
| APTOS 2019 | DR Grading (5-class) | 3662 (train) + 1928 (test) = 5590 | 3662 (train indexed) | 3662 | — | **CONNECTED** | ✅ READY (previously trained) |
| DRIVE | Vessel Segmentation | 20 (train) + 20 (test) = 40 | 20 (train indexed) | 20 vessel masks (train) | 20 FOV masks (train), 20 FOV masks (test) | **CONNECTED** | ✅ READY (train only indexed) |
| IDRiD (B) | DR Grading (5-class) | 413 (train) + 103 (test) = 516 | 413 (train indexed) | 413 (train) + 103 (test) | 413 DME risk annotations (train) | **CONNECTED** | ✅ READY (train only indexed) |
| IDRiD (A) | Lesion Segmentation | 54 (train) + 27 (test) = 81 | 54 (train) | — | 241 masks (MA/HE/EX/SE/OD) | Not yet indexed by loader | 🔶 Partial readiness |
| IDRiD (C) | OD + Fovea Localiz. | 413 (train) + 103 (test) = 516 | 413 (train) | — | 515 OD + 515 Fovea coords (train) | Not yet indexed by loader | 🔶 Partial readiness |
| Messidor-2 | External Validation | 1748 | 1748 | 0 (no DR grades in release) | 874 image-pair annotations | **PARTIAL** | 🔶 Images available, no DR labels |

---

## 1. APTOS 2019

**Filesystem path:** `data/aptos2019/`  
**Archives used:** `data/aptos2019/aptos2019.zip` (extracted previously)  
**Extraction:** COMPLETE (done in prior session)

| Metric | Value |
|---|---|
| Training images | 3662 |
| Test images | 1928 |
| Valid training images | 3662 |
| Corrupt (0-byte) images | 0 |
| Missing files | 0 |
| Labels (`train.csv`) | 3662 |
| Labels matched to images | 3662 (100%) |
| Unmatched labels | 0 |
| Unmatched images | 0 |
| SQLite status | `CONNECTED` |
| last_scanned | 2026-09-17 |

**Grade distribution (training set):**

| Grade | Class | Count |
|---|---|---|
| 0 | No DR | 1805 |
| 1 | Mild | 370 |
| 2 | Moderate | 999 |
| 3 | Severe | 193 |
| 4 | Proliferative | 295 |

**Notes:** APTOS was not modified in this session. All 3662 images remain intact. The ONNX classification model trained on this dataset is already deployed.

---

## 2. DRIVE

**Filesystem path:** `data/drive/`  
**Archives used:** `data/drive/test.zip`, `data/drive/training.zip`  
**Extraction:** SUCCESS — exit code 0

### Structure Extracted

```
data/drive/
├── training/
│   ├── images/         20 × .tif  (IDs: 21–40)
│   ├── 1st_manual/     20 × .gif  (vessel ground truth masks)
│   └── mask/           20 × .gif  (FOV masks)
└── test/
    ├── images/         20 × .tif  (IDs: 01–20)
    └── mask/           20 × .gif  (FOV masks — NO vessel GT for test set)
```

| Metric | Value |
|---|---|
| Total fundus images on disk | 40 (20 training + 20 test) |
| Training images (indexed) | 20 |
| Training vessel masks (1st_manual) | 20 |
| Training FOV masks | 20 |
| Test images (on disk, not indexed) | 20 |
| Test FOV masks | 20 |
| Test vessel masks | 0 (not publicly released) |
| Corrupt images | 0 |
| Image/mask correspondence errors | 0 |
| SQLite status | `CONNECTED` |

**Loader fix applied:** `BaseLoader.ts` extended to include `.gif` in the image extension whitelist so DRIVE vessel masks (standard DRIVE format) are correctly scanned and registered.

**Loader counts (SQLite):**
- `dataset_files` type=fundus: 20
- `dataset_files` type=mask: 20
- `dataset_labels`: 20 (mask registrations, dr_grade=-1 as vessel task)
- `dataset_annotations`: 20 (VESSEL_MASK annotations linking image ↔ mask path)

---

## 3. IDRiD

**Filesystem path:** `data/idrid/`  
**Archives used:**
- `data/idrid/A. Segmentation.zip`
- `data/idrid/B. Disease Grading.zip`
- `data/idrid/C. Localization.zip`  

**Extraction:** SUCCESS — all 3 archives extracted, exit code 0

### Structure Extracted

```
data/idrid/
├── A. Segmentation/
│   ├── 1. Original Images/
│   │   ├── a. Training Set/    54 × .jpg
│   │   └── b. Testing Set/     27 × .jpg
│   └── 2. All Segmentation Groundtruths/
│       ├── a. Training Set/
│       │   ├── 1. Microaneurysms/    54 × .tif (MA)
│       │   ├── 2. Haemorrhages/      53 × .tif (HE)
│       │   ├── 3. Hard Exudates/     54 × .tif (EX)
│       │   ├── 4. Soft Exudates/     26 × .tif (SE — sparse, not all images)
│       │   └── 5. Optic Disc/        54 × .tif (OD)
│       └── b. Testing Set/
│           ├── 1. Microaneurysms/    27 × .tif
│           ├── 2. Haemorrhages/      27 × .tif
│           ├── 3. Hard Exudates/     27 × .tif
│           ├── 4. Soft Exudates/     14 × .tif
│           └── 5. Optic Disc/        27 × .tif
├── B. Disease Grading/
│   ├── 1. Original Images/
│   │   ├── a. Training Set/    413 × .jpg
│   │   └── b. Testing Set/     103 × .jpg
│   └── 2. Groundtruths/
│       ├── a. IDRiD_Disease Grading_Training Labels.csv   (413 records)
│       └── b. IDRiD_Disease Grading_Testing Labels.csv    (103 records)
└── C. Localization/
    ├── 1. Original Images/
    │   ├── a. Training Set/    413 × .jpg
    │   └── b. Testing Set/     103 × .jpg
    └── 2. Groundtruths/
        ├── 1. Optic Disc Center Location/
        │   ├── a. IDRiD_OD_Center_Training Set_Markups.csv   (515 records)
        │   └── b. IDRiD_OD_Center_Testing Set_Markups.csv
        └── 2. Fovea Center Location/
            ├── IDRiD_Fovea_Center_Training Set_Markups.csv   (515 records)
            └── IDRiD_Fovea_Center_Testing Set_Markups.csv
```

### B. Disease Grading — Detailed Validation

| Metric | Value |
|---|---|
| Total images on disk | 516 (413 training + 103 test) |
| Training images | 413 |
| Training labels | 413 |
| Testing images | 103 |
| Testing labels | 103 |
| Images without label | 0 |
| Labels without image | 0 |
| DME risk annotations (training) | 413 |
| Corrupt images | 0 |
| SQLite status | `CONNECTED` |

**Grade distribution (training, actual from CSV):**

| Grade | Class | Count |
|---|---|---|
| 0 | No DR | 134 |
| 1 | Mild | 20 |
| 2 | Moderate | 136 |
| 3 | Severe | 74 |
| 4 | Proliferative | 49 |

### A. Segmentation — Detailed Validation

| Lesion type | Training masks | Testing masks | Notes |
|---|---|---|---|
| Microaneurysms (MA) | 54 | 27 | Complete |
| Haemorrhages (HE) | 53 | 27 | 1 training image has no HE mask (normal — no HE present) |
| Hard Exudates (EX) | 54 | 27 | Complete |
| Soft Exudates (SE) | 26 | 14 | Sparse — SE only present in subset of images |
| Optic Disc (OD) | 54 | 27 | Complete |

**Note on SE sparsity:** 28 of 54 training images have no SE mask. This is correct per the IDRiD dataset specification — SE masks are only provided when soft exudates are actually present.

### C. Localization — Detailed Validation

| Type | Training records | Testing records |
|---|---|---|
| Optic Disc center (X,Y) | 515 | 515 (Note: 515 exact rows in CSV, exceeds 413/103 images) |
| Fovea center (X,Y) | 515 | 515 (Note: 515 exact rows in CSV, exceeds 413/103 images) |

**Loader fix applied:** `IdridLoader.ts` path strings updated from incorrect legacy paths (`Disease Grading/Original Images/Training Set`) to actual extracted paths (`B. Disease Grading/1. Original Images/a. Training Set` and `B. Disease Grading/2. Groundtruths/...`). This is a path-only fix. No data was moved.

**Note:** IDRiD Segmentation (A) and Localization (C) data are present on disk but are not yet indexed by the current loader (which focuses on B. Disease Grading). These will require loader extensions in a future phase.

---

## 4. Messidor-2

**Filesystem path:** `data/messidor2/`  
**Archives used:** `IMAGES.zip.001`, `IMAGES.zip.002`, `IMAGES.zip.003`, `IMAGES.zip.004` (multipart)  
**Exact filenames present:**
- `IMAGES.zip.001`
- `IMAGES.zip.002`
- `IMAGES.zip.003`
- `IMAGES.zip.004`
- `messidor-2.csv`
- `IMAGES/` (directory)

**Original parts:** Preserved on disk, not renamed or deleted  
**Extraction method:** Python binary concatenation (parts streamed in sequence → temp_combined.zip → extracted → temp file deleted)  
**Extraction:** SUCCESS — exit code 0

| Metric | Value |
|---|---|
| Images extracted to `IMAGES/` | 1748 |
| Image formats | 1058 × .png, 690 × .jpg |
| Corrupt (0-byte) images | 0 |
| `messidor-2.csv` present | YES |

### messidor-2.csv Analysis

| Field | Value |
|---|---|
| Delimiter | `;` (semicolon) |
| Columns | `left`, `right` |
| Rows | 874 (image pairs) |
| Images referenced | 1748 (874 left + 874 right) |
| Images found in IMAGES/ | 1748 (100% match) |
| Missing images from CSV | 0 |
| Images in IMAGES/ not in CSV | 0 |

**Important:** `messidor-2.csv` contains only **left/right image pairing** metadata. It does **not** contain adjudicated DR grade labels. The standard Messidor-2 image release does not bundle DR grade ground truth. This is not a data error — it is a known limitation of the public Messidor-2 release.

**SQLite status:** `PARTIAL` — images fully present and indexed, but DR labels = 0.

**Loader fix applied:** `MessidorLoader.ts` updated to read `messidor-2.csv` (actual filename) instead of `messidor_data.csv` (expected by old loader). Correctly annotates image-pairing metadata and accurately sets `PARTIAL` status. No fabricated grades.

---

## SQLite State After Final Validation

| Dataset ID | Status | Images (dataset_files type=fundus) | Masks (dataset_files type=mask) | Labels (dataset_labels) | Annotations (dataset_annotations) |
|---|---|---|---|---|---|
| aptos2019 | CONNECTED | 3662 (train) | 0 | 3662 | 0 |
| drive | CONNECTED | 20 (train) | 20 (train) | 20 (vessel masks) | 20 |
| idrid | CONNECTED | 413 (train) | 0 | 413 | 413 |
| messidor2 | PARTIAL | 1748 (all) | 0 | 0 | 874 |

*Note: Database counts strictly represent files successfully indexed by current loaders. For example, DRIVE testing images (20) are currently on the filesystem but unindexed in SQLite.*

**Stale records:** None — `clearDBRecords()` is called before each `validate()` run.  
**Orphaned records:** None — all records tied to validated dataset IDs with FK constraints.  
**Duplicate records:** None — cleared before each validation.

---

## Loader Modifications Summary

| File | Change | Reason |
|---|---|---|
| `server/src/datasets/loaders/BaseLoader.ts` | Added `.gif` to extension whitelist | DRIVE vessel masks are `.gif` per standard release |
| `server/src/datasets/loaders/IdridLoader.ts` | Updated 2 path strings to match actual extracted structure | ZIP extracts to `B. Disease Grading/1. Original Images/a. Training Set` not `Disease Grading/Original Images/Training Set` |
| `server/src/datasets/loaders/MessidorLoader.ts` | Updated CSV filename (`messidor-2.csv`), removed fake DR grade parsing, added image-pair annotation | Actual release has `messidor-2.csv` with pairing data, no DR labels |

No source dataset files were moved, renamed, or deleted.

---

## Training Readiness Assessment

| Dataset | Use Case | Readiness |
|---|---|---|
| APTOS 2019 | DR classification (5-class) | ✅ CONNECTED — model already trained |
| DRIVE | Retinal vessel segmentation | ✅ CONNECTED — 20 image/mask pairs with perfect correspondence |
| IDRiD B | DR classification (5-class, Indian population) | ✅ CONNECTED — 413 labeled images |
| IDRiD A | Lesion segmentation (MA/HE/EX/SE/OD) | 🔶 Data present, loader not yet extended |
| IDRiD C | OD/Fovea localization | 🔶 Data present, loader not yet extended |
| Messidor-2 | External validation / generalization | 🔶 1748 images available, no DR labels — use for distribution analysis only |

---

## What Was NOT Done

- No model training performed
- No fake labels, masks, or annotations generated
- No dataset files deleted or renamed
- No npm run dev fixes
- No Grad-CAM implementation
- No Simulink integration
- No metric fabrication
