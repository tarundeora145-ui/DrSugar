# DR-SUGAR — DRIVE Segmentation Readiness Report

> [!WARNING]
> **Audit only. No training performed. No datasets modified. No models overwritten.**

---

## 1. Dataset Structure

```
data/drive/
├── training/
│   ├── images/        20 × .tif  (fundus images, RGB)
│   ├── 1st_manual/    20 × .gif  (vessel ground-truth binary masks)
│   └── mask/          20 × .gif  (field-of-view / FOV masks)
└── test/
    ├── images/        20 × .tif  (fundus images, RGB)
    └── mask/          20 × .gif  (FOV masks — NO vessel GT)
```

---

## 2. Image & Mask Counts

| Set | Images | Vessel GT | FOV Masks |
|---|---|---|---|
| Training | **20** | **20** | **20** |
| Test | **20** | **0** (none — by design) | **20** |

---

## 3. Correspondence Validation

All 20 training images have a correctly named, matching vessel GT mask and FOV mask.
Naming convention verified: `{ID}_training.tif` ↔ `{ID}_manual1.gif` ↔ `{ID}_training_mask.gif`.

All 20 test images have correctly named FOV masks.
Naming convention: `{ID}_test.tif` ↔ `{ID}_test_mask.gif`.

**Result: PASS — No correspondence errors or missing files.**

---

## 4. Image Dimensions

| Property | Value |
|---|---|
| Training image size | **(584, 565)** H×W, RGB |
| Vessel GT mask size | **(584, 565)** H×W |
| FOV mask size | **(584, 565)** H×W |
| All dimensions consistent | **YES** — all 20 training sets are identical |
| Mask pixel values | `{0, 255}` — clean binary, no corruption |
| FOV pixel values | `{0, 255}` — clean binary, no corruption |

---

## 5. FOV & Foreground Statistics

The FOV (retinal field-of-view) mask defines the valid circular region within each image. Vessel statistics are computed **inside** the FOV only.

| Statistic | Value |
|---|---|
| Mean FOV coverage (% of full image) | **~68.9%** |
| Vessel foreground (% inside FOV) — Min | **8.74%** (Image 31) |
| Vessel foreground (% inside FOV) — Max | **16.79%** (Image 24) |
| Vessel foreground (% inside FOV) — Mean | **12.54%** |
| Vessel foreground (% inside FOV) — Median | **12.52%** |

> [!NOTE]
> The vessel-to-background ratio (~1:8) represents moderate class imbalance. Standard loss functions (Binary Cross-Entropy) will work but may under-segment thin vessels. A combined loss (BCE + Dice) is recommended.

**Per-image Details:**

| ID | Image HW | FOV% | Vessel% (in FOV) | Mask Values | FOV Values |
|---|---|---|---|---|---|
| 21 | (584, 565) | 68.4% | 10.93% | [0, 255] | [0, 255] |
| 22 | (584, 565) | 69.0% | 13.09% | [0, 255] | [0, 255] |
| 23 | (584, 565) | 69.2% |  9.52% | [0, 255] | [0, 255] |
| 24 | (584, 565) | 69.0% | 16.79% | [0, 255] | [0, 255] |
| 25 | (584, 565) | 68.9% | 13.93% | [0, 255] | [0, 255] |
| 26 | (584, 565) | 68.2% | 12.25% | [0, 255] | [0, 255] |
| 27 | (584, 565) | 69.0% | 12.76% | [0, 255] | [0, 255] |
| 28 | (584, 565) | 68.9% | 14.18% | [0, 255] | [0, 255] |
| 29 | (584, 565) | 68.9% | 12.21% | [0, 255] | [0, 255] |
| 30 | (584, 565) | 68.9% | 11.39% | [0, 255] | [0, 255] |
| 31 | (584, 565) | 69.0% |  8.74% | [0, 255] | [0, 255] |
| 32 | (584, 565) | 68.2% | 12.00% | [0, 255] | [0, 255] |
| 33 | (584, 565) | 69.0% | 11.72% | [0, 255] | [0, 255] |
| 34 | (584, 565) | 68.7% | 14.25% | [0, 255] | [0, 255] |
| 35 | (584, 565) | 69.0% | 12.57% | [0, 255] | [0, 255] |
| 36 | (584, 565) | 68.9% | 15.79% | [0, 255] | [0, 255] |
| 37 | (584, 565) | 68.9% | 12.69% | [0, 255] | [0, 255] |
| 38 | (584, 565) | 68.6% | 12.59% | [0, 255] | [0, 255] |
| 39 | (584, 565) | 68.9% | 12.46% | [0, 255] | [0, 255] |
| 40 | (584, 565) | 68.8% | 11.01% | [0, 255] | [0, 255] |

---

## 6. Test Set — Important Constraint

> [!IMPORTANT]
> The DRIVE test set has **no public vessel ground-truth**. The official first annotator's labels are withheld by the DRIVE challenge organizers. The 20 test images **MUST NOT** be used for supervised model selection, threshold tuning, or final performance reporting unless a separate evaluation server is used.

**Correct usage:**
- **Training & validation:** Use the 20 training images only (e.g., with an 18/2 or 16/4 split, or 5-fold cross-validation within the 20 training images).
- **Test set:** Use only for qualitative visualization of predictions. Do not report test metrics as supervised evaluation.

---

## 7. Existing Code Status

### Backend — TypeScript (Dataset Registration)
| File | Status |
|---|---|
| `server/src/datasets/loaders/DriveLoader.ts` | **Structural code only.** Handles DB registration, file scanning, and correspondence linking. Does **not** contain any segmentation model, training, or inference logic. Correctly notes vessel masks as `VESSEL_MASK` annotations with DR grade = -1 (unspecified). |

### ML Scripts — Python
| File | Status |
|---|---|
| `scripts/train_dr_grading.py` | APTOS grading script. No DRIVE code. |
| `scripts/train_idrid_grading*.py` | IDRiD grading scripts. No DRIVE code. |
| `scripts/export_*.py` | ONNX exporters for APTOS/IDRiD. No DRIVE code. |
| `scripts/prepare_idrid_dataset.py` | IDRiD preparation. No DRIVE code. |

> [!NOTE]
> **There is no existing DRIVE segmentation training script.** No `UNet`, no vessel segmentation loader, no training loop, and no DRIVE-specific model exists anywhere in the codebase. Starting from scratch is required.

### Inference Backend — TypeScript
`server/src/ml/inference.ts` is specific to APTOS/IDRiD classification (ONNX, 224×224 input, 5-class softmax). It has no segmentation inference pathway.

---

## 8. GPU & Environment Status

| Item | Status |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA | 13.2 |
| PyTorch | 2.14.0+cu132 |
| Torchvision | 0.29.0+cu132 |
| CUDA Available | **True** |
| onnxruntime | 1.30.0 ✓ |
| onnx | 1.22.0 ✓ |
| Pillow | 12.3.0 ✓ |
| numpy | 2.5.3 ✓ |
| pandas | 3.0.5 ✓ |
| scikit-learn | 1.9.1 ✓ |
| `segmentation_models_pytorch` | **NOT installed** |
| `albumentations` | **NOT installed** |
| `opencv-python (cv2)` | **NOT installed** |

---

## 9. Recommended Next Experiment — DRIVE V1

### Architecture
**U-Net** with an encoder-decoder structure is the de-facto standard for DRIVE vessel segmentation. Given the fully controlled IDRiD experiments using `MobileNetV3-Small`, a matching approach is:
- **Encoder:** `MobileNetV3-Small` (ImageNet pretrained) used as the encoder backbone.
- **Decoder:** U-Net style upsampling decoder with skip connections.
- This can be implemented directly in PyTorch without external segmentation libraries.

### Cross-Validation Strategy
With only 20 training images, standard held-out val splits are extremely small. Recommended:
- **5-fold cross-validation** within the 20 training images (4 folds of 16 train + 4 val).
- Report mean ± std of Dice score across folds.

### Evaluation Metrics (within training set cross-validation only)
- **Dice Coefficient** (primary)
- **IoU / Jaccard Index**
- **Sensitivity / Recall** (vessel pixel-level)
- **Specificity** (background pixel-level)
- **AUC-ROC** on pixel probabilities

### Loss Function
- **BCE + Dice Loss** combined — standard for this class of imbalance.

---

## 10. Blockers

| Item | Severity | Resolution |
|---|---|---|
| No DRIVE training script exists | **HIGH** — required before training | Write `scripts/train_drive_segmentation.py` |
| `opencv-python` not installed | LOW — optional for augmentation/visualization | Install only if needed; `Pillow` + `torchvision` transforms can replace it |
| `albumentations` not installed | LOW | Not required; `torchvision.transforms` and `torchvision.transforms.v2` are available |
| `segmentation_models_pytorch` not installed | LOW | Not required; U-Net can be implemented directly in PyTorch |
| No public DRIVE test vessel GT | **MUST understand** — test set cannot be used for metric evaluation | Cross-validate strictly within 20 training images |

---

## 11. Summary

**The DRIVE dataset is fully provisioned and verified:**
- 20/20 training images, vessel masks, and FOV masks all present, correctly named, dimensionally consistent, and corruption-free.
- 20/20 test images and FOV masks present.
- All masks are clean binary ({0, 255}).
- Vessel foreground is ~12.5% within FOV (moderate imbalance — manageable with Dice/BCE combined loss).

**The environment is GPU-ready** with PyTorch 2.14.0+cu132 and CUDA 13.2 on the RTX 5060.

**There is no existing segmentation code.** The DRIVE V1 training script must be written from scratch.
