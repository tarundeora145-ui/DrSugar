# APTOS DR Classification Training Report

**Generated:** 2026-09-17T21:33:53 (IST)  
**Model:** DR-SUGAR-Baseline-MobileNetV3 v1.0.0

---

## Dataset

| Field | Value |
|---|---|
| Source | APTOS 2019 Blindness Detection |
| Total valid images | 3662 |
| Label file | `train.csv` |
| Images directory | `data/aptos2019/train_images/` |
| Missing files | 0 |
| Corrupt files (0-byte) | 0 |

---

## Split

| Split | Count | Ratio |
|---|---|---|
| Train | 2929 | 80% |
| Validation | 366 | 10% |
| Test | 367 | 10% |
| **Total** | **3662** | **100%** |

- **Random seed:** 42
- **Strategy:** Stratified by DR severity class
- **Leakage check:** Test set held out completely. No model selection performed on test set.

---

## Class Distribution

| Class | Label | Overall | Train | Val | Test |
|---|---|---|---|---|---|
| 0 | No DR | 1805 | 1444 | 180 | 181 |
| 1 | Mild | 370 | 296 | 37 | 37 |
| 2 | Moderate | 999 | 799 | 100 | 100 |
| 3 | Severe | 193 | 154 | 20 | 19 |
| 4 | Proliferative | 295 | 236 | 29 | 30 |

**Imbalance note:** Class 0 (No DR) dominates (~49%). Class 3 (Severe) is rarest (~5%).  
No synthetic oversampling was used. The model was trained with uniform CrossEntropyLoss. Class imbalance is acknowledged as a known limitation.

---

## Preprocessing

Both PyTorch training and Node.js inference use an identical pipeline (`v1-224-imagenet`):

1. Load image and convert to RGB
2. Resize to **224 × 224** (cover/squash — no padding)
3. Convert to float tensor `[0.0, 1.0]` (divide by 255)
4. Normalize per-channel with ImageNet statistics:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
5. Tensor layout: `[Batch, Channel, Height, Width]` = `[1, 3, 224, 224]`

---

## Model Architecture

| Field | Value |
|---|---|
| Architecture | MobileNetV3-Small |
| Parameters | ~2.54M |
| Pretrained weights | ImageNet (DEFAULT via torchvision) |
| Output head | Linear(1024 → 5) replacing final classifier layer |
| Input size | 224 × 224 × 3 |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Epochs trained | 3 |
| Best model selected by | Max Val Macro-F1 (handles imbalance better than accuracy) |

---

## Training Results

All values from actual execution logs.

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro-F1 |
|---|---|---|---|---|
| 1 | 0.7772 | 0.7334 | 0.7459 | 0.4136 |
| 2 | 0.5451 | 0.6715 | 0.7732 | 0.5428 |
| 3 | 0.4523 | 0.6145 | 0.7678 | **0.5545** ← Best |

Best checkpoint saved after epoch 3 (`best_model.pth`).

---

## Held-Out Test Results

Evaluated on `split_test.csv` (367 images, never used during training or model selection).

| Metric | Value |
|---|---|
| Accuracy | **0.7902** |
| Macro Precision | 0.6268 |
| Macro Recall | 0.5669 |
| Macro F1 | **0.5726** |

### Per-Class Performance (from confusion matrix)

| Class | Predicted correctly |
|---|---|
| 0 (No DR) | 178 / 181 |
| 1 (Mild) | 11 / 37 |
| 2 (Moderate) | 81 / 100 |
| 3 (Severe) | 4 / 19 |
| 4 (Proliferative) | 16 / 30 |

---

## Confusion Matrix (5-class)

Rows = True label, Columns = Predicted label (classes 0–4):

|  | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|---|---|---|---|---|---|
| **True 0** | 178 | 2 | 0 | 0 | 1 |
| **True 1** | 5 | 11 | 19 | 1 | 1 |
| **True 2** | 0 | 0 | 81 | 6 | 13 |
| **True 3** | 1 | 0 | 8 | 4 | 6 |
| **True 4** | 0 | 1 | 8 | 5 | 16 |

---

## Referable DR Results

**Definition:** Referable DR = predicted severity ≥ 2 (Moderate, Severe, or Proliferative)

| Metric | Value |
|---|---|
| TP (True Positive) | 147 |
| TN (True Negative) | 196 |
| FP (False Positive) | 22 |
| FN (False Negative) | 2 |
| **Sensitivity** | **0.9866** |
| **Specificity** | **0.8991** |
| Precision | 0.8698 |
| F1 (Referable DR) | 0.9245 |

---

## ONNX Verification

| Check | Result |
|---|---|
| Export format | ONNX opset 18 (legacy trace path, dynamo=False) |
| ONNX checker | PASSED |
| PyTorch vs ONNX max diff | **0.00000489** |
| Tolerance threshold | atol=1e-4, rtol=1e-3 |
| Assertion | PASSED |

---

## Node Inference Verification

Test image: `data/aptos2019/test_images/0005cfc8afb6.png` (236,005 bytes)

| Check | Result |
|---|---|
| Image loaded | YES |
| Preprocessing succeeded | YES |
| ONNX model loaded (onnxruntime-node) | YES |
| Inference executed | YES |
| Predicted class | 2 (Moderate) |
| Probabilities | [0.0365, 0.1795, **0.4929**, 0.0461, 0.2450] |
| Probabilities sum | **1.000000** |
| referableDR | **true** (class ≥ 2) |
| Model version returned | 1.0.0 |

---

## SQLite Verification

Record persisted to `screenings` table:

| Field | Value |
|---|---|
| id | 1 |
| patient_id | test-patient-node-inference |
| image_path | `C:\...\data\aptos2019\test_images\0005cfc8afb6.png` |
| status | COMPLETED |
| result_grade | 2 |
| created_at | 2026-09-17 16:07:37 |

---

## Explainability

Grad-CAM and gradient-based explanation: **NOT IMPLEMENTED** in this phase.  
Status: `EXPLANATION UNAVAILABLE`

---

## Limitations

1. **3 epochs only** — CPU training constrained epochs to 3. A GPU-trained model over 15–30 epochs would yield substantially higher macro-F1, particularly for rare classes (Severe, Mild).
2. **Class imbalance** — No class weighting or oversampling applied. Classes 1 (Mild) and 3 (Severe) have low per-class recall.
3. **No augmentation** — Training used basic resize + normalize only. Data augmentation (flips, rotations, colour jitter) would improve generalization.
4. **Sensitivity target** — The referable DR sensitivity of **98.66%** exceeds the project target of >90%. Specificity of **89.91%** is close to the >85% target. These are baseline results, not clinical validation.
5. **Not clinically validated** — This is a research prototype. All predictions require ophthalmologist review.

---

## Reproducibility

| Parameter | Value |
|---|---|
| Random seed | 42 (fixed in `train_test_split`) |
| PyTorch | 2.14.0+cpu |
| torchvision | 0.29.0 |
| ONNX | 1.22.0 |
| onnxruntime | 1.30.0 |
| Python | 3.14.3 |
| Split files | `data/aptos2019/split_{train,val,test}.csv` |
| Model checkpoint | `models/dr_classification/best_model.pth` |
| ONNX model | `models/dr_classification/model.onnx` |
| Metadata | `models/dr_classification/metadata.json` |
