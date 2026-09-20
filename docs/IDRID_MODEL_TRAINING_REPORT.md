# DR-SUGAR — IDRiD Disease Grading Model Training Report

> [!WARNING]  
> **Research/prototype model evaluation only.** This model is NOT clinically validated. Do not use for actual medical diagnosis.

---

## 1. Dataset & Splits

**Dataset**: IDRiD B. Disease Grading  
**Total Images Evaluated**: 516

An 80/20 train/validation split was created from the official 413-image training dataset to ensure the 103-image official test set remained completely sequestered until the final evaluation phase.

* **Training Set**: 330 images
* **Validation Set**: 83 images
* **Held-Out Test Set**: 103 images

**Class Distribution (Total Training Set - 413)**:
- Grade 0: 134
- Grade 1: 20
- Grade 2: 136
- Grade 3: 74
- Grade 4: 49

---

## 2. Model Architecture & Preprocessing

* **Architecture**: MobileNetV3-Small
* **Pre-training**: ImageNet defaults
* **Classification Head**: 5-class linear layer
* **Image Resizing**: 224x224
* **Normalization**: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
* **Augmentation**: Horizontal Flip, Rotation (±10°), ColorJitter (brightness 0.1, contrast 0.1)

---

## 3. Training Configuration

* **Hardware**: CPU
* **Random Seed**: 42
* **Loss Function**: Class-Weighted Cross Entropy (to combat extreme Grade 1 imbalance)
* **Optimizer**: Adam
* **Learning Rate**: 1e-4
* **Batch Size**: 16
* **Epochs**: 15
* **Early Stopping Criterion**: Best Validation Macro F1 score

---

## 4. Final Held-Out Test Metrics (103 Images)

The model was evaluated exactly once on the official test set.

| Metric | Score |
|---|---|
| **Accuracy** | 45.63% |
| **Macro Precision** | 41.14% |
| **Macro Recall** | 36.07% |
| **Macro F1** | 32.76% |
| **Weighted F1** | 42.45% |

### Per-Class Results

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| **0 - No DR** | 65.22% | 44.12% | 52.63% |
| **1 - Mild** | 0.00% | 0.00% | 0.00% |
| **2 - Moderate** | 43.40% | 71.88% | 54.12% |
| **3 - Severe** | 66.67% | 10.53% | 18.18% |
| **4 - Proliferative**| 30.43% | 53.85% | 38.89% |

### Referable DR Task (Severity ≥ 2)

* **Sensitivity (Recall)**: 90.63%
* **Specificity**: 46.15%
* **Precision**: 73.42%
* **F1 Score**: 81.12%

*Confusion Matrix (Referable DR)*:
- True Positive (TP): 58
- True Negative (TN): 18
- False Positive (FP): 21
- False Negative (FN): 6

---

## 5. Confusion Matrix (5-Class)

| True \ Pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | 15 | 1 | 17 | 0 | 1 |
| **1** | 2 | 0 | 3 | 0 | 0 |
| **2** | 4 | 0 | 23 | 1 | 4 |
| **3** | 0 | 0 | 6 | 2 | 11 |
| **4** | 2 | 0 | 4 | 0 | 7 |

---

## 6. Calibration & Error Analysis

* **Calibration**: Calibration not implemented. Raw softmax probabilities provided.
* **Error Analysis**: Detailed image-level predictions, probabilities, and confidence scores have been saved to `models/idrid/test_predictions.csv` for downstream error analysis.

---

## 7. ONNX & Backend Integration Verification

* **ONNX Export**: SUCCESS. The ONNX checker passed. PyTorch vs. ONNX output mismatch was measured at a maximum of `0.00000280`, which is well within acceptable tolerance thresholds.
* **Node Inference Verification**: SUCCESS. The existing inference backend correctly loaded the `models/idrid/model.onnx` file. 
* **SQLite Persistence**: SUCCESS. `test_idrid_inference.ts` successfully executed a full forward pass on an IDRiD image and persisted the result to the `screenings` table (ID 3).

---

## 8. Limitations

1. **Class 1 (Mild) Extinction**: Despite class-weighting, the model failed to recall any Grade 1 cases in the test set. This is a common artifact of extreme class imbalance (only 20 instances in training) combined with brief training duration on a small dataset.
2. **CPU Constraints**: Training was artificially constrained to 15 epochs on a lightweight architecture (MobileNetV3) due to lack of CUDA acceleration on the host hardware.
3. **Over-prediction of Grade 2**: The model heavily biases toward Grade 2 for uncertain non-referable cases, sacrificing specificity for higher sensitivity. 

## 9. Reproducibility Instructions

```bash
# 1. Generate stratified dataset splits (seed=42)
python scripts/prepare_idrid_dataset.py

# 2. Train the model
python scripts/train_idrid_grading.py

# 3. Export to ONNX and verify tolerances
python scripts/export_idrid_model.py

# 4. Run Node Inference integration test
npx tsx server/src/ml/test_idrid_inference.ts
```
