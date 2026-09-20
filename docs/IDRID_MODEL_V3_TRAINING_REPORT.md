# DR-SUGAR — IDRiD Disease Grading V3 Training Report

> [!WARNING]
> **Research/prototype model evaluation only. Not clinically validated.** Do not use for actual medical diagnosis. This report strictly separates the validation selection metrics from the locked official test set evaluation.

---

## 1. V3 Objective & Strategy
The V3 experiment aimed to leverage GPU acceleration to explore advanced imbalance strategies over a longer training horizon. 
The IDRiD dataset is heavily imbalanced, especially against Grade 1 (only 16 training samples). 

**V3 Strategy:**
- **Architecture:** `MobileNetV3-Small` (ImageNet initialization, identical to V1/V2)
- **Imbalance Handling:** Combined batch-level `WeightedRandomSampler` with instance-level `Focal Loss` ($\gamma=2.0$).
- **Optimization:** `AdamW` (LR=$10^{-4}$, Weight Decay=$10^{-2}$) with `ReduceLROnPlateau` scheduling.
- **Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU using Mixed Precision (AMP).

---

## 2. Experimental Setup

| Parameter | Configuration |
|---|---|
| GPU / CUDA | NVIDIA RTX 5060 (CUDA 13.2, PyTorch 2.14.0+cu132) |
| Dataset Split | 80/20 train/val (stratified), 103 test (locked) |
| Training Size | 330 images |
| Class Distribution | Grade 0: 107, 1: 16, 2: 109, 3: 59, 4: 39 |
| Preprocessing | 224x224 RGB, ImageNet normalization |
| Augmentation | Horizontal flip, Rotation ($\pm10^\circ$), Mild brightness/contrast jitter |
| Max Epochs | 80 (Early Stopping Patience: 12) |
| Best Epoch Selection | Highest Validation Macro F1 |

---

## 3. Training & Validation Results
Training stopped early at **Epoch 32** (best checkpoint at **Epoch 20**).

| Validation Metric | V1 | V2 | V3 (Best) |
|---|---|---|---|
| **Macro F1** | 0.4546 | 0.5519 | **0.6081** |
| Accuracy | - | - | 0.6747 |
| Weighted F1 | - | - | 0.6702 |
| Expected Calib. Error (ECE)| - | - | 0.1008 |
| Grade 1 Val F1 | 0.00 | 0.105 | **0.400** |
| Referable Sensitivity | - | - | 0.9038 |
| Referable Specificity | - | - | 0.8065 |

> [!TIP]
> **Validation Impact:** The combination of Focal Loss and AdamW successfully pushed the validation Macro F1 past 0.60, and importantly, forced the model to learn Grade 1 features (Val F1 0.40), whereas earlier models completely ignored Grade 1.

---

## 4. Locked Official Test Set Results (One-Shot Evaluation)
After model selection, V3 was evaluated exactly once on the official 103-image test set.

### 5-Class Grading Metrics
| Metric | Test Result |
|---|---:|
| Accuracy | 50.49% |
| Macro Precision | 39.49% |
| Macro Recall | 37.07% |
| Macro F1 | 37.08% |
| Weighted F1 | 49.16% |
| Expected Calib. Error (ECE)| 0.0705 |

**Per-Class Test F1:**
- Grade 0: 0.64
- Grade 1: 0.00
- Grade 2: 0.56
- Grade 3: 0.38
- Grade 4: 0.26

### Referable DR (Binary: $\ge$ Grade 2)
While 5-class grading struggles, the model exhibits strong binary Referable DR detection:
| Metric | Result |
|---|---:|
| Sensitivity | 90.62% |
| Specificity | 64.10% |
| Precision | 80.56% |
| F1 Score | 85.29% |
| ROC-AUC | 0.9050 |

---

## 5. Error Analysis & Calibration
An analysis of the 103 test predictions (`models/idrid/v3/test_predictions.csv`):

- **Grade 1 Recall:** **0/5 correctly predicted.** Despite Val F1 hitting 0.40, the model completely fails to generalize Grade 1 features to the unseen test set. 5 images is a tiny sample size, but the failure is absolute.
- **Grade 2 Overprediction:** **27 images.** The model heavily biases toward Grade 2 when uncertain.
- **High-Confidence Errors:** **2 images.** Only 2 images were incorrectly predicted with a confidence $\ge$ 80%, indicating the Focal Loss prevented severe overconfidence on incorrect classes.
- **Low-Confidence Predictions:** **10 images.** 10 images had a max confidence $< 0.40$, correctly reflecting the model's uncertainty.
- **Calibration (ECE):** The Expected Calibration Error is quite low on the test set (**0.0705**), meaning the model's output probabilities are relatively well-calibrated to its actual accuracy.

### Test Confusion Matrix
```text
[[20  2 12  0  0]   <-- True Grade 0
 [ 3  0  2  0  0]   <-- True Grade 1 (0 correct, 3 pred as 0, 2 pred as 2)
 [ 4  0 23  3  2]   <-- True Grade 2
 [ 0  0  8  6  5]   <-- True Grade 3
 [ 1  1  5  3  3]]  <-- True Grade 4
```

---

## 6. ONNX Export Validation
The V3 model was exported to ONNX (`models/idrid/v3/model.onnx`).
A numerical validation check between the PyTorch output and the ONNX Runtime output was performed using an actual IDRiD test image.
- **Maximum Probability Difference:** `2.41e-06`
- **Result:** **PASSED** (well below the $10^{-5}$ tolerance).

---

## 7. Conclusions
1. **GPU Success:** The RTX 5060 setup with Mixed Precision and AdamW provided a stable, fast training environment.
2. **Imbalance Handling:** Focal Loss + WeightedRandomSampler significantly improved *validation* performance, particularly forcing the model to address the minority Grade 1 class during training.
3. **Generalization Gap:** The strong validation improvements did *not* translate to the official test set for the minority classes. 330 training images are simply insufficient to learn robust, generalizable features for 5-class grading without severe overfitting to the validation split.
4. **Referable DR Strengths:** The model is highly effective (0.90 AUC, 0.90 Sensitivity) at the binary task of separating "Safe" (Grades 0/1) from "Referable" (Grades 2/3/4), making it a viable baseline for binary screening prototypes despite the 5-class failure.
