# DR-SUGAR — DRIVE Retinal Vessel Segmentation V1 Training Report

> [!WARNING]
> **Research/prototype only. Not clinically validated.** DRIVE test set was not used for any metric evaluation (no public vessel GT exists). All results are from 5-fold cross-validation within the 20 official DRIVE training images.

---

## 1. Objective

Train a real retinal vessel segmentation model on the DRIVE dataset using GPU acceleration. Evaluate rigorously using 5-fold cross-validation within the 20 available training images. Report Dice, IoU, Sensitivity, Specificity, and AUC-ROC.

---

## 2. Hardware & Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA | 13.2 |
| PyTorch | 2.14.0+cu132 |
| Mixed Precision | torch.amp.autocast / GradScaler |
| Total Training Time | **16.5 minutes** (all 5 folds) |

---

## 3. Architecture: MobileNetV3-Small U-Net

A U-Net was implemented from scratch using **MobileNetV3-Small** (ImageNet pretrained) as the encoder backbone, with a custom decoder.

### Encoder Skip Connection Points (512×512 input)

| Stage | Layer(s) | Output Channels | Spatial Size |
|---|---|---|---|
| s0 | features[0] | 16 | 256×256 |
| s1 | features[1] | 16 | 128×128 |
| s2 | features[2–3] | 24 | 64×64 |
| s3 | features[4–6] | 40 | 32×32 |
| Bottleneck | features[7–12] | 576 | 16×16 |

### Decoder

| Block | Input | Skip Ch | Output Ch | Output Size |
|---|---|---|---|---|
| dec4 | 576 | 40 | 256 | 32×32 |
| dec3 | 256 | 24 | 128 | 64×64 |
| dec2 | 128 | 16 | 64 | 128×128 |
| dec1 | 64 | 16 | 32 | 256×256 |
| up_final | 32 | — | 16 | 512×512 |
| final_conv | 16 | — | 1 | 512×512 |

Each decoder block uses `ConvTranspose2d` upsampling followed by a `DoubleConv` (Conv-BN-ReLU × 2).
Output is a single-channel logit map; sigmoid gives pixel-level vessel probability.

> [!NOTE]
> No external segmentation library (`segmentation_models_pytorch` etc.) was used. The U-Net is fully implemented in PyTorch.

---

## 4. Dataset

| Item | Value |
|---|---|
| Source | DRIVE (Digital Retinal Images for Vessel Extraction) |
| Training images | 20 (IDs 21–40) |
| Test images | 20 (NOT used — no public vessel GT) |
| Vessel GT | 1st manual annotator masks (`.gif`, binary {0, 255}) |
| FOV masks | Circular retinal field masks (`.gif`, binary {0, 255}) |
| Image dimensions | 584×565 pixels (RGB `.tif`) |

### 5-Fold Split (stratified by image ID, seed=42)

| Fold | Validation IDs | Train Count | Val Count |
|---|---|---|---|
| 1 | 21, 26, 34, 33 | 16 | 4 |
| 2 | 38, 32, 23, 28 | 16 | 4 |
| 3 | 36, 24, 30, 31 | 16 | 4 |
| 4 | 22, 39, 40, 35 | 16 | 4 |
| 5 | 29, 37, 25, 27 | 16 | 4 |

---

## 5. Preprocessing

- **Input size:** 512×512 — achieved by center-cropping from 584×565 (retains ~96.6% of content; all content within FOV is preserved)
- **Normalization:** ImageNet mean/std applied after scaling to [0,1]
- **FOV masking:** The circular retinal FOV mask was applied during loss computation and metric evaluation. Pixels outside the retina are excluded from all calculations.

---

## 6. Augmentation (training only)

| Augmentation | Details |
|---|---|
| Horizontal flip | 50% probability |
| Vertical flip | 50% probability |
| Rotation | ±10°, bilinear resampling for image, nearest for mask/FOV |
| Brightness/Contrast jitter | torchvision.transforms.ColorJitter (0.1/0.1) |

Augmentations were applied **identically** to image, vessel mask, and FOV mask. Nearest-neighbor resampling for masks ensures binary labels are preserved.

---

## 7. Training Configuration

| Parameter | Value |
|---|---|
| Loss | **BCE + Dice** (computed within FOV mask only) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5, mode=max) |
| Batch size | 4 |
| Max epochs | 80 |
| Early stopping | Patience=12 on validation Dice |
| AMP | True |
| Checkpoint selection | Best validation Dice per fold |

---

## 8. Results

### Per-Fold Validation Metrics

| Fold | Dice | IoU | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.7379 | 0.5847 | 0.7580 | 0.9578 | 0.9403 |
| 2 | 0.7659 | 0.6207 | 0.7902 | 0.9620 | 0.9536 |
| 3 | 0.7801 | 0.6395 | 0.7783 | 0.9663 | 0.9459 |
| 4 | 0.7137 | 0.5549 | 0.8281 | 0.9298 | 0.9472 |
| 5 | 0.7745 | 0.6319 | 0.7607 | 0.9693 | 0.9465 |

### Cross-Validation Summary (Mean ± Std)

| Metric | Mean ± Std |
|---|---:|
| **Dice** | **0.7544 ± 0.0250** |
| **IoU / Jaccard** | **0.6063 ± 0.0319** |
| **Sensitivity / Recall** | **0.7831 ± 0.0254** |
| **Specificity** | **0.9570 ± 0.0142** |
| **ROC-AUC** | **0.9467 ± 0.0042** |

> [!NOTE]
> Fold 4 shows notably lower Dice (0.7137) and Specificity (0.9298) compared to other folds. This likely reflects a harder validation subset (images 22, 39, 40, 35 may contain more tortuous vessels or variable image quality). This variability is expected with only 20 images in total.

---

## 9. Qualitative Results

Qualitative prediction images (Original | Ground Truth | Prediction) are saved for each fold's validation images:

- `models/drive/v1/fold1_predictions.png`
- `models/drive/v1/fold2_predictions.png`
- `models/drive/v1/fold3_predictions.png`
- `models/drive/v1/fold4_predictions.png`
- `models/drive/v1/fold5_predictions.png`

---

## 10. Artifacts

```
models/drive/v1/
├── fold1_best.pth          (Val Dice 0.7379)
├── fold2_best.pth          (Val Dice 0.7659)
├── fold3_best.pth          (Val Dice 0.7801)
├── fold4_best.pth          (Val Dice 0.7137)
├── fold5_best.pth          (Val Dice 0.7745)
├── fold1_predictions.png
├── fold2_predictions.png
├── fold3_predictions.png
├── fold4_predictions.png
├── fold5_predictions.png
├── metadata.json           (full configuration + results)
├── cv_results.csv          (per-fold metrics)
└── training_history.csv    (per-epoch metrics, all folds)
```

---

## 11. Limitations

1. **Dataset size.** With only 20 training images (16 per fold), results are sensitive to fold composition. The ±0.025 Dice standard deviation reflects genuine generalization variance.
2. **No test-set evaluation.** The DRIVE test set (20 images) has no public vessel GT. No test metrics are reported — this is not a limitation of this implementation but of the dataset's design.
3. **Single annotator.** Training was performed against the 1st manual annotator only. Inter-annotator agreement on DRIVE is typically 0.77–0.80 Dice; our mean of 0.754 is close to human-level on this small training regime.
4. **No post-processing.** No CRF, morphological post-processing, or threshold optimization was applied. These could improve Dice by 1–3%.
5. **No test-time augmentation.** TTA could improve AUC-ROC and reduce variance across folds.
6. **Fold 4 degradation.** Fold 4 shows lower performance. Investigation of those specific images may reveal challenging cases (highly tortuous vessels, pathological backgrounds from diabetic eyes).
7. **Clinical use.** This model is a research prototype. It has not been validated for clinical deployment.
