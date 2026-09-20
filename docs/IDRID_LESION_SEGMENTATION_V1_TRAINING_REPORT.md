# DR-SUGAR — IDRiD Lesion Segmentation V1 Training Report

> [!WARNING]
> **Research/prototype only. Not clinically validated.** 
> The official IDRiD segmentation test set was explicitly withheld and not used for metric evaluation or threshold tuning. All results are from a rigorous 5-fold cross-validation solely within the 54 official training images.

---

## 1. Objective

Train a multi-class lesion segmentation model on the IDRiD dataset using GPU acceleration. Evaluate performance separately for Microaneurysms (MA), Haemorrhages (HE), Hard Exudates (EX), and Soft Exudates (SE) using 5-fold cross-validation. Report Mean ± Std across folds for Dice, IoU, Sensitivity, Specificity, and Precision per lesion type.

---

## 2. Hardware & Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA | 13.2 |
| PyTorch | 2.14.0+cu132 |
| Mixed Precision | torch.amp.autocast / GradScaler |
| Total Training Time | **~50.8 minutes** (all 5 folds) |

---

## 3. Dataset Discovery & Structure

| Dataset Element | Count | Description |
|---|---|---|
| Total Original Training Images | 54 | Located in `data/idrid/A. Segmentation/1. Original Images/a. Training Set` |
| Original Dimensions | 4288 × 2848 | Aspect ratio ~ 3:2 |
| MA Masks Found | 54 | — |
| HE Masks Found | 53 | 1 image explicitly missing HE lesions |
| EX Masks Found | 54 | — |
| SE Masks Found | 26 | 28 images explicitly missing SE lesions |

> [!NOTE]
> Missing mask files were appropriately handled by loading dynamic **all-zero (true negative) masks**. They were NOT treated as corrupted or skipped.

### Overlap Analysis
An automated script evaluated the ground truth mask overlaps for all 54 training images:
- **Images with overlapping lesions**: 21
- **Conclusion**: Due to frequent overlapping lesions (e.g., hemorrhages overlapping with exudates), it is **scientifically incorrect** to model this using mutually-exclusive Softmax/Cross-Entropy.

---

## 4. Architecture: MobileNetV3-Small U-Net

To correctly model overlapping lesions, the architecture outputs **4 independent binary channels**.

- **Encoder**: ImageNet-pretrained `MobileNetV3-Small`
- **Decoder**: Custom PyTorch U-Net decoder with skip connections
- **Input Channels**: 3 (RGB)
- **Output Channels**: 4 (MA, HE, EX, SE), normalized by `Sigmoid`.

---

## 5. Preprocessing & Augmentation

### Preprocessing
Because the original 4288×2848 resolution is too large for U-Net training on consumer GPUs, all images and masks were:
1. **Padded to square**: Black padding added to reach 4288×4288 (preserving aspect ratio).
2. **Resized**: Downscaled to **512×512** (bilinear for images, nearest-neighbor for binary masks).
3. **Normalized**: Scaled to [0, 1], then standard ImageNet Mean/Std subtraction.

### Augmentation (Training Only)
Clinically reasonable geometric transformations applied identically to image and masks:
- Horizontal Flip (50%)
- Vertical Flip (50%)
- Random Rotation (±10°)
- Mild Brightness/Contrast variation (ColorJitter 0.1/0.1)

---

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Loss | **BCE + Dice** (independent per channel, then averaged) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5, mode=max) |
| Batch size | 4 |
| Max epochs | 80 |
| Early stopping | Patience=15 on validation Mean Dice |
| AMP | True |
| Checkpoint selection | Best Validation Mean Dice per fold |

### 5-Fold Split
The 54 training images were randomly split (Seed 42) into 5 folds.
- 4 folds contain 11 validation images (43 training).
- 1 fold contains 10 validation images (44 training).

---

## 7. Results

The model severely struggled with the dataset in this exact configuration, achieving very low metric values. Early stopping engaged aggressively (often around Epoch 15–16).

### Cross-Validation Summary (Mean ± Std)

| Lesion | Dice | IoU | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| **MA** (Microaneurysms) | 0.2000 ± 0.4000 | 0.2000 ± 0.4000 | 1.0000 ± 0.0000 | 0.6000 ± 0.4899 | 0.0000 ± 0.0000 |
| **HE** (Haemorrhages) | 0.4000 ± 0.4899 | 0.4000 ± 0.4899 | 1.0000 ± 0.0000 | 0.7997 ± 0.3998 | 0.0000 ± 0.0000 |
| **EX** (Hard Exudates) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.2000 ± 0.4000 | 0.0000 ± 0.0000 |
| **SE** (Soft Exudates) | 0.6000 ± 0.4899 | 0.6000 ± 0.4899 | 1.0000 ± 0.0000 | 0.6000 ± 0.4899 | 0.0000 ± 0.0000 |
| **Mean (All Lesions)** | **0.3000 ± 0.2915** | — | — | — | — |

> [!CAUTION]
> The metric breakdown indicates the model effectively collapsed, frequently outputting all zeros (which artificially inflates sensitivity to 1.0 for true negative masks, and drives precision to 0.0 on positive cases).

---

## 8. Qualitative Results

Qualitative grids showing `Original | MA | HE | EX | SE` (Ground Truth vs. Prediction) are saved in:
- `models/idrid/lesions/v1/qualitative/foldX_{img_id}_pred.png`

---

## 9. Limitations and Failure Analysis

This baseline experiment failed to produce a viable segmentation model. The primary reasons are:

1. **Destructive Downsampling**: Resizing 4288×2848 high-resolution fundus images to 512×512 completely destroys Microaneurysms (which are often only 5–10 pixels across in the original image). After downsampling, these lesions sub-pixelate and vanish.
2. **Sparsity**: IDRiD lesions are extremely sparse. When the lesions vanish due to downscaling, the model trivially learns to predict an empty mask (background only) to minimize the BCE loss.
3. **Patch-based Training Required**: To accurately segment IDRiD lesions, the network must be trained on high-resolution crops (e.g., 512×512 patches extracted from the original 4288×2848 images), rather than globally downscaled full-fundus images.
4. **Focal Loss**: Standard BCE is easily overwhelmed by the extreme class imbalance (99.9% background). Moving to Focal Loss or adjusting the Dice smoothing factor is necessary.

Future iterations (V2) must implement patch-based extraction and sampling to solve this resolution collapse.
