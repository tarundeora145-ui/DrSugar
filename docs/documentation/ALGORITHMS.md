# Algorithmic Specification

This document details all algorithms, mathematical models, preprocessing procedures, explainability methods, and metrics implemented in **DR-SUGAR**.

---

## 1. Primary DR Classification Algorithm

### 1.1 Model Architecture
- **Model Backbone**: `MobileNetV3-Small` (2.54M parameters).
- **Target Function**: 5-class ordinal severity classification of Diabetic Retinopathy.
- **Classes**:
  - `0`: No DR
  - `1`: Mild DR
  - `2`: Moderate DR
  - `3`: Severe DR
  - `4`: Proliferative DR

### 1.2 Image Preprocessing Pipeline
To prevent distribution shift between training and ONNX runtime inference:
1. **Color Space**: RGB.
2. **Resizing**: Resize input image to `224 x 224` pixels using bilinear interpolation with `fit: fill`.
   $$\text{Resize}_{\text{torchvision}}((224, 224)) \equiv \text{Sharp.resize}(224, 224, \{\text{fit}: \text{'fill'}\})$$
   *Note: No aspect cropping or padding is applied, matching training input behavior.*
3. **Channel Normalization**:
   Channel values $x \in [0, 255]$ are scaled to $[0.0, 1.0]$ and normalized per-channel using ImageNet mean ($\mu$) and standard deviation ($\sigma$):
   $$\hat{x}_{c, i, j} = \frac{\frac{x_{c, i, j}}{255.0} - \mu_c}{\sigma_c}$$
   Where:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 1.3 Probability & Softmax Computation
Output logits $z = [z_0, z_1, z_2, z_3, z_4]$ are transformed into probabilities $p_i$ via numerically stable Softmax:
$$p_i = \frac{e^{z_i - \max(z)}}{\sum_{j=0}^{4} e^{z_j - \max(z)}}$$

### 1.4 Decision Logic
- **Predicted Grade**:
  $$\hat{y} = \arg\max_{i \in \{0,..,4\}} p_i$$
- **Confidence Score**:
  $$C = \max_{i} p_i = p_{\hat{y}}$$
- **Referable DR Threshold**:
  $$\text{ReferableDR} = \begin{cases} 1 & \text{if } \hat{y} \ge 2 \\ 0 & \text{if } \hat{y} < 2 \end{cases}$$

---

## 2. Explainability Algorithm (Grad-CAM)

### 2.1 Gradient-Weighted Class Activation Mapping
Grad-CAM computes the visual attribution heatmap for a target class $k$ (predicted grade $\hat{y}$) from feature map activations $A^k$ at layer `features[-1]` of MobileNetV3.

1. **Neuron Importance Weights ($\alpha_c^k$)**:
   $$\alpha_c^k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial z_k}{\partial A_{i, j}^c}$$
   Where $Z = H \times W$ is the spatial area of the feature map.

2. **Weighted Activation Map ($L_{\text{Grad-CAM}}^k$)**:
   $$L_{\text{Grad-CAM}}^k = \text{ReLU}\left( \sum_{c} \alpha_c^k A^c \right)$$

3. **Normalization & Visual Overlay**:
   The activation map $L^k$ is normalized to $[0, 1]$, colorized with OpenCV `COLORMAP_JET`, and blended with original fundus image $I$ at alpha weight $\alpha = 0.5$:
   $$I_{\text{overlay}} = 0.5 \cdot I + 0.5 \cdot \text{ColorMap}(L^k)$$

---

## 3. Lesion Segmentation Algorithm (IDRiD V3)

### 3.1 Model & Multi-Class Micro-Structures
- **Architecture**: PyTorch U-Net (`models/idrid/lesions/v3/fold1_best.pth`).
- **Input Patch Processing**: 512x512 bilinear resized retinal patches.
- **Output Channels**: 4 distinct pathological lesion types:
  1. `MA`: Microaneurysms
  2. `HE`: Hemorrhages
  3. `EX`: Hard Exudates
  4. `SE`: Soft Exudates / Cotton Wool Spots

### 3.2 Area Coverage Calculation
For each lesion channel $m \in \{\text{MA}, \text{HE}, \text{EX}, \text{SE}\}$, the binary lesion mask is thresholded at $\tau = 0.5$:
$$\text{Mask}_m(x, y) = \mathbb{I}\left( P_m(x, y) \ge 0.5 \right)$$
Surface area coverage percentage is computed over total image pixels $N_{\text{pixels}} = H \times W$:
$$\text{AreaPct}_m = \left( \frac{\sum_{x,y} \text{Mask}_m(x, y)}{N_{\text{pixels}}} \right) \times 100$$

---

## 4. Retinal Vessel Segmentation Algorithm (DRIVE V1)

### 4.1 Architecture & Vessel Extraction
- **Architecture**: PyTorch U-Net trained on DRIVE V1 dataset (`models/drive/v1/fold1_best.pth`).
- **Center Crop / Padding Preprocessing**:
  Images are center-cropped or padded to 512x512 dimensions, normalized, and processed via sigmoid activation:
  $$P_{\text{vessel}}(x, y) = \sigma\left( \text{Model}(I(x, y)) \right)$$

### 4.2 Vascular Coverage Calculation
Binary vessel mask:
$$\text{VesselMask}(x, y) = \mathbb{I}\left( P_{\text{vessel}}(x, y) \ge 0.5 \right)$$
Vascular coverage metric:
$$\text{VesselCoveragePct} = \left( \frac{\sum_{x,y} \text{VesselMask}(x, y)}{H \times W} \right) \times 100$$

---

## 5. Deterministic Queue & Network Simulation Algorithm

The simulation module models PHC screening queue dynamics without invoking external proprietary solvers:

1. **Total Annual Images**:
   $$N_{\text{images}} = N_{\text{patients}} \times N_{\text{images/patient}}$$
2. **Total Upload Volume (GB)**:
   $$V_{\text{data\_GB}} = \frac{N_{\text{images}} \times S_{\text{image\_MB}}}{1024}$$
3. **Image Transmission Latency (seconds)**:
   $$T_{\text{upload}} = \frac{S_{\text{image\_MB}} \times 8}{\text{Bandwidth}_{\text{Mbps}}}$$
4. **Total Turnaround Time (seconds)**:
   $$T_{\text{turnaround}} = T_{\text{upload}} + T_{\text{AI\_processing}}$$
5. **Annual Triaged Referrals**:
   $$N_{\text{referred}} = N_{\text{patients}} \times \text{ReferralRate}$$
6. **Daily Specialist Review Workload**:
   $$W_{\text{specialist/day}} = \frac{N_{\text{referred}}}{365}$$

---

## 6. Mathematical Evaluation Metrics

- **Sensitivity (Recall on Referable DR)**:
  $$\text{Sensitivity} = \frac{TP}{TP + FN}$$
- **Specificity (True Negative Rate)**:
  $$\text{Specificity} = \frac{TN}{TN + FP}$$
- **Precision (Positive Predictive Value)**:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **F1 Score**:
  $$F1 = 2 \times \frac{\text{Precision} \times \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}$$
- **Macro F1 Score**:
  $$\text{Macro F1} = \frac{1}{K} \sum_{k=0}^{K-1} F1_k$$
