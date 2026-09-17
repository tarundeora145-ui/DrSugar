# DR Model Specification

## Task
**5-Class Diabetic Retinopathy Severity Grading**

## Architecture
- **Base Model**: MobileNetV3-Small (lightweight for efficient CPU inference)
- **Framework**: PyTorch → exported to ONNX (opset 14)

## Class Mapping
| Class ID | Label | Referable DR |
|---|---|---|
| 0 | No DR | No |
| 1 | Mild | No |
| 2 | Moderate | Yes |
| 3 | Severe | Yes |
| 4 | Proliferative | Yes |

## Preprocessing Requirements
Both training (Python `torchvision.transforms`) and inference (Node.js `sharp` + Math) strictly adhere to:
1. **Resize**: 224x224 (cover/squash)
2. **Channel Order**: RGB
3. **Scaling**: Pixels normalized `[0, 1]` (divided by 255.0)
4. **Standardization**: 
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
5. **Tensor Shape**: `[1, 3, 224, 224]` (Batch, Channel, Height, Width)

## Artifact Output
- `models/dr_classification/model.onnx` (Binary weights)
- `models/dr_classification/metadata.json` (Training metadata and metrics)
