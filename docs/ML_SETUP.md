# DR-SUGAR ML Setup

## 1. Environment Preparation
The ML pipeline runs in two distinct environments:
- **Training (Offline):** Python 3 + PyTorch
- **Inference (Production):** Node.js + ONNX Runtime

### Training Setup (Windows)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision onnx onnxruntime pandas scikit-learn pillow
```

### Node.js Inference Setup
The server automatically installs the required packages (`onnxruntime-node` and `sharp`) during `npm install`.

## 2. Dataset Acquisition
We use the **APTOS 2019 Blindness Detection Dataset** for the primary DR severity classification.

1. Create a Kaggle account and agree to the dataset terms.
2. Download the dataset from: `https://www.kaggle.com/c/aptos2019-blindness-detection/data`
3. Extract the contents directly into `data/aptos2019/`.
4. Ensure the structure looks exactly like this:
   ```
   data/aptos2019/
   ├── train.csv
   ├── test.csv
   ├── train_images/
   │   ├── 000c1434d8d7.png
   │   └── ...
   └── test_images/
   ```

## 3. Dataset Preparation
Once the data is downloaded, run the split script:
```bash
python scripts/prepare_dr_dataset.py
```
This generates `split_train.csv`, `split_val.csv`, `split_test.csv`, and `split_metadata.json`.

## 4. Model Training
```bash
python scripts/train_dr_grading.py
```
This trains a MobileNetV3 model on the APTOS dataset and saves the best model to `models/dr_classification/best_model.pth`.

## 5. ONNX Export
```bash
python scripts/export_dr_model.py
```
This exports the PyTorch model to `model.onnx` and automatically validates the numerical outputs against the original PyTorch model.
