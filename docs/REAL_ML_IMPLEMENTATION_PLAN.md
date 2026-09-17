# DR-SUGAR Real ML Implementation Plan

## Phase 1 — Environment
- **Task**: Set up the production inference environment in Node.js and the offline training environment in Python.
- **Files to Create/Change**: `server/package.json` (add `onnxruntime-node`, `jimp` or `sharp` for preprocessing), `requirements.txt` (for offline PyTorch training).
- **Dependencies**: `onnxruntime-node`, `torch`, `torchvision`, `onnx`.
- **Input**: None.
- **Output**: Functional runtimes capable of loading an ONNX file.
- **Test**: Run a script that imports `onnxruntime-node` without crashing.
- **Acceptance Criteria**: The server environment supports ML execution without MATLAB.

## Phase 2 — Dataset Preparation
- **Task**: Download and extract the real APTOS 2019 dataset to `data/aptos2019/`.
- **Files to Create/Change**: `scripts/download_datasets.py` or manual instructions.
- **Dependencies**: Kaggle CLI or manual download.
- **Input**: Kaggle API credentials.
- **Output**: Populated `data/aptos2019/` directory containing `.png` images and `train.csv`.
- **Test**: Run the existing dataset scanner in the backend to verify the dataset transitions from `MISSING` to `CONNECTED`.
- **Acceptance Criteria**: The SQLite database correctly registers actual APTOS images and labels.

## Phase 3 — Preprocessing
- **Task**: Implement real image preprocessing (resizing, center cropping, normalization) in Node.js matching the PyTorch `transforms`.
- **Files to Create/Change**: `server/src/ml/preprocessing.ts`.
- **Dependencies**: `sharp` or `jimp`.
- **Input**: Raw fundus image buffer.
- **Output**: `Float32Array` tensor matching `[1, 3, 224, 224]` dimensions.
- **Test**: Compare a preprocessed image tensor against a PyTorch reference output.
- **Acceptance Criteria**: Output exactly matches the shape and scaling expected by the trained model.

## Phase 4 — DR Classification
- **Task**: Train (offline in Python) a baseline ResNet or MobileNet model on APTOS 2019 for 5-class DR severity. Export to ONNX.
- **Files to Create/Change**: `scripts/train_dr_grading.py`, `models/dr_classification/metadata.json`, `models/dr_classification/model.onnx`.
- **Dependencies**: PyTorch.
- **Input**: APTOS 2019 images.
- **Output**: A trained `.onnx` model artifact and metadata.
- **Test**: Evaluate the model locally on a hold-out test set in Python.
- **Acceptance Criteria**: Model achieves >80% accuracy on validation set and successfully exports to ONNX.

## Phase 5 — Inference Service
- **Task**: Build the Node.js ML execution service.
- **Files to Create/Change**: `server/src/ml/inference.ts`, `server/src/routes/screening.ts`.
- **Dependencies**: `onnxruntime-node`.
- **Input**: Preprocessed `Float32Array` tensor.
- **Output**: Prediction class, probabilities array, and referable DR boolean.
- **Test**: Pass an image via POST `/api/screening` and verify the JSON response contains real probabilities summing to 1.0.
- **Acceptance Criteria**: The API dynamically loads the ONNX model and removes the `MODEL UNAVAILABLE` stub.

## Phase 6 — Explainability
- **Task**: Generate Grad-CAM heatmaps from the ONNX model.
- **Files to Create/Change**: `server/src/ml/gradcam.ts`, update ONNX export to include intermediate convolutional layer outputs.
- **Dependencies**: `onnxruntime-node` (executing a dual-output model).
- **Input**: Input tensor and target class.
- **Output**: A resized spatial heatmap array.
- **Test**: Verify the heatmap correctly aligns with high-contrast regions (optic disc/vessels/lesions).
- **Acceptance Criteria**: The `/api/screening/:id/explainability` route returns real heatmap data instead of `NOT EVALUATED`.

## Phase 7 — SQLite Integration
- **Task**: Connect the real inference outputs to the existing database schema.
- **Files to Create/Change**: `server/src/routes/screening.ts`.
- **Dependencies**: `better-sqlite3`.
- **Input**: Inference service output.
- **Output**: Insert statements for `screenings`, `screening_analysis`, and `screening_quality`.
- **Test**: Perform a screening and verify the exact probability vectors are present in the SQLite database.
- **Acceptance Criteria**: No fabricated results are inserted; database schema matches model contract.

## Phase 8 — Validation
- **Task**: Compute real metrics using the Validation UI based on actual DB records.
- **Files to Create/Change**: `server/src/routes/validation.ts`.
- **Dependencies**: None.
- **Input**: Query across `screenings` vs `dataset_labels`.
- **Output**: Confusion matrix, TP/TN/FP/FN, Sensitivity, Specificity.
- **Test**: Run evaluation on the APTOS valid split and verify the metrics reflect the model's actual performance.
- **Acceptance Criteria**: Validation UI removes `NOT EVALUATED` and displays mathematically verified metrics.

## Phase 9 — Lesion Analysis
- **Task**: (Deferred to MVP v2) Train a semantic segmentation model on IDRiD for microaneurysms/hemorrhages.
- **Files to Create/Change**: `scripts/train_lesions.py`.
- **Acceptance Criteria**: ONNX model capable of outputting probability masks for lesions.

## Phase 10 — Vessel Segmentation
- **Task**: (Deferred to MVP v3) Train a UNet on DRIVE dataset for retinal vessels.
- **Files to Create/Change**: `scripts/train_vessels.py`.
- **Acceptance Criteria**: ONNX model capable of outputting a crisp binary vessel mask.

## Phase 11 — MATLAB/Simulink Integration
- **Task**: (Deferred or Deprecated) Since MATLAB is unavailable on the target environment, the Simulink deployment simulation will either require translation to a Python queueing simulation (`simpy`) or a MATLAB installation.
- **Files to Create/Change**: To be decided based on Simulink licensing.

## Phase 12 — Production Hardening
- **Task**: Add authentication and secure the ONNX execution paths.
- **Files to Create/Change**: `server/src/middleware/auth.ts`.
- **Acceptance Criteria**: API routes are protected.
