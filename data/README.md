# DR-SUGAR Datasets

This directory is strictly meant to contain the datasets utilized by the DR-SUGAR screening platform.

## Supported Datasets & Installation

1. **APTOS 2019 Blindness Detection**
   - **Path**: `data/aptos2019/`
   - **Structure Expected**:
     - `train_images/`
     - `train.csv`
   - **Download Command (requires Kaggle CLI)**:
     ```bash
     kaggle competitions download -c aptos2019-blindness-detection
     ```
     Extract the files into `data/aptos2019/`.

2. **IDRiD (Indian Diabetic Retinopathy Image Dataset)**
   - **Path**: `data/idrid/`
   - **Structure Expected**:
     - `Disease Grading/`
     - `Localization/`

3. **DRIVE (Digital Retinal Images for Vessel Extraction)**
   - **Path**: `data/drive/`
   - **Structure Expected**:
     - `training/`
     - `test/`

4. **Messidor-2**
   - **Path**: `data/messidor2/`
   - **Structure Expected**:
     - `IMAGES/`
     - `messidor_data.csv`

> **Note on Safety:** The DR-SUGAR backend performs rigorous deep validation and path traversal checks. It will not fabricate missing data. Ensure the datasets perfectly follow these folder structures.
