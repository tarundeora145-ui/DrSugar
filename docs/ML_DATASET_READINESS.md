# DR-SUGAR Dataset Readiness Audit

## Current SQLite Status
The `datasets` table in SQLite is structurally ready and the loaders (`AptosLoader.ts`, etc.) are functional. However, because the actual files are missing from the disk, the loaders will correctly evaluate their status as `MISSING`.

## APTOS 2019
- **Dataset Path**: `data/aptos2019/` (MISSING)
- **Accessible Files**: 0
- **Label Availability**: UNAVAILABLE
- **Annotation Availability**: UNAVAILABLE
- **Current SQLite Status**: MISSING
- **Usable ML Tasks**: None currently. (Intended for DR grading).
- **Blockers**: The dataset archive must be downloaded and extracted into `data/aptos2019/`.

## IDRiD
- **Dataset Path**: `data/idrid/` (MISSING)
- **Accessible Files**: 0
- **Label Availability**: UNAVAILABLE
- **Annotation Availability**: UNAVAILABLE
- **Current SQLite Status**: MISSING
- **Usable ML Tasks**: None currently. (Intended for DR grading and lesion segmentation).
- **Blockers**: The dataset archive must be downloaded and extracted into `data/idrid/`.

## DRIVE
- **Dataset Path**: `data/drive/` (MISSING)
- **Accessible Files**: 0
- **Label Availability**: UNAVAILABLE (Masks unavailable)
- **Annotation Availability**: UNAVAILABLE
- **Current SQLite Status**: MISSING
- **Usable ML Tasks**: None currently. (Intended for vessel segmentation).
- **Blockers**: The dataset archive must be downloaded and extracted into `data/drive/`.

## Messidor-2
- **Dataset Path**: `data/messidor2/` (MISSING)
- **Accessible Files**: 0
- **Label Availability**: UNAVAILABLE
- **Annotation Availability**: UNAVAILABLE
- **Current SQLite Status**: MISSING
- **Usable ML Tasks**: None currently. (Intended for external robustness).
- **Blockers**: The dataset archive must be downloaded and extracted into `data/messidor2/`.

## Conclusion
**DO NOT INVENT LABELS.** 
The database will continue to reflect reality (MISSING) until the physical datasets are placed in the `data/` directory. No training or validation can occur until these datasets are provisioned.
