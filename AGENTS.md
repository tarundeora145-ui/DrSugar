# DR-SUGAR — AGENTS.md

## PROJECT IDENTITY

Project name:
DR-SUGAR

Full name:
Explainable AI for Diabetic Retinopathy Screening

Purpose:
Build a functional, explainable diabetic-retinopathy screening prototype for resource-constrained and rural healthcare environments.

This is a Smart India Hackathon project.

---

# NON-NEGOTIABLE RULES

1. Never fabricate data.
2. Never fabricate AI predictions.
3. Never fabricate model confidence.
4. Never fabricate sensitivity, specificity, AUROC, F1, accuracy, or clinical validation.
5. Never fabricate dataset connectivity.
6. Never fabricate lesion annotations.
7. Never fabricate simulation results.
8. Never claim clinical diagnosis.
9. Never claim regulatory approval.
10. Never claim production readiness unless objectively demonstrated.
11. Preserve working functionality.
12. Prefer small, testable changes over large rewrites.
13. Inspect before editing.
14. Test after editing.
15. Do not silently replace missing functionality with mock medical results.

---

# DEVELOPMENT PRIORITY

Use this order:

1. Existing functionality
2. Correctness
3. Data integrity
4. Backend reliability
5. Actual dataset integration
6. Medical-image processing
7. Explainability
8. Validation
9. Simulink
10. UI polish

---

# PRODUCT PIPELINE

Fundus Image
→ Image Quality Assessment
→ Enhancement
→ Retinal Analysis
→ Lesion Analysis
→ DR Severity Assessment
→ Referable DR
→ Explainability
→ Clinical Review
→ Report

---

# DATASETS

APTOS 2019
Source:
https://www.kaggle.com/c/aptos2019-blindness-detection

Use for:
DR severity grading.

IDRiD
Source:
https://ieeedataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

Use for:
DR grading
lesion annotations
retinal analysis
explainability
optic disc/fovea information where available

DRIVE
Source:
https://drive.grand-challenge.org/

Use for:
retinal vessel segmentation

Messidor-2
Source:
https://www.adcis.net/en/third-party/messidor2/

Use for:
external robustness/generalization testing where valid ground truth exists.

Do not blindly merge these datasets into one model.

---

# DATASET STORAGE

Actual datasets belong in:

data/aptos2019/
data/idrid/
data/drive/
data/messidor2/

Do not put large datasets in src/.

Do not put image binaries into SQLite.

---

# DATABASE

Use SQLite:

database/dr_sugar.db

SQLite stores metadata, provenance, screening records, validation runs, and report metadata.

Do not store retinal image binaries as SQLite blobs unless there is a compelling technical reason.

---

# BACKEND

The backend is responsible for:

- dataset discovery
- dataset validation
- filesystem access
- image streaming
- dataset metadata
- ZIP extraction where appropriate
- SQLite operations
- model status
- screening metadata
- report metadata

The frontend must not access arbitrary local filesystem paths directly.

---

# DATASET STATUS

Allowed:

CONNECTED
PARTIAL
MISSING
INVALID

CONNECTED means the actual required files were found and validated.

Never set CONNECTED merely because metadata exists.

---

# AI

If actual trained model weights exist, use them.

If model weights do not exist:

MODEL UNAVAILABLE

Do not create fake predictions.

---

# MEDICAL SAFETY

Use language such as:

AI-assisted screening
Model prediction
Requires ophthalmologist review
Research prototype

Avoid:

Confirmed diagnosis
Guaranteed diagnosis
Clinically validated
Production-ready

unless actual evidence supports the claim.

---

# DESIGN

The design reference is provided in DESIGN.md.

Adapt its visual language to DR-SUGAR.

Core design:

- pure black void
- white typography
- gray secondary text
- electric violet action accent
- amber emphasis
- restrained teal
- huge lightweight typography
- generous whitespace
- asymmetric two-column layouts
- minimal borders
- minimal shadows
- sparse UI

Do not copy the reference brand.

Do not use:
Dala
Sakura
Sublime
DesignLayer

Create an original DR-SUGAR identity inspired by the visual language.

---

# APPLICATION ROUTES

/

Introduction

/dashboard
Operational Dashboard

/screening
Screening

/reports
Reports

/simulation
Simulation

/validation
Datasets + Validation

---

# ENGINEERING RULE

For every task:

INSPECT
→ PLAN
→ IMPLEMENT
→ RUN
→ TEST
→ FIX
→ VERIFY

Do not skip verification.

---

# FINAL REPORT FORMAT

After completing a task, report:

Implemented
Tested
Changed Files
Known Issues
Not Implemented
Run Commands

Do not claim success when something was only mocked.
