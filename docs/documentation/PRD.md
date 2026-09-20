# Product Requirements Document (PRD)

## 1. Product Overview
**DR-SUGAR** (*Explainable AI for Diabetic Retinopathy Screening*) is an open, verifiable research prototype designed for resource-constrained, low-bandwidth, and rural healthcare environments in India. It assists healthcare operators and ophthalmologists by providing rapid 2-phase AI screening: immediate 5-class DR severity classification (<80 ms) followed by background multi-modal explainability evidence (Grad-CAM activation overlays, IDRiD lesion segmentation, and DRIVE retinal vessel structure mapping).

---

## 2. Problem Statement
Diabetic Retinopathy (DR) is a leading cause of preventable blindness worldwide. In rural healthcare networks across India, millions of diabetic patients require mandatory annual retinal screenings. However, the severe shortage of trained ophthalmologists creates massive referral backlogs. While black-box AI models offer high theoretical accuracy, clinicians reject uncalibrated predictions that lack anatomical evidence, leading to clinical distrust and diagnostic inertia.

---

## 3. Background / Rural Healthcare Context
- **Primary Health Centers (PHCs)**: Operating under variable power, limited internet bandwidth (1–2 Mbps), and operated by non-ophthalmic healthcare staff.
- **Teleophthalmology Bottlenecks**: Transmitting high-resolution fundus images to urban tertiary centers causes turnaround delays of days to weeks.
- **Explainability Gap**: Ophthalmologists require verifiable evidence (lesion location, hemorrhage extent, vascular density) to confirm referral urgency without re-evaluating every raw scan from scratch.

---

## 4. Product Vision
To provide a transparent, fast, and clinically auditable AI screening companion that empowers rural screening operators to triage referable diabetic retinopathy at point-of-care while giving ophthalmologists complete anatomical reasoning behind every prediction.

---

## 5. Goals
- **Sub-Second Primary Inference**: Deliver 5-class DR grade (0–4), confidence score, and referable DR status in <80 ms via ONNX Runtime.
- **Asynchronous Explainability**: Generate background visual evidence (Grad-CAM heatmaps, lesion area coverage %, vessel density) within 3–5 seconds without blocking the operator UI.
- **Zero Fabrication**: Enforce strict truthfulness—never mock medical predictions, dataset statuses, or clinical validation scores.
- **Bilingual Accessibility**: Full support for English and Hindi (हिन्दी) in patient-facing screening reports.
- **High Sensitivity Triage**: Achieve >95% Sensitivity on referable DR (Grade ≥ 2) on independent test splits.

---

## 6. Non-Goals
- **Clinical Diagnosis**: DR-SUGAR is an AI-assisted screening prototype, not a standalone diagnostic system or regulatory-approved medical device (FDA/CDSCO).
- **Direct EHR Integration**: No automated HL7/FHIR sync in the current prototype release.
- **Autonomous Prescription**: System never prescribes treatments or medication.

---

## 7. Target Users
1. **Screening Operator / PHC Technician**: Non-specialist healthcare worker who captures fundus photographs, uploads images, and records basic demographic details.
2. **Ophthalmologist / Specialist**: Eye care physician who reviews triaged screenings, examines Grad-CAM overlays and lesion statistics, and issues formal clinical referrals.
3. **Patient**: Individual receiving the preliminary screening summary report in their preferred language (English or Hindi).

---

## 8. User Personas
- **Persona A: Sunita (PHC Health Worker, Rural MP)**: Needs a fast, simple interface that accepts camera uploads, verifies image quality, and produces a printable patient summary in Hindi.
- **Persona B: Dr. R. Sharma (Consultant Ophthalmologist, District Hospital)**: Requires high-sensitivity triage (minimal false negatives) with verifiable visual heatmaps and lesion area percentages to rapidly confirm referred cases.

---

## 9. User Journeys

```
[Patient Arrival at PHC]
       │
       ▼
[Demographics & Preferred Language Selection (EN / HI)]
       │
       ▼
[Retinal Fundus Image Capture & Upload]
       │
       ▼
[ONNX Primary Inference (<80 ms)] ──► Immediate UI Feedback (Grade 0-4, Confidence, Referable Status)
       │
       ▼ (Asynchronous Background IPC Worker)
[Multi-Modal Evidence Generation (3-5s)]
   ├── Grad-CAM Model Activation Heatmap
   ├── IDRiD Lesion Segmentation (MA, HE, EX, SE % Coverage)
   └── DRIVE Retinal Vessel Segmentation (% Vascular Density)
       │
       ▼
[Database Persistence (SQLite WAL)]
       │
       ▼
[Clinical & Patient Report Generation (Print / PDF / Hindi Localized)]
       │
       ▼
[Specialist Referral Triage / Routine Care Scheduling]
```

---

## 10. Functional Requirements

| ID | Module | Feature | Implementation Status |
|---|---|---|---|
| **FR-01** | Upload | Retinal fundus image upload (JPEG/PNG, 10MB limit) | **IMPLEMENTED** |
| **FR-02** | Demographics | Patient name, age, gender, and preferred language (EN/HI) input | **IMPLEMENTED** |
| **FR-03** | Primary Classification | 5-Class DR Grading (Grade 0–4) + Softmax Probabilities + Confidence | **IMPLEMENTED** |
| **FR-04** | Referable Triage | Referable DR flag (Grade ≥ 2 triggers Specialist Referral recommendation) | **IMPLEMENTED** |
| **FR-05** | Explainability | Grad-CAM feature activation map overlay on retinal image | **IMPLEMENTED** |
| **FR-06** | Lesion Segmentation | IDRiD V3 multi-class lesion segmentation (% area for MA, HE, EX, SE) | **IMPLEMENTED** |
| **FR-07** | Vessel Segmentation | DRIVE V1 retinal vessel mask extraction (% vascular coverage) | **IMPLEMENTED** |
| **FR-08** | Asynchronous Pipeline | Non-blocking IPC worker for background evidence processing | **IMPLEMENTED** |
| **FR-09** | Patient Report | Bilingual patient summary report (English & Hindi) with print support | **IMPLEMENTED** |
| **FR-10** | Doctor Report | Detailed clinical breakdown for ophthalmologist review | **IMPLEMENTED** |
| **FR-11** | Dataset Explorer | Sample image viewer with ground-truth grades and annotations | **IMPLEMENTED** |
| **FR-12** | Model Validation | Real-time clinical metric calculation (Sensitivity, Specificity, AUROC, F1, 5x5 CM) | **IMPLEMENTED** |
| **FR-13** | Network Simulation | Deterministic queue & bandwidth throughput calculator for PHC deployments | **IMPLEMENTED** |
| **FR-14** | Theme System | High-contrast Dark Mode (Default) and Light Mode toggle | **IMPLEMENTED** |

---

## 11. Non-Functional Requirements
- **Truthfulness & Data Integrity**: Model metrics and dataset connectivity reflect actual code execution and disk contents. No synthetic metrics or fabricated predictions.
- **Reliability & Worker Recovery**: The background Python worker automatically recovers and restarts upon process crashes without dropping server requests.
- **Maintainability**: Clean modular separation across React frontend, Express API routes, SQLite storage, ONNX primary inference, and PyTorch IPC workers.

---

## 12. Performance Requirements
Based on empirical bench-marking on the local test suite:
- **ONNX Primary Classification Latency**: `< 80 ms` (mean ~75 ms post pre-warming).
- **Background Evidence Generation**: `3.0 – 4.5 seconds` total for Grad-CAM + Lesion + Vessel segmentation.
- **API Response Time**: `< 100 ms` for primary processing response.
- **Concurrent Request Handling**: Asynchronous IPC queue prevents Express main thread blocking.

---

## 13. UX Requirements
- **Minimalist Aesthetic**: Pure void background in dark mode, crisp typography, no decorative animations, no pill buttons, no emoji icons.
- **Immediate UI Feedback**: Primary classification displays immediately while evidence indicators pulse and transition to "Ready" dynamically.

---

## 14. Accessibility Requirements
- **Theme Support**: High-contrast Dark Mode and Light Mode with CSS custom properties.
- **Responsive Layouts**: Accessible across desktop workstations and mobile tablet devices (1536x826 viewport optimized).

---

## 15. Language Requirements
- **Supported Languages**:
  1. **English (en)** (Default)
  2. **Hindi (हिन्दी) (hi)**
- **Scope**: Patient-facing report (`PatientReport.tsx`) translates all labels, clinical recommendations, next steps, disclaimers, and date locales (`hi-IN`).

---

## 16. Data Requirements
- **Supported Datasets**:
  1. **APTOS 2019**: 3,662 images (DR Severity Grading)
  2. **IDRiD**: 516 disease grading images + 1,476 segmentation files (Lesions & Grading)
  3. **DRIVE**: 100 images (Retinal Vessel Segmentation)
  4. **Messidor-2**: 1,748 fundus images (External Generalization)
- **Database**: SQLite with WAL mode (`database/dr_sugar.db`).

---

## 17. Success Criteria
- Primary classification latency under 100 ms.
- Referable DR Sensitivity exceeding 95% on test benchmarks (APTOS 2019 test split: **98.66%**).
- Zero server crashes during background IPC execution.
- 100% localization accuracy in Hindi patient reports.

---

## 18. Limitations
- **Prototype Status**: Research prototype designed for demonstration and hackathon evaluation.
- **Image Formats**: Supports JPEG and PNG fundus images up to 10 MB.
- **Single Camera Calibration**: Models trained on standardized 45° FOV fundus images.

---

## 19. Future Enhancements
- Automated DICOM / PACS image ingestion.
- Multi-GPU parallel worker scaling.
- Automated PDF report export with digital signature verification.
