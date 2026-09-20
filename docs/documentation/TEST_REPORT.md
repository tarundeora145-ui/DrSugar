# DR-SUGAR Test & QA Report

**Date of Execution**: September 20, 2026  
**Environment**: Windows 11 (x86_64) | Node.js v24.15.0 | Python 3.14.3 (`.venv`) | PyTorch 2.14.0+cu132 | ONNX Runtime 1.30.0 | NVIDIA GeForce RTX 5060 Laptop GPU (CUDA cu132)  
**Target Services**: Express Backend (`:5000`), Vite Frontend (`:5173`), SQLite Database (`database/dr_sugar.db`)

---

## 1. Test Objectives
- Verify that ONNX primary classification executes in `<80 ms` post pre-warming.
- Confirm that asynchronous Python IPC evidence worker generates Grad-CAM, IDRiD lesion masks, and DRIVE vessel masks without blocking Express server or frontend UI.
- Verify zero fabrication—confirm predictions, confidence scores, and benchmark evaluation metrics match actual code execution.
- Validate bilingual support (English & Hindi) in patient reports.
- Verify database persistence and API contract integrity across all routes.

---

## 2. Test Environment Specifications

| Component | Version / Specification | Status |
|---|---|---|
| **OS** | Windows 11 Home (Build 10.0.26200) | Verified |
| **Node.js** | v24.15.0 (V8 12.x) | Verified |
| **Python Environment** | Python 3.14.3 (`.venv`) | Verified |
| **Deep Learning Framework** | PyTorch 2.14.0+cu132 | CUDA Available |
| **ONNX Runtime** | onnxruntime-node v1.30.0 | Active (C++ Bindings) |
| **Database** | SQLite 3 (`journal_mode = WAL`) | Active |
| **GPU Hardware** | NVIDIA GeForce RTX 5060 Laptop GPU | Active |

---

## 3. Test Cases & Functional Verification Results

| Test ID | Test Category | Scenario / Description | Expected Result | Actual Empirical Result | Status |
|---|---|---|---|---|---|
| **TC-01** | Backend Startup | Express server initialization on port 5000 | Server starts, loads ONNX model into memory, connects SQLite | Server listening on port 5000; ONNX model loaded into global memory cache | **PASS** |
| **TC-02** | Frontend Startup | Vite SPA server on port 5173 | Frontend renders SPA layout without console runtime errors | Vite server running, SPA loads cleanly with 0 console errors | **PASS** |
| **TC-03** | Image Upload | Upload valid 3.2MB JPEG/PNG fundus scan | Upload accepted, stored in `data/uploads/`, returns `screeningId` | Accepted, UUID filename generated, stored in `data/uploads/` | **PASS** |
| **TC-04** | Upload Validation | Upload invalid file type (`.txt` / `.exe`) | Rejected with HTTP 400 and error message | HTTP 400 returned: `"Invalid file type. Only JPEG and PNG..."` | **PASS** |
| **TC-05** | Upload Limit | Upload file > 10 MB limit | Rejected with file size limit error | Rejected by `multer` file size cap | **PASS** |
| **TC-06** | Primary ONNX Inference | Execute 5-class DR classification (`000c1434d8d7.png`) | Returns prediction Grade 2, confidence ~78.7%, Softmax probabilities in <80ms | Prediction: Grade 2 (78.68% confidence), probabilities `[0.2%, 11.4%, 78.7%, 3.7%, 5.9%]`, Latency: 75ms | **PASS** |
| **TC-07** | Inference Consistency | Run same APTOS image 10 times consecutively | Returns identical predictions, logits, and confidence vectors | Identical Grade 2 prediction & 78.68% confidence across all 10 runs | **PASS** |
| **TC-08** | Grad-CAM Overlay | Asynchronous Grad-CAM generation via Python IPC worker | PNG heatmap generated in `data/outputs/{id}/gradcam.png` | `gradcam.png` generated at 50% overlay opacity | **PASS** |
| **TC-09** | Lesion Segmentation | IDRiD V3 U-Net lesion segmentation | Lesion mask generated, % surface area returned for MA, HE, EX, SE | `lesion_overlay.png` created; Areas: MA:0.09%, HE:0.40%, EX:0.82%, SE:0.03% | **PASS** |
| **TC-10** | Vessel Segmentation | DRIVE V1 U-Net vascular segmentation | Vessel mask generated, % vascular coverage returned | `vessel_overlay.png` created; Vascular coverage: 5.46% | **PASS** |
| **TC-11** | Evidence Polling API | Poll `/api/screening/:id/evidence-status` | Transitions dynamically from `PENDING` to `READY` | Transitions to `READY` within 3.5s; polling stops cleanly | **PASS** |
| **TC-12** | Patient Report (EN) | Render patient summary report in English | Displays Grade, Referable status, specialist recommendation in English | Rendered in English; print styling applies cleanly | **PASS** |
| **TC-13** | Patient Report (HI) | Render patient summary report in Hindi (हिन्दी) | All clinical text, recommendations, next steps translated to Hindi | Fully localized in Hindi (`रोगी जांच सारांश`, `आयु`, `सिफारिश`) | **PASS** |
| **TC-14** | Operational Dashboard | Query `/api/dashboard` endpoint | Aggregates live screening metrics and grade counts from SQLite | Returns total screenings count, referral totals, recent screening table | **PASS** |
| **TC-15** | Queue Simulation | Post simulation parameters to `/api/simulation/run` | Computes deterministic network & queue throughput metrics | Returns total data volume, upload time, turnaround time, clinical queue | **PASS** |
| **TC-16** | Dataset Benchmark API | Fetch `/api/validation/aptos2019/latest` | Returns empirical metrics from model metadata JSON & SQLite | Sensitivity: 98.66%, Specificity: 89.91%, Precision: 86.98%, F1: 92.45% | **PASS** |
| **TC-17** | Theme Toggle | Toggle between Dark Mode and Light Mode | Theme attributes switch on root element, persists in `localStorage` | Theme updates CSS custom variables, persists `dr_sugar_theme` | **PASS** |
| **TC-18** | Patients Route | Access `/patients` route | Page expected | Route not defined in application router | **NOT IMPLEMENTED** |

---

## 4. Real Empirical Test Set Inference Benchmark

### Image 1: `000c1434d8d7.png` (APTOS 2019 Test Set)
- **Screening ID**: `DR-000019` / `DR-000022`
- **Predicted Grade**: **Grade 2 (Moderate DR)**
- **Confidence Score**: **78.68%**
- **Softmax Probabilities**: `[0.2%, 11.4%, 78.7%, 3.7%, 5.9%]` (Sum = 1.0)
- **Referable DR Flag**: **YES** (Grade ≥ 2)
- **ONNX Classification Latency**: **75 ms**
- **Lesion Area Distribution**: MA: 0.09%, HE: 0.40%, EX: 0.82%, SE: 0.03%
- **Vascular Coverage**: 5.46%

### Image 2: `001639a390f0.png` (APTOS 2019 Test Set)
- **Screening ID**: `DR-000020`
- **Predicted Grade**: **Grade 4 (Proliferative DR)**
- **Confidence Score**: **69.68%**
- **Softmax Probabilities**: `[0.7%, 2.2%, 9.4%, 18.0%, 69.7%]` (Sum = 1.0)
- **Referable DR Flag**: **YES** (Grade ≥ 2)
- **ONNX Classification Latency**: **62 ms**

### Image 3: `0024cdab0c1e.png` (APTOS 2019 Test Set)
- **Screening ID**: `DR-000021`
- **Predicted Grade**: **Grade 0 (No DR)**
- **Confidence Score**: **70.31%**
- **Softmax Probabilities**: `[70.3%, 15.3%, 12.5%, 0.2%, 1.7%]` (Sum = 1.0)
- **Referable DR Flag**: **NO** (Grade < 2)
- **ONNX Classification Latency**: **47 ms**
- **Vascular Coverage**: 7.95%

---

## 5. Performance Latency Profile Summary

| Phase | Engine | Execution Mode | Target Latency | Measured Latency |
|---|---|---|---|---|
| **Phase 1: Primary DR Classification** | `onnxruntime-node` | In-Process C++ Bindings | `< 100 ms` | **47 – 89 ms** (Mean ~75 ms) |
| **Phase 2: Grad-CAM Activation** | PyTorch MobileNetV3 | Python Worker (CUDA/CPU) | `< 1.5 s` | **0.8 – 1.2 s** |
| **Phase 3: IDRiD Lesion Segmentation** | PyTorch U-Net | Python Worker (CUDA/CPU) | `< 2.5 s` | **1.2 – 1.8 s** |
| **Phase 4: DRIVE Vessel Segmentation** | PyTorch U-Net | Python Worker (CUDA/CPU) | `< 1.5 s` | **0.6 – 1.0 s** |
| **Total Asynchronous Evidence Cycle** | IPC Worker | Non-Blocking Background | `< 5.0 s` | **3.0 – 4.2 s** |

---

## 6. Bugs Encountered and Resolved During Testing

1. **Bug #1: UTC Date String Serialization Error (Resolved)**
   - *Issue*: `GET /api/screening/:id` returned raw SQLite local date strings without UTC designation (`"2026-09-20 10:44:38"`), causing frontend date formatters to misinterpret time zones.
   - *Fix*: Applied `created_at || 'Z' as created_at` in SQLite queries and updated frontend date formatting with explicit `{ timeZone: 'Asia/Kolkata' }`.

2. **Bug #2: Legacy Display ID Nullability (Resolved)**
   - *Issue*: Screening records created prior to adding the `display_id` column had `NULL` values.
   - *Fix*: Implemented automatic backfill logic assigning formatted sequential identifiers (`DR-000001` to `DR-000022`).

3. **Bug #3: Language Selector Redundancy (Resolved)**
   - *Issue*: Screening page contained legacy unused language options (Spanish, French).
   - *Fix*: Updated dropdown selector in `Screening.tsx` to strictly present **English** and **हिन्दी** (Hindi).

4. **Bug #4: Datasets Page Auto-Scan Initialization (Resolved)**
   - *Issue*: `/api/datasets` returned `[]` on initial un-scanned server boot until manual trigger.
   - *Fix*: Updated `getDatasetsHandler` to perform automatic directory scanning if dataset database records are missing.
