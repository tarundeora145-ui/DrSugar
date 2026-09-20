# High Level Design (HLD)

## 1. System Overview
**DR-SUGAR** is architected as an end-to-end explainable AI screening system. The architecture separates fast interactive tasks (primary classification, UI updates, patient intake) from heavy computational workloads (multi-modal deep learning feature extraction, visual heatmaps, pixel-level segmentation).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           REACT FRONTEND (SPA)                          │
│   Intro | Dashboard | Screening | Reports | Simulation | Validation     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ REST HTTP (JSON / Multipart)
┌────────────────────────────────────▼────────────────────────────────────┐
│                           EXPRESS API BACKEND                           │
│  - Routes & Middleware      - Image Static Server (/uploads, /outputs)   │
│  - SQLite Database Sync     - Multi-Threaded Request Dispatcher       │
└───────────┬────────────────────────────────────────────────┬────────────┘
            │ In-Process C++ Bindings                        │ Stdin/Stdout IPC
┌───────────▼───────────┐                        ┌───────────▼────────────┐
│ ONNX RUNTIME INFERENCE │                        │ PERSISTENT PYTHON IPC  │
│  MobileNetV3 (5-Class)│                        │    EVIDENCE WORKER     │
│  Latency: < 80 ms     │                        │  - PyTorch CUDA/CPU    │
└───────────────────────┘                        │  - Grad-CAM Engine     │
                                                 │  - IDRiD Lesion Engine │
                                                 │  - DRIVE Vessel Engine │
                                                 └───────────┬────────────┘
                                                             │ Static PNG & JSON
                                                 ┌───────────▼────────────┐
                                                 │ DATA OUTPUTS STORAGE   │
                                                 │ data/outputs/{id}/     │
                                                 └────────────────────────┘
```

---

## 2. Architectural Goals
- **Decoupled 2-Phase Execution**: Primary classification completes synchronously within milliseconds; secondary evidence generates asynchronously in the background.
- **IPC Worker Isolation**: ML evidence algorithms (PyTorch) execute in a dedicated Python child process to isolate memory, GPU context, and heavy dependencies from the Express Node.js event loop.
- **Single Source of Truth**: All screening records, quality metrics, evidence statuses, and validation benchmarks persist in an embedded SQLite database with Write-Ahead Logging (WAL).

---

## 3. Major Components

### 3.1 React Frontend
- **Framework**: React 19 SPA with Client-Side Routing via React Router DOM v7.
- **Theme Provider**: Supports dynamic high-contrast Dark Mode (Default) and Light Mode.
- **Pages**:
  - `Introduction.tsx`: Hero vision, pipeline breakdown, clinical transparency specs.
  - `Screening.tsx`: Multipart image upload, patient demographics intake, real-time stage progress polling, interactive Grad-CAM and lesion overlay viewer.
  - `Reports.tsx`: Historical screening registry with search and referral filtering.
  - `PatientReport.tsx`: Patient-facing summary localized in English and Hindi (हिन्दी) with print formatting.
  - `DoctorReport.tsx`: Specialized clinical breakdown for ophthalmologist review.
  - `Dashboard.tsx`: Operational triage statistics and DR grade distributions.
  - `Simulation.tsx`: Deployment queue & network throughput modeling tool.
  - `Validation.tsx`: Benchmark dataset exploration and empirical model validation dashboard.

### 3.2 Express API Backend
- **Framework**: Express 5.
- **Role**: Coordinates image uploads (`multer`), executes in-process ONNX model classification (`onnxruntime-node`), manages database CRUD operations (`better-sqlite3`), and dispatches background jobs to the Python evidence worker.

### 3.3 Database Layer
- **Engine**: SQLite 3 (`database/dr_sugar.db`).
- **Pragma**: `journal_mode = WAL` for concurrent read/write performance.
- **Tables**: `screenings`, `screening_quality`, `screening_analysis`, `datasets`, `dataset_files`, `dataset_labels`, `dataset_annotations`, `reports`, `validation_runs`.

### 3.4 Primary ONNX Classification Engine
- **Model**: `MobileNetV3-Small` exported to ONNX (`models/dr_classification/model.onnx`).
- **Execution**: In-process via `onnxruntime-node` C++ bindings.
- **Preprocessing**: `Sharp` resizes image to `224x224 RGB` with `fit: fill` (no aspect cropping) and applies ImageNet normalization.

### 3.5 Persistent Python Evidence Worker
- **Script**: `scripts/evidence_worker.py`.
- **IPC Mechanism**: Express spawns the process on startup; request jobs are piped over `stdin` as JSON lines, and results are emitted over `stdout` as JSON lines.
- **Sub-Engines**:
  1. **Grad-CAM Engine**: Generates class-activation visual heatmaps overlaid at 50% opacity.
  2. **Lesion Segmentation Engine**: PyTorch U-Net trained on IDRiD V3, calculates % surface area for MA, HE, EX, and SE.
  3. **Vessel Segmentation Engine**: PyTorch U-Net trained on DRIVE V1, extracts binary vascular mask and calculates % vessel coverage.

---

## 4. Component Responsibilities & Interfaces

```
[Screening Page] ──► POST /api/screening/upload ──► Store Upload File ──► Return Screening ID
       │
       ▼
[Screening Page] ──► POST /api/screening/:id/process
                           │
                           ├──► ONNX Primary Inference (<80ms)
                           ├──► Save Classification to SQLite
                           ├──► Return Immediate Response (Grade 0-4, Conf, Referable Flag)
                           │
                           └──► Enqueue Job to EvidenceWorker (Background)
                                      │
                                      ▼
                               [Python Evidence Worker]
                                  ├── Grad-CAM Heatmap
                                  ├── Lesion Segmentation
                                  └── Vessel Mask
                                      │
                                      ▼
                               [Save PNG Artifacts & Update SQLite]
```

---

## 5. Background Processing Architecture

```
Express Backend (Node.js)                         Python Worker (PyTorch / CUDA)
       │                                                      │
       │─── Spawn process (`scripts/evidence_worker.py`) ────►│
       │                                                      │ (Loads models into memory)
       │◄─── Emit {"status": "READY"} ────────────────────────│
       │                                                      │
[Incoming Job]                                                │
       │─── Write JSON to stdin ─────────────────────────────►│
       │    {"id": "28", "image": "...", "class_idx": 2}      │ (Generates Grad-CAM, Lesion, Vessel)
       │                                                      │
       │◄─── Emit JSON to stdout ─────────────────────────────│
       │    {"id": "28", "status": "OK", "gradcam_path":...}  │
       │                                                      │
[Update SQLite status = 'COMPLETED']                          │
```

---

## 6. Persistence & Storage Structure

```
c:/Users/Tarun/Desktop/sugar/
├── data/
│   ├── uploads/                # Original uploaded fundus images
│   ├── outputs/
│   │   └── {screening_id}/     # Generated visual evidence artifacts
│   │       ├── gradcam.png
│   │       ├── lesion_overlay.png
│   │       └── vessel_overlay.png
│   ├── aptos2019/             # APTOS 2019 dataset files
│   ├── idrid/                 # IDRiD dataset files
│   ├── drive/                 # DRIVE dataset files
│   └── messidor2/             # Messidor-2 dataset files
├── database/
│   └── dr_sugar.db            # SQLite relational database file
└── models/
    ├── dr_classification/     # ONNX primary model + weights + metadata
    ├── idrid/                 # IDRiD grading & lesion models
    └── drive/                 # DRIVE vessel segmentation model
```

---

## 7. Security & Input Hardening
- **Upload Restrictions**: Capped at 10 MB per file; MIME-type validated against `image/jpeg` and `image/png`.
- **Database Sanitization**: All database queries parameterize inputs via `better-sqlite3` prepared statements.
- **Path Resolution Hardening**: Path operations verify resolved output targets remain inside designated workspace subdirectories.

---

## 8. Deployment Model
Current deployment configuration is an **on-premise / local workstation server** setup suitable for rural PHC edge deployments:
- Single Node.js Express process serving static frontend assets and REST endpoints.
- Persistent Python background worker handling local GPU/CPU ML inference.
- Embedded zero-config SQLite database requiring no external database server administration.
