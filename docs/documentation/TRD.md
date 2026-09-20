# Technical Requirements Document (TRD)

## 1. Technical Overview
**DR-SUGAR** is architected as a decoupled, multi-runtime application:
1. **Frontend**: Single-Page Application (SPA) built with React 19, TypeScript, Vite, and Tailwind CSS v4.
2. **Backend API**: Node.js REST server using Express 5, TypeScript, and `better-sqlite3` with Write-Ahead Logging (WAL).
3. **Primary ML Inference**: High-performance in-process C++ bindings via `onnxruntime-node` (MobileNetV3 5-class classification).
4. **Secondary Evidence Worker**: Asynchronous Python worker process running PyTorch with CUDA/CPU acceleration communicating over `stdin`/`stdout` JSON IPC.

---

## 2. Technology Stack & Installed Versions

| Layer | Component / Tool | Version | Purpose |
|---|---|---|---|
| **Frontend** | React | `^19.2.8` | UI Component Framework |
| | React Router DOM | `^7.18.4` | Client-Side Routing (`/`, `/dashboard`, `/screening`, `/reports`, `/simulation`, `/validation`) |
| | TypeScript | `~6.0.2` | Static Type Safety |
| | Vite | `^8.3.0` | Frontend Bundler & Dev Server |
| | Tailwind CSS | `^4.3.3` | Styling Engine with `@theme` Custom Properties |
| | Lucide React | `^1.47.0` | UI Icons |
| **Backend** | Node.js | `>=20.0.0` | Server Runtime |
| | Express | `^5.2.1` | HTTP REST API Server |
| | better-sqlite3 | `^13.0.3` | Synchronous High-Performance SQLite Binding |
| | onnxruntime-node | `^1.30.0` | In-Process C++ ONNX Model Execution |
| | Sharp | `^0.35.4` | Image Resizing & Preprocessing (224x224 RGB) |
| | Multer | `^2.4.0` | Multipart Form Image Upload Handling |
| **ML & IPC** | Python | `3.10+` | Evidence Generation Runtime (`.venv`) |
| | PyTorch | `2.x` | Deep Learning Framework |
| | torchvision | `0.x` | Image Transforms & Vision Architectures |
| | PIL / Pillow | `10.x` | Image Manipulation & Overlay Generation |
| | NumPy | `1.24+` | Tensor Computations & Mask Array Operations |
| **Database** | SQLite | `3.x` | Local Embedded Relational Database |

---

## 3. System Requirements
- **Operating System**: Windows 10/11, Ubuntu 22.04 LTS, or macOS 14+.
- **Node.js Environment**: Node.js v20.0+ and npm v10.0+.
- **Python Environment**: Python 3.10+ virtual environment (`.venv`) with PyTorch installed.
- **Storage**: Minimum 5 GB free disk space (for dataset storage in `data/` and ONNX/PyTorch models in `models/`).

---

## 4. Hardware Requirements

### Minimum Requirements (CPU Mode)
- **CPU**: Dual-Core x86_64 / ARM64 processor (2.0 GHz+).
- **RAM**: 8 GB system memory.
- **Inference Speed**: Primary classification ~75 ms (ONNX CPU); Evidence worker ~3.5 s (PyTorch CPU).

### Recommended Requirements (GPU Accelerated Mode)
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 7.5+, 4GB+ VRAM).
- **RAM**: 16 GB system memory.
- **Inference Speed**: Primary classification ~20 ms; Evidence worker ~0.8 s (PyTorch CUDA).

---

## 5. API Requirements & Route Specifications

| Method | Endpoint | Description | Request Body / Params | Response Payload |
|---|---|---|---|---|
| `GET` | `/api/health` | Server health check | None | `{ status: 'OK', timestamp }` |
| `POST` | `/api/screening/upload` | Multipart fundus image upload | `file`, `patient_name`, `patient_age`, `patient_gender`, `preferred_language` | `{ success: true, screeningId }` |
| `POST` | `/api/screening/:id/process` | Trigger ONNX classification & enqueue evidence | `id` (Param) | `{ success: true, status: 'CLASSIFIED', classification_ms, data: { prediction, probabilities, confidence, referable_dr } }` |
| `GET` | `/api/screening/:id/evidence-status` | Lightweight polling endpoint for evidence status | `id` (Param) | `{ gradcam: { status, url }, lesion: { status, url, data }, vessel: { status, url, data } }` |
| `GET` | `/api/screening/:id` | Fetch full screening record | `id` (Param) | `{ success: true, data: ScreeningRecord }` |
| `GET` | `/api/reports` | List all historical screenings | None | `Array<ScreeningSummary>` |
| `GET` | `/api/dashboard` | Aggregated operational metrics | None | `{ totalScreenings, referableCount, normalCount, avgConfidence, gradeCounts }` |
| `GET` | `/api/datasets` | List registered datasets & statuses | None | `Array<DatasetSummary>` |
| `GET` | `/api/datasets/scan` | Re-scan disk for dataset files | None | `{ success: true, scanned, validation }` |
| `GET` | `/api/validation/:datasetId/latest` | Fetch latest validation benchmark | `datasetId` (Param) | `{ success: true, data: ValidationRunRecord }` |
| `POST` | `/api/validation/:datasetId/run` | Execute validation evaluation | `datasetId` (Param) | `{ success: true, data: ValidationRunRecord }` |
| `POST` | `/api/simulation/run` | Execute queue simulation | `{ patientsPerYear, imagesPerPatient, imageSizeMB, bandwidthMbps, processingTimeS, reviewCapacity }` | `{ success: true, data: SimulationMetrics }` |

---

## 6. Database Requirements & Relational Schema

Database File: `database/dr_sugar.db` (SQLite 3 with WAL Pragma).

```sql
CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('CONNECTED', 'PARTIAL', 'MISSING', 'INVALID')),
    last_scanned DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dataset_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE dataset_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    dr_grade INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE dataset_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    annotation_type TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE screenings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_id TEXT,
    patient_id TEXT,
    patient_name TEXT,
    patient_age INTEGER,
    patient_gender TEXT,
    preferred_language TEXT DEFAULT 'en',
    image_path TEXT NOT NULL,
    status TEXT NOT NULL,
    result_grade INTEGER,
    probabilities TEXT,
    confidence REAL,
    referable_dr INTEGER,
    model_version TEXT,
    gradcam_path TEXT,
    lesion_path TEXT,
    lesion_data TEXT,
    vessel_path TEXT,
    vessel_data TEXT,
    evidence_status TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    results TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);
```

---

## 7. Machine Learning Requirements

### Primary Classification Model
- **Format**: ONNX (`models/dr_classification/model.onnx`).
- **Base Architecture**: `MobileNetV3-Small` (2.54M parameters).
- **Input Dimension**: `[1, 3, 224, 224]` Float32 tensor.
- **Normalization**: ImageNet Mean `[0.485, 0.456, 0.406]` / Std `[0.229, 0.224, 0.225]`.
- **Output**: 5-class logit vector `[Grade 0, Grade 1, Grade 2, Grade 3, Grade 4]`.
- **Inference Runtime**: `onnxruntime-node` C++ bindings.

### Secondary Evidence Models
- **Grad-CAM Model**: PyTorch MobileNetV3 (`models/dr_classification/best_model.pth`), target feature layer `features[-1]`.
- **Lesion Model**: PyTorch U-Net (`models/idrid/lesions/v3/fold1_best.pth`), outputs 4 lesion probability channels (`MA`, `HE`, `EX`, `SE`).
- **Vessel Model**: PyTorch U-Net (`models/drive/v1/fold1_best.pth`), outputs binary vessel mask.

---

## 8. File Storage Requirements
- `data/uploads/`: Stores raw uploaded patient retinal fundus photographs.
- `data/outputs/{id}/`: Stores generated visual evidence artifacts (`gradcam.png`, `lesion_overlay.png`, `vessel_overlay.png`).
- `data/aptos2019/`, `data/idrid/`, `data/drive/`, `data/messidor2/`: Store local benchmark dataset files.

---

## 9. Security Requirements
- **Input Filtering**: Strict MIME-type checking on uploads (JPEG/PNG only) and file size capping (10 MB max).
- **Path Traversal Protection**: Explicit `path.resolve` containment checks preventing arbitrary file access outside `data/` directories.
- **SQL Injection Prevention**: Parameterized queries using `better-sqlite3` prepared statements.

---

## 10. Error Handling & Recovery
- **Model Warmup Safeguard**: Express server fails gracefully at startup if primary ONNX model files are missing.
- **Worker Auto-Restart**: If the Python evidence worker crashes or exits, `EvidenceWorker` automatically restarts the process after a 2-second delay.
