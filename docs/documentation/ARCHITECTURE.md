# Architecture Specification

This document provides visual architectural specifications for **DR-SUGAR**, detailing system components, pipeline interactions, screening data flow, and deployment models.

---

## 1. High-Level Component Architecture

```
                  ┌─────────────────────────────────────────┐
                  │            Client Web Browser           │
                  │   React 19 SPA (Vite + Tailwind CSS v4) │
                  └────────────────────┬────────────────────┘
                                       │ HTTP / REST API
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │           Express Backend API           │
                  │             (Node.js / TS)              │
                  └────────┬───────────────────────┬────────┘
                           │                       │
         In-Process C++    │                       │ JSON Line IPC
         Binding           │                       │ via Stdin/Stdout
                           ▼                       ▼
           ┌──────────────────────┐    ┌──────────────────────────┐
           │ ONNX Runtime Engine  │    │ Persistent Python Worker │
           │ MobileNetV3 (5-Class)│    │ (PyTorch + CUDA / CPU)   │
           └──────────────────────┘    └───────────┬──────────────┘
                                                   │
                                                   ├── Grad-CAM Heatmaps
                                                   ├── Lesion Segmentation
                                                   └── Vessel Segmentation
                                                   │
                                                   ▼
┌───────────────────────┐              ┌──────────────────────────┐
│   SQLite Database     │◄─────────────┤  Local File Storage      │
│ (database/dr_sugar.db)│              │  (data/uploads, outputs) │
└───────────────────────┘              └──────────────────────────┘
```

---

## 2. Frontend → Backend → ML Architecture Detail

```
┌────────────────────────────────────────────────────────────────────────┐
│ FRONTEND LAYER (React 19)                                             │
│                                                                        │
│  [Screening.tsx] ───► [ThemeContext.tsx] ───► [PatientReport.tsx]      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTP POST / GET
┌──────────────────────────────────▼─────────────────────────────────────┐
│ BACKEND ROUTING LAYER (Express 5)                                      │
│                                                                        │
│  /api/screening/upload ──► /api/screening/:id/process ──► /api/reports │
└─────────────────┬──────────────────────┬───────────────────────────────┘
                  │                      │
                  ▼                      ▼
┌──────────────────────────┐  ┌──────────────────────────────────────────┐
│ PRIMARY ONNX CLASSIFIER  │  │ ASYNCHRONOUS EVIDENCE WORKER             │
│ (onnxruntime-node)       │  │ (scripts/evidence_worker.py)             │
│                          │  │                                          │
│ - Sharp Resize (224x224) │  │ ┌──────────────────────────────────────┐ │
│ - ImageNet Normalization │  │ │ Grad-CAM Engine (MobileNetV3)        │ │
│ - Logits + Softmax       │  │ ├──────────────────────────────────────┤ │
│ - Latency: < 80 ms       │  │ │ Lesion Engine (IDRiD U-Net Fold 1)    │ │
└──────────────────────────┘  │ ├──────────────────────────────────────┤ │
                              │ │ Vessel Engine (DRIVE U-Net Fold 1)     │ │
                              │ └──────────────────────────────────────┘ │
                              └──────────────────────────────────────────┘
```

---

## 3. Asynchronous Screening Workflow

```
User (Operator)            Express API                  ONNX Classifier           Python IPC Worker            SQLite DB
       │                        │                              │                         │                         │
       │── Upload Image ───────►│                              │                         │                         │
       │                        │── Insert Record (UPLOADED) ─────────────────────────────────────────────────────►│
       │◄── Return Screening ID ┤                              │                         │                         │
       │                        │                              │                         │                         │
       │── Trigger Process ────►│                              │                         │                         │
       │                        │── Read Image Buffer ────────►│                         │                         │
       │                        │◄── Return Logits & Softmax ──┤                         │                         │
       │                        │── Update DB (CLASSIFIED) ───────────────────────────────────────────────────────►│
       │◄── Return Primary Res ─┤ (Grade, Conf, Referable DR)  │                         │                         │
       │    (UI Displays Grade) │                              │                         │                         │
       │                        │── Enqueue Background Job ─────────────────────────────►│                         │
       │                        │                              │                         │ (Grad-CAM + Lesions)    │
       │                        │◄── Emit OK JSON line ──────────────────────────────────┤                         │
       │                        │── Save PNGs & Update DB (COMPLETED) ───────────────────────────────────────────►│
       │                        │                              │                         │                         │
       │── Poll Evidence Status►│                              │                         │                         │
       │◄── Return READY Status ┤                              │                         │                         │
       │    (UI Shows Heatmaps) │                              │                         │                         │
```

---

## 4. Database Relationship Architecture

```
  ┌─────────────────────────┐             ┌─────────────────────────┐
  │        datasets         │             │      dataset_files      │
  ├─────────────────────────┤             ├─────────────────────────┤
  │ PK  id                  │1           *│ PK  id                  │
  │     name                ├────────────►│ FK  dataset_id          │
  │     path                │             │     filename            │
  │     status              │             │     relative_path       │
  │     last_scanned        │             │     type                │
  └────────────┬────────────┘             └─────────────────────────┘
               │
               │1                         ┌─────────────────────────┐
               │                          │     dataset_labels      │
               │                          ├─────────────────────────┤
               │                         *│ PK  id                  │
               ├─────────────────────────►│ FK  dataset_id          │
               │                          │     filename            │
               │                          │     dr_grade            │
               │                          └─────────────────────────┘
               │
               │                          ┌─────────────────────────┐
               │                          │   dataset_annotations   │
               │                         *├─────────────────────────┤
               └─────────────────────────►│ PK  id                  │
                                          │ FK  dataset_id          │
                                          │     filename            │
                                          │     annotation_type     │
                                          │     data (JSON)         │
                                          └─────────────────────────┘

  ┌─────────────────────────┐             ┌─────────────────────────┐
  │       screenings        │             │    screening_quality    │
  ├─────────────────────────┤             ├─────────────────────────┤
  │ PK  id                  │1           *│ PK  id                  │
  │     display_id          ├────────────►│ FK  screening_id        │
  │     patient_name        │             │     focus, sharpness... │
  │     patient_age         │             │     status              │
  │     patient_gender      │             └─────────────────────────┘
  │     preferred_language  │
  │     image_path          │             ┌─────────────────────────┐
  │     status              │             │   screening_analysis    │
  │     result_grade        │            *├─────────────────────────┤
  │     probabilities (JSON)├────────────►│ PK  id                  │
  │     confidence          │             │ FK  screening_id        │
  │     referable_dr        │             │     optic_disc, fovea...│
  │     gradcam_path        │             └─────────────────────────┘
  │     lesion_path         │
  │     lesion_data (JSON)  │             ┌─────────────────────────┐
  │     vessel_path         │             │         reports         │
  │     vessel_data (JSON)  │            *├─────────────────────────┤
  │     evidence_status     │────────────►│ PK  id                  │
  └─────────────────────────┘             │ FK  screening_id        │
                                          │     report_data (JSON)  │
                                          └─────────────────────────┘
```

---

## 5. Standalone Deployment Model

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      RURAL HEALTHCARE EDGE NODE                           │
│                                                                           │
│  ┌────────────────────────┐            ┌───────────────────────────────┐  │
│  │   Vite Static Bundle   │            │   Node.js Server Process      │  │
│  │  React 19 SPA Assets   ├───────────►│  Express 5 API (Port 5000)    │  │
│  └────────────────────────┘            └──────────────┬────────────────┘  │
│                                                       │                   │
│                                                       ├─► SQLite Database │
│                                                       │   (dr_sugar.db)   │
│                                                       │                   │
│                                                       ├─► ONNX Runtime    │
│                                                       │   (C++ Binding)   │
│                                                       │                   │
│                                                       └─► Python Worker   │
│                                                           (IPC Child Proc)│
└───────────────────────────────────────────────────────────────────────────┘
```
