# DR-SUGAR Current State Audit

## Executive Summary
This audit evaluated the DR-SUGAR repository against its core rules: "Never fabricate data/outputs," "Preserve working functionality," and "Implement the complete workflow architecture without faking AI." The workspace represents a highly structured, honest full-stack prototype. It successfully bridges a modern React frontend and a Node/SQLite backend while deliberately leaving the clinical AI, MATLAB engines, and Simulink components as structural "UNAVAILABLE" stubs until real models are integrated.

## Current Architecture
The architecture strictly follows a decoupled client-server model:
- **Client**: Vite + React + Tailwind CSS (using sparse, lightweight typography, organic canvases, and asymmetric layouts).
- **Server**: Express.js + Node.js routing layer.
- **Persistence**: SQLite (via `better-sqlite3` in WAL mode).
- **Engines (Pending)**: MATLAB (`matlab/`) and Simulink (`simulink/`) acting as computational boundaries.

## Frontend
- **Verified from source**: The `frontend/` directory is a complete Vite/React application. It implements the exact `DESIGN.md` aesthetics via vast typography and zero card-grids.
- **Key Components**: `RetinalConstellation.tsx` (procedural canvas), `DataExplorer.tsx` (dataset interactions), `ExplainabilityPanel.tsx` (heatmap logic), `SimulationDashboard.tsx` (rural deployment config), `ValidationDashboard.tsx` (metrics UI).
- **Routing**: `App.tsx` correctly maps `/`, `/dashboard`, `/screening`, `/reports`, `/simulation`, and `/validation`.

## Backend
- **Verified from source**: `server/src/` contains an Express API.
- **Routes**: `datasets.ts`, `health.ts`, `models.ts`, `screening.ts`, `simulation.ts`, `validation.ts`.
- **Integrity**: Endpoint controllers rigorously reject fabricating numbers. For instance, `/api/simulation/run` always throws an intentional payload error (`success: false, message: 'SIMULATION UNAVAILABLE'`) to prevent hallucinated queue metrics.

## Database
- **Verified from source**: `server/src/database/db.ts` successfully initializes an SQLite db (`database/dr_sugar.db`).
- **Tables Extracted**: `datasets`, `dataset_files`, `dataset_labels`, `dataset_annotations`, `screenings`, `screening_quality`, `screening_analysis`, `reports`, `validation_runs`.
- **Usage**: SQLite is actively used to persist dataset filesystem scans, screening session logs, and validation runs. It is not currently storing image blobs (honoring project rules).

## Dataset System
- **Verified from source**: `server/src/datasets/loaders/` contains `AptosLoader.ts`, `DriveLoader.ts`, `IdridLoader.ts`, and `MessidorLoader.ts`.
- **Functional**: The dataset scanning is real. It traverses local directories, validates image readability, maps them to labels (if present), and updates SQLite status (`CONNECTED`, `PARTIAL`, `MISSING`, `INVALID`) based on actual disk reality. There is zero fake/hard-coded dataset availability.

## AI/ML
- **Verified from source**: Absolutely no AI inference code is executed.
- **Integration**: The `/api/models` and `/api/screening/:id/explainability` routes exist but return `MODEL UNAVAILABLE` and `NOT EVALUATED`. The UI gracefully handles these fallbacks. 

## MATLAB
- **Verified from source**: Structural `.m` files exist in `matlab/classification/`, `matlab/lesions/`, `matlab/preprocessing/`, `matlab/quality/`, and `matlab/segmentation/`. 
- **Connection**: There is currently **no connection** between Node and MATLAB. The `.m` files are isolated architectural stubs.

## Simulink
- **Verified from source**: `simulink/run_deployment_simulation.m` exists.
- **Connection**: The Node backend API (`/api/simulation/run`) has no child process bridging to Simulink. The MATLAB script itself immediately throws an explicit error if run, preventing pseudo-queueing.

## Reports
- **Verified from source**: Frontend has `Reports.tsx`. Backend has `reports` table in SQLite. 
- **Implementation**: Structural layout exists, but no deep clinical PDF generation or HL7 reporting is wired up yet. 

## Validation
- **Verified from source**: `validation.ts` routes and `ValidationDashboard.tsx` exist.
- **Implementation**: The UI maps the 90% Sensitivity / 85% Specificity targets and Referable DR (Level ≥ 2) thresholds, but explicitly states `NOT EVALUATED` and nullifies the Confusion Matrix rather than faking data.

## Dependency/Code Relationships
- **Frontend ↔ Backend**: API calls are explicitly made using standard `fetch` hooks over HTTP (`http://localhost:5000`).
- **Backend ↔ Filesystem**: Handled safely via Node `fs/promises` inside the loaders.
- **Backend ↔ MATLAB**: Completely disconnected.

## Implemented vs Mocked
- **Implemented**: Full UI shell, SQLite schema, real dataset parsing and validation, real local file streaming.
- **Mocked/Stubs**: AI Inference, Explanations (Grad-CAM), Clinical Validation metrics, Queue Simulation. (These are stubbed strictly to return failure states, avoiding hallucination).

## Conflicts Created by Earlier Prompts
- **None**: Because the earlier instructions strictly forbade fabricating data or diverging from the rules, the architecture successfully scaled without introducing contradictory code. The UI anticipates data that the backend formally refuses to fake.

## Missing Components
- MATLAB Engine API for Node (or Python middleware) to execute `.m` scripts dynamically.
- A functional PyTorch/ONNX backend if MATLAB is not used for inference.
- Authentication/RBAC.

## Broken Components
- **None structurally broken**. The "errors" thrown in Simulation and Models are intentional architectural constraints.

## Files to Preserve
- `AGENTS.md`, `DESIGN.md` (Project rules).
- All `frontend/` UI shells and `RetinalConstellation.tsx`.
- All `server/src/datasets/loaders/` logic.
- `server/src/database/db.ts`.

## Files Requiring Changes
- `server/src/routes/screening.ts`: Must be wired to an actual ML inference pipeline.
- `server/src/routes/simulation.ts`: Must spawn a MATLAB instance to run `run_deployment_simulation.m`.

## Recommended Reconciliation Plan
1. **Engine Integration**: Establish a middleware bridging Node.js to the MATLAB/Simulink executables.
2. **Model Loading**: Integrate real `.onnx` or `.pt` weights into the inference paths.
3. **Database Seeding**: Allow users to initiate real screenings through the UI that persist physical image paths to SQLite.

## Risks
- If the MATLAB Engine API proves too brittle for Node.js, the backend may need to orchestrate a Python/FastAPI microservice for inference.
