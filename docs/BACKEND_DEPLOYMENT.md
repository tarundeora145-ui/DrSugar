# DR-SUGAR Backend Deployment Guide

This document explains how to deploy the persistent Express + SQLite backend for DR-SUGAR.
The frontend has already been deployed to Vercel and expects this backend to be available over HTTPS.

## Architecture Context

```text
                    ┌────────────────────┐
                    │      Vercel        │
                    │ React + Vite       │
                    └─────────┬──────────┘
                              │
                       HTTPS API calls
                              │
                              ▼
                    ┌────────────────────┐
                    │ Persistent Backend │
                    │ Express + Node     │
                    ├────────────────────┤
                    │ SQLite             │
                    │ ONNX Runtime       │
                    │ Python Worker      │
                    │ Models             │
                    │ Uploads            │
                    └────────────────────┘
```

The backend **cannot** be deployed to Vercel or any other serverless environment because it requires:
1. A persistent filesystem for SQLite (`dr_sugar.db`)
2. A persistent filesystem for uploading patient images and generating evidence (Grad-CAM, Lesion masks, etc.)
3. A long-running Python child process (`evidence_worker.py`) that stays resident in memory to avoid cold starts.

## Host Requirements

Your deployment host (e.g. AWS EC2, DigitalOcean Droplet, Render Background Worker, Railway) must provide:

- **Node.js**: v18 or newer
- **Python**: 3.9 or newer
- **Persistent Disk**: Required for SQLite, image uploads, and outputs.
- **RAM**: At least 2GB (ONNX Runtime + Python worker consume memory)

### Required Python Packages
```bash
pip install torch torchvision numpy opencv-python pillow
```

## Environment Variables

Configure the following environment variables in your production host:

- `PORT`: The port the Express server will listen on (default: `5000`).
- `NODE_ENV`: Set to `production`.
- `FRONTEND_URL`: Set to the Vercel URL (e.g., `https://dr-sugar.vercel.app`) for strict CORS, or `*` to allow all.
- `DATABASE_PATH`: Absolute path to where `dr_sugar.db` should be stored (e.g. `/app/data/db/dr_sugar.db`).
- `OUTPUT_DIR`: Absolute path for generated Grad-CAM/Lesion evidence outputs.
- `UPLOAD_DIR`: Absolute path for patient uploaded images.
- `MODELS_DIR`: Absolute path to the folder containing `dr_classification/model.onnx` and its `metadata.json`.
- `DATA_DIR`: Absolute path to the training/validation datasets folder (if hosting datasets).
- `PYTHON_BIN`: Path to the Python executable to run the worker (e.g. `/usr/bin/python3`).
- `EVIDENCE_WORKER_PATH`: Absolute path to `scripts/evidence_worker.py`.

*Note: Never expose API keys or passwords in the client frontend.*

## Build and Start

1. Install dependencies:
   ```bash
   npm install
   ```

2. Build the TypeScript source:
   ```bash
   npm run build
   ```

3. Start the production server:
   ```bash
   npm start
   ```

## Model Files Requirement

You **must** upload the `models/` directory to the server. The ONNX Runtime requires `models/dr_classification/model.onnx` and `models/dr_classification/metadata.json` to perform the phase-1 instantaneous classification.

**DO NOT UPLOAD:** The `/data/aptos2019/`, `/data/idrid/`, etc. directories. These are gigabytes of training data and are strictly not needed for production screening endpoints.

## Frontend Connection

Once the backend is live, update the Vercel frontend by setting this environment variable in the Vercel dashboard:

```env
VITE_API_URL=https://your-deployed-backend.example.com/api
```

This will automatically route all dashboard, screening, reporting, and validation calls to this persistent API.

## Health Endpoint

You can monitor the backend using the health endpoint:

```
GET https://your-deployed-backend.example.com/api/health
```

Returns:
```json
{
  "status": "ok",
  "timestamp": "2023-10-25T12:00:00Z",
  "service": "DR-SUGAR API"
}
```
