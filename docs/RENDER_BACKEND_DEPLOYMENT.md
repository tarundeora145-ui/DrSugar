# DR-SUGAR Render Deployment Guide

This document explains how to deploy the persistent Express + SQLite backend for DR-SUGAR specifically on Render.

## Render Service Type
**Persistent Web Service** (must have a persistent disk mounted).

## Root Directory
`server`

## Build Command
```bash
npm install && npm run build && pip install -r requirements.txt
```

## Start Command
```bash
npm start
```

## Environment Variables

Configure these in the Render dashboard:

- `PORT` (Render sets this automatically)
- `NODE_ENV`: `production`
- `FRONTEND_URL`: `https://dr-sugar.vercel.app` (or your actual Vercel URL)
- `DATABASE_PATH`: `/var/data/dr_sugar.db` (must be on the persistent disk)
- `OUTPUT_DIR`: `/var/data/outputs`
- `UPLOAD_DIR`: `/var/data/uploads`
- `MODELS_DIR`: `/opt/render/project/src/models`
- `PYTHON_BIN`: `python3` (or the exact path to Python 3 on the Render instance)
- `EVIDENCE_WORKER_PATH`: `/opt/render/project/src/scripts/evidence_worker.py`

*(Note: Never expose API keys or passwords in the client frontend. DR-SUGAR does not require any secrets.)*

## Persistent Disk
A persistent disk MUST be configured on Render because the backend requires persistent storage for:
- SQLite database (`dr_sugar.db`)
- Patient uploads (`uploads/`)
- Grad-CAM, lesion, and vessel generated outputs (`outputs/`)

Mount path should be (for example): `/var/data`.

## Python
The evidence worker uses Python. 
Required packages are listed in `server/requirements.txt`:
- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `pillow`

## Models
The deployed backend must have the required trained model artifacts. You must upload or keep these in source control:
- `models/dr_classification/model.onnx`
- `models/dr_classification/metadata.json`
- `models/idrid/lesions/v3/fold1_best.pth`
- `models/drive/v1/fold1_best.pth`

## Datasets
**Training/testing datasets are NOT deployed.**
Do NOT upload `data/aptos2019/`, `data/idrid/`, `data/drive/`, `data/messidor2/`, or `data/uploads/`.

## Frontend
Once the Render backend is live, update the Vercel frontend by setting this environment variable:
`VITE_API_URL=https://<your-render-backend-url>/api`

## Health Check
You can verify the backend is running at:
`GET /api/health`
