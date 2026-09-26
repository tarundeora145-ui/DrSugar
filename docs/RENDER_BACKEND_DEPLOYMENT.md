# DR-SUGAR Render Deployment Guide

This document explains how to deploy the persistent Express + SQLite backend for DR-SUGAR specifically on Render.

## Render Service Type
**Web Service** (Free Tier or Paid)

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
- `MODELS_DIR`: `/opt/render/project/src/models`
- `PYTHON_BIN`: `python3` (or the exact path to Python 3 on the Render instance)
- `EVIDENCE_WORKER_PATH`: `/opt/render/project/src/scripts/evidence_worker.py`

*(Note: Data paths like `DATABASE_PATH`, `OUTPUT_DIR`, and `UPLOAD_DIR` will automatically fall back to an ephemeral `./runtime/` directory inside your app if not provided. If you upgrade to a paid persistent disk, you can explicitly set them to `/var/data/...`)*

*(Note: Never expose API keys or passwords in the client frontend. DR-SUGAR does not require any secrets.)*

## Storage & Ephemeral Data (Free Tier)
Because Render Free does not support persistent disks, the backend is configured to safely fall back to using `./runtime/` directories for data if `NODE_ENV=production` and paths aren't manually overridden.
- SQLite database (`./runtime/dr_sugar.db`)
- Patient uploads (`./runtime/uploads/`)
- Grad-CAM, lesion, and vessel generated outputs (`./runtime/outputs/`)

**Warning**: On the Free Tier, any uploaded images, generated evidence, and SQLite data will be wiped whenever Render spins down or redeploys the instance. If you need permanence, upgrade your instance and mount a persistent disk (e.g. to `/var/data`) and override the environment variables above.

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
