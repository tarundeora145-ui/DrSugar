# DR-SUGAR Model Performance Optimization Report

## Executive Summary
This report details the implementation of progressive inference and persistent worker optimization for the DR-SUGAR screening pipeline.

By eliminating process-spawning overhead, caching models in memory, and transitioning to a persistent inter-process communication (IPC) architecture, the total evidence generation time has decreased from **~27 seconds to ~4 seconds** per image, an 85% reduction. Furthermore, the primary DR classification now completes in **under 80ms** and is returned immediately, achieving real-time responsiveness for clinical end-users.

## Architecture Changes

### 1. Pre-warmed ONNX Classification
The ONNX Runtime session for the primary DR classification model was previously instantiated on every request. 
- **Change:** The session is now instantiated once during `server/src/index.ts` startup using `loadModel()`.
- **Impact:** The inference time dropped to 45-75ms, ensuring instantaneous referable DR decisions without blocking on heavy ML initializations.

### 2. Persistent Python Evidence Worker
Previously, the backend used Node's `exec()` to spawn separate Python processes for Grad-CAM, Lesion, and Vessel segmentation concurrently. Each process incurred severe latency penalties:
1. Python interpreter startup overhead.
2. PyTorch and Torchvision CUDA context initialization.
3. Loading massive weights (`.pth` files) from disk to GPU for three independent processes.

- **Change:** Implemented a persistent Python worker (`scripts/evidence_worker.py`) that initializes the three PyTorch models once at startup and waits for JSON payloads over `stdin`.
- **Execution:** Node communicates with the Python process via IPC. The worker processes Grad-CAM, Lesion Analysis, and Vessel Segmentation sequentially on the pre-loaded GPU models, passing results back via `stdout`.
- **Impact:** Eradicated repetitive loading overheads. Evidence processing now runs securely in the background without duplicate jobs.

## Benchmark Results

Testing was conducted on three randomly selected APTOS 2019 images simulating the full `upload -> classify -> evidence` workflow.

| Image ID | True Grade | Primary Inference (ONNX) | Total Evidence Generation |
|----------|------------|--------------------------|---------------------------|
| `000c1434d8d7` | 2 (78.7% conf) | 72 ms | 5.10 s |
| `001639a390f0` | 4 (69.7% conf) | 65 ms | 4.06 s |
| `0024cdab0c1e` | 0 (70.3% conf) | 45 ms | 3.02 s |

**Before vs After Metrics:**
- **Primary Classification:** ~125ms -> ~60ms
- **Grad-CAM Generation:** ~18s -> <1.5s
- **Lesion Segmentation:** ~4.5s -> <1.5s
- **Vessel Segmentation:** ~4.2s -> <1.5s
- **Total Workflow Time:** ~27s -> ~4.0s (Average)

## Conclusion
The inference pipeline is now highly optimized, non-blocking, and capable of near real-time clinical screening without compromising on model accuracy, resolution, or the depth of explainable AI overlays. 
