# DR-SUGAR ML Environment Audit

## Runtime Availability
**VERIFIED**: Node.js v24.15.0 and npm 11.12.1 are installed and available.

## Python
**VERIFIED**: Python 3.14.3 and pip 25.3 are installed and available.

## PyTorch
**UNAVAILABLE**: PyTorch is not currently installed in the Python environment.

## CUDA/GPU
**UNKNOWN**: Cannot verify CUDA status via PyTorch as PyTorch is unavailable. (Requires installation of `torch` to check `torch.cuda.is_available()`).

## MATLAB
**UNAVAILABLE**: The `matlab` command is not recognized in the system path.

## MATLAB Toolboxes
**UNAVAILABLE**: Cannot verify toolboxes (Deep Learning, Image Processing, Computer Vision, Medical Imaging, Statistics/Machine Learning) since MATLAB is inaccessible.

## Simulink
**UNAVAILABLE**: Cannot verify Simulink presence.

## Existing Model Files
**UNAVAILABLE**: No `.onnx`, `.pt`, `.pth`, or `.mat` model weight files exist in the repository.

## Existing ML Code
**PARTIAL / STRUCTURAL STUB**:
- **MATLAB scripts** (`matlab/**/*.m`) exist as structural stubs for preprocessing, quality assessment, segmentation, lesion detection, and DR grading, but they contain no active deep learning execution logic and cannot be run.
- **Node Backend** (`server/src/routes/models.ts`, `screening.ts`) contains endpoints that intentionally return `MODEL UNAVAILABLE`, correctly honoring the no-fake-metrics rule.
- **Explainability**: UI components (`ExplainabilityPanel.tsx`) exist but backend explainability endpoints return `NOT EVALUATED`.

## Recommended Runtime
**Python/PyTorch OR ONNX Runtime**
*Reasoning*: MATLAB is completely unavailable on this machine. Node.js is present, and Python is present. Given the absence of MATLAB, Option A (MATLAB-based inference) is impossible. Option B (Python/PyTorch) or Option C (ONNX Runtime via Node.js or Python) are the only viable paths. ONNX Runtime in Node.js would remove the need for a secondary Python server, making it highly recommended for production deployment, provided we can train or acquire real ONNX weights.

## Blockers
- MATLAB is absent, invalidating the current `matlab/` directory architectural stubs.
- No model weights exist.
- Python lacks `torch` and `onnxruntime`.
