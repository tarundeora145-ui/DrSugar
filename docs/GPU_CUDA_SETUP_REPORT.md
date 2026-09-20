# DR-SUGAR — GPU/CUDA Environment Setup Report

## Hardware Verification
- **GPU Name:** NVIDIA GeForce RTX 5060 Laptop GPU
- **Architecture:** Blackwell (Compute Capability 12.0 / `sm_120`)
- **NVIDIA Driver Version:** 610.62
- **Supported CUDA UMD:** 13.3

## Python & Environment Verification
- **Python Version:** 3.14.3
- **Initial PyTorch Version:** `2.14.0+cpu` (CPU-only build)
- **Initial CUDA Availability:** `False`

## The Problem
The initially installed PyTorch was a `+cpu` build. Furthermore, standard PyTorch 2.14 wheels built against CUDA 12.6 (`+cu126`) do *not* support the new Blackwell `sm_120` architecture found in the RTX 5060. PyTorch `__init__.py` actively warned that `sm_120` is not compatible with `cu126` and explicitly recommended upgrading to `cu130` or `cu132`.

## The Solution
We located and installed the `cu132` wheels for Windows Python 3.14, which correctly bundle the CUDA 13.2 runtime that supports `sm_120`.

### Commands Executed
```powershell
pip install "torch==2.14.0+cu132" "torchvision==0.29.0+cu132" --index-url https://download.pytorch.org/whl/cu132 --no-deps
```

## Post-Installation Verification
- **PyTorch Version:** `2.14.0+cu132`
- **CUDA Build:** `13.2`
- **CUDA Availability:** `True`
- **GPU Detected:** `NVIDIA GeForce RTX 5060 Laptop GPU`

### GPU Smoke Test
A test performing matrix multiplication on `1000x1000` random tensors explicitly assigned to `device='cuda'` completed successfully.
- **Result:** SUCCESS
- **GPU Memory Used (PyTorch allocator):** 43.44 MB

## Training Script Compatibility
Both `scripts/train_idrid_grading.py` and `scripts/train_idrid_grading_v2.py` were inspected.
Both scripts correctly implement standard PyTorch device-agnostic logic:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```
No modifications to the ML source code were required to enable CUDA acceleration. The environment is now fully GPU-ready for future ML training phases.
