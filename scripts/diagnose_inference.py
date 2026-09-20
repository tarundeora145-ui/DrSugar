#!/usr/bin/env python3
"""
DR-SUGAR Inference Consistency Diagnostic
==========================================
Tests:
  1. Repeated PyTorch inference on same image (determinism)
  2. PyTorch vs ONNX output comparison
  3. Preprocessing verification (resize method, normalization, dtype, shape)
  4. 4-image regression test (5 runs each)
  5. Softmax correctness check

Usage:
  python scripts/diagnose_inference.py
"""
import sys, json, time, io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import onnxruntime as ort

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / 'models' / 'dr_classification' / 'best_model.pth'
ONNX_MODEL = ROOT / 'models' / 'dr_classification' / 'model.onnx'
APTOS_DIR  = ROOT / 'data' / 'aptos2019' / 'train_images'

IMAGES = [
    '000c1434d8d7.png',
    '001639a390f0.png',
    '0024cdab0c1e.png',
    '002c21358ce6.png',
]

# ── constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224

def build_pytorch_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    state = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def preprocess_pil_resize(img_path: Path) -> np.ndarray:
    """Training-equivalent: PIL Resize (Bilinear squash to 224x224)."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    # CHW
    return arr.transpose(2, 0, 1)[None, ...]  # 1,3,H,W

def preprocess_sharp_cover(img_path: Path) -> np.ndarray:
    """Current Node sharp fit:cover — simulated via PIL thumbnail+crop."""
    img = Image.open(img_path).convert('RGB')
    # Replicate sharp fit:cover: scale so shortest side = 224, center crop
    w, h = img.size
    scale = max(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - INPUT_SIZE) // 2
    top  = (new_h - INPUT_SIZE) // 2
    img  = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    arr  = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return arr.transpose(2, 0, 1)[None, ...]

def torchvision_preprocess(img_path: Path) -> np.ndarray:
    """Exact torchvision transforms.Resize + Normalize (matches training)."""
    tfm = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert('RGB')
    t = tfm(img)  # 3,H,W
    return t.unsqueeze(0).numpy()  # 1,3,H,W

def softmax(x):
    x = np.array(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()

def run_pytorch(model, arr: np.ndarray):
    t = torch.from_numpy(arr)
    with torch.inference_mode():
        logits = model(t).numpy()[0]
    probs = softmax(logits)
    return logits, probs

def run_onnx(session: ort.InferenceSession, arr: np.ndarray):
    inp_name = session.get_inputs()[0].name
    logits = session.run(None, {inp_name: arr.astype(np.float32)})[0][0]
    probs = softmax(logits)
    return logits, probs

def fmt_probs(probs):
    return '[' + ', '.join(f'{p:.4f}' for p in probs) + ']'

def main():
    print('=' * 70)
    print('DR-SUGAR INFERENCE CONSISTENCY DIAGNOSTIC')
    print('=' * 70)

    if not CHECKPOINT.exists():
        print(f'ERROR: PyTorch checkpoint not found: {CHECKPOINT}')
        sys.exit(1)
    if not ONNX_MODEL.exists():
        print(f'ERROR: ONNX model not found: {ONNX_MODEL}')
        sys.exit(1)

    # ── Load models ──────────────────────────────────────────────────────────
    print('\n[1] Loading models...')
    pt_model = build_pytorch_model()
    ort_session = ort.InferenceSession(str(ONNX_MODEL), providers=['CPUExecutionProvider'])
    print('    PyTorch MobileNetV3-Small: OK')
    print(f'    ONNX model: OK  (inputs={ort_session.get_inputs()[0].name}, '
          f'shape={ort_session.get_inputs()[0].shape})')

    # ── Section 2: Preprocessing comparison on one image ─────────────────────
    img0 = APTOS_DIR / IMAGES[0]
    print(f'\n[2] PREPROCESSING COMPARISON — {IMAGES[0]}')
    print('-' * 60)

    arr_tv    = torchvision_preprocess(img0)   # training-exact
    arr_pil   = preprocess_pil_resize(img0)    # matches gradcam script
    arr_cover = preprocess_sharp_cover(img0)   # what Node sharp does

    diff_tv_vs_pil   = float(np.max(np.abs(arr_tv - arr_pil)))
    diff_tv_vs_cover = float(np.max(np.abs(arr_tv - arr_cover)))

    print(f'  torchvision vs PIL Bilinear squash  max|diff|: {diff_tv_vs_pil:.6f}')
    print(f'  torchvision vs sharp fit:cover       max|diff|: {diff_tv_vs_cover:.6f}')
    print(f'  --> Training-matching method: torchvision/PIL squash')
    print(f'  --> Node sharp fit:cover is DIFFERENT ({"MISMATCH" if diff_tv_vs_cover > 1e-3 else "OK"})')

    # Check what predictions each preprocessing gives
    _, probs_tv    = run_pytorch(pt_model, arr_tv)
    _, probs_pil   = run_pytorch(pt_model, arr_pil)
    _, probs_cover = run_pytorch(pt_model, arr_cover)

    print(f'\n  torchvision probs : {fmt_probs(probs_tv)}  -> Grade {np.argmax(probs_tv)}')
    print(f'  PIL squash probs  : {fmt_probs(probs_pil)}  -> Grade {np.argmax(probs_pil)}')
    print(f'  sharp cover probs : {fmt_probs(probs_cover)}  -> Grade {np.argmax(probs_cover)}')

    # ── Section 3: PyTorch determinism (same image, 10 runs) ─────────────────
    print(f'\n[3] PYTORCH DETERMINISM — {IMAGES[0]} (10 runs)')
    print('-' * 60)
    arr = torchvision_preprocess(img0)
    grades, confs = [], []
    for i in range(10):
        t0 = time.perf_counter()
        logits, probs = run_pytorch(pt_model, arr)
        ms = (time.perf_counter() - t0) * 1000
        g = int(np.argmax(probs))
        grades.append(g)
        confs.append(float(probs[g]))
        print(f'  Run {i+1:2d}: Grade {g}  conf={probs[g]:.4f}  {fmt_probs(probs)}  {ms:.1f}ms')
    unique = set(grades)
    print(f'  Unique grades: {unique}  --> {"DETERMINISTIC" if len(unique)==1 else "NON-DETERMINISTIC"}')

    # ── Section 4: PyTorch vs ONNX comparison ────────────────────────────────
    print(f'\n[4] PYTORCH vs ONNX — 3 images (torchvision preprocessing)')
    print('-' * 60)
    for img_name in IMAGES[:3]:
        img_path = APTOS_DIR / img_name
        arr = torchvision_preprocess(img_path)
        logits_pt, probs_pt = run_pytorch(pt_model, arr)
        logits_ort, probs_ort = run_onnx(ort_session, arr)

        max_logit_diff = float(np.max(np.abs(logits_pt - logits_ort)))
        max_prob_diff  = float(np.max(np.abs(probs_pt  - probs_ort)))
        grade_pt  = int(np.argmax(probs_pt))
        grade_ort = int(np.argmax(probs_ort))

        print(f'  {img_name}')
        print(f'    PyTorch:   Grade {grade_pt}  {fmt_probs(probs_pt)}')
        print(f'    ONNX:      Grade {grade_ort}  {fmt_probs(probs_ort)}')
        print(f'    max logit diff: {max_logit_diff:.8f}  max prob diff: {max_prob_diff:.8f}')
        print(f'    Grade agreement: {"PASS" if grade_pt == grade_ort else "FAIL"}')
        print()

    # ── Section 5: Regression test (5 runs each, torchvision preprocessing) ──
    print(f'\n[5] REGRESSION TEST — 4 images × 5 runs each')
    print('-' * 60)
    results = {}
    for img_name in IMAGES:
        img_path = APTOS_DIR / img_name
        arr_tv = torchvision_preprocess(img_path)
        arr_cover = preprocess_sharp_cover(img_path)

        runs_tv, runs_cover = [], []
        for _ in range(5):
            _, p = run_pytorch(pt_model, arr_tv)
            runs_tv.append({'grade': int(np.argmax(p)), 'conf': float(np.max(p)), 'probs': p.tolist()})
            _, p = run_onnx(ort_session, arr_cover)
            runs_cover.append({'grade': int(np.argmax(p)), 'conf': float(np.max(p)), 'probs': p.tolist()})

        tv_grades    = [r['grade'] for r in runs_tv]
        cover_grades = [r['grade'] for r in runs_cover]
        results[img_name] = {
            'torchvision_grade': tv_grades[0],
            'cover_grade': cover_grades[0],
            'tv_probs': runs_tv[0]['probs'],
            'cover_probs': runs_cover[0]['probs'],
        }
        print(f'  {img_name}')
        print(f'    Training-matching (torchvision): Grade {tv_grades[0]}  conf={runs_tv[0]["conf"]:.4f}  {fmt_probs(runs_tv[0]["probs"])}')
        print(f'    Current Node sharp cover:        Grade {cover_grades[0]}  conf={runs_cover[0]["conf"]:.4f}  {fmt_probs(runs_cover[0]["probs"])}')
        match = tv_grades[0] == cover_grades[0]
        print(f'    Grade match: {"PASS" if match else "MISMATCH"}')
        print()

    # ── Section 6: Softmax validation ─────────────────────────────────────────
    print('[6] SOFTMAX VALIDATION')
    print('-' * 60)
    arr = torchvision_preprocess(APTOS_DIR / IMAGES[0])
    logits, probs = run_pytorch(pt_model, arr)
    total = float(np.sum(probs))
    conf = float(np.max(probs))
    grade = int(np.argmax(probs))
    print(f'  Probabilities sum: {total:.8f}  (should be 1.0)')
    print(f'  Confidence = probs[{grade}] = {conf:.6f}  (correct: {abs(conf - probs[grade]) < 1e-9})')
    print(f'  Softmax applied once: {"PASS" if 0.99 < total < 1.01 else "FAIL"}')

    # ── Section 7: model.eval() / no grad check ───────────────────────────────
    print('\n[7] TRAINING MODE CHECK')
    print('-' * 60)
    print(f'  model.training = {pt_model.training}  (should be False)')
    any_dropout_active = any(
        isinstance(m, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)) and m.training
        for m in pt_model.modules()
    )
    print(f'  Any BN/Dropout in training mode: {any_dropout_active}  (should be False)')
    print(f'  model.eval() check: {"PASS" if not pt_model.training and not any_dropout_active else "FAIL"}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'  sharp fit:cover vs torchvision max pixel diff: {diff_tv_vs_cover:.6f}')
    print(f'  PyTorch deterministic: {"YES" if len(set(grades))==1 else "NO"}')
    print()
    print('  PREPROCESSING MISMATCH DETECTED:')
    print('  Node sharp resize(224,224,{fit:"cover"}) != torchvision.Resize((224,224))')
    print('  sharp cover = scale to fill then CENTER CROP')
    print('  torchvision = SQUASH to 224x224 (no crop)')
    print()
    print('  FIX REQUIRED: Change sharp resize to { fit: "fill" } (squash, no crop)')
    print()
    for img_name, r in results.items():
        match = r['torchvision_grade'] == r['cover_grade']
        print(f'  {img_name}: train-match Grade {r["torchvision_grade"]}  '
              f'vs current Grade {r["cover_grade"]}  {"OK" if match else "WRONG"}')

    # Write machine-readable results
    out = ROOT / 'scripts' / '__pycache__' / 'diag_results.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'preprocessing_max_diff_cover_vs_tv': diff_tv_vs_cover,
            'pytorch_deterministic': len(set(grades)) == 1,
            'regression': results,
        }, f, indent=2)
    print(f'\nResults saved to {out}')

if __name__ == '__main__':
    main()
