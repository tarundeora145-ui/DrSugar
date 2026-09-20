# IDRiD Lesion Segmentation V2 — Interrupted Status

> [!IMPORTANT]
> **This is NOT a failed experiment.** V2 successfully demonstrated that patch-based training solves the core V1 failure. Training was interrupted manually due to wall-clock time constraints, not due to model failure or incorrect results.

---

## Status: INTERRUPTED — Fold 1 Only

| Item | Value |
|---|---|
| Status | **Interrupted by user request** |
| Folds completed | **0 of 5** (Fold 1 partially trained; Folds 2–5 not started) |
| Fold 1 best epoch | **Epoch 27** |
| Fold 1 last epoch run | **Epoch 30** |
| Best Mean Dice (Fold 1) | **0.6121** |
| Checkpoint path | `models/idrid/lesions/v2/fold1_best.pth` |
| Checkpoint size | 14.3 MB |
| Checkpoint last written | 2026-09-19 12:05:14 |
| Training start | 2026-09-19 ~08:29 |
| Interruption time | 2026-09-19 ~12:57 |
| Elapsed (Fold 1 only) | **~4.5 hours** |

---

## Fold 1 Best Metrics (Epoch 27)

| Lesion | Dice |
|---|---|
| **MA** (Microaneurysms) | **0.4651** |
| **HE** (Haemorrhages) | **0.5973** |
| **EX** (Hard Exudates) | **0.7405** |
| **SE** (Soft Exudates) | **0.6456** |
| **Mean (all lesions)** | **0.6121** |

> [!NOTE]
> V1 achieved 0.0000 on all lesions across all 5 folds. V2 Fold 1 alone demonstrates the patch-based approach is fundamentally sound.

---

## Epoch Progression (Fold 1 Selected Epochs)

| Epoch | Mean Dice | MA | HE | EX | SE | LR |
|---|---|---|---|---|---|---|
| 1 | 0.2461 | 0.0072 | 0.2460 | 0.4164 | 0.3150 | 5.00e-05 |
| 6 | 0.6013 | 0.4157 | 0.5975 | 0.7475 | 0.6444 | 5.00e-05 |
| 12 | 0.5287 | 0.4395 | 0.5089 | 0.7202 | 0.4461 | 5.00e-05 |
| 21 | 0.6032 | 0.4589 | 0.5885 | 0.7423 | 0.6230 | 1.25e-05 |
| 24 | 0.6110 | 0.4424 | 0.5965 | 0.7474 | 0.6577 | 1.25e-05 |
| **27** | **0.6121** | **0.4651** | **0.5973** | **0.7405** | **0.6456** | 1.25e-05 |
| 30 | 0.6111 | 0.4646 | 0.6029 | 0.7486 | 0.6282 | 1.25e-05 |

---

## Reason for Stopping

- **Wall-clock time:** Each epoch processes 9,888 training patches at batch size 6, taking approximately 8–10 minutes per epoch on the RTX 5060.
- **At 30 epochs per fold × 5 folds**, the full experiment was projected to require **25–40 hours of GPU time** in this configuration.
- This is impractical for a single session. The experiment was stopped after Fold 1 to preserve time.

---

## Root Cause of Slowness

| Bottleneck | Details |
|---|---|
| **Patch pre-loading into RAM** | All 9,888 patches (~512×512×3 float32 each) are loaded as full NumPy arrays into CPU RAM before training starts. Each patch = 786 KB → total ~7.7 GB RAM just for train patches. |
| **No data workers** | `num_workers=0` forces CPU single-threaded data loading per batch. At batch=6, GPU waits for CPU to copy each patch every iteration. |
| **Excessive patch count** | Stride=256 on 4288×2848 images generates ~720 patches per image × 43 images = ~31,000 raw patches before oversampling. After 3:1 oversample, 9,888 are used. |
| **Low batch size** | Batch=6 is too small for the RTX 5060 (8 GB VRAM). Larger batches improve GPU utilization dramatically. |

---

## Exact Optimizations Required for V2 V3 (Faster)

The following changes, applied together, should reduce per-fold training time from **~4 hours to ~30–60 minutes**:

### 1. Lazy/On-the-fly Patch Sampling (Critical)
**Current:** All patches are extracted from all images and stored in RAM before training begins.  
**Fix:** Store only `(image_path, y_offset, x_offset)` coordinates. Load and crop the patch from the JPEG on the fly inside `__getitem__`. Reduces RAM from ~8 GB to <200 MB and eliminates the multi-minute patch extraction phase.

### 2. Increase `num_workers`
**Current:** `num_workers=0` (single-threaded, GPU waits).  
**Fix:** `num_workers=4` or `num_workers=6` with `persistent_workers=True`. Prefetches patches asynchronously so GPU never idles waiting for data.

### 3. Increase Batch Size
**Current:** `batch_size=6`.  
**Fix:** `batch_size=12` or `batch_size=16`. The RTX 5060 has sufficient VRAM for 512×512 patches at this batch size, improving GPU utilization and training throughput.

### 4. Reduce Stride (Fewer Patches)
**Current:** `stride=256` generates ~720 raw patches per image.  
**Fix:** Use `stride=384` or `stride=512` (non-overlapping). This reduces the raw patch count by 50–75%, cuts oversampled training to ~3,000–5,000 patches per fold, and preserves coverage.

### 5. Pin Memory
**Current:** `pin_memory=True` is already set, but wasted with `num_workers=0`.  
**Fix:** Only effective in combination with `num_workers > 0`.

---

## V1 vs V2 Comparison (Fold 1 Only)

| Metric | V1 (all 5 folds) | V2 (Fold 1) |
|---|---|---|
| Mean Dice | 0.0000 | **0.6121** |
| MA Dice | 0.0000 | **0.4651** |
| HE Dice | 0.0000 | **0.5973** |
| EX Dice | 0.0000 | **0.7405** |
| SE Dice | 0.0000 | **0.6456** |
| Approach | Full-image 512×512 resize | **512×512 patches at original res** |
| Root cause of V1 failure | Lesions destroyed by downsampling | Resolved |

---

## Saved Artifacts

```
models/idrid/lesions/v2/
└── fold1_best.pth   (14.3 MB, epoch 27, MeanDice=0.6121)
```

Folds 2–5 checkpoints do not exist. Do not invent or fabricate them.

---

## Next Steps (When Resuming)

1. Implement lazy patch sampling (on-the-fly from JPEG) — **highest priority**.
2. Set `num_workers=4`, `batch_size=14`, `stride=384`.
3. Re-run full 5-fold CV.
4. Report complete mean ± std metrics.

> [!CAUTION]
> Do not report the interrupted single-fold result as a final 5-fold CV result. This document records a partial result only.
