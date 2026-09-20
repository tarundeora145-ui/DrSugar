"""
DR-SUGAR — IDRiD Lesion Segmentation V3
================================================
V2 proof-of-concept: patch-based training works (Fold 1 Mean Dice 0.6121).
V2 problem: ~9,888 patches preloaded into RAM, num_workers=0, batch=6 → ~10 min/epoch.

V3 efficiency fixes:
  1. Lazy patch loading: store (path, y, x) coordinates; load+crop in __getitem__.
  2. stride=384  →  ~3,000-4,500 training patches per fold (vs 9,888).
  3. num_workers=4, persistent_workers=True, prefetch_factor=2  →  GPU stays fed.
  4. batch_size=14  →  better GPU utilisation on RTX 5060.
  5. max_epochs=40  →  V2 Fold 1 peaked at epoch 27; 40 is a safe ceiling.

Architecture / loss unchanged from V2:
  MobileNetV3-Small encoder + U-Net decoder, 4-channel sigmoid output.
  Focal BCE + Dice per channel, averaged.

Research/prototype only. Not clinically validated.
"""

import sys
import json
import time
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(levelname)s:root:%(message)s')

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

# ============================================================
# Config
# ============================================================
PATCH_SIZE  = 512
STRIDE      = 384        # Was 256 in V2 → fewer patches
BATCH_SIZE  = 14         # Increased from V2's 6
NUM_WORKERS = 4
MAX_EPOCHS  = 40         # V2 peaked at ep 27; 40 is safe ceiling
LR          = 5e-5       # Same as validated in V2
WEIGHT_DECAY = 1e-2
ES_PATIENCE  = 10
SCHED_PATIENCE = 5
SCHED_FACTOR   = 0.5
N_FOLDS      = 5
START_FOLD = 5   # Resume from this fold (1-indexed). Folds before this must have checkpoints on disk.

MIN_POS_PIXELS = 10      # pixels across all 4 channels to count as positive patch
POS_NEG_RATIO  = 2       # 2 positives per 1 negative (reduced from V2's 3:1)

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75

BASE_DIR  = Path('data/idrid/A. Segmentation')
IMG_DIR   = BASE_DIR / '1. Original Images' / 'a. Training Set'
GT_BASE   = BASE_DIR / '2. All Segmentation Groundtruths' / 'a. Training Set'
SAVE_DIR  = Path('models/idrid/lesions/v3')
QUAL_DIR  = SAVE_DIR / 'qualitative'

LESIONS = [
    ('1. Microaneurysms', 'MA'),
    ('2. Haemorrhages',   'HE'),
    ('3. Hard Exudates',  'EX'),
    ('4. Soft Exudates',  'SE'),
]
N_CLASSES = len(LESIONS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# Patch Coordinate Building (lazy — no pixel data in RAM)
# ============================================================

def build_patch_coords(img_paths, oversample=True):
    """
    Returns list of (img_path, y, x) coordinate tuples.
    Positive patches (any lesion pixel >= MIN_POS_PIXELS) are oversampled POS_NEG_RATIO:1.
    No pixel data is stored — images are loaded on demand in __getitem__.
    """
    pos_coords = []
    neg_coords = []

    for img_path in img_paths:
        img_path = Path(img_path)
        img_id   = img_path.stem

        # Load masks at full res to determine patch positivity
        masks = []
        for l_dir, l_abbr in LESIONS:
            mpath = GT_BASE / l_dir / f'{img_id}_{l_abbr}.tif'
            if mpath.exists():
                m = np.array(Image.open(mpath))
                masks.append((m[:, :, 0] if m.ndim == 3 else m) > 0)
            else:
                img_size = Image.open(img_path).size   # (W, H)
                masks.append(np.zeros((img_size[1], img_size[0]), dtype=bool))

        # Stack to get union mask for positivity check
        union = np.stack(masks, axis=0).any(axis=0)   # H, W bool
        H, W  = union.shape

        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                patch_union = union[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                if patch_union.sum() >= MIN_POS_PIXELS:
                    pos_coords.append((img_path, y, x))
                else:
                    neg_coords.append((img_path, y, x))

    if not oversample:
        return pos_coords + neg_coords

    n_neg = len(neg_coords)
    n_pos_target = n_neg * POS_NEG_RATIO
    if 0 < len(pos_coords) < n_pos_target:
        rng = np.random.default_rng(SEED)
        extra = rng.choice(len(pos_coords), size=n_pos_target - len(pos_coords), replace=True)
        combined = pos_coords + [pos_coords[i] for i in extra] + neg_coords
    else:
        combined = pos_coords + neg_coords

    logging.info(f"  Coords: {len(pos_coords)} pos (os to {len(combined)-n_neg}), "
                 f"{n_neg} neg, total={len(combined)}")
    return combined


# ============================================================
# Dataset — lazy: load patch from JPEG in __getitem__
# ============================================================

class LazyPatchDataset(Dataset):
    def __init__(self, coords, augment=False):
        self.coords  = coords   # list of (Path, y, x)
        self.augment = augment

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        img_path, y0, x0 = self.coords[idx]
        img_id = Path(img_path).stem

        # Load full-res image and crop patch
        with Image.open(img_path) as f:
            img_full = np.array(f.convert('RGB'))
        img_p = img_full[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE].copy()

        # Load masks and crop
        masks_p = np.zeros((N_CLASSES, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for i, (l_dir, l_abbr) in enumerate(LESIONS):
            mpath = GT_BASE / l_dir / f'{img_id}_{l_abbr}.tif'
            if mpath.exists():
                with Image.open(mpath) as f:
                    m = np.array(f)
                if m.ndim == 3:
                    m = m[:, :, 0]
                masks_p[i] = (m[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] > 0).astype(np.float32)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img_p    = img_p[:, ::-1, :].copy()
                masks_p  = masks_p[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                img_p    = img_p[::-1, :, :].copy()
                masks_p  = masks_p[:, ::-1, :].copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img_p   = np.rot90(img_p,   k=k, axes=(0, 1)).copy()
                masks_p = np.rot90(masks_p, k=k, axes=(1, 2)).copy()
            jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
            img_p = np.array(jitter(Image.fromarray(img_p)))

        # Normalize
        img_f = img_p.astype(np.float32) / 255.0
        for c in range(3):
            img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        img_t   = torch.from_numpy(img_f.transpose(2, 0, 1))
        masks_t = torch.from_numpy(masks_p)
        return img_t, masks_t


# ============================================================
# Model
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MobileNetV3UNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        f = models.mobilenet_v3_small(weights=w).features
        self.enc0 = f[0]
        self.enc1 = f[1]
        self.enc2 = nn.Sequential(*f[2:4])
        self.enc3 = nn.Sequential(*f[4:7])
        self.enc4 = nn.Sequential(*f[7:13])
        self.dec4 = DecoderBlock(576, 40, 256)
        self.dec3 = DecoderBlock(256, 24, 128)
        self.dec2 = DecoderBlock(128, 16, 64)
        self.dec1 = DecoderBlock(64,  16, 32)
        self.up_final   = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.final_conv = nn.Sequential(DoubleConv(16, 16), nn.Conv2d(16, n_classes, 1))

    def forward(self, x):
        s0 = self.enc0(x); s1 = self.enc1(s0)
        s2 = self.enc2(s1); s3 = self.enc3(s2); b = self.enc4(s3)
        d = self.dec4(b, s3); d = self.dec3(d, s2)
        d = self.dec2(d, s1); d = self.dec1(d, s0)
        return self.final_conv(self.up_final(d))


# ============================================================
# Loss
# ============================================================

def focal_bce(logits, targets, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t    = torch.exp(-bce)
    a_t    = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def dice_loss(logits, targets, smooth=1.0):
    p  = torch.sigmoid(logits)
    pf = p.view(p.size(0), p.size(1), -1)
    tf = targets.view(targets.size(0), targets.size(1), -1)
    i  = (pf * tf).sum(2)
    return 1.0 - ((2*i + smooth) / (pf.sum(2) + tf.sum(2) + smooth)).mean()


def combined_loss(logits, targets):
    return focal_bce(logits, targets) + dice_loss(logits, targets)


# ============================================================
# Metrics
# ============================================================

def channel_metrics(probs, targets, thr=0.5):
    preds = (probs >= thr).astype(np.float32)
    TP = ((preds==1)&(targets==1)).sum()
    TN = ((preds==0)&(targets==0)).sum()
    FP = ((preds==1)&(targets==0)).sum()
    FN = ((preds==0)&(targets==1)).sum()
    n_pos = targets.sum()
    if n_pos == 0:
        return (1.0 if FP==0 else 0.0,
                1.0 if FP==0 else 0.0,
                float('nan'), float((TN+1e-7)/(TN+FP+1e-7)), float('nan'))
    dice = float((2*TP+1e-7)/(2*TP+FP+FN+1e-7))
    iou  = float((TP+1e-7)/(TP+FP+FN+1e-7))
    sens = float((TP+1e-7)/(TP+FN+1e-7))
    spec = float((TN+1e-7)/(TN+FP+1e-7))
    prec = float((TP+1e-7)/(TP+FP+1e-7))
    return dice, iou, sens, spec, prec


def evaluate_fold(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_probs   = [[] for _ in range(N_CLASSES)]
    all_targets = [[] for _ in range(N_CLASSES)]
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = combined_loss(logits, masks)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            tgts  = masks.cpu().numpy()
            for c in range(N_CLASSES):
                all_probs[c].append(probs[:, c].ravel())
                all_targets[c].append(tgts[:, c].ravel())
    avg_loss = total_loss / max(len(loader), 1)
    metrics  = {'loss': avg_loss}
    mean_d   = 0.0; valid_n = 0
    for i, (_, abbr) in enumerate(LESIONS):
        p = np.concatenate(all_probs[i])
        t = np.concatenate(all_targets[i])
        d, iou, sens, spec, prec = channel_metrics(p, t)
        metrics[f'{abbr}_dice']        = d
        metrics[f'{abbr}_iou']         = iou
        metrics[f'{abbr}_sensitivity'] = sens
        metrics[f'{abbr}_specificity'] = spec
        metrics[f'{abbr}_precision']   = prec
        if not np.isnan(d):
            mean_d += d; valid_n += 1
    metrics['mean_dice'] = mean_d / max(valid_n, 1)
    return metrics


# ============================================================
# Qualitative
# ============================================================

def save_qualitative(model, val_paths, device, fold_idx, n=2):
    model.eval()
    QUAL_DIR.mkdir(parents=True, exist_ok=True)
    for img_path in list(val_paths)[:n]:
        img_id   = Path(img_path).stem
        img_full = np.array(Image.open(img_path).convert('RGB'))
        H, W     = img_full.shape[:2]
        masks_full = np.zeros((N_CLASSES, H, W), dtype=np.float32)
        for i, (l_dir, l_abbr) in enumerate(LESIONS):
            mpath = GT_BASE / l_dir / f'{img_id}_{l_abbr}.tif'
            if mpath.exists():
                m = np.array(Image.open(mpath))
                if m.ndim == 3: m = m[:,:,0]
                masks_full[i] = (m > 0).astype(np.float32)
        # Find best MA patch
        best, bsc = (0, 0), -1
        for y in range(0, H-PATCH_SIZE+1, STRIDE):
            for x in range(0, W-PATCH_SIZE+1, STRIDE):
                sc = masks_full[0, y:y+PATCH_SIZE, x:x+PATCH_SIZE].sum()*10 + masks_full[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE].sum()
                if sc > bsc: bsc = sc; best = (y, x)
        y0, x0 = best
        ip = img_full[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        mp = masks_full[:, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        img_f = ip.astype(np.float32) / 255.0
        for c in range(3):
            img_f[:,:,c] = (img_f[:,:,c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        with torch.no_grad():
            logit = model(torch.from_numpy(img_f.transpose(2,0,1)).unsqueeze(0).to(device))
        preds = (torch.sigmoid(logit).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)*255
        orig_pil = Image.fromarray(ip)
        pw = PATCH_SIZE
        grid = Image.new('RGB', (pw*3, pw*N_CLASSES))
        for i, (_, abbr) in enumerate(LESIONS):
            grid.paste(orig_pil, (0, pw*i))
            grid.paste(Image.fromarray((mp[i]*255).astype(np.uint8)).convert('RGB'), (pw, pw*i))
            grid.paste(Image.fromarray(preds[i]).convert('RGB'), (pw*2, pw*i))
        out = QUAL_DIR / f'fold{fold_idx+1}_{img_id}.png'
        grid.save(out)
        logging.info(f"  Saved qualitative: {out}")


# ============================================================
# Train one fold
# ============================================================

def train_fold(fold_idx, train_paths, val_paths, device, nw):
    logging.info(f"\n{'='*60}")
    logging.info(f"FOLD {fold_idx+1}/{N_FOLDS}  train={len(train_paths)} val={len(val_paths)}")
    logging.info(f"{'='*60}")

    t_coord = time.time()
    train_coords = build_patch_coords(train_paths, oversample=True)
    val_coords   = build_patch_coords(val_paths,   oversample=False)
    logging.info(f"  Coord build: {time.time()-t_coord:.1f}s | train={len(train_coords)} val={len(val_coords)}")

    train_ds = LazyPatchDataset(train_coords, augment=True)
    val_ds   = LazyPatchDataset(val_coords,   augment=False)

    ldr_kw = dict(num_workers=nw, pin_memory=True, drop_last=True)
    if nw > 0:
        ldr_kw['persistent_workers'] = True
        ldr_kw['prefetch_factor']    = 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **ldr_kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **{k: v for k,v in ldr_kw.items() if k != 'drop_last'})

    model     = MobileNetV3UNet(N_CLASSES, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
    scaler    = torch.amp.GradScaler('cuda')

    best_dice     = -1.0
    no_improve    = 0
    history       = []
    ckpt          = SAVE_DIR / f'fold{fold_idx+1}_best.pth'
    t_start       = time.time()

    steps_ep = len(train_loader)
    logging.info(f"  Steps/epoch: {steps_ep} | batches of {BATCH_SIZE}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = combined_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        tl = running / steps_ep
        vm = evaluate_fold(model, val_loader, device)
        vd = vm['mean_dice']
        scheduler.step(vd)

        logging.info(
            f"  Ep {epoch+1:02d} | TrLoss={tl:.4f} | "
            f"Dice={vd:.4f} MA={vm['MA_dice']:.4f} HE={vm['HE_dice']:.4f} "
            f"EX={vm['EX_dice']:.4f} SE={vm['SE_dice']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )
        history.append({'epoch': epoch+1, 'train_loss': tl, **vm})

        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), ckpt)
            logging.info(f"  -> Checkpoint saved (dice={best_dice:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                logging.info(f"  Early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - t_start
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    final = evaluate_fold(model, val_loader, device)
    save_qualitative(model, val_paths, device, fold_idx, n=2)
    logging.info(f"  Fold {fold_idx+1} done in {elapsed/60:.1f} min | Best dice {best_dice:.4f}")
    return final, history, elapsed


# ============================================================
# Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    device    = torch.device('cuda')
    gpu_name  = torch.cuda.get_device_name(0)
    vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    QUAL_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(list(IMG_DIR.glob('*.jpg')))
    if len(all_paths) != 54:
        raise ValueError(f"Expected 54 training images, got {len(all_paths)}")

    rng = np.random.default_rng(SEED)
    shuffled = all_paths.copy()
    rng.shuffle(shuffled)
    folds = [list(shuffled[i::N_FOLDS]) for i in range(N_FOLDS)]

    # Test num_workers — fall back if multiprocessing fails on Windows
    nw = NUM_WORKERS
    try:
        _td = LazyPatchDataset([(all_paths[0], 0, 0)], augment=False)
        _dl = DataLoader(_td, batch_size=1, num_workers=nw, prefetch_factor=2, persistent_workers=True)
        _ = next(iter(_dl))
        del _td, _dl
    except Exception as e:
        logging.warning(f"num_workers={nw} failed ({e}), falling back to 2")
        nw = 2

    print("\n" + "="*50)
    print(f"PATCHES/FOLD:  ~{(len(all_paths)//N_FOLDS * 4 * POS_NEG_RATIO + len(all_paths)//N_FOLDS * 2):.0f} est.")
    print(f"BATCH:         {BATCH_SIZE}")
    print(f"WORKERS:       {nw}")
    print(f"STRIDE:        {STRIDE}")
    print(f"EPOCH LIMIT:   {MAX_EPOCHS}")
    print(f"GPU:           {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print("="*50 + "\n")

    logging.info("="*60)
    logging.info("IDRiD LESION SEGMENTATION V3 -- EFFICIENT PATCH TRAINING")
    logging.info("="*60)
    logging.info(f"  GPU:      {gpu_name}  ({vram_gb:.1f} GB)")
    logging.info(f"  Batch:    {BATCH_SIZE} | Workers: {nw} | Stride: {STRIDE}")
    logging.info(f"  MaxEpoch: {MAX_EPOCHS} | ES patience: {ES_PATIENCE}")
    logging.info(f"  Loss:     FocalBCE(g={FOCAL_GAMMA},a={FOCAL_ALPHA}) + Dice")
    for i, f in enumerate(folds):
        logging.info(f"  Fold {i+1} val: {[p.stem for p in f]}")
    logging.info("="*60)

    all_metrics  = []
    all_histories = []
    total_start  = time.time()

    for fold_idx in range(N_FOLDS):
        val_paths   = folds[fold_idx]
        seen = set(); train_paths = []
        for fi, fl in enumerate(folds):
            if fi == fold_idx: continue
            for p in fl:
                if p not in seen: seen.add(p); train_paths.append(p)

        ckpt_path = SAVE_DIR / f'fold{fold_idx+1}_best.pth'

        if fold_idx + 1 < START_FOLD:
            # Fold already completed — re-evaluate saved checkpoint for metrics
            logging.info(f"\nFOLD {fold_idx+1} already complete — re-evaluating checkpoint {ckpt_path}")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Expected checkpoint {ckpt_path} but it is missing.")
            # Build val loader (no training)
            val_coords = build_patch_coords(val_paths, oversample=False)
            val_ds     = LazyPatchDataset(val_coords, augment=False)
            ldr_kw = dict(num_workers=nw, pin_memory=True)
            if nw > 0:
                ldr_kw['persistent_workers'] = True
                ldr_kw['prefetch_factor']    = 2
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **ldr_kw)
            model = MobileNetV3UNet(N_CLASSES, pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            metrics = evaluate_fold(model, val_loader, device)
            metrics['fold']            = fold_idx + 1
            metrics['training_time_s'] = 0.0  # already counted
            all_metrics.append(metrics)
            logging.info(f"  Re-eval MeanDice={metrics['mean_dice']:.4f} "
                         f"MA={metrics['MA_dice']:.4f} HE={metrics['HE_dice']:.4f} "
                         f"EX={metrics['EX_dice']:.4f} SE={metrics['SE_dice']:.4f}")
            continue

        metrics, history, elapsed = train_fold(fold_idx, train_paths, val_paths, device, nw)
        metrics['fold']           = fold_idx + 1
        metrics['training_time_s'] = elapsed
        all_metrics.append(metrics)
        all_histories.extend([{**h, 'fold': fold_idx+1} for h in history])

    total_elapsed = time.time() - total_start

    logging.info("\n" + "="*60)
    logging.info("CROSS-VALIDATION SUMMARY (Mean +/- Std)")
    logging.info("="*60)
    summary = {}
    for abbr in ['MA','HE','EX','SE']:
        for met in ['dice','iou','sensitivity','specificity','precision']:
            key  = f'{abbr}_{met}'
            vals = [m[key] for m in all_metrics if not np.isnan(m.get(key, float('nan')))]
            mn   = float(np.mean(vals)) if vals else float('nan')
            sd   = float(np.std(vals))  if vals else float('nan')
            summary[key] = {'mean': mn, 'std': sd, 'per_fold': vals}
            logging.info(f"  {key:>22}: {mn:.4f} +/- {sd:.4f}")
    md_vals = [m['mean_dice'] for m in all_metrics]
    summary['mean_dice'] = {'mean': float(np.mean(md_vals)), 'std': float(np.std(md_vals))}
    logging.info(f"  {'mean_dice':>22}: {summary['mean_dice']['mean']:.4f} +/- {summary['mean_dice']['std']:.4f}")

    pd.DataFrame(all_metrics).to_csv(SAVE_DIR / 'cv_results.csv', index=False)
    pd.DataFrame(all_histories).to_csv(SAVE_DIR / 'training_history.csv', index=False)

    meta = {
        "model_name":  "DR-SUGAR-IDRiD-Lesions-V3",
        "version":     "3.0.0",
        "architecture": "MobileNetV3-Small + U-Net decoder (4 channels)",
        "patch_size":  PATCH_SIZE, "stride": STRIDE,
        "batch_size":  BATCH_SIZE, "num_workers": nw,
        "max_epochs":  MAX_EPOCHS, "es_patience": ES_PATIENCE,
        "loss": f"FocalBCE(g={FOCAL_GAMMA},a={FOCAL_ALPHA})+Dice",
        "optimizer":   f"AdamW(lr={LR},wd={WEIGHT_DECAY})",
        "scheduler":   f"ReduceLROnPlateau(p={SCHED_PATIENCE},f={SCHED_FACTOR})",
        "seed": SEED, "amp": True, "gpu": gpu_name,
        "total_training_time_s": total_elapsed,
        "cv_summary": summary, "per_fold_metrics": all_metrics,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "disclaimer":  "Research/prototype only. Not clinically validated.",
    }
    with open(SAVE_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    logging.info(f"\nAll artifacts in {SAVE_DIR}/")
    logging.info(f"Total time: {total_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
