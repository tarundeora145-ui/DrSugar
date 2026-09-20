"""
DR-SUGAR — IDRiD Lesion Segmentation V2
================================================
V1 Failure Root Cause:
  Resizing 4288x2848 -> 512x512 destroyed tiny lesions (esp. MA).
  MA pixels: ~0.057% of full image = ~4px after downscale. Invisible.

V2 Fix: HIGH-RESOLUTION PATCH-BASED TRAINING
  - Extract 512x512 patches at the ORIGINAL resolution.
  - Oversample patches containing lesion pixels (esp. MA).
  - No data leakage: images are split first, then patches extracted.

Architecture: MobileNetV3-Small encoder + U-Net decoder
Output: 4 independent binary channels (MA, HE, EX, SE) via Sigmoid
Loss: Focal BCE + Dice per channel, then averaged across channels
Optimizer: AdamW with CosineAnnealingLR
AMP: enabled
Seed: 42

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
PATCH_SIZE   = 512
STRIDE       = 256           # Overlapping patches
BATCH_SIZE   = 6
MAX_EPOCHS   = 80
LR           = 5e-5
WEIGHT_DECAY = 1e-2
EARLY_STOP_PATIENCE = 15
SCHEDULER_PATIENCE  = 6
SCHEDULER_FACTOR    = 0.5
N_FOLDS      = 5

# Positive patch oversampling: must have >= MIN_POS_PIXELS to be a positive patch
MIN_POS_PIXELS = 10          # At least 10 lesion pixels across all 4 channels
# Oversampling ratio: include N_POS_OVERSAMPLE positive patches for each 1 negative
N_POS_OVERSAMPLE = 3         # 3:1 positive:negative ratio

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75           # Positive class weight for Focal loss

BASE_DIR    = Path('data/idrid/A. Segmentation')
TRAIN_IMG   = BASE_DIR / '1. Original Images' / 'a. Training Set'
GT_BASE     = BASE_DIR / '2. All Segmentation Groundtruths' / 'a. Training Set'
SAVE_DIR    = Path('models/idrid/lesions/v2')
QUAL_DIR    = SAVE_DIR / 'qualitative'

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
# Patch Extraction
# ============================================================

def load_image_and_masks(img_path: Path):
    """Load full-res image (H,W,3 uint8) and 4-channel mask (4,H,W float32)."""
    img = np.array(Image.open(img_path).convert('RGB'))  # H,W,3
    H, W = img.shape[:2]
    masks = np.zeros((N_CLASSES, H, W), dtype=np.float32)
    img_id = img_path.stem
    for i, (l_dir, l_abbr) in enumerate(LESIONS):
        mask_path = GT_BASE / l_dir / f'{img_id}_{l_abbr}.tif'
        if mask_path.exists():
            m = np.array(Image.open(mask_path))
            if m.ndim == 3:
                m = m[:, :, 0]
            masks[i] = (m > 0).astype(np.float32)
    return img, masks


def extract_patches(img, masks, stride=STRIDE, patch_size=PATCH_SIZE):
    """
    Extract all (patch_size x patch_size) patches with given stride.
    Returns list of (img_patch, mask_patch, has_positive).
    """
    H, W = img.shape[:2]
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            ip = img[y:y+patch_size, x:x+patch_size]        # H,W,3
            mp = masks[:, y:y+patch_size, x:x+patch_size]   # 4,H,W
            has_pos = mp.sum() >= MIN_POS_PIXELS
            patches.append((ip, mp, bool(has_pos)))
    return patches


def build_patch_list(img_paths, oversample=True):
    """
    Build a list of (img_array, mask_array) patches for a set of images.
    Oversample positive patches N_POS_OVERSAMPLE:1 against negatives.
    """
    pos_patches = []
    neg_patches = []
    for img_path in img_paths:
        img, masks = load_image_and_masks(img_path)
        for ip, mp, has_pos in extract_patches(img, masks):
            if has_pos:
                pos_patches.append((ip, mp))
            else:
                neg_patches.append((ip, mp))

    if not oversample:
        return pos_patches + neg_patches

    # Oversample: target N_POS_OVERSAMPLE positive for every 1 negative
    n_neg = len(neg_patches)
    n_pos_target = n_neg * N_POS_OVERSAMPLE
    if len(pos_patches) < n_pos_target and len(pos_patches) > 0:
        rng = np.random.default_rng(SEED)
        extra_idx = rng.choice(len(pos_patches), size=n_pos_target - len(pos_patches), replace=True)
        oversampled = [pos_patches[i] for i in extra_idx]
        combined = pos_patches + oversampled + neg_patches
    else:
        combined = pos_patches + neg_patches

    logging.info(f"  Patch pool: {len(pos_patches)} pos (oversampled to {len(combined)-n_neg}), {n_neg} neg, total={len(combined)}")
    return combined


# ============================================================
# Dataset
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, patches, augment=False):
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_np, mask_np = self.patches[idx]
        img_np = img_np.copy()
        mask_np = mask_np.copy()

        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img_np  = img_np[:, ::-1, :].copy()
                mask_np = mask_np[:, :, ::-1].copy()
            # Vertical flip
            if np.random.rand() > 0.5:
                img_np  = img_np[::-1, :, :].copy()
                mask_np = mask_np[:, ::-1, :].copy()
            # Random 90-degree rotation (low distortion for lesion preservation)
            k = np.random.randint(0, 4)
            if k > 0:
                img_np  = np.rot90(img_np,  k=k, axes=(0, 1)).copy()
                mask_np = np.rot90(mask_np, k=k, axes=(1, 2)).copy()
            # Mild brightness/contrast
            jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
            img_np = np.array(jitter(Image.fromarray(img_np)))

        # Normalize
        img_f = img_np.astype(np.float32) / 255.0
        for c in range(3):
            img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        img_t  = torch.from_numpy(img_f.transpose(2, 0, 1))  # C,H,W
        mask_t = torch.from_numpy(mask_np)                    # 4,H,W
        return img_t, mask_t


# ============================================================
# Model: MobileNetV3-Small U-Net
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MobileNetV3UNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        feats = models.mobilenet_v3_small(weights=weights).features

        self.enc0 = feats[0]                     # 16ch, /2
        self.enc1 = feats[1]                     # 16ch, /2
        self.enc2 = nn.Sequential(*feats[2:4])   # 24ch, /2
        self.enc3 = nn.Sequential(*feats[4:7])   # 40ch, same
        self.enc4 = nn.Sequential(*feats[7:13])  # 576ch, /2

        self.dec4 = DecoderBlock(576, 40, 256)
        self.dec3 = DecoderBlock(256, 24, 128)
        self.dec2 = DecoderBlock(128, 16, 64)
        self.dec1 = DecoderBlock(64,  16, 32)

        self.up_final  = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(DoubleConv(16, 16), nn.Conv2d(16, n_classes, 1))

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b  = self.enc4(s3)
        d4 = self.dec4(b,  s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        out = self.up_final(d1)
        return self.final_conv(out)  # B, n_classes, H, W


# ============================================================
# Focal BCE + Dice Loss
# ============================================================

def focal_bce_loss(logits, targets, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """Focal binary cross-entropy. Shape: B,C,H,W -> scalar."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    pf = probs.view(probs.size(0), probs.size(1), -1)
    tf = targets.view(targets.size(0), targets.size(1), -1)
    inter = (pf * tf).sum(dim=2)
    return 1.0 - ((2 * inter + smooth) / (pf.sum(dim=2) + tf.sum(dim=2) + smooth)).mean()


def combined_loss(logits, targets):
    return focal_bce_loss(logits, targets) + dice_loss(logits, targets)


# ============================================================
# Metrics
# ============================================================

def compute_channel_metrics(probs, targets, threshold=0.5):
    preds = (probs >= threshold).astype(np.float32)
    TP = ((preds == 1) & (targets == 1)).sum()
    TN = ((preds == 0) & (targets == 0)).sum()
    FP = ((preds == 1) & (targets == 0)).sum()
    FN = ((preds == 0) & (targets == 1)).sum()

    n_pos_gt = targets.sum()
    if n_pos_gt == 0:
        # Absent lesion: Dice = 1.0 if no FP, else 0.0
        dice = 1.0 if FP == 0 else 0.0
        iou  = 1.0 if FP == 0 else 0.0
        sens = float('nan')   # Undefined — no positives to recall
        prec = float('nan')   # Undefined
    else:
        dice = float((2 * TP + 1e-7) / (2 * TP + FP + FN + 1e-7))
        iou  = float((TP + 1e-7) / (TP + FP + FN + 1e-7))
        sens = float((TP + 1e-7) / (TP + FN + 1e-7))
        prec = float((TP + 1e-7) / (TP + FP + 1e-7))

    spec = float((TN + 1e-7) / (TN + FP + 1e-7))
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
            probs = torch.sigmoid(logits).cpu().numpy()   # B,4,H,W
            tgts  = masks.cpu().numpy()
            for c in range(N_CLASSES):
                all_probs[c].append(probs[:, c].ravel())
                all_targets[c].append(tgts[:, c].ravel())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = {'loss': avg_loss}
    mean_dice = 0.0
    valid_dice_count = 0

    for i, (_, abbr) in enumerate(LESIONS):
        p = np.concatenate(all_probs[i])
        t = np.concatenate(all_targets[i])
        dice, iou, sens, spec, prec = compute_channel_metrics(p, t)
        metrics[f'{abbr}_dice']        = dice
        metrics[f'{abbr}_iou']         = iou
        metrics[f'{abbr}_sensitivity'] = sens
        metrics[f'{abbr}_specificity'] = spec
        metrics[f'{abbr}_precision']   = prec
        if not np.isnan(dice):
            mean_dice += dice
            valid_dice_count += 1

    metrics['mean_dice'] = mean_dice / max(valid_dice_count, 1)
    return metrics


# ============================================================
# Qualitative
# ============================================================

def save_qualitative(model, img_paths, device, save_dir, fold_idx, n=2):
    """
    For up to n validation images, save a grid:
    Row 1: Original patch | GT MA  | Pred MA
    Row 2: Original patch | GT HE  | Pred HE
    Row 3: Original patch | GT EX  | Pred EX
    Row 4: Original patch | GT SE  | Pred SE
    Select a lesion-containing patch for visualization.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in list(img_paths)[:n]:
        img_id = img_path.stem
        img_full, masks_full = load_image_and_masks(img_path)
        H, W = img_full.shape[:2]

        # Find a patch with the most lesion pixels (prefer MA)
        best_patch = None
        best_score = -1
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                mp = masks_full[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                score = mp[0].sum() * 10 + mp.sum()  # Weight MA heavily
                if score > best_score:
                    best_score = score
                    best_patch = (y, x)

        if best_patch is None:
            continue
        y0, x0 = best_patch
        ip  = img_full[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        mp  = masks_full[:, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]

        # Normalize for inference
        img_f = ip.astype(np.float32) / 255.0
        for c in range(3):
            img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(img_t)
        preds = (torch.sigmoid(logit).squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255

        orig_pil = Image.fromarray(ip)
        pw = PATCH_SIZE
        grid = Image.new('RGB', (pw * 3, pw * N_CLASSES))
        for i, (_, abbr) in enumerate(LESIONS):
            gt_pil   = Image.fromarray((mp[i] * 255).astype(np.uint8)).convert('RGB')
            pred_pil = Image.fromarray(preds[i]).convert('RGB')
            grid.paste(orig_pil, (0,      pw * i))
            grid.paste(gt_pil,   (pw,     pw * i))
            grid.paste(pred_pil, (pw * 2, pw * i))

        out_path = save_dir / f'fold{fold_idx+1}_{img_id}.png'
        grid.save(out_path)
        logging.info(f"  Saved qualitative: {out_path}")


# ============================================================
# Train Fold
# ============================================================

def train_fold(fold_idx, train_paths, val_paths, device):
    logging.info(f"\n{'='*60}")
    logging.info(f"FOLD {fold_idx+1}/{N_FOLDS}  |  train_imgs={len(train_paths)}  val_imgs={len(val_paths)}")
    logging.info(f"{'='*60}")

    logging.info("  Extracting patches (train)...")
    train_patches = build_patch_list(train_paths, oversample=True)
    logging.info("  Extracting patches (val, no oversample)...")
    val_patches   = build_patch_list(val_paths,   oversample=False)
    logging.info(f"  Train patches: {len(train_patches)}, Val patches: {len(val_patches)}")

    train_ds = PatchDataset(train_patches, augment=True)
    val_ds   = PatchDataset(val_patches,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = MobileNetV3UNet(N_CLASSES, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    scaler    = torch.amp.GradScaler('cuda')

    best_val_dice     = -1.0
    epochs_no_improve = 0
    history           = []
    fold_ckpt         = SAVE_DIR / f'fold{fold_idx+1}_best.pth'
    t_start           = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = combined_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss  = running_loss / len(train_loader)
        val_metrics = evaluate_fold(model, val_loader, device)
        val_dice    = val_metrics['mean_dice']
        scheduler.step(val_dice)

        logging.info(
            f"  Ep {epoch+1:03d} | TrLoss={train_loss:.4f} | "
            f"MeanDice={val_dice:.4f} | "
            f"MA={val_metrics['MA_dice']:.4f} HE={val_metrics['HE_dice']:.4f} "
            f"EX={val_metrics['EX_dice']:.4f} SE={val_metrics['SE_dice']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append({'epoch': epoch + 1, 'train_loss': train_loss, **val_metrics})

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), fold_ckpt)
            logging.info(f"  -> Checkpoint saved (MeanDice={best_val_dice:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - t_start
    logging.info(f"  Fold {fold_idx+1} done in {elapsed/60:.1f} min. Best Mean Val Dice: {best_val_dice:.4f}")

    model.load_state_dict(torch.load(fold_ckpt, map_location=device, weights_only=True))
    final_metrics = evaluate_fold(model, val_loader, device)

    # Qualitative on val images (not val patches)
    save_qualitative(model, val_paths, device, QUAL_DIR, fold_idx, n=3)

    return final_metrics, history, elapsed


# ============================================================
# Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device   = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    QUAL_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(list(TRAIN_IMG.glob('*.jpg')))
    if len(all_paths) != 54:
        raise ValueError(f"Expected 54 training images, found {len(all_paths)}")

    rng = np.random.default_rng(SEED)
    shuffled = all_paths.copy()
    rng.shuffle(shuffled)
    folds = [list(shuffled[i::N_FOLDS]) for i in range(N_FOLDS)]

    logging.info("=" * 60)
    logging.info("IDRiD LESION SEGMENTATION V2 — CONFIGURATION")
    logging.info("=" * 60)
    logging.info(f"  GPU:           {gpu_name}")
    logging.info(f"  PyTorch:       {torch.__version__}")
    logging.info(f"  Patch size:    {PATCH_SIZE}x{PATCH_SIZE}")
    logging.info(f"  Stride:        {STRIDE}")
    logging.info(f"  Min pos px:    {MIN_POS_PIXELS}")
    logging.info(f"  Oversample:    {N_POS_OVERSAMPLE}:1 pos:neg")
    logging.info(f"  Loss:          Focal BCE (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}) + Dice")
    logging.info(f"  Optimizer:     AdamW (lr={LR}, wd={WEIGHT_DECAY})")
    logging.info(f"  Scheduler:     ReduceLROnPlateau (patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR})")

    logging.info(f"  Max Epochs:    {MAX_EPOCHS}")
    logging.info(f"  ES Patience:   {EARLY_STOP_PATIENCE}")
    logging.info(f"  Batch size:    {BATCH_SIZE}")
    logging.info(f"  N Folds:       {N_FOLDS}")
    for i, f in enumerate(folds):
        ids = [p.stem for p in f]
        logging.info(f"  Fold {i+1} val: {ids}")
    logging.info("=" * 60)

    all_fold_metrics = []
    all_histories    = []
    total_start      = time.time()

    for fold_idx in range(N_FOLDS):
        val_paths   = folds[fold_idx]
        train_paths = [p for fold in folds for p in fold if fold != folds[fold_idx]]
        # Deduplicate while preserving order
        seen = set()
        train_paths_dedup = []
        for p in train_paths:
            if p not in seen:
                seen.add(p)
                train_paths_dedup.append(p)
        train_paths = train_paths_dedup

        metrics, history, elapsed = train_fold(fold_idx, train_paths, val_paths, device)
        metrics['fold']              = fold_idx + 1
        metrics['training_time_s']   = elapsed
        all_fold_metrics.append(metrics)
        all_histories.extend([{**h, 'fold': fold_idx + 1} for h in history])

    total_elapsed = time.time() - total_start

    logging.info("\n" + "=" * 60)
    logging.info("CROSS-VALIDATION SUMMARY (Mean +/- Std)")
    logging.info("=" * 60)

    summary = {}
    lesion_abbrs = ['MA', 'HE', 'EX', 'SE']
    for abbr in lesion_abbrs:
        for metric in ['dice', 'iou', 'sensitivity', 'specificity', 'precision']:
            key = f'{abbr}_{metric}'
            vals = [m[key] for m in all_fold_metrics if not np.isnan(m.get(key, float('nan')))]
            mean = float(np.mean(vals)) if vals else float('nan')
            std  = float(np.std(vals))  if vals else float('nan')
            summary[key] = {'mean': mean, 'std': std, 'per_fold': vals}
            logging.info(f"  {key:>22}: {mean:.4f} +/- {std:.4f}")

    mean_dice_vals = [m['mean_dice'] for m in all_fold_metrics]
    summary['mean_dice'] = {'mean': float(np.mean(mean_dice_vals)), 'std': float(np.std(mean_dice_vals))}
    logging.info(f"  {'mean_dice':>22}: {summary['mean_dice']['mean']:.4f} +/- {summary['mean_dice']['std']:.4f}")

    pd.DataFrame(all_fold_metrics).to_csv(SAVE_DIR / 'cv_results.csv', index=False)
    pd.DataFrame(all_histories).to_csv(SAVE_DIR / 'training_history.csv', index=False)

    metadata = {
        "model_name":        "DR-SUGAR-IDRiD-Lesions-V2",
        "version":           "2.0.0",
        "architecture":      "MobileNetV3-Small encoder + U-Net decoder",
        "task":              "retinal_lesion_segmentation_patch_based",
        "dataset":           "IDRiD A. Segmentation",
        "n_training_images": 54,
        "test_set_used":     False,
        "patch_size":        PATCH_SIZE,
        "stride":            STRIDE,
        "oversample_ratio":  N_POS_OVERSAMPLE,
        "min_pos_pixels":    MIN_POS_PIXELS,
        "loss":              f"FocalBCE(gamma={FOCAL_GAMMA},alpha={FOCAL_ALPHA}) + Dice",
        "optimizer":         f"AdamW(lr={LR}, wd={WEIGHT_DECAY})",
        "scheduler":         f"ReduceLROnPlateau(patience={SCHEDULER_PATIENCE},factor={SCHEDULER_FACTOR})",
        "max_epochs":        MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "batch_size":        BATCH_SIZE,
        "seed":              SEED,
        "amp":               True,
        "gpu":               gpu_name,
        "pytorch":           torch.__version__,
        "total_training_time_s": total_elapsed,
        "cv_summary":        summary,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "disclaimer":        "Research/prototype only. Not clinically validated.",
    }

    with open(SAVE_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logging.info(f"\nAll artifacts saved to {SAVE_DIR}/")
    logging.info(f"Total training time: {total_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
