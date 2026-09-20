"""
DR-SUGAR — IDRiD Lesion Segmentation V1
================================================
Architecture:
  U-Net with MobileNetV3-Small ImageNet-pretrained encoder.
  Decoder uses skip connections at 4 scale levels.
  Input: 512x512 (Original 4288x2848 is padded to square and resized).
  Output: 4 independent binary channels (MA, HE, EX, SE).

Data:
  54 IDRiD A. Segmentation training images.
  5-fold cross-validation.
  Missing masks are treated as all-zero true negatives.
  IDRiD test set NOT used.

Loss: Average of independent BCE + Dice per channel.
Optimizer: AdamW with ReduceLROnPlateau.
Mixed precision: torch.amp.autocast / GradScaler.
Seed: 42.

Research/prototype only. Not clinically validated.
"""

import sys
import os
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
from PIL import Image, ImageOps

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
IMG_SIZE   = 512
BATCH_SIZE = 4
MAX_EPOCHS = 80
LR         = 1e-4
WEIGHT_DECAY = 1e-2
EARLY_STOP_PATIENCE = 15
SCHEDULER_PATIENCE  = 5
SCHEDULER_FACTOR    = 0.5
N_FOLDS    = 5

BASE_DIR = Path('data/idrid/A. Segmentation')
TRAIN_IMG_DIR = BASE_DIR / '1. Original Images' / 'a. Training Set'
TRAIN_GT_BASE = BASE_DIR / '2. All Segmentation Groundtruths' / 'a. Training Set'

LESIONS = [
    ('1. Microaneurysms', 'MA'),
    ('2. Haemorrhages', 'HE'),
    ('3. Hard Exudates', 'EX'),
    ('4. Soft Exudates', 'SE')
]
N_CLASSES = len(LESIONS)

SAVE_DIR = Path('models/idrid/lesions/v1')
QUAL_DIR = SAVE_DIR / 'qualitative'

# ============================================================
# Dataset
# ============================================================

def pad_to_square(img):
    """Pad PIL image to square (black padding)."""
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    pad_w = (size - w) // 2
    pad_h = (size - h) // 2
    padding = (pad_w, pad_h, size - w - pad_w, size - h - pad_h)
    return ImageOps.expand(img, border=padding, fill=0)

class IDRiDLesionDataset(Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, img_paths, augment=False):
        self.img_paths = img_paths
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = Path(self.img_paths[idx])
        img_id = img_path.stem

        img_pil = Image.open(img_path).convert('RGB')
        
        # Load the 4 masks
        mask_pils = []
        for l_dir, l_abbr in LESIONS:
            mask_path = TRAIN_GT_BASE / l_dir / f'{img_id}_{l_abbr}.tif'
            if mask_path.exists():
                m_pil = Image.open(mask_path).convert('L')
            else:
                m_pil = Image.new('L', img_pil.size, 0)
            mask_pils.append(m_pil)

        # Pad to square and resize
        img_pil = pad_to_square(img_pil).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        for i in range(N_CLASSES):
            mask_pils[i] = pad_to_square(mask_pils[i]).resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)

        img = np.array(img_pil)
        masks = np.stack([(np.array(m) > 127).astype(np.float32) for m in mask_pils], axis=0) # Shape: (4, H, W)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                masks = masks[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                img = img[::-1, :, :].copy()
                masks = masks[:, ::-1, :].copy()
            angle = np.random.uniform(-10, 10)
            img = np.array(Image.fromarray(img).rotate(angle, resample=Image.BILINEAR))
            for i in range(N_CLASSES):
                m_rot = Image.fromarray((masks[i] * 255).astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
                masks[i] = (np.array(m_rot) > 127).astype(np.float32)
            
            jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
            img = np.array(jitter(Image.fromarray(img)))

        # Normalize
        img_f = img.astype(np.float32) / 255.0
        for c in range(3):
            img_f[:, :, c] = (img_f[:, :, c] - self.IMAGENET_MEAN[c]) / self.IMAGENET_STD[c]
        
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)) # C,H,W
        masks_t = torch.from_numpy(masks) # 4,H,W

        return img_t, masks_t, img_id

# ============================================================
# U-Net with MobileNetV3-Small Encoder
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
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV3UNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        feats = backbone.features

        self.enc0 = feats[0]            # 16ch, 256
        self.enc1 = feats[1]            # 16ch, 128
        self.enc2 = nn.Sequential(*feats[2:4])  # 24ch, 64
        self.enc3 = nn.Sequential(*feats[4:7])  # 40ch, 32
        self.enc4 = nn.Sequential(*feats[7:13]) # 576ch, 16

        self.dec4 = DecoderBlock(576, 40, 256)
        self.dec3 = DecoderBlock(256, 24, 128)
        self.dec2 = DecoderBlock(128, 16, 64)
        self.dec1 = DecoderBlock(64, 16, 32)
        
        self.up_final = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, n_classes, 1),
        )

    def forward(self, x):
        s0 = self.enc0(x)   # 256x256
        s1 = self.enc1(s0)  # 128x128
        s2 = self.enc2(s1)  # 64x64
        s3 = self.enc3(s2)  # 32x32
        b  = self.enc4(s3)  # 16x16

        d4 = self.dec4(b,  s3)  
        d3 = self.dec3(d4, s2)  
        d2 = self.dec2(d3, s1)  
        d1 = self.dec1(d2, s0)  

        out = self.up_final(d1) 
        out = self.final_conv(out)
        return out  # logits, shape: B, 4, H, W

# ============================================================
# Loss
# ============================================================

def bce_dice_loss(logits, targets, smooth=1.0):
    """BCE + Dice computed independently for each channel, then averaged."""
    # logits/targets: B, 4, H, W
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    
    probs = torch.sigmoid(logits)
    # compute Dice per channel, per item in batch
    # flatten spatial dims
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)
    targets_flat = targets.view(targets.size(0), targets.size(1), -1)
    
    intersection = (probs_flat * targets_flat).sum(dim=2)
    dice_score = (2 * intersection + smooth) / (probs_flat.sum(dim=2) + targets_flat.sum(dim=2) + smooth)
    dice_loss = 1.0 - dice_score.mean()
    
    return bce + dice_loss

# ============================================================
# Metrics
# ============================================================

def compute_channel_metrics(probs, targets, threshold=0.5):
    """Compute metrics for a single channel across the dataset."""
    preds = (probs >= threshold).astype(np.float32)
    
    TP = ((preds == 1) & (targets == 1)).sum()
    TN = ((preds == 0) & (targets == 0)).sum()
    FP = ((preds == 1) & (targets == 0)).sum()
    FN = ((preds == 0) & (targets == 1)).sum()

    # If targets are all zero, dice/iou defaults to 1 if no FP, 0 if FP
    if targets.sum() == 0:
        dice = 1.0 if FP == 0 else 0.0
        iou  = 1.0 if FP == 0 else 0.0
        sens = 1.0 # True negative image, recall is nominally 1
        prec = 0.0
    else:
        dice = (2 * TP + 1e-7) / (2 * TP + FP + FN + 1e-7)
        iou  = (TP + 1e-7) / (TP + FP + FN + 1e-7)
        sens = (TP + 1e-7) / (TP + FN + 1e-7)
        prec = (TP + 1e-7) / (TP + FP + 1e-7)
        
    spec = (TN + 1e-7) / (TN + FP + 1e-7)
    
    return float(dice), float(iou), float(sens), float(spec), float(prec)

def evaluate_fold(model, loader, device):
    model.eval()
    total_loss = 0.0
    
    # store probs and targets to compute global metrics
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks)
            total_loss += loss.item()
            
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
    avg_loss = total_loss / max(len(loader), 1)
    
    all_probs = np.concatenate(all_probs, axis=0) # N, 4, H, W
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = {'loss': avg_loss}
    mean_dice = 0.0
    
    for i, (_, abbr) in enumerate(LESIONS):
        dice, iou, sens, spec, prec = compute_channel_metrics(all_probs[:, i], all_targets[:, i])
        metrics[f'{abbr}_dice'] = dice
        metrics[f'{abbr}_iou'] = iou
        metrics[f'{abbr}_sensitivity'] = sens
        metrics[f'{abbr}_specificity'] = spec
        metrics[f'{abbr}_precision'] = prec
        mean_dice += dice
        
    metrics['mean_dice'] = mean_dice / N_CLASSES
    return metrics

# ============================================================
# Qualitative
# ============================================================

def save_qualitative(model, dataset, device, save_dir, fold_idx, n=2):
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    for idx in range(min(n, len(dataset))):
        img_t, mask_t, img_id = dataset[idx]
        with torch.no_grad():
            logit = model(img_t.unsqueeze(0).to(device))
        probs = torch.sigmoid(logit).squeeze().cpu().numpy() # 4, H, W
        preds = (probs >= 0.5).astype(np.uint8) * 255
        
        img_np = img_t.numpy().transpose(1, 2, 0)
        img_np = (img_np * STD + MEAN).clip(0, 1)
        orig_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        w, h = orig_pil.size
        
        # Grid: [Orig] [MA GT] [MA Pred] | [HE GT] [HE Pred] | [EX GT] [EX Pred] | [SE GT] [SE Pred]
        # Let's make a vertical stack of rows for each lesion
        grid = Image.new('RGB', (w * 3, h * 4))
        
        for i, (_, abbr) in enumerate(LESIONS):
            gt_pil = Image.fromarray((mask_t[i].numpy() * 255).astype(np.uint8)).convert('RGB')
            pred_pil = Image.fromarray(preds[i]).convert('RGB')
            
            y_offset = h * i
            grid.paste(orig_pil, (0, y_offset))
            grid.paste(gt_pil, (w, y_offset))
            grid.paste(pred_pil, (w*2, y_offset))
            
        out_path = save_dir / f'fold{fold_idx+1}_{img_id}_pred.png'
        grid.save(out_path)

# ============================================================
# Train Fold
# ============================================================

def train_fold(fold_idx, train_paths, val_paths, device):
    logging.info(f"\n{'='*60}")
    logging.info(f"FOLD {fold_idx+1}/{N_FOLDS}  |  train={len(train_paths)}  val={len(val_paths)}")
    logging.info(f"{'='*60}")

    train_ds = IDRiDLesionDataset(train_paths, augment=True)
    val_ds   = IDRiDLesionDataset(val_paths, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = MobileNetV3UNet(N_CLASSES, pretrained=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    scaler = torch.amp.GradScaler('cuda')

    best_val_dice = -1.0
    epochs_no_improve = 0
    history = []
    fold_ckpt = SAVE_DIR / f'fold{fold_idx+1}_best.pth'
    t_start = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_fold(model, val_loader, device)
        val_dice = val_metrics['mean_dice']
        scheduler.step(val_dice)

        logging.info(f"  Ep {epoch+1:03d} | TrLoss={train_loss:.4f} | ValMeanDice={val_dice:.4f} | LR={optimizer.param_groups[0]['lr']:.2e}")
        
        history.append({'epoch': epoch+1, 'train_loss': train_loss, **val_metrics})

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), fold_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - t_start
    logging.info(f"  Fold {fold_idx+1} training complete in {elapsed/60:.1f} min. Best Mean Val Dice: {best_val_dice:.4f}")

    model.load_state_dict(torch.load(fold_ckpt, map_location=device, weights_only=True))
    final_metrics = evaluate_fold(model, val_loader, device)
    
    save_qualitative(model, val_ds, device, QUAL_DIR, fold_idx, n=2)

    return final_metrics, history, elapsed

# ============================================================
# Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    QUAL_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(list(TRAIN_IMG_DIR.glob('*.jpg')))
    if len(all_paths) != 54:
        raise ValueError(f"Expected 54 training images, found {len(all_paths)}")

    np.random.seed(SEED)
    shuffled_paths = all_paths.copy()
    np.random.shuffle(shuffled_paths)
    folds = [shuffled_paths[i::N_FOLDS] for i in range(N_FOLDS)]

    logging.info("=" * 60)
    logging.info("IDRiD LESION SEGMENTATION V1 — CONFIGURATION")
    logging.info("=" * 60)
    logging.info(f"  GPU:         {gpu_name}")
    logging.info(f"  Input size:  {IMG_SIZE}x{IMG_SIZE} (padded to square, resized)")
    logging.info(f"  Images:      {len(all_paths)}")
    logging.info(f"  Classes:     4 (MA, HE, EX, SE)")
    logging.info("=" * 60)

    all_fold_metrics = []
    all_histories = []
    total_start = time.time()

    for fold_idx in range(N_FOLDS):
        val_paths = folds[fold_idx]
        train_paths = [p for p in shuffled_paths if p not in val_paths]
        metrics, history, elapsed = train_fold(fold_idx, train_paths, val_paths, device)
        metrics['fold'] = fold_idx + 1
        metrics['training_time_s'] = elapsed
        all_fold_metrics.append(metrics)
        all_histories.extend([{**h, 'fold': fold_idx + 1} for h in history])

    total_elapsed = time.time() - total_start

    logging.info("\n" + "=" * 60)
    logging.info("CROSS-VALIDATION SUMMARY (Mean ± Std)")
    logging.info("=" * 60)
    
    summary = {}
    for abbr in ['mean', 'MA', 'HE', 'EX', 'SE']:
        prefix = f'{abbr}_' if abbr != 'mean' else 'mean_'
        metric_keys = [k for k in all_fold_metrics[0].keys() if k.startswith(prefix)]
        for k in metric_keys:
            vals = [m[k] for m in all_fold_metrics]
            mean, std = np.mean(vals), np.std(vals)
            summary[k] = {'mean': float(mean), 'std': float(std)}
            logging.info(f"  {k:>15}: {mean:.4f} ± {std:.4f}")

    pd.DataFrame(all_fold_metrics).to_csv(SAVE_DIR / 'cv_results.csv', index=False)
    pd.DataFrame(all_histories).to_csv(SAVE_DIR / 'training_history.csv', index=False)

    metadata = {
        "model_name": "DR-SUGAR-IDRiD-Lesions-V1",
        "training_time_s": total_elapsed,
        "cv_summary": summary,
        "architecture": "MobileNetV3-Small encoder + U-Net decoder",
        "input_size": IMG_SIZE,
        "n_folds": N_FOLDS
    }
    with open(SAVE_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"\nAll artifacts saved to {SAVE_DIR}/")

if __name__ == '__main__':
    main()
