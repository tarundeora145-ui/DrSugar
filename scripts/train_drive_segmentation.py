"""
DR-SUGAR — DRIVE Retinal Vessel Segmentation V1
================================================
Architecture:
  U-Net with MobileNetV3-Small ImageNet-pretrained encoder.
  Decoder uses skip connections at 4 scale levels.
  Input: 512x512 (center-cropped from 584x565 DRIVE images).

Data:
  20 DRIVE training images with vessel GT.
  5-fold cross-validation (16 train / 4 val per fold).
  FOV mask used to exclude outside-retina pixels from loss and metrics.
  DRIVE test set NOT used — no public vessel GT available.

Loss: Combined BCE + Dice.
Optimizer: AdamW with ReduceLROnPlateau.
Mixed precision: torch.amp.autocast / GradScaler.
Seed: 42.

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
IMG_SIZE   = 512       # Center-crop target (from 584x565)
BATCH_SIZE = 4         # Small batch: 512x512 images are large
MAX_EPOCHS = 80
LR         = 1e-4
WEIGHT_DECAY = 1e-2
EARLY_STOP_PATIENCE = 12
SCHEDULER_PATIENCE  = 5
SCHEDULER_FACTOR    = 0.5
N_FOLDS    = 5

TRAIN_IMG  = Path('data/drive/training/images')
TRAIN_MASK = Path('data/drive/training/1st_manual')
TRAIN_FOV  = Path('data/drive/training/mask')
SAVE_DIR   = Path('models/drive/v1')

# ============================================================
# Dataset
# ============================================================

class DRIVEDataset(Dataset):
    """DRIVE retinal vessel segmentation dataset."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, img_ids, img_dir, mask_dir, fov_dir,
                 augment=False, img_size=IMG_SIZE):
        self.img_ids = img_ids
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.fov_dir  = Path(fov_dir)
        self.augment  = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.img_ids)

    def _center_crop(self, arr, size):
        h, w = arr.shape[:2]
        top  = (h - size) // 2
        left = (w - size) // 2
        if arr.ndim == 3:
            return arr[top:top+size, left:left+size, :]
        return arr[top:top+size, left:left+size]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img  = np.array(Image.open(self.img_dir  / f'{img_id}_training.tif').convert('RGB'))
        mask = np.array(Image.open(self.mask_dir / f'{img_id}_manual1.gif').convert('L'))
        fov  = np.array(Image.open(self.fov_dir  / f'{img_id}_training_mask.gif').convert('L'))

        img  = self._center_crop(img,  self.img_size)
        mask = self._center_crop(mask, self.img_size)
        fov  = self._center_crop(fov,  self.img_size)

        # Binarize
        mask = (mask > 127).astype(np.float32)
        fov  = (fov  > 127).astype(np.float32)

        # Augmentation (applied identically to image, mask, and fov)
        if self.augment:
            if np.random.rand() > 0.5:
                img  = img[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
                fov  = fov[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img  = img[::-1, :, :].copy()
                mask = mask[::-1, :].copy()
                fov  = fov[::-1, :].copy()
            angle = np.random.uniform(-10, 10)
            img_pil  = Image.fromarray(img).rotate(angle, resample=Image.BILINEAR)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
            fov_pil  = Image.fromarray((fov  * 255).astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
            img  = np.array(img_pil)
            mask = (np.array(mask_pil) > 127).astype(np.float32)
            fov  = (np.array(fov_pil)  > 127).astype(np.float32)
            # Mild brightness/contrast jitter on image only
            jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
            img = np.array(jitter(Image.fromarray(img)))

        # Normalize image
        img_f = img.astype(np.float32) / 255.0
        for c in range(3):
            img_f[:, :, c] = (img_f[:, :, c] - self.IMAGENET_MEAN[c]) / self.IMAGENET_STD[c]
        img_t  = torch.from_numpy(img_f.transpose(2, 0, 1))      # C,H,W
        mask_t = torch.from_numpy(mask).unsqueeze(0)              # 1,H,W
        fov_t  = torch.from_numpy(fov).unsqueeze(0)               # 1,H,W

        return img_t, mask_t, fov_t


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
        # Handle slight size mismatches from integer division
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV3UNet(nn.Module):
    """
    U-Net segmentation model using MobileNetV3-Small as the encoder.

    Encoder skip connection points (at 512x512 input):
      s0: after features[0]  → 16ch, 256x256
      s1: after features[1]  → 16ch, 128x128
      s2: after features[3]  → 24ch,  64x64
      s3: after features[6]  → 40ch,  32x32
      bottleneck: after features[12] → 576ch, 16x16

    Decoder upsamples back: 16x16 → 32 → 64 → 128 → 256 → 512
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        feats = backbone.features

        # Encoder stages (split backbone into segments)
        self.enc0 = feats[0]            # -> 16ch, /2  (256)
        self.enc1 = feats[1]            # -> 16ch, /2  (128)
        self.enc2 = nn.Sequential(*feats[2:4])  # -> 24ch, /2 (64)
        self.enc3 = nn.Sequential(*feats[4:7])  # -> 40ch, same (32)
        self.enc4 = nn.Sequential(*feats[7:13]) # -> 576ch, /2 (16)

        # Extra downsample to get the /4 stride after enc0→enc1
        # enc0 outputs /2 (256), enc1 outputs /4 (128)

        # Decoder
        # bottleneck 576ch @ 16 → up → 32, skip with enc3 40ch
        self.dec4 = DecoderBlock(576, 40, 256)
        # 256ch @ 32 → up → 64, skip with enc2 24ch
        self.dec3 = DecoderBlock(256, 24, 128)
        # 128ch @ 64 → up → 128, skip with enc1 16ch
        self.dec2 = DecoderBlock(128, 16, 64)
        # 64ch @ 128 → up → 256, skip with enc0 16ch
        self.dec1 = DecoderBlock(64, 16, 32)
        # 32ch @ 256 → up → 512 (original size), no skip
        self.up_final = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        # Encoder
        s0 = self.enc0(x)   # 16ch, 256x256
        s1 = self.enc1(s0)  # 16ch, 128x128
        s2 = self.enc2(s1)  # 24ch,  64x64
        s3 = self.enc3(s2)  # 40ch,  32x32
        b  = self.enc4(s3)  # 576ch, 16x16

        # Decoder
        d4 = self.dec4(b,  s3)  # → 256ch, 32x32
        d3 = self.dec3(d4, s2)  # → 128ch, 64x64
        d2 = self.dec2(d3, s1)  # →  64ch, 128x128
        d1 = self.dec1(d2, s0)  # →  32ch, 256x256

        out = self.up_final(d1) # →  16ch, 512x512
        out = self.final_conv(out)
        return out  # logits, shape: B,1,H,W


# ============================================================
# Loss
# ============================================================

def bce_dice_loss(logits, targets, fov, smooth=1.0):
    """
    Combined BCE + Dice loss, computed only within the FOV mask.
    """
    # Flatten and apply FOV
    fov_flat    = fov.view(-1).bool()
    logits_flat = logits.view(-1)[fov_flat]
    targets_flat = targets.view(-1)[fov_flat]

    bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat)

    probs = torch.sigmoid(logits_flat)
    intersection = (probs * targets_flat).sum()
    dice = 1.0 - (2 * intersection + smooth) / (probs.sum() + targets_flat.sum() + smooth)

    return bce + dice


# ============================================================
# Metrics
# ============================================================

def compute_metrics(logits, targets, fov, threshold=0.5):
    """Compute Dice, IoU, Sensitivity, Specificity and collect probs+labels for AUC."""
    probs   = torch.sigmoid(logits)
    fov_b   = fov.bool()

    probs_np   = probs[fov_b].detach().cpu().numpy()
    targets_np = targets[fov_b].detach().cpu().numpy()
    preds_np   = (probs_np >= threshold).astype(np.float32)

    TP = ((preds_np == 1) & (targets_np == 1)).sum()
    TN = ((preds_np == 0) & (targets_np == 0)).sum()
    FP = ((preds_np == 1) & (targets_np == 0)).sum()
    FN = ((preds_np == 0) & (targets_np == 1)).sum()

    dice        = (2 * TP + 1e-7) / (2 * TP + FP + FN + 1e-7)
    iou         = (TP + 1e-7) / (TP + FP + FN + 1e-7)
    sensitivity = (TP + 1e-7) / (TP + FN + 1e-7)
    specificity = (TN + 1e-7) / (TN + FP + 1e-7)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'probs': probs_np,
        'labels': targets_np,
    }


# ============================================================
# Evaluation loop
# ============================================================

def evaluate_fold(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    dice_list, iou_list, sens_list, spec_list = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for imgs, masks, fovs in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            fovs  = fovs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = bce_dice_loss(logits, masks, fovs)
            total_loss += loss.item()
            m = compute_metrics(logits, masks, fovs)
            dice_list.append(m['dice'])
            iou_list.append(m['iou'])
            sens_list.append(m['sensitivity'])
            spec_list.append(m['specificity'])
            all_probs.extend(m['probs'].tolist())
            all_labels.extend(m['labels'].tolist())

    avg_loss = total_loss / max(len(loader), 1)

    from sklearn.metrics import roc_auc_score
    roc_auc = None
    if len(set(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'dice': float(np.mean(dice_list)),
        'iou': float(np.mean(iou_list)),
        'sensitivity': float(np.mean(sens_list)),
        'specificity': float(np.mean(spec_list)),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
    }


# ============================================================
# Qualitative visualization helper
# ============================================================

def save_qualitative(model, dataset, device, save_path, n=4):
    """Save a grid: [Original | GT | Prediction] for n images."""
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    rows = []
    for idx in range(min(n, len(dataset))):
        img_t, mask_t, fov_t = dataset[idx]
        with torch.no_grad():
            logit = model(img_t.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logit).squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255

        # Denormalize image
        img_np = img_t.numpy().transpose(1, 2, 0)  # H,W,C
        img_np = (img_np * STD + MEAN).clip(0, 1)
        img_np = (img_np * 255).astype(np.uint8)

        gt_np = (mask_t.squeeze().numpy() * 255).astype(np.uint8)

        orig_pil = Image.fromarray(img_np)
        gt_pil   = Image.fromarray(gt_np).convert('RGB')
        pred_pil = Image.fromarray(pred).convert('RGB')

        # Concatenate horizontally
        w, h = orig_pil.size
        row  = Image.new('RGB', (w * 3, h))
        row.paste(orig_pil, (0, 0))
        row.paste(gt_pil,   (w, 0))
        row.paste(pred_pil, (w * 2, 0))
        rows.append(row)

    # Stack rows vertically
    total_h = sum(r.height for r in rows)
    grid = Image.new('RGB', (rows[0].width, total_h))
    y = 0
    for r in rows:
        grid.paste(r, (0, y))
        y += r.height
    grid.save(save_path)
    logging.info(f"Saved qualitative image: {save_path}")


# ============================================================
# Train one fold
# ============================================================

def train_fold(fold_idx, train_ids, val_ids, device, save_dir):
    logging.info(f"\n{'='*60}")
    logging.info(f"FOLD {fold_idx+1}/{N_FOLDS}  |  train={train_ids}  val={val_ids}")
    logging.info(f"{'='*60}")

    train_ds = DRIVEDataset(train_ids, TRAIN_IMG, TRAIN_MASK, TRAIN_FOV, augment=True)
    val_ds   = DRIVEDataset(val_ids,   TRAIN_IMG, TRAIN_MASK, TRAIN_FOV, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = MobileNetV3UNet(pretrained=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    scaler = torch.amp.GradScaler('cuda')

    best_val_dice     = 0.0
    epochs_no_improve = 0
    history           = []
    fold_ckpt         = save_dir / f'fold{fold_idx+1}_best.pth'
    t_start           = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, masks, fovs in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            fovs  = fovs.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = bce_dice_loss(logits, masks, fovs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_fold(model, val_loader, device)
        val_dice    = val_metrics['dice']
        scheduler.step(val_dice)

        logging.info(
            f"  Ep {epoch+1:03d} | TrLoss={train_loss:.4f} | "
            f"ValDice={val_dice:.4f} | ValIoU={val_metrics['iou']:.4f} | "
            f"Sens={val_metrics['sensitivity']:.4f} | Spec={val_metrics['specificity']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_dice': val_dice,
            'val_iou': val_metrics['iou'],
            'val_sensitivity': val_metrics['sensitivity'],
            'val_specificity': val_metrics['specificity'],
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), fold_ckpt)
            logging.info(f"  -> Best checkpoint saved (Dice={best_val_dice:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - t_start
    logging.info(f"  Fold {fold_idx+1} training complete in {elapsed/60:.1f} min. Best Val Dice: {best_val_dice:.4f}")

    # Reload best and do final eval + qualitative
    model.load_state_dict(torch.load(fold_ckpt, map_location=device, weights_only=True))
    final_metrics = evaluate_fold(model, val_loader, device)
    logging.info(f"  Fold {fold_idx+1} Final Metrics: {final_metrics}")

    # Save qualitative predictions
    save_qualitative(
        model, val_ds, device,
        save_dir / f'fold{fold_idx+1}_predictions.png',
        n=4
    )

    return final_metrics, history, elapsed


# ============================================================
# Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DRIVE V1 requires GPU training.")

    device   = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # All 20 training image IDs
    all_ids = [str(i) for i in range(21, 41)]

    # Deterministic 5-fold split (sequential — no shuffling needed for 20 images)
    np.random.seed(SEED)
    shuffled_ids = all_ids.copy()
    np.random.shuffle(shuffled_ids)
    folds = [shuffled_ids[i::N_FOLDS] for i in range(N_FOLDS)]

    logging.info("=" * 60)
    logging.info("DRIVE VESSEL SEGMENTATION V1 — CONFIGURATION")
    logging.info("=" * 60)
    logging.info(f"  GPU:         {gpu_name}")
    logging.info(f"  CUDA:        {torch.version.cuda}")
    logging.info(f"  PyTorch:     {torch.__version__}")
    logging.info(f"  Architecture: MobileNetV3-Small encoder + U-Net decoder")
    logging.info(f"  Input size:  {IMG_SIZE}x{IMG_SIZE} (center-cropped from 584x565)")
    logging.info(f"  Loss:        BCE + Dice (FOV-masked)")
    logging.info(f"  Optimizer:   AdamW (lr={LR}, wd={WEIGHT_DECAY})")
    logging.info(f"  AMP:         True")
    logging.info(f"  Max Epochs:  {MAX_EPOCHS}")
    logging.info(f"  ES Patience: {EARLY_STOP_PATIENCE}")
    logging.info(f"  N Folds:     {N_FOLDS}")
    logging.info(f"  Output:      {SAVE_DIR}")
    for i, f in enumerate(folds):
        logging.info(f"  Fold {i+1} val IDs: {f}")
    logging.info("=" * 60)

    all_fold_metrics = []
    all_histories    = []
    total_start = time.time()

    for fold_idx in range(N_FOLDS):
        val_ids   = folds[fold_idx]
        train_ids = [x for x in shuffled_ids if x not in val_ids]
        metrics, history, elapsed = train_fold(fold_idx, train_ids, val_ids, device, SAVE_DIR)
        metrics['fold'] = fold_idx + 1
        metrics['training_time_s'] = elapsed
        all_fold_metrics.append(metrics)
        all_histories.extend([{**h, 'fold': fold_idx + 1} for h in history])

    total_elapsed = time.time() - total_start

    # ---- Summary ----
    logging.info("\n" + "=" * 60)
    logging.info("CROSS-VALIDATION SUMMARY")
    logging.info("=" * 60)
    metric_keys = ['dice', 'iou', 'sensitivity', 'specificity', 'roc_auc']
    summary = {}
    for k in metric_keys:
        vals = [m[k] for m in all_fold_metrics if m[k] is not None]
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        summary[k] = {'mean': mean, 'std': std, 'per_fold': vals}
        logging.info(f"  {k:>15}: {mean:.4f} ± {std:.4f}  (per fold: {[f'{v:.4f}' for v in vals]})")

    # ---- Save results ----
    cv_df = pd.DataFrame(all_fold_metrics)
    cv_df.to_csv(SAVE_DIR / 'cv_results.csv', index=False)
    logging.info(f"\nSaved cv_results.csv")

    pd.DataFrame(all_histories).to_csv(SAVE_DIR / 'training_history.csv', index=False)

    metadata = {
        "model_name": "DR-SUGAR-DRIVE-MobileNetV3UNet-V1",
        "version": "1.0.0",
        "architecture": "MobileNetV3-Small encoder + U-Net decoder",
        "task": "retinal_vessel_segmentation",
        "dataset": "DRIVE",
        "n_training_images": 20,
        "n_test_images": 20,
        "test_vessel_gt_available": False,
        "evaluation": "5-fold cross-validation (training set only)",
        "input_size": IMG_SIZE,
        "preprocessing": "Center-crop 512x512, ImageNet normalization",
        "augmentation": ["horizontal_flip", "vertical_flip", "rotation_10deg", "mild_brightness_contrast"],
        "loss": "BCE + Dice (FOV-masked)",
        "optimizer": f"AdamW(lr={LR}, wd={WEIGHT_DECAY})",
        "scheduler": f"ReduceLROnPlateau(patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR})",
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "amp": True,
        "gpu": gpu_name,
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "total_training_time_s": total_elapsed,
        "cv_summary": summary,
        "per_fold_metrics": all_fold_metrics,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "disclaimer": "Research/prototype only. Not clinically validated. DRIVE test set not used for metrics.",
    }

    with open(SAVE_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logging.info(f"\nAll artifacts saved to {SAVE_DIR}/")
    logging.info(f"Total training time: {total_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
