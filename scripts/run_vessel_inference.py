#!/usr/bin/env python3
"""
DR-SUGAR — DRIVE Vessel Segmentation Inference
================================================
Runs retinal vessel segmentation on a fundus image using the trained
MobileNetV3-Small + U-Net decoder checkpoint (DRIVE V1, Fold 1).

Processes via a single 512×512 center-crop (matching DRIVE training).
For larger images the crop is taken from the center; for smaller images
the image is padded with zeros.

Usage:
    python scripts/run_vessel_inference.py \
        --image data/uploads/<filename> \
        --output-dir data/outputs/<id>/ \
        --checkpoint models/drive/v1/fold1_best.pth

Outputs JSON to stdout.

Research/prototype only. Not clinically validated.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

IMG_SIZE      = 512
THRESHOLD     = 0.5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Vessel overlay: bright cyan at 60% opacity
VESSEL_COLOR  = (0, 210, 255)
VESSEL_ALPHA  = 0.55


# ============================================================
# Model — identical to train_drive_segmentation.py
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
    def __init__(self):
        super().__init__()
        feats = models.mobilenet_v3_small(weights=None).features
        self.enc0 = feats[0]
        self.enc1 = feats[1]
        self.enc2 = nn.Sequential(*feats[2:4])
        self.enc3 = nn.Sequential(*feats[4:7])
        self.enc4 = nn.Sequential(*feats[7:13])
        self.dec4 = DecoderBlock(576, 40, 256)
        self.dec3 = DecoderBlock(256, 24, 128)
        self.dec2 = DecoderBlock(128, 16, 64)
        self.dec1 = DecoderBlock(64, 16, 32)
        self.up_final = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            DoubleConv(16, 16),
            nn.Conv2d(16, 1, 1),
        )

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
        out = self.final_conv(out)
        return out  # logits, shape: B,1,H,W


def load_model(checkpoint_path: str) -> nn.Module:
    model = MobileNetV3UNet()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def center_crop_or_pad(arr: np.ndarray, size: int) -> tuple[np.ndarray, tuple]:
    """
    Center-crop or pad an H×W×3 array to size×size.
    Returns: (cropped/padded array, (top, left, orig_h, orig_w) for reverse mapping)
    """
    h, w = arr.shape[:2]

    if h >= size and w >= size:
        top  = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size], (top, left, h, w)
    else:
        # Pad with zeros
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        top  = (size - h) // 2 if h < size else 0
        left = (size - w) // 2 if w < size else 0
        paste_h = min(h, size)
        paste_w = min(w, size)
        padded[top:top+paste_h, left:left+paste_w] = arr[:paste_h, :paste_w]
        return padded, (-top, -left, h, w)


def normalize_img(arr_rgb: np.ndarray) -> torch.Tensor:
    f = arr_rgb.astype(np.float32) / 255.0
    for c in range(3):
        f[:, :, c] = (f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return torch.from_numpy(f.transpose(2, 0, 1)).unsqueeze(0)


def create_vessel_overlay(original_img: Image.Image, vessel_mask: np.ndarray) -> Image.Image:
    """Blend vessel mask (H×W bool) over original image in cyan."""
    orig_arr = np.array(original_img.convert('RGB'), dtype=np.float32)
    H, W = orig_arr.shape[:2]

    # Resize mask back to original image size if needed
    if vessel_mask.shape != (H, W):
        mask_pil = Image.fromarray((vessel_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((W, H), Image.NEAREST)
        vessel_mask = np.array(mask_pil) > 127

    r, g, b = VESSEL_COLOR
    overlay = orig_arr.copy()
    overlay[vessel_mask, 0] = overlay[vessel_mask, 0] * (1 - VESSEL_ALPHA) + r * VESSEL_ALPHA
    overlay[vessel_mask, 1] = overlay[vessel_mask, 1] * (1 - VESSEL_ALPHA) + g * VESSEL_ALPHA
    overlay[vessel_mask, 2] = overlay[vessel_mask, 2] * (1 - VESSEL_ALPHA) + b * VESSEL_ALPHA

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description='DRIVE Vessel Segmentation Inference')
    parser.add_argument('--image',      required=True, help='Input retinal image path')
    parser.add_argument('--output-dir', required=True, help='Output directory for overlay')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path (.pth)')
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(json.dumps({"error": f"Image not found: {args.image}"}))
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(json.dumps({"error": f"Checkpoint not found: {args.checkpoint}"}))
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(args.output_dir) / 'vessel_overlay.png')

    # Load model
    model = load_model(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load image
    img = Image.open(args.image).convert('RGB')
    image_rgb = np.array(img, dtype=np.uint8)
    orig_h, orig_w = image_rgb.shape[:2]

    # Center-crop to 512×512 (matching DRIVE training preprocessing)
    cropped, (top, left, _, _) = center_crop_or_pad(image_rgb, IMG_SIZE)
    tensor = normalize_img(cropped).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()  # H×W

    # Threshold
    vessel_mask_crop = probs >= THRESHOLD

    # Map back to full image coordinates
    vessel_mask_full = np.zeros((orig_h, orig_w), dtype=bool)
    # top/left is offset within original where crop starts
    crop_h = min(IMG_SIZE, orig_h)
    crop_w = min(IMG_SIZE, orig_w)

    if top >= 0:
        dst_y0 = top
        src_y0 = 0
    else:
        dst_y0 = 0
        src_y0 = -top

    if left >= 0:
        dst_x0 = left
        src_x0 = 0
    else:
        dst_x0 = 0
        src_x0 = -left

    h_copy = min(IMG_SIZE - src_y0, orig_h - dst_y0)
    w_copy = min(IMG_SIZE - src_x0, orig_w - dst_x0)

    if h_copy > 0 and w_copy > 0:
        vessel_mask_full[dst_y0:dst_y0+h_copy, dst_x0:dst_x0+w_copy] = \
            vessel_mask_crop[src_y0:src_y0+h_copy, src_x0:src_x0+w_copy]

    # Calculate coverage
    total_pixels = orig_h * orig_w
    vessel_coverage_pct = float(vessel_mask_full.sum() / total_pixels * 100)

    # Create overlay
    overlay_img = create_vessel_overlay(img, vessel_mask_full)
    overlay_img.save(output_path, 'PNG')

    result = {
        "status": "OK",
        "vessel_path": output_path,
        "vessel_coverage_pct": round(vessel_coverage_pct, 4),
        "model": "DRIVE-V1-Fold1",
        "threshold": THRESHOLD
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
