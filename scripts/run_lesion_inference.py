#!/usr/bin/env python3
"""
DR-SUGAR — IDRiD Lesion Segmentation Inference
================================================
Runs lesion segmentation on a retinal fundus image using the trained
MobileNetV3-Small + U-Net decoder checkpoint (V3, Fold 1).

Processes the image via overlapping 512x512 patches (stride=384) and
stitches results by averaging overlapping regions.

Output channels:
  0: Microaneurysms (MA)   — red
  1: Haemorrhages (HE)     — orange  
  2: Hard Exudates (EX)    — yellow
  3: Soft Exudates (SE)    — cyan

Usage:
    python scripts/run_lesion_inference.py \
        --image data/uploads/<filename> \
        --output-dir data/outputs/<id>/ \
        --checkpoint models/idrid/lesions/v3/fold1_best.pth

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

PATCH_SIZE    = 512
STRIDE        = 384
THRESHOLD     = 0.5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
N_CLASSES     = 4

# Channel → color (RGBA overlay)
LESION_COLORS = [
    (220,  50,  50, 180),   # MA  — red
    (255, 140,   0, 180),   # HE  — orange
    (255, 220,   0, 180),   # EX  — yellow
    (  0, 220, 220, 180),   # SE  — cyan
]
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']


# ============================================================
# Model — identical to train_idrid_lesion_segmentation_v3.py
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
    def __init__(self, n_classes, pretrained=False):
        super().__init__()
        f = models.mobilenet_v3_small(weights=None).features
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


def load_model(checkpoint_path: str) -> nn.Module:
    model = MobileNetV3UNet(n_classes=N_CLASSES, pretrained=False)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_patch(patch_rgb: np.ndarray) -> torch.Tensor:
    """Convert H×W×3 uint8 numpy array to normalized 1×3×H×W tensor."""
    arr = patch_rgb.astype(np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def run_patch_inference(model: nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """
    Sliding window inference on full-resolution image.
    Returns: prob_map shape (N_CLASSES, H, W) float32 in [0,1].
    """
    H, W = image_rgb.shape[:2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    accumulator = np.zeros((N_CLASSES, H, W), dtype=np.float64)
    count_map   = np.zeros((H, W), dtype=np.float64)

    # Build patch coordinates
    y_starts = list(range(0, max(1, H - PATCH_SIZE + 1), STRIDE))
    x_starts = list(range(0, max(1, W - PATCH_SIZE + 1), STRIDE))

    # Ensure we cover the full image
    if not y_starts or y_starts[-1] + PATCH_SIZE < H:
        y_starts.append(max(0, H - PATCH_SIZE))
    if not x_starts or x_starts[-1] + PATCH_SIZE < W:
        x_starts.append(max(0, W - PATCH_SIZE))

    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + PATCH_SIZE, H)
                x1 = min(x0 + PATCH_SIZE, W)

                patch = image_rgb[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]

                # Pad if necessary
                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                tensor = normalize_patch(patch).to(device)
                logits = model(tensor)
                probs  = torch.sigmoid(logits)[0].cpu().numpy()  # N_CLASSES×512×512

                # Only accumulate the valid (unpadded) region
                accumulator[:, y0:y1, x0:x1] += probs[:, :ph, :pw]
                count_map[y0:y1, x0:x1] += 1.0

    # Avoid division by zero
    count_map = np.maximum(count_map, 1.0)
    prob_map = (accumulator / count_map[np.newaxis, :, :]).astype(np.float32)
    return prob_map


def create_lesion_overlay(original_img: Image.Image, prob_map: np.ndarray, threshold: float = THRESHOLD) -> Image.Image:
    """Create color overlay of all lesion predictions on the original image."""
    orig_w, orig_h = original_img.size
    orig_rgb = np.array(original_img.convert('RGB'), dtype=np.uint8)

    # Start with original image as RGBA
    overlay = np.array(original_img.convert('RGBA'), dtype=np.float32)

    for ch_idx, color in enumerate(LESION_COLORS):
        # Resize probability map channel to original size
        ch_prob = prob_map[ch_idx]  # H×W
        ch_pil  = Image.fromarray((ch_prob * 255).astype(np.uint8))
        ch_pil  = ch_pil.resize((orig_w, orig_h), Image.BILINEAR)
        ch_arr  = np.array(ch_pil, dtype=np.float32) / 255.0

        mask = ch_arr >= threshold

        if mask.any():
            r, g, b, a = color
            alpha_factor = (a / 255.0) * ch_arr  # proportional to probability
            for c_idx, c_val in enumerate([r, g, b]):
                overlay[:, :, c_idx] = np.where(
                    mask,
                    overlay[:, :, c_idx] * (1 - alpha_factor) + c_val * alpha_factor,
                    overlay[:, :, c_idx]
                )

    result = np.clip(overlay[:, :, :3], 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def main():
    parser = argparse.ArgumentParser(description='IDRiD Lesion Segmentation Inference')
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
    output_path = str(Path(args.output_dir) / 'lesion_overlay.png')

    # Load model
    model = load_model(args.checkpoint)

    # Load image
    img = Image.open(args.image).convert('RGB')
    image_rgb = np.array(img, dtype=np.uint8)
    H, W = image_rgb.shape[:2]

    # Run patch inference
    prob_map = run_patch_inference(model, image_rgb)

    # Calculate area percentages per channel
    total_pixels = H * W
    lesion_data = {}
    for i, name in enumerate(LESION_NAMES):
        resized_pil = Image.fromarray((prob_map[i] * 255).astype(np.uint8))
        resized_pil = resized_pil.resize((W, H), Image.BILINEAR)
        resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0
        mask = resized_arr >= THRESHOLD
        pct  = float(mask.sum() / total_pixels * 100)
        lesion_data[name] = round(pct, 4)

    # Create overlay
    overlay_img = create_lesion_overlay(img, prob_map)
    overlay_img.save(output_path, 'PNG')

    result = {
        "status": "OK",
        "lesion_path": output_path,
        "lesion_areas_pct": lesion_data,
        "model": "IDRiD-Lesions-V3-Fold1",
        "threshold": THRESHOLD
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
