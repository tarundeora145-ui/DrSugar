#!/usr/bin/env python3
"""
DR-SUGAR — Grad-CAM Generator
==============================
Generates a real gradient-weighted class activation map using the
original PyTorch MobileNetV3-Small DR classification checkpoint.

Usage:
    python scripts/generate_gradcam.py \
        --image data/uploads/<filename> \
        --output data/outputs/<id>/gradcam.png \
        --checkpoint models/dr_classification/best_model.pth \
        [--class-idx N]  # Optional: override predicted class

Outputs JSON to stdout:
    {"gradcam_path": "...", "predicted_class": N, "confidence": 0.xx}

Research/prototype only. Not clinically validated.
"""

import sys
import json
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

# Force UTF-8 on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224
N_CLASSES     = 5


def build_model(checkpoint_path: str) -> nn.Module:
    """Reconstruct exact architecture used during training."""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, N_CLASSES)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image_path: str):
    """Apply exact production preprocessing (224x224, ImageNet norm)."""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    # Resize to 224x224 (center crop match of training)
    img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # 1,3,H,W
    return tensor, img, original_size


def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply jet colormap to a [0,1] float grayscale array. Returns uint8 RGB."""
    # Jet: blue→cyan→green→yellow→red
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def generate_gradcam(model: nn.Module, tensor: torch.Tensor, class_idx: int | None = None):
    """
    Compute Grad-CAM for the given input tensor.
    Returns: (cam_normalized np.ndarray H×W float32, predicted_class int, confidence float)
    """
    # Storage for hook outputs
    feature_maps = {}
    gradients = {}

    def forward_hook(module, input, output):
        feature_maps['target'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['target'] = grad_output[0].detach()

    # Hook model.features[-1] — the last convolutional block
    target_layer = model.features[-1]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor.requires_grad_(True)
        logits = model(tensor)  # 1×5

        probs = softmax_np(logits.detach().cpu().numpy()[0])

        if class_idx is None:
            class_idx = int(np.argmax(probs))

        confidence = float(probs[class_idx])

        # Backward on the target class score
        model.zero_grad()
        class_score = logits[0, class_idx]
        class_score.backward()

        # Grad-CAM weights: global average pool of gradients
        grads = gradients['target'][0]     # C×h×w
        feats = feature_maps['target'][0]  # C×h×w

        weights = grads.mean(dim=(1, 2))   # C

        cam = torch.zeros(feats.shape[1:], dtype=torch.float32, device=feats.device)
        for i, w in enumerate(weights):
            cam += w * feats[i]

        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), class_idx, confidence

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def create_overlay(original_img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend Grad-CAM heatmap onto original image."""
    orig_w, orig_h = original_img.size

    # Resize CAM to original image dimensions
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((orig_w, orig_h), Image.LANCZOS)
    cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0

    # Apply jet colormap
    heatmap_rgb = apply_jet_colormap(cam_arr)
    heatmap_pil = Image.fromarray(heatmap_rgb, mode='RGB')

    # Blend
    orig_arr = np.array(original_img.convert('RGB'), dtype=np.float32)
    heat_arr = np.array(heatmap_pil, dtype=np.float32)
    blended = (1 - alpha) * orig_arr + alpha * heat_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM for DR classification')
    parser.add_argument('--image',      required=True, help='Input retinal image path')
    parser.add_argument('--output',     required=True, help='Output overlay image path')
    parser.add_argument('--checkpoint', required=True, help='PyTorch checkpoint path (.pth)')
    parser.add_argument('--class-idx',  type=int, default=None, help='Override predicted class index')
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(json.dumps({"error": f"Image not found: {args.image}"}))
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(json.dumps({"error": f"Checkpoint not found: {args.checkpoint}"}))
        sys.exit(1)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(args.checkpoint)

    # Preprocess image
    tensor, original_img, original_size = preprocess(args.image)

    # Generate Grad-CAM
    cam, predicted_class, confidence = generate_gradcam(model, tensor, args.class_idx)

    # Create and save overlay
    overlay = create_overlay(original_img, cam, alpha=0.5)
    overlay.save(args.output, 'PNG')

    result = {
        "gradcam_path": args.output,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "status": "OK"
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
