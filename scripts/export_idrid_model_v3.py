"""
DR-SUGAR — IDRiD V3 ONNX Export & Validation
=============================================
Converts the best V3 MobileNetV3-Small model to ONNX.
Validates numerical agreement between PyTorch and ONNX Runtime.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import onnx
import onnxruntime as ort

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(levelname)s:root:%(message)s')

NUM_CLASSES = 5
MODEL_DIR = Path('models/idrid/v3')
PTH_PATH = MODEL_DIR / 'best_model.pth'
ONNX_PATH = MODEL_DIR / 'model.onnx'

# IDRiD test image for validation
TEST_IMG_DIR = Path('data/idrid/B. Disease Grading/1. Original Images/b. Testing Set')
TEST_IMG_NAME = 'IDRiD_001.jpg'


def export_and_validate():
    if not PTH_PATH.exists():
        logging.error(f"PyTorch checkpoint not found: {PTH_PATH}")
        return

    # 1. Load PyTorch model
    device = torch.device('cpu')  # Export on CPU for portability
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(PTH_PATH, map_location=device))
    model.eval()

    # 2. Export to ONNX
    logging.info("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 3. Verify ONNX structure
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    logging.info(f"ONNX model saved and verified: {ONNX_PATH}")

    # 4. Numerical validation (PyTorch vs ONNX)
    test_img_path = TEST_IMG_DIR / TEST_IMG_NAME
    if not test_img_path.exists():
        logging.warning(f"Test image {test_img_path} not found. Skipping numerical validation.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(test_img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    logging.info(f"\n--- Validation with {TEST_IMG_NAME} ---")
    
    # PyTorch inference
    with torch.no_grad():
        pt_outputs = model(input_tensor)
        pt_probs = torch.softmax(pt_outputs, dim=1).numpy()[0]
    
    # ONNX inference
    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Calculate Softmax for ONNX outputs
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    ort_probs = softmax(ort_outputs[0])[0]
    
    max_diff = np.max(np.abs(pt_probs - ort_probs))
    
    logging.info(f"PyTorch Probabilities: {pt_probs}")
    logging.info(f"ONNX Probabilities:    {ort_probs}")
    logging.info(f"Maximum difference:    {max_diff:.8e}")
    
    if max_diff < 1e-5:
        logging.info("Validation PASSED: ONNX outputs match PyTorch outputs.")
    else:
        logging.error("Validation FAILED: Discrepancy between ONNX and PyTorch.")

if __name__ == '__main__':
    export_and_validate()
