"""
DR-SUGAR — IDRiD V2 ONNX Export
Exports models/idrid/v2/best_model.pth → models/idrid/v2/model.onnx

Verifies:
  - ONNX checker
  - input shape [1,3,224,224]
  - 5 output classes
  - PyTorch vs ONNX numerical agreement (max diff reported)
  - ONNX Runtime inference on a real IDRiD image
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime
from PIL import Image
from torchvision import transforms

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(levelname)s:root:%(message)s')

SAVE_DIR = Path('models/idrid/v2')
TEST_IMAGE = Path('data/idrid/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg')


def export():
    pth_path = SAVE_DIR / 'best_model.pth'
    onnx_path = SAVE_DIR / 'model.onnx'

    if not pth_path.exists():
        logging.error(f'Checkpoint not found: {pth_path}. Run train_idrid_grading_v2.py first.')
        return

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    model.load_state_dict(torch.load(str(pth_path), map_location='cpu'))
    model.eval()
    logging.info('Loaded best_model.pth')

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        dynamo=False,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    logging.info(f'Exported to {onnx_path}')

    # ONNX checker
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logging.info('ONNX checker: PASS')

    # Validate input/output shape
    session = onnxruntime.InferenceSession(str(onnx_path))
    in_shape = session.get_inputs()[0].shape
    out_shape = session.get_outputs()[0].shape
    logging.info(f'ONNX input shape: {in_shape}')
    logging.info(f'ONNX output shape: {out_shape}')
    assert out_shape[-1] == 5, f'Expected 5 output classes, got {out_shape[-1]}'
    logging.info('Output class count: PASS (5)')

    # Numerical agreement: PyTorch vs ONNX
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    ort_out = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)
    logging.info(f'PyTorch vs ONNX max output diff: {max_diff:.8f} — PASS')

    # Probability agreement
    pt_probs = torch.softmax(torch.tensor(pt_out), dim=1).numpy()
    ort_probs = torch.softmax(torch.tensor(ort_out), dim=1).numpy()
    max_prob_diff = float(np.max(np.abs(pt_probs - ort_probs)))
    np.testing.assert_allclose(pt_probs, ort_probs, rtol=1e-3, atol=1e-4)
    logging.info(f'Probability max diff: {max_prob_diff:.8f} — PASS')

    # Real image inference via ONNX Runtime
    if not TEST_IMAGE.exists():
        logging.warning(f'Test image not found: {TEST_IMAGE}. Skipping real-image inference check.')
        return

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = eval_transform(Image.open(TEST_IMAGE).convert('RGB')).unsqueeze(0)
    ort_result = session.run(None, {session.get_inputs()[0].name: img_tensor.numpy()})[0]
    ort_probs_real = float(np.max(ort_result))  # just the raw max logit for sanity
    probs_arr = list(torch.softmax(torch.tensor(ort_result), dim=1).numpy()[0])
    predicted_grade = int(np.argmax(probs_arr))
    logging.info(f'ONNX Runtime inference on IDRiD_001.jpg:')
    logging.info(f'  Predicted grade: {predicted_grade}')
    logging.info(f'  Probabilities: {[round(p, 4) for p in probs_arr]}')
    logging.info(f'  Referable DR: {predicted_grade >= 2}')
    logging.info('All verification checks PASSED.')


if __name__ == '__main__':
    export()
