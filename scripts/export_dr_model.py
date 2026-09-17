import os
import sys
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import onnx
import onnxruntime
import numpy as np
import logging

# Force UTF-8 stdout to prevent UnicodeEncodeError on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO)

def export_model():
    model_path = Path('models/dr_classification/best_model.pth')
    onnx_path = 'models/dr_classification/model.onnx'
    
    if not model_path.exists():
        logging.error('Model weights not found. Train the model first.')
        return
        
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Use legacy trace-based export (dynamo=False) — stable, no onnxscript dependency.
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        dynamo=False,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    logging.info('Model exported to %s', onnx_path)
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logging.info('ONNX model checker passed.')
    
    # Compare PyTorch vs ONNX outputs on the same dummy input
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
        
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_out = ort_outs[0]
    
    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-04)
    logging.info('PyTorch vs ONNX max diff: %.8f — PASS', max_diff)

if __name__ == '__main__':
    export_model()
