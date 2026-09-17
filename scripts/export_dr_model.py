import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def export_model():
    model_path = 'models/dr_classification/best_model.pth'
    onnx_path = 'models/dr_classification/model.onnx'
    
    if not Path(model_path).exists():
        logging.error("Model weights not found. Train the model first.")
        return
        
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Dummy input matching [1, 3, 224, 224]
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    logging.info(f"Model exported to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model checker passed.")
    
    # Compare outputs
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
        
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_out = ort_outs[0]
    
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    logging.info(f"Output comparison passed! Max diff: {np.max(np.abs(torch_out - ort_out)):.6f}")

if __name__ == "__main__":
    from pathlib import Path
    export_model()
