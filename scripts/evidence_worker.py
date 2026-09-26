#!/usr/bin/env python3
import sys
import json
import torch
import traceback
from pathlib import Path
import numpy as np
from PIL import Image

# Add scripts directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

import generate_gradcam
import run_lesion_inference
import run_vessel_inference

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
    eprint("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import os
    models_dir = os.getenv("MODELS_DIR", str(Path(__file__).parent.parent / "models"))
    
    gradcam_model = generate_gradcam.build_model(str(Path(models_dir) / "dr_classification" / "best_model.pth")).to(device)
    lesion_model = run_lesion_inference.load_model(str(Path(models_dir) / "idrid" / "lesions" / "v3" / "fold1_best.pth")).to(device)
    vessel_model = run_vessel_inference.load_model(str(Path(models_dir) / "drive" / "v1" / "fold1_best.pth")).to(device)
    
    eprint("Models loaded. Ready for IPC.")
    
    # Send ready signal
    print(json.dumps({"status": "READY"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            image_path = req['image']
            out_dir = req['out_dir']
            class_idx = req['class_idx']
            id = req['id']
            
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # --- GradCAM ---
            tensor_g, original_img_g, _ = generate_gradcam.preprocess(image_path)
            tensor_g = tensor_g.to(device)
            cam, _, confidence = generate_gradcam.generate_gradcam(gradcam_model, tensor_g, class_idx)
            overlay_g = generate_gradcam.create_overlay(original_img_g, cam, alpha=0.5)
            gradcam_out = out_path / "gradcam.png"
            overlay_g.save(gradcam_out, 'PNG')
            
            # --- Lesion ---
            img_rgb_l = np.array(Image.open(image_path).convert('RGB'))
            H_l, W_l = img_rgb_l.shape[:2]
            prob_map_l = run_lesion_inference.run_patch_inference(lesion_model, img_rgb_l)
            
            lesion_areas_pct = {}
            total_pixels_l = H_l * W_l
            for i, name in enumerate(run_lesion_inference.LESION_NAMES):
                resized_pil = Image.fromarray((prob_map_l[i] * 255).astype(np.uint8))
                resized_pil = resized_pil.resize((W_l, H_l), Image.BILINEAR)
                resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0
                mask_l = resized_arr >= run_lesion_inference.THRESHOLD
                pct = float(mask_l.sum() / total_pixels_l * 100)
                lesion_areas_pct[name] = round(pct, 4)
                
            overlay_l = run_lesion_inference.create_lesion_overlay(Image.open(image_path).convert('RGB'), prob_map_l)
            lesion_out = out_path / "lesion_overlay.png"
            overlay_l.save(lesion_out, 'PNG')
            
            # --- Vessel ---
            img_rgb_v = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
            orig_h_v, orig_w_v = img_rgb_v.shape[:2]
            cropped_v, (top_v, left_v, _, _) = run_vessel_inference.center_crop_or_pad(img_rgb_v, run_vessel_inference.IMG_SIZE)
            tensor_v = run_vessel_inference.normalize_img(cropped_v).to(device)
            with torch.no_grad():
                logits_v = vessel_model(tensor_v)
                probs_v  = torch.sigmoid(logits_v)[0, 0].cpu().numpy()
            
            vessel_mask_crop = probs_v >= run_vessel_inference.THRESHOLD
            vessel_mask_full = np.zeros((orig_h_v, orig_w_v), dtype=bool)
            crop_h = min(run_vessel_inference.IMG_SIZE, orig_h_v)
            crop_w = min(run_vessel_inference.IMG_SIZE, orig_w_v)
            dst_y0 = top_v if top_v >= 0 else 0
            src_y0 = 0 if top_v >= 0 else -top_v
            dst_x0 = left_v if left_v >= 0 else 0
            src_x0 = 0 if left_v >= 0 else -left_v
            h_copy = min(run_vessel_inference.IMG_SIZE - src_y0, orig_h_v - dst_y0)
            w_copy = min(run_vessel_inference.IMG_SIZE - src_x0, orig_w_v - dst_x0)
            if h_copy > 0 and w_copy > 0:
                vessel_mask_full[dst_y0:dst_y0+h_copy, dst_x0:dst_x0+w_copy] = \
                    vessel_mask_crop[src_y0:src_y0+h_copy, src_x0:src_x0+w_copy]
            
            total_pixels = orig_h_v * orig_w_v
            coverage_pct = float(vessel_mask_full.sum() / total_pixels * 100)
            overlay_v = run_vessel_inference.create_vessel_overlay(Image.open(image_path).convert('RGB'), vessel_mask_full)
            vessel_out = out_path / "vessel_overlay.png"
            overlay_v.save(vessel_out, 'PNG')
            
            # Send result
            result = {
                "id": id,
                "status": "OK",
                "gradcam_path": str(gradcam_out.as_posix()),
                "confidence": float(confidence),
                "lesion_path": str(lesion_out.as_posix()),
                "lesion_areas_pct": lesion_areas_pct,
                "vessel_path": str(vessel_out.as_posix()),
                "vessel_coverage_pct": float(coverage_pct)
            }
            print(json.dumps(result), flush=True)
            
        except Exception as e:
            eprint(f"Error processing {line}: {traceback.format_exc()}")
            print(json.dumps({"id": req.get("id"), "status": "ERROR", "message": str(e)}), flush=True)

if __name__ == '__main__':
    main()
