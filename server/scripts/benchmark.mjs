import fs from 'fs';
import path from 'path';

const API_BASE = 'http://localhost:5000/api';

async function uploadImage(imagePath) {
  const formData = new FormData();
  const fileBuffer = fs.readFileSync(imagePath);
  const blob = new Blob([fileBuffer], { type: 'image/png' });
  formData.append('image', blob, path.basename(imagePath));
  formData.append('patient_name', 'Bench Test');

  const res = await fetch(`${API_BASE}/screening/upload`, {
    method: 'POST',
    body: formData
  });
  const data = await res.json();
  return data.screeningId;
}

async function processScreening(id) {
  const t0 = Date.now();
  const res = await fetch(`${API_BASE}/screening/${id}/process`, { method: 'POST' });
  const data = await res.json();
  const t1 = Date.now();
  return { data, timeMs: t1 - t0 };
}

async function waitForEvidence(id) {
  const t0 = Date.now();
  while (true) {
    const res = await fetch(`${API_BASE}/screening/${id}/evidence-status`);
    const data = await res.json();
    if (data.gradcam.status === 'READY' && data.lesion.status === 'READY' && data.vessel.status === 'READY') {
      const t1 = Date.now();
      return t1 - t0;
    }
    await new Promise(r => setTimeout(r, 500));
  }
}

async function runBenchmark() {
  const images = [
    '000c1434d8d7.png',
    '001639a390f0.png',
    '0024cdab0c1e.png'
  ];
  
  for (let i = 0; i < images.length; i++) {
    const imgName = images[i];
    const imagePath = path.join(process.cwd(), '..', 'data', 'aptos2019', 'train_images', imgName);
    
    console.log(`\n--- Benchmarking ${imgName} ---`);
    const id = await uploadImage(imagePath);
    console.log(`Uploaded. ID: ${id}`);
    
    const { data, timeMs } = await processScreening(id);
    console.log(`Primary Inference (ONNX): ${timeMs}ms`);
    console.log(`Result: Grade ${data.data.prediction} (Conf: ${(data.data.confidence*100).toFixed(1)}%)`);
    
    const evTime = await waitForEvidence(id);
    console.log(`Total Evidence Generation (Background Worker): ${evTime}ms`);
  }
}

runBenchmark().catch(console.error);
