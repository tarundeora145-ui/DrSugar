// API consistency test — runs same image 10 times via the real backend API
// and checks that results are identical each time.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const API_BASE = 'http://localhost:5000/api';
const ROOT = path.join(__dirname, '..', '..');

const IMAGES = [
  '000c1434d8d7.png',
  '0024cdab0c1e.png',
];

async function uploadAndProcess(imgName) {
  const imgPath = path.join(ROOT, 'data', 'aptos2019', 'train_images', imgName);
  const buf = fs.readFileSync(imgPath);
  const blob = new Blob([buf], { type: 'image/png' });
  const form = new FormData();
  form.append('image', blob, imgName);
  form.append('patient_name', 'Test');

  const upRes = await fetch(`${API_BASE}/screening/upload`, { method: 'POST', body: form });
  const upData = await upRes.json();
  if (!upData.success) throw new Error(`Upload failed: ${JSON.stringify(upData)}`);

  const id = upData.screeningId;
  const t0 = Date.now();
  const procRes = await fetch(`${API_BASE}/screening/${id}/process`, { method: 'POST' });
  const procData = await procRes.json();
  const latency = Date.now() - t0;

  if (!procData.success) throw new Error(`Process failed: ${JSON.stringify(procData)}`);
  return { id, latency, ...procData.data };
}

async function main() {
  console.log('DR-SUGAR API CONSISTENCY TEST');
  console.log('='.repeat(60));

  for (const img of IMAGES) {
    console.log(`\nImage: ${img} — 10 repeated API calls`);
    console.log('-'.repeat(60));

    const results = [];
    for (let i = 1; i <= 10; i++) {
      try {
        const r = await uploadAndProcess(img);
        results.push(r);
        const probs = r.probabilities.map(p => p.toFixed(4)).join(', ');
        console.log(`  Run ${String(i).padStart(2)}: Grade ${r.prediction}  conf=${r.confidence.toFixed(4)}  [${probs}]  ${r.latency}ms`);
      } catch (e) {
        console.log(`  Run ${i}: ERROR — ${e.message}`);
      }
    }

    const grades = results.map(r => r.prediction);
    const unique = [...new Set(grades)];
    const confs = results.map(r => r.confidence);
    const confRange = Math.max(...confs) - Math.min(...confs);

    console.log(`  Unique grades: [${unique.join(', ')}]  → ${unique.length === 1 ? 'DETERMINISTIC ✓' : 'NON-DETERMINISTIC ✗'}`);
    console.log(`  Confidence range: ${confRange.toFixed(6)}  → ${confRange < 1e-4 ? 'STABLE ✓' : 'UNSTABLE ✗'}`);
    console.log(`  Avg latency: ${Math.round(results.reduce((s, r) => s + r.latency, 0) / results.length)}ms`);
  }
}

main().catch(console.error);
