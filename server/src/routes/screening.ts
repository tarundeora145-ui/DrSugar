import { Router, RequestHandler } from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import db from '../database/db';
import { runInference } from '../ml/inference';

const router = Router();

// ── directories ─────────────────────────────────────────────────────────────
const rootDir = process.cwd();
const isProd = process.env.NODE_ENV === 'production';
const defaultOutputDir = isProd ? path.join(rootDir, 'runtime', 'outputs') : path.join(rootDir, '..', 'data', 'outputs');
const defaultUploadDir = isProd ? path.join(rootDir, 'runtime', 'uploads') : path.join(rootDir, '..', 'data', 'uploads');
const outputDir = process.env.OUTPUT_DIR || defaultOutputDir;
const uploadDir  = process.env.UPLOAD_DIR || defaultUploadDir;
for (const d of [outputDir, uploadDir]) {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
}

// ── Python Worker (Linux compatible) ────────────────────────────────────────
const pythonExe = process.env.PYTHON_BIN || path.join(rootDir, '..', '.venv', 'Scripts', 'python.exe');
const workerScript = process.env.EVIDENCE_WORKER_PATH || path.join(rootDir, '..', 'scripts', 'evidence_worker.py');

// ── multer ───────────────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadDir),
  filename:    (_req,  file, cb) => {
    const suffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, suffix + path.extname(file.originalname));
  },
});

const fileFilter = (_req: any, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  if (['image/jpeg', 'image/png', 'image/jpg'].includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only JPEG and PNG images are accepted.'));
  }
};

const upload = multer({ storage, fileFilter, limits: { fileSize: 10 * 1024 * 1024 } });

// ── helpers ──────────────────────────────────────────────────────────────────

// ── Persistent Python Worker ──────────────────────────────────────────────────
class EvidenceWorker {
  private worker: ChildProcess | null = null;
  private queue: any[] = [];
  private isProcessing = false;

  constructor() {
    this.startWorker();
  }

  private startWorker() {
    console.log('[evidence] Starting persistent Python worker...');
    this.worker = spawn(pythonExe, [workerScript]);
    
    // Using readline to parse stdout JSON lines
    const readline = require('readline');
    const rl = readline.createInterface({ input: this.worker.stdout! });
    
    rl.on('line', (line: string) => {
      try {
        const res = JSON.parse(line);
        if (res.status === 'READY') {
          console.log('[evidence] Worker is ready.');
          this.processNext();
        } else if (res.status === 'OK' || res.status === 'ERROR') {
          if (res.id) this.handleResult(res);
          this.isProcessing = false;
          this.processNext();
        }
      } catch (err) {
        console.error('[evidence] Error parsing worker output:', line);
      }
    });

    this.worker.stderr!.on('data', (data) => {
      console.error(`[evidence stderr] ${data.toString().trim()}`);
    });

    this.worker.on('exit', (code) => {
      console.error(`[evidence] Worker exited with code ${code}. Restarting...`);
      this.isProcessing = false;
      setTimeout(() => this.startWorker(), 2000);
    });
  }

  private handleResult(res: any) {
    if (res.status === 'ERROR') {
      console.error(`[evidence] Error processing id ${res.id}:`, res.message);
      return;
    }
    
    try {
      saveEvidenceField(res.id, 'gradcam', { gradcam_path: res.gradcam_path }, 'gradcam');
      saveEvidenceField(res.id, 'lesion', { lesion_areas_pct: res.lesion_areas_pct }, 'lesion');
      saveEvidenceField(res.id, 'vessel', { vessel_coverage_pct: res.vessel_coverage_pct }, 'vessel');
      
      // Mark as COMPLETED since all three are done by the worker
      db.prepare("UPDATE screenings SET status = 'COMPLETED' WHERE id = ?").run(res.id);
    } catch (e) {
      console.error(`[evidence] Failed to save results for id ${res.id}`, e);
    }
  }

  private processNext() {
    if (this.isProcessing || this.queue.length === 0 || !this.worker) return;
    this.isProcessing = true;
    const req = this.queue.shift();
    this.worker.stdin!.write(JSON.stringify(req) + '\n');
  }

  public enqueue(id: string, imagePath: string, predictedClass: number, outDir: string) {
    this.queue.push({
      id,
      image: imagePath,
      class_idx: predictedClass,
      out_dir: outDir
    });
    this.processNext();
  }
}

const worker = new EvidenceWorker();


/** Persist the result of one evidence script into the DB. */
function saveEvidenceField(
  id: string,
  field: 'gradcam' | 'lesion' | 'vessel',
  result: any,
  evidenceKey: string,
): void {
  if (!result) return;

  const updates: Record<string, string> = {};
  if (field === 'gradcam') {
    updates.gradcam_path = `outputs/${id}/gradcam.png`;
  } else if (field === 'lesion') {
    updates.lesion_path = `outputs/${id}/lesion_overlay.png`;
    updates.lesion_data = JSON.stringify(result.lesion_areas_pct ?? {});
  } else {
    updates.vessel_path = `outputs/${id}/vessel_overlay.png`;
    updates.vessel_data = JSON.stringify({ coverage: result.vessel_coverage_pct ?? 0 });
  }

  // Also update evidence_status JSON
  const existing = db.prepare('SELECT evidence_status FROM screenings WHERE id = ?').get(id) as any;
  let evStatus: Record<string, string> = {};
  try { if (existing?.evidence_status) evStatus = JSON.parse(existing.evidence_status); } catch {}
  evStatus[evidenceKey] = 'READY';

  const setClause = Object.keys(updates)
    .map(k => `${k} = ?`)
    .join(', ');
  const vals = Object.values(updates);
  vals.push(JSON.stringify(evStatus), id);

  db.prepare(`UPDATE screenings SET ${setClause}, evidence_status = ? WHERE id = ?`).run(...vals);
}

/**
 * Push job to persistent evidence worker.
 */
function generateEvidenceInBackground(
  id: string,
  absImagePath: string,
  predictedClass: number,
): void {
  const outDir = path.join(outputDir, id);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const row = db.prepare('SELECT evidence_status FROM screenings WHERE id = ?').get(id) as any;
  let evStatus: any = {};
  try { evStatus = JSON.parse(row?.evidence_status || '{}'); } catch {}

  // Prevent duplicate jobs
  if (evStatus.gradcam === 'PENDING' || evStatus.gradcam === 'READY') {
    return;
  }

  db.prepare('UPDATE screenings SET evidence_status = ? WHERE id = ?')
    .run(JSON.stringify({ gradcam: 'PENDING', lesion: 'PENDING', vessel: 'PENDING' }), id);

  worker.enqueue(id, absImagePath, predictedClass, outDir);
}

// ── route handlers ────────────────────────────────────────────────────────────

const uploadHandler: RequestHandler = (req, res) => {
  if (!req.file) {
    res.status(400).json({ success: false, message: 'No image uploaded' });
    return;
  }

  const { patient_name, patient_age, patient_gender, preferred_language } = req.body;

  const countRow = db.prepare('SELECT COUNT(*) as count FROM screenings').get() as any;
  const count = countRow.count || 0;
  const displayId = `DR-${String(count + 1).padStart(6, '0')}`;

  const info = db.prepare(`
    INSERT INTO screenings (
      display_id, image_path, status, patient_name, patient_age, patient_gender, preferred_language
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(
    displayId,
    req.file.filename,
    'UPLOADED',
    patient_name || null,
    patient_age  || null,
    patient_gender || null,
    preferred_language || 'en',
  );

  res.json({ success: true, screeningId: info.lastInsertRowid });
};

/**
 * POST /:id/process
 *
 * Phase 1 (sync,  ~0.5s): ONNX classification → returns result immediately.
 * Phase 2 (async, background): Grad-CAM + lesion + vessel update DB as they finish.
 */
const processHandler: RequestHandler = async (req, res) => {
  const id = req.params.id as string;

  try {
    const record = db.prepare('SELECT * FROM screenings WHERE id = ?').get(id) as any;
    if (!record) {
      res.status(404).json({ success: false, message: 'Screening not found' });
      return;
    }

    const imagePath = path.join(uploadDir, record.image_path);
    if (!fs.existsSync(imagePath)) {
      res.status(404).json({ success: false, message: 'Image file not found on disk' });
      return;
    }

    // ── Phase 1: ONNX classification ────────────────────────────────────────
    db.prepare("UPDATE screenings SET status = 'PROCESSING' WHERE id = ?").run(id);

    const t0      = Date.now();
    const result  = await runInference(fs.readFileSync(imagePath));
    const classMs = Date.now() - t0;

    const referableDR = result.prediction >= 2 ? 1 : 0;

    // Persist classification result; evidence fields remain NULL until background fills them
    db.prepare(`
      UPDATE screenings
      SET status       = 'CLASSIFIED',
          result_grade = ?,
          probabilities = ?,
          confidence    = ?,
          referable_dr  = ?,
          model_version = 'DR-SUGAR-V1'
      WHERE id = ?
    `).run(
      result.prediction,
      JSON.stringify(result.probabilities),
      result.confidence,
      referableDR,
      id,
    );

    // ── Return primary result immediately ────────────────────────────────────
    res.json({
      success: true,
      status: 'CLASSIFIED',
      classification_ms: classMs,
      data: {
        prediction:   result.prediction,
        probabilities: result.probabilities,
        confidence:   result.confidence,
        referable_dr: referableDR,
      },
    });

    // ── Phase 2: evidence in background (after response already sent) ────────
    setTimeout(() => {
      generateEvidenceInBackground(id, imagePath, result.prediction);
    }, 100);

  } catch (error: any) {
    db.prepare("UPDATE screenings SET status = 'FAILED' WHERE id = ?").run(id);
    // Response may already be sent — guard against double-send
    if (!res.headersSent) {
      res.status(500).json({ success: false, status: 'FAILED', message: error.message });
    }
  }
};

/**
 * GET /:id/evidence-status
 * Lightweight polling endpoint — returns which evidence pieces are ready.
 */
const evidenceStatusHandler: RequestHandler = (req, res) => {
  const id = req.params.id as string;
  const row = db.prepare(
    'SELECT status, evidence_status, gradcam_path, lesion_path, vessel_path, lesion_data, vessel_data FROM screenings WHERE id = ?'
  ).get(id) as any;

  if (!row) {
    res.status(404).json({ success: false, message: 'Screening not found' });
    return;
  }

  let ev: Record<string, string> = {};
  try { if (row.evidence_status) ev = JSON.parse(row.evidence_status); } catch {}

  let lesionData = null;
  let vesselData = null;
  try { if (row.lesion_data) lesionData = JSON.parse(row.lesion_data); } catch {}
  try { if (row.vessel_data) vesselData = JSON.parse(row.vessel_data); } catch {}

  res.json({
    success: true,
    screeningStatus: row.status,
    gradcam:  { status: ev.gradcam ?? 'PENDING', url: row.gradcam_path ? `/${row.gradcam_path}` : null },
    lesion:   { status: ev.lesion  ?? 'PENDING', url: row.lesion_path  ? `/${row.lesion_path}`  : null, data: lesionData },
    vessel:   { status: ev.vessel  ?? 'PENDING', url: row.vessel_path  ? `/${row.vessel_path}`  : null, data: vesselData },
  });
};

const statusHandler: RequestHandler = (req, res) => {
  const id  = req.params.id as string;
  const row = db.prepare(`
    SELECT *,
      created_at || 'Z' as created_at
    FROM screenings WHERE id = ?
  `).get(id) as any;

  if (!row) {
    res.status(404).json({ success: false, message: 'Screening not found' });
    return;
  }

  try { if (row.probabilities)  row.probabilities  = JSON.parse(row.probabilities);  } catch {}
  try { if (row.lesion_data)    row.lesion_data    = JSON.parse(row.lesion_data);    } catch {}
  try { if (row.vessel_data)    row.vessel_data    = JSON.parse(row.vessel_data);    } catch {}
  try { if (row.evidence_status) row.evidence_status = JSON.parse(row.evidence_status); } catch {}

  res.json({ success: true, data: row });
};

const explainabilityHandler: RequestHandler = (req, res) => {
  const id  = req.params.id as string;
  const row = db.prepare(`
    SELECT *,
      created_at || 'Z' as created_at
    FROM screenings WHERE id = ?
  `).get(id) as any;

  if (!row) {
    res.status(404).json({ success: false, message: 'Screening not found' });
    return;
  }

  let lesionData = null, vesselData = null;
  try { if (row.lesion_data) lesionData = JSON.parse(row.lesion_data); } catch {}
  try { if (row.vessel_data) vesselData = JSON.parse(row.vessel_data); } catch {}

  res.json({
    success: true,
    gradCamStatus: row.gradcam_path ? 'AVAILABLE' : 'GRAD-CAM UNAVAILABLE',
    gradCamUrl:    row.gradcam_path ? `/${row.gradcam_path}` : null,
    lesionEvidence: lesionData,
    lesionUrl:     row.lesion_path  ? `/${row.lesion_path}`  : null,
    vesselMask:    vesselData,
    vesselUrl:     row.vessel_path  ? `/${row.vessel_path}`  : null,
    landmarks:     null,
    confidence:    row.confidence   ?? null,
    message:       row.gradcam_path ? 'Explainability data loaded.' : 'No model activations exist.',
  });
};

// ── route registration ────────────────────────────────────────────────────────
router.post('/upload', (req, res, next) => {
  upload.single('image')(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      return res.status(400).json({ success: false, message: `Upload error: ${err.message}` });
    } else if (err) {
      return res.status(400).json({ success: false, message: err.message });
    }
    return uploadHandler(req, res, next);
  });
});
router.post('/:id/process',           processHandler);
router.get('/:id/evidence-status',    evidenceStatusHandler);
router.get('/:id/explainability',     explainabilityHandler);
router.get('/:id',                    statusHandler);

export default router;
