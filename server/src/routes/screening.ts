import { Router, Request, Response, RequestHandler } from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import db from '../database/db';

const router = Router();

// Ensure upload directory exists
const uploadDir = path.join(process.cwd(), '..', 'data', 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Multer config
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

const uploadHandler: RequestHandler = (req, res) => {
  if (!req.file) {
    res.status(400).json({ success: false, message: 'No image uploaded' });
    return;
  }

  const stmt = db.prepare(`
    INSERT INTO screenings (image_path, status)
    VALUES (?, ?)
  `);
  const info = stmt.run(req.file.filename, 'UPLOADED');

  res.json({ success: true, screeningId: info.lastInsertRowid });
};

const processHandler: RequestHandler = (req, res) => {
  const { id } = req.params;
  
  // Here we pretend to start processing, but immediately fail since no model is present.
  const stmt = db.prepare(`
    UPDATE screenings 
    SET status = ? 
    WHERE id = ?
  `);
  stmt.run('MODEL UNAVAILABLE', id);

  res.json({ 
    success: false, 
    status: 'MODEL UNAVAILABLE',
    message: 'No trained model weights are currently connected. AI prediction is offline.'
  });
};

const statusHandler: RequestHandler = (req, res) => {
  const { id } = req.params;
  const stmt = db.prepare('SELECT * FROM screenings WHERE id = ?');
  const record = stmt.get(id);

  if (!record) {
    res.status(404).json({ success: false, message: 'Screening not found' });
    return;
  }

  res.json({ success: true, data: record });
};

const explainabilityHandler: RequestHandler = (req, res) => {
  // We check for the model's output in the database for the given screening ID.
  // Since no model exists, this endpoint explicitly refuses to return fabricated heatmaps.
  res.json({
    success: false,
    gradCamStatus: 'GRAD-CAM UNAVAILABLE',
    lesionEvidence: null,
    vesselMask: null,
    landmarks: null,
    confidence: null,
    message: 'No model activations exist. Cannot generate Explainability data.'
  });
};

router.post('/upload', upload.single('image'), uploadHandler);
router.post('/:id/process', processHandler);
router.get('/:id', statusHandler);
router.get('/:id/explainability', explainabilityHandler);

export default router;
