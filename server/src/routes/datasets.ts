import { Router, Request, Response, RequestHandler } from 'express';
import fs from 'fs';
import path from 'path';
import db from '../database/db';
import { AptosLoader } from '../datasets/loaders/AptosLoader';
import { IdridLoader } from '../datasets/loaders/IdridLoader';
import { DriveLoader } from '../datasets/loaders/DriveLoader';
import { MessidorLoader } from '../datasets/loaders/MessidorLoader';

const router = Router();

const DATA_DIR = path.resolve(__dirname, '../../../data');

const getLoaders = () => [
  new AptosLoader(DATA_DIR),
  new IdridLoader(DATA_DIR),
  new DriveLoader(DATA_DIR),
  new MessidorLoader(DATA_DIR)
];

const scanDatasetsHandler: RequestHandler = (req, res) => {
  try {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }

    const loaders = getLoaders();
    const results = loaders.map(loader => loader.scan());
    
    // Also fetch updated statuses from DB to return
    const stmt = db.prepare('SELECT * FROM datasets');
    const rows = stmt.all();

    res.json({ success: true, scanned: rows, validation: results });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to scan datasets' });
  }
};

const getDatasetsHandler: RequestHandler = (req, res) => {
  const stmt = db.prepare('SELECT * FROM datasets');
  const rows = stmt.all();
  res.json(rows);
};

const validateDatasetHandler: RequestHandler = (req, res) => {
  const id = req.params.id as string;
  const loaders = getLoaders();
  const loader = loaders.find(l => l.id === id); // id is protected, but we can check if it matches the class initialization, or just create it directly

  if (!loader) {
    res.status(404).json({ error: 'Dataset not found or unsupported' });
    return;
  }

  try {
    const result = loader.validate();
    res.json({ success: true, datasetId: id, result });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
};

const getDatasetImagesHandler: RequestHandler = (req, res) => {
  const id = req.params.id as string;
  const files = db.prepare("SELECT * FROM dataset_files WHERE dataset_id = ? AND type = 'fundus'").all(id) as any[];
  const labels = db.prepare('SELECT * FROM dataset_labels WHERE dataset_id = ?').all(id) as any[];
  const annotations = db.prepare('SELECT * FROM dataset_annotations WHERE dataset_id = ?').all(id) as any[];

  // Join them in memory
  const results = files.map(file => {
    const fileLabels = labels.filter(l => l.filename === file.filename);
    const drGrade = fileLabels.length > 0 ? fileLabels[0].dr_grade : null;
    const fileAnnotations = annotations.filter(a => a.filename === file.filename).map(a => ({
      type: a.annotation_type,
      data: JSON.parse(a.data)
    }));
    
    return {
      ...file,
      dr_grade: drGrade,
      annotations: fileAnnotations
    };
  });

  res.json(results);
};

const streamImageHandler: RequestHandler = (req, res) => {
  const id = req.params.id as string;
  const imageName = req.params.imageName as string;
  
  const targetPath = path.join(DATA_DIR, id, imageName);
  const absoluteTargetPath = path.resolve(targetPath);

  const safeDatasetDir = path.resolve(path.join(DATA_DIR, id));
  if (!absoluteTargetPath.startsWith(safeDatasetDir + path.sep) && absoluteTargetPath !== safeDatasetDir) {
    res.status(403).send('Forbidden: Invalid path requested');
    return;
  }

  if (!fs.existsSync(absoluteTargetPath)) {
    res.status(404).send('Image not found');
    return;
  }

  const stream = fs.createReadStream(absoluteTargetPath);
  stream.on('error', () => res.status(500).send('Error reading file'));
  stream.pipe(res);
};

router.get('/', getDatasetsHandler);
router.get('/scan', scanDatasetsHandler);
router.post('/refresh', scanDatasetsHandler);
router.post('/:id/validate', validateDatasetHandler);
router.get('/:id/images', getDatasetImagesHandler);
router.get('/:id/image/:imageName', streamImageHandler);

export default router;
