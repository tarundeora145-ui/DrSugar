import { Router, RequestHandler } from 'express';
import db from '../database/db';

const router = Router();

const getLatestValidation: RequestHandler = (req, res) => {
  const { datasetId } = req.params;
  
  const stmt = db.prepare('SELECT * FROM validation_runs WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1');
  const record = stmt.get(datasetId);

  if (!record) {
    res.json({
      success: true,
      data: {
        status: 'NOT EVALUATED',
        model_version: 'NONE',
        results: null
      }
    });
    return;
  }

  res.json({ success: true, data: record });
};

const runValidation: RequestHandler = (req, res) => {
  const { datasetId } = req.params;

  // We explicitly reject fabricating a validation run since there is no model.
  // We log the attempt as 'NOT EVALUATED' to preserve truthfulness.
  const stmt = db.prepare(`
    INSERT INTO validation_runs (dataset_id, model_version, status, results)
    VALUES (?, ?, ?, ?)
  `);

  const emptyResults = JSON.stringify({
    accuracy: null,
    sensitivity: null,
    specificity: null,
    precision: null,
    recall: null,
    f1: null,
    auroc: null,
    confusion_matrix: {
      tp: null, tn: null, fp: null, fn: null
    }
  });

  const info = stmt.run(datasetId, 'NONE', 'NOT EVALUATED', emptyResults);

  res.json({
    success: false,
    message: 'MODEL UNAVAILABLE: Cannot perform clinical validation. Run recorded as NOT EVALUATED.',
    runId: info.lastInsertRowid
  });
};

router.get('/:datasetId/latest', getLatestValidation);
router.post('/:datasetId/run', runValidation);

export default router;
