import { Router, RequestHandler } from 'express';
import db from '../database/db';

const router = Router();

const getReportsHandler: RequestHandler = (req, res) => {
  try {
    const stmt = db.prepare(`
      SELECT 
        id, 
        display_id,
        patient_name, 
        patient_id, 
        result_grade, 
        referable_dr, 
        status, 
        created_at || 'Z' as created_at
      FROM screenings 
      WHERE status = 'COMPLETED' 
      ORDER BY created_at DESC
    `);
    const records = stmt.all();

    res.json({ success: true, data: records });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};

router.get('/', getReportsHandler);

export default router;
