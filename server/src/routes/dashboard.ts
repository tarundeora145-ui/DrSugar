import { Router, RequestHandler } from 'express';
import db from '../database/db';

const router = Router();

const getDashboardMetrics: RequestHandler = (req, res) => {
  try {
    const totalRow = db.prepare('SELECT COUNT(*) as total FROM screenings').get() as any;
    const pendingRow = db.prepare('SELECT COUNT(*) as pending FROM screenings WHERE referable_dr = 1').get() as any;
    
    const recentScreenings = db.prepare(`
      SELECT 
        id, 
        display_id,
        patient_name,
        result_grade,
        referable_dr,
        status,
        created_at || 'Z' as created_at
      FROM screenings
      ORDER BY created_at DESC
      LIMIT 5
    `).all();

    res.json({
      success: true,
      data: {
        totalScreenings: totalRow.total || 0,
        pendingReview: pendingRow.pending || 0,
        systemStatus: 'ONLINE',
        recentScreenings
      }
    });
  } catch (err: any) {
    res.status(500).json({ success: false, message: err.message });
  }
};

router.get('/', getDashboardMetrics);

export default router;
