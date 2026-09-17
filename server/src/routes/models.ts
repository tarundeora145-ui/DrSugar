import { Router, Request, Response, RequestHandler } from 'express';

const router = Router();

const statusHandler: RequestHandler = (req, res) => {
  res.json({
    status: 'MODEL UNAVAILABLE',
    message: 'No trained model weights are currently connected. AI prediction is offline.'
  });
};

router.get('/status', statusHandler);

export default router;
