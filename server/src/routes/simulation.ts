import { Router, RequestHandler } from 'express';

const router = Router();

const runSimulation: RequestHandler = (req, res) => {
  const params = req.body;
  
  // Strict rule: "Do not fabricate simulation outputs. 
  // If MATLAB/Simulink is unavailable, create the model structure and clearly report 
  // that execution requires MATLAB/Simulink."
  
  console.log('Received Simulation Request with params:', params);

  res.json({
    success: false,
    message: 'SIMULATION UNAVAILABLE: Execution requires active MATLAB/Simulink SimEvents environment. Cannot fabricate queueing results.'
  });
};

router.post('/run', runSimulation);

export default router;
