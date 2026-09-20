import { Router, RequestHandler } from 'express';

const router = Router();

const runSimulation: RequestHandler = (req, res) => {
  const params = req.body;
  
  // Deterministic Prototype Simulation Math
  const { patientsPerYear, imagesPerPatient, imageSizeMB, bandwidthMbps, processingTimeS, reviewCapacity } = params;
  
  const totalImages = patientsPerYear * imagesPerPatient;
  const totalDataMB = totalImages * imageSizeMB;
  const networkTimeS = (totalDataMB * 8) / bandwidthMbps;
  
  const aiCapacityImages = (365 * 24 * 60 * 60) / processingTimeS; 
  
  res.json({
    success: true,
    data: {
      totalImages,
      totalDataTB: (totalDataMB / 1024 / 1024).toFixed(2),
      networkDays: (networkTimeS / (3600 * 24)).toFixed(1),
      aiCapacityLimit: aiCapacityImages > totalImages ? 'OK' : 'BOTTLENECK',
      clinicalQueue: totalImages > reviewCapacity ? 'OVERLOADED' : 'SUSTAINABLE'
    }
  });
};

router.post('/run', runSimulation);

export default router;
