import express, { Application } from 'express';
import cors from 'cors';
import { config } from './config/env';
import healthRouter from './routes/health';
import datasetsRouter from './routes/datasets';
import screeningRouter from './routes/screening';
import validationRouter from './routes/validation';
import simulationRouter from './routes/simulation';
import reportsRouter from './routes/reports';
import dashboardRouter from './routes/dashboard';
import { errorHandler } from './middleware/errorHandler';
import path from 'path';
import './database/db'; // Initialize database
import { loadModel } from './ml/inference';

const app: Application = express();

// Middleware
app.use(cors({
  origin: process.env.FRONTEND_URL || '*'
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/health', healthRouter);
app.use('/api/datasets', datasetsRouter);
app.use('/api/screening', screeningRouter);
app.use('/api/validation', validationRouter);
app.use('/api/simulation', simulationRouter);
app.use('/api/reports', reportsRouter);
app.use('/api/dashboard', dashboardRouter);

const isProd = process.env.NODE_ENV === 'production';
const defaultOutputDir = isProd ? path.join(process.cwd(), 'runtime', 'outputs') : path.join(process.cwd(), '..', 'data', 'outputs');
const defaultUploadDir = isProd ? path.join(process.cwd(), 'runtime', 'uploads') : path.join(process.cwd(), '..', 'data', 'uploads');
const outputDir = process.env.OUTPUT_DIR || defaultOutputDir;
const uploadDir = process.env.UPLOAD_DIR || defaultUploadDir;
app.use('/outputs', express.static(outputDir));
app.use('/uploads', express.static(uploadDir));

// Global Error Handler
app.use(errorHandler);

app.listen(config.port, () => {
  console.log(`Server running on port ${config.port} in ${config.nodeEnv} mode.`);
});

loadModel()
  .then(() => {
    console.log('ONNX classification model loaded.');
  })
  .catch((err) => {
    console.warn('ONNX model unavailable — screening inference disabled:', err.message);
  });
