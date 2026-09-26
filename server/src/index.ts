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
app.use(cors());
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

// Static routes
app.use('/outputs', express.static(path.join(process.cwd(), '..', 'data', 'outputs')));
app.use('/uploads', express.static(path.join(process.cwd(), '..', 'data', 'uploads')));

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
