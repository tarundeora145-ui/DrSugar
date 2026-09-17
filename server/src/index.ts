import express, { Application } from 'express';
import cors from 'cors';
import { config } from './config/env';
import healthRouter from './routes/health';
import datasetsRouter from './routes/datasets';
import modelsRouter from './routes/models';
import screeningRouter from './routes/screening';
import validationRouter from './routes/validation';
import simulationRouter from './routes/simulation';
import { errorHandler } from './middleware/errorHandler';
import './database/db'; // Initialize database

const app: Application = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/health', healthRouter);
app.use('/api/datasets', datasetsRouter);
app.use('/api/models', modelsRouter);
app.use('/api/screening', screeningRouter);
app.use('/api/validation', validationRouter);
app.use('/api/simulation', simulationRouter);

// Global Error Handler
app.use(errorHandler);

app.listen(config.port, () => {
  console.log(`Server running on port ${config.port} in ${config.nodeEnv} mode.`);
});
