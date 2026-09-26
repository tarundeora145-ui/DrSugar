import { Request, Response, NextFunction } from 'express';

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  console.error(`[Error] ${req.method} ${req.path} — ${err.message}`);
  
  res.status(500).json({
    success: false,
    message: 'Internal Server Error',
  });
};
