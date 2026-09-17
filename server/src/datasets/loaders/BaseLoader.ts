import fs from 'fs';
import path from 'path';
import db from '../../database/db';

export interface ValidationResult {
  status: 'CONNECTED' | 'PARTIAL' | 'MISSING' | 'INVALID';
  messages: string[];
  imageCount: number;
  labelCount: number;
}

export abstract class BaseLoader {
  public readonly id: string;
  protected name: string;
  protected datasetPath: string;

  constructor(id: string, name: string, dataDir: string) {
    this.id = id;
    this.name = name;
    this.datasetPath = path.join(dataDir, id);
  }

  // Ensure record exists in DB
  protected initializeDBRecord() {
    const stmt = db.prepare(`
      INSERT INTO datasets (id, name, path, status, last_scanned) 
      VALUES (?, ?, ?, 'MISSING', CURRENT_TIMESTAMP)
      ON CONFLICT(id) DO NOTHING
    `);
    stmt.run(this.id, this.name, this.datasetPath);
  }

  protected updateDBStatus(status: string) {
    const stmt = db.prepare(`
      UPDATE datasets SET status = ?, last_scanned = CURRENT_TIMESTAMP WHERE id = ?
    `);
    stmt.run(status, this.id);
  }

  protected clearDBRecords() {
    db.prepare('DELETE FROM dataset_files WHERE dataset_id = ?').run(this.id);
    db.prepare('DELETE FROM dataset_labels WHERE dataset_id = ?').run(this.id);
    db.prepare('DELETE FROM dataset_annotations WHERE dataset_id = ?').run(this.id);
  }

  protected insertDBFile(filename: string, relativePath: string, type: string) {
    db.prepare(`
      INSERT INTO dataset_files (dataset_id, filename, relative_path, type)
      VALUES (?, ?, ?, ?)
    `).run(this.id, filename, relativePath, type);
  }

  protected insertDBLabel(filename: string, drGrade: number) {
    db.prepare(`
      INSERT INTO dataset_labels (dataset_id, filename, dr_grade)
      VALUES (?, ?, ?)
    `).run(this.id, filename, drGrade);
  }

  protected insertDBAnnotation(filename: string, type: string, data: string) {
    db.prepare(`
      INSERT INTO dataset_annotations (dataset_id, filename, annotation_type, data)
      VALUES (?, ?, ?, ?)
    `).run(this.id, filename, type, data);
  }

  protected checkFileExists(relativePath: string): boolean {
    return fs.existsSync(path.join(this.datasetPath, relativePath));
  }

  protected checkDirectoryExists(relativePath: string): boolean {
    const fullPath = path.join(this.datasetPath, relativePath);
    return fs.existsSync(fullPath) && fs.statSync(fullPath).isDirectory();
  }

  protected getImagesInDir(relativePath: string): string[] {
    const fullPath = path.join(this.datasetPath, relativePath);
    if (!fs.existsSync(fullPath)) return [];
    
    return fs.readdirSync(fullPath).filter(file => {
      const ext = path.extname(file).toLowerCase();
      if (!['.png', '.jpg', '.jpeg', '.tif', '.tiff'].includes(ext)) return false;
      const stat = fs.statSync(path.join(fullPath, file));
      return stat.isFile() && stat.size > 0;
    });
  }

  abstract scan(): ValidationResult;
  abstract validate(): ValidationResult;
}
