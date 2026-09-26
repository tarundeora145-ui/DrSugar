import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

// Point to the root database directory
const dbDir = path.resolve(__dirname, '../../../database');
if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

const dbPath = path.join(dbDir, 'dr_sugar.db');

// Initialize database
const db = new Database(dbPath);

// Enable WAL mode for better performance
db.pragma('journal_mode = WAL');

// Initial setup - Full Schema
db.exec(`
  CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('CONNECTED', 'PARTIAL', 'MISSING', 'INVALID')),
    last_scanned DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS dataset_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS dataset_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    dr_grade INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS dataset_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    annotation_type TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
  );

    CREATE TABLE IF NOT EXISTS screenings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      display_id TEXT,
      patient_id TEXT,
      patient_name TEXT,
      patient_age INTEGER,
      patient_gender TEXT,
      preferred_language TEXT DEFAULT 'en',
      image_path TEXT NOT NULL,
      status TEXT NOT NULL,
      result_grade INTEGER,
      probabilities TEXT,
      confidence REAL,
      referable_dr INTEGER,
      model_version TEXT,
      gradcam_path TEXT,
      lesion_path TEXT,
      lesion_data TEXT,
      vessel_path TEXT,
      vessel_data TEXT,
      evidence_status TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

  CREATE TABLE IF NOT EXISTS screening_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id INTEGER NOT NULL,
    focus REAL,
    sharpness REAL,
    illumination REAL,
    contrast REAL,
    exposure REAL,
    fov REAL,
    gradeability REAL,
    status TEXT CHECK(status IN ('GOOD', 'BORDERLINE', 'UNGRADABLE')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(screening_id) REFERENCES screenings(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS screening_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id INTEGER NOT NULL,
    optic_disc_detected BOOLEAN,
    fovea_detected BOOLEAN,
    vessel_density REAL,
    ma_count INTEGER,
    exudate_area REAL,
    hemorrhage_area REAL,
    neo_detected BOOLEAN,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(screening_id) REFERENCES screenings(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id INTEGER NOT NULL,
    report_data TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(screening_id) REFERENCES screenings(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    results TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
  );
`);

try {
  db.exec('ALTER TABLE screenings ADD COLUMN evidence_status TEXT');
} catch (err) {
  // Column already exists
}

try {
  db.exec('ALTER TABLE screenings ADD COLUMN display_id TEXT');
} catch (err) {
  // Column already exists
}

export default db;
