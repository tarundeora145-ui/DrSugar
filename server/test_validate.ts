import path from 'path';
import { AptosLoader } from './src/datasets/loaders/AptosLoader';
import db from './src/database/db';
import fs from 'fs';

const DATA_DIR = path.resolve(__dirname, '../data');
const loader = new AptosLoader(DATA_DIR);
const result = loader.validate();

console.log('--- APTOS 2019 VALIDATE RESULT ---');
console.log(JSON.stringify(result, null, 2));

const dbRecord = db.prepare('SELECT * FROM datasets WHERE id = ?').get('aptos2019');
console.log('--- SQLITE RECORD ---');
console.log(JSON.stringify(dbRecord, null, 2));

// check for unmatched images/labels
const unmatchedLabels = db.prepare(`
  SELECT l.filename 
  FROM dataset_labels l 
  LEFT JOIN dataset_files f ON l.filename = f.filename AND l.dataset_id = f.dataset_id 
  WHERE f.id IS NULL AND l.dataset_id = 'aptos2019'
`).all() as any[];

const unmatchedImages = db.prepare(`
  SELECT f.filename 
  FROM dataset_files f 
  LEFT JOIN dataset_labels l ON f.filename = l.filename AND f.dataset_id = l.dataset_id 
  WHERE l.id IS NULL AND f.dataset_id = 'aptos2019'
`).all() as any[];

console.log('--- MISMATCH REPORT ---');
console.log(`Labels without images: \${unmatchedLabels.length}`);
console.log(`Images without labels: \${unmatchedImages.length}`);

// check for corrupt files by trying to read headers or just file sizes
const trainImagesDir = path.join(DATA_DIR, 'aptos2019', 'train_images');
let corruptCount = 0;
if (fs.existsSync(trainImagesDir)) {
  const files = fs.readdirSync(trainImagesDir);
  for (const f of files) {
    try {
      const stats = fs.statSync(path.join(trainImagesDir, f));
      if (stats.size === 0) corruptCount++;
    } catch (e) {
      corruptCount++;
    }
  }
}
console.log(`Corrupt (0 byte) images: \${corruptCount}`);

