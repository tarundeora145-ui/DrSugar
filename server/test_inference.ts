import fs from 'fs';
import path from 'path';
import { runInference } from './src/ml/inference';
import db from './src/database/db';

async function testInference() {
  const imagePath = path.resolve(__dirname, '../data/aptos2019/test_images/0005cfc8afb6.png');
  
  if (!fs.existsSync(imagePath)) {
    console.error(`Test image not found: ${imagePath}`);
    return;
  }
  
  const buffer = fs.readFileSync(imagePath);
  console.log(`Loaded test image (size: ${buffer.length} bytes). Running inference...`);
  
  try {
    const result = await runInference(buffer);
    console.log('--- INFERENCE RESULT ---');
    console.log(JSON.stringify(result, null, 2));
    
    // Verify probabilities sum to ~1.0
    const probSum = result.probabilities.reduce((a, b) => a + b, 0);
    console.log(`Probabilities sum: ${probSum.toFixed(6)} (should be ~1.0)`);
    console.log(`Predicted class: ${result.prediction} (valid: ${result.prediction >= 0 && result.prediction <= 4})`);
    console.log(`referableDR: ${result.referableDR} (should be ${result.prediction >= 2})`);
    
    // Persist to SQLite
    const insertStmt = db.prepare(`
      INSERT INTO screenings (patient_id, image_path, status, result_grade)
      VALUES (?, ?, ?, ?)
    `);
    const info = insertStmt.run('test-patient-node-inference', imagePath, 'COMPLETED', result.prediction);
    
    console.log(`\nInserted screening into SQLite (ID: ${info.lastInsertRowid})`);
    
    const dbRecord = db.prepare('SELECT * FROM screenings WHERE id = ?').get(info.lastInsertRowid);
    console.log('--- SQLITE RECORD ---');
    console.log(JSON.stringify(dbRecord, null, 2));
    
    console.log('\n--- INFERENCE TEST: PASSED ---');
    
  } catch (err: any) {
    if (err.message === 'MODEL UNAVAILABLE') {
      console.error('ERROR: MODEL UNAVAILABLE — no model.onnx or metadata.json found.');
    } else {
      console.error('Inference failed:', err.message);
    }
  }
}

testInference();
