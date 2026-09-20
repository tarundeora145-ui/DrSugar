import * as fs from 'fs';
import * as path from 'path';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import db from '../database/db';

const modelPath = path.resolve(__dirname, '../../../models/idrid/model.onnx');
const metadataPath = path.resolve(__dirname, '../../../models/idrid/metadata.json');

async function preprocessImage(imageBuffer: Buffer): Promise<Float32Array> {
  const { data } = await sharp(imageBuffer)
    .resize(224, 224, { fit: 'cover' })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const tensorData = new Float32Array(3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 224 * 224; i++) {
    tensorData[i] = (data[i * 3] / 255.0 - mean[0]) / std[0];
    tensorData[224 * 224 + i] = (data[i * 3 + 1] / 255.0 - mean[1]) / std[1];
    tensorData[2 * 224 * 224 + i] = (data[i * 3 + 2] / 255.0 - mean[2]) / std[2];
  }

  return tensorData;
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map(x => x / sum);
}

async function testInference() {
  const session = await ort.InferenceSession.create(modelPath);
  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));

  // Test on one IDRiD image (from training set)
  const testImagePath = path.resolve(__dirname, '../../../data/idrid/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg');
  const imageBuffer = fs.readFileSync(testImagePath);

  const tensorData = await preprocessImage(imageBuffer);
  const tensor = new ort.Tensor('float32', tensorData, [1, 3, 224, 224]);
  
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;
  
  const outputData = await session.run(feeds);
  const outputTensor = outputData[session.outputNames[0]];
  const logits = Array.from(outputTensor.data as Float32Array);
  
  const probs = softmax(logits);
  const prediction = probs.indexOf(Math.max(...probs));
  const confidence = probs[prediction];
  const referableDR = prediction >= 2;

  console.log("--- NODE INFERENCE VERIFICATION ---");
  console.log("Image: IDRiD_001.jpg");
  console.log("Predicted Grade:", prediction);
  console.log("Five Probabilities:", probs);
  console.log("Referable Status:", referableDR);
  console.log("Model Version:", metadata.version);
  
  // Persist a real screening result to SQLite if the existing screening schema supports it.
  const stmt = db.prepare(`
    INSERT INTO screenings (patient_id, image_path, status, result_grade)
    VALUES (?, ?, ?, ?)
  `);
  
  const result = stmt.run('IDRiD-TEST-1', 'data/idrid/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg', 'COMPLETED', prediction);
  console.log(`Persisted screening result to SQLite. ID: ${result.lastInsertRowid}`);
  
  // Do not close db as it is a singleton used by other modules, but for this script it's fine.
}

testInference().catch(console.error);
