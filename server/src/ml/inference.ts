import * as fs from 'fs';
import * as path from 'path';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';

export interface InferenceResult {
  prediction: number;
  probabilities: number[];
  confidence: number;
  referableDR: boolean;
  modelVersion: string;
  preprocessingVersion: string;
  timestamp: string;
}

// Global session cache to avoid reloading
let session: ort.InferenceSession | null = null;
let metadata: any = null;

export async function loadModel() {
  const modelPath = path.resolve(__dirname, '../../../models/dr_classification/model.onnx');
  const metadataPath = path.resolve(__dirname, '../../../models/dr_classification/metadata.json');
  
  if (!fs.existsSync(modelPath) || !fs.existsSync(metadataPath)) {
    throw new Error('MODEL UNAVAILABLE');
  }

  if (!session) {
    session = await ort.InferenceSession.create(modelPath);
    metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  }
}

export async function preprocessImage(imageBuffer: Buffer): Promise<Float32Array> {
  const { data, info } = await sharp(imageBuffer)
    .resize(224, 224, { fit: 'cover' })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const tensorData = new Float32Array(3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 224 * 224; i++) {
    // RGB layout, normalized
    tensorData[i] = (data[i * 3] / 255.0 - mean[0]) / std[0]; // R
    tensorData[224 * 224 + i] = (data[i * 3 + 1] / 255.0 - mean[1]) / std[1]; // G
    tensorData[2 * 224 * 224 + i] = (data[i * 3 + 2] / 255.0 - mean[2]) / std[2]; // B
  }

  return tensorData;
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map(x => x / sum);
}

export async function runInference(imageBuffer: Buffer): Promise<InferenceResult> {
  if (!session || !metadata) {
    await loadModel();
  }

  const tensorData = await preprocessImage(imageBuffer);
  const tensor = new ort.Tensor('float32', tensorData, [1, 3, 224, 224]);
  
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session!.inputNames[0]] = tensor;
  
  const outputData = await session!.run(feeds);
  const outputTensor = outputData[session!.outputNames[0]];
  const logits = Array.from(outputTensor.data as Float32Array);
  
  const probs = softmax(logits);
  const prediction = probs.indexOf(Math.max(...probs));
  const confidence = probs[prediction];
  
  return {
    prediction,
    probabilities: probs,
    confidence,
    referableDR: prediction >= 2,
    modelVersion: metadata.version || '1.0.0',
    preprocessingVersion: metadata.preprocessing_version || 'v1-224-imagenet',
    timestamp: new Date().toISOString()
  };
}
