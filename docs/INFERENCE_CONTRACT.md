# Inference Contract

The Node.js ML Service (`server/src/ml/inference.ts`) establishes a strict contract for inputs and outputs. 

## Inputs
- **Image**: Raw `Buffer` loaded from disk or memory (PNG/JPEG format).

## Processing
1. Image is resized and normalized into a `Float32Array` tensor of shape `[1, 3, 224, 224]`.
2. Passed to `onnxruntime-node` session.
3. Raw logits are returned and passed through a Softmax function.

## Output Contract
The `runInference` function returns a JSON object matching the following TypeScript interface:

```typescript
export interface InferenceResult {
  prediction: number;       // The integer class ID (0, 1, 2, 3, 4)
  probabilities: number[];  // Array of 5 floating point numbers summing to ~1.0
  confidence: number;       // The probability of the predicted class
  referableDR: boolean;     // True if prediction >= 2
  modelVersion: string;     // The version string from metadata.json
  preprocessingVersion: string; // The preprocessing string from metadata.json
  timestamp: string;        // ISO-8601 string of execution time
}
```

## Error Handling
- If the model is missing (no `.onnx` or `metadata.json` present), the service will throw `Error('MODEL UNAVAILABLE')`.
- Endpoints must catch this exact error and forward it cleanly to the frontend without failing the entire request loop.
