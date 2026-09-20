const fs = require('fs');
const path = require('path');
const { runInference } = require('../server/dist/ml/inference.js');

async function main() {
  try {
    const imagePath = path.resolve(__dirname, '../data/aptos2019/train_images/000c1434d8d7.png');
    if (!fs.existsSync(imagePath)) {
      console.error('Image not found at', imagePath);
      process.exit(1);
    }
    
    console.log('IMAGE:', imagePath);
    console.log('MODEL: models/dr_classification/model.onnx');
    console.log('INPUT SHAPE: [1, 3, 224, 224]');
    console.log('OUTPUT SHAPE: [1, 5]');
    
    const imageBuffer = fs.readFileSync(imagePath);
    
    const startTime = Date.now();
    const result = await runInference(imageBuffer);
    const inferenceTime = Date.now() - startTime;
    
    console.log('PREDICTED GRADE:', result.prediction);
    console.log('PROBABILITIES:');
    console.log('[' + result.probabilities.map(p => p.toFixed(4)).join(', ') + ']');
    console.log('CONFIDENCE:', result.confidence.toFixed(4));
    console.log('REFERABLE DR:', result.referableDR);
    console.log('INFERENCE TIME:', inferenceTime + 'ms');
  } catch (error) {
    console.error('Error during inference test:', error);
  }
}

main();
