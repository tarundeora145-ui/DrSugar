import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { BaseLoader, ValidationResult } from './BaseLoader';

export class AptosLoader extends BaseLoader {
  constructor(dataDir: string) {
    super('aptos2019', 'APTOS 2019', dataDir);
  }

  scan(): ValidationResult {
    this.initializeDBRecord();
    
    if (!this.checkDirectoryExists('')) {
      this.updateDBStatus('MISSING');
      return { status: 'MISSING', messages: ['Dataset directory missing. Download from Kaggle: `kaggle competitions download -c aptos2019-blindness-detection`'], imageCount: 0, labelCount: 0 };
    }

    const hasImages = this.checkDirectoryExists('train_images');
    const hasLabels = this.checkFileExists('train.csv');

    if (hasImages && hasLabels) {
      this.updateDBStatus('CONNECTED');
      return { status: 'CONNECTED', messages: ['Dataset detected'], imageCount: 0, labelCount: 0 };
    } else if (hasImages || hasLabels) {
      this.updateDBStatus('PARTIAL');
      return { status: 'PARTIAL', messages: ['Dataset partially detected'], imageCount: 0, labelCount: 0 };
    } else {
      this.updateDBStatus('MISSING');
      return { status: 'MISSING', messages: ['Dataset directory empty'], imageCount: 0, labelCount: 0 };
    }
  }

  validate(): ValidationResult {
    const result: ValidationResult = { status: 'INVALID', messages: [], imageCount: 0, labelCount: 0 };
    this.initializeDBRecord();
    this.clearDBRecords();

    if (!this.checkDirectoryExists('')) {
      result.status = 'MISSING';
      result.messages.push('Dataset directory missing.');
      this.updateDBStatus(result.status);
      return result;
    }

    const images = this.getImagesInDir('train_images');
    result.imageCount = images.length;

    if (images.length === 0) {
      result.messages.push('No valid images found in train_images/.');
    } else {
      images.forEach(img => this.insertDBFile(img, `train_images/\${img}`, 'fundus'));
    }

    let parsedLabels: any[] = [];
    if (this.checkFileExists('train.csv')) {
      try {
        const fileContent = fs.readFileSync(path.join(this.datasetPath, 'train.csv'));
        parsedLabels = parse(fileContent, { columns: true, skip_empty_lines: true });
        result.labelCount = parsedLabels.length;
        
        parsedLabels.forEach(record => {
          // APTOS uses id_code (no extension usually) and diagnosis
          const filename = `\${record.id_code}.png`; 
          this.insertDBLabel(filename, parseInt(record.diagnosis, 10));
        });
      } catch (err) {
        result.messages.push('Error parsing train.csv.');
      }
    } else {
      result.messages.push('train.csv is missing.');
    }

    if (result.imageCount > 0 && result.labelCount > 0 && result.messages.length === 0) {
      result.status = 'CONNECTED';
    } else if (result.imageCount > 0 || result.labelCount > 0) {
      result.status = 'PARTIAL';
    } else {
      result.status = 'MISSING';
    }

    this.updateDBStatus(result.status);
    return result;
  }
}
