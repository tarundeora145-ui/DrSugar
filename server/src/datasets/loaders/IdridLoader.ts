import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { BaseLoader, ValidationResult } from './BaseLoader';

export class IdridLoader extends BaseLoader {
  constructor(dataDir: string) {
    super('idrid', 'IDRiD', dataDir);
  }

  scan(): ValidationResult {
    this.initializeDBRecord();
    if (!this.checkDirectoryExists('')) {
      this.updateDBStatus('MISSING');
      return { status: 'MISSING', messages: ['Dataset directory missing'], imageCount: 0, labelCount: 0 };
    }
    const hasImages = this.checkDirectoryExists('Disease Grading');
    const status = hasImages ? 'CONNECTED' : 'PARTIAL';
    this.updateDBStatus(status);
    return { status, messages: ['Scan complete'], imageCount: 0, labelCount: 0 };
  }

  validate(): ValidationResult {
    const result: ValidationResult = { status: 'INVALID', messages: [], imageCount: 0, labelCount: 0 };
    this.initializeDBRecord();
    this.clearDBRecords();

    if (!this.checkDirectoryExists('')) {
      result.status = 'MISSING';
      result.messages.push('Dataset missing.');
      this.updateDBStatus(result.status);
      return result;
    }

    // Checking Disease Grading folder
    const images = this.getImagesInDir('Disease Grading/Original Images/Training Set');
    result.imageCount += images.length;
    images.forEach(img => this.insertDBFile(img, `Disease Grading/Original Images/Training Set/\${img}`, 'fundus'));

    if (images.length === 0) {
      result.messages.push('No images found in Disease Grading/Original Images/Training Set');
    }

    if (this.checkFileExists('Disease Grading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv')) {
      try {
        const fileContent = fs.readFileSync(path.join(this.datasetPath, 'Disease Grading/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'));
        const parsedLabels = parse(fileContent, { columns: true, skip_empty_lines: true });
        result.labelCount += parsedLabels.length;
        parsedLabels.forEach((record: any) => {
          const filename = `\${record['Image name']}.jpg`;
          this.insertDBLabel(filename, parseInt(record['Retinopathy grade'], 10));
          
          if (record['Risk of macular edema ']) {
            this.insertDBAnnotation(filename, 'DME', JSON.stringify({ risk: parseInt(record['Risk of macular edema '], 10) }));
          }
        });
      } catch (err) {
        result.messages.push('Error parsing IDRiD labels.');
      }
    } else {
      result.messages.push('Labels CSV missing.');
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
