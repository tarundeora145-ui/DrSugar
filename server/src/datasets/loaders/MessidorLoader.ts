import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { BaseLoader, ValidationResult } from './BaseLoader';

export class MessidorLoader extends BaseLoader {
  constructor(dataDir: string) {
    super('messidor2', 'Messidor-2', dataDir);
  }

  scan(): ValidationResult {
    this.initializeDBRecord();
    if (!this.checkDirectoryExists('')) {
      this.updateDBStatus('MISSING');
      return { status: 'MISSING', messages: ['Dataset directory missing'], imageCount: 0, labelCount: 0 };
    }
    const hasImages = this.checkDirectoryExists('IMAGES');
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

    const images = this.getImagesInDir('IMAGES');
    result.imageCount = images.length;
    images.forEach(img => this.insertDBFile(img, `IMAGES/\${img}`, 'fundus'));

    if (this.checkFileExists('messidor_data.csv')) {
      try {
        const fileContent = fs.readFileSync(path.join(this.datasetPath, 'messidor_data.csv'));
        const parsedLabels = parse(fileContent, { columns: true, skip_empty_lines: true });
        result.labelCount = parsedLabels.length;
        parsedLabels.forEach((record: any) => {
          this.insertDBLabel(record.image_id, parseInt(record.adjudicated_dr_grade, 10));
          this.insertDBAnnotation(record.image_id, 'GROUND_TRUTH', JSON.stringify({ available: true, gradable: record.adjudicated_gradable }));
        });
      } catch (err) {
        result.messages.push('Error parsing Messidor CSV.');
      }
    } else {
      result.messages.push('messidor_data.csv missing.');
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
