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
    images.forEach(img => this.insertDBFile(img, `IMAGES/${img}`, 'fundus'));

    // Actual CSV is messidor-2.csv with semicolon-delimited left;right image pairs.
    // This file contains image pairing metadata only — it does NOT contain
    // adjudicated DR grades. DR ground truth labels are not bundled with the
    // standard Messidor-2 image release. labelCount stays 0.
    if (this.checkFileExists('messidor-2.csv')) {
      try {
        const fileContent = fs.readFileSync(path.join(this.datasetPath, 'messidor-2.csv'), 'utf8');
        const lines = fileContent.split('\n').filter(l => l.trim() && !l.startsWith('left'));
        // Each line is: left_image;right_image — register as pairing annotations only
        lines.forEach(line => {
          const parts = line.split(';');
          const left = parts[0]?.trim();
          const right = parts[1]?.trim();
          if (left) this.insertDBAnnotation(left, 'LEFT_RIGHT_PAIR', JSON.stringify({ left, right: right || null }));
        });
        result.messages.push(`messidor-2.csv present: ${lines.length} image pairs found. No adjudicated DR grades in this release.`);
      } catch (err) {
        result.messages.push('Error parsing messidor-2.csv.');
      }
    } else {
      result.messages.push('messidor-2.csv missing.');
    }

    // PARTIAL: images found but no DR grade labels available in this release
    if (result.imageCount > 0) {
      result.status = 'PARTIAL';
    } else {
      result.status = 'MISSING';
    }

    this.updateDBStatus(result.status);
    return result;
  }
}
