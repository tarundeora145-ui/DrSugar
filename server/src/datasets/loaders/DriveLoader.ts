import { BaseLoader, ValidationResult } from './BaseLoader';

export class DriveLoader extends BaseLoader {
  constructor(dataDir: string) {
    super('drive', 'DRIVE', dataDir);
  }

  scan(): ValidationResult {
    this.initializeDBRecord();
    if (!this.checkDirectoryExists('')) {
      this.updateDBStatus('MISSING');
      return { status: 'MISSING', messages: ['Dataset directory missing'], imageCount: 0, labelCount: 0 };
    }
    const hasImages = this.checkDirectoryExists('training');
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

    const images = this.getImagesInDir('training/images');
    result.imageCount = images.length;
    images.forEach(img => this.insertDBFile(img, `training/images/\${img}`, 'fundus'));

    const masks = this.getImagesInDir('training/1st_manual');
    result.labelCount = masks.length;
    masks.forEach(mask => {
      this.insertDBFile(mask, `training/1st_manual/\${mask}`, 'mask');
      this.insertDBLabel(mask, -1); // Unspecified DR grade for vessel masks

      const baseNum = mask.split('_')[0];
      const correspondingImg = images.find(img => img.startsWith(baseNum + '_'));
      if (correspondingImg) {
        this.insertDBAnnotation(correspondingImg, 'VESSEL_MASK', JSON.stringify({ maskPath: `training/1st_manual/\${mask}` }));
      }
    });

    if (result.imageCount > 0 && result.labelCount > 0) {
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
