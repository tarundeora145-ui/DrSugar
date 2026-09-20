import path from 'path';
import db from './src/database/db';
import { DriveLoader } from './src/datasets/loaders/DriveLoader';

const DATA_DIR = path.resolve(__dirname, '../data');
const loader = new DriveLoader(DATA_DIR);
const result = loader.validate();
console.log('DRIVE validate result:', JSON.stringify(result, null, 2));

const masks = db.prepare("SELECT COUNT(*) as c FROM dataset_files WHERE dataset_id='drive' AND type='mask'").get() as any;
const imgs  = db.prepare("SELECT COUNT(*) as c FROM dataset_files WHERE dataset_id='drive' AND type='fundus'").get() as any;
const anns  = db.prepare("SELECT COUNT(*) as c FROM dataset_annotations WHERE dataset_id='drive'").get() as any;
const lbls  = db.prepare("SELECT COUNT(*) as c FROM dataset_labels WHERE dataset_id='drive'").get() as any;
console.log(`DB fundus=${imgs.c} masks=${masks.c} annotations=${anns.c} labels=${lbls.c}`);
