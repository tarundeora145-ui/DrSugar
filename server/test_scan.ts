import path from 'path';
import { AptosLoader } from './src/datasets/loaders/AptosLoader';
import db from './src/database/db';

const DATA_DIR = path.resolve(__dirname, '../data');
const loader = new AptosLoader(DATA_DIR);
const result = loader.scan();

console.log('--- APTOS 2019 SCAN RESULT ---');
console.log(JSON.stringify(result, null, 2));

const dbRecord = db.prepare('SELECT * FROM datasets WHERE id = ?').get('aptos2019');
console.log('--- SQLITE RECORD ---');
console.log(JSON.stringify(dbRecord, null, 2));
