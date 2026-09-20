import path from 'path';
import fs from 'fs';
import db from './src/database/db';
import { AptosLoader } from './src/datasets/loaders/AptosLoader';
import { DriveLoader } from './src/datasets/loaders/DriveLoader';
import { IdridLoader } from './src/datasets/loaders/IdridLoader';
import { MessidorLoader } from './src/datasets/loaders/MessidorLoader';

const DATA_DIR = path.resolve(__dirname, '../data');

function validateDataset(name: string, LoaderClass: any) {
    console.log(`\n=== Validating ${name} ===`);
    const loader = new LoaderClass(DATA_DIR);
    const result = loader.validate();
    console.log(JSON.stringify(result, null, 2));

    const dbRecord = db.prepare('SELECT * FROM datasets WHERE id = ?').get(loader.id);
    console.log('--- SQLITE RECORD ---');
    console.log(JSON.stringify(dbRecord, null, 2));
    
    return result;
}

validateDataset('APTOS 2019', AptosLoader);
validateDataset('DRIVE', DriveLoader);
validateDataset('IDRiD', IdridLoader);
validateDataset('Messidor-2', MessidorLoader);
