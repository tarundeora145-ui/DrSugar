const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const Database = require('../server/node_modules/better-sqlite3');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const serverPath = path.resolve(__dirname, '../server/dist/index.js');
  const imagePath = path.resolve(__dirname, '../data/aptos2019/train_images/000c1434d8d7.png');
  
  if (!fs.existsSync(imagePath)) {
    console.error('Image not found');
    process.exit(1);
  }

  // Start the server
  const server = spawn('node', [serverPath], { cwd: path.resolve(__dirname, '../server') });
  
  server.stdout.on('data', (data) => {
    console.log(`[SERVER] ${data.toString().trim()}`);
  });
  
  server.stderr.on('data', (data) => {
    console.error(`[SERVER ERROR] ${data.toString().trim()}`);
  });

  // Wait for server to start
  await sleep(3000);
  
  try {
    console.log('Sending upload request...');
    // Create multipart form data manually for fetch
    const fileData = fs.readFileSync(imagePath);
    const boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW';
    let body = Buffer.concat([
      Buffer.from(`--${boundary}\r\n`),
      Buffer.from('Content-Disposition: form-data; name="image"; filename="000c1434d8d7.png"\r\n'),
      Buffer.from('Content-Type: image/png\r\n\r\n'),
      fileData,
      Buffer.from(`\r\n--${boundary}--\r\n`)
    ]);

    const uploadRes = await fetch('http://localhost:5000/api/screening/upload', {
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`
      },
      body
    });
    
    const uploadData = await uploadRes.json();
    console.log('Upload result:', uploadData);
    
    if (!uploadData.success) {
      throw new Error('Upload failed');
    }
    
    const id = uploadData.screeningId;
    
    console.log(`Sending process request for ID ${id}...`);
    const processRes = await fetch(`http://localhost:5000/api/screening/${id}/process`, {
      method: 'POST'
    });
    
    const processData = await processRes.json();
    console.log('Process result:', JSON.stringify(processData, null, 2));
    
    console.log('Checking SQLite database...');
    const db = new Database(path.resolve(__dirname, '../database/dr_sugar.db'));
    const record = db.prepare('SELECT * FROM screenings WHERE id = ?').get(id);
    console.log('Database record:', record);
    
  } catch (err) {
    console.error('Error during API test:', err);
  } finally {
    server.kill();
  }
}

main();
