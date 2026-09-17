const fs = require('fs');
const path = require('path');
const dir = 'src/datasets/loaders';

const files = fs.readdirSync(dir).filter(f => f.endsWith('.ts'));

for (const file of files) {
  const filePath = path.join(dir, file);
  let content = fs.readFileSync(filePath, 'utf8');
  // Replace the escaped backticks string `\`` with just ```
  content = content.replace(/\\`/g, '`');
  fs.writeFileSync(filePath, content);
}
console.log('Fixed backticks');
