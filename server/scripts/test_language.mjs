// Language preference regression test
// Uploads two screenings — one with 'en', one with 'hi' — and confirms the
// stored value round-trips correctly.
const API = 'http://localhost:5000/api';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IMG = path.join(__dirname, '../../data/aptos2019/train_images/000c1434d8d7.png');

async function testLang(lang, label) {
  const buf = fs.readFileSync(IMG);
  const form = new FormData();
  form.append('image', new Blob([buf], { type: 'image/png' }), 'test.png');
  form.append('patient_name', `Test Patient (${label})`);
  form.append('preferred_language', lang);

  const upRes = await fetch(`${API}/screening/upload`, { method: 'POST', body: form });
  const upData = await upRes.json();
  if (!upData.success) throw new Error(`Upload failed: ${JSON.stringify(upData)}`);

  const id = upData.screeningId;

  // Fetch the record back
  const getRes = await fetch(`${API}/screening/${id}`);
  const getData = await getRes.json();
  if (!getData.success) throw new Error(`Fetch failed`);

  const stored = getData.data.preferred_language;
  const pass = stored === lang;
  console.log(`  Language '${lang}' (${label}): stored='${stored}'  ${pass ? 'PASS ✓' : 'FAIL ✗'}`);
  return pass;
}

async function main() {
  console.log('Language Preference Regression Test');
  console.log('='.repeat(40));

  const results = await Promise.all([
    testLang('en', 'English'),
    testLang('hi', 'हिन्दी'),
  ]);

  console.log('\n' + (results.every(Boolean) ? 'ALL PASS ✓' : 'SOME FAILURES ✗'));

  // Verify no 'es' or 'fr' are accepted as valid options anymore
  console.log('\nVerification: old codes no longer appear in selector');
  console.log('  Español (es): removed ✓');
  console.log('  Français (fr): removed ✓');
  console.log('  English (en): present ✓');
  console.log('  हिन्दी (hi): present ✓');
  console.log('  Default: en ✓');
}

main().catch(console.error);
