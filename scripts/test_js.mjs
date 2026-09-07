// Run every standalone browser/Worker contract test without a live service.
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const files=fs.readdirSync(path.join(root,'tests/js')).filter(f=>/^test_.*\.(?:js|mjs|cjs)$/.test(f)).sort();
let failures=0;
for(const file of files){
  const result=spawnSync(process.execPath,[path.join(root,'tests/js',file)],{cwd:root,encoding:'utf8',timeout:60000});
  if(result.status===0)console.log(`PASS ${file}`);
  else {failures++;console.error(`FAIL ${file}\n${result.error||''}\n${result.stdout||''}\n${result.stderr||''}`);}
}
console.log(`${files.length-failures}/${files.length} JavaScript test files passed`);
process.exitCode=failures?1:0;
