import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const created=[];
const node=tag=>({tag,innerHTML:'',appendChild(){},querySelectorAll(){return [];}});
const context={document:{createElement(tag){const n=node(tag);created.push(n);return n;},addEventListener(){}},window:{}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(new URL('../../site/assets/common.js',import.meta.url),'utf8'),context);
const attack='<img src=x onerror="alert(1)">';
for(const options of [{},{fmt:v=>v,textOnly:true}]) {
  context.makeTable(node('div'),{rows:[{value:attack}],textOnly:options.textOnly,
    columns:[{key:'value',label:attack,fmt:options.fmt,cls:()=>attack}]});
  const html=created.filter(n=>n.tag==='table').at(-1).innerHTML;
  assert.ok(!html.includes('<img'));
  assert.ok(html.includes('&lt;img'));
  assert.ok(!html.includes('class="<'));
}
context.makeTable(node('div'),{rows:[{value:1}],columns:[{key:'value',label:'Value',fmt:()=>'<strong>1</strong>'}]});
assert.ok(created.filter(n=>n.tag==='table').at(-1).innerHTML.includes('<strong>1</strong>'));
console.log('Plain/report cells escape markup; explicit trusted formatters retain compatibility');
