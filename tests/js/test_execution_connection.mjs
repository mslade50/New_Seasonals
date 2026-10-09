import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

let clock=1_000_000;const logs=[];
const source=fs.readFileSync(new URL('../../execution-broker/src/index.js',import.meta.url),'utf8')
  .replace(/^import[\s\S]*?;\r?\n/gm,'').split('export default')[0].replace('export class ExecBroker','class ExecBroker');
const c={Response,URL,console:{info:(...args)=>logs.push(args)},Date:class extends Date { static now(){return clock;} },
  DurableObject:class {constructor(ctx,env){this.ctx=ctx;this.env=env;}}};
vm.createContext(c);vm.runInContext(source+'\nglobalThis.Broker=ExecBroker;',c);
const values=new Map();
const storage={get:async k=>values.get(k),put:async(k,v)=>values.set(k,v),delete:async k=>values.delete(k)};
const messages=[];let attachment={connectedAt:clock};
const ws={send:x=>messages.push(JSON.parse(x)),serializeAttachment:x=>attachment=x,
  deserializeAttachment:()=>attachment,close(){}};
const broker=new c.Broker({storage,getWebSockets:()=>[ws]}, {STATUS_TOKEN:'status-fixture-secret',AGENT_TOKEN:'agent-fixture-secret'});
await broker.webSocketMessage(ws,JSON.stringify({type:'heartbeat',session:'a'.repeat(32),seq:1}));
assert.equal(messages.at(-1).seq,1);assert.equal(messages.at(-1).session,'a'.repeat(32));
await broker.webSocketMessage(ws,JSON.stringify({type:'heartbeat'}));
assert.equal(messages.at(-1).seq,undefined);assert.equal(messages.at(-1).of,'heartbeat');
const request={url:'https://fixture.invalid/status',headers:{get:()=> 'Bearer status-fixture-secret'}};
let status=await (await broker.fetch(request)).json();assert.equal(status.online,true);
clock+=30_001;
status=await (await broker.fetch(request)).json();assert.equal(status.online,false);
assert.equal(status.stale_after_ms,30_000);
await broker.webSocketClose(ws,1006,'Bearer agent-fixture-secret https://secret:password@fixture.invalid',false);
await broker.webSocketError(ws,new Error('status-fixture-secret transport failed'));
status=await (await broker.fetch(request)).json();
assert.equal(status.last_connection_close.code,1006);assert.equal(status.last_connection_close.was_clean,false);
assert.equal(status.last_connection_error.at,clock);
assert.equal(status.last_connection_close.socket_connected_at,1_000_000);
assert.doesNotMatch(JSON.stringify(status),/agent-fixture-secret|status-fixture-secret|password/);
assert.equal(logs.length,2);assert.doesNotMatch(JSON.stringify(logs),/agent-fixture-secret|status-fixture-secret|password/);
assert.equal(messages.length,2); // diagnostics never construct/deliver commands
console.log('PASS correlated and legacy ACKs, unchanged offline threshold, redacted close/error diagnostics');
