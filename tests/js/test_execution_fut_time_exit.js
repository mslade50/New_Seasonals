"use strict";

// CBOT grains and CME livestock are closed at 15:59 ET. The executor moves their
// TIME exit to the session close minus one minute; the ticket must show that
// time in ET, and every other future must still read 15:59.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "execution.js"),
  "utf8",
);
const context = {
  console,
  document: { addEventListener() {} },
  window: {},
  location: { search: "" },
  URLSearchParams,
  Intl,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "execution.js" });

function jsonExpr(expr) {
  return JSON.parse(vm.runInContext(`JSON.stringify(${expr})`, context));
}

const XK = '{symbol:"XK",ib_symbol:"YK",trading_class:"XK",exchange:"CBOT",multiplier:1000,min_tick:0.00125,'
  + 'price_magnifier:100,time_exit_session_close:"13:20",time_exit_session_tz:"US/Central"}';
const LE = '{symbol:"LE",ib_symbol:"LE",trading_class:"LE",exchange:"CME",multiplier:400,min_tick:0.00025,'
  + 'price_magnifier:100,time_exit_session_close:"13:05",time_exit_session_tz:"US/Central"}';
const MES = '{symbol:"MES",ib_symbol:"MES",trading_class:"MES",exchange:"CME",multiplier:5,min_tick:0.25}';
vm.runInContext(`FUT_SPECS = {XK: ${XK}, LE: ${LE}, MES: ${MES}}`, context);

// Clock on both sides of the November and March US clock changes.
for (const day of ["2026-11-03", "2030-11-01", "2030-11-04", "2031-03-07", "2031-03-10", "2030-07-09"]) {
  assert.strictEqual(jsonExpr(`futTimeExitClock(futSpec("XK"), "${day}")`), "14:19", day);
  assert.strictEqual(jsonExpr(`futTimeExitClock(futSpec("LE"), "${day}")`), "14:04", day);
  assert.strictEqual(jsonExpr(`futTimeExitClock(futSpec("MES"), "${day}")`), "15:59", day);
}
assert.strictEqual(jsonExpr('futTimeExitClock(null, "2026-11-03")'), "15:59");
assert.strictEqual(jsonExpr('futTimeExitClock({time_exit_session_close:"16:00",time_exit_session_tz:"US/Central"}, "2026-11-03")'), "15:59");

// A position may carry the IB root (YK) rather than the page alias (XK).
assert.strictEqual(jsonExpr('futTimeExitSpec("YK").symbol'), "XK");

// The agent preview's TIME leg, as the page renders it.
const xkLeg = "TIME    SELL 2  MKT @ 2026-11-03 15:59  (OCA)";
const xkFut = '{symbol:"XK",sec_type:"FUT"}';
assert.strictEqual(jsonExpr(`truthfulTimeLeg(${JSON.stringify(xkLeg)}, ${xkFut})`),
  "TIME    SELL 2  MKT @ 2026-11-03 14:19 ET  (OCA)");
assert.strictEqual(jsonExpr(`truthfulTimeLeg("NEAR TIME    SELL 1  MKT @ 2026-11-03 15:59  (OCA)", ${xkFut})`),
  "NEAR TIME    SELL 1  MKT @ 2026-11-03 14:19 ET  (OCA)");
const attachLeg = "TIME    SELL 2 YK  MKT @ 2026-11-03 15:59  (OCA GTC)";
assert.strictEqual(jsonExpr(`truthfulTimeLeg(${JSON.stringify(attachLeg)}, {symbol:"YK",sec_type:"FUT"})`),
  "TIME    SELL 2 YK  MKT @ 2026-11-03 14:19 ET  (OCA GTC)");
const mesLeg = "TIME    SELL 2  MKT @ 2026-11-03 15:59  (OCA)";
assert.strictEqual(jsonExpr(`truthfulTimeLeg(${JSON.stringify(mesLeg)}, {symbol:"MES",sec_type:"FUT"})`), mesLeg);
// Stocks and the open clock are untouched.
assert.strictEqual(jsonExpr(`truthfulTimeLeg(${JSON.stringify(xkLeg)}, {symbol:"XK",sec_type:"STK"})`), xkLeg);
assert.strictEqual(jsonExpr(`truthfulTimeLeg(${JSON.stringify(xkLeg)}, {symbol:"XK",sec_type:"FUT",time_stop_at:"open"})`), xkLeg);

assert.strictEqual(jsonExpr(`timeExitSuffix({symbol:"XK",sec_type:"FUT",time_stop:"2026-11-03"})`), " 14:19 ET");
assert.strictEqual(jsonExpr(`timeExitSuffix({symbol:"MES",sec_type:"FUT",time_stop:"2026-11-03"})`), "");

console.log("PASS execution futures time-exit clock");
