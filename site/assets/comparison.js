"use strict";
/* Catalog order_ref tags -> display name. Accepts the strategies.json payload
   (or null). Only these tags leave the "excluded" bucket for Actual-only. */
function catalogTagMap(catalog) {
  const map=new Map();
  for(const s of (catalog && Array.isArray(catalog.strategies))?catalog.strategies:[]) {
    for(const tag of Array.isArray(s.order_ref_tags)?s.order_ref_tags:[]) {
      const key=String(tag||"").trim();
      if(key && !map.has(key))map.set(key,s.name||s.id||key);
    }
  }
  return map;
}
/* Exact tag first; a catalog tag ending in * matches by prefix (Pitch-*),
   grouped under the wildcard tag. Returns [groupTag, name] or null. */
function catalogTagLookup(tags, strategy) {
  if(!strategy)return null;
  if(tags.has(strategy))return [strategy,tags.get(strategy)];
  for(const [tag,name] of tags) {
    if(tag.endsWith("*") && tag.length>1 && strategy.startsWith(tag.slice(0,-1)))return [tag,name];
  }
  return null;
}
function fillNotional(fill) {
  const qty=Math.abs(Number(fill.qty)), price=Number(fill.price);
  if(!Number.isFinite(qty) || !Number.isFinite(price))return null;
  const secType=String(fill.sec_type||"STK").toUpperCase();
  if(secType==="STK")return qty*price;
  const multiplier=Number(fill.multiplier);
  return (secType==="FUT"||secType==="OPT") && multiplier>0 ? qty*price*multiplier : null;
}
function comparisonRows(trades, fills, strategies, from, to, catalogTags) {
  const rows=new Map(strategies.map(name=>[name,{strategy:name,theo:0,theo_closed:0,actual:null,actual_fills:0,actual_missing:0,commissions:null}]));
  const tags=catalogTags instanceof Map ? catalogTags : new Map([...(catalogTags||[])].map(t=>[t,t]));
  const actualOnly=new Map();
  let excluded=0, missingPnl=0;
  const c=trades.columns || {};
  for(let i=0;i<(trades.n||0);i++) {
    const row=rows.get(c.Strategy?.[i]), date=c.Exit_Date?.[i], pnl=c.PnL_flat?.[i];
    if(row && !c.Open?.[i] && date>=from && date<=to && Number.isFinite(pnl)) {
      row.theo+=pnl;row.theo_closed++;
    }
  }
  for(const fill of fills || []) {
    if(fill.account_key!=="primary")continue;
    const timestamp=new Date(fill.time);
    if(!Number.isFinite(timestamp.valueOf())){excluded++;continue;}
    const date=timestamp.toLocaleDateString("en-CA",{timeZone:"America/New_York"});
    if(date<from || date>to)continue;
    const parts=String(fill.order_ref || "").split("|");
    const strategy=fill.strategy || (parts.length>=4?parts[2].trim():"");
    const row=rows.get(strategy);
    if(!row) {
      const hit=catalogTagLookup(tags,strategy);
      if(!hit){excluded++;continue;}
      const [groupTag,name]=hit;
      if(!actualOnly.has(groupTag))actualOnly.set(groupTag,{strategy:groupTag,name,fills:0,symbols:new Set(),buys:0,sells:0,notional:0,notional_unknown:0,actual:null,actual_missing:0,commissions:null});
      const only=actualOnly.get(groupTag);
      only.fills++;
      if(fill.symbol)only.symbols.add(String(fill.local_symbol||fill.symbol));
      const side=String(fill.side||"").toUpperCase();
      if(side==="BOT"||side==="BUY")only.buys++;
      else if(side==="SLD"||side==="SELL")only.sells++;
      const notional=fillNotional(fill);
      if(notional==null)only.notional_unknown++;else only.notional+=notional;
      if(Number.isFinite(fill.realized_pnl))only.actual=(only.actual??0)+fill.realized_pnl;
      else {missingPnl++;only.actual_missing++;}
      if(Number.isFinite(fill.commission))only.commissions=(only.commissions??0)+fill.commission;
      continue;
    }
    row.actual_fills++;
    if(Number.isFinite(fill.realized_pnl))row.actual=(row.actual??0)+fill.realized_pnl;
    else {missingPnl++;row.actual_missing++;}
    if(Number.isFinite(fill.commission))row.commissions=(row.commissions??0)+fill.commission;
  }
  const actualOnlyRows=[...actualOnly.values()].map(r=>({...r,symbols:[...r.symbols].sort().join(", ")}))
    .sort((a,b)=>a.strategy.localeCompare(b.strategy));
  return {rows:[...rows.values()].filter(r=>r.theo_closed||r.actual_fills),actualOnly:actualOnlyRows,excluded,missingPnl};
}
let comparisonBusy=false;
async function refreshComparison() {
  if(comparisonBusy)return;
  comparisonBusy=true;
  const status=document.getElementById("compare-status"),button=document.getElementById("compare-refresh");
  button.disabled=true;
  try {
    const snapshot=await loadSiteSnapshot(async meta=>{
      const catalogP=(meta.payloads||{}).strategies===false?Promise.resolve(null)
        :fetchSitePayload(meta,"data/strategies.json").catch(()=>null);
      const [trades,catalog]=await Promise.all([fetchSitePayload(meta,"data/trades.json"),catalogP]);
      return {trades,catalog};
    });
    const broker=await fetchJSON("/exec-fills");
    if(!Array.isArray(broker.fills))throw Error("Broker fills unavailable");
    const days=Number(document.getElementById("compare-days").value);
    const to=new Date().toLocaleDateString("en-CA",{timeZone:"America/New_York"});
    const fromDate=new Date(to+"T00:00:00Z");fromDate.setUTCDate(fromDate.getUTCDate()-(days-1));
    const from=fromDate.toISOString().slice(0,10);
    const names=(snapshot.meta.strategies||[]).map(s=>s.Strategy);
    const result=comparisonRows(snapshot.trades,broker.fills,names,from,to,catalogTagMap(snapshot.catalog));
    const coverage=broker.completeness || {}, account=coverage.accounts?.primary;
    const verified=account?.complete===true && !coverage.truncated && !coverage.merge_error;
    makeTable(document.getElementById("compare-table"),{rows:result.rows,textOnly:true,columns:[
      {key:"strategy",label:"Strategy",align:"l"},{key:"theo_closed",label:"Model closes"},
      {key:"theo",label:"Model realized · $750k",fmt:v=>fmt.money(v)},
      {key:"actual_fills",label:"Actual executions"},
      {key:"actual",label:"Broker reported subtotal",fmt:v=>v==null?"Unknown":fmt.money(v)},
      {key:"actual_missing",label:"Fills without PnL"},
      {key:"commissions",label:"Reported fees",fmt:v=>v==null?"Unknown":fmt.money(v,2)}
    ]});
    renderActualOnly(result.actualOnly,snapshot.catalog);
    status.textContent=from+" to "+to+" · "+(verified?"Current Primary fill source verified":"Primary fill coverage unverified or incomplete")+" · "+result.excluded+" unmatched executions excluded · "+result.missingPnl+" executions without reported realized PnL. Model through "+snapshot.meta.ledger_last_signal+".";
    status.className=verified?"cap":"err";
    setAsof("Fetched "+new Date().toLocaleTimeString());
  } catch(error) {
    status.textContent="Refresh failed: "+error.message+". Any table below is from the previous successful refresh.";
    status.className="err";
  } finally {comparisonBusy=false;button.disabled=false;}
}
function renderActualOnly(rows,catalog) {
  const card=document.getElementById("compare-actual-only"),note=document.getElementById("compare-actual-only-note");
  if(!card)return;
  if(!catalog){card.style.display="none";return;}
  card.style.display="";
  if(note)note.textContent=rows.length?"Fills tagged with a catalog strategy that has no ledger replay. There is no model column: these strategies are measured on broker fills only."
    :"No fills from catalog strategies without a ledger replay in this window.";
  makeTable(document.getElementById("compare-actual-only-table"),{rows,textOnly:true,columns:[
    {key:"name",label:"Strategy",align:"l"},{key:"strategy",label:"Tag",align:"l"},
    {key:"fills",label:"Executions"},{key:"symbols",label:"Symbols",align:"l"},
    {key:"buys",label:"Buys"},{key:"sells",label:"Sells"},
    {key:"notional",label:"Notional",fmt:(v,r)=>r.notional_unknown&&r.notional_unknown>=r.fills?"Unknown"
      :fmt.money(v)+(r.notional_unknown?" + "+r.notional_unknown+" unpriced":"")},
    {key:"actual",label:"Broker reported subtotal",fmt:v=>v==null?"Unknown":fmt.money(v)},
    {key:"actual_missing",label:"Fills without PnL"},
    {key:"commissions",label:"Reported fees",fmt:v=>v==null?"Unknown":fmt.money(v,2)}
  ]});
}
document.addEventListener("DOMContentLoaded",()=>{
  renderNav("index.html");
  document.getElementById("compare-refresh").addEventListener("click",refreshComparison);
  document.getElementById("compare-days").addEventListener("change",refreshComparison);
  refreshComparison();
});
