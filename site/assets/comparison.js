"use strict";
function comparisonRows(trades, fills, strategies, from, to) {
  const rows=new Map(strategies.map(name=>[name,{strategy:name,theo:0,theo_closed:0,actual:null,actual_fills:0,actual_missing:0,commissions:null}]));
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
    if(!row){excluded++;continue;}
    row.actual_fills++;
    if(Number.isFinite(fill.realized_pnl))row.actual=(row.actual??0)+fill.realized_pnl;
    else {missingPnl++;row.actual_missing++;}
    if(Number.isFinite(fill.commission))row.commissions=(row.commissions??0)+fill.commission;
  }
  return {rows:[...rows.values()].filter(r=>r.theo_closed||r.actual_fills),excluded,missingPnl};
}
let comparisonBusy=false;
async function refreshComparison() {
  if(comparisonBusy)return;
  comparisonBusy=true;
  const status=document.getElementById("compare-status"),button=document.getElementById("compare-refresh");
  button.disabled=true;
  try {
    const snapshot=await loadSiteSnapshot(async meta=>({trades:await fetchSitePayload(meta,"data/trades.json")}));
    const broker=await fetchJSON("/exec-fills");
    if(!Array.isArray(broker.fills))throw Error("Broker fills unavailable");
    const days=Number(document.getElementById("compare-days").value);
    const to=new Date().toLocaleDateString("en-CA",{timeZone:"America/New_York"});
    const fromDate=new Date(to+"T00:00:00Z");fromDate.setUTCDate(fromDate.getUTCDate()-(days-1));
    const from=fromDate.toISOString().slice(0,10);
    const names=(snapshot.meta.strategies||[]).map(s=>s.Strategy);
    const result=comparisonRows(snapshot.trades,broker.fills,names,from,to);
    const coverage=broker.completeness || {}, account=coverage.accounts?.primary;
    const verified=account?.complete===true && !coverage.truncated && !coverage.merge_error;
    makeTable(document.getElementById("compare-table"),{rows:result.rows,columns:[
      {key:"strategy",label:"Strategy",align:"l"},{key:"theo_closed",label:"Model closes"},
      {key:"theo",label:"Model realized · $750k",fmt:v=>fmt.money(v)},
      {key:"actual_fills",label:"Actual executions"},
      {key:"actual",label:"Broker reported subtotal",fmt:v=>v==null?"Unknown":fmt.money(v)},
      {key:"actual_missing",label:"Fills without PnL"},
      {key:"commissions",label:"Reported fees",fmt:v=>v==null?"Unknown":fmt.money(v,2)}
    ]});
    status.textContent=from+" to "+to+" · "+(verified?"Current Primary fill source verified":"Primary fill coverage unverified or incomplete")+" · "+result.excluded+" unmatched executions excluded · "+result.missingPnl+" executions without reported realized PnL. Model through "+snapshot.meta.ledger_last_signal+".";
    status.className=verified?"cap":"err";
    setAsof("Fetched "+new Date().toLocaleTimeString());
  } catch(error) {
    status.textContent="Refresh failed: "+error.message+". Any table below is from the previous successful refresh.";
    status.className="err";
  } finally {comparisonBusy=false;button.disabled=false;}
}
document.addEventListener("DOMContentLoaded",()=>{
  renderNav("index.html");
  document.getElementById("compare-refresh").addEventListener("click",refreshComparison);
  document.getElementById("compare-days").addEventListener("change",refreshComparison);
  refreshComparison();
});
