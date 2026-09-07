"use strict";
(() => {
  let busy=false;
  async function refresh() {
    const output=document.getElementById("expected-exit-message");
    if(!output || busy)return;
    busy=true;
    try {
      const response=await fetch("/expected-exit-status",{cache:"no-store",signal:AbortSignal.timeout(15000)});
      if(!response.ok)throw Error("unavailable");
      const report=await response.json();
      if(report.schema_version!==1 || !Array.isArray(report.obligations))throw Error("invalid");
      const at=new Date(report.generated_at);
      if(!Number.isFinite(at.valueOf()))throw Error("invalid time");
      const counts=report.counts;
      output.textContent=(report.stale?"Report needs refresh · ":"")+counts.missed+" overdue · "+counts.unable_to_verify+" unverified · "+counts.pending+" pending · "+counts.resolved+" resolved. Checked "+at.toLocaleString("en-US",{timeZone:"America/New_York"})+" ET.";
      output.className=report.stale || counts.missed || counts.unable_to_verify?"err":"cap";
      const rows=report.obligations.filter(row=>row.status!=="resolved");
      makeTable(document.getElementById("expected-exit-table"),{rows,columns:[
        {key:"symbol",label:"Symbol",align:"l"},{key:"strategy",label:"Strategy",align:"l"},
        {key:"status",label:"Status",align:"l"},{key:"remaining_tagged_qty",label:"Tagged shares",fmt:v=>v==null?"Unknown":String(v)},
        {key:"deadline",label:"Deadline",align:"l",fmt:v=>v?new Date(v).toLocaleString("en-US",{timeZone:"America/New_York"})+" ET":"Unknown"},
        {key:"detail",label:"Detail",align:"l"}
      ]});
    } catch {
      output.textContent="Expected exits cannot currently be verified. Any table below is from the previous successful check.";
      output.className="err";
    } finally {busy=false;}
  }
  document.addEventListener("DOMContentLoaded",()=>{
    document.getElementById("expected-exit-refresh")?.addEventListener("click",refresh);
    refresh();setInterval(()=>{if(!document.hidden)refresh();},60000);
  });
})();
