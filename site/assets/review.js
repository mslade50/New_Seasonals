import * as C from './review-core.js';
const $=id=>document.getElementById(id), esc=x=>String(x??'Not specified').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x?new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'}).format(new Date(x)):'Manual venue window unverified';
const labels={pending:'Pending review',approved_review:'Approved',rejected:'Rejected',expired:'Expired',superseded:'Superseded'};
let data=null, rows=[],events=[],staging=[], verified=false, busy=false, generation=0, serverOffset=0;
const now=()=>new Date(Date.now()+serverOffset).toISOString();
const notice=text=>{$('notice').textContent=text;};
const state=row=>row.current?C.view(row.envelope,events,now()):{...C.view(row.envelope,events,now()),status:'superseded'};
function field(k,v){return `<div><dt>${esc(k)}</dt><dd>${esc(v===''?'Not specified':v)}</dd></div>`;}
function trade(o){return `<section class="trade"><h3>Leg ${esc(o.Leg)} · ${esc(o.Action)} ${esc(o.Quantity)} ${esc(o.Ticker)} <span class="pill">${esc(o.Sec_Type)}</span></h3><dl>${field('Entry',`${o.Order_Type} ${o.Limit_Price || 'Auction / open derived'} · ${o.TIF}`)}${field('Entry rule',`${o.Entry_Anchor || o.Entry_Type} ${o.Entry_Offset_ATR || ''} ATR`)}${field('Execute on',o.Execute_On)}${field('Stop',o.Stop_Price|| (o.Stop_ATR?`${o.Stop_ATR} ATR · price at entry`:'No price stop specified'))}${field('Target',o.Target_Price||(o.Target_ATR?`${o.Target_ATR} ATR · price at entry`:'No target specified'))}${field('Time exit',`${o.Time_Exit_Date} ${o.Time_Exit_Order}`)}${field('Entry expiry',o.Entry_Expire_Date)}${field('Stated risk amount',o.Risk_Amt)}${field('Notional',o.Notional)}</dl><details><summary>Full published specification · all fields</summary><dl>${Object.entries(o).map(([k,v])=>field(k,typeof v==='object'?JSON.stringify(v):v)).join('')}</dl></details></section>`;}
function statusText(job){
 const labels={queued:'Queued',scheduled:'Queued for the open',processing:'Staging',pushed:'Staging',delivered:'Staging',working:'Staged',filled:'Filled',partially_filled:'Partially filled',closed:'Closed',partially_closed:'Partially closed',cancelled:'Cancelled',rejected:'Blocked',error:'Blocked',dry_run:'Blocked',expired:'Expired',delivery_unknown:'Needs checking',unknown:'Needs checking',needs_reconciliation:'Needs checking'};
 return `${job.account==='primary'?'Primary':'PA'}: ${labels[job.state]||'Needs checking'}${job.detail?' — '+job.detail:''}`;
}
function card(row){
 const e=row.envelope,p=e.payload,s=state(row),pending=s.status==='pending',eligible=pending&&verified&&!data.read_only&&!busy;
 const jobs=staging.filter(j=>j.proposal_id===e.id),evidence=p.evidence||{};
 const accounts=(p.execution_accounts||[]).map(a=>a==='primary'?'Primary':'PA').join(' + ');
 const approved=s.status==='approved_review',automatic=s.event?.scope==='review_and_stage';
 const progress=jobs.map(j=>`<p role="status"><strong>${esc(statusText(j))}</strong>${j.quantities?.length?'<br>'+j.quantities.map(q=>`${esc(q.action)} ${esc(q.quantity)} ${esc(q.ticker)}`).join(' · '):''}</p>`).join('');
 return `<article class="proposal" id="${esc(e.id)}"><div class="cardhead"><div><div class="product">${p.product==='pitch'?'DAILY PITCH':'DAILY SEASONAL'} · ${esc(p.grade)} · ${p.orders.length} leg(s)</div><h2>${esc(p.title)}</h2></div><span class="pill ${s.status}">${labels[s.status]}</span></div><p class="thesis">${esc(p.thesis)}</p><div class="metadata"><div><span>Accounts</span><b>${esc(accounts||p.account)}</b></div><div><span>Approve by</span><b>${fmt(p.review_deadline)}</b></div><div><span>Evidence</span><b>${esc(evidence.summary||JSON.stringify(evidence))}${evidence.n!=null?` · N=${esc(evidence.n)}`:''}</b></div></div>${p.orders.map(trade).join('')}<p class="small"><strong>What kills it:</strong> ${esc(p.what_kills_it)}</p><p>Yes stages every leg for ${esc(accounts||'the published accounts')}, using the agent’s recommended risk and each account’s equity. Auction orders execute at their stated open or close. The share counts above are the publisher’s reference sizes.</p>${pending&&row.staging_block?`<p role="alert">${esc(row.staging_block)}</p>`:''}${approved&&!automatic?'<p>This earlier approval recorded research only. It did not authorize automatic staging.</p>':''}${automatic?progress||'<p role="status">Approved · queued for staging.</p>':''}<div class="actions">${pending?`<button data-id="${esc(e.id)}" data-decision="approve_review" ${eligible&&!row.staging_block?'':'disabled'}>Yes — stage orders</button><button class="reject" data-id="${esc(e.id)}" data-decision="reject" ${eligible?'':'disabled'}>No — pass</button>`:''}<a class="link" href="review.html?date=${encodeURIComponent(p.source_date)}#${encodeURIComponent(e.id)}">Link to proposal</a></div></article>`;
}
function render(){if(!data)return;
 const nav=document.querySelector('#topbar a[href="review.html"]');
 if(nav){const pending=rows.filter(r=>state(r).status==='pending').length,incomplete=data.products.some(p=>['missing','stale'].includes(p.status));nav.textContent=verified?`Review (${pending}${incomplete?' · incomplete':''})`:'Review (?)';}
 $('clock').textContent=`Checked ${fmt(data.server_time)}`;
 $('counts').innerHTML=['pending','approved_review','rejected','expired'].map(s=>`<div class="metric"><b>${rows.filter(r=>state(r).status===s).length}</b><span>${labels[s]}</span></div>`).join('');
 const filtered=rows.filter(r=>($('filter').value==='all'||state(r).status===$('filter').value)&&($('product').value==='all'||r.envelope.payload.product===$('product').value)).sort((a,b)=>Date.parse(a.envelope.payload.review_deadline)-Date.parse(b.envelope.payload.review_deadline));
 const focus=document.activeElement?.dataset;
 $('queue').innerHTML=filtered.length?filtered.map(card).join(''):'<div class="empty">No proposals in this view. Check feed status above and All proposals before treating this as a stand-down.</div>';
 $('queue').querySelectorAll('[data-decision]').forEach(b=>b.addEventListener('click',()=>submitDecision(b.dataset.id,b.dataset.decision)));
 if(focus?.id)[...$('queue').querySelectorAll('[data-decision]')].find(b=>b.dataset.id===focus.id&&b.dataset.decision===focus.decision)?.focus();
 $('audit').innerHTML=events.length?[...events].sort((a,b)=>b.at.localeCompare(a.at)).map(e=>`<li><strong>${e.decision==='approve_review'?'Review approved':'Proposal rejected'}</strong> · ${esc(e.actor)}<small>${fmt(e.at)} · revision ${e.revision} · ${e.scope==='review_and_stage'?'orders queued for '+e.accounts.map(a=>a==='primary'?'Primary':'PA').join(' + '):'no orders submitted'}</small><div class="mono">${esc(e.proposal_id)}</div>${e.reason?`<p>${esc(e.reason)}</p>`:''}</li>`).join(''):'<li>No review decisions recorded for this date.</li>';
}
async function jsonRequest(path,options={}){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),12000);try{const r=await fetch(path,{cache:'no-store',redirect:'error',credentials:'same-origin',...options,signal:controller.signal});if(!r.ok||!(r.headers.get('content-type')||'').includes('application/json')){let detail=`HTTP ${r.status}. Sign in and reload if your session expired.`;try{detail=(await r.json()).error||detail;}catch{}throw Error(detail);}return await r.json();}finally{clearTimeout(timer);}}
async function load(){const token=++generation;verified=false;render();$('refresh').disabled=true;
 try{const date=$('review-date').value||new URLSearchParams(location.search).get('date')||'';const fresh=await jsonRequest(`/review-inbox${date?'?date='+encodeURIComponent(date):''}`);const freshRows=fresh.products.flatMap(p=>p.proposals||[]);await Promise.all(freshRows.map(r=>C.verify(r.envelope)));if(token!==generation)return;
  data=fresh;rows=freshRows;events=fresh.products.flatMap(p=>p.events||[]);staging=fresh.products.flatMap(p=>p.staging||[]);serverOffset=Date.parse(fresh.server_time)-Date.now();verified=true;$('review-date').value=fresh.date;$('review-date').max=fresh.today;$('sign-in').hidden=true;
  $('feed-health').innerHTML=fresh.products.map(p=>`<p><strong>${p.product==='pitch'?'Daily Pitch':'Daily Seasonal'}:</strong> ${p.status==='missing'?'Not published / unavailable — not a stand-down':p.status==='stale'?'Awaiting confirmed delivery reconciliation — decisions blocked':p.stand_down?`Delivered stand-down: ${esc(p.stand_down_reason)}`:`${p.status==='historical'?'Historical (read-only)':'Delivery confirmed'} · ${p.proposals.filter(r=>r.current).length} current proposal(s)`}</p>`).join('')+(fresh.read_only?'<p><strong>History is read-only. Return to today for pending reviews.</strong></p>':'');
  render();
 }catch(e){if(token!==generation)return;verified=false;$('feed-health').textContent='Review state could not be verified. Decisions are disabled; previously loaded details are for reference only.';$('sign-in').hidden=false;notice(e.message);render();}
 finally{if(token===generation)$('refresh').disabled=false;}
}
function savedIntent(){try{return JSON.parse(sessionStorage.getItem('review-pending-intent'));}catch{return null;}}
async function submitDecision(id,decision){
 if(!verified||busy||data.read_only)return;
 const row=rows.find(r=>r.envelope.id===id);if(!row?.current||state(row).status!=='pending'||decision==='approve_review'&&row.staging_block)return;
 const e=row.envelope,pending=savedIntent();
 const command=pending?.proposal_id===id&&pending.decision===decision?pending:{id:crypto.randomUUID(),product:e.payload.product,proposal_id:e.id,proposal_hash:e.hash,expected_revision:state(row).revision,decision,reason:decision==='reject'?'Declined':'',confirmed:true,stage:decision==='approve_review'};
 busy=true;render();
 try{
  try{sessionStorage.setItem('review-pending-intent',JSON.stringify(command));}catch{}
  const result=await jsonRequest(`/review-inbox?date=${encodeURIComponent(e.payload.source_date)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command)});
  if(result.event?.id!==command.id||result.execution!==(command.stage?'queued':'not_submitted'))throw Error('Unexpected confirmation. Check status.');
  try{sessionStorage.removeItem('review-pending-intent');}catch{}$('filter').value='all';await load();
  notice(command.stage?'Approved. Orders queued; staging status appears on the idea.':'Passed. No orders submitted.');
 }catch(error){verified=false;notice(`${error.message} Check status before retrying; your decision may already be recorded.`);}
 finally{busy=false;render();}
}
renderNav('review.html');$('filter').addEventListener('change',render);$('product').addEventListener('change',render);$('review-date').addEventListener('change',()=>{$('filter').value='all';load();});$('refresh').addEventListener('click',load);
if(location.hash)$('filter').value='all';await load();if(location.hash)document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
setInterval(()=>{if(!busy)load();},10000);document.addEventListener('visibilitychange',()=>{if(!document.hidden&&!busy)load();});
