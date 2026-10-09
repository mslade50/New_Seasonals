const {WINDOWS, finite, datesOf, studyOf, anchorsOf, era, seriesFor,
  episodeOutcome, pathsFor, pathEnvelope, drawdownSample, signalAt,
  signalChanges, dialDelta, ruleChecks, resolveState, returnSample, returnStats, filterOutcomes, downsideCells, validFullSample, validNonoverlapSample, episodeChartRange, assertSharedRisk} = await import(new URL(`./risk_lab_core.js${new URL(import.meta.url).search}`, import.meta.url));

const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, decimals=1) => finite(v) ? v.toLocaleString('en-US',{minimumFractionDigits:decimals,maximumFractionDigits:decimals}) : '—';
const pct = (v, decimals=1, signed=false) => finite(v) ? `${signed && v>0?'+':''}${num(v*100,decimals)}%` : '—';
const signed = v => finite(v) ? `${v>0?'+':''}${num(v)}` : '—';
const color = v => finite(v) ? (v<0?'down':v>0?'up':'') : '';
const CONFIG = {responsive:true,displaylogo:false,displayModeBar:false,scrollZoom:false};
const PALETTE = ['#dba869','#83b6d9','#9e94d4','#91c7ab','#d38d9d','#8caacb','#c3be7f','#b99484'];
let data, state, mode = document.body.dataset.mode || 'personal';
let sequence = 0;
const activeAnchors = () => anchorsOf(data,state.sample);
const sampleLabel = () => state.sample==='all'?'All matching dates':state.sample==='nonoverlap'?'Non-overlapping episodes':'Legacy overlap-reduced episodes';
const sampleMethod = () => state.sample==='all'
  ? 'All qualifying dates are included, even consecutive days. Nearby dates share future market moves, so this larger count is not a count of independent events.'
  : state.sample==='nonoverlap'
  ? 'Retains the first qualifying date, then requires more than 21 trading sessions before the next retained date. The same dates are used across 5/10/21; their outcome windows do not overlap. Separate windows do not guarantee independent events.'
  : 'Legacy sample: more than 10 trading sessions between retained dates reduces overlap; 21-session windows can still overlap.';

function layout(extra={}) {
  const base={paper_bgcolor:'transparent',plot_bgcolor:'transparent',
    font:{family:'Inter, Segoe UI, sans-serif',color:'#a8b5c9',size:11},
    margin:{l:45,r:30,t:16,b:35},height:310,
    xaxis:{gridcolor:'#283346',showgrid:false,zeroline:false},
    yaxis:{gridcolor:'#28334677',zeroline:false},
    legend:{orientation:'h',y:1.13,x:0,font:{size:10}},
    hoverlabel:{bgcolor:'#141c28',bordercolor:'#283346'}};
  return {...base,...extra,xaxis:{...base.xaxis,...extra.xaxis},yaxis:{...base.yaxis,...extra.yaxis}};
}
function plot(id,traces,extra={},config={}) {
  if (!window.Plotly) return Promise.resolve();
  return Plotly.react($(id),traces,layout(extra),{...CONFIG,...config});
}
function vline(date,color='#efc478',label=null) {
  return {type:'line',xref:'x',yref:'paper',x0:date,x1:date,y0:0,y1:1,
    line:{color,width:1.3,dash:label?'dot':'solid'}};
}
function buttons(id,values,current,handler) {
  $(id).innerHTML = values.map(v=>`<button type="button" aria-pressed="${v===current}" data-value="${v}">${esc(v)}${typeof v==='number'?'d':''}</button>`).join('');
  for (const button of $(id).querySelectorAll('button')) button.addEventListener('click',()=>handler(button.dataset.value));
}
function keepState() {
  const url=new URL(location.href);
  for(const [key,value] of Object.entries({window:state.window,atr:state.threshold,asset:state.asset,episode:state.selected,range:state.range,outcomes:state.outcomeFilter,overview:state.overviewRange,sample:state.sample})) {
    if(value===null)url.searchParams.delete(key);else url.searchParams.set(key,value);
  }
  history.replaceState(null,'',url);
}
function showInspector(title,html) {
  $('inspectorTitle').textContent=title;
  $('inspectorBody').innerHTML=html;
  $('inspector').showModal();
}
function facts(items) {
  return `<dl class="inspector-facts">${items.map(([a,b])=>`<dt>${esc(a)}</dt><dd>${b}</dd>`).join('')}</dl>`;
}
function summary() {
  const sz=data.sizing_state || {}, study=studyOf(data);
  const changes=dialDelta(data,1);
  const active=(data.signals || []).filter(s=>s.on).length;
  $('liveSummary').innerHTML=`<div class="summary-grid">
    <article class="summary-card featured"><div class="label">MAIN RISK DIAL <span class="pill">Latest snapshot</span></div><div class="number">${num(sz.score)} <small>/ 100</small></div><div class="micro">${changes?`${signed(changes.value)} points since ${esc(changes.from)}`:'Prior observation unavailable'} · ${esc(sz.asof || data.asof)}</div></article>
    <article class="summary-card"><div class="label">ACTIVE SIGNALS</div><div class="number">${active} <small>/ ${(data.signals || []).length}</small></div><div class="micro">Activity count, not a score decomposition<br>Includes the 5d-only put/call signal</div></article>
    <article class="summary-card"><div class="label">RETURN SAMPLE</div><div class="number" id="analogCount"></div><div class="micro" id="analogCountNote"></div></article></div>`;
  if(mode==='personal' && sz.throttle_on!=null) {
    $('liveSummary').insertAdjacentHTML('beforeend',`<details class="policy"><summary>Personal policy in this snapshot · ${sz.throttle_on?'reduced-size bands active':'full-size bands'} · ${sz.days_in_state??'—'} sessions in state</summary><div class="policy-grid">${(sz.throttled || []).map(s=>`<span class="pill">${esc(s.strategy)} · ${num(s.mult,2)}×</span>`).join('') || 'No banded strategy is reduced.'}</div><p class="micro">Threshold ${num(sz.threshold,0)} · ${num(Math.abs(sz.gap_to_threshold))} points ${sz.score>=sz.threshold?'above':'below'} · Exposure leg ${num(sz.exposure?.mult,2)}× as of ${esc(sz.exposure?.asof || 'unavailable')}. This is a snapshot of existing policy, not a new sizing recommendation.</p></details>`);
  }
}
function provenance() {
  $('provenanceSummary').textContent=`Snapshot: ${data.asof} · Saved dial: ${data.sizing_state?.asof || 'unavailable'} · Observation dates & model history`;
  const ad=data.atr_downside || {};
  $('provenanceBody').innerHTML=facts([
    ['Market observation',esc(data.asof)],['Saved main dial',esc(data.sizing_state?.asof || 'unavailable')],
    ['Snapshot generated',esc(data.built_at)],['Data source','Authoritative cloud snapshot; saved main dial and adjusted market prices'],
    ['Dial history','Before 2026-07-02: frozen reconstruction. From 2026-07-02: recorded observations.'],
    ['Model changes','2026-09-17: recorded NYSE main score introduced. 2026-09-18: five-session EMA trigger.'],
    ['Signal history','Reconstructed under the snapshot rules; not an archive of previously published signals.'],
    ['Downside baseline',`${esc(ad.data_from || '?')} through ${esc(ad.data_through || '?')} · precomputed full-history baseline`],
    ['Availability','Per-input timestamps and historical contribution breakdowns are not supplied for every component. Missing values stay unavailable.'],
    ['SPY market context',`${num(data.spy_last,2)} · ${esc(data.price_ctx?.regime_label || '')}. SPY is the broad-market ETF used to measure these historical returns; its price supplies chart context, not another risk score.`],
    ['Volatility context',`VIX ${num(data.vol_kpi?.vix)} / VIX3M ${num(data.vol_kpi?.vix3m)} · ratio ${num(data.vol_kpi?.term_ratio,2)} as of ${esc(data.vol_kpi?.asof || data.asof)}. These compare options-implied volatility over roughly one month and three months; a ratio above 1 means near-term volatility is higher. This ratio is background context, not the dial itself. <a href="https://www.cboe.com/tradable_products/vix/term_structure" target="_blank" rel="noopener noreferrer">Cboe definitions ↗</a>`]]);
  const c=signalChanges(data), d1=dialDelta(data,1), d5=dialDelta(data,5);
  $('changes').innerHTML=`<strong>What changed?</strong><span>Saved dial: <b>${d1?signed(d1.value):'—'} / 1 session</b> · ${d5?signed(d5.value):'—'} / 5 sessions</span><span class="quiet">Signal reconstruction ${esc(c.from || '—')} → ${esc(c.to || '—')}:</span>${c.changes.length?c.changes.map(s=>`<button class="text-button" data-open-signal="${esc(s.name)}">${esc(s.name)} ${s.on?'activated':'cleared'} ↗</button>`).join(''):'<span class="quiet">no on/off transitions</span>'}${d1?.crossesEra||d5?.crossesEra?'<span class="pill">Comparison crosses a model era</span>':''}`;
}
function renderEvidence() {
  const w=state.window, study=studyOf(data), st=returnStats(data,w,state.sample);
  const sample=drawdownSample(data,w,state.threshold,state.sample);
  const outcomes=returnSample(data,w,state.sample), completed=outcomes.completed.length;
  const total=outcomes.all.length, unavailable=total-completed-outcomes.pending;
  $('analogCount').innerHTML=`${total} <small>${state.sample==='all'?'matching dates':'episodes'}</small>`;
  $('analogCountNote').textContent=`${sampleLabel()} · ${completed} completed ${w}-session returns`;
  for(const id of ['sampleMode','downsideSampleMode'])$(id).value=state.sample;
  $('sampleSummary').innerHTML=`<div class="sample-heading"><strong>${esc(sampleLabel())}</strong><span>Dial ${num(study.band_low)}–${num(study.band_high)} · ±5 points · selected before outcomes</span></div><div class="sample-counts"><div><b>${total}</b><span>selected ${state.sample==='all'?'dates':'episodes'}</span></div><div class="completed"><b>${completed}</b><span>completed ${w}d returns</span></div><div><b>${outcomes.pending}</b><span>incomplete</span></div><div><b>${unavailable}</b><span>unavailable</span></div></div><p class="micro"><b>Return statistics use n = ${completed}, not ${total}.</b> ${sampleMethod()} Available dial / price history: ${esc(data.dates?.[0] || '?')}–${esc(data.asof)}. Matching dates span ${esc(activeAnchors()[0] || '—')}–${esc(activeAnchors().at(-1) || '—')}.</p>`;
  buttons('windowButtons',WINDOWS,w,value=>{state.window=Number(value);refresh();});
  $('cohortDescription').innerHTML=validFullSample(data)?`<b>${anchorsOf(data,'all').length} matching dates → ${validNonoverlapSample(data)?anchorsOf(data,'nonoverlap').length:'unavailable'} non-overlapping episodes.</b> The sample selector changes return cards, statistics, red match lines, downside frequencies, date inspection and paths together. Matches share the main dial level, not an identical combination of signals.`:`<b>Full sample unavailable in this snapshot.</b> Showing ${anchorsOf(data).length} original overlap-reduced episodes. A matching cloud-generated sample is required to enable all dates.`;
  const cards=[
    {id:'returns',label:`${w}-session mean return`,value:pct(st?.mean,2,true),class:color(st?.mean),sub:`n = ${completed} completed · baseline ${pct(st?.uncond_mean,2,true)}`},
    {id:'distribution',label:`${w}-session median return`,value:pct(st?.median,2,true),class:color(st?.median),sub:`n = ${completed} · middle 50%: ${pct(st?.q25)} to ${pct(st?.q75)}`},
    {id:'returns',label:'Positive / negative / flat',value:`${outcomes.positive} / ${outcomes.negative} / ${outcomes.flat}`,sub:`All ${completed} completed ${w}-session outcomes`},
    {id:'returns',label:'Completed / all matches',value:`${completed} / ${outcomes.all.length}`,sub:`${outcomes.pending} incomplete · ${outcomes.all.length-completed-outcomes.pending} unavailable · all retained in the list`}
  ];
  $('evidenceCards').innerHTML=cards.map(c=>`<button class="evidence-card" data-inspect="${c.id}"><div class="label">${esc(c.label)} ⓘ</div><div class="number ${c.class||''}"${c.id==='distribution'?' style="font-size:21px"':''}>${esc(c.value)}</div><div class="micro">${esc(c.sub)}</div></button>`).join('');
  $('returnsTable').innerHTML=`<table><thead><tr><th>Window</th><th>Mean</th><th>Baseline</th><th>Difference</th><th>Completed n / selected</th><th>Incomplete</th></tr></thead><tbody>${WINDOWS.map(window=>{const r=returnStats(data,window,state.sample),o=returnSample(data,window,state.sample);return `<tr class="${window===w?'selected':''}"><td><button class="text-button" data-window="${window}">${window} sessions ↗</button></td><td class="${color(r?.mean)}">${pct(r?.mean,2,true)}</td><td>${pct(r?.uncond_mean,2,true)}</td><td>${r?`${signed(100*(r.mean-r.uncond_mean))} pp`:'—'}</td><td><b>n = ${o.completed.length}</b> / ${o.all.length}</td><td>${o.pending}</td></tr>`;}).join('')}</tbody></table><p class="micro">${esc(sampleLabel())}. Each row uses its own completed n. Below ${study.min_samples || 5} completed observations: summaries withheld. Overlapping dates and windows are not independent events.</p>`;
  const ad=data.atr_downside || {}, cells=downsideCells(data,state.sample);
  const available=cells.filter(c=>finite(c.delta)), elevated=cells.filter(c=>c.elevated);
  $('downsideSuite').classList.toggle('elevated',elevated.length>0);
  $('downsideHeadline').innerHTML=available.length?`<div class="risk-alert"><strong>${elevated.length?'Elevated downside across the suite':'Downside compared with the long-run baseline'}</strong><span>${elevated.length} of ${available.length} available window / ATR combinations are at least 8 percentage points above baseline.</span></div>`:'<p class="quiet">Downside outcomes unavailable for this snapshot. A matching downside sample is required.</p>';
  $('downsideDescription').innerHTML=`<b>${esc(sampleLabel())}: the same ${total} selected dates as the returns and timeline.</b> Each rate counts threshold breaches divided by all completed, eligible matches, including those that never breached. A positive ending return can still include an intraday drawdown. Incomplete windows and missing low/ATR inputs are excluded from that window’s denominator.`;
  $('downsideCounts').innerHTML=WINDOWS.map(window=>{const dd=drawdownSample(data,window,state.threshold,state.sample);return `<div><b>n = ${dd.n} / ${dd.selected}</b><span>eligible ${window}d / selected</span><span>${dd.pending} incomplete · ${dd.unavailable} unavailable</span><span>Baseline n = ${ad.baseline_n?.[`${window}d`]??'—'}</span></div>`;}).join('');
  $('atrTable').innerHTML=`<table class="risk-matrix"><thead><tr><th>Window / eligible matches</th>${[1,2,3,5].map(k=>`<th>≥${k} ATR</th>`).join('')}</tr></thead><tbody>${WINDOWS.map(window=>{const dd=drawdownSample(data,window,state.threshold,state.sample);return `<tr class="${window===w?'selected':''}"><td>${window} sessions <span class="baseline">n = ${dd.n} of ${dd.selected} selected</span></td>${[1,2,3,5].map(k=>{const c=cells.find(c=>c.window===window&&c.threshold===k);return `<td class="${c.elevated?'risk-elevated':''}"><button class="cell-button ${window===w&&k===state.threshold?'active':''}" data-atr-window="${window}" data-atr-threshold="${k}" aria-label="Inspect ${window}-session downside of at least ${k} ATR, ${c.hits} of ${c.n} eligible matches breached">${finite(c.value)?`${num(c.value,0)}%`:'—'}</button><span class="baseline">${c.hits} / ${c.n} hit</span><span class="baseline">${finite(c.baseline)?`${num(c.baseline,0)}% baseline`:'—'}</span><span class="risk-delta">${finite(c.delta)?`${signed(c.delta)} pp`:''}</span></td>`;}).join('')}</tr>`;}).join('')}</tbody></table>`;
  const selected=cells.find(c=>c.window===w&&c.threshold===state.threshold);
  $('atrSelection').innerHTML=`<b>${w} sessions / ≥${state.threshold} ATR:</b> ${finite(selected.value)?`${num(selected.value)}%`:'unavailable'} · <b>${selected.hits} hits / ${selected.n} eligible matches</b> from ${total} selected dates. ${sample.n-sample.hits} eligible matches did not hit this threshold. Long-run baseline ${finite(selected.baseline)?`${num(selected.baseline)}%`:'unavailable'}${finite(selected.delta)?`; ${signed(selected.delta)} percentage points difference`:''}. Same ±5 sample as the rest of the page.`;
  $('outcomeFilter').value=state.outcomeFilter;
  const visible=filterOutcomes(outcomes.all,state.outcomeFilter).slice().reverse();
  $('episodeListCap').textContent=`${sampleLabel()} · showing ${visible.length} of ${total} selected dates · ${completed} completed returns, ${outcomes.positive} positive, ${outcomes.negative} negative, ${outcomes.flat} flat, ${outcomes.pending} incomplete, ${unavailable} unavailable. No downside threshold is applied to this list.`;
  $('episodesTable').innerHTML=`<table><thead><tr><th>Matched close</th><th>History basis</th><th>${w}d SPY return</th><th>Outcome</th><th>Max low-touch downside</th><th>Availability</th></tr></thead><tbody>${visible.map(r=>{const date=r.date,dd=sample.all.find(r=>r.anchor_date===date);return `<tr class="${date===state.selected?'selected':''}"><td><button class="text-button" data-episode="${esc(date)}">${esc(date)} ↗</button></td><td class="micro">${esc(era(date))}</td><td class="${color(r.value)}">${pct(r.value,2,true)}</td><td class="${color(r.value)}">${r.status!=='complete'?'Pending / unavailable':r.value>0?'Positive':r.value<0?'Negative':'Flat'}</td><td>${finite(dd?.max_drawdown_atr)?`${num(dd.max_drawdown_atr,2)} ATR`:'—'}</td><td>${r.status==='complete'?'Complete':r.status==='incomplete'?`${r.available} / ${w} sessions available`:'Price coverage unavailable'}</td></tr>`;}).join('') || '<tr><td colspan="6" class="quiet">No matched episodes fit this outcome filter.</td></tr>'}</tbody></table><p class="micro">The sample is chosen before outcomes. Positive close-to-close returns can coexist with intraday drawdowns. Returns measure adjusted close to adjusted close. Downside details use the same selected dates and require a completed low-price window and a valid starting ATR.</p>`;
}
function inspect(kind) {
  const w=state.window,s=studyOf(data),r=returnStats(data,w,state.sample),dd=drawdownSample(data,w,state.threshold,state.sample);
  const outcomes=returnSample(data,w,state.sample);
  const coverage=`${data.dates?.[0] || '?'} through ${data.asof}`;
  if(kind==='returns'||kind==='distribution') {
    showInspector(kind==='returns'?`${w}-session SPY returns`:'Historical return distribution',
      `<p class="quiet">${esc(sampleLabel())}. Selecting a date, outcome filter or chart range does not change this sample.</p><div class="inspector-stat">${kind==='returns'?pct(r?.mean,2,true):`${pct(r?.q25)} to ${pct(r?.q75)}`}</div>`+facts([
        ['Selection',`Main dial ${num(s.current_score)} ±5 points; ${sampleMethod()}`],
        ['Completed n / selected',`n = ${outcomes.completed.length} / ${outcomes.all.length}`],
        ['Incomplete / unavailable',`${outcomes.pending} / ${outcomes.all.length-outcomes.completed.length-outcomes.pending}`],
        ['Full / non-overlapping match counts',`${anchorsOf(data,'all').length} matching dates / ${validNonoverlapSample(data)?anchorsOf(data,'nonoverlap').length:'unavailable'} non-overlapping episodes`],
        ['Coverage',esc(coverage)],['Unconditional mean',pct(r?.uncond_mean,2,true)],
        ['Return baseline n',String(r?.baseline_n ?? 'Not supplied in this saved summary')],
        ['Difference from baseline',r?`${signed(100*(r.mean-r.uncond_mean))} percentage points`:'Unavailable'],
        ['Median',pct(r?.median,2,true)],['Negative outcomes',pct(r?.pct_neg,0)],
        ['10th / 90th percentiles',`${pct(r?.q10,2)} / ${pct(r?.q90,2)}`],
        ['Worst / best',`${pct(r?.worst,2)} / ${pct(r?.best,2)}`]])+
      '<p class="rule-note">These are descriptive historical outcomes, not a forecast interval or calibrated confidence measure. Long windows can overlap, and a small sample can be driven by one episode. Individual matched dates are listed on the page.</p>');
  } else if(kind==='drawdown'||kind==='iv') {
    const listed=kind==='iv'?dd.breached:dd.rows;
    showInspector(kind==='iv'?'VIX behavior after a threshold breach':`All selected outcomes · ≥${state.threshold} ATR within ${w} sessions`,
      `<p class="quiet">${esc(sampleLabel())} · same dial ±5 dates as returns and the timeline.</p><div class="inspector-stat">${kind==='iv'?(finite(dd.ivMean)?`${signed(dd.ivMean)} VIX points`:'Unavailable'):pct(dd.rate,1)}</div>`+facts([
        ['Selected / eligible',`${dd.selected} selected dates / ${dd.n} completed, eligible windows`],
        ['Incomplete / unavailable',`${dd.pending} / ${dd.unavailable}`],
        ['Hit / did not hit',`${dd.hits} / ${dd.n-dd.hits}`],
        ['Frequency',`${dd.hits} hits ÷ ${dd.n} eligible matches = ${pct(dd.rate,1)}`],
        ['VIX observations among breaches',String(dd.ivN)],
        ['Conditional VIX mean / median',`${signed(dd.ivMean)} / ${signed(dd.ivMedian)} points`],
        ['Downside definition',`Matched SPY close minus lowest subsequent intraday low in ${w} sessions, floored at zero, divided by Wilder ATR(14) fixed at the matched close`],
        ['VIX timing',`${esc(data.downside_samples?.iv_basis || 'VIX')} from the following session through the date of the worst SPY low`]])+
      `<p class="rule-note">Downside frequencies include every eligible match, whether or not it hit the threshold. Only the VIX average is conditional on breaches; missing VIX does not change the downside denominator.</p><h3>${kind==='iv'?'Matches that hit the threshold':'Every selected match'}</h3><div class="table-scroll"><table><thead><tr><th>Matched date</th><th>Downside</th><th>Threshold outcome</th></tr></thead><tbody>${listed.map(row=>`<tr><td><button class="text-button" data-dialog-episode="${esc(row.anchor_date)}">${esc(row.anchor_date)} ↗</button></td><td>${finite(row.max_drawdown_atr)?`${num(row.max_drawdown_atr,2)} ATR`:'—'}</td><td>${row.status==='complete'?(row.breaches?.[state.threshold]?'Hit':'Did not hit'):row.status==='incomplete'?'Incomplete':'Unavailable'}</td></tr>`).join('') || '<tr><td colspan="3">No eligible observations supplied.</td></tr>'}</tbody></table></div>`);
    for(const b of $('inspectorBody').querySelectorAll('[data-dialog-episode]'))b.addEventListener('click',()=>{$('inspector').close();selectEpisode(b.dataset.dialogEpisode,true);});
  } else {
    const ad=data.atr_downside || {};
    showInspector('Downside sample & method',facts([
      ['Selected sample',`${esc(sampleLabel())} · ${dd.selected} dates selected before outcomes, main dial ${num(s.band_low)}–${num(s.band_high)}`],
      ['Shared dates','The return cards, downside frequencies, timeline and episode list follow the same sample selector.'],
      ['Frequency',`${dd.hits} breaches / all ${dd.n} eligible ${w}-session windows; non-breaches stay in the denominator.`],
      ['Incomplete / unavailable',`${dd.pending} / ${dd.unavailable}. Missing lows or starting ATR make downside unavailable, even if a closing return exists.`],
      ['Baseline n',`${ad.baseline_n?.[`${w}d`]??'—'} completed ${w}-session market windows`],
      ['Baseline coverage',`${esc(ad.data_from || '?')} through ${esc(ad.data_through || '?')}. This long-run baseline can cover a longer period than the dial sample.`],
      ['Overlap',sampleMethod()]])+
      '<p class="rule-note">The sample is selected by the dial before looking at outcomes. VIX after a threshold breach is a separate, conditional summary.</p>');
  }
}

function selectionOptions() {
  $('episodeSelect').innerHTML='<option value="">Latest market context</option>'+activeAnchors().slice().reverse().map(date=>`<option value="${esc(date)}">${esc(date)} · ${esc(era(date))}</option>`).join('');
  $('assetSelect').innerHTML=(data.price_explorer?.assets || ['SPY']).map(a=>`<option value="${esc(a)}">${esc(a)}</option>`).join('');
  const options=`<option value="all" ${validFullSample(data)?'':'disabled'}>All matching dates (${validFullSample(data)?anchorsOf(data,'all').length:'unavailable'})</option><option value="nonoverlap" ${validNonoverlapSample(data)?'':'disabled'}>Non-overlapping episodes (${validNonoverlapSample(data)?anchorsOf(data,'nonoverlap').length:'unavailable'})</option>${state.sample==='reduced'?`<option value="reduced">Legacy overlap-reduced episodes (${anchorsOf(data).length})</option>`:''}`;
  for(const id of ['sampleMode','downsideSampleMode'])$(id).innerHTML=options;
}
function selectEpisode(date,scroll=false) {
  if(date!==null && !activeAnchors().includes(date))return;
  state.selected=date;
  state.range=date?'Episode':state.range==='Episode'?'1Y':state.range;
  refresh();
  if(scroll)$('episode-review').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth',block:'start'});
}
function renderReview() {
  const date=state.selected || data.asof,w=state.window,sz=data.sizing_state || {};
  $('reviewTitle').textContent=state.selected?`Episode · ${date}`:'Explore the latest market context';
  $('episodeSelect').value=state.selected || '';$('assetSelect').value=state.asset;
  const anchors=activeAnchors(),i=anchors.indexOf(state.selected);
  $('prevEpisode').disabled=!anchors.length || i===0;
  $('nextEpisode').disabled=!anchors.length || i===anchors.length-1;
  $('clearSelection').disabled=!state.selected;
  $('selectionBanner').innerHTML=state.selected?`<b>Historical inspection: ${esc(date)}.</b> ${esc(era(date))}. Latest reading remains ${num(sz.score)} as of ${esc(sz.asof)}. Signal history is reconstructed; this is not historical page replay.`:`<b>Latest context: ${esc(date)}.</b> Select a historical match from the table, marker, or episode menu. The evidence above remains based on the latest dial.`;
  const spark=sz.spark || {},index=spark.dates?.indexOf(date),score=index>=0?spark.ma[index]:null;
  const out=state.selected?episodeOutcome(data,date,w,'SPY',state.sample):null;
  const row=drawdownSample(data,w,state.threshold,state.sample).rows.find(r=>r.anchor_date===date);
  const active=(data.signals || []).filter(s=>signalAt(data.signal_detail?.[s.name],date));
  $('episodeDetail').innerHTML=`<h3>${state.selected?'Matched episode':'Latest saved reading'}</h3><div class="micro">${esc(date)} · ${esc(era(date))}</div><div class="detail-number">${num(score)} <span class="quiet">main dial</span></div>
    <div class="detail-row"><span>${w}-session SPY return</span><b class="${color(out?.value)}">${pct(out?.value,2,true)}</b></div>
    <div class="detail-row"><span>Worst low-touch downside</span><b>${finite(row?.max_drawdown_atr)?`${num(row.max_drawdown_atr,2)} ATR`:'—'}</b></div>
    <div class="detail-row"><span>Sessions to worst low</span><b>${finite(row?.sessions_to_low)?num(row.sessions_to_low,0):'—'}</b></div>
    <div class="detail-row"><span>VIX close → peak*</span><b>${finite(row?.iv_start_close)?num(row.iv_start_close):'—'} → ${finite(row?.iv_peak)?num(row.iv_peak):'—'}</b></div>
    <p class="micro">${out?.status==='incomplete'?`Only ${out.available} of ${w} future sessions are available. Incomplete outcomes are not included in the completed summary.`:state.selected?'* Peak VIX through the window’s worst SPY low, not the full-window VIX maximum.':'Future outcomes are unavailable for the latest observation.'}${state.selected&&row?.status==='unavailable'?' Downside inputs are unavailable for this window.':''}</p>
    <div class="detail-heading">SIGNALS ON · RECONSTRUCTED</div><div class="detail-signals">${active.length?active.map(s=>`<button class="text-button" data-open-signal="${esc(s.name)}">${esc(s.name)} ↗</button>`).join(''):'<span class="quiet">No active periods in the supplied history.</span>'}</div>`;
}
async function renderMatches() {
  const s=seriesFor(data), anchors=activeAnchors();
  if(!s)return;
  const start=state.overviewRange==='All'?s.dates[0]:`${Number(data.asof.slice(0,4))-(state.overviewRange==='3Y'?3:1)}${data.asof.slice(4)}`;
  const markers=anchors.filter(date=>s.dates.includes(date));
  const shapes=anchors.map(date=>vline(date,'#f07575'));
  if(state.selected)shapes.push({...vline(state.selected),line:{color:'#efc478',width:2.5}});
  const prices=s.close.filter((v,i)=>finite(v)&&s.dates[i]>=start);
  const lo=prices.length?Math.min(...prices):0, hi=prices.length?Math.max(...prices):1, pad=Math.max((hi-lo)*.08,1);
  buttons('overviewRangeButtons',['1Y','3Y','All'],state.overviewRange,value=>{state.overviewRange=value;keepState();renderMatches();});
  await plot('matchesChart',[
    {x:s.dates,y:s.close,name:'SPY close',line:{color:'#7caaff',width:1.5},hovertemplate:'%{x}<br>SPY %{y:.2f}<extra></extra>'},
    {x:markers,y:markers.map(date=>s.close[s.dates.indexOf(date)]),customdata:markers,name:'Selected sample matches',mode:'markers',type:'scatter',marker:{symbol:'triangle-down',size:10,color:'#f07575'},hovertemplate:'Analog %{x}<br>SPY %{y:.2f}<extra>Click to inspect</extra>'}
  ],{height:330,margin:{l:45,r:20,t:30,b:35},uirevision:`analogs-${state.overviewRange}`,xaxis:{range:[start,data.asof]},yaxis:{range:[lo-pad,hi+pad],title:{text:'SPY',font:{size:10}}},shapes});
  $('matchesSubhead').textContent=`${sampleLabel()} · ${anchors.length} selected dates marked by red vertical lines`;
  $('matchesCaption').textContent=`${anchors.length} selected analog dates across ${s.dates[0]}–${data.asof}. ${sampleMethod()} Click a red triangle to inspect; a gold line marks the selected date. Episode selection does not zoom this overview.`;
  const chart=$('matchesChart');
  if(!chart._selectionBound&&typeof chart.on==='function') {
    chart.on('plotly_click',e=>{const date=e.points?.find(p=>p.data.name==='Selected sample matches')?.customdata;if(date)selectEpisode(date,true);});
    chart._selectionBound=true;
  }
}
async function renderPrices() {
  const s=seriesFor(data,state.asset);
  if(!s)return;
  const context=episodeChartRange(s.dates,state.selected);
  if(!context)return;
  const {start,end,range}=context, lows=(s.low || s.close).slice(start,end+1).filter(finite),highs=(s.high || s.close).slice(start,end+1).filter(finite);
  const extent=lows.length&&highs.length?[Math.min(...lows),Math.max(...highs)]:null;
  const yr=extent?[extent[0]-(extent[1]-extent[0])*.06,extent[1]+(extent[1]-extent[0])*.06]:undefined;
  const traces=s.open&&s.high&&s.low?[{type:'candlestick',x:s.dates,open:s.open,high:s.high,low:s.low,close:s.close,name:state.asset,
    increasing:{line:{color:'#79d8b2',width:1},fillcolor:'#79d8b2'},decreasing:{line:{color:'#e19696',width:1},fillcolor:'#e19696'}}]:[{x:s.dates,y:s.close,name:state.asset,line:{color:'#7caaff'}}];
  const markers=activeAnchors().filter(date=>s.dates.includes(date));
  traces.push({x:markers,y:markers.map(date=>(s.high || s.close)[s.dates.indexOf(date)]),customdata:markers,
    type:'scatter',mode:'markers',name:'Selected sample matches',marker:{symbol:'triangle-down',size:8,color:'#efc478'},hovertemplate:'Match %{x}<extra></extra>'});
  await plot('priceChart',traces,{height:350,showlegend:false,xaxis:{range,rangeslider:{visible:false},rangebreaks:[{bounds:['sat','mon']}]},yaxis:{range:yr,gridcolor:'#28334677',title:{text:state.asset,font:{size:11}}},shapes:context.matched?[vline(state.selected)]:[]});
  $('priceCaption').textContent=`Six-month context (about 126 trading sessions): ${range[0]}–${range[1]}. ${context.matched?'Centered on the match where history allows; recent matches use more earlier history.':state.selected?'The match date is unavailable for this asset; showing its latest context.':'Latest available history.'} Select a match marker to inspect. Changing the asset preserves the SPY-selected sample.`;
  const chart=$('priceChart');
  if(!chart._selectionBound&&typeof chart.on==='function') {
    chart.on('plotly_click',e=>{const date=e.points?.find(p=>p.data.name==='Selected sample matches')?.customdata;if(date)selectEpisode(date);});
    chart._selectionBound=true;
  }
}
async function renderDial() {
  const s=data.sizing_state?.spark || {},spy=seriesFor(data);
  let start=state.range==='All'?s.dates?.[0]:`${Number(data.asof.slice(0,4))-(state.range==='3Y'?3:1)}${data.asof.slice(4)}`;
  let end=data.asof;
  if(state.range==='Episode'&&state.selected) {
    const context=episodeChartRange(data.dates,state.selected);
    if(context)[start,end]=context.range;
  }
  const visiblePrices=(spy?.close || []).filter((v,i)=>finite(v)&&spy.dates[i]>=start&&spy.dates[i]<=end);
  const lo=visiblePrices.length?Math.min(...visiblePrices):0,hi=visiblePrices.length?Math.max(...visiblePrices):1;
  const pad=Math.max((hi-lo)*.08,1);
  const shapes=[vline('2026-07-02','#698199','era'),vline('2026-09-17','#698199','era'),vline('2026-09-18','#698199','era')];
  if(state.selected)shapes.push(vline(state.selected));
  if(mode==='personal'&&finite(data.sizing_state?.threshold))shapes.push({type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:data.sizing_state.threshold,y1:data.sizing_state.threshold,line:{color:'#efc47877',dash:'dash',width:1}});
  buttons('rangeButtons',state.selected?['Episode','1Y','3Y','All']:['1Y','3Y','All'],state.range,value=>{state.range=value;keepState();renderDial();});
  await plot('dialChart',[{x:s.dates,y:s.ma,name:'Main dial · saved history',line:{color:'#efc478',width:2},hovertemplate:'%{x}<br>Main dial %{y:.1f}<extra></extra>'},
    {x:spy?.dates,y:spy?.close,name:'SPY',yaxis:'y2',line:{color:'#7caaff',width:1.3}}],{
    height:280,margin:{l:45,r:50,t:30,b:30},xaxis:{range:[start,end],showgrid:false},
    yaxis:{range:[0,Math.max(100,...(s.ma || []).filter(finite))],gridcolor:'#28334677',title:{text:'Dial',font:{size:10}}},
    yaxis2:{overlaying:'y',side:'right',showgrid:false,range:[lo-pad,hi+pad],title:{text:'SPY',font:{size:10}}},shapes});
}
function renderPaths() {
  if(!$('pathsDetails').open)return;
  const paths=pathsFor(data,state.window,state.sample),env=pathEnvelope(paths,state.window);
  const traces=[{x:env.map(e=>e.day),y:env.map(e=>e.q25),line:{width:0},showlegend:false,hoverinfo:'skip'},
    {x:env.map(e=>e.day),y:env.map(e=>e.q75),fill:'tonexty',fillcolor:'#7caaff16',line:{width:0},name:'Middle 50% of available paths',hoverinfo:'skip'}];
  paths.forEach((p,i)=>traces.push({x:p.x,y:p.y,name:`${p.date}${p.complete?'':' · partial'}`,showlegend:paths.length<=12||state.selected===p.date,customdata:p.x.map(()=>p.date),line:{color:state.selected===p.date?'#efc478':PALETTE[i%PALETTE.length],width:state.selected===p.date?2.5:1, dash:p.complete?'solid':'dot'},opacity:state.selected&&state.selected!==p.date?.55:.85,connectgaps:false,hovertemplate:`${p.date}<br>Session %{x}: %{y:.2f}<extra></extra>`}));
  traces.push({x:env.map(e=>e.day),y:env.map(e=>e.median),customdata:env.map(e=>e.n),name:'Median of available paths',line:{color:'#e7edf7',width:2},hovertemplate:'Session %{x}<br>Median %{y:.2f}<br>%{customdata} available paths<extra></extra>'});
  plot('pathsChart',traces,{height:370,margin:{l:45,r:20,t:35,b:60},xaxis:{title:{text:'Trading sessions after matched close'},range:[0,state.window]},yaxis:{title:{text:'SPY · matched close = 100'},gridcolor:'#28334677'},legend:{orientation:'h',y:-.2,font:{size:9}}});
  $('pathsCaption').textContent=`${sampleLabel()} · ${paths.length} paths begin at 100; ${paths.filter(p=>p.complete).length} reach session ${state.window}. Partial paths stop at their last observation. The median and middle 50% use available paths at each session, so membership can change. These are historical paths, not forecast or confidence bands.`;
}
function renderSignals() {
  $('signalsDate').textContent=`Current rules · as of ${data.asof}`;
  $('signalsList').innerHTML=(data.signals || []).map((signal,i)=>{
    const detail=data.signal_detail?.[signal.name] || {},rule=ruleChecks(data,signal.name), value=detail.current?.value;
    return `<details class="signal-card" id="signal-${i}" data-signal-index="${i}"><summary><strong>${esc(signal.name)}</strong><span class="pill ${signal.on?'on':'off'}">${esc(signal.badge || (signal.on?'ON':'OFF'))}</span><span class="signal-value">${num(value,detail.metric?.decimals??1)} ${esc(detail.metric?.unit || '')}</span></summary><div class="signal-content"><ul class="rule-list">${rule.checks.map(c=>`<li><span class="check-mark ${c.pass===true?'pass':c.pass===false?'fail':''}" aria-label="${c.pass===true?'Met':c.pass===false?'Not met':'Unavailable'}">${c.pass===true?'✓':c.pass===false?'×':'?'}</span><span>${esc(c.label)}</span><span class="rule-value">${finite(c.value)?num(c.value,Math.abs(c.value)<10?2:1):'Unavailable'} ${esc(c.unit)} · needs ${esc(c.operator)} ${c.threshold}</span></li>`).join('')}</ul><p class="rule-note">${esc(rule.note)}</p><p class="micro">Source summary: ${esc(rule.reported || 'Unavailable')}${rule.version?` · Rule ${esc(rule.version)}`:''}</p><div class="signal-chart" id="signal-chart-${i}"></div>${signal.on?signalDownside(signal.name):''}</div></details>`;
  }).join('');
  for(const details of $('signalsList').querySelectorAll('details'))details.addEventListener('toggle',()=>{if(details.open)renderSignalChart(Number(details.dataset.signalIndex));});
}
function signalDownside(name) {
  const ad=data.atr_downside || {},s=ad.signals?.[name];
  if(!s?.episode)return '';
  return `<h3>Downside after a fresh trigger</h3><p class="micro">${s.n_episodes} activation runs across ${esc(ad.data_from)}–${esc(ad.data_through)}. Completed counts for each window are not supplied; the total activation count is not a per-window denominator.</p><div class="table-scroll"><table><thead><tr><th>Window</th>${(ad.mults || []).map(k=>`<th>≥${k} ATR</th>`).join('')}</tr></thead><tbody>${WINDOWS.map(w=>`<tr><td>${w}d</td>${(ad.mults || []).map(k=>`<td>${finite(s.episode[`${w}d`]?.[k])?`${num(s.episode[`${w}d`][k],0)}%`:'—'}<span class="baseline">${finite(ad.baseline?.[`${w}d`]?.[k])?`${num(ad.baseline[`${w}d`][k],0)}% baseline`:'—'}</span></td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function renderSignalChart(i) {
  const signal=data.signals[i],detail=data.signal_detail?.[signal.name],metric=detail?.metric;
  if(!metric?.values){$(`signal-chart-${i}`).innerHTML='<p class="micro">Metric history unavailable.</p>';return;}
  const spy=seriesFor(data),context=state.selected?episodeChartRange(data.dates,state.selected):null;
  const start=context?context.range[0]:`${Number(data.asof.slice(0,4))-1}${data.asof.slice(4)}`;
  const end=context?context.range[1]:data.asof;
  const shapes=(detail.periods || []).map(p=>({type:'rect',xref:'x',yref:'paper',x0:p[0],x1:p[1],y0:0,y1:1,line:{width:0},fillcolor:'#efc4780b',layer:'below'}));
  if(state.selected)shapes.push(vline(state.selected));
  for(const threshold of metric.thresholds || [])shapes.push({type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:threshold.value,y1:threshold.value,line:{color:'#efc47888',width:1,dash:'dot'}});
  plot(`signal-chart-${i}`,[{x:data.dates,y:metric.values,name:metric.label,line:{color:'#efc478',width:1.5},connectgaps:false},
    {x:spy?.dates,y:spy?.close,name:'SPY',yaxis:'y2',line:{color:'#7caaff77',width:1}}],{height:245,xaxis:{range:[start,end],showgrid:false},yaxis:{gridcolor:'#28334677',title:{text:metric.unit,font:{size:10}}},yaxis2:{overlaying:'y',side:'right',showgrid:false},margin:{l:45,r:40,t:30,b:30},shapes});
}
function openSignal(name) {
  const i=(data.signals || []).findIndex(s=>s.name===name);if(i<0)return;
  const el=$(`signal-${i}`);el.open=true;renderSignalChart(i);el.scrollIntoView({block:'center',behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'});
}
async function refresh() {
  keepState();renderEvidence();renderReview();
  const run=++sequence;
  try { await Promise.all([renderPrices(),renderDial(),renderMatches()]); if(run===sequence)renderPaths(); }
  catch(error) { $('loadError').hidden=false; $('loadError').textContent=`Chart unavailable: ${error.message}`; }
  for(const details of $('signalsList').querySelectorAll('details[open]'))renderSignalChart(Number(details.dataset.signalIndex));
}
function bind() {
  for(const id of ['sampleMode','downsideSampleMode'])$(id).addEventListener('change',e=>{
    state.sample=e.target.value;
    if(state.selected&&!activeAnchors().includes(state.selected)) {
      state.selected=null;
      if(state.range==='Episode')state.range='1Y';
    }
    selectionOptions();refresh();
  });
  $('outcomeFilter').addEventListener('change',e=>{state.outcomeFilter=e.target.value;keepState();renderEvidence();});
  $('episodeSelect').addEventListener('change',e=>selectEpisode(e.target.value || null));
  $('assetSelect').addEventListener('change',e=>{state.asset=e.target.value;refresh();});
  $('clearSelection').addEventListener('click',()=>selectEpisode(null));
  $('prevEpisode').addEventListener('click',()=>{const a=activeAnchors(),i=a.indexOf(state.selected);selectEpisode(a[i<0?a.length-1:Math.max(0,i-1)]);});
  $('nextEpisode').addEventListener('click',()=>{const a=activeAnchors(),i=a.indexOf(state.selected);selectEpisode(a[i<0?0:Math.min(a.length-1,i+1)]);});
  $('pathsDetails').addEventListener('toggle',renderPaths);
  $('dialMethod').addEventListener('click',()=>inspect('dial'));
  $('closeInspector').addEventListener('click',()=>$('inspector').close());
  document.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    if(b.dataset.inspect)inspect(b.dataset.inspect);
    else if(b.dataset.episode)selectEpisode(b.dataset.episode,true);
    else if(b.dataset.window){state.window=Number(b.dataset.window);refresh();}
    else if(b.dataset.atrWindow){state.window=Number(b.dataset.atrWindow);state.threshold=Number(b.dataset.atrThreshold);refresh();}
    else if(b.dataset.openSignal)openSignal(b.dataset.openSignal);
  });
}
async function init() {
  try {
    if(mode==='personal') {
      renderNav('risk.html');
      const snapshot=await loadSiteSnapshot(async meta=>({risk:await fetchSitePayload(meta,'data/risk.json')}));
      data=snapshot.risk;
    } else {
      data=await fetchJSON('data/risk.json');
      assertSharedRisk(data);
    }
    const activeNames=new Set(['Distribution Dominance','VIX Range Compression','Defensive Leadership',
      'Low Absorption Ratio','Seasonal Rank Divergence','Dispersion','Equity P/C Complacency','NYSE Net Highs']);
    data={...data,signals:(data.signals||[]).filter(s=>activeNames.has(s.name))};
    setAsof(`as of ${data.asof}`);
    state=resolveState(data,new URLSearchParams(location.search));
    $('app').hidden=false;
    if(!window.Plotly)throw new Error('The local chart library is unavailable.');
    summary();provenance();selectionOptions();renderSignals();bind();await refresh();
    window.__riskLab={mode,asof:data.asof,ready:true};
  } catch(error) {
    $('loadError').hidden=false;$('loadError').textContent=`Risk inputs unavailable: ${error.message}`;
  }
}
init();
