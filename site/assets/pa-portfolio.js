/* PA account view on the existing private Portfolio page. Pure API is also
   exported for Node regression tests; no capital is uploaded or URL-encoded. */
(function () {
  'use strict';
  const sum = a => a.reduce((x, y) => x + y, 0);
  const sd = a => {
    if (a.length < 2) return null;
    const mean = sum(a) / a.length;
    return Math.sqrt(sum(a.map(x => (x - mean) ** 2)) / (a.length - 1));
  };
  function validate(data) {
    const c = data && data.config;
    if (!data || data.version !== 1 || !c || !(c.primary_anchor > 0) ||
        !(c.risk_multiplier > 0) || !Number.isFinite(c.primary_anchor) ||
        !Number.isFinite(c.risk_multiplier) || !Array.isArray(data.dates) || !data.dates.length ||
        !Array.isArray(data.trades)) throw new Error('Invalid PA replay payload');
    if (data.dates.some((d, i) => !/^\d{4}-\d{2}-\d{2}$/.test(d) || (i && d <= data.dates[i - 1])))
      throw new Error('Invalid PA session calendar');
    for (const t of data.trades) {
      if (![t.stage, t.entry, t.exit, t.primary_qty].every(Number.isInteger) ||
          t.stage < 0 || t.entry < t.stage || t.exit < t.entry || t.exit >= data.dates.length ||
          t.primary_qty < 1 || !Number.isFinite(t.risk_per_share) || t.risk_per_share < 0 ||
          !Array.isArray(t.unit_pnl) || t.unit_pnl.length !== t.exit - t.entry + 1 ||
          !t.unit_pnl.every(Number.isFinite)) throw new Error('Invalid PA trade MTM vector');
    }
  }
  function quantity(primaryQty, equity, config) {
    return Math.floor(primaryQty * config.risk_multiplier * equity / config.primary_anchor);
  }
  function replay(data, capital, {compound = true, start = data.dates[0]} = {}) {
    validate(data);
    if (!Number.isFinite(capital) || capital <= 0) throw new Error('Enter positive PA starting equity');
    const begin = data.dates.findIndex(d => d >= start);
    if (begin < 0) throw new Error('Start date is after the replay history');
    const pnl = new Array(data.dates.length).fill(0);
    const staged = new Map();
    for (const t of data.trades) {
      if (t.stage < begin) continue; // start with no inherited holdings
      if (!staged.has(t.stage)) staged.set(t.stage, []);
      staged.get(t.stage).push(t);
    }
    const equity = [], returns = [], dd = [], orders = [];
    let current = capital, peak = capital;
    for (let i = begin; i < data.dates.length; i++) {
      // All orders staged today see the SAME prior-session equity. Holding
      // quantities do not change when later daily equity moves.
      const sizingCapital = compound ? Math.max(current, 0) : capital;
      for (const t of staged.get(i) || []) {
        const qty = quantity(t.primary_qty, sizingCapital, data.config);
        if (qty < 1) continue;
        const tradePnl = sum(t.unit_pnl) * qty;
        orders.push({...t, qty, sizing_equity: sizingCapital,
          pa_risk: qty * t.risk_per_share, pa_pnl: tradePnl});
        t.unit_pnl.forEach((v, k) => { pnl[t.entry + k] += v * qty; });
      }
      returns.push(current > 0 ? pnl[i] / current : null);
      current += pnl[i]; // allow insolvency to remain visible; do not hide losses
      peak = Math.max(peak, current);
      equity.push(current);
      dd.push(current / peak - 1);
    }
    return {dates: data.dates.slice(begin), equity, returns, dd,
      pnl: pnl.slice(begin), orders, capital, compound};
  }
  function periodReturns(run, key) {
    const groups = [];
    let prior = run.capital;
    run.dates.forEach((date, i) => {
      const label = key(date);
      let g = groups[groups.length - 1];
      if (!g || g.period !== label) {
        g = {period: label, start: prior, end: run.equity[i], dates: []};
        groups.push(g);
      }
      g.end = run.equity[i]; g.dates.push(date); prior = run.equity[i];
    });
    return groups.map(g => ({...g, return: g.start > 0 ? g.end / g.start - 1 : null}));
  }
  function week(date) {
    const d = new Date(date + 'T00:00:00Z');
    d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
    return d.toISOString().slice(0, 10);
  }
  function metrics(run) {
    const r = run.returns.filter(Number.isFinite);
    const daily = sd(r), weekly = periodReturns(run, week);
    const monthly = periodReturns(run, d => d.slice(0, 7));
    const annual = periodReturns(run, d => d.slice(0, 4));
    const trough = run.dd.indexOf(Math.min(...run.dd));
    let peak = -1;
    for (let i = 0; i < trough; i++) if (Math.abs(run.dd[i]) < 1e-12) peak = i;
    let recovery = -1;
    for (let i = trough + 1; i < run.dd.length; i++) if (run.dd[i] >= -1e-12) { recovery = i; break; }
    const end = run.equity[run.equity.length - 1];
    return {daily, weeklyApprox: daily == null ? null : daily * Math.sqrt(5),
      monthlyApprox: daily == null ? null : daily * Math.sqrt(21),
      annualized: daily == null ? null : daily * Math.sqrt(252),
      weeklyEmpirical: sd(weekly.slice(1, -1).map(g => g.return).filter(Number.isFinite)),
      monthlyEmpirical: sd(monthly.slice(1, -1).map(g => g.return).filter(Number.isFinite)),
      total: end / run.capital - 1,
      cagr: end > 0 ? (end / run.capital) ** (252 / run.dates.length) - 1 : null,
      maxDD: run.dd[trough], peak: peak < 0 ? 'Starting equity' : run.dates[peak],
      trough: run.dates[trough], recovery: recovery < 0 ? null : run.dates[recovery],
      underwaterSessions: recovery < 0 ? run.dates.length - 1 - peak : recovery - peak,
      monthly, annual, insolvent: run.equity.some(v => v <= 0)};
  }
  const api = {quantity, validate, replay, metrics, periodReturns};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;

  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const pct = value => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
  const money = value => fmt.money(value, 0);
  const table = (headers, rows) => `<div class="pa-table-wrap"><table class="tbl"><thead><tr>${headers.map((h,i) => `<th${i===0?' class="l"':''}>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((v,i) => `<td${i===0?' class="l"':''}>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  let payload = null, latest = null;
  function plot(id, traces, ytitle, extra = {}) {
    return Plotly.newPlot(id, traces, {paper_bgcolor:'transparent', plot_bgcolor:'transparent',
      font:{color:'#9daec0'}, margin:{l:60,r:25,t:25,b:40},
      xaxis:{type:'date',gridcolor:'#27303e'}, yaxis:{title:ytitle,gridcolor:'#27303e'}, ...extra}, {responsive:true, displayModeBar:false});
  }
  function render() {
    const root = document.getElementById('paPortfolio');
    const status = document.getElementById('paStatus');
    if (!payload || root.hidden) return;
    try {
      const capital = Number(document.getElementById('paCapital').value);
      const compound = document.getElementById('paSizing').value === 'scaled';
      latest = replay(payload, capital, {compound, start:document.getElementById('paStart').value});
      try {
        sessionStorage.setItem('paPortfolioCapital', String(capital));
        sessionStorage.setItem('paPortfolioOptions', JSON.stringify({
          start:document.getElementById('paStart').value, compound}));
      } catch (_) {}
      const m = metrics(latest);
      status.textContent = `${latest.orders.length.toLocaleString()} PA orders; ${latest.dates[0]}–${latest.dates.at(-1)}. ${compound ? 'New orders use prior-session PA equity' : 'New orders use fixed PA starting equity'}.`;
      const items = [['Total return', pct(m.total)], ['Annual growth', pct(m.cagr)],
        ['Annualized volatility', pct(m.annualized)], ['Worst EOD drawdown', pct(m.maxDD)]];
      document.getElementById('paKpis').innerHTML = items.map(([k,v]) => `<div class="card"><span class="cap">${k}</span><h2>${v}</h2></div>`).join('');
      document.getElementById('paRisk').innerHTML = table(['Return SD','Square-root approximation','Empirical calendar periods'],[
        ['Weekly',pct(m.weeklyApprox),pct(m.weeklyEmpirical)],
        ['Monthly',pct(m.monthlyApprox),pct(m.monthlyEmpirical)],
        ['Annualized',pct(m.annualized),'Daily SD × √252'],
      ]) + `<p class="cap">Square-root scaling assumes weak serial dependence. Empirical SD uses nonoverlapping interior calendar weeks/months; the first and last period are omitted. Volatility is an outcome of the sized replay, not a configured volatility target.</p>`;
      document.getElementById('paEpisode').textContent = `Worst drawdown: ${pct(m.maxDD)}, from ${m.peak} to ${m.trough}. ${m.recovery ? 'Recovered ' + m.recovery : 'Still below that peak at sample end'}; ${m.underwaterSessions} exchange sessions underwater. ${m.insolvent ? 'This scenario exhausted its capital.' : ''}`;
      plot('paEquity',[{x:latest.dates,y:latest.equity,type:'scatter',mode:'lines',name:'PA equity',line:{color:'#58c4ac'}}],'PA equity ($)');
      plot('paDD',[{x:latest.dates,y:latest.dd.map(v=>v*100),type:'scatter',mode:'lines',fill:'tozeroy',name:'Drawdown',line:{color:'#e7957b'}}],'Drawdown (%)');
      plot('paMonthly',[{x:m.monthly.map(v=>v.period+'-01'),y:m.monthly.map(v=>v.return==null?null:v.return*100),type:'bar',name:'Monthly return',marker:{color:m.monthly.map(v=>v.return>=0?'#58c4ac':'#e7957b')}}],'Monthly return (%)');
      document.getElementById('paAnnual').innerHTML = table(['Year','Return','Ending equity'],m.annual.map(g=>[g.period,pct(g.return),money(g.end)]));
      document.getElementById('paTrades').innerHTML = table(['Signal','Entry','Exit / mark','Status','Strategy','Ticker','Side','Main unsplit qty','PA qty','PA risk budget','PA P&L'],
        latest.orders.slice().reverse().slice(0,200).map(t=>[t.signal,payload.dates[t.entry],payload.dates[t.exit],t.exit_type,t.strategy,t.ticker,t.direction,t.primary_qty,t.qty,money(t.pa_risk),money(t.pa_pnl)]));
    } catch (error) { status.textContent = error.message; latest = null;
      for (const id of ['paKpis','paRisk','paEpisode','paEquity','paDD','paMonthly','paAnnual','paTrades']) document.getElementById(id).replaceChildren(); }
  }
  function download() {
    if (!latest) return;
    const headers = ['Signal','Staging date','Entry','Exit / mark','Status','Strategy','Ticker','Direction','Main unsplit qty','PA qty','Sizing equity','Risk budget','PnL'];
    const cells = v => '"' + String(v).replace(/"/g,'""') + '"';
    const rows = latest.orders.map(t=>[t.signal,payload.dates[t.stage],payload.dates[t.entry],payload.dates[t.exit],t.exit_type,t.strategy,t.ticker,t.direction,t.primary_qty,t.qty,t.sizing_equity,t.pa_risk,t.pa_pnl]);
    const url = URL.createObjectURL(new Blob([[headers,...rows].map(r=>r.map(cells).join(',')).join('\r\n')],{type:'text/csv'}));
    const a = document.createElement('a'); a.href=url; a.download='pa-portfolio-replay.csv'; a.click(); URL.revokeObjectURL(url);
  }
  document.addEventListener('DOMContentLoaded', async () => {
    const root = document.getElementById('paPortfolio');
    if (!root) return;
    renderNav('pa-portfolio.html');
    root.innerHTML = `<h1>Portfolio — PA</h1><p class="sub">Current staging formula, applied to a fresh engine replay of the strategy book.</p>
      <div class="filters"><label>PA starting equity ($) <input id="paCapital" type="number" min="1" step="0.01" placeholder="Enter privately"></label>
      <label>Replay start <input id="paStart" type="date"></label>
      <label>Sizing <select id="paSizing"><option value="scaled">Scaled with PA equity</option><option value="flat">Fixed PA equity</option></select></label>
      <button class="btn" id="paCalculate">Calculate</button><button class="btn ghost" id="paExport">Download PA trades</button></div>
      <p class="cap" id="paFormula"></p><p class="cap">Your capital stays in this browser tab. Start with an empty book. Orders staged before the replay start are excluded; changing the start date begins a new capital scenario.</p>
      <p class="cap" id="paStatus">Loading the PA replay…</p>
      <div id="paKpis" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px"></div>
      <section><h2>Equity and drawdowns</h2><div class="chart" id="paEquity"></div><div class="chart" id="paDD"></div><p class="cap" id="paEpisode"></p></section>
      <section><h2>Realized portfolio risk</h2><div id="paRisk"></div></section>
      <section><h2>Calendar returns</h2><div class="chart" id="paMonthly"></div><div id="paAnnual"></div></section>
      <section><h2>PA trade sizing</h2><p class="cap">Latest 200 orders shown; download includes all. Risk budget is an assigned sizing amount, not a guaranteed loss ceiling.</p><div id="paTrades"></div></section>
      <p class="cap" id="paLimits"></p>`;
    document.getElementById('paCalculate').addEventListener('click',render);
    document.getElementById('paExport').addEventListener('click',download);
    for (const id of ['paCapital','paStart','paSizing']) document.getElementById(id).addEventListener('change',render);
    try {
      const snapshot = await loadSiteSnapshot(async meta => {
        if (meta.pa_portfolio_version !== 1 || !meta.payloads.pa_portfolio) throw new Error('PA replay is not available in this deployment');
        return {pa: await fetchSitePayload(meta,'data/pa_portfolio.json')};
      });
      payload = snapshot.pa; validate(payload);
      setAsof(`PA replay through ${payload.asof}`);
      const c = payload.config;
      document.getElementById('paFormula').textContent = `PA qty = floor(Main unsplit qty × ${c.risk_multiplier} × PA equity / ${money(c.primary_anchor)}). Main per-strategy daily risk cap: ${c.per_strategy_daily_cap_bps} bps before PA scaling. PA OVS remains one far-target order.`;
      const start = document.getElementById('paStart'); start.value = payload.dates[0]; start.min=payload.dates[0]; start.max=payload.dates.at(-1);
      document.getElementById('paLimits').textContent = payload.limitations.join(' ');
      try {
        document.getElementById('paCapital').value = sessionStorage.getItem('paPortfolioCapital') || '';
        const options = JSON.parse(sessionStorage.getItem('paPortfolioOptions') || '{}');
        if (typeof options.start === 'string' && options.start >= start.min && options.start <= start.max)
          start.value = options.start;
        if (options.compound === false) document.getElementById('paSizing').value = 'flat';
      } catch (_) {}
      document.getElementById('paStatus').textContent = 'Enter PA starting equity to calculate. Prior-session EOD equity approximates live premarket NLV.';
      if (document.getElementById('paCapital').value) render();
    } catch (error) { document.getElementById('paStatus').textContent = error.message; }
  });
})();
