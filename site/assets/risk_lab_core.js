// Pure research-view calculations. No scoring or portfolio changes.
export const WINDOWS = [5, 10, 21];
export const finite = x => typeof x === 'number' && Number.isFinite(x);
export const datesOf = (d, s) => Array.isArray(s?.dates) ? s.dates : d.dates || [];
export const studyOf = d => d.forward_returns?.['63d'] || {};
export const anchorsOf = (d, sample = 'reduced') => sample === 'all'
  ? validFullSample(d) ? d.return_samples.all.episode_dates : []
  : sample === 'nonoverlap' ? validNonoverlapSample(d) ? d.return_samples.nonoverlap.episode_dates : []
  : sample === 'reduced' ? studyOf(d).episode_dates || [] : [];
export function validFullSample(d) {
  const s=d.return_samples, study=studyOf(d);
  return !!(s?.version===1 && s.asof===d.asof && s.score_asof===d.sizing_state?.asof
    && finite(s.current_score) && finite(study.current_score)
    && Math.abs(s.current_score-study.current_score)<1e-9
    && s.band_low===study.band_low && s.band_high===study.band_high
    && Array.isArray(s.all?.episode_dates));
}
export function validNonoverlapSample(d) {
  if (!validFullSample(d) || d.return_samples.nonoverlap_gap !== Math.max(...WINDOWS)
    || !Array.isArray(d.return_samples.nonoverlap?.episode_dates)) return false;
  const dates=seriesFor(d)?.dates || d.dates || [], all=d.return_samples.all.episode_dates;
  let previous=-Infinity;
  return d.return_samples.nonoverlap.episode_dates.every(date=>{
    const pos=dates.indexOf(date), valid=pos>=0 && all.includes(date) && pos-previous>Math.max(...WINDOWS);
    previous=pos;
    return valid;
  });
}
// Roughly six trading months. Backfill context at either history boundary.
export function episodeChartRange(dates, selectedDate = null) {
  if (!dates?.length) return null;
  const pos=selectedDate ? dates.indexOf(selectedDate) : -1;
  const span=126, last=dates.length-1;
  let start=pos>=0 ? Math.max(0,pos-span/2) : Math.max(0,last-span);
  let end=Math.min(last,start+span);
  start=Math.max(0,end-span);
  return {start,end,range:[dates[start],dates[end]],matched:pos>=0};
}
export function assertSharedRisk(d) {
  const banned=['banded_strategies','basis','throttled','threshold','throttle_on',
    'gap_to_threshold','days_in_state','episodes','exposure','sleeve'];
  if (!d.shared_redacted || d.trade_console || banned.some(k=>k in (d.sizing_state||{})))
    throw new Error('Shared risk inputs failed the privacy check.');
}
export function era(date) {
  if (date < '2026-07-02') return 'Reconstructed history';
  if (date < '2026-09-17') return 'Recorded · legacy dial';
  if (date < '2026-09-18') return 'Recorded · NYSE v1';
  return 'Recorded · NYSE EMA5';
}
export function seriesFor(d, asset = 'SPY') {
  const series = d.price_explorer?.series?.[asset];
  if (series) return {...series, dates: datesOf(d, series)};
  return asset === 'SPY' && d.spy_series
    ? {...d.spy_series, dates: datesOf(d, d.spy_series)} : null;
}
export function episodeOutcome(d, date, window = 21, asset = 'SPY', sample = 'reduced') {
  const recorded=validFullSample(d)?d.return_samples?.[sample]?.outcomes?.[window]?.find(r=>r.date===date):null;
  if(asset==='SPY' && recorded) return {...recorded};
  const s = seriesFor(d, asset);
  if (!s) return {date, status: 'unavailable', value: null};
  const pos = s.dates.indexOf(date);
  if (pos < 0 || !finite(s.close[pos]) || s.close[pos] <= 0)
    return {date, status: 'unavailable', value: null};
  const end = pos + window;
  if (end >= s.dates.length) return {date, status: 'incomplete', value: null,
    available: s.dates.length - 1 - pos};
  if (!finite(s.close[end])) return {date, status: 'unavailable', value: null};
  return {date, status: 'complete', value: s.close[end] / s.close[pos] - 1,
    endDate: s.dates[end], start: s.close[pos], finish: s.close[end]};
}
export function pathsFor(d, window = 21, sample = 'reduced') {
  const s = seriesFor(d);
  if (!s) return [];
  return anchorsOf(d, sample).map(date => {
    const pos = s.dates.indexOf(date);
    if (pos < 0 || !finite(s.close[pos]) || s.close[pos] <= 0) return null;
    const available = Math.min(window, s.dates.length - pos - 1);
    return {date, complete: available === window && finite(s.close[pos + window]),
      x: Array.from({length: available + 1}, (_, i) => i),
      y: s.close.slice(pos, pos + available + 1)
        .map(v => finite(v) ? v / s.close[pos] * 100 : null)};
  }).filter(Boolean);
}
export function quantile(values, q) {
  const a = values.filter(finite).sort((a,b) => a-b);
  if (!a.length) return null;
  const p = (a.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
  return a[lo] + (a[hi] - a[lo]) * (p - lo);
}
export function pathEnvelope(paths, window) {
  return Array.from({length: window + 1}, (_, i) => {
    const values = paths.map(p => p.y[i]).filter(finite);
    return {day: i, n: values.length, median: quantile(values,.5),
      q25: quantile(values,.25), q75: quantile(values,.75)};
  });
}
export function downsideStudy(d, sample = 'all') {
  const s=d.downside_samples, returns=d.return_samples;
  if(!validFullSample(d) || !s || s.version!==1 || s.asof!==returns.asof
    || s.score_asof!==returns.score_asof || s.current_score!==returns.current_score
    || s.band_low!==returns.band_low || s.band_high!==returns.band_high
    || JSON.stringify(s[sample]?.episode_dates)!==JSON.stringify(anchorsOf(d,sample)))return null;
  return s[sample];
}
export function drawdownSample(d, window, threshold, sample = 'all') {
  const saved=downsideStudy(d,sample)?.windows?.[window];
  const rows=saved?.outcomes || [];
  const all=rows.filter(r=>r.status==='complete');
  const breached=all.filter(r=>r.breaches?.[threshold]===true);
  const iv=breached.map(r=>r.iv_change_points).filter(finite);
  return {rows,all,breached,n:all.length,hits:breached.length,
    selected:anchorsOf(d,sample).length,
    pending:rows.filter(r=>r.status==='incomplete').length,
    unavailable:saved?rows.filter(r=>r.status==='unavailable').length:anchorsOf(d,sample).length,
    rate:all.length?breached.length/all.length:null,
    ivN:iv.length,ivMean:iv.length?iv.reduce((a,b)=>a+b,0)/iv.length:null,ivMedian:quantile(iv,.5)};
}
export function returnSample(d, window, sample = 'reduced') {
  const all = anchorsOf(d, sample).map(date => episodeOutcome(d, date, window, 'SPY', sample));
  const completed = all.filter(r => r.status === 'complete');
  return {all, completed, positive: completed.filter(r => r.value > 0).length,
    negative: completed.filter(r => r.value < 0).length,
    flat: completed.filter(r => r.value === 0).length,
    pending: all.filter(r => r.status === 'incomplete').length};
}
export function returnStats(d, window, sample = 'reduced') {
  if (validFullSample(d) && d.return_samples?.[sample]) return d.return_samples[sample].returns?.[window] || null;
  return sample==='reduced' ? studyOf(d).returns?.[window] || null : null;
}
export function filterOutcomes(rows, filter) {
  if (filter === 'up') return rows.filter(r => r.status === 'complete' && r.value > 0);
  if (filter === 'down') return rows.filter(r => r.status === 'complete' && r.value < 0);
  if (filter === 'pending') return rows.filter(r => r.status !== 'complete');
  return rows;
}
export function downsideCells(d, sample = 'all') {
  const ad=d.atr_downside || {};
  return WINDOWS.flatMap(window=>[1,2,3,5].map(threshold=>{
    const dd=drawdownSample(d,window,threshold,sample);
    const value=finite(dd.rate)?100*dd.rate:null;
    const baseline=ad.baseline?.[`${window}d`]?.[threshold];
    const delta=finite(value)&&finite(baseline)?value-baseline:null;
    return {window,threshold,value,baseline,delta,hits:dd.hits,n:dd.n,
      elevated:finite(delta)&&delta>=8};
  }));
}
export function signalAt(detail, date) {
  if (!detail || !date) return null;
  return (detail.periods || []).some(p => date >= p[0] && date <= p[1]);
}
export function signalChanges(d) {
  const dates = d.dates || [];
  if (dates.length < 2) return {from: null, to: dates.at(-1), changes: []};
  const from = dates.at(-2), to = dates.at(-1);
  return {from, to, changes: (d.signals || []).flatMap(s => {
    const detail = d.signal_detail?.[s.name];
    const values = detail?.metric?.values;
    // A missing endpoint is unknown, never a newly cleared rule.
    if (!detail || !finite(values?.at(-1)) || !finite(values?.at(-2))) return [];
    const before = signalAt(detail, from), after = signalAt(detail, to);
    return before === after ? [] : [{name: s.name, on: after}];
  })};
}
export function dialDelta(d, sessions) {
  const s = d.sizing_state?.spark;
  if (!s || !finite(s.ma?.at(-1))) return null;
  const end = s.ma.length - 1, start = end - sessions;
  if (start < 0 || !finite(s.ma[start])) return null;
  return {value: s.ma[end] - s.ma[start], from: s.dates[start], to: s.dates[end],
    crossesEra: era(s.dates[start]) !== era(s.dates[end])};
}
export function check(label, value, operator, threshold, unit = '') {
  const known = finite(value);
  const operations = {'<':(a,b)=>a<b, '<=':(a,b)=>a<=b, '>':(a,b)=>a>b, '>=':(a,b)=>a>=b};
  return {label, value: known ? value : null, operator, threshold, unit,
    pass: known && operations[operator] ? operations[operator](value, threshold) : null};
}
export function ruleChecks(d, name) {
  const detail = d.signal_detail?.[name] || {}, value = detail.current?.value;
  const summary = detail.current?.summary || '';
  const reported = regex => {
    const match = summary.match(regex);
    return match ? Number(match[1]) : null;
  };
  const spy = seriesFor(d), closes = spy?.close || [];
  const last = closes.at(-1), year = closes.slice(-252).filter(finite);
  const near = finite(last) && year.length === 252 ? 100*(1-last/Math.max(...year)) : null;
  const ma50 = closes.slice(-50);
  const above50 = finite(last) && ma50.length === 50 && ma50.every(finite)
    ? last - ma50.reduce((a,b)=>a+b,0)/50 : null;
  const nearCheck = () => check('SPY below 252-session closing high', near, '<=', 2, '%');
  let checks = [], note = 'Current snapshot values; checklist does not recalculate the dial.';
  if (name === 'VIX Range Compression') {
    checks = [check('21-session VIX range percentile',value,'<',15,'pctile'),
      check('Consecutive compressed sessions (reported)',reported(/compressed (\d+)d/),'>=',10,'sessions'),
      check('VIX close',d.vol_kpi?.vix,'>',13),
      check('Five-session VIX change (reported)',reported(/5d change ([+-]?[\d.]+)/),'<',0,'pts')];
  } else if (name === 'Distribution Dominance') {
    checks = [check('Distribution / accumulation ratio',value,'>',3.75),
      check('SPY close minus 50-session average',above50,'>',0), nearCheck()];
    note = 'Base rule needs all three conditions. Elevated branch: ratio > 6 AND within 2% of the high; the 50-session filter is not required for that branch.';
  } else if (name === 'Defensive Leadership') {
    checks = [check('50-session risk-on minus risk-off spread',value,'<',-10,'pp'),
      check('200-session spread (rounded reported value)',reported(/200d spread: ([+-]?[\d.]+)/),'<',0,'pp'),nearCheck()];
    note = 'All conditions are required. Within 1% of the high gives the elevated tier. The 200-session value is rounded in the source summary; use the recorded signal status at boundaries.';
  } else if (name === 'Low Absorption Ratio') {
    checks = [check('Absorption ratio percentile',value,'<',10,'pctile'),nearCheck()];
  } else if (name === 'Seasonal Rank Divergence') {
    checks = [check('Risk-off minus risk-on seasonal spread',value,'>',10,'pp'),nearCheck()];
  } else if (name === 'Dispersion') {
    checks = [check('Composite dispersion percentile',value,'>',85,'pctile'),
      check('Sessions since a 10% correction (signal basis)',null,'>=',200,'sessions')];
    note = 'Also requires at least 200 sessions since the signal’s expanding-high correction measure, then more than 10 sessions between retained fires. The context strip uses a different correction basis; it is not substituted here.';
  } else if (name === 'Equity P/C Complacency') {
    checks = [check('Equity put/call percentile',value,'<',10,'pctile')];
    note = 'Requires a fresh put/call observation. This component contributes to the 5-session composite only; it has no direct weight in the main 63-session dial.';
  } else if (name === 'NYSE Net Highs') {
    checks = [check('Five-session EMA of net new highs',value,'<',0,'issues'),
      check('Distance below 252-session closing high',near,'<=',3,'%'),
      check('Complete breadth history',null,'>=',77,'sessions')];
    note = 'Full severity within 2%; partial severity from 2–3%. EMA ≥ 0 resets the NYSE memory. Breadth completeness is not exposed by this snapshot; it remains unknown here. The recorded main score includes a floor at the base dial.';
  }
  return {checks, note, reported: summary, version: detail.rule_version};
}
export function resolveState(d, params = new URLSearchParams()) {
  const assets = d.price_explorer?.assets || ['SPY'];
  const horizon = Number(params.get('window'));
  const threshold = Number(params.get('atr'));
  const asset = assets.includes(params.get('asset')) ? params.get('asset') : 'SPY';
  const requested = params.get('episode');
  const requestedSample=params.get('sample');
  const sample = ['nonoverlap','reduced'].includes(requestedSample) && validNonoverlapSample(d)
    ? 'nonoverlap' : requestedSample==='reduced' ? 'reduced'
    : validFullSample(d) ? 'all' : 'reduced';
  const selected = anchorsOf(d,sample).includes(requested) ? requested : null;
  return {window: WINDOWS.includes(horizon) ? horizon : 21,
    threshold: [1,2,3,5].includes(threshold) ? threshold : 2,
    asset, sample, selected,
    outcomeFilter: ['all','up','down','pending'].includes(params.get('outcomes')) ? params.get('outcomes') : 'all',
    overviewRange: ['1Y','3Y','All'].includes(params.get('overview')) ? params.get('overview') : 'All',
    range: ['1Y','3Y','All'].includes(params.get('range')) ? params.get('range')
      : params.get('range') === 'Episode' && selected ? 'Episode' : '1Y'};
}
