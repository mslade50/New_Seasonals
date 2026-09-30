"""Causal, research-only rising-SMA200 range undercut/reclaim study."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scratch.fu_range_reclaim.execution import evaluate, block_ci


@dataclass(frozen=True)
class Rules:
    range_bars: int = 40
    frozen_bars: int = 5
    sma_bars: int = 200
    slope_bars: int = 20
    require_current_sma_rise: bool = True
    pivot_radius: int = 2
    support_tests: int = 3
    resistance_tests: int = 2
    test_spacing: int = 5
    support_span: int = 15
    support_tolerance_atr: float = .35
    resistance_tolerance_atr: float = .5
    rebound_atr: float = 1.
    min_width_atr: float = 2.
    max_width_atr: float = 8.
    max_width_pct: float = .15
    max_range_drift_atr: float = 1.
    max_sweep_atr: float = .5
    cooldown: int = 20


def features(df, rules=Rules()):
    d = df.copy()
    prev = d.Close.shift()
    tr = pd.concat([d.High-d.Low, (d.High-prev).abs(), (d.Low-prev).abs()], axis=1).max(axis=1)
    d['atr'] = tr.rolling(14, min_periods=14).mean()
    d['sma200'] = d.Close.rolling(rules.sma_bars, min_periods=rules.sma_bars).mean()
    return d


def separated_tests(candidates, spacing, high=None, floor=None, rebound=None):
    selected = []
    for p in candidates:
        p = int(p)
        if selected and p-selected[-1] < spacing:
            continue
        if selected and high is not None and np.max(high[selected[-1]+1:p]) < floor+rebound:
            continue
        selected.append(p)
    return selected


def detect(df, rules=Rules()):
    d = features(df, rules)
    r, n, freeze = rules.pivot_radius, rules.range_bars, rules.frozen_bars
    formation = n-freeze
    if formation <= 2*r or rules.test_spacing < 2:
        raise ValueError('Formation window or test spacing is invalid.')
    lo, hi, cl = (d[k].to_numpy(float) for k in ['Low', 'High', 'Close'])
    floor = d.Low.rolling(formation).min().shift(freeze+1)
    ceiling = d.High.rolling(formation).max().shift(freeze+1)
    ref = d.atr.shift(freeze+1)
    width = ceiling-floor
    trend = ((d.Close > d.sma200) & (d.Close.shift() > d.sma200.shift())
             & (d.sma200.shift() > d.sma200.shift(rules.slope_bars+1))
             & (floor > d.sma200.shift()))
    if rules.require_current_sma_rise:
        trend &= d.sma200 > d.sma200.shift()
    mask = (trend & (d.Low < floor) & (d.Low >= floor-rules.max_sweep_atr*ref)
            & (d.Close > floor) & (d.Close <= ceiling) & (d.Close >= (d.High+d.Low)/2)
            & (width >= rules.min_width_atr*ref) & (width <= rules.max_width_atr*ref)
            & (width/floor <= rules.max_width_pct))
    # These arrays identify pivot locations; only already-confirmed locations
    # (p + r <= formation cutoff) may be consumed below.
    pl, ph = pd.Series(True, index=d.index), pd.Series(True, index=d.index)
    for k in range(1, r+1):
        pl &= (d.Low < d.Low.shift(k)) & (d.Low < d.Low.shift(-k))
        ph &= (d.High > d.High.shift(k)) & (d.High > d.High.shift(-k))
    low_pivots, high_pivots = np.flatnonzero(pl), np.flatnonzero(ph)
    x = np.arange(n, dtype=float)-(n-1)/2
    last, found = -10**9, []
    for i in np.flatnonzero(mask.to_numpy()):
        if i-last < rules.cooldown:
            continue
        start, cutoff = i-n, i-freeze-1
        if start < r:
            continue
        support, resistance, a = float(floor.iloc[i]), float(ceiling.iloc[i]), float(ref.iloc[i])
        if np.min(lo[cutoff+1:i]) < support or np.max(hi[cutoff+1:i]) > resistance:
            continue
        prior = cl[start:i]
        if np.min(prior) < support or np.max(prior) > resistance:
            continue
        drift = float(np.dot(x, prior-prior.mean())/np.dot(x, x)*(n-1))
        if abs(drift) > rules.max_range_drift_atr*a:
            continue
        lp = low_pivots[(low_pivots >= start) & (low_pivots+r <= cutoff)]
        lp = lp[lo[lp] <= support+rules.support_tolerance_atr*a]
        support_i = separated_tests(lp, rules.test_spacing, hi, support, rules.rebound_atr*a)
        if len(support_i) < rules.support_tests or support_i[-1]-support_i[0] < rules.support_span:
            continue
        hp = high_pivots[(high_pivots >= start) & (high_pivots+r <= cutoff)]
        hp = hp[hi[hp] >= resistance-rules.resistance_tolerance_atr*a]
        resistance_i = separated_tests(hp, rules.test_spacing)
        if len(resistance_i) < rules.resistance_tests:
            continue
        found.append(dict(signal_i=int(i), range_start_i=int(start), known_i=int(cutoff),
                          support=support, resistance=resistance, atr_ref=a, sweep_low=float(lo[i]),
                          sma200=float(d.sma200.iloc[i]), sma200_prior=float(d.sma200.iloc[i-1]),
                          sma200_20ago=float(d.sma200.iloc[i-rules.slope_bars-1]),
                          support_tests=support_i, resistance_tests=resistance_i,
                          range_width_atr=(resistance-support)/a, range_drift_atr=drift/a))
        last = int(i)
    return found


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def chart(df, s, future=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    d = features(df)
    i = s['signal_i']
    start, end = max(0, s['range_start_i']-40), min(len(d), i+(21 if future else 1))
    sub = d.iloc[start:end]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k, b in enumerate(sub.itertuples()):
        color = '#11856a' if b.Close >= b.Open else '#c64b55'
        ax.vlines(k, b.Low, b.High, color=color, lw=.8)
        ax.add_patch(Rectangle((k-.3, min(b.Open, b.Close)), .6,
                               max(abs(b.Close-b.Open), b.Close*.00002), color=color))
    ax.plot(np.arange(len(sub)), sub.sma200, color='#b48910', lw=1.6, label='SMA 200')
    ax.hlines(s['support'], s['range_start_i']-start, end-start-1, color='#2565bc', ls='--', label='Established support')
    ax.hlines(s['resistance'], s['range_start_i']-start, end-start-1, color='#7351a7', ls='--', label='Range ceiling')
    for key, color, label, field in [('support_tests', '#2565bc', 'Prior support tests', 'Low'),
                                     ('resistance_tests', '#7351a7', 'Prior ceiling tests', 'High')]:
        idx = np.array(s[key], dtype=int)
        ax.scatter(idx-start, d[field].iloc[idx], s=75, facecolors='none', edgecolors=color, lw=1.7, label=label)
    ax.axvspan(s['known_i']-start, i-start, color='#2565bc', alpha=.055)
    ax.axvline(i-start, color='#172736', ls=':', label='Undercut / reclaim')
    ticks = np.arange(0, len(sub), max(1, len(sub)//9))
    ax.set_xticks(ticks, [sub.index[t].strftime('%b %d, %Y') for t in ticks], rotation=20, ha='right')
    ax.set_title(f"{s['ticker']} | {d.index[i].date()} | {len(s['support_tests'])} prior support tests | "
                 + ('subsequent bars shown' if future else 'as known at signal close'))
    ax.legend(ncol=3, fontsize=8, loc='best')
    ax.grid(alpha=.15)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return buf.getvalue()


STYLE = '<style>body{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 24px;color:#24364a;background:#f9fbfd}h1{font-size:30px}p,li{line-height:1.5}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #d7e0eb;text-align:right}th{background:#eaf0f6}img{max-width:100%;margin:12px 0}details{margin:12px 0 35px}a{color:#2565bc}.scroll{overflow:auto}</style>'


def run_detect(args):
    if args.out.exists():
        raise SystemExit('Choose a new run directory; existing research is preserved.')
    args.out.mkdir(parents=True)
    from strategy_config import LIQUID_PLUS_COMMODITIES, CSV_UNIVERSE
    liquid = set(LIQUID_PLUS_COMMODITIES)
    universe = sorted(t for t in liquid | set(CSV_UNIVERSE) if not any(x in t for x in ['^', '=', '/', '-']))
    stat = args.prices.stat()
    raw = pd.read_parquet(args.prices, columns=['ticker', 'date', 'Open', 'High', 'Low', 'Close'],
                          filters=[('ticker', 'in', universe)])
    after = args.prices.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source cache changed during loading; retry in a fresh directory.')
    raw.date = pd.to_datetime(raw.date).dt.normalize()
    if raw.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate ticker sessions.')
    values = raw[['Open', 'High', 'Low', 'Close']]
    valid = (np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
             & (raw.High >= values.max(axis=1)) & (raw.Low <= values.min(axis=1))
             & (raw.date.dt.weekday < 5))
    invalid = raw.loc[~valid]
    invalid.to_csv(args.out/'invalid_rows.csv', index=False)
    bad = set(invalid.ticker)
    raw = raw.loc[~raw.ticker.isin(bad)].sort_values(['ticker', 'date'])
    raw.to_parquet(args.out/'prices_snapshot.parquet', index=False)
    frames = {t: g.set_index('date') for t, g in raw.groupby('ticker')}
    if 'SPY' not in frames:
        raise ValueError('SPY unavailable or invalid; cannot benchmark this run.')
    rows, gaps, coverage = [], [], []
    for num, (ticker, d) in enumerate(frames.items(), start=1):
        tier = 'liquid' if ticker in liquid else 'overflow'
        coverage.append(dict(ticker=ticker, tier=tier, first=str(d.index.min().date()), last=str(d.index.max().date()), rows=len(d)))
        gap = d.Open/d.Close.shift()-1
        for date, val in gap[gap.abs() >= .4].items():
            gaps.append(dict(ticker=ticker, date=str(date.date()), overnight_return=float(val)))
        for s in detect(d):
            if d.index[s['signal_i']] < pd.Timestamp('2000-01-01'):
                continue
            rows.append({**s, 'ticker': ticker, 'tier': tier, 'signal_date': str(d.index[s['signal_i']].date()),
                         'range_start': str(d.index[s['range_start_i']].date()),
                         'known_date': str(d.index[s['known_i']].date()),
                         'support_dates': [str(d.index[k].date()) for k in s['support_tests']]})
        if num % 200 == 0:
            print(f'Detected {num}/{len(frames)} names; {len(rows)} signals', flush=True)
    rows.sort(key=lambda s: (s['signal_date'], s['ticker']))
    (args.out/'signals.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(args.out/'signals.csv', index=False)
    pd.DataFrame(coverage).to_csv(args.out/'coverage.csv', index=False)
    pd.DataFrame(gaps).to_csv(args.out/'large_gaps.csv', index=False)
    selected = rows[-15:][::-1]
    gallery = '<!doctype html><meta charset="utf-8"><title>Rising 200-day range reclaim — visual review</title>'+STYLE
    gallery += '<h1>Rising 200-day range reclaim</h1><p>Latest 15 signals selected by date. Initial charts stop at the signal close; subsequent prices are hidden below each chart. Blue circles mark prior support tests. Blue shading marks the five-session period after the range boundaries were established.</p>'
    gallery += f'<p>{len(rows)} signals across {len(frames)} eligible names; data through {raw.date.max().date()}. Adjusted prices from a saved research snapshot. Current-universe selection and corporate-action limitations apply.</p>'
    for rank, s in enumerate(selected, start=1):
        picture = chart(frames[s['ticker']], s)
        (args.out/f'example_{rank:02}.png').write_bytes(picture)
        encoded = base64.b64encode(picture).decode()
        later = base64.b64encode(chart(frames[s['ticker']], s, future=True)).decode()
        gallery += f'<h2>{html.escape(s["ticker"])} · {s["signal_date"]}</h2><p>Support {s["support"]:.3f}; ceiling {s["resistance"]:.3f}; SMA200 {s["sma200"]:.3f}. Support-test dates: {", ".join(s["support_dates"])}.</p><img alt="Pattern as of signal" src="data:image/png;base64,{encoded}"><details><summary>Show subsequent prices</summary><img alt="Subsequent prices" src="data:image/png;base64,{later}"></details>'
    (args.out/'gallery.html').write_text(gallery, encoding='utf-8')
    manifest = dict(rules=asdict(Rules()), source=str(args.prices.resolve()), source_size=stat.st_size,
                    source_mtime_ns=stat.st_mtime_ns, snapshot_sha256=sha(args.out/'prices_snapshot.parquet'),
                    code_hashes={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                    universe=universe, excluded_invalid=sorted(bad), invalid_rows=len(invalid),
                    missing=sorted(set(universe)-set(frames)-bad), rows=len(raw), tickers=len(frames),
                    first=str(raw.date.min()), last=str(raw.date.max()), signal_count=len(rows),
                    gallery_selection=[dict(ticker=s['ticker'], date=s['signal_date']) for s in selected],
                    generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
                    versions=dict(python=sys.version, pandas=pd.__version__, numpy=np.__version__))
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['tickers', 'last', 'signal_count', 'invalid_rows']}, indent=2))
    print(pd.DataFrame(selected)[['ticker', 'signal_date', 'support', 'resistance', 'sma200', 'support_dates']].to_string(index=False) if selected else 'No signals with the frozen rules.')


def run_evaluate(args):
    if (args.out/'trades.csv').exists():
        raise SystemExit('Existing trade results are preserved.')
    manifest = json.loads((args.out/'manifest.json').read_text())
    if sha(args.out/'prices_snapshot.parquet') != manifest['snapshot_sha256']:
        raise ValueError('Research snapshot hash changed.')
    for filename, expected in manifest['code_hashes'].items():
        if sha(Path(__file__).parent/filename) != expected:
            raise ValueError(f'Research source changed since detection: {filename}')
    signals = json.loads((args.out/'signals.json').read_text())
    raw = pd.read_parquet(args.out/'prices_snapshot.parquet')
    frames = {t: g.set_index('date').sort_index() for t, g in raw.groupby('ticker')}
    spy = frames['SPY']
    rows = []
    for ticker in sorted({s['ticker'] for s in signals}):
        events = [s for s in signals if s['ticker'] == ticker]
        for horizon, stop in [(5, False), (10, False), (20, False), (10, True)]:
            for trade in evaluate(frames[ticker], events, horizon, stop):
                en, ex = trade['entry_date'], trade['exit_date']
                if en not in spy.index or ex not in spy.index:
                    raise ValueError(f'Missing SPY dates for {ticker} {en} {ex}')
                trade['spy_excess'] = trade['gross']-(spy.at[ex, 'Close']/spy.at[en, 'Open']-1)
                trade['policy'] = 'stop10' if stop else f'fixed{horizon}'
                rows.append(trade)
    t = pd.DataFrame(rows)
    t.to_csv(args.out/'trades.csv', index=False)
    summaries = []
    if not t.empty:
        for (policy, tier), g in t.groupby(['policy', 'tier']):
            for period, q in [('all', g), ('2000-2017', g[g.signal_date.dt.year < 2018]),
                              ('2018-2022', g[g.signal_date.dt.year.between(2018, 2022)]),
                              ('2023+', g[g.signal_date.dt.year >= 2023])]:
                if q.empty:
                    continue
                low, high = block_ci(q, 'net')
                elo, ehi = block_ci(q, 'spy_excess')
                summaries.append(dict(policy=policy, tier=tier, period=period, n=len(q),
                                      mean_pct=100*q.net.mean(), median_pct=100*q.net.median(),
                                      win_pct=100*(q.net > 0).mean(), spy_excess_pct=100*q.spy_excess.mean(),
                                      ci_low_pct=100*low, ci_high_pct=100*high,
                                      spy_ci_low_pct=100*elo, spy_ci_high_pct=100*ehi,
                                      cost0_pct=100*q.gross.mean(), cost25_pct=100*q.gross.mean()-.25))
        for _, g in t.groupby(['ticker', 'policy']):
            assert (g.entry_date.iloc[1:].to_numpy() > g.exit_date.iloc[:-1].to_numpy()).all()
        assert (t.entry_date > t.signal_date).all()
        assert np.allclose(t.net, t.exit/t.entry-1-.001)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out/'summary.csv', index=False)
    report = '<!doctype html><meta charset="utf-8"><title>Range reclaim v2 results</title>'+STYLE
    report += '<h1>Range reclaim v2 — research results</h1><p><a href="gallery.html">Review the annotated signal gallery</a></p><p>Next-open entry, fixed horizons or separate stop-only execution, 10 bps round-trip cost. Monthly block-bootstrap 95% intervals. Current-universe selection and unresolved corporate-action risks; retrospective slices, no parameter optimization. Event returns are not portfolio returns.</p>'
    report += '<div class="scroll">'+summary.round(3).to_html(index=False)+'</div>'
    (args.out/'results.html').write_text(report, encoding='utf-8')
    (args.out/'evaluation_audit.json').write_text(json.dumps(dict(trades=len(t), checks='timing, nonoverlap, cost identity, SPY dates and snapshot hash passed'), indent=2))
    print(summary.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['detect', 'evaluate'])
    parser.add_argument('--prices', type=Path, default=ROOT/'data/master_prices.parquet')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (run_detect if args.mode == 'detect' else run_evaluate)(args)


if __name__ == '__main__':
    main()
