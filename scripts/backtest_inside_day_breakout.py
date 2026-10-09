"""Research-only inside-day buy-stop sweep using the repo's shared indicators.

Entry extension to pages/backtester.py: tomorrow-only stop at signal High;
gap fills at max(Open, signal High). Exit walk copies its next-session arming,
stop-first OHLC priority, gap-aware fills, entry+H time stop and 5bps/side cost.
ATR is deliberately frozen at SIGNAL close, not the unavailable entry-day ATR.
No strategy-book, scanner, broker, cache, or site writes.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from indicators import calculate_indicators
from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES, SPOT_TO_TRADEABLE

MAX_HOLD = 21


@dataclass(frozen=True)
class ExitSpec:
    hold: int
    stop: float | str | None
    target: float | None
    target_in_r: bool = False

    @property
    def name(self):
        stop = 'none' if self.stop is None else str(self.stop)
        target = 'none' if self.target is None else f'{self.target:g}{"R" if self.target_in_r else "ATR"}'
        return f'H{self.hold}_S{stop}_T{target}'


def exit_grid():
    specs = [ExitSpec(h, s, t) for h in (2, 5, 10, 21)
             for s in (None, 0.5, 1., 1.5, 2.) for t in (None, 1., 2., 3., 4.)]
    specs += [ExitSpec(h, 'signal_low', t, True) for h in (2, 5, 10, 21)
              for t in (None, 1., 2., 3., 4.)]
    return specs


def signal_mask(df, use_126):
    """Use existing rank_ret_* columns, including their 252-observation warmup."""
    signal = (df.rank_ret_252d.between(50, 90)
              & (df.rank_ret_5d < 95) & (df.rank_ret_10d < 95)
              & (df.rank_ret_21d < 95)
              & (df.High < df.High.shift()) & (df.Low > df.Low.shift())
              & (df.Close > df.SMA10) & (df.Close > df.EMA21))
    if use_126:
        signal &= df.rank_ret_126d.between(50, 90)
    return signal.fillna(False)


def buy_stop_fill(signal_high, next_open, next_high):
    # No tick offset: levels are relative and recomputed on adjusted bars.
    return np.where(next_high >= signal_high, np.maximum(next_open, signal_high), np.nan)


def valid_ohlc(df):
    bars = df[['Open', 'High', 'Low', 'Close']]
    tolerance = 1e-5 + 1e-6 * df.Close.abs()
    return (np.isfinite(bars).all(axis=1) & (bars > 0).all(axis=1)
            & (df.High >= bars.max(axis=1) - tolerance)
            & (df.Low <= bars.min(axis=1) + tolerance))


def simulate(meta, paths, spec, *, entry_day=False, cost_bps=5.):
    """Vectorized copy of the single-strategy daily exit walk; no future ATR."""
    n = len(meta)
    entry, atr = meta.entry.to_numpy(), meta.atr.to_numpy()
    if spec.stop == 'signal_low':
        stop = meta.signal_low.to_numpy()
        risk = entry - stop
    else:
        risk = atr * (float(spec.stop) if spec.stop is not None else 1.)
        stop = entry - risk if spec.stop is not None else np.full(n, -np.inf)
    if not (np.isfinite(atr).all() and (atr > 0).all()
            and np.isfinite(risk).all() and (risk > 0).all()):
        raise ValueError('ATR and reporting/stop risk must be finite and positive')
    target = (entry + spec.target * (risk if spec.target_in_r else atr)
              if spec.target is not None else np.full(n, np.inf))
    px = paths['Close'][:, spec.hold].copy()
    day = np.full(n, spec.hold, dtype=int)
    reason = np.full(n, 'Time', dtype='U6')
    alive = np.ones(n, dtype=bool)
    for d in range(0 if entry_day else 1, spec.hold + 1):
        stop_hit = alive & (paths['Low'][:, d] <= stop)
        target_hit = alive & ~stop_hit & (paths['High'][:, d] >= target)
        px[stop_hit] = (stop[stop_hit] if d == 0 else
                        np.minimum(paths['Open'][stop_hit, d], stop[stop_hit]))
        px[target_hit] = np.maximum(paths['Open'][target_hit, d], target[target_hit])
        day[stop_hit | target_hit] = d
        reason[stop_hit], reason[target_hit] = 'Stop', 'Target'
        alive &= ~(stop_hit | target_hit)
    slip = cost_bps / 10000
    pnl = px * (1 - slip) - entry * (1 + slip)
    return pd.DataFrame({'exit': px, 'exit_day': day, 'reason': reason,
                         'net_atr': pnl / atr, 'net_r': pnl / risk,
                         'net_pct': 100 * pnl / (entry * (1 + slip)),
                         'risk_atr': risk / atr})


def one_position_mask(meta, result):
    """Same signal-date <= last-exit rejection as pages/backtester.run_engine."""
    accepted = np.zeros(len(meta), dtype=bool)
    last_exit = {}
    for i, (ticker, signal_idx, entry_idx, exit_day) in enumerate(zip(
            meta.ticker, meta.signal_idx, meta.entry_idx, result.exit_day)):
        if signal_idx <= last_exit.get(ticker, -1):
            continue
        accepted[i] = True
        last_exit[ticker] = entry_idx + exit_day
    return accepted


def metrics(result):
    if result.empty:
        return {'trades': 0}
    r = result.net_atr
    losses = -r[r < 0].sum()
    return {'trades': len(result), 'avg_atr': r.mean(), 'median_atr': r.median(),
            'avg_r': result.net_r.mean(), 'avg_pct': result.net_pct.mean(),
            'win_pct': 100 * (r > 0).mean(), 'pf_atr': r[r > 0].sum() / losses if losses else np.nan,
            'p05_atr': r.quantile(.05), 'worst_atr': r.min(),
            'avg_hold': result.exit_day.mean(),
            'stop_pct': 100 * (result.reason == 'Stop').mean(),
            'target_pct': 100 * (result.reason == 'Target').mean()}


def month_bootstrap(meta, result):
    """Resample calendar months, preserving simultaneous and serial clusters."""
    d = pd.DataFrame({'month': meta.signal_date.dt.to_period('M'), 'r': result.net_atr})
    blocks = d.groupby('month').r.agg(['sum', 'count']).to_numpy()
    rng = np.random.default_rng(20261006)
    draw = rng.integers(0, len(blocks), (2000, len(blocks)))
    sampled = blocks[draw].sum(axis=1)
    return np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975]).tolist()


def prepare(prices, start, end, out, exclusions=()):
    wanted = set(CSV_UNIVERSE) | set(LIQUID_PLUS_COMMODITIES)
    # Avoid double-counting spot indices and their actual tradeable ETFs.
    wanted -= set(SPOT_TO_TRADEABLE)
    requested = wanted.copy()
    wanted -= set(exclusions)
    snapshot_stat = prices.stat()
    snapshot_hash = hashlib.sha256(prices.read_bytes()).hexdigest()
    raw = pd.read_parquet(prices)
    if prices.stat().st_mtime_ns != snapshot_stat.st_mtime_ns:
        raise ValueError('Price file changed while loading; retry from a stable snapshot')
    raw.date = pd.to_datetime(raw.date).dt.normalize()
    raw = raw[raw.ticker.isin(wanted) & (raw.date <= end)].copy()
    if raw.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate ticker dates; resolve source ambiguity before testing')
    coverage, events, path_parts = [], [], {c: [] for c in ('Open', 'High', 'Low', 'Close', 'date')}
    for k, (ticker, g) in enumerate(raw.groupby('ticker', sort=True)):
        g = g.sort_values('date').set_index('date')
        valid = valid_ohlc(g)
        # Preserve calendar positions. A bad bar stays as NaN rather than
        # compressing the history and changing the meaning of tomorrow/hold H.
        g.loc[~valid, ['Open', 'High', 'Low', 'Close']] = np.nan
        df = calculate_indicators(g.drop(columns='ticker'), {}, ticker)
        base, strict = signal_mask(df, False), signal_mask(df, True)
        idx = np.flatnonzero(base & (df.index >= start))
        mature = idx[idx + 1 + MAX_HOLD < len(df)]
        clean = np.array([valid.iloc[i-1:i+MAX_HOLD+2].all() for i in mature])
        invalid_windows = int((~clean).sum())
        mature = mature[clean]
        fill = buy_stop_fill(df.High.to_numpy()[mature], df.Open.to_numpy()[mature + 1],
                             df.High.to_numpy()[mature + 1])
        selected = mature[np.isfinite(fill)]
        fill = fill[np.isfinite(fill)]
        usable_risk = (np.isfinite(df.ATR.to_numpy()[selected])
                       & (df.ATR.to_numpy()[selected] > 0)
                       & (fill > df.Low.to_numpy()[selected]))
        invalid_risk = int((~usable_risk).sum())
        selected, fill = selected[usable_risk], fill[usable_risk]
        coverage.append({'ticker': ticker, 'bars': len(df), 'invalid_bars': int((~valid).sum()),
                         'invalid_signal_windows': invalid_windows,
                         'excluded_nonpositive_risk': invalid_risk,
                         'first': df.index.min(), 'last': df.index.max(),
                         'signals_no126': len(idx), 'signals_126': int(strict.iloc[idx].sum()),
                         'mature_signals_no126': len(mature), 'fills_no126': len(selected),
                         'mature_signals_126': int(strict.iloc[mature].sum()),
                         'fills_126': int(strict.iloc[selected].sum())})
        if len(selected):
            e = pd.DataFrame({'ticker': ticker, 'tier': 'Liquid' if ticker in LIQUID_PLUS_COMMODITIES else 'Overflow',
                              'signal_date': df.index[selected], 'signal_idx': selected, 'entry_idx': selected + 1,
                              'entry_date': df.index[selected + 1], 'entry': fill,
                              'signal_high': df.High.to_numpy()[selected],
                              'signal_low': df.Low.to_numpy()[selected], 'atr': df.ATR.to_numpy()[selected],
                              'use126': strict.to_numpy()[selected]})
            # Comparator: random next-open entries, same ticker/year/horizon,
            # conditioned on the shared momentum/MA gates but no inside-day rule.
            control = (df.rank_ret_252d.between(50, 90)
                       & (df.rank_ret_5d < 95) & (df.rank_ret_10d < 95) & (df.rank_ret_21d < 95)
                       & (df.Close > df.SMA10) & (df.Close > df.EMA21))
            for h in (2, 5, 10, 21):
                p = df.Close.shift(-(h + 1)) * .9995 - df.Open.shift(-1) * 1.0005
                ret = 100 * p / (df.Open.shift(-1) * 1.0005)
                for use126 in (False, True):
                    mask = control & (df.index >= start)
                    if use126:
                        mask &= df.rank_ret_126d.between(50, 90)
                    yearly = ret[mask].groupby(ret[mask].index.year).mean()
                    e[f'control_{h}_{int(use126)}'] = pd.Series(e.signal_date.dt.year).map(yearly).to_numpy()
            events.append(e)
            positions = selected[:, None] + 1 + np.arange(MAX_HOLD + 1)[None, :]
            for col in path_parts:
                values = df.index.to_numpy() if col == 'date' else df[col].to_numpy(dtype=float)
                path_parts[col].append(values[positions])
        if k % 100 == 0:
            print(f'Indicators {k+1}/{raw.ticker.nunique()}: {ticker}', flush=True)
    if not events:
        raise ValueError('No mature fills')
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(out / 'coverage.csv', index=False)
    meta = pd.concat(events, ignore_index=True)
    paths = {c: np.concatenate(parts) for c, parts in path_parts.items()}
    meta.to_parquet(out / 'events.parquet', index=False)
    np.savez_compressed(out / 'paths.npz', **paths)
    missing = sorted(wanted - set(coverage.ticker))
    manifest = {'prices': str(prices.resolve()), 'bytes': snapshot_stat.st_size,
                'price_sha256': snapshot_hash,
                'start': str(start.date()), 'end': str(end.date()), 'universe_requested': len(requested),
                'excluded_tickers': sorted(set(exclusions)),
                'covered': len(coverage), 'missing': missing, 'events': len(meta),
                'shared_indicators_sha256': hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return meta, paths


def sweep(meta, paths, out):
    rows, annual, tiers = [], [], []
    for use126 in (True, False):
        variant = 'With126' if use126 else 'Without126'
        keep = meta.use126.to_numpy() if use126 else np.ones(len(meta), bool)
        m = meta[keep].reset_index(drop=True)
        p = {c: a[keep] for c, a in paths.items()}
        for spec in exit_grid():
            r = simulate(m, p, spec)
            accepted = one_position_mask(m, r)
            for mode, mask in (('one_position', accepted), ('all_signals', np.ones(len(m), bool))):
                mm, rr = m[mask].reset_index(drop=True), r[mask].reset_index(drop=True)
                stats = {'filter': variant, 'exit': spec.name, 'mode': mode,
                         'hold': spec.hold, 'stop': str(spec.stop), 'target': str(spec.target),
                         'target_in_r': spec.target_in_r, **metrics(rr)}
                discovery = mm.signal_date.dt.year <= 2017
                stats['discovery_avg_atr'] = rr.loc[discovery, 'net_atr'].mean()
                stats['later_avg_atr'] = rr.loc[~discovery, 'net_atr'].mean()
                stats['later_trades'] = int((~discovery).sum())
                if spec.stop is None and spec.target is None:
                    stats['control_avg_pct'] = mm[f'control_{spec.hold}_{int(use126)}'].mean()
                    stats['control_edge_pct'] = stats['avg_pct'] - stats['control_avg_pct']
                rows.append(stats)
                if mode == 'one_position':
                    for year, ii in mm.groupby(mm.signal_date.dt.year).groups.items():
                        annual.append({'filter': variant, 'exit': spec.name, 'year': year, **metrics(rr.loc[ii])})
                    for tier, ii in mm.groupby('tier').groups.items():
                        tiers.append({'filter': variant, 'exit': spec.name, 'tier': tier, **metrics(rr.loc[ii])})
        print(f'Swept {variant}: {len(m):,} mature fills', flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(annual).to_csv(out / 'annual.csv', index=False)
    pd.DataFrame(tiers).to_csv(out / 'tiers.csv', index=False)
    return summary


def selected_details(meta, paths, summary, out):
    details, intervals, sensitivity = [], [], []
    primary = summary[summary['mode'] == 'one_position']
    for variant in ('With126', 'Without126'):
        part = primary[primary['filter'] == variant]
        names = set(part.nlargest(3, 'avg_atr').exit) | set(part.nlargest(1, 'discovery_avg_atr').exit)
        names |= {ExitSpec(h, None, None).name for h in (2, 5, 10, 21)}
        names |= {ExitSpec(h, 1., 2.).name for h in (2, 5, 21)}
        names |= {ExitSpec(h, 'signal_low', 2., True).name for h in (2, 5, 21)}
        names.add(ExitSpec(21, 2., None).name)
        keep = meta.use126.to_numpy() if variant == 'With126' else np.ones(len(meta), bool)
        m = meta[keep].reset_index(drop=True)
        p = {c: a[keep] for c, a in paths.items()}
        for spec in exit_grid():
            if spec.name not in names:
                continue
            r = simulate(m, p, spec)
            accepted = one_position_mask(m, r)
            mm, rr = m[accepted].reset_index(drop=True), r[accepted].reset_index(drop=True)
            lo, hi = month_bootstrap(mm, rr)
            intervals.append({'filter': variant, 'exit': spec.name, 'ci_low_atr': lo, 'ci_high_atr': hi})
            rr['exit_date'] = p['date'][accepted][np.arange(len(rr)), rr.exit_day]
            details.append(pd.concat([mm.assign(filter=variant, exit_spec=spec.name), rr], axis=1))
            if spec.stop is not None or spec.target is not None:
                same_day = simulate(m, p, spec, entry_day=True)
                accepted0 = one_position_mask(m, same_day)
                sensitivity.append({'filter': variant, 'exit': spec.name, 'arming': 'entry_day_pessimistic',
                                    **metrics(same_day[accepted0])})
    pd.concat(details, ignore_index=True).to_parquet(out / 'selected_trades.parquet', index=False)
    pd.DataFrame(intervals).to_csv(out / 'confidence.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out / 'entry_day_sensitivity.csv', index=False)


def report(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    primary = summary[summary['mode'] == 'one_position'].copy()
    confidence = pd.read_csv(out / 'confidence.csv')
    primary = primary.merge(confidence, on=['filter', 'exit'], how='left')
    columns = ['filter', 'exit', 'trades', 'avg_atr', 'avg_pct', 'win_pct', 'pf_atr',
               'discovery_avg_atr', 'later_avg_atr', 'p05_atr', 'ci_low_atr', 'ci_high_atr']
    sections = []
    for variant in ('With126', 'Without126'):
        p = primary[primary['filter'] == variant]
        sections.append(f'<h2>{variant}: top 12 by net ATR per trade (exploratory)</h2>'
                        + p.nlargest(12, 'avg_atr')[columns].to_html(index=False, float_format=lambda x: f'{x:.3f}'))
    time_only = primary[(primary.stop == 'None') & (primary.target == 'None')]
    sections.append('<h2>No stop / no target</h2>' + time_only[columns + ['control_avg_pct', 'control_edge_pct']].to_html(index=False, float_format=lambda x: f'{x:.3f}'))
    sections.append('<h2>All primary configurations</h2>' + primary[columns].to_html(index=False, float_format=lambda x: f'{x:.3f}'))
    manifest = json.loads((out / 'manifest.json').read_text())
    event_dates = pd.read_parquet(out / 'events.parquet', columns=['signal_date']).signal_date
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, variant in zip(axes, ('With126', 'Without126')):
        for stop, target, label in [('None', 'None', 'Time only'), ('None', '1.0', 'No stop; 1 ATR target'),
                                    ('None', '2.0', 'No stop; 2 ATR target'), ('1.0', '2.0', '1 ATR stop; 2 ATR target')]:
            series = primary[(primary['filter'] == variant) & (primary.stop == stop)
                             & (primary.target == target)].sort_values('hold')
            ax.plot(series.hold, series.avg_atr, marker='o', label=label)
        ax.axhline(0, color='#555', linewidth=.8)
        ax.set_title('126-day filter' if variant == 'With126' else 'Without 126-day filter')
        ax.set_xlabel('Time exit: sessions after entry')
        ax.set_xticks([2, 5, 10, 21])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Average net ATR per trade')
    axes[1].legend(fontsize=9)
    fig.suptitle('Inside-day buy stop — one position per ticker, 5 bps per side')
    fig.tight_layout()
    fig.savefig(out / 'exit_comparison.png', dpi=160)
    plt.close(fig)
    methodology = '''Percentiles: indicators.calculate_indicators; N-session percentage return ranked against its expanding history, minimum 252 valid observations, ties averaged. Bands 50–90 inclusive; 5/10/21 ranks strictly below 95. Strict inside day; close above both SMA10 and EMA21. Buy stop at signal high, valid T+1 only; gaps fill at T+1 open. No arbitrary price/volume filters. Current configured CSV universe plus liquid/commodities, removing duplicate ^GSPC/^NDX aliases. Historical membership is unavailable: survivorship and current-universe selection bias remain.
    Signal-day simple ATR14 is frozen; adjusted bars as used by the engines. Corporate-action basis risk remains in this rolling-adjusted cache. Nonpositive/nonfinite/inconsistent OHLC bars are marked unavailable without compressing calendar positions, and affected signal/execution windows are excluded; coverage.csv records these exclusions. Fills with unavailable/nonpositive ATR or zero signal-low risk are excluded from the common comparison sample. Primary exits arm the next session, matching the repo convention. Stop wins a daily bar containing both stop and target; stops gap down to open, targets gap up to open. Time stop at entry index + H, so H=2 includes entry plus two subsequent sessions. Signal-low stops target multiples of actual entry-to-signal-low risk. No-stop R is an ATR reporting unit, not a bounded risk. Fixed stop R uses its stop distance. Main comparisons use net ATR and percentage returns, not potentially misleading differing R denominators.
    5 bps entry and 5 bps exit slippage. No commissions or impact model. One position per ticker, excluding signals on/before its last exit, matches the existing single-strategy framework. All-signal sensitivity also exported. No account sizing, pooled caps or portfolio CAGR modeled: these are trade-level outcomes. Every event has 21 subsequent sessions, avoiding partial terminal trades and holding-horizon sample drift. Dates after the last completed US session are excluded. Delisted names retain their available historical bars.
    240 filter/exit combinations are exploratory. Discovery <=2017 and later >=2018 are retrospective slices, not untouched out-of-sample tests. Reported selected confidence intervals bootstrap calendar-month blocks with fixed seed, 2,000 draws; they do not correct for searching the exit grid. Comparator for time-only trades: next-open entries with identical momentum/MA gates but no inside-day condition, weighted to each strategy trade's ticker and signal year; useful context, not a randomized matched experiment. Entry-day arming sensitivity is a pessimistic daily-bar assumption; intraday ordering cannot be established from OHLC.'''
    page = '<!doctype html><html><meta charset="utf-8"><title>Inside-day breakout research</title><style>body{font:15px system-ui;margin:32px;color:#1b263b}table{border-collapse:collapse;font-size:13px}th,td{padding:6px 10px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef2f8}p{max-width:1150px;line-height:1.6}h2{margin-top:32px}</style><h1>Inside-day buy-stop breakout</h1>'
    page += f'<p>Research through {html.escape(manifest["end"])}; {manifest["covered"]} tickers; {manifest["events"]:,} mature filled events before the 126-day gate and overlap rejection. Common signal sample: {event_dates.min().date()}–{event_dates.max().date()}.</p>'
    if manifest.get('excluded_tickers'):
        page += f'<p>Price-history audit exclusions: {html.escape(", ".join(manifest["excluded_tickers"]))}. See price_quality_audit.md for the evidence and limitations.</p>'
    page += '<img src="exit_comparison.png" alt="Exit comparison" style="width:100%;max-width:1100px">'
    page += '<h2>Assumptions and limitations</h2>' + ''.join(f'<p>{html.escape(x.strip())}</p>' for x in methodology.split('\n'))
    page += ''.join(sections) + '</html>'
    (out / 'report.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=ROOT / 'data/master_prices.parquet')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/inside-day-breakout')
    parser.add_argument('--start', default='2000-01-01')
    parser.add_argument('--end', default=str((pd.Timestamp.now(tz='America/New_York').date()
                                            - pd.Timedelta(days=1))))
    parser.add_argument('--reuse-events', action='store_true')
    parser.add_argument('--exclude-tickers', nargs='*', default=[],
                        help='Explicit research-only exclusions for unusable source histories')
    args = parser.parse_args()
    today = pd.Timestamp.now(tz='America/New_York').normalize().tz_localize(None)
    if pd.Timestamp(args.end) >= today:
        parser.error('--end must precede today to exclude potentially incomplete bars')
    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        parser.error('--start must precede --end')
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'artifacts'):
        parser.error('Research outputs must stay under artifacts/')
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse_events:
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['start'] != args.start or manifest['end'] != args.end:
            parser.error('Cached event dates do not match requested dates')
        if manifest.get('excluded_tickers', []) != sorted(set(args.exclude_tickers)):
            parser.error('Cached event universe exclusions differ; regenerate events')
        if manifest['shared_indicators_sha256'] != hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest():
            parser.error('Shared indicators changed; regenerate events')
        if (manifest['prices'] != str(args.prices.resolve()) or
                manifest['price_sha256'] != hashlib.sha256(args.prices.read_bytes()).hexdigest()):
            parser.error('Input prices changed; regenerate events')
        meta = pd.read_parquet(out / 'events.parquet')
        with np.load(out / 'paths.npz') as cached:
            paths = {c: cached[c] for c in cached.files}
    else:
        meta, paths = prepare(args.prices, pd.Timestamp(args.start), pd.Timestamp(args.end), out,
                              args.exclude_tickers)
    summary = sweep(meta, paths, out)
    selected_details(meta, paths, summary, out)
    report(summary, out)
    print(summary[summary['mode'] == 'one_position'].nlargest(12, 'avg_atr').to_string(index=False), flush=True)
    print(f'Report: {out / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
