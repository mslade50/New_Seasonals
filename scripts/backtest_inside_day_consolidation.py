"""Count-first consolidation research; strict <=100 raw signals in EACH year.

Reuses the inside-day study's audited execution paths and shared indicators.
Does not mutate production data, strategy settings, orders, or live systems.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from indicators import calculate_indicators, consolidation_features
from scripts.backtest_inside_day_breakout import (
    ExitSpec, exit_grid, metrics, month_bootstrap, one_position_mask,
    signal_mask, simulate, valid_ohlc,
)
from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES, SPOT_TO_TRADEABLE


def episode_mask(selected, ticker_codes, bar_idx, lookback):
    """Shared first-instance convention: no raw qualifying signal in prior L-1 bars."""
    idx = np.flatnonzero(selected)
    if lookback <= 1 or not len(idx):
        return selected.copy()
    same = ticker_codes[idx[1:]] == ticker_codes[idx[:-1]]
    gap = bar_idx[idx[1:]] - bar_idx[idx[:-1]]
    first = np.r_[True, ~same | (gap >= lookback)]
    out = np.zeros(len(selected), dtype=bool)
    out[idx[first]] = True
    return out


def annual_count_gate(years, selected, cap=100, min_signals=250):
    counts = np.bincount(years[selected], minlength=27)
    return bool(counts.max(initial=0) <= cap and counts.sum() >= min_signals), counts


def profiles():
    # A predeclared, count-first grid. No percentile/MA baseline retuning.
    for family in ('quiet_inside', 'narrowest_range', 'vol_squeeze', 'double_inside'):
        volume = (.8, .6, .4, .25) if family == 'quiet_inside' else (None, .8, .6, .4)
        special = (.7, .5, .35) if family == 'quiet_inside' else (
            ('nr7', 'nr10') if family == 'narrowest_range' else (
                (.85, .7, .55) if family == 'vol_squeeze' else (True,)))
        for v, s, comp, near, trend, use126, no_high, episode in itertools.product(
                volume, special, (2., 1.5, 1.), (5., 3., 1.),
                ('trend_rising50', 'trend_stack'), (False, True), (False, True), (1, 10, 21)):
            yield {'family': family, 'volume': v, 'special': s, 'range5': comp,
                   'near_high_pct': near, 'trend': trend, 'use126': use126,
                   'exclude_new_close_high': no_high, 'episode': episode}


def profile_mask(features, profile):
    p = profile
    mask = ((features.range5_atr <= p['range5'])
            & (features.distance_high252_pct <= p['near_high_pct'])
            & features[p['trend']].astype(bool)).to_numpy()
    if p['volume'] is not None:
        mask &= (features.quiet_volume_ratio <= p['volume']).to_numpy()
    if p['use126']:
        mask &= features.use126.to_numpy(bool)
    if p['exclude_new_close_high']:
        mask &= features.not_new_closing_high252.to_numpy(bool)
    if p['family'] == 'quiet_inside':
        mask &= (features.day_range_atr <= p['special']).to_numpy()
    elif p['family'] == 'narrowest_range':
        if p['special'] is not None:
            mask &= features[p['special']].to_numpy(bool)
    elif p['family'] == 'vol_squeeze':
        mask &= (features.atr5_to_atr21 <= p['special']).to_numpy()
    else:
        mask &= features.double_inside.to_numpy(bool)
    return mask


def description(p):
    pieces = ['Close above rising SMA50' if p['trend'] == 'trend_rising50'
              else 'Close > SMA50 > SMA200; SMA50 rising',
              f'5-day high-low span <= {p["range5"]:g} ATR',
              f'Close within {p["near_high_pct"]:g}% of 52w intraday high']
    if p['volume'] is not None:
        pieces.append(f'Volume <= {p["volume"]:g}x 63d average')
    if p['family'] == 'quiet_inside':
        pieces.append(f'Today range <= {p["special"]:g} ATR')
    elif p['family'] == 'narrowest_range':
        pieces.append(f'Range narrowest of last {7 if p["special"] == "nr7" else 10} bars')
    elif p['family'] == 'vol_squeeze':
        pieces.append(f'ATR5 / ATR21 <= {p["special"]:g}')
    else:
        pieces.append('Two consecutive strict inside days')
    if p['use126']:
        pieces.append('126d return rank 50–90')
    if p['exclude_new_close_high']:
        pieces.append('Close below prior 252d closing high')
    if p['episode'] > 1:
        pieces.append(f'No qualifying signal in previous {p["episode"]-1} ticker sessions')
    return '; '.join(pieces)


def prepare(prices, base, out):
    original = json.loads((base / 'manifest.json').read_text())
    price_bytes = prices.read_bytes()
    digest = hashlib.sha256(price_bytes).hexdigest()
    if digest != original['price_sha256']:
        raise ValueError('Prices differ from audited entry/exit paths; regenerate original study first')
    events = pd.read_parquet(base / 'events.parquet')
    lookup = pd.Series(np.arange(len(events)), index=pd.MultiIndex.from_frame(events[['ticker', 'signal_date']]))
    wanted = (set(CSV_UNIVERSE) | set(LIQUID_PLUS_COMMODITIES)) - set(SPOT_TO_TRADEABLE)
    wanted -= set(original['excluded_tickers'])
    raw = pd.read_parquet(io.BytesIO(price_bytes))
    raw.date = pd.to_datetime(raw.date).dt.normalize()
    raw = raw[raw.ticker.isin(wanted) & (raw.date <= original['end'])]
    if raw.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate ticker dates')
    frames = []
    for k, (ticker, g) in enumerate(raw.groupby('ticker', sort=True)):
        g = g.sort_values('date').set_index('date').drop(columns='ticker')
        g.loc[~valid_ohlc(g), ['Open', 'High', 'Low', 'Close']] = np.nan
        df = calculate_indicators(g, {}, ticker)
        mask = signal_mask(df, False) & (df.index >= original['start'])
        f = consolidation_features(df).loc[mask].copy()
        f['ticker'], f['signal_date'] = ticker, f.index
        f['signal_idx'] = np.flatnonzero(mask)
        f['use126'] = signal_mask(df, True).loc[mask].to_numpy()
        keys = pd.MultiIndex.from_frame(f[['ticker', 'signal_date']])
        f['event_idx'] = lookup.reindex(keys, fill_value=-1).to_numpy()
        frames.append(f.reset_index(drop=True))
        if k % 100 == 0:
            print(f'Consolidation features {k+1}/{len(wanted)}: {ticker}', flush=True)
    features = pd.concat(frames, ignore_index=True)
    features.to_parquet(out / 'signal_features.parquet', index=False)
    linked = features[features.event_idx >= 0]
    assert set(linked.event_idx) == set(range(len(events))), 'Audited events must all map to raw signals'
    np.testing.assert_array_equal(linked.use126.to_numpy(), events.iloc[linked.event_idx].use126.to_numpy())
    manifest = {**original, 'raw_base_signals': len(features),
                'indicator_source_sha256': hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return features, events


def screen(features, events, paths, out):
    ticker_codes = pd.factorize(features.ticker)[0]
    bars = features.signal_idx.to_numpy()
    years = features.signal_date.dt.year.to_numpy() - 2000
    seen, candidates, count_rows, results = set(), {}, [], []
    tried, passed, too_few = 0, 0, 0
    specs = [ExitSpec(h, None, None) for h in (2, 5, 10, 21)]
    specs += [ExitSpec(5, None, 2.), ExitSpec(5, 1., 2.), ExitSpec(10, 1., 2.),
              ExitSpec(21, 2., None)]
    for profile in profiles():
        tried += 1
        selected = episode_mask(profile_mask(features, profile), ticker_codes, bars, profile['episode'])
        admissible, counts = annual_count_gate(years, selected)
        if counts.sum() < 250 or counts[18:].sum() < 50:
            too_few += 1
            continue
        if not admissible:
            continue
        passed += 1
        fingerprint = hashlib.sha256(np.packbits(selected).tobytes()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        name = f'C{len(candidates)+1:04d}'
        event_idx = features.loc[selected & (features.event_idx >= 0), 'event_idx'].to_numpy(int)
        # Preserve ticker/signal order for the inherited one-position walk.
        event_idx.sort()
        if len(event_idx) < 100:
            continue
        candidates[name] = {'profile': profile, 'event_idx': event_idx, 'selected': np.flatnonzero(selected)}
        count_rows.append({'candidate': name, **profile, 'rules': description(profile),
                           'signals': int(counts.sum()), 'max_year_signals': int(counts.max()),
                           'mean_year_signals': float(counts[2:26].mean()),
                           'later_signals': int(counts[18:].sum()), 'mature_fills': len(event_idx),
                           **{f'signals_{2000+i}': int(c) for i, c in enumerate(counts)}})
        m = events.iloc[event_idx].reset_index(drop=True)
        pp = {k: a[event_idx] for k, a in paths.items()}
        for spec in specs:
            r = simulate(m, pp, spec)
            accepted = one_position_mask(m, r)
            mm, rr = m[accepted].reset_index(drop=True), r[accepted].reset_index(drop=True)
            discovery = mm.signal_date.dt.year <= 2017
            later = ~discovery
            results.append({'candidate': name, 'exit': spec.name, **metrics(rr),
                            'discovery_trades': int(discovery.sum()),
                            'discovery_avg_atr': rr.loc[discovery, 'net_atr'].mean(),
                            'later_trades': int(later.sum()), 'later_avg_atr': rr.loc[later, 'net_atr'].mean(),
                            'later_avg_pct': rr.loc[later, 'net_pct'].mean()})
    if not candidates:
        raise ValueError('No profile meets the count and sample-size constraints')
    counts = pd.DataFrame(count_rows)
    summary = pd.DataFrame(results)
    counts.to_csv(out / 'candidates.csv', index=False)
    summary.to_csv(out / 'screen_results.csv', index=False)
    print(f'Grid {tried}; count-qualified {passed}; unique usable sets {len(candidates)}; thin {too_few}', flush=True)
    (out / 'search.json').write_text(json.dumps({'profiles_tested': tried, 'count_qualified': passed,
                                               'unique_usable': len(candidates), 'too_thin': too_few}, indent=2))
    return candidates, counts, summary


def finalists(candidates, counts, summary, events, paths, out):
    # Family leaders use only <=2017 5/10d returns. A separate consistency
    # shortlist below uses both eras; the entire exercise is retrospective.
    selection = summary[summary.exit.isin(['H5_Snone_Tnone', 'H10_Snone_Tnone'])]
    selection = selection[(selection.discovery_trades >= 50) & (selection.later_trades >= 30)]
    score = selection.groupby('candidate').discovery_avg_atr.mean().rename('selection_score')
    ranking = counts.merge(score, on='candidate').sort_values('selection_score', ascending=False)
    chosen = list(ranking.groupby('family', sort=False).head(2).candidate)
    basis = {name: 'early-period family leader' for name in chosen}
    # A separate, explicitly retrospective consistency shortlist. Unlike the
    # early leaders, these use BOTH eras and must not be called validation.
    for h in (5, 10, 21):
        stable = summary[(summary.exit == f'H{h}_Snone_Tnone')
                         & (summary.discovery_trades >= 75) & (summary.later_trades >= 50)].copy()
        if stable.empty:
            continue
        stable['consistency_score'] = np.minimum(stable.discovery_avg_atr, stable.later_avg_atr).round(8)
        stable = stable.merge(counts, on='candidate')
        stable['trend_order'] = (stable.trend != 'trend_stack').astype(int)
        best = stable.sort_values(['consistency_score', 'episode', 'trend_order'],
                                  ascending=[False, True, True]).iloc[0].candidate
        if best not in chosen:
            chosen.append(best)
        basis[best] = f'retrospective early/later consistency, H{h}'
    ranking = counts.merge(score, on='candidate', how='left')
    ranking['selection_basis'] = ranking.candidate.map(basis)
    rows, annual, trades, intervals, signals = [], [], [], [], []
    for name in chosen:
        item = candidates[name]
        ii = item['event_idx']
        m = events.iloc[ii].reset_index(drop=True)
        pp = {k: a[ii] for k, a in paths.items()}
        selected_signals = pd.read_parquet(out / 'signal_features.parquet').iloc[item['selected']].copy()
        signals.append(selected_signals.assign(candidate=name))
        for spec in exit_grid():
            r = simulate(m, pp, spec)
            accepted = one_position_mask(m, r)
            mm, rr = m[accepted].reset_index(drop=True), r[accepted].reset_index(drop=True)
            later = mm.signal_date.dt.year >= 2018
            rows.append({'candidate': name, 'exit': spec.name, **metrics(rr),
                         'discovery_avg_atr': rr.loc[~later, 'net_atr'].mean(),
                         'later_avg_atr': rr.loc[later, 'net_atr'].mean(),
                         'later_trades': int(later.sum()), 'later_avg_pct': rr.loc[later, 'net_pct'].mean()})
            for year, jj in mm.groupby(mm.signal_date.dt.year).groups.items():
                annual.append({'candidate': name, 'exit': spec.name, 'year': year, **metrics(rr.loc[jj])})
            if spec in [ExitSpec(h, None, None) for h in (2, 5, 10, 21)] + [ExitSpec(5, 1., 2.)]:
                lo, hi = month_bootstrap(mm, rr)
                later_lo, later_hi = month_bootstrap(mm[later].reset_index(drop=True), rr[later].reset_index(drop=True))
                intervals.append({'candidate': name, 'exit': spec.name, 'ci_low_atr': lo, 'ci_high_atr': hi,
                                  'later_ci_low_atr': later_lo, 'later_ci_high_atr': later_hi})
                rr['exit_date'] = pp['date'][accepted][np.arange(len(rr)), rr.exit_day]
                rr['control_edge_pct'] = (rr.net_pct - mm[f'control_{spec.hold}_{int(item["profile"]["use126"])}']
                                          if spec.stop is None and spec.target is None else np.nan)
                trades.append(pd.concat([mm.assign(candidate=name, exit_spec=spec.name), rr], axis=1))
    pd.DataFrame(rows).to_csv(out / 'exit_sweep.csv', index=False)
    pd.DataFrame(annual).to_csv(out / 'annual.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'confidence.csv', index=False)
    pd.concat(trades, ignore_index=True).to_parquet(out / 'selected_trades.parquet', index=False)
    pd.concat(signals, ignore_index=True).to_csv(out / 'selected_signals.csv', index=False)
    ranking[ranking.candidate.isin(chosen)].sort_values('candidate').to_csv(out / 'finalists.csv', index=False)
    return chosen


def sensitivity(features, candidates, chosen, events, paths, out):
    """Fixed-exit ablations and a calendar check; no selection from these runs."""
    from equity_sessions import calendar

    closes = calendar().schedule['close'].dt.tz_convert('America/New_York')
    early_dates = closes[closes.dt.hour < 16].index.tz_localize(None).normalize()
    regular = ~features.signal_date.isin(early_dates).to_numpy()
    ticker_codes = pd.factorize(features.ticker)[0]
    bars = features.signal_idx.to_numpy()
    years = features.signal_date.dt.year.to_numpy() - 2000
    rows = []
    for name in chosen:
        profile = candidates[name]['profile']
        variants = [('original', profile, False), ('exclude early-close signal days', profile, True)]
        # Hold every other clause fixed, including the original base signal.
        if (profile['family'] == 'narrowest_range' and profile['volume'] == .4
                and profile['special'] == 'nr7' and not profile['use126']
                and profile['episode'] == 1):
            variants += [(label, {**profile, **change}, False) for label, change in [
                ('remove quiet-volume filter', {'volume': None}),
                ('remove NR7 filter', {'special': None}),
                ('remove five-day span filter', {'range5': float('inf')}),
                ('remove near-high filter', {'near_high_pct': float('inf')}),
                ('allow new closing highs', {'exclude_new_close_high': False}),
                ('add 126d rank band', {'use126': True}),
                ('volume threshold 0.6x', {'volume': .6}),
                ('volume threshold 0.25x', {'volume': .25}),
                ('within 1% of high', {'near_high_pct': 1.}),
                ('within 5% of high', {'near_high_pct': 5.}),
                ('five-day span 1.5 ATR', {'range5': 1.5}),
                ('five-day span 1 ATR', {'range5': 1.}),
            ]]
        for label, p, exclude_early in variants:
            selected = profile_mask(features, p)
            if exclude_early:
                selected &= regular
            selected = episode_mask(selected, ticker_codes, bars, p['episode'])
            _, counts = annual_count_gate(years, selected, min_signals=0)
            ii = np.sort(features.loc[selected & (features.event_idx >= 0), 'event_idx'].to_numpy(int))
            m = events.iloc[ii].reset_index(drop=True)
            pp = {k: a[ii] for k, a in paths.items()}
            for hold in (5, 10, 21):
                spec = ExitSpec(hold, None, None)
                r = simulate(m, pp, spec)
                accepted = one_position_mask(m, r)
                mm, rr = m[accepted].reset_index(drop=True), r[accepted].reset_index(drop=True)
                later = mm.signal_date.dt.year >= 2018
                rows.append({'candidate': name, 'change': label, 'hold': hold,
                             'signals': int(counts.sum()), 'max_year_signals': int(counts.max()),
                             'passes_annual_cap': bool(counts.max() <= 100), **metrics(rr),
                             'later_trades': int(later.sum()),
                             'later_avg_atr': rr.loc[later, 'net_atr'].mean(),
                             'later_avg_pct': rr.loc[later, 'net_pct'].mean()})
    pd.DataFrame(rows).to_csv(out / 'sensitivity.csv', index=False)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    counts = pd.read_csv(out / 'finalists.csv')
    rows = pd.read_csv(out / 'exit_sweep.csv')
    ci = pd.read_csv(out / 'confidence.csv')
    display = rows[rows.exit.isin(['H2_Snone_Tnone', 'H5_Snone_Tnone', 'H10_Snone_Tnone', 'H21_Snone_Tnone', 'H5_S1.0_T2ATR'])]
    display = display.merge(ci, on=['candidate', 'exit'])
    stable_names = counts[counts.selection_basis.str.startswith('retrospective')].candidate
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for name in stable_names:
        d = display[(display.candidate == name) & display.exit.str.endswith('_Snone_Tnone')].copy()
        d['hold'] = d.exit.str.extract(r'H(\d+)').astype(int)
        d = d.sort_values('hold')
        for ax, col in zip(axes, ('avg_atr', 'later_avg_atr')):
            ax.plot(d.hold, d[col], marker='o', label=name)
    for ax, title in zip(axes, ('Full history', '2018–2026 (retrospective)')):
        ax.set_title(title)
        ax.set_xticks([2, 5, 10, 21])
        ax.set_xlabel('Sessions after entry, no stop or target')
        ax.axhline(0, color='#555', linewidth=.8)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel('Average net ATR per trade')
    fig.suptitle('Consolidation profiles: every historical year <=100 raw signals')
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)
    page = '<!doctype html><html><meta charset="utf-8"><title>Consolidation inside-day research</title><style>body{font:15px system-ui;margin:32px;color:#172b4d}p{max-width:1100px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f8}h2{margin-top:32px}</style><h1>Quiet consolidation within uptrends</h1>'
    page += '<p>Original inside-day/252d-rank/short-rank/SMA10/EMA21 rule retained. 126d band tested with and without. Tomorrow-only stop at signal high; 5 bps per side; signal-day ATR; inherited next-session exit arming and entry+H timing. Universe and price-quality exclusions match the prior study.</p>'
    page += '<p>The annual limit counts ALL qualifying signals before fills, overlap rejection, and terminal-path availability. Every shown profile naturally has at most 100 signals in every observed calendar year. No top-N truncation or hindsight picking of individual trades. 2026 is partial through October 5; a guaranteed future maximum would require a separate causal annual budget.</p>'
    page += '<p>Count-first parameter search, minimum 250 signals total and 50 since 2018, and at least 100 mature fills. Identical signal sets deduplicated. Finalists selected using average 5/10-day expectancy through 2017, requiring at least 50 early and 30 later trades and taking up to two per family; later-period returns are not used to re-rank. This is retrospective research, not untouched out-of-sample evidence. Multiple testing, current-universe survivorship, corporate-action uncertainty, and sparse clustered observations remain. Confidence intervals resample calendar-month blocks and do not correct for the parameter search.</p>'
    page += '<p>A separate consistency shortlist selects the strongest minimum of early/later expectancy at 5/10/21 sessions, requiring 75 early and 50 later trades. These picks explicitly use BOTH eras and are exploratory. Ties prefer no episode-spacing rule and the stronger SMA50/SMA200 trend stack. They are labeled separately from the early-period family leaders.</p>'
    page += '<img src="comparison.png" alt="Consolidation return comparison" style="width:100%;max-width:1100px">'
    page += '<p>Volume is signal-day volume / trailing 63-day mean; unavailable/zero volume cannot pass quiet-volume filters. A 5-day range means max High minus min Low over five completed bars, divided by signal ATR14. Rising SMA50 means higher than 21 sessions earlier. NR7/NR10 allow ties. No-new-closing-high means close strictly below the highest close in the PREVIOUS 252 sessions. Episode spacing follows the repo first-instance convention.</p>'
    page += '<h2>Search accounting</h2><pre>' + html.escape((out / 'search.json').read_text()) + '</pre>'
    for _, c in counts.iterrows():
        name = c.candidate
        page += f'<h2>{html.escape(name)} — {html.escape(c.family)}</h2><p>{html.escape(c.selection_basis)}.</p><p>{html.escape(c.rules)}</p>'
        page += f'<p>{int(c.signals)} raw signals; maximum {int(c.max_year_signals)} in one year; mean {c.mean_year_signals:.1f} per completed year 2002–2025; {int(c.mature_fills)} mature fills.</p>'
        columns = ['exit', 'trades', 'avg_atr', 'avg_pct', 'win_pct', 'pf_atr', 'later_trades', 'later_avg_atr', 'later_avg_pct', 'later_ci_low_atr', 'later_ci_high_atr']
        page += display[display.candidate == name][columns].to_html(index=False, float_format=lambda x: f'{x:.3f}')
    page += '<h2>Annual raw signal counts</h2>' + counts[['candidate']+[f'signals_{y}' for y in range(2002, 2027)]].to_html(index=False)
    page += '<h2>Filter ablations and shortened-session sensitivity</h2><p>Each row changes one clause at fixed time exits, with the one-position rule replayed after the change. Early-close signal days are identified by the shared XNYS calendar. These checks do not select finalists. Rows exceeding 100 signals in any year are inadmissible under the requested constraint.</p>'
    sensitivity_rows = pd.read_csv(out / 'sensitivity.csv')
    for name in stable_names:
        page += f'<h3>{html.escape(name)}</h3>'
        hold = int(counts.set_index('candidate').loc[name, 'selection_basis'].rsplit('H', 1)[1])
        page += sensitivity_rows[(sensitivity_rows.candidate == name) & (sensitivity_rows.hold == hold)].to_html(index=False, float_format=lambda x: f'{x:.3f}')
    (out / 'report.html').write_text(page + '</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'artifacts/inside-day-breakout')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/inside-day-consolidation')
    parser.add_argument('--prices', type=Path, default=ROOT / 'data/master_prices.parquet')
    parser.add_argument('--reuse-features', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'artifacts'):
        parser.error('Outputs must stay under artifacts/')
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse_features:
        features = pd.read_parquet(out / 'signal_features.parquet')
        events = pd.read_parquet(args.base / 'events.parquet')
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['price_sha256'] != hashlib.sha256(args.prices.read_bytes()).hexdigest():
            parser.error('Price snapshot changed; regenerate features and audited paths')
        if manifest['indicator_source_sha256'] != hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest():
            parser.error('Indicators changed; regenerate features')
    else:
        features, events = prepare(args.prices, args.base, out)
    with np.load(args.base / 'paths.npz') as cache:
        paths = {k: cache[k] for k in cache.files}
    candidates, counts, summary = screen(features, events, paths, out)
    chosen = finalists(candidates, counts, summary, events, paths, out)
    sensitivity(features, candidates, chosen, events, paths, out)
    report(out)
    print('Finalists:', ', '.join(chosen), flush=True)
    print(summary[summary.candidate.isin(chosen)].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
