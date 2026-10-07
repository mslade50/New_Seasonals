"""Predeclared impulse/flag momentum research on the frozen stock universe.

Reuses shared indicators, NYSE sessions and the existing daily execution walk.
Entry rules are selected on completed signal bars. This runner does not modify
the live strategy book, broker state, production caches or scheduled scanners.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from indicators import calculate_indicators, impulse_momentum_features
from scripts.backtest_inside_day_breakout import ExitSpec, buy_stop_fill, metrics, one_position_mask, simulate
from scripts.backtest_inside_day_filter_audit import market_open_hours, sharpe
from scripts.backtest_smooth_momentum import research_bars, marks
from trading_calendar import TRADING_DAY

START = pd.Timestamp('2005-01-01')
MAX_HOLD = 21


def definitions():
    out = []
    for family, market, ranked in itertools.product(('flag5', 'flag10', 'dip2', 'break21'), (False, True), (False, True)):
        for entry in (('open', 'limit') if family == 'break21' else ('open', 'stop')):
            out.append({'name': f'F{len(out)+1:03d}', 'family': family, 'market': market,
                        'top': 1 if ranked else None, 'entry': entry})
    return out


def context_mask(f):
    return (f.trend_stack & f.above_ema21 & f.beta126.between(1.2, 3.)
            & (f.efficiency63 >= .15) & (f.largest_step_share63 <= .15)
            & (f.relative_return126 > 0) & (f.dollar_volume63 >= 20_000_000)
            & (f.signal_close >= 10)).fillna(False)


def signal_selection(f, p):
    selected = context_mask(f)
    if p['market']:
        selected &= f.market_above200
    if p['family'].startswith('flag'):
        move = f.impulse5_before3_atr >= 3 if p['family'] == 'flag5' else f.impulse10_before3_atr >= 4
        selected &= (move & f.pullback3_atr.between(-1.5, -.25) & (f.range3_atr <= 2.)
                     & f.drawdown8_atr.between(.25, 1.5) & (f.quiet_rate3 <= .7))
    elif p['family'] == 'dip2':
        selected &= ((f.move21_atr >= 3) & f.pullback2_atr.between(-1.5, -.5)
                     & (f.day_range_atr <= 1.5) & (f.quiet_volume_rate_ratio <= .7))
    else:
        selected &= ((f.signal_close > f.previous_high21) & (f.close_location >= .75)
                     & (f.today_move_atr >= .75) & (f.quiet_volume_rate_ratio >= 1.5))
    selected = selected.fillna(False).to_numpy()
    if p['top']:
        ranked = f.loc[selected].assign(score=lambda x: x.efficiency63 * x.move21_atr)
        ranked = ranked.sort_values(['signal_date', 'score', 'ticker'], ascending=[True, False, True])
        chosen = ranked.groupby('signal_date', sort=False).head(p['top']).index
        selected[:] = False
        selected[chosen] = True
    return selected


def entry_prices(m, paths, p):
    if p['entry'] == 'open':
        return paths['Open'][:, 0].copy()
    if p['entry'] == 'limit':
        limit = m.signal_close.to_numpy() - .25 * m.atr.to_numpy()
        return np.where(paths['Low'][:, 0] <= limit, np.minimum(paths['Open'][:, 0], limit), np.nan)
    lookback = 2 if p['family'] == 'dip2' else 3
    return buy_stop_fill(m[f'box_high{lookback}'].to_numpy(), paths['Open'][:, 0], paths['High'][:, 0])


def prepare(source, out):
    old = json.loads((source / 'manifest.json').read_text())
    snapshot = (source / 'prices.parquet').read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == old['price_sha256']
    raw = pd.read_parquet(io.BytesIO(snapshot))
    raw.date = pd.to_datetime(raw.date).dt.normalize()
    assert not raw.duplicated(['ticker', 'date']).any()
    hours = market_open_hours()
    hours = hours.reindex(pd.date_range(hours.index.min(), hours.index.max(), freq=TRADING_DAY))
    spy = raw.loc[raw.ticker == 'SPY'].sort_values('date').set_index('date').Close.reindex(hours.index)
    end = pd.Timestamp(old['end'])
    frames, parts = [], {k: [] for k in ('Open', 'High', 'Low', 'Close', 'date')}
    offset, coverage = 0, []
    for n, (ticker, g) in enumerate(raw[raw.ticker.isin(old['fixed_universe']) & (raw.date <= end)].groupby('ticker', sort=True)):
        df, valid = research_bars(g, hours)
        df = calculate_indicators(df, {}, ticker)
        f = impulse_momentum_features(df, spy, hours)
        f['ticker'], f['signal_date'], f['signal_idx'] = ticker, df.index, np.arange(len(df))
        f['signal_close'], f['signal_high'], f['signal_low'], f['atr'] = df.Close, df.High, df.Low, df.ATR
        union = np.zeros(len(f), bool)
        for p in definitions():
            # Pool every available trigger before ranking across all stocks.
            union |= signal_selection(f, {**p, 'top': None})
        union &= df.index >= START
        if ticker == 'RAMP':
            union &= df.index < pd.Timestamp('2026-05-18')
        if ticker == 'CBZ':
            union &= df.index < pd.Timestamp('2026-07-29')
        idx = np.flatnonzero(union)
        chosen = idx[idx + 1 + MAX_HOLD < len(df)]
        chosen = chosen[np.array([valid.iloc[j:j+MAX_HOLD+2].all() for j in chosen], bool)]
        chosen = chosen[np.isfinite(df.ATR.to_numpy()[chosen]) & (df.ATR.to_numpy()[chosen] > 0)]
        ff = f.iloc[idx].copy()
        ff['event_idx'] = -1
        ff.loc[df.index[chosen], 'event_idx'] = offset + np.arange(len(chosen))
        position = chosen[:, None] + 1 + np.arange(MAX_HOLD + 1)[None, :]
        for k in parts:
            values = df.index.to_numpy() if k == 'date' else df[k].to_numpy(float)
            parts[k].append(values[position])
        frames.append(ff.reset_index(drop=True))
        coverage.append({'ticker': ticker, 'raw_pool': len(idx), 'mature_clean_paths': len(chosen)})
        offset += len(chosen)
        if n % 100 == 0:
            print(f'Prepared {n+1}/{old["covered"]}: {ticker}', flush=True)
    f = pd.concat(frames, ignore_index=True)
    paths = {k: np.concatenate(v) for k, v in parts.items()}
    f.to_parquet(out / 'features.parquet', index=False)
    np.savez_compressed(out / 'paths.npz', **paths)
    pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
    manifest = {**old, 'source_snapshot': str(source), 'rules': definitions(),
                'indicator_source_sha256': hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest(),
                'research_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'notes': ['Fixed current stock universe, not point-in-time membership',
                          'Predeclared impulse/flag, shallow dip and confirmed 21-session breakout hypotheses',
                          'Causal signal-date beta/smoothness/liquidity; tomorrow-only orders',
                          'Common mature clean 21-session cohort; RAMP/CBZ exclusions retained; wider corporate-action audit incomplete',
                          'Targets/stops arm the session after entry, matching existing daily engine']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return f, paths, manifest


def evaluate(f, paths, manifest, out):
    sessions = pd.date_range(START, manifest['end'], freq=TRADING_DAY)
    locations = sessions.get_indexer(pd.to_datetime(paths['date'].ravel())).reshape(paths['date'].shape)
    assert (locations >= 0).all() and (np.diff(locations, axis=1) == 1).all()
    rows, annual, trades, daily = [], [], [], {}
    for p in definitions():
        raw = f.loc[signal_selection(f, p)]
        m = raw[raw.event_idx >= 0].reset_index(drop=True).copy()
        pp = {k: v[m.event_idx.to_numpy(int)] for k, v in paths.items()}
        fill = entry_prices(m, pp, p)
        usable = np.isfinite(fill) & (fill > 0)
        m = m.loc[usable].reset_index(drop=True)
        m['entry'], m['entry_idx'] = fill[usable], m.signal_idx + 1
        pp = {k: v[usable] for k, v in pp.items()}
        m['entry_date'] = pp['date'][:, 0]
        for hold, stop in itertools.product((3, 5, 10, 21), (None, 2.)):
            spec = ExitSpec(hold, stop, None)
            r = simulate(m, pp, spec)
            accepted = one_position_mask(m, r)
            mm, rr = m.loc[accepted].reset_index(drop=True), r.loc[accepted].reset_index(drop=True)
            ap = {k: v[accepted] for k, v in pp.items()}
            key = p['name'] + '_' + spec.name
            pnl, live = marks(mm, rr, ap, sessions)
            daily[key] = np.array([pnl, live])
            row = {'key': key, 'candidate': p['name'], 'family': p['family'], 'entry': p['entry'],
                   'top': p['top'], 'raw_signals': len(raw), **metrics(rr)}
            for era, select, lo, hi in [('train', mm.signal_date < '2018-01-01', 2005, 2017),
                                      ('recent', mm.signal_date >= '2018-01-01', 2018, 2025),
                                      ('full', pd.Series(True, index=mm.index), 2005, 2025)]:
                tt, rt = mm.loc[select].reset_index(drop=True), rr.loc[select].reset_index(drop=True)
                ep, el = marks(tt, rt, {k: v[select.to_numpy()] for k, v in ap.items()}, sessions)
                period = (sessions.year >= lo) & (sessions.year <= (hi if era == 'train' else 2026))
                count = tt.signal_date.dt.year.value_counts().reindex(range(lo, hi+1), fill_value=0)
                row.update({era+'_tim': sharpe(ep[el]), era+'_all': sharpe(ep[period]),
                            era+'_pf': metrics(rt).get('pf_atr', np.nan), era+'_trades': len(tt),
                            era+'_mean_trades_year': count.mean(), era+'_min_year': int(count.min()),
                            era+'_max_year': int(count.max()), era+'_mean_beta': tt.beta126.mean(),
                            era+'_mean_efficiency': tt.efficiency63.mean()})
            row['frequency_ok'] = (row['recent_mean_trades_year'] > 10 and row['full_max_year'] <= 200)
            rows.append(row)
            for year in range(2005, 2027):
                select = mm.signal_date.dt.year == year
                annual.append({'key': key, 'year': year, 'trades': int(select.sum()),
                               'raw_signals': int((raw.signal_date.dt.year == year).sum()),
                               'net_atr': float(rr.loc[select, 'net_atr'].sum()),
                               'calendar_pnl_atr': float(pnl[sessions.year == year].sum())})
            trades.append(pd.concat([mm[['ticker','signal_date','entry_date','entry','atr','beta126','efficiency63']], rr], axis=1)
                          .assign(key=key, exit_date=ap['date'][np.arange(len(rr)), rr.exit_day]))
        print(f'Evaluated {p["name"]}: {p["family"]}/{p["entry"]}, top={p["top"]}, {len(raw)} raw signals', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / 'results.csv', index=False)
    pd.DataFrame(annual).to_csv(out / 'annual.csv', index=False)
    pd.concat(trades, ignore_index=True).to_parquet(out / 'trades.parquet', index=False)
    np.savez_compressed(out / 'daily.npz', dates=sessions.to_numpy(), **daily)
    early = results[(results.train_mean_trades_year > 10) & (results.train_max_year <= 200)]
    leaders = early.sort_values('train_tim', ascending=False).groupby('family', sort=False).head(1)
    leaders.to_csv(out / 'early_selected_family_leaders.csv', index=False)
    columns = ['key','family','train_tim','recent_tim','full_tim','recent_pf','recent_mean_trades_year','recent_min_year','full_max_year']
    print('EARLY SELECTED LEADERS\n' + leaders[columns].to_string(index=False), flush=True)
    print('EXPLORATORY RECENT LEADERS\n' + results[results.frequency_ok].sort_values('recent_tim', ascending=False).head(10)[columns].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/smooth-momentum-expanded')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/momentum-impulse')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'artifacts'):
        parser.error('Output must be under artifacts/')
    if out == args.source.resolve():
        parser.error('Output must differ from the retained source snapshot')
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['rules'] == definitions()
        assert manifest['indicator_source_sha256'] == hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest()
        assert manifest['research_source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        f = pd.read_parquet(out / 'features.parquet')
        with np.load(out / 'paths.npz') as z:
            paths = {k: z[k] for k in z.files}
    else:
        f, paths, manifest = prepare(args.source.resolve(), out)
    evaluate(f, paths, manifest, out)


if __name__ == '__main__':
    main()
