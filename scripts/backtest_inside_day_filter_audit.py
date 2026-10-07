"""Ablate consolidation gates, preserving the 40% volume choice.

Research only. Uses the prior audited price snapshot/universe; no live changes.
All combinations within the five added gates and seven original gates are evaluated.
Fixed alternative signal ideas are disclosed separately, without retuning.
The selected consolidation breakout uses 40% volume per market-open hour.
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
from equity_sessions import calendar
from indicators import calculate_indicators, consolidation_audit_features
from scripts.backtest_inside_day_breakout import (
    ExitSpec, buy_stop_fill, metrics, one_position_mask, simulate, valid_ohlc,
)
from trading_calendar import TRADING_DAY

BASE_FLAGS = ('rank252', 'rank5', 'rank10', 'rank21', 'inside', 'sma10', 'ema21')
EXTRA_FLAGS = ('trend', 'near_high', 'range5', 'nr7', 'not_closing_high')
SELECTED_RULE_NAME = 'session_adjusted_volume'
SELECTED_HOLD_DAYS = 10
START, END, MAX_HOLD = pd.Timestamp('2002-01-01'), pd.Timestamp('2026-10-05'), 21


def flag_frame(df, hours):
    f = consolidation_audit_features(df, hours)
    return pd.DataFrame({
        'rank252': df.rank_ret_252d.between(50, 90),
        'rank5': df.rank_ret_5d < 95, 'rank10': df.rank_ret_10d < 95,
        'rank21': df.rank_ret_21d < 95,
        'inside': (df.High < df.High.shift()) & (df.Low > df.Low.shift()),
        'sma10': df.Close > df.SMA10, 'ema21': df.Close > df.EMA21,
        'trend': f.trend_stack, 'near_high': f.distance_high252_pct <= 3,
        'range5': f.range5_atr <= 2, 'nr7': f.nr7,
        'not_closing_high': f.not_new_closing_high252,
        'quiet': f.quiet_volume_ratio <= .4,
        'quiet_rate': f.quiet_volume_rate_ratio <= .4,
        'nr10': f.nr10, 'double_inside': f.double_inside,
        'squeeze': f.atr5_to_atr21 <= .7,
        'shelf': (f.range5_atr <= 1.5) & (f.range10_atr <= 2) & (f.close_position5 >= .65),
        'near_high5': f.distance_high252_pct <= 5,
        'small_day': f.day_range_atr <= .75,
        'ema21_touch': (df.Low <= df.EMA21) & (df.Close > df.EMA21),
        'failed_low': (df.Low < df.Low.shift()) & (df.Close > df.Low.shift()),
    }, index=df.index).fillna(False).astype(bool)


def definitions():
    rows = []
    for keep in itertools.product((False, True), repeat=len(EXTRA_FLAGS)):
        flags = [name for name, included in zip(EXTRA_FLAGS, keep) if included]
        removed = [name for name, included in zip(EXTRA_FLAGS, keep) if not included]
        rows.append({'name': 'baseline' if not removed else 'drop_' + '_'.join(removed),
                     'group': 'added_gate_factorial', 'required': ['quiet', *BASE_FLAGS, *flags],
                     'description': 'Drop ' + ', '.join(removed) if removed else 'Full raw-volume reference'})
    for keep in itertools.product((False, True), repeat=len(BASE_FLAGS)):
        removed = [name for name, included in zip(BASE_FLAGS, keep) if not included]
        if not removed:
            continue  # The full rule is already the baseline above.
        name = 'base_drop_all' if len(removed)==len(BASE_FLAGS) else 'base_drop_' + '_'.join(removed)
        rows.append({'name': name, 'group': 'original_gate_factorial',
                     'required': ['quiet', *EXTRA_FLAGS, *[x for x, included in zip(BASE_FLAGS, keep) if included]],
                     'description': 'Full rule except original ' + ', '.join(removed)})
    original_ex_inside = [x for x in BASE_FLAGS if x != 'inside']
    common = ['quiet', 'trend', 'near_high', 'range5', 'not_closing_high']
    ideas = [
        ('NR10', [*BASE_FLAGS, *common, 'nr10'], 'Replace NR7 with NR10; other rules fixed'),
        ('double_inside', [*BASE_FLAGS, *common, 'double_inside'], 'Replace NR7 with two consecutive inside days'),
        ('squeeze', [*BASE_FLAGS, *common, 'squeeze'], 'Replace NR7 with ATR5/ATR21 <= 0.7'),
        ('multi_day_shelf', [*original_ex_inside, 'quiet', 'trend', 'near_high', 'not_closing_high', 'shelf'],
         'No inside/NR7 gate; 5d span <=1.5 ATR, 10d span <=2 ATR, close in upper 35% of 5d shelf'),
        ('EMA21_pullback', ['quiet', 'trend', 'near_high5', 'range5', 'small_day', 'ema21_touch', 'not_closing_high'],
         'Quiet touch/reclaim of EMA21; rising SMA50>SMA200, within 5% of annual high, day range <=0.75 ATR, 5d span <=2 ATR'),
        ('failed_downside_break', ['quiet', 'trend', 'near_high5', 'range5', 'small_day', 'failed_low', 'ema21', 'not_closing_high'],
         'Quiet lower low reclaiming prior low; rising SMA50>SMA200, above EMA21, near annual high'),
        ('session_adjusted_volume', ['quiet_rate', *BASE_FLAGS, *EXTRA_FLAGS],
         'Full rule; 40% volume ceiling measured per market-open hour'),
        ('exclude_yearend_week', ['quiet', *BASE_FLAGS, *EXTRA_FLAGS, 'not_yearend'],
         'Full rule excluding last five NYSE sessions of December'),
    ]
    rows += [{'name': name, 'group': 'fixed_alternative', 'required': required, 'description': desc}
             for name, required, desc in ideas]
    return rows


def selected_mask(flags, required):
    return flags[list(required)].all(axis=1).to_numpy()


def selected_definition():
    """Approved research signal; retain all other price gates and the 10d exit."""
    return next(rule for rule in definitions() if rule['name'] == SELECTED_RULE_NAME)


def market_open_hours():
    schedule = calendar().schedule
    hours = (schedule['close'] - schedule['open']).dt.total_seconds() / 3600
    hours.index = hours.index.tz_localize(None).normalize()
    return hours


def consolidation_breakout_mask(df, hours=None):
    """Selected price/volume gates on shared indicators, using known NYSE hours.

    Universe and acquisition exclusions remain separate in prepare(). Unknown
    session lengths cannot qualify for the per-hour quiet-volume requirement.
    """
    flags = flag_frame(df, market_open_hours() if hours is None else hours)
    return pd.Series(selected_mask(flags, selected_definition()['required']), index=df.index)


def write_selected_outputs(rows, signals, trades, out):
    """Save an explicit selected research snapshot alongside the comparisons."""
    selected_row = rows[(rows.rule == SELECTED_RULE_NAME) & (rows.hold == SELECTED_HOLD_DAYS)]
    if len(selected_row) != 1:
        raise ValueError('Selected research result must have exactly one matching exit')
    selected_signals = signals[signals.rule == SELECTED_RULE_NAME].copy()
    selected_trades = trades[(trades.rule == SELECTED_RULE_NAME) & (trades.hold == SELECTED_HOLD_DAYS)].copy()
    row = selected_row.iloc[0].to_dict()
    assert len(selected_signals) == row['signals'] and len(selected_trades) == row['trades']
    selected_signals.to_csv(out / 'selected_signals.csv', index=False)
    selected_trades.to_parquet(out / 'selected_trades.parquet', index=False)
    metadata = {
        'name': 'Consolidation Breakout', 'status': 'selected_research',
        'rule': selected_definition(), 'volume_ceiling': .4, 'volume_lookback_sessions': 63,
        'volume_basis': 'per_market_open_hour',
        'volume_formula': '(Volume / session_hours) / trailing_63_session_mean(Volume / session_hours)',
        'entry': 'Tomorrow-only buy stop at signal high; gap fill at max(open, signal high)',
        'hold_days': SELECTED_HOLD_DAYS, 'time_exit': 'Entry session index + hold_days',
        'stop': None, 'target': None, 'cost_bps_per_side': 5,
        'annual_signal_ceiling': 100, 'metrics': row,
        'corporate_action_review': 'Partial; RAMP excluded from 2026-05-18 in the audited pool',
    }
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        metadata['price_sha256'] = manifest['price_sha256']
        metadata['data_end'] = manifest['end']
    (out / 'selected_strategy.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def cache_covers_rules(cached, proposed, available_columns):
    """A new conjunction is covered if an old conjunction is less restrictive.

    This permits more ablations from a pool that already included the fully
    relaxed parent. It never permits a new, broader or unavailable condition.
    """
    old = [set(r['required']) for r in cached]
    available = set(available_columns)
    return all(set(r['required']) <= available
               and any(parent <= set(r['required']) for parent in old) for r in proposed)


def prepare(prices, base, out):
    original = json.loads((base / 'manifest.json').read_text())
    snapshot = prices.read_bytes()
    digest = hashlib.sha256(snapshot).hexdigest()
    if digest != original['price_sha256']:
        raise ValueError('Price snapshot differs from the audited prior study')
    wanted = set(pd.read_csv(base / 'coverage.csv').ticker)
    raw = pd.read_parquet(io.BytesIO(snapshot))
    raw.date = pd.to_datetime(raw.date).dt.normalize()
    raw = raw[raw.ticker.isin(wanted) & (raw.date <= END)]
    if raw.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate ticker dates')
    hours = market_open_hours()
    yearend = set()
    for _, g in hours[hours.index.month == 12].groupby(hours[hours.index.month == 12].index.year):
        yearend.update(g.index[-5:])
    rules = definitions()
    frames, events, parts = [], [], {x: [] for x in ('Open', 'High', 'Low', 'Close', 'date')}
    offset = 0
    for k, (ticker, g) in enumerate(raw.groupby('ticker', sort=True)):
        g = g.sort_values('date').set_index('date').drop(columns='ticker')
        valid = valid_ohlc(g)
        g.loc[~valid, ['Open', 'High', 'Low', 'Close']] = np.nan
        df = calculate_indicators(g, {}, ticker)
        flags = flag_frame(df, hours)
        flags['not_yearend'] = ~flags.index.isin(yearend)
        union = np.zeros(len(flags), bool)
        for rule in rules:
            union |= selected_mask(flags, rule['required'])
        union &= (df.index >= START) & (df.index <= END)
        # Confirmed post-announcement RAMP exclusion, retaining pre-deal history.
        if ticker == 'RAMP':
            union &= df.index < pd.Timestamp('2026-05-18')
        idx = np.flatnonzero(union)
        f = flags.iloc[idx].copy()
        f['ticker'], f['signal_date'], f['signal_idx'] = ticker, f.index, idx
        f['event_idx'] = -1
        mature = idx[idx + 1 + MAX_HOLD < len(df)]
        good = np.array([valid.iloc[i-1:i+MAX_HOLD+2].all() for i in mature], dtype=bool)
        mature = mature[good]
        fill = buy_stop_fill(df.High.to_numpy()[mature], df.Open.to_numpy()[mature+1], df.High.to_numpy()[mature+1])
        usable = np.isfinite(fill) & np.isfinite(df.ATR.to_numpy()[mature]) & (df.ATR.to_numpy()[mature] > 0)
        usable &= fill > df.Low.to_numpy()[mature]
        chosen, fill = mature[usable], fill[usable]
        if len(chosen):
            f.loc[df.index[chosen], 'event_idx'] = offset + np.arange(len(chosen))
            events.append(pd.DataFrame({'ticker': ticker, 'signal_date': df.index[chosen],
                'signal_idx': chosen, 'entry_idx': chosen+1, 'entry_date': df.index[chosen+1],
                'entry': fill, 'signal_high': df.High.to_numpy()[chosen],
                'signal_low': df.Low.to_numpy()[chosen], 'atr': df.ATR.to_numpy()[chosen]}))
            positions = chosen[:, None] + 1 + np.arange(MAX_HOLD+1)[None, :]
            for col in parts:
                values = df.index.to_numpy() if col == 'date' else df[col].to_numpy(float)
                parts[col].append(values[positions])
            offset += len(chosen)
        frames.append(f.reset_index(drop=True))
        if k % 100 == 0:
            print(f'Audit features {k+1}/{len(wanted)}: {ticker}', flush=True)
    features = pd.concat(frames, ignore_index=True)
    meta = pd.concat(events, ignore_index=True)
    paths = {k: np.concatenate(v) for k, v in parts.items()}
    features.to_parquet(out / 'features.parquet', index=False)
    meta.to_parquet(out / 'events.parquet', index=False)
    np.savez_compressed(out / 'paths.npz', **paths)
    manifest = {'price_sha256': digest, 'start': str(START.date()), 'end': str(END.date()),
                'universe': len(wanted), 'raw_union_signals': len(features), 'union_fills': len(meta),
                'excluded_histories': original['excluded_tickers'],
                'deal_exclusion': {'RAMP': '2026-05-18'}, 'merger_audit': 'partial',
                'indicator_source_sha256': hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest(),
                'rules': rules}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return features, meta, paths


def daily_mtm(meta, result, paths, hold, sessions):
    close = paths['Close'][:, :hold+1].astype(float)
    atr, entry = meta.atr.to_numpy(float), meta.entry.to_numpy(float)
    changes = np.empty_like(close)
    changes[:, 0] = (close[:, 0] - entry*1.0005) / atr
    changes[:, 1:] = np.diff(close, axis=1) / atr[:, None]
    changes[:, -1] -= close[:, -1]*.0005 / atr
    rounding = result.net_atr.to_numpy() - changes.sum(axis=1)
    expected = (entry*1.0005 - (meta.entry.to_numpy()*1.0005).astype(float)) / atr
    np.testing.assert_allclose(rounding, expected, atol=1e-9)
    changes[:, -1] += rounding
    loc = sessions.get_indexer(pd.to_datetime(paths['date'][:, :hold+1].ravel()))
    if (loc < 0).any():
        raise ValueError('Accepted paths contain non-NYSE sessions')
    pnl = np.zeros(len(sessions))
    np.add.at(pnl, loc, changes.ravel())
    live = np.zeros(len(sessions), bool)
    live[loc] = True
    np.testing.assert_allclose(pnl.sum(), result.net_atr.sum(), atol=1e-8)
    return pnl, live


def sharpe(pnl):
    return float(np.sqrt(252)*np.mean(pnl)/np.std(pnl, ddof=1)) if len(pnl)>1 and np.std(pnl)>0 else np.nan


def evaluate(features, events, paths, out):
    sessions = pd.date_range(START, END, freq=TRADING_DAY)
    rows, annual, baseline, selected_rows, trade_rows = [], [], {}, [], []
    for rule in definitions():
        selected = selected_mask(features, rule['required'])
        counts = features.loc[selected].signal_date.dt.year.value_counts()
        ii = np.sort(features.loc[selected & (features.event_idx>=0), 'event_idx'].to_numpy(int))
        m = events.iloc[ii].reset_index(drop=True)
        pp = {k: v[ii] for k, v in paths.items()}
        selected_rows.append(features.loc[selected, ['ticker', 'signal_date', 'event_idx']].assign(rule=rule['name']))
        for hold in (5, 10, 21):
            r = simulate(m, pp, ExitSpec(hold, None, None))
            keep = one_position_mask(m, r)
            mm, rr = m[keep].reset_index(drop=True), r[keep].reset_index(drop=True)
            accepted_paths = {k: v[keep] for k, v in pp.items()}
            exit_dates = accepted_paths['date'][np.arange(len(rr)), rr.exit_day]
            trade_rows.append(pd.concat([mm, rr], axis=1).assign(rule=rule['name'], hold=hold, exit_date=exit_dates))
            pnl, active = daily_mtm(mm, rr, accepted_paths, hold, sessions)
            recent = (mm.signal_date >= '2018-01-01').to_numpy()
            later_pnl, later_active = daily_mtm(mm[recent].reset_index(drop=True), rr[recent].reset_index(drop=True),
                                               {k: v[recent] for k, v in accepted_paths.items()}, hold, sessions)
            period = sessions >= '2018-01-01'
            row = {'rule': rule['name'], 'group': rule['group'], 'description': rule['description'],
                   'hold': hold, 'signals': int(counts.sum()), 'max_year_signals': int(counts.max()) if len(counts) else 0,
                   'passes_annual_cap': bool(len(counts)==0 or counts.max()<=100),
                   'signals_2025': int(counts.get(2025, 0)), 'signals_2026': int(counts.get(2026, 0)),
                   **metrics(rr), 'sharpe_all': sharpe(pnl), 'sharpe_tim': sharpe(pnl[active]),
                   'later_trades': int(recent.sum()), 'later_avg_atr': rr.loc[recent, 'net_atr'].mean(),
                   'later_pf': metrics(rr.loc[recent]).get('pf_atr', np.nan),
                   'later_sharpe': sharpe(later_pnl[period]),
                   'later_sharpe_tim': sharpe(later_pnl[later_active])}
            rows.append(row)
            for year in range(2002, 2027):
                year_mask = sessions.year == year
                annual.append({'rule': rule['name'], 'hold': hold, 'year': year,
                               'raw_signals': int(counts.get(year, 0)), 'calendar_net_atr': float(pnl[year_mask].sum())})
            if rule['name']=='baseline' and hold==10:
                assert len(mm)==423 and counts.sum()==662 and counts.max()==74
                prior = pd.read_csv(ROOT / 'artifacts/inside-day-consolidation/daily_mtm.csv', index_col='date', parse_dates=True)
                np.testing.assert_allclose(pnl, prior.daily_pnl_per_unit_atr_risk.reindex(sessions), atol=1e-8)
                baseline = {'row': row, 'pnl': pnl.tolist()}
        print(f'{rule["name"]}: {counts.sum()} raw signals; max {counts.max() if len(counts) else 0}/year', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / 'results.csv', index=False)
    pd.DataFrame(annual).to_csv(out / 'annual.csv', index=False)
    signals = pd.concat(selected_rows, ignore_index=True)
    trades = pd.concat(trade_rows, ignore_index=True)
    signals.to_parquet(out / 'selected_signals.parquet', index=False)
    trades.to_parquet(out / 'trades.parquet', index=False)
    write_selected_outputs(results, signals, trades, out)
    (out / 'evaluated_rules.json').write_text(json.dumps(definitions(), indent=2))
    (out / 'baseline_verification.json').write_text(json.dumps(baseline, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=ROOT / 'data/master_prices.parquet')
    parser.add_argument('--base', type=Path, default=ROOT / 'artifacts/inside-day-breakout')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/inside-day-filter-audit')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'artifacts'):
        parser.error('Output must remain under artifacts/')
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['indicator_source_sha256'] != hashlib.sha256((ROOT / 'indicators.py').read_bytes()).hexdigest():
            parser.error('Indicators changed; rebuild features')
        f, m = pd.read_parquet(out / 'features.parquet'), pd.read_parquet(out / 'events.parquet')
        if not cache_covers_rules(manifest['rules'], definitions(), f.columns):
            parser.error('New definitions exceed cached candidate coverage; rebuild features')
        with np.load(out / 'paths.npz') as cache:
            paths = {k: cache[k] for k in cache.files}
    else:
        f, m, paths = prepare(args.prices, args.base, out)
    rows = evaluate(f, m, paths, out)
    print(rows[rows.hold==10].to_string(index=False), flush=True)


if __name__=='__main__':
    main()
