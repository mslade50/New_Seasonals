"""Single source of truth for signal filters (consolidated 2026-07-16).

Until now the filter stack was implemented twice and hand-synced:
`daily_scan.check_signal` (~420 lines, decides what STAGES LIVE) and
`pages/strat_backtester.get_historical_mask` (~580 lines, decides what the
LEDGER books). Drift between them produced real bugs (the use_ath/52wh case,
the atr_sznl fail-open case, the SPY/QQQ ATR-floor exemption below). Both
now delegate here: `check_signal` is `evaluate_filter_mask(...).iloc[-1]`
in live mode, `get_historical_mask` is the same call in backtest mode —
so scan/ledger parity is structural, not a discipline.

Semantics are the SCAN's (live truth) wherever the two disagreed, with two
deliberate mode differences and one scan-only strip:

- ``mode='live'`` (daily_scan): dial_filters FAIL CLOSED — missing/stale
  fragility cache or missing column rejects the signal (never trade an
  unknown regime through an explicitly configured dial gate). Staleness =
  cache older than FRAG_STALE_TD business days vs the signal date.
- ``mode='backtest'`` (engine/ledger): dial_filters pass through NaN /
  missing dials (pre-2016 history is ungradeable point-in-time; penalizing
  it would be lookahead in reverse).
- T+1 gates (``use_t1_open_filter``, ``use_t1_gap_kill``) evaluate NextOpen
  and belong to the ENGINE's mask; the live scan cannot see tomorrow's open
  — it stamps the specs for order_staging to enforce at the IBKR open. The
  check_signal wrapper strips them (daily_scan side); backtest mode keeps
  them.

Scan-truth unifications adopted here (each was a live-vs-ledger divergence;
config audit 2026-07-16 — only the first two bind on the current book):
- ETF_ATR_EXEMPT (SPY/QQQ/IWM/DIA) waive min_atr_pct — the engine used to
  enforce the floor and silently dropped low-vol-regime index trades that
  live stages (MonFri Reversion, Monday Dip, Indices Oversold Bounce).
- VIX filter reads 0.0 when the column is missing (fail toward reject when
  vix_min > 0) instead of silently passing.
- month filter (scan-only feature) now exists in the mask.
- or_filter_groups fail CLOSED when a group resolves to nothing; missing
  columns inside a group evaluate at the scan's neutral 50.0.
- ref/xsec rank filters evaluate missing columns at neutral 50.0.
- perf legacy (use_perf_rank) merges into perf_filters and
  perf_first_instance applies to the COMBINED condition (scan semantics).
- 52w_first_instance default True (scan default; every book config is
  explicit so this only guards future configs).
- gap filter falls back to GapCount_21 then zeros, never KeyError.
- ma-dist filter supports the 52-Week-High / All-Time-High targets and the
  use_dist_filter alias; degenerate rows (ATR/MA/Close <= 0) evaluate at 0.
- DOW filter with an empty allowed_days list rejects (scan behavior).
- range-in-ATR passes through rows with ATR <= 0 (scan behavior).

Engine-only capabilities that remain available to any caller: 52w_lag,
transition_filters, or-group superset, ma_touch. The scan gains them for
free the day a config uses them.

Guards: tests/test_filters_consolidation.py (semantics),
tests/test_atr_sznl_parity.py; ship-time proof:
scratch/verify_filters_consolidation.py (old-vs-new booleans across the
full book on real data, both sides).
"""
import os

import numpy as np
import pandas as pd

# Index/mega ETFs exempt from the min ATR%-of-price floor: their ATR% sits
# below single-stock floors in calm regimes by construction, and the floor
# exists to exclude untradeably-dead names, not SPY. (Moved from daily_scan
# 2026-07-16; max_atr_pct still applies to them.)
ETF_ATR_EXEMPT = {'SPY', 'QQQ', 'IWM', 'DIA'}

# Dial gate staleness: reject (live mode) when the fragility cache's last
# row is more than this many business days older than the signal date.
FRAG_STALE_TD = 3

_FRAG_DF_CACHE = {}


def get_fragility_df_cached():
    """Load data/rd2_fragility.parquet once per process. None if missing."""
    if 'loaded' in _FRAG_DF_CACHE:
        return _FRAG_DF_CACHE['loaded']
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, 'data', 'rd2_fragility.parquet')
        df = pd.read_parquet(path) if os.path.exists(path) else None
        if df is not None:
            df.index = pd.to_datetime(df.index).normalize()
            try:
                df.index = df.index.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            df = df.sort_index()
    except Exception:
        df = None
    _FRAG_DF_CACHE['loaded'] = df
    return df


def _col_or(df, col, fill, n):
    return df[col].values if col in df.columns else np.full(n, fill, dtype=float)


def evaluate_filter_mask(df, params, sznl_map=None, ticker_name="UNK", mode="backtest"):
    """Boolean mask over df's rows: True where every configured filter passes.

    The last row is "today" for the live scan; the engine consumes the whole
    series as candidate signal dates. NaN comparisons resolve False (reject)
    everywhere except the documented pass-throughs.
    """
    n = len(df)
    conditions = []

    # Trend filter
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        conditions.append(df['Close'].values > df['SMA200'].values)
    elif trend_opt == "Price > Rising 200 SMA":
        sma200 = df['SMA200'].values
        conditions.append((df['Close'].values > sma200) & (sma200 > np.roll(sma200, 1)))
    elif trend_opt == "Not Below Declining 200 SMA":
        sma200 = df['SMA200'].values
        conditions.append(~((df['Close'].values < sma200) & (sma200 < np.roll(sma200, 1))))
    elif trend_opt == "Price < 200 SMA":
        conditions.append(df['Close'].values < df['SMA200'].values)
    elif trend_opt == "Price < Falling 200 SMA":
        sma200 = df['SMA200'].values
        conditions.append((df['Close'].values < sma200) & (sma200 < np.roll(sma200, 1)))
    elif ("Market" in trend_opt or "SPY" in trend_opt) and 'Market_Above_SMA200' in df.columns:
        if ">" in trend_opt:
            conditions.append(df['Market_Above_SMA200'].values.astype(bool))
        elif "<" in trend_opt:
            conditions.append(~df['Market_Above_SMA200'].values.astype(bool))

    # Price/volume/age liquidity gates
    conditions.append(df['Close'].values >= params.get('min_price', 0))
    conditions.append(df['vol_ma'].values >= params.get('min_vol', 0))
    conditions.append(df['age_years'].values >= params.get('min_age', 0))
    conditions.append(df['age_years'].values <= params.get('max_age', 100))

    if 'ATR_Pct' in df.columns:
        atr_pct = df['ATR_Pct'].values
        # ETF_ATR_EXEMPT waives the MIN floor only (scan semantics — the
        # engine used to enforce it and drop index trades live stages)
        if str(ticker_name).upper() in ETF_ATR_EXEMPT:
            conditions.append(atr_pct <= params.get('max_atr_pct', 1000))
        else:
            conditions.append((atr_pct >= params.get('min_atr_pct', 0))
                              & (atr_pct <= params.get('max_atr_pct', 1000)))

    if params.get('require_close_gt_open', False):
        conditions.append(df['Close'].values > df['Open'].values)

    # Breakout mode
    bk_mode = params.get('breakout_mode', 'None')
    if bk_mode == "Close > Prev Day High":
        conditions.append(df['Close'].values > df['PrevHigh'].values)
    elif bk_mode == "Close < Prev Day Low":
        conditions.append(df['Close'].values < df['PrevLow'].values)

    # Candle range filter (% of day's range)
    if params.get('use_range_filter', False):
        rn_val = df['RangePct'].values * 100
        conditions.append((rn_val >= params.get('range_min', 0))
                          & (rn_val <= params.get('range_max', 100)))

    # Range in ATR filter — rows with ATR <= 0 pass through (scan behavior)
    if params.get('use_range_atr_filter', False):
        if 'range_in_atr' in df.columns:
            ria = df['range_in_atr'].values.astype(float)
        else:
            atr_v = df['ATR'].values.astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ria = (df['High'].values - df['Low'].values) / atr_v
        logic = params.get('range_atr_logic', 'Between')
        if logic == '>':
            cond = ria > params.get('range_atr_min', 0)
        elif logic == '<':
            cond = ria < params.get('range_atr_max', 99)
        else:
            cond = ((ria >= params.get('range_atr_min', 0))
                    & (ria <= params.get('range_atr_max', 99)))
        atr_ok = df['ATR'].values.astype(float) > 0
        conditions.append(np.where(atr_ok, cond, True))

    # Day of week filter — empty allowed_days REJECTS (scan behavior)
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        if 'DayOfWeekVal' in df.columns:
            dows = df['DayOfWeekVal'].values
        else:
            dows = df.index.dayofweek.values
        if allowed:
            conditions.append(np.isin(dows, allowed))
        else:
            conditions.append(np.zeros(n, dtype=bool))

    # Presidential cycle filter
    if 'allowed_cycles' in params:
        allowed = params['allowed_cycles']
        if allowed and len(allowed) < 4:
            conditions.append(np.isin(df.index.year % 4, allowed))

    # Month filter (was scan-only until 2026-07-16)
    if params.get('use_month_filter', False):
        allowed_months = params.get('allowed_months', list(range(1, 13)))
        conditions.append(np.isin(df.index.month, allowed_months))

    # Gap filter — falls back GapCount_{lb} -> GapCount_21 -> zeros
    if params.get('use_gap_filter', False):
        lookback = params.get('gap_lookback', 21)
        col_name = f'GapCount_{lookback}'
        if col_name not in df.columns:
            col_name = 'GapCount_21'
        g_val = _col_or(df, col_name, 0.0, n)
        g_thresh = params.get('gap_thresh', 0)
        g_logic = params.get('gap_logic', '>')
        if g_logic == ">":
            conditions.append(g_val > g_thresh)
        elif g_logic == "<":
            conditions.append(g_val < g_thresh)
        elif g_logic == "=":
            conditions.append(g_val == g_thresh)

    # Accumulation / distribution count filters (skip when column missing —
    # both implementations agreed)
    if params.get('use_acc_count_filter', False):
        col = f"AccCount_{params.get('acc_count_window', 21)}"
        if col in df.columns:
            v = df[col].values
            logic = params.get('acc_count_logic', '=')
            t = params.get('acc_count_thresh', 0)
            if logic == ">":
                conditions.append(v > t)
            elif logic == "<":
                conditions.append(v < t)
            elif logic == "=":
                conditions.append(v == t)

    if params.get('use_dist_count_filter', False):
        col = f"DistCount_{params.get('dist_count_window', 21)}"
        if col in df.columns:
            v = df[col].values
            logic = params.get('dist_count_logic', '>')
            t = params.get('dist_count_thresh', 0)
            if logic == ">":
                conditions.append(v > t)
            elif logic == "<":
                conditions.append(v < t)
            elif logic == "=":
                conditions.append(v == t)

    # Performance rank filters. Scan semantics: the legacy single-window
    # (use_perf_rank) merges into the list, and perf_first_instance applies
    # to the COMBINED condition of all perf filters.
    _perf_filters = list(params.get('perf_filters', []))
    if params.get('use_perf_rank', False):
        legacy = {
            'window': params['perf_window'],
            'logic': params['perf_logic'],
            'thresh': params['perf_thresh'],
            'consecutive': params.get('perf_consecutive', 1),
        }
        already = any(pf['window'] == legacy['window']
                      and pf['logic'] == legacy['logic']
                      and pf['thresh'] == legacy['thresh']
                      for pf in _perf_filters)
        if not already:
            _perf_filters.append(legacy)

    if _perf_filters:
        combined_perf = np.ones(n, dtype=bool)
        for pf in _perf_filters:
            col = f"rank_ret_{pf['window']}d"
            col_vals = df[col].values
            thresh_max = pf.get('thresh_max', 100.0)
            if pf['logic'] == '<':
                cond = col_vals < pf['thresh']
            elif pf['logic'] == 'Between':
                cond = (col_vals >= pf['thresh']) & (col_vals <= thresh_max)
            elif pf['logic'] == 'Not Between':
                # NaN comparisons return False on both branches -> NaN excluded
                cond = (col_vals < pf['thresh']) | (col_vals > thresh_max)
            else:
                cond = col_vals > pf['thresh']
            consec = pf.get('consecutive', 1)
            if consec > 1:
                cond = pd.Series(cond).rolling(consec).sum().values == consec
            combined_perf = combined_perf & np.nan_to_num(
                np.asarray(cond, dtype=float), nan=0.0).astype(bool)
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            s = pd.Series(combined_perf, index=df.index)
            prev_inst = s.shift(1).rolling(lookback).sum()
            combined_perf = combined_perf & (prev_inst.values == 0)
        conditions.append(combined_perf)

    # MA consecutive filters
    for f in params.get('ma_consec_filters', []) or []:
        col = f"SMA{f['length']}"
        if col in df.columns:
            if f['logic'] == 'Above':
                cond = df['Close'].values > df[col].values
            else:
                cond = df['Close'].values < df[col].values
            consec = f.get('consec', 1)
            if consec > 1:
                cond = pd.Series(cond).rolling(consec).sum().values == consec
            conditions.append(cond)

    # Seasonal filter
    if params.get('use_sznl', False):
        sznl_vals = df['Sznl'].values
        if params['sznl_logic'] == '<':
            cond_s = sznl_vals < params['sznl_thresh']
        else:
            cond_s = sznl_vals > params['sznl_thresh']
        if params.get('sznl_first_instance', False):
            lookback = params.get('sznl_lookback', 21)
            s = pd.Series(cond_s, index=df.index)
            prev = s.shift(1).rolling(lookback).max().fillna(0)
            cond_s = cond_s & (prev.values == 0)
        conditions.append(cond_s)

    # Market seasonal filter
    if params.get('use_market_sznl', False):
        mkt_sznl = df['Mkt_Sznl_Ref'].values
        if params['market_sznl_logic'] == '<':
            conditions.append(mkt_sznl < params['market_sznl_thresh'])
        else:
            conditions.append(mkt_sznl > params['market_sznl_thresh'])

    # ATR seasonal rank filters — fail CLOSED on a missing rank column,
    # 'consecutive' honored (aligned 2026-07-16; tests/test_atr_sznl_parity.py)
    for asf in params.get('atr_sznl_filters', []) or []:
        col = f"atr_sznl_{asf['window']}d"
        if col not in df.columns:
            conditions.append(np.zeros(n, dtype=bool))
            continue
        vals = df[col].values
        logic = asf.get('logic', '>')
        if logic == '<':
            cond = vals < asf['thresh']
        elif logic == '>':
            cond = vals > asf['thresh']
        elif logic == 'Between':
            cond = (vals >= asf['thresh']) & (vals <= asf.get('thresh_max', 100.0))
        else:
            continue
        consec = int(asf.get('consecutive', 1) or 1)
        if consec > 1:
            cond = pd.Series(cond, index=df.index).rolling(consec).sum().eq(consec).values
        conditions.append(cond)

    # Risk dial filters — THE mode-dependent gate.
    dial_filters = params.get('dial_filters', [])
    if dial_filters:
        frag_df = get_fragility_df_cached()
        if mode == 'live':
            # Fail closed: missing/stale cache or missing column rejects.
            ok = np.zeros(n, dtype=bool)
            passed = False
            if frag_df is not None and not frag_df.empty:
                signal_date = df.index[-1]
                try:
                    signal_date = pd.Timestamp(signal_date).normalize().tz_localize(None)
                except (TypeError, AttributeError):
                    signal_date = pd.Timestamp(signal_date).normalize()
                _last_frag = pd.Timestamp(frag_df.index[-1]).normalize()
                try:
                    fresh = np.busday_count(_last_frag.date(), signal_date.date()) <= FRAG_STALE_TD
                except (ValueError, AttributeError):
                    fresh = False
                if fresh:
                    passed = True
                    for dfil in dial_filters:
                        dial_col = dfil.get('dial')
                        if dial_col not in frag_df.columns:
                            passed = False
                            break
                        win = max(1, int(dfil.get('window', 1)))
                        ds = frag_df[dial_col]
                        if win > 1:
                            ds = ds.rolling(win, min_periods=win).mean()
                        try:
                            val = float(ds.reindex([signal_date], method='ffill').iloc[0])
                        except (IndexError, KeyError):
                            passed = False
                            break
                        if pd.isna(val):
                            passed = False
                            break
                        thresh = float(dfil.get('thresh', 0))
                        logic = dfil.get('logic', '>')
                        if ((logic == '>' and not val > thresh)
                                or (logic == '<' and not val < thresh)
                                or (logic == '>=' and not val >= thresh)
                                or (logic == '<=' and not val <= thresh)):
                            passed = False
                            break
            if passed:
                ok = np.ones(n, dtype=bool)
            conditions.append(ok)
        else:
            # Backtest: NaN/missing dial passes through (pre-2016 history is
            # ungradeable PIT; don't penalize it).
            for dfil in dial_filters:
                dial_col = dfil.get('dial')
                if frag_df is None or dial_col not in frag_df.columns:
                    continue
                win = max(1, int(dfil.get('window', 1)))
                ds = frag_df[dial_col]
                if win > 1:
                    ds = ds.rolling(win, min_periods=win).mean()
                aligned = ds.reindex(df.index, method='ffill').values
                thresh = float(dfil.get('thresh', 0))
                logic = dfil.get('logic', '>')
                if logic == '>':
                    cond_d = aligned > thresh
                elif logic == '<':
                    cond_d = aligned < thresh
                elif logic == '>=':
                    cond_d = aligned >= thresh
                elif logic == '<=':
                    cond_d = aligned <= thresh
                else:
                    cond_d = np.ones(n, dtype=bool)
                cond_d = np.where(np.isnan(aligned), True, cond_d)
                conditions.append(cond_d)

    # 52-week high/low filter — first_instance defaults TRUE (scan default)
    if params.get('use_52w', False):
        if params['52w_type'] == 'New 52w High':
            cond_52 = df['is_52w_high'].values.copy()
        else:
            cond_52 = df['is_52w_low'].values.copy()
        if params.get('52w_lag', 0) > 0:
            cond_52 = np.roll(cond_52, params['52w_lag'])
            cond_52[:params['52w_lag']] = False
        if params.get('52w_first_instance', True):
            lookback = params.get('52w_lookback', 21)
            s = pd.Series(cond_52, index=df.index)
            prev = s.shift(1).rolling(lookback).sum()
            cond_52 = cond_52 & (prev.values == 0)
        conditions.append(cond_52)

    if params.get('exclude_52w_high', False):
        conditions.append(~df['is_52w_high'].values)

    # ATH filter
    if params.get('use_ath', False):
        if params.get('ath_type') == 'Today is ATH':
            conditions.append(df['is_ath'].values)
        else:
            conditions.append(~df['is_ath'].values)

    # Trailing 52w high / low / ATH filters
    if params.get('use_recent_52w', False):
        lb = params.get('recent_52w_lookback', 21)
        m = df['is_52w_high'].rolling(window=lb, min_periods=1).max().astype(bool).values
        conditions.append(~m if params.get('recent_52w_invert', False) else m)

    if params.get('use_recent_52w_low', False):
        lb = params.get('recent_52w_low_lookback', 21)
        m = df['is_52w_low'].rolling(window=lb, min_periods=1).max().astype(bool).values
        conditions.append(~m if params.get('recent_52w_low_invert', False) else m)

    if params.get('use_recent_ath', False):
        lb = params.get('ath_lookback_days', 21)
        m = df['is_ath'].rolling(window=lb, min_periods=1).max().astype(bool).values
        conditions.append(~m if params.get('recent_ath_invert', False) else m)

    # Volume filters
    if params.get('vol_gt_prev', False):
        conditions.append(df['Volume'].values > np.roll(df['Volume'].values, 1))

    if params.get('use_vol', False):
        conditions.append(df['vol_ratio'].values > params['vol_thresh'])

    if params.get('use_vol_rank', False):
        v = df['vol_ratio_10d_rank'].values
        if params['vol_rank_logic'] == ">":
            conditions.append(v > params['vol_rank_thresh'])
        elif params['vol_rank_logic'] == "<":
            conditions.append(v < params['vol_rank_thresh'])

    # MA/level distance filter — percent-space. Scan superset: SMA/EMA plus
    # 52-Week-High / All-Time-High targets and the use_dist_filter alias;
    # degenerate rows (ATR/MA/Close <= 0) evaluate at 0 (scan behavior).
    if params.get('use_ma_dist_filter', False) or params.get('use_dist_filter', False):
        ma_type = params.get('dist_ma_type', 'SMA 200')
        special = {"52-Week High": "High_52w", "All-Time High": "ATH_Level"}
        ma_col = special.get(ma_type, str(ma_type).replace(" ", ""))
        if ma_col in df.columns:
            close_v = df['Close'].values.astype(float)
            atr_v = df['ATR'].values.astype(float)
            ma_v = df[ma_col].values.astype(float)
            valid = (atr_v > 0) & (ma_v > 0) & (close_v > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                dist_val = ((close_v - ma_v) / ma_v) / (atr_v / close_v)
            dist_val = np.where(valid, dist_val, 0.0)
            d_logic = params.get('dist_logic', 'Between')
            if d_logic == "Greater Than (>)":
                conditions.append(dist_val > params.get('dist_min', 0))
            elif d_logic == "Less Than (<)":
                conditions.append(dist_val < params.get('dist_max', 0))
            elif d_logic == "Between":
                conditions.append((dist_val >= params.get('dist_min', 0))
                                  & (dist_val <= params.get('dist_max', 0)))

    # VIX filter — missing column evaluates at 0.0 (scan semantics: fails
    # toward reject when vix_min > 0 instead of silently passing)
    if params.get('use_vix_filter', False):
        vix_vals = _col_or(df, 'VIX_Value', 0.0, n)
        conditions.append((vix_vals >= params.get('vix_min', 0))
                          & (vix_vals <= params.get('vix_max', 100)))

    # Today's return in ATR terms — prefer the precomputed column (scan
    # basis); identical formula as fallback
    if params.get('use_today_return', False):
        if 'today_return_atr' in df.columns:
            dr = df['today_return_atr'].values
        else:
            prev_close = np.roll(df['Close'].values, 1)
            dr = (df['Close'].values - prev_close) / df['ATR'].values
        conditions.append((dr >= params.get('return_min', -100))
                          & (dr <= params.get('return_max', 100)))

    if params.get('use_atr_ret_filter', False):
        if 'today_return_atr' in df.columns:
            ar = df['today_return_atr'].values
        else:
            prev_close = np.roll(df['Close'].values, 1)
            ar = (df['Close'].values - prev_close) / df['ATR'].values
        conditions.append((ar >= params.get('atr_ret_min', -100))
                          & (ar <= params.get('atr_ret_max', 100)))

    # T+1 Open filter — ENGINE-ONLY semantics: needs NextOpen (tomorrow's
    # open), which the live scan cannot know; daily_scan's wrapper strips
    # these flags and stamps the specs for order_staging instead.
    if params.get('use_t1_open_filter', False):
        _t1_filters = params.get('t1_open_filters', []) or []
        if _t1_filters and 'NextOpen' in df.columns:
            _next_open = df['NextOpen'].values
            _atr_vals = df['ATR'].values
            for _f in _t1_filters:
                _ref_col = _f.get('reference')
                _atr_offset = float(_f.get('atr_offset', 0) or 0)
                _logic = _f.get('logic')
                if _ref_col not in df.columns or _logic not in ('>', '>=', '<', '<='):
                    continue
                _threshold = df[_ref_col].values + (_atr_offset * _atr_vals)
                with np.errstate(invalid='ignore'):
                    if _logic == '>':
                        _cond = _next_open > _threshold
                    elif _logic == '>=':
                        _cond = _next_open >= _threshold
                    elif _logic == '<':
                        _cond = _next_open < _threshold
                    else:
                        _cond = _next_open <= _threshold
                conditions.append(_cond)

    # Monday-gap kill — engine-only for the same NextOpen reason; live it is
    # enforced by order_staging via the MonGapKill_* stamps + Signal_Date.
    if params.get('use_t1_gap_kill', False) and 'NextOpen' in df.columns:
        _kill_wds = params.get('t1_gap_kill_signal_weekdays', []) or []
        _kill_atr = float(params.get('t1_gap_kill_atr', 0) or 0)
        if _kill_wds and _kill_atr > 0:
            _kill_dir = params.get('t1_gap_kill_dir', 'up')
            _no = df['NextOpen'].values
            _close = df['Close'].values
            _atr = df['ATR'].values
            if 'DayOfWeekVal' in df.columns:
                _dows = df['DayOfWeekVal'].values
            else:
                _dows = df.index.dayofweek.values
            _wd_gated = np.isin(_dows, _kill_wds)
            with np.errstate(invalid='ignore'):
                if _kill_dir == 'down':
                    _gap_kill = _no < (_close - (_kill_atr * _atr))
                else:
                    _gap_kill = _no > (_close + (_kill_atr * _atr))
            _gap_kill = np.nan_to_num(_gap_kill, nan=False).astype(bool)
            conditions.append(~(_wd_gated & _gap_kill))

    # MA touch filter (engine feature; no live config uses it yet)
    if params.get('use_ma_touch', False):
        ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50",
                      "SMA 100": "SMA100", "SMA 200": "SMA200",
                      "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21"}
        ma_target = ma_col_map.get(params.get('ma_touch_type'))
        if ma_target and ma_target in df.columns:
            ma_series = df[ma_target]
            ma_vals = ma_series.values
            direction = params.get('trade_direction', 'Long')
            slope_lookback = params.get('ma_slope_days', 20)
            untested_lookback = params.get('ma_untested_days', 50)
            if direction == 'Long':
                is_slope_ok = ma_vals > np.roll(ma_vals, 1)
            else:
                is_slope_ok = ma_vals < np.roll(ma_vals, 1)
            if slope_lookback > 1:
                is_slope_ok = pd.Series(is_slope_ok).rolling(slope_lookback).sum().values == slope_lookback
            if untested_lookback > 0:
                if direction == 'Long':
                    was_untested = (df['Low'] > ma_series).shift(1).rolling(untested_lookback).min().values == 1.0
                else:
                    was_untested = (df['High'] < ma_series).shift(1).rolling(untested_lookback).min().values == 1.0
                was_untested = np.nan_to_num(was_untested, nan=False).astype(bool)
            else:
                was_untested = np.ones(n, dtype=bool)
            if direction == 'Long':
                touched_today = df['Low'].values <= ma_vals
            else:
                touched_today = df['High'].values >= ma_vals
            conditions.append(is_slope_ok & was_untested & touched_today)

    # Reference ticker filter — missing column evaluates at neutral 50.0
    # (scan semantics)
    if params.get('use_ref_ticker_filter', False) and params.get('ref_filters'):
        for rf in params['ref_filters']:
            col = f"Ref_rank_ret_{rf['window']}d"
            col_vals = _col_or(df, col, 50.0, n)
            if rf['logic'] == '<':
                conditions.append(col_vals < rf['thresh'])
            elif rf['logic'] == '>':
                conditions.append(col_vals > rf['thresh'])

    # Cross-sectional rank filter — missing column evaluates at neutral 50.0
    if params.get('use_xsec_filter', False) and params.get('xsec_filters'):
        for xf in params['xsec_filters']:
            col = f"xsec_rank_ret_{xf['window']}d"
            col_vals = _col_or(df, col, 50.0, n)
            if xf['logic'] == '<':
                conditions.append(col_vals < xf['thresh'])
            elif xf['logic'] == '>':
                conditions.append(col_vals > xf['thresh'])
            elif xf['logic'] == 'Between':
                conditions.append((col_vals >= xf['thresh'])
                                  & (col_vals <= xf.get('thresh_max', 100.0)))

    # Transition / crossover filter (engine feature — the scan gains it the
    # day a config uses it)
    for tf in params.get('transition_filters', []) or []:
        st_type = tf.get('series_type', 'perf')
        win = tf['window']
        if st_type == 'xsec':
            col = f"xsec_rank_ret_{win}d"
        elif st_type == 'atr_sznl':
            col = f"atr_sznl_{win}d"
        else:
            col = f"rank_ret_{win}d"
        if col not in df.columns:
            continue
        col_vals = df[col].values.astype(float)

        def _logic_mask(vals, logic, thresh):
            if logic == '<':
                return vals < thresh
            if logic == '<=':
                return vals <= thresh
            if logic == '>':
                return vals > thresh
            if logic == '>=':
                return vals >= thresh
            return np.zeros_like(vals, dtype=bool)

        to_mask = _logic_mask(col_vals, tf.get('to_logic', '>'), tf.get('to_thresh', 0.0))
        from_mask = _logic_mask(col_vals, tf.get('from_logic', '<'), tf.get('from_thresh', 0.0))
        within = max(1, int(tf.get('within_days', 5)))
        from_series = pd.Series(from_mask.astype(float), index=df.index)
        from_rolling = from_series.rolling(within, min_periods=1).max().shift(1).fillna(0).values.astype(bool)
        conditions.append(to_mask & from_rolling)

    # OR filter groups — a group that resolves to NO usable condition fails
    # CLOSED, and missing columns evaluate at neutral 50.0 (scan semantics)
    for group in params.get('or_filter_groups', []) or []:
        group_masks = []
        for cond in group:
            ctype = cond.get('type', 'perf')
            window = cond['window']
            logic = cond['logic']
            thresh = cond['thresh']
            if ctype == 'perf':
                col = f"rank_ret_{window}d"
            elif ctype == 'xsec':
                col = f"xsec_rank_ret_{window}d"
            else:
                continue
            col_vals = _col_or(df, col, 50.0, n)
            if logic == '<':
                group_masks.append(col_vals < thresh)
            elif logic == '>':
                group_masks.append(col_vals > thresh)
        if group_masks:
            combined_or = np.zeros(n, dtype=bool)
            for m in group_masks:
                combined_or = combined_or | np.nan_to_num(m, nan=0).astype(bool)
            conditions.append(combined_or)
        else:
            conditions.append(np.zeros(n, dtype=bool))

    # Combine — NaN in any condition resolves False (reject)
    if conditions:
        combined = np.ones(n, dtype=bool)
        for cond in conditions:
            cond_arr = np.asarray(cond)
            cond_arr = np.where(np.isnan(cond_arr.astype(float)), False, cond_arr)
            combined = combined & cond_arr.astype(bool)
        return pd.Series(combined, index=df.index)
    return pd.Series(True, index=df.index)


# Keys the live scan must strip: they evaluate NextOpen, which doesn't exist
# yet at scan time — order_staging enforces them at the real T+1 open.
_LIVE_STRIP_KEYS = ('use_t1_open_filter', 'use_t1_gap_kill')


def live_signal_mask(df, params, sznl_map=None, ticker=None):
    """The full live-mode signal mask (T+1 gates stripped). check_signal_live
    is its last row; the signal-recency ladder counts its trailing window."""
    if any(params.get(k) for k in _LIVE_STRIP_KEYS):
        params = dict(params)
        for k in _LIVE_STRIP_KEYS:
            params[k] = False
    return evaluate_filter_mask(df, params, sznl_map=sznl_map,
                                ticker_name=(ticker or 'UNK'), mode='live')


def recency_prior_from_mask(mask, window_td):
    """Signal days in the trailing window_td sessions BEFORE the last bar
    (the last bar itself — today's signal — is excluded)."""
    return int(mask.iloc[-(int(window_td) + 1):-1].sum())


def check_signal_live(df, params, sznl_map=None, ticker=None):
    """daily_scan's signal check: the shared mask's last row, live mode,
    with the T+1 gates stripped (stamped for order_staging instead)."""
    return bool(live_signal_mask(df, params, sznl_map=sznl_map,
                                 ticker=ticker).iloc[-1])
