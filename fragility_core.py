"""fragility_core — the dial scoring core, streamlit-free.

Extracted verbatim from pages/risk_dashboard_v2.py on 2026-07-16
(RISK_DIALS_2026-07-16.md A3). The page re-imports everything here, so both
`from pages.risk_dashboard_v2 import compute_horizon_fragility` and
`from fragility_core import compute_horizon_fragility` resolve to the same
objects. The three consumers of the duplicated scoring pipeline (the page,
daily_risk_report, weekly_market_rundown) all call compute_fragility_bundle.

Behavior contract: functions are MOVED, not modified — the golden values in
tests/test_fragility_core.py lock the scoring math.
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
HORIZON_STATS_PATH = os.path.join(DATA_DIR, "signal_horizon_stats.json")


def load_horizon_stats() -> dict | None:
    """Load backtested signal horizon stats from JSON."""
    if not os.path.exists(HORIZON_STATS_PATH):
        return None
    with open(HORIZON_STATS_PATH, 'r') as f:
        return json.load(f)


def _signal_edge(stats: dict, signal_key: str, horizon: str) -> float:
    """Return the downside edge (positive = worse) for a signal at a horizon."""
    sig = stats.get('signals', {}).get(signal_key, {})
    dm = sig.get('horizons', {}).get(horizon, {}).get('diff_mean', 0)
    if dm is None:
        return 0.0
    return max(0.0, -dm)


HORIZON_DAYS = {'5d': 5, '21d': 21, '63d': 63}

# Drawdown level at which decay weight reaches 0 (per horizon)
HORIZON_DECAY_DD = {'5d': 0.05, '21d': 0.10, '63d': 0.20}

# Calm duration multiplier: extended periods without corrections build stored energy
# Thresholds from empirical percentiles of streak lengths (trading days)
CALM_STREAK_THRESHOLDS = {
    'corr_5pct':  {'p85': 94,  'p90': 125, 'p95': 152},
    'corr_10pct': {'p85': 203, 'p90': 292, 'p95': 468},
}
CALM_STREAK_MULTIPLIERS = {'p85': 1.05, 'p90': 1.10, 'p95': 1.20}


def _calm_mult_for_streak(streak_len, thresholds):
    """Return calm duration multiplier for a single streak length."""
    if streak_len >= thresholds['p95']:
        return 1.20
    if streak_len >= thresholds['p90']:
        return 1.10
    if streak_len >= thresholds['p85']:
        return 1.05
    return 1.0


def _compute_calm_multiplier_scalar(spy_close: pd.Series) -> float:
    """
    Compute calm duration multiplier (scalar) for current point in time.
    Product of 5% and 10% correction streak multipliers (max 1.44x).
    """
    if spy_close is None or len(spy_close) < 252:
        return 1.0
    high_52w = spy_close.rolling(252).max()
    drawdown = spy_close / high_52w - 1

    # Current streak: days since last 5% correction from 52w high
    corr_5 = drawdown <= -0.05
    if corr_5.any():
        streak_5 = len(spy_close) - 1 - spy_close.index.get_loc(corr_5[corr_5].index[-1])
    else:
        streak_5 = len(spy_close)

    # Current streak: days since last 10% correction from 52w high
    corr_10 = drawdown <= -0.10
    if corr_10.any():
        streak_10 = len(spy_close) - 1 - spy_close.index.get_loc(corr_10[corr_10].index[-1])
    else:
        streak_10 = len(spy_close)

    m5 = _calm_mult_for_streak(streak_5, CALM_STREAK_THRESHOLDS['corr_5pct'])
    m10 = _calm_mult_for_streak(streak_10, CALM_STREAK_THRESHOLDS['corr_10pct'])
    return m5 * m10


def _compute_calm_multiplier_series(spy_close: pd.Series) -> pd.Series:
    """
    Compute calm duration multiplier as a time series (vectorized).
    Product of 5% and 10% correction streak multipliers.
    """
    if spy_close is None or len(spy_close) < 252:
        return pd.Series(1.0, index=spy_close.index if spy_close is not None else [])
    high_52w = spy_close.rolling(252).max()
    drawdown = spy_close / high_52w - 1

    # Streak counters for 5% corrections
    corr_5 = (drawdown <= -0.05).astype(int)
    # Group by cumulative correction events; count days since last correction
    grp_5 = corr_5.cumsum()
    streak_5 = grp_5.groupby(grp_5).cumcount()
    # Before first correction ever, streak = cumulative index position
    never_5 = grp_5 == 0
    streak_5[never_5] = range(never_5.sum())

    # Streak counters for 10% corrections
    corr_10 = (drawdown <= -0.10).astype(int)
    grp_10 = corr_10.cumsum()
    streak_10 = grp_10.groupby(grp_10).cumcount()
    never_10 = grp_10 == 0
    streak_10[never_10] = range(never_10.sum())

    t5 = CALM_STREAK_THRESHOLDS['corr_5pct']
    t10 = CALM_STREAK_THRESHOLDS['corr_10pct']

    m5 = np.where(streak_5 >= t5['p95'], 1.20,
         np.where(streak_5 >= t5['p90'], 1.10,
         np.where(streak_5 >= t5['p85'], 1.05, 1.0)))

    m10 = np.where(streak_10 >= t10['p95'], 1.20,
          np.where(streak_10 >= t10['p90'], 1.10,
          np.where(streak_10 >= t10['p85'], 1.05, 1.0)))

    return pd.Series(m5 * m10, index=spy_close.index)


def _days_since_last_fire(signal_history: pd.Series) -> int | None:
    """
    Return trading days since the signal last fired (was True).
    Returns 0 if signal is currently ON, None if it never fired.
    """
    if signal_history is None or signal_history.empty:
        return None
    try:
        fire_mask = signal_history.astype(bool)
    except (ValueError, TypeError):
        return None
    if not fire_mask.any():
        return None
    last_fire_idx = fire_mask[fire_mask].index[-1]
    # Count trading days from last fire to end of series
    return len(signal_history.loc[last_fire_idx:]) - 1


def _signal_decay_weight(sig: dict, horizon: str, spy_pct_from_high: float) -> float:
    """
    Compute effective weight (0-1) for a signal on a given horizon dial.

    - If currently ON: 1.0 (full weight)
    - If OFF: linear decay based on remaining fraction of the horizon window,
      modulated by SPY proximity to highs.

    Drawdown sensitivity varies by horizon:
      5d:  decay → 0 at  5% drawdown (short-term thesis already invalidated)
      21d: decay → 0 at 10% drawdown
      63d: decay → 0 at 20% drawdown (structural concerns persist longer)

    spy_pct_from_high: positive value, e.g. 0.03 means SPY is 3% below 52w high.
    """
    if sig.get('on'):
        return 1.0

    days_since = _days_since_last_fire(sig.get('signal_history'))
    if days_since is None or days_since == 0:
        return 0.0

    h_days = HORIZON_DAYS.get(horizon, 21)
    remaining_frac = max(0.0, (h_days - days_since) / h_days)
    if remaining_frac == 0.0:
        return 0.0

    dd_zero = HORIZON_DECAY_DD.get(horizon, 0.10)
    spy_factor = max(0.0, 1.0 - (spy_pct_from_high / dd_zero))

    return remaining_frac * spy_factor


def _compute_decay_metadata(sig: dict, spy_pct_from_high: float) -> dict | None:
    """
    Return decay metadata for a signal that is OFF but still contributing weight.

    Returns None if signal is ON, never fired, or fully expired on all horizons.
    Otherwise returns {days_since, horizons: {5d/21d/63d: {weight, remaining_days}}, max_remaining_days}.
    """
    if sig.get('on'):
        return None

    days_since = _days_since_last_fire(sig.get('signal_history'))
    if days_since is None or days_since == 0:
        return None

    horizons = {}
    for h_label, h_days in HORIZON_DAYS.items():
        w = _signal_decay_weight(sig, h_label, spy_pct_from_high)
        remaining = max(0, h_days - days_since)
        horizons[h_label] = {'weight': w, 'remaining_days': remaining}

    # If no horizon still has weight, signal is fully expired
    if all(h['weight'] == 0.0 for h in horizons.values()):
        return None

    max_remaining = max(h['remaining_days'] for h in horizons.values() if h['weight'] > 0)
    return {
        'days_since': days_since,
        'horizons': horizons,
        'max_remaining_days': max_remaining,
    }


def compute_horizon_fragility(
    signals_ordered: dict,
    regime_mult: float,
    horizon_stats: dict,
    price_ctx: dict,
    spy_close: pd.Series = None,
) -> dict:
    """
    Compute 0-100 fragility scores for 3 horizons (5d, 21d, 63d).

    Each signal's contribution is weighted by its backtested edge
    (how much worse than baseline forward returns are when signal active).
    Only the seven BASE signal edges are consumed. The stats JSON also
    carries a "Distribution Dominance (Elevated)" entry — that field is
    research/display reference only and does NOT feed this composite.

    Signals that recently turned OFF decay linearly over their horizon window,
    modulated by SPY proximity to highs. Score further scaled by calm duration
    multiplier (extended periods without corrections amplify fragility).
    """
    horizons = ['5d', '21d', '63d']
    stats = horizon_stats

    da = signals_ordered.get('Distribution Dominance', {})
    vrc = signals_ordered.get('VIX Range Compression', {})
    dl = signals_ordered.get('Defensive Leadership', {})
    fomc = signals_ordered.get('Pre-FOMC Rally', {})
    ar = signals_ordered.get('Low Absorption Ratio', {})
    srd = signals_ordered.get('Seasonal Rank Divergence', {})
    disp = signals_ordered.get('Dispersion', {})

    # SPY distance from highs (positive = below high)
    dd = price_ctx.get('drawdown')
    spy_pct_from_high = abs(dd) if dd is not None and dd < 0 else 0.0

    scores = {}
    for h in horizons:
        active_weight = 0.0

        # D/A — use base edge (matches timeseries computation)
        da_w = _signal_decay_weight(da, h, spy_pct_from_high)
        if da_w > 0:
            active_weight += _signal_edge(stats, 'Distribution Dominance', h) * da_w

        vrc_w = _signal_decay_weight(vrc, h, spy_pct_from_high)
        if vrc_w > 0:
            active_weight += _signal_edge(stats, 'VIX Range Compression', h) * vrc_w

        dl_w = _signal_decay_weight(dl, h, spy_pct_from_high)
        if dl_w > 0:
            active_weight += _signal_edge(stats, 'Defensive Leadership', h) * dl_w

        fomc_w = _signal_decay_weight(fomc, h, spy_pct_from_high)
        if fomc_w > 0:
            active_weight += _signal_edge(stats, 'Pre-FOMC Rally', h) * fomc_w

        ar_w = _signal_decay_weight(ar, h, spy_pct_from_high)
        if ar_w > 0:
            active_weight += _signal_edge(stats, 'Low Absorption Ratio', h) * ar_w

        srd_w = _signal_decay_weight(srd, h, spy_pct_from_high)
        if srd_w > 0:
            active_weight += _signal_edge(stats, 'Seasonal Rank Divergence', h) * srd_w

        disp_w = _signal_decay_weight(disp, h, spy_pct_from_high)
        if disp_w > 0:
            active_weight += _signal_edge(stats, 'Dispersion', h) * disp_w

        # Dynamic max_weight: FOMC is calendar-dependent — only include its
        # edge in the denominator when it's contributing (ON or decaying).
        # Otherwise its large 5d edge (47% of total) prevents the dial from
        # reaching meaningful levels on the ~95% of days FOMC can't fire.
        max_weight = (
            _signal_edge(stats, 'Distribution Dominance', h)
            + _signal_edge(stats, 'VIX Range Compression', h)
            + _signal_edge(stats, 'Defensive Leadership', h)
            + _signal_edge(stats, 'Low Absorption Ratio', h)
            + _signal_edge(stats, 'Seasonal Rank Divergence', h)
            + _signal_edge(stats, 'Dispersion', h)
        )
        if fomc_w > 0:
            max_weight += _signal_edge(stats, 'Pre-FOMC Rally', h)

        if max_weight > 0:
            calm_mult = _compute_calm_multiplier_scalar(spy_close) if spy_close is not None else 1.0
            score = (active_weight / max_weight) * 80 * regime_mult * calm_mult
        else:
            score = 0.0

        scores[h] = max(0.0, score)

    return scores



def compute_fragility_timeseries(
    signals_ordered: dict,
    spy_close: pd.Series,
    horizon_stats: dict,
) -> pd.DataFrame:
    """
    Compute historical fragility scores for all 3 horizons (5d, 21d, 63d).

    Returns DataFrame with columns ['5d', '21d', '63d'], indexed by date.
    """
    # Build boolean fire DataFrame from signal histories
    fires = {}
    for name, sig in signals_ordered.items():
        h = sig.get('signal_history')
        if h is not None and not h.empty:
            fires[name] = h.astype(bool)
    if not fires:
        return pd.DataFrame(columns=['5d', '21d', '63d'], index=spy_close.index)
    fire_df = pd.DataFrame(fires).reindex(spy_close.index).fillna(False).astype(bool)

    # Price context vectors
    ret_12m = spy_close / spy_close.shift(252) - 1
    sma_200 = spy_close.rolling(200).mean()
    extension_200d = spy_close / sma_200 - 1
    high_52w = spy_close.rolling(252).max()
    drawdown = spy_close / high_52w - 1

    # Vectorized regime multiplier
    m = pd.Series(1.0, index=spy_close.index)
    m = m + np.where(ret_12m > 0.25, 0.25,
            np.where(ret_12m > 0.15, 0.10,
            np.where(ret_12m < -0.05, -0.15, 0.0)))
    m = m + np.where(extension_200d > 0.10, 0.25,
            np.where(extension_200d > 0.05, 0.10,
            np.where(extension_200d < -0.02, -0.15, 0.0)))
    m = m + np.where(drawdown > -0.02, 0.10,
            np.where(drawdown < -0.10, -0.20, 0.0))
    regime_mult = m.clip(0.6, 1.8)
    calm_mult = _compute_calm_multiplier_series(spy_close)

    spy_pct_from_high = (-drawdown).clip(lower=0.0)

    signal_names = list(signals_ordered.keys())
    result = {}

    for horizon, h_days in HORIZON_DAYS.items():
        edges = {name: _signal_edge(horizon_stats, name, horizon) for name in signal_names}

        active_weight = pd.Series(0.0, index=spy_close.index)
        fomc_weight_series = pd.Series(0.0, index=spy_close.index)

        for name in signal_names:
            if name not in fire_df.columns:
                continue
            edge = edges[name]
            if edge == 0.0:
                continue

            sig_on = fire_df[name]
            fire_int = sig_on.astype(int)
            group = fire_int.cumsum()
            days_since = group.groupby(group).cumcount()
            ever_fired = group > 0
            days_since = days_since.where(ever_fired, other=np.nan)

            remaining_frac = ((h_days - days_since) / h_days).clip(0.0, 1.0)
            dd_zero = HORIZON_DECAY_DD.get(horizon, 0.10)
            spy_factor = (1.0 - spy_pct_from_high / dd_zero).clip(0.0, 1.0)

            weight = np.where(
                sig_on, 1.0,
                np.where(ever_fired & (remaining_frac > 0), remaining_frac * spy_factor, 0.0),
            )
            active_weight += edge * weight

            if name == 'Pre-FOMC Rally':
                fomc_weight_series = pd.Series(weight, index=spy_close.index)

        # Dynamic max_weight: exclude FOMC edge on days it can't contribute
        base_max = sum(e for n, e in edges.items() if n != 'Pre-FOMC Rally')
        fomc_edge = edges.get('Pre-FOMC Rally', 0.0)
        max_weight = base_max + np.where(fomc_weight_series > 0, fomc_edge, 0.0)
        max_weight = np.maximum(max_weight, 1e-9)  # avoid division by zero

        result[horizon] = ((active_weight / max_weight) * 80 * regime_mult * calm_mult).clip(0.0)

    return pd.DataFrame(result, index=spy_close.index)




def compute_fragility_bundle(signals_ordered, regime_mult, price_ctx,
                             spy_close, ts_write_path=None):
    """The scoring pipeline shared by the page, daily_risk_report and
    weekly_market_rundown (previously three hand-kept copies).

    Returns dict: horizon_stats, h_scores (5d-smoothed latest), h_scores_10d,
    frag_df (raw timeseries). All None-safe when the stats JSON is missing.

    ts_write_path: optional explicit destination for the RAW timeseries
    (rd2_fragility_ts.parquet — research/ML only, NEVER a sizing input).
    Written with vintage metadata; the write is a caller decision now, not a
    side effect buried in a cached compute path.
    """
    horizon_stats = load_horizon_stats()
    h_scores = None
    h_scores_10d = None
    frag_df = None
    if horizon_stats is not None:
        h_scores = compute_horizon_fragility(
            signals_ordered, regime_mult, horizon_stats, price_ctx, spy_close)
        frag_df = compute_fragility_timeseries(
            signals_ordered, spy_close, horizon_stats)
        if frag_df is not None and len(frag_df) >= 1:
            # 5d moving average for dial display (smooths day-to-day noise)
            h_scores = frag_df.rolling(5, min_periods=1).mean().iloc[-1].to_dict()
            h_scores_10d = frag_df.rolling(10, min_periods=1).mean().iloc[-1].to_dict()
        if ts_write_path and frag_df is not None and not frag_df.empty:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(frag_df)
                md = dict(table.schema.metadata or {})
                md[b"fragility_basis"] = b"raw_recompute"
                md[b"fragility_generated"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S").encode()
                md[b"fragility_note"] = (b"full recompute vintage - research/ML "
                                         b"only, never a sizing input")
                pq.write_table(table.replace_schema_metadata(md), ts_write_path)
            except Exception:
                frag_df.to_parquet(ts_write_path)
    return {
        "horizon_stats": horizon_stats,
        "h_scores": h_scores,
        "h_scores_10d": h_scores_10d,
        "frag_df": frag_df,
    }
