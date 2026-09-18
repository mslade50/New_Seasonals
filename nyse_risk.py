"""NYSE warning layered onto the existing main dial, with a recovery reset.

The base series is already smoothed 5 then 10 sessions. Only the NYSE
contribution has resettable smoothing; the existing components stay intact.

Trigger series, 2026-09-18: a five-period EMA of `nyse_net` rather than the raw
one-day print. Both halves read the EMA, so arming needs a persistently negative
stretch near the high and a recovery needs the EMA back to zero or above. A
single non-negative day inside a negative stretch no longer clears the component
or its smoothing queues. The definition and the session alignment are the ones
measured in `scratch/nyse_smoothing_study/`; severity tiers are unchanged.
"""
from collections import deque

import numpy as np
import pandas as pd

MODEL_VERSION = "nyse-reset-floor-v2-ema5"
MAIN_COLUMN = "main_score"
SIGNAL_NAME = "NYSE Net Highs"
EMA_SPAN = 5
# First session the explicit main score was ever written (the v1 migration).
MIGRATION_START = pd.Timestamp("2026-09-17")
# First session scored on the EMA5 trigger. Rows saved before it were minted on
# the raw one-day print and keep that value forever; see
# docs/nyse_risk_dial_2026-09-17.md for the definitional-vintage note.
BASIS_V2_START = pd.Timestamp("2026-09-18")


def smooth_nyse_net(net_highs):
    """Five-period EMA of NYSE net new highs on the caller's session calendar.

    `ewm(span=5, adjust=False)`, the same construction the smoothing study
    measured. A missing breadth reading blanks the EMA for its whole trailing
    window, so an unknown session can neither arm the warning nor confirm a
    recovery. That five-session blackout is also the warm-up: the EMA is
    undefined until five consecutive readings exist.
    """
    net = pd.Series(net_highs, dtype=float)
    complete = (net.notna().astype(int)
                .rolling(EMA_SPAN, min_periods=EMA_SPAN).min().eq(1))
    return net.ewm(span=EMA_SPAN, adjust=False).mean().where(complete)


def main_dial_from_frame(frame):
    """Preserve the saved legacy vintage before the new main column begins."""
    frame = frame.sort_index()
    legacy = frame["63d"].dropna().rolling(10, min_periods=1).mean()
    if MAIN_COLUMN in frame:
        return frame[MAIN_COLUMN].combine_first(legacy).dropna()
    return legacy


def warning_severity(net_highs, distance):
    """Severity from the SMOOTHED breadth series; callers pass the raw print.

    2% is partial; 3% is inclusive. Missing observations remain unknown, and an
    incomplete EMA window counts as missing.
    """
    net = smooth_nyse_net(net_highs)
    valid = net.notna() & distance.notna()
    return pd.Series(np.where((net < 0) & (distance < .02), 1.,
                     np.where((net < 0) & (distance <= .03), .6, 0.)),
                     index=distance.index).where(valid)


def compute_nyse_main(base_main, spy_close, net_highs, stats):
    """Causal reconstruction, matching the reviewed reset-and-floor study.

    Use only exact-session breadth. Unknown readings cannot confirm recovery.
    Following a gap, use the base score until the possible 63-session fade and
    both smoothing windows have passed (same conservative study eligibility).

    Arming and recovery both read the five-period EMA (2026-09-18), so a lone
    non-negative print inside a negative stretch no longer wipes the state.
    """
    from fragility_core import ACTIVE_RISK_SIGNALS, _signal_edge, _compute_calm_multiplier_series
    spy = spy_close.sort_index().dropna()
    net = net_highs.reindex(spy.index)
    smoothed = smooth_nyse_net(net)
    distance = (1 - spy / spy.rolling(252).max()).clip(lower=0)
    severity = warning_severity(net, distance)
    ret = spy / spy.shift(252) - 1
    extension = spy / spy.rolling(200).mean() - 1
    regime = pd.Series(1., index=spy.index)
    regime += np.where(ret > .25, .25, np.where(ret > .15, .10, np.where(ret < -.05, -.15, 0)))
    regime += np.where(extension > .10, .25, np.where(extension > .05, .10, np.where(extension < -.02, -.15, 0)))
    regime += np.where(distance < .02, .10, np.where(distance > .10, -.20, 0))
    ceiling = 80 * regime.clip(.6, 1.8) * _compute_calm_multiplier_series(spy)
    weight = _signal_edge(stats, "Low Absorption Ratio", "63d")
    total = sum(_signal_edge(stats, name, "63d") for name in ACTIVE_RISK_SIGNALS)
    alpha = weight / (total + weight) if total + weight > 0 else 0.
    first, second = deque(maxlen=5), deque(maxlen=10)
    last_i, last_value = None, 0.
    contributions, effective, resets = [], [], []
    for i, (sev, sm, dd, cap) in enumerate(zip(severity, smoothed, distance, ceiling)):
        reset = pd.notna(sm) and sm >= 0
        if reset:
            last_i, last_value = None, 0.
            first, second = deque([0.] * 5, maxlen=5), deque([0.] * 10, maxlen=10)
        if not reset and pd.notna(sev) and sev > 0:
            last_i, last_value, eff = i, float(sev), float(sev)
        elif last_i is not None and pd.notna(dd):
            eff = last_value * max(0., 1 - (i - last_i) / 63) * max(0., 1 - dd / .20)
        else:
            eff = 0.
        first.append(alpha * cap * eff)
        second.append(sum(first) / len(first))
        contributions.append(sum(second) / len(second))
        effective.append(eff)
        resets.append(reset)
    eligible = severity.notna().astype(int).rolling(64, min_periods=64).min().eq(1)
    eligible = eligible.astype(int).rolling(5, min_periods=5).min().eq(1)
    eligible = eligible.astype(int).rolling(10, min_periods=10).min().eq(1)
    base = base_main.reindex(spy.index)
    contribution = pd.Series(contributions, index=spy.index)
    expanded = (1 - alpha) * base + contribution
    score = pd.Series(np.maximum(base, expanded), index=spy.index).where(eligible, base)
    return pd.DataFrame({MAIN_COLUMN: score, "base_main": base,
                         "nyse_contribution": contribution, "nyse_effective": effective,
                         "nyse_severity": severity, "nyse_reset": resets,
                         "nyse_available": eligible, "nyse_net": net,
                         "nyse_net_ema5": smoothed,
                         "distance_from_high": distance}, index=spy.index)


def append_main_scores(frame, existing, spy_close, net_highs, stats, run_date):
    """Add today's explicit main score without changing saved prior decisions.

    Newly bootstrapped history remains legacy; historical new-model studies
    are separate artifacts. No fabricated inverse-smoothed 63d values.
    """
    base = frame["63d"].dropna().rolling(10, min_periods=1).mean()
    calculated = compute_nyse_main(base, spy_close, net_highs, stats)
    out = frame.copy()
    if MAIN_COLUMN not in out:
        out[MAIN_COLUMN] = np.nan
    # A delayed producer may append multiple dates. The first migration starts
    # on the run date; subsequent runs resume after the last explicit main row.
    prior = existing.get(MAIN_COLUMN, pd.Series(dtype=float)).dropna() if existing is not None else pd.Series(dtype=float)
    start = max(MIGRATION_START, pd.Timestamp(run_date).normalize())
    writable = out.index >= start
    if not prior.empty:
        writable |= out.index > prior.index.max()
    idx = out.index[writable].intersection(calculated.index)
    if not prior.empty:
        # Definitional-vintage freeze for the v1 -> v2 (EMA5) trigger change.
        # The AM correction legitimately refreshes the previous session's row
        # from settled prices, and on the changeover that would have rescored a
        # saved raw-trigger row under the new basis. A row already saved before
        # BASIS_V2_START keeps its raw-trigger value; only v2 rows may refresh.
        idx = idx.difference(prior.index[prior.index < BASIS_V2_START])
    out.loc[idx, MAIN_COLUMN] = calculated.loc[idx, MAIN_COLUMN]
    return out, calculated


def load_nyse_signal(spy_close, path):
    """Display metadata. The weighted calculation remains a separate layer."""
    from pathlib import Path
    spy = spy_close.dropna().sort_index()
    if spy.empty:
        return {"on": False, "available": False, "detail": "SPY history unavailable"}
    breadth = pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()
    net = breadth.get("nyse_net", pd.Series(dtype=float)).reindex(spy.index)
    smoothed = smooth_nyse_net(net)
    distance = (1 - spy / spy.rolling(252).max()).clip(lower=0)
    severity = warning_severity(net, distance)
    valid = pd.notna(severity.iloc[-1])
    complete = bool(severity.notna().astype(int).rolling(77, min_periods=77).min().eq(1).iloc[-1])
    on = complete and severity.iloc[-1] > 0
    detail = (f"NYSE net highs 5d EMA {smoothed.iloc[-1]:+.0f} (raw {net.iloc[-1]:+.0f}); SPY {distance.iloc[-1]:.2%} below its 252-session closing high; signal {severity.iloc[-1]:.1f}x."
              if valid else "NYSE reading missing for the latest SPY session; existing dial retained.")
    if valid and not complete:
        detail += " Recent breadth history is incomplete; existing dial retained."
    fired = np.flatnonzero(severity.fillna(0).gt(0).to_numpy())
    recovered = np.flatnonzero(smoothed.ge(0).to_numpy())
    cleared = bool(len(recovered) and (not len(fired) or recovered[-1] >= fired[-1]))
    return {"on": bool(on), "available": complete, "detail": detail,
            "recovery_cleared": cleared,
            "summary": detail, "signal_history": severity.fillna(0).gt(0),
            "net_highs_ema5": smoothed, "raw_net": net, "severity": severity,
            "explanation": "More NYSE new lows than highs near an index high can reveal weakening participation. The trigger is a five-session EMA of net new highs, so one non-negative print no longer clears the warning; the EMA reaching zero or above clears it and its smoothing memory."}
