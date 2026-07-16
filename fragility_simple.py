"""Simple-dial shadow: equal-weight fragility composite (pre-registered).

Spec (RISK_DIALS_2026-07-16.md A6, registered before any parallel history
exists): score = mean over the 7 signals of a per-signal weight that is 1.0
while the signal is ON and decays LINEARLY to 0 over 63 trading days after it
turns off, scaled to 0-100. Deliberately absent vs the incumbent composite:
diff_mean edge weights, regime multiplier, calm-duration multipliers, the x80
scale factor, the drawdown-modulated decay, and the dynamic FOMC denominator
(FOMC sits in the fixed denominator like every other signal). ~35 fitted or
hand-set parameters removed.

Pre-registered threshold rule (no scanning): at evaluation time the shadow's
gate threshold is the percentile of its 10d-MA history that matches the
incumbent gate's ON rate (fraction of days the incumbent 63d 10d-MA >= 50)
over the common window. Agreement is computed on the LIVE basis: 5d-smoothed
series, then 10d MA — same convention as rd2_fragility.parquet.

The shadow is written to data/rd2_fragility_simple.parquet (its own file —
tests/test_fragility_append.py freezes the main parquet's schema) and CHANGES
NOTHING until a PIT re-run gates a swap decision, ~2027 at earliest.

This module is streamlit-free by design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECAY_TD = 63
SIMPLE_CACHE_NAME = "rd2_fragility_simple.parquet"


def _decayed_weight(history: pd.Series, index: pd.Index) -> pd.Series:
    """1.0 while ON, linear decay to 0 over DECAY_TD sessions after OFF."""
    h = history.reindex(index)
    h = h.astype(object).where(pd.notna(h), False).astype(bool)
    pos = pd.Series(np.arange(len(index), dtype=float), index=index)
    last_on = pos.where(h.to_numpy()).ffill()
    days_since = pos - last_on
    weight = (1.0 - days_since / DECAY_TD).clip(lower=0.0)
    weight[h.to_numpy()] = 1.0
    return weight.fillna(0.0)


def compute_simple_dial(signals_ordered: dict, index: pd.Index) -> pd.DataFrame:
    """Equal-weight decayed composite on the given date index.

    Returns a one-column DataFrame ('simple', 0-100). Signals with no usable
    history are EXCLUDED from the denominator (a missing series is a data
    problem, not evidence of calm) — the count used is stamped by the caller.
    """
    weights = []
    for sig in signals_ordered.values():
        history = (sig or {}).get("signal_history")
        if history is None or not hasattr(history, "empty") or history.empty:
            continue
        weights.append(_decayed_weight(history, index))
    if not weights:
        return pd.DataFrame(columns=["simple"])
    score = sum(weights) / len(weights) * 100.0
    out = pd.DataFrame({"simple": score.astype(float)}, index=index)
    out.attrs["n_signals"] = len(weights)
    return out
