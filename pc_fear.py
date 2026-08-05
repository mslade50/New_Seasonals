"""Equity put/call fear state for the fear-conditioned fragility bands.

Shipped 2026-08-05 per McKinley (ahead of the prereg gates — recorded in
scratch/ultracode_research/family_pc_fear_band_prereg_2026-08-05.md rev 3).
The dip-buy family carriers select their frag band TABLE by this state:

  fear ON  (lag-1 pctile > 85):  [[0, 50, 1.25], [50, 999, 1.0]]
  fear OFF:                      [[0, 50, 1.0],  [50, 999, 0.0]]
  STALE (> 3 bd) / no data:      execution['frag_risk_bands'] (incumbent
                                 0.25x table — exactly the pre-2026-08-05
                                 book; the unambiguous fail-closed state)

Definition (lag-1 by construction — measured 2026-08-05: the 21:30 UTC
nightly scrape captures only D-1, CBOE's daily page for D populates later):
the state for signal date D uses the newest cached row dated <= D - 1 bday.
Statistic: trailing-252d percentile rank (inclusive) of the 10d MA of the
CBOE equity put/call ratio from data/cboe_putcall.parquet.

Aligned sites — change together:
- strategy_config.PC_FEAR_BANDS (source of truth, carried by the 6 family
  band strategies via execution['pc_fear_bands'])
- daily_scan sizing 2b (frag_band_mult table selection + Sizing notes +
  email liveness footnote)
- pages/strat_backtester.py sizing 3b3 (frag_band_mult_at, point-in-time
  lag-1 replay; pc_fear_enabled=False for counterfactual passes)
- scripts/build_trade_ledger.py pcfear shadow pass (leg-C evidence accrual)
- Guard: tests/test_pc_fear_bands.py
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
PC_CACHE_PATH = os.path.join(_ROOT, "data", "cboe_putcall.parquet")

MA_DAYS = 10
RANK_WINDOW = 252
FEAR_PCT_THRESHOLD = 85.0
STALE_BD = 3  # newest usable row older than this many bdays vs signal date -> stale

_CACHE: dict = {}


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def pct_series(cache_path: str | None = None) -> pd.Series | None:
    """Trailing-252d percentile rank of the 10d-MA equity P/C, indexed by the
    DATA date (no lag applied here). None when the cache is missing/unusable."""
    key = cache_path or PC_CACHE_PATH
    if key in _CACHE:
        return _CACHE[key]
    series = None
    try:
        if os.path.exists(key):
            eq = pd.read_parquet(key)["equity"].dropna().sort_index()
            eq.index = pd.to_datetime(eq.index)
            try:
                eq.index = eq.index.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            ma = eq.rolling(MA_DAYS, min_periods=MA_DAYS).mean()
            series = _rolling_pct_rank(ma, RANK_WINDOW).dropna()
    except Exception:
        series = None
    _CACHE[key] = series
    return series


def fear_state_asof(signal_date, cache_path: str | None = None) -> dict:
    """Fear state for a signal date. Lag-1: only rows dated <= signal - 1 bday
    are eligible; stale when the newest eligible row is > STALE_BD bdays old.

    Returns {'state': 'on'|'off'|'stale', 'pct': float|None,
             'data_date': date|None, 'age_bd': int|None}.
    """
    out = {"state": "stale", "pct": None, "data_date": None, "age_bd": None}
    series = pct_series(cache_path)
    if series is None or series.empty:
        return out
    ts = pd.Timestamp(signal_date).normalize()
    cutoff = ts - pd.tseries.offsets.BDay(1)
    eligible = series.loc[:cutoff]
    if eligible.empty:
        return out
    data_ts = eligible.index[-1]
    age = int(np.busday_count(data_ts.date(), ts.date()))
    pct = float(eligible.iloc[-1])
    out.update(data_date=data_ts.date(), age_bd=age, pct=pct)
    if age > STALE_BD:
        return out
    out["state"] = "on" if pct > FEAR_PCT_THRESHOLD else "off"
    return out


def select_bands(execution: dict | None, state: str):
    """Band table for a strategy given the fear state. Strategies without
    pc_fear_bands keep their plain frag_risk_bands (or None). Stale/unknown
    states fail CLOSED to frag_risk_bands — the incumbent book."""
    if not execution:
        return None
    spec = execution.get("pc_fear_bands")
    if not spec:
        return execution.get("frag_risk_bands")
    if state == "on":
        return spec.get("on")
    if state == "off":
        return spec.get("off")
    return execution.get("frag_risk_bands")


def band_mult(bands, frag_score) -> float:
    """[lo, hi, mult] first-match lookup; 1.0 on no bands / no score. Same
    semantics as daily_scan.frag_band_mult's original loop."""
    if not bands or frag_score is None:
        return 1.0
    for lo, hi, mult in bands:
        if lo <= frag_score < hi:
            return float(mult)
    return 1.0


def sizing_note(state: dict, frag_score, mult: float) -> str:
    """Human-readable Sizing-notes fragment, e.g.
    'P/C 88%ile (fear ON) + dial 41 -> 1.25x' or
    'P/C stale (2026-07-28, 5 bd) -> incumbent bands'."""
    if state["state"] == "stale":
        detail = (f"{state['data_date']}, {state['age_bd']} bd"
                  if state["data_date"] else "no data")
        return f"P/C stale ({detail}) -> incumbent bands"
    dial = f"{frag_score:.0f}" if frag_score is not None else "n/a"
    return (f"P/C {state['pct']:.0f}%ile (fear {state['state'].upper()}) "
            f"+ dial {dial} -> {mult:.2f}x")
