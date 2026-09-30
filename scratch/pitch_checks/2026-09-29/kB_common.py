"""Shared helpers for the kB_ checks of 2026-09-29 (C4 Brazil vote, C5 USDJPY book close, C8 crude into NFP)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

BAR = pd.Timestamp("2026-09-28")


def nyse_panel(tickers: list[str], ffill: tuple[str, ...] = ()) -> pd.DataFrame:
    """Close panel on SPY's (NYSE) calendar; ffill FX/futures that trade NYSE holidays."""
    raw = close_panel(sorted(set(tickers) | {"SPY"}))
    idx = raw["SPY"].dropna().index
    px = raw.reindex(idx).copy()
    for t in ffill:
        full = raw[t].dropna()
        px[t] = full.reindex(idx.union(full.index)).ffill(limit=5).reindex(idx)
    return px.loc[:BAR]


def month_end_positions(idx: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the last session of each COMPLETED month."""
    per = pd.DatetimeIndex(idx).to_period("M")
    nxt = np.r_[per[1:] != per[:-1], False]
    return np.flatnonzero(nxt)


def cell(vals, label: str) -> dict:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if len(v):
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v) - w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s


def welch(a, b) -> tuple[float, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return 100 * (a.mean() - b.mean()), (a.mean() - b.mean()) / se


def pos_before(idx: pd.DatetimeIndex, d) -> int:
    """Position of the last session strictly before d."""
    return int(idx.searchsorted(pd.Timestamp(d))) - 1


def pos_after(idx: pd.DatetimeIndex, d) -> int:
    """Position of the first session strictly after d."""
    return int(idx.searchsorted(pd.Timestamp(d), side="right"))


def span_ret(s: np.ndarray, e: int, x: int) -> float:
    if e < 0 or x >= len(s) or np.isnan(s[e]) or np.isnan(s[x]) or s[e] <= 0:
        return np.nan
    return s[x] / s[e] - 1.0
