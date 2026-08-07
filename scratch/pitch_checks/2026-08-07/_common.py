"""Shared helpers for the 2026-08-07 daily-pitch falsification checks.

Read-only. Nothing here writes to data/ or touches the live book.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRICES = ROOT / "data" / "master_prices.parquet"
EVENTS = ROOT / "data" / "macro_events.csv"


def load_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Per-ticker OHLCV frames indexed by date (adjusted basis, as cached)."""
    mp = pd.read_parquet(PRICES)
    mp = mp[mp["ticker"].isin(tickers)].copy()
    mp["date"] = pd.to_datetime(mp["date"])
    out = {}
    for t, g in mp.groupby("ticker"):
        g = g.drop(columns=["ticker"]).sort_values("date").set_index("date")
        out[t] = g[~g.index.duplicated(keep="last")]
    return out


def close_panel(tickers: list[str]) -> pd.DataFrame:
    """Aligned close panel (inner-join on the union index, forward-fill free)."""
    px = load_prices(tickers)
    return pd.DataFrame({t: px[t]["Close"] for t in px}).dropna(how="all")


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder-14 ATR (the pitch convention, not the scanner's simple mean)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def zscore(s: pd.Series, n: int = 10) -> pd.Series:
    """z10 as pitch_state defines it: n-day return over its trailing-year sd."""
    r = s.pct_change(n)
    return (r - r.rolling(252).mean()) / r.rolling(252).std()


def pct_rank(s: pd.Series, n: int, lookback: int = 252) -> pd.Series:
    """Trailing-`lookback` percentile rank of the n-day return (0-100)."""
    r = s.pct_change(n)
    return r.rolling(lookback).rank(pct=True) * 100.0


def fwd_ret(s: pd.Series, h: int) -> pd.Series:
    """Close-to-close forward return over h sessions, aligned to the anchor day."""
    return s.shift(-h) / s - 1.0


def declusters(idx: pd.DatetimeIndex, min_gap_td: int, all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Keep the first event of each cluster; `min_gap_td` in trading days."""
    pos = pd.Series(range(len(all_dates)), index=all_dates)
    keep, last = [], -10**9
    for d in sorted(idx):
        p = pos.get(d)
        if p is None:
            continue
        if p - last >= min_gap_td:
            keep.append(d)
            last = p
    return pd.DatetimeIndex(keep)


def summarize(vals: np.ndarray, label: str = "") -> dict:
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return {"label": label, "n": 0}
    sd = v.std(ddof=1) if n > 1 else np.nan
    t = v.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return {
        "label": label,
        "n": n,
        "mean_pct": 100 * v.mean(),
        "median_pct": 100 * float(np.median(v)),
        "hit": 100 * float((v > 0).mean()),
        "t": t,
        "worst_pct": 100 * v.min(),
        "best_pct": 100 * v.max(),
        "sd_pct": 100 * sd,
    }


def show(rows: list[dict], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    df = pd.DataFrame(rows)
    if df.empty:
        print("  (empty)")
        return
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    print(df.to_string(index=False))


def era_split(dates: pd.DatetimeIndex, vals: np.ndarray, cut: str = "2018-01-01") -> list[dict]:
    d = pd.DatetimeIndex(dates)
    m = d < pd.Timestamp(cut)
    return [summarize(np.asarray(vals)[m], f"pre-{cut[:4]}"),
            summarize(np.asarray(vals)[~m], f"{cut[:4]}+")]


def load_events(kinds: list[str] | None = None) -> pd.DataFrame:
    e = pd.read_csv(EVENTS)
    e["date"] = pd.to_datetime(e["date"])
    if kinds:
        e = e[e["event"].isin(kinds)]
    return e.sort_values("date").reset_index(drop=True)


def bootstrap_p_le0(vals: np.ndarray, n_boot: int = 5000, seed: int = 42) -> float:
    """P(mean <= 0) under a simple iid bootstrap. Use on declustered episodes only."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float((means <= 0).mean())
