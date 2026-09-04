"""Shared loaders + stats for the 2026-08-06 pitch checks (bake-off variant
opus_xhigh).

HARD RULE for this run: the cache already carries a 2026-08-06 bar for a
couple of tickers, but the pitch is written pre-market on 2026-08-06. Every
loader here truncates at ASOF_EXCL so today's realized bar can never leak
into a check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRICES = ROOT / "data" / "master_prices.parquet"
ASOF_EXCL = pd.Timestamp("2026-08-06")   # bars STRICTLY BEFORE this date
LAST_BAR = pd.Timestamp("2026-08-05")


def load(tickers: list[str]) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(PRICES)
    df = df[df["date"] < ASOF_EXCL]
    out = {}
    for t in tickers:
        sub = df[df["ticker"] == t].sort_values("date").set_index("date")
        if sub.empty:
            print(f"  !! {t}: no rows in master_prices")
            continue
        out[t] = sub[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return out


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    atr = np.full(len(c), np.nan)
    if len(c) > n:
        atr[n] = np.nanmean(tr[1:n + 1])
        for i in range(n + 1, len(c)):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return pd.Series(atr, index=df.index)


def ret(close: pd.Series, k: int) -> pd.Series:
    return close.pct_change(k) * 100.0


def pct_rank(s: pd.Series, window: int = 252) -> pd.Series:
    """Trailing percentile rank of the CURRENT value within the trailing
    window (inclusive), matching build_pitch_state's tape ranks."""
    return s.rolling(window, min_periods=60).rank(pct=True) * 100.0


def z10(close: pd.Series) -> pd.Series:
    r = close.pct_change()
    return (close.pct_change(10) / (r.rolling(63).std() * np.sqrt(10)))


def fwd(close: pd.Series, k: int) -> pd.Series:
    """Forward k-session close-to-close return in percent, signal-day aligned
    (entry at the signal close, exit k sessions later)."""
    return (close.shift(-k) / close - 1.0) * 100.0


def fwd_from_next_open(df: pd.DataFrame, k: int) -> pd.Series:
    """Enter MOO the session AFTER the signal, exit MOC k sessions after entry."""
    entry = df["Open"].shift(-1)
    exit_ = df["Close"].shift(-k)
    return (exit_ / entry - 1.0) * 100.0


def declusterize(dates: pd.DatetimeIndex, gap_td: int = 5) -> np.ndarray:
    """Boolean mask keeping the first signal of each cluster (episodes)."""
    keep, last = [], None
    for d in dates:
        if last is None or (d - last).days > gap_td * 1.6:
            keep.append(True)
            last = d
        else:
            keep.append(False)
    return np.array(keep)


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def describe(name: str, x, baseline=None) -> dict:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    d = {"cohort": name, "n": len(x),
         "avg": round(float(x.mean()), 3) if len(x) else float("nan"),
         "med": round(float(np.median(x)), 3) if len(x) else float("nan"),
         "hit": round(float((x > 0).mean() * 100), 1) if len(x) else float("nan"),
         "t": round(tstat(x), 2),
         "worst": round(float(x.min()), 2) if len(x) else float("nan"),
         "best": round(float(x.max()), 2) if len(x) else float("nan")}
    if baseline is not None:
        b = np.asarray(pd.Series(baseline).dropna(), dtype=float)
        d["base_avg"] = round(float(b.mean()), 3)
        d["lift"] = round(d["avg"] - d["base_avg"], 3)
    return d


def show(rows: list[dict]) -> None:
    print(pd.DataFrame(rows).to_string(index=False))


def era_split(dates, values, cut="2018-01-01") -> list[dict]:
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates)).dropna()
    return [describe(f"pre-{cut[:4]}", s[s.index < cut]),
            describe(f"{cut[:4]}+", s[s.index >= cut])]
