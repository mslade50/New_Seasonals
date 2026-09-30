"""kB shared helpers (2026-09-30 checker B): NYSE month-ends, stats line,
SPDR spread construction, ex-ante beta. Re-uses the 2026-09-17 k1 construction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

SPDR9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
OUT = Path(__file__).resolve().parent


def nyse_index() -> pd.DatetimeIndex:
    return load_prices(["SPY"])["SPY"].index


def month_end_positions(idx: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the last NYSE session of each COMPLETED month (the last
    row is excluded: 2026-09-30 is not in the cache)."""
    per = pd.DatetimeIndex(idx).to_period("M")
    nxt = np.r_[per[1:] != per[:-1], False]
    return np.flatnonzero(nxt)


def me_table(idx: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for m in month_end_positions(idx):
        d = idx[m]
        rows.append({"me_pos": m, "me_date": d, "month": d.month, "year": d.year,
                     "qe": d.month in (3, 6, 9, 12), "midterm": d.year % 4 == 2})
    return pd.DataFrame(rows)


def stats_line(vals, dates, label):
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    v = v[ok]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v) - w}"
        r["sign_p"] = sign_test(w, len(v))
    return r


def spread_at(P: np.ndarray, rank_row: np.ndarray, p0: int, p1: int, k: int = 2):
    """Bottom-k minus top-k (by rank_row) return from position p0 to p1.
    Positive = the REVERSAL pair (long losers, short winners) earns."""
    if p1 >= len(P) or p0 < 0:
        return np.nan, np.nan, np.nan
    f = P[p1] / P[p0] - 1
    ok = ~np.isnan(rank_row) & ~np.isnan(f)
    if ok.sum() < 2 * k + 1:
        return np.nan, np.nan, np.nan
    rr, ff = rank_row[ok], f[ok]
    o = np.argsort(rr)
    lo, hi = ff[o[:k]].mean(), ff[o[-k:]].mean()
    return lo - hi, lo, hi


def exante_beta(y: pd.Series, x: pd.Series, win: int = 252) -> pd.Series:
    """Trailing-`win` daily OLS beta of y on x, known at each close."""
    ry, rx = y.pct_change(fill_method=None), x.pct_change(fill_method=None)
    cov = ry.rolling(win, min_periods=200).cov(rx)
    var = rx.rolling(win, min_periods=200).var()
    return cov / var
