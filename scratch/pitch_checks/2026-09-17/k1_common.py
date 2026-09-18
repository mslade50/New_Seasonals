"""K1 shared calendar helpers: NYSE month-ends from the SPY index."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

SPDR9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def nyse_index() -> pd.DatetimeIndex:
    return load_prices(["SPY"])["SPY"].index


def month_end_positions(idx: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the last NYSE session of each COMPLETED month."""
    idx = pd.DatetimeIndex(idx)
    per = idx.to_period("M")
    nxt = np.r_[per[1:] != per[:-1], False]  # last row: month not complete
    return np.flatnonzero(nxt)


def anchors(idx: pd.DatetimeIndex, offset_signal: int = -10):
    """DataFrame of month-ends with signal date (ME + offset_signal),
    month, quarter flag, midterm flag."""
    me = month_end_positions(idx)
    rows = []
    for m in me:
        s = m + offset_signal
        if s < 0:
            continue
        d = idx[m]
        rows.append({"me_pos": m, "sig_pos": s, "me_date": d,
                     "sig_date": idx[s], "month": d.month,
                     "qe": d.month in (3, 6, 9, 12),
                     "midterm": d.year % 4 == 2, "year": d.year})
    return pd.DataFrame(rows)


def stats_line(vals, dates, label):
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    v, d = v[ok], pd.DatetimeIndex(dates)[ok]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = sign_test(w, len(v))
    return r
