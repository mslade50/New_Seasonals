"""Follow-ups: (1) era split + episode listing for the XLU-continuation
short (cell B of xlu_washout.py, sign-inverted candidate); (2) is the
Aug-NFP TLT week just August bond seasonality, and is that seasonal
era-stable? No bar after 2026-08-05 is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
CUTOFF = pd.Timestamp("2026-08-05")

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])


def load(tkr: str) -> pd.Series:
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[:CUTOFF, "Close"]


def stats(x: pd.Series, label: str) -> None:
    x = x.dropna()
    if len(x) < 3:
        print(f"{label:52s} N={len(x)}")
        return
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{label:52s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
          f"  hit {(x > 0).mean():.2f}  worst {x.min()*1e4:+.0f}")


# ---------- 1. XLU cell B era split ----------
xlu, spy = load("XLU"), load("SPY")
df = pd.DataFrame({"xlu": xlu, "spy": spy}).dropna()
r21 = df.pct_change(21)
rank_xlu = r21["xlu"].rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)
rank_spy = r21["spy"].rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)
near_high = df["spy"] / df["spy"].rolling(252).max() - 1 >= -0.02
mB = ((rank_xlu < 10) & (rank_spy > 50) & near_high).fillna(False)

out, last = [], None
for i, v in enumerate(mB.values):
    if v and (last is None or i - last > 10):
        out.append(mB.index[i])
        last = i
days = pd.DatetimeIndex(out)
fwd5 = df.shift(-5) / df - 1
sprd = (fwd5["xlu"] - fwd5["spy"]).reindex(days)
print("=== XLU-SPY 5d spread, cell B (washout + SPY near high) ===")
stats(sprd, "all")
stats(sprd[sprd.index >= "2018-01-01"], "2018+")
stats(sprd[sprd.index < "2018-01-01"], "pre-2018")
print("episode values (bps):")
for d, v in sprd.dropna().items():
    print(f"  {d:%Y-%m-%d} {v*1e4:+7.0f}")

# ---------- 2. TLT August seasonality ----------
tlt = load("TLT")
ret = tlt.pct_change().dropna()
aug = ret[ret.index.month == 8]
print("\n=== TLT daily returns, August vs all ===")
stats(aug, "all August days")
stats(ret, "all days (control)")
for lo, hi, lab in [("2002", "2013", "2002-2012"), ("2013", "2020", "2013-2019"),
                    ("2020", "2027", "2020+")]:
    stats(aug[(aug.index >= lo) & (aug.index < hi)], f"August {lab}")
print("August monthly total by year:")
yr = aug.groupby(aug.index.year).apply(lambda x: (1 + x).prod() - 1)
print("  " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in yr.items()))
mid = yr[yr.index % 4 == 2]
print(f"  midterm Augusts: {' '.join(f'{y}:{v*100:+.1f}%' for y, v in mid.items())}")
