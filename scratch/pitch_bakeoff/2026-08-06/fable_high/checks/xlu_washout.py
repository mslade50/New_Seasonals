"""Check: long XLU after a relative washout while SPY is strong.

Candidate: XLU 21d rank 6.0 / z10 -1.87 as of 2026-08-05 while SPY 21d
rank 73.8 and 0.2% off its 52w high. Cells:
  A. XLU 21d-return 252d-rank < 10 AND SPY 21d rank > 50 -> fwd 5/10d
     XLU outright and XLU-minus-SPY spread
  B. same + SPY within 2% of its 252d high (today's exact shape)
  C. control: XLU rank < 10 alone; XLU unconditional drift
Declustered (10 td). Era split. No bar after 2026-08-05 is used.
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


xlu, spy = load("XLU"), load("SPY")
df = pd.DataFrame({"xlu": xlu, "spy": spy}).dropna()

r21 = df.pct_change(21)
rank_xlu = r21["xlu"].rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)
rank_spy = r21["spy"].rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)
near_high = df["spy"] / df["spy"].rolling(252).max() - 1 >= -0.02

fwd = {h: df.shift(-h) / df - 1 for h in (5, 10)}


def decluster(mask: pd.Series, gap: int = 10) -> pd.Series:
    out, last = [], None
    for d, v in mask.items():
        if v and (last is None or (mask.index.get_loc(d) - last) > gap):
            out.append(d)
            last = mask.index.get_loc(d)
    return pd.DatetimeIndex(out)


def stats(x: pd.Series, label: str) -> None:
    x = x.dropna()
    if len(x) < 3:
        print(f"{label:56s} N={len(x)}")
        return
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{label:56s} {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  N {len(x):3d}"
          f"  hit {(x > 0).mean():.2f}  worst {x.min()*1e4:+.0f}")


mA = (rank_xlu < 10) & (rank_spy > 50)
mB = mA & near_high
mC = rank_xlu < 10

for name, m in [("A rel-washout", mA), ("B + SPY near high", mB), ("C xlu washout alone", mC)]:
    days = decluster(m.fillna(False))
    print(f"=== cell {name}: {len(days)} declustered episodes ===")
    for h in (5, 10):
        out = fwd[h]["xlu"].reindex(days)
        spread = (fwd[h]["xlu"] - fwd[h]["spy"]).reindex(days)
        stats(out, f"  XLU fwd {h}d")
        stats(spread, f"  XLU-SPY spread fwd {h}d")
    out10 = fwd[10]["xlu"].reindex(days)
    stats(out10[out10.index >= "2018-01-01"], "  XLU fwd 10d, 2018+")
    stats(out10[out10.index < "2018-01-01"], "  XLU fwd 10d, pre-2018")
    if name.startswith("B"):
        print("  episodes:", " ".join(d.strftime("%Y-%m-%d") for d in days))

stats(fwd[10]["xlu"].dropna(), "XLU unconditional fwd 10d")
print("current: rank_xlu=%.1f rank_spy=%.1f near_high=%s"
      % (rank_xlu.iloc[-1], rank_spy.iloc[-1], bool(near_high.iloc[-1])))
