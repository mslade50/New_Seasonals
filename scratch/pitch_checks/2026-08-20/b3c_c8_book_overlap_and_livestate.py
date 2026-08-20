"""C8 round 2, continued: the book-overlap check (the ledger column is
'Signal Date', not 'Signal_Date' — b3b's zero-row answer was a column-name
miss, not a finding) and the JOINT live state.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["USO", "XLE", "CL=F", "XOP"])
d = px.index
jh = pd.DatetimeIndex(sorted(set(load_events(["jackson_hole"])["date"])
                             & set(d)))
jh = jh[jh <= d[-1]]
pos = pd.Series(range(len(d)), index=d)
entry = d[np.clip(d.get_indexer(jh) - 6, 0, len(d) - 1)]

# ------------------------------------------------------------ book overlap
tr = pd.read_parquet("data/backtest_trades_full.parquet")
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
ENERGY = {"USO", "XLE", "XOP", "OIH", "XOM", "CVX", "COP", "OXY", "SLB",
          "EOG", "PSX", "VLO", "MPC", "HAL", "DVN", "FANG", "APA", "HES",
          "MRO", "PXD", "KMI", "WMB", "OKE", "BKR", "ERX", "ERY", "GUSH",
          "DRIP", "DIG", "DUG", "UCO", "SCO", "UNG", "NRGU", "TRGP", "CTRA",
          "EQT", "AR", "RRC", "SWN", "CHK", "NOV", "FTI", "WHD", "LBRT"}
win = set()
for a in entry:
    p = pos.get(a)
    if p is None:
        continue
    for k in range(0, 12):
        if p + k < len(d):
            win.add(d[p + k])
sub = tr[tr["Signal Date"].isin(win) & tr["Ticker"].isin(ENERGY)]
print("=" * 78)
print("BOOK OVERLAP: energy signals inside a JH-6 .. JH+4 window (h=10 hold)")
print("=" * 78)
print(f"  ledger rows {len(tr)};  energy signals in window: {len(sub)}")
if len(sub):
    print("\n  by strategy x direction:")
    print(sub.groupby(["Strategy", "Direction"])
          .agg(n=("Ticker", "size"), avgR=("R_Multiple", "mean"),
               totR=("R_Multiple", "sum")).round(3).to_string())
    print(f"\n  DIRECTION SPLIT: {sub['Direction'].value_counts().to_dict()}")
    print(f"  tickers: {sorted(sub['Ticker'].unique())}")
    print(f"\n  the book is net {'SHORT' if (sub['Direction']=='Short').sum() > (sub['Direction']=='Long').sum() else 'LONG'} "
          f"energy in this window by signal count.")
    print("\n  same for USO/XLE/XOP only:")
    s2 = sub[sub["Ticker"].isin(["USO", "XLE", "XOP"])]
    if len(s2):
        print(s2.groupby(["Strategy", "Direction"])
              .agg(n=("Ticker", "size"), avgR=("R_Multiple", "mean")).round(3)
              .to_string())

# ------------------------------------------------------- joint live state
print("\n\n" + "=" * 78)
print("JOINT LIVE STATE: USO 63d rank 4.4 (deep floor) AND XLE within 2% of "
      "its 52w high. How many of the 26 JH anchors look like today?")
print("=" * 78)
r63 = pct_rank(px["USO"], 63)
xhi = rolling_on_valid(px["XLE"], lambda x: x.rolling(252).max())
xoff = (px["XLE"] / xhi - 1.0) * 100
tbl = pd.DataFrame({"USO_63d_rank": r63.reindex(entry),
                    "XLE_off_high_pct": xoff.reindex(entry)}).dropna(how="all")
tbl["h10_USO"] = 100 * fwd_lag(px["USO"], 10, lag=0).reindex(entry)
tbl["h10_CL"] = 100 * fwd_lag(px["CL=F"], 10, lag=0).reindex(entry)
tbl.index = tbl.index.year
print(tbl.round(2).to_string())
joint = tbl[(tbl.USO_63d_rank <= 25) & (tbl.XLE_off_high_pct >= -2.0)]
print(f"\n  anchors matching BOTH live legs: {len(joint)}")
print(f"  live today: USO 63d rank {r63.dropna().iloc[-1]:.1f}, "
      f"XLE off-high {xoff.dropna().iloc[-1]:.2f}%")
loose = tbl[(tbl.USO_63d_rank <= 40) & (tbl.XLE_off_high_pct >= -5.0)]
print(f"  anchors matching a LOOSENED version (rank<=40, XLE within 5%): "
      f"{len(loose)}  years {list(loose.index)}")
if len(loose):
    print(f"    their h=10 USO mean {loose.h10_USO.mean():+.3f}%  "
          f"values {[round(v,1) for v in loose.h10_USO.dropna()]}")
