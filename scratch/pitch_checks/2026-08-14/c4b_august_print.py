"""C4 addendum — isolate the cell that is actually LIVE: the AUGUST NVDA print.

c4 grouped by ANCHOR month, which splits the August prints across months 7
and 8. This regroups by PRINT month, which is the cell today sits in, and
adds drop-best and the laggard cross inside it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import close_panel, pct_rank, show, sign_test, summarize  # noqa: E402

ASOF = pd.Timestamp("2026-08-13")
K, H = 9, 7
px = close_panel(["SMH", "QQQ", "NVDA"])
px = px[px.index <= ASOF]
idx = px.index

earn = pd.read_parquet("data/earnings_calendar.parquet", columns=["ticker", "date"])
earn["date"] = pd.to_datetime(earn["date"])
nv = earn[earn["ticker"] == "NVDA"]["date"].sort_values()
nv = nv[(nv >= idx[0]) & (nv <= idx[-1] + pd.Timedelta(days=20))]
pE = np.searchsorted(idx.values, nv.values, side="left")
ok = (pE > 0) & (pE < len(idx))
pE, nvd = pE[ok], pd.DatetimeIndex(nv.values[ok])

a = pE - K
e_, x_ = a + 1, a + 1 + H
m = (a >= 260) & (x_ < len(idx))
c = px["SMH"].values
r = c[x_[m]] / c[e_[m]] - 1.0
pm = nvd[m]                       # PRINT month, the right grouping
anc = idx[a[m]]
good = ~np.isnan(r)
r, pm, anc = r[good], pm[good], anc[good]

print("=" * 74)
print("SMH into the NVDA print, grouped by PRINT month (k=9, h=7, exit pE-1)")
print("=" * 74)
rows = []
for mo in sorted(set(pm.month)):
    sel = pm.month == mo
    if sel.sum() < 3:
        continue
    rows.append(summarize(r[sel], f"print month {mo:02d} (N={int(sel.sum())})"))
rows.append(summarize(r, f"ALL prints (N={len(r)})"))
show(rows, "by print month — today is an AUGUST print (2026-08-26)")

aug = pm.month == 8
va = r[aug]
w = int((va > 0).sum())
print(f"\nAUGUST print cell: N={len(va)}  mean {100*va.mean():+.3f}%  "
      f"median {100*np.median(va):+.3f}%  record {w}-{len(va)-w}  "
      f"sign p = {sign_test(w, len(va)):.4f}")
print(f"  drop-best  {100*np.sort(va)[:-1].mean():+.3f}%   "
      f"drop-worst {100*np.sort(va)[1:].mean():+.3f}%")
print("  per-year:", {int(y): round(100 * v, 2) for y, v in zip(pm[aug].year, va)})

base = (px["SMH"].shift(-(1 + H)) / px["SMH"].shift(-1) - 1.0).dropna()
print(f"\n  SMH unconditional h=7 all days: {100*base.mean():+.3f}%")
print(f"  SMH unconditional h=7 August days: "
      f"{100*base[base.index.month == 8].mean():+.3f}%")
print(f"  August-print cell EDGE vs all days = {100*(va.mean()-base.mean()):+.3f}pp")

sel20 = aug & (pm.year >= 2020)
v20 = r[sel20]
print(f"\n  2020+ August prints (the era NVDA drives the complex): N={len(v20)} "
      f"mean {100*v20.mean():+.3f}%  record "
      f"{int((v20>0).sum())}-{len(v20)-int((v20>0).sum())}")
print("   ", {int(y): round(100 * v, 2) for y, v in zip(pm[sel20].year, v20)})

r63 = pct_rank(px["SMH"], 63).reindex(idx)
lag = r63.values[a[m]][good]
sel = aug & (lag < 25)
print(f"\n  August print AND SMH rank63 < 25 (today = {r63.iloc[-1]:.1f}): "
      f"N={int(sel.sum())} mean {100*r[sel].mean():+.3f}%" if sel.sum() else
      "\n  August print AND rank63<25: N=0")
if sel.sum():
    print("   ", {int(y): round(100 * v, 2) for y, v in zip(pm[sel].year, r[sel])})
