"""b0 - recon: vehicle depth, JH anchor alignment, dial vintage facts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

TK = ["TLT", "IEF", "LQD", "^TNX", "GLD", "GDX", "SLV", "DX-Y.NYB", "UUP",
      "SPY", "XLV", "XLU", "QQQ", "IWM"]
px = close_panel(TK)
print("panel span", px.index[0].date(), px.index[-1].date(), len(px))
for t in px.columns:
    s = px[t].dropna()
    print(f"  {t:10s} {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}")

# --- JH anchor alignment: anchor = JH_pos - 11 (mirrors today exactly) ---
jh = load_events(["jackson_hole"])["date"]
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
rows = []
for d in jh:
    # position of the JH day (or next trading day if holiday/weekend)
    p = pos.get(d)
    if p is None:
        later = idx[idx >= d]
        if len(later) == 0:
            rows.append((d.date(), None, None, None, "no bar"))
            continue
        p = pos[later[0]]
    a = p - 11
    rows.append((d.date(), idx[p].date(), idx[a].date(), idx[p].weekday(),
                 idx[a].weekday()))
print("\nJH -> anchor(-11td) alignment  (jh_csv, jh_bar, anchor, jh_wd, anch_wd)")
for r in rows:
    print("  ", r)

# today's mirror
print("\ntoday: last bar", idx[-1].date(), " JH 2026-08-28")
print("  today is 08-13; anchor position mirrors JH_pos-11")

# --- dial vintage facts ---
f = pd.read_parquet("data/rd2_fragility.parquet")
print("\nrd2_fragility:", f.shape, f.columns.tolist())
print("  index", f.index[0].date(), "..", f.index[-1].date())
pit = f.index >= pd.Timestamp("2026-07-02")
print(f"  rows pre-2026-07-02 (RECOMPUTE vintage): {(~pit).sum()} of {len(f)} "
      f"= {100*(~pit).mean():.1f}%")
ma = f["63d"].rolling(10).mean()
print("  ma10_63d tail:\n", ma.tail(3))
d21 = ma - ma.shift(21)
print("  live delta21 =", round(float(d21.iloc[-1]), 2),
      " ma now", round(float(ma.iloc[-1]), 2),
      " 21 ago", round(float(ma.shift(21).iloc[-1]), 2))
