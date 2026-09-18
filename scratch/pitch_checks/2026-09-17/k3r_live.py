"""K3R step 1 - is each candidate LIVE on the 2026-09-16 signal close?
c5: trailing-252 percentile of (GDX 5d - GLD 5d); at-the-low flag.
c6: VLO 252-high flag (close and intraday-high forms), USO 1d return.
c8: IEF and TLT 5d pct_rank (pitch_lab.pct_rank).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

BAR = pd.Timestamp("2026-09-16")

# ---------------- c5
px = close_panel(["GDX", "GLD"]).dropna()
px = px[px.index <= BAR]
g5 = px["GDX"].pct_change(5)
l5 = px["GLD"].pct_change(5)
s5 = g5 - l5
rank = s5.rolling(252).rank(pct=True) * 100
prior_min = s5.shift(1).rolling(251).min()
print("=== c5 GDX/GLD 5d spread ===")
print(f"last bar {px.index[-1].date()}  GDX 5d {100*g5.iloc[-1]:+.2f}%  GLD 5d {100*l5.iloc[-1]:+.2f}%  "
      f"spread {100*s5.iloc[-1]:+.2f}pp")
print(f"trailing-252 pctile rank of spread {rank.iloc[-1]:.2f} (threshold: at the 252 low = rank {100/252:.2f})")
print(f"prior-251 min spread {100*prior_min.iloc[-1]:+.2f}pp on "
      f"{s5.iloc[-252:-1].idxmin().date()}; live AT low? {bool(s5.iloc[-1] <= prior_min.iloc[-1])}")
print(f"required further underperformance to arm: {100*(prior_min.iloc[-1]-s5.iloc[-1]):+.2f}pp")
print(f"sessions in prior 251 with a LOWER spread: {int((s5.iloc[-252:-1] < s5.iloc[-1]).sum())}")
for d in s5.index[-6:]:
    print(f"  {d.date()} spread {100*s5[d]:+.2f}pp rank {rank[d]:.1f}")
print(f"pct_rank(GDX,5) {pct_rank(px['GDX'],5).iloc[-1]:.1f}  pct_rank(GLD,5) {pct_rank(px['GLD'],5).iloc[-1]:.1f}")

# ---------------- c6
P = load_prices(["VLO", "USO", "XLE"])
vlo = P["VLO"][P["VLO"].index <= BAR]
uso = P["USO"][P["USO"].index <= BAR]
xle = P["XLE"][P["XLE"].index <= BAR]
print("\n=== c6 VLO / USO ===")
print(f"last bars VLO {vlo.index[-1].date()} USO {uso.index[-1].date()} XLE {xle.index[-1].date()}")
vc = vlo["Close"] if "Close" in vlo else vlo["close"]
cols = {c.lower(): c for c in vlo.columns}
vh = vlo[cols["high"]]
vc = vlo[cols["close"]]
uc = uso[{c.lower(): c for c in uso.columns}["close"]]
xc = xle[{c.lower(): c for c in xle.columns}["close"]]
print(f"VLO close {vc.iloc[-1]:.2f}  prior-251 max close {vc.iloc[-252:-1].max():.2f}  "
      f"close-252-high flag {bool(vc.iloc[-1] >= vc.iloc[-252:-1].max())}")
print(f"VLO high {vh.iloc[-1]:.2f}  prior-251 max high {vh.iloc[-252:-1].max():.2f}  "
      f"intraday-252-high flag {bool(vh.iloc[-1] >= vh.iloc[-252:-1].max())}")
print(f"VLO 1d {100*vc.pct_change().iloc[-1]:+.2f}%   USO 1d {100*uc.pct_change().iloc[-1]:+.2f}% "
      f"(threshold <= -3.00%)   XLE 1d {100*xc.pct_change().iloc[-1]:+.2f}%")
print(f"c6 live (close-high AND USO<=-3%): "
      f"{bool(vc.iloc[-1] >= vc.iloc[-252:-1].max() and uc.pct_change().iloc[-1] <= -0.03)}")

# ---------------- c8
rp = close_panel(["IEF", "TLT"]).dropna()
rp = rp[rp.index <= BAR]
ri = pct_rank(rp["IEF"], 5)
rt = pct_rank(rp["TLT"], 5)
print("\n=== c8 IEF / TLT ===")
print(f"last bar {rp.index[-1].date()}  IEF 5d {100*rp['IEF'].pct_change(5).iloc[-1]:+.2f}% rank {ri.iloc[-1]:.2f} (<=5)  "
      f"TLT 5d {100*rp['TLT'].pct_change(5).iloc[-1]:+.2f}% rank {rt.iloc[-1]:.2f} (>=20)")
print(f"c8 live: {bool(ri.iloc[-1] <= 5 and rt.iloc[-1] >= 20)}")
print(f"IEF close {rp['IEF'].iloc[-1]:.3f} prior-251 min {rp['IEF'].iloc[-252:-1].min():.3f}")
