"""W28 re-run on the 09-28 bar: adapted from 2026-09-24/kC_s1_slv_w28_b.py (BAR moved, same definitions).

Confirms the 09-28 first-break status on the entry's own mask and reproduces the in-sample FIRST record.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

BAR = pd.Timestamp("2026-09-28")
px = close_panel(["GLD", "SLV", "GDX"]).dropna().loc[:BAR]
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in ["GLD", "SLV", "GDX"]}
LEGS = [("SLV", -1.0)]
faith = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
prior = faith.shift(1).rolling(5).max().fillna(0).astype(bool)
first = faith & ~prior

last = px.index[-1]
print(f"last bar {last.date()} | GLD {100 * r1['GLD'][last]:+.2f}% SLV {100 * r1['SLV'][last]:+.2f}% GDX {100 * r1['GDX'][last]:+.2f}%")
print(f"faithful={bool(faith[last])} prior-5 faithful={bool(prior[last])} FIRST={bool(first[last])}")
print("prior 5 sessions:", [(str(d.date()), bool(faith[d])) for d in px.index[-6:-1]])

r = vehicle_ret(px, LEGS, 1, 1)
d = px.index[first.values & r.notna().values]
v = r.loc[d].values
w = int((v > 0).sum())
base = r.dropna()
print(f"in-sample FIRST breaks, short SLV h=1 lag 1: mean {100 * v.mean():+.3f}% on {w}-{len(v) - w}, "
      f"sign p vs down-rate {sign_test(w, len(v), float((base > 0).mean())):.4f} (down-rate {100 * (base > 0).mean():.2f}%)")
oos = [x for x in px.index[first.values] if x >= pd.Timestamp("2026-09-24")]
print("OOS first breaks from 2026-09-24:", [str(x.date()) for x in oos],
      "-> scoring window: short SLV from the next close (09-29) to the close after (09-30)")
print("last 2026 first breaks with lag-1 h=1 outcome:",
      [(str(x.date()), f"{100 * r[x]:+.2f}%" if pd.notna(r[x]) else "open") for x in px.index[first.values] if x >= pd.Timestamp("2026-01-01")])
