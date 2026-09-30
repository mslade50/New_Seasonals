"""C9 round 1: long XLE / short USO when USO has outrun XLE by >= 15pp over 21d.
Live: USO +16.47% vs XLE -0.03% (gap +16.5pp). h=5/10, lag=1.
Must confront the dead 63d version (b5_xle_uso_divergence, registry ~777-790):
overlap with the 63d state, bear-tape selection, leg attribution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["XLE", "USO", "SPY"]).dropna()
idx = px.index
r21 = px.pct_change(21)
r63 = px.pct_change(63)
gap21 = 100 * (r21["USO"] - r21["XLE"])
gap63 = 100 * (r63["USO"] - r63["XLE"])
gap21_rank = gap21.rolling(252).rank(pct=True) * 100
spy_bear = px["SPY"] < px["SPY"].rolling(200).mean()
print("LIVE:", {k: round(float(v.iloc[-1]), 2) for k, v in
                (("gap21", gap21), ("gap21_rank252", gap21_rank), ("gap63", gap63))},
      "SPY<200d:", bool(spy_bear.iloc[-1]), "last bar", idx[-1].date())
print("gap21 last 8:", gap21.tail(8).round(2).to_dict())

gate = gap21 >= 15
variants = {"gap21>=10": gap21 >= 10, "gap21>=12.5": gap21 >= 12.5,
            "gap21>=17.5": gap21 >= 17.5, "gap21>=20": gap21 >= 20,
            "gap21 rank252>=95": gap21_rank >= 95,
            "gap21>=15 & gap63<18 (NOT the 63d state)": gate & (gap63 < 18),
            "gap21>=15 & gap63>=18 (inside 63d state)": gate & (gap63 >= 18),
            "gap21>=15 & SPY>200d": gate & ~spy_bear,
            "gap63>=18 (dead parent)": gap63 >= 18}

for h in (5, 10):
    for lbl, legs in (("eq-dollar", [("XLE", 1.0), ("USO", -1.0)]),
                      ("beta-neutral 0.5", [("XLE", 1.0), ("USO", -0.5)])):
        battery(px, gate, legs, h, f"C9 long XLE / short USO {lbl}", cost_bps=4.0,
                variants=variants, event_kinds=("nfp",))
    # leg attribution
    rows = []
    for lbl, legs in (("long XLE alone", [("XLE", 1.0)]), ("short USO alone", [("USO", -1.0)]),
                      ("pair eq-dollar", [("XLE", 1.0), ("USO", -1.0)])):
        ret = vehicle_ret(px, legs, h)
        s = idx[gate.reindex(idx, fill_value=False).values & ret.notna().values]
        e = declusters(s, h, idx)
        v = ret.loc[e].values
        r = summarize(v, f"{lbl} (N={len(e)})")
        r["ctrl_all"] = round(100 * ret.mean(), 3)
        r["sign_p"] = sign_test(int((v > 0).sum()), len(v))
        rows.append(r)
    show(rows, f"h={h} leg attribution on gap21>=15 episodes")

# structural confrontation with the 63d kill
g = gate.reindex(idx, fill_value=False)
print("\nP(gap63>=18 | gap21>=15) =", round(float((gap63[g] >= 18).mean()), 3),
      "  P(gap21>=15 | gap63>=18) =", round(float(gate[gap63 >= 18].mean()), 3))
print("corr(gap21, gap63) =", round(float(gap21.corr(gap63)), 3))
print("SPY<200d share: trigger days", round(float(spy_bear[g].mean()), 3),
      " base rate", round(float(spy_bear[spy_bear.index >= idx[200]].mean()), 3))
ep = declusters(idx[g.values], 10, idx)
print("episodes (gap21>=15, gap 10):", len(ep))
for d in ep:
    print(f"  {d.date()} gap21 {gap21[d]:+.1f} gap63 {gap63[d]:+.1f} SPY<200d {bool(spy_bear[d])}")
