"""C4 round 2 -- make the apples-to-apples reproduction honest, then the
high-dial subset.

b2f used "HYG within 0.25% of its trailing-252 high" and did not land on the
2026-08-26 registry numbers (+0.615pp at >=2%, -0.042pp in the 1-2% band). The
registry entry says "HYG at a FRESH 52w high", i.e. the close IS the running
252d maximum, which is a strictly tighter object. Both definitions are run here
side by side so the IWM-vs-SPY comparison is on ONE definition, and the gate is
measured against the SAME baseline the registry used (the HYG-high parent).

Then: does the IWM version survive on the high-dial subset that describes today
(ma10 of the 63d dial = 87.6)?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

INDEXES = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM"]
px = close_panel(["HYG"] + INDEXES)
idx = px.index


def dist_from_high(t):
    s = px[t]
    return s / rolling_on_valid(s, lambda x: x.rolling(252).max()) - 1.0


hyg_d = dist_from_high("HYG")
dists = {t: dist_from_high(t) for t in INDEXES}
DEFS = {"FRESH 52w high (close == 252d max)": hyg_d >= -1e-9,
        "within 0.25% of 252d high": hyg_d >= -0.0025}
DEPTH = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1.0)]

print("=== 1. SPY vs IWM depth split, BOTH HYG definitions, gate measured "
      "against the HYG-high PARENT (the registry's baseline) ===")
for dname, hmask in DEFS.items():
    print(f"\n--- HYG definition: {dname}  ({int(hmask.sum())} days) ---")
    for tkr in ("SPY", "IWM"):
        live = -100 * float(dists[tkr].iloc[-1])
        for h in (3, 5):
            r = fwd_lag(px[tkr], h, 1)
            ok = r.notna()
            pe = declusters(idx[hmask.values & ok.values], max(h, 5), idx)
            pm = 100 * float(np.nanmean(r.loc[pe].values))
            rows = []
            for lo, hi in DEPTH:
                m = hmask & (dists[tkr] <= -lo) & (dists[tkr] > -hi)
                dd = idx[m.values & ok.values]
                if len(dd) < 2:
                    rows.append({"label": f"{100*lo:.0f}-{100*hi:.0f}%", "n": 0})
                    continue
                e = declusters(dd, max(h, 5), idx)
                s = summarize(r.loc[e].values, f"{100*lo:.0f}-{100*hi:.0f}% off")
                s["gate_vs_parent_pp"] = round(s["mean_pct"] - pm, 3)
                s["n_days"] = len(dd)
                s["LIVE"] = "<<<" if (lo <= live / 100.0 < hi) else ""
                rows.append(s)
            show(rows, f"{tkr} h={h} | parent {pm:+.3f}% | LIVE depth {live:.2f}%")

print("\n=== 2. HIGH-DIAL SUBSET: does the IWM cell work at today's regime? ===")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean().dropna()
ma = ma.reindex(idx)
cell = (hyg_d >= -0.0025) & (dists["IWM"] <= -0.02)
for h in (3, 5, 10):
    r = fwd_lag(px["IWM"], h, 1)
    ok = r.notna()
    rows = []
    for lo, hi in [(0, 30), (30, 50), (50, 70), (70, 200)]:
        m = cell & (ma >= lo) & (ma < hi)
        dd = idx[m.fillna(False).values & ok.values]
        if len(dd) < 2:
            rows.append({"label": f"dial [{lo},{hi})", "n": 0, "n_days": len(dd)})
            continue
        e = declusters(dd, max(h, 5), idx)
        s = summarize(r.loc[e].values, f"dial [{lo},{hi})")
        s["n_days"] = len(dd)
        s["LIVE"] = "<<< 87.6" if lo <= 87.6 < hi else ""
        rows.append(s)
    # and the tightest live-matched slice
    m = cell & (ma >= 80)
    dd = idx[m.fillna(False).values & ok.values]
    if len(dd) >= 2:
        e = declusters(dd, max(h, 5), idx)
        s = summarize(r.loc[e].values, "dial >= 80 (live-matched)")
        s["n_days"] = len(dd)
        rows.append(s)
        print(f"  h={h} dial>=80 episode dates/returns: " +
              ", ".join(f"{str(x.date())} {100*y:+.2f}%"
                        for x, y in zip(e, r.loc[e].values)))
    show(rows, f"LONG IWM h={h}, cell split by the 10d-MA 63d fragility dial")

print("\n=== 3. the literal live cell (IWM 2-5% off) vs the pooled >=2% form ===")
for h in (3, 5, 10):
    r = fwd_lag(px["IWM"], h, 1)
    ok = r.notna()
    rows = []
    for lbl, m in [("IWM 2.0-5.0% off (LIVE band)",
                    (hyg_d >= -0.0025) & (dists["IWM"] <= -0.02) & (dists["IWM"] > -0.05)),
                   ("IWM >5% off (the band that pays)",
                    (hyg_d >= -0.0025) & (dists["IWM"] <= -0.05)),
                   ("IWM >=2% off (pitched, pools both)",
                    (hyg_d >= -0.0025) & (dists["IWM"] <= -0.02))]:
        dd = idx[m.values & ok.values]
        e = declusters(dd, max(h, 5), idx)
        v = r.loc[e].values
        s = summarize(v, lbl)
        s["n_days"] = len(dd)
        s["vs_alldays_pp"] = round(s["mean_pct"] - 100 * float(r[ok].mean()), 3)
        s["bps"] = round(100 * float(np.nanmean(v)) * 100, 1)
        s["x_cost"] = round(100 * float(np.nanmean(v)) * 100 / 2.0, 1)
        rows.append(s)
    show(rows, f"LONG IWM h={h} (2 bp round trip, 5x bar)")
