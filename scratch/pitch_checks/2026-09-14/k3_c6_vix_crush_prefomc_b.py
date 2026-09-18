import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

# Parent test after the FOMC gate failed to filter: beta-hedged SHORT SVXY (+ long b x SPY)
# after ANY >=10% one-day VIX crush, 2018-03+. Is the residual worth its two-leg round trip?
raw = close_panel(["SPY", "^VIX", "SVXY"])
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)
px["^VIX"] = px["^VIX"].ffill(limit=2)
vchg = px["^VIX"] / px["^VIX"].shift(1) - 1
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
pos, _ = anchor_positions(cal, fomc, 0)
fps = np.sort(np.array(pos))
nxt = np.array([next((f - i for f in fps if f > i), 999) for i in range(len(cal))])
k = pd.Series(nxt, index=cal)
post18 = pd.Series(cal >= pd.Timestamp("2018-03-01"), index=cal)

for thr in [-0.08, -0.10, -0.12]:
    rows = []
    for h in [1, 2, 3]:
        rs = vehicle_ret(px, [("SVXY", 1.0)], h)
        rspy = vehicle_ret(px, [("SPY", 1.0)], h)
        ok = rs.notna() & rspy.notna() & post18
        b = np.polyfit(rspy[ok].values, rs[ok].values, 1)[0]
        hedged_short = -(rs - b * rspy)
        for lbl, m in [("crush any", vchg <= thr), ("crush & FOMC k=3", (vchg <= thr) & (k == 3)),
                       ("crush & FOMC 1..3", (vchg <= thr) & k.between(1, 3)),
                       ("crush NOT FOMC 1..3", (vchg <= thr) & ~k.between(1, 3))]:
            d = declusters(cal[(m & ok).values], max(h, 3), cal)
            v = hedged_short.loc[d].values
            x = summarize(v, f"thr {thr} h={h} {lbl}")
            wn = int((v > 0).sum())
            x["rec"] = f"{wn}-{len(v)-wn}"
            x["sign_p"] = round(sign_test(wn, len(v)), 4) if len(v) else np.nan
            x["raw_shortSVXY"] = round(-100 * rs.loc[d].mean(), 3)
            x["beta"] = round(b, 2)
            x["alldays_hedged"] = round(100 * hedged_short[ok].mean(), 3)
            x["x_cost_12bps"] = round(100 * 100 * np.mean(v) / 12.0, 1) if len(v) else np.nan
            rows.append(x)
    show(rows, f"beta-hedged SHORT SVXY after VIX crush <= {thr}, 2018-03+")
