"""C1/C15 join: the ONE construction that needs a single round trip --
buy at 15:00 on ME-0, sell MOO on ME+1.  Combines the (negative) last hour
with the (positive) overnight, so it must be measured, not inferred."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import *  # noqa
from pitch_lab import load_prices, summarize, show, sign_test
import intraday_data as idl

for t in ("SPY", "IWM", "QQQ"):
    b = idl.get_intraday(t).copy()
    b["d"] = b["ts"].dt.normalize()
    rec = {}
    for d, g in b.groupby("d", sort=True):
        if len(g) < 20:
            continue
        g = g.sort_values("ts")
        m = g["ts"].dt.strftime("%H:%M") == "15:00"
        if not m.any():
            continue
        rec[d] = (float(g["open"].iloc[0]), float(g.loc[m, "open"].iloc[0]),
                  float(g["close"].iloc[-1]))
    f = pd.DataFrame(rec, index=["o", "p15", "c"]).T.sort_index()
    f["r_1500_nextopen"] = f["o"].shift(-1) / f["p15"] - 1.0
    f["r_lh"] = f["c"] / f["p15"] - 1.0
    f["r_on"] = f["o"].shift(-1) / f["c"] - 1.0
    d = load_prices([t])[t]
    ym = pd.Series(d.index.year * 100 + d.index.month, index=d.index)
    me = pd.DatetimeIndex(ym.groupby(ym.values).apply(lambda s: s.index[-1]).values)
    me = pd.DatetimeIndex([x for x in me if x in f.index])
    s = f.loc[me, "r_1500_nextopen"].dropna()
    a = f["r_1500_nextopen"].dropna()
    ex = 1e4 * (s.mean() - a.mean())
    w = int((s > 0).sum()); bp = float((a > 0).mean())
    print(f"\n{t}: ME-0 15:00 -> ME+1 OPEN (ONE round trip, ~5 bps)")
    show([summarize(s.values, f"ME-0 (N={len(s)})"),
          summarize(a.values, f"all sessions (N={len(a)})")])
    print(f"  excess {ex:+.2f} bps -> {abs(ex)/5:.2f}x cost | record {w}-{len(s)-w} "
          f"vs base {100*bp:.1f}%, sign p {sign_test(w, len(s), p=bp):.4f}")
    print(f"  decomposition: last hour {1e4*f.loc[me,'r_lh'].mean():+.2f} bps + "
          f"overnight {1e4*f.loc[me,'r_on'].mean():+.2f} bps")
    for lo in (2013, 2018, 2020):
        v = s[s.index.year >= lo]; c = a[a.index.year >= lo]
        print(f"  {lo}+: excess {1e4*(v.mean()-c.mean()):+.2f} bps  N={len(v)}")
    aug = s[s.index.month == 8]
    print(f"  AUGUST: {1e4*aug.mean():+.2f} bps on N={len(aug)}; "
          f"AUG x midterm {1e4*aug[aug.index.year % 4 == 2].mean():+.2f} bps "
          f"on N={len(aug[aug.index.year % 4 == 2])}")
