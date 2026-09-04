"""Jackson Hole lands 2026-08-28, five sessions out.

The engine's event lane only anchors k=1..3 td before a scheduled event, so a
k=5 event is invisible to the sweep. Measured here from the session five
trading days before each prior Jackson Hole, lag=0, so h=5 IS the symposium
session's own close-to-close move and h=1..4 is the run into it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, fwd_ret, summarize, sign_test, cluster_note

px = load_prices(["SPY", "QQQ", "^VIX", "TLT", "GC=F"])
ASOF = pd.Timestamp("2026-08-21")
jh = pd.DatetimeIndex(load_events(["jackson_hole"])["date"])
jh = jh[jh <= ASOF]
print("prior symposium dates:", [str(d.date()) for d in jh])

for t in ["SPY", "^VIX", "TLT", "GC=F"]:
    s = px[t]["Close"].astype(float).loc[:ASOF]
    idx = s.index
    pos = pd.Series(range(len(idx)), index=idx)
    anchors = []
    for d in jh:
        if d not in pos.index:
            d2 = idx[idx.searchsorted(d) - 1] if idx.searchsorted(d) else None
            if d2 is None:
                continue
            d = d2
        p = pos[d] - 5
        if p >= 0:
            anchors.append(idx[p])
    a = pd.DatetimeIndex(anchors)
    print(f"\n########## {t}: from 5 sessions before the symposium ##########")
    rows = []
    for h in (1, 3, 5, 6, 10):
        f = fwd_ret(s, h)
        v = f.reindex(a).dropna()
        r = summarize(v.values, f"h={h}" + (" (symposium day)" if h == 5 else ""))
        if r["n"]:
            up = int((v.values > 0).sum())
            r["record"] = f"{up}-{r['n'] - up}"
            r["sign_p"] = round(sign_test(up, r["n"]), 4)
            r["ctl_all"] = round(100 * f.dropna().mean(), 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = [c for c in ["label", "n", "mean_pct", "median_pct", "hit", "t",
                        "record", "sign_p", "ctl_all"] if c in df]
    print(df[keep].round(3).to_string(index=False))

# the symposium session by itself
print("\n########## the symposium session's OWN move ##########")
for t in ["SPY", "QQQ", "^VIX", "TLT"]:
    s = px[t]["Close"].astype(float).loc[:ASOF]
    r1 = s.pct_change(fill_method=None)
    v = r1.reindex(jh).dropna()
    up = int((v.values > 0).sum())
    st = summarize(v.values, t)
    print(f"   {t:5s} n={st['n']:2d} mean={st['mean_pct']:+.3f}% med={st['median_pct']:+.3f}% "
          f"record {up}-{st['n']-up} p={sign_test(up, st['n']):.4f} "
          f"worst={st['worst_pct']:+.2f}% best={st['best_pct']:+.2f}%")
    if t == "SPY":
        print("      by year:", {int(d.year): round(100 * x, 2) for d, x in v.items()})
        print("      concentration:", cluster_note(v.index, v.values, k=2))
