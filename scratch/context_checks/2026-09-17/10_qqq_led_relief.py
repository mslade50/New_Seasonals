"""Today's relief rally was large-cap only: QQQ +1.73%, IWM +0.53%.

A 120 bp one-session spread with BOTH legs positive, on the day after an FOMC
decision, with the VIX down 12.8%. Does a tech-led up day mean the laggard
catches up, or that it keeps lagging? This is the today-lane companion to the
post-witching IWM cell and it must be measured on its own, not assumed from it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, declusters, cluster_note,
)

px = close_panel(["QQQ", "IWM", "SPY"])
q, i, s = px["QQQ"].dropna(), px["IWM"].dropna(), px["SPY"].dropna()
common = q.index.intersection(i.index).intersection(s.index)
rq = (q / q.shift(1) - 1.0).reindex(common)
ri = (i / i.shift(1) - 1.0).reindex(common)
spread = rq - ri
print(f"today: QQQ {100 * rq.iloc[-1]:+.2f}%, IWM {100 * ri.iloc[-1]:+.2f}%, "
      f"spread {100 * spread.iloc[-1]:+.2f}pp")
print(f"spread percentile today: {100 * (spread < spread.iloc[-1]).mean():.1f}")

mask = (spread >= 0.01) & (rq > 0) & (ri > 0)
d = common[mask.fillna(False).values]
print(f"\nQQQ beats IWM by 100bp+ with both up: {len(d)} sessions of {len(common)}")
epi = declusters(d, 5, common)
print(f"  declustered (5td): {len(epi)}; by year "
      f"{dict(pd.Series(1, index=epi).groupby(epi.year).sum())}")

for h in (1, 5, 10):
    fq = (q.shift(-h) / q - 1.0).reindex(common)
    fi = (i.shift(-h) / i - 1.0).reindex(common)
    fsp = fi - fq                      # IWM minus QQQ going forward
    rows = []
    for lab, dd in [("all such sessions", d), ("declustered", epi)]:
        e = pd.DatetimeIndex(dd).intersection(fsp.dropna().index)
        r = summarize(fsp.loc[e].values, lab)
        if r["n"]:
            k = int((fsp.loc[e] > 0).sum())
            r["record"] = f"{k}-{r['n'] - k} IWM ahead"
            r["sign_p_iwm_lags"] = round(sign_test(r["n"] - k, r["n"]), 4)
        rows.append(r)
    rows.append(summarize(fsp.dropna().values, "CTL every session"))
    show(rows, f"IWM minus QQQ over the next {h} sessions")

print("\n=== and the outright legs, declustered ===")
for h in (1, 5):
    rows = []
    for tkr, ser in [("QQQ", q), ("IWM", i), ("SPY", s)]:
        f = (ser.shift(-h) / ser - 1.0).reindex(common)
        e = epi.intersection(f.dropna().index)
        r = summarize(f.loc[e].values, f"{tkr} h={h}")
        k = int((f.loc[e] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
        r["ctl_pct"] = round(100 * f.dropna().mean(), 3)
        rows.append(r)
    show(rows, f"outright, h={h}")

print("\n=== narrower: the same day also has the VIX down 8%+ ===")
vx = close_panel(["^VIX"])["^VIX"].dropna().reindex(common)
rv = vx / vx.shift(1) - 1.0
m2 = mask & (rv <= -0.08)
d2 = common[m2.fillna(False).values]
print(f"  n={len(d2)}: {[str(x.date()) for x in d2]}")
for h in (1, 5):
    fi = (i.shift(-h) / i - 1.0).reindex(common)
    fq = (q.shift(-h) / q - 1.0).reindex(common)
    e = pd.DatetimeIndex(d2).intersection((fi - fq).dropna().index)
    v = (fi - fq).loc[e]
    k = int((v > 0).sum())
    print(f"  IWM-QQQ h={h}: n={len(v)} mean {100 * v.mean():+.2f}pp "
          f"{k}-{len(v) - k} IWM ahead")
    vi = fi.loc[e]
    ki = int((vi > 0).sum())
    print(f"  IWM outright h={h}: mean {100 * vi.mean():+.2f}% {ki}-{len(vi) - ki} up")

print("\n=== era split on the declustered IWM-minus-QQQ h=5 cell ===")
f5 = ((i.shift(-5) / i - 1.0) - (q.shift(-5) / q - 1.0)).reindex(common)
e = epi.intersection(f5.dropna().index)
for lab, m in [("pre-2018", e < pd.Timestamp("2018-01-01")),
               ("2018+", e >= pd.Timestamp("2018-01-01"))]:
    v = f5.loc[e[m]]
    k = int((v > 0).sum())
    print(f"  {lab:<10} n={len(v):<3} mean {100 * v.mean():+.3f}pp  {k}-{len(v) - k} IWM ahead")
print(f"  concentration: {cluster_note(e, f5.loc[e].values, k=2)}")
