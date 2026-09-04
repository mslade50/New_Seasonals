"""Is the post-August-expiration week cherry-picked out of twelve months?

Drill 02 found SPY/QQQ/IWM strong in the five sessions after AUGUST
expiration and flat after expiration generally. August is 1 of 12, so the
month grid is the control that decides whether that is a finding or a pick.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, fwd_ret, summarize, sign_test

px = load_prices(["SPY", "QQQ", "IWM", "^VIX"])
ASOF = pd.Timestamp("2026-08-21")
opex_all = pd.DatetimeIndex(load_events(["opex"])["date"])
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

for t in ["SPY", "QQQ", "^VIX"]:
    s = px[t]["Close"].astype(float).loc[:ASOF]
    f5 = fwd_ret(s, 5)
    base = 100 * f5.dropna().mean()
    rows = []
    for m in range(1, 13):
        o = pd.DatetimeIndex([d for d in opex_all
                              if d in s.index and d <= ASOF and d.month == m])
        v = f5.reindex(o).dropna()
        if not len(v):
            continue
        up = int((v.values > 0).sum())
        r = summarize(v.values, MONTHS[m - 1])
        r["record"] = f"{up}-{r['n'] - up}"
        r["sign_p"] = round(sign_test(up, r["n"]), 4)
        r["edge_pct"] = round(r["mean_pct"] - base, 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = ["label", "n", "mean_pct", "median_pct", "hit", "t", "record", "sign_p", "edge_pct"]
    print(f"\n########## {t}: 5 sessions after expiration, by month "
          f"(all-days control {base:+.3f}%) ##########")
    print(df[keep].round(3).to_string(index=False))
    aug = df[df.label == "Aug"].iloc[0]
    rank = int((df["mean_pct"] > aug["mean_pct"]).sum()) + 1
    print(f"   August ranks {rank} of {len(df)} by mean; "
          f"{int((df['hit'] >= aug['hit']).sum())} months at or above its hit rate")

# how often does a random month beat August's hit rate by chance?
print("\n########## SPY: hit-rate spread across months ##########")
s = px["SPY"]["Close"].astype(float).loc[:ASOF]
f5 = fwd_ret(s, 5)
hits = []
for m in range(1, 13):
    o = pd.DatetimeIndex([d for d in opex_all if d in s.index and d <= ASOF and d.month == m])
    v = f5.reindex(o).dropna()
    hits.append((MONTHS[m - 1], len(v), round(100 * float((v.values > 0).mean()), 1)))
for name, n, h in sorted(hits, key=lambda x: -x[2]):
    print(f"   {name} n={n} hit={h}%")
