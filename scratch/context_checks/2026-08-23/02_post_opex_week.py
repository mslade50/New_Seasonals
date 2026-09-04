"""The forward side of monthly expiration, which the engine never sweeps.

`E:opex` anchors k=1..3 BEFORE the expiration bar and goes dark once it
passes. Friday 2026-08-21 was August expiration, so Monday is the first
post-expiration session and the sweep has nothing to say about it.

Measured from the EXPIRATION CLOSE forward, lag=0, so h=1 is the Monday.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, fwd_ret, summarize, sign_test, cluster_note

TK = ["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]
px = load_prices(TK)
ASOF = pd.Timestamp("2026-08-21")
opex_all = pd.DatetimeIndex(load_events(["opex"])["date"])


def table(t, hs=(1, 2, 3, 5), sub=None, label=""):
    s = px[t]["Close"].astype(float).loc[:ASOF]
    idx = s.index
    ox = pd.DatetimeIndex([d for d in opex_all if d in idx and d <= ASOF])
    if sub is not None:
        ox = ox[sub(ox)]
    rows = []
    for h in hs:
        f = fwd_ret(s, h)
        v = f.reindex(ox).dropna()
        r = summarize(v.values, f"{label} h={h}")
        if r["n"]:
            up = int((v.values > 0).sum())
            r["record"] = f"{up}-{r['n'] - up}"
            r["sign_p"] = round(sign_test(up, r["n"]), 4)
            base = f.dropna()
            r["ctl_all_pct"] = round(100 * base.mean(), 3)
            r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = [c for c in ["label", "n", "mean_pct", "median_pct", "hit", "t",
                        "record", "sign_p", "ctl_all_pct", "edge_pct"] if c in df]
    print(df[keep].round(3).to_string(index=False))
    return ox


for t in ["SPY", "QQQ", "IWM", "^VIX"]:
    print(f"\n########## {t} from the expiration close ##########")
    table(t, label="every opex")
    table(t, sub=lambda o: o.month == 8, label="AUGUST opex")

print("\n########## SPY, August opex, per episode (h=5) ##########")
s = px["SPY"]["Close"].astype(float).loc[:ASOF]
f5 = fwd_ret(s, 5)
ox = pd.DatetimeIndex([d for d in opex_all if d in s.index and d <= ASOF and d.month == 8])
v = f5.reindex(ox).dropna()
for d, x in v.items():
    print(f"   {d.date()}  {100*x:+6.2f}%")
print("  concentration:", cluster_note(v.index, v.values, k=2))
cut = pd.Timestamp("2018-01-01")
for lab, w in [("pre-2018", v[v.index < cut]), ("2018+", v[v.index >= cut])]:
    up = int((w.values > 0).sum())
    st = summarize(w.values, lab)
    print(f"  {lab:9s} n={st['n']:2d} mean={st['mean_pct']:+.3f}% med={st['median_pct']:+.3f}% "
          f"record {up}-{st['n']-up} sign_p={sign_test(up, st['n']):.4f}")

print("\n########## midterm years only, August opex, h=5 ##########")
for t in ["SPY", "QQQ", "IWM"]:
    s = px[t]["Close"].astype(float).loc[:ASOF]
    f = fwd_ret(s, 5)
    o = pd.DatetimeIndex([d for d in opex_all
                          if d in s.index and d <= ASOF and d.month == 8 and d.year % 4 == 2])
    w = f.reindex(o).dropna()
    up = int((w.values > 0).sum())
    st = summarize(w.values, t)
    print(f"  {t:5s} n={st['n']:2d} mean={st['mean_pct']:+.3f}% med={st['median_pct']:+.3f}% "
          f"record {up}-{st['n']-up} sign_p={sign_test(up, st['n']):.4f} "
          f"years={[d.year for d in w.index]}")
