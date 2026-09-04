"""The sweep's strongest cell: bitcoin z10 >= 2 stretched up (n=294, t=3.15,
BH pass, era stable). Bitcoin closed today at z10 3.19, well past the trigger.
Does the continuation survive conditioning on the deeper stretch, and does it
survive a decluster? The sweep counts overlapping days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, summarize, sign_test, fwd_ret, declusters,
                       local_control, era_split, cluster_note)  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["BTC-USD"])
c = px["BTC-USD"]["Close"].loc[:ASOF]
idx = c.index
print("panel", idx[0].date(), "->", idx[-1].date(), len(idx), "bars (7-day calendar)")

# z10 exactly as build_context_state / build_pitch_state._metrics_for define it:
# 10d return over 21d daily vol scaled to 10d
r1 = c.pct_change()
z10 = c.pct_change(10) / (r1.rolling(21).std() * np.sqrt(10))
print(f"live z10 {z10.iloc[-1]:.2f}, 10d return {100*c.pct_change(10).iloc[-1]:.2f}%, "
      f"21d {100*c.pct_change(21).iloc[-1]:.2f}%")


def block(mask, label, gap):
    dts = idx[mask.reindex(idx).fillna(False)]
    dts = pd.DatetimeIndex([d for d in dts if d <= ASOF])
    dc = declusters(dts, gap, idx) if gap else dts
    print(f"-- {label}: raw n={len(dts)}, decluster@{gap} n={len(dc)}")
    ctrl = local_control(idx, dc, 126)
    for h in (1, 5, 10, 21):
        f = fwd_ret(c, h)
        v = f.reindex(dc).dropna()
        if len(v) < 4:
            print(f"   h{h}: n={len(v)} too few")
            continue
        st = summarize(v.values, "")
        up = int((v.values > 0).sum())
        cs = summarize(f.reindex(ctrl).dropna().values, "")
        a = summarize(f.dropna().values, "")
        print(f"   h{h:<3} n={st['n']:<3} mean {st['mean_pct']:>7.2f}%  med {st['median_pct']:>7.2f}%  "
              f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  t {st['t']:>5.2f} | "
              f"local ctrl {cs['mean_pct']:>6.2f}% | all {a['mean_pct']:>6.2f}%")
        if h in (1, 21):
            print("      era:", [(e['label'], e['n'], round(e.get('mean_pct', float('nan')), 2)) for e in era_split(v.index, v.values)])
            print("      ", cluster_note(v.index, v.values))


block(z10 >= 2, "z10 >= 2 (the sweep's cell, overlapping days)", 0)
print()
block(z10 >= 2, "z10 >= 2, declustered at 10 bars", 10)
print()
block(z10 >= 3, "z10 >= 3, the live reading, declustered at 10 bars", 10)
print()
print("=== how rare is z10 >= 3 in bitcoin, and when ===")
d3 = idx[(z10 >= 3).reindex(idx).fillna(False)]
d3c = declusters(d3, 10, idx)
print(f"raw {len(d3)} bars, {len(d3c)} episodes; by year "
      f"{dict(pd.Series(d3c.year).value_counts().sort_index())}")
