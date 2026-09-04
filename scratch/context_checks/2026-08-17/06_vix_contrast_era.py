"""Era and concentration check on the contrast found in script 05.

VIX +5%+ alongside a deep selloff mean-reverts hard (VIX h5 -3.45%, 237-445, t=-4.94).
VIX +5%+ on a shallow down day like today does not (VIX h5 +1.06%, at the
unconditional +1.01%). Before publishing a contrast, check both halves hold in both
eras and that neither is carried by a handful of episodes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, declusters, summarize, era_split, sign_test, cluster_note,
)

px = close_panel(["SPY", "^GSPC", "^VIX"]).dropna(how="all")
spx_r = px["^GSPC"] / px["^GSPC"].shift(1) - 1.0
vix_r = px["^VIX"] / px["^VIX"].shift(1) - 1.0
common = spx_r.dropna().index.intersection(vix_r.dropna().index)

shallow = common[((vix_r.reindex(common) >= 0.05) & (spx_r.reindex(common) > -0.01)
                  & (spx_r.reindex(common) < 0)).values]
deep = common[((vix_r.reindex(common) >= 0.05) & (spx_r.reindex(common) <= -0.01)).values]


def line(d):
    if d.get("n", 0) == 0:
        return "n=0"
    up = int(round(d["hit"] / 100 * d["n"]))
    return (f"n={d['n']:5d} mean={d['mean_pct']:+.3f}% med={d['median_pct']:+.3f}% "
            f"hit={d['hit']:5.1f}% t={d['t']:+.2f} rec={up}-{d['n']-up} "
            f"signp={sign_test(up, d['n']):.4f}")


for name, idx in [("shallow (SPX 0 to -1%)", shallow), ("deep (SPX <= -1%)", deep)]:
    print(f"\n=== {name}, ^VIX forward ===")
    for h in [1, 5, 10]:
        v = fwd_ret(px["^VIX"], h).reindex(idx).dropna()
        print(f"  h{h:<3d} {line(summarize(v.values))}")
    v5 = fwd_ret(px["^VIX"], 5).reindex(idx).dropna()
    print("  era split h5:")
    for part in era_split(v5.index, v5.values):
        u = int(round(part["hit"] / 100 * part["n"]))
        print(f"    {part.get('label',''):12s} n={part['n']:4d} "
              f"mean={part['mean_pct']:+.3f}% med={part['median_pct']:+.3f}% "
              f"rec={u}-{part['n']-u} t={part['t']:+.2f}")
    print(f"  {cluster_note(v5.index, v5.values, k=2)}")
    dec = declusters(idx, 5, px.index)
    vd = fwd_ret(px["^VIX"], 5).reindex(dec).dropna()
    print(f"  declustered 5td: {line(summarize(vd.values))}")

print("\n=== the contrast itself, by era (VIX h5) ===")
for cut, lbl in [(None, "full")]:
    pass
for era_lo, era_hi, lbl in [("1999-01-01", "2018-01-01", "pre-2018"),
                            ("2018-01-01", "2027-01-01", "2018+")]:
    a = shallow[(shallow >= era_lo) & (shallow < era_hi)]
    b = deep[(deep >= era_lo) & (deep < era_hi)]
    va = fwd_ret(px["^VIX"], 5).reindex(a).dropna()
    vb = fwd_ret(px["^VIX"], 5).reindex(b).dropna()
    sa, sb = summarize(va.values), summarize(vb.values)
    print(f"  {lbl:9s} shallow {sa['mean_pct']:+.2f}% (n={sa['n']}, med "
          f"{sa['median_pct']:+.2f}%) vs deep {sb['mean_pct']:+.2f}% "
          f"(n={sb['n']}, med {sb['median_pct']:+.2f}%)  "
          f"gap {sa['mean_pct']-sb['mean_pct']:+.2f}pp")

print("\n=== unconditional ^VIX for reference ===")
for h in [1, 5, 10]:
    print(f"  h{h:<3d} {line(summarize(fwd_ret(px['^VIX'], h).dropna().values))}")
