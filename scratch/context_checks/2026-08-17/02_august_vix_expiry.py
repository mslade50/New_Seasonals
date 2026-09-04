"""The August inversion found in script 01.

Base cell: 319 anchors 2 td before a VIX expiry, SPY +0.190% t=2.87, QQQ +0.216%
hit 58.9% sign p 0.0008, era-stable and stronger 2018+, local control +0.03%, no
concentration. Clears BH on QQQ.

The 26 August anchors go the other way: SPY 11-15, QQQ 11-15, IWM 9-17, ^VIX +2.07%.
Tomorrow is an August one. Questions: is that just August, is it era-stable, and is
the vol side the cleaner statement?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, summarize, era_split, sign_test, cluster_note,
)

px = close_panel(["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]).dropna(how="all")
dates = px.index
ev = load_events(["vix_expiry"])
ev = ev[(ev["date"] >= dates[0]) & (ev["date"] <= dates[-1])]

anchors = []
for d in ev["date"]:
    pos = dates.searchsorted(d)
    if pos < len(dates) and pos - 2 >= 0:
        anchors.append(dates[pos - 2])
anchors = pd.DatetimeIndex(sorted(set(anchors)))
aug = anchors[anchors.month == 8]
non_aug = anchors[anchors.month != 8]


def line(d):
    if d.get("n", 0) == 0:
        return "n=0"
    up = int(round(d["hit"] / 100 * d["n"]))
    return (f"n={d['n']:5d} mean={d['mean_pct']:+.3f}% med={d['median_pct']:+.3f}% "
            f"hit={d['hit']:5.1f}% t={d['t']:+.2f} rec={up}-{d['n']-up} "
            f"signp={sign_test(up, d['n']):.4f}")


print("=== the three-way split, h1 ===")
for t in ["SPY", "QQQ", "IWM", "^VIX"]:
    f = fwd_ret(px[t], 1)
    print(f"{t}")
    print(f"   august anchors   {line(summarize(f.reindex(aug).dropna().values))}")
    print(f"   other anchors    {line(summarize(f.reindex(non_aug).dropna().values))}")
    allaug = f[f.index.month == 8].dropna()
    print(f"   ALL august days  {line(summarize(allaug.values))}")
    print(f"   all days         {line(summarize(f.dropna().values))}")

print("\n=== is the August cell distinct from August itself? ===")
# august days that are NOT anchors
for t in ["SPY", "QQQ", "^VIX"]:
    f = fwd_ret(px[t], 1)
    aug_days = f.index[(f.index.month == 8)]
    aug_non_anchor = aug_days.difference(aug)
    a = summarize(f.reindex(aug).dropna().values)
    b = summarize(f.reindex(aug_non_anchor).dropna().values)
    print(f"{t:6s} anchor {a['mean_pct']:+.3f}% (n={a['n']}) vs "
          f"other-august {b['mean_pct']:+.3f}% (n={b['n']})  "
          f"gap {a['mean_pct']-b['mean_pct']:+.3f}pp")

print("\n=== August anchors, era split ===")
for t in ["SPY", "QQQ", "IWM", "^VIX"]:
    f = fwd_ret(px[t], 1).reindex(aug).dropna()
    for part in era_split(f.index, f.values):
        print(f"{t:6s} {part.get('label',''):12s} {line(part)}")

print("\n=== August anchors, concentration ===")
for t in ["SPY", "QQQ", "^VIX"]:
    f = fwd_ret(px[t], 1).reindex(aug).dropna()
    print(f"{t:6s} {cluster_note(f.index, f.values, k=2)}")

print("\n=== every August anchor, episode by episode ===")
vix_r = px["^VIX"] / px["^VIX"].shift(1) - 1.0
spx_r = px["^GSPC"] / px["^GSPC"].shift(1) - 1.0
f_spy, f_qqq, f_vix = fwd_ret(px["SPY"], 1), fwd_ret(px["QQQ"], 1), fwd_ret(px["^VIX"], 1)
print(f"{'anchor':12s} {'anchorVIX':>9s} {'anchorSPX':>9s} | {'SPY h1':>8s} "
      f"{'QQQ h1':>8s} {'VIX h1':>8s}")
for d in aug:
    print(f"{str(d.date()):12s} {100*vix_r.get(d,np.nan):+8.1f}% "
          f"{100*spx_r.get(d,np.nan):+8.2f}% | {100*f_spy.get(d,np.nan):+7.2f}% "
          f"{100*f_qqq.get(d,np.nan):+7.2f}% {100*f_vix.get(d,np.nan):+7.1f}%")

print("\n=== August anchors where vol was already bid on the anchor (VIX up) ===")
sub = aug[vix_r.reindex(aug) > 0]
print(f"n = {len(sub)}: {[str(x.date()) for x in sub]}")
for t in ["SPY", "QQQ", "^VIX"]:
    print(f"{t:6s} {line(summarize(fwd_ret(px[t], 1).reindex(sub).dropna().values))}")

print("\n=== August anchors, VIX up AND index down (tonight's shape) ===")
sub2 = aug[(vix_r.reindex(aug) > 0) & (spx_r.reindex(aug) < 0)]
print(f"n = {len(sub2)}: {[str(x.date()) for x in sub2]}")
for t in ["SPY", "QQQ", "^VIX"]:
    print(f"{t:6s} {line(summarize(fwd_ret(px[t], 1).reindex(sub2).dropna().values))}")

print("\n=== August anchors, horizons (h1 = the day before expiry, h2 = expiry) ===")
for h in [1, 2, 3, 5]:
    parts = []
    for t in ["SPY", "QQQ", "^VIX"]:
        d = summarize(fwd_ret(px[t], h).reindex(aug).dropna().values)
        parts.append(f"{t} {d['mean_pct']:+.2f}% hit={d['hit']:.0f}%")
    print(f"h{h}: " + " | ".join(parts))

print("\n=== midterm August anchors ===")
mid_aug = aug[aug.year % 4 == 2]
print(f"n = {len(mid_aug)}: {[str(x.date()) for x in mid_aug]}")
for t in ["SPY", "QQQ", "^VIX"]:
    print(f"{t:6s} {line(summarize(fwd_ret(px[t], 1).reindex(mid_aug).dropna().values))}")
