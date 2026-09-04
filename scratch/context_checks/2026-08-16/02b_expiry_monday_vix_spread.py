"""Two follow-ons to 02: what VIX does on the August expiry-week Monday, and whether the
QQQ-minus-SPY August Monday spread from 01 is really the same 26 sessions.

The all-months version of this anchor has VIX UP 58% at h1 (t 3.15). The engine's August
subset has it down 69% of the time. If both hold, the August session is the sign flip.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, era_split, cluster_note, load_events  # noqa

px = close_panel(["SPY", "QQQ", "^VIX", "^VIX3M"])
idx = px.index
ev = load_events(["vix_expiry"])
vx = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == "vix_expiry", "date"])))

anchors = []
for e in vx:
    loc = idx.searchsorted(e)
    if loc >= len(idx) or idx[loc] != e or loc - 3 < 0:
        continue
    anchors.append(idx[loc - 3])
anchors = pd.DatetimeIndex(sorted(set(anchors)))

nxt = pd.Series(idx, index=idx).shift(-1)
is_anchor = pd.Series(idx.isin(anchors), index=idx)
mon = (nxt.dt.weekday == 0).fillna(False)
aug = (nxt.dt.month == 8).fillna(False)

cell = (is_anchor & mon & aug).values
allm = (is_anchor & mon).values
other_mon = (~is_anchor & mon).values

print("VIX, h1")
for name, m in [("August expiry-week Monday", cell),
                ("expiry-week Monday, all months", allm),
                ("every other Monday", other_mon)]:
    v = fwd_ret(px["^VIX"], 1)[m].dropna()
    s = summarize(v.values, name)
    up = int((v > 0).sum())
    print(f"  {name:32s} n {s['n']:4d}  mean {s['mean_pct']:+.2f}%  med {s['median_pct']:+.2f}%"
          f"  up {up}-{s['n']-up} ({s['hit']:.1f}%)  t {s['t']:+.2f}  signp {sign_test(up, s['n']):.4f}")

v = fwd_ret(px["^VIX"], 1)[cell].dropna()
print("  era:", [f"{e['label']} n {e['n']} mean {e['mean_pct']:+.2f}% up {e['hit']:.1f}%"
                 for e in era_split(v.index, v.values)])
print("  the 26:", ", ".join(f"{d.year}:{100*x:+.1f}" for d, x in zip(v.index, v.values)))

# joint: SPY up and VIX down on the same session
spy = fwd_ret(px["SPY"], 1)
vix = fwd_ret(px["^VIX"], 1)
joint = ((spy > 0) & (vix < 0))
for name, m in [("August expiry-week Monday", cell), ("expiry-week Monday, all months", allm),
                ("every other Monday", other_mon)]:
    sub = joint[m].dropna()
    print(f"  SPY up and VIX down together, {name:32s} {int(sub.sum())}/{len(sub)} = {100*sub.mean():.1f}%")
base = joint.dropna()
print(f"  SPY up and VIX down together, every session               "
      f"{int(base.sum())}/{len(base)} = {100*base.mean():.1f}%")

# QQQ minus SPY: is the August Monday spread the expiry week or the rest of August
print("\nQQQ minus SPY, h1")
sp = fwd_ret(px["QQQ"], 1) - fwd_ret(px["SPY"], 1)
aug_mon_all = (mon & aug).values
aug_mon_nonexp = (mon & aug & ~is_anchor).values
for name, m in [("August expiry-week Monday", cell),
                ("August Mondays, other weeks", aug_mon_nonexp),
                ("all August Mondays", aug_mon_all),
                ("Mondays outside August", (mon & ~aug).values)]:
    v = sp[m].dropna()
    s = summarize(v.values, name)
    up = int((v > 0).sum())
    print(f"  {name:32s} n {s['n']:4d}  mean {s['mean_pct']:+.3f}%  hit {s['hit']:.1f}%"
          f"  {up}-{s['n']-up}  t {s['t']:+.2f}  signp {sign_test(up, s['n']):.4f}")

# and the term structure state going in, since tonight enters with VIX at 14.63
ts = (px["^VIX"] / px["^VIX3M"]).dropna()
print(f"\nVIX/VIX3M on the 2026-08-14 close: {ts.iloc[-1]:.3f}"
      f"   (trailing 252d percentile {100*(ts.iloc[-252:] < ts.iloc[-1]).mean():.0f})")
v = fwd_ret(px["SPY"], 1)
sub = pd.Series(v[cell].dropna())
tsv = ts.reindex(sub.index)
lo = sub[tsv < 0.90]
print(f"  of the 26 August anchors, {len(lo)} entered with VIX/VIX3M below 0.90: "
      f"{int((lo>0).sum())}-{len(lo)-int((lo>0).sum())}, mean {100*lo.mean():+.3f}%")
