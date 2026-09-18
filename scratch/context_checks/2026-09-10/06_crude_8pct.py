"""Crude closed +8.20% at 103.93. The bar is clean: volume 407k against a 241k 20-day median,
gap only +0.86%, so the move traded rather than rolled (contrast KC/SI/PL/PA/GC/HG/CT today).
What follows an 8% crude session, and does it matter that this one happened in an UPTREND?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, summarize, era_split, sign_test, cluster_note, declusters, local_control

px = load_prices(['CL=F','^GSPC','SPY','^VIX','XLE'])
cl = px['CL=F']['Close'].dropna()
d = cl.index[cl.index >= pd.Timestamp('1999-01-01')]
cl = cl.reindex(d)
r = cl.pct_change()
vol = px['CL=F']['Volume'].reindex(d)

# clean-bar filter: today's volume at least half the trailing 20d median (drops dead-contract seams)
volmed = vol.rolling(21).median().shift(1)
clean = (vol >= 0.5 * volmed) | volmed.isna()

r252 = cl / cl.rolling(252).min() - 1          # distance above the 252d low
hi252 = cl / cl.rolling(252).max() - 1         # distance below the 252d high
rank21 = cl.pct_change(21).rolling(252).rank(pct=True) * 100

for thr in (0.05, 0.06, 0.07, 0.08):
    m = (r >= thr) & clean
    print(f"CL=F sessions >= +{100*thr:.0f}% (clean bars only): {int(m.sum())}")
print()

trig = d[((r >= 0.08) & clean).fillna(False)]
trig = declusters(trig, 5, d)
print(f"+8% or better, declustered 5td: n={len(trig)}   last 6: {[str(x.date()) for x in trig[-6:]]}")

def fwd(sel, sym, h, src=None):
    s = (src if src is not None else px[sym]['Close'].dropna())
    out = []
    for a in sel:
        if a not in s.index: continue
        i = s.index.get_loc(a)
        if i + h < len(s): out.append((a, s.iloc[i+h]/s.iloc[i]-1))
    return pd.DatetimeIndex([o[0] for o in out]), np.array([o[1] for o in out])

def show(lab, sel, sym, h):
    dd, v = fwd(sel, sym, h)
    if len(v) < 4: print(f"  {lab:46} {sym:6} h{h:<2} n={len(v)}"); return None
    st = summarize(v); w = int((v > 0).sum())
    print(f"  {lab:46} {sym:6} h{h:<2} n={len(v):3} mean={st['mean_pct']:+7.3f}% up={w:3}/{len(v):3} "
          f"t={st['t']:+6.2f} signp={sign_test(w,len(v)):.4f}")
    return dd, v

print("\n=== forward from a +8% crude session ===")
for h in (1, 5, 21):
    show("crude +8% day", trig, 'CL=F', h)
    show("crude +8% day", trig, '^GSPC', h)
print()

# tonight's distinguishing feature: the spike came in an uptrend, not off a crash low
print("state tonight: 21d return rank", round(rank21.iloc[-1], 1),
      "| above 252d low", f"{100*r252.iloc[-1]:.1f}%", "| below 252d high", f"{100*hi252.iloc[-1]:.1f}%")
up_trend = trig[(hi252.reindex(trig) > -0.15).fillna(False)]
dn_trend = trig[(hi252.reindex(trig) <= -0.15).fillna(False)]
print(f"\nof the {len(trig)} spikes, {len(up_trend)} came within 15% of a 252d high, {len(dn_trend)} deeper than that")
print("  within 15% of the high:", [str(x.date()) for x in up_trend])
print()
for h in (1, 5, 21):
    show("+8% within 15% of a 252d high", up_trend, 'CL=F', h)
    show("+8% deeper than 15% below the high", dn_trend, 'CL=F', h)
print()
for h in (1, 5, 21):
    show("+8% within 15% of a 252d high", up_trend, '^GSPC', h)

print("\n=== CONTROLS ===")
for sym in ('CL=F','^GSPC'):
    for h in (1, 5, 21):
        dd, v = fwd(d, sym, h)
        print(f"  baseline {sym:6} h{h:<2} n={len(v):5} mean={summarize(v)['mean_pct']:+7.3f}% up={100*(v>0).mean():5.1f}%")
lc = local_control(d, trig, 126)
for h in (1, 5):
    dd, v = fwd(lc, 'CL=F', h)
    print(f"  local +/-126td control CL=F h{h}: n={len(v)} mean={summarize(v)['mean_pct']:+.3f}% up={100*(v>0).mean():.1f}%")

out = show("+8% within 15% of a 252d high", up_trend, 'CL=F', 5)
if out:
    dd, v = out
    print("  episodes:", ", ".join(f"{x.date()} {100*y:+.1f}%" for x, y in zip(dd, v)))
    print("  era:", [(e['label'], e['n'], round(e['mean_pct'],3)) for e in era_split(dd, v)])
    print("  concentration:", cluster_note(dd, v))
