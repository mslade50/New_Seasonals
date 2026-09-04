"""Gold into a CPI print when gold is already extended.

Engine: GC=F on the print session n=309, +0.140%, 176-132, t=2.29, era-stable,
BH pass. Separately P4 says gold's own z10 stretch (+2.14 tonight) predicts
nothing on its own: n=286, +0.005%, t=0.06, era-UNSTABLE.

Gold enters tonight +8.12% over 5 sessions, 5d rank 97.6, 21d rank 90.9,
z10 +2.14. So: does the print-day drift survive the stretch, or is the
stretch where it dies?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["GC=F", "SI=F"])
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev["date"]).unique()))

for tk in ("GC=F", "SI=F"):
    c = px[tk]["Close"].dropna()
    d = c.index
    anc = []
    for x in cpi:
        pos = d.searchsorted(x)
        if pos <= 0 or pos >= len(d) or d[pos] != x:
            continue
        anc.append(d[pos - 1])
    anc = pd.DatetimeIndex(anc)

    r = c.pct_change()
    nxt = r.shift(-1)
    r5 = c.pct_change(5)
    rank5 = r5.rolling(252, min_periods=252).rank(pct=True) * 100
    z10 = (c.pct_change(10) / (r.rolling(21).std() * np.sqrt(10)))

    base = nxt.reindex(anc).dropna()
    s = summarize(base.values, "all CPI")
    print(f"\n=== {tk} ===")
    print(f"  ALL CPI prints: n={s['n']} mean {s['mean_pct']:+.3f}% "
          f"up {(base>0).sum()}-{(base<0).sum()} t={s['t']:+.2f} "
          f"sign p {sign_test(int((base>0).sum()), int(len(base))):.4f}")

    for name, mask in [("5d rank >= 90", rank5 >= 90), ("5d rank >= 95", rank5 >= 95),
                       ("z10 >= 1.5", z10 >= 1.5), ("z10 >= 2.0", z10 >= 2.0),
                       ("5d rank < 90", rank5 < 90)]:
        sel = pd.DatetimeIndex([x for x in anc if bool(mask.get(x, False))])
        v = nxt.reindex(sel).dropna()
        if len(v) < 5:
            print(f"  {name:<14} n={len(v)} too small")
            continue
        ss = summarize(v.values, name)
        print(f"  {name:<14} n={ss['n']:<4} mean {ss['mean_pct']:+7.3f}%  "
              f"up {(v>0).sum():>3}-{(v<0).sum():<3} {ss['hit']:5.1f}%  t={ss['t']:+5.2f}  "
              f"sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}")

    # tonight's state on gold: 5d rank >= 95 AND z10 >= 1.5
    sel = pd.DatetimeIndex([x for x in anc
                            if bool((rank5 >= 95).get(x, False)) and bool((z10 >= 1.5).get(x, False))])
    v = nxt.reindex(sel).dropna()
    if len(v) >= 5:
        ss = summarize(v.values, "tonight")
        print(f"  TONIGHT (5d rank>=95 AND z10>=1.5): n={ss['n']} mean {ss['mean_pct']:+.3f}% "
              f"up {(v>0).sum()}-{(v<0).sum()} t={ss['t']:+.2f} "
              f"sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}")
        print("   episodes:", [(str(x.date()), round(y*100, 2)) for x, y in v.items()])
        print("   era:", [(e['label'], e['n'], round(e['mean_pct'], 3)) for e in era_split(v.index, v.values)])
        # control: same stretch, no print next session
        allst = pd.DatetimeIndex([x for x in c.index
                                  if bool((rank5 >= 95).get(x, False)) and bool((z10 >= 1.5).get(x, False))])
        vc = nxt.reindex(allst.difference(anc)).dropna()
        sc = summarize(vc.values, "ctrl")
        print(f"   same stretch, NO print next: n={sc['n']} mean {sc['mean_pct']:+.3f}% hit {sc['hit']:.1f}%")
        print(f"   edge: {ss['mean_pct']-sc['mean_pct']:+.3f}%")
