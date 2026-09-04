"""Tonight's context brief scored the August expiry-week Monday for SPY at
20-6, mean +0.43% from the Friday close (N=26, sign_p 0.0047). A post drafted
Saturday can only enter MOO on the Monday, so the tradeable slice is the
OPEN-to-close leg, not the close-to-close cell. If the weekend gap carries the
whole move, the idea does not ship and the stat runs alone.

Cells:
  - cc: Friday close -> Monday close (recheck of the brief's cell)
  - gap: Friday close -> Monday open (what MOO forfeits)
  - oc: Monday open -> Monday close (what MOO captures)
Control: same three legs on every non-expiry-week Monday.
Anchor = session 3 td before each VIX expiry when the next session is a
Monday, matching the engine's E:vix_expiry k=3 definition verbatim.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_events, load_prices, sign_test, summarize  # noqa

px = {t: load_prices([t])[t] for t in ("SPY", "QQQ")}

ev = load_events(["vix_expiry"])
vx = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == "vix_expiry", "date"])))

for tkr, df in px.items():
    close = df["Close"].dropna()
    opn = df["Open"].reindex(close.index)
    idx = close.index

    anchors = []
    for e in vx:
        loc = idx.searchsorted(e)
        if loc >= len(idx) or idx[loc] != e or loc - 3 < 0:
            continue
        anchors.append(idx[loc - 3])
    anchors = pd.DatetimeIndex(sorted(set(anchors)))

    nxt = pd.Series(idx, index=idx).shift(-1)
    is_anchor = pd.Series(idx.isin(anchors), index=idx)
    mon_next = (nxt.dt.weekday == 0).fillna(False)
    aug_exp = (is_anchor & mon_next & (nxt.dt.month == 8)).values
    other_mon = (~is_anchor & mon_next).values

    cc = close.pct_change().shift(-1)          # anchor-indexed: next session cc
    gap = (opn / close.shift(1) - 1).shift(-1)  # next session's open vs anchor close
    oc = (close / opn - 1).shift(-1)            # next session's open -> close

    print("=" * 78)
    print(f"{tkr}: August expiry-week Mondays (anchor-indexed)")
    for name, ser in (("cc  Fri close->Mon close", cc),
                      ("gap Fri close->Mon open ", gap),
                      ("oc  Mon open->Mon close ", oc)):
        for label, mask in ((f"aug expiry Mon", aug_exp), ("other Mondays ", other_mon)):
            v = ser[mask].dropna()
            s = summarize(v.values, label)
            up = int((v > 0).sum())
            print(f"  {name} | {label}  n {s['n']:4d}  mean {s['mean_pct']:+.3f}%"
                  f"  med {s['median_pct']:+.3f}%  hit {s['hit']:.1f}%"
                  f"  {up}-{s['n']-up}  signp {sign_test(up, s['n']):.4f}")
    v = oc[aug_exp].dropna()
    print("  oc per-anchor:", ", ".join(
        f"{d.date()}:{100*x:+.2f}" for d, x in zip(v.index, v.values)))
    print(f"  oc worst {100*v.min():+.2f}%  best {100*v.max():+.2f}%")
