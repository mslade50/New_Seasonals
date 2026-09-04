"""Is the expiry-week cell anything more than a Monday?

The engine fired `E:vix_expiry` at k=3, which for a Wednesday expiry is the Friday close, so h1
is the Monday of expiry week. That cell is confounded with the plain Monday cell by construction.
Three questions:
  1. expiry-week Mondays vs all other Mondays (the Monday effect held constant)
  2. the August subset, which the engine flagged at 80.8% for QQQ on 26 anchors
  3. where in expiry week the drift actually sits, h1 through h5
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, era_split, cluster_note, load_events  # noqa


TICKERS = ["QQQ", "SPY", "^GSPC", "IWM", "^VIX"]
px = close_panel(TICKERS)
idx = px.index

ev = load_events(["vix_expiry", "opex"])
vx = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == "vix_expiry", "date"])))
print(f"vix expiries loaded: {len(vx)}  {vx.min().date()} .. {vx.max().date()}")

# anchor = the session 3 td before each expiry, matching the engine
pos = {d: i for i, d in enumerate(idx)}
anchors = []
for e in vx:
    loc = idx.searchsorted(e)
    if loc >= len(idx) or idx[loc] != e:
        continue
    if loc - 3 < 0:
        continue
    anchors.append(idx[loc - 3])
anchors = pd.DatetimeIndex(sorted(set(anchors)))
anchors = anchors[anchors <= idx[-1]]
print(f"k3 anchors: {len(anchors)}   weekday mix: "
      f"{dict(pd.Series(anchors.weekday).value_counts().sort_index())}  (4 = Friday)")

nxt = pd.Series(idx, index=idx).shift(-1)
nxt_wd = nxt.dt.weekday
nxt_mo = nxt.dt.month

is_anchor = pd.Series(idx.isin(anchors), index=idx)
is_mon_anchor = (nxt_wd == 0).fillna(False)

expiry_mon = (is_anchor & is_mon_anchor).values          # expiry-week Mondays
other_mon = (~is_anchor & is_mon_anchor).values          # every other Monday
aug_expiry_mon = (is_anchor & is_mon_anchor & (nxt_mo == 8)).values
aug_other_mon = (~is_anchor & is_mon_anchor & (nxt_mo == 8)).values

print(f"\nexpiry-week Mondays {expiry_mon.sum()}   other Mondays {other_mon.sum()}"
      f"   August expiry-week Mondays {aug_expiry_mon.sum()}   August other Mondays {aug_other_mon.sum()}")


def row(tkr, name, mask, h=1):
    v = fwd_ret(px[tkr], h)[mask].dropna()
    if not len(v):
        return None
    s = summarize(v.values, name)
    up = int((v > 0).sum())
    print(f"  {name:30s} n {s['n']:4d}  mean {s['mean_pct']:+.3f}%  med {s['median_pct']:+.3f}%"
          f"  hit {s['hit']:.1f}%  t {s['t']:+.2f}  {up}-{s['n']-up}  signp {sign_test(up, s['n']):.4f}")
    return v


for tkr in ["QQQ", "SPY", "^GSPC", "IWM"]:
    print("=" * 84)
    print(f"{tkr}  h1")
    row(tkr, "expiry-week Monday", expiry_mon)
    row(tkr, "any other Monday", other_mon)
    v_aug = row(tkr, "August expiry-week Monday", aug_expiry_mon)
    row(tkr, "August, other Mondays", aug_other_mon)
    if v_aug is not None:
        print("   era:", [f"{e['label']} n {e['n']} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}%"
                          for e in era_split(v_aug.index, v_aug.values)])
        print("   cluster:", cluster_note(v_aug.index, v_aug.values, k=2))
        mid = v_aug[v_aug.index.year % 4 == 2]
        mu = int((mid > 0).sum())
        print(f"   midterm: n {len(mid)} {mu}-{len(mid)-mu} mean {100*mid.mean():+.3f}%")
        print("   the 26 August anchors:",
              ", ".join(f"{d.date()}:{100*x:+.2f}" for d, x in zip(v_aug.index, v_aug.values)))

# where in expiry week the move sits, all months, SPY and QQQ
print("\n" + "=" * 84)
print("cumulative from the Friday close through expiry week (all months)")
for tkr in ["SPY", "QQQ", "^VIX"]:
    print(f"  {tkr}")
    for h in (1, 2, 3, 4, 5):
        v = fwd_ret(px[tkr], h)[expiry_mon].dropna()
        s = summarize(v.values, f"h{h}")
        up = int((v > 0).sum())
        print(f"    h{h}  n {s['n']:4d}  mean {s['mean_pct']:+.3f}%  hit {s['hit']:.1f}%"
              f"  t {s['t']:+.2f}  {up}-{s['n']-up}")

# the same-week increments, so the Monday is not double counted downstream
print("\nper-session increments inside expiry week, SPY / QQQ (mean %, hit)")
for tkr in ["SPY", "QQQ"]:
    out = []
    for h in (1, 2, 3, 4, 5):
        a = fwd_ret(px[tkr], h)[expiry_mon]
        b = fwd_ret(px[tkr], h - 1)[expiry_mon] if h > 1 else pd.Series(0.0, index=a.index)
        inc = ((1 + a) / (1 + b) - 1).dropna()
        out.append(f"d{h} {100*inc.mean():+.3f}% ({100*(inc>0).mean():.0f}%)")
    print(f"  {tkr}: " + "  ".join(out))
