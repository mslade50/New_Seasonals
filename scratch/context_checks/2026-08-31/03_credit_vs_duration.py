"""The conjunction the sweep cannot see: credit at its highs, duration at its lows.

On 2026-08-31 HYG closed 0.14% below its 52-week HIGH while TLT closed 1.44%
above its 52-week LOW, IEF 0.61% above and LQD 0.75% above, on the same day the
10-year printed a 52-week high in yield.

That is a pure rates move with no credit component. The sweep enumerates single
triggers, so this state never fires anything. Question: how common is it, and
what has followed for stocks and for the two bond legs?

Cell: HYG within 1% of its trailing-252 max AND TLT within 3% of its
trailing-252 min, same session. Novelty-filtered to first occurrence in 30+
calendar days so a multi-month standoff counts once, matching the engine's
_first_in_calendar_days convention for state triggers.

Convention: lag=0 close-to-close from the anchor close, h=1 is the next session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, era_split, fwd_ret,  # noqa: E402
                       local_control, sign_test, show, summarize)

px = close_panel(["HYG", "TLT", "SPY", "IEF", "LQD", "^TNX", "^VIX"])
px = px[px.index >= "2008-01-01"]  # HYG inception 2007-04
sub = px[["HYG", "TLT", "SPY"]].dropna()

hyg, tlt = sub["HYG"], sub["TLT"]
hyg_dist = hyg / hyg.rolling(252, min_periods=252).max() - 1.0
tlt_dist = tlt / tlt.rolling(252, min_periods=252).min() - 1.0

state = (hyg_dist >= -0.01) & (tlt_dist <= 0.03)
print(f"raw state days: {int(state.sum())} of {len(sub)} "
      f"({100*state.mean():.1f}%) since {sub.index[0].date()}")
print(f"today HYG dist to 52w high {100*hyg_dist.iloc[-1]:+.2f}%, "
      f"TLT dist above 52w low {100*tlt_dist.iloc[-1]:+.2f}%, "
      f"state = {bool(state.iloc[-1])}")


def first_in_calendar_days(mask, days=30):
    out = pd.Series(False, index=mask.index)
    last = None
    for d in mask.index[mask.fillna(False).values]:
        if last is None or (d - last).days > days:
            out.loc[d] = True
        last = d
    return out


epi = first_in_calendar_days(state, 30)
ed = epi.index[epi.values]
print(f"\nepisodes (first in 30+ calendar days): {len(ed)}")
print("  ", [str(d.date()) for d in ed])

hist = ed[ed < pd.Timestamp("2026-08-31")]
print(f"  with forward data: {len(hist)}")

print("\n--- raw-state-day version (no novelty filter), for the record ---")
sd = state.index[state.values]
sd = sd[sd < pd.Timestamp("2026-08-31")]
for h in (1, 5, 10, 21):
    f = fwd_ret(sub["SPY"], h).dropna()
    d = pd.DatetimeIndex(sd).intersection(f.index)
    v = f.loc[d].values
    up = int((v > 0).sum())
    print(f"  SPY h={h:2d} n={len(v):3d} {up}-{len(v)-up} up mean {100*v.mean():+.3f}% "
          f"t {summarize(v)['t']:+.2f}")

HS = (1, 3, 5, 10, 21)
for tkr in ["SPY", "HYG", "TLT"]:
    s = px[tkr].dropna()
    rows = []
    for h in HS:
        f = fwd_ret(s, h).dropna()
        d = pd.DatetimeIndex(hist).intersection(f.index)
        r = summarize(f.loc[d].values, f"h={h}")
        if r.get("n"):
            ctl = local_control(f.index, d, 126)
            r["all_days_pct"] = round(100 * f.mean(), 3)
            r["local_ctl_pct"] = round(100 * f.loc[ctl].mean(), 3)
            r["edge_vs_local"] = round(r["mean_pct"] - 100 * f.loc[ctl].mean(), 3)
            v = f.loc[d].values
            r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(r)
    show(rows, f"{tkr} after credit-high / duration-low")

print("\n=== episode detail: SPY and HYG over the next 21 sessions ===")
f21s = fwd_ret(px["SPY"], 21)
f21h = fwd_ret(px["HYG"], 21)
f5s = fwd_ret(px["SPY"], 5)
for d in hist:
    a = f5s.get(d, np.nan)
    b = f21s.get(d, np.nan)
    c = f21h.get(d, np.nan)
    print(f"  {d.date()}  SPY 5d {100*a:+6.2f}%  SPY 21d {100*b:+6.2f}%  HYG 21d {100*c:+6.2f}%")

print("\n=== era split, SPY h=21 ===")
f = fwd_ret(px["SPY"], 21).dropna()
d = pd.DatetimeIndex(hist).intersection(f.index)
if len(d):
    show(era_split(d, f.loc[d].values), "SPY h=21")
    print("  ", cluster_note(d, f.loc[d].values, k=2))
