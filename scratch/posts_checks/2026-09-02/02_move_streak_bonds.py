"""Idea candidate for Thursday 2026-09-03: bonds after a 5-session MOVE up streak.

Tonight's brief: MOVE has 5 straight up closes (+14.79%, 5d rank 93) while
VIX sits at a 63d rank of 7. Across 55 streak episodes MOVE was lower a week
later 39 times, median -3.31%. The brief says equities are empty inside a
month. Bonds were not reported. Question: is TLT / IEF a tradeable long
from tomorrow's close (lag-1) or tomorrow's open?

Episode = the FIRST session a MOVE up-streak reaches 5 (one per streak).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, era_split, fwd_lag, load_prices, local_control, sign_test,
    summarize, wilder_atr,
)

ASOF = pd.Timestamp("2026-09-02")
px = load_prices(["^MOVE", "TLT", "IEF", "SPY", "^VIX"])
mv = px["^MOVE"]["Close"].dropna()
up = (mv > mv.shift(1)).astype(int)
streak = up.groupby((up == 0).cumsum()).cumsum()
first5 = mv.index[(streak == 5).values]
first5 = first5[first5 < ASOF] if streak.iloc[-1] != 5 else first5[first5 <= ASOF]
print(f"MOVE {mv.iloc[-1]:.2f} on {mv.index[-1].date()}  current streak {int(streak.iloc[-1])}  "
      f"episodes (streak hits 5): {len(first5)}  last {first5[-3:].date.tolist()}")
epi = first5[first5 < ASOF]


def block(name, r, s, h, lag):
    r = r.dropna()
    if len(r) == 0:
        print(f"  {name:<40} n=0")
        return r
    st = summarize(r.values)
    nup = int((r > 0).sum())
    allr = fwd_lag(s, h, lag).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name:<40} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(r)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(r)):.4f}  "
          f"| all {100*allr.mean():+.3f}% hit {100*(allr>0).mean():.1f}%  local {100*loc.mean():+.3f}% "
          f"hit {100*(loc>0).mean():.1f}%  | worst {st['worst_pct']:+.2f}% ({r.idxmin().date()})")
    return r


def splits(r):
    r = r.dropna()
    v = r.values
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1)) for e in era_split(r.index, v)])
    print("    concentration:", cluster_note(r.index, v))


print("\n=== MOVE itself, lag-0 (brief: 39-16 lower at h5, median -3.31%) ===")
for h in (1, 5, 10):
    block(f"MOVE lag0 h{h}", fwd_lag(mv, h, 0).reindex(epi), mv, h, 0)

for tk in ("TLT", "IEF", "SPY"):
    c = px[tk]["Close"].dropna()
    o = px[tk]["Open"].reindex(c.index)
    e = epi[epi.isin(c.index)]
    atr = pd.Series(wilder_atr(px[tk]["High"], px[tk]["Low"], px[tk]["Close"]), index=px[tk].index).reindex(c.index)
    print(f"\n=== {tk}  (tonight {c.iloc[-1]:.2f}, Wilder-14 ATR {atr.iloc[-1]:.4f} = {100*atr.iloc[-1]/c.iloc[-1]:.2f}%) ===")
    print("  -- lag-0 from the streak close --")
    for h in (1, 5, 10):
        block(f"{tk} lag0 h{h}", fwd_lag(c, h, 0).reindex(e), c, h, 0)
    print("  -- lag-1 (MOC D+1 entry) --")
    for h in (1, 3, 5, 10):
        r = block(f"{tk} lag1 h{h}", fwd_lag(c, h, 1).reindex(e), c, h, 1)
        if tk in ("TLT", "IEF") and h in (5, 10):
            splits(r)
    print("  -- MOO D+1 entry -> close D+1+h --")
    p = pd.Series(np.arange(len(c)), index=c.index)
    for h in (1, 5):
        oc = pd.Series({d: c.iloc[p[d] + h] / o.iloc[p[d] + 1] - 1 for d in e if p[d] + h < len(c)})
        st = summarize(oc.values)
        nup = int((oc > 0).sum())
        print(f"  {tk} MOO D+1 -> close D+{h}                n={st['n']}  mean={st['mean_pct']:+.3f}%  "
              f"{nup}-{len(oc)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(oc)):.4f}  worst {st['worst_pct']:+.2f}%")

# the live crossing: streak AND VIX 63d rank <= 10
vix = px["^VIX"]["Close"].dropna()
from pitch_lab import pct_rank  # noqa: E402
vr = pct_rank(vix, 63, 252)
calm = epi[(vr.reindex(epi) <= 15).fillna(False).values]
print(f"\n=== streak episodes with VIX 63d rank <= 15 (tonight {vr.iloc[-1]:.1f}): {len(calm)} ===")
for tk in ("TLT", "SPY"):
    c = px[tk]["Close"].dropna()
    for h in (5, 10):
        block(f"{tk} lag1 h{h}, calm-VIX subset", fwd_lag(c, h, 1).reindex(calm[calm.isin(c.index)]), c, h, 1)
