"""Idea candidate for Tuesday 2026-09-01: long SPY after a MOVE jump off a low base.

Tonight's context brief (drill 07/07b) found: MOVE +6.13% from a 21-day rank
of 23, and across 58 such episodes SPY was higher ten sessions later 45 times
(77.6% vs 60.9% base), lag-0 from the anchor close, t 1.86 because of one
-9.46% tail (May 2022).

That is a context number. An IDEA needs the pitch convention: entry the
session AFTER the signal (MOC tomorrow), forward returns from that close,
declustered, with controls, eras, midterm cut and concentration. Both entry
forms are measured (MOC t+1 and MOO t+1), horizons 5/10/21, and the exact
state tonight also carries a 52-week yield high on the 10-year, so the
conjunction is printed for the record even though its n will be tiny.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_lag, load_prices,
    local_control, sign_test, summarize, wilder_atr,
)

ASOF = pd.Timestamp("2026-08-31")
px = close_panel(["^MOVE", "SPY", "^TNX"])
px = px[px.index >= "1999-01-01"]
mv = px["^MOVE"].dropna()
r1 = mv.pct_change()
r21 = mv.pct_change(21)
rank21 = r21.rolling(252, min_periods=252).apply(
    lambda w: 100.0 * (w[-1] > w[:-1]).mean(), raw=True)
state = (r1 >= 0.05) & (rank21 <= 33.3)
print(f"tonight MOVE {mv.iloc[-1]:.2f}  1d {100*r1.iloc[-1]:+.2f}%  rank21 {rank21.iloc[-1]:.1f}  "
      f"state={bool(state.iloc[-1])}  bar {mv.index[-1].date()}")


def first_in_calendar_days(mask, days=30):
    out = pd.Series(False, index=mask.index)
    last = None
    for d in mask.index[mask.fillna(False).values]:
        if last is None or (d - last).days > days:
            out.loc[d] = True
        last = d
    return out


ed = first_in_calendar_days(state, 30)
hist = ed.index[ed.values]
hist = hist[hist < ASOF]
print(f"episodes (30-cal-day novelty): {len(hist)}  {hist[0].date()}..{hist[-1].date()}")

spyf = load_prices(["SPY"])["SPY"]
spy = spyf["Close"].dropna()
atr = pd.Series(wilder_atr(spyf["High"], spyf["Low"], spyf["Close"]), index=spyf.index)
print(f"tonight SPY close {spy.iloc[-1]:.2f}  Wilder-14 ATR {atr.iloc[-1]:.4f} "
      f"({100*atr.iloc[-1]/spy.iloc[-1]:.2f}%)  bar {spy.index[-1].date()}")
pos = pd.Series(np.arange(len(spy)), index=spy.index)
opn = spyf["Open"].reindex(spy.index)


def block(name, r, s, h, lag=1):
    r = r.dropna()
    if len(r) == 0:
        print(f"  {name:<40} n=0")
        return r
    st = summarize(r.values)
    nup = int((r > 0).sum())
    allr = fwd_lag(s, h, lag).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name:<40} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(r)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(r)):.5f}  "
          f"| all {100*allr.mean():+.3f}% hit {100*(allr>0).mean():.1f}%  local {100*loc.mean():+.3f}% "
          f"hit {100*(loc>0).mean():.1f}%  | worst {st['worst_pct']:+.2f}% ({r.idxmin().date()})")
    return r


def splits(r):
    r = r.dropna()
    v = r.values
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1)) for e in era_split(r.index, v)])
    for label, m in (("pre-2018", r.index < "2018-01-01"), ("2018+", r.index >= "2018-01-01")):
        s = r[m]
        up = int((s > 0).sum())
        print(f"    {label}: {up}-{len(s)-up}  mean {100*s.mean():+.3f}%  sp={sign_test(up, len(s)):.4f}")
    print("    concentration:", cluster_note(r.index, v))
    mid = r[[d.year % 4 == 2 for d in r.index]]
    nu = int((mid > 0).sum())
    if len(mid):
        print(f"    midterm n={len(mid)} mean={100*mid.mean():+.3f}% {nu}-{len(mid)-nu} sp={sign_test(nu, len(mid)):.4f}")


print("\n=== A. MOC entry t+1 (pitch convention, lag-1 close-to-close) ===")
d = pd.DatetimeIndex(hist).intersection(spy.index)
for h in (1, 3, 5, 10, 21):
    r = fwd_lag(spy, h, 1).reindex(d)
    block(f"SPY lag1 h{h}", r, spy, h)
r10 = fwd_lag(spy, 10, 1).reindex(d)
splits(r10)

print("\n=== A2. lag-0 reproduction of the brief's number (should be 45-13) ===")
block("SPY lag0 h10", fwd_lag(spy, 10, 0).reindex(d), spy, 10, 0)
print("  (the lag-1 cell above is the tradeable one; lag-0 includes tomorrow's session, which is unknown)")

print("\n=== B. decluster the anchors at 10 and 21 sessions (h10 windows overlap otherwise) ===")
for gap in (10, 21):
    dd = declusters(d, gap, spy.index)
    block(f"SPY lag1 h10, min-gap {gap}", fwd_lag(spy, 10, 1).reindex(dd), spy, 10)

print("\n=== C. MOO entry t+1 -> MOC t+10 (open-anchored form) ===")
out = {}
for dt in d:
    i = pos[dt]
    if i + 10 < len(spy):
        out[dt] = spy.iloc[i + 10] / opn.iloc[i + 1] - 1
rmoo = pd.Series(out)
block("SPY MOO t+1 -> close t+10", rmoo, spy, 10)
print(f"  entry-session cost check (close t -> open t+1): "
      f"{100*np.mean([opn.iloc[pos[x]+1]/spy.iloc[pos[x]]-1 for x in d if pos[x]+1 < len(spy)]):+.3f}%")

print("\n=== D. the losers on the lag-1 h10 cell, named ===")
lo = r10.dropna().sort_values()
for dt, x in lo.head(8).items():
    print(f"  {dt.date()} {100*x:+.2f}%")
print(f"  losers <= -2%: {int((lo <= -0.02).sum())} of {len(lo)}")

print("\n=== E. the exact conjunction tonight: also a 10y 52-week yield high (n will be tiny) ===")
tnx = px["^TNX"].dropna()
hi252 = tnx.rolling(252, min_periods=200).max()
near = (tnx >= hi252 * 0.98).reindex(d).fillna(False)
conj = r10[near.values]
block("SPY lag1 h10, MOVE jump AND 10y within 2% of 52w high", conj, spy, 10)
print("   dates:", [(x.date().isoformat(), round(100 * y, 2)) for x, y in conj.dropna().items()])

print("\n=== F. the unconditioned parent: any MOVE +5% day, lag-1 h10, min-gap 10 ===")
par = declusters(mv.index[(r1 >= 0.05).fillna(False).values], 10, spy.index)
par = par[par < ASOF].intersection(spy.index)
block("SPY lag1 h10, any MOVE +5% day", fwd_lag(spy, 10, 1).reindex(par), spy, 10)
hi = declusters(mv.index[((r1 >= 0.05) & (rank21 > 66.6)).fillna(False).values], 10, spy.index)
hi = hi[hi < ASOF].intersection(spy.index)
block("SPY lag1 h10, MOVE +5% from TOP-third rank", fwd_lag(spy, 10, 1).reindex(hi), spy, 10)
