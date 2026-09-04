"""Idea check: long SPY into/through a monthly opex entered on a weak week.

Tonight's state: tomorrow (2026-08-21) is monthly opex and SPY walks in
with a 5d return in the 9.5th percentile of its trailing year. The context
brief's lag-0 cell (opex entered on 5d rank < 20, expiry bar + following
week) ran n=57, +1.22%, 68.4% up, t=3.31. A POST idea has to survive the
pitch doctrine instead: tradeable entries only (MOO on the opex day, or
MOC opex day), controls, era split, concentration, and the reference-class
kill that the tape is 8% ABOVE its 200d (if the edge lives in bear-market
rebounds, tonight is not in the reference class).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, era_split, fwd_lag, load_events, load_prices,
    local_control, pct_rank, sign_test, summarize, wilder_atr,
)

raw = load_prices(["SPY"])["SPY"]
spy = raw["Close"]
atr = pd.Series(wilder_atr(raw["High"], raw["Low"], raw["Close"]), index=raw.index)
rank5 = pct_rank(spy, 5)

print("SPY panel", spy.index[0].date(), "->", spy.index[-1].date(), "n", len(spy))
print("tonight: close %.2f | 5d rank %.1f | Wilder-14 ATR %.4f | 200d dist %+.1f%%"
      % (spy.iloc[-1], rank5.iloc[-1], atr.iloc[-1],
         (spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1) * 100))

# anchor = the session BEFORE each monthly opex (so lag-1 entry = the opex day)
opex = load_events(["opex"])["date"]
pos = pd.Series(range(len(spy)), index=spy.index)
anchors = []
for d in opex:
    p = pos.get(d)
    if p is not None and p > 0:
        anchors.append(spy.index[p - 1])
anchors = pd.DatetimeIndex(anchors)

weak = rank5.reindex(anchors) < 20
trig = anchors[weak.fillna(False)]
trig = trig[trig < spy.index[-1]]
print(f"\ntrigger: day before monthly opex with SPY 5d rank < 20 -> n={len(trig)} "
      f"(of {len(anchors)} opex anchors)")

# MOC form: buy the opex close, hold h sessions (fwd_lag, the honest default)
for h in (1, 3, 5, 10):
    f = fwd_lag(spy, h).reindex(trig).dropna()
    if not len(f):
        continue
    s = summarize(f.values)
    nup = int((f > 0).sum())
    allc = summarize(fwd_lag(spy, h).dropna().values)
    loc = summarize(fwd_lag(spy, h).reindex(local_control(spy.index, trig, 126)).dropna().values)
    print(f"  MOC-opex h{h:<2} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(f)-nup} up  "
          f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}  "
          f"| all {allc['mean_pct']:+.3f}%  local {loc['mean_pct']:+.3f}%")

# MOO form: buy the opex OPEN, exit at the close h sessions after the opex day
opn = raw["Open"]
def moo_ret(h: int) -> pd.Series:
    entry = opn.shift(-1)                       # opex-day open
    exit_ = spy.shift(-(1 + h))                 # close h sessions after opex day
    return (exit_ / entry - 1.0)

for h in (0, 3, 5):
    f = moo_ret(h).reindex(trig).dropna()
    s = summarize(f.values)
    nup = int((f > 0).sum())
    allc = summarize(moo_ret(h).dropna().values)
    print(f"  MOO-opex h{h:<2} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(f)-nup} up  "
          f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}  | all {allc['mean_pct']:+.3f}%")

f5 = fwd_lag(spy, 5).reindex(trig).dropna()
print("\nera h5 MOC:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                          round(e.get("hit", np.nan), 1)) for e in era_split(f5.index, f5.values)])
print("concentration h5 MOC:", cluster_note(f5.index, f5.values))

# reference-class kill attempt: tonight SPY is ABOVE its 200d. If the edge is
# all below-200d (bear rebound) episodes, tonight doesn't qualify.
sma200 = spy.rolling(200).mean()
above = (spy / sma200 - 1).reindex(trig) > 0
for lab, sub in (("above 200d", trig[above.fillna(False)]),
                 ("below 200d", trig[~above.fillna(True)])):
    fs = fwd_lag(spy, 5).reindex(sub).dropna()
    if not len(fs):
        print(f"  {lab}: n=0")
        continue
    s = summarize(fs.values)
    nup = int((fs > 0).sum())
    print(f"  {lab}: n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(fs)-nup} up  "
          f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(fs)):.4f}  worst={s['worst_pct']:+.2f}%")

# second kill attempt: does it survive midterm years (tonight's regime)?
mid = pd.DatetimeIndex([d for d in trig if d.year % 4 == 2])
fm = fwd_lag(spy, 5).reindex(mid).dropna()
if len(fm):
    s = summarize(fm.values)
    nup = int((fm > 0).sum())
    print(f"  midterm yrs: n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(fm)-nup} up  "
          f"sign_p={sign_test(nup, len(fm)):.4f}")
