"""Stat verifications (2026-09-16) for tonight's queue.
1. ^TNX closed at/above 5.0 tonight: last prior close >= 5.0 (the brief says 2007-07-19).
2. Quad witching (tomorrow+1, 2026-09-18): S&P on the day, and the Monday after, vs
   ordinary Fridays / Mondays; September quad witching alone.
3. VIX the session after a decision-day VIX rise with the S&P down < 1% (the brief's cell),
   reproduced, with the other-Wednesday control.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-16")
px = load_prices(["^TNX", "^GSPC", "^VIX", "SPY"])
nyse = px["SPY"]["Close"].dropna().index
nyse = nyse[nyse <= ASOF]
C = {t: px[t]["Close"].astype(float).reindex(nyse) for t in px}


def rec(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def line(label, v, ctrl=None):
    v = v.dropna()
    up, dn, n = rec(v.values)
    s = summarize(v.values)
    extra = ""
    if ctrl is not None:
        cv = ctrl.dropna()
        cu, cd, cn = rec(cv.values)
        extra = f" | ctrl n={cn} {cu}-{cd} mean {100*cv.mean():+.3f}% med {100*cv.median():+.3f}%"
    print(f"  {label}: n={n} {up}-{dn} mean {s['mean_pct']:+.3f}% med {s['median_pct']:+.3f}% "
          f"signp_up {sign_test(up, n):.4f} signp_dn {sign_test(dn, n):.4f} worst {s['worst_pct']:+.2f}% best {s['best_pct']:+.2f}%{extra}")


# 1. ten-year
tnx = px["^TNX"]["Close"].astype(float)
tnx = tnx[tnx.index <= ASOF]
print("TNX tonight:", tnx.index[-1].date(), round(tnx.iloc[-1], 3))
prior = tnx[(tnx >= 5.0) & (tnx.index < tnx.index[-1])]
print("last prior close >= 5.0:", prior.index[-1].date() if len(prior) else None, round(prior.iloc[-1], 3) if len(prior) else None)
print("closes >= 5.0 since 2007-07-19 (before tonight):", int(((tnx >= 5.0) & (tnx.index > "2007-07-19") & (tnx.index < tnx.index[-1])).sum()))
print("sessions since:", int(((tnx.index > prior.index[-1]) & (tnx.index <= tnx.index[-1])).sum()) if len(prior) else None)

# 2. quad witching
spx = C["^GSPC"]
r1 = spx / spx.shift(1) - 1
qw = load_events(["quad_witching"])
qwd = pd.DatetimeIndex([d for d in pd.to_datetime(qw["date"].unique()) if d in nyse and d < ASOF])
print("\nquad witching days:", len(qwd), qwd[0].date(), "->", qwd[-1].date())
fri = nyse[(nyse.weekday == 4) & (~nyse.isin(qwd))]
line("S&P on quad witching day", r1.reindex(qwd), r1.reindex(fri))
sep = qwd[qwd.month == 9]
line("September quad witching day", r1.reindex(sep), r1.reindex(fri[fri.month == 9]))
# the Monday after
nxt = pd.DatetimeIndex([nyse[nyse.searchsorted(d) + 1] for d in qwd if nyse.searchsorted(d) + 1 < len(nyse)])
mon = nyse[(nyse.weekday == 0) & (~nyse.isin(nxt))]
line("session after quad witching", r1.reindex(nxt), r1.reindex(mon))
nxt_sep = pd.DatetimeIndex([nyse[nyse.searchsorted(d) + 1] for d in sep if nyse.searchsorted(d) + 1 < len(nyse)])
line("session after September quad witching", r1.reindex(nxt_sep), r1.reindex(mon[mon.month == 9]))
# week after (5 sessions from qw close)
r5 = spx.shift(-5) / spx - 1
line("5 sessions after quad witching close", r5.reindex(qwd), r5.reindex(fri))
line("5 sessions after September quad witching close", r5.reindex(sep), r5.reindex(fri[fri.month == 9]))
v = r5.reindex(sep).dropna()
print("   Sept episodes (5d after):", [(str(d.date()), round(100 * x, 2)) for d, x in v.items()])
v = r1.reindex(nxt_sep).dropna()
print("   Sept episodes (next session):", [(str(d.date()), round(100 * x, 2)) for d, x in v.items()])

# 3. VIX after a decision-day rise, S&P down < 1%
vix = C["^VIX"]
vr1 = vix / vix.shift(1) - 1
vfwd = vix.shift(-1) / vix - 1
ev = load_events(["fomc_decision"])
dec = pd.DatetimeIndex([d for d in pd.to_datetime(ev["date"].unique()) if d in nyse and d < ASOF])
sel = dec[((vr1.reindex(dec) > 0) & (r1.reindex(dec) < 0) & (r1.reindex(dec) > -0.01)).values]
wed = nyse[(nyse.weekday == 2) & (~nyse.isin(dec))]
wsel = wed[((vr1.reindex(wed) > 0) & (r1.reindex(wed) < 0) & (r1.reindex(wed) > -0.01)).values]
print("\nVIX next session after decision-day VIX up & S&P in (-1%, 0):")
line("decision days", vfwd.reindex(sel), vfwd.reindex(wsel))
line("  2018+", vfwd.reindex(sel[sel >= "2018-01-01"]))
print("tonight: VIX %.2f (%+.2f%%), S&P %+.2f%%" % (vix.iloc[-1], 100 * vr1.iloc[-1], 100 * r1.iloc[-1]))
