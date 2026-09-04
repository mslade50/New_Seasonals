"""September calendar cells for the stat post + reply ammunition.

  a. September as a month, SPY, by cycle year (midterm vs the rest), plus the
     "worst month" folklore against the other eleven months.
  b. The last session of August (Monday 08-31), own return, vs other months'
     last sessions and vs all days.
  c. The first session of September (Tue 09-01), own return.
  d. The week before Labor Day (Mon-Fri ending the Friday before the first
     Monday of September), SPY and IWM, vs random 5-session windows.
  e. September NFP day (Fri 09-04) own return, all years and midterm.
All lag 0 own-session returns unless stated: these are context, not entries.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_events, load_prices, sign_test, summarize  # noqa: E402

px = load_prices(["SPY", "IWM", "QQQ", "^VIX"])
spy = px["SPY"]["Close"].dropna()
iwm = px["IWM"]["Close"].dropna()
print(f"SPY history {spy.index[0].date()} .. {spy.index[-1].date()}")


def fmt(lab: str, r: pd.Series) -> None:
    st = summarize(r.values)
    nu = int((r > 0).sum())
    print(f"  {lab:<40} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nu}-{len(r)-nu} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sign_p={sign_test(nu, len(r)):.4f}  "
          f"worst={st['worst_pct']:+.2f}% ({r.idxmin().date() if len(r) else ''})")


# a. monthly returns
m = spy.resample("ME").last()
mr = m.pct_change().dropna()
mr = mr[mr.index.year < 2026] if False else mr
mr = mr[:-1] if mr.index[-1].month == 8 and mr.index[-1].year == 2026 else mr  # August 2026 still open
print("\n=== a. SPY monthly returns ===")
sep = mr[mr.index.month == 9]
fmt("september, all years", sep)
fmt("other 11 months", mr[mr.index.month != 9])
fmt("september, midterm years", sep[sep.index.year % 4 == 2])
fmt("september, non-midterm", sep[sep.index.year % 4 != 2])
print("   september by year:", [(d.year, round(100 * x, 1)) for d, x in sep.items()])
rank = mr.groupby(mr.index.month).mean().rank()
print("   month rank by mean (1=worst):", {k: int(v) for k, v in rank.items()})
print("   month means:", {k: round(100 * v, 2) for k, v in mr.groupby(mr.index.month).mean().items()})
sep_iwm = iwm.resample("ME").last().pct_change().dropna()
sep_iwm = sep_iwm[sep_iwm.index.month == 9]
fmt("IWM september", sep_iwm)
fmt("IWM september midterm", sep_iwm[sep_iwm.index.year % 4 == 2])

# b/c. last session of August, first of September
d1 = spy.pct_change()
ym = spy.index.to_period("M")
pos = pd.Series(np.arange(len(spy)), index=spy.index)
last_pos = pos.groupby(ym).transform("max")
first_pos = pos.groupby(ym).transform("min")
is_last = (pos == last_pos) & (ym != ym[-1])
is_first = pos == first_pos
print("\n=== b. last session of the month, own return (lag 0) ===")
fmt("last session, all months", d1[is_last])
fmt("last session of AUGUST", d1[is_last & (spy.index.month == 8)])
fmt("last of august, midterm", d1[is_last & (spy.index.month == 8) & (spy.index.year % 4 == 2)])
fmt("all days", d1.dropna())
print("   last of august by year:", [(d.year, round(100 * x, 2)) for d, x in d1[is_last & (spy.index.month == 8)].items()])
print("\n=== c. first session of the month, own return ===")
fmt("first session, all months", d1[is_first])
fmt("first session of SEPTEMBER", d1[is_first & (spy.index.month == 9)])
print("   first of september by year:", [(d.year, round(100 * x, 2)) for d, x in d1[is_first & (spy.index.month == 9)].items()])
# the 3-session TOM window own return (lag 0, ME-1 close -> day 3 close) for comparison
tom = (spy.shift(-3) / spy - 1)[is_last]
fmt("TOM lag0: ME-1 close -> day3 close, all", tom.dropna())
fmt("TOM lag0, august -> september", tom[tom.index.month == 8].dropna())

# d. week before Labor Day
print("\n=== d. week before Labor Day (5 sessions ending the Friday before) ===")
rows_spy, rows_iwm = {}, {}
for yr in range(int(spy.index[0].year), 2026):
    sept = pd.date_range(f"{yr}-09-01", f"{yr}-09-07")
    labor = [d for d in sept if d.weekday() == 0][0]
    end = labor - pd.Timedelta(days=3)  # the Friday before
    # find the last session <= end and the session 5 earlier
    for src, out in ((spy, rows_spy), (iwm, rows_iwm)):
        sub = src[src.index <= end]
        if len(sub) < 7:
            continue
        p = len(sub) - 1
        out[sub.index[p]] = sub.iloc[p] / sub.iloc[p - 5] - 1
r_spy = pd.Series(rows_spy)
r_iwm = pd.Series(rows_iwm)
fmt("SPY week before labor day", r_spy)
fmt("SPY same, midterm", r_spy[r_spy.index.year % 4 == 2])
fmt("SPY any 5 sessions", (spy / spy.shift(5) - 1).dropna())
fmt("IWM week before labor day", r_iwm)
fmt("IWM any 5 sessions", (iwm / iwm.shift(5) - 1).dropna())
print("   SPY by year:", [(d.year, round(100 * x, 2)) for d, x in r_spy.items()])

# e. September NFP day
print("\n=== e. NFP day own return ===")
nfp = load_events(["nfp"])["date"]
nfp = nfp[nfp.isin(spy.index)]
r_nfp = d1.reindex(nfp).dropna()
fmt("all NFP days", r_nfp)
r_sep = r_nfp[r_nfp.index.month == 9]
fmt("september NFP", r_sep)
fmt("september NFP, midterm", r_sep[r_sep.index.year % 4 == 2])
print("   sept NFP by year:", [(d.year, round(100 * x, 2)) for d, x in r_sep.items()])
vix = px["^VIX"]["Close"].pct_change()
fmt("VIX on september NFP", vix.reindex(r_sep.index).dropna())
