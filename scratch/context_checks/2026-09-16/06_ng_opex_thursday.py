"""E:opex|NG=F|k2 cleared BH: natgas fell 182 of 312 sessions two before monthly opex, -0.40% (published 08-19).
The k2 anchor is always a Wednesday, so h1 is always the third Thursday: the weekly EIA storage report day. Is this
a Thursday / storage-day effect, a mid-month effect, or genuinely opex-adjacent? Controls: all Thursdays, the
other Thursdays of the month, the third Wednesday and Friday. Era, September-only, roll-seam screen (gap vs
intraday) on the losers."""
from ctx_common import *

px = load_prices(['NG=F', 'SPY'])
ng = px['NG=F'].dropna(subset=['Close'])
c = ng['Close']
r = c / c.shift(1) - 1
ev = load_events(['opex'])
op = pd.DatetimeIndex(ev['date'])
idx = c.index
pos = pd.Series(range(len(idx)), index=idx)
third_thu = []
for o in op:
    if o in pos.index and pos[o] >= 1 and o < TODAY:
        third_thu.append(idx[pos[o] - 1])
third_thu = pd.DatetimeIndex(third_thu)
third_thu = third_thu[third_thu.weekday == 3]
past = idx[(idx < TODAY) & (idx >= '2000-01-01')]
thu = past[past.weekday == 3]
other_thu = thu[~thu.isin(third_thu)]
line("NG=F third Thursday (day before opex)", r.reindex(third_thu), third_thu, show_era=True)
line("NG=F other Thursdays", r.reindex(other_thu), other_thu, show_era=True)
line("NG=F all sessions", r.reindex(past))
wed = pd.DatetimeIndex([idx[pos[x] - 1] for x in third_thu if pos[x] >= 1])
fri = pd.DatetimeIndex([idx[pos[x] + 1] for x in third_thu if pos[x] + 1 < len(idx)])
line("NG=F the Wednesday before", r.reindex(wed))
line("NG=F opex Friday", r.reindex(fri))
line("third Thursday, September only", r.reindex(third_thu[third_thu.month == 9]), third_thu[third_thu.month == 9])
line("third Thursday 2018+", r.reindex(third_thu[third_thu >= '2018-01-01']))
line("third Thursday 2022+", r.reindex(third_thu[third_thu >= '2022-01-01']))
gap = ng['Open'] / c.shift(1) - 1
g3 = gap.reindex(third_thu)
print("third-Thursday sessions with |gap| > 3%:", int((g3.abs() > 0.03).sum()))
clean = third_thu[(g3.abs() <= 0.03).values]
line("third Thursday ex |gap|>3%", r.reindex(clean), clean, show_era=True)
by_year = r.reindex(third_thu).groupby(third_thu.year).mean() * 100
print("per-year mean%:", by_year.round(2).to_dict())
