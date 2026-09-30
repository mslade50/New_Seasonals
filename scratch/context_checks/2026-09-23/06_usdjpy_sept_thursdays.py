"""Engine BH pass: USDJPY (JPY=X) up on 72 of 112 September Thursdays, mean +0.07%.
Is this September, or every Thursday? Is it the Tokyo-fix / Gotobi calendar, or the dollar's
September drift? Hit rate with magnitude, or hit rate alone?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["JPY=X", "DX-Y.NYB"])
s = px["JPY=X"].dropna()
idx = s.index
r = s.pct_change().dropna()
wd = r.index.weekday
rows = [summarize(r.values, "all days")]
for d, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
    rows.append(summarize(r[wd == d].values, f"all {nm}"))
sep = r.index.month == 9
rows.append(summarize(r[sep].values, "all Sept days"))
rows.append(summarize(r[sep & (wd == 3)].values, "Sept Thu"))
rows.append(summarize(r[~sep & (wd == 3)].values, "non-Sept Thu"))
for m in range(1, 13):
    rows.append(summarize(r[(r.index.month == m) & (wd == 3)].values, f"Thu month {m}"))
show(rows, "USDJPY daily % by slot")
st = r[sep & (wd == 3)]
for part in era_split(st.index, st.values):
    print(part)
up = int((st > 0).sum())
print("Sept Thu up", up, "of", len(st), "sign p vs 0.5", sign_test(up, len(st)),
      "vs all-Thu hit", round((r[wd == 3] > 0).mean(), 3), "p", sign_test(up, len(st), float((r[wd == 3] > 0).mean())))
# Late-September Thursdays (day >= 22) only, like tomorrow
late = st[st.index.day >= 22]
print("late Sept Thu:", summarize(late.values, "late"), "up", int((late > 0).sum()), "of", len(late))
# Gotobi: days of month divisible by 5 (Tokyo fix dollar demand); tomorrow is the 24th -> not gotobi
gotobi = (st.index.day % 5 == 0)
print("Sept Thu gotobi:", summarize(st[gotobi].values), "non-gotobi:", summarize(st[~gotobi].values))
print("by year up count:", st.groupby(st.index.year).apply(lambda x: f"{(x > 0).sum()}/{len(x)}").tail(12).to_dict())
