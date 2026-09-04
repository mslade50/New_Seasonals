"""Probe 2: is `date` in earnings_calendar an ANNOUNCEMENT date or a fiscal
period end? An earnings-anchored cell is worthless if the anchor is a quarter
end. Check day-of-month distribution and month-end share by era."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

e = pd.read_parquet("data/earnings_calendar.parquet")
e["date"] = pd.to_datetime(e["date"])

# month-end share by year: a fiscal-period-end stamp lands on the last calendar
# day of a month; a real announcement almost never does.
e["is_me"] = e["date"].dt.is_month_end
by_yr = e.groupby(e["date"].dt.year)["is_me"].agg(["mean", "size"])
print("share of rows landing on a calendar month END, by year:")
print((by_yr.assign(mean=lambda d: (100 * d["mean"]).round(1))
       .loc[1998:2027]).to_string())

for t in ["TJX", "ROST", "NVDA", "TGT", "WMT", "AAPL"]:
    s = e[e["ticker"] == t].sort_values("date")
    print(f"\n{t}: month-end share {100*s['is_me'].mean():.1f}%")
    print("  2010s sample:", [str(d.date()) for d in s[(s["date"] >= "2012-01-01")
                                                       & (s["date"] <= "2014-12-31")]["date"]])
    print("  recent:", [str(d.date()) for d in s[s["date"] >= "2024-01-01"]["date"]])

# where does the announcement-date era begin?
print("\nmonth-end share by year, ex-1985..1997:")
me = e.groupby(e["date"].dt.year)["is_me"].mean()
for y in range(1998, 2028):
    if y in me:
        print(f"  {y}: {100*me[y]:5.1f}%")
