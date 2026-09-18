"""CPI landing on a FRIDAY (2026-09-11), with PPI the day before it.

Two questions:
  1. How rare is a Friday CPI, and does the print session behave differently?
  2. How rare is PPI immediately preceding CPI (Thu then Fri)? Usually PPI follows.
Anchor convention: the session k td before the event, so for the Fri CPI the k3
anchor is TODAY and h=3 is the print itself.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, load_events, fwd_ret, anchor_positions,
                       declusters, summarize, era_split, sign_test,
                       cluster_note, show)

px = close_panel(["^GSPC", "SPY", "^VIX", "TLT", "GC=F", "^TNX", "IEF"])
ev = load_events(["cpi", "ppi"])
cpi = ev[ev["event"] == "cpi"].copy()
ppi = ev[ev["event"] == "ppi"].copy()
cpi = cpi[cpi["date"] <= "2026-09-11"]
cpi["dow"] = cpi["date"].dt.day_name()
print("CPI prints by weekday, 2000-2026:")
print(cpi["dow"].value_counts().to_string())
fri = cpi[cpi["dow"] == "Friday"]
print(f"\nFriday CPIs: {len(fri)}")
print(fri[["date"]].assign(d=fri["date"].dt.date)["d"].tolist())

# PPI the session immediately before CPI
pset = set(ppi["date"])
bdays_before = []
for d in cpi["date"]:
    prev = d - pd.tseries.offsets.BDay(1)
    if prev in pset:
        bdays_before.append(d)
print(f"\nCPI prints with PPI on the immediately preceding business day: "
      f"{len(bdays_before)} of {len(cpi)}")
print("most recent 8:", [str(x.date()) for x in bdays_before[-8:]])

# Both at once: Friday CPI with PPI on the Thursday
both = [d for d in bdays_before if d.day_name() == "Friday"]
print(f"\nFriday CPI with Thursday PPI: {len(both)} -> {[str(x.date()) for x in both]}")

print("\n" + "=" * 70)
print("The CPI SESSION itself: Friday CPIs vs all other CPIs")
print("=" * 70)
idx = px.index
for sub in ["^GSPC", "^VIX", "TLT", "GC=F"]:
    r1 = px[sub].pct_change()
    rows = []
    for lab, dd in [("Friday CPI", fri["date"]), ("non-Friday CPI",
                    cpi[cpi["dow"] != "Friday"]["date"]),
                    ("all sessions", None)]:
        if dd is None:
            v = r1.dropna().values
        else:
            d = pd.DatetimeIndex(dd).intersection(r1.dropna().index)
            v = r1.loc[d].values
        s = summarize(v, lab)
        up = int((v > 0).sum())
        s["record"] = f"{up}-{len(v)-up}"
        s["sign_p"] = round(sign_test(up, len(v)), 4)
        rows.append(s)
    show(rows, f"{sub} on the CPI print session")

print("\n" + "=" * 70)
print("The k3 run-up cell (today's anchor), split by CPI weekday")
print("=" * 70)
for sub in ["^GSPC", "^VIX"]:
    rows = []
    for lab, dd in [("Friday CPI", fri["date"]),
                    ("non-Friday CPI", cpi[cpi["dow"] != "Friday"]["date"])]:
        pos, kept = anchor_positions(px.index, pd.DatetimeIndex(dd), offset=-3)
        anchors = px.index[pos]          # kept is the EVENT date; pos is the anchor
        for h in (1, 3):
            f = fwd_ret(px[sub], h)
            d = pd.DatetimeIndex(anchors).intersection(f.dropna().index)
            v = f.loc[d].values
            s = summarize(v, f"{lab} h={h}")
            if s["n"]:
                up = int((v > 0).sum())
                s["record"] = f"{up}-{s['n']-up}"
                s["sign_p"] = round(sign_test(up, s["n"]), 4)
            rows.append(s)
    show(rows, f"{sub} from the k3 anchor")
