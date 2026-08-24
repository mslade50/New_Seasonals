"""C10 round 2 -- episode structure, era concentration and the later-day cell.

a3 established the kills. This closes the two remaining ways the candidate
could come back: (i) "the freshness leg only fails TODAY, so park it", which
needs the later-day cell measured on the C10 rung rather than assumed from
W5's TLT version, and (ii) "the rung is fine, 2022 just dominates", which
needs the drop-2022 and era numbers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 230)
TK = ["TLT", "IEF", "LQD", "AGG"]
raw = load_prices(TK)
idx = None
for t in TK:
    i = raw[t]["Close"].dropna().index
    idx = i if idx is None else idx.intersection(i)
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx) for t in TK}).dropna()
idx = px.index
dist = pd.DataFrame({t: (px[t] / rolling_on_valid(px[t], lambda x: x.rolling(252).min()) - 1) * 100
                     for t in TK})

M = (dist["IEF"] <= 1.0) & (dist["LQD"] <= 1.0)
trig = idx[M.values]
first = declusters(trig, 10, idx)
later = trig.difference(first)
print("C10 rung (IEF<=1 & LQD<=1): %d trigger days, %d episode-first, %d later"
      % (len(trig), len(first), len(later)))

for veh, cost in (("IEF", 3.0), ("TLT", 3.0)):
    r = fwd_lag(px[veh], 1, 1)
    base = r.dropna()
    rows = []
    for lbl, d in (("ALL trigger days", trig), ("EPISODE-FIRST", first),
                   ("LATER days in episode", later)):
        d = d.intersection(base.index)
        s = summarize(r.loc[d].values, f"{veh}: {lbl}")
        s["excess_pp"] = round(s["mean_pct"] - 100 * base.mean(), 4)
        s["x_cost"] = round(100 * s["mean_pct"] / cost, 1)
        rows.append(s)
    show(rows, f"freshness attribution on the C10 rung, long {veh} h=1")

print("\nEPISODE-FIRST days by regime year:")
r_i = fwd_lag(px["IEF"], 1, 1)
r_t = fwd_lag(px["TLT"], 1, 1)
f = first.intersection(r_i.dropna().index)
by = pd.DataFrame({"IEF_pct": 100 * r_i.loc[f].values,
                   "TLT_pct": 100 * r_t.loc[f].values}, index=f)
print(by.groupby(by.index.year).agg(["count", "mean"]).round(3).to_string())
print("\n  2022 is %d of %d episodes (%.0f%%)"
      % ((f.year == 2022).sum(), len(f), 100 * (f.year == 2022).mean()))
ex22 = f[f.year != 2022]
bi = r_i.dropna().mean()
bt = r_t.dropna().mean()
show([summarize(r_i.loc[f].values, "IEF fresh, all episodes"),
      summarize(r_i.loc[ex22].values, "IEF fresh, DROP 2022"),
      summarize(r_t.loc[f].values, "TLT fresh, all episodes"),
      summarize(r_t.loc[ex22].values, "TLT fresh, DROP 2022")],
     "drop the dominant regime year")
print("  IEF fresh ex-2022 excess %+.4fpp = %.1fx the 3 bps round trip"
      % (100 * (r_i.loc[ex22].mean() - bi), 100 * 100 * (r_i.loc[ex22].mean() - bi) / 3.0))
print("  TLT fresh ex-2022 excess %+.4fpp = %.1fx"
      % (100 * (r_t.loc[ex22].mean() - bt), 100 * 100 * (r_t.loc[ex22].mean() - bt) / 3.0))

# the live episode
pos = pd.Series(range(len(idx)), index=idx)
recent = trig[trig >= pd.Timestamp("2026-07-01")]
print("\nthe LIVE episode: C10-rung trigger days since 2026-07-01:")
print("  " + ", ".join(str(d.date()) for d in recent))
print("  most recent episode-first day: %s (%d sessions before the 2026-08-21 bar)"
      % (first[-1].date(), pos[idx[-1]] - pos[first[-1]]))
print("  W5's freshness rule needs >= 10 sessions since the previous TRIGGER day; "
      "the previous trigger day is %s, %d session(s) back."
      % (trig[trig < idx[-1]][-1].date(),
         pos[idx[-1]] - pos[trig[trig < idx[-1]][-1]]))

# era split on the parent leg that C10 claims is the anchor
print("\nthe IEF leg alone, by era (C10's claimed anchor):")
mi = dist["IEF"] <= 1.0
di = idx[mi.values & r_i.notna().values]
show([summarize(r_i.loc[di].values, "IEF<=1% -> long IEF, all"),
      summarize(r_i.loc[di[di.year < 2015]].values, "  pre-2015"),
      summarize(r_i.loc[di[di.year >= 2015]].values, "  2015+"),
      summarize(r_i.loc[di[di.year >= 2022]].values, "  2022+"),
      summarize(r_i.dropna().values, "CTRL-b all days")])
