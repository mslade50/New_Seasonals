"""The engine's seasonal doy cell: ^GSPC 6-0 up in midterm years at the Aug-17 trading day of
year, sign p 0.0156. Six observations. Two questions before that can be said out loud:
  1. is it one session or a window, i.e. what does the rest of late August do in midterms
  2. how much of it is the 2000-2025 midterm sample being six specific years
Also the plain calendar question: the second half of August in midterm years, which is the
famous cell this one sits inside.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, cluster_note  # noqa

px = close_panel(["^GSPC", "SPY", "QQQ", "IWM"])
idx = px.index

# trading day of year for every session, then the anchors matching this year's next session
tdoy = pd.Series(index=idx, dtype=int)
for y, g in pd.Series(idx, index=idx).groupby(idx.year):
    tdoy.loc[g.index] = np.arange(1, len(g) + 1)

nxt_date = pd.Timestamp("2026-08-17")
this_year = idx[idx.year == 2026]
target_tdoy = int(np.searchsorted(this_year, nxt_date) + 1)
print(f"next session {nxt_date.date()} is trading day {target_tdoy} of 2026")

# anchor = the session before a matching doy, +/- 2 as the engine does, one pick per year
anchors = []
for y in range(1999, 2026):
    g = idx[idx.year == y]
    if len(g) < target_tdoy + 6:
        continue
    cand = g[max(0, target_tdoy - 3):target_tdoy + 2]
    pick = min(cand, key=lambda d: abs((d - pd.Timestamp(f"{y}-08-17")).days))
    loc = idx.get_loc(pick)
    anchors.append(idx[loc - 1])
anchors = pd.DatetimeIndex(anchors)
mid = anchors[anchors.year % 4 == 2]
print(f"anchors {len(anchors)} ({anchors[0].date()} .. {anchors[-1].date()}), midterm {list(mid.year)}")

for tkr in ["^GSPC", "SPY", "QQQ", "IWM"]:
    print("=" * 78)
    print(tkr)
    for h in (1, 5, 10, 21):
        r = fwd_ret(px[tkr], h)
        a = r.reindex(anchors).dropna()
        m = r.reindex(mid).dropna()
        au, mu = int((a > 0).sum()), int((m > 0).sum())
        print(f"  h{h:<3d} all years n {len(a):2d} {au}-{len(a)-au} mean {100*a.mean():+.2f}% "
              f"med {100*a.median():+.2f}%   |   midterm n {len(m)} {mu}-{len(m)-mu} "
              f"mean {100*m.mean():+.2f}% signp {sign_test(mu, len(m)):.4f}")
    r1 = fwd_ret(px[tkr], 1).reindex(mid).dropna()
    print("   midterm h1 by year:", ", ".join(f"{d.year}:{100*x:+.2f}" for d, x in zip(r1.index, r1.values)))

# the window this sits inside: mid-August to end-September in midterm years, one read per year
print("\n" + "=" * 78)
print("^GSPC from this anchor forward, midterm years only, cumulative")
r = px["^GSPC"]
rows = []
for d in mid:
    loc = idx.get_loc(d)
    row = {"year": d.year}
    for h in (1, 5, 10, 21, 42):
        if loc + h < len(idx):
            row[f"h{h}"] = 100 * (r.iloc[loc + h] / r.iloc[loc] - 1)
    rows.append(row)
df = pd.DataFrame(rows).set_index("year")
print(df.round(2).to_string())
print("\nmeans:", df.mean().round(2).to_dict())
print("up counts:", {c: f"{int((df[c] > 0).sum())}-{int((df[c] < 0).sum())}" for c in df.columns})

# and the all-years version of the same forward window, as the control
print("\n^GSPC same anchors, ALL years, cumulative means")
rows = []
for d in anchors:
    loc = idx.get_loc(d)
    row = {"year": d.year}
    for h in (1, 5, 10, 21, 42):
        if loc + h < len(idx):
            row[f"h{h}"] = 100 * (r.iloc[loc + h] / r.iloc[loc] - 1)
    rows.append(row)
dfa = pd.DataFrame(rows).set_index("year")
print("means:", dfa.mean().round(2).to_dict())
print("up counts:", {c: f"{int((dfa[c] > 0).sum())}-{int((dfa[c] < 0).sum())}" for c in dfa.columns})

# does the midterm h1 survive dropping its best year
r1 = fwd_ret(px["^GSPC"], 1).reindex(mid).dropna()
print("\nmidterm h1 cluster:", cluster_note(r1.index, r1.values, k=1))
print(f"  drop the best year: n {len(r1)-1} mean {100*r1.drop(r1.idxmax()).mean():+.3f}%")
print(f"  overlap with the expiry-week Monday cell: "
      f"{[str(d.date()) for d in mid]}")
