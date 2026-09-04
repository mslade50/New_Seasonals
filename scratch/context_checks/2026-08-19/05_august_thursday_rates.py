"""Two BH-passing cells point the same way: TLT up on 68 of 107 August Thursdays
and the 10y yield down on 74 of 117, with the Aug-20 day-of-year cell agreeing.
Both carry era_stable=False from the engine, so test the era split first, then
ask whether the cross (August AND Thursday) is doing any work."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, era_split, fwd_ret, sign_test, summarize,
)

px = close_panel(["TLT", "IEF", "^TNX", "SPY"])
subs = {"TLT": px["TLT"].dropna(), "IEF": px["IEF"].dropna(),
        "^TNX": px["^TNX"].dropna()}


def cell(s, dates, lab, quiet=False):
    f = fwd_ret(s, 1).reindex(pd.DatetimeIndex(dates)).dropna()
    if len(f) == 0:
        return None
    st = summarize(f.values)
    nup = int((f > 0).sum())
    if not quiet:
        print(f"  {lab:<40} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  "
              f"{nup}-{len(f)-nup} up  t={st['t']:+.2f}  "
              f"up_p={sign_test(nup, len(f)):.4f}  dn_p={sign_test(len(f)-nup, len(f)):.4f}")
    return f


for name, s in subs.items():
    idx = s.index[:-1]
    print(f"\n=== {name}  ({s.index[0].date()} -> {s.index[-1].date()}) ===")
    # the anchor is the Wednesday, so h1 is the Thursday session
    wed = idx[idx.weekday == 2]
    aug_wed = wed[wed.month == 8]
    f = cell(s, aug_wed, "August Wednesdays -> Thursday session")
    cell(s, wed[wed.month != 8], "Wednesdays in other months")
    cell(s, idx[(idx.weekday != 2) & (idx.month == 8)], "August, other weekdays")
    cell(s, idx, "all sessions")
    if f is not None and len(f) >= 10:
        eras = era_split(f.index, f.values)
        print("    era:", [(e["label"], e["n"], round(e["mean_pct"], 3),
                            f"{int((f[f.index < pd.Timestamp('2018-01-01')] > 0).sum() if e['label'].startswith('pre') else (f[f.index >= pd.Timestamp('2018-01-01')] > 0).sum())}up")
                           for e in eras])
        for lab, m in (("pre-2018", f.index < pd.Timestamp("2018-01-01")),
                       ("2018+", f.index >= pd.Timestamp("2018-01-01"))):
            v = f[m]
            if len(v) == 0:
                continue
            nup = int((v > 0).sum())
            st = summarize(v.values)
            print(f"    {lab:<9} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
                  f"{nup}-{len(v)-nup} up  t={st['t']:+.2f}  "
                  f"up_p={sign_test(nup, len(v)):.4f}")
        print("   ", cluster_note(f.index, f.values))
        byyr = {}
        for d, v in f.items():
            byyr.setdefault(d.year, []).append(v)
        pos = sum(1 for y in byyr if np.mean(byyr[y]) > 0)
        print(f"    positive in {pos} of {len(byyr)} Augusts")

print("\n=== the day-of-year corroboration: sessions near Aug 20 ===")
for name, s in subs.items():
    idx = s.index[:-1]
    near = idx[(idx.month == 8) & (idx.day >= 18) & (idx.day <= 22)]
    cell(s, near, f"{name}: Aug 18-22 anchors -> next session")

print("\n=== how much of the August-Thursday tilt is just late August? ===")
for name, s in subs.items():
    idx = s.index[:-1]
    wed = idx[(idx.weekday == 2) & (idx.month == 8)]
    cell(s, wed[wed.day <= 15], f"{name}: Aug Wed, first half")
    cell(s, wed[wed.day > 15], f"{name}: Aug Wed, second half")

print("\n=== equity leg on the same anchors, for contrast ===")
spy = px["SPY"].dropna()
idx = spy.index[:-1]
wed = idx[idx.weekday == 2]
cell(spy, wed[wed.month == 8], "SPY: August Wednesdays -> Thursday")
cell(spy, idx, "SPY: all sessions")
