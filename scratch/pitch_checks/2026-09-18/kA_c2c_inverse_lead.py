"""c2 by-product: form C (one-day ^VIX crush >= 10% on opex-3..opex-1) was
the most WRONG-signed form for the short (h=2 short SPY -0.535% on 40, 11-29,
t -3.16). Is the INVERSE (long SPY from the opex close) a real lead or a
cell? Offset ladder with the condition relocated to anchor-3..anchor-1,
non-opex placebo, era / midterm / Sept, signed concentration, neighbours.
Reported as a LEAD only; nothing here is the pitched c2 direction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from kA_common import *  # noqa
import numpy as np
import pandas as pd

px = build_panel()
cal = px.index
vix = px["^VIX"]
v1 = vix / vix.shift(1) - 1
opex = pd.DatetimeIndex(sorted(set(load_events(["opex"])["date"]) & set(cal)))
opex = opex[opex < pd.Timestamp("2026-09-18")]
is_opex = pd.Series(cal.isin(opex), index=cal)


def long_spy(h):
    return vehicle_ret(px, [("SPY", 1.0)], h, lag=0)


def crush_prior(thr, win=3):
    return v1.shift(1).rolling(win).min() <= thr


for h in (1, 2, 3, 5):
    r = long_spy(h)
    rows = []
    for lbl, m in [("C opex & 1d crush<=-10 in prior3", is_opex & crush_prior(-0.10)),
                   ("C opex & crush<=-12", is_opex & crush_prior(-0.12)),
                   ("C opex & crush<=-8", is_opex & crush_prior(-0.08)),
                   ("C opex & crush in prior2", is_opex & crush_prior(-0.10, 2)),
                   ("C opex & crush in prior5", is_opex & crush_prior(-0.10, 5)),
                   ("CTRL all opex", is_opex),
                   ("PLACEBO non-opex crush prior3", ~is_opex & crush_prior(-0.10)),
                   ("CTRL all days", pd.Series(True, index=cal))]:
        d = cal[(m & r.notna()).values]
        if "PLACEBO" in lbl or "all days" in lbl:
            d = declusters(d, h, cal)
        rows.append(rec_row(r.loc[d].values, lbl, 3.0))
    show(rows, f"LONG SPY from the opex close, h={h}")

print("\n=== offset ladder (anchor opex+k, crush on anchor-3..anchor-1), long SPY ===")
cp = crush_prior(-0.10)
for h in (2, 3, 5):
    r = long_spy(h)
    rows = []
    for k in range(-5, 6):
        p = cal.get_indexer(opex) + k
        p = p[(p >= 0) & (p < len(cal))]
        a = cal[p]
        a = a[cp.reindex(a).fillna(False).values]
        v = r.reindex(a).dropna().values
        w = int((v > 0).sum())
        rows.append({"k": k, "n": len(v), "mean_pct": 100 * v.mean(), "hit": 100 * (v > 0).mean(),
                     "rec": f"{w}-{len(v)-w}"})
    df = pd.DataFrame(rows)
    df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
    print(f"h={h}")
    print(df.round(3).to_string(index=False))
    print(f"  TRUE k=0 ranks {int(df.loc[df.k == 0, 'rank'].iloc[0])} of {len(df)}")

print("\n=== C long SPY h=2: era / midterm / Sept / concentration ===")
r = long_spy(2)
d = cal[(is_opex & cp & r.notna()).values]
v = r.loc[d]
show([rec_row(v[v.index < "2018-01-01"].values, "pre-2018", 3.0),
      rec_row(v[v.index >= "2018-01-01"].values, "2018+", 3.0),
      rec_row(v[v.index.year % 4 == 2].values, "midterm", 3.0),
      rec_row(v[v.index.year % 4 != 2].values, "non-midterm", 3.0),
      rec_row(v[v.index.month == 9].values, "September", 3.0),
      rec_row(v[v.index.month != 9].values, "ex-September", 3.0),
      rec_row(v[v.index.month.isin([3, 6, 9, 12])].values, "quad months", 3.0)])
print("  ", signed_concentration(v.index, v.values))
for x, y in v.items():
    print(f"   {x.date()} min1d {100*v1.shift(1).rolling(3).min()[x]:+.1f}%  long SPY h2 {100*y:+.2f}%")
