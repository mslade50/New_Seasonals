"""C6 -- one-day ^SKEW rise >= 3% on a session the ^VIX FELL.

A CO-MOVEMENT trigger, not a level or a rank trigger. Both of ^SKEW's level
and rank forms are closed in both tails (2026-08-10, 08-12, 08-14).

Mechanism claimed: skew up while ATM vol falls = demand for tail protection
financed by selling the body = POSITIONING, not fear. Fear moves both up.

Kill tests, in order:
 1. full battery, SPY both signs, cost 3 bps; SVXY at 6 bps with the 2018
    leverage-change era SPLIT (never pooled: -1.0x pre-Feb-2018, -0.5x after)
 2. GATE ATTRIBUTION: skew-alone / vix-alone / joint. A joint that pays no
    more than the better parent is decoration; a joint that pays LESS is a
    negative interaction and by registry rule is NOT parkable.
 3. threshold ladder: skew 2/2.5/3/4/5%, vix down-at-all / >=1% / >=2%
 4. LAGGING-MARKER offset ladder k=-5..+5 around the trigger
 5. era split, midterm split, concentration, sign test, decluster 5/10/21
 6. SPY distance-from-52w-high conditioning; where does today (-1.10%) sit
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["SPY", "SVXY"]
px_map = load_prices(TK)
raw = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
raw["date"] = pd.to_datetime(raw["date"])


def series(t, col="Close"):
    g = raw[raw.ticker == t].sort_values("date").set_index("date")
    g = g[~g.index.duplicated(keep="last")]
    return g[col]


skew = series("^SKEW")
vix = series("^VIX")
spy = px_map["SPY"]["Close"]
svxy = px_map["SVXY"]["Close"]

# aligned on SPY's calendar (the trade calendar)
idx = spy.index
sk = skew.reindex(idx)
vx = vix.reindex(idx)

sk_chg = sk / sk.shift(1) - 1.0
vx_chg = vx / vx.shift(1) - 1.0

px = pd.DataFrame({"SPY": spy}).dropna()
IDX = px.index
sk_chg = sk_chg.reindex(IDX)
vx_chg = vx_chg.reindex(IDX)

print("=" * 78)
print("C6  SKEW up >= 3% AND VIX down.  Live 2026-08-28: SKEW +3.97%, VIX -0.55%")
print("=" * 78)
print(f"^SKEW coverage {skew.index.min().date()} .. {skew.index.max().date()} "
      f"({len(skew)} bars);  ^VIX {vix.index.min().date()} .. {vix.index.max().date()}")

M_SKEW = (sk_chg >= 0.03).fillna(False)
M_VIXD = (vx_chg < 0).fillna(False)
JOINT = M_SKEW & M_VIXD
print(f"\ncounts:  skew>=3% {int(M_SKEW.sum())}   vix down {int(M_VIXD.sum())}   "
      f"JOINT {int(JOINT.sum())}   (all days {len(IDX)})")
print("today's row reproduces:",
      bool(JOINT.get(pd.Timestamp('2026-08-28'), False)))

# ---------------------------------------------------------------- 1. battery
for sign, lbl in ((1.0, "LONG SPY"), (-1.0, "SHORT SPY")):
    for h in (5,):
        battery(px, JOINT, [("SPY", sign)], h,
                f"C6 {lbl} | skew>=3% & vix down", 3.0,
                variants={
                    "skew>=2%  & vixdn": ((sk_chg >= 0.02) & M_VIXD).fillna(False),
                    "skew>=2.5%& vixdn": ((sk_chg >= 0.025) & M_VIXD).fillna(False),
                    "skew>=3%  & vixdn": JOINT,
                    "skew>=4%  & vixdn": ((sk_chg >= 0.04) & M_VIXD).fillna(False),
                    "skew>=5%  & vixdn": ((sk_chg >= 0.05) & M_VIXD).fillna(False),
                    "skew>=3% & vix<=-1%": ((sk_chg >= 0.03) & (vx_chg <= -0.01)).fillna(False),
                    "skew>=3% & vix<=-2%": ((sk_chg >= 0.03) & (vx_chg <= -0.02)).fillna(False),
                }, min_gap=5)

# ------------------------------------------------- 2. GATE ATTRIBUTION
print("\n" + "=" * 78)
print("2. GATE ATTRIBUTION  (episode level, min_gap 5, LONG SPY)")
print("=" * 78)
for h in (1, 3, 5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    valid = ret.notna()
    rows = []
    for lbl, m in (("(a) skew>=3% only", M_SKEW),
                   ("(b) vix down only", M_VIXD),
                   ("(c) JOINT", JOINT),
                   ("(d) skew>=3% & vix UP", (M_SKEW & ~M_VIXD).fillna(False)),
                   ("--- all days", pd.Series(True, index=IDX))):
        s = IDX[m.reindex(IDX, fill_value=False).values & valid.values]
        e = declusters(s, 5, IDX)
        r = summarize(ret.loc[e].values, lbl)
        r["n_days"] = len(s)
        rows.append(r)
    show(rows, f"h={h} LONG SPY")
    a = rows[0].get("mean_pct", np.nan)
    b = rows[1].get("mean_pct", np.nan)
    c = rows[2].get("mean_pct", np.nan)
    print(f"  JOIN VALUE h={h}: joint {c:+.3f}% vs better parent "
          f"{max(a, b):+.3f}%  ->  {c - max(a, b):+.3f}pp")

# ---------------------------------------------------- 4. lagging-marker ladder
print("\n" + "=" * 78)
print("4. LAGGING-MARKER OFFSET LADDER (long SPY h=5, episodes min_gap 5)")
print("   registry shape to look for: monotone decay INTO the anchor")
print("=" * 78)
ret5 = vehicle_ret(px, [("SPY", 1.0)], 5, lag=1)
valid5 = ret5.notna()
trig = IDX[JOINT.reindex(IDX, fill_value=False).values]
rows = []
pos = pd.Series(range(len(IDX)), index=IDX)
for k in range(-5, 6):
    shifted = []
    for d in trig:
        p = pos.get(d)
        if p is None:
            continue
        q = p + k
        if 0 <= q < len(IDX):
            shifted.append(IDX[q])
    sd = pd.DatetimeIndex(sorted(set(shifted)))
    sd = sd[valid5.reindex(sd, fill_value=False).values]
    e = declusters(sd, 5, IDX)
    r = summarize(ret5.loc[e].values, f"offset k={k:+d}")
    rows.append(r)
show(rows, "offset ladder")

# ------------------------------------------------- 5. midterm / era / SVXY
print("\n" + "=" * 78)
print("5. MIDTERM SPLIT + YEAR TABLE (long SPY h=5 episodes)")
print("=" * 78)
epi = declusters(IDX[JOINT.reindex(IDX, fill_value=False).values & valid5.values], 5, IDX)
v = ret5.loc[epi].values
mid = np.array([(d.year % 4) == 2 for d in epi])
show([summarize(v[mid], f"midterm yrs (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm split")
yr = pd.Series(100 * v, index=epi).groupby(epi.year).agg(["count", "sum", "mean"])
print("\nyear table (pp):")
print(yr.round(2).to_string())

for mg in (5, 10, 21):
    e = declusters(IDX[JOINT.reindex(IDX, fill_value=False).values & valid5.values], mg, IDX)
    print(f"  decluster min_gap={mg}: N={len(e)}  "
          f"mean {100*ret5.loc[e].mean():+.3f}%  hit {100*(ret5.loc[e]>0).mean():.1f}%")

# ------------------------------------------------- SVXY, eras NOT pooled
print("\n" + "=" * 78)
print("SVXY expression, ERAS SPLIT (SVXY was -1.0x until 2018-02-05, -0.5x after)")
print("=" * 78)
pxs = pd.DataFrame({"SVXY": svxy}).dropna()
IS = pxs.index
for h in (3, 5, 10):
    r = vehicle_ret(pxs, [("SVXY", 1.0)], h, lag=1)
    vd = r.notna()
    s = IS[JOINT.reindex(IS, fill_value=False).values & vd.values]
    for lo, hi, lbl in ((pd.Timestamp("2011-01-01"), pd.Timestamp("2018-02-05"), "SVXY -1.0x era"),
                        (pd.Timestamp("2018-02-06"), pd.Timestamp("2030-01-01"), "SVXY -0.5x era")):
        ss = s[(s >= lo) & (s <= hi)]
        e = declusters(ss, 5, IS)
        base = IS[(IS >= lo) & (IS <= hi) & vd.values]
        rr = summarize(r.loc[e].values, f"h={h} {lbl} (N={len(e)})")
        rr["ctl_all_days"] = round(100 * r.loc[base].mean(), 3) if len(base) else np.nan
        show([rr], "")

# ------------------------------------- 6. SPY distance-from-52w-high condition
print("\n" + "=" * 78)
print("6. CONDITION ON SPY DISTANCE FROM 52w HIGH (today: -1.10%)")
print("=" * 78)
hi52 = rolling_on_valid(spy, lambda x: x.rolling(252).max())
dist = (spy / hi52 - 1.0) * 100.0  # negative = below high
dist = dist.reindex(IDX)
trig_all = IDX[JOINT.reindex(IDX, fill_value=False).values & valid5.values]
d_at = dist.loc[trig_all]
print(f"trigger-day distance distribution: min {d_at.min():.2f}%  "
      f"med {d_at.median():.2f}%  max {d_at.max():.2f}%")
buckets = [(-100, -5, "> 5% below high"), (-5, -2, "2-5% below"),
           (-2, 0.001, "within 2% of high")]
rows = []
for lo, hi, lbl in buckets:
    m = (d_at > lo) & (d_at <= hi)
    sel = trig_all[m.values]
    e = declusters(sel, 5, IDX)
    r = summarize(ret5.loc[e].values, f"{lbl} (days {int(m.sum())})")
    rows.append(r)
show(rows, "long SPY h=5 by distance bucket")
print("today's bucket = 'within 2% of high' (-1.10%)")
