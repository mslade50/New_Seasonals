"""c8 round 1: LONG TLT after a +1.0% to +1.5% session from within 4% of its
trailing-252 closing low. Entry MOC D+1 (lag=1). Watchlist 17's band, read long.
Signal 2026-09-17: TLT +1.11%, 1.33% above its low, FOMC k=+1.

1. LIVE verify
2. RECOMPUTE the ladder (watchlist 17's rung numbers are internally inconsistent)
   long side, gap 10 as in 2026-08-20's script and gap 5, h=1,2,3,5
3. battery h=1,2,3,5 on the band
4. LAG PROFILE lag 0/1/2
5. per-session increments from the entry close
6. FOMC adjacency: thrust on FOMC decision day (k=0) or k=+1 vs not
7. distance-to-low neighbours (2/3/4/6/8%)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

COST = 3.0
GAP = 10
px = panel(["TLT", "IEF", "SPY"], "TLT")
tlt = px["TLT"]
d1 = dret(tlt)
dist = tlt / tlt.rolling(252).min() - 1.0
d = px.index
L = [("TLT", 1.0)]


def band(lo, hi, low=0.04):
    m = (d1 >= lo) & (d1 < hi) & (dist <= low)
    return m.fillna(False)


cell = band(0.010, 0.015)
last = d[-1]
print(f"panel {d[0].date()}..{last.date()}  LIVE {last.date()}: TLT d1 {100*d1.iloc[-1]:+.2f}%  "
      f"above 252 low {100*dist.iloc[-1]:.2f}%  fired={bool(cell.iloc[-1])}   day-level triggers {int(cell.sum())}")

# FOMC k
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
pos, kept = anchor_positions(d, fomc, 0)
fpos = np.array(sorted(set(pos)))
k_since = np.full(len(d), 999)
j = -1
for i in range(len(d)):
    while j + 1 < len(fpos) and fpos[j + 1] <= i:
        j += 1
    if j >= 0:
        k_since[i] = i - fpos[j]
k_since = pd.Series(k_since, index=d)
print(f"  live FOMC k at signal: {k_since.iloc[-1]}")

print("\n" + "=" * 100)
print("2. LADDER, LONG TLT (episode level). Recomputed.")
rungs = {"[1.00,1.25)": band(.01, .0125), "[1.25,1.50)": band(.0125, .015), "[1.00,1.50) CELL": cell,
         ">=1.00 (parent)": band(.01, 9), ">=1.50": band(.015, 9), ">=1.75": band(.0175, 9),
         ">=2.00": band(.02, 9), "[1.50,2.00)": band(.015, .02)}
for gap in (10, 5):
    rows = []
    for lbl, m in rungs.items():
        row = {"rung": lbl}
        for h in (1, 2, 3, 5):
            s, _, _ = cellstats(px, m, L, h, "", gap)
            row[f"h{h}"] = round(s.get("mean_pct", np.nan), 3)
            row[f"rec{h}"] = s.get("rec", "")
            row[f"p{h}"] = s.get("p_coin", np.nan)
        row["n"] = s.get("n", 0)
        rows.append(row)
    print(f"\n--- gap {gap} ---  (TLT own all-days h1/2/3/5 = "
          + "/".join(f"{100*vehicle_ret(px, L, h).mean():.3f}" for h in (1, 2, 3, 5)) + ")")
    print(pd.DataFrame(rows).to_string(index=False))

for h in (1, 2, 3, 5):
    battery(px, cell, L, h, f"c8 LONG TLT [1.0,1.5) within 4% of low h={h}", COST, min_gap=GAP,
            variants={"low 2%": band(.01, .015, .02), "low 3%": band(.01, .015, .03),
                      "low 6%": band(.01, .015, .06), "low 8%": band(.01, .015, .08),
                      "band [0.75,1.5)": band(.0075, .015), "band [1.0,1.75)": band(.01, .0175)},
            event_kinds=("nfp", "cpi"))

print("\n" + "=" * 100)
print("4. LAG PROFILE (episode, gap 10)")
rows = []
for h in (1, 2, 3, 5):
    for lag in (0, 1, 2):
        s, _, _ = cellstats(px, cell, L, h, f"h={h} lag={lag}", GAP, lag, COST)
        rows.append(s)
show(rows)

print("\n" + "=" * 100)
print("5. PER-SESSION INCREMENTS from the entry close (cell episodes)")
epi = declusters(d[cell.values], GAP, d)
paths = episode_paths(px, epi, L, 6)
inc = paths.diff(axis=1)
inc[1] = paths[1]
for k in paths.columns:
    v = inc[k].dropna().values
    print(f"  session +{k}: mean {100*v.mean():+.3f}pp hit {100*(v>0).mean():.1f}%  cum {100*paths[k].mean():+.3f}%  N={len(v)}")

print("\n" + "=" * 100)
print("6. FOMC ADJACENCY (thrust session is FOMC k=0 or k=+1)")
adj = k_since.isin([0, 1])
rows = []
for h in (1, 2, 3, 5):
    for lbl, m in [("FOMC k in {0,1}", cell & adj), ("not FOMC-adjacent", cell & ~adj),
                   ("parent >=1.0 & FOMC k in {0,1}", band(.01, 9) & adj)]:
        s, _, _ = cellstats(px, m.fillna(False), L, h, f"h={h} {lbl}", GAP)
        rows.append(s)
show(rows)
s, epi, vals = cellstats(px, (cell & adj).fillna(False), L, 2, "", GAP)
print("  FOMC-adjacent episodes h=2:", [(str(x.date()), round(100 * v, 2)) for x, v in zip(epi, vals)])

print("\n" + "=" * 100)
print("7. SIGNED CONCENTRATION + by-year + midterm (cell, gap 10)")
for h in (1, 2, 3, 5):
    s, epi, vals = cellstats(px, cell, L, h, "", GAP)
    print(f"  h={h}: {s['rec']} p_coin {s['p_coin']} mean {s['mean_pct']:+.3f}% edge {s['edge_pp']:+.3f}pp")
    print("   ", signed_conc(epi, vals, f"h={h}"))
s, epi, vals = cellstats(px, cell, L, 2, "", GAP)
yrs = pd.Series(vals, index=pd.DatetimeIndex(epi).year)
print("   h=2 by year (n, sum pp):", {int(y): (int(g.size), round(100 * g.sum(), 2)) for y, g in yrs.groupby(level=0)})
mt = midterm_mask(d)
for h in (2, 5):
    for lbl, m in [("midterm", cell & mt), ("non-midterm", cell & ~mt)]:
        s, _, _ = cellstats(px, m, L, h, "", GAP)
        print(f"  {lbl} h={h}: N={s.get('n')} mean {s.get('mean_pct', np.nan):+.3f}% rec {s.get('rec')}")
