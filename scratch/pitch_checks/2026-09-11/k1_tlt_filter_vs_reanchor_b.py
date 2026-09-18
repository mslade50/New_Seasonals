"""A1 ROUND 2, part 3 -- decompose the join's value into FILTERING and
RE-ANCHORING, which is the whole verdict.

The parent (TLT <= 0.5% above its trailing-252 low, first trigger in >= 10 td)
has 17 anchors, 16 with a resolvable h=1 lag=1 return, mean -0.229%.
The cell (add IEF <= 1.0% and LQD <= 1.0%) has 11 anchors, 10 resolvable,
mean +0.470%.

Two things happened on the way from one to the other:
  FILTERING    -- some parent anchors have no cell counterpart at all
  RE-ANCHORING -- the rest survive but on a LATER date, because tightening the
                  price rung delays the first qualifying session

This splits the +0.699pp gap between the two. Same-episode matching: a parent
anchor is matched to the first cell anchor within 21 td forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

raw = load_prices(["TLT", "IEF", "LQD"])
IDX = raw["TLT"].index
POS = pd.Series(range(len(IDX)), index=IDX)
R1 = fwd_lag(raw["TLT"]["Close"].reindex(IDX), 1, 1)


def above_low(t, n=252):
    s = raw[t]["Close"]
    return ((s / s.rolling(n).min() - 1.0) * 100).reindex(IDX)


TL, IE, LQ = above_low("TLT"), above_low("IEF"), above_low("LQD")


def first_in(mask, gap=10):
    days = IDX[mask.reindex(IDX, fill_value=False).values]
    keep, last = [], -10 ** 9
    for d in days:
        p = int(POS[d])
        if p - last >= gap:
            keep.append(d)
        last = p
    return pd.DatetimeIndex(keep)


a_par = first_in((TL <= 0.5).fillna(False))
a_cell = first_in(((TL <= 0.5) & (IE <= 1.0) & (LQ <= 1.0)).fillna(False))

matched, unmatched = [], []
for pa in a_par:
    cands = [c for c in a_cell if 0 <= int(POS[c]) - int(POS[pa]) <= 21]
    (matched if cands else unmatched).append((pa, cands[0] if cands else None))

par_all = R1.loc[a_par].dropna()
unm = R1.loc[[p for p, _ in unmatched]].dropna()
mat_at_parent = R1.loc[[p for p, _ in matched]].dropna()
mat_at_cell = R1.loc[[c for _, c in matched]].dropna()

print("=" * 78)
print("DECOMPOSITION OF THE JOIN'S +0.699pp")
print("=" * 78)
print(f"  A. PARENT, all anchors                  N={len(par_all):2d}  "
      f"{100*par_all.mean():+.3f}%")
print(f"  B. parent anchors the join DELETES      N={len(unm):2d}  "
      f"{100*unm.mean():+.3f}%   <- what filtering removes")
print(f"  C. parent anchors the join KEEPS,")
print(f"     measured on the PARENT's date        N={len(mat_at_parent):2d}  "
      f"{100*mat_at_parent.mean():+.3f}%")
print(f"  D. the SAME episodes, measured on the")
print(f"     CELL's (later) date                  N={len(mat_at_cell):2d}  "
      f"{100*mat_at_cell.mean():+.3f}%   <- what the cell reports")
print()
print(f"  value of FILTERING    (A -> C) = "
      f"{100*(mat_at_parent.mean() - par_all.mean()):+.3f}pp")
print(f"  value of RE-ANCHORING (C -> D) = "
      f"{100*(mat_at_cell.mean() - mat_at_parent.mean()):+.3f}pp")
tot = 100 * (mat_at_cell.mean() - par_all.mean())
fil = 100 * (mat_at_parent.mean() - par_all.mean())
rea = 100 * (mat_at_cell.mean() - mat_at_parent.mean())
print(f"  TOTAL                          = {tot:+.3f}pp   "
      f"-> re-anchoring is {100*rea/tot:.0f}% of it, filtering {100*fil/tot:.0f}%")

print("\n  matched pairs, parent date -> cell date:")
for pa, ca in matched:
    sh = int(POS[ca]) - int(POS[pa])
    print(f"    {pa.date()} ({100*R1.get(pa, np.nan):+.3f}%) -> "
          f"{ca.date()} ({100*R1.get(ca, np.nan):+.3f}%)  shift {sh} td")
print("  deleted parent anchors:")
for pa, _ in unmatched:
    print(f"    {pa.date()}  {100*R1.get(pa, np.nan):+.3f}%")

print("\n" + "=" * 78)
print("CONTROL: does a PURE DELAY of the parent anchor reproduce the cell?")
print("=" * 78)
print("  match the join's own shift distribution (0,0,0,0,0,1,2,4,9,11,17 td)")
shifts = sorted(int(POS[c]) - int(POS[p]) for p, c in matched)
print(f"  observed shifts: {shifts}  median {int(np.median(shifts))}")
rng = np.random.default_rng(7)
pool = np.array(shifts)
sims = []
ppos = [int(POS[p]) for p, _ in matched]
for _ in range(20000):
    s = rng.choice(pool, size=len(ppos), replace=True)
    vals = [R1.iloc[q] for q in (np.array(ppos) + s) if q < len(IDX)]
    v = np.array([x for x in vals if not np.isnan(x)])
    if len(v):
        sims.append(v.mean())
sims = np.array(sims)
obs = mat_at_cell.mean()
print(f"  RANDOM re-anchoring using the SAME shift pool: mean of means "
      f"{100*sims.mean():+.3f}%, 95th pctile {100*np.quantile(sims, 0.95):+.3f}%")
print(f"  the cell's actual {100*obs:+.3f}%  ->  "
      f"P(random shift >= cell) = {(sims >= obs).mean():.4f}")
print("  (a random delay drawn from the join's own shift distribution is the")
print("   null 'the join is nothing but a delay'. A small p here says the join")
print("   picked BETTER dates than a random delay; a large p says it did not.)")
