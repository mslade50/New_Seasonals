"""A1 ROUND 2, part 2 -- IS THE THREE-WAY JOIN A FILTER OR AN ANCHOR SWAP?

k1_tlt_ig_floor_b.py showed the join does not DELETE episodes, it MOVES them:
  fresh anchors, TLT<=0.5 alone     : 17 dates, mean -0.229%
  fresh anchors, TLT & LQD          : 10 dates, mean +0.156%
  fresh anchors, all three (CELL)   : 10 dates, mean +0.470%
The IEF leg changes the episode COUNT by ZERO (10 -> 10) and the mean by
+0.314pp of a +0.453pp headline. The 2026-09-07 checker found exactly this on
one anchor (Sept 2022, -1.03% -> +1.68%) and filed it in the registry as a trap.
This script asks how many of the ten anchors it happens to.

Mechanism matters here: the freshness rule anchors on the FIRST trigger day in
>= 10 sessions. Tightening the price rung DELAYS the first qualifying day, so a
"join" can move an anchor forward into the part of a selloff that already
bottomed. That is a look-ahead-flavoured selection even though every input is
point-in-time, because the thing being selected is WHEN to call the episode.

Also here:
  - the subset permutation: the cell picks 10 of 77 trigger days. How often
    does a random 10-subset of the same trigger population beat +0.470%?
  - the same, restricted to "fresh-shaped" subsets (one draw per episode).
  - what the parent's OWN anchors pay when carried forward N sessions, which
    is the anchor swap expressed as a pure timing experiment.
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
BASE = R1.dropna().mean()


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


CELL = ((TL <= 0.5) & (IE <= 1.0) & (LQ <= 1.0)).fillna(False)
PARENT = (TL <= 0.5).fillna(False)
TLT_LQD = ((TL <= 0.5) & (LQ <= 1.0)).fillna(False)

a_cell = first_in(CELL)
a_par = first_in(PARENT)
a_tl = first_in(TLT_LQD)

print("=" * 78)
print("1. ANCHOR-BY-ANCHOR: where did the join MOVE the anchor to?")
print("=" * 78)
print(f"{'parent anchor':<14}{'r%':>8}   {'cell anchor':<14}{'r%':>8}"
      f"{'shift td':>10}{'delta pp':>10}")
rows = []
used = set()
for pa in a_par:
    # the cell anchor that lands inside the same episode: nearest cell anchor
    # within 21 td forward of the parent anchor
    cands = [c for c in a_cell if 0 <= int(POS[c]) - int(POS[pa]) <= 21]
    ca = cands[0] if cands else None
    rp = R1.get(pa, np.nan)
    rc = R1.get(ca, np.nan) if ca is not None else np.nan
    if ca is not None:
        used.add(ca)
    sh = (int(POS[ca]) - int(POS[pa])) if ca is not None else None
    rows.append((pa, rp, ca, rc, sh))
    print(f"{str(pa.date()):<14}{100*rp:>8.3f}   "
          f"{str(ca.date()) if ca is not None else '-':<14}"
          f"{(100*rc if ca is not None else float('nan')):>8.3f}"
          f"{(sh if sh is not None else -1):>10}"
          f"{(100*(rc-rp) if ca is not None else float('nan')):>10.3f}")
orphan = [c for c in a_cell if c not in used]
print(f"  cell anchors with no parent anchor inside 21 td: "
      f"{[str(d.date()) for d in orphan]}")
matched = [(p, rp, c, rc, s) for p, rp, c, rc, s in rows if c is not None]
shifted = [x for x in matched if x[4] != 0]
same = [x for x in matched if x[4] == 0]
print(f"\n  matched pairs: {len(matched)}   SAME day: {len(same)}   "
      f"SHIFTED: {len(shifted)}")
if shifted:
    dp = np.array([x[3] - x[1] for x in shifted if not np.isnan(x[3])])
    print(f"  on the {len(dp)} shifted anchors the join moved the return "
          f"{100*dp.mean():+.3f}pp on average "
          f"(median {100*np.median(dp):+.3f}pp, "
          f"{int((dp > 0).sum())} of {len(dp)} improved)")
    print(f"  shift sizes (td): {[x[4] for x in shifted]}")
if same:
    ds = np.array([x[1] for x in same if not np.isnan(x[1])])
    print(f"  on the {len(ds)} anchors the join did NOT move, the return is "
          f"{100*ds.mean():+.3f}%")

print("\n" + "=" * 78)
print("2. TIMING EXPERIMENT -- carry the PARENT's anchors forward k sessions")
print("=" * 78)
print("  If the join's value is information, delaying the parent anchor by a")
print("  fixed number of sessions should NOT reproduce it.")
par_pos = [int(POS[d]) for d in a_par]
for k in range(0, 13, 2):
    vals = []
    for p in par_pos:
        q = p + k
        if q < len(IDX):
            vals.append(R1.iloc[q])
    v = np.array([x for x in vals if not np.isnan(x)])
    w = int((v > 0).sum())
    print(f"    parent anchors + {k:2d} td: N={len(v):2d} {100*v.mean():+.3f}% "
          f"hit {100*(v>0).mean():.1f}% sign p {sign_test(w, len(v)):.4f}")
print("  and the cell's own anchors for contrast:")
vc = R1.loc[a_cell].dropna()
print(f"    CELL anchors        : N={len(vc)} {100*vc.mean():+.3f}% "
      f"hit {100*(vc>0).mean():.1f}%")
print("  median shift of the join was printed above; compare that row.")

print("\n" + "=" * 78)
print("3. SUBSET PERMUTATION -- the cell picks 10 of 77 trigger days")
print("=" * 78)
trig = IDX[CELL.values]
tv = R1.loc[trig].dropna()
print(f"  trigger-day population: N={len(tv)} mean {100*tv.mean():+.3f}%")
rng = np.random.default_rng(42)
obs = float(R1.loc[a_cell].dropna().mean())
draws = rng.choice(tv.values, size=(20000, len(vc)), replace=False
                   if False else True).mean(axis=1)
# without replacement, properly
idxs = np.array([rng.permutation(len(tv))[:len(vc)] for _ in range(20000)])
draws = tv.values[idxs].mean(axis=1)
print(f"  P(random {len(vc)}-subset of the trigger days >= the cell's "
      f"{100*obs:+.3f}%) = {(draws >= obs).mean():.4f}")
print(f"  null-max style: 95th pctile of random subsets = "
      f"{100*np.quantile(draws, 0.95):+.3f}%")

print("\n" + "=" * 78)
print("4. THE HONEST CELL: what survives if the IEF anchor swap is removed?")
print("=" * 78)
for lbl, a in (("all three legs (as pitched)", a_cell),
               ("TLT & LQD only", a_tl),
               ("TLT alone", a_par)):
    v = R1.loc[a].dropna()
    w = int((v > 0).sum())
    edge = 100 * (v.mean() - BASE)
    print(f"  {lbl:<28s} N={len(v):2d} {100*v.mean():+.3f}% "
          f"excess {edge:+.3f}pp hit {100*(v>0).mean():.1f}% "
          f"sign p {sign_test(w, len(v)):.4f}  -> "
          f"{abs(edge)*100/3.0:.1f}x a 3 bps round trip")
print(f"  anchors TLT&LQD : {[str(d.date()) for d in a_tl]}")
print(f"  anchors ALL     : {[str(d.date()) for d in a_cell]}")
print(f"  differing dates : "
      f"{[str(d.date()) for d in pd.DatetimeIndex(a_cell).symmetric_difference(a_tl)]}")
