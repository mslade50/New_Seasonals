"""N6: is August+midterm special, or the luckiest of ~48 month x cycle cells?

N5 found the cell at N=5: long DX after a weak NFP close, August, midterm,
+1.424% at h=5 with a 100% hit rate and t=2.893. That number cannot be read
naked. With 12 months x 4 cycle-year types there are 48 such cells, each
holding ~5-7 NFPs, and SOME cell has to be the best one. The question is
whether August+midterm is more extreme than the best cell luck alone produces.

Two tests:
  1. the full 48-cell grid, so the cell is ranked rather than admired alone
  2. a permutation test that reassigns NFP dates to cells at random, preserving
     cell sizes, and records the MAX cell mean each time. That is the null
     distribution of "the best cell you would find by chance."

Also isolates the broad effect underneath: long DX after ANY weak NFP close,
N=160, which is the statistically defensible version of the same intuition.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

RAW = load_prices(["DX-Y.NYB"])["DX-Y.NYB"]
DX = RAW.dropna(subset=["Close"]).copy()
CAL = DX.index
POS = pd.Series(range(len(CAL)), index=CAL)
EV = load_events()
NFP = [d for d in EV.loc[EV.event == "nfp", "date"] if d in POS.index]
C = DX["Close"]
WEAK = C < C.shift(1)          # the loosest definition, the one that "worked"
H = 5

rows = []
for d in NFP:
    p = POS[d]
    if p + H >= len(CAL):
        continue
    rows.append({"date": d, "month": d.month, "cyc": d.year % 4,
                 "weak": bool(WEAK.get(d, False)),
                 "ret": C.iloc[p + H] / C.iloc[p] - 1.0})
df = pd.DataFrame(rows)
wk = df[df.weak].reset_index(drop=True)

CYC = {0: "election", 1: "post-elec", 2: "midterm", 3: "pre-elec"}
print(f"weak-close NFPs with a full h={H} window: N = {len(wk)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 94)
print(f"TEST 1 - the full month x cycle grid (weak-close NFPs, long DX, h={H})")
print("=" * 94)
grid = []
for m in range(1, 13):
    for c in range(4):
        v = wk[(wk.month == m) & (wk.cyc == c)]["ret"].values
        if len(v) == 0:
            continue
        grid.append({"month": m, "cycle": CYC[c], "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1)})
g = pd.DataFrame(grid).sort_values("mean_pct", ascending=False).reset_index(drop=True)
print(f"  populated cells: {len(g)}")
print("\n  TOP 8 BY MEAN:")
print(g.head(8).to_string(index=True))
print("\n  WHERE AUGUST x MIDTERM SITS:")
aug = g[(g.month == 8) & (g.cycle == "midterm")]
rank = aug.index[0] + 1 if len(aug) else None
print(aug.to_string(index=False))
print(f"    rank {rank} of {len(g)} populated cells")

n100 = g[(g.hit == 100.0) & (g.n >= 5)]
print(f"\n  cells with a 100% hit rate and N>=5: {len(n100)}")
print(n100.to_string(index=False))
print("    ^ if this is more than one, a 100% hit on N=5 is not remarkable")

# ---------------------------------------------------------------------------
print("\n" + "=" * 94)
print("TEST 2 - permutation: how extreme is the BEST cell under pure chance?")
print("=" * 94)
sizes = [len(wk[(wk.month == m) & (wk.cyc == c)])
         for m in range(1, 13) for c in range(4)]
sizes = [s for s in sizes if s > 0]
obs_aug = 100 * wk[(wk.month == 8) & (wk.cyc == 2)]["ret"].mean()
obs_max = g["mean_pct"].max()
vals = wk["ret"].values

rng = np.random.default_rng(42)
max_means, aug_means = [], []
n_aug = len(wk[(wk.month == 8) & (wk.cyc == 2)])
for _ in range(20000):
    perm = rng.permutation(vals)
    i, cell_means = 0, []
    for s in sizes:
        cell_means.append(100 * perm[i:i + s].mean())
        i += s
    max_means.append(max(cell_means))
    aug_means.append(100 * rng.choice(vals, n_aug, replace=False).mean())

max_means = np.array(max_means)
aug_means = np.array(aug_means)
print(f"  observed August x midterm mean : {obs_aug:+.3f}%  (N={n_aug})")
print(f"  observed BEST cell in the grid : {obs_max:+.3f}%")
print(f"\n  null distribution of the BEST-of-{len(sizes)}-cells mean:")
for q in (50, 75, 90, 95, 99):
    print(f"    {q}th pctile = {np.percentile(max_means, q):+.3f}%")
p_family = (max_means >= obs_aug).mean()
print(f"\n  P(best cell by CHANCE >= our August x midterm {obs_aug:+.3f}%) "
      f"= {p_family:.3f}")
print(f"    ^ this is the family-wise p-value. It is the honest one.")
p_naive = (aug_means >= obs_aug).mean()
print(f"  P(a random N={n_aug} draw >= {obs_aug:+.3f}%) = {p_naive:.3f}")
print(f"    ^ the naive p-value, which ignores that we PICKED this cell")

# ---------------------------------------------------------------------------
print("\n" + "=" * 94)
print("TEST 3 - the broad effect underneath (the defensible version)")
print("=" * 94)
v = wk["ret"].values
allv = df["ret"].values
print(f"  long DX after ANY weak NFP close, h={H}: "
      f"N={len(v)} mean {100*v.mean():+.4f}% hit {100*(v>0).mean():.1f}% "
      f"t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.3f}")
print(f"  all NFP control                        : "
      f"N={len(allv)} mean {100*allv.mean():+.4f}%")
r = (C.shift(-H) / C - 1.0).dropna()
print(f"  DX unconditional all-days control      : "
      f"N={len(r)} mean {100*r.mean():+.4f}%")
print(f"  edge vs all-NFP control: {100*(v.mean()-allv.mean()):+.4f}pp")
print(f"  bootstrap P(mean<=0) = {bootstrap_p_le0(v):.4f}")
print(f"  DX futures round trip ~1.5 bps -> {100*v.mean()/0.015:.1f}x cost")
dts = pd.DatetimeIndex(wk["date"])
for s in era_split(dts, v):
    print(f"    era {s['label']:<10} n={s['n']:<4} mean={s['mean_pct']:+.4f} "
          f"t={s['t']:+.2f}")
mid = wk.cyc == 2
print(f"    midterm     n={int(mid.sum()):<4} "
      f"mean={100*v[mid.values].mean():+.4f}")
print(f"    non-midterm n={int((~mid).sum()):<4} "
      f"mean={100*v[~mid.values].mean():+.4f}")
