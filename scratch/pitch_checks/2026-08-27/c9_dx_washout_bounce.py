"""C9 -- DX 21d PIT rank <= 5 AND 5d PIT rank >= 60. Long the dollar.

The BARE washout parent (watchlist 27) is already dead in midterms:
DX -0.479% over 22 episodes at 36.4% hit, bootstrap P(mean<=0) 0.970.
C9's ENTIRE claim is that the bounce conditioner is INDEPENDENT of the cycle
split. So the midterm x bounce cross is the whole test, and it is run FIRST.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

TK = ["DX-Y.NYB", "UUP", "SPY"]
px = close_panel(TK)
dx = px["DX-Y.NYB"]
r21k = pct_rank(dx, 21)
r5k = pct_rank(dx, 5)
print("today: DX r21 rank %.1f, r5 rank %.1f" % (r21k.iloc[-1], r5k.iloc[-1]))

parent = (r21k <= 5).fillna(False)
mask = (parent & (r5k >= 60)).fillna(False)
print("\n=== counts ===")
print("parent (r21<=5):", int(parent.sum()), "days |",
      len(declusters(px.index[parent], 10, px.index)), "episodes")
print("C9 (r21<=5 & r5>=60):", int(mask.sum()), "days |",
      len(declusters(px.index[mask], 10, px.index)), "episodes")
print("gate keeps %.1f%% of parent days" % (100 * mask.sum() / max(1, parent.sum())))
print("by year:", {int(k): int(v) for k, v in
                   mask.groupby(px.index.year).sum().items() if v})

# ---------- THE test: midterm x bounce, run first ----------
print("\n\n########## MIDTERM x BOUNCE CROSS (the whole question) ##########")
for h in (3, 5, 10):
    ret = fwd_lag(dx, h, 1)
    ok = ret.notna().values
    rows = []
    for nm, m in (("parent r21<=5", parent), ("C9 r21<=5 & r5>=60", mask),
                  ("anti-C9 r21<=5 & r5<60", (parent & (r5k < 60)).fillna(False))):
        s = px.index[m.values & ok]
        e = declusters(s, 10, px.index)
        mid = np.array([d.year % 4 == 2 for d in e])
        rows.append(summarize(ret.loc[e].values, f"{nm} ALL (N={len(e)})"))
        rows.append(summarize(ret.loc[e[mid]].values,
                              f"{nm} MIDTERM (N={int(mid.sum())})"))
        rows.append(summarize(ret.loc[e[~mid]].values,
                              f"{nm} non-mid (N={int((~mid).sum())})"))
    base = ret.dropna()
    rows.append(summarize(base.values, f"all days (N={len(base)})"))
    bm = np.array([d.year % 4 == 2 for d in base.index])
    rows.append(summarize(base.values[bm], f"all days MIDTERM (N={int(bm.sum())})"))
    show(rows, f"DX long, h={h}")
    # sign tests on the midterm cell
    s = px.index[mask.values & ok]
    e = declusters(s, 10, px.index)
    mid = np.array([d.year % 4 == 2 for d in e])
    v = ret.loc[e[mid]].values
    if len(v):
        w = int((v > 0).sum())
        print(f"  h={h} C9-MIDTERM record {w}-{len(v)-w}, sign p={sign_test(w, len(v)):.4f}, "
              f"bootstrap P(mean<=0)={bootstrap_p_le0(v):.3f}")

variants = {
    "parent r21<=5 (NO gate)": parent,
    "C9 r21<=5 & r5>=60": mask,
    "r21<=5 & r5>=75": (parent & (r5k >= 75)).fillna(False),
    "r21<=5 & r5>=50": (parent & (r5k >= 50)).fillna(False),
    "r21<=2 & r5>=60": ((r21k <= 2) & (r5k >= 60)).fillna(False),
    "r21<=10 & r5>=60": ((r21k <= 10) & (r5k >= 60)).fillna(False),
    "all days": pd.Series(True, index=px.index),
}
for h in (3, 5, 10):
    battery(px, mask, [("DX-Y.NYB", 1.0)], h, "C9 LONG DX | r21<=5 & r5>=60",
            0.75, variants=variants, min_gap=10, event_kinds=("jackson_hole",))

print("\n\n### UUP vehicle (documented 1.3-1.4 bp per 5td worse) ###")
for h in (5, 10):
    battery(px, mask, [("UUP", 1.0)], h, "C9 LONG UUP", 3.0,
            min_gap=10, event_kinds=("jackson_hole",))
