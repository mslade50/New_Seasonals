"""C9 round 2 -- the search charge C9 inherits, and the support it does not have.

Round 1 established:
  - the joint mask is THREE days in 20 years, TWO declustered episodes
  - the LEVEL trigger overlaps the killed RANK form on 125 of 138 days (91%)
  - the parent (GLD r21 rank >= 95) is NEGATIVE against its own drift at h=5
    (-0.410pp) and h=10 (-0.942pp)
  - the 2026-08-21 drawdown state reproduces: GLD >10% below its 52w high pays
    -0.149% (h=3, 37.5% hit) against +0.313% for the complement
  - and NONE of the three joint days had GLD more than 10% off its high, which
    is where GLD is today (-14.63%)

This round prices the search. Because the LEVEL mask is 91% the same days as
the RANK mask, C9 is not a fresh look at gold-versus-rates -- it is one more
cell drawn from the same grid whose rotation charge was already measured at
P(max t >= 2.06) = 0.937 on 2026-08-19. Re-measured here on THIS construction:
a grid of (gold rank rung) x (yield condition) x (vehicle) x (horizon), with a
circular-rotation null.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

px = close_panel(["GLD", "GDX", "^TNX", "DX-Y.NYB"]).dropna(how="any")
idx = px.index
gld, gdx, tnx = px["GLD"], px["GDX"], px["^TNX"]
r21 = pct_rank(gld, 21, 252)
tnx_off = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0
chg21 = tnx - tnx.shift(21)
rank21 = pct_rank(tnx, 21, 252)
gld_off = gld / rolling_on_valid(gld, lambda x: x.rolling(252).max()) - 1.0

YIELD_CONDS = {
    "TNX at 52wh (LEVEL, C9)": tnx_off >= -0.0025,
    "TNX within 1% of 52wh": tnx_off >= -0.01,
    "TNX within 2% of 52wh": tnx_off >= -0.02,
    "TNX chg21>=+0.20 (W14)": chg21 >= 0.20,
    "TNX rank21>=65 (RANK)": rank21 >= 65,
    "TNX rank21>=80": rank21 >= 80,
}
GOLD_RUNGS = {"r21>=90": r21 >= 90, "r21>=95": r21 >= 95, "r21>=98": r21 >= 98}
VEH = {"GLD": [("GLD", 1.0)], "GDX": [("GDX", 1.0)]}
HS = (1, 2, 3, 5, 10)

print("=" * 105)
print("1. THE GRID.  %d yield conditions x %d gold rungs x %d vehicles x %d horizons "
      "= %d cells" % (len(YIELD_CONDS), len(GOLD_RUNGS), len(VEH), len(HS),
                      len(YIELD_CONDS) * len(GOLD_RUNGS) * len(VEH) * len(HS)))
print("=" * 105)
rets = {(vk, h): vehicle_ret(px, lg, h, 1) for vk, lg in VEH.items() for h in HS}
rows = []
for yk, ym in YIELD_CONDS.items():
    for gk, gm in GOLD_RUNGS.items():
        m = (ym & gm).reindex(idx, fill_value=False).values
        for (vk, h), ret in rets.items():
            sig = idx[m & ret.notna().values]
            if len(sig) == 0:
                continue
            e = declusters(sig, max(h, 5), idx)
            v = ret.loc[e].values
            v = v[~np.isnan(v)]
            if len(v) < 2:
                continue
            t = (v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std(ddof=1) > 0 else np.nan
            rows.append({"yield": yk, "gold": gk, "veh": vk, "h": h, "n": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "hit": round(100 * (v > 0).mean(), 0), "t": round(t, 2)})
g = pd.DataFrame(rows)
print("  cells with N>=2: %d ; cells with N>=8: %d" % (len(g), int((g.n >= 8).sum())))
print("\n  top 8 by |t| across the WHOLE grid (this is what a search would have found):")
print(g.reindex(g["t"].abs().sort_values(ascending=False).index).head(8).to_string(index=False))
print("\n  the C9 cell itself:")
print(g[(g["yield"].str.startswith("TNX at 52wh")) & (g.gold == "r21>=95")].to_string(index=False))
print("\n  cells with N>=8 only (the ones a sample-size floor would keep):")
print(g[g.n >= 8].reindex(g[g.n >= 8]["t"].abs().sort_values(ascending=False).index)
      .head(10).to_string(index=False))

print("\n" + "=" * 105)
print("2. ROTATION CHARGE.  Circular-shift the YIELD condition (the added gate),")
print("   keep the gold state fixed, recompute the grid max |t| among N>=8 cells.")
print("=" * 105)


def grid_max(shift):
    best = 0.0
    for yk, ym in YIELD_CONDS.items():
        yv = np.roll(ym.reindex(idx, fill_value=False).values, shift)
        for gk, gm in GOLD_RUNGS.items():
            m = yv & gm.reindex(idx, fill_value=False).values
            for (vk, h), ret in rets.items():
                sig = idx[m & ret.notna().values]
                if len(sig) < 8:
                    continue
                e = declusters(sig, max(h, 5), idx)
                v = ret.loc[e].values
                v = v[~np.isnan(v)]
                if len(v) < 8 or v.std(ddof=1) == 0:
                    continue
                t = abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
                best = max(best, t)
    return best


obs = grid_max(0)
rng = np.random.default_rng(7)
sh = rng.integers(252, len(idx) - 252, size=200)
null = np.array([grid_max(int(s)) for s in sh])
print("  observed grid max |t| (N>=8 cells) = %.2f" % obs)
print("  rotation null (200 shifts): median %.2f, 95th %.2f" % (np.median(null), np.percentile(null, 95)))
print("  P(rotated grid max |t| >= observed) = %.3f" % float((null >= obs).mean()))

print("\n" + "=" * 105)
print("3. SUPPORT.  Where is today inside this grid?")
print("=" * 105)
C9 = (tnx_off >= -0.0025) & (r21 >= 95)
print("  C9 days ever: %d -> %s" % (int(C9.sum()), ", ".join(str(d.date()) for d in idx[C9.values])))
print("  of those, how many had GLD >10%% below its 52w high (today -14.63%%): %d"
      % int((C9 & (gld_off <= -0.10)).sum()))
print("  of those, how many had TNX chg21 <= +0.15pt (today +0.035): %d"
      % int((C9 & (chg21 <= 0.15)).sum()))
print("  of those, how many had DX 21d rank <= 5 (today 0.4): %d"
      % int((C9 & (pct_rank(px['DX-Y.NYB'], 21, 252) <= 5)).sum()))
JOINT = C9 & (gld_off <= -0.10) & (chg21 <= 0.15)
print("  today's THREE-WAY literal state (C9 + deep drawdown + slow grind): %d days ever"
      % int(JOINT.sum()))

print("\n" + "=" * 105)
print("4. WHAT THE PARENT ACTUALLY SAYS ABOUT TODAY (the only populated read)")
print("=" * 105)
for h in (1, 2, 3, 5, 10):
    ret = vehicle_ret(px, [("GLD", 1.0)], h, 1)
    m = ((r21 >= 95) & (gld_off <= -0.10)).reindex(idx, fill_value=False).values
    sig = idx[m & ret.notna().values]
    e = declusters(sig, max(h, 5), idx)
    v = ret.loc[e].values
    w = int((v > 0).sum())
    base = ret.dropna()
    print("  h=%2d  gold thrust + GLD >10%% off its high: N=%2d  mean %+.3f%%  hit %.0f%%  "
          "sign p %.4f  ctrl-b %+.3f%%  EDGE %+.3f pp"
          % (h, len(v), 100 * v.mean(), 100 * (v > 0).mean(), sign_test(w, len(v)),
             100 * base.mean(), 100 * (v.mean() - base.mean())))
