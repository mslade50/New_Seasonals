"""C1 round 1: Long GLD on a miner-led thrust the metal has not joined.

Watchlist entry (2026-08-11) claims GDX 5d rank >= 95 while GLD 5d rank < 95
pays +0.832% at h=5, excess +0.602, sign p 0.013; both-thrust pays +0.239%.
Re-verify from scratch on today's data, then attack:
  1. live state
  2. the cell vs three controls (battery)
  3. gate attribution: does "GLD has NOT joined" actually filter?
  4. threshold grid: GLD ceiling 85/90/95/97 x GDX floor 90/95/97
  5. concentration
  6. is today's trend state (GLD 16% below its 52w high, 63d rank 34) inside
     the historical support of the cell's episodes?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5

px = close_panel(["GLD", "GDX", "SPY", "SLV", "GDXJ", "NEM"]).loc[:ASOF]
idx = px.index

gdx5 = pct_rank(px["GDX"], 5)
gld5 = pct_rank(px["GLD"], 5)
gld63 = pct_rank(px["GLD"], 63)
gld_hi = rolling_on_valid(px["GLD"], lambda x: x.rolling(252).max())
gld_dd = px["GLD"] / gld_hi - 1.0
gdx_hi = rolling_on_valid(px["GDX"], lambda x: x.rolling(252).max())
gdx_dd = px["GDX"] / gdx_hi - 1.0

print("=" * 100)
print("C1-0  LIVE STATE as of", ASOF.date())
print("=" * 100)
print(f"  GDX 5d rank  = {gdx5.loc[ASOF]:.1f}   (need >= 95)")
print(f"  GLD 5d rank  = {gld5.loc[ASOF]:.1f}   (need <  95)")
print(f"  GLD 63d rank = {gld63.loc[ASOF]:.1f}")
print(f"  GLD 5d ret   = {100*(px['GLD'].iloc[-1]/px['GLD'].iloc[-6]-1):.2f}%   "
      f"GDX 5d ret = {100*(px['GDX'].iloc[-1]/px['GDX'].iloc[-6]-1):.2f}%")
print(f"  GLD dist 52wh= {100*gld_dd.loc[ASOF]:.2f}%   GDX dist 52wh = {100*gdx_dd.loc[ASOF]:.2f}%")

mask = ((gdx5 >= 95) & (gld5 < 95)).fillna(False)
both = ((gdx5 >= 95) & (gld5 >= 95)).fillna(False)
gdx_only = (gdx5 >= 95).fillna(False)

print(f"\n  trigger days (GDX>=95 & GLD<95): {int(mask.sum())}   "
      f"both-thrust days: {int(both.sum())}   GDX>=95 any: {int(gdx_only.sum())}")
print(f"  fires TODAY: {bool(mask.loc[ASOF])}")

# ---------------------------------------------------------------- 1. battery
variants = {}
for gc in (85, 90, 95, 97, 100):
    variants[f"GLD<{gc} x GDX>=95"] = ((gdx5 >= 95) & (gld5 < gc)).fillna(False)
for gf in (90, 93, 97):
    variants[f"GLD<95 x GDX>={gf}"] = ((gdx5 >= gf) & (gld5 < 95)).fillna(False)

battery(px, mask, [("GLD", 1.0)], H,
        "C1  long GLD | GDX 5d rank>=95 & GLD 5d rank<95", cost_bps=3.0,
        variants=variants, min_gap=5, event_kinds=("cpi", "ppi"))

# ------------------------------------------------- 2. gate attribution
print("\n" + "=" * 100)
print("C1-2  GATE ATTRIBUTION: does the 'GLD has NOT joined' leg filter?")
print("=" * 100)
leg = fwd_lag(px["GLD"], H, lag=1)
base = leg.dropna()
rows = []
for lbl, m in (("GDX>=95 ALONE (gate off)", gdx_only),
               ("GDX>=95 & GLD<95 (the cell)", mask),
               ("GDX>=95 & GLD>=95 (both thrust)", both),
               ("GLD<95 alone (no GDX cond)", (gld5 < 95).fillna(False))):
    e = declusters(idx[(m & leg.notna()).values], 5, idx)
    if len(e) < 3:
        rows.append({"label": lbl, "n": len(e)})
        continue
    v = leg.loc[e].values
    r = summarize(v, lbl)
    r["excess_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    r["signp"] = round(sign_test(int((v > 0).sum()), len(e)), 4)
    r["bootp"] = round(bootstrap_p_le0(v), 3)
    rows.append(r)
show(rows, "gate attribution, h=5 episodes (min_gap 5)")
print(f"  GLD all-days drift h={H}: {100*base.mean():+.3f}%  (N={len(base)})")

# Welch of cell vs the gate-off parent
e_cell = declusters(idx[(mask & leg.notna()).values], 5, idx)
e_par = declusters(idx[(gdx_only & leg.notna()).values], 5, idx)
a, b = leg.loc[e_cell].values, leg.loc[e_par].values
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  cell vs gate-off parent: {100*(a.mean()-b.mean()):+.3f}pp  welch t {(a.mean()-b.mean())/se:+.2f}")
e_both = declusters(idx[(both & leg.notna()).values], 5, idx)
c = leg.loc[e_both].values
se2 = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
print(f"  cell vs both-thrust:     {100*(a.mean()-c.mean()):+.3f}pp  welch t {(a.mean()-c.mean())/se2:+.2f}")

# ------------------------------------------------- 3. full 2-D grid
print("\n" + "=" * 100)
print("C1-3  THRESHOLD GRID (episode mean %, h=5, min_gap 5)  [N in brackets]")
print("=" * 100)
hdr = "  GDX floor \\ GLD ceil " + "".join(f"{gc:>16}" for gc in (85, 90, 95, 97, 100))
print(hdr)
for gf in (90, 93, 95, 97):
    line = f"  {gf:>19}  "
    for gc in (85, 90, 95, 97, 100):
        m = ((gdx5 >= gf) & (gld5 < gc)).fillna(False)
        e = declusters(idx[(m & leg.notna()).values], 5, idx)
        if len(e) < 3:
            line += f"{'n=' + str(len(e)):>16}"
        else:
            v = leg.loc[e].values
            line += f"{100*v.mean():>+9.3f}[{len(e):>3}]"
    print(line)

# ------------------------------------------------- 4. trend-state support
print("\n" + "=" * 100)
print("C1-4  IS TODAY'S TREND STATE INSIDE THE CELL'S HISTORICAL SUPPORT?")
print("     today: GLD -16.26% off 52wh, 63d rank 34.1 -- a thrust inside a drawdown")
print("=" * 100)
e = e_cell
tab = pd.DataFrame({
    "gld_dd_pct": 100 * gld_dd.loc[e].values,
    "gld_rk63": gld63.loc[e].values,
    "fwd_pct": 100 * leg.loc[e].values,
}, index=e)
print(f"  episode GLD distance-from-52wh: median {tab['gld_dd_pct'].median():.2f}%  "
      f"mean {tab['gld_dd_pct'].mean():.2f}%  min {tab['gld_dd_pct'].min():.2f}%")
print(f"  episodes with dd <= -10%: {int((tab['gld_dd_pct'] <= -10).sum())} of {len(tab)}  "
      f"({100*(tab['gld_dd_pct'] <= -10).mean():.1f}%)")
print(f"  episode GLD 63d rank: median {tab['gld_rk63'].median():.1f}  "
      f"episodes with rk63 < 50: {int((tab['gld_rk63'] < 50).sum())} of {len(tab)}")
for lbl, sub in (("dd <= -10% (today's half)", tab[tab["gld_dd_pct"] <= -10]),
                 ("dd >  -10%", tab[tab["gld_dd_pct"] > -10]),
                 ("rk63 < 50 (today's half)", tab[tab["gld_rk63"] < 50]),
                 ("rk63 >= 50", tab[tab["gld_rk63"] >= 50])):
    if len(sub) == 0:
        print(f"  {lbl:<28} N=0")
        continue
    v = sub["fwd_pct"].values / 100
    print(f"  {lbl:<28} N={len(sub):<4} mean {100*v.mean():+7.3f}%  hit {100*(v>0).mean():5.1f}%  "
          f"sign p {sign_test(int((v>0).sum()), len(v)):.4f}  worst {100*v.min():+7.2f}%")

# ------------------------------------------------- 5. horizon scan
print("\n" + "=" * 100)
print("C1-5  HORIZON SCAN on the cell")
print("=" * 100)
show(horizon_scan(px, e_cell, [("GLD", 1.0)], hs=(1, 2, 3, 5, 10), min_gap=5),
     "cell by horizon (episodes)")
