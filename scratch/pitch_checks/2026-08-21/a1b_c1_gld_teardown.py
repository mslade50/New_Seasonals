"""C1 round 2: teardown of the GLD miner-thrust cell.

Round 1 said: cell N=75 episodes +0.853% (excess +0.619, sign p 0.010) but
(a) the gate over its own parent is +0.313pp at welch t +0.86, (b) day-level
era split is +1.412% pre-2018 against +0.122% from 2018, and (c) today's
trend half (GLD 63d rank < 50) pays +0.421% at a 52.8% hit, sign p 0.434.

Round 2 crosses those, prices the live GLD 5d rank band (86.9, i.e. the top
of the "has not joined" range), checks mid-cluster entry, and asks whether
GLD is the right leg at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5

px = close_panel(["GLD", "GDX", "SLV", "GDXJ", "NEM", "SPY"]).loc[:ASOF]
idx = px.index
gdx5 = pct_rank(px["GDX"], 5)
gld5 = pct_rank(px["GLD"], 5)
gld63 = pct_rank(px["GLD"], 63)
gdx21 = pct_rank(px["GDX"], 21)
gld_hi = rolling_on_valid(px["GLD"], lambda x: x.rolling(252).max())
gld_dd = px["GLD"] / gld_hi - 1.0

mask = ((gdx5 >= 95) & (gld5 < 95)).fillna(False)
leg = fwd_lag(px["GLD"], H, lag=1)
base = leg.dropna()
epi = declusters(idx[(mask & leg.notna()).values], 5, idx)
v = leg.loc[epi].values


def line(lbl, sub_mask_on_epi):
    s = np.asarray(sub_mask_on_epi, bool)
    if s.sum() == 0:
        print(f"  {lbl:<40} N=0")
        return
    x = v[s]
    print(f"  {lbl:<40} N={len(x):<4} mean {100*x.mean():+7.3f}%  hit {100*(x>0).mean():5.1f}%  "
          f"sign p {sign_test(int((x>0).sum()), len(x)):.4f}  worst {100*x.min():+7.2f}%  "
          f"excess {100*x.mean()-100*base.mean():+7.3f}%")


print("=" * 100)
print("C1b-1  ERA x TREND CROSS.  today = 2018+ AND GLD 63d rank 34.1 (<50)")
print("=" * 100)
post = np.array([d >= pd.Timestamp("2018-01-01") for d in epi])
weak = (gld63.loc[epi].values < 50)
line("ALL episodes", np.ones(len(v), bool))
line("pre-2018", ~post)
line("2018+", post)
line("2018+ AND rk63<50  <- TODAY'S CELL", post & weak)
line("2018+ AND rk63>=50", post & ~weak)
line("pre-2018 AND rk63<50", ~post & weak)
print()
deep = (gld_dd.loc[epi].values <= -0.10)
line("2018+ AND dd<=-10%  <- TODAY (dd -16.3%)", post & deep)
line("2018+ AND dd>-10%", post & ~deep)
line("2018+ AND rk63<50 AND dd<=-10%  <- TODAY", post & weak & deep)

print("\n" + "=" * 100)
print("C1b-2  GLD 5d RANK BAND.  today = 86.9, the TOP of the 'has not joined' range")
print("=" * 100)
g5 = gld5.loc[epi].values
for lo, hi in ((0, 50), (50, 70), (70, 85), (85, 95)):
    line(f"GLD 5d rank [{lo},{hi})", (g5 >= lo) & (g5 < hi))
print(f"  monotone? the cell's own conditioner should pay MORE the LESS GLD has joined.")

print("\n" + "=" * 100)
print("C1b-3  GDX MAGNITUDE SUPPORT.  today GDX 21d rank 99.6, 5d +13.12%, 21d +30.2%")
print("=" * 100)
g21 = gdx21.loc[epi].values
print(f"  episode GDX 21d rank: median {np.nanmedian(g21):.1f}   "
      f"episodes with rk21 >= 99: {int((g21 >= 99).sum())} of {len(g21)}")
for lo, hi in ((0, 70), (70, 90), (90, 99), (99, 101)):
    line(f"GDX 21d rank [{lo},{hi})", (g21 >= lo) & (g21 < hi))
gdx5r = _valid_pct_change(px["GDX"], 5).loc[epi].values
print(f"\n  episode GDX 5d return: median {100*np.nanmedian(gdx5r):.2f}%   today +13.12%")
big = gdx5r >= 0.10
line("GDX 5d ret >= +10% (today +13.1%)", big)
line("GDX 5d ret <  +10%", ~big)

print("\n" + "=" * 100)
print("C1b-4  MID-CLUSTER ENTRY.  how deep into the GDX run is today?")
print("=" * 100)
trig_days = idx[(mask & leg.notna()).values]
pos = pd.Series(range(len(idx)), index=idx)
# how many of the prior 10 sessions were also trigger days
prior = []
for d in trig_days:
    p = pos[d]
    w = mask.iloc[max(0, p - 10):p]
    prior.append(int(w.sum()))
prior = np.array(prior)
dv = leg.loc[trig_days].values
for lo, hi in ((0, 1), (1, 3), (3, 11)):
    s = (prior >= lo) & (prior < hi)
    if s.sum() == 0:
        continue
    x = dv[s]
    print(f"  prior-10td trigger count [{lo},{hi}): N={len(x):<4} mean {100*x.mean():+7.3f}%  "
          f"hit {100*(x>0).mean():5.1f}%  (DAY level)")
p_today = pos[ASOF]
print(f"  TODAY prior-10td trigger count = {int(mask.iloc[max(0,p_today-10):p_today].sum())}")
print(f"  last 12 sessions GDX5 / GLD5 ranks:")
for d in idx[-12:]:
    print(f"     {d.date()}  GDX5 {gdx5.loc[d]:5.1f}  GLD5 {gld5.loc[d]:5.1f}  "
          f"{'TRIGGER' if mask.loc[d] else ''}")

print("\n" + "=" * 100)
print("C1b-5  IS GLD THE RIGHT LEG?  same trigger, every vehicle in the complex")
print("=" * 100)
rows = []
for tkr in ("GLD", "SLV", "GDX", "GDXJ", "NEM"):
    lg = fwd_lag(px[tkr], H, lag=1)
    e = declusters(idx[(mask & lg.notna()).values], 5, idx)
    x = lg.loc[e].values
    b = lg.dropna()
    r = summarize(x, tkr)
    r["own_drift"] = round(100 * b.mean(), 3)
    r["excess_pct"] = round(r["mean_pct"] - 100 * b.mean(), 3)
    r["signp"] = round(sign_test(int((x > 0).sum()), len(x)), 4)
    rows.append(r)
show(rows, "same trigger, h=5 episodes, each vehicle")

print("\n" + "=" * 100)
print("C1b-6  DECLUSTER SENSITIVITY (episode stat must not depend on min_gap)")
print("=" * 100)
for g in (5, 10, 21, 42):
    e = declusters(idx[(mask & leg.notna()).values], g, idx)
    x = leg.loc[e].values
    print(f"  min_gap {g:>2}: N={len(e):<4} mean {100*x.mean():+7.3f}%  "
          f"excess {100*x.mean()-100*base.mean():+7.3f}%  hit {100*(x>0).mean():5.1f}%  "
          f"sign p {sign_test(int((x>0).sum()), len(x)):.4f}")
