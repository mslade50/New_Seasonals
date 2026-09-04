"""C11 round-2 probe: C11 AS FRAMED is dead (the VIX<=35 half subtracts edge:
joint h=5 excess +0.175% over 104 episodes vs SKEW>=95 ALONE +0.363% over 185).
So the only thing left standing is a DIFFERENT idea -- long SPY after a 5-day
^SKEW spike, no vol condition. This script tries to kill THAT before anything
gets reported as a survivor.

Attacks:
1. is it just "SPY dipped and bounced"? condition on SPY's own 5d return; if
   the SKEW spike only pays when SPY fell, the signal is the dip, not the skew.
2. the registry's own warning: the OPPOSITE tail of this instrument
   (^SKEW bottom decile at a 52w SPY high) read +0.410% at sign p 0.0205 and
   died three ways, one of which was dropping three months. Same test here.
3. concentration, best-year share, decluster at a longer gap, era, midterm.
4. horizon shape: where in the hold does the edge actually appear?
5. does it survive the SPY-near-high state that is live TODAY?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "^SKEW", "^VIX"])
px = px.dropna(subset=["SPY"])          # index-only bars break rolling windows
idx = px.index
sk5 = pct_rank(px["^SKEW"], 5)
spy5 = px["SPY"].pct_change(5)
spy_dist_high = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

MASK = (sk5 >= 95).reindex(idx, fill_value=False).fillna(False)
print(f"TODAY: SKEW rank5 {sk5.iloc[-1]:.1f}  SPY 5d {100*spy5.iloc[-1]:+.2f}%  "
      f"SPY dist high {100*spy_dist_high.iloc[-1]:+.2f}%  live={bool(MASK.iloc[-1])}")
depth = 0
for i in range(len(idx) - 1, -1, -1):
    if bool(MASK.iloc[i]):
        depth += 1
    else:
        break
print(f"cluster depth today (consecutive SKEW rank5>=95 days): {depth}\n")

H = 5
ret = vehicle_ret(px, [("SPY", 1.0)], H, 1)
valid = ret.notna()
sig = idx[MASK.values & valid.values]
base_all = ret[valid].dropna()
base_hit = float((base_all > 0).mean())


def line(lbl, dts):
    dts = pd.DatetimeIndex(dts)
    if len(dts) == 0:
        print(f"  {lbl:44s} n=0")
        return
    v = ret.loc[dts].values
    w = int((v > 0).sum())
    print(f"  {lbl:44s} n={len(v):4d}  mean {100*v.mean():+6.3f}%  "
          f"excess {100*(v.mean()-base_all.mean()):+6.3f}%  hit {100*w/len(v):5.1f}%  "
          f"signp {sign_test(w, len(v), base_hit):.4f}  bootP(<=0) {bootstrap_p_le0(v):.3f}")


print(f"=== all-days control: mean {100*base_all.mean():+.3f}%, hit {100*base_hit:.1f}% "
      f"(N={len(base_all)}) ===\n")

print("=== 1. DECLUSTER SENSITIVITY (is the episode count real?) ===")
for gap in (5, 10, 21, 42):
    line(f"SKEW rank5>=95, min_gap={gap}td", declusters(sig, gap, idx))

epi = declusters(sig, 5, idx)
v = ret.loc[epi].values

print("\n=== 2. IS IT THE SKEW SPIKE OR THE SPY DIP? ===")
s5 = spy5.reindex(epi).values
for lbl, m in (("SPY 5d <= -1% (dip)", s5 <= -0.01),
               ("SPY 5d in (-1%, +1%)", (s5 > -0.01) & (s5 < 0.01)),
               ("SPY 5d >= +1% (rally)", s5 >= 0.01),
               ("SPY 5d >= 0 (TODAY: -0.10%, flat)", s5 >= 0)):
    line(lbl, epi[m])
# the control question: does a SPY dip alone do this, with no skew condition?
dip_only = idx[(spy5 <= -0.01).reindex(idx, fill_value=False).values & valid.values]
line("CONTROL: SPY 5d <= -1%, NO skew condition", declusters(dip_only, 5, idx))

print("\n=== 3. CONCENTRATION / DROP-THE-BEST (the registry's 3-month test) ===")
print("  " + cluster_note(epi, v))
yr = pd.DatetimeIndex(epi).year
by_yr = pd.Series(v).groupby(yr.values).sum().sort_values(ascending=False)
print(f"  best-year share: {by_yr.index[0]} = {100*by_yr.iloc[0]:+.2f}pp of "
      f"{100*v.sum():+.2f}pp ({100*by_yr.iloc[0]/v.sum():.0f}%); "
      f"top-3 years {[(int(y), round(100*r,1)) for y, r in by_yr.head(3).items()]}")
ym = pd.PeriodIndex(epi, freq="M")
by_m = pd.Series(v).groupby(ym.values).sum().sort_values(ascending=False)
worst3 = list(by_m.head(3).index)
keep = ~pd.Series(ym.values).isin(worst3).values
print(f"  DROP the 3 best MONTHS {[str(m) for m in worst3]}: "
      f"mean {100*v[keep].mean():+.3f}% excess "
      f"{100*(v[keep].mean()-base_all.mean()):+.3f}% n={int(keep.sum())} "
      f"hit {100*(v[keep]>0).mean():.1f}%")
ordv = np.argsort(-v)
k2 = np.ones(len(v), bool)
k2[ordv[:2]] = False
print(f"  DROP top-2 episodes: mean {100*v[k2].mean():+.3f}% excess "
      f"{100*(v[k2].mean()-base_all.mean()):+.3f}%")
print(f"  episode year histogram: {dict(pd.Series(v).groupby(yr.values).count())}")

print("\n=== 4. ERA / MIDTERM / TODAY'S STATE ===")
line("pre-2018", epi[yr < 2018])
line("2018+", epi[yr >= 2018])
line("2013+ (post-SKEW-popularity)", epi[yr >= 2013])
line("midterm years (TODAY)", epi[(yr % 4) == 2])
line("non-midterm", epi[(yr % 4) != 2])
nh = spy_dist_high.reindex(epi).values >= -0.01
line("SPY within 1% of 52w high (TODAY)", epi[nh])
line("SPY not near high", epi[~nh])
vx = pct_rank(px["^VIX"], 5).reindex(epi).values
line("VIX rank5 <= 35 (the C11 joint cell)", epi[vx <= 35])
line("VIX rank5 > 35 (what C11 threw away)", epi[vx > 35])

print("\n=== 5. HORIZON SHAPE (edge vs all-days, episode level) ===")
show(horizon_scan(px, sig, [("SPY", 1.0)], hs=(1, 2, 3, 4, 5, 6, 7, 8, 10),
                  min_gap=5), "SKEW rank5>=95 alone")

print("\n=== 6. THRESHOLD NEIGHBOURS ===")
for cut in (85, 90, 92, 95, 98, 99):
    m = (sk5 >= cut).reindex(idx, fill_value=False).fillna(False)
    line(f"SKEW rank5 >= {cut}", declusters(idx[m.values & valid.values], 5, idx))

print("\n=== 7. CPI/PPI IN HOLD (today has both) ===")
fl = event_in_window(epi, idx, H, 1, ("cpi", "ppi"))
line("cpi or ppi IN hold", epi[fl])
line("neither in hold", epi[~fl])
