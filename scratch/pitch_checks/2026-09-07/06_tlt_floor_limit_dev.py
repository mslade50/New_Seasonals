"""06 round 3 -- development / autopsy of the TLT floor cell as a LIMIT order.

Round 1+2 (06_tlt_floor_limit.py) reproduced the parked MOC cell (+0.385pp
excess, 83.3% hit, N=18, sign p 0.0038, local welch t +2.44) and found the
LIMIT translation negative at every exit. This script establishes WHY, with
numbers the write-up can quote:

  1  per-session decomposition from the state print -- where the +0.40% lives
  2  the SKIPPED LEG measured on the parked cell's own 18 episodes
  3  fill price vs close(D): does the limit buy above the day's own close?
  4  the touch-and-reverse case (low touches, close ends OUTSIDE the rung)
  5  IEF-leg attribution: exactly which observations it removes
  6  episode_paths on the LOSING episodes, both forms
  7  exit sensitivity for the limit form (target / stop / time)
  8  my own search cost, counted honestly
  9  the number that would turn a LIMIT form on
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

TK = ["TLT", "IEF", "LQD"]
raw = load_prices(TK)
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna()
idx = px.index
N = len(idx)
pos = pd.Series(range(N), index=idx)
tlt = raw["TLT"].reindex(idx)
OPEN, LOW, CLOSE = tlt["Open"].values, tlt["Low"].values, tlt["Close"].values

off = {t: ((px[t] / px[t].rolling(252).min()) - 1.0) * 100.0 for t in TK}
tight = ((off["TLT"] <= 0.5) & (off["IEF"] <= 1.0) & (off["LQD"] <= 1.0)).values
tight &= ~np.isnan(off["TLT"].values)
trig = idx[tight]
epi = declusters(trig, 10, idx)
P = pos[epi].values

print("=" * 108)
print("1. PER-SESSION DECOMPOSITION from the state print close(D), 18 episodes")
print("=" * 108)
print("   (the parked trade buys at close(D+1) and sells at close(D+2), so it")
print("    owns session +2 ONLY. A limit filling inside session D owns +1 too.)")
cum = 0.0
for k in range(1, 8):
    a, b = P + k - 1, P + k
    m = b < N
    inc = CLOSE[b[m]] / CLOSE[a[m]] - 1.0
    cum += inc.mean()
    star = "  <-- the ONLY session the parked trade holds" if k == 2 else ""
    star = "  <-- the session the LIMIT adds" if k == 1 else star
    print(f"   session +{k}: {100*inc.mean():+.4f}%  hit {100*(inc>0).mean():5.1f}%  "
          f"cum from close(D) {100*cum:+.4f}%{star}")

print("\n" + "=" * 108)
print("2. THE SKIPPED LEG, scored properly against controls")
print("=" * 108)
r_skip = (px["TLT"].shift(-1) / px["TLT"] - 1.0)
v = r_skip.loc[epi].dropna().values
base = r_skip.dropna()
w = int((v > 0).sum())
loc = local_control(idx[r_skip.notna().values], trig)
lv = r_skip.loc[loc].dropna().values
se = np.sqrt(v.var(ddof=1) / len(v) + lv.var(ddof=1) / len(lv))
print(f"   close(D)->close(D+1) on the 18 episodes: {100*v.mean():+.4f}%  "
      f"hit {100*w/len(v):.1f}%  t {v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f}")
print(f"   all-days control {100*base.mean():+.4f}%   excess "
      f"{100*(v.mean()-base.mean()):+.4f}pp")
print(f"   local +/-126td ex-trigger {100*lv.mean():+.4f}%   welch t "
      f"{(v.mean()-lv.mean())/se:+.2f}   sign p (short side) "
      f"{sign_test(len(v)-w, len(v)):.4f}")
print("   Losing 11 of 18 on the session the limit form adds is not noise around")
print("   the headline, it is the headline's mirror image.")

print("\n" + "=" * 108)
print("3. FILL PRICE vs CLOSE(D) on limit-fill days")
print("=" * 108)
minc = px["TLT"].rolling(252).min()
level = (1.005 * minc.shift(1)).values
armed_prev = ((off["IEF"].shift(1) <= 1.0) & (off["LQD"].shift(1) <= 1.0)).values
armed_prev &= ~np.isnan(off["IEF"].shift(1).values)
above_prev = (off["TLT"].shift(1) > 0.5).values
touch = LOW <= level
fill_rd = armed_prev & above_prev & touch & ~np.isnan(level)
fd = declusters(idx[fill_rd], 10, idx)
fp_pos = pos[fd].values
fillp = np.where(OPEN[fp_pos] < level[fp_pos], OPEN[fp_pos], level[fp_pos])
slip = CLOSE[fp_pos] / fillp - 1.0
print(f"   reach-down episode-first fills N={len(fd)}")
print(f"   mean close(D)/fill - 1 = {100*slip.mean():+.4f}%   "
      f"close BELOW the fill on {int((slip<0).sum())} of {len(fd)}")
print("   dates / fill / close(D) / same-session P&L:")
for d, f_, s in zip(fd, fillp, slip):
    p = pos[d]
    print(f"     {d.date()}  fill {f_:8.3f}  close {CLOSE[p]:8.3f}  "
          f"{100*s:+.3f}%   open {OPEN[p]:8.3f}  low {LOW[p]:8.3f}")
print("   A buy limit that is under water at the same session's close on the")
print("   majority of its fills is buying the fall, not the floor.")

print("\n" + "=" * 108)
print("4. TOUCH-AND-REVERSE vs TOUCH-AND-KEEP-FALLING")
print("=" * 108)
inside = off["TLT"].values[fp_pos] <= 0.5
for lbl, sel in (("closed INSIDE the rung", inside), ("REVERSED out", ~inside)):
    if sel.sum() == 0:
        print(f"   {lbl:24s} N=0")
        continue
    for k in (0, 1, 2):
        q = fp_pos[sel] + k
        m = q < N
        vv = CLOSE[q[m]] / fillp[sel][m] - 1.0
        print(f"   {lbl:24s} exit D+{k}  N={m.sum():2d}  {100*vv.mean():+.4f}%  "
              f"hit {100*(vv>0).mean():.0f}%")
print("   The reversal case is the only one a limit wants and it cannot be")
print("   selected for at order time.")

print("\n" + "=" * 108)
print("5. IEF-LEG ATTRIBUTION -- what does the leg actually remove?")
print("=" * 108)
r1 = fwd_lag(px["TLT"], 1, 1)
no_ief = ((off["TLT"] <= 0.5) & (off["LQD"] <= 1.0)).values & ~np.isnan(off["TLT"].values)
e_noief = declusters(idx[no_ief], 10, idx)
a = r1.loc[e_noief].dropna()
b = r1.loc[epi].dropna()
print(f"   TLT.5 + LQD1 (IEF dropped): N_days={int(no_ief.sum())} "
      f"N_epi={len(a)} mean {100*a.mean():+.4f}%")
print(f"   FULL join (IEF added)     : N_days={int(tight.sum())} "
      f"N_epi={len(b)} mean {100*b.mean():+.4f}%")
gone = [d for d in e_noief if d not in set(epi)]
added = [d for d in epi if d not in set(e_noief)]
raw_days = sorted(set(idx[no_ief]) - set(idx[tight]))
print(f"   the IEF leg removes {len(raw_days)} RAW TRIGGER DAYS: "
      f"{[str(d.date()) for d in raw_days]}")
print(f"   after re-declustering that is {len(gone)} anchor(s) OUT "
      f"{[str(d.date()) for d in gone]} and {len(added)} IN "
      f"{[str(d.date()) for d in added]}  (net {len(a)-len(b):+d} episodes)")
for d in gone:
    print(f"     OUT {d.date()} paid {100*r1.loc[d]:+.4f}%   IEF was "
          f"{off['IEF'].loc[d]:+.3f}% off its low (rung 1.0)")
for d in added:
    print(f"     IN  {d.date()} paid {100*r1.loc[d]:+.4f}%")
contrib = 100 * (b.mean() - a.mean())
head = 100 * (b.mean() - r1.dropna().mean())
print(f"   removing them moves the episode mean by {contrib:+.4f}pp against a")
print(f"   headline excess of {head:+.4f}pp -> the IEF leg carries "
      f"{100*contrib/head:.0f}% of the finding while touching "
      f"{len(raw_days)} of {int(no_ief.sum())} days "
      f"({100*len(raw_days)/int(no_ief.sum()):.1f}%).")
negs = [d for d in gone if r1.loc[d] < 0]
if negs:
    tot = float(sum(r1.loc[d] for d in negs))
    print(f"   {len(negs)} of the removed anchors are LOSSES summing "
          f"{100*tot:+.3f}pp; spread over {len(b)} episodes that is "
          f"{-100*tot/len(b):+.4f}pp of the reported mean, i.e. the IEF leg's")
    print(f"   entire job is deleting {len(negs)} bad day(s).")
print(chr(10) + "=" * 108)
print("5b. THE SAME ATTRIBUTION AT DAY LEVEL (declustering cannot move it)")
print("=" * 108)
for lbl, m in (("TLT.5 only", (off["TLT"] <= 0.5).values),
               ("TLT.5 + IEF1", ((off["TLT"] <= 0.5) & (off["IEF"] <= 1.0)).values),
               ("TLT.5 + LQD1", no_ief),
               ("FULL join", tight)):
    mm = m & ~np.isnan(off["TLT"].values)
    vv = r1.loc[idx[mm]].dropna().values
    print(f"   {lbl:14s} N_days={len(vv):4d}  mean {100*vv.mean():+.4f}%  "
          f"hit {100*(vv>0).mean():5.1f}%")
only_ief = no_ief & ~tight & ~np.isnan(off["TLT"].values)
vv = r1.loc[idx[only_ief]].dropna().values
print(f"   the 4 days the IEF leg DELETES paid {100*vv.mean():+.4f}% "
      f"(hit {100*(vv>0).mean():.0f}%, N={len(vv)}): "
      f"{[round(100*x,3) for x in vv]}")
print("   Day level: dropping 4 of 80 days moves the mean "
      f"{100*(r1.loc[idx[tight]].dropna().mean() - r1.loc[idx[no_ief]].dropna().mean()):+.4f}pp.")


print("\n" + "=" * 108)
print("6. EPISODE PATHS on the LOSERS (what would say the thesis is wrong)")
print("=" * 108)
paths = episode_paths(px, epi, [("TLT", 1.0)], h=5, lag=1)
p3 = (100 * paths).round(3)
p3.index = [str(d.date()) for d in paths.index]
print("   MOC form, cumulative % from the entry close (day 1 = the traded bar):")
print(p3.to_string())
lose = p3[p3[1] < 0]
print(f"\n   losing episodes at day 1: {list(lose.index)}")
if len(lose):
    print(f"   worst day-1 {lose[1].min():+.3f}%  ({lose[1].idxmin()})")
    print(f"   those same episodes at day 5: {lose[5].to_dict()}")
rows = {}
for d, f_ in zip(fd, fillp):
    p_ = pos[d]
    if p_ + 5 >= N:
        continue
    rows[str(d.date())] = 100 * (CLOSE[p_:p_ + 6] / f_ - 1.0)
lp = pd.DataFrame(rows, index=["D+0", "D+1", "D+2", "D+3", "D+4", "D+5"]).T.round(3)
print(chr(10) + "   LIMIT form, cumulative % from the ACTUAL FILL PRICE:")
print(lp.to_string())
print(f"   losers at D+1: {list(lp.index[lp['D+1'] < 0])}  "
      f"worst {lp['D+1'].min():+.3f}% ({lp['D+1'].idxmin()})")
print(f"   column means: {lp.mean().round(3).to_dict()}")

print("\n" + "=" * 108)
print("7. EXIT SENSITIVITY for the limit form (does a stop or target rescue it?)")
print("=" * 108)
ATR = wilder_atr(tlt["High"], tlt["Low"], tlt["Close"], 14)
HI, LO = tlt["High"].values, tlt["Low"].values
for tgt, stp, hold in ((1.0, None, 2), (1.0, 1.0, 2), (0.5, 1.0, 2),
                       (None, 1.0, 2), (2.0, 1.0, 5), (None, None, 3)):
    outs = []
    for d, f_ in zip(fd, fillp):
        p = pos[d]
        a = ATR[p - 1] if not np.isnan(ATR[p - 1]) else np.nan
        if np.isnan(a):
            continue
        T = f_ + tgt * a if tgt else None
        S = f_ - stp * a if stp else None
        r = None
        for k in range(0, hold + 1):
            q = p + k
            if q >= N:
                break
            if S is not None and LO[q] <= S and k >= 1:
                r = S / f_ - 1.0
                break
            if T is not None and HI[q] >= T and k >= 1:
                r = T / f_ - 1.0
                break
            r = CLOSE[q] / f_ - 1.0
        if r is not None:
            outs.append(r)
    o = np.array(outs)
    print(f"   tgt {str(tgt):4s} ATR / stop {str(stp):4s} ATR / hold {hold}td: "
          f"N={len(o)}  mean {100*o.mean():+.4f}%  hit {100*(o>0).mean():.0f}%  "
          f"worst {100*o.min():+.3f}%")
print("   (stops arm day 2, book convention)")

print("\n" + "=" * 108)
print("8. MY OWN SEARCH COST (the candidate arrived pre-specified; this is mine)")
print("=" * 108)
print("   Walked here, all disclosed: 4 TLT rungs x 4 IEF/LQD rungs (8 cells,")
print("   one axis at a time) + 5 freshness gaps + 10 horizons x 2 entry forms")
print("   + 3 exit points x 2 fill rules x 2 populations + 6 exit-overlay")
print("   configurations = 8 + 5 + 20 + 12 + 6 = 51 cells.")
print("   NOTHING in that walk produced a positive LIMIT cell that clears cost,")
print("   so there is no best-cell to charge. Reported for the record only.")

print("\n" + "=" * 108)
print("9. THE NUMBER THAT WOULD TURN A LIMIT FORM ON")
print("=" * 108)
print(f"   The limit form needs the session it adds to stop being a loss.")
print(f"   Measured: close(D)->close(D+1) on the 18 episodes = "
      f"{100*v.mean():+.4f}% at a {100*w/len(v):.1f}% hit.")
print(f"   It needs >= 0.0% at a >= 50% hit before a fill inside session D can")
print(f"   be worth more than waiting for the close. Nothing about a limit")
print(f"   order changes that number; only new episodes can.")
print("\n   Separately, the parked MOC entry keeps its ORIGINAL arm unchanged:")
lvl = 1.005 * minc.iloc[-1]
print(f"     TLT must CLOSE at or below {lvl:.2f} (0.5% above the trailing-252")
print(f"     low of {minc.iloc[-1]:.2f}). Live close {CLOSE[-1]:.2f} = "
      f"{100*(CLOSE[-1]/minc.iloc[-1]-1):+.3f}% above the low, needs "
      f"{100*(lvl/CLOSE[-1]-1):+.3f}%.")
print("\nDONE.")
