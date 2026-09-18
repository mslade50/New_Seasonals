"""06 round 1+2 -- KILL ATTEMPT on watchlist entry 5 expressed as a LIMIT order.

Parked cell (2026-08-12, a6b_c4_tight_horizon_freshness.py): TLT within 0.5% of
its trailing-252 low AND IEF within 1.0% AND LQD within 1.0%, restricted to
EPISODE-FIRST days (first trigger in >= 10 td), long TLT h=1 lag=1 MOC. Claim:
+0.354pp excess, 82.4% hit, N=17, sign p 0.0101, ex-2022 +0.339% at 90%.

Today the freshness leg clears for the first time since it was parked (last
tight fire 2026-08-18, 13 sessions back) and only the TLT price leg is missing:
TLT +1.444% above its 252d low against the <= 0.5% rung. The candidate reaches
for it with a resting BUY LIMIT at the arming level (~81.44 = close - 1.25 ATR).

That entry form is NOT what the parked evidence measured, and this script's job
is to find out whether it survives the translation. Sections:

  A  reproduce the parked cell exactly (h=1,2,3,5)
  B  the SKIPPED LEG: parked form is lag=1, so close(D)->close(D+1) is never
     traded. The limit form owns it. What does it pay?
  C  the LIMIT form, no lookahead: level from D-1 closes, IEF/LQD state from
     D-1 closes, TLT LOW on D touches. Exits MOC D / D+1 / D+2.
  D  fill rate, unconditional and conditional, and the WHOLE-VARIANT compare
     (per-order expectation incl. zeros for unfilled). No marginal decomposition.
  E  gate attribution: drop each leg in turn, count what it removes
  F  credit conditioner (HYG near its 52w high vs stressed)
  G  era / rate regime / cycle year
  H  concentration + declustering
  I  definition neighbours (rung ladder, freshness ladder)
  J  horizon scan both forms
  K  macro print inside the hold
  L  cost
  M  book overlap (backtest_trades_full.parquet, TLT)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

TK = ["TLT", "IEF", "LQD", "HYG"]
raw = load_prices(TK + ["^TNX"])
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna(subset=["TLT", "IEF", "LQD"])
px = px[["TLT", "IEF", "LQD", "HYG"]]
idx = px.index
N = len(idx)
pos = pd.Series(range(N), index=idx)

tlt = raw["TLT"].reindex(idx)
OPEN, HIGH, LOW, CLOSE = (tlt["Open"].values, tlt["High"].values,
                          tlt["Low"].values, tlt["Close"].values)
ATR = wilder_atr(tlt["High"], tlt["Low"], tlt["Close"], 14)

tnx = raw["^TNX"]["Close"].reindex(idx).ffill()

RUNG = {"TLT": 0.5, "IEF": 1.0, "LQD": 1.0}
off = {t: ((px[t] / px[t].rolling(252).min()) - 1.0) * 100.0 for t in TK}

print("=" * 108)
print("LIVE READING  asof", idx[-1].date())
for t in TK:
    print(f"  {t:4s} close {px[t].iloc[-1]:8.3f}  off 252d low "
          f"{off[t].iloc[-1]:+.3f}%   off 252d high "
          f"{100*(px[t].iloc[-1]/px[t].rolling(252).max().iloc[-1]-1):+.3f}%")
print(f"  TLT Wilder-14 ATR {ATR[-1]:.4f} ({100*ATR[-1]/CLOSE[-1]:.3f}% of price)")
lvl_live = 1.005 * px["TLT"].rolling(252).min().iloc[-1]
print(f"  0.5% arming level = {lvl_live:.4f}   close-1.25ATR = "
      f"{CLOSE[-1]-1.25*ATR[-1]:.4f}   distance {100*(lvl_live/CLOSE[-1]-1):+.3f}% "
      f"= {(CLOSE[-1]-lvl_live)/ATR[-1]:.2f} ATR")
print(f"  ^TNX {tnx.iloc[-1]:.3f}  252d chg {100*(tnx.iloc[-1]-tnx.shift(252).iloc[-1]):+.1f}bp")

tight = np.ones(N, bool)
for t, k in RUNG.items():
    tight &= (off[t] <= k).values
tight &= ~np.isnan(off["TLT"].values)
trig = idx[tight]
epi = declusters(trig, 10, idx)
print(f"\n  tight-rung trigger DAYS {len(trig)}   episode-FIRST (gap>=10) {len(epi)}")
print(f"  last tight fire {trig[-1].date()}  ({pos[idx[-1]]-pos[trig[-1]]} sessions ago)")

# ---------------------------------------------------------------- A
print("\n" + "=" * 108)
print("A. REPRODUCE THE PARKED CELL  (long TLT, lag=1 MOC, episode-FIRST)")
print("=" * 108)
base_all = {}
for h in (1, 2, 3, 5):
    r = fwd_lag(px["TLT"], h, 1)
    b = r.dropna()
    base_all[h] = b.mean()
    v = r.loc[epi].dropna().values
    w = int((v > 0).sum())
    print(f"  h={h:2d}  N={len(v):3d}  mean {100*v.mean():+.4f}%  "
          f"excess {100*(v.mean()-b.mean()):+.4f}pp  hit {100*w/len(v):5.1f}%  "
          f"t {v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f}  "
          f"signp(vs base) {sign_test(w, len(v), float((b>0).mean())):.4f}  "
          f"signp(0.5) {sign_test(w, len(v)):.4f}")
r1 = fwd_lag(px["TLT"], 1, 1)
b1 = r1.dropna()
v1 = r1.loc[epi].dropna().values
print(f"\n  PARKED CLAIM: N=17, excess +0.354pp, hit 82.4%, sign p 0.0101")
print(f"  REPRODUCED  : N={len(v1)}, excess {100*(v1.mean()-b1.mean()):+.4f}pp, "
      f"hit {100*(v1>0).mean():.1f}%, sign p "
      f"{sign_test(int((v1>0).sum()), len(v1), float((b1>0).mean())):.4f}")
print("  episode dates:", ", ".join(str(d.date()) for d in epi))
ex22 = np.array([r1.loc[d] for d in epi if d.year != 2022 and not np.isnan(r1.loc[d])])
print(f"  ex-2022: N={len(ex22)} mean {100*ex22.mean():+.4f}% hit {100*(ex22>0).mean():.1f}%")
print(f"  concentration: {cluster_note(epi, v1)}")
loc = local_control(idx[r1.notna().values], trig)
lv = r1.loc[loc].dropna().values
se = np.sqrt(v1.var(ddof=1)/len(v1) + lv.var(ddof=1)/len(lv))
print(f"  local +/-126td ex-trigger control {100*lv.mean():+.4f}%  welch t "
      f"{(v1.mean()-lv.mean())/se:+.2f}")

# ---------------------------------------------------------------- B
print("\n" + "=" * 108)
print("B. THE SKIPPED LEG  close(D) -> close(D+1), which lag=1 never trades")
print("=" * 108)
r0 = px["TLT"].shift(-1) / px["TLT"] - 1.0
s0 = r0.loc[epi].dropna().values
b0 = r0.dropna()
print(f"  episode-first D->D+1 : N={len(s0)} mean {100*s0.mean():+.4f}% "
      f"excess {100*(s0.mean()-b0.mean()):+.4f}pp hit {100*(s0>0).mean():.1f}%")
print("  The limit form fills INSIDE session D, so it owns this leg plus the")
print("  parked one. If it is negative the translation dilutes the parked edge.")
r2c = px["TLT"].shift(-2) / px["TLT"] - 1.0   # close D -> close D+2, lag=0 h=2
s2 = r2c.loc[epi].dropna().values
print(f"  episode-first D->D+2 : N={len(s2)} mean {100*s2.mean():+.4f}% "
      f"excess {100*(s2.mean()-r2c.dropna().mean()):+.4f}pp hit {100*(s2>0).mean():.1f}%")

# ---------------------------------------------------------------- C
print("\n" + "=" * 108)
print("C. THE LIMIT FORM, NO LOOKAHEAD")
print("=" * 108)
minc = px["TLT"].rolling(252).min()
level = (1.005 * minc.shift(1)).values          # knowable pre-open on D
armed_prev = np.ones(N, bool)                   # IEF/LQD state at D-1 close
for t in ("IEF", "LQD"):
    armed_prev &= (off[t].shift(1) <= RUNG[t]).values
armed_prev &= ~np.isnan(off["IEF"].shift(1).values)
tlt_above_prev = (off["TLT"].shift(1) > RUNG["TLT"]).values   # reach-down order
touch = LOW <= level
ok = ~np.isnan(level)

place_all = armed_prev & ok                     # order placed, any TLT position
place_rd = place_all & tlt_above_prev           # TODAY's configuration
fill_all = place_all & touch
fill_rd = place_rd & touch


def fillpx(m):
    return np.where(OPEN[m] < level[m], OPEN[m], level[m])


def limit_rows(mask, tag, gap=10):
    d = idx[mask]
    if len(d) == 0:
        print(f"  {tag}: NO FILLS EVER")
        return None
    dd = declusters(d, gap, idx)
    p = pos[dd].values
    fp = np.where(OPEN[p] < level[p], OPEN[p], level[p])
    out = {"dates": dd, "fill": fp, "p": p}
    print(f"\n  --- {tag}: {len(d)} fill days, {len(dd)} episode-first (gap {gap})")
    for lbl, k in (("exit MOC D  ", 0), ("exit MOC D+1", 1), ("exit MOC D+2", 2)):
        q = p + k
        m = q < N
        v = CLOSE[q[m]] / fp[m] - 1.0
        w = int((v > 0).sum())
        # matched control: same-length holds from the fill price basis is not
        # definable for all days, so control = all-day close-to-close over k+? .
        ctl = (px["TLT"].shift(-k) / px["TLT"] - 1.0).dropna() if k else None
        cs = f" ctl(all days close->close+{k}) {100*ctl.mean():+.4f}%" if k else ""
        print(f"      {lbl}  N={m.sum():3d}  mean {100*v.mean():+.4f}%  "
              f"hit {100*w/len(v):5.1f}%  t "
              f"{v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f}  "
              f"signp {sign_test(w, len(v)):.4f}  worst {100*v.min():+.2f}%{cs}")
        out[f"h{k}"] = v
        out[f"h{k}_dates"] = dd[m]
    # where did the close on D end up?
    inside = off["TLT"].values[p] <= RUNG["TLT"]
    print(f"      of {len(p)} fills, close(D) ended INSIDE the 0.5% rung on "
          f"{int(inside.sum())} ({100*inside.mean():.0f}%)")
    gapped = OPEN[p] < level[p]
    print(f"      gapped below the level at the open on {int(gapped.sum())} "
          f"({100*gapped.mean():.0f}%)  -> fill better than the limit")
    return out


L_all = limit_rows(fill_all, "limit, any TLT position at D-1")
L_rd = limit_rows(fill_rd, "limit, REACH-DOWN only (TLT above rung at D-1) = today")

print("\n  conservative variant: fill AT THE LEVEL even when the day gaps below")
for tag, mask in (("any", fill_all), ("reach-down", fill_rd)):
    d = declusters(idx[mask], 10, idx)
    p = pos[d].values
    for k in (0, 1, 2):
        q = p + k
        m = q < N
        v = CLOSE[q[m]] / level[p[m]] - 1.0
        print(f"    {tag:11s} exit MOC D+{k}  N={m.sum():3d} mean "
              f"{100*v.mean():+.4f}%  hit {100*(v>0).mean():.1f}%")

# ---------------------------------------------------------------- D
print("\n" + "=" * 108)
print("D. FILL RATE + WHOLE-VARIANT COMPARISON")
print("=" * 108)
uncond_lvl = 1.005 * minc.shift(1).values
u_ok = ~np.isnan(uncond_lvl)
print(f"  UNCONDITIONAL, limit at the 0.5%-above-252d-low level:")
print(f"    days the level exists {int(u_ok.sum())}, of which the low touches "
      f"{int((touch & u_ok).sum())}  = {100*(touch & u_ok).sum()/u_ok.sum():.1f}%")
rd_u = u_ok & tlt_above_prev
print(f"    reach-down days only (TLT above rung at D-1): {int(rd_u.sum())}, "
      f"touched {int((touch & rd_u).sum())} = {100*(touch&rd_u).sum()/rd_u.sum():.2f}%")
atrlvl = CLOSE - 1.25 * np.nan_to_num(ATR, nan=np.nan)
atrlvl = np.concatenate([[np.nan], atrlvl[:-1]])   # yesterday's close-1.25ATR
t2 = LOW <= atrlvl
o2 = ~np.isnan(atrlvl)
print(f"  UNCONDITIONAL, limit at prior close - 1.25 ATR: {int((t2&o2).sum())}/"
      f"{int(o2.sum())} = {100*(t2&o2).sum()/o2.sum():.2f}% of all sessions fill")
print(f"  CONDITIONAL on the IEF/LQD floor armed at D-1:")
print(f"    order days {int(place_all.sum())}, fills {int(fill_all.sum())} = "
      f"{100*fill_all.sum()/max(place_all.sum(),1):.1f}%")
print(f"    reach-down order days {int(place_rd.sum())}, fills "
      f"{int(fill_rd.sum())} = {100*fill_rd.sum()/max(place_rd.sum(),1):.1f}%")

print("\n  WHOLE-VARIANT per-ORDER expectation (unfilled orders count as 0.00%),")
print("  episode-first order days, both variants scored over their OWN order set:")
# MOC variant: order set = tight-rung episode-first close days, always trades
print(f"    MOC-always  (tight rung close D, buy MOC D+1, exit MOC D+2): "
      f"orders {len(v1)}  per-order {100*v1.mean():+.4f}%")
for tag, pl, fi in (("any", place_all, fill_all), ("reach-down", place_rd, fill_rd)):
    od = declusters(idx[pl], 10, idx)
    p = pos[od].values
    for k in (0, 1, 2):
        q = p + k
        m = q < N
        f = touch[p[m]]
        fp = np.where(OPEN[p[m]] < level[p[m]], OPEN[p[m]], level[p[m]])
        v = np.where(f, CLOSE[q[m]] / fp - 1.0, 0.0)
        print(f"    LIMIT {tag:11s} exit MOC D+{k}: orders {m.sum():3d}  fills "
              f"{int(f.sum()):3d}  per-order {100*v.mean():+.4f}%  "
              f"per-fill {100*v[f].mean() if f.sum() else float('nan'):+.4f}%")

print("\n  SAME-POPULATION entry-form compare (limit fill days only, whole")
print("  variants over one date set -- NOT a marginal-fill decomposition):")
if L_rd is not None:
    dd = L_rd["dates"]
    p = pos[dd].values
    fp = L_rd["fill"]
    for k in (1, 2):
        q = p + k
        m = q < N
        a = CLOSE[q[m]] / fp[m] - 1.0                 # limit fill -> MOC D+k
        b = CLOSE[q[m]] / CLOSE[p[m]] - 1.0           # MOC D      -> MOC D+k
        print(f"    exit MOC D+{k}: LIMIT {100*a.mean():+.4f}%  vs  "
              f"MOC-at-close-D {100*b.mean():+.4f}%  diff "
              f"{100*(a.mean()-b.mean()):+.4f}pp (N={m.sum()})")

# ---------------------------------------------------------------- E
print("\n" + "=" * 108)
print("E. GATE ATTRIBUTION -- does each leg filter anything?")
print("=" * 108)


def cell(mask_arr, tag, h=1, lag=1):
    d = idx[mask_arr]
    if len(d) == 0:
        print(f"  {tag:38s} N_days=  0  DEAD")
        return
    e = declusters(d, 10, idx)
    r = fwd_lag(px["TLT"], h, lag)
    v = r.loc[e].dropna().values
    b = r.dropna()
    if len(v) == 0:
        print(f"  {tag:38s} N_days={len(d):4d}  no scored episodes")
        return
    w = int((v > 0).sum())
    print(f"  {tag:38s} N_days={len(d):4d} N_epi={len(v):3d}  "
          f"mean {100*v.mean():+.4f}%  excess {100*(v.mean()-b.mean()):+.4f}pp  "
          f"hit {100*w/len(v):5.1f}%  signp {sign_test(w, len(v)):.4f}")


tlt_only = (off["TLT"] <= 0.5).values & ~np.isnan(off["TLT"].values)
ief_lqd = ((off["IEF"] <= 1.0) & (off["LQD"] <= 1.0)).values & ~np.isnan(off["IEF"].values)
cell(tight, "FULL join TLT.5 / IEF1 / LQD1")
cell(tlt_only, "TLT<=0.5 ALONE")
cell(ief_lqd, "IEF+LQD only (no TLT leg)")
cell(tlt_only & (off["IEF"] <= 1.0).values, "TLT.5 + IEF1 (LQD dropped)")
cell(tlt_only & (off["LQD"] <= 1.0).values, "TLT.5 + LQD1 (IEF dropped)")
print(f"\n  leg removal counts on the TLT<=0.5 base ({int(tlt_only.sum())} days):")
print(f"    adding IEF<=1.0 removes "
      f"{int(tlt_only.sum()) - int((tlt_only & (off['IEF']<=1.0).values).sum())} days")
print(f"    adding LQD<=1.0 removes "
      f"{int(tlt_only.sum()) - int((tlt_only & (off['LQD']<=1.0).values).sum())} days")
print(f"    the full join removes {int(tlt_only.sum())-int(tight.sum())} of "
      f"{int(tlt_only.sum())}")

# ---------------------------------------------------------------- F
print("\n" + "=" * 108)
print("F. CREDIT CONDITIONER -- is there any credit content?")
print("=" * 108)
hyg_off_hi = ((px["HYG"] / px["HYG"].rolling(252).max()) - 1.0) * 100.0
hh = hyg_off_hi.reindex(idx)
for thr in (-1.0, -2.0, -5.0):
    m_hi = (hh >= thr).values
    a = [d for d in epi if bool(m_hi[pos[d]])]
    b = [d for d in epi if not bool(m_hi[pos[d]]) and not np.isnan(hh.values[pos[d]])]
    for lbl, s in (("HYG near high", a), ("HYG stressed", b)):
        v = r1.loc[s].dropna().values if len(s) else np.array([])
        if len(v):
            print(f"  HYG within {abs(thr):.0f}% of 52w high | {lbl:14s} N={len(v):3d} "
                  f"mean {100*v.mean():+.4f}%  hit {100*(v>0).mean():.1f}%")
        else:
            print(f"  HYG within {abs(thr):.0f}% of 52w high | {lbl:14s} N=0")
print("  LQD-on-IEF regression (the 'duration wearing a credit label' test):")
for h in (1, 5, 10):
    ri = fwd_lag(px["IEF"], h, 1)
    rl = fwd_lag(px["LQD"], h, 1)
    m = ri.notna() & rl.notna()
    beta = np.polyfit(ri[m].values, rl[m].values, 1)
    resid = rl.loc[epi].dropna().values - (beta[0]*ri.loc[epi].dropna().values + beta[1])
    print(f"    h={h:2d}  LQD = {beta[0]:.3f}*IEF {100*beta[1]:+.4f}pp   "
          f"episode residual {100*resid.mean():+.4f}pp (N={len(resid)})")

# ---------------------------------------------------------------- G
print("\n" + "=" * 108)
print("G. ERA / RATE REGIME / CYCLE")
print("=" * 108)
show(era_split(epi, v1), "episode era split")
d252 = (tnx - tnx.shift(252)).reindex(idx)
rise = (d252 > 0).values
a = [d for d in epi if bool(rise[pos[d]])]
b = [d for d in epi if not bool(rise[pos[d]])]
for lbl, s in (("RISING 252d yield", a), ("FALLING 252d yield", b)):
    v = r1.loc[s].dropna().values if len(s) else np.array([])
    if len(v):
        print(f"  {lbl:20s} N={len(v):3d}  mean {100*v.mean():+.4f}%  hit "
              f"{100*(v>0).mean():.1f}%  dates {[str(x.date()) for x in s]}")
    else:
        print(f"  {lbl:20s} N=0")
print(f"  live 252d yield change {100*(tnx.iloc[-1]-tnx.shift(252).iloc[-1]):+.1f}bp "
      f"(RISING)")
cyc = {}
for d in epi:
    cyc.setdefault(d.year % 4, []).append(r1.loc[d])
for k in sorted(cyc):
    v = np.array([x for x in cyc[k] if not np.isnan(x)])
    lab = {0: "election", 1: "post-elec", 2: "MIDTERM", 3: "pre-elec"}[k]
    print(f"  cycle {k} ({lab:9s}) N={len(v):2d} mean {100*v.mean():+.4f}% "
          f"hit {100*(v>0).mean():.0f}%")

# ---------------------------------------------------------------- H
print("\n" + "=" * 108)
print("H. CONCENTRATION")
print("=" * 108)
for k in (1, 2, 3):
    print(f"  top{k}: {cluster_note(epi, v1, k)}")
byyr = pd.Series(v1, index=[d.year for d in epi[:len(v1)]]).groupby(level=0)
print("  by year:", {y: round(100*g.sum(), 3) for y, g in byyr})

# ---------------------------------------------------------------- I
print("\n" + "=" * 108)
print("I. DEFINITION NEIGHBOURS  (MY search -- report as such)")
print("=" * 108)
print("  TLT rung ladder (IEF/LQD held at 1.0):")
for k in (0.25, 0.5, 0.75, 1.0):
    m = ((off["TLT"] <= k) & (off["IEF"] <= 1.0) & (off["LQD"] <= 1.0)).values
    m &= ~np.isnan(off["TLT"].values)
    cell(m, f"    TLT<={k}")
print("  IEF/LQD rung ladder (TLT held at 0.5):")
for k in (0.5, 1.0, 1.5, 2.0):
    m = ((off["TLT"] <= 0.5) & (off["IEF"] <= k) & (off["LQD"] <= k)).values
    m &= ~np.isnan(off["TLT"].values)
    cell(m, f"    IEF/LQD<={k}")
print("  freshness gap ladder (tight rung, h=1 lag=1):")
for g in (1, 5, 10, 21, 42):
    e = declusters(trig, g, idx)
    v = r1.loc[e].dropna().values
    print(f"    gap {g:2d}td  N={len(v):3d}  mean {100*v.mean():+.4f}%  "
          f"excess {100*(v.mean()-b1.mean()):+.4f}pp  hit {100*(v>0).mean():5.1f}%  "
          f"signp {sign_test(int((v>0).sum()), len(v)):.4f}")

# ---------------------------------------------------------------- J
print("\n" + "=" * 108)
print("J. HORIZON SCAN (MY search -- report as such)")
print("=" * 108)
show(horizon_scan(px, epi, [("TLT", 1.0)], hs=tuple(range(1, 11)), lag=1,
                  min_gap=10), "MOC form (parked), episode-first")
if L_rd is not None:
    rows = []
    p = pos[L_rd["dates"]].values
    fp = L_rd["fill"]
    for k in range(0, 11):
        q = p + k
        m = q < N
        v = CLOSE[q[m]] / fp[m] - 1.0
        rows.append({"label": f"exit MOC D+{k}", "n": int(m.sum()),
                     "mean_pct": 100*v.mean(), "hit": 100*(v > 0).mean(),
                     "t": v.mean()/(v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else np.nan,
                     "worst_pct": 100*v.min()})
    show(rows, "LIMIT form (reach-down), episode-first")

# ---------------------------------------------------------------- K
print("\n" + "=" * 108)
print("K. MACRO PRINT INSIDE THE HOLD  (PPI 09-10 = +2td, CPI 09-11 = +3td)")
print("=" * 108)
for h, lag in ((1, 1), (2, 1), (3, 1)):
    fl = event_in_window(epi, idx, h, lag, ("cpi", "ppi", "nfp", "fomc_decision"))
    r = fwd_lag(px["TLT"], h, lag)
    v = r.loc[epi].values
    ok2 = ~np.isnan(v)
    print(f"  MOC form h={h}: print IN  N={int((fl&ok2).sum()):2d} "
          f"mean {100*np.nanmean(v[fl&ok2]) if (fl&ok2).sum() else float('nan'):+.4f}% "
          f"hit {100*(v[fl&ok2]>0).mean() if (fl&ok2).sum() else float('nan'):.0f}%"
          f"   |  OUT N={int(((~fl)&ok2).sum()):2d} "
          f"mean {100*np.nanmean(v[(~fl)&ok2]):+.4f}% "
          f"hit {100*(v[(~fl)&ok2]>0).mean():.0f}%")
if L_rd is not None:
    dd = L_rd["dates"]
    p = pos[dd].values
    fp = L_rd["fill"]
    ev = load_events(["cpi", "ppi", "nfp", "fomc_decision"])["date"].values.astype("datetime64[ns]")
    for k in (1, 2):
        q = p + k
        m = q < N
        inw = np.array([bool(((ev > np.datetime64(idx[a])) &
                              (ev <= np.datetime64(idx[bq]))).any())
                        for a, bq in zip(p[m], q[m])])
        v = CLOSE[q[m]] / fp[m] - 1.0
        for lbl, sel in (("print IN ", inw), ("print OUT", ~inw)):
            if sel.sum():
                print(f"  LIMIT exit D+{k} {lbl} N={int(sel.sum()):2d} mean "
                      f"{100*v[sel].mean():+.4f}% hit {100*(v[sel]>0).mean():.0f}%")

# ---------------------------------------------------------------- L
print("\n" + "=" * 108)
print("L. COST")
print("=" * 108)
print("  TLT round trip ~2-3 bps all in (1 leg).")
for lbl, val in (("MOC form h=1 episode mean", 100*v1.mean()),):
    print(f"  {lbl}: {100*val:.1f} bps -> {100*val/3:.1f}x a 3 bp round trip")
if L_rd is not None:
    for k in (0, 1, 2):
        vv = L_rd.get(f"h{k}")
        if vv is not None and len(vv):
            print(f"  LIMIT reach-down exit D+{k}: {100*100*vv.mean():.1f} bps -> "
                  f"{100*100*vv.mean()/3:.1f}x a 3 bp round trip")

# ---------------------------------------------------------------- M
print("\n" + "=" * 108)
print("M. BOOK OVERLAP -- does the systematic book hold TLT on these dates?")
print("=" * 108)
LP = Path("data/backtest_trades_full.parquet")
if LP.exists():
    led = pd.read_parquet(LP)
    cols = [c for c in led.columns]
    tk = "Ticker" if "Ticker" in cols else [c for c in cols if "icker" in c][0]
    t_ = led[led[tk] == "TLT"]
    print(f"  ledger rows total {len(led)}, TLT rows {len(t_)}")
    if len(t_):
        sc = "Strategy_Name" if "Strategy_Name" in cols else "Strategy"
        print(t_.groupby([sc, "Direction" if "Direction" in cols else sc]).size()
              .to_string())
        dc = "Entry_Date" if "Entry_Date" in cols else "Signal_Date"
        xc = "Exit_Date" if "Exit_Date" in cols else None
        ed = pd.to_datetime(t_[dc])
        print(f"  TLT entries span {ed.min().date()} .. {ed.max().date()}")
        hits = 0
        for d in epi:
            m = (ed <= d) & (pd.to_datetime(t_[xc]) >= d) if xc else (ed == d)
            hits += int(m.sum())
        print(f"  book TLT positions open on a tight-rung episode-first day: {hits}")
        print(f"  most recent TLT trades:")
        print(t_.sort_values(dc).tail(6)[[sc, dc] + ([xc] if xc else []) +
              [c for c in ("Direction", "R_Multiple", "PnL") if c in cols]].to_string(index=False))
else:
    print("  ledger parquet ABSENT")
print("\nDONE.")
