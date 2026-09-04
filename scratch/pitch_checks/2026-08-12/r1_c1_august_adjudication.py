"""C1 RED TEAM r1 -- adjudicate the August cross, and re-derive the two
load-bearing numbers from scratch (not from the prior report).

Three jobs:
  A. Independent re-derivation: live cell N/mean/hit/base, and the 2018+ gate
     lift + Welch t. Loads master_prices.parquet and macro_events.csv directly.
  B. Is the month effect a MECHANISM or a LABEL? Build a MONTH-matched control
     (and a month x tdom double control) and re-grade the live cell against it.
     If TLT's own August drift already explains the parent's August, the month
     charge is being double-counted.
  C. The 0-for-4 August cross, both directions, WITH the multiplicity charge
     the checker owes for having found the month rather than pre-specified it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from math import comb

ROOT_P = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------
# A. INDEPENDENT RE-DERIVATION (own loader, own event parse)
# --------------------------------------------------------------------------
mp = pd.read_parquet(ROOT_P / "data" / "master_prices.parquet")
tl = mp[mp["ticker"] == "TLT"].copy()
tl["date"] = pd.to_datetime(tl["date"])
tl = tl.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
idx = tl.index
c = tl["Close"].values.astype(float)
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
ok = ~np.isnan(d1)
base_hit = float((d1[ok] > 0).mean())
base_mean = float(d1[ok].mean())

ecsv = pd.read_csv(ROOT_P / "data" / "macro_events.csv")
ecsv["date"] = pd.to_datetime(ecsv["date"])


def sess(kind):
    out = set()
    for x in ecsv.loc[ecsv["event"] == kind, "date"]:
        p = int(idx.searchsorted(x, "left"))
        if 0 <= p < N:
            out.add(p)
    return out


PPI, CPI = sess("ppi"), sess("cpi")
ppi_l = sorted(p for p in PPI if 1 <= p < N and ok[p])
v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
mo, yr = dt.month.values, dt.year.values
L = np.array([(p - 1) in CPI for p in ppi_l])          # CPI on the eve = LIVE

print("=" * 100)
print("A. INDEPENDENT RE-DERIVATION")
print("=" * 100)
print(f"TLT bars {idx[0].date()}..{idx[-1].date()}  N={N}")
print(f"TLT unconditional daily mean {100*base_mean:+.4f}%  base hit {100*base_hit:.2f}%")
print(f"PPI print sessions with a prior bar: N={len(ppi_l)}  "
      f"{dt[0].date()}..{dt[-1].date()}")
print(f"  of which CPI printed on the eve (LIVE STATE): {int(L.sum())}")


def rep(x, lbl, p0=None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"cell": lbl, "N": 0}
    w = int((x > 0).sum())
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return {"cell": lbl, "N": len(x), "mean_bps": round(1e4 * x.mean(), 2),
            "hit": round(100 * w / len(x), 1),
            "t": round(x.mean() / (sd / np.sqrt(len(x))), 2),
            "signp": round(sign_test(w, len(x), p0 if p0 else base_hit), 4)}


print("\n  cell table (raw print-session close-to-close):")
print(pd.DataFrame([
    rep(v, "PARENT all PPI prints"),
    rep(v[L], "*** LIVE: CPI on the eve ***"),
    rep(v[~L], "no CPI on the eve"),
]).to_string(index=False))

print("\n  --- the 2018+ gate lift, re-derived ---")
cpi_next = sorted(p + 1 for p in CPI if 1 <= p + 1 < N and ok[p + 1])
vc = np.array([d1[q] for q in cpi_next])
yc = np.array([idx[q].year for q in cpi_next])
gp = np.array([q in PPI for q in cpi_next])
m18 = yc >= 2018
a, b = vc[m18 & gp], vc[m18 & ~gp]
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(pd.DataFrame([
    rep(vc[m18 & gp], "CPI+1 2018+ WITH a PPI (live)"),
    rep(vc[m18 & ~gp], "CPI+1 2018+ NO PPI (counterfactual)"),
    rep(vc[m18], "CPI+1 2018+ pooled"),
]).to_string(index=False))
print(f"  GATE LIFT = {1e4*(a.mean()-b.mean()):+.2f} bps   Welch t "
      f"{(a.mean()-b.mean())/se:+.3f}   [prior report: +42.70 bps, t +2.38]")

print("\n  --- WHAT IS THE COUNTERFACTUAL ARM, ACTUALLY? ---")
print("  In 2018+ the BLS almost always schedules PPI the session AFTER CPI.")
print("  So 'CPI+1 with no PPI' is not a neutral control: it is the months")
print("  where the scheduling was the OTHER way round. Check where PPI sat.")
rel = []
for q, hasppi in zip(cpi_next, gp):
    if idx[q].year < 2018 or hasppi:
        continue
    p_cpi = q - 1
    near = [k for k in range(-6, 7) if (p_cpi + k) in PPI]
    rel.append(near[0] if near else 99)
rel = np.array(rel)
print("  offset of the nearest PPI relative to the CPI session, no-PPI arm 2018+:")
print("  " + pd.Series(rel).value_counts().sort_index().to_string().replace("\n", "\n  "))
print(f"  -> PPI printed BEFORE the CPI in "
      f"{int((rel < 0).sum())} of {len(rel)} of the counterfactual months.")

# --------------------------------------------------------------------------
# B. MONTH: MECHANISM OR LABEL?
# --------------------------------------------------------------------------
print("\n" + "=" * 100)
print("B. IS THE MONTH EFFECT A MECHANISM OR A LABEL?")
print("=" * 100)

# every session touched by a PPI print is excluded from the controls
in_ev = np.zeros(N, bool)
for p in ppi_l:
    in_ev[p] = True
ctrl_ok = ok & ~in_ev
mo_all = np.array([d.month for d in idx])
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
tdom = ym.groupby(ym.values).cumcount().values + 1

print("\n1. TLT's OWN unconditional daily drift by month (PPI sessions removed)")
rows = []
for m in range(1, 13):
    s = d1[ctrl_ok & (mo_all == m)]
    e = v[mo == m]
    rows.append({"month": m, "ctrl_N": len(s), "ctrl_bps": round(1e4 * s.mean(), 2),
                 "ctrl_hit": round(100 * (s > 0).mean(), 1),
                 "ppi_N": len(e), "ppi_bps": round(1e4 * e.mean(), 2),
                 "excess_bps": round(1e4 * (e.mean() - s.mean()), 2)})
mdf = pd.DataFrame(rows)
print(mdf.to_string(index=False))
aug_ctrl = d1[ctrl_ok & (mo_all == 8)].mean()
print(f"\n  TLT's own August non-PPI daily drift = {1e4*aug_ctrl:+.2f} bps")
print(f"  PARENT August PPI sessions           = {1e4*v[mo==8].mean():+.2f} bps")
print(f"  PARENT August EXCESS over own month  = "
      f"{1e4*(v[mo==8].mean()-aug_ctrl):+.2f} bps")
print(f"  PARENT other-months EXCESS           = "
      f"{1e4*(v[mo!=8].mean()-d1[ctrl_ok & (mo_all!=8)].mean()):+.2f} bps")
print("  -> if the August excess is comparable to the other-months excess, the")
print("     parent's low August RAW mean is TLT's own August, not the event's.")
print(f"  corr(TLT own-month drift, PPI-session month mean) over 12 months = "
      f"{np.corrcoef(mdf['ctrl_bps'], mdf['ppi_bps'])[0,1]:+.3f}")

print("\n2. THE MONTH-MATCHED CONTROL applied to the live cell")
mctrl = {m: d1[ctrl_ok & (mo_all == m)].mean() for m in range(1, 13)}
ex_m = v - np.array([mctrl[m] for m in mo])
print(pd.DataFrame([
    rep(v, "raw parent"),
    rep(ex_m, "MONTH-matched excess: parent"),
    rep(ex_m[L], "*** MONTH-matched excess: LIVE CELL ***"),
    rep(ex_m[~L], "MONTH-matched excess: not live"),
    rep(ex_m[L & (mo == 8)], "MONTH-matched: live AND August (N=4)"),
]).to_string(index=False))
al, bl = ex_m[L], ex_m[~L]
sem = np.sqrt(al.var(ddof=1) / len(al) + bl.var(ddof=1) / len(bl))
print(f"  month-matched gate lift = {1e4*(al.mean()-bl.mean()):+.2f} bps  "
      f"Welch t {(al.mean()-bl.mean())/sem:+.2f}")
print(f"  bootstrap P(mean<=0) month-matched live cell = {bootstrap_p_le0(al):.3f}")

print("\n3. MONTH x TDOM double control (the strictest reasonable control)")
bk = {}
for m in range(1, 13):
    for j in range(1, 24):
        s = d1[ctrl_ok & (mo_all == m) & (tdom == j)]
        if len(s) >= 8:
            bk[(m, j)] = s.mean()
ex_mt, keep = [], []
for i, p in enumerate(ppi_l):
    k = (int(mo_all[p]), int(tdom[p]))
    if k in bk:
        ex_mt.append(v[i] - bk[k])
        keep.append(i)
ex_mt = np.array(ex_mt)
keep = np.array(keep)
Lk = L[keep]
print(pd.DataFrame([
    rep(ex_mt, "month x tdom excess: parent"),
    rep(ex_mt[Lk], "*** month x tdom excess: LIVE CELL ***"),
    rep(ex_mt[~Lk], "month x tdom excess: not live"),
]).to_string(index=False))
print(f"  coverage: {len(keep)} of {len(ppi_l)} prints had a >=8-obs bucket")

print("\n4. THE PRIOR REPORT'S CLAIM 'AUGUST IS THE NULL MONTH' -- checked")
means = np.array([v[mo == m].mean() for m in range(1, 13)])
order = np.argsort(means)
print("  parent monthly means, worst to best (bps):")
print("  " + "  ".join(f"{m+1}:{1e4*means[m]:+.0f}" for m in order))
print(f"  August's rank among the 12 = {int((means <= means[7]).sum())}/12  "
      f"(1 = worst).  December {1e4*means[11]:+.0f}, February {1e4*means[1]:+.0f} "
      f"are worse.")
aw = int((v[mo == 8] > 0).sum())
print(f"  August parent sign test vs base: {aw}/{len(v[mo==8])} wins, "
      f"P(<= {aw}) = {sign_test(len(v[mo==8])-aw, len(v[mo==8]), 1-base_hit):.4f}")
print("  -> the permutation P=0.025 is about the DISPERSION of all 12 months.")
print("     It is not evidence about August, which is mid-pack and whose own")
print("     hit rate is indistinguishable from TLT's base rate.")

# --------------------------------------------------------------------------
# C. THE 0-FOR-4 CROSS, WITH ITS MULTIPLICITY CHARGE
# --------------------------------------------------------------------------
print("\n" + "=" * 100)
print("C. THE AUGUST CROSS: 0-FOR-4 AT -0.85%")
print("=" * 100)
liv = L & (mo == 8)
print("  the four observations:")
for i in np.where(liv)[0]:
    p = ppi_l[i]
    print(f"    {dt[i].date()}  print {100*v[i]:+.3f}%   eve(CPI day) "
          f"{100*d1[p-1]:+.3f}%   midterm={bool(yr[i] % 4 == 2)}")
h = 0.636
print(f"\n  P(0 wins in 4 | cell hit 63.6%) = {(1-h)**4:.4f}")
print(f"  P(0 wins in 4 | TLT base {base_hit:.3f}) = {(1-base_hit)**4:.4f}")

cnt = pd.Series(mo[L]).value_counts().sort_index()
print("\n  live-cell observation count by month:")
print("  " + cnt.to_string().replace("\n", "\n  "))
p_any = 1.0
for m, n in cnt.items():
    p_any *= (1 - (1 - h) ** n)
print(f"\n  MULTIPLICITY CHARGE (the month was found, not pre-specified):")
print(f"    P(NO month is a 0-for-its-own-N shutout | 63.6%) = {p_any:.4f}")
print(f"    P(at least one month shows a shutout)           = {1-p_any:.4f}")

rng = np.random.default_rng(7)
sims_min, sims_shut = [], 0
res = v[L] - v[L].mean()
for _ in range(20000):
    sh = rng.permutation(v[L])
    mm = np.array([sh[mo[L] == m].mean() if (mo[L] == m).sum() else np.nan
                   for m in range(1, 13)])
    sims_min.append(np.nanmin(mm))
    if any(((sh[mo[L] == m] > 0).sum() == 0) for m in range(1, 13)
           if (mo[L] == m).sum() >= 4):
        sims_shut += 1
sims_min = np.array(sims_min)
obs_min = np.nanmin([v[L][mo[L] == m].mean() if (mo[L] == m).sum() else np.nan
                     for m in range(1, 13)])
print(f"\n  PERMUTATION within the live cell (labels shuffled across months):")
print(f"    observed WORST monthly mean = {100*obs_min:+.4f}%  (August)")
print(f"    P(some month is at least this bad by chance) = "
      f"{(sims_min <= obs_min).mean():.4f}")
print(f"    P(some month with N>=4 is a shutout by chance) = "
      f"{sims_shut/20000:.4f}")

print("\n  SYMMETRY CHECK -- the best month, same treatment:")
best_m = int(np.nanargmax([v[L][mo[L] == m].mean() if (mo[L] == m).sum() else np.nan
                           for m in range(1, 13)])) + 1
bm = v[L][mo[L] == best_m]
print(f"    best live-cell month = {best_m}: N={len(bm)} mean {100*bm.mean():+.3f}% "
      f"hit {100*(bm>0).mean():.0f}%")
print("    Nobody would ship a 4-observation month. The same rule forbids")
print("    killing a 55-observation cell on one.")

print("\n  SHRINKAGE: how much do 4 observations move a 55-observation mean?")
oth = v[L & (mo != 8)]
sd_c = v[L].std(ddof=1)
print(f"    live cell ex-August: N={len(oth)} mean {1e4*oth.mean():+.2f} bps "
      f"hit {100*(oth>0).mean():.1f}%")
print(f"    live cell all      : N={len(v[L])} mean {1e4*v[L].mean():+.2f} bps")
print(f"    one-session sd of the cell = {100*sd_c:.3f}%; the SE of a 4-obs mean "
      f"is {100*sd_c/2:.3f}%")
print(f"    the August mean sits {(v[liv].mean()-oth.mean())/(sd_c/2):+.2f} SE "
      f"from the ex-August mean.")
tau = oth.std(ddof=1) / np.sqrt(len(oth))
shrunk = (v[liv].mean() / (sd_c**2 / 4) + oth.mean() / tau**2) / \
         (1 / (sd_c**2 / 4) + 1 / tau**2)
print(f"    precision-weighted blend of the two = {1e4*shrunk:+.2f} bps")

print("\n" + "=" * 100)
print("D. THE CONDITIONER THE PRIOR CHECKER DID NOT CHARGE: MIDTERM")
print("=" * 100)
mid = (yr % 4) == 2
print(pd.DataFrame([
    rep(v[mid], "PARENT midterm"),
    rep(v[~mid], "PARENT non-midterm"),
    rep(v[L & mid], "LIVE cell midterm (today)"),
    rep(v[L & ~mid], "LIVE cell non-midterm"),
    rep(v[L & mid & (mo == 8)], "LIVE cell midterm AND August"),
]).to_string(index=False))
am, bm2 = v[L & mid], v[L & ~mid]
sm = np.sqrt(am.var(ddof=1) / len(am) + bm2.var(ddof=1) / len(bm2))
print(f"  midterm gap inside the live cell = {1e4*(am.mean()-bm2.mean()):+.2f} bps "
      f"Welch t {(am.mean()-bm2.mean())/sm:+.2f}")
print(f"  midterm gap in the PARENT        = "
      f"{1e4*(v[mid].mean()-v[~mid].mean()):+.2f} bps  <- ~zero, so the cell's")
print("     midterm dip has no parent-level support either.")
