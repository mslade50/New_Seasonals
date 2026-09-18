"""kD round 2: c10 LONG dollar after the FOMC decision close.

A. decluster/concentration (top-2 share, drop best 2, drop best year, drop worst year)
B. horizon neighbours h=1..5, raw + tdom-matched excess
C. placebo ladder k=-5..+5 (h=3), rank of k=0 on raw mean AND tdomX
D. family charge: permutation of the 30-cell parent family kA walked
   (TLT, DX-Y.NYB, GLD) x (long, short) x h=1..5 on kA's decision sample
   (decisions with a 252-bar ^TNX eve, N=204 on DX). Statistic: max over cells
   of |t of raw mean| (sign flip makes the 30 signed cells 15 |t| cells);
   observed = DX long h=3 t. Null: each decision replaced by a random
   same-tdom date from non-FOMC-window sessions within +/-252td, same date
   draw shared across vehicles (preserves cross-vehicle correlation). A
   second run uses tdom-matched excess t as the statistic. Sidak bound too.
E. mechanism (a): dollar announcement-session return (eve close -> D close),
   raw and tdom-matched
F. mechanism (b): slope of h=3 post return on the announcement-session
   return, terciles, vs the unconditional all-days slope of r(t,t+3) on
   r(t-1,t) (generic DX reversal); plus the pre-meeting 3d run-up D-3..D
G. reference class: same h=3 long-dollar construction after CPI and NFP
   release closes, tdom-matched
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from math import erf, sqrt

rng = np.random.default_rng(7)
pxd = load_prices(["DX-Y.NYB", "UUP", "TLT", "GLD", "^TNX"])
tnx = pxd["^TNX"]["Close"].dropna()
mx = tnx.rolling(252).max()
fomc_all = load_events(["fomc_decision"])["date"]
fomc_all = pd.DatetimeIndex(fomc_all[fomc_all <= pd.Timestamp("2026-09-15")])
fomcA = pd.DatetimeIndex([d for d in fomc_all
                          if len(tnx.index[tnx.index < d]) and
                          not np.isnan(mx.get(tnx.index[tnx.index < d][-1], np.nan))])
dx = pxd["DX-Y.NYB"]["Close"].dropna()
idx = dx.index


def tdom_of(ix):
    ym = pd.Series(ix.year * 100 + ix.month, index=ix)
    return ym.groupby(ym.values).cumcount().values + 1


TD = tdom_of(idx)
POS = pd.Series(np.arange(len(idx)), index=idx)


def excl_mask(dates, n, pre=5, post=5):
    m = np.zeros(n, bool)
    for d in dates:
        if d in POS.index:
            p = POS[d]
            m[max(0, p - pre):p + post + 1] = True
    return m


def fwd(s, h):
    return (s.shift(-h) / s - 1.0).values


def tdomX(r, dp, excl):
    ok = ~np.isnan(r)
    b = {j: np.nanmean(r[(TD == j) & ~excl & ok]) for j in np.unique(TD)}
    return np.array([r[p] - b[TD[p]] for p in dp]), b


def tstat(v):
    v = v[~np.isnan(v)]
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


def row(lbl, v, x=None):
    o = summarize(v, lbl)
    wn = int((v > 0).sum())
    o["rec"] = f"{wn}-{len(v) - wn}"
    o["sign_p"] = sign_test(wn, len(v))
    if x is not None:
        o["tdomX"] = 100 * x.mean()
        o["X_t"] = tstat(x)
        we = int((x > 0).sum())
        o["X_sign_p"] = sign_test(we, len(x))
    return o


EX = excl_mask(fomc_all, len(idx))
dates = [d for d in fomc_all if d in POS.index]
dp = np.array([POS[d] for d in dates])
r3 = fwd(dx, 3)
ok3 = ~np.isnan(r3[dp])
dates3 = pd.DatetimeIndex(np.array(dates)[ok3])
dp3 = dp[ok3]
v3 = r3[dp3]
x3, _ = tdomX(r3, dp3, EX)

# ---------------- A. concentration
print("=" * 90, "\nA. concentration (DX long h=3, all decisions N=%d)" % len(v3))
print(" ", cluster_note(dates3, v3))
o = np.argsort(-v3)
yrs = dates3.year.values
by = pd.Series(v3).groupby(yrs).sum().sort_values()
rows = [row("all", v3, x3), row("drop best 2", v3[o[2:]], x3[o[2:]]),
        row(f"drop best yr {by.index[-1]}", v3[yrs != by.index[-1]], x3[yrs != by.index[-1]]),
        row(f"drop worst yr {by.index[0]}", v3[yrs != by.index[0]], x3[yrs != by.index[0]])]
show(rows)
print("  yearly sums (pp):", {int(k): round(100 * val, 2) for k, val in by.items()})
pos_years = int((by > 0).sum())
print(f"  positive years {pos_years}/{len(by)}")

# ---------------- B. horizons
hrows = []
for h in range(1, 6):
    r = fwd(dx, h)
    okh = ~np.isnan(r[dp])
    x, _ = tdomX(r, dp[okh], EX)
    hrows.append(row(f"h={h}", r[dp[okh]], x))
show(hrows, "B. horizon neighbours, DX long, entry D close")

# ---------------- C. placebo ladder
lad = []
for k in range(-5, 6):
    pk = dp + k
    pk = pk[(pk >= 0) & (pk < len(idx))]
    pk = pk[~np.isnan(r3[pk])]
    x, _ = tdomX(r3, pk, EX)
    lad.append({"k": k, "n": len(pk), "mean": 100 * r3[pk].mean(), "tdomX": 100 * x.mean(),
                "t": tstat(r3[pk]), "X_t": tstat(x), "hit": 100 * (r3[pk] > 0).mean()})
lad = pd.DataFrame(lad)
k0 = lad[lad.k == 0].iloc[0]
print("\nC. placebo ladder h=3 (k = entry offset from decision close)")
print(lad.round(3).to_string(index=False))
print(f"  k=0 rank by raw mean {int((lad['mean'] > k0['mean']).sum()) + 1}/11, "
      f"by tdomX {int((lad['tdomX'] > k0['tdomX']).sum()) + 1}/11")
nov = lad[lad.k.isin([-5, -4, 3, 4, 5])]
print(f"  non-overlapping-with-k0 rungs (-5,-4,+3,+4,+5): mean tdomX {nov.tdomX.mean():+.3f}%")

# ---------------- D. family permutation
VEH = {"TLT": pxd["TLT"]["Close"].dropna(), "DX": dx, "GLD": pxd["GLD"]["Close"].dropna()}
RET = {}
for vk, s in VEH.items():
    s2 = s.reindex(idx)  # align on DX calendar; NaN where vehicle missing
    for h in range(1, 6):
        RET[(vk, h)] = (s2.shift(-h) / s2 - 1.0).values
EXF = excl_mask(fomc_all, len(idx))
BUCK = {}
for key, r in RET.items():
    ok = ~np.isnan(r)
    BUCK[key] = np.array([np.nanmean(r[(TD == j) & ~EXF & ok]) if ((TD == j) & ~EXF & ok).any()
                          else np.nan for j in range(1, 24)])
dpA = np.array([POS[d] for d in fomcA if d in POS.index])


def fam_stats(pp):
    tr, tx = {}, {}
    for key, r in RET.items():
        v = r[pp]
        m = ~np.isnan(v)
        v = v[m]
        tr[key] = tstat(v)
        x = v - BUCK[key][TD[pp[m]] - 1]
        tx[key] = tstat(x)
    return tr, tx


obs_r, obs_x = fam_stats(dpA)
print("\nD. family: observed t (raw) per cell, long sign (short = mirror), kA sample N_dx=%d" % len(dpA))
print(pd.DataFrame({"raw_t": obs_r, "tdomX_t": obs_x}).round(2).to_string())
T_obs_r = obs_r[("DX", 3)]
T_obs_x = obs_x[("DX", 3)]
max_obs_r = max(abs(v) for v in obs_r.values())
cand = np.where(~EXF & (np.arange(len(idx)) < len(idx) - 6))[0]
cand_by_td = {j: cand[TD[cand] == j] for j in range(1, 24)}
NPERM = 2000
mr, mxx, MDX = np.empty(NPERM), np.empty(NPERM), np.empty(NPERM)
WHO = {}
for i in range(NPERM):
    pp = []
    for p in dpA:
        c = cand_by_td[TD[p]]
        c = c[np.abs(c - p) <= 252]
        pp.append(rng.choice(c))
    pp = np.array(pp)
    tr, tx = fam_stats(pp)
    mr[i] = max(abs(v) for v in tr.values())
    mxx[i] = max(abs(v) for v in tx.values())
    arg = max(tr, key=lambda kk: abs(tr[kk]))
    WHO[arg[0]] = WHO.get(arg[0], 0) + 1
    MDX[i] = max(abs(tr[("DX", hh)]) for hh in range(1, 6))
p_r = (1 + (mr >= abs(T_obs_r)).sum()) / (NPERM + 1)
p_x = (1 + (mxx >= abs(T_obs_x)).sum()) / (NPERM + 1)
p_x5 = (1 + (mxx >= abs(obs_x[("DX", 5)])).sum()) / (NPERM + 1)
print(f"  raw-null diagnosis: vehicle holding the null max|t_raw| {WHO}; DX-only 5-cell null "
      f"max|t_raw| median {np.median(MDX):.2f}, P(>= {abs(T_obs_r):.2f}) {(1 + (MDX >= abs(T_obs_r)).sum()) / (NPERM + 1):.4f}")
print(f"  if h=5 is pitched instead: P(max|t_tdomX| >= {abs(obs_x[('DX', 5)]):.2f}) = {p_x5:.4f}")
p1 = 2 * (1 - 0.5 * (1 + erf(abs(T_obs_r) / sqrt(2))))
print(f"  observed DX long h3: raw t {T_obs_r:.3f} (family max |t| {max_obs_r:.3f}), tdomX t {T_obs_x:.3f}")
print(f"  permutation (tdom-matched random dates, {NPERM} draws): P(max|t_raw| over 30 cells >= "
      f"{abs(T_obs_r):.2f}) = {p_r:.4f};  P(max|t_tdomX| >= {abs(T_obs_x):.2f}) = {p_x:.4f}")
print(f"  null max|t| quantiles raw 50/90/95: {np.percentile(mr, [50, 90, 95]).round(2)}; "
      f"tdomX: {np.percentile(mxx, [50, 90, 95]).round(2)}")
print(f"  Sidak on raw t: p_single(2-sided normal) {p1:.4f} -> 30 cells {1 - (1 - p1) ** 30:.3f}, "
      f"150 cells (x5 gate rungs) {1 - (1 - p1) ** 150:.3f}")

# ---------------- E. mechanism (a): announcement session
r1 = (dx / dx.shift(1) - 1.0).values
ok1 = ~np.isnan(r1[dp])
xa, _ = tdomX(r1, dp[ok1], EX)
va = r1[dp[ok1]]
yrsa = pd.DatetimeIndex(np.array(dates)[ok1]).year.values
show([row("ALL eve->D (dollar)", va, xa),
      row("pre-2008", va[yrsa < 2008], xa[yrsa < 2008]),
      row("2008-2017", va[(yrsa >= 2008) & (yrsa < 2018)], xa[(yrsa >= 2008) & (yrsa < 2018)]),
      row("2018+", va[yrsa >= 2018], xa[yrsa >= 2018])],
     "E. mechanism (a): dollar return on the announcement session (eve close -> D close)")
rpre = fwd(dx, 3)
pre_p = dp - 3
xp, _ = tdomX(rpre, pre_p, EX)
show([row("pre-meeting D-3 close -> D close", rpre[pre_p], xp)], "E2. pre-meeting 3d run-in (dollar)")

# ---------------- F. mechanism (b): slope
m = ok1 & ok3
a_ = r1[dp[m]]
b_ = r3[dp[m]]
slope = np.polyfit(a_, b_, 1)[0]
cc = np.corrcoef(a_, b_)[0, 1]
tsl = cc * np.sqrt((len(a_) - 2) / (1 - cc ** 2))
q = np.quantile(a_, [1 / 3, 2 / 3])
terc = [row("weak ann. dollar (T1)", b_[a_ <= q[0]]), row("mid (T2)", b_[(a_ > q[0]) & (a_ <= q[1])]),
        row("strong ann. dollar (T3)", b_[a_ > q[1]])]
show(terc, "F. mechanism (b): h=3 post return by announcement-session tercile")
allok = ~np.isnan(r1) & ~np.isnan(r3)
ga, gb = r1[allok], r3[allok]
gsl = np.polyfit(ga, gb, 1)[0]
gcc = np.corrcoef(ga, gb)[0, 1]
print(f"  FOMC slope of post-h3 on ann-session: {slope:+.3f} (corr {cc:+.3f}, t {tsl:+.2f}, n {len(a_)})")
print(f"  unconditional all-days slope r(t,t+3) on r(t-1,t): {gsl:+.3f} (corr {gcc:+.3f}, n {len(ga)})")
# does the pre-meeting 3d run-in predict?
m2 = ~np.isnan(rpre[pre_p]) & ok3
c2 = np.corrcoef(rpre[pre_p][m2], r3[dp][m2])[0, 1]
print(f"  corr(pre-meeting D-3..D dollar, post h3): {c2:+.3f} n {m2.sum()}")
# residual edge after removing the generic reversal
resid = b_ - gsl * a_
print(f"  post-h3 mean {100 * b_.mean():+.3f}% ; after subtracting generic reversal x ann-session: "
      f"{100 * resid.mean():+.3f}%")

# ---------------- G. reference class CPI / NFP
print("\nG. reference class: long DX h=3 from the release-day close")
ref = []
for kind in ("cpi", "nfp", "ppi"):
    ev = load_events([kind])["date"]
    ev = pd.DatetimeIndex(ev[(ev <= pd.Timestamp("2026-09-15")) & (ev >= idx[0])]).unique()
    ev = [d for d in ev if d in POS.index and d not in set(fomc_all)]
    ep = np.array([POS[d] for d in ev])
    ep = ep[~np.isnan(r3[ep])]
    exk = excl_mask(pd.DatetimeIndex(idx[ep]), len(idx)) | EX
    xk, _ = tdomX(r3, ep, exk)
    yk = idx[ep].year.values
    ref.append(row(f"{kind} N={len(ep)}", r3[ep], xk))
    ref.append(row(f"{kind} 2018+", r3[ep][yk >= 2018], xk[yk >= 2018]))
    ann = r1[ep]
    xa2, _ = tdomX(r1, ep, exk)
    ref.append(row(f"{kind} release session (dollar)", ann[~np.isnan(ann)], xa2[~np.isnan(ann)]))
ref.append(row("FOMC N=%d" % len(v3), v3, x3))
show(ref)
