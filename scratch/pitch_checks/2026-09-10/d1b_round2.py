"""D1 round 2 -- the questions round 1 left open, and a properly specified charge.

Round 1 established, on entry #33's OWN definition:
  * the parked arm reproduces EXACTLY: n=31, 25-6, +1.722%, t 4.943, sign p
    0.0004, boot 0.0000;
  * the -0.5x era alone is n=19, 14-5, +0.988%, t 3.091, sign p 0.032;
  * the placebo ladder puts the true k=-2 anchor FIRST of 11 in both samples;
  * every one of the 21 dial-covered armed anchors carried ma10(63d) <= 68.0
    with a MEDIAN of 1.2, against a live 87.66;
  * in the -0.5x era the (10,15] half of the band is DEAD (+0.188%, 4-3).

This round:
 A. CHARGE ON THE RIGHT STATISTIC. Round 1 maximised the MEAN over the grid,
    which is dominated by h=10 cells whose sd is 4x h=1's. Redo on the
    t-statistic, against BOTH defended cells' own t.
 B. THE DIAL, which is the live tape's only genuinely out-of-sample axis.
 C. SPY RESIDUAL with a standard error, and the equivalence to watchlist #35.
 D. DEFINITION NEIGHBOURS: the production absolute-range/504d percentile, the
    SPY-calendar reindex, and the (5,10] / (10,15] boundary.
 E. DECLUSTER ORDER + episode-level view (rule 4).
 F. The short ^VIX matched leg (settles the parked '+4.181%, t 4.552').
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, bootstrap_p_le0,
                       declusters)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
RELEVER = pd.Timestamp("2018-02-28")

px = close_panel(["SVXY", "^VIX", "^VIX3M", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]


def relpct(v):
    rng = (rolling_on_valid(v, lambda x: x.rolling(21).max())
           - rolling_on_valid(v, lambda x: x.rolling(21).min()))
    return rolling_on_valid(rng / rolling_on_valid(v, lambda x: x.rolling(21).mean()),
                            lambda x: x.rolling(252).rank(pct=True) * 100)


REL = relpct(vix)                                   # entry #33's definition
REL_SPY = relpct(px["^VIX"].reindex(cal))
absr = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
        - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL_PROD = rolling_on_valid(absr, lambda x: x.rolling(504).apply(
    lambda w: 100.0 * (w[:-1] < w[-1]).mean(), raw=True))

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALLP = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
posn = pd.Series(range(len(cal)), index=cal)
rows = []
for kind in KINDS:
    p, kept = anchor_positions(cal, EV[kind], -2)
    for i, ap in enumerate(p):
        d0 = kept[i]
        nxt = ALLP[ALLP > d0]
        rw = 99 if len(nxt) == 0 else int(posn.get(nxt[0], 0) - posn.get(d0, 0))
        rows.append({"anchor": cal[ap], "runway_td": rw})
F = pd.DataFrame(rows).set_index("anchor").sort_index().groupby(level=0).min()
F["rel"] = REL.reindex(F.index).values
F["rel_spy"] = REL_SPY.reindex(F.index).values
F["rel_prod"] = REL_PROD.reindex(F.index).values
for h in range(1, 11):
    F[f"s{h}"] = fwd_lag(px["SVXY"].dropna(), h, lag=1).reindex(F.index).values
    F[f"p{h}"] = fwd_lag(px["SPY"].dropna(), h, lag=1).reindex(F.index).values
    F[f"v{h}"] = (-fwd_lag(vix.dropna(), h, lag=1)).reindex(F.index).values
CL = F[F["runway_td"] >= 3]
MATCH = CL[CL["s1"].notna()]
DEF = MATCH[(MATCH["rel"] > 5) & (MATCH["rel"] <= 15)]
POST = DEF[DEF.index >= RELEVER]


def cc(v, label):
    v = pd.Series(v).dropna()
    st = summarize(v.values, label)
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        st["rec"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    return st


def tstat(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return np.nan
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


print("=" * 122)
print("A. CHARGED PERMUTATION ON THE T-STATISTIC")
print("=" * 122)
BANDS = [(0, 5), (5, 10), (10, 15), (5, 15), (5, 20), (5, 25), (10, 20),
         (15, 30), (0, 15), (0, 101)]
RWS = [1, 2, 3, 4]
HS = list(range(1, 11))
print(f"  disclosed walk = {len(BANDS)} bands x {len(RWS)} runway rungs x {len(HS)} "
      f"horizons x 2 vehicles = {len(BANDS)*len(RWS)*len(HS)*2} cells (floor)")
print("  DEFENDED statistic: the t of the MEAN 1-session lag-1 long-SVXY return")
print("  over anchors with rel in (5,15], runway>=3, k=-2.")
t_all = tstat(DEF["s1"].values)
t_post = tstat(POST["s1"].values)
print(f"    all-era defended t = {t_all:.3f} (n={len(DEF)})")
print(f"    -0.5x defended t   = {t_post:.3f} (n={len(POST)})")

# precompute anchor POSITIONS per (band, runway) once, then circularly shift
cpos = pd.Series(range(len(cal)), index=cal)
anchor_pos = cpos.reindex(F.index).values.astype(int)
rel_v = F["rel"].values
rw_v = F["runway_td"].values
svc_pos = set(cpos.reindex(px["SVXY"].dropna().index).dropna().astype(int).tolist())
post_start = int(cpos.reindex([cal[cal.searchsorted(RELEVER)]]).iloc[0])
sub = {}
for lo, hi in BANDS:
    for rw in RWS:
        m = (rel_v > lo) & (rel_v <= hi) & (rw_v >= rw) & ~np.isnan(rel_v)
        sub[(lo, hi, rw)] = anchor_pos[m]
        sub[("post", lo, hi, rw)] = anchor_pos[m & (anchor_pos >= post_start)]
svxy_a = {h: fwd_lag(px["SVXY"].dropna(), h, lag=1).reindex(cal).values for h in HS}
nvix_a = {h: (-fwd_lag(vix.dropna(), h, lag=1)).reindex(cal).values for h in HS}
n_all = len(cal)
rng = np.random.default_rng(7)
NB = 4000


def run_charge(defended_t, keyfn, label):
    unch = 0
    nulls = np.empty(NB)
    for b in range(NB):
        sh = int(rng.integers(21, n_all - 21))
        best = -9.9
        for lo, hi in BANDS:
            for rw in RWS:
                ap = sub[keyfn(lo, hi, rw)]
                if len(ap) < 10:
                    continue
                src = (ap + sh) % n_all
                for h in HS:
                    for nm, arrs in (("svxy", svxy_a), ("nvix", nvix_a)):
                        v = arrs[h][src]
                        v = v[~np.isnan(v)]
                        if len(v) < 10:
                            continue
                        tt = tstat(v)
                        if not np.isnan(tt):
                            if tt > best:
                                best = tt
                            if (nm == "svxy" and h == 1 and rw == 3
                                    and (lo, hi) == (5, 15) and tt >= defended_t):
                                unch += 1
        nulls[b] = best
    print(f"  {label}")
    print(f"    UNCHARGED p = {unch/NB:.4f}   CHARGED p = {(nulls >= defended_t).mean():.4f}")
    print(f"    null-max t distribution: p50 {np.percentile(nulls,50):.2f}  "
          f"p90 {np.percentile(nulls,90):.2f}  p95 {np.percentile(nulls,95):.2f}  "
          f"max {nulls.max():.2f}   vs defended {defended_t:.2f}")


run_charge(t_all, lambda lo, hi, rw: (lo, hi, rw), "ALL-ERA sample")
run_charge(t_post, lambda lo, hi, rw: ("post", lo, hi, rw), "-0.5x ERA sample")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("B. THE DIAL -- the live tape's out-of-sample axis")
print("=" * 122)
frag = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean()
live = float(ma.dropna().iloc[-1])
print(f"  live ma10(63d) = {live:.2f}; the armed cell's dial-covered anchors max at 68.0")
print("  1) ALL clear-calendar k=-2 anchors (any rel band), -0.5x era, by dial band:")
M = MATCH[MATCH.index >= RELEVER].copy()
M["dial"] = ma.reindex(M.index).values
rowsd = []
for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 101)):
    m = (M["dial"] >= lo) & (M["dial"] < hi)
    rowsd.append(cc(M.loc[m, "s1"].values, f"anchors, dial [{lo},{hi})"))
show(rowsd, "no armed episode has ever occurred above 68; this is the nearest read")
print("  2) EVERY -0.5x-era session, next-session long SVXY, by dial band "
      "(the widest available read on today's dial):")
s1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
J = pd.concat([s1.rename("s1"), ma.rename("dial")], axis=1).dropna()
J = J[J.index >= RELEVER]
rowsd = []
for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 101)):
    m = (J["dial"] >= lo) & (J["dial"] < hi)
    rowsd.append(cc(J.loc[m, "s1"].values, f"all sessions, dial [{lo},{hi})"))
show(rowsd, "-0.5x era, day level (overlap-free at h=1)")
print(f"  sessions in the -0.5x era at dial >= 80: {int((J['dial']>=80).sum())}; "
      f"at dial >= 85: {int((J['dial']>=85).sum())}")
print("  3) the entry's settled claim was measured on buckets <40 / 40-70 / 70+.")
print("     Its own 70+ bucket, re-measured on TODAY's data, and the 80+ slice:")
show([cc(J.loc[J["dial"] >= 70, "s1"].values, "all -0.5x sessions, dial >= 70"),
      cc(J.loc[J["dial"] >= 80, "s1"].values, "all -0.5x sessions, dial >= 80"),
      cc(J.loc[J["dial"] >= 85, "s1"].values, "all -0.5x sessions, dial >= 85")],
     "the live band")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("C. SPY RESIDUAL, with a standard error, and the #35 equivalence")
print("=" * 122)
for lab, sub_ in (("all-era", DEF), ("-0.5x era", POST)):
    s = sub_.dropna(subset=["s1", "p1"])
    lo = RELEVER if "0.5" in lab else pd.Timestamp("2000-01-01")
    hi = pd.Timestamp("2099-01-01") if "0.5" in lab else RELEVER
    sp = fwd_lag(px["SPY"].dropna(), 1, lag=1)
    sv = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
    j = pd.concat([sp.rename("p"), sv.rename("s")], axis=1).dropna()
    j = j[(j.index >= lo) & (j.index < hi)]
    bb, aa = np.polyfit(j["p"].values, j["s"].values, 1)
    resid = s["s1"].values - (aa + bb * s["p1"].values)
    se = resid.std(ddof=1) / np.sqrt(len(resid))
    print(f"  {lab}: n={len(s)}  era-wide SVXY = {100*aa:+.3f}% + {bb:.2f}*SPY")
    print(f"     cell raw {100*s['s1'].mean():+.3f}%  |  SPY leg {100*s['p1'].mean():+.3f}% "
          f"(era uncond {100*j['p'].mean():+.3f}%)  |  beta-charged ALPHA "
          f"{100*resid.mean():+.3f}pp  t {resid.mean()/se:+.2f}  "
          f"sign p {sign_test(int((resid>0).sum()), len(resid)):.4f} "
          f"({int((resid>0).sum())}-{int((resid<0).sum())})")
print("  the SPY leg on the SAME anchors is watchlist entry #35's cell:")
show([cc(DEF["p1"].values, "long SPY, same gate, all-era"),
      cc(POST["p1"].values, "long SPY, same gate, 2018-02-28+")],
     "#33 is a ~1.5x levered version of #35, which the verdicts blocked on the dial")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("D. DEFINITION NEIGHBOURS")
print("=" * 122)
print("  live readings: entry def %.3f | SPY-calendar %.3f | production(abs/504) %.3f"
      % (float(REL.dropna().iloc[-1]), float(REL_SPY.dropna().iloc[-1]),
         float(REL_PROD.dropna().iloc[-1])))
for nm, col in (("entry definition", "rel"), ("SPY-calendar reindex", "rel_spy"),
                ("production abs-range/504d", "rel_prod")):
    d5 = MATCH[(MATCH[col] > 5) & (MATCH[col] <= 15)]
    d5p = d5[d5.index >= RELEVER]
    show([cc(d5["s1"].values, f"{nm}: (5,15] all-era"),
          cc(d5p["s1"].values, f"{nm}: (5,15] -0.5x era")], nm)
print("  BOUNDARY: today reads 9.921 on the entry definition, 0.079 points below")
print("  the (10,15] edge. -0.5x era, the two halves and a sliding window:")
M5 = MATCH[MATCH.index >= RELEVER]
show([cc(M5.loc[(M5["rel"] > 5) & (M5["rel"] <= 10), "s1"].values, "(5,10] -0.5x"),
      cc(M5.loc[(M5["rel"] > 10) & (M5["rel"] <= 15), "s1"].values, "(10,15] -0.5x"),
      cc(M5.loc[(M5["rel"] > 8) & (M5["rel"] <= 12), "s1"].values, "(8,12] -0.5x (straddles it)")],
     "in the tradeable era the band is NOT a plateau")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("E. DECLUSTER ORDER + episode view (rule 4)")
print("=" * 122)
gap = 10
ftd = declusters(DEF.index, gap, cal)
allc = declusters(MATCH.index, gap, cal)
dtf = pd.DatetimeIndex([d for d in allc if 5 < float(F.loc[d, "rel"]) <= 15])
print(f"  filter-then-decluster (gap {gap}): N={len(ftd)}  "
      f"{100*DEF.loc[ftd,'s1'].mean():+.3f}%  t {tstat(DEF.loc[ftd,'s1'].values):.2f}")
print(f"  decluster-then-filter (gap {gap}): N={len(dtf)}  "
      f"{100*F.loc[dtf,'s1'].mean():+.3f}%  t {tstat(F.loc[dtf,'s1'].values):.2f}")
print(f"    FTD only: {[str(x.date()) for x in ftd.difference(dtf)]}")
print(f"    DTF only: {[str(x.date()) for x in dtf.difference(ftd)]}")
for g in (5, 10, 21, 42):
    e = declusters(DEF.index, g, cal)
    ep = declusters(POST.index, g, cal)
    print(f"  gap {g:2d}: all-era N={len(e):2d} {100*DEF.loc[e,'s1'].mean():+.3f}% "
          f"t {tstat(DEF.loc[e,'s1'].values):.2f}  |  -0.5x N={len(ep):2d} "
          f"{100*POST.loc[ep,'s1'].mean():+.3f}% t {tstat(POST.loc[ep,'s1'].values):.2f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 122)
print("F. THE SHORT ^VIX LEG (parked claim: +4.181%, t 4.552)")
print("=" * 122)
show([cc(DEF["v1"].values, "short ^VIX on the MATCHED 31 anchors"),
      cc(CL.loc[(CL["rel"] > 5) & (CL["rel"] <= 15), "v1"].values,
         "short ^VIX, full history (65 anchors)"),
      cc(POST["v1"].values, "short ^VIX, -0.5x era window (19)")],
     "which subset produced the parked number")
