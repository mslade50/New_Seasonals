"""C8 -- XLE 5d return rank <= 20 inside a 21d rank >= 65 thrust, within 3% of
its trailing-252 high.  "The pullback inside the thrust, near a high."

Two adjacent registry cells bracket this one and it has to be shown to be
neither:
  (a) 2026-08-17 "5-day complex-wide energy THRUST into a 52w high" -- rank form
      +0.715% is an intersection artifact; C8 is the opposite side.
  (b) watchlist 25 "sector washing out (5d rank <= 5) within 5% of its 52w high"
      pays a POOLED +0.900% at h=7 across nine SPDRs, XLI ranks 2 of 9, Cochran
      Q p 0.789 (homogeneous).  C8 is a LOOSER rung (5d rank <= 20) of that
      family, so the honest form is POOLED OR NOTHING.

Therefore this script runs the nine-SPDR family, reports Cochran Q / I^2 /
XLE's rank by |t| / permutation max-of-9, tests whether the 21d-thrust clause
adds anything over the bare 5-day washout, tests whether the near-high clause
is a bull-tape selector (fraction of trigger days above SPY's 200d vs base
rate -- the registry measured 100.0% vs 71.6% on the tighter rung, a fatal
tell), and strips the crude beta.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (load_prices, pct_rank, fwd_lag, summarize, show,  # noqa: E402
                       declusters, local_control, cluster_note, sign_test,
                       bootstrap_p_le0)

SPDR = ["XLE", "XLB", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLF"]
EXTRA = ["SPY", "USO", "CL=F", "XOP"]
ASOF = pd.Timestamp("2026-08-27")
HS = (1, 2, 3, 5, 7, 10)
R5_MAX, R21_MIN, NEAR = 20.0, 65.0, 0.03

px = load_prices(SPDR + EXTRA)
S = {t: px[t]["Close"].dropna().loc[:ASOF] for t in px}
have_crude = "CL=F" in S and len(S["CL=F"]) > 1000
crude = "CL=F" if have_crude else ("USO" if "USO" in S else None)
print(f"crude proxy: {crude}")

spy = S["SPY"]
spy200 = spy.rolling(200).mean()
spy_above200 = (spy > spy200)


def masks(t):
    s = S[t]
    r5 = pct_rank(s, 5)
    r21 = pct_rank(s, 21)
    dist = s / s.rolling(252).max() - 1.0
    m_full = (r5 <= R5_MAX) & (r21 >= R21_MIN) & (dist >= -NEAR)
    m_bare = (r5 <= R5_MAX)                                  # no thrust, no high
    m_nothrust = (r5 <= R5_MAX) & (dist >= -NEAR)            # drop thrust clause
    m_nohigh = (r5 <= R5_MAX) & (r21 >= R21_MIN)             # drop near-high clause
    return dict(full=m_full, bare=m_bare, no_thrust=m_nothrust, no_high=m_nohigh), \
        dict(r5=r5, r21=r21, dist=dist)


print("\n=== C8.0  live state, all nine SPDRs ===")
MK, AUX = {}, {}
for t in SPDR:
    MK[t], AUX[t] = masks(t)
    a = AUX[t]
    print(f"  {t:<5} 5d rank {a['r5'].iloc[-1]:5.1f}  21d rank {a['r21'].iloc[-1]:5.1f}  "
          f"dist52wH {100*a['dist'].iloc[-1]:+6.2f}%  FULL MASK LIVE="
          f"{bool(MK[t]['full'].iloc[-1])}")


def cellstats(t, mkey, h, label=None, min_gap=None):
    s = S[t]
    r = fwd_lag(s, h, 1)
    m = MK[t][mkey].reindex(s.index, fill_value=False)
    dts = s.index[m.values & r.notna().values]
    epi = declusters(dts, min_gap or h, s.index)
    d = summarize(r.loc[dts].values, label or f"{t} {mkey} day")
    e = summarize(r.loc[epi].values, f"{t} {mkey} epi")
    return d, e, dts, epi, r


# --------------------------------------------------------------------------
print("\n=== C8.1  XLE alone: full cell vs gate-attribution controls ===")
for h in HS:
    rows = []
    for mk in ("full", "no_thrust", "no_high", "bare"):
        d, e, dts, epi, r = cellstats("XLE", mk, h)
        e["n_days"] = len(dts)
        rows.append(e)
    r = fwd_lag(S["XLE"], h, 1)
    rows.append(summarize(r.dropna().values, "XLE all days"))
    _, _, dtsF, _, _ = cellstats("XLE", "full", h)
    loc = local_control(S["XLE"].index[r.notna().values], dtsF)
    rows.append(summarize(r.loc[loc].values, "CTRL-c local +/-126td"))
    show(rows, f"XLE gate attribution, h={h}")

# --------------------------------------------------------------------------
print("\n=== C8.2  THE FAMILY: same rule on all nine SPDRs, pooled + per-name ===")
for h in (3, 5, 7, 10):
    rows, pooled, ts, ses, means = [], [], [], [], []
    for t in SPDR:
        d, e, dts, epi, r = cellstats(t, "full", h)
        if e["n"] == 0:
            rows.append({"label": t, "n": 0})
            continue
        e["label"] = t
        e["n_days"] = len(dts)
        base = fwd_lag(S[t], h, 1).dropna()
        e["ctl_pct"] = round(100 * base.mean(), 3)
        e["edge_pct"] = round(e["mean_pct"] - 100 * base.mean(), 3)
        rows.append(e)
        v = r.loc[epi].values
        pooled.append(v)
        means.append(v.mean())
        ses.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
        ts.append(e["t"])
    allv = np.concatenate([p for p in pooled if len(p)])
    rows.append(summarize(allv, f"POOLED nine SPDRs (N={len(allv)})"))
    show(rows, f"family, h={h}")
    # Cochran Q / I^2 on the per-name episode means
    mu = np.asarray(means, float)
    se = np.asarray(ses, float)
    ok = np.isfinite(mu) & np.isfinite(se) & (se > 0)
    w = 1.0 / se[ok] ** 2
    mbar = float((w * mu[ok]).sum() / w.sum())
    Q = float((w * (mu[ok] - mbar) ** 2).sum())
    dfq = int(ok.sum()) - 1
    from math import erf, exp
    # chi-square survival via Wilson-Hilferty
    if dfq > 0:
        z = ((Q / dfq) ** (1 / 3) - (1 - 2 / (9 * dfq))) / np.sqrt(2 / (9 * dfq))
        pQ = 0.5 * (1 - erf(z / np.sqrt(2)))
    else:
        pQ = np.nan
    I2 = max(0.0, (Q - dfq) / Q) * 100 if Q > 0 else 0.0
    tabs = np.abs(np.asarray(ts, float))
    xle_rank = int(np.sum(tabs > tabs[SPDR.index("XLE")])) + 1
    print(f"  Cochran Q = {Q:.2f} on df={dfq}, p = {pQ:.3f}   I^2 = {I2:.1f}%   "
          f"XLE ranks {xle_rank} of {len(SPDR)} by |t|")
    print(f"  pooled mean {100*allv.mean():+.3f}%  t = "
          f"{allv.mean()/(allv.std(ddof=1)/np.sqrt(len(allv))):+.2f}  "
          f"(names ARE homogeneous -> pooled or nothing)"
          if pQ > 0.10 else
          f"  pooled mean {100*allv.mean():+.3f}%  HETEROGENEOUS (p={pQ:.3f})")

# --------------------------------------------------------------------------
print("\n=== C8.3  permutation max-of-9: is XLE's |t| just the best of nine? ===")
H_PERM = 7
rng = np.random.default_rng(7)
obs_t = []
for t in SPDR:
    _, e, _, _, _ = cellstats(t, "full", H_PERM)
    obs_t.append(abs(e["t"]) if e["n"] > 1 else 0.0)
obs_max = float(np.nanmax(obs_t))
xle_t = obs_t[SPDR.index("XLE")]
perm = []
for _ in range(400):
    mx = 0.0
    for t in SPDR:
        s = S[t]
        r = fwd_lag(s, H_PERM, 1)
        m = MK[t]["full"].reindex(s.index, fill_value=False).values
        k = int(rng.integers(60, len(s) - 60))
        mr = np.roll(m, k)                       # circular block shift of the mask
        dts = s.index[mr & r.notna().values]
        epi = declusters(dts, H_PERM, s.index)
        if len(epi) < 3:
            continue
        v = r.loc[epi].values
        sd = v.std(ddof=1)
        if sd > 0:
            tt = abs(v.mean() / (sd / np.sqrt(len(v))))
            mx = max(mx, tt)
    perm.append(mx)
perm = np.asarray(perm)
print(f"  h={H_PERM}: observed max|t| over nine names = {obs_max:.2f} "
      f"(XLE's own |t| = {xle_t:.2f})")
print(f"  circular-shift null max|t|: median {np.median(perm):.2f}  p95 "
      f"{np.percentile(perm,95):.2f}   P(null max >= XLE |t|) = "
      f"{(perm>=xle_t).mean():.3f}   P(null max >= observed max) = "
      f"{(perm>=obs_max).mean():.3f}")

# --------------------------------------------------------------------------
print("\n=== C8.4  is the near-high clause a bull-tape selector? ===")
base_rate = float(spy_above200.dropna().mean())
print(f"  base rate SPY > 200d over the full sample = {100*base_rate:.1f}%")
for t in SPDR:
    for mk in ("full", "bare"):
        m = MK[t][mk].reindex(spy_above200.index, fill_value=False)
        sel = spy_above200[m.values & spy_above200.notna().values]
        if len(sel) == 0:
            continue
        if t == "XLE" or mk == "full":
            print(f"  {t:<5} {mk:<9} trigger days above SPY 200d: "
                  f"{100*sel.mean():5.1f}%  (n={len(sel)})")

# --------------------------------------------------------------------------
print("\n=== C8.5  crude-beta residual: is this levered crude with a producer label? ===")
if crude:
    cs = S[crude]
    idx = S["XLE"].index.intersection(cs.index)
    xr = S["XLE"].reindex(idx).pct_change()
    cr = cs.reindex(idx).pct_change()
    ok = xr.notna() & cr.notna()
    beta_c = float(np.polyfit(cr[ok], xr[ok], 1)[0])
    print(f"  XLE daily beta on {crude} = {beta_c:.3f}  (n={int(ok.sum())})")
    for h in (5, 7, 10):
        rx = fwd_lag(S["XLE"].reindex(idx), h, 1)
        rc = fwd_lag(cs.reindex(idx), h, 1)
        resid = rx - beta_c * rc
        m = MK["XLE"]["full"].reindex(idx, fill_value=False)
        dts = idx[m.values & resid.notna().values]
        epi = declusters(dts, h, idx)
        rows = [summarize(rx.loc[epi].dropna().values, f"XLE raw h={h}"),
                summarize(rc.loc[epi].dropna().values, f"{crude} raw h={h}"),
                summarize(resid.loc[epi].dropna().values,
                          f"XLE ex-crude residual h={h}"),
                summarize(resid.dropna().values, f"residual ALL days h={h}")]
        show(rows, f"crude attribution, h={h}")
else:
    print("  no crude series available in the cache")

# --------------------------------------------------------------------------
print("\n=== C8.6  XLE episodes / concentration / era / midterm (h=7) ===")
d, e, dts, epi, r = cellstats("XLE", "full", 7)
v = r.loc[epi].values
print(f"  {len(dts)} days -> {len(epi)} episodes; span "
      f"{epi[0].date()} .. {epi[-1].date()}")
print(f"  {cluster_note(epi, v)}")
w = int((v > 0).sum())
print(f"  record {w}-{len(v)-w}  sign p = {sign_test(max(w,len(v)-w), len(v)):.3f}  "
      f"bootstrap P(mean<=0) = {bootstrap_p_le0(v):.3f}")
yrs = pd.DatetimeIndex(epi).year
show([summarize(v[yrs < 2018], "pre-2018"), summarize(v[yrs >= 2018], "2018+"),
      summarize(v[yrs % 4 == 2], "midterm"), summarize(v[yrs % 4 != 2], "non-mid")],
     "era + midterm (XLE h=7 episodes)")

print("\n=== C8.7  threshold neighbours on XLE (h=7 episodes) ===")
s = S["XLE"]
r5, r21 = pct_rank(s, 5), pct_rank(s, 21)
dist = s / s.rolling(252).max() - 1.0
rr = fwd_lag(s, 7, 1)
rows = []
for a in (5, 10, 15, 20, 25, 30):
    for b in (50, 65, 80):
        m = (r5 <= a) & (r21 >= b) & (dist >= -NEAR)
        dts2 = s.index[m.reindex(s.index, fill_value=False).values & rr.notna().values]
        if len(dts2) == 0:
            continue
        ep2 = declusters(dts2, 7, s.index)
        x = summarize(rr.loc[ep2].values, f"r5<={a} r21>={b}")
        x["n_days"] = len(dts2)
        rows.append(x)
show(rows, "XLE r5 x r21 grid")
rows = []
for nh in (0.01, 0.02, 0.03, 0.05, 0.10, 1.0):
    m = (r5 <= R5_MAX) & (r21 >= R21_MIN) & (dist >= -nh)
    dts2 = s.index[m.reindex(s.index, fill_value=False).values & rr.notna().values]
    ep2 = declusters(dts2, 7, s.index)
    x = summarize(rr.loc[ep2].values, f"within {100*nh:.0f}% of high")
    x["n_days"] = len(dts2)
    rows.append(x)
show(rows, "XLE near-high threshold (h=7)")
print("\n  cost bar: sector ETF ~6 bps round trip. Need >= 30 bps episode mean.")
