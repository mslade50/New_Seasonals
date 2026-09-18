"""kR round 2 on c10 (long dollar MOC on the FOMC decision close).

A. The live intersection: SEP meeting AND dollar bid into the eve (DX r5 >= 75,
   UUP r5 >= 75). Within-SEP difference test (label permutation, 20000, seed
   11), interaction test (SEP hi-lo minus nonSEP hi-lo, r5 label permuted
   within the SEP/nonSEP strata), episode list, announcement-session return
   in that slice.
B. Permutation re-run of kD's family charge with a new fixed seed
   (20260916, 2000 draws, vectorized). Statistic = max over cells of |t of
   tdom-matched excess|; p = P(null max >= |observed t of THIS cell|).
   B1 = kD's 30 cells (TLT/DX/GLD x sign x h1..5, ALL decisions N=204).
   B2 = the rungs kA actually walked with n >= 30 (ALL, thr20, NOT_w2.0).
   B3 = all 8 kA rungs (n >= 5), subsets held fixed per decision.
C. Sizing: h=5 return in units of eve Wilder-14 ATR (as % of price).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pxd = load_prices(["DX-Y.NYB", "UUP", "TLT", "GLD", "^TNX"])
FOMC = load_events(["fomc_decision"])["date"]
FOMC = pd.DatetimeIndex(FOMC[FOMC <= pd.Timestamp("2026-09-15")])
SEP_EARLY = pd.DatetimeIndex(["2011-04-27", "2011-06-22", "2011-11-02", "2012-01-25",
                              "2012-04-25", "2012-06-20", "2012-09-13", "2012-12-12"])


def is_sep(d):
    return (d in SEP_EARLY) or (d.year >= 2013 and d.month in (3, 6, 9, 12))


def tdom_of(ix):
    ym = pd.Series(ix.year * 100 + ix.month, index=ix)
    return ym.groupby(ym.values).cumcount().values + 1


def tst(x):
    x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def row(lbl, v, x):
    n = len(x)
    if n == 0:
        return {"label": lbl, "n": 0}
    we = int((x > 0).sum())
    return {"label": lbl, "n": n, "raw%": 100 * v.mean(), "tdomX%": 100 * x.mean(), "X_t": tst(x),
            "X_rec": f"{we}-{n - we}", "X_sign_p": sign_test(we, n)}


def cell(tkr, h):
    s = pxd[tkr]["Close"].dropna()
    idx = s.index
    td = tdom_of(idx)
    pos = pd.Series(np.arange(len(s)), index=idx)
    dd = pd.DatetimeIndex([d for d in FOMC if d in pos.index])
    dp = np.array([pos[d] for d in dd])
    ex = np.zeros(len(s), bool)
    for p in dp:
        ex[max(0, p - 5):p + 6] = True
    r = (s.shift(-h) / s - 1.0).values
    ok = ~np.isnan(r)
    b = {j: np.nanmean(r[(td == j) & ~ex & ok]) for j in np.unique(td)}
    k = ok[dp]
    dp = dp[k]
    return dd[k], dp, r[dp], np.array([r[p] - b[td[p]] for p in dp]), s


def eve_val(ser, d):
    e = ser.index[ser.index < d]
    return ser[e[-1]] if len(e) else np.nan


# ------------------------------------------------------------------ A
print("=" * 100, "\nA. LIVE INTERSECTION: SEP x dollar bid into the eve\n" + "=" * 100)
dx_r5 = pct_rank(pxd["DX-Y.NYB"]["Close"].dropna(), 5)
uu_r5 = pct_rank(pxd["UUP"]["Close"].dropna(), 5)
rng = np.random.default_rng(11)
for tkr in ("DX-Y.NYB", "UUP"):
    for rk_name, rk in (("DX r5", dx_r5), ("UUP r5", uu_r5)):
        for h in (3, 5):
            dd, dp, v, x, s = cell(tkr, h)
            r5 = np.array([eve_val(rk, d) for d in dd])
            g = ~np.isnan(r5)
            dd, v, x, r5 = dd[g], v[g], x[g], r5[g]
            sep = np.array([is_sep(d) for d in dd])
            hi = r5 >= 75
            a, b = x[sep & hi], x[sep & ~hi]
            d_obs = a.mean() - b.mean()
            c, e = x[~sep & hi], x[~sep & ~hi]
            i_obs = d_obs - (c.mean() - e.mean())
            NP = 20000
            cnt_d = cnt_i = 0
            xs, xn = x[sep], x[~sep]
            hs, hn = hi[sep], hi[~sep]
            for _ in range(NP):
                ps = rng.permutation(hs)
                pn = rng.permutation(hn)
                dd_ = xs[ps].mean() - xs[~ps].mean()
                cnt_d += abs(dd_) >= abs(d_obs) - 1e-15
                ii = dd_ - (xn[pn].mean() - xn[~pn].mean())
                cnt_i += abs(ii) >= abs(i_obs) - 1e-15
            cc = np.corrcoef(r5[sep], x[sep])[0, 1]
            show([row(f"SEP & {rk_name}>=75", v[sep & hi], a), row(f"SEP & {rk_name}<75", v[sep & ~hi], b),
                  row(f"nonSEP & {rk_name}>=75", v[~sep & hi], c), row(f"nonSEP & {rk_name}<75", v[~sep & ~hi], e)],
                 f"A. {tkr} h={h} by {rk_name}")
            print(f"    within-SEP diff {100 * d_obs:+.3f}pp perm p(2s) {(1 + cnt_d) / (NP + 1):.3f};  "
                  f"interaction {100 * i_obs:+.3f}pp perm p(2s) {(1 + cnt_i) / (NP + 1):.3f};  "
                  f"within-SEP corr(r5, tdomX) {cc:+.3f} n {sep.sum()}")
            if tkr == "DX-Y.NYB" and rk_name == "DX r5":
                ann = (s / s.shift(1) - 1.0)
                idxs = np.where(sep & hi)[0]
                print("    SEP & hi episodes (date, eve r5, ann-session %, h%):  " + "; ".join(
                    f"{dd[i].date()} {r5[i]:.0f} {100 * ann[dd[i]]:+.2f} {100 * v[i]:+.2f}" for i in idxs))
                yrs = dd.year.values
                m1, m2 = sep & hi & (yrs < 2018), sep & hi & (yrs >= 2018)
                show([row("SEP&hi pre-2018", v[m1], x[m1]), row("SEP&hi 2018+", v[m2], x[m2])])

# ------------------------------------------------------------------ B
print("\n" + "=" * 100, "\nB. FAMILY PERMUTATION RE-RUN (seed 20260916, 2000 draws)\n" + "=" * 100)
dx = pxd["DX-Y.NYB"]["Close"].dropna()
idx = dx.index
TD = tdom_of(idx)
POS = pd.Series(np.arange(len(idx)), index=idx)
tnx = pxd["^TNX"]["Close"].dropna()
mx = tnx.rolling(252).max()
chg21 = tnx - tnx.shift(21)
fomcA = [d for d in FOMC if len(tnx.index[tnx.index < d]) and
         not np.isnan(mx.get(tnx.index[tnx.index < d][-1], np.nan)) and d in POS.index]
dpA = np.array([POS[d] for d in fomcA])
ev = [tnx.index[tnx.index < d][-1] for d in fomcA]
tv = np.array([tnx[e] for e in ev])
tm = np.array([mx[e] for e in ev])
c21 = np.array([chg21[e] for e in ev])
dist = tv / tm - 1.0
GATES = {"ALL": np.ones(len(dpA), bool), "at_max": tv >= tm - 1e-9, "w0.5": dist >= -0.005,
         "w1.0": dist >= -0.01, "w2.0": dist >= -0.02, "thr20": c21 >= 0.20,
         "NOT_w2.0": dist < -0.02}
GATES["at_max&thr20"] = GATES["at_max"] & GATES["thr20"]
print("N decisions", len(dpA), {k: int(m.sum()) for k, m in GATES.items()})
EX = np.zeros(len(idx), bool)
for d in FOMC:
    if d in POS.index:
        p = POS[d]
        EX[max(0, p - 5):p + 6] = True
RET, BUCK = {}, {}
for vk, tk in (("TLT", "TLT"), ("DX", "DX-Y.NYB"), ("GLD", "GLD")):
    s2 = pxd[tk]["Close"].dropna().reindex(idx)
    for h in range(1, 6):
        r = (s2.shift(-h) / s2 - 1.0).values
        ok = ~np.isnan(r)
        RET[(vk, h)] = r
        BUCK[(vk, h)] = np.array([np.nanmean(r[(TD == j) & ~EX & ok]) if ((TD == j) & ~EX & ok).any()
                                  else np.nan for j in range(1, 24)])


def tmat(X):
    n = (~np.isnan(X)).sum(axis=-1)
    m = np.nanmean(X, axis=-1)
    sd = np.nanstd(X, axis=-1, ddof=1)
    return m / (sd / np.sqrt(n))


obs = {}
for key, r in RET.items():
    x = r[dpA] - BUCK[key][TD[dpA] - 1]
    for gk, gm in GATES.items():
        obs[(key[0], key[1], gk)] = tmat(x[gm][None, :])[0]
print(f"observed tdomX t: DX h3 ALL {obs[('DX', 3, 'ALL')]:.3f}, DX h5 ALL {obs[('DX', 5, 'ALL')]:.3f}")
top = sorted(obs.items(), key=lambda kv: -abs(kv[1]) if not np.isnan(kv[1]) else 0)[:8]
print("observed largest |t| cells over all rungs:", [(k, round(float(t), 2)) for k, t in top])

NPERM = 2000
prng = np.random.default_rng(20260916)
cand = np.where(~EX & (np.arange(len(idx)) < len(idx) - 6))[0]
PP = np.empty((NPERM, len(dpA)), int)
for j, p in enumerate(dpA):
    c = cand[(TD[cand] == TD[p]) & (np.abs(cand - p) <= 252)]
    PP[:, j] = c[prng.integers(0, len(c), size=NPERM)]
FAM = {"B1 ALL only (kD family)": ["ALL"], "B2 rungs n>=30": ["ALL", "thr20", "NOT_w2.0"],
       "B3 all 8 rungs": list(GATES)}
MAXT = {k: np.zeros(NPERM) for k in FAM}
for key, r in RET.items():
    X = r[PP] - BUCK[key][TD[PP] - 1]
    for gk, gm in GATES.items():
        t = np.abs(tmat(X[:, gm]))
        t = np.nan_to_num(t, nan=0.0)
        for fk, rungs in FAM.items():
            if gk in rungs:
                MAXT[fk] = np.maximum(MAXT[fk], t)
for fk in FAM:
    mt = MAXT[fk]
    p3 = (1 + (mt >= abs(obs[("DX", 3, "ALL")])).sum()) / (NPERM + 1)
    p5 = (1 + (mt >= abs(obs[("DX", 5, "ALL")])).sum()) / (NPERM + 1)
    print(f"  {fk}: null max|t| 50/90/95 {np.percentile(mt, [50, 90, 95]).round(2)}  "
          f"P(>= DX h3 {abs(obs[('DX', 3, 'ALL')]):.2f}) {p3:.4f}  P(>= DX h5 {abs(obs[('DX', 5, 'ALL')]):.2f}) {p5:.4f}")

# ------------------------------------------------------------------ C
print("\n" + "=" * 100, "\nC. SIZING: h=5 returns in eve Wilder-14 ATR units\n" + "=" * 100)
for tkr in ("DX-Y.NYB", "UUP"):
    f = pxd[tkr].dropna(subset=["Close"])
    pc = f["Close"].shift(1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - pc).abs(), (f["Low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atrp = atr / f["Close"]
    for h in (3, 5):
        dd, dp, v, x, s = cell(tkr, h)
        a = np.array([eve_val(atrp, d) for d in dd])
        z = v / a
        yrs = dd.year.values
        sep = np.array([is_sep(d) for d in dd])
        print(f"  {tkr} h={h}: sd raw {100 * v.std(ddof=1):.3f}%  sd in ATR units all {z.std(ddof=1):.2f} "
              f"(2011+ {z[yrs >= 2011].std(ddof=1):.2f}, 2018+ {z[yrs >= 2018].std(ddof=1):.2f}, SEP {z[sep].std(ddof=1):.2f})  "
              f"mean {z.mean():+.2f} ATR  p5/p1 {np.percentile(z, 5):.2f}/{np.percentile(z, 1):.2f} ATR  worst {z.min():.2f}")
    print(f"  {tkr} eve 2026-09-15: close {f['Close'].iloc[-1]:.3f} ATR {atr.iloc[-1]:.4f} ({100 * atrp.iloc[-1]:.3f}%)")
