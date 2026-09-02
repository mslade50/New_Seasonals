"""Unconstrained growth, part 1: the log-growth curve g(m) for the book and
the drawdown DISTRIBUTION at each multiple m of current sizing.

m = multiple of CURRENT sizing (the ledger is built at GRM 1.5, so GRM = 1.5 m).
Series: dist/data/strategy_daily.json total_flat (flat $750k MTM, 2003-01-21 ..
2026-08-07) -> r_t = pnl / 750k.  Fat tails preserved three ways: exact
empirical E[log(1+m r)], a stationary block bootstrap (mean block 10 sessions),
and a Student-t / GPD-tail semi-parametric fit.  Haircuts shrink the daily mean
by 0 / 25 / 50 %.  Writes unconstrained_growth_01_growth.json beside this file.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
NAV = 750_000.0
GRM_NOW = 1.5
RNG = np.random.default_rng(20260902)
M_GRID = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0])
M_FINE = np.round(np.arange(0.25, 30.01, 0.25), 2)
OUT: dict = {"m_grid": M_GRID.tolist(), "grm_now": GRM_NOW, "note": "m = multiple of current sizing; GRM = 1.5*m"}

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
tot = pd.Series(sd["total_flat"], index=pd.to_datetime(sd["dates"]), dtype=float)
tot = tot[tot.index <= "2026-08-07"]
r_all = (tot / NAV).astype(float)
print(f"daily series {r_all.index.min().date()} .. {r_all.index.max().date()} N={len(r_all)}")

def growth_curve(r: np.ndarray, m_grid: np.ndarray) -> np.ndarray:
    g = np.empty(len(m_grid))
    for i, m in enumerate(m_grid):
        x = 1.0 + m * r
        g[i] = -np.inf if (x <= 0).any() else 252.0 * np.mean(np.log(x))
    return g

def quad_curve(r: np.ndarray, m_grid: np.ndarray) -> np.ndarray:
    mu, v = r.mean(), r.var()
    return 252.0 * (m_grid * mu - 0.5 * m_grid**2 * v)

def summarize_curve(m_grid, g):
    i = int(np.nanargmax(g)); m_star = float(m_grid[i])
    def g_at(mm):
        j = int(np.argmin(np.abs(m_grid - mm))); return float(g[j])
    return dict(m_star=m_star, grm_star=m_star * GRM_NOW, g_star=float(g[i]),
                g_at_current=g_at(1.0), g_quarter=g_at(m_star / 4), g_half=g_at(m_star / 2), g_3q=g_at(3 * m_star / 4),
                m_quarter=m_star / 4, m_half=m_star / 2, m_3q=3 * m_star / 4)

# ------------------------------------------------------------ 1. analytic curves by window x haircut
windows = {"2003+": r_all, "2016+": r_all[r_all.index >= "2016-01-01"], "2021+": r_all[r_all.index >= "2021-01-01"]}
haircuts = [0.0, 0.25, 0.5]
OUT["analytic"] = {}
print("\n=== 1. exact empirical growth g(m) = 252*E[log(1+m r)] (ann log growth) ===")
for wname, r in windows.items():
    rv = r.values
    mu, sig = rv.mean(), rv.std()
    base = dict(days=len(rv), ann_ret=mu * 252, ann_vol=sig * np.sqrt(252), sharpe=mu / sig * np.sqrt(252),
                worst_day=float(rv.min()), best_day=float(rv.max()), skew=float(stats.skew(rv)), kurt=float(stats.kurtosis(rv)),
                kelly_quad=float(mu / sig**2))
    OUT["analytic"][wname] = {"base": base, "haircut": {}}
    for h in haircuts:
        rh = rv - h * mu
        g = growth_curve(rh, M_FINE); q = quad_curve(rh, M_FINE)
        s = summarize_curve(M_FINE, g); sq = summarize_curve(M_FINE, q)
        ruin_m = float(M_FINE[np.isinf(g)].min()) if np.isinf(g).any() else None
        s.update(quad_m_star=sq["m_star"], quad_g_star=sq["g_star"], first_ruin_m=ruin_m,
                 curve={f"{m:g}": (None if np.isinf(gg) else float(gg)) for m, gg in zip(M_GRID, growth_curve(rh, M_GRID))})
        OUT["analytic"][wname]["haircut"][f"{h:g}"] = s
        print(f"{wname:6s} h={h:.2f}: Sharpe {base['sharpe']:.2f}  m*={s['m_star']:.2f} (GRM {s['grm_star']:.2f}) g*={s['g_star']:.1%}  "
              f"g(now)={s['g_at_current']:.1%} g(m*/4)={s['g_quarter']:.1%} g(m*/2)={s['g_half']:.1%} g(3m*/4)={s['g_3q']:.1%}  "
              f"quad m*={sq['m_star']:.2f}  ruin from m={ruin_m}")

# ------------------------------------------------------------ 2. parametric tails: Student-t and empirical body + GPD tails
print("\n=== 2. parametric / semi-parametric tails (2016+ and 2003+, no haircut) ===")
OUT["parametric"] = {}
for wname in ["2003+", "2016+"]:
    rv = windows[wname].values
    df_t, loc_t, sc_t = stats.t.fit(rv)
    # growth under t: integrate log(1+m x) over x > -1/m; report daily ruin mass separately
    xs = np.linspace(-0.5, 0.5, 200_001)
    pdf = stats.t.pdf(xs, df_t, loc_t, sc_t); dx = xs[1] - xs[0]
    tcurve = {}
    for m in M_GRID:
        ok = xs > -1.0 / m + 1e-9
        g_trunc = 252.0 * np.sum(np.log1p(m * xs[ok]) * pdf[ok]) * dx
        p_ruin_day = float(stats.t.cdf(-1.0 / m, df_t, loc_t, sc_t))
        tcurve[f"{m:g}"] = dict(g_trunc=float(g_trunc), p_ruin_day=p_ruin_day, p_ruin_year=1 - (1 - p_ruin_day) ** 252)
    # semi-parametric: empirical body, GPD beyond the 2.5% / 97.5% quantiles
    lo_q, hi_q = np.quantile(rv, 0.025), np.quantile(rv, 0.975)
    lo_exc = lo_q - rv[rv < lo_q]; hi_exc = rv[rv > hi_q] - hi_q
    c_lo, _, s_lo = stats.genpareto.fit(lo_exc, floc=0); c_hi, _, s_hi = stats.genpareto.fit(hi_exc, floc=0)
    n_sim = 2_000_000
    u = RNG.random(n_sim); body = rv[(rv >= lo_q) & (rv <= hi_q)]
    sim = RNG.choice(body, n_sim)
    lo_mask = u < 0.025; hi_mask = u > 0.975
    sim[lo_mask] = lo_q - stats.genpareto.rvs(c_lo, loc=0, scale=s_lo, size=lo_mask.sum(), random_state=RNG)
    sim[hi_mask] = hi_q + stats.genpareto.rvs(c_hi, loc=0, scale=s_hi, size=hi_mask.sum(), random_state=RNG)
    sim = sim - (sim.mean() - rv.mean())  # re-centre to the sample mean
    gpd_curve = {}
    for m in M_FINE:
        x = 1 + m * sim; ruin = (x <= 0).mean()
        gpd_curve[float(m)] = (252 * np.mean(np.log(x[x > 0])), ruin)
    gvals = np.array([v[0] if v[1] == 0 else -np.inf for v in gpd_curve.values()])
    s = summarize_curve(M_FINE, gvals)
    OUT["parametric"][wname] = dict(student_t=dict(df=float(df_t), loc=float(loc_t), scale=float(sc_t), curve=tcurve),
                                   gpd=dict(xi_lower=float(c_lo), beta_lower=float(s_lo), xi_upper=float(c_hi), beta_upper=float(s_hi),
                                            m_star=s["m_star"], g_star=s["g_star"], g_at_current=s["g_at_current"],
                                            curve={f"{m:g}": dict(g=float(gpd_curve[float(m)][0]), p_ruin_day=float(gpd_curve[float(m)][1])) for m in M_GRID}))
    print(f"{wname}: t df={df_t:.2f} scale={sc_t*1e4:.1f}bps | GPD xi_lo={c_lo:.2f} xi_hi={c_hi:.2f} | GPD-tail m*={s['m_star']:.2f} g*={s['g_star']:.1%}")
    print("   t-tail daily ruin P(m r < -1):", {k: f"{v['p_ruin_year']:.2%}/yr" for k, v in tcurve.items() if v["p_ruin_year"] > 1e-4})

# ------------------------------------------------------------ 3. stationary block bootstrap: growth + drawdown distribution
def stationary_bootstrap_idx(n: int, paths: int, T: int, mean_block: float, rng) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((paths, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, paths)
    for t in range(1, T):
        new = rng.random(paths) < p
        idx[:, t] = np.where(new, rng.integers(0, n, paths), (idx[:, t - 1] + 1) % n)
    return idx

def dd_stats(eq: np.ndarray):
    """eq: paths x T equity (starting at 1).  Returns maxDD (positive fraction), trough idx, recovery days (nan if censored), longest underwater."""
    peak = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
    dd = 1.0 - eq / peak
    maxdd = dd.max(axis=1); trough = dd.argmax(axis=1)
    P, T = eq.shape
    pk = peak[np.arange(P), trough]
    tt = np.arange(T)[None, :]
    mask = (tt >= trough[:, None]) & (eq >= pk[:, None])
    has = mask.any(axis=1)
    rec = np.where(has, mask.argmax(axis=1) - trough, np.nan).astype(float)
    uw = (dd > 0).astype(np.int64)
    c = np.cumsum(uw, axis=1)
    run = c - np.maximum.accumulate(np.where(uw == 0, c, 0), axis=1)
    longest = run.max(axis=1).astype(float)
    return maxdd, rec, longest

PATHS = 3000
OUT["bootstrap"] = {}
print("\n=== 3. stationary block bootstrap (mean block 10, %d paths): growth + drawdown distribution ===" % PATHS)
for wname in ["2016+", "2003+"]:
    rv = windows[wname].values; n = len(rv); mu = rv.mean()
    for h in haircuts:
        rh = rv - h * mu
        key = f"{wname}|h{h:g}"
        OUT["bootstrap"][key] = {}
        for T, tag in [(252, "1y"), (756, "3y")]:
            idx = stationary_bootstrap_idx(n, PATHS, T, 10.0, RNG)
            R = rh[idx]
            rows = {}
            for m in M_GRID:
                x = 1.0 + m * R
                ruined = (x <= 0).any(axis=1)
                lg = np.where(x > 0, np.log(np.maximum(x, 1e-300)), -50.0)
                growth = lg.sum(axis=1) / T * 252
                eq = np.exp(np.cumsum(lg, axis=1))
                maxdd, rec, longest = dd_stats(eq)
                # flat (additive) basis drawdown in % of the fixed $750k
                cum = np.cumsum(m * R, axis=1); flat_dd = (np.maximum.accumulate(np.maximum(cum, 0), axis=1) - cum).max(axis=1)
                term = eq[:, -1]
                rows[f"{m:g}"] = dict(
                    growth_mean=float(np.mean(growth[~ruined])) if (~ruined).any() else None, growth_median=float(np.median(growth)),
                    growth_p05=float(np.quantile(growth, 0.05)), p_ruin=float(ruined.mean()),
                    p_end_below_start=float((term < 1).mean()), terminal_median=float(np.median(term)), terminal_p05=float(np.quantile(term, 0.05)),
                    maxdd_median=float(np.median(maxdd)), maxdd_mean=float(maxdd.mean()), maxdd_p95=float(np.quantile(maxdd, 0.95)),
                    maxdd_p99=float(np.quantile(maxdd, 0.99)),
                    p_dd_gt_10=float((maxdd > .10).mean()), p_dd_gt_20=float((maxdd > .20).mean()), p_dd_gt_30=float((maxdd > .30).mean()),
                    p_dd_gt_40=float((maxdd > .40).mean()), p_dd_gt_50=float((maxdd > .50).mean()),
                    flat_dd_median=float(np.median(flat_dd)), flat_dd_p95=float(np.quantile(flat_dd, 0.95)),
                    p_flat_dd_gt_20=float((flat_dd > .20).mean()), p_flat_dd_gt_30=float((flat_dd > .30).mean()), p_flat_dd_gt_50=float((flat_dd > .50).mean()),
                    recover_days_median=float(np.nanmedian(rec)) if np.isfinite(rec).any() else None,
                    recover_days_mean=float(np.nanmean(rec)) if np.isfinite(rec).any() else None,
                    recover_days_p95=float(np.nanquantile(rec, 0.95)) if np.isfinite(rec).any() else None,
                    p_unrecovered_at_horizon=float(np.isnan(rec).mean()),
                    longest_underwater_median=float(np.median(longest)), longest_underwater_p95=float(np.quantile(longest, 0.95)))
            OUT["bootstrap"][key][tag] = rows
            if tag == "3y":
                gm = {m: rows[f"{m:g}"]["growth_mean"] for m in M_GRID}
                best = max((m for m in M_GRID if gm[m] is not None), key=lambda m: gm[m])
                print(f"{key} {tag}: bootstrap growth-max m={best:g} (GRM {best*GRM_NOW:.2f}), g={gm[best]:.1%}; "
                      f"at m=1 g={gm[1.0]:.1%} medDD={rows['1']['maxdd_median']:.1%} P(DD>20)={rows['1']['p_dd_gt_20']:.1%}")
            if tag == "1y":
                for m in [1.0, 2.0, 3.0, 5.0, 8.0]:
                    q = rows[f"{m:g}"]
                    print(f"   1y m={m:g}: medDD {q['maxdd_median']:.1%} p95DD {q['maxdd_p95']:.1%} P>20 {q['p_dd_gt_20']:.1%} P>30 {q['p_dd_gt_30']:.1%} "
                          f"P>50 {q['p_dd_gt_50']:.1%} rec med {q['recover_days_median']} unrec {q['p_unrecovered_at_horizon']:.0%} ruin {q['p_ruin']:.2%}")

# ------------------------------------------------------------ 4. historical (actual path) drawdowns at each m, compounded and flat
print("\n=== 4. actual-path drawdowns 2016+ and 2003+ at each m ===")
OUT["historical"] = {}
for wname in ["2016+", "2003+"]:
    rv = windows[wname].values
    rows = {}
    for m in M_GRID:
        x = 1 + m * rv
        if (x <= 0).any():
            rows[f"{m:g}"] = dict(ruin=True); continue
        eq = np.cumprod(x); peak = np.maximum.accumulate(eq); dd = 1 - eq / peak
        cum = np.cumsum(m * rv); fdd = (np.maximum.accumulate(np.maximum(cum, 0)) - cum)
        yrs = len(rv) / 252
        rows[f"{m:g}"] = dict(ruin=False, cagr=float(eq[-1] ** (1 / yrs) - 1), maxdd_comp=float(dd.max()), maxdd_flat_pct_of_750k=float(fdd.max()),
                              worst_day=float(m * rv.min()), terminal_multiple=float(eq[-1]))
    OUT["historical"][wname] = rows
    print(wname, {k: (f"cagr {v['cagr']:.0%} DD {v['maxdd_comp']:.0%}" if not v["ruin"] else "RUIN") for k, v in rows.items() if k in ["1", "2", "3", "5", "8", "12", "20"]})

# ------------------------------------------------------------ 5. fractional-Kelly base rates (continuous-time lognormal)
frac = {}
for c in [0.1, 0.25, 0.5, 0.75, 1.0]:
    frac[f"{c:g}"] = dict(share_of_max_growth=2 * c - c**2,
                         p_ever_dd={f"{x:g}": (1 - x) ** (2 / c - 1) for x in [0.2, 0.3, 0.5]})
OUT["fractional_kelly_theory"] = frac
print("\n=== 5. fractional Kelly theory: share of max growth, P(ever losing x) = (1-x)^(2/c-1) ===")
for c, v in frac.items():
    print(f"  c={c}: growth share {v['share_of_max_growth']:.0%}, P(ever DD>=20/30/50%) = "
          + ", ".join(f"{float(p):.1%}" for p in v["p_ever_dd"].values()))

json.dump(OUT, open(HERE / "unconstrained_growth_01_growth.json", "w"), indent=1, default=float)
print("\nwrote", HERE / "unconstrained_growth_01_growth.json")
