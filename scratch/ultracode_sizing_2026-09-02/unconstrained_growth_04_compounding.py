"""Unconstrained growth, part 4: compounding (risk sized off live NAV) versus
the flat $750k basis, and the sequencing risk of switching.  Block-bootstrap
paths (2016+ / 2003+, haircut 0 / 50%) at m in {1, 1.5, 2}; policies: flat
(fixed $ risk), full compounding, quarterly rebase, ratchet (rebase up only),
half-compounding.  Also the GPD-tail growth curve under the mean haircuts,
which part 1 only ran unhaircut.  Writes unconstrained_growth_04_compounding.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
NAV = 750_000.0; GRM_NOW = 1.5
RNG = np.random.default_rng(20260902)
OUT: dict = {}
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
tot = pd.Series(sd["total_flat"], index=pd.to_datetime(sd["dates"]), dtype=float)
tot = tot[tot.index <= "2026-08-07"]
r_all = (tot / NAV).astype(float)
windows = {"2016+": r_all[r_all.index >= "2016-01-01"].values, "2003+": r_all.values}

def sb_idx(n, paths, T, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((paths, T), dtype=np.int64); idx[:, 0] = rng.integers(0, n, paths)
    for t in range(1, T):
        new = rng.random(paths) < p
        idx[:, t] = np.where(new, rng.integers(0, n, paths), (idx[:, t - 1] + 1) % n)
    return idx

def maxdd(W):
    pk = np.maximum.accumulate(W, axis=1); return (1 - W / pk).max(axis=1)
def max_dollar_dd(W):
    pk = np.maximum.accumulate(W, axis=1); return (pk - W).max(axis=1)      # in units of initial capital

def run_policies(R, m):
    P, T = R.shape
    flat = 1 + np.cumsum(m * R, axis=1)
    comp = np.cumprod(1 + m * R, axis=1)
    # quarterly rebase, ratchet, half: iterate in 63-day blocks
    q = np.ones(P); rat = np.ones(P); hf = np.ones(P)
    Wq = np.empty_like(R); Wr = np.empty_like(R); Wh = np.empty_like(R)
    wq = np.ones(P); wr = np.ones(P); wh = np.ones(P)
    for t in range(T):
        if t % 63 == 0 and t > 0:
            q = wq.copy(); rat = np.maximum(rat, wr); hf = 0.5 + 0.5 * wh
        wq = wq + q * m * R[:, t]; wr = wr + rat * m * R[:, t]; wh = wh + hf * m * R[:, t]
        Wq[:, t] = wq; Wr[:, t] = wr; Wh[:, t] = wh
    return dict(flat=flat, comp=comp, quarterly=Wq, ratchet=Wr, half=Wh)

def summarize(W, name):
    term = W[:, -1]
    return dict(policy=name, terminal_median=float(np.median(term)), terminal_mean=float(term.mean()), terminal_p05=float(np.quantile(term, .05)), terminal_p95=float(np.quantile(term, .95)),
                p_loss=float((term < 1).mean()), p_ruin=float((W.min(axis=1) <= 0).mean()),
                maxdd_pct_of_peak_median=float(np.median(maxdd(np.maximum(W, 1e-9)))), maxdd_pct_of_peak_p95=float(np.quantile(maxdd(np.maximum(W, 1e-9)), .95)),
                max_dollar_dd_median=float(np.median(max_dollar_dd(W))), max_dollar_dd_p95=float(np.quantile(max_dollar_dd(W), .95)),
                p_dollar_dd_gt_20pct_initial=float((max_dollar_dd(W) > .2).mean()), p_dollar_dd_gt_50pct_initial=float((max_dollar_dd(W) > .5).mean()))

PATHS = 3000
OUT["policies"] = {}
print("=== compounding vs flat: block bootstrap, terminal wealth (x initial) and drawdowns ===")
for wname, rv in windows.items():
    n = len(rv); mu = rv.mean()
    for h in [0.0, 0.5]:
        rh = rv - h * mu
        for T, tag in [(756, "3y"), (2520, "10y")]:
            idx = sb_idx(n, PATHS, T, 10.0, RNG); R = rh[idx]
            for m in [1.0, 1.5, 2.0]:
                pol = run_policies(R, m)
                rows = [summarize(W, k) for k, W in pol.items()]
                # pairwise sequencing regret: comp vs flat on the same path
                d = pol["comp"][:, -1] - pol["flat"][:, -1]
                y1dd = maxdd(np.maximum(pol["comp"][:, :252], 1e-9))
                bad = y1dd > 0.15
                seq = dict(p_comp_below_flat=float((d < 0).mean()), median_gain_comp_minus_flat=float(np.median(d)), p05_gain=float(np.quantile(d, .05)),
                           p_first_year_dd_gt_15=float(bad.mean()),
                           given_bad_first_year_median_comp=float(np.median(pol["comp"][bad, -1])) if bad.any() else None,
                           given_bad_first_year_median_flat=float(np.median(pol["flat"][bad, -1])) if bad.any() else None,
                           effective_m_in_nav_terms_flat_at_end_median=float(m / np.median(pol["flat"][:, -1])))
                key = f"{wname}|h{h:g}|{tag}|m{m:g}"
                OUT["policies"][key] = dict(rows=rows, sequencing=seq)
                if tag == "10y" or (tag == "3y" and m == 1.5):
                    f = rows[0]; c = rows[1]; hh = rows[4]
                    print(f"{key:22s} flat med {f['terminal_median']:5.2f}x DD$ {f['max_dollar_dd_median']:.0%}/{f['max_dollar_dd_p95']:.0%} | comp med {c['terminal_median']:6.2f}x p5 {c['terminal_p05']:5.2f} DD% {c['maxdd_pct_of_peak_median']:.0%}/{c['maxdd_pct_of_peak_p95']:.0%} DD$ {c['max_dollar_dd_p95']:.0%} | "
                          f"half med {hh['terminal_median']:5.2f}x | P(comp<flat) {seq['p_comp_below_flat']:.0%} | bad-yr1 {seq['p_first_year_dd_gt_15']:.0%}: comp {seq['given_bad_first_year_median_comp']} flat {seq['given_bad_first_year_median_flat']} | eff m flat end {seq['effective_m_in_nav_terms_flat_at_end_median']:.2f}")

# ------------------------------------------------------------ GPD-tail growth with haircuts
print("\n=== GPD-tail growth optimum under mean haircuts ===")
M_FINE = np.round(np.arange(0.25, 30.01, 0.25), 2)
OUT["gpd_haircut"] = {}
for wname, rv in windows.items():
    lo_q, hi_q = np.quantile(rv, 0.025), np.quantile(rv, 0.975)
    c_lo, _, s_lo = stats.genpareto.fit(lo_q - rv[rv < lo_q], floc=0); c_hi, _, s_hi = stats.genpareto.fit(rv[rv > hi_q] - hi_q, floc=0)
    n_sim = 2_000_000; u = RNG.random(n_sim); body = rv[(rv >= lo_q) & (rv <= hi_q)]
    sim = RNG.choice(body, n_sim); lo = u < .025; hi = u > .975
    sim[lo] = lo_q - stats.genpareto.rvs(c_lo, loc=0, scale=s_lo, size=lo.sum(), random_state=RNG)
    sim[hi] = hi_q + stats.genpareto.rvs(c_hi, loc=0, scale=s_hi, size=hi.sum(), random_state=RNG)
    sim = sim - (sim.mean() - rv.mean())
    OUT["gpd_haircut"][wname] = {}
    for h in [0.0, 0.25, 0.5]:
        s = sim - h * rv.mean()
        g = []
        for m in M_FINE:
            x = 1 + m * s
            g.append(-np.inf if (x <= 0).mean() > 0 else 252 * np.mean(np.log(x)))
        g = np.array(g); i = int(np.argmax(g))
        ruin_m = float(M_FINE[np.isinf(g)].min()) if np.isinf(g).any() else None
        OUT["gpd_haircut"][wname][f"{h:g}"] = dict(m_star=float(M_FINE[i]), grm_star=float(M_FINE[i] * GRM_NOW), g_star=float(g[i]), g_at_1=float(g[3]), g_at_1p5=float(g[5]), g_at_2=float(g[7]), first_ruin_m=ruin_m,
                                                  sim_worst_day=float(s.min()), sim_p_day_below_minus_5pct=float((s < -.05).mean()))
        print(f"{wname} h={h:g}: GPD-tail m*={M_FINE[i]:.2f} (GRM {M_FINE[i]*1.5:.1f}) g*={g[i]:.0%}; g(m=1)={g[3]:.1%} g(1.5)={g[5]:.1%} g(2)={g[7]:.1%}; ruin from m={ruin_m}; sim worst day {s.min():.1%}")

json.dump(OUT, open(HERE / "unconstrained_growth_04_compounding.json", "w"), indent=1, default=float)
print("\nwrote", HERE / "unconstrained_growth_04_compounding.json")
