"""Growth-maximizer lens, part 2: the growth curve and drawdown distribution at
the MEASURED haircut (estimation_haircut: 40% central, 27-71% range) instead
of the 0/25/50% grid, plus the Kaminski-Lo test (daily P&L autocorrelation)
that decides whether any drawdown-triggered de-risking can have positive
expected value.  Daily series = dist/data/strategy_daily.json total_flat on
the flat $750k basis (ends 2026-08-07).  Writes growthmax_2_growth_dd_acf.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
NAV = 750_000.0
RNG = np.random.default_rng(20260902)
OUT: dict = {}

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
tot = pd.Series(sd["total_flat"], index=pd.to_datetime(sd["dates"]), dtype=float)
r_all = (tot / NAV)
r_all = r_all[r_all.index <= "2026-08-07"]
WINDOWS = {"2003+": r_all[r_all.index >= "2003-01-01"], "2016+": r_all[r_all.index >= "2016-01-01"]}
M_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0]
HAIRCUTS = [0.0, 0.27, 0.40, 0.50, 0.71]


def growth(r: np.ndarray, m: float) -> float:
    x = 1 + m * r
    if (x <= 0).any():
        return float("-inf")
    return float(np.log(x).mean() * 252)


print("=== 1. annualised log growth g(m) by window x haircut (exact E[log(1+m r)]) ===")
OUT["growth"] = {}
for wn, r in WINDOWS.items():
    rv = r.values
    OUT["growth"][wn] = {}
    for hc in HAIRCUTS:
        rr = rv - rv.mean() * hc
        row = {f"{m:g}": growth(rr, m) for m in M_GRID}
        # m* on a fine grid
        grid = np.linspace(0.25, 25, 400)
        gs = np.array([growth(rr, m) for m in grid])
        row["m_star"] = float(grid[int(np.nanargmax(gs))]); row["g_star"] = float(np.nanmax(gs))
        OUT["growth"][wn][f"hc{hc:g}"] = row
        print(f"  {wn} haircut {hc:>4.0%}: " + " ".join(f"m{m:g}={row[f'{m:g}']*100:5.1f}%" for m in M_GRID) + f" | m*={row['m_star']:.1f} g*={row['g_star']*100:.0f}%")
    s = rv.mean() / rv.std() * np.sqrt(252)
    OUT["growth"][wn]["sharpe"] = float(s); OUT["growth"][wn]["ann_vol"] = float(rv.std() * np.sqrt(252)); OUT["growth"][wn]["N"] = int(len(rv))
    print(f"  {wn}: Sharpe {s:.2f}, vol {rv.std()*np.sqrt(252):.1%}, N {len(rv)}")


# ---------------------------------------------------------------- 2. drawdown distribution by m at the measured haircut
def stationary_idx(n, paths, T, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((paths, T), dtype=np.int64); idx[:, 0] = rng.integers(0, n, paths)
    for t in range(1, T):
        new = rng.random(paths) < p
        idx[:, t] = np.where(new, rng.integers(0, n, paths), (idx[:, t - 1] + 1) % n)
    return idx


PATHS = 3000
print("\n=== 2. block-bootstrap (mean block 10, 3000 paths) compounded maxDD by m, haircuts 40% / 50% / 71% ===")
OUT["drawdown"] = {}
for wn, r in WINDOWS.items():
    rv = r.values; n = len(rv)
    OUT["drawdown"][wn] = {}
    for T, tag in [(252, "1y"), (756, "3y")]:
        idx = stationary_idx(n, PATHS, T, 10.0, RNG)
        for hc in [0.40, 0.50, 0.71]:
            rr = rv - rv.mean() * hc
            R = rr[idx]
            OUT["drawdown"][wn][f"{tag}_hc{hc:g}"] = {}
            for m in M_GRID:
                eq = np.cumprod(1 + m * R, axis=1)
                peak = np.maximum.accumulate(eq, axis=1)
                dd = (eq / peak - 1).min(axis=1)
                term = eq[:, -1]
                ruin = (eq <= 0).any(axis=1).mean()
                row = dict(dd_median=float(-np.median(dd)), dd_p95=float(-np.percentile(dd, 5)), p_dd10=float((dd < -.10).mean()), p_dd15=float((dd < -.15).mean()),
                           p_dd20=float((dd < -.20).mean()), p_dd30=float((dd < -.30).mean()), p_dd50=float((dd < -.50).mean()),
                           terminal_median=float(np.median(term)), terminal_p5=float(np.percentile(term, 5)), p_terminal_below_1=float((term < 1).mean()), p_ruin=float(ruin))
                OUT["drawdown"][wn][f"{tag}_hc{hc:g}"][f"{m:g}"] = row
                if tag == "1y" and hc == 0.40 or (tag == "3y" and hc == 0.40 and m in (1.5, 2.0, 2.5, 3.0)):
                    print(f"  {wn} {tag} hc{hc:g} m={m:<4g} DD median {row['dd_median']:5.1%} p95 {row['dd_p95']:5.1%} P>10 {row['p_dd10']:4.0%} P>15 {row['p_dd15']:4.0%} P>20 {row['p_dd20']:4.0%} P>30 {row['p_dd30']:4.0%} P>50 {row['p_dd50']:4.1%} | terminal med {row['terminal_median']:.2f} p5 {row['terminal_p5']:.2f} P(<1) {row['p_terminal_below_1']:.0%}")

# ---------------------------------------------------------------- 3. actual path at each m, 2003+ and 2016+
print("\n=== 3. actual historical path, compounded, at each m (no haircut) ===")
OUT["actual_path"] = {}
for wn, r in WINDOWS.items():
    OUT["actual_path"][wn] = {}
    for m in M_GRID:
        eq = (1 + m * r).cumprod(); dd = (eq / eq.cummax() - 1).min(); worst = (m * r).min()
        yrs = (r.index[-1] - r.index[0]).days / 365.25
        OUT["actual_path"][wn][f"{m:g}"] = dict(maxdd=float(-dd), worst_day=float(-worst), cagr=float(eq.iloc[-1] ** (1 / yrs) - 1), terminal=float(eq.iloc[-1]))
    print(f"  {wn}: " + " | ".join(f"m{m:g}: DD {OUT['actual_path'][wn][f'{m:g}']['maxdd']:.0%} worst {OUT['actual_path'][wn][f'{m:g}']['worst_day']:.1%} CAGR {OUT['actual_path'][wn][f'{m:g}']['cagr']:.0%}" for m in [1, 1.5, 2, 2.5, 3]))

# ---------------------------------------------------------------- 4. Kaminski-Lo: autocorrelation and drawdown-state forecast content
print("\n=== 4. daily P&L autocorrelation (Kaminski-Lo condition for a stop-loss / drawdown cut to pay) ===")
OUT["acf"] = {}
for wn, r in WINDOWS.items():
    x = r.values; n = len(x)
    ac = [float(pd.Series(x).autocorr(k)) for k in range(1, 21)]
    se = 1 / np.sqrt(n)
    lb = n * (n + 2) * sum(a * a / (n - k) for k, a in enumerate(ac[:10], start=1))
    # weekly (5d) and monthly (21d) non-overlapping sums
    w5 = pd.Series(x).groupby(np.arange(n) // 5).sum(); w21 = pd.Series(x).groupby(np.arange(n) // 21).sum()
    OUT["acf"][wn] = dict(lags=ac, se=float(se), ljung_box_q10=float(lb), acf_5d_sum=float(w5.autocorr(1)), acf_21d_sum=float(w21.autocorr(1)),
                          sum_lag1_5=float(sum(ac[:5])), sum_lag1_20=float(sum(ac)))
    print(f"  {wn}: lag1..5 {[round(a,3) for a in ac[:5]]} (se {se:.3f}); sum(1-5) {sum(ac[:5]):+.3f}; sum(1-20) {sum(ac):+.3f}; LB(10) {lb:.1f}; weekly-sum acf {w5.autocorr(1):+.3f}; monthly-sum acf {w21.autocorr(1):+.3f}")
    # drawdown-state conditional forward returns (does being in a drawdown forecast anything?)
    eq = r.cumsum(); ddst = eq - eq.cummax()
    fwd21 = r.rolling(21).sum().shift(-21); fwd5 = r.rolling(5).sum().shift(-5)
    states = {"dd<2%": ddst > -0.02, "dd 2-5%": (ddst <= -0.02) & (ddst > -0.05), "dd 5-10%": (ddst <= -0.05) & (ddst > -0.10), "dd>10%": ddst <= -0.10}
    tab = {}
    for k, mk in states.items():
        f21 = fwd21[mk].dropna(); f5 = fwd5[mk].dropna()
        tab[k] = dict(days=int(mk.sum()), fwd21_ann=float(f21.mean() * 12), fwd5_ann=float(f5.mean() * 50), fwd21_sharpe=float(f21.mean() / f21.std() * np.sqrt(12)) if len(f21) > 30 else None)
        print(f"     state {k:9s} days {int(mk.sum()):5d} fwd21 ann {tab[k]['fwd21_ann']:+.1%} fwd5 ann {tab[k]['fwd5_ann']:+.1%}")
    OUT["acf"][wn]["drawdown_state_forward"] = tab
    # trailing-21d return tercile -> forward 21d (momentum in P&L?)
    tr21 = r.rolling(21).sum()
    q = tr21.quantile([1 / 3, 2 / 3])
    for lab, mk in [("bottom tercile", tr21 <= q.iloc[0]), ("top tercile", tr21 > q.iloc[1])]:
        f = fwd21[mk].dropna()
        OUT["acf"][wn][f"trailing21_{lab.split()[0]}_fwd21_ann"] = float(f.mean() * 12)
        print(f"     trailing-21d {lab:14s}: fwd21 ann {f.mean()*12:+.1%}")

# ---------------------------------------------------------------- 5. GBM fractional-Kelly cross-check at the measured haircut
print("\n=== 5. fraction of (haircut) Kelly at each GRM and lifetime P(DD >= x) under GBM ===")
OUT["kelly_fraction"] = {}
for wn, r in WINDOWS.items():
    s_raw = r.mean() / r.std() * np.sqrt(252); vol = r.std() * np.sqrt(252)
    OUT["kelly_fraction"][wn] = {}
    for hc in [0.0, 0.40, 0.71]:
        s = s_raw * (1 - hc)
        kelly_lev = s / vol            # multiple of current size that is full Kelly (GBM)
        for m in [1.0, 1.5, 2.0, 2.5, 3.0]:
            c = m / kelly_lev
            p20 = 0.8 ** (2 / c - 1); p30 = 0.7 ** (2 / c - 1); p50 = 0.5 ** (2 / c - 1)
            OUT["kelly_fraction"][wn][f"hc{hc:g}_m{m:g}"] = dict(kelly_fraction=float(c), growth_share=float(c * (2 - c)), p_ever_dd20=float(p20), p_ever_dd30=float(p30), p_ever_dd50=float(p50), ann_vol=float(m * vol))
        print(f"  {wn} hc{hc:g}: Sharpe {s:.2f} full-Kelly m {kelly_lev:.1f}; at m=1.5 c={1.5/kelly_lev:.3f} P(ever DD20) {0.8**(2/(1.5/kelly_lev)-1):.1%}; m=2 c={2/kelly_lev:.3f} P(ever DD20) {0.8**(2/(2/kelly_lev)-1):.1%}; m=3 c={3/kelly_lev:.3f} P(ever DD20) {0.8**(2/(3/kelly_lev)-1):.1%}")

json.dump(OUT, open(HERE / "growthmax_2_growth_dd_acf.json", "w"), indent=1, default=float)
print("\nwrote growthmax_2_growth_dd_acf.json")
