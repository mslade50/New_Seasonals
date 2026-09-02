"""Within-strategy adds, step 2: marginal edge and marginal sleeve risk of a leg by
(a) open legs, (b) open legs in the same sector, (c) same ticker, (d) trailing-63d
correlation with the open names, (e) stack age. Episode-clustered t vs the solo
bucket, LOYO floor, and the ex-post Euler share of sleeve variance per bucket
(cov(leg MTM, sleeve MTM) / var(sleeve) on the leg's open days) so that PnL per
unit of MARGINAL variance can be compared with PnL per unit of ATR risk.
Writes within_strategy_adds_marginal.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
df = pd.read_parquet(OUT / "within_strategy_adds_features.parquet")
M = pd.read_parquet(OUT / "within_strategy_adds_mtm.parquet")
RES: dict = {}

# ---- episodes: connected chains of overlapping legs within a strategy
def episodes(g):
    g = g.sort_values("Entry Date")
    ep = np.zeros(len(g), dtype=int); cur = 0; cur_exit = None
    for k, (a, b) in enumerate(zip(g["Entry Date"], g["Exit Date"])):
        if cur_exit is None or a > cur_exit:
            cur += 1; cur_exit = b
        else:
            cur_exit = max(cur_exit, b)
        ep[k] = cur
    return pd.Series(ep, index=g.index)
df["episode"] = 0
for s, g in df.groupby("Strategy"):
    df.loc[g.index, "episode"] = episodes(g).values

def cl_se(x, c):
    x = np.asarray(x, float); mu = x.mean()
    resid = pd.Series(x - mu).groupby(np.asarray(c)).sum()
    return float(np.sqrt((resid ** 2).sum()) / len(x)), mu

# ---- ex-post Euler share per leg
euler = {}
for s, g in df.groupby("Strategy"):
    m = M[M.Strategy == s]
    sleeve = m.groupby("date").pnl.sum()
    var_s = sleeve.var()
    for idx, mm in m.groupby("idx"):
        x = mm.set_index("date").pnl
        y = sleeve.reindex(x.index)
        if len(x) >= 2:
            cov = float(np.mean((x - x.mean()) * (y - y.mean())))
        else:
            cov = float(x.iloc[0] * y.iloc[0]) if len(x) else 0.0
        # contribution to sleeve variance over the leg's life, in $^2 per day; sum over days = leg's Euler share of sleeve sum-of-squares
        euler[idx] = dict(euler_var=float(((x - 0) * (y - 0)).sum()), days=len(x), leg_sd=float(x.std()) if len(x) > 1 else abs(float(x.iloc[0])))
E = pd.DataFrame(euler).T
df = df.join(E)

# ---- bucket definitions
def bucketize(s):
    g = df[df.Strategy == s].copy()
    g["b_open"] = pd.cut(g.n_open, [-1, 0, 1, 2, 5, 12, 99], labels=["0", "1", "2", "3-5", "6-12", "13+"])
    g["b_sec"] = np.where(g.n_open == 0, "solo", np.where(g.n_same_sector == 0, "0 same-sec", np.where(g.n_same_sector == 1, "1 same-sec", "2+ same-sec")))
    g["b_tk"] = np.where(g.n_open == 0, "solo", np.where(g.n_same_ticker == 0, "0 same-tk", "1+ same-tk"))
    g["b_rho"] = np.where(g.n_open == 0, "solo", pd.cut(g.rho63_mean, [-1.1, 0.2, 0.4, 0.6, 1.01], labels=["rho<.2", ".2-.4", ".4-.6", ">.6"]).astype(str))
    g["b_age"] = np.where(g.n_open == 0, "solo", pd.cut(g.stack_age_td, [-1, 0, 2, 5, 10, 999], labels=["same-day", "1-2td", "3-5td", "6-10td", "11+td"]).astype(str))
    g["b_sameday"] = np.where(g.n_open == 0, "solo", np.where(g.same_day_prior >= 1, "same-day cluster", "earlier-day stack"))
    return g

def table(g, col, solo_label):
    rows = []
    solo = g[g[col] == solo_label]
    se0, mu0 = cl_se(solo.R_Multiple, solo.episode) if len(solo) > 1 else (np.nan, np.nan)
    for b, x in g.groupby(col, observed=True):
        if len(x) == 0:
            continue
        se, mu = cl_se(x.R_Multiple, x.episode) if len(x) > 1 else (np.nan, float(x.R_Multiple.mean()))
        t = (mu - mu0) / np.sqrt(se ** 2 + se0 ** 2) if b != solo_label and np.isfinite(se) and np.isfinite(se0) and (se + se0) > 0 else np.nan
        # LOYO floor of (bucket - solo) t
        floor = np.nan
        if b != solo_label and len(x) >= 8:
            ts = []
            for y in sorted(x.yr.unique()):
                xx, ss = x[x.yr != y], solo[solo.yr != y]
                if len(xx) > 3 and len(ss) > 3:
                    a1, m1 = cl_se(xx.R_Multiple, xx.episode); a0, m0 = cl_se(ss.R_Multiple, ss.episode)
                    ts.append((m1 - m0) / np.sqrt(a1 ** 2 + a0 ** 2) if (a1 + a0) > 0 else np.nan)
            floor = float(np.nanmin(ts)) if ts else np.nan
        risk = x.Risk_flat_750k.sum(); pnl = x.PnL_flat_750k.sum()
        ev = x.euler_var.sum(); ev_tot = g.euler_var.sum()
        rows.append(dict(bucket=str(b), N=int(len(x)), episodes=int(x.episode.nunique()), avgR=float(mu), sdR=float(x.R_Multiple.std()),
                         win=float((x.R_Multiple > 0).mean()), t_vs_solo=float(t) if np.isfinite(t) else None,
                         loyo_floor_t=float(floor) if np.isfinite(floor) else None, pnl=float(pnl), risk=float(risk),
                         pnl_per_risk=float(pnl / risk) if risk else None, share_pnl=float(pnl / g.PnL_flat_750k.sum()),
                         share_risk=float(risk / g.Risk_flat_750k.sum()), share_euler_var=float(ev / ev_tot) if ev_tot else None,
                         pnl_per_euler=float(pnl / ev * 1e6) if ev else None,
                         unit_avgR=float((x.unit_pnl / x.unit_risk).mean()), size_mult=float(x.Size_Mult.mean()),
                         rho63=float(x.rho63_mean.mean()) if x.rho63_mean.notna().any() else None,
                         yrs_pos=f"{int((x.groupby('yr').R_Multiple.mean() > 0).sum())}/{x.yr.nunique()}"))
    return rows

for s in sorted(df.Strategy.unique()):
    g = bucketize(s)
    RES[s] = {}
    print(f"\n==================== {s}  N={len(g)}  episodes={g.episode.nunique()}  ({g.yr.min()}-{g.yr.max()})")
    for col, lab, solo in [("b_open", "(a) open legs", "0"), ("b_sec", "(b) same-sector open legs", "solo"), ("b_tk", "(c) same ticker", "solo"),
                           ("b_rho", "(d) rho63 with open names", "solo"), ("b_age", "(e) stack age", "solo"), ("b_sameday", "(e2) same-day vs earlier", "solo")]:
        rows = table(g, col, solo)
        RES[s][col] = rows
        T = pd.DataFrame(rows)[["bucket", "N", "episodes", "avgR", "win", "t_vs_solo", "loyo_floor_t", "pnl", "pnl_per_risk", "share_pnl", "share_risk", "share_euler_var", "pnl_per_euler", "rho63", "yrs_pos"]]
        print(f"--- {lab}"); print(T.to_string(index=False))
    # 2-way: open legs x same-sector share (conditional on n_open >= 1)
    x = g[g.n_open >= 1].copy()
    x["sec_share"] = np.where(x.n_same_sector / x.n_open >= 0.5, "sec-share>=50%", "sec-share<50%")
    x["depth"] = np.where(x.n_open >= 3, "3+ open", "1-2 open")
    two = x.groupby(["depth", "sec_share"], observed=True).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum"), rho=("rho63_mean", "mean")).reset_index()
    print("--- depth x sector-share:"); print(two.round(3).to_string(index=False))
    RES[s]["depth_x_sector"] = two.round(4).to_dict("records")
    # rho as a continuous predictor of R among adds (Spearman), and partial of sector given rho
    if x.rho63_mean.notna().sum() > 20:
        sp = float(x.rho63_mean.rank().corr(x.R_Multiple.rank()))
        sp_sec = float(x.n_same_sector.rank().corr(x.R_Multiple.rank()))
        sp_n = float(x.n_open.rank().corr(x.R_Multiple.rank()))
        sp_age = float(x.stack_age_td.rank().corr(x.R_Multiple.rank()))
        print(f"Spearman(R, rho63)={sp:+.3f}  Spearman(R, n_same_sector)={sp_sec:+.3f}  Spearman(R, n_open)={sp_n:+.3f}  Spearman(R, age)={sp_age:+.3f}  (adds only, N={len(x)})")
        RES[s]["spearman_adds"] = dict(rho63=sp, n_same_sector=sp_sec, n_open=sp_n, stack_age=sp_age, N=int(len(x)))

# ---- realised sleeve variance scaling: sd(sleeve)/(sd1*sqrt(n)) recomputed from the leg MTM (ex-post rho by open count), all strategies
print("\n==== sleeve daily sd by open-leg count (from leg MTM), implied pairwise rho ====")
imp = {}
for s in sorted(df.Strategy.unique()):
    m = M[M.Strategy == s]
    sleeve = m.groupby("date").pnl.sum()
    nleg = m.groupby("date").idx.nunique()
    p = pd.DataFrame({"pnl": sleeve, "n": nleg})
    p = p[p.index >= "2010-01-01"]
    one = p[p.n == 1].pnl.std()
    rows = []
    for lo, hi, lab in [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 99, "13+")]:
        q = p[(p.n >= lo) & (p.n <= hi)]
        if len(q) < 25:
            continue
        n = q.n.mean(); rho = (q.pnl.var() / (n * one ** 2) - 1) / (n - 1) if n > 1 else np.nan
        rows.append(dict(legs=lab, days=int(len(q)), n_mean=float(n), sd_vs_sqrtn=float(q.pnl.std() / (one * np.sqrt(n))), implied_rho=float(rho), mean_per_leg=float(q.pnl.mean() / n), sd_per_leg=float(q.pnl.std() / n)))
    imp[s] = rows
    print(s); print(pd.DataFrame(rows).round(3).to_string(index=False))
RES["_implied_rho_from_leg_mtm"] = imp

df.to_parquet(OUT / "within_strategy_adds_features.parquet")
json.dump(RES, open(OUT / "within_strategy_adds_marginal.json", "w"), indent=1, default=float)
print("\nwrote within_strategy_adds_marginal.json")
