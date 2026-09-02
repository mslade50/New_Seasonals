"""Flow-conditional sizing, part 3: walk-forward replay of flow-conditional multipliers
per family, WITH the per-strategy 250 bps daily cap re-applied, plus the cap-interaction
accounting (how much of the flow effect the cap already captures / fights).

Usage: python flow_conditional_04_walkforward.py [fills|candidates]
Reads flow_trades_<src>.parquet (from part 1). Writes flow_conditional_walkforward_<src>.json
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from flow_conditional_lib import build_trade_mtm, dd_stats, FAMILIES, OUT_DIR, NAV

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
SRC = sys.argv[1] if len(sys.argv) > 1 else "candidates"
OUT: dict = {"source": SRC}
CAP_BPS = 250.0
CAP_D = CAP_BPS / 1e4 * NAV

tr = pd.read_parquet(OUT_DIR / f"flow_trades_{SRC}.parquet")
tr = tr.sort_values(["Signal Date", "Strategy", "Ticker"]).reset_index(drop=True)
tr["nominal"] = tr["RiskBps"] / 1e4 * NAV * tr["SizeMult"]          # pre-cap staged risk of this (filled) row
tr["cap_bound"] = tr["cap_scale"] < 0.999
# placed-total estimate per (strategy, day): exact on bound days (cap / scale), lower-bounded by filled nominal otherwise
grp = tr.groupby(["Strategy", "Signal Date"])
tr["fills_nominal_day"] = grp["nominal"].transform("sum")
tr["placed_est"] = np.where(tr["cap_bound"], CAP_D / tr["cap_scale"].clip(lower=1e-6), tr["fills_nominal_day"])
tr.loc[tr["Strategy"] == "Overbot Vol Spike", "placed_est"] = np.maximum(tr.loc[tr["Strategy"] == "Overbot Vol Spike", "placed_est"], tr.loc[tr["Strategy"] == "Overbot Vol Spike", "fills_nominal_day"])
print(f"trades {len(tr)}; cap-bound rows {tr.cap_bound.mean():.1%}; cap-bound (strategy,day) groups {tr[tr.cap_bound].groupby(['Strategy','Signal Date']).ngroups}")

days, MTM = build_trade_mtm(tr)
print(f"MTM matrix {MTM.shape}; reconciliation check: {np.abs(MTM.sum(axis=1) - tr.PnL.values).max():.2f}")


def apply_rule(df: pd.DataFrame, mult: np.ndarray, reapply_cap: bool = True) -> np.ndarray:
    """Return per-row risk factor (new_risk / booked_risk) for a per-row pre-cap multiplier, re-applying the daily cap."""
    if not reapply_cap:
        return np.asarray(mult, dtype=float)      # rule on top of the booked cap scale (cap not re-applied)
    # group-level: cap scale after the rule = min(1, cap / (placed * mult_group)); mult is constant within (strategy, day) for flow vars,
    # but allow row-varying mult by using the risk-weighted mean multiplier of the group on the placed estimate.
    d = df[["Strategy", "Signal Date", "nominal", "placed_est", "cap_scale"]].copy()
    d["m"] = mult
    d["mw"] = d["m"] * d["nominal"]
    g = d.groupby(["Strategy", "Signal Date"])
    mbar = g["mw"].transform("sum") / g["nominal"].transform("sum")
    new_scale = np.minimum(1.0, CAP_D / (d["placed_est"].values * mbar.values))
    new_risk = d["nominal"].values * d["m"].values * new_scale
    return new_risk / (d["nominal"].values * d["cap_scale"].values)


def score(df: pd.DataFrame, factor: np.ndarray, idx: np.ndarray) -> dict:
    """factor multiplies booked PnL/Risk row-wise; idx = row positions into MTM."""
    pnl = df["PnL"].values * factor
    risk = df["Risk"].values * factor
    daily = pd.Series((MTM[idx] * factor[:, None]).sum(axis=0), index=days)
    daily = daily[(daily.index >= df["Signal Date"].min()) & (daily.index <= df["ExitDate"].max())]
    st = dd_stats(daily)
    st.update(pnl=float(pnl.sum()), risk=float(risk.sum()), pnl_per_risk=float(pnl.sum() / risk.sum()) if risk.sum() else np.nan,
              avg_factor=float(factor.mean()))
    return st


def fit_mults(train: pd.DataFrame, var: str, shrink: float, lo_clip: float, hi_clip: float, mode: str, min_n: int = 60):
    """Tercile thresholds + multipliers from training trades: mult_b = 1 + shrink * (rpr_b / rpr_all - 1), on PnL-per-risk."""
    if len(train) < min_n or train[var].nunique() < 3:
        return None
    q1, q2 = train[var].quantile(1 / 3), train[var].quantile(2 / 3)
    b = np.where(train[var] <= q1, 0, np.where(train[var] <= q2, 1, 2))
    base = train["PnL"].sum() / train["Risk"].sum()
    mults = []
    for k in range(3):
        g = train[b == k]
        if len(g) < 15 or base <= 0:
            mults.append(1.0); continue
        rel = (g["PnL"].sum() / g["Risk"].sum()) / base
        m = 1 + shrink * (rel - 1)
        if mode == "up_only":
            m = max(1.0, m)
        elif mode == "down_only":
            m = min(1.0, m)
        mults.append(float(np.clip(m, lo_clip, hi_clip)))
    return dict(q1=float(q1), q2=float(q2), mults=mults)


def bucket_mult(df: pd.DataFrame, var: str, fit: dict) -> np.ndarray:
    b = np.where(df[var] <= fit["q1"], 0, np.where(df[var] <= fit["q2"], 1, 2))
    return np.array(fit["mults"])[b]


TEST_YEARS = list(range(2010, 2027))
VARS = ["s1", "s5", "s21", "f5", "f21", "f21_rel", "b5", "nstrat1"]
MODES = ["two_sided", "up_only", "down_only"]

# ------------------------------------------------------------------ 1. walk-forward per family x variable x mode
print("\n=== 1. walk-forward (expanding, annual re-fit, 2010-2026), rule fit per FAMILY, cap re-applied ===")
res_rows, year_rows = [], []
for f in FAMILIES:
    F = tr[tr.family == f]
    fidx = F.index.values
    base = score(F, np.ones(len(F)), fidx)
    for var in VARS:
        for mode in MODES:
            fac = np.ones(len(F)); fac_nocap = np.ones(len(F)); fac_eq = np.ones(len(F))
            fits = {}
            for y in TEST_YEARS:
                trn = F[(F.year < y) & (F.year >= 2005)]
                te_mask = (F.year == y).values
                if te_mask.sum() == 0:
                    continue
                fit = fit_mults(trn, var, shrink=0.5, lo_clip=0.5, hi_clip=1.5, mode=mode)
                if fit is None:
                    continue
                fits[y] = fit
                m_all = bucket_mult(F, var, fit)
                # risk-normalise: scale so training deployed risk under the rule == baseline training risk (keeps total risk comparable)
                trn_mask = ((F.year < y) & (F.year >= 2005)).values
                m_tr = m_all[trn_mask]
                norm = trn["Risk"].sum() / (trn["Risk"].values * apply_rule(trn, m_tr)).sum() if trn_mask.sum() else 1.0
                r_all = apply_rule(F, m_all)
                fac[te_mask] = r_all[te_mask]
                fac_eq[te_mask] = apply_rule(F, m_all * norm)[te_mask]
                fac_nocap[te_mask] = (m_all / F["cap_scale"].values * F["cap_scale"].values)[te_mask]  # rule w/o cap re-application, keeps booked cap
            oos = (F.year >= 2010).values
            Fo, io = F[oos], fidx[oos]
            b0 = score(Fo, np.ones(oos.sum()), io)
            s1 = score(Fo, fac[oos], io)
            s2 = score(Fo, fac_eq[oos], io)
            s3 = score(Fo, fac_nocap[oos], io)
            # per-year comparison
            yb = []
            for y in TEST_YEARS:
                m = (Fo.year == y).values
                if m.sum() == 0:
                    continue
                pb, pr = Fo["PnL"].values[m].sum(), (Fo["PnL"].values[m] * fac_eq[oos][m]).sum()
                yb.append((y, pb, pr))
                year_rows.append(dict(family=f, var=var, mode=mode, year=y, base_pnl=pb, rule_pnl=pr))
            yb = pd.DataFrame(yb, columns=["y", "b", "r"])
            better = int((yb.r > yb.b).sum())
            worst_rel = float(((yb.r - yb.b) / yb.b.abs().clip(lower=1000)).min())
            res_rows.append(dict(family=f, var=var, mode=mode, N=int(oos.sum()), base_pnl=b0["pnl"], base_sharpe=b0["sharpe"], base_maxdd=b0["maxdd"], base_worst21=b0["worst21"],
                                 rule_pnl=s1["pnl"], rule_risk_ratio=s1["risk"] / b0["risk"], rule_sharpe=s1["sharpe"], rule_maxdd=s1["maxdd"],
                                 eq_pnl=s2["pnl"], eq_risk_ratio=s2["risk"] / b0["risk"], eq_sharpe=s2["sharpe"], eq_maxdd=s2["maxdd"], eq_worst21=s2["worst21"],
                                 eq_pnl_per_risk=s2["pnl_per_risk"], base_pnl_per_risk=b0["pnl_per_risk"],
                                 nocap_pnl=s3["pnl"], nocap_sharpe=s3["sharpe"], years_better=better, years=len(yb), worst_year_rel=worst_rel,
                                 last_fit=fits.get(2026) or fits.get(max(fits)) if fits else None))
WF = pd.DataFrame(res_rows)
show = ["family", "var", "mode", "N", "base_pnl", "eq_pnl", "eq_risk_ratio", "base_sharpe", "eq_sharpe", "base_maxdd", "eq_maxdd", "base_pnl_per_risk", "eq_pnl_per_risk", "years_better", "years", "worst_year_rel", "nocap_pnl"]
for f in FAMILIES:
    print(f"\n--- {f} ---")
    print(WF[WF.family == f][show].round(3).to_string(index=False))
OUT["walkforward"] = WF.drop(columns=["last_fit"]).round(4).to_dict("records")
OUT["walkforward_last_fit"] = {f"{r.family}|{r['var']}|{r['mode']}": r["last_fit"] for _, r in WF.iterrows()}
OUT["walkforward_years"] = pd.DataFrame(year_rows).round(1).to_dict("records")

# ------------------------------------------------------------------ 2. cap interaction accounting
print("\n=== 2. the 250 bps per-strategy cap vs flow: where it binds, what it costs, what it saves ===")
rows = []
for f in FAMILIES:
    F = tr[tr.family == f]
    for var in ["s1", "s5", "f5"]:
        q1, q2 = F[var].quantile(1 / 3), F[var].quantile(2 / 3)
        b = np.where(F[var] <= q1, "lo", np.where(F[var] <= q2, "mid", "hi"))
        for lab in ["lo", "mid", "hi"]:
            g = F[b == lab]
            if len(g) == 0:
                continue
            unc = g["PnL"] / g["cap_scale"]
            rows.append(dict(family=f, var=var, bucket=lab, N=len(g), share_cap_bound=float(g.cap_bound.mean()), mean_cap_scale=float(g.cap_scale.mean()),
                             avgR=float(g.R.mean()), pnl_capped=float(g.PnL.sum()), pnl_uncapped=float(unc.sum()),
                             cap_cost=float(unc.sum() - g.PnL.sum()), risk_capped=float(g.Risk.sum()), risk_uncapped=float((g.Risk / g.cap_scale).sum()),
                             rpr_capped=float(g.PnL.sum() / g.Risk.sum()), rpr_uncapped=float(unc.sum() / (g.Risk / g.cap_scale).sum())))
CI = pd.DataFrame(rows)
print(CI.round(3).to_string(index=False))
OUT["cap_by_flow"] = CI.round(4).to_dict("records")

# per-strategy: cap-bound days' avgR vs unbound, and worst-day protection (realized-at-exit day pnl capped vs uncapped)
rows = []
for s, g in tr.groupby("Strategy"):
    bnd, unb = g[g.cap_bound], g[~g.cap_bound]
    day_c = g.groupby("Signal Date")["PnL"].sum(); day_u = (g["PnL"] / g["cap_scale"]).groupby(g["Signal Date"]).sum()
    rows.append(dict(strategy=s, N=len(g), bound_share=float(g.cap_bound.mean()), bound_days=int(bnd["Signal Date"].nunique()),
                     avgR_bound=float(bnd.R.mean()) if len(bnd) else np.nan, avgR_unbound=float(unb.R.mean()),
                     pnl_capped=float(g.PnL.sum()), pnl_uncapped=float((g.PnL / g.cap_scale).sum()),
                     worst_signal_day_capped=float(day_c.min()), worst_signal_day_uncapped=float(day_u.min()),
                     mean_scale_when_bound=float(bnd.cap_scale.mean()) if len(bnd) else np.nan))
CS = pd.DataFrame(rows)
print("\nper strategy:\n", CS.round(3).to_string(index=False))
OUT["cap_by_strategy"] = CS.round(4).to_dict("records")

# ------------------------------------------------------------------ 3. cap-relief variants: raise the cap on high-flow days (walk-forward: fixed rule, no fit)
print("\n=== 3. cap-relief variants (no fitting): per-strategy cap x k on days in the family's top flow tercile (thresholds from < 2010 data) ===")
rows = []
for f in FAMILIES:
    F = tr[(tr.family == f)]
    Fo = F[F.year >= 2010]; io = Fo.index.values
    base = score(Fo, np.ones(len(Fo)), io)
    for var in ["s1", "f5"]:
        thr = F[F.year < 2010][var].quantile(2 / 3)
        hi = (Fo[var] > thr).values
        for k in [1.5, 2.0, 99.0]:
            # new cap on hi-flow days: scale = min(1, k*cap/placed) -> factor relative to booked
            new_scale = np.where(hi, np.minimum(1.0, k * CAP_D / Fo["placed_est"].values), Fo["cap_scale"].values)
            fac = new_scale / Fo["cap_scale"].values
            s = score(Fo, fac, io)
            rows.append(dict(family=f, var=var, k=k, hi_share=float(hi.mean()), base_pnl=base["pnl"], pnl=s["pnl"], d_pnl=s["pnl"] - base["pnl"],
                             risk_ratio=s["risk"] / base["risk"], base_sharpe=base["sharpe"], sharpe=s["sharpe"], base_maxdd=base["maxdd"], maxdd=s["maxdd"],
                             base_worst21=base["worst21"], worst21=s["worst21"], pnl_per_risk=s["pnl_per_risk"], base_ppr=base["pnl_per_risk"]))
CR = pd.DataFrame(rows)
print(CR.round(3).to_string(index=False))
OUT["cap_relief"] = CR.round(4).to_dict("records")

# ------------------------------------------------------------------ 4. book-level: combine the best-supported per-family rule (pre-specified: f5 two-sided, eq-risk) and report the book
print("\n=== 4. book-level walk-forward: f5 two-sided rule on dip_buy / oversold_hold / short_fade, breakout down-only on f5; cap re-applied ===")
fac = np.ones(len(tr))
for f in FAMILIES:
    F = tr[tr.family == f]; fidx = F.index.values
    mode = "down_only" if f == "breakout" else "two_sided"
    for y in TEST_YEARS:
        trn = F[(F.year < y) & (F.year >= 2005)]; te = (F.year == y).values
        fit = fit_mults(trn, "f5", 0.5, 0.5, 1.5, mode)
        if fit is None or te.sum() == 0:
            continue
        m_all = bucket_mult(F, "f5", fit)
        trn_mask = ((F.year < y) & (F.year >= 2005)).values
        norm = trn["Risk"].sum() / (trn["Risk"].values * apply_rule(trn, m_all[trn_mask])).sum()
        fac[fidx[te]] = apply_rule(F, m_all * norm)[te]
oos = (tr.year >= 2010).values
B0 = score(tr[oos], np.ones(oos.sum()), tr.index.values[oos]); B1 = score(tr[oos], fac[oos], tr.index.values[oos])
print("baseline:", {k: round(v, 3) for k, v in B0.items()})
print("flow f5 :", {k: round(v, 3) for k, v in B1.items()})
yrs = []
for y in TEST_YEARS:
    m = (tr.year == y).values & oos
    yrs.append(dict(year=y, base=float(tr.PnL.values[m].sum()), rule=float((tr.PnL.values[m] * fac[m]).sum())))
Yb = pd.DataFrame(yrs); print(Yb.round(0).to_string(index=False)); print("years better:", int((Yb.rule > Yb.base).sum()), "of", len(Yb))
OUT["book_f5"] = dict(baseline=B0, rule=B1, years=Yb.round(0).to_dict("records"))

json.dump(OUT, open(OUT_DIR / f"flow_conditional_walkforward_{SRC}.json", "w"), indent=1, default=float)
print(f"wrote flow_conditional_walkforward_{SRC}.json")
