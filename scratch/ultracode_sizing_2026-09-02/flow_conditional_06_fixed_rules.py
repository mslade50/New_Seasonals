"""Flow-conditional sizing, part 5: replay of FIXED, live-implementable rules (no fitting inside the
replay; multipliers pre-set to 0.75 / 1.0 / 1.25, thresholds = integer candidate counts), with the
per-strategy 250 bps daily cap re-applied, sensitivity to thresholds/multipliers, dial gating for the
dip-buy family, and the flow-aware cap variant. 2010-2026 on the candidate-based flow.
Reads flow_trades_candidates.parquet. Writes flow_conditional_fixed_rules.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from flow_conditional_lib import build_trade_mtm, dd_stats, FAMILIES, OUT_DIR, NAV, ROOT

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
CAP_D = 250 / 1e4 * NAV
tr = pd.read_parquet(OUT_DIR / "flow_trades_candidates.parquet").sort_values(["Signal Date", "Strategy", "Ticker"]).reset_index(drop=True)
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
tr["dial"] = frag["63d"].rolling(10).mean().shift(1).reindex(tr["Signal Date"]).values
tr["nominal"] = tr["RiskBps"] / 1e4 * NAV * tr["SizeMult"]
tr["cap_bound"] = tr["cap_scale"] < 0.999
g = tr.groupby(["Strategy", "Signal Date"])
tr["fills_nominal_day"] = g["nominal"].transform("sum")
tr["placed_est"] = np.where(tr["cap_bound"], CAP_D / tr["cap_scale"].clip(lower=1e-6), tr["fills_nominal_day"])
days, MTM = build_trade_mtm(tr)
oos = (tr.year >= 2010).values
T = tr[oos].reset_index(drop=True); M = MTM[oos]


def apply(df, mult, cap_mult=None):
    """row factor (new risk / booked risk) after re-applying the per-strategy cap (cap_mult raises the cap on the row's day)."""
    d = df[["Strategy", "Signal Date", "nominal", "placed_est", "cap_scale"]].copy()
    d["m"] = mult; d["mw"] = d["m"] * d["nominal"]
    gg = d.groupby(["Strategy", "Signal Date"])
    mbar = gg["mw"].transform("sum") / gg["nominal"].transform("sum")
    cm = np.ones(len(d)) if cap_mult is None else np.asarray(cap_mult, dtype=float)
    new_scale = np.minimum(1.0, cm * CAP_D / (d["placed_est"].values * mbar.values))
    return d["nominal"].values * d["m"].values * new_scale / (d["nominal"].values * d["cap_scale"].values)


def score(df, fac, rows):
    daily = pd.Series((M[rows] * fac[:, None]).sum(axis=0), index=days)
    daily = daily[daily.index >= "2010-01-01"]
    st = dd_stats(daily)
    pnl = df["PnL"].values * fac; risk = df["Risk"].values * fac
    st.update(pnl=float(pnl.sum()), risk=float(risk.sum()), pnl_per_risk=float(pnl.sum() / risk.sum()), avg_factor=float(fac.mean()))
    yrs = pd.DataFrame({"y": df["year"].values, "b": df["PnL"].values, "r": pnl}).groupby("y").sum()
    st.update(years_better=int((yrs.r > yrs.b).sum()), years=len(yrs), worst_year_rel=float(((yrs.r - yrs.b) / yrs.b.abs().clip(lower=1000)).min()),
              years_table={int(k): [float(v.b), float(v.r)] for k, v in yrs.iterrows()})
    return st


def fixed_rule(df, var, t_lo, t_hi, m_lo, m_hi, gate_dial=None):
    m = np.where(df[var] <= t_lo, m_lo, np.where(df[var] >= t_hi, m_hi, 1.0)).astype(float)
    if gate_dial is not None:   # no flow adjustment when the dial is >= gate (frag bands own that zone); NaN dial (pre-2016) -> rule applies
        m = np.where(df["dial"].values >= gate_dial, 1.0, m)
    return m


RULES = {  # family: (var, t_lo, t_hi) from the 2026 expanding-window terciles, rounded to integers
    "dip_buy": ("f5", 3, 6), "oversold_hold": ("f5", 2, 7), "short_fade": ("f5", 23, 104), "breakout": ("f5", 1, 3),
}
print("=== fixed rules: mult 0.75 at flow <= t_lo, 1.25 at flow >= t_hi, else 1.0; cap re-applied; 2010-2026 ===")
rows = []
for f in FAMILIES:
    var, t_lo, t_hi = RULES[f]
    F = T[T.family == f]; ridx = F.index.values
    base = score(F, np.ones(len(F)), ridx)
    variants = {
        "baseline": (np.ones(len(F)), None),
        "flow 0.75/1.25": (apply(F, fixed_rule(F, var, t_lo, t_hi, 0.75, 1.25)), None),
        "flow 0.75/1.25, eq-risk": (None, None),
        "flow up-only 1.25": (apply(F, fixed_rule(F, var, t_lo, t_hi, 1.0, 1.25)), None),
        "flow down-only 0.75": (apply(F, fixed_rule(F, var, t_lo, t_hi, 0.75, 1.0)), None),
        "flow 0.5/1.5": (apply(F, fixed_rule(F, var, t_lo, t_hi, 0.5, 1.5)), None),
        "flow 0.75/1.25 + cap x1.5 on hi": (apply(F, fixed_rule(F, var, t_lo, t_hi, 0.75, 1.25), cap_mult=np.where(F[var] >= t_hi, 1.5, 1.0)), None),
        "cap x1.5 on hi only": (apply(F, np.ones(len(F)), cap_mult=np.where(F[var] >= t_hi, 1.5, 1.0)), None),
        "flow 0.75/1.25, dial<50 gate": (apply(F, fixed_rule(F, var, t_lo, t_hi, 0.75, 1.25, gate_dial=50)), None),
        f"thr sens: t_hi={t_hi-1}": (apply(F, fixed_rule(F, var, t_lo, t_hi - 1, 0.75, 1.25)), None),
        f"thr sens: t_hi={t_hi+1}": (apply(F, fixed_rule(F, var, t_lo, t_hi + 1, 0.75, 1.25)), None),
        f"thr sens: t_lo={t_lo+1}": (apply(F, fixed_rule(F, var, t_lo + 1, t_hi, 0.75, 1.25)), None),
    }
    fac_main = variants["flow 0.75/1.25"][0]
    variants["flow 0.75/1.25, eq-risk"] = (fac_main * (F["Risk"].sum() / (F["Risk"].values * fac_main).sum()), None)
    variants["cap x2.0 on hi only"] = (apply(F, np.ones(len(F)), cap_mult=np.where(F[var] >= t_hi, 2.0, 1.0)), None)
    variants["flow up-only 1.25 + cap x1.5 on hi"] = (apply(F, fixed_rule(F, var, t_lo, t_hi, 1.0, 1.25), cap_mult=np.where(F[var] >= t_hi, 1.5, 1.0)), None)
    # cap-absorption: same rule, cap NOT re-applied (rule rides on the booked cap scale) -> how much of the rule's up-sizing the cap eats
    m_up = fixed_rule(F, var, t_lo, t_hi, 1.0, 1.25)
    variants["flow up-only 1.25, cap NOT re-applied"] = (m_up, None)
    hi = (F[var] >= t_hi).values
    absorbed = 1 - ((apply(F, m_up)[hi] - 1).sum() / (m_up[hi] - 1).sum())
    OUT.setdefault("cap_absorbs_share_of_upsize", {})[f] = float(absorbed)
    print(f"{f}: on hi-flow days the cap re-application removes {absorbed:.1%} of the 1.25x up-size (booked mean cap scale on those days {F.cap_scale.values[hi].mean():.3f})")
    for lab, (fac, _) in variants.items():
        s = score(F, fac, ridx)
        rows.append(dict(family=f, rule=lab, var=var, t_lo=t_lo, t_hi=t_hi, N=len(F), share_lo=float((F[var] <= t_lo).mean()), share_hi=float((F[var] >= t_hi).mean()),
                         pnl=s["pnl"], d_pnl_pct=(s["pnl"] / base["pnl"] - 1) * 100, risk_ratio=s["risk"] / base["risk"], pnl_per_risk=s["pnl_per_risk"],
                         sharpe=s["sharpe"], maxdd=s["maxdd"], worst21=s["worst21"], years_better=s["years_better"], years=s["years"], worst_year_rel=s["worst_year_rel"]))
        OUT.setdefault("years", {})[f"{f}|{lab}"] = s["years_table"]
R = pd.DataFrame(rows)
for f in FAMILIES:
    print(f"\n--- {f} ---")
    print(R[R.family == f].drop(columns=["family", "var", "t_lo", "t_hi", "N"]).round(3).to_string(index=False))
OUT["fixed_rules"] = R.round(4).to_dict("records")

# ---- book-level combination: dip_buy + oversold_hold + short_fade with their fixed rules (breakout untouched), cap re-applied
print("\n=== book: fixed flow rules on the three reverting families, breakout untouched ===")
fac = np.ones(len(T))
for f in ["dip_buy", "oversold_hold", "short_fade"]:
    var, t_lo, t_hi = RULES[f]
    F = T[T.family == f]
    fac[F.index.values] = apply(F, fixed_rule(F, var, t_lo, t_hi, 0.75, 1.25, gate_dial=50 if f == "dip_buy" else None))
b0 = score(T, np.ones(len(T)), T.index.values); b1 = score(T, fac, T.index.values)
fac_eq = fac * (T["Risk"].sum() / (T["Risk"].values * fac).sum()); b2 = score(T, fac_eq, T.index.values)
fac_up = np.ones(len(T))
for f in ["dip_buy", "oversold_hold", "short_fade"]:
    var, t_lo, t_hi = RULES[f]; F = T[T.family == f]
    fac_up[F.index.values] = apply(F, fixed_rule(F, var, t_lo, t_hi, 1.0, 1.25, gate_dial=50 if f == "dip_buy" else None))
b3 = score(T, fac_up, T.index.values)
fac_cap = np.ones(len(T)); fac_both = np.ones(len(T))
for f in ["dip_buy", "oversold_hold", "short_fade"]:
    var, t_lo, t_hi = RULES[f]; F = T[T.family == f]; cm = np.where(F[var] >= t_hi, 1.5, 1.0)
    fac_cap[F.index.values] = apply(F, np.ones(len(F)), cap_mult=cm)
    fac_both[F.index.values] = apply(F, fixed_rule(F, var, t_lo, t_hi, 1.0, 1.25, gate_dial=50 if f == "dip_buy" else None), cap_mult=cm)
b4 = score(T, fac_cap, T.index.values); b5 = score(T, fac_both, T.index.values)
fac_dn = np.ones(len(T))
for f in ["dip_buy", "oversold_hold", "short_fade"]:
    var, t_lo, t_hi = RULES[f]; F = T[T.family == f]
    fac_dn[F.index.values] = apply(F, fixed_rule(F, var, t_lo, t_hi, 0.75, 1.0))
b6 = score(T, fac_dn, T.index.values)
for lab, s in [("baseline", b0), ("flow 0.75/1.25 (raw)", b1), ("flow 0.75/1.25 (eq-risk)", b2), ("flow up-only 1.25", b3),
               ("cap x1.5 on hi-flow days", b4), ("up-only 1.25 + cap x1.5", b5), ("flow down-only 0.75", b6)]:
    print(f"{lab:28s} pnl {s['pnl']:>12,.0f}  risk x{s['risk']/b0['risk']:.3f}  pnl/risk {s['pnl_per_risk']:.3f}  sharpe {s['sharpe']:.3f}  maxDD {s['maxdd']:>10,.0f}  worst21 {s['worst21']:>10,.0f}  yrs better {s['years_better']}/{s['years']}  worst yr {s['worst_year_rel']:+.2f}")
    OUT.setdefault("book", {})[lab] = {k: v for k, v in s.items() if k != "years_table"}
    OUT["book"][lab]["years_table"] = s["years_table"]

# ---- bootstrap on the book-level gain: resample years (block = calendar year) of the eq-risk rule minus baseline
yt = pd.DataFrame(b2["years_table"]).T; yt.columns = ["b", "r"]; d = (yt.r - yt.b).values
rng = np.random.default_rng(5); bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(5000)])
print(f"eq-risk gain {d.sum():,.0f} over {len(d)} years; year-bootstrap P(gain<=0) = {(bs <= 0).mean():.4f}; 5th pct {np.percentile(bs, 5):,.0f}")
OUT["book"]["year_bootstrap"] = dict(gain=float(d.sum()), p_le_0=float((bs <= 0).mean()), p5=float(np.percentile(bs, 5)))

json.dump(OUT, open(OUT_DIR / "flow_conditional_fixed_rules.json", "w"), indent=1, default=float)
print("wrote flow_conditional_fixed_rules.json")
