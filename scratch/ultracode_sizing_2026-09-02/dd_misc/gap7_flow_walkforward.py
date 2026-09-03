"""GAP 7: are the flow thresholds (dip_buy >= 6, oversold_hold >= 7, short_fade >= 104)
in-sample at the level?  Re-set the terciles WALK-FORWARD (expanding window of
trades through year Y-1 defines year Y's terciles, trade-level quantiles exactly as
flow_conditional_04's fit_mults did) and recompute (a) the top-vs-bottom tercile
PnL-per-unit-risk ratio, (b) the 1.2x up-size replay with the 250 cap re-applied,
(c) per-year flips, (d) hi-flow episode counts, (e) the 2026 OVS cluster days.
Reads flow_trades_candidates.parquet (f5 = family trailing-5d raw candidate count,
inclusive, lib FAMILY membership) and flow_candidates.parquet. Writes gap7_flow_walkforward.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIZ = HERE.parent
sys.path.insert(0, str(SIZ))
from flow_conditional_lib import (build_trade_mtm, dd_stats, episodes, cluster_boot_diff, daily_counts, trailing,  # noqa: E402
                                  load_candidates, FAMILY, OUT_DIR, NAV, BDAYS, ROOT)

pd.set_option("display.width", 250, "display.max_columns", 40)
OUT: dict = {}
CAP_D = 250 / 1e4 * NAV
FAMS = ["dip_buy", "oversold_hold", "short_fade"]
FIXED = {"dip_buy": (3, 6), "oversold_hold": (2, 7), "short_fade": (23, 104)}
UP = 1.2

tr = pd.read_parquet(OUT_DIR / "flow_trades_candidates.parquet").sort_values(["Signal Date", "Strategy", "Ticker"]).reset_index(drop=True)
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
tr["dial"] = frag["63d"].rolling(10).mean().shift(1).reindex(tr["Signal Date"]).values
tr["nominal"] = tr["RiskBps"] / 1e4 * NAV * tr["SizeMult"]
tr["cap_bound"] = tr["cap_scale"] < 0.999
g = tr.groupby(["Strategy", "Signal Date"])
tr["fills_nominal_day"] = g["nominal"].transform("sum")
tr["placed_est"] = np.where(tr["cap_bound"], CAP_D / tr["cap_scale"].clip(lower=1e-6), tr["fills_nominal_day"])
print(f"trades {len(tr)} years {tr.year.min()}-{tr.year.max()} last signal {tr['Signal Date'].max().date()}")
OUT["meta"] = dict(trades=int(len(tr)), last_signal=str(tr["Signal Date"].max().date()))

# ---------------------------------------------------------------- membership check: brief WP8 vs lib FAMILY
cand = load_candidates()
WP8 = {"Weak Close Decent Sznls": "dip_buy", "SPY QQQ MonFri Reversion": "dip_buy", "Monday Dip": "dip_buy", "Indices Oversold Bounce": "dip_buy",
       "Monthly Weak Close": "dip_buy", "St OS Sznl": "dip_buy", "Oversold Low Volume": "oversold_hold", "LT Trend ST OS": "oversold_hold",
       "Overbot Vol Spike": "short_fade", "3x ETF Overbot Fade": "short_fade", "ATR Extended Gap Up": "short_fade",
       "3x Bear ETF Overbot Fade": "bear_etf_fade", "3x Leader Gap Fade": "bear_etf_fade", "52wh Breakout": "breakout", "Sector BO": "breakout"}
diffs = {s: (FAMILY.get(s), WP8.get(s)) for s in FAMILY if FAMILY.get(s) != WP8.get(s)}
print("\nfamily membership differences (study lib -> brief WP8):", diffs)
OUT["membership_diffs"] = diffs
cand2 = cand.copy(); cand2["family"] = cand2["strategy"].map(WP8)
cf_lib = trailing(daily_counts(cand, "family"), 5); cf_wp8 = trailing(daily_counts(cand2, "family"), 5)
memb = {}
for f in FAMS:
    a = cf_lib[f].reindex(tr.loc[tr.family == f, "Signal Date"]).values
    b = cf_wp8[f].reindex(tr.loc[tr.family == f, "Signal Date"]).values
    t_lo, t_hi = FIXED[f]
    memb[f] = dict(q2_lib=float(np.quantile(a, 2 / 3)), q2_wp8=float(np.quantile(b, 2 / 3)), share_hi_lib=float((a >= t_hi).mean()), share_hi_wp8=float((b >= t_hi).mean()),
                   share_days_hi_lib=float((cf_lib[f][cf_lib.index >= "2005-01-01"] >= t_hi).mean()), share_days_hi_wp8=float((cf_wp8[f][cf_wp8.index >= "2005-01-01"] >= t_hi).mean()))
    print(f"  {f:14s} trade-level q2: lib {memb[f]['q2_lib']:.1f} vs WP8 {memb[f]['q2_wp8']:.1f}; share of trades >= {t_hi}: lib {memb[f]['share_hi_lib']:.1%} vs WP8 {memb[f]['share_hi_wp8']:.1%}")
OUT["membership_threshold_shift"] = memb

# ---------------------------------------------------------------- walk-forward terciles
YEARS = list(range(2010, 2027))
tr["hi_wf"] = False; tr["lo_wf"] = False; tr["q2_wf"] = np.nan; tr["q1_wf"] = np.nan
wf_thr = {}
for f in FAMS:
    F = tr[tr.family == f]
    for y in YEARS:
        trn = F[(F.year >= 2005) & (F.year < y)]
        te = F.index[F.year == y]
        if len(trn) < 60 or len(te) == 0:
            continue
        q1, q2 = trn["f5"].quantile(1 / 3), trn["f5"].quantile(2 / 3)
        wf_thr.setdefault(f, {})[y] = dict(q1=float(q1), q2=float(q2), n_train=int(len(trn)))
        tr.loc[te, "hi_wf"] = (F.loc[te, "f5"] > q2).values
        tr.loc[te, "lo_wf"] = (F.loc[te, "f5"] <= q1).values
        tr.loc[te, "q2_wf"] = q2; tr.loc[te, "q1_wf"] = q1
tr["hi_fx"] = [v >= FIXED[f][1] if f in FIXED else False for v, f in zip(tr.f5, tr.family)]
tr["lo_fx"] = [v <= FIXED[f][0] if f in FIXED else False for v, f in zip(tr.f5, tr.family)]
OUT["wf_thresholds"] = wf_thr
print("\nwalk-forward q2 (top-tercile line, strictly above) by year:")
for f in FAMS:
    print(f"  {f:14s} " + " ".join(f"{y}:{wf_thr[f][y]['q2']:.0f}" for y in YEARS if y in wf_thr.get(f, {})) + f"  | fixed >= {FIXED[f][1]}")

oos = (tr.year >= 2010).values
T = tr[oos].reset_index(drop=True)
T["ep"] = 0
for f in FAMS:
    m = (T.family == f).values
    T.loc[m, "ep"] = episodes(T.loc[m, "Signal Date"])


def rpr(d):
    return float(d["PnL"].sum() / d["Risk"].sum()) if d["Risk"].sum() > 0 else np.nan


print("\n=== (a) top vs bottom tercile PnL per unit risk, 2010-2026, walk-forward vs fixed thresholds ===")
OUT["tercile"] = {}
for f in FAMS:
    F = T[T.family == f]
    row = {}
    for lab, hi, lo in [("wf", F.hi_wf.values, F.lo_wf.values), ("fixed", F.hi_fx.values, F.lo_fx.values)]:
        H, L = F[hi], F[lo]
        cb = cluster_boot_diff(H.R.values, H.ep.values, L.R.values, L.ep.values)
        row[lab] = dict(N_hi=int(len(H)), N_lo=int(len(L)), share_hi=float(hi.mean()), rpr_hi=rpr(H), rpr_lo=rpr(L), ratio=rpr(H) / rpr(L) if rpr(L) > 0 else np.nan,
                        avgR_hi=float(H.R.mean()), avgR_lo=float(L.R.mean()), t_episode=cb["t"], p=cb["p"], ep_hi=int(H.ep.nunique()), ep_lo=int(L.ep.nunique()))
    OUT["tercile"][f] = row
    for lab in ["wf", "fixed"]:
        r = row[lab]
        print(f"  {f:14s} {lab:5s}: hi N={r['N_hi']:4d} ({r['share_hi']:.0%}) rpr {r['rpr_hi']:.2f} avgR {r['avgR_hi']:.2f} | lo N={r['N_lo']:4d} rpr {r['rpr_lo']:.2f} avgR {r['avgR_lo']:.2f} | ratio {r['ratio']:.2f} | episode t {r['t_episode']:.2f} (ep hi {r['ep_hi']}, lo {r['ep_lo']})")
# also the plan's own basis: 2005-2026 with the fixed thresholds
OUT["tercile_2005_fixed"] = {}
for f in FAMS:
    F = tr[(tr.family == f) & (tr.year >= 2005)]
    H, L = F[F.hi_fx], F[F.lo_fx]
    OUT["tercile_2005_fixed"][f] = dict(N_hi=int(len(H)), N_lo=int(len(L)), rpr_hi=rpr(H), rpr_lo=rpr(L), ratio=rpr(H) / rpr(L))
    print(f"  (2005+ fixed) {f:14s} ratio {rpr(H)/rpr(L):.2f}  rpr hi {rpr(H):.2f} lo {rpr(L):.2f}  N {len(H)}/{len(L)}")

# ---------------------------------------------------------------- (b) 1.2x up-size replay, cap re-applied
days, MTM = build_trade_mtm(T)
print("MTM reconciliation residual:", float(np.abs(MTM.sum(axis=1) - T.PnL.values).max()))


def apply(df, mult):
    d = df[["Strategy", "Signal Date", "nominal", "placed_est", "cap_scale"]].copy()
    d["m"] = mult; d["mw"] = d["m"] * d["nominal"]
    gg = d.groupby(["Strategy", "Signal Date"])
    mbar = gg["mw"].transform("sum") / gg["nominal"].transform("sum")
    new_scale = np.minimum(1.0, CAP_D / (d["placed_est"].values * mbar.values))
    return d["nominal"].values * d["m"].values * new_scale / (d["nominal"].values * d["cap_scale"].values)


def score(df, fac, rows):
    daily = pd.Series((MTM[rows] * fac[:, None]).sum(axis=0), index=days)
    daily = daily[daily.index >= "2010-01-01"]
    st = dd_stats(daily)
    pnl = df["PnL"].values * fac; risk = df["Risk"].values * fac
    st.update(pnl=float(pnl.sum()), risk=float(risk.sum()), pnl_per_risk=float(pnl.sum() / risk.sum()))
    yrs = pd.DataFrame({"y": df["year"].values, "b": df["PnL"].values, "r": pnl}).groupby("y").sum()
    st.update(years_better=int((yrs.r > yrs.b).sum()), years=len(yrs), years_table={int(k): [float(v.b), float(v.r)] for k, v in yrs.iterrows()})
    return st


def hi_mask(df, kind):
    hi = df["hi_wf"].values if kind == "wf" else df["hi_fx"].values
    gate = np.where(df["family"].values == "dip_buy", ~(df["dial"].values >= 50), True)   # dip_buy: no up-size at dial >= 50
    return hi & gate


print(f"\n=== (b) up-only {UP}x replay, 2010-2026, cap re-applied; dip_buy gated dial<50 ===")
OUT["upsize"] = {}
fac_all = {"wf": np.ones(len(T)), "fixed": np.ones(len(T))}
for f in FAMS:
    F = T[T.family == f]; ridx = F.index.values
    base = score(F, np.ones(len(F)), ridx)
    OUT["upsize"][f] = {"baseline": {k: v for k, v in base.items() if k != "years_table"}}
    for kind in ["wf", "fixed"]:
        m = np.where(hi_mask(F, kind), UP, 1.0)
        fac = apply(F, m); fac_all[kind][ridx] = fac
        s = score(F, fac, ridx)
        s_eq = score(F, fac * (F.Risk.sum() / (F.Risk.values * fac).sum()), ridx)
        OUT["upsize"][f][kind] = {k: v for k, v in s.items() if k != "years_table"} | dict(eq_pnl=s_eq["pnl"], eq_sharpe=s_eq["sharpe"], eq_maxdd=s_eq["maxdd"], years_table=s["years_table"],
                                                                                      share_hi_after_gate=float(hi_mask(F, kind).mean()))
        print(f"  {f:14s} {kind:5s}: dPnL {s['pnl']/base['pnl']-1:+.2%} risk x{s['risk']/base['risk']:.3f} pnl/risk {base['pnl_per_risk']:.3f}->{s['pnl_per_risk']:.3f} "
              f"sharpe {base['sharpe']:.2f}->{s['sharpe']:.2f} maxDD {base['maxdd']:,.0f}->{s['maxdd']:,.0f} worst21 {base['worst21']:,.0f}->{s['worst21']:,.0f} yrs better {s['years_better']}/{s['years']} | eq-risk dPnL {s_eq['pnl']/base['pnl']-1:+.2%}")
B0 = score(T, np.ones(len(T)), T.index.values)
OUT["upsize"]["book"] = {"baseline": {k: v for k, v in B0.items() if k != "years_table"}}
for kind in ["wf", "fixed"]:
    s = score(T, fac_all[kind], T.index.values)
    OUT["upsize"]["book"][kind] = {k: v for k, v in s.items() if k != "years_table"} | dict(years_table=s["years_table"])
    print(f"  {'BOOK':14s} {kind:5s}: dPnL {s['pnl']/B0['pnl']-1:+.2%} risk x{s['risk']/B0['risk']:.3f} sharpe {B0['sharpe']:.3f}->{s['sharpe']:.3f} maxDD {B0['maxdd']:,.0f}->{s['maxdd']:,.0f} worst21 {B0['worst21']:,.0f}->{s['worst21']:,.0f} yrs better {s['years_better']}/{s['years']}")

# ---------------------------------------------------------------- (c) per-year table
print("\n=== (c) per family per year: WF q2 line, hi-share (wf/fixed), hi-vs-lo rpr ratio (wf/fixed), up-size dPnL (wf/fixed) ===")
OUT["per_year"] = {}
for f in FAMS:
    F = T[T.family == f]
    rows = []
    for y in YEARS:
        Y = F[F.year == y]
        if len(Y) == 0:
            continue
        rec = dict(year=y, N=int(len(Y)), q2_wf=float(Y.q2_wf.iloc[0]) if Y.q2_wf.notna().any() else np.nan, fixed_hi=FIXED[f][1],
                   share_hi_wf=float(Y.hi_wf.mean()), share_hi_fx=float(Y.hi_fx.mean()), n_hi_wf=int(Y.hi_wf.sum()), n_hi_fx=int(Y.hi_fx.sum()))
        for lab, hi, lo in [("wf", Y.hi_wf, Y.lo_wf), ("fx", Y.hi_fx, Y.lo_fx)]:
            rec[f"rpr_hi_{lab}"] = rpr(Y[hi]) if hi.any() else np.nan
            rec[f"rpr_lo_{lab}"] = rpr(Y[lo]) if lo.any() else np.nan
            rec[f"hi_gt_lo_{lab}"] = bool(rec[f"rpr_hi_{lab}"] > rec[f"rpr_lo_{lab}"]) if np.isfinite(rec[f"rpr_hi_{lab}"]) and np.isfinite(rec[f"rpr_lo_{lab}"]) else None
        yt_wf = OUT["upsize"][f]["wf"]["years_table"].get(y); yt_fx = OUT["upsize"][f]["fixed"]["years_table"].get(y)
        rec["dpnl_wf"] = yt_wf[1] - yt_wf[0] if yt_wf else np.nan
        rec["dpnl_fx"] = yt_fx[1] - yt_fx[0] if yt_fx else np.nan
        rec["hi_pnl_wf"] = float(Y[Y.hi_wf].PnL.sum()); rec["hi_pnl_fx"] = float(Y[Y.hi_fx].PnL.sum())
        rec["flip_class"] = bool(rec["n_hi_wf"] != rec["n_hi_fx"])
        rec["flip_verdict"] = bool(rec["hi_gt_lo_wf"] != rec["hi_gt_lo_fx"]) if (rec["hi_gt_lo_wf"] is not None and rec["hi_gt_lo_fx"] is not None) else None
        rows.append(rec)
    D = pd.DataFrame(rows)
    OUT["per_year"][f] = D.to_dict("records")
    print(f"\n--- {f} (fixed >= {FIXED[f][1]}) ---")
    print(D[["year", "N", "q2_wf", "n_hi_wf", "n_hi_fx", "rpr_hi_wf", "rpr_lo_wf", "rpr_hi_fx", "rpr_lo_fx", "hi_gt_lo_wf", "hi_gt_lo_fx", "dpnl_wf", "dpnl_fx", "flip_verdict"]].round(2).to_string(index=False))
    print(f"  years hi>lo: wf {int(D.hi_gt_lo_wf.fillna(False).sum())}/{int(D.hi_gt_lo_wf.notna().sum())}, fixed {int(D.hi_gt_lo_fx.fillna(False).sum())}/{int(D.hi_gt_lo_fx.notna().sum())}; "
          f"verdict flips {int(D.flip_verdict.fillna(False).sum())}; up-size years better: wf {int((D.dpnl_wf > 0).sum())}, fixed {int((D.dpnl_fx > 0).sum())}")

# ---------------------------------------------------------------- (d) hi-flow episode counts
print("\n=== (d) hi-flow episodes (runs of hi-flow trade signal dates with gaps <= 5 td), 2010-2026 and 2005-2026 ===")
OUT["episodes"] = {}
for f in FAMS:
    rec = {}
    for lab, D in [("2010+", T[T.family == f]), ("2005+", tr[(tr.family == f) & (tr.year >= 2005)])]:
        for kind, col in [("wf", "hi_wf"), ("fixed", "hi_fx")]:
            H = D[D[col]]
            if kind == "wf" and lab == "2005+":
                H = D[D[col] & (D.year >= 2010)]
            n_ep = int(len(set(episodes(H["Signal Date"])))) if len(H) else 0
            n_days = int(H["Signal Date"].nunique())
            rec[f"{lab}|{kind}"] = dict(trades=int(len(H)), signal_days=n_days, episodes=n_ep)
    # calendar-day episodes from the family daily count series (fixed threshold), 2005+
    cs = cf_lib[f][(cf_lib.index >= "2005-01-01") & (cf_lib.index <= "2026-09-01")]
    hd = cs[cs >= FIXED[f][1]].index
    rec["calendar_days_2005+_fixed"] = dict(days=int(len(hd)), episodes=int(len(set(episodes(pd.Series(hd))))) if len(hd) else 0)
    OUT["episodes"][f] = rec
    print(f"  {f:14s} " + " | ".join(f"{k}: {v['trades']}tr/{v['signal_days']}d/{v['episodes']}ep" if 'trades' in v else f"{k}: {v['days']}d/{v['episodes']}ep" for k, v in rec.items()))

# ---------------------------------------------------------------- (e) 2026 OVS cluster days and the short_fade verdict
print("\n=== (e) 2026 short_fade hi-flow days, OVS same-day counts, and the verdict with / without 2026 ===")
sf26 = tr[(tr.family == "short_fade") & (tr.year == 2026)]
ovs_day = cand[(cand.strategy == "Overbot Vol Spike")].groupby("signal_date").size()
day26 = sf26.groupby("Signal Date").agg(N=("R", "size"), pnl=("PnL", "sum"), risk=("Risk", "sum"), f5=("f5", "first"), hi_fx=("hi_fx", "any"), hi_wf=("hi_wf", "any"))
day26["ovs_cands"] = ovs_day.reindex(day26.index).fillna(0).astype(int).values
day26["rpr"] = day26.pnl / day26.risk
cl = day26[day26.ovs_cands >= 5]
print("2026 short_fade fill days with >= 5 OVS candidates the same day:")
print(cl.round(2).to_string())
OUT["ovs_2026_cluster_days"] = cl.reset_index().assign(**{"Signal Date": cl.index.astype(str)}).to_dict("records")
hi26 = day26[day26.hi_fx]
print(f"2026 hi-flow (fixed) short_fade fill days: {len(hi26)}, PnL {hi26.pnl.sum():,.0f}, rpr {hi26.pnl.sum()/hi26.risk.sum() if hi26.risk.sum() else float('nan'):.2f}; "
      f"all 2026 short_fade: N {len(sf26)} PnL {sf26.PnL.sum():,.0f} rpr {rpr(sf26):.2f}")
OUT["short_fade_2026"] = dict(hi_days=int(len(hi26)), hi_pnl=float(hi26.pnl.sum()), hi_risk=float(hi26.risk.sum()), all_N=int(len(sf26)), all_pnl=float(sf26.PnL.sum()), all_rpr=rpr(sf26))
verd = {}
for lab, D in [("2005-2025 fixed", tr[(tr.family == "short_fade") & (tr.year >= 2005) & (tr.year <= 2025)]),
               ("2005-2026 fixed", tr[(tr.family == "short_fade") & (tr.year >= 2005)]),
               ("2010-2025 wf", T[(T.family == "short_fade") & (T.year <= 2025)]), ("2010-2026 wf", T[T.family == "short_fade"])]:
    col_hi, col_lo = ("hi_fx", "lo_fx") if "fixed" in lab else ("hi_wf", "lo_wf")
    H, L = D[D[col_hi]], D[D[col_lo]]
    epH = episodes(H["Signal Date"]); epL = episodes(L["Signal Date"])
    cb = cluster_boot_diff(H.R.values, epH, L.R.values, epL)
    verd[lab] = dict(N_hi=int(len(H)), rpr_hi=rpr(H), rpr_lo=rpr(L), ratio=rpr(H) / rpr(L), avgR_hi=float(H.R.mean()), avgR_lo=float(L.R.mean()), t=cb["t"], p=cb["p"])
    print(f"  {lab:16s}: hi N {len(H):4d} rpr {rpr(H):.2f} lo rpr {rpr(L):.2f} ratio {rpr(H)/rpr(L):.2f} avgR hi/lo {H.R.mean():.2f}/{L.R.mean():.2f} episode t {cb['t']:.2f}")
OUT["short_fade_verdict"] = verd
# the 2026 hi-flow cell alone vs the rest of 2026
h26 = sf26[sf26.hi_fx]; r26 = sf26[~sf26.hi_fx]
print(f"  2026 hi-flow trades N {len(h26)} avgR {h26.R.mean():.2f} rpr {rpr(h26):.2f} | 2026 non-hi N {len(r26)} avgR {r26.R.mean():.2f} rpr {rpr(r26):.2f}")
OUT["short_fade_2026"].update(hi_N=int(len(h26)), hi_avgR=float(h26.R.mean()) if len(h26) else None, hi_rpr=rpr(h26) if len(h26) else None, rest_N=int(len(r26)), rest_avgR=float(r26.R.mean()), rest_rpr=rpr(r26))

json.dump(OUT, open(HERE / "gap7_flow_walkforward.json", "w"), indent=1, default=float)
print("wrote gap7_flow_walkforward.json")
