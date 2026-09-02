"""Cycle/macro regime conditioning, part 3: how much of each regime effect is
ALREADY captured by the live controls (frag_risk_bands, pc_fear_bands,
dial_filters, cycle_risk_mults), plus a PIT-weights dial rebuild for the
dial cells (2018+).
(A) day-level cross-tab of every regime bucket vs the dial (2016-07+);
(B) regime effects re-estimated on trades OUTSIDE the existing controls
    (dial < 50 or pre-dial era; Size_Mult >= 0.99);
(C) family carriers: regime cells split by dial >= 50 (the zeroed/quartered
    part) vs < 50;
(D) OVS cycle tilt audit + what the same tilt does on the other strategies;
(E) PIT dial (scratch/pit_reestimate.py method) vs current weights on the
    65+ / 50-65 cells for OLV, LT Trend ST OS, OVS, and the vol regimes'
    overlap with the PIT dial.
Writes cycle_macro_03_overlap.json beside this file."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_macro_lib import (FAMILY_BAND, HERE, ROOT, REGIME_COLS, attach_trade_regimes, build_regimes, cluster_t, episode_ids,
                             jsonable, load_ledger, loyo, welch_t)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
led = load_ledger()
R = build_regimes()
T = attach_trade_regimes(led, R)
KEY_AXES = ["cycle", "vix_lvl", "rv21", "spy_dd", "vix_ts", "credit21", "tnx_chg63", "pc_fear", "vol_ratio", "mom12_1"]

# ---------------------------------------------------------------- A. regime x dial cross-tab (days)
print("=== A. P(dial >= 50 | regime bucket) and P(bucket | dial >= 50), days 2016-07+ (current-weights dial) ===")
W = R[(R.index >= "2016-07-20") & R["dial_val"].notna()]
hi = W["dial_val"] >= 50
ct_rows = []
for col in KEY_AXES:
    for b, g in W.groupby(col):
        if b == "nan":
            continue
        ct_rows.append(dict(regime=col, bucket=b, days=len(g), p_dial_ge50=float((g["dial_val"] >= 50).mean()), p_dial_ge65=float((g["dial_val"] >= 65).mean()),
                            share_of_hi_dial_days=float(((W[col] == b) & hi).sum() / hi.sum()), share_of_days=float(len(g) / len(W))))
CT = pd.DataFrame(ct_rows); print(CT.round(3).to_string(index=False))
OUT["regime_vs_dial_days"] = jsonable(CT.round(4).to_dict("records"))
# the P/C fear state vs VIX (the book's fear switch vs the vol axis)
pcv = pd.crosstab(W["pc_fear"], W["vix_lvl"], normalize="columns").round(3)
print("\nP(pc_fear state | VIX level), 2016-07+:\n", pcv.to_string())
OUT["pc_fear_vs_vix"] = jsonable(pcv.to_dict())

# ---------------------------------------------------------------- B. residual effects outside the existing controls
print("\n=== B. regime effects on trades OUTSIDE live controls: (i) dial<50 or pre-2016; (ii) Size_Mult>=0.99 ===")
def cell_stats(d, col, b):
    m = d[col] == b
    g, rest = d[m], d[~m]
    if len(g) < 10 or len(rest) < 10:
        return None
    ep = episode_ids(R[col] == b, gap=21).reindex(d["Signal Date"]).values
    cl = np.where(m.values, "E" + pd.Series(ep).astype(str), "M" + d["ym"].values)
    beta, t, G = cluster_t(d["R_Multiple"].values, m.values.astype(float), cl)
    lo = loyo(d, m)
    return dict(N=int(len(g)), avgR=float(g["R_Multiple"].mean()), avgR_rest=float(rest["R_Multiple"].mean()), diff=beta, t_cluster=t, n_clusters=G,
                sd_ratio=float(g["R_Multiple"].std() / d["R_Multiple"].std()), yr_pos=lo["yr_pos"], yr_neg=lo["yr_neg"])
sub_defs = {"all": T, "dial_lt50_or_predial": T[(T["dial_val"] < 50) | T["dial_val"].isna()], "sizemult_ge_0.99": T[T["Size_Mult"] >= 0.99],
            "dial_lt50_2016plus": T[(T["dial_val"] < 50)]}
res_rows = []
for col in KEY_AXES:
    for b in sorted(T[col].unique()):
        if b == "nan":
            continue
        for s in ["ALL"] + sorted(T["Strat2"].unique()):
            row = dict(regime=col, bucket=b, strategy=s)
            ok = False
            for lab, sub in sub_defs.items():
                d = sub if s == "ALL" else sub[sub["Strat2"] == s]
                d = d[d[col] != "nan"]
                c = cell_stats(d, col, b)
                if c:
                    ok = True
                    row.update({f"{lab}_{k}": v for k, v in c.items()})
            if ok:
                res_rows.append(row)
RS = pd.DataFrame(res_rows)
OUT["residual_effects"] = jsonable(RS.to_dict("records"))
show = RS[(RS.strategy == "ALL")][["regime", "bucket", "all_N", "all_diff", "all_t_cluster", "dial_lt50_or_predial_N", "dial_lt50_or_predial_diff", "dial_lt50_or_predial_t_cluster",
                                   "sizemult_ge_0.99_N", "sizemult_ge_0.99_diff", "sizemult_ge_0.99_t_cluster", "dial_lt50_2016plus_N", "dial_lt50_2016plus_diff", "dial_lt50_2016plus_t_cluster"]]
print(show.round(3).to_string(index=False))
print("\n-- per-strategy cells with |t| >= 2 on the full sample: do they survive outside the controls? --")
ps = RS[(RS.strategy != "ALL") & (RS["all_t_cluster"].abs() >= 2) & (RS["all_N"] >= 15)].sort_values("all_t_cluster")
print(ps[["regime", "bucket", "strategy", "all_N", "all_diff", "all_t_cluster", "dial_lt50_or_predial_N", "dial_lt50_or_predial_diff", "dial_lt50_or_predial_t_cluster",
          "sizemult_ge_0.99_N", "sizemult_ge_0.99_diff", "sizemult_ge_0.99_t_cluster"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- C. family carriers: regime cells split by dial >= 50
print("\n=== C. family band carriers (2016-07+): regime cell avgR split by dial >= 50 (the band-controlled part) ===")
fam = T[T["Strat2"].isin(FAMILY_BAND) & T["dial_val"].notna()]
fam_rows = []
for col in ["vix_lvl", "rv21", "spy_dd", "vix_ts", "credit21", "pc_fear", "cycle"]:
    for b, g in fam.groupby(col):
        if b == "nan" or len(g) < 10:
            continue
        h, l = g[g["dial_val"] >= 50], g[g["dial_val"] < 50]
        fam_rows.append(dict(regime=col, bucket=b, N=len(g), share_dial_ge50=float(len(h) / len(g)), avgR_all=float(g["R_Multiple"].mean()),
                             avgR_dial_ge50=float(h["R_Multiple"].mean()) if len(h) >= 3 else np.nan, N_ge50=len(h),
                             avgR_dial_lt50=float(l["R_Multiple"].mean()) if len(l) >= 3 else np.nan, N_lt50=len(l),
                             mean_size_mult=float(g["Size_Mult"].mean()), pnl=float(g["PnL_flat_750k"].sum())))
FM = pd.DataFrame(fam_rows); print(FM.round(3).to_string(index=False))
OUT["family_by_dial"] = jsonable(FM.round(4).to_dict("records"))
# what the family's ledger PnL by regime would be with the band removed (R x nominal risk) vs with it
fam2 = fam.copy(); fam2["risk_nominal"] = fam2["Risk_flat_750k"] / fam2["Size_Mult"].replace(0, np.nan)
fam2["pnl_unbanded"] = fam2["R_Multiple"] * fam2["risk_nominal"]
cmp = fam2.groupby("vix_lvl").agg(pnl_live=("PnL_flat_750k", "sum"), pnl_unbanded=("pnl_unbanded", "sum"), N=("R_Multiple", "size"))
print("\nfamily PnL by VIX level, live sizing vs Size_Mult stripped (all overlays):\n", cmp.round(0).to_string())
OUT["family_vix_live_vs_unbanded"] = jsonable(cmp.round(0).reset_index().to_dict("records"))

# ---------------------------------------------------------------- D. OVS cycle tilt audit
print("\n=== D. OVS cycle tilt (0.75x midterm): per-path, year-clustered; and the same tilt on the other strategies ===")
ovs = T[T["Strategy"] == "Overbot Vol Spike"]
def yc(d):
    y = d.groupby("yr").agg(avgR=("R_Multiple", "mean"), N=("R_Multiple", "size"), cyc=("cycle", "first"))
    y = y[y.N >= 8]; mid, oth = y[y.cyc == "midterm"], y[y.cyc != "midterm"]
    return dict(mid_years=int(len(mid)), oth_years=int(len(oth)), mid_avgR=float(mid.avgR.mean()), oth_avgR=float(oth.avgR.mean()), t_year=welch_t(mid.avgR, oth.avgR),
                mid_by_year={int(k): round(float(v), 2) for k, v in mid.avgR.items()})
d_rows = []
for lab, d in [("OVS all", ovs), ("OVS path 1", ovs[ovs["Risk bps"] >= 30]), ("OVS path 2", ovs[ovs["Risk bps"] < 30])]:
    r = yc(d); d_rows.append(dict(label=lab, **r)); print(lab, r)
# the cycle mult as seen in the ledger Size_Mult
print("OVS Size_Mult mean by cycle:", ovs.groupby("cycle")["Size_Mult"].mean().round(3).to_dict())
OUT["ovs_cycle"] = jsonable(d_rows)
# midterm avgR by strategy, year-clustered, restricted to 2006+ (when OLV etc exist) -- from part 2 too; here the trade-weighted PnL impact of a 0.75x midterm tilt
imp = []
for s, d in T.groupby("Strat2"):
    m = d["cycle"] == "midterm"
    if m.sum() < 10:
        continue
    pnl_mid = d.loc[m, "PnL_flat_750k"].sum(); risk_mid = d.loc[m, "Risk_flat_750k"].sum()
    imp.append(dict(strategy=s, N_mid=int(m.sum()), pnl_mid=float(pnl_mid), ppr_mid=float(pnl_mid / risk_mid), ppr_other=float(d.loc[~m, "PnL_flat_750k"].sum() / d.loc[~m, "Risk_flat_750k"].sum()),
                    cost_of_0p75=float(-0.25 * pnl_mid), already_tilted=bool(s.startswith("OVS"))))
IM = pd.DataFrame(imp); print(IM.round(3).to_string(index=False))
OUT["midterm_tilt_impact"] = jsonable(IM.round(4).to_dict("records"))

# ---------------------------------------------------------------- E. PIT dial rebuild
print("\n=== E. PIT-weights dial (pit_reestimate method) vs current weights, 2018+ trade cells ===")
try:
    import os
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    class _NoOp:
        def __getattr__(self, name): return self
        def __call__(self, *a, **k): return self
        def __bool__(self): return False
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cache_data(self, *a, **k):
            def deco(fn): return fn
            return deco
        cache_resource = cache_data
    sys.modules["streamlit"] = _NoOp()
    from pages.risk_dashboard_v2 import compute_fragility_timeseries
    raw = pickle.load(open(ROOT / "scratch/pit_signals.pkl", "rb"))
    fires, spy_c = raw["fires"], raw["spy_close"]
    fires.index = pd.to_datetime(fires.index); spy_c.index = pd.to_datetime(spy_c.index)
    HZ = {"5d": 5, "21d": 21, "63d": 63}
    def estimate_stats(end):
        end = pd.Timestamp(end); out = {}
        for name in fires.columns:
            hor = {}
            for hl, h in HZ.items():
                fwd = (spy_c.shift(-h) / spy_c - 1.0) * 100.0
                valid = fwd.index[fwd.index <= end - pd.Timedelta(days=int(h * 1.6))]
                fwd = fwd.reindex(valid).dropna(); f = fires[name].reindex(fwd.index).fillna(False)
                hor[hl] = {"diff_mean": float(fwd[f].mean() - fwd.mean()) if f.sum() >= 10 else None}
            out[name] = {"horizons": hor}
        return {"signals": out}
    sig_dict = {n: {"signal_history": fires[n]} for n in fires.columns}
    frames = []
    for year in range(2018, 2027):
        vint = estimate_stats(f"{year - 1}-12-31")
        fr = compute_fragility_timeseries(sig_dict, spy_c, vint)
        frames.append(fr[fr.index.year == year])
    pit_raw = pd.concat(frames).sort_index()
    def live_basis(frame):
        s = frame["63d"].rolling(5, min_periods=1).mean().dropna().rolling(10, min_periods=1).mean()
        s.index = pd.to_datetime(s.index).normalize()
        return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill(limit=5)
    pit = live_basis(pit_raw)
    pit.to_frame("pit_dial").to_parquet(HERE / "cycle_macro_pit_dial.parquet")
    T["pit_dial"] = pit.reindex(T["Signal Date"]).values
    cur = T["dial_val"]
    both = T.dropna(subset=["pit_dial", "dial_val"])
    print(f"PIT vs current at trade signal dates (2018+): corr {both.pit_dial.corr(both.dial_val):.3f}; >=50 agreement {((both.pit_dial>=50)==(both.dial_val>=50)).mean()*100:.1f}%; >=65 agreement {((both.pit_dial>=65)==(both.dial_val>=65)).mean()*100:.1f}%")
    OUT["pit_meta"] = dict(span=[pit.index.min(), pit.index.max()], corr_at_signals=float(both.pit_dial.corr(both.dial_val)),
                           agree50=float(((both.pit_dial >= 50) == (both.dial_val >= 50)).mean()), agree65=float(((both.pit_dial >= 65) == (both.dial_val >= 65)).mean()))
    prow = []
    d18 = T[(T["yr"] >= 2018) & T["pit_dial"].notna()]
    for s in ["ALL", "Oversold Low Volume", "LT Trend ST OS", "OVS path 1", "OVS path 2", "52wh Breakout", "SPY QQQ MonFri Reversion", "Weak Close Decent Sznls", "Indices Oversold Bounce", "Monday Dip"]:
        d = d18 if s == "ALL" else d18[d18["Strat2"] == s]
        for series, lab in [("pit_dial", "PIT"), ("dial_val", "current")]:
            for lo_, hi_ in [(50, 65), (65, 999), (50, 999)]:
                m = (d[series] >= lo_) & (d[series] < hi_)
                ctrl = d[series] < 50
                if m.sum() < 8 or ctrl.sum() < 8:
                    continue
                # cluster hi trades by episode of >= lo_ (from the daily PIT/current series), controls by month
                if series == "pit_dial":
                    mask_daily = (pit >= lo_) & (pit < hi_)
                else:
                    mask_daily = (R["dial_val"] >= lo_) & (R["dial_val"] < hi_)
                ep = episode_ids(mask_daily, gap=21).reindex(d["Signal Date"]).values
                dd = d[m | ctrl]
                x = m[m | ctrl].values.astype(float)
                cl = np.where(x == 1, "E" + pd.Series(ep[(m | ctrl).values]).astype(str), "M" + dd["ym"].values)
                beta, t, G = cluster_t(dd["R_Multiple"].values, x, cl)
                # drop-best-episode
                epi = pd.Series(ep[(m | ctrl).values], index=dd.index)[x == 1]
                epm = dd.loc[epi.index].groupby(epi)["R_Multiple"].mean()
                worst_drop = np.nan
                if len(epm) >= 2:
                    best = epm.idxmin() if beta < 0 else epm.idxmax()   # the episode that most supports the sign
                    keep = dd.index.difference(epi[epi == best].index)
                    dk = dd.loc[keep]; xk = x[dd.index.get_indexer(keep)]
                    if xk.sum() >= 5:
                        worst_drop = float(dk.loc[xk == 1, "R_Multiple"].mean() - dk.loc[xk == 0, "R_Multiple"].mean())
                prow.append(dict(strategy=s, series=lab, cell=f"[{lo_},{hi_})", N=int(m.sum()), N_ctrl=int(ctrl.sum()), avgR=float(d.loc[m, "R_Multiple"].mean()),
                                 avgR_lt50=float(d.loc[ctrl, "R_Multiple"].mean()), diff=beta, t_cluster=t, n_episodes=int(len(epm)), diff_drop_best_episode=worst_drop))
    PR = pd.DataFrame(prow); print(PR.round(3).to_string(index=False))
    OUT["pit_cells"] = jsonable(PR.round(4).to_dict("records"))
    # vol regimes vs PIT dial (days)
    Rp = R.join(pit.rename("pit_dial")).dropna(subset=["pit_dial"])
    vr = []
    for col in ["vix_lvl", "rv21", "spy_dd", "vix_ts"]:
        for b, g in Rp.groupby(col):
            if b == "nan":
                continue
            vr.append(dict(regime=col, bucket=b, days=len(g), p_pit_ge50=float((g.pit_dial >= 50).mean()), p_pit_ge65=float((g.pit_dial >= 65).mean())))
    VR = pd.DataFrame(vr); print(VR.round(3).to_string(index=False)); OUT["regime_vs_pit_dial_days"] = jsonable(VR.round(4).to_dict("records"))
except Exception as e:  # noqa
    import traceback
    traceback.print_exc()
    OUT["pit_error"] = repr(e)

json.dump(jsonable(OUT), open(HERE / "cycle_macro_03_overlap.json", "w"), indent=1)
print("\nwrote", HERE / "cycle_macro_03_overlap.json")
