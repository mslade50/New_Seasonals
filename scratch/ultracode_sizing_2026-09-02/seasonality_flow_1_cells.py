"""seasonality_flow_1_cells.py (2026-09-02): trade-level seasonality of signal FLOW,
QUALITY and VARIANCE per strategy and for the book.

Cells: calendar month, quarter, Nov-Apr vs May-Oct, week-of-month, turn-of-month,
opex week, day-of-week, earnings season (fixed base-rate + data-driven), holiday
adjacency. Per cell: N, avgR, sdR, R per unit risk, sum PnL, flow ratio
(signal share / session share). Tests: episode-clustered t (5-session gap
episodes within strategy) AND year-paired t (one obs per year), then BH-FDR
and Bonferroni within each (dimension) family across strategies.
OVS scale-out tranches are collapsed to one row per fill (R = sum pnl / sum risk).
Writes seasonality_flow_cells.json.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from seasonality_flow_common import (HERE, ROOT, MONTHS, DOW, load_ledger, load_spy, trading_calendar,
                                     episodes, cluster_diff_t, year_paired_t, bh_fdr, summarize, jdump)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
led = load_ledger()
# collapse scale-out tranches (OVS near/far) to one row per fill
key = ["Strategy", "Tier", "Ticker", "sig", "Entry Date"]
g = led.groupby(key, as_index=False).agg(pnl=("pnl", "sum"), risk=("risk", "sum"), yr=("yr", "first"),
                                          exit=("Exit Date", "first"), n_tr=("R", "size"))
g["R"] = g["pnl"] / g["risk"]
led = g
print(f"ledger rows after tranche collapse: {len(led)}")

earn = pd.read_parquet(ROOT / "data/earnings_calendar.parquet", columns=["date"])
spy = load_spy()
cal = trading_calendar(spy.index, earn)
cal = cal[cal.index >= "2003-01-01"]
led = led[led["sig"].isin(cal.index)].copy()
F = cal.loc[led["sig"]].reset_index(drop=True)
for c in ["month", "mname", "quarter", "half", "wom", "tom", "opex_week", "post_opex_week", "dow", "dname",
          "eseason_fixed", "eseason_data", "holiday_adj"]:
    led[c] = F[c].values
led["dname"] = led["dname"].astype(str)
led["tom"] = np.where(led["tom"], "TOM", "mid")
led["opex"] = np.where(led["opex_week"], "opex", np.where(led["post_opex_week"], "post_opex", "other"))
led["eseason_fixed"] = np.where(led["eseason_fixed"], "season", "off")
led["eseason_data"] = np.where(led["eseason_data"], "season", "off")
led["quarter"] = "Q" + led["quarter"].astype(str)

DIMS = {"month": "mname", "quarter": "quarter", "half": "half", "wom": "wom", "tom": "tom", "opex": "opex",
        "dow": "dname", "eseason_fixed": "eseason_fixed", "eseason_data": "eseason_data", "holiday": "holiday_adj"}
CAL_COL = {"month": ("mname", None), "quarter": ("quarter", lambda s: "Q" + s.astype(str)), "half": ("half", None),
           "wom": ("wom", None), "tom": ("tom", lambda s: np.where(s, "TOM", "mid")),
           "opex": (None, None), "dow": ("dname", None),
           "eseason_fixed": ("eseason_fixed", lambda s: np.where(s, "season", "off")),
           "eseason_data": ("eseason_data", lambda s: np.where(s, "season", "off")), "holiday": ("holiday_adj", None)}
cal["opex"] = np.where(cal["opex_week"], "opex", np.where(cal["post_opex_week"], "post_opex", "other"))
cal["quarter_s"] = "Q" + cal["quarter"].astype(str)
cal["tom_s"] = np.where(cal["tom"], "TOM", "mid")
cal["ef_s"] = np.where(cal["eseason_fixed"], "season", "off")
cal["ed_s"] = np.where(cal["eseason_data"], "season", "off")
CALSER = {"month": "mname", "quarter": "quarter_s", "half": "half", "wom": "wom", "tom": "tom_s", "opex": "opex",
          "dow": "dname", "eseason_fixed": "ef_s", "eseason_data": "ed_s", "holiday": "holiday_adj"}

strategies = sorted(led["Strategy"].unique())
groups = {s: led[led["Strategy"] == s] for s in strategies}
groups["BOOK"] = led
rows = []
for sname, df in groups.items():
    df = df.copy()
    base = summarize(df)
    # episodes: within strategy (BOOK: within each strategy, then offset so ids are unique)
    if sname == "BOOK":
        eid = np.zeros(len(df), dtype=np.int64)
        off = 0
        for s, idx in df.groupby("Strategy").indices.items():
            e = episodes(df.iloc[idx]["sig"], 5, cal.index)
            eid[idx] = e + off
            off += e.max() + 1
    else:
        eid = episodes(df["sig"], 5, cal.index)
    df["ep"] = eid
    span = cal[(cal.index >= df["sig"].min()) & (cal.index <= df["sig"].max())]
    for dim, col in DIMS.items():
        cells = sorted(df[col].unique(), key=lambda v: (MONTHS.index(v) if v in MONTHS else (DOW.index(v) if v in DOW else str(v))))
        for cell in cells:
            m = (df[col] == cell).values
            sub = df[m]
            st = summarize(sub)
            t_ep, p_ep, g_ep = cluster_diff_t(df["R"].values, m, df["ep"].values)
            t_yr, p_yr, n_yr = year_paired_t(df, "R", m, df["yr"].values)
            sess_share = float((span[CALSER[dim]] == cell).mean()) if len(span) else np.nan
            sig_share = float(m.mean())
            n_yrs_with = int(sub["yr"].nunique())
            rows.append(dict(strategy=sname, dim=dim, cell=str(cell), N=st["N"], n_years=n_yrs_with,
                             avgR=st["avgR"], sdR=st["sdR"], win=st["win"], R_per_risk=st["R_per_risk"],
                             sum_pnl=st["sum_pnl"], avgR_rest=float(df.loc[~m, "R"].mean()) if (~m).sum() else np.nan,
                             sdR_rest=float(df.loc[~m, "R"].std(ddof=1)) if (~m).sum() > 1 else np.nan,
                             R_per_risk_rest=float(df.loc[~m, "pnl"].sum() / df.loc[~m, "risk"].sum()) if (~m).sum() else np.nan,
                             flow_ratio=sig_share / sess_share if sess_share else np.nan,
                             sig_share=sig_share, sess_share=sess_share,
                             t_episode=t_ep, p_episode=p_ep, n_episodes=g_ep,
                             t_year=t_yr, p_year=p_yr, n_years_paired=n_yr,
                             worst_R=float(sub["R"].min()) if len(sub) else np.nan,
                             pnl_share=st["sum_pnl"] / base["sum_pnl"] if base["sum_pnl"] else np.nan))
C = pd.DataFrame(rows)
# multiple comparisons: family = (dim) across all strategies + book; also per-strategy-within-dim
for fam_col, tag in [(["dim"], "fam_dim"), (["dim", "strategy"], "fam_strat")]:
    for (keys), idx in C.groupby(fam_col).indices.items():
        sub = C.iloc[idx]
        for pcol in ["p_episode", "p_year"]:
            C.loc[sub.index, f"q_{pcol}_{tag}"] = bh_fdr(sub[pcol].values)
            ntest = int(sub[pcol].notna().sum())
            C.loc[sub.index, f"bonf_{pcol}_{tag}"] = np.minimum(sub[pcol].values * max(ntest, 1), 1.0)
C["survives_fdr10_year"] = (C["q_p_year_fam_dim"] < 0.10)
C["survives_bonf05_year"] = (C["bonf_p_year_fam_dim"] < 0.05)
C["survives_fdr10_episode"] = (C["q_p_episode_fam_dim"] < 0.10)
C["survives_fdr10_strat_year"] = (C["q_p_year_fam_strat"] < 0.10)

print("\n=== BOOK by month ===")
print(C[(C.strategy == "BOOK") & (C.dim == "month")][["cell", "N", "n_years", "avgR", "sdR", "R_per_risk", "flow_ratio", "sum_pnl", "t_episode", "p_episode", "t_year", "p_year", "q_p_year_fam_dim", "worst_R"]].to_string(index=False))
for dim in ["quarter", "half", "wom", "tom", "opex", "dow", "eseason_fixed", "eseason_data", "holiday"]:
    print(f"\n=== BOOK by {dim} ===")
    print(C[(C.strategy == "BOOK") & (C.dim == dim)][["cell", "N", "avgR", "sdR", "R_per_risk", "flow_ratio", "sum_pnl", "t_episode", "p_episode", "t_year", "p_year", "q_p_year_fam_dim"]].to_string(index=False))

print("\n=== per-strategy month grid: avgR (N) ===")
M = C[C.dim == "month"].pivot(index="strategy", columns="cell", values="avgR")[MONTHS]
Nn = C[C.dim == "month"].pivot(index="strategy", columns="cell", values="N")[MONTHS]
print(M.round(2).to_string())
print(Nn.to_string())
print("\n=== per-strategy month grid: flow ratio ===")
print(C[C.dim == "month"].pivot(index="strategy", columns="cell", values="flow_ratio")[MONTHS].round(2).to_string())

print("\n=== cells surviving FDR10 (year-paired, family=dim) ===")
surv = C[C["survives_fdr10_year"] & (C.N >= 10)].sort_values("q_p_year_fam_dim")
print(surv[["strategy", "dim", "cell", "N", "n_years", "avgR", "avgR_rest", "R_per_risk", "flow_ratio", "t_year", "p_year", "q_p_year_fam_dim", "bonf_p_year_fam_dim", "t_episode", "p_episode"]].to_string(index=False))
print("\n=== cells surviving FDR10 (episode-clustered, family=dim) ===")
surv2 = C[C["survives_fdr10_episode"] & (C.N >= 10)].sort_values("q_p_episode_fam_dim")
print(surv2[["strategy", "dim", "cell", "N", "n_years", "avgR", "avgR_rest", "R_per_risk", "flow_ratio", "t_year", "p_year", "t_episode", "p_episode", "q_p_episode_fam_dim"]].to_string(index=False))

print("\n=== nominal p<0.05 on BOTH tests, N>=15 (candidates) ===")
cand = C[(C.p_year < 0.05) & (C.p_episode < 0.05) & (C.N >= 15)].sort_values("p_year")
print(cand[["strategy", "dim", "cell", "N", "n_years", "avgR", "avgR_rest", "R_per_risk", "flow_ratio", "t_year", "p_year", "t_episode", "p_episode", "q_p_year_fam_dim"]].to_string(index=False))

# flow x quality screen (month cells): flow high (>1.25) and quality poor (avgR < avgR_rest - 0.15) / good
mm = C[(C.dim == "month") & (C.N >= 10)].copy()
mm["dR"] = mm["avgR"] - mm["avgR_rest"]
hi_poor = mm[(mm.flow_ratio > 1.25) & (mm.dR < -0.15)].sort_values("dR")
hi_good = mm[(mm.flow_ratio > 1.25) & (mm.dR > 0.15)].sort_values("dR", ascending=False)
print("\n=== FLOW HIGH + QUALITY POOR (month cells, N>=10) ===")
print(hi_poor[["strategy", "cell", "N", "n_years", "flow_ratio", "avgR", "avgR_rest", "R_per_risk", "sum_pnl", "t_year", "p_year", "q_p_year_fam_dim"]].to_string(index=False))
print("\n=== FLOW HIGH + QUALITY GOOD (month cells, N>=10) ===")
print(hi_good[["strategy", "cell", "N", "n_years", "flow_ratio", "avgR", "avgR_rest", "R_per_risk", "sum_pnl", "t_year", "p_year", "q_p_year_fam_dim"]].to_string(index=False))

# variance seasonality at trade level: sdR by month vs rest, Levene (Brown-Forsythe) with year-clustered rank test
from scipy import stats as sps
var_rows = []
for sname, df in groups.items():
    for dim in ["month", "quarter", "half", "eseason_data"]:
        col = DIMS[dim]
        for cell in sorted(df[col].unique()):
            m = (df[col] == cell).values
            if m.sum() < 10 or (~m).sum() < 10:
                continue
            a, b = df.loc[m, "R"].values, df.loc[~m, "R"].values
            W, p = sps.levene(a, b, center="median")
            # year-paired sd difference
            yy = df.assign(c=m).groupby(["yr", "c"])["R"].std().unstack().dropna()
            if len(yy) >= 3 and yy.shape[1] == 2:
                dd = yy[True] - yy[False]
                t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if dd.std(ddof=1) > 0 else np.nan
                py = 2 * sps.t.sf(abs(t), len(dd) - 1) if np.isfinite(t) else np.nan
            else:
                t, py = np.nan, np.nan
            var_rows.append(dict(strategy=sname, dim=dim, cell=str(cell), N=int(m.sum()), sdR=float(a.std(ddof=1)),
                                 sdR_rest=float(b.std(ddof=1)), sd_ratio=float(a.std(ddof=1) / b.std(ddof=1)),
                                 levene_p=float(p), t_year_sd=t, p_year_sd=py, n_years=int(len(yy)) if len(yy) else 0))
V = pd.DataFrame(var_rows)
for dim, idx in V.groupby("dim").indices.items():
    V.loc[V.index[idx], "q_levene_fam"] = bh_fdr(V.iloc[idx]["levene_p"].values)
    V.loc[V.index[idx], "q_year_sd_fam"] = bh_fdr(V.iloc[idx]["p_year_sd"].values)
print("\n=== VARIANCE seasonality (trade R): cells with Levene q<0.10 or year-paired p<0.05 ===")
vv = V[((V.q_levene_fam < 0.10) | (V.p_year_sd < 0.05)) & (V.N >= 15)].sort_values("levene_p")
print(vv[["strategy", "dim", "cell", "N", "sdR", "sdR_rest", "sd_ratio", "levene_p", "q_levene_fam", "t_year_sd", "p_year_sd", "n_years"]].to_string(index=False))

OUT = dict(
    meta=dict(rows_after_tranche_collapse=int(len(led)), span=[str(led["sig"].min().date()), str(led["sig"].max().date())],
              episode_gap_sessions=5, tests="episode-clustered OLS t on cell dummy; year-paired t of cell-vs-complement means",
              mc_families="fam_dim = all cells in one dimension across strategies+book; fam_strat = cells within (dim, strategy)"),
    cells=C.round(5).to_dict("records"),
    survivors_fdr10_year=surv.round(4).to_dict("records"),
    survivors_fdr10_episode=surv2.round(4).to_dict("records"),
    candidates_both_p05=cand.round(4).to_dict("records"),
    flow_high_quality_poor=hi_poor.round(4).to_dict("records"),
    flow_high_quality_good=hi_good.round(4).to_dict("records"),
    variance_cells=V.round(5).to_dict("records"),
    n_tests=dict(total=int(C["p_year"].notna().sum()), per_dim=C.groupby("dim")["p_year"].apply(lambda s: int(s.notna().sum())).to_dict()),
)
jdump(OUT, HERE / "seasonality_flow_cells.json")
C.to_csv(HERE / "seasonality_flow_cells.csv", index=False)
print("\nwrote", HERE / "seasonality_flow_cells.json")
