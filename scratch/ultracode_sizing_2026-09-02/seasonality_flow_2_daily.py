"""seasonality_flow_2_daily.py (2026-09-02): daily MTM PnL and VOLATILITY seasonality
per strategy and for the book (flat $750k). Cells: month, quarter, half, wom, tom,
opex, dow, earnings season (data), holiday adjacency. Per cell: mean bps/day, sd
bps/day, ann Sharpe, worst day, PnL share, active-day share; year-paired t on the
mean and on the sd (one obs per year), Levene on daily returns; BH-FDR per dimension.
Plus: drawdown-trough month histogram, worst-21d-window start months, and the
persistence of the month-vol ranking across years (mean pairwise Spearman).
Writes seasonality_flow_daily.json.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps
from seasonality_flow_common import (HERE, ROOT, NAV, MONTHS, DOW, load_strategy_daily, load_spy, trading_calendar,
                                     bh_fdr, jdump, maxdd)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
strat, tot = load_strategy_daily()
earn = pd.read_parquet(ROOT / "data/earnings_calendar.parquet", columns=["date"])
cal = trading_calendar(load_spy().index, earn)
cal["opex"] = np.where(cal["opex_week"], "opex", np.where(cal["post_opex_week"], "post_opex", "other"))
cal["quarter_s"] = "Q" + cal["quarter"].astype(str)
cal["tom_s"] = np.where(cal["tom"], "TOM", "mid")
cal["ed_s"] = np.where(cal["eseason_data"], "season", "off")
DIMS = {"month": "mname", "quarter": "quarter_s", "half": "half", "wom": "wom", "tom": "tom_s", "opex": "opex",
        "dow": "dname", "eseason_data": "ed_s", "holiday": "holiday_adj"}
idx = tot.index.intersection(cal.index)
tot = tot.reindex(idx)
strat = strat.reindex(idx)
cal = cal.loc[idx]
series = {s: strat[s] for s in strat.columns}
series["BOOK"] = tot

rows = []
for sname, ser in series.items():
    r = (ser / NAV).astype(float)
    active = ser != 0
    first = ser[active].index.min() if active.any() else None
    r = r[r.index >= first]
    c = cal.loc[r.index]
    yr = c["year"].values
    for dim, col in DIMS.items():
        vals = c[col].values
        cells = sorted(set(vals), key=lambda v: (MONTHS.index(v) if v in MONTHS else (DOW.index(v) if v in DOW else str(v))))
        for cell in cells:
            m = vals == cell
            a, b = r.values[m], r.values[~m]
            if m.sum() < 20 or (~m).sum() < 20:
                continue
            act_share = float((ser.reindex(r.index).values[m] != 0).mean())
            # year-paired mean and sd
            df = pd.DataFrame({"r": r.values, "c": m, "y": yr})
            gm = df.groupby(["y", "c"])["r"].mean().unstack().dropna()
            gs = df.groupby(["y", "c"])["r"].std().unstack().dropna()
            def pt(g):
                if len(g) < 3 or g.shape[1] < 2:
                    return np.nan, np.nan, len(g)
                d = g[True] - g[False]
                if d.std(ddof=1) == 0:
                    return np.nan, np.nan, len(d)
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                return float(t), float(2 * sps.t.sf(abs(t), len(d) - 1)), int(len(d))
            tm, pm, ny = pt(gm)
            ts, ps, nys = pt(gs)
            lev = sps.levene(a, b, center="median").pvalue
            rows.append(dict(strategy=sname, dim=dim, cell=str(cell), days=int(m.sum()), n_years=ny,
                             mean_bps=float(a.mean() * 1e4), sd_bps=float(a.std(ddof=1) * 1e4),
                             sharpe=float(a.mean() / a.std(ddof=1) * np.sqrt(252)) if a.std(ddof=1) > 0 else np.nan,
                             mean_bps_rest=float(b.mean() * 1e4), sd_bps_rest=float(b.std(ddof=1) * 1e4),
                             sharpe_rest=float(b.mean() / b.std(ddof=1) * np.sqrt(252)) if b.std(ddof=1) > 0 else np.nan,
                             sd_ratio=float(a.std(ddof=1) / b.std(ddof=1)) if b.std(ddof=1) > 0 else np.nan,
                             kelly_ratio=float((a.mean() / a.var()) / (b.mean() / b.var())) if b.mean() > 0 and a.var() > 0 else np.nan,
                             worst_day_pct=float(a.min() * 100), cvar5_bps=float(np.mean(np.sort(a)[: max(1, int(0.05 * len(a)))]) * 1e4),
                             pnl_share=float(a.sum() / r.sum()) if r.sum() != 0 else np.nan, day_share=float(m.mean()),
                             active_share=act_share, t_year_mean=tm, p_year_mean=pm, t_year_sd=ts, p_year_sd=ps, levene_p=float(lev)))
D = pd.DataFrame(rows)
for dim, ix in D.groupby("dim").indices.items():
    for pc in ["p_year_mean", "p_year_sd", "levene_p"]:
        D.loc[D.index[ix], f"q_{pc}"] = bh_fdr(D.iloc[ix][pc].values)

print("=== BOOK daily by month ===")
print(D[(D.strategy == "BOOK") & (D.dim == "month")][["cell", "days", "mean_bps", "sd_bps", "sharpe", "sd_ratio", "kelly_ratio", "worst_day_pct", "cvar5_bps", "pnl_share", "t_year_mean", "p_year_mean", "t_year_sd", "p_year_sd", "levene_p", "q_levene_p"]].to_string(index=False))
for dim in ["quarter", "half", "wom", "tom", "opex", "dow", "eseason_data", "holiday"]:
    print(f"\n=== BOOK daily by {dim} ===")
    print(D[(D.strategy == "BOOK") & (D.dim == dim)][["cell", "days", "mean_bps", "sd_bps", "sharpe", "sd_ratio", "kelly_ratio", "worst_day_pct", "pnl_share", "t_year_mean", "p_year_mean", "t_year_sd", "p_year_sd", "levene_p"]].to_string(index=False))

print("\n=== per-strategy daily Sharpe by month ===")
print(D[D.dim == "month"].pivot(index="strategy", columns="cell", values="sharpe")[MONTHS].round(2).to_string())
print("\n=== per-strategy daily sd_bps by month ===")
print(D[D.dim == "month"].pivot(index="strategy", columns="cell", values="sd_bps")[MONTHS].round(1).to_string())
print("\n=== per-strategy mean_bps by month ===")
print(D[D.dim == "month"].pivot(index="strategy", columns="cell", values="mean_bps")[MONTHS].round(1).to_string())

print("\n=== daily cells surviving FDR10 on MEAN (year-paired) ===")
sm = D[(D.q_p_year_mean < 0.10)].sort_values("q_p_year_mean")
print(sm[["strategy", "dim", "cell", "days", "n_years", "mean_bps", "mean_bps_rest", "sharpe", "sharpe_rest", "t_year_mean", "p_year_mean", "q_p_year_mean"]].to_string(index=False))
print("\n=== daily cells surviving FDR10 on VARIANCE (Levene) ===")
sv = D[(D.q_levene_p < 0.10)].sort_values("q_levene_p")
print(sv[["strategy", "dim", "cell", "days", "n_years", "sd_bps", "sd_bps_rest", "sd_ratio", "mean_bps", "mean_bps_rest", "kelly_ratio", "levene_p", "q_levene_p", "t_year_sd", "p_year_sd"]].to_string(index=False))
print("\n=== nominal p<0.05 on mean (year-paired), days>=100 ===")
nm = D[(D.p_year_mean < 0.05) & (D.days >= 100)].sort_values("p_year_mean")
print(nm[["strategy", "dim", "cell", "days", "n_years", "mean_bps", "mean_bps_rest", "sharpe", "sharpe_rest", "t_year_mean", "p_year_mean", "q_p_year_mean"]].to_string(index=False))

# ---- drawdown geography
c = tot.cumsum()
dd = c - c.cummax()
trough_months = []
in_dd = False
cur_min, cur_date = 0, None
for d, v in dd.items():
    if v < 0 and not in_dd:
        in_dd, cur_min, cur_date = True, v, d
    elif v < 0 and v < cur_min:
        cur_min, cur_date = v, d
    elif v == 0 and in_dd:
        in_dd = False
        if cur_min < -0.02 * NAV:
            trough_months.append((cur_date, cur_min))
tm = pd.DataFrame(trough_months, columns=["trough", "depth"])
tm["month"] = tm["trough"].dt.month
th = tm.groupby("month")["depth"].agg(["size", "sum", "min"]).reindex(range(1, 13)).fillna(0)
th.index = MONTHS
print("\n=== drawdown troughs (> 2% NAV) by trough month: count, total depth, deepest ===")
print(th.to_string())
w21 = tot.rolling(21).sum()
worst = w21.nsmallest(60)
# decluster: keep windows at least 42 sessions apart
kept = []
for d, v in worst.items():
    if all(abs((d - k).days) > 60 for k, _ in kept):
        kept.append((d, v))
    if len(kept) >= 15:
        break
wk = pd.DataFrame(kept, columns=["end", "pnl21"])
wk["end_month"] = wk["end"].dt.month
print("\n=== 15 worst declustered 21d windows: end month ===")
print(wk.to_string(index=False))
print(wk["end_month"].value_counts().sort_index().to_string())

# ---- persistence of month-vol ranking across years (book and per strategy)
pers = {}
for sname, ser in series.items():
    r = ser / NAV
    r = r[r != 0] if sname != "BOOK" else r
    if len(r) < 500:
        continue
    cc = cal.loc[r.index]
    tab = pd.DataFrame({"r": r.values, "y": cc["year"].values, "m": cc["month"].values}).groupby(["y", "m"])["r"].std().unstack()
    tab = tab.dropna(thresh=10)
    if len(tab) < 4:
        continue
    rk = tab.rank(axis=1)
    cors = []
    yrs = list(tab.index)
    for i in range(len(yrs)):
        for j in range(i + 1, len(yrs)):
            a, b = rk.iloc[i], rk.iloc[j]
            ok = a.notna() & b.notna()
            if ok.sum() >= 8:
                cors.append(sps.spearmanr(a[ok], b[ok]).correlation)
    # mean-vol by month pooled vs year-demeaned
    pers[sname] = dict(years=len(tab), mean_pairwise_spearman=float(np.nanmean(cors)) if cors else np.nan,
                       n_pairs=len(cors), month_sd_rank_mean=rk.mean().round(2).to_dict())
print("\n=== persistence of month-vol ranking across years (mean pairwise Spearman of within-year month-sd ranks) ===")
for k, v in pers.items():
    print(f"{k:28s} years={v['years']:3d} rho={v['mean_pairwise_spearman']:.3f}")

OUT = dict(cells=D.round(5).to_dict("records"), fdr_mean=sm.round(4).to_dict("records"), fdr_var=sv.round(4).to_dict("records"),
           nominal_mean=nm.round(4).to_dict("records"), dd_troughs_by_month=th.reset_index().rename(columns={"index": "month"}).to_dict("records"),
           worst21_windows=wk.to_dict("records"), vol_rank_persistence=pers)
jdump(OUT, HERE / "seasonality_flow_daily.json")
D.to_csv(HERE / "seasonality_flow_daily.csv", index=False)
print("wrote", HERE / "seasonality_flow_daily.json")
