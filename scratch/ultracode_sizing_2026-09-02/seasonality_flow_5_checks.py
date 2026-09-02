"""seasonality_flow_5_checks.py (2026-09-02): robustness checks on the cells that
looked interesting in studies 1-4.
1. January book variance: which years drive it (episodic vs persistent)?
2. December lower trade-R variance: composition (OVS share) or within-strategy?
3. OVS same-day 5+ cluster cell: per-episode list, LOYO stability, and how much of
   that cell's staged risk the 250 bps per-strategy daily cap already trims.
4. Friday MTM weakness: by strategy and by year.
5. September: book by year (the external prior says bad; the ledger says fine).
6. Nov-Apr vs May-Oct on the daily basis by year: Kelly ratio stability.
Writes seasonality_flow_checks.json.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps
from seasonality_flow_common import (HERE, NAV, MONTHS, load_ledger, load_strategy_daily, load_spy, trading_calendar, jdump)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
led = load_ledger()
key = ["Strategy", "Tier", "Ticker", "sig", "Entry Date"]
g = led.groupby(key, as_index=False).agg(pnl=("pnl", "sum"), risk=("risk", "sum"), yr=("yr", "first"), exit=("Exit Date", "first"))
g["R"] = g["pnl"] / g["risk"]
led = g
led["month"] = led["sig"].dt.month
strat, tot = load_strategy_daily()
cal = trading_calendar(load_spy().index)
OUT = {}

# 1. January variance by year
r = tot / NAV
rj = r[r.index.month == 1]; rr = r[r.index.month != 1]
jan = rj.groupby(rj.index.year).agg(["std", "mean", "min", "size"])
rest = rr.groupby(rr.index.year).agg(["std"])
jan["sd_ratio_vs_rest_of_year"] = jan["std"] / rest["std"]
jan[["std", "mean", "min"]] *= 1e4
print("=== 1. January book daily sd (bps) by year vs rest of that year ===")
print(jan.round(2).to_string())
print("years with Jan sd ratio > 1.3:", int((jan["sd_ratio_vs_rest_of_year"] > 1.3).sum()), "of", len(jan), "; median ratio", round(jan["sd_ratio_vs_rest_of_year"].median(), 2))
OUT["january_variance_by_year"] = jan.round(4).reset_index().rename(columns={"index": "year"}).to_dict("records")
OUT["january_sd_ratio_median"] = float(jan["sd_ratio_vs_rest_of_year"].median())

# 2. December sdR composition
dec = led[led.month == 12]
print("\n=== 2. December trade-R variance: composition ===")
print("OVS share of Dec signals:", round((dec.Strategy == "Overbot Vol Spike").mean(), 3), " vs all-months:", round((led.Strategy == "Overbot Vol Spike").mean(), 3))
comp = []
for s, df in led.groupby("Strategy"):
    a, b = df[df.month == 12]["R"], df[df.month != 12]["R"]
    if len(a) >= 10:
        comp.append(dict(strategy=s, N_dec=len(a), sdR_dec=float(a.std()), sdR_rest=float(b.std()), ratio=float(a.std() / b.std()),
                         levene_p=float(sps.levene(a, b, center="median").pvalue)))
comp = pd.DataFrame(comp)
print(comp.round(3).to_string(index=False))
# within-strategy standardized R: does Dec still show lower variance?
led["Rz"] = led.groupby("Strategy")["R"].transform(lambda s: (s - s.mean()) / s.std())
a, b = led[led.month == 12]["Rz"], led[led.month != 12]["Rz"]
lev = sps.levene(a, b, center="median").pvalue
print(f"within-strategy standardized R: Dec sd {a.std():.3f} vs rest {b.std():.3f}, Levene p={lev:.3f}")
OUT["december_variance_composition"] = dict(per_strategy=comp.round(4).to_dict("records"), ovs_share_dec=float((dec.Strategy == "Overbot Vol Spike").mean()),
                                            standardized_sd_dec=float(a.std()), standardized_sd_rest=float(b.std()), levene_p_standardized=float(lev))

# 3. OVS same-day 5+ cell
ovs = led[led.Strategy == "Overbot Vol Spike"].copy()
ovs["n_day"] = ovs.groupby("sig")["sig"].transform("size")
ovs["risk_day_bps"] = ovs.groupby("sig")["risk"].transform("sum") / NAV * 1e4
big = ovs[ovs.n_day >= 5]
ep = big.groupby("sig").agg(n=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum"), risk_bps=("risk_day_bps", "first"))
ep["yr"] = ep.index.year
print("\n=== 3. OVS days with >= 5 signals: per-day ===")
print(ep.round(2).to_string())
yr = big.groupby("yr").agg(n=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum"))
print(yr.round(2).to_string())
loyo = []
for y in sorted(ovs["yr"].unique()):
    tr = ovs[ovs.yr != y]
    d = tr[tr.n_day >= 5]["R"].mean() - tr[tr.n_day < 5]["R"].mean()
    loyo.append(d)
print(f"LOYO diff (5+ minus <5) min={min(loyo):.3f} max={max(loyo):.3f}; years where 5+ cell avgR > 0: {(yr.avgR > 0).sum()}/{len(yr)}")
print("staged risk on 5+ days (bps): median", round(ep.risk_bps.median(), 1), " p90", round(ep.risk_bps.quantile(.9), 1), " max", round(ep.risk_bps.max(), 1),
      " (ledger risk is POST-cap; days at ~250 are cap-bound)")
print("days with risk >= 240 bps:", int((ep.risk_bps >= 240).sum()), "of", len(ep))
OUT["ovs_cluster_days"] = dict(per_day=ep.round(3).reset_index().to_dict("records"), per_year=yr.round(3).reset_index().to_dict("records"),
                               loyo_diff_min=float(min(loyo)), loyo_diff_max=float(max(loyo)),
                               risk_bps_median=float(ep.risk_bps.median()), risk_bps_max=float(ep.risk_bps.max()), days_cap_bound=int((ep.risk_bps >= 240).sum()))
# by n_day finer
fine = ovs.groupby(pd.cut(ovs.n_day, [0, 1, 2, 3, 4, 6, 9, 15, 100])).agg(N=("R", "size"), avgR=("R", "mean"), sdR=("R", "std"), pnl=("pnl", "sum"), days=("sig", "nunique"))
print(fine.round(3).to_string())
OUT["ovs_by_sameday_count"] = fine.round(4).reset_index().astype({"n_day": str}).to_dict("records")

# 4. Friday MTM weakness
c = pd.DataFrame(index=tot.index); c["dow"] = tot.index.dayofweek; c["month"] = tot.index.month
fri = c["dow"].values == 4
print("\n=== 4. Friday daily MTM by strategy (bps/day, Fri vs other) ===")
rows = []
for s in list(strat.columns) + ["BOOK"]:
    ser = (strat[s] if s != "BOOK" else tot) / NAV * 1e4
    act = ser != 0
    ser = ser[ser.index >= ser[act].index.min()]
    f = ser.index.dayofweek.values == 4
    a, b = ser[f], ser[~f]
    yy = pd.DataFrame({"v": ser.values, "f": f, "y": ser.index.year}).groupby(["y", "f"])["v"].mean().unstack().dropna()
    d = yy[True] - yy[False]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    rows.append(dict(strategy=s, fri_mean=float(a.mean()), other_mean=float(b.mean()), fri_sd=float(a.std()), other_sd=float(b.std()),
                     t_year=float(t), years_fri_worse=int((d < 0).sum()), years=int(len(d))))
FR = pd.DataFrame(rows)
print(FR.round(3).to_string(index=False))
OUT["friday_mtm"] = FR.round(4).to_dict("records")

# 5. September by year
sep = led[led.month == 9].groupby("yr").agg(N=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum"))
print("\n=== 5. September (signal month) by year ===")
print(sep.round(2).to_string())
print("Sep years with positive pnl:", int((sep.pnl > 0).sum()), "/", len(sep), " avgR>0:", int((sep.avgR > 0).sum()))
OUT["september_by_year"] = sep.round(3).reset_index().to_dict("records")

# 6. halves on the daily basis, by year: Kelly ratio stability
c["half"] = np.where(c["month"].isin([11, 12, 1, 2, 3, 4]), "NovApr", "MayOct")
df = pd.DataFrame({"r": r.values, "half": c["half"].values, "y": r.index.year})
hy = df.groupby(["y", "half"])["r"].agg(["mean", "std"]).unstack()
hy["kelly_NovApr"] = hy[("mean", "NovApr")] / hy[("std", "NovApr")] ** 2
hy["kelly_MayOct"] = hy[("mean", "MayOct")] / hy[("std", "MayOct")] ** 2
hy["kelly_ratio_MayOct_over_NovApr"] = hy["kelly_MayOct"] / hy["kelly_NovApr"]
hy["sd_ratio"] = hy[("std", "MayOct")] / hy[("std", "NovApr")]
hy["mean_ratio"] = hy[("mean", "MayOct")] / hy[("mean", "NovApr")]
print("\n=== 6. May-Oct vs Nov-Apr daily basis by year ===")
print(hy[["sd_ratio", "mean_ratio", "kelly_ratio_MayOct_over_NovApr"]].round(2).to_string())
kr = hy["kelly_ratio_MayOct_over_NovApr"].replace([np.inf, -np.inf], np.nan).dropna()
print("median Kelly ratio MayOct/NovApr:", round(kr.median(), 2), " years MayOct sd lower:", int((hy["sd_ratio"] < 1).sum()), "/", len(hy),
      " years MayOct mean lower:", int((hy["mean_ratio"] < 1).sum()))
OUT["halves_by_year"] = dict(median_kelly_ratio=float(kr.median()), years_sd_lower=int((hy["sd_ratio"] < 1).sum()), years=int(len(hy)),
                             years_mean_lower=int((hy["mean_ratio"] < 1).sum()),
                             rows=[dict(year=int(y), sd_ratio=float(a), mean_ratio=float(b), kelly_ratio=float(k) if np.isfinite(k) else None)
                                   for y, a, b, k in zip(hy.index, hy["sd_ratio"], hy["mean_ratio"], hy["kelly_ratio_MayOct_over_NovApr"])])
jdump(OUT, HERE / "seasonality_flow_checks.json")
print("wrote", HERE / "seasonality_flow_checks.json")
