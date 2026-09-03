"""GAP 14: the 'midterm base rate' reading. Per strategy avgR and PnL per unit risk in
midterm years (year % 4 == 2, by signal date) vs other years 2003-2026; 2026 alone vs the
prior midterms (2006/2010/2014/2018/2022), calendar-aligned (Jan-Aug) and full-year; OLV
and LT Trend 2026 YTD vs their own history and vs the tilt's 2027-Q1 retirement thresholds
(OLV PnL/risk < 0.5, LT Trend < 0.15). Ledger collapsed to trades (OVS tranches summed),
flat $750k. Writes gap14_midterm.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIZ = HERE.parent
sys.path.insert(0, str(SIZ))
from flow_conditional_lib import load_ledger  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40)
OUT: dict = {}
led = load_ledger()
led["month"] = led["Signal Date"].dt.month
led["midterm"] = (led.year % 4 == 2)
led["cycle"] = led.year % 4
PRIOR_MID = [2006, 2010, 2014, 2018, 2022]
CUT_MONTH = 8   # 2026 runs through Aug (last signal 2026-08-28)


def stats(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return dict(N=0, avgR=np.nan, rpr=np.nan, win=np.nan, pnl=0.0, risk=0.0)
    return dict(N=int(len(g)), avgR=float(g.R.mean()), rpr=float(g.PnL.sum() / g.Risk.sum()), win=float((g.R > 0).mean()), pnl=float(g.PnL.sum()), risk=float(g.Risk.sum()),
                se=float(g.R.std() / np.sqrt(len(g))) if len(g) > 1 else np.nan)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


strats = ["BOOK"] + sorted(led.Strategy.unique())
rows = []
for s in strats:
    g = led if s == "BOOK" else led[led.Strategy == s]
    mid, oth = g[g.midterm], g[~g.midterm]
    pm = g[g.year.isin(PRIOR_MID)]
    pm_ja = pm[pm.month <= CUT_MONTH]
    y26 = g[g.year == 2026]
    non_mid_ja = oth[oth.month <= CUT_MONTH]
    # annual PnL/risk by year for the rank of 2026
    yr = g.groupby("year").agg(pnl=("PnL", "sum"), risk=("Risk", "sum"), N=("R", "size"))
    yr["rpr"] = yr.pnl / yr.risk
    yr_ja = g[g.month <= CUT_MONTH].groupby("year").agg(pnl=("PnL", "sum"), risk=("Risk", "sum"), N=("R", "size")); yr_ja["rpr"] = yr_ja.pnl / yr_ja.risk
    yr_ja = yr_ja[yr_ja.N >= 3]
    rank26 = int((yr_ja.rpr < yr_ja.rpr.get(2026, np.nan)).sum()) + 1 if 2026 in yr_ja.index else None
    pm_yrs = {int(y): float(yr_ja.rpr[y]) for y in PRIOR_MID if y in yr_ja.index}
    r = dict(strategy=s, N_all=len(g), N_mid=len(mid), N_oth=len(oth),
             avgR_mid=stats(mid)["avgR"], avgR_oth=stats(oth)["avgR"], rpr_mid=stats(mid)["rpr"], rpr_oth=stats(oth)["rpr"], t_mid_vs_oth=welch(mid.R, oth.R),
             N_26=len(y26), avgR_26=stats(y26)["avgR"], rpr_26=stats(y26)["rpr"], win_26=stats(y26)["win"], pnl_26=stats(y26)["pnl"],
             avgR_priormid_JA=stats(pm_ja)["avgR"], rpr_priormid_JA=stats(pm_ja)["rpr"], N_priormid_JA=len(pm_ja),
             avgR_nonmid_JA=stats(non_mid_ja)["avgR"], rpr_nonmid_JA=stats(non_mid_ja)["rpr"],
             t_26_vs_priormid_JA=welch(y26.R, pm_ja.R), t_26_vs_nonmid_JA=welch(y26.R, non_mid_ja.R), t_26_vs_all_prior=welch(y26.R, g[g.year < 2026].R),
             priormid_JA_rpr_by_year=pm_yrs, priormid_JA_rpr_min=min(pm_yrs.values()) if pm_yrs else np.nan, priormid_JA_rpr_max=max(pm_yrs.values()) if pm_yrs else np.nan,
             rank_2026_JA_of=f"{rank26}/{len(yr_ja)}" if rank26 else None)
    r["within_prior_midterm_range"] = bool(r["priormid_JA_rpr_min"] <= r["rpr_26"] <= r["priormid_JA_rpr_max"]) if pm_yrs and np.isfinite(r["rpr_26"]) else None
    r["explained_by_base_rate"] = ("yes" if r["within_prior_midterm_range"] else ("below" if (pm_yrs and np.isfinite(r["rpr_26"]) and r["rpr_26"] < r["priormid_JA_rpr_min"]) else "above")) if pm_yrs and np.isfinite(r["rpr_26"]) else "n/a"
    rows.append(r)
D = pd.DataFrame(rows)
OUT["per_strategy"] = D.to_dict("records")
print("=== midterm vs other, 2003-2026 (signal-date year % 4 == 2) ===")
print(D[["strategy", "N_mid", "N_oth", "avgR_mid", "avgR_oth", "rpr_mid", "rpr_oth", "t_mid_vs_oth"]].round(3).to_string(index=False))
print("\n=== 2026 (Jan-Aug) vs prior midterms Jan-Aug (2006/10/14/18/22) and vs non-midterm Jan-Aug ===")
print(D[["strategy", "N_26", "avgR_26", "rpr_26", "pnl_26", "N_priormid_JA", "avgR_priormid_JA", "rpr_priormid_JA", "rpr_nonmid_JA", "t_26_vs_priormid_JA", "t_26_vs_nonmid_JA", "priormid_JA_rpr_min", "priormid_JA_rpr_max", "rank_2026_JA_of", "explained_by_base_rate"]].round(2).to_string(index=False))

# ---- prior-midterm Jan-Aug per year, book and the two names
print("\n=== PnL/risk Jan-Aug by midterm year (book, OLV, LT Trend, OVS) ===")
OUT["midterm_years_JA"] = {}
for s in ["BOOK", "Oversold Low Volume", "LT Trend ST OS", "Overbot Vol Spike", "Weak Close Decent Sznls", "SPY QQQ MonFri Reversion"]:
    g = led if s == "BOOK" else led[led.Strategy == s]
    ja = g[g.month <= CUT_MONTH]
    t = ja[ja.year.isin(PRIOR_MID + [2026])].groupby("year").agg(N=("R", "size"), avgR=("R", "mean"), pnl=("PnL", "sum"), risk=("Risk", "sum"))
    t["rpr"] = t.pnl / t.risk
    OUT["midterm_years_JA"][s] = t.round(4).reset_index().to_dict("records")
    print(f"  {s}: " + "  ".join(f"{int(y)}: N{int(r.N)} rpr {r.rpr:+.2f} avgR {r.avgR:+.2f}" for y, r in t.iterrows()))

# ---- OLV and LT Trend in depth
print("\n=== OLV and LT Trend: 2026 YTD vs own history, by tier, vs the 2027-Q1 retirement thresholds ===")
OUT["olv_lt"] = {}
THR = {"Oversold Low Volume": 0.5, "LT Trend ST OS": 0.15}
for s in ["Oversold Low Volume", "LT Trend ST OS"]:
    g = led[led.Strategy == s]
    rec = {}
    for lab, h in [("all 2003-2025", g[g.year < 2026]), ("non-midterm", g[~g.midterm & (g.year < 2026)]), ("prior midterms", g[g.year.isin(PRIOR_MID)]),
                   ("prior midterms Jan-Aug", g[g.year.isin(PRIOR_MID) & (g.month <= CUT_MONTH)]), ("2024", g[g.year == 2024]), ("2025", g[g.year == 2025]),
                   ("2026 YTD", g[g.year == 2026]), ("2026 Liquid", g[(g.year == 2026) & (g.Tier == "Liquid")]), ("2026 Overflow", g[(g.year == 2026) & (g.Tier == "Overflow")]),
                   ("2026 Jan-Apr", g[(g.year == 2026) & (g.month <= 4)]), ("2026 May-Aug", g[(g.year == 2026) & (g.month >= 5)]),
                   ("hist Liquid", g[(g.year < 2026) & (g.Tier == "Liquid")]), ("hist Overflow", g[(g.year < 2026) & (g.Tier == "Overflow")])]:
        st = stats(h); rec[lab] = st
        print(f"  {s:20s} {lab:24s} N {st['N']:4d} avgR {st['avgR']:+.3f} rpr {st['rpr']:+.3f} win {st['win']:.0%} pnl {st['pnl']:>10,.0f}")
    yr = g.groupby("year").agg(N=("R", "size"), pnl=("PnL", "sum"), risk=("Risk", "sum"), avgR=("R", "mean")); yr["rpr"] = yr.pnl / yr.risk
    rec["by_year"] = yr.round(4).reset_index().to_dict("records")
    below = yr[(yr.rpr < THR[s]) & (yr.N >= 5)]
    rec["years_below_threshold"] = [int(y) for y in below.index]
    rec["threshold"] = THR[s]
    print(f"  {s}: years (N>=5) with full-year PnL/risk below the retirement line {THR[s]}: {[int(y) for y in below.index]} of {int((yr.N >= 5).sum())}; 2026 YTD rpr {rec['2026 YTD']['rpr']:+.3f}")
    print("   by year: " + " ".join(f"{int(y)}:{r.rpr:+.2f}({int(r.N)})" for y, r in yr.iterrows()))
    OUT["olv_lt"][s] = rec

# ---- OVS same-day cluster days in 2026 (OVS fills >= 5 the same signal day) for GAP 7's footnote
ovs = led[led.Strategy == "Overbot Vol Spike"]
d = ovs.groupby("Signal Date").agg(N=("R", "size"), pnl=("PnL", "sum"), risk=("Risk", "sum"), avgR=("R", "mean"))
d["rpr"] = d.pnl / d.risk
c26 = d[(d.index.year == 2026) & (d.N >= 5)]
print("\n=== OVS fill-cluster days (>= 5 OVS fills) in 2026 ===")
print(c26.round(2).to_string())
call = d[d.N >= 5]
print(f"all-history OVS cluster days (>=5 fills): {len(call)} days, avgR {call.avgR.mean():.2f}, rpr {call.pnl.sum()/call.risk.sum():.2f}, share positive {(call.pnl > 0).mean():.0%}; non-cluster fills avgR {ovs[~ovs['Signal Date'].isin(call.index)].R.mean():.2f}")
OUT["ovs_cluster_2026"] = c26.reset_index().assign(**{"Signal Date": c26.index.astype(str)}).round(4).to_dict("records")
OUT["ovs_cluster_all"] = dict(days=int(len(call)), avgR=float(call.avgR.mean()), rpr=float(call.pnl.sum() / call.risk.sum()), share_pos=float((call.pnl > 0).mean()))

json.dump(OUT, open(HERE / "gap14_midterm.json", "w"), indent=1, default=float)
print("wrote gap14_midterm.json")
