"""seasonality_flow_6_family.py (2026-09-02): the one directionally-consistent
calendar pattern in the per-strategy tables -- the DIP-BUY family is weaker in
May-Oct in every member (each cell n.s. alone). Pooled family test with
episode-clustered and year-paired t, LOYO of the family gap, and a trade_mtm
replay of 'dip-buy family 0.75x in May-Oct' / '1.25x in Nov-Apr' (vol-matched).
Also the earnings-season book effect split by single-stock vs index strategies.
Writes seasonality_flow_family.json.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps
from seasonality_flow_common import (HERE, ROOT, NAV, load_ledger, load_trade_mtm, load_spy, trading_calendar, episodes,
                                     cluster_diff_t, year_paired_t, summarize, jdump, perf)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
DIP = ["Weak Close Decent Sznls", "Indices Oversold Bounce", "Monday Dip", "SPY QQQ MonFri Reversion", "St OS Sznl", "Monthly Weak Close", "52wh Breakout", "LT Trend ST OS"]
DIP_CORE = ["Weak Close Decent Sznls", "Indices Oversold Bounce", "Monday Dip", "SPY QQQ MonFri Reversion", "St OS Sznl", "Monthly Weak Close"]
INDEX_STRATS = ["Indices Oversold Bounce", "SPY QQQ MonFri Reversion", "Monday Dip", "Monthly Weak Close", "3x ETF Overbot Fade", "3x Bear ETF Overbot Fade", "3x Leader Gap Fade", "Sector BO"]
led = load_ledger()
key = ["Strategy", "Tier", "Ticker", "sig", "Entry Date"]
g = led.groupby(key, as_index=False).agg(pnl=("pnl", "sum"), risk=("risk", "sum"), yr=("yr", "first"), trade_ids=("trade_id", list))
g["R"] = g["pnl"] / g["risk"]
led = g
earn = pd.read_parquet(ROOT / "data/earnings_calendar.parquet", columns=["date"])
cal = trading_calendar(load_spy().index, earn)
cal = cal[cal.index >= "2003-01-01"]
led = led[led["sig"].isin(cal.index)].copy()
F = cal.loc[led["sig"]]
led["half"] = F["half"].values
led["eseason"] = F["eseason_data"].values
led["month"] = F["month"].values
led["ep"] = 0
off = 0
for s, ix in led.groupby("Strategy").indices.items():
    e = episodes(led.iloc[ix]["sig"], 5, cal.index)
    led.loc[led.index[ix], "ep"] = e + off
    off += e.max() + 1
OUT = {}


def family_test(df, mask_name, m, label):
    st_in, st_out = summarize(df[m]), summarize(df[~m])
    t_ep, p_ep, g_ep = cluster_diff_t(df["R"].values, m, df["ep"].values)
    t_yr, p_yr, n_yr = year_paired_t(df, "R", m, df["yr"].values)
    loyo = []
    for y in sorted(df["yr"].unique()):
        d = df[df.yr != y]
        mm = m[(df.yr != y).values]
        loyo.append(d[mm]["R"].mean() - d[~mm]["R"].mean())
    # per-strategy sign consistency
    signs = {}
    for s, dd in df.groupby("Strategy"):
        ms = m[(df.Strategy == s).values]
        if ms.sum() >= 5 and (~ms).sum() >= 5:
            signs[s] = float(dd[ms]["R"].mean() - dd[~ms]["R"].mean())
    res = dict(label=label, N_in=st_in["N"], N_out=st_out["N"], avgR_in=st_in["avgR"], avgR_out=st_out["avgR"], sdR_in=st_in["sdR"], sdR_out=st_out["sdR"],
               R_per_risk_in=st_in["R_per_risk"], R_per_risk_out=st_out["R_per_risk"], pnl_in=st_in["sum_pnl"], pnl_out=st_out["sum_pnl"],
               t_episode=t_ep, p_episode=p_ep, n_episodes_in=g_ep, t_year=t_yr, p_year=p_yr, n_years=n_yr,
               loyo_diff_min=float(min(loyo)), loyo_diff_max=float(max(loyo)), per_strategy_diff=signs,
               n_strats_same_sign=int(sum(1 for v in signs.values() if np.sign(v) == np.sign(st_in["avgR"] - st_out["avgR"]))), n_strats=len(signs))
    print(f"\n{label}: in N={res['N_in']} avgR {res['avgR_in']:.3f} (R/risk {res['R_per_risk_in']:.3f}) vs out N={res['N_out']} avgR {res['avgR_out']:.3f} (R/risk {res['R_per_risk_out']:.3f})"
          f"\n  t_ep={t_ep:.2f} p={p_ep:.3f} (episodes {g_ep}); t_year={t_yr:.2f} p={p_yr:.3f} (years {n_yr}); LOYO diff [{min(loyo):.3f}, {max(loyo):.3f}]; strats same sign {res['n_strats_same_sign']}/{len(signs)}")
    print("  per-strategy diff:", {k: round(v, 2) for k, v in signs.items()})
    return res


fam = led[led.Strategy.isin(DIP_CORE)]
OUT["dipbuy_core_mayoct"] = family_test(fam, "half", (fam["half"] == "MayOct").values, "DIP-BUY CORE (6 strats) May-Oct vs Nov-Apr")
fam2 = led[led.Strategy.isin(DIP)]
OUT["dipbuy_ext_mayoct"] = family_test(fam2, "half", (fam2["half"] == "MayOct").values, "DIP-BUY EXT (+52wh, LT Trend) May-Oct vs Nov-Apr")
# finer: which months carry it
mo = fam.groupby("month").apply(lambda d: pd.Series(dict(N=len(d), avgR=d.R.mean(), R_per_risk=d.pnl.sum() / d.risk.sum(), pnl=d.pnl.sum())))
print("\ndip-buy core by month:\n", mo.round(3).to_string())
OUT["dipbuy_core_by_month"] = mo.round(4).reset_index().to_dict("records")
# summer sub-windows: Jun-Sep vs May/Oct
for lab, months in [("Jun-Sep", [6, 7, 8, 9]), ("May-Oct", [5, 6, 7, 8, 9, 10]), ("Jul-Sep", [7, 8, 9]), ("Aug-Sep", [8, 9])]:
    OUT[f"dipbuy_core_{lab}"] = family_test(fam, lab, fam["month"].isin(months).values, f"DIP-BUY CORE {lab} vs rest")
# non-family control: does the rest of the book show the same? (OVS, OLV, 3x fades, ATR ext, Sector BO)
rest = led[~led.Strategy.isin(DIP)]
OUT["nonfamily_mayoct"] = family_test(rest, "half", (rest["half"] == "MayOct").values, "NON-FAMILY (OVS, OLV, fades, ATR, Sector) May-Oct vs Nov-Apr")

# earnings season split
ss = led[~led.Strategy.isin(INDEX_STRATS)]
ix = led[led.Strategy.isin(INDEX_STRATS)]
OUT["eseason_singlestock"] = family_test(ss, "eseason", ss["eseason"].values, "SINGLE-STOCK strats, earnings season vs off")
OUT["eseason_index"] = family_test(ix, "eseason", ix["eseason"].values, "INDEX/ETF strats, earnings season vs off")

# replay: dip-buy family 0.75x May-Oct; 1.25x Nov-Apr; and combined re-balance (0.8 / 1.2) -- full sample (priors, unfitted in form) + 2010+
dates, mtm = load_trade_mtm()
N_DAYS = len(dates)
def book_from(df, mults):
    out = np.zeros(N_DAYS)
    for tids, m in zip(df["trade_ids"], mults):
        for t in tids:
            s, v = mtm[t]
            out[s:s + len(v)] += v * m
    return pd.Series(out, index=dates)
base = book_from(led, np.ones(len(led)))
def cmp(alt, lo, hi):
    b = base[(base.index >= lo) & (base.index <= hi)]; a = alt[(alt.index >= lo) & (alt.index <= hi)]
    pb, pa = perf(b), perf(a)
    pv = perf(a * (pb["ann_vol_pct"] / pa["ann_vol_pct"]))
    yb, ya = b.groupby(b.index.year).sum(), a.groupby(a.index.year).sum()
    d = ya - yb
    return dict(window=[lo, hi], d_pnl_pct=(pa["total_pnl"] / pb["total_pnl"] - 1) * 100, d_sharpe=pa["sharpe"] - pb["sharpe"], d_sharpe_volmatched=pv["sharpe"] - pb["sharpe"],
                d_maxdd_pts=pa["maxdd_pct"] - pb["maxdd_pct"], d_pnl_over_maxdd_pct=(pa["pnl_over_maxdd"] / pb["pnl_over_maxdd"] - 1) * 100,
                years_better=int((d > 0).sum()), years=int(len(d)), base=pb, alt=pa)
rep = {}
isdip = led.Strategy.isin(DIP_CORE).values
for name, fn in {"dip_mayoct_0.75x": lambda h: 0.75 if h == "MayOct" else 1.0, "dip_mayoct_0.5x": lambda h: 0.5 if h == "MayOct" else 1.0,
                 "dip_novapr_1.25x": lambda h: 1.25 if h == "NovApr" else 1.0, "dip_rebalance_0.8_1.2": lambda h: 0.8 if h == "MayOct" else 1.2}.items():
    mults = np.where(isdip, [fn(h) for h in led["half"]], 1.0)
    alt = book_from(led, mults)
    rep[name] = {w: cmp(alt, *w.split("_")) for w in ["2003-01-01_2026-08-07", "2010-01-01_2026-08-07"]}
    for w, r in rep[name].items():
        print(f"{name:24s} {w[:4]}-{w[11:15]}: dPnL {r['d_pnl_pct']:6.2f}%  dSharpe {r['d_sharpe']:+.3f}  dSh_vm {r['d_sharpe_volmatched']:+.3f}  dMaxDD {r['d_maxdd_pts']:+.2f}  dPnL/DD {r['d_pnl_over_maxdd_pct']:+.1f}%  yrs+ {r['years_better']}/{r['years']}")
# walk-forward version: apply the family cut only if the family gap through Y-1 is negative (a self-gating rule)
mults = np.ones(len(led))
gate_years = []
for Y in range(2010, 2027):
    fit = fam[fam.yr < Y]
    gap = fit[fit.half == "MayOct"]["R"].mean() - fit[fit.half == "NovApr"]["R"].mean()
    on = gap < -0.1
    gate_years.append((Y, round(float(gap), 3), bool(on)))
    if on:
        m = isdip & (led.yr == Y).values & (led.half == "MayOct").values
        mults[m] = 0.75
alt = book_from(led, mults)
rep["dip_mayoct_0.75x_walkforward_gated"] = {"2010-01-01_2026-08-07": cmp(alt, "2010-01-01", "2026-08-07"), "gate": gate_years}
r = rep["dip_mayoct_0.75x_walkforward_gated"]["2010-01-01_2026-08-07"]
print(f"walk-forward gated 0.75x: dPnL {r['d_pnl_pct']:.2f}% dSharpe {r['d_sharpe']:+.3f} dSh_vm {r['d_sharpe_volmatched']:+.3f} dMaxDD {r['d_maxdd_pts']:+.2f} yrs+ {r['years_better']}/{r['years']}; gate:", gate_years)
OUT["replays"] = rep
# family sleeve daily-basis by half (Sharpe / Kelly ratio)
from seasonality_flow_common import load_strategy_daily
strat, tot = load_strategy_daily()
sleeve = strat[[c for c in DIP_CORE if c in strat.columns]].sum(axis=1) / NAV
h = np.where(sleeve.index.month.isin([11, 12, 1, 2, 3, 4]), "NovApr", "MayOct")
a, b = sleeve[h == "MayOct"], sleeve[h == "NovApr"]
OUT["dipbuy_sleeve_daily"] = dict(mayoct=dict(mean_bps=float(a.mean() * 1e4), sd_bps=float(a.std() * 1e4), sharpe=float(a.mean() / a.std() * np.sqrt(252))),
                                  novapr=dict(mean_bps=float(b.mean() * 1e4), sd_bps=float(b.std() * 1e4), sharpe=float(b.mean() / b.std() * np.sqrt(252))),
                                  kelly_ratio_mayoct_over_novapr=float((a.mean() / a.var()) / (b.mean() / b.var())))
print("\ndip-buy sleeve daily:", OUT["dipbuy_sleeve_daily"])
jdump(OUT, HERE / "seasonality_flow_family.json")
print("wrote", HERE / "seasonality_flow_family.json")
