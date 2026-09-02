"""Cycle/macro regime conditioning, part 4: (A) fixed candidate rules replayed
on the ledger (realized-at-exit daily series, raw and risk-matched, LOYO
sign years); (B) mechanism families (mean-reversion vs momentum) pooled by
vol/drawdown regime with year-sign counts and a family-level walk-forward;
(C) LT Trend ST OS at PIT dial >= 50: LOYO + episode table (the one dial
cell that met the plan's P2 gate in part 3).
Writes cycle_macro_04_rules.json beside this file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_macro_lib import (HERE, NAV, attach_trade_regimes, build_regimes, cluster_t, episode_ids, jsonable, load_ledger, loyo, welch_t)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
led = load_ledger()
R = build_regimes()
T = attach_trade_regimes(led, R)
T["exit_d"] = T["Exit Date"]
pit = pd.read_parquet(HERE / "cycle_macro_pit_dial.parquet")["pit_dial"]
T["pit_dial"] = pit.reindex(T["Signal Date"]).values

MR = ["Oversold Low Volume", "OVS path 1", "OVS path 2", "Indices Oversold Bounce", "SPY QQQ MonFri Reversion", "Monday Dip",
      "Weak Close Decent Sznls", "LT Trend ST OS", "St OS Sznl", "Monthly Weak Close", "3x ETF Overbot Fade", "3x Bear ETF Overbot Fade", "3x Leader Gap Fade", "ATR Extended Gap Up"]
MOM = ["52wh Breakout", "Sector BO"]
T["family"] = np.where(T["Strat2"].isin(MOM), "momentum", "mean_reversion")

# ---------------------------------------------------------------- A. fixed rules
def st(x):
    eq = x.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(ann=x.mean() * 252 * 100, vol=x.std() * np.sqrt(252) * 100, sharpe=x.mean() / x.std() * np.sqrt(252), maxdd=dd * 100, worst=x.min() * 100)

def replay(mult: pd.Series, d: pd.DataFrame, since="2003-01-01") -> dict:
    d = d[d["Signal Date"] >= since]; mult = mult[d.index]
    pnl, risk = d["PnL_flat_750k"] * mult, d["Risk_flat_750k"] * mult
    bp, br = d["PnL_flat_750k"], d["Risk_flat_750k"]
    idx = pd.bdate_range(d["exit_d"].min(), d["exit_d"].max())
    daily = pnl.groupby(d["exit_d"]).sum().reindex(idx).fillna(0) / NAV
    bdaily = bp.groupby(d["exit_d"]).sum().reindex(idx).fillna(0) / NAV
    scale = br.sum() / risk.sum()
    a, b, rm = st(daily), st(bdaily), st(daily * scale)
    yb, yn = 0, 0
    for y, g in d.groupby("yr"):
        mm = mult[g.index]
        if (mm != 1).sum() == 0:
            continue
        yn += 1; yb += int((g["PnL_flat_750k"] * mm).sum() / (g["Risk_flat_750k"] * mm).sum() > g["PnL_flat_750k"].sum() / g["Risk_flat_750k"].sum())
    return dict(d_pnl=float(pnl.sum() - bp.sum()), d_pnl_riskmatched=float(pnl.sum() * scale - bp.sum()), risk_ratio=float(risk.sum() / br.sum()),
                base_ppr=float(bp.sum() / br.sum()), ppr=float(pnl.sum() / risk.sum()), base_sharpe=b["sharpe"], sharpe_raw=a["sharpe"], sharpe_rm=rm["sharpe"],
                base_maxdd=b["maxdd"], maxdd_raw=a["maxdd"], maxdd_rm=rm["maxdd"], base_vol=b["vol"], vol_raw=a["vol"], base_worst=b["worst"], worst_rm=rm["worst"],
                years_touched=yn, years_ppr_better=yb, trades_touched=int((mult != 1).sum()))

ones = pd.Series(1.0, index=T.index)
def rule(name, mask, m, strategies=None, since="2003-01-01"):
    mult = ones.copy()
    sel = mask if strategies is None else (mask & T["Strat2"].isin(strategies))
    mult[sel] = m
    r = replay(mult, T, since); r["rule"] = name; r["n_trades"] = int(sel.sum()); r["cell_avgR"] = float(T.loc[sel, "R_Multiple"].mean())
    return r
rules = [
    rule("midterm 0.75x, whole book ex-OVS (OVS already 0.75x)", T["cycle"] == "midterm", 0.75, [s for s in T["Strat2"].unique() if not s.startswith("OVS")]),
    rule("midterm 0.75x, whole book incl OVS (stacks on live tilt)", T["cycle"] == "midterm", 0.75),
    rule("midterm 0.5x, whole book ex-OVS", T["cycle"] == "midterm", 0.5, [s for s in T["Strat2"].unique() if not s.startswith("OVS")]),
    rule("post-election 1.25x, whole book", T["cycle"] == "post_election", 1.25),
    rule("OLV 0.5x when VIX < 15", T["vix_lvl"] == "<15", 0.5, ["Oversold Low Volume"]),
    rule("OLV 0.5x when SPY within 3% of 252d high", T["spy_dd"] == "<3", 0.5, ["Oversold Low Volume"]),
    rule("OLV 1.5x when SPY 3-10% off high", T["spy_dd"] == "3-10", 1.5, ["Oversold Low Volume"]),
    rule("52wh 0.5x when HYG/LQD widening (21d < -1.5%)", T["credit21"] == "widening", 0.5, ["52wh Breakout"]),
    rule("52wh 0.75x when SPY rv21 < 12", T["rv21"] == "<12", 0.75, ["52wh Breakout"]),
    rule("52wh 0.5x when SPY rv21 < 12", T["rv21"] == "<12", 0.5, ["52wh Breakout"]),
    rule("IOB 0.5x when HYG/LQD widening", T["credit21"] == "widening", 0.5, ["Indices Oversold Bounce"]),
    rule("ATR Ext Gap Up 0.5x when rv21 < 12", T["rv21"] == "<12", 0.5, ["ATR Extended Gap Up"]),
    rule("book 0.8x when SPY within 3% of high", T["spy_dd"] == "<3", 0.8),
    rule("book 1.25x when SPY 3-10% off high", T["spy_dd"] == "3-10", 1.25),
    rule("book 0.8x at <3 AND 1.25x at 3-10", T["spy_dd"] == "<3", 0.8),  # placeholder, replaced below
    rule("book 0.75x when VIX < 15", T["vix_lvl"] == "<15", 0.75),
    rule("book 1.25x when VIX >= 20", T["vix_lvl"].isin(["20-30", "30+"]), 1.25),
    rule("book 0.75x when rv21 < 12", T["rv21"] == "<12", 0.75),
    rule("MR family 0.75x when rv21 < 12", T["rv21"] == "<12", 0.75, MR),
    rule("MR family 1.25x when rv21 20-30", T["rv21"] == "20-30", 1.25, MR),
    rule("MR family 0.8x at spy_dd<3", T["spy_dd"] == "<3", 0.8, MR),
    rule("MOM family 0.5x when credit widening", T["credit21"] == "widening", 0.5, MOM),
    rule("LT Trend ST OS 0.5x at PIT dial >= 50 (2018+)", T["pit_dial"] >= 50, 0.5, ["LT Trend ST OS"], since="2018-01-01"),
    rule("LT Trend ST OS 0.5x at current dial >= 65 (2016+)", T["dial_val"] >= 65, 0.5, ["LT Trend ST OS"], since="2016-07-20"),
]
# two-sided dd rule
mult = ones.copy(); mult[T["spy_dd"] == "<3"] = 0.8; mult[T["spy_dd"] == "3-10"] = 1.25
r = replay(mult, T); r["rule"] = "book 0.8x at <3 AND 1.25x at 3-10"; r["n_trades"] = int((mult != 1).sum()); r["cell_avgR"] = np.nan
rules = [x for x in rules if x["rule"] != "book 0.8x at <3 AND 1.25x at 3-10"] + [r]
RU = pd.DataFrame(rules)
cols = ["rule", "n_trades", "cell_avgR", "d_pnl", "d_pnl_riskmatched", "risk_ratio", "base_ppr", "ppr", "base_sharpe", "sharpe_raw", "sharpe_rm", "base_maxdd", "maxdd_raw", "maxdd_rm", "years_touched", "years_ppr_better"]
print("=== A. fixed rules on the ledger (realized-at-exit daily; rm = risk-matched to baseline risk) ===")
print(RU[cols].round(3).to_string(index=False))
OUT["fixed_rules"] = jsonable(RU.round(4).to_dict("records"))

# ---------------------------------------------------------------- B. mechanism families by regime
print("\n=== B. mechanism families by vol/drawdown regime (trade level, episode-clustered, LOYO year signs) ===")
frows = []
for fam, d in T.groupby("family"):
    for col in ["rv21", "vix_lvl", "spy_dd", "vix_ts", "credit21", "cycle", "mom12_1", "spy_200"]:
        dd = d[d[col] != "nan"]
        for b in sorted(dd[col].unique()):
            m = dd[col] == b
            if m.sum() < 15 or (~m).sum() < 15:
                continue
            ep = episode_ids(R[col] == b, gap=21).reindex(dd["Signal Date"]).values
            cl = np.where(m.values, "E" + pd.Series(ep).astype(str), "M" + dd["ym"].values)
            beta, t, G = cluster_t(dd["R_Multiple"].values, m.values.astype(float), cl)
            lo = loyo(dd, m)
            g = dd[m]
            frows.append(dict(family=fam, regime=col, bucket=b, N=int(m.sum()), avgR=float(g["R_Multiple"].mean()), avgR_rest=float(dd.loc[~m, "R_Multiple"].mean()),
                              sdR=float(g["R_Multiple"].std()), sd_ratio=float(g["R_Multiple"].std() / dd["R_Multiple"].std()), diff=beta, t_cluster=t, n_clusters=G,
                              ppr=float(g["PnL_flat_750k"].sum() / g["Risk_flat_750k"].sum()), **lo))
FB = pd.DataFrame(frows)
print(FB[["family", "regime", "bucket", "N", "avgR", "avgR_rest", "sd_ratio", "diff", "t_cluster", "n_clusters", "yr_pos", "yr_neg", "loyo_min", "loyo_max"]].round(3).to_string(index=False))
OUT["family_regime"] = jsonable(FB.round(4).to_dict("records"))

# family-level walk-forward (table fit per family, shrunk N0=50, clip [0.5, 1.5])
print("\n=== family-level walk-forward, 2010-2026 ===")
N0, LO, HI = 50, 0.5, 1.5
def fit_family(tr, axis):
    tab = {}
    for f, g in tr.groupby("family"):
        mu, va = g["R_Multiple"].mean(), g["R_Multiple"].var()
        if mu <= 0 or len(g) < 30:
            continue
        for b, gb in g.groupby(axis):
            if b == "nan" or len(gb) < 8:
                continue
            k = (gb["R_Multiple"].mean() / gb["R_Multiple"].var()) / (mu / va)
            tab[(f, b)] = float(np.clip((len(gb) * k + N0) / (len(gb) + N0), LO, HI))
    return tab
wf = []
for axis in ["rv21", "vix_lvl", "spy_dd", "vix_ts", "credit21", "cycle"]:
    mults, parts = [], []
    for Y in range(2010, 2027):
        tr = T[(T["yr"] < Y) & (T[axis] != "nan")]; te = T[T["yr"] == Y]
        tab = fit_family(tr, axis)
        mults.append(pd.Series([tab.get((f, b), 1.0) for f, b in zip(te["family"], te[axis])], index=te.index)); parts.append(te)
    mult = pd.concat(mults); d = pd.concat(parts)
    r = replay(mult, d); r["axis"] = axis; wf.append(r)
WFF = pd.DataFrame(wf)
print(WFF[["axis", "trades_touched", "d_pnl_riskmatched", "risk_ratio", "base_ppr", "ppr", "base_sharpe", "sharpe_rm", "base_maxdd", "maxdd_rm", "years_touched", "years_ppr_better"]].round(3).to_string(index=False))
OUT["family_walk_forward"] = jsonable(WFF.round(4).to_dict("records"))

# ---------------------------------------------------------------- C. LT Trend ST OS at PIT dial >= 50
print("\n=== C. LT Trend ST OS, PIT dial >= 50 (2018+): LOYO + per-episode table ===")
lt = T[(T["Strat2"] == "LT Trend ST OS") & (T["yr"] >= 2018) & T["pit_dial"].notna()].copy()
m = lt["pit_dial"] >= 50
lo = loyo(lt, m)
ep = episode_ids(pit >= 50, gap=21).reindex(lt["Signal Date"]).values
lt["ep"] = np.where(m.values, ep, -1)
epi = lt[m].groupby("ep").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), first=("Signal Date", "min"), last=("Signal Date", "max"), pnl=("PnL_flat_750k", "sum"))
print("LOYO:", lo); print(epi.to_string())
by_year = lt.groupby(["yr", m.rename("hi")])["R_Multiple"].agg(["mean", "size"]).unstack()
print(by_year.round(2).to_string())
# same cell on the daily basis (active-day Sharpe) for the strategy
OUT["lt_trend_pit50"] = jsonable(dict(N_hi=int(m.sum()), N_lo=int((~m).sum()), avgR_hi=float(lt.loc[m, "R_Multiple"].mean()), avgR_lo=float(lt.loc[~m, "R_Multiple"].mean()),
                                      loyo=lo, episodes=epi.reset_index().to_dict("records"), by_year=by_year.round(3).reset_index().values.tolist()))
# and the same strategy's pre-2018 trades at current-weights dial >= 50 (2016-17, recompute vintage) for completeness
lt2 = T[(T["Strat2"] == "LT Trend ST OS") & (T["yr"].between(2016, 2017)) & T["dial_val"].notna()]
print("2016-17 (current weights, recompute vintage): hi N", int((lt2.dial_val >= 50).sum()), "avgR", round(float(lt2.loc[lt2.dial_val >= 50, "R_Multiple"].mean()), 3) if (lt2.dial_val >= 50).sum() else None,
      "| lo N", int((lt2.dial_val < 50).sum()), "avgR", round(float(lt2.loc[lt2.dial_val < 50, "R_Multiple"].mean()), 3))

json.dump(jsonable(OUT), open(HERE / "cycle_macro_04_rules.json", "w"), indent=1)
print("\nwrote", HERE / "cycle_macro_04_rules.json")
