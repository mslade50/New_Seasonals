"""Cycle/macro regime conditioning, part 5: single-strategy walk-forward for
the candidate cells (does a per-strategy regime multiplier survive when it
is fit on the strategy's own trades before year Y and applied to year Y?),
plus signal FLOW by regime (trades per 252 regime-days) per strategy, and
book open-risk by regime. Writes cycle_macro_05_single_wf.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_macro_lib import HERE, NAV, REGIME_COLS, attach_trade_regimes, build_regimes, jsonable, load_ledger

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
led = load_ledger()
R = build_regimes()
T = attach_trade_regimes(led, R)
T["exit_d"] = T["Exit Date"]
pit = pd.read_parquet(HERE / "cycle_macro_pit_dial.parquet")["pit_dial"]
T["pit_dial"] = pit.reindex(T["Signal Date"]).values
T["pit_bucket"] = np.where(T["pit_dial"].isna(), "nan", np.where(T["pit_dial"] >= 50, ">=50", "<50"))

N0, LO, HI = 30, 0.5, 1.5
def fit_one(tr, axis):
    mu, va = tr["R_Multiple"].mean(), tr["R_Multiple"].var()
    tab = {}
    if mu <= 0 or len(tr) < 25:
        return tab
    for b, gb in tr.groupby(axis):
        if b == "nan" or len(gb) < 6 or gb["R_Multiple"].var() == 0:
            continue
        k = (gb["R_Multiple"].mean() / gb["R_Multiple"].var()) / (mu / va)
        tab[b] = float(np.clip((len(gb) * k + N0) / (len(gb) + N0), LO, HI))
    return tab

def single_wf(strategy, axis, start=2010):
    d = T[(T["Strat2"] == strategy) & (T[axis] != "nan")].copy()
    mults, parts = [], []
    for Y in range(start, 2027):
        tr, te = d[d["yr"] < Y], d[d["yr"] == Y]
        if len(te) == 0:
            continue
        tab = fit_one(tr, axis)
        mults.append(pd.Series([tab.get(b, 1.0) for b in te[axis]], index=te.index)); parts.append(te)
    if not parts:
        return None
    mult = pd.concat(mults); e = pd.concat(parts)
    pnl, risk = e["PnL_flat_750k"] * mult, e["Risk_flat_750k"] * mult
    bp, br = e["PnL_flat_750k"], e["Risk_flat_750k"]
    yb = yn = 0; ylist = []
    for y, g in e.groupby("yr"):
        mm = mult[g.index]
        if (mm != 1).sum() == 0 or g["Risk_flat_750k"].sum() == 0:
            continue
        a = (g["PnL_flat_750k"] * mm).sum() / (g["Risk_flat_750k"] * mm).sum(); b = g["PnL_flat_750k"].sum() / g["Risk_flat_750k"].sum()
        yn += 1; yb += int(a > b); ylist.append(dict(year=int(y), ppr=float(a), base_ppr=float(b), n=int(len(g))))
    # strategy-only realized daily series
    idx = pd.bdate_range(e["exit_d"].min(), e["exit_d"].max())
    daily = pnl.groupby(e["exit_d"]).sum().reindex(idx).fillna(0) / NAV; bdaily = bp.groupby(e["exit_d"]).sum().reindex(idx).fillna(0) / NAV
    scale = br.sum() / risk.sum()
    def sh(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan
    def mdd(x):
        eq = x.cumsum(); return float((eq - eq.cummax()).min() * 100)
    return dict(strategy=strategy, axis=axis, N=int(len(e)), years=yn, years_ppr_better=yb, base_ppr=float(bp.sum() / br.sum()), ppr=float(pnl.sum() / risk.sum()),
                d_ppr_pct=float((pnl.sum() / risk.sum()) / (bp.sum() / br.sum()) - 1) * 100, risk_ratio=float(risk.sum() / br.sum()),
                d_pnl_rm=float(pnl.sum() * scale - bp.sum()), base_sharpe=sh(bdaily), sharpe_rm=sh(daily * scale), base_maxdd=mdd(bdaily), maxdd_rm=mdd(daily * scale),
                mean_mult=float(mult.mean()), share_scaled=float((mult != 1).mean()), yearly=ylist)

pairs = [("Oversold Low Volume", "spy_dd"), ("Oversold Low Volume", "vix_lvl"), ("Oversold Low Volume", "rv21"), ("Oversold Low Volume", "cycle"),
         ("52wh Breakout", "rv21"), ("52wh Breakout", "credit21"), ("52wh Breakout", "spy_dd"), ("52wh Breakout", "vix_lvl"),
         ("Indices Oversold Bounce", "credit21"), ("Indices Oversold Bounce", "vix_ts"), ("Indices Oversold Bounce", "vix_lvl"),
         ("OVS path 1", "spy_dd"), ("OVS path 1", "vix_lvl"), ("OVS path 1", "rv21"), ("OVS path 1", "cycle"), ("OVS path 1", "tnx_chg63"),
         ("LT Trend ST OS", "cycle"), ("LT Trend ST OS", "rv21"), ("LT Trend ST OS", "pit_bucket"),
         ("SPY QQQ MonFri Reversion", "cycle"), ("SPY QQQ MonFri Reversion", "rv21"), ("SPY QQQ MonFri Reversion", "credit21"),
         ("Weak Close Decent Sznls", "credit21"), ("Weak Close Decent Sznls", "rv21"), ("Monday Dip", "rv21"),
         ("ATR Extended Gap Up", "rv21"), ("ATR Extended Gap Up", "vix_lvl"), ("3x ETF Overbot Fade", "vix_lvl"), ("3x ETF Overbot Fade", "cycle"),
         ("Sector BO", "spy_dd"), ("St OS Sznl", "vix_ts")]
rows = []
for s, a in pairs:
    r = single_wf(s, a, start=2019 if a == "pit_bucket" else 2010)
    if r:
        rows.append(r)
SW = pd.DataFrame(rows)
print("=== single-strategy walk-forward (fit on own trades < Y, apply Y; shrink N0=30, clip [0.5,1.5]) ===")
print(SW[["strategy", "axis", "N", "years", "years_ppr_better", "base_ppr", "ppr", "d_ppr_pct", "risk_ratio", "d_pnl_rm", "base_sharpe", "sharpe_rm", "base_maxdd", "maxdd_rm", "share_scaled"]].round(3).to_string(index=False))
OUT["single_wf"] = jsonable(SW.to_dict("records"))

# ---------------------------------------------------------------- signal flow by regime
print("\n=== signal flow: trades per 252 regime-days, per strategy (2005+) ===")
Rx = R[R.index >= "2005-01-01"]
Tx = T[T["Signal Date"] >= "2005-01-01"]
flow = {}
for col in ["cycle", "vix_lvl", "rv21", "spy_dd", "vix_ts", "credit21", "pc_fear", "dial"]:
    days = Rx[col].value_counts()
    cnt = Tx.groupby(["Strat2", col]).size().unstack().fillna(0)
    cnt = cnt[[c for c in cnt.columns if c != "nan"]]
    per = cnt.div(days.reindex(cnt.columns), axis=1) * 252
    per.loc["BOOK"] = per.sum()
    flow[col] = per.round(1)
    print(f"-- {col} --\n{per.round(1).to_string()}")
OUT["signal_flow"] = jsonable({k: v.reset_index().to_dict("records") for k, v in flow.items()})
# open risk (bps) by regime day: sum of Risk_flat over open trades, lag-1 regime
idx = pd.bdate_range("2005-01-01", "2026-09-01")
open_risk = pd.Series(0.0, index=idx)
for a, b, r in zip(Tx["Entry Date"], Tx["Exit Date"], Tx["Risk_flat_750k"]):
    open_risk[(idx >= a) & (idx <= b)] += r
orr = pd.DataFrame({"open_bps": open_risk / NAV * 1e4}).join(R.shift(1)[["vix_lvl", "rv21", "spy_dd", "cycle", "dial"]])
orisk = {}
for col in ["vix_lvl", "rv21", "spy_dd", "cycle", "dial"]:
    g = orr.groupby(col)["open_bps"].agg(["mean", "median", lambda s: s.quantile(.9)])
    g.columns = ["mean", "median", "p90"]; g = g.drop("nan", errors="ignore")
    orisk[col] = g.round(1); print(f"-- open risk bps by {col} --\n{g.round(1).to_string()}")
OUT["open_risk_by_regime"] = jsonable({k: v.reset_index().to_dict("records") for k, v in orisk.items()})

json.dump(jsonable(OUT), open(HERE / "cycle_macro_05_single_wf.json", "w"), indent=1)
print("\nwrote", HERE / "cycle_macro_05_single_wf.json")
