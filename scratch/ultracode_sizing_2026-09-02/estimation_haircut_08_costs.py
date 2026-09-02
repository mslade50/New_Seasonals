"""Cost realism: what the ledger assumes vs what the order types actually pay.

Engine assumptions (pages/strat_backtester.py): limit entries fill AT the limit
whenever the bar touches it; target limits fill AT the target whenever touched;
time exits at the Close (live: MKT at 15:59 or MOC); stops at worse of stop/open
+3 bps, +10 bps more on a gap; NO commissions; adjusted (total-return) bars.

Measured here, per trade from master_prices bars on the entry/exit dates:
  (1) marginal entry fills: touch depth = how far the bar traded THROUGH the limit
      (long: (limit - Low)/ATR). A touch of < 0.02 ATR is a fill that live only
      gets with queue priority. Share + avgR of that cell, vs the rest.
  (2) marginal target fills: same for Target exits ((High - target)/ATR for longs).
  (3) commissions: IBKR pro tiered ~ $0.0035-0.005/sh + exchange/clearing fees,
      floor $1; modeled at $0.005/sh, floor $1, both sides -> bps of notional.
  (4) spread/impact on MKT exits: half-spread proxy by tier (liquid 1.5 bps,
      overflow 6 bps; ADV-based when available) applied to Time exits (MKT at close)
      and to EOD-DD/vol-confirm MOO exits; limit entries/targets pay none.
  (5) shorts: borrow on a 2-day hold ~ negligible for GC names; HTB overflow names
      assumed 5% ann on 5% of overflow shorts (scenario, not measured).
Converts every cost into R (cost_bps / risk_bps_per_share) per strategy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
res: dict = {}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    led[c] = pd.to_datetime(led[c])
tick = sorted(set(led["Ticker"]))
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"],
                   filters=[("ticker", "in", tick)]).to_pandas()
px["date"] = pd.to_datetime(px["date"])
px = px.set_index(["ticker", "date"]).sort_index()
ent = led.join(px[["Open", "High", "Low", "Close", "Volume"]].add_prefix("e_"), on=["Ticker", "Entry Date"])
ent = ent.join(px[["Open", "High", "Low", "Close"]].add_prefix("x_"), on=["Ticker", "Exit Date"])
# 63d ADV in dollars at entry (for spread proxy)
adv = px["Close"].mul(px["Volume"]).groupby(level=0).transform(lambda s: s.rolling(63, min_periods=20).mean())
ent = ent.join(adv.rename("adv63"), on=["Ticker", "Entry Date"])

is_long = ent["Direction"].eq("Long")
crit = ent["Entry Criteria"].astype(str)
ent["entry_kind"] = np.where(crit.str.contains("Persistent"), "limit_persistent", np.where(crit.str.contains("Limit"), "limit_open", np.where(crit.str.contains("Close"), "moc", "other")))
res["entry_kind_counts"] = ent["entry_kind"].value_counts().to_dict()
print("entry kinds:", res["entry_kind_counts"])
print("entry criteria strings:", ent["Entry Criteria"].value_counts().head(12).to_dict())

# (1) marginal entry fills: depth the bar traded through the fill price
lim = ent[ent["entry_kind"].str.startswith("limit")].copy()
lim["touch_atr"] = np.where(lim["Direction"].eq("Long"), (lim["Entry Price"] - lim["e_Low"]) / lim["ATR"], (lim["e_High"] - lim["Entry Price"]) / lim["ATR"])
lim["touch_bps"] = np.where(lim["Direction"].eq("Long"), (lim["Entry Price"] / lim["e_Low"] - 1) * 1e4, (lim["e_High"] / lim["Entry Price"] - 1) * 1e4)
# a persistent limit can fill on a day after Entry Date? Entry Date IS the fill day in the ledger, so the bar is right.
lim = lim[lim["touch_atr"].notna()]
neg = (lim["touch_atr"] < -1e-6).mean()
res["entry_touch_check"] = {"N": int(len(lim)), "share_negative_touch(fill_outside_bar)": float(neg)}
print("entry touch depth check: N", len(lim), "negative share", round(neg, 4))
lim = lim[lim["touch_atr"] >= -1e-6]
for thr_name, thr in [("<=0.02 ATR", 0.02), ("<=0.05 ATR", 0.05), ("<=10 bps", None)]:
    if thr is None:
        marg = lim["touch_bps"] <= 10
    else:
        marg = lim["touch_atr"] <= thr
    d = {"share": float(marg.mean()), "N_marginal": int(marg.sum()), "avgR_marginal": float(lim.loc[marg, "R_Multiple"].mean()),
         "avgR_rest": float(lim.loc[~marg, "R_Multiple"].mean()), "avgR_all": float(lim["R_Multiple"].mean())}
    # if marginal fills only fill half the time live, the realized avgR of the population = weighted
    d["avgR_if_marginal_fill_50pct"] = float((0.5 * marg.sum() * d["avgR_marginal"] + (~marg).sum() * d["avgR_rest"]) / (0.5 * marg.sum() + (~marg).sum()))
    d["avgR_if_marginal_fill_0pct"] = d["avgR_rest"]
    res[f"entry_marginal_{thr_name}"] = d
    print(f"entry marginal {thr_name}:", {k: round(v, 3) for k, v in d.items()})
per = {}
for s, g in lim.groupby("Strategy"):
    marg = g["touch_atr"] <= 0.02
    per[s] = {"N": int(len(g)), "share_marginal": float(marg.mean()), "avgR_marginal": float(g.loc[marg, "R_Multiple"].mean()) if marg.sum() else None,
              "avgR_rest": float(g.loc[~marg, "R_Multiple"].mean()), "avgR_all": float(g["R_Multiple"].mean()),
              "median_touch_atr": float(g["touch_atr"].median())}
res["entry_marginal_by_strategy"] = per
print(pd.DataFrame(per).T.to_string())

# (2) marginal target fills
tg = ent[ent["Exit Type"].eq("Target")].copy()
tg["ttouch_atr"] = np.where(tg["Direction"].eq("Long"), (tg["x_High"] - tg["Exit Price"]) / tg["ATR"], (tg["Exit Price"] - tg["x_Low"]) / tg["ATR"])
tg = tg[tg["ttouch_atr"].notna() & (tg["ttouch_atr"] >= -1e-6)]
marg = tg["ttouch_atr"] <= 0.02
# if a marginal target does not fill, the trade continues; approximate the give-back as the
# difference between the target R and the R the trade would have had at the next close exit:
# unknown without a replay, so report the share and a scenario (target -> time exit at that day's close)
alt_R = np.where(tg["Direction"].eq("Long"), (tg["x_Close"] - tg["Entry Price"]), (tg["Entry Price"] - tg["x_Close"])) / (tg["stop_atr"] * tg["ATR"])
tg["alt_R_close_same_day"] = alt_R
res["target_marginal"] = {"N_targets": int(len(tg)), "share_marginal_0.02atr": float(marg.mean()), "N_marginal": int(marg.sum()),
                          "R_at_target_marginal": float(tg.loc[marg, "R_Multiple"].mean()), "R_if_closed_same_day_instead": float(tg.loc[marg, "alt_R_close_same_day"].mean()),
                          "share_all_trades": float(marg.sum() / len(ent))}
print("target marginal:", res["target_marginal"])
tper = {}
for s, g in tg.groupby("Strategy"):
    mm = g["ttouch_atr"] <= 0.02
    tper[s] = {"N_targets": int(len(g)), "share_marginal": float(mm.mean()), "R_target": float(g.loc[mm, "R_Multiple"].mean()) if mm.sum() else None, "R_alt_close": float(g.loc[mm, "alt_R_close_same_day"].mean()) if mm.sum() else None}
res["target_marginal_by_strategy"] = tper

# (3)+(4) commissions and MKT-exit spread, in R
ent["risk_bps_per_share"] = ent["stop_atr"] * ent["ATR"] / ent["Entry Price"] * 1e4
sh = ent["Shares_flat"].clip(lower=1)
comm_side = np.maximum(1.0, 0.005 * sh)
ent["comm_bps"] = 2 * comm_side / (sh * ent["Entry Price"]) * 1e4
# half-spread proxy from ADV: 1.5 bps above $100M ADV, 3 bps 25-100M, 6 bps 5-25M, 12 bps below
adv_b = ent["adv63"].fillna(2e7)
ent["half_spread_bps"] = np.select([adv_b >= 1e8, adv_b >= 2.5e7, adv_b >= 5e6], [1.5, 3.0, 6.0], 12.0)
mkt_exit = ent["Exit Type"].isin(["Time", "EOD-DD"])  # MKT at close / MOC; Stop already carries 13 bps; Target = limit
ent["exit_cost_bps"] = np.where(mkt_exit, ent["half_spread_bps"], 0.0)
# a 'Stop' exit for OLV (vol-confirm) is a MOO -> pays half-spread too, but engine already charges 3 bps slip: add the excess only
ent["cost_bps_total"] = ent["comm_bps"] + ent["exit_cost_bps"]
ent["cost_R"] = ent["cost_bps_total"] / ent["risk_bps_per_share"]
summ = ent.groupby("Strategy").agg(N=("cost_R", "size"), comm_bps=("comm_bps", "mean"), exit_bps=("exit_cost_bps", "mean"), risk_bps_ps=("risk_bps_per_share", "median"),
                                   cost_R=("cost_R", "mean"), avgR=("R_Multiple", "mean"))
summ["cost_share_of_avgR"] = summ["cost_R"] / summ["avgR"]
print("\ncommission + MKT-exit cost in R by strategy:\n", summ.round(3).to_string())
res["cost_by_strategy"] = summ.round(4).to_dict("index")
book_cost_R = float(ent["cost_R"].mean())
res["book_cost_R_per_trade"] = book_cost_R
res["book_avgR"] = float(ent["R_Multiple"].mean())
res["book_cost_share"] = book_cost_R / res["book_avgR"]
# by tier
tier = ent.groupby("Tier").agg(N=("cost_R", "size"), comm_bps=("comm_bps", "mean"), exit_bps=("exit_cost_bps", "mean"), cost_R=("cost_R", "mean"), avgR=("R_Multiple", "mean"), risk_bps_ps=("risk_bps_per_share", "median"))
res["cost_by_tier"] = tier.round(4).to_dict("index")
print("\nby tier:\n", tier.round(3))
# short borrow scenario: overflow shorts, 5% of them HTB at 5% ann for hold days
sho = ent[ent["Direction"].eq("Short")]
hold = (sho["Exit Date"] - sho["Entry Date"]).dt.days.clip(lower=1)
borrow_bps = 0.05 * 0.05 * hold / 365 * 1e4  # expected: 5% names x 5% fee
res["short_borrow_scenario_R"] = float((borrow_bps / sho["risk_bps_per_share"]).mean())
print("short borrow scenario R/trade:", round(res["short_borrow_scenario_R"], 4))

# dividend basis: shorts held over an ex-date pay the dividend; adjusted bars embed it correctly
# (the short's adjusted return already includes the price drop), so no extra cost; long total-return
# accounting is correct. Flag only: ledger rows spanning an ex-div in a short (count from adjusted vs raw not available here).
res["notes"] = [
    "Engine charges NO commissions and NO spread on limit entries, target limits or time exits; only stops carry 3/13 bps.",
    "Adjusted (total-return) bars make long dividends a real gain and short dividends a real cost -- basis is correct, not optimistic.",
    "Limit fills at a bar touch overstate live fill rates: a touch of <=0.02 ATR needs queue priority; measured share and R above.",
]
(OUT / "estimation_haircut_costs.json").write_text(json.dumps(res, indent=1, default=str))
