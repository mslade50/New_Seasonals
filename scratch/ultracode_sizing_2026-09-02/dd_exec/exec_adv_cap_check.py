"""Execution-reality checks for the 2026-09-02 sizing plan (read-only).

(d) ADV participation of 2025-2026 ledger orders at GRM 1.5 and 1.875
    (x1.25) against the 21d median dollar ADV from master_prices.
(e) How often the 250 bps per-strategy cap already binds, and what the
    GRM step is worth once the cap is held fixed in effective bps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
LEDGER = ROOT / "data" / "backtest_trades_full.parquet"
PRICES = ROOT / "data" / "master_prices.parquet"
SHEETS = ROOT / "scratch" / "ultracode_sizing_2026-09-02" / "sheets_Signals.csv"

NAV = 750_000.0
CAP_BPS = 250.0
CAP_D = NAV * CAP_BPS / 1e4
STEP = 1.875 / 1.5

led = pd.read_parquet(LEDGER)
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    led[c] = pd.to_datetime(led[c])

# ---------------------------------------------------------------- (d) ADV
recent = led[led["Signal Date"] >= "2025-01-01"].copy()
tickers = sorted(recent["Ticker"].unique())
tbl = pq.read_table(PRICES, columns=["ticker", "date", "Close", "Volume"],
                    filters=[("ticker", "in", tickers), ("date", ">=", pd.Timestamp("2024-10-01"))])
px = tbl.to_pandas()
px["dv"] = px["Close"] * px["Volume"]
px = px.sort_values(["ticker", "date"])
px["adv21"] = px.groupby("ticker")["dv"].transform(lambda s: s.rolling(21, min_periods=15).median())
adv = px.set_index(["ticker", "date"])["adv21"]

# one order per (strategy, ticker, signal, entry) - OVS tranches split one fill into two rows
orders = (recent.groupby(["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date", "Direction"], as_index=False)
          .agg(shares=("Shares_flat", "sum"), entry=("Entry Price", "first")))


def adv_at(row):
    try:
        s = adv.loc[row["Ticker"]]
    except KeyError:
        return np.nan
    s = s[s.index <= row["Signal Date"]]
    return float(s.iloc[-1]) if len(s) else np.nan


orders["adv21"] = orders.apply(adv_at, axis=1)
orders["notional_15"] = orders["shares"] * orders["entry"]
orders["part_15"] = orders["notional_15"] / orders["adv21"]
orders["part_1875"] = orders["part_15"] * STEP

LEV3X = {"3x ETF Overbot Fade", "3x Bear ETF Overbot Fade", "3x Leader Gap Fade"}
orders["Class"] = np.where(orders["Strategy"].isin(LEV3X), "3x ETF", orders["Tier"])

rows = []
for (strat, cls), g in orders.groupby(["Strategy", "Class"]):
    g = g.dropna(subset=["part_15"])
    if g.empty:
        continue
    rows.append({
        "Strategy": strat, "Class": cls, "n_orders": len(g),
        "adv21_med_musd": round(g["adv21"].median() / 1e6, 1),
        "part_p50_1.875": round(g["part_1875"].median() * 100, 3),
        "part_p90_1.875": round(g["part_1875"].quantile(0.9) * 100, 2),
        "part_max_1.875": round(g["part_1875"].max() * 100, 2),
        ">1%@1.5": round((g["part_15"] > 0.01).mean() * 100, 1),
        ">1%@1.875": round((g["part_1875"] > 0.01).mean() * 100, 1),
        ">2%@1.875": round((g["part_1875"] > 0.02).mean() * 100, 1),
        ">5%@1.875": round((g["part_1875"] > 0.05).mean() * 100, 1),
        "n>5%@1.875": int((g["part_1875"] > 0.05).sum()),
    })
adv_tbl = pd.DataFrame(rows).sort_values(["Class", ">1%@1.875"], ascending=[True, False])
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n=== (d) ADV participation, 2025-01..2026-08 ledger orders, 21d median $ADV ===")
print(adv_tbl.to_string(index=False))
worst = orders.dropna(subset=["part_15"]).sort_values("part_1875", ascending=False).head(15)
print("\nTop 15 orders by participation at 1.875:")
print(worst[["Strategy", "Class", "Ticker", "Signal Date", "shares", "entry", "adv21", "part_1875"]]
      .assign(adv21=lambda d: (d["adv21"] / 1e6).round(1), part_1875=lambda d: (d["part_1875"] * 100).round(2))
      .to_string(index=False))
# 0.4% rule for sub-0.4R strategies
sub04 = {"LT Trend ST OS", "St OS Sznl", "Weak Close Decent Sznls", "Indices Oversold Bounce"}
g = orders[orders["Strategy"].isin(sub04)].dropna(subset=["part_15"])
print(f"\n0.4% rule carriers (LT Trend/St OS/WCDS/IOB): n={len(g)}, share >0.4% at 1.875 = "
      f"{(g['part_1875'] > 0.004).mean() * 100:.1f}%  (>1%: {(g['part_1875'] > 0.01).mean() * 100:.1f}%)")

# ---------------------------------------------------------------- (e) cap
led["precap"] = NAV * led["Risk bps"] / 1e4 * led["Size_Mult"]
led["realized"] = led["Risk_flat_750k"]
day = (led.groupby(["Strategy", "Signal Date"], as_index=False)
       .agg(precap=("precap", "sum"), realized=("realized", "sum"), pnl=("PnL_flat_750k", "sum"), n=("trade_id", "count")))
day["scale"] = day["realized"] / day["precap"]
day["bound_now"] = day["scale"] < 0.999
# placed risk: exact when bound (cap/scale); lower bound (filled precap sum) otherwise
day["placed_lb"] = np.where(day["bound_now"], CAP_D / day["scale"], day["precap"])
day["bound_1875_lb"] = day["placed_lb"] * STEP > CAP_D
# realized multiple of the GRM step on this strategy-day (optimistic: placed = filled precap when unbound)
day["mult_1875"] = np.minimum(STEP, CAP_D / day["placed_lb"])
day["mult_1875"] = np.where(day["bound_now"], 1.0, day["mult_1875"])  # already at the cap -> no growth


def cap_summary(d, label):
    tot_pnl = d["pnl"].sum()
    pos = d[d["pnl"] > 0]["pnl"].sum()
    out = {
        "label": label,
        "strategy_days": len(d),
        "share_days_bound_now": round(d["bound_now"].mean() * 100, 1),
        "share_days_bound_1875_lb": round(d["bound_1875_lb"].mean() * 100, 1),
        "share_pnl_on_bound_days_now": round(d.loc[d["bound_now"], "pnl"].sum() / tot_pnl * 100, 1),
        "share_pnl_on_bound_days_1875_lb": round(d.loc[d["bound_1875_lb"], "pnl"].sum() / tot_pnl * 100, 1),
        "pnl_weighted_step_multiple_upper": round((d["pnl"] * d["mult_1875"]).sum() / tot_pnl, 3),
        "note": "step multiple 1.25 = fully linear; bound-now days get 1.0",
    }
    return out


print("\n=== (e) 250 bps per-strategy cap: binding now vs at GRM 1.875 (ledger, flat basis) ===")
res_e = []
for label, d in [("2003+", day), ("2016+", day[day["Signal Date"] >= "2016-01-01"]),
                 ("2025+", day[day["Signal Date"] >= "2025-01-01"])]:
    r = cap_summary(d, label)
    res_e.append(r)
    print(r)
print("\nPer strategy, 2016+: share of strategy-days bound now / at 1.875 (lower bound), PnL share on bound days now")
d16 = day[day["Signal Date"] >= "2016-01-01"]
ps = (d16.groupby("Strategy").apply(lambda g: pd.Series({
    "days": len(g),
    "bound_now_%": round(g["bound_now"].mean() * 100, 1),
    "bound_1875_lb_%": round(g["bound_1875_lb"].mean() * 100, 1),
    "pnl_share_bound_now_%": round(g.loc[g["bound_now"], "pnl"].sum() / g["pnl"].sum() * 100, 1) if g["pnl"].sum() else np.nan,
    "step_mult_upper": round((g["pnl"] * g["mult_1875"]).sum() / g["pnl"].sum(), 3) if g["pnl"].sum() else np.nan,
    "pnl_k": round(g["pnl"].sum() / 1e3, 0),
})).sort_values("pnl_share_bound_now_%", ascending=False))
print(ps.to_string())

# Live staged (sheets) placed risk vs ledger filled, same strategy-days, 2026-03..08
try:
    sh = pd.read_csv(SHEETS)
    sh["Date"] = pd.to_datetime(sh["Date"], errors="coerce")
    sh = sh[sh["Strategy_Name"].isin(led["Strategy"].unique())]
    # one row per (strategy, ticker, date) - AM+PM bookends restage the same signal
    sh = sh.sort_values("Scan_Timestamp").drop_duplicates(["Strategy_Name", "Ticker", "Date"], keep="last")
    staged = sh.groupby(["Strategy_Name", "Date"], as_index=False)["Risk_Amt"].sum().rename(
        columns={"Strategy_Name": "Strategy", "Date": "Signal Date", "Risk_Amt": "staged"})
    staged["bound_now"] = staged["staged"] > CAP_D
    staged["bound_1875"] = staged["staged"] * STEP > CAP_D
    non_ovs = staged[staged["Strategy"] != "Overbot Vol Spike"]
    print("\n=== live STAGED risk per strategy-day (Trade_Signals_Log 2026-03-26..08-31, non-OVS; OVS is pre-gap) ===")
    print(non_ovs.groupby("Strategy").agg(days=("staged", "size"), staged_med=("staged", "median"),
                                         staged_max=("staged", "max"),
                                         bound_now=("bound_now", "sum"), bound_1875=("bound_1875", "sum")).round(0).to_string())
    m = non_ovs.merge(day[["Strategy", "Signal Date", "precap"]], on=["Strategy", "Signal Date"], how="left")
    m["fill_ratio"] = m["precap"] / m["staged"]
    print("\nfilled(ledger precap)/staged ratio by strategy (median over strategy-days with a ledger fill):")
    print(m.dropna().groupby("Strategy")["fill_ratio"].median().round(2).to_string())
except Exception as e:  # noqa: BLE001
    print("sheets check failed:", e)

json.dump({"adv": adv_tbl.to_dict(orient="records"), "cap": res_e,
           "cap_by_strategy_2016": ps.reset_index().to_dict(orient="records")},
          open(OUT / "exec_adv_cap_check.json", "w"), indent=1, default=str)
print("\nwrote", OUT / "exec_adv_cap_check.json")
