"""Survivorship bias of the ledger universe.

(A) master_prices per-ticker first/last bar: how many ledger tickers are dead
    (history preserved but no live bar), universe availability by year.
(B) ledger trades on dead-in-history tickers vs alive ones, by strategy.
(C) liquid vs overflow avgR by strategy (same rules, different tiers).
(D) collapse proxy: trades whose ticker's adjusted close falls below 50% / 30%
    of the entry price inside the next 252 sessions (the population survivorship
    removes is the tail of this one), avgR by strategy family and direction.
Writes estimation_haircut_survivorship.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    led[c] = pd.to_datetime(led[c])
res: dict = {}

# ---- (A) per-ticker span in the cache
meta = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date"]).to_pandas()
span = meta.groupby("ticker")["date"].agg(["min", "max", "size"])
last_bar = span["max"].max()
span["dead"] = span["max"] < (last_bar - pd.Timedelta(days=20))
led_tickers = set(led["Ticker"])
liquid = set(sc.LIQUID_PLUS_COMMODITIES)
csvu = set(sc.CSV_UNIVERSE)
res["cache"] = {
    "n_tickers": int(len(span)), "last_bar": str(last_bar.date()),
    "n_dead_in_cache": int(span["dead"].sum()),
    "dead_tickers": sorted(span.index[span["dead"]].tolist()),
    "n_ledger_tickers": len(led_tickers),
    "n_ledger_tickers_dead": int(span.loc[span.index.isin(led_tickers), "dead"].sum()),
    "csv_universe_n": len(csvu), "liquid_n": len(liquid),
}
# universe availability by year: how many CSV_UNIVERSE names have bars in each year
yrs = list(range(2003, 2027))
avail = {}
for y in yrs:
    avail[y] = int(((span["min"] <= pd.Timestamp(f"{y}-12-31")) & (span["max"] >= pd.Timestamp(f"{y}-01-01")) & span.index.isin(csvu)).sum())
res["csv_universe_names_with_bars_by_year"] = avail
print("cache:", {k: v for k, v in res["cache"].items() if k != "dead_tickers"})
print("dead tickers:", res["cache"]["dead_tickers"])
print("universe names with bars by year:", avail)

# ---- (B) trades on dead tickers
led["ticker_dead"] = led["Ticker"].map(span["dead"]).fillna(False).astype(bool)
led["ticker_last_bar"] = led["Ticker"].map(span["max"])
g = led.groupby("ticker_dead")["R_Multiple"].agg(["size", "mean", "std"])
print("\ntrades on dead vs alive tickers:\n", g)
res["dead_vs_alive"] = {str(k): {"N": int(v["size"]), "avgR": float(v["mean"]), "sdR": float(v["std"])} for k, v in g.iterrows()}
gd = led[led["ticker_dead"]].groupby(["Strategy", "Direction"])["R_Multiple"].agg(["size", "mean"])
print(gd)
res["dead_by_strategy"] = {f"{a}|{b}": {"N": int(v["size"]), "avgR": float(v["mean"])} for (a, b), v in gd.iterrows()}

# ---- (C) liquid vs overflow
gt = led.groupby(["Strategy", "Tier"])["R_Multiple"].agg(["size", "mean", "std"])
res["tier_split"] = {f"{a}|{b}": {"N": int(v["size"]), "avgR": float(v["mean"]), "sdR": float(v["std"])} for (a, b), v in gt.iterrows()}
print("\ntier split:\n", gt)

# ---- (D) collapse proxy on adjusted closes
tick = sorted(led_tickers)
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", tick)]).to_pandas()
px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["ticker", "date"])
# forward 252d min and forward 252d close relative to each date
px["fmin252"] = px.groupby("ticker")["Close"].transform(lambda s: s[::-1].rolling(252, min_periods=20).min()[::-1].shift(-1))
px["f252"] = px.groupby("ticker")["Close"].transform(lambda s: s.shift(-252))
px["fmin63"] = px.groupby("ticker")["Close"].transform(lambda s: s[::-1].rolling(63, min_periods=10).min()[::-1].shift(-1))
key = px.set_index(["ticker", "date"])[["Close", "fmin252", "f252", "fmin63"]]
j = led.join(key, on=["Ticker", "Entry Date"], how="left")
j["fwd_min252_ratio"] = j["fmin252"] / j["Close"]
j["fwd_min63_ratio"] = j["fmin63"] / j["Close"]
j["fwd252_ratio"] = j["f252"] / j["Close"]
cov = j["fwd_min252_ratio"].notna().mean()
print(f"\ncollapse proxy coverage: {cov:.1%} of trades have fwd-252 data")


def bucket(r):
    if pd.isna(r):
        return "na"
    if r < 0.3:
        return "<0.30"
    if r < 0.5:
        return "0.30-0.50"
    if r < 0.7:
        return "0.50-0.70"
    return ">=0.70"


j["collapse_bkt"] = j["fwd_min252_ratio"].map(bucket)
tab = j.groupby(["Direction", "collapse_bkt"])["R_Multiple"].agg(["size", "mean"])
print("\navgR by forward-252d min/entry bucket (all strategies):\n", tab)
res["collapse_proxy_by_direction"] = {f"{a}|{b}": {"N": int(v["size"]), "avgR": float(v["mean"])} for (a, b), v in tab.iterrows()}
tab2 = j.groupby(["Strategy", "collapse_bkt"])["R_Multiple"].agg(["size", "mean"])
res["collapse_proxy_by_strategy"] = {f"{a}|{b}": {"N": int(v["size"]), "avgR": float(v["mean"])} for (a, b), v in tab2.iterrows()}
print(tab2.to_string())
# 63d version too (closer to the hold horizons of the long strategies)
j["collapse63_bkt"] = j["fwd_min63_ratio"].map(bucket)
tab3 = j.groupby(["Direction", "collapse63_bkt"])["R_Multiple"].agg(["size", "mean"])
res["collapse63_proxy_by_direction"] = {f"{a}|{b}": {"N": int(v["size"]), "avgR": float(v["mean"])} for (a, b), v in tab3.iterrows()}
print("\n63d version:\n", tab3)

# share of long trades in collapse buckets and the avgR gap -> a per-unit "missing population" effect
longs = j[j["Direction"] == "Long"]
allR = longs["R_Multiple"].mean()
col = longs[longs["fwd_min252_ratio"] < 0.5]
res["long_collapse_summary"] = {"N_long": int(len(longs)), "avgR_long": float(allR),
                                "N_collapse50": int(len(col)), "avgR_collapse50": float(col["R_Multiple"].mean()) if len(col) else None,
                                "share_collapse50": float(len(col) / len(longs))}
print(res["long_collapse_summary"])
# per strategy long collapse gap
per = {}
for s, gg in longs.groupby("Strategy"):
    c = gg[gg["fwd_min252_ratio"] < 0.5]
    per[s] = {"N": int(len(gg)), "avgR": float(gg["R_Multiple"].mean()), "N_collapse50": int(len(c)),
              "avgR_collapse50": float(c["R_Multiple"].mean()) if len(c) else None, "share": float(len(c) / len(gg))}
res["long_collapse_by_strategy"] = per
print(pd.DataFrame(per).T.to_string())
(OUT / "estimation_haircut_survivorship.json").write_text(json.dumps(res, indent=1, default=str))
