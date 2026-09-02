"""Unconstrained growth, part 3: the liquidity boundary.  Per-trade
participation = position notional / trailing-20d dollar ADV of the ticker at
entry (master_prices Volume x Close, read with column + ticker filters), scaled
by the multiple m of current sizing; square-root impact cost
  cost = k * sigma_daily * sqrt(participation), charged twice (entry + exit)
and the growth-relevant drag c(m) it implies for the book.  Writes
unconstrained_growth_03_liquidity.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
NAV = 750_000.0
GRM_NOW = 1.5
M_GRID = np.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
K_IMPACT = 1.0      # sqrt-impact coefficient (Almgren 2005 ~0.6-0.9 temporary+permanent; 1.0 is a conservative practitioner default)
OUT: dict = {"m_grid": M_GRID.tolist(), "k_impact": K_IMPACT}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["notional"] = led["Entry Price"] * led["Shares_flat"]
alias = {"^GSPC": "SPY", "^NDX": "QQQ"}
led["px_ticker"] = led["Ticker"].map(lambda t: alias.get(t, t))
tickers = sorted(led["px_ticker"].unique())
print(f"{len(led)} trades, {len(tickers)} tickers")

tbl = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close", "Volume"],
                    filters=[("ticker", "in", tickers)])
px = tbl.to_pandas(strings_to_categorical=True)
px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["ticker", "date"])
px["dv"] = px["Close"] * px["Volume"]
g = px.groupby("ticker", observed=True)
px["adv20"] = g["dv"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
px["sig20"] = g["Close"].transform(lambda s: s.pct_change().rolling(20, min_periods=10).std().shift(1))
px["ticker"] = px["ticker"].astype(str)
key = px.set_index(["ticker", "date"])[["adv20", "sig20"]]
led = led.join(key, on=["px_ticker", "Entry Date"])
miss = led["adv20"].isna().mean()
print(f"ADV lookup missing for {miss:.1%} of trades (index proxies aside)")
led = led.dropna(subset=["adv20", "sig20"])
led = led[led["adv20"] > 0]
led["part"] = led["notional"] / led["adv20"]
led["cost1"] = 2 * K_IMPACT * led["sig20"] * np.sqrt(led["part"]) * led["notional"]   # round-trip $ cost at m=1
led["cost_R1"] = led["cost1"] / led["Risk_flat_750k"]                                   # in R units at m=1
led["yr"] = led["Exit Date"].dt.year

# ------------------------------------------------------------ 1. participation distribution per strategy / tier at m=1
print("\n=== 1. participation of position notional vs 20d $ADV at m=1 (median / p90 / p99 / max, %) ===")
rows = []
for (s, t), d in led.groupby(["Strategy", "Tier"]):
    rows.append(dict(strategy=s, tier=t, n=len(d), part_med=d["part"].median(), part_p90=d["part"].quantile(.9), part_p99=d["part"].quantile(.99), part_max=d["part"].max(),
                     adv_med_musd=d["adv20"].median() / 1e6, cost_R_med=d["cost_R1"].median(), cost_R_mean=d["cost_R1"].mean(), avgR=d["R_Multiple"].mean(),
                     share_over_1pct=(d["part"] > .01).mean(), share_over_5pct=(d["part"] > .05).mean()))
P1 = pd.DataFrame(rows).sort_values("part_p90", ascending=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.4f}".format)
print(P1.to_string(index=False))
OUT["participation_at_1x"] = P1.round(5).to_dict("records")

# ------------------------------------------------------------ 2. impact-adjusted edge by m, per strategy and book (2016+)
print("\n=== 2. impact drag vs m: net avgR = avgR - cost_R1*sqrt(m) (cost per $ risk grows as sqrt(m)); book drag in % NAV / yr ===")
W = led[led["Exit Date"] >= "2016-01-01"]
years = (W["Exit Date"].max() - W["Exit Date"].min()).days / 365.25
OUT["impact_by_m"] = {}
for m in M_GRID:
    part_m = W["part"] * m
    cost_m = 2 * K_IMPACT * W["sig20"] * np.sqrt(part_m) * W["notional"] * m      # $ round trip at m
    gross_m = W["PnL_flat_750k"] * m
    drag_pct_nav_yr = cost_m.sum() / years / NAV
    gross_pct_nav_yr = gross_m.sum() / years / NAV
    per_strat = {}
    for s, d in W.groupby("Strategy"):
        cm = 2 * K_IMPACT * d["sig20"] * np.sqrt(d["part"] * m) * d["notional"] * m
        per_strat[s] = dict(net_avgR=float(((d["PnL_flat_750k"] * m - cm) / (d["Risk_flat_750k"] * m)).mean()), gross_avgR=float(d["R_Multiple"].mean()),
                            share_part_over_5=float((d["part"] * m > .05).mean()), share_part_over_25=float((d["part"] * m > .25).mean()))
    OUT["impact_by_m"][f"{m:g}"] = dict(drag_pct_nav_yr=float(drag_pct_nav_yr), gross_pct_nav_yr=float(gross_pct_nav_yr), net_pct_nav_yr=float(gross_pct_nav_yr - drag_pct_nav_yr),
                                        drag_share_of_gross=float(drag_pct_nav_yr / gross_pct_nav_yr), share_trades_part_over_5=float((part_m > .05).mean()),
                                        share_trades_part_over_25=float((part_m > .25).mean()), share_trades_part_over_100=float((part_m > 1).mean()), per_strategy=per_strat)
    print(f"m={m:4g}: gross {gross_pct_nav_yr:6.1%}/yr drag {drag_pct_nav_yr:6.2%}/yr ({drag_pct_nav_yr/gross_pct_nav_yr:5.1%} of gross) | trades >5% ADV {(part_m>.05).mean():5.1%}, >25% {(part_m>.25).mean():5.1%}, >100% {(part_m>1).mean():4.1%}")
# per-strategy m at which net avgR hits zero: avgR = mean(cost_R1)*sqrt(m) -> m0 = (avgR / mean cost_R1)^2
print("\nper-strategy m at which sqrt-impact eats the whole edge (2016+):")
z = {}
for s, d in W.groupby("Strategy"):
    c = d["cost_R1"].mean(); a = d["R_Multiple"].mean()
    m0 = (a / c) ** 2 if c > 0 and a > 0 else np.inf
    z[s] = dict(avgR=float(a), cost_R_at_1x=float(c), m_zero_edge=float(m0), grm_zero_edge=float(m0 * GRM_NOW), n=int(len(d)),
                m_half_edge=float((a / 2 / c) ** 2) if c > 0 and a > 0 else None)
    print(f"  {s:28s} avgR {a:5.2f} cost_R(1x) {c:5.3f} -> edge halves at m={z[s]['m_half_edge']:.1f}, zero at m={m0:.1f}")
OUT["m_zero_edge_by_strategy"] = z
book_a = W["R_Multiple"].mean(); book_c = W["cost_R1"].mean()
OUT["book_m_zero_edge"] = dict(avgR=float(book_a), cost_R_at_1x=float(book_c), m_half_edge=float((book_a / 2 / book_c) ** 2), m_zero_edge=float((book_a / book_c) ** 2))
print(f"  BOOK: avgR {book_a:.2f} cost_R(1x) {book_c:.3f} -> half edge at m={(book_a/2/book_c)**2:.1f}, zero at m={(book_a/book_c)**2:.1f}")

# ------------------------------------------------------------ 3. sensitivity to k and to a 10% ADV hard participation limit (fills capped)
print("\n=== 3. sensitivity: k=0.5 and k=1.5; and a 10%-of-ADV participation cap (position clipped, edge per $ kept) ===")
OUT["sensitivity"] = {}
for k in [0.5, 1.0, 1.5]:
    d = {}
    for m in [1, 2, 3, 5, 8]:
        cost_m = 2 * k * W["sig20"] * np.sqrt(W["part"] * m) * W["notional"] * m
        d[f"{m:g}"] = float(cost_m.sum() / years / NAV)
    OUT["sensitivity"][f"k{k:g}_drag_pct_nav_yr"] = d
    print(f"  k={k}: drag %NAV/yr by m:", {kk: f"{v:.2%}" for kk, v in d.items()})
capd = {}
for m in M_GRID:
    clip = np.minimum(1.0, 0.10 / (W["part"] * m))          # fraction of the intended position that fits under 10% ADV
    eff = float(np.average(clip, weights=W["Risk_flat_750k"]))
    capd[f"{m:g}"] = dict(effective_mult=float(m * eff), share_clipped=float((clip < 1).mean()), risk_weighted_fill=eff)
OUT["adv_cap_10pct"] = capd
print("  10% ADV cap: effective multiple realised:", {k: f"{v['effective_mult']:.2f}" for k, v in capd.items()})

json.dump(OUT, open(HERE / "unconstrained_growth_03_liquidity.json", "w"), indent=1, default=float)
print("\nwrote", HERE / "unconstrained_growth_03_liquidity.json")
