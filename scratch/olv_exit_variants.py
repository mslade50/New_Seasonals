"""Re-run the production engine on OLV-only to test shorter hold / tighter target.

Variants (entry/signal logic identical; only exit params change):
  baseline : hold 10d, tgt 2.5 ATR   (current production)
  A        : hold  5d, tgt 2.5 ATR
  B        : hold 10d, tgt 1.25 ATR
  C        : hold  5d, tgt 1.25 ATR
Stop stays 1.25 ATR, fill_window 3, day-2 stop arming throughout.

R_Multiple is sizing/cap-independent, so the R-based shape is exact even though
this isolates OLV (the live daily cap is shared across the whole book).
"""
import copy
import datetime
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)
from daily_portfolio_report import build_full_strategy_book, OVERFLOW_TICKERS, OVERFLOW_ELIGIBLE

DATA_START = datetime.date(2000, 1, 1)
BT_START = datetime.date(2003, 1, 1)

full_book = build_full_strategy_book()
olv_book = [s for s in full_book if s["name"] == "Oversold Low Volume"]
print(f"OLV passes in book: {len(olv_book)} "
      f"(bps: {[s['execution']['risk_bps'] for s in olv_book]})")

tickers = set()
for s in olv_book:
    tickers.update(s["universe_tickers"])
tickers.update(["SPY", "^VIX"])
print(f"Loading {len(tickers)} tickers ...")
md = data_provider.get_history(list(tickers), start=DATA_START.strftime("%Y-%m-%d"))

vix_df = md.get("^VIX")
vix_series = None
if vix_df is not None and not vix_df.empty:
    vd = vix_df.copy()
    if isinstance(vd.columns, pd.MultiIndex):
        vd.columns = vd.columns.get_level_values(0)
    vd.columns = [c.capitalize() for c in vd.columns]
    vix_series = vd["Close"]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
print("Precomputing indicators (cache-backed) ...")
processed = precompute_all_indicators(md, olv_book, sznl_map, vix_series, atr_sznl_map)


def run_variant(hold, tgt):
    book = copy.deepcopy(olv_book)
    for s in book:
        s["execution"]["hold_days"] = hold
        s["execution"]["tgt_atr"] = tgt
    cands, sigdata = generate_candidates_fast(processed, book, sznl_map, BT_START)
    sig = process_signals_fast(cands, sigdata, processed, book, ACCOUNT_VALUE,
                               cap_bps=250, overflow_active=True)
    # normalize raw engine columns -> analysis columns (mirrors build_trade_ledger)
    sig = sig.rename(columns={"Date": "Signal Date", "Price": "Entry Price",
                              "PnL": "PnL_c", "Risk $": "Risk_c"})
    sig["R_Multiple"] = sig["PnL_c"] / sig["Risk_c"].replace(0, np.nan)
    _of = set(OVERFLOW_TICKERS)
    sig["Tier"] = np.where(sig["Strategy"].isin(OVERFLOW_ELIGIBLE) & sig["Ticker"].isin(_of),
                           "Overflow", "Liquid")
    return sig


VARIANTS = {
    "baseline 10d/2.5": (10, 2.5),
    "A: 5d/2.5":        (5, 2.5),
    "B: 10d/1.25":      (10, 1.25),
    "C: 5d/1.25":       (5, 1.25),
}

results = {name: run_variant(h, t) for name, (h, t) in VARIANTS.items()}


def max_dd_R(df):
    d = df.sort_values("Exit Date")
    eq = d["R_Multiple"].astype(float).cumsum()
    return float((eq - eq.cummax()).min())


def stats(df):
    R = df["R_Multiple"].astype(float)
    wins, losses = R[R > 0], R[R < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() else np.inf
    # avg hold actually realized
    hold = (pd.to_datetime(df["Exit Date"]) - pd.to_datetime(df["Entry Date"])).dt.days
    exit_mix = df["Exit Type"].value_counts(normalize=True) * 100
    return {
        "N": len(df),
        "Win%": 100 * (R > 0).mean(),
        "AvgR": R.mean(),
        "MedR": R.median(),
        "TotR": R.sum(),
        "StdR": R.std(),
        "AvgR/Std": R.mean() / R.std() if R.std() else np.nan,
        "PF": pf,
        "MaxDD_R": max_dd_R(df),
        "WorstR": R.min(),
        "P95": R.quantile(0.95),
        "AvgHoldD": hold.mean(),
        "%Tgt": exit_mix.get("Target", 0.0),
        "%Stop": exit_mix.get("Stop", 0.0),
        "%Time": exit_mix.get("Time", 0.0),
    }


pd.set_option("display.width", 240, "display.float_format", lambda x: f"{x:.2f}")
rows = {name: stats(df) for name, df in results.items()}
tbl = pd.DataFrame(rows).T
print("\n=== WHOLE OLV BOOK (liquid + overflow) ===")
print(tbl.to_string())

for tier in ["Liquid", "Overflow"]:
    rows_t = {name: stats(df[df["Tier"] == tier]) for name, df in results.items()}
    tt = pd.DataFrame(rows_t).T
    print(f"\n=== {tier} tier ===")
    print(tt[["N", "Win%", "AvgR", "TotR", "AvgR/Std", "PF", "MaxDD_R", "%Tgt", "%Stop", "%Time"]].to_string())

# Year-by-year total R
print("\n=== Total R by exit year ===")
yr = {}
for name, df in results.items():
    s = df.assign(y=pd.to_datetime(df["Exit Date"]).dt.year).groupby("y")["R_Multiple"].sum()
    yr[name] = s
print(pd.DataFrame(yr).fillna(0).to_string())

# Equity curves
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
colors = {"baseline 10d/2.5": "#444", "A: 5d/2.5": "#1f77b4",
          "B: 10d/1.25": "#2ca02c", "C: 5d/1.25": "#d62728"}
for name, df in results.items():
    d = df.sort_values("Exit Date")
    eq = d["R_Multiple"].astype(float).cumsum()
    ax1.plot(pd.to_datetime(d["Exit Date"]).values, eq.values, label=name, color=colors[name], lw=1.5)
    dd = eq - eq.cummax()
    ax2.plot(pd.to_datetime(d["Exit Date"]).values, dd.values, label=name, color=colors[name], lw=1.1)
ax1.set_title("OLV cumulative R — exit-param variants"); ax1.legend(); ax1.grid(alpha=.3); ax1.set_ylabel("cumulative R")
ax2.set_title("Underwater (R below peak)"); ax2.legend(); ax2.grid(alpha=.3); ax2.set_ylabel("R")
fig.tight_layout()
out = os.path.join(ROOT, "scratch", "olv_exit_variants.png")
fig.savefig(out, dpi=110)
print(f"\nSaved -> {out}")
