"""OVS hold-extension counterfactual (2026-07-11).

Question: what if OVS trades that are LOSING at the T+2 time exit were held
to T+5 instead (target still live during the extension, no stop -- OVS has
none)? Three variants:
  1. Apply to every losing T+2 time exit
  2. Only when the signal-date 252d rank is on the LOW side of the barbell (<65)
  3. Only on low-vol names (ATR / signal close < 3%)

Method: replay off the full ledger (data/backtest_trades_full.parquet).
Affected set = Exit Type 'Time' with R_Multiple < 0 (EOD-DD and Target exits
never reach a losing T+2 close). For each, walk the 3 bars after the original
exit on the ticker's own calendar: short target (entry - tgt_atr*ATR) fills at
target if Low <= tgt (engine convention, strat_backtester ~line 2100), else
exit at the T+5 close. delta_R = (T2_close - new_exit) / ATR (all OVS short).
Both sides of the delta use the CURRENT price cache so recent-window
re-adjustment can't leak in as fake PnL; trades whose cache T+2 close drifted
>1% from the ledger exit price are dropped and counted.

252d rank recomputed per indicators.py: expanding pct-rank of 252d return,
min_periods=252, valued at signal date.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
ovs = ledger[ledger["Strategy"] == "Overbot Vol Spike"].copy()
assert (ovs["Direction"] == "Short").all(), "OVS should be all short"
for c in ("Signal Date", "Entry Date", "Exit Date"):
    ovs[c] = pd.to_datetime(ovs[c])

print(f"OVS trades: {len(ovs)}  |  exit types: {ovs['Exit Type'].value_counts().to_dict()}")

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
tickers = ovs["Ticker"].unique()
prices = prices[prices["ticker"].isin(tickers)]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

# ---- 252d rank at signal date (indicators.py convention) ----
rank252 = {}
for t, df in px.items():
    ret = df["Close"].pct_change(252, fill_method=None)
    rank252[t] = ret.expanding(min_periods=252).rank(pct=True) * 100.0

def rank_at(t: str, d: pd.Timestamp) -> float:
    s = rank252.get(t)
    if s is None or d not in s.index:
        return np.nan
    return s.loc[d]

ovs["rank252"] = [rank_at(t, d) for t, d in zip(ovs["Ticker"], ovs["Signal Date"])]
ovs["atr_pct"] = ovs["ATR"] / ovs["Signal Close"] * 100.0

# ---- counterfactual extension for losing T+2 time exits ----
EXT_DAYS = 3  # T+2 -> T+5

affected = (ovs["Exit Type"] == "Time") & (ovs["R_Multiple"] < 0)
delta = pd.Series(0.0, index=ovs.index)
n_censored = n_basis_drop = 0
new_exit_type = pd.Series("", index=ovs.index)

for idx, row in ovs[affected].iterrows():
    df = px.get(row["Ticker"])
    if df is None or row["Exit Date"] not in df.index:
        n_censored += 1
        continue
    pos = df.index.get_loc(row["Exit Date"])
    cache_t2_close = df["Close"].iloc[pos]
    if abs(cache_t2_close / row["Exit Price"] - 1) > 0.01:
        n_basis_drop += 1
        continue
    ext = df.iloc[pos + 1 : pos + 1 + EXT_DAYS]
    if len(ext) < EXT_DAYS:
        n_censored += 1
        continue
    tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
    new_exit = ext["Close"].iloc[-1]
    etype = "Time5"
    for _, day in ext.iterrows():
        if day["Low"] <= tgt:
            new_exit = tgt
            etype = "Target"
            break
    delta.loc[idx] = (cache_t2_close - new_exit) / row["ATR"]
    new_exit_type.loc[idx] = etype

modifiable = affected & (delta != 0.0)
print(f"Losing T+2 time exits: {affected.sum()}  |  replayed: {modifiable.sum()}"
      f"  |  censored (no fwd bars): {n_censored}  |  basis-dropped: {n_basis_drop}")

def pf(r: pd.Series) -> float:
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return g / l if l > 0 else np.inf

def scenario(name: str, mask: pd.Series) -> None:
    apply = modifiable & mask
    base, new = ovs["R_Multiple"], ovs["R_Multiple"] + delta.where(apply, 0.0)
    d = delta[apply]
    in_slice = mask.sum()
    print(f"\n=== {name} ===")
    print(f"  trades in slice: {in_slice}  |  extensions applied: {apply.sum()}"
          f"  ({(new_exit_type[apply] == 'Target').sum()} hit target during extension)")
    if apply.sum():
        print(f"  delta R on extended trades: mean {d.mean():+.3f}  median {d.median():+.3f}"
              f"  improved {(d > 0).mean() * 100:.0f}%  p5 {d.quantile(0.05):+.2f}"
              f"  min {d.min():+.2f}  max {d.max():+.2f}  sum {d.sum():+.1f}R")
    print(f"  OVS book   totR {base.sum():+.1f} -> {new.sum():+.1f}"
          f"  |  avgR {base.mean():+.3f} -> {new.mean():+.3f}"
          f"  |  win% {(base > 0).mean() * 100:.1f} -> {(new > 0).mean() * 100:.1f}"
          f"  |  PF {pf(base):.2f} -> {pf(new):.2f}"
          f"  |  worst {base.min():.2f} -> {new.min():.2f}")
    if apply.sum():
        yr = ovs.loc[apply].assign(d=d)["Signal Date"].dt.year
        by_year = d.groupby(yr).agg(["count", "sum"])
        bad = by_year[by_year["sum"] < -1.0]
        print(f"  years hurt worst: "
              + ", ".join(f"{y}: {r['sum']:+.1f}R/{int(r['count'])}t" for y, r in bad.iterrows())
              if len(bad) else "  no year loses more than 1R")

all_mask = pd.Series(True, index=ovs.index)
scenario("1. Extend ALL losing T+2 exits to T+5", all_mask)

print(f"\nrank252 coverage: {ovs['rank252'].notna().mean() * 100:.0f}% "
      f"| barbell split: low(<65) {(ovs['rank252'] < 65).sum()}, "
      f"high(>95) {(ovs['rank252'] > 95).sum()}, "
      f"mid/NaN {(~((ovs['rank252'] < 65) | (ovs['rank252'] > 95))).sum()}")
scenario("2a. Only low-side 252d entries (rank < 65)", ovs["rank252"] < 65)
scenario("2b. contrast: high-side entries (rank > 95)", ovs["rank252"] > 95)
scenario("2c. deeper low (rank < 35)", ovs["rank252"] < 35)

print(f"\natr_pct distribution: p25 {ovs['atr_pct'].quantile(0.25):.2f} "
      f"median {ovs['atr_pct'].median():.2f} p75 {ovs['atr_pct'].quantile(0.75):.2f} "
      f"| <3%: {(ovs['atr_pct'] < 3).sum()} trades")
scenario("3a. Only low-vol names (ATR% < 3)", ovs["atr_pct"] < 3)
scenario("3b. contrast: ATR% >= 3", ovs["atr_pct"] >= 3)
