"""Crisis-alpha track: shared data prep.

Builds and saves (to this dir):
  ca_prices.parquet  — daily Close/Open panel for the overlay instruments
  ca_frag.parquet    — daily 63d MA10 fragility (live sizing basis)
  ca_book_daily.parquet — book daily realized PnL (flat 750k, exit-date attributed),
                          split OVS / non-OVS, plus frag-at-signal per trade
  ca_book_monthly.parquet — book monthly return series (PnL/750k by exit month)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

TICKERS = ["SPY", "QQQ", "UVXY", "GLD", "TLT", "IEF", "DBC", "UUP", "TMF",
           "^VIX", "^VIX3M", "^IRX"]

mp = pd.read_parquet(ROOT / "data/master_prices.parquet")
mp = mp[mp["ticker"].isin(TICKERS)].copy()
mp["date"] = pd.to_datetime(mp["date"]).dt.normalize()
close = mp.pivot_table(index="date", columns="ticker", values="Close")
open_ = mp.pivot_table(index="date", columns="ticker", values="Open")
panel = pd.concat({"Close": close, "Open": open_}, axis=1)
panel.to_parquet(OUT / "ca_prices.parquet")
print("panel:", panel.shape, panel.index.min().date(), "->", panel.index.max().date())
for t in TICKERS:
    s = close[t].dropna()
    print(f"  {t}: {s.index.min().date()} -> {s.index.max().date()}")

# fragility — live sizing basis: 63d, 10d MA
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()
frag_ma.name = "frag63_ma10"
frag_ma.to_frame().to_parquet(OUT / "ca_frag.parquet")
print("\nfrag63_ma10:", frag_ma.index.min().date(), "->", frag_ma.index.max().date(),
      f" mean={frag_ma.mean():.1f} p90={frag_ma.quantile(.9):.1f}")
print("days >=50:", (frag_ma >= 50).sum(), " >=55:", (frag_ma >= 55).sum(),
      " >=60:", (frag_ma >= 60).sum(), "of", len(frag_ma))

# ledger
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"]).dt.normalize()
led["Exit Date"] = pd.to_datetime(led["Exit Date"]).dt.normalize()
led = led.dropna(subset=["Exit Date", "PnL_flat_750k"])
led["is_ovs"] = led["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
print("\nledger:", len(led), "trades", led["Exit Date"].min().date(), "->",
      led["Exit Date"].max().date())
print("Size_Mult describe:", led["Size_Mult"].describe().round(3).to_dict())

# as-of frag at signal date (ffill limit 5 days like live)
led = led.sort_values("Signal Date")
frag_df = frag_ma.rename_axis("date").reset_index()
led["frag_sig"] = pd.merge_asof(
    led[["Signal Date"]], frag_df, left_on="Signal Date", right_on="date",
    tolerance=pd.Timedelta(days=5),
)["frag63_ma10"].values

keep = ["trade_id", "Strategy", "Ticker", "Direction", "Signal Date", "Exit Date",
        "R_Multiple", "PnL_flat_750k", "Risk_flat_750k", "is_ovs", "frag_sig"]
led[keep].to_parquet(OUT / "ca_book_daily.parquet")

# monthly book series
led["ym"] = led["Exit Date"].dt.to_period("M")
mon = led.groupby("ym").agg(pnl=("PnL_flat_750k", "sum"), n=("trade_id", "count"),
                            totR=("R_Multiple", "sum"))
mon["ret"] = mon["pnl"] / 750_000
# fill missing months with zero (book idle)
full_idx = pd.period_range(mon.index.min(), mon.index.max(), freq="M")
mon = mon.reindex(full_idx).fillna({"pnl": 0, "n": 0, "totR": 0, "ret": 0})
mon.to_parquet(OUT / "ca_book_monthly.parquet")
print("\nbook monthly:", len(mon), "months, mean ret %/mo:",
      round(mon["ret"].mean() * 100, 2), " (2003-01..):", mon.index.min(), mon.index.max())
