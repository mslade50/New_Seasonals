"""Top-20 most counterproductive OVS hold extensions (any-loss trigger)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
ovs = ledger[ledger["Strategy"] == "Overbot Vol Spike"].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    ovs[c] = pd.to_datetime(ovs[c])
ovs["atr_pct"] = ovs["ATR"] / ovs["Signal Close"] * 100.0

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
prices = prices[prices["ticker"].isin(ovs["Ticker"].unique())]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

rows = []
affected = (ovs["Exit Type"] == "Time") & (ovs["R_Multiple"] < 0)
for idx, row in ovs[affected].iterrows():
    df = px.get(row["Ticker"])
    if df is None or row["Exit Date"] not in df.index:
        continue
    pos = df.index.get_loc(row["Exit Date"])
    t2_close = df["Close"].iloc[pos]
    if abs(t2_close / row["Exit Price"] - 1) > 0.01:
        continue
    ext = df.iloc[pos + 1 : pos + 4]
    if len(ext) < 3:
        continue
    tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
    ex_d, ex_p = ext.index[-1], ext["Close"].iloc[-1]
    for d, day in ext.iterrows():
        if day["Low"] <= tgt:
            ex_d, ex_p = d, tgt
            break
    delta = (t2_close - ex_p) / row["ATR"]
    rows.append({
        "Ticker": row["Ticker"], "Tier": row["Tier"],
        "Signal": row["Signal Date"].date(), "T+2 exit": row["Exit Date"].date(),
        "New exit": ex_d.date(),
        "R@T+2": row["R_Multiple"], "deltaR": delta,
        "R@T+5": row["R_Multiple"] + delta,
        "ATR%": row["atr_pct"],
        "move%": (ex_p / t2_close - 1) * 100,  # stock kept rising this much
        "$impact_flat": delta * row["Risk_flat_750k"],
    })

res = pd.DataFrame(rows).sort_values("deltaR").head(20)
pd.set_option("display.width", 200)
print(res.to_string(index=False,
                    formatters={"R@T+2": "{:+.2f}".format, "deltaR": "{:+.2f}".format,
                                "R@T+5": "{:+.2f}".format, "ATR%": "{:.1f}".format,
                                "move%": "{:+.1f}".format, "$impact_flat": "{:,.0f}".format}))
print(f"\nsum of top-20 damage: {res['deltaR'].sum():+.1f}R  "
      f"(vs +81.9R total from all 351 extensions)")
print(f"ATR%<3 among top 20: {(res['ATR%'].astype(float) < 3).sum()}")
