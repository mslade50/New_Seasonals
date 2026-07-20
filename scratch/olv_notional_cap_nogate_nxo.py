"""Bounding case for the 50%-NAV OLV single-stock notional cap: sector loss
gate fully OFF and the candidate nxo_15 exit rule (volume-confirmed close,
exit next open) — longest holds, deepest stacks. Entries = no-gate pass."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
ETFS = {"CEF", "DBC", "EWZ", "GDX", "GLD", "ITA", "KRE", "OIH", "SLV", "USO", "BP"}
ETFS.discard("BP")  # BP is a single stock
STOP_SLIP_BPS = 3.0
STOP_ATR, TGT_ATR, HOLD, FILL_WINDOW, LIMIT_MULT = 1.25, 2.5, 10, 3, 0.25

ng = pd.read_parquet(ROOT / "data" / "backtest_trades_nogate.parquet")
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    ng[c] = pd.to_datetime(ng[c])
ng["notional"] = ng["Shares_flat"].abs() * ng["Entry Price"]

tickers = sorted(ng["Ticker"].unique())
px = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     filters=[("ticker", "in", tickers)])
px["date"] = pd.to_datetime(px["date"]).dt.normalize()
frames = {}
for tkr, g in px.groupby("ticker"):
    g = g.sort_values("date").drop_duplicates("date").set_index("date")
    ranges = pd.concat([g["High"] - g["Low"],
                        (g["High"] - g["Close"].shift(1)).abs(),
                        (g["Low"] - g["Close"].shift(1)).abs()], axis=1)
    g["ATR"] = ranges.max(axis=1).rolling(14).mean()
    g["vol_med20"] = g["Volume"].rolling(20).median().shift(1)
    frames[tkr] = g


def sim_nxo(tkr, signal_date):
    df = frames.get(tkr)
    if df is None or signal_date not in df.index:
        return None
    sidx = df.index.get_loc(signal_date)
    atr = df.iloc[sidx]["ATR"]
    if pd.isna(atr) or atr <= 0:
        return None
    limit = df.iloc[sidx]["Close"] - LIMIT_MULT * atr
    entry_idx = entry_price = None
    for i in range(sidx + 1, min(sidx + 1 + FILL_WINDOW, len(df))):
        r = df.iloc[i]
        if r["Open"] < limit:
            entry_idx, entry_price = i, r["Open"]; break
        if r["Low"] <= limit:
            entry_idx, entry_price = i, limit; break
    if entry_idx is None:
        return None
    hold = max(1, HOLD - (entry_idx - sidx - 1))
    stop_level = entry_price - STOP_ATR * atr
    tgt_level = entry_price + TGT_ATR * atr
    risk_unit = STOP_ATR * atr
    max_exit_idx = min(entry_idx + hold, len(df) - 1)
    exit_idx, exit_price = max_exit_idx, df.iloc[max_exit_idx]["Close"]
    for ci in range(entry_idx + 1, max_exit_idx + 1):
        r = df.iloc[ci]
        if r["High"] >= tgt_level:
            exit_idx, exit_price = ci, tgt_level; break
        if r["Close"] <= stop_level:
            volofk = (r["Volume"] / r["vol_med20"]) if r["vol_med20"] and r["vol_med20"] > 0 else np.nan
            if not pd.isna(volofk) and volofk >= 1.5:
                if ci + 1 < len(df):
                    exit_idx, exit_price = ci + 1, df.iloc[ci + 1]["Open"] * (1 - STOP_SLIP_BPS / 1e4)
                else:
                    exit_idx, exit_price = ci, r["Close"] * (1 - STOP_SLIP_BPS / 1e4)
                break
    return {"Entry Date": df.index[entry_idx], "Exit Date": df.index[exit_idx],
            "R": (exit_price - entry_price) / risk_unit}


recs = []
for _, s in ng.iterrows():
    out = sim_nxo(s["Ticker"], s["Signal Date"])
    if out is None:
        continue
    out.update({"Ticker": s["Ticker"], "Signal Date": s["Signal Date"],
                "notional": s["notional"], "PnL": out["R"] * s["Risk_flat_750k"]})
    recs.append(out)
d = pd.DataFrame(recs)
print(f"simulated {len(d)}/{len(ng)} no-gate trades under nxo_15")

stocks = d[~d["Ticker"].isin(ETFS)]

cap = 0.50 * NAV
lost = 0.0
binds = []
peaks = {}
for tkr, g in stocks.groupby("Ticker"):
    open_legs = []
    peak = 0.0
    for _, r in g.sort_values(["Entry Date", "Signal Date"]).iterrows():
        open_legs = [(x, n) for x, n in open_legs if x > r["Entry Date"]]
        used = sum(n for _, n in open_legs)
        peak = max(peak, used + r["notional"])
        room = max(0.0, cap - used)
        clip = min(1.0, room / r["notional"]) if r["notional"] > 0 else 1.0
        if clip < 1.0:
            binds.append({"Ticker": tkr, "Entry Date": r["Entry Date"].date(),
                          "clip": round(clip, 2), "PnL": round(r["PnL"]),
                          "lost": r["PnL"] * (1 - clip)})
            lost += r["PnL"] * (1 - clip)
        open_legs.append((r["Exit Date"], r["notional"] * clip))
    peaks[tkr] = peak
pk = pd.Series(peaks).sort_values(ascending=False)
print(f"\nno-gate + nxo_15, single stocks: tickers over 50% NAV uncapped: {(pk > cap).sum()}")
print("  " + ", ".join(f"{t} {v/NAV:.0%}" for t, v in pk.head(8).items()))
b = pd.DataFrame(binds)
print(f"cap 50%: binds {len(b)} legs, foregone ${lost:,.0f} "
      f"(segment total ${stocks['PnL'].sum():,.0f})")
if len(b):
    print(b.to_string(index=False))
