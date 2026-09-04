"""Forensic 2: EWZ bars around 6/9 in the cache, ledger trade, and ATR/limit recompute."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MP = ROOT / "data" / "master_prices.parquet"
LEDGER = ROOT / "data" / "backtest_trades_full.parquet"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 60)

mp = pd.read_parquet(MP)
ewz = mp[mp["ticker"] == "EWZ"].copy()
ewz["date"] = pd.to_datetime(ewz["date"])
ewz = ewz.sort_values("date").set_index("date")

print("EWZ cache last date:", ewz.index.max())
print("\nEWZ bars late May -> mid June 2026 (CACHE = adjusted as of Jun 11 rebuild):")
window = ewz.loc["2026-05-26":"2026-06-12"][["Open", "High", "Low", "Close", "Volume"]]
print(window.round(4))

# Recompute ATR(14) on the CACHED (current-adjusted) series, the way calculate_indicators does
high_low = ewz["High"] - ewz["Low"]
high_close = (ewz["High"] - ewz["Close"].shift()).abs()
low_close = (ewz["Low"] - ewz["Close"].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
ewz["ATR"] = tr.rolling(14).mean()

print("\n--- Limit recompute on the CACHED adjusted series ---")
sig_day = pd.Timestamp("2026-06-09")
if sig_day in ewz.index:
    sig_close = ewz.loc[sig_day, "Close"]
    sig_atr = ewz.loc[sig_day, "ATR"]
    print(f"6/9 adj Close={sig_close:.4f}  adj ATR={sig_atr:.4f}")
    for k in (0.25, 0.5, 1.0):
        print(f"  Long limit close-{k}ATR = {sig_close - k*sig_atr:.4f}")
    # forward bars
    fwd = ewz.loc[sig_day:].head(8)[["Open", "High", "Low", "Close"]]
    print("\nForward bars (adjusted) from 6/9:")
    print(fwd.round(4))
else:
    print("6/9 not in cache index")

# Ledger
print("\n" + "=" * 70)
print("LEDGER: any EWZ trades entered around 6/9/2026")
led = pd.read_parquet(LEDGER)
print("ledger cols:", list(led.columns))
tcol = "Ticker" if "Ticker" in led.columns else [c for c in led.columns if c.lower()=="ticker"][0]
ewzt = led[led[tcol].astype(str).str.upper() == "EWZ"].copy()
# date columns
dcols = [c for c in led.columns if "date" in c.lower() or "entry" in c.lower() or "exit" in c.lower()]
print("date-ish cols:", dcols)
if not ewzt.empty:
    show = [c for c in led.columns if any(k in c.lower() for k in
            ["ticker","entry","exit","date","price","strategy","direction","r","fill"])]
    recent = ewzt
    for c in dcols:
        try:
            ewzt[c] = pd.to_datetime(ewzt[c])
        except Exception:
            pass
    # filter to 2026
    edate = [c for c in dcols if "entry" in c.lower()]
    if edate:
        ec = edate[0]
        recent = ewzt[ewzt[ec] >= pd.Timestamp("2026-05-15")]
    print(recent[show].to_string())
else:
    print("No EWZ trades in ledger.")
