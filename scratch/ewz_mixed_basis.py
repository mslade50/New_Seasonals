"""Forensic part 2: reconstruct the live 33.51 limit and test the
scale-invariance / mixed-basis theories with EWZ real numbers.

The live scan on 2026-06-09 staged a persistent -k*ATR limit anchored to
that day's Close, using that day's ATR (REL_CLOSE). Both Close and ATR were
read from whatever basis the cache had ON 6/9 (pre any later ex-div).

The backtester re-run reads the CURRENT cache (re-adjusted after any
dividend that went ex after 6/9). We compare:
  (A) limit implied by the CURRENT-cache 6/9 close+ATR  vs current bars
  (B) limit FIXED at 33.51 (live, old basis)            vs current bars
"""
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
MP = ROOT / "data" / "master_prices.parquet"
LIMIT_LIVE = 33.51

df = pd.read_parquet(MP)
ewz = df[df["ticker"] == "EWZ"].copy()
ewz = ewz.set_index(pd.to_datetime(ewz["date"])).sort_index()

# --- reconstruct ATR(14) exactly as indicators.py does ---
high_low = ewz["High"] - ewz["Low"]
high_close = (ewz["High"] - ewz["Close"].shift()).abs()
low_close = (ewz["Low"] - ewz["Close"].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
ewz["ATR"] = tr.rolling(14).mean()

sig_date = pd.Timestamp("2026-06-09")
sig_close = ewz.loc[sig_date, "Close"]
sig_atr = ewz.loc[sig_date, "ATR"]

print("=" * 72)
print("CURRENT-CACHE 6/9 anchor (re-adjusted basis as of last parquet write)")
print("=" * 72)
print(f"  Close(6/9) = {sig_close:.4f}")
print(f"  ATR14(6/9) = {sig_atr:.4f}")
for k in (0.25, 0.5, 0.75, 1.0):
    lp = sig_close - k * sig_atr
    lp_s = sig_close + k * sig_atr
    print(f"  long limit  -{k:>4} ATR = {lp:.4f}   short limit +{k:>4} ATR = {lp_s:.4f}")

implied_k = (sig_close - LIMIT_LIVE) / sig_atr
print(f"\n  k that reproduces 33.51 off CURRENT close = {implied_k:.4f}")

print("\n" + "=" * 72)
print("Back out the LIVE (pre-ex) basis from the live limit 33.51")
print("=" * 72)
for k in (0.25, 0.5, 0.75, 1.0):
    cur_limit = sig_close - k * sig_atr
    f = cur_limit / LIMIT_LIVE
    print(f"  k=-{k}: current-basis limit={cur_limit:.4f} -> implied f={f:.5f} "
          f"(div ~ {(1-f)*100:.3f}% of pre-ex close)")

print("\n" + "=" * 72)
print("FORWARD BARS (current cache) vs the two candidate limits")
print("=" * 72)
fwd = ewz.loc["2026-06-10":"2026-06-30", ["Open", "High", "Low", "Close", "ATR"]].copy()
fwd["hits_LIVE_33.51"] = fwd["Low"] <= LIMIT_LIVE
cur_limit_025 = sig_close - 0.25 * sig_atr
fwd["cur_-0.25_lim"] = round(cur_limit_025, 4)
fwd["hits_cur_-0.25"] = fwd["Low"] <= cur_limit_025
print(fwd.round(4).to_string())

print(f"\nLIVE limit 33.51 -> any current bar Low <= 33.51? "
      f"{bool(fwd['hits_LIVE_33.51'].any())}")
print(f"CURRENT -0.25ATR limit {cur_limit_025:.4f} -> any current bar Low <= it? "
      f"{bool(fwd['hits_cur_-0.25'].any())}")
