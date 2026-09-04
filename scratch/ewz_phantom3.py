"""Forensic part 3: pin the 33.51 limit + persistence window phantom-fill math.

OLV = 'Oversold Low Volume', Long, entry = 'Limit Order -0.25 ATR (Persistent)'.
limit = sig_close - 0.25*ATR   (computed at signal bar, from df)
fill if any forward bar within holding window has Low <= limit.
"""
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 220)
ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
LIMIT_LIVE = 33.51

df = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
ewz = df[df["ticker"] == "EWZ"].copy()
ewz = ewz.set_index(pd.to_datetime(ewz["date"])).sort_index()

# Recompute ATR(14, Wilder) on the ADJUSTED cache the way backtester does.
h, l, c = ewz["High"], ewz["Low"], ewz["Close"]
pc = c.shift(1)
tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
ewz["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

print("Adjusted cache + recomputed ATR around the signal dates:")
w = ewz.loc["2026-06-03":"2026-06-12"]
print(w[["Open", "High", "Low", "Close", "ATR"]].round(4).to_string())

print("\n--- Candidate signal bars and their -0.25 ATR persistent limit (ADJ cache) ---")
for d in ["2026-06-05", "2026-06-08", "2026-06-09"]:
    if d in ewz.index.strftime("%Y-%m-%d").tolist():
        row = ewz.loc[d]
        lim = row["Close"] - 0.25 * row["ATR"]
        print(f"sig {d}: close={row['Close']:.4f} atr={row['ATR']:.4f} -> limit={lim:.4f}")

# The LIVE staged limit was 33.51 on a 6/9 signal. In RAW (un-adjusted live) basis:
# raw 6/9 close = 33.92, raw ATR ~? -> 33.92 - 0.25*ATR. Solve implied ATR:
raw_close_69 = 33.92
implied_atr = (raw_close_69 - LIMIT_LIVE) / 0.25
print(f"\nLIVE limit {LIMIT_LIVE} from raw 6/9 close {raw_close_69}: implied 0.25*ATR={raw_close_69 - LIMIT_LIVE:.4f} -> ATR={implied_atr:.4f}")

# Now the crux: simulate the persistent limit fill check in THREE bases.
# (a) live raw basis: limit 33.51 vs RAW forward lows
# (b) current adj cache basis: re-derive limit on adj cache, vs adj forward lows (SCALE-CONSISTENT)
# (c) MIXED: live limit 33.51 (raw) vs CURRENT adj-cache forward lows (post-ex re-adjusted)
raw_lows = {  # from yfinance auto_adjust=False
    "2026-06-10": 33.63, "2026-06-11": 33.86, "2026-06-12": 34.84,
    "2026-06-15": 34.55, "2026-06-16": 34.23, "2026-06-17": 33.97,
}
f = 0.99057  # pre-ex multiplicative factor (Adj/Close)
print("\n--- Forward-bar low comparison for a 6/9 signal, limit 33.51 ---")
print(f"{'date':12} {'raw_low':>9} {'<=33.51?':>9} {'adj_low(f*raw)':>15} {'<=33.51?':>9}")
for d, rl in raw_lows.items():
    # bars 6/10..6/12 are pre-ex so get scaled by f; 6/15+ are at/after ex (f=1)
    adj_low = rl * (f if d < "2026-06-15" else 1.0)
    print(f"{d:12} {rl:9.4f} {str(rl <= LIMIT_LIVE):>9} {adj_low:15.4f} {str(adj_low <= LIMIT_LIVE):>9}")

print("\nNOTE: cache last written Jun 11 06:47, so today's local cache 6/10 low still pre-re-adjust.")
print("After the 6/15 ex-div is ingested by the nightly rebuild, ALL pre-6/15 bars scale by ~0.99057.")
