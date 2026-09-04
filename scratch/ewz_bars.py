from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
ewz = df[df["ticker"] == "EWZ"].copy()
ewz["date"] = pd.to_datetime(ewz["date"])
ewz = ewz.set_index("date").sort_index()
win = ewz.loc["2026-05-20":"2026-06-30", ["Open", "High", "Low", "Close", "Volume"]]
print("LOCAL CACHE (written Jun 11, pre-ex) EWZ bars:")
print(win.to_string())
print("\nLast cache date for EWZ:", ewz.index.max())

# Recompute ATR(14) Wilder/simple to match indicators.py (rolling(14).mean of TR)
h, l, c = ewz["High"], ewz["Low"], ewz["Close"]
pc = c.shift(1)
tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
ewz["ATR"] = tr.rolling(14).mean()
print("\nATR(14) simple-rolling around signal:")
print(ewz.loc["2026-06-01":"2026-06-12", ["Close", "ATR"]].to_string())

# The reported staged order: signal 6/8, staged 6/9, OLV -0.25 ATR Persistent.
for sig in ["2026-06-08", "2026-06-09"]:
    if pd.Timestamp(sig) in ewz.index:
        sc = ewz.loc[sig, "Close"]
        sa = ewz.loc[sig, "ATR"]
        lim = sc - 0.25 * sa
        print(f"\nsignal {sig}: Close={sc:.4f} ATR={sa:.4f} -> -0.25ATR limit={lim:.4f} (round2={round(lim,2)})")
