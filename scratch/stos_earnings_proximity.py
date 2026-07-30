"""St OS Sznl: performance by proximity to earnings (signal date vs nearest announcement)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
sys.path.insert(0, str(ROOT))

from earnings_filter import load_earnings_dates_map, signed_offset

df = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
s = df[df["Strategy"] == "St OS Sznl"].copy()
s["Signal Date"] = pd.to_datetime(s["Signal Date"])

emap = load_earnings_dates_map()

def offset_for(row):
    dates = emap.get(row["Ticker"])
    if dates is None or len(dates) == 0:
        return np.nan
    return signed_offset(row["Signal Date"], dates)

s["earn_off"] = s.apply(offset_for, axis=1)

def summarize(sub: pd.DataFrame, label: str) -> None:
    if len(sub) == 0:
        print(f"{label:28s} N=0")
        return
    r = sub["R_Multiple"]
    pnl = sub["PnL_flat_750k"].sum()
    print(f"{label:28s} N={len(sub):3d}  avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
          f"win={100*(r>0).mean():.1f}%  totR={r.sum():+.1f}  PnL flat=${pnl:+,.0f}")

print(f"Total St OS Sznl trades: {len(s)}  ({s['Signal Date'].min().date()} .. {s['Signal Date'].max().date()})")
print(f"Tickers with no earnings data (pass-through under a blackout): "
      f"{s['earn_off'].isna().sum()} trades, {sorted(s.loc[s['earn_off'].isna(),'Ticker'].unique())}")
print()

has = s[s["earn_off"].notna()]
summarize(s, "ALL trades")
summarize(has, "  with earnings data")
summarize(has[has["earn_off"].abs() <= 10], "  WITHIN +-10 TD")
summarize(has[has["earn_off"].abs() > 10], "  OUTSIDE +-10 TD")
summarize(s[(s["earn_off"].isna()) | (s["earn_off"].abs() > 10)], "  SURVIVES blackout (out+NaN)")
print()
print("Finer buckets (signed offset, negative = earnings AHEAD):")
for lo, hi, lab in [(-10, -6, "-10..-6 (pre)"), (-5, -1, "-5..-1 (pre)"), (0, 0, "0 (day-of)"),
                    (1, 5, "+1..+5 (post)"), (6, 10, "+6..+10 (post)")]:
    summarize(has[(has["earn_off"] >= lo) & (has["earn_off"] <= hi)], f"  {lab}")
print()
print("Within-window trades detail:")
w = has[has["earn_off"].abs() <= 10].sort_values("Signal Date")
cols = ["Ticker", "Tier", "Signal Date", "earn_off", "Exit Type", "R_Multiple", "PnL_flat_750k"]
print(w[cols].to_string(index=False))
