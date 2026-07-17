"""NAV-level vol estimate for the resized overlay stack:
trend sleeve at 0.30x NAV + dial-gated SPY sleeve at 0.25x NAV.

Trend sleeve reconstructed per trend_sleeve.py conventions (month-end combo:
12-1 momentum > 0 AND close > 10-mo MA; inverse 63d-vol weights capped 20%;
cash earns nothing here — conservative for vol purposes it doesn't matter).
Approximation, not the production script — good for a vol estimate.
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
            "GLD", "SLV", "DBC", "TLT", "LQD"]
TREND_F = 0.30
DIAL_F = 0.25

mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                     filters=[("ticker", "in", UNIVERSE + ["^IRX"])])
mp["date"] = pd.to_datetime(mp["date"])
closes = (mp[mp["ticker"] != "^IRX"]
          .pivot_table(index="date", columns="ticker", values="Close")
          .sort_index().ffill())
closes = closes.dropna(how="any")          # common history of all 12
daily_ret = closes.pct_change()

m_close = closes.resample("ME").last()
mom = m_close / m_close.shift(12) - 1
ma10 = m_close.rolling(10).mean()
sig = (mom > 0) & (m_close > ma10)

vol63 = daily_ret.rolling(63).std()
vol_m = vol63.resample("ME").last()

w = (1.0 / vol_m).where(sig)
w = w.div(w.sum(axis=1), axis=0).clip(upper=0.20).fillna(0.0)
# renormalization after cap is NOT done in prod (cash otherwise) — keep caps

m_ret = m_close.pct_change()
trend_sleeve = (w.shift(1) * m_ret).sum(axis=1).dropna()
trend_sleeve = trend_sleeve.loc["2008":]

# dial sleeve (final spec, variant C) — reuse the event loop
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dev", os.path.join(_ROOT, "scratch", "dial_exit_variants.py"))
dev = importlib.util.module_from_spec(spec)
import sys as _sys
_sys.modules["dev"] = dev
spec.loader.exec_module(dev)
dial_r, _, _, _ = dev.run("C")
dial_m = (1 + dial_r).resample("ME").prod() - 1

print(f"trend sleeve (1.0x sleeve): {trend_sleeve.index.min():%Y-%m} -> "
      f"{trend_sleeve.index.max():%Y-%m}, ann vol "
      f"{trend_sleeve.std() * np.sqrt(12) * 100:.1f}%, "
      f"ann ret {trend_sleeve.mean() * 12 * 100:.1f}%")
print(f"dial sleeve (1.0x sleeve, 2016+): ann vol "
      f"{dial_m.std() * np.sqrt(12) * 100:.1f}%")

both = pd.DataFrame({"trend": trend_sleeve, "dial": dial_m}).dropna()
corr = both["trend"].corr(both["dial"])
print(f"monthly correlation (overlap {both.index.min():%Y-%m}+, "
      f"N={len(both)}): {corr:+.2f}")

t_nav = TREND_F * trend_sleeve
d_nav = DIAL_F * dial_m
print(f"\nNAV-level, trend at {TREND_F:.0%}: ann vol "
      f"{t_nav.std() * np.sqrt(12) * 100:.2f}%")
print(f"NAV-level, dial at {DIAL_F:.0%}:  ann vol "
      f"{d_nav.std() * np.sqrt(12) * 100:.2f}%")
combo = (t_nav.reindex(both.index) + d_nav.reindex(both.index))
print(f"combined overlay (overlap period): ann vol "
      f"{combo.std() * np.sqrt(12) * 100:.2f}%")
