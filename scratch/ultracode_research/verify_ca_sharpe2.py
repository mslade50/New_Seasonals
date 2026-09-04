"""Sharpe ladder including the partial 2026-07 month (their apparent basis)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
NAV = 750_000.0

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"])
uvxy = mp[mp["ticker"] == "UVXY"].set_index("date")["Close"].sort_index()
lev = pd.Series(np.where(uvxy.index < pd.Timestamp("2018-02-28"), 2.0, 1.5),
                index=uvxy.index)
r_proxy = (uvxy.pct_change() / lev).dropna()

fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
basis = fr["63d"].rolling(10, min_periods=1).mean()

def build_gate(b, on_t, off_t):
    on = False
    out = []
    for v in b.values:
        if not on and v >= on_t:
            on = True
        elif on and v < off_t:
            on = False
        out.append(on)
    return pd.Series(out, index=b.index)

g = build_gate(basis, 55, 50).reindex(r_proxy.index).ffill().fillna(False).infer_objects(copy=False)
pos = g.shift(1).fillna(False).astype(float)
pnl = pos * r_proxy * 0.05 * NAV - pos.diff().abs().fillna(0) * 0.05 * NAV * 0.001
sleeve_m = pnl.loc["2016-07-01":].resample("ME").sum()

tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
tr["Exit Date"] = pd.to_datetime(tr["Exit Date"])
t16 = tr[tr["Signal Date"] >= "2016-07-05"].copy()
sig_frag = basis.reindex(pd.date_range(basis.index[0], basis.index[-1], freq="D")).ffill(limit=5)
f = sig_frag.reindex(t16["Signal Date"]).values
mult = np.where(t16["Strategy"].values == "Overbot Vol Spike", 1.0,
                1.0 - 0.5 * np.clip((np.nan_to_num(f, nan=0.0) - 50) / 10, 0, 1))
t16["pnl_thr"] = t16["PnL_flat_750k"] * mult

def monthly(s, col):
    m = s.groupby(s["Exit Date"].dt.to_period("M"))[col].sum()
    m.index = m.index.to_timestamp("M")
    return m

base_m = monthly(t16, "PnL_flat_750k")
thr_m = monthly(t16, "pnl_thr")
idx = base_m.index  # includes partial 2026-07
hed = sleeve_m.reindex(idx).fillna(0)

def sh(x):
    return x.mean() / x.std() * np.sqrt(12)

print("incl partial 2026-07:")
print("  baseline", round(sh(base_m), 2), "| throttle", round(sh(thr_m), 2),
      "| throttle+VXXP", round(sh(thr_m + hed), 2),
      "| baseline+VXXP", round(sh(base_m + hed), 2))
