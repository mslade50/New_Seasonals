"""How long should the OLV (Oversold Low Volume) persistent entry limit stay live?

Production model: GTC limit at close - 0.25 ATR, live for the full hold window
(hold_days=10 trading days). Fill on day i shortens the hold to 10-(i-signal).
Engine: pages/strat_backtester.py persistent-limit block (~line 1642).

Because OLV has max_one_pos=False, dropping a late fill never unblocks a later
signal, and R-multiple / flat-$750k PnL are sizing-path-invariant. So capping
the order-live window to T+N is EXACTLY the subset of ledger trades whose fill
landed on or before trading day N. No re-run needed; this is exact on R and on
the flat-$750k dollar basis (additive). Compounded $ is path-dependent and only
indicative.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data_provider

STRAT = "Oversold Low Volume"

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = trades[trades["Strategy"] == STRAT].copy()
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    olv[c] = pd.to_datetime(olv[c])

# --- per-ticker bar index (what the engine actually counts) ---
# The persistent limit is live for 10 of the TICKER'S OWN bars, not 10 NYSE
# calendar days. Gappy overflow names can therefore fill many calendar days
# out while still being <=10 bars. Compute lag on each ticker's own series.
tickers = sorted(olv["Ticker"].unique())
hist = data_provider.get_history(tickers, start="2003-01-01")
bar_pos = {}
for t in tickers:
    df = hist.get(t)
    if df is None or df.empty:
        continue
    idx = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    bar_pos[t] = pd.Series(np.arange(len(idx)), index=idx)

def lag_bars(tk, sig, ent):
    bp = bar_pos.get(tk)
    if bp is None:
        return np.nan
    sig, ent = pd.Timestamp(sig).normalize(), pd.Timestamp(ent).normalize()
    try:
        return int(bp.loc[ent] - bp.loc[sig])
    except KeyError:
        # fall back to as-of positions (entry/signal should both be real bars)
        try:
            return int(bp.index.searchsorted(ent) - bp.index.searchsorted(sig))
        except Exception:
            return np.nan

olv["fill_lag"] = [lag_bars(t, s, e)
                   for t, s, e in zip(olv["Ticker"], olv["Signal Date"], olv["Entry Date"])]
_nan = olv["fill_lag"].isna().sum()
if _nan:
    print(f"NOTE: {_nan} trades unmatched to price history (dropped from lag analysis)")
    olv = olv.dropna(subset=["fill_lag"])
olv["fill_lag"] = olv["fill_lag"].astype(int)

# sanity: persistent limit is live T+1..T+10, so fill_lag should be in [1,10]
bad = olv[(olv["fill_lag"] < 1) | (olv["fill_lag"] > 10)]
if len(bad):
    print(f"WARNING: {len(bad)} trades with fill_lag outside [1,10]:")
    print(bad[["Ticker", "Signal Date", "Entry Date", "fill_lag"]].to_string())
    olv = olv[(olv["fill_lag"] >= 1) & (olv["fill_lag"] <= 10)]

N = len(olv)
print(f"OLV trades: {N}  ({olv['Signal Date'].min().date()} -> {olv['Signal Date'].max().date()})")
print(f"Tier mix: {olv['Tier'].value_counts().to_dict()}")

# ---------- per-fill-day marginal profile ----------
print("\n" + "=" * 78)
print("MARGINAL profile by fill day (the trades that fill EXACTLY on day d)")
print("=" * 78)
print(f"{'day':>3} {'N':>5} {'%cum':>6} {'win%':>6} {'avgR':>7} {'medR':>7} "
      f"{'totR':>8} {'avgRet%':>8} {'flat$/750k':>12} {'exit mix'}")
for d in range(1, 11):
    g = olv[olv["fill_lag"] == d]
    if len(g) == 0:
        print(f"{d:>3} {0:>5}")
        continue
    cum = (olv["fill_lag"] <= d).mean()
    r = g["R_Multiple"]
    mix = g["Exit Type"].value_counts().to_dict()
    print(f"{d:>3} {len(g):>5} {cum:>5.0%} {(r>0).mean():>6.1%} {r.mean():>+7.3f} "
          f"{r.median():>+7.3f} {r.sum():>+8.1f} {g['Return_Pct'].mean():>+8.2f} "
          f"{g['PnL_flat_750k'].sum():>12,.0f}  {mix}")

# ---------- cumulative windows (the actual question) ----------
print("\n" + "=" * 78)
print("CUMULATIVE: results if the order is left live only through day N (T+N)")
print("=" * 78)
print(f"{'window':>7} {'N':>5} {'%of':>5} {'win%':>6} {'avgR':>7} {'totR':>8} "
      f"{'expR':>7} {'PF':>6} {'avgRet%':>8} {'flat$750k':>12} {'cmp$':>12}")

def pf(r):
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    return g / l if l > 0 else float("inf")

for w in [3, 5, 7, 10]:
    g = olv[olv["fill_lag"] <= w]
    r = g["R_Multiple"]
    print(f"  T+{w:<4} {len(g):>5} {len(g)/N:>4.0%} {(r>0).mean():>6.1%} "
          f"{r.mean():>+7.3f} {r.sum():>+8.1f} {r.mean():>+7.3f} {pf(r):>6.2f} "
          f"{g['Return_Pct'].mean():>+8.2f} {g['PnL_flat_750k'].sum():>12,.0f} "
          f"{g['PnL_compounded'].sum():>12,.0f}")

# ---------- what you give up by cutting (days 4-10 and 6-10) ----------
print("\n" + "=" * 78)
print("WHAT THE LATE FILLS CONTRIBUTE (the trades you'd DROP)")
print("=" * 78)
for lo, hi, lbl in [(4, 10, "days 4-10 (dropped if T+3)"),
                    (6, 10, "days 6-10 (dropped if T+5)"),
                    (8, 10, "days 8-10 (dropped if T+7)")]:
    g = olv[(olv["fill_lag"] >= lo) & (olv["fill_lag"] <= hi)]
    r = g["R_Multiple"]
    if len(g) == 0:
        print(f"  {lbl}: none"); continue
    print(f"  {lbl}: N={len(g):>3}  win%={(r>0).mean():>5.1%}  avgR={r.mean():>+.3f}  "
          f"totR={r.sum():>+6.1f}  flat$={g['PnL_flat_750k'].sum():>+10,.0f}")

# ---------- tier split of the cumulative windows ----------
print("\n" + "=" * 78)
print("BY TIER")
print("=" * 78)
for tier in ["Liquid", "Overflow"]:
    sub = olv[olv["Tier"] == tier]
    if len(sub) == 0:
        continue
    print(f"\n  {tier} (N={len(sub)}):")
    for w in [3, 5, 7, 10]:
        g = sub[sub["fill_lag"] <= w]
        r = g["R_Multiple"]
        print(f"    T+{w:<3} N={len(g):>3} win%={(r>0).mean():>5.1%} "
              f"avgR={r.mean():>+.3f} totR={r.sum():>+6.1f}")
