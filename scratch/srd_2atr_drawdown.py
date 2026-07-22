"""Odds of a >=2 ATR adverse move within each forward window after the Seasonal
Rank Divergence trigger, vs unconditional baselines.

Reconstructs the exact SRD mask (pages/risk_dashboard_v2.compute_seasonal_divergence_signal
logic) using the committed seasonal CSVs + SPY OHLC from data/master_prices.parquet.

Move definition (per day i, horizon N): anchor at the FIRE-DAY CLOSE (Close_i),
then over the forward days i+1..i+N take the largest drop to any subsequent
regular CLOSE: maxdrop = max_j (Close_i - Close_{i+j}). Close-to-close, downside
only, fixed anchor (no interim high reset, no intraday lows). Expressed in ATR
units using the Wilder-14 ATR as of day i. Flag = maxdrop/ATR_i >= 2.0.
"""
import os
import sys

class _NoOp:
    def __getattr__(self, n): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data

sys.modules["streamlit"] = _NoOp()

ROOT = r"C:\Users\mckin\UsersmckinNew_Seasonals"
sys.path.insert(0, os.path.join(ROOT, "pages"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import risk_dashboard_v2 as rd

HORIZONS = {"5d": 5, "10d": 10, "21d": 21, "42d": 42, "63d": 63}
ATR_N = 14
THRESH_ATR = 2.0

# ---- SPY OHLC ----
mp = pd.read_parquet(os.path.join(ROOT, "data", "master_prices.parquet"))
spy = mp[mp["ticker"] == "SPY"].copy()
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")[["Open", "High", "Low", "Close"]].dropna()
print(f"SPY OHLC: {spy.index[0].date()} -> {spy.index[-1].date()}  ({len(spy)} days)")

H = spy["High"].to_numpy(float)
L = spy["Low"].to_numpy(float)
C = spy["Close"].to_numpy(float)
n = len(spy)

# ---- Wilder ATR-14 ----
prev_c = np.concatenate([[np.nan], C[:-1]])
tr = np.maximum.reduce([H - L, np.abs(H - prev_c), np.abs(L - prev_c)])
atr = np.full(n, np.nan)
# seed with simple mean of first ATR_N TRs (skip TR[0] which has no prev close)
if n > ATR_N:
    atr[ATR_N] = np.nanmean(tr[1:ATR_N + 1])
    for i in range(ATR_N + 1, n):
        atr[i] = (atr[i - 1] * (ATR_N - 1) + tr[i]) / ATR_N

# ---- forward max close-to-close drop from fire-day close, vectorized ----
def maxdrop_for_horizon(N):
    maxdrop = np.zeros(n)    # anchored at Close_i
    valid = np.ones(n, bool)
    for j in range(1, N + 1):
        Cs = np.concatenate([C[j:], np.full(j, np.nan)])   # Close[i+j]
        maxdrop = np.fmax(maxdrop, C - Cs)                 # downside only
    valid[n - N:] = False    # incomplete forward window
    return maxdrop, valid

# ---- SRD mask (exact replication) ----
spread = rd._load_seasonal_spread()
assert spread is not None, "seasonal spread failed to load"
spread.index = pd.to_datetime(spread.index)
common = spread.index.intersection(spy.index)
spread_al = spread.loc[common]
spy_c = spy["Close"].loc[common]
high_52w_c = spy_c.rolling(252, min_periods=60).max()
near_high_common = spy_c >= high_52w_c * 0.98
srd_common = (spread_al > 10) & near_high_common

srd = pd.Series(False, index=spy.index)
srd.loc[srd_common.index] = srd_common.values.astype(bool)
srd = srd.to_numpy()

# near-high over the FULL spy index (control group)
hi52 = spy["Close"].rolling(252, min_periods=60).max()
near_high_all = (spy["Close"] >= hi52 * 0.98).to_numpy()

atr_valid = np.isfinite(atr) & (atr > 0)

# episodes: first day of each contiguous SRD run
srd_prev = np.concatenate([[False], srd[:-1]])
episode_first = srd & ~srd_prev

print(f"\nSRD active days: {srd.sum()}   episodes: {episode_first.sum()}   "
      f"(spread covers {common[0].date()} -> {common[-1].date()})")
print(f"near-high days (all): {near_high_all.sum()}   total valid ATR days: {atr_valid.sum()}\n")

def rate(mask, flag, valid):
    m = mask & valid
    if m.sum() == 0:
        return np.nan, 0
    return 100.0 * flag[m].mean(), int(m.sum())

rows = []
for label, N in HORIZONS.items():
    maxdd, wvalid = maxdrop_for_horizon(N)
    valid = wvalid & atr_valid
    ddatr = np.where(valid, maxdd / atr, np.nan)
    flag = ddatr >= THRESH_ATR

    r_all, n_all = rate(np.ones(n, bool), flag, valid)
    r_nh, n_nh = rate(near_high_all, flag, valid)
    r_srd, n_srd = rate(srd, flag, valid)
    r_ep, n_ep = rate(episode_first, flag, valid)

    # median maxDD in ATR for SRD vs all (context)
    med_srd = np.nanmedian(ddatr[srd & valid]) if (srd & valid).sum() else np.nan
    med_all = np.nanmedian(ddatr[valid])
    med_nh = np.nanmedian(ddatr[near_high_all & valid])

    rows.append({
        "horizon": label,
        "P(>=2ATR) all%": round(r_all, 1),
        "P near-high%": round(r_nh, 1),
        "P SRD-day%": round(r_srd, 1),
        "P SRD-episode%": round(r_ep, 1),
        "medDD/ATR all": round(med_all, 2),
        "medDD/ATR nearHi": round(med_nh, 2),
        "medDD/ATR SRD": round(med_srd, 2),
        "n_srd": n_srd, "n_ep": n_ep,
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(df.to_string(index=False))

print("\nInterpretation columns:")
print("  P(>=2ATR) all%   = unconditional base rate, every market day")
print("  P near-high%     = fair control: days SPY within 2% of 52w high")
print("  P SRD-day%       = conditional on SRD active (day-level, overlapping)")
print("  P SRD-episode%   = conditional on SRD, first day of each run only (overlap-free)")
