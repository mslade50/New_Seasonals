"""Preview of the proposed signal-card downside table for Seasonal Rank Divergence.
Anchor = fire-day CLOSE. Two candidate measures, ATR multiples [1,2,3,5]:
  - LOW-TOUCH : subsequent intraday LOW reaches close - k*ATR   (did price trade down k ATR)
  - CLOSE     : subsequent CLOSE reaches close - k*ATR          (did price close down k ATR)
ATR = Wilder-14 at the fire day. SPY 2000-2026.
"""
import os, sys

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
sys.path.insert(0, os.path.join(ROOT, "pages")); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import risk_dashboard_v2 as rd

HORIZONS = {"5d": 5, "10d": 10, "21d": 21, "42d": 42, "63d": 63}
MULTS = [1, 2, 3, 5]
ATR_N = 14

mp = pd.read_parquet(os.path.join(ROOT, "data", "master_prices.parquet"))
spy = mp[mp["ticker"] == "SPY"].copy()
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")[["Open","High","Low","Close"]].dropna()
Hh = spy["High"].to_numpy(float); Ll = spy["Low"].to_numpy(float); Cc = spy["Close"].to_numpy(float)
n = len(spy)

prev_c = np.concatenate([[np.nan], Cc[:-1]])
tr = np.maximum.reduce([Hh-Ll, np.abs(Hh-prev_c), np.abs(Ll-prev_c)])
atr = np.full(n, np.nan); atr[ATR_N] = np.nanmean(tr[1:ATR_N+1])
for i in range(ATR_N+1, n):
    atr[i] = (atr[i-1]*(ATR_N-1) + tr[i]) / ATR_N
atr_valid = np.isfinite(atr) & (atr > 0)

# SRD mask (exact)
spread = rd._load_seasonal_spread(); spread.index = pd.to_datetime(spread.index)
common = spread.index.intersection(spy.index)
spy_c = spy["Close"].loc[common]
near = spy_c >= spy_c.rolling(252, min_periods=60).max()*0.98
srd_c = (spread.loc[common] > 10) & near
srd = pd.Series(False, index=spy.index); srd.loc[srd_c.index] = srd_c.values.astype(bool); srd = srd.to_numpy()
ep = srd & ~np.concatenate([[False], srd[:-1]])
hi52 = (spy["Close"] >= spy["Close"].rolling(252, min_periods=60).max()*0.98).to_numpy()

def worst_drop(N, use_low):
    """max (Close_i - X_{i+j})/ATR_i over j=1..N, X=Low (touch) or Close."""
    worst = np.zeros(n); valid = np.ones(n, bool)
    src = Ll if use_low else Cc
    for j in range(1, N+1):
        Xs = np.concatenate([src[j:], np.full(j, np.nan)])
        worst = np.fmax(worst, Cc - Xs)
    valid[n-N:] = False
    return worst, valid

def table(mask, use_low):
    out = {}
    for lab, N in HORIZONS.items():
        drop, wv = worst_drop(N, use_low)
        v = wv & atr_valid & mask
        dd = drop[v] / atr[v]
        out[lab] = {k: round(100*np.mean(dd >= k), 1) for k in MULTS}
    return pd.DataFrame(out).T[MULTS]

print(f"SRD active days={srd.sum()}  episodes={ep.sum()}  (2001-2026)\n")
for use_low, name in [(True, "LOW-TOUCH (intraday low reaches -k ATR)"),
                      (False, "CLOSE (a close reaches -k ATR)")]:
    print(f"=== {name} ===")
    print("  P(decline >= k ATR), % — SRD episode-first:")
    print(table(ep, use_low).to_string());
    print("  baseline (all market):")
    print(table(np.ones(n, bool), use_low).to_string()); print()
