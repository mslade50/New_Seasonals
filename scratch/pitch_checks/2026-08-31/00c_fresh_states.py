"""Enumerate the fresh/rare states on the 2026-08-28 tape. Survey input, not a check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-28")

# --- 1. watchlist 21 trigger: ^TNX vs trailing-252 high (PIT)
px = load_prices(["^TNX", "TLT", "IEF", "GLD", "SLV", "GDX", "NEM", "SPY", "QQQ", "SMH",
                  "DX-Y.NYB", "UUP", "^VIX", "^VIX3M", "^SKEW", "XLU", "VNQ", "HYG", "LQD"])
tnx = px["^TNX"]["Close"]
roll_hi = tnx.rolling(252).max()
frac = tnx / roll_hi
print("=== ^TNX PIT proximity to trailing-252 high")
print(tnx.tail(6).to_string())
for d in tnx.index[-6:]:
    print(f"  {d.date()}  tnx={tnx[d]:.3f}  hi252={roll_hi[d]:.3f}  frac={100*frac[d]:.3f}%  (rung >= 99.75)")

# --- 2. metals flush magnitude
print("\n=== 2026-08-28 one-day moves in ATR units, metals complex")
for t in ["GLD", "SLV", "GDX", "NEM"]:
    d = px[t]
    atr = wilder_atr(d["High"], d["Low"], d["Close"], 14)
    r1 = d["Close"].pct_change()
    x = r1 / (atr / d["Close"].shift(1))
    print(f"  {t}: ret1d={100*r1[ASOF]:.2f}%  atr_units={x[ASOF]:.2f}  "
          f"pctile of |atr_units| trailing-252 = {100*(x.loc[:ASOF].tail(252) <= x[ASOF]).mean():.1f}")
    # how many days since 2010 were <= this in atr units
    sub = x.loc["2010":]
    print(f"       days since 2010 with atr_units <= today: {(sub <= x[ASOF]).sum()} of {sub.notna().sum()}")

# --- 3. joint metals flush: GLD, SLV, GDX all down >2% same day
gl, sl, gd = (px[t]["Close"].pct_change() for t in ["GLD", "SLV", "GDX"])
joint = (gl <= -0.02) & (sl <= -0.02) & (gd <= -0.02)
print(f"\n=== joint GLD/SLV/GDX all <= -2% in one day: {int(joint.sum())} days since {joint.index[joint][0].date() if joint.any() else 'n/a'}")
print("   last 12:", [str(d.date()) for d in joint.index[joint][-12:]])

# --- 4. month position of today
spy = px["SPY"]["Close"]
idx = spy.index
mkey = pd.Series(idx.year * 100 + idx.month, index=idx)
last_of_month = mkey != mkey.shift(-1)
print(f"\n=== is 2026-08-28 the last August session in the cache? {bool(last_of_month.get(ASOF, False))}")
print("   sessions after ASOF in cache:", [str(d.date()) for d in idx[idx > ASOF]])

# --- 5. growth/value 63d rank split
print("\n=== 63d return rank, growth vs value complex (from tape)")
import json
T = json.load(open(Path(__file__).resolve().parents[3] / "data" / "pitch_tape.json"))["tickers"]
for t in ["QQQ", "^NDX", "XLK", "SMH", "XLY", "XLC", "SPY", "XLF", "XLV", "XLE", "XLP", "XLI", "IWM"]:
    r = T[t]
    print(f"  {t:<6} r63={r['rank_63d']:>5.1f} ret63={r['ret_63d']:>7.2f}% r21={r['rank_21d']:>5.1f} r5={r['rank_5d']:>5.1f}")

# --- 6. VIX3M at a 52w low?
v3 = px["^VIX3M"]["Close"]
print(f"\n=== ^VIX3M {v3[ASOF]:.2f}, trailing-252 min {v3.loc[:ASOF].tail(252).min():.2f}, "
      f"is it the min? {v3[ASOF] <= v3.loc[:ASOF].tail(252).min() + 1e-9}")
vix = px["^VIX"]["Close"]
print(f"    VIX/VIX3M = {vix[ASOF]/v3[ASOF]:.4f}; trailing-252 pctile of ratio = "
      f"{100*((vix/v3).loc[:ASOF].tail(252) <= (vix/v3)[ASOF]).mean():.1f}")
