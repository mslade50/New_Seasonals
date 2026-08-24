"""b0 — recon: reproduce the tape's live state for C3 (copper thrust) and C7
(energy z10 cluster) BEFORE measuring anything.

Verifies:
  1. tape z10 definition reproduced exactly for the five energy names
  2. FCX's live state: 5d thrust, fresh-52w-high, volume ratio, denominator roll
  3. what vehicles exist in master_prices and how deep
  4. the C7 complex membership + its usable span (all-names-valid)
  5. earnings distance for FCX (the single-name tail the ETFs do not carry)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, close_panel, rolling_on_valid  # noqa: E402

pd.set_option("display.width", 200)

# --- the tape's own z10, reproduced -----------------------------------------
def tape_z10(close: pd.Series, n: int = 10) -> pd.Series:
    """build_pitch_state._metrics_for: n-day return / (21d sd * sqrt(n))."""
    r = close.pct_change(n)
    vol21 = close.pct_change().rolling(21).std()
    return r / (vol21 * np.sqrt(n))


ENERGY = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
          "HAL", "WMB"]
COPPER = ["FCX", "COPX", "HG=F", "XME", "XLB", "SCCO", "TECK", "RIO", "BHP",
          "VALE", "HBM"]

px = load_prices(sorted(set(ENERGY + COPPER + ["SPY"])))

print("=== 1. tape z10 reproduction (tape bar 2026-08-21) ===")
TAPE_Z10 = {"VLO": 2.56, "COP": 2.35, "XLE": 2.18, "XOP": 2.10, "CVX": 2.04,
            "USO": 1.18, "OXY": 1.32, "SLB": 0.65, "EOG": 1.63, "HAL": 1.48,
            "WMB": 0.02, "FCX": 1.00, "XME": 0.36, "XLB": 0.32}
rows = []
for t, want in TAPE_Z10.items():
    z = tape_z10(px[t]["Close"]).iloc[-1]
    rows.append({"ticker": t, "tape": want, "mine": round(float(z), 2),
                 "match": abs(float(z) - want) < 0.011})
print(pd.DataFrame(rows).to_string(index=False))
print("all match:", all(r["match"] for r in rows))

print("\n=== 2. FCX live state ===")
c = px["FCX"]["Close"]
v = px["FCX"]["Volume"]
r5 = c.pct_change(5).iloc[-1] * 100
hi252 = c.rolling(252).max().iloc[-1]
print(f"  5d return          {r5:+.2f}%   (tape 15.30)")
print(f"  close / 252d max   {c.iloc[-1] / hi252 - 1:+.4%}  (tape dist_52w_high 0.00)")
print(f"  1d return          {c.pct_change(1).iloc[-1]:+.2%}   (tape 7.64)")
print(f"  vol / 63d mean     {v.iloc[-1] / v.rolling(63).mean().iloc[-1]:.2f}  (tape 2.0)")
print(f"  above 200d         {c.iloc[-1] / c.rolling(200).mean().iloc[-1] - 1:+.2%}")
# denominator roll: how much of today's 5d return is today's own move vs the
# t-5 bar rolling off?
own = c.pct_change(1).iloc[-1]
roll_off = c.pct_change(1).iloc[-5]  # the bar leaving the window tomorrow
print(f"  own-day move {own:+.2%} vs the bar rolling out of the window "
      f"{roll_off:+.2%}  -> own move dominates: {abs(own) > abs(roll_off)}")

print("\n=== 3. vehicle depth ===")
for t in COPPER + ENERGY:
    if t not in px:
        print(f"  {t:8s} MISSING")
        continue
    d = px[t].index
    print(f"  {t:8s} n={len(d):6d}  {d[0].date()} .. {d[-1].date()}")

print("\n=== 4. C7 complex usable span ===")
pan = close_panel(ENERGY)
z = pd.DataFrame({t: tape_z10(px[t]["Close"]) for t in ENERGY}).reindex(pan.index)
allvalid = z.notna().all(axis=1)
first = allvalid.idxmax()
print(f"  complex = {ENERGY}")
print(f"  first day all 11 have a valid z10: {first.date()}")
print(f"  usable days: {int(allvalid.sum())}  "
      f"(of {len(pan)} panel rows since 2000)")
holes = pan.isna().sum()
print("  per-ticker NaN rows inside the panel (calendar mismatch check):")
print("   ", {t: int(h) for t, h in holes.items() if h})
cnt = (z >= 2.0).sum(axis=1).where(allvalid)
print(f"  today's count (z10>=2, tape bar): {int(cnt.iloc[-1])} "
      f"-> {sorted(z.columns[(z.iloc[-1] >= 2.0).values])}")
print("  count distribution over usable days:")
print(cnt.value_counts().sort_index().to_string())

print("\n=== 5. FCX earnings distance ===")
try:
    ec = pd.read_parquet("data/earnings_calendar.parquet")
    col = "date" if "date" in ec.columns else ec.columns[0]
    f = ec[ec["ticker"] == "FCX"].copy()
    f[col] = pd.to_datetime(f[col])
    nxt = f[f[col] >= pd.Timestamp("2026-08-21")].sort_values(col).head(2)
    prv = f[f[col] < pd.Timestamp("2026-08-21")].sort_values(col).tail(1)
    print("  prior:", prv[col].dt.date.tolist(), " next:", nxt[col].dt.date.tolist())
except Exception as exc:  # noqa: BLE001
    print("  earnings lookup failed:", exc)
