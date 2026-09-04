"""Replay-parity gate for frag_risk_bands + sector_loss_gate (roadmap step 3).

Compares the rebuilt ledger against the pre-change snapshot:
  1. FAMILY4 trades signaled at 63d-MA10 >= 50 must show Size_Mult scaled by
     exactly 0.25 vs their old value; family trades below 50 unchanged.
  2. OVS trades in [21,44) must scale by exactly 0.75; outside, unchanged.
  3. OLV loses ~12 net-losing trades to the sector gate (study cells).
  4. No other strategy's trade count changes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
old = pd.read_parquet(ROOT / "scratch" / "ledger_pre_frag_bands.parquet")
new = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
s = frag["63d"].dropna().rolling(10, min_periods=1).mean()
s.index = pd.to_datetime(s.index).normalize()
grid = pd.date_range(s.index.min(), s.index.max(), freq="D")
sf = s.reindex(grid).ffill(limit=5)

for df in (old, new):
    df["Signal Date"] = pd.to_datetime(df["Signal Date"])

KEY = ["Strategy", "Ticker", "Signal Date"]
m = old.merge(new, on=KEY, suffixes=("_old", "_new"), how="outer", indicator=True)
m["score"] = sf.reindex(m["Signal Date"].dt.normalize()).values

FAMILY4 = {"Weak Close Decent Sznls", "SPY QQQ MonFri Reversion",
           "Monday Dip", "Indices Oversold Bounce"}

both = m[m._merge == "both"].copy()
both["ratio"] = both.Size_Mult_new / both.Size_Mult_old

fails = 0

# 1. family scaling
fam = both[both.Strategy.isin(FAMILY4) & both.score.notna()]
hi = fam[fam.score >= 50]
lo = fam[fam.score < 50]
ok_hi = np.isclose(hi.ratio, 0.25, atol=1e-6).all()
ok_lo = np.isclose(lo.ratio, 1.0, atol=1e-6).all()
print(f"1. FAMILY4 >=50: N={len(hi)}, all ratio 0.25: {ok_hi} | "
      f"<50: N={len(lo)}, all 1.0: {ok_lo}")
fails += (not ok_hi) + (not ok_lo)

# pre-2016 family trades (no score) must be unchanged
fam_ns = both[both.Strategy.isin(FAMILY4) & both.score.isna()]
ok_ns = np.isclose(fam_ns.ratio, 1.0, atol=1e-6).all()
print(f"   FAMILY4 no-score (pre-2016): N={len(fam_ns)}, all 1.0: {ok_ns}")
fails += (not ok_ns)

# 2. OVS mid-band
ovs = both[(both.Strategy == "Overbot Vol Spike") & both.score.notna()]
mid = ovs[(ovs.score >= 21) & (ovs.score < 44)]
out = ovs[(ovs.score < 21) | (ovs.score >= 44)]
ok_mid = np.isclose(mid.ratio, 0.75, atol=1e-6).all()
ok_out = np.isclose(out.ratio, 1.0, atol=1e-6).all()
print(f"2. OVS [21,44): N={len(mid)}, all ratio 0.75: {ok_mid} | "
      f"outside: N={len(out)}, all 1.0: {ok_out}")
fails += (not ok_mid) + (not ok_out)

# 3. OLV sector gate drops
olv_dropped = m[(m._merge == "left_only") & (m.Strategy == "Oversold Low Volume")]
olv_added = m[(m._merge == "right_only") & (m.Strategy == "Oversold Low Volume")]
print(f"3. OLV dropped: {len(olv_dropped)} trades, {olv_dropped.R_Multiple_old.sum():+.1f}R "
      f"(study: 12, -2.7R) | spuriously added: {len(olv_added)}")
jun = olv_dropped[(olv_dropped["Signal Date"] >= "2026-05-25") & (olv_dropped["Signal Date"] <= "2026-06-30")]
print(f"   dropped in the June-2026 window: {len(jun)} ({jun.R_Multiple_old.sum():+.1f}R)")

# 4. other strategies: counts unchanged
others = m[~m.Strategy.isin(FAMILY4 | {"Overbot Vol Spike", "Oversold Low Volume"})]
odd = others[others._merge != "both"]
print(f"4. non-target strategies with count changes: {len(odd)}")
fails += (len(odd) > 0)

# R_Multiple must be untouched everywhere (R is per unit risk)
r_drift = (both.R_Multiple_new - both.R_Multiple_old).abs().max()
print(f"5. max |R_Multiple drift| on matched trades: {r_drift:.6f}")
fails += (r_drift > 1e-6)

print("\nPARITY:", "PASS" if fails == 0 else f"FAIL ({fails} checks)")
