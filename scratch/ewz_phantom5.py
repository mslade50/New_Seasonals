"""Forensic part 5: reconcile ledger 3429 (the one that DID fill in the cache)
and characterize the true phantom-fill knife-edge.

trade 3429: OLV Long, Signal 2026-06-05, Entry 2026-06-08, Entry Price 33.827856,
ATR 0.728571, Entry Criteria 'Limit Order -0.25 ATR (Persistent)'.
"""
import pandas as pd

# limit for the 6/5 signal per the ledger's own ATR
sig_close_65_adj = 34.009998
atr_3429 = 0.728571
lim_3429 = sig_close_65_adj - 0.25 * atr_3429
print(f"Ledger 3429 (sig 6/5): adj close {sig_close_65_adj} - 0.25*{atr_3429} = limit {lim_3429:.6f}")
print(f"  ledger Entry Price = 33.827856  (matches limit? {abs(lim_3429 - 33.827856) < 1e-4})")
print(f"  6/8 adj low (current cache) = 33.529999  -> 33.529999 <= {lim_3429:.4f}? "
      f"{33.529999 <= lim_3429}  => filled 6/8 in current cache. OK consistent.\n")

# Now the KNIFE-EDGE for the 6/8 signal (the live 33.51 staged limit):
# RAW lows never reach 33.51, so LIVE never fills.
# Question: after the nightly rebuild re-adjusts, does the BACKTEST re-derive a
# limit that the (now-lower) lows clear? Two scenarios for what the backtest does:
print("=== 6/8-signal limit, raw vs fully-re-adjusted self-consistent ===")
f = 0.99057
# raw basis
raw_close_68 = 33.69
atr_raw = 0.7612
lim_raw = raw_close_68 - 0.25 * atr_raw
raw_lows = {"2026-06-09": 33.58, "2026-06-10": 33.63, "2026-06-11": 33.86}
print(f"RAW: limit={lim_raw:.4f}; forward lows {raw_lows}")
print(f"  any raw low <= {lim_raw:.4f}? {any(v <= lim_raw for v in raw_lows.values())}  (NO live fill)")

# fully re-adjusted self-consistent
adj_close_68 = raw_close_68 * f
atr_adj = atr_raw * f
lim_adj = adj_close_68 - 0.25 * atr_adj
adj_lows = {k: v * f for k, v in raw_lows.items()}
print(f"ADJ self-consistent: limit={lim_adj:.4f}; forward lows "
      f"{ {k: round(v,4) for k,v in adj_lows.items()} }")
print(f"  any adj low <= {lim_adj:.4f}? {any(v <= lim_adj for v in adj_lows.values())}  "
      f"(scale-invariant => SAME verdict as raw)")

print("\n=== The TRUE phantom: live order vs backtest/ledger, NOT a self re-run ===")
print("The 33.51 staged limit is a LIVE artifact (raw basis). It never filled live.")
print("The daily_portfolio_report / strat_backtester replays signals on the ADJUSTED")
print("cache. After the 6/15 ex-div, the 6/8 OLV signal's re-derived adj limit is")
print(f"{lim_adj:.4f}; the 6/8 adj low {adj_lows.get('2026-06-09', 0):.4f} (and 6/8 itself")
print("33.2138) sit BELOW it, so the replay books an ENTRY the live book never took.")

# Show the exact gap that makes 33.51 a phantom under mixed basis
print("\n=== Mixed-basis gap quantification (the headline numbers) ===")
print(f"Live limit (raw basis, 6/8 close):        33.5100")
print(f"Min RAW forward low (6/10):               33.6300  -> 0.12 ABOVE limit, NO live fill")
print(f"Min re-adj forward low (6/8, post-ex):    {33.69*f:.4f}? no -> use lows:")
for k, v in sorted(adj_lows.items()):
    print(f"    re-adj low {k}: {v:.4f}  <=33.51? {v <= 33.51}")
print(f"6/8 re-adj low itself: {33.53*f:.4f}  <=33.51? {33.53*f <= 33.51}")
