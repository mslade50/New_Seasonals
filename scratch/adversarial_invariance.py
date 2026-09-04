"""Adversarial: try to BREAK scale-invariance of the within-run check.
Two attack vectors:
 (1) rounding of the limit to 2dp (verify_fills line 330) vs unrounded bars
 (2) a self-consistent ADJ re-run where rounding tips a knife-edge into a fill
Also confirm the MIXED-basis is what actually flips no-fill->fill for EWZ.
"""
f = 33.6001/33.92
atr_raw = 0.731428
mult = 0.25
c_raw = 33.69

# RAW low forward
raw_lows = {'6/9':33.58, '6/10':33.625}
adj_lows = {k: v*f for k,v in raw_lows.items()}

# ---- WITHIN-RUN, ADJUSTED basis, recompute limit, NO rounding (backtester engines) ----
limit_adj_unrounded = (c_raw*f) - mult*(atr_raw*f)
print("=== within-run ADJ, unrounded (backtest engines) ===")
for k in raw_lows:
    print(f"  {k}: adj_low={adj_lows[k]:.4f} <= limit {limit_adj_unrounded:.4f} ? {adj_lows[k] <= limit_adj_unrounded}")

# ---- WITHIN-RUN, ADJUSTED basis, recompute limit, ROUNDED to 2dp (verify_fills) ----
# verify_fills would, on a post-ex re-pull, read signal_close & atr from the SHEET (frozen pre-ex),
# NOT from re-adjusted data. So a "self-consistent ADJ" verify_fills run does NOT exist in practice.
# But test the hypothetical: if the sheet held adjusted values, limit rounds:
limit_adj_rounded = round(limit_adj_unrounded, 2)
print(f"\n=== within-run ADJ, ROUNDED limit={limit_adj_rounded} ===")
for k in raw_lows:
    print(f"  {k}: adj_low={adj_lows[k]:.4f} <= {limit_adj_rounded} ? {adj_lows[k] <= limit_adj_rounded}")

# ---- THE ACTUAL verify_fills PATH: frozen sheet Entry=33.69 ATR=0.73 (rounded in sheet!) ----
# Sheet stores ATR rounded to 2dp = 0.73, Entry rounded = 33.69
sheet_entry = 33.69
sheet_atr = 0.73   # rounded as stored
limit_frozen = round(sheet_entry - mult*sheet_atr, 2)
print(f"\n=== verify_fills REAL: frozen sheet Entry={sheet_entry} ATR={sheet_atr} -> limit={limit_frozen} ===")
print("   (note: uses rounded ATR 0.73, not 0.7314)")
for k in raw_lows:
    print(f"  {k}: ADJ low={adj_lows[k]:.4f} <= {limit_frozen} ? {adj_lows[k] <= limit_frozen}  | RAW low={raw_lows[k]:.4f} <= {limit_frozen} ? {raw_lows[k] <= limit_frozen}")

# Knife-edge cushion analysis
true_low_min = min(raw_lows.values())
print(f"\nCushion (true raw low - limit) = {true_low_min:.4f} - {limit_frozen} = {true_low_min - limit_frozen:.4f}")
print(f"Dividend downshift on ~$33.6 low = low*(1-f) = {true_low_min*(1-f):.4f}")
print(f"Downshift {true_low_min*(1-f):.4f} > cushion {true_low_min-limit_frozen:.4f} ? -> phantom: {true_low_min*(1-f) > (true_low_min-limit_frozen)}")
