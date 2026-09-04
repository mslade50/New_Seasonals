"""Decisive test: does the STANDALONE backtester (recompute-each-run, single
adjusted basis) phantom-fill this EWZ trade? Compare two consistent runs:
  RUN-RAW : pre-ex basis (what live saw 6/9)
  RUN-ADJ : post-15 ex-div basis (what a re-download gives now)
For each, recompute limit = sig_close - k*ATR and test forward-bar lows.
Also test the persistence-window straddle (signal 6/9, holding_days forward,
ex-div 6/15 lands INSIDE the window)."""
import numpy as np
import pandas as pd

# Real EWZ bars (from live yfinance), raw and adjusted (factor f=0.990570 pre-6/15)
# date: (raw O,H,L,C) and (adj O,H,L,C)
raw = {
 "2026-06-03": (35.01,35.13,34.45,34.64),
 "2026-06-04": (34.90,35.03,34.68,34.78),
 "2026-06-05": (34.29,34.60,33.95,34.01),
 "2026-06-08": (33.98,34.10,33.53,33.69),
 "2026-06-09": (33.99,34.28,33.58,33.92),
 "2026-06-10": (33.74,34.07,33.63,33.78),
 "2026-06-11": (34.00,34.93,33.86,34.81),
 "2026-06-12": (34.99,35.22,34.84,35.10),
 "2026-06-15": (35.46,35.62,34.55,34.64),  # EX-DIV today, NOT scaled
 "2026-06-16": (34.44,34.59,34.23,34.41),
}
f = 0.990570
def adj(d, t):
    o,h,l,c = raw[t]
    return (o*f,h*f,l*f,c*f) if d=="adj" and t < "2026-06-15" else (o,h,l,c)

def run(basis, k=0.5, holding_days=10, sig="2026-06-09"):
    dates = sorted(raw.keys())
    # build a tiny df with a stable ATR proxy: use the 0.82-ish live ATR for the
    # sig bar so the limit lands ~33.51 in RAW basis (matches the live order).
    # We test scale-invariance, so the exact ATR value doesn't change the verdict;
    # we scale it by f in the adj basis exactly as calculate_indicators would.
    atr_raw = 0.82
    atr = atr_raw * (f if (basis=="adj" and sig < "2026-06-15") else 1.0)
    o,h,l,c = adj(basis, sig)
    sig_close = c
    limit = sig_close - k*atr
    si = dates.index(sig)
    fill = None
    for w in range(1, holding_days+1):
        if si+w >= len(dates): break
        d = dates[si+w]
        bo,bh,bl,bc = adj(basis, d)
        if bl <= limit:
            fill = (d, bl, limit); break
    return sig_close, atr, limit, fill

print("RUN-RAW (live/pre-ex basis, single consistent series):")
sc,a,lim,fil = run("raw")
print(f"  sig_close={sc:.4f} ATR={a:.4f} limit={lim:.4f} -> fill={fil}")

print("\nRUN-ADJ (post-15 ex-div basis, single consistent re-download):")
sc,a,lim,fil = run("adj")
print(f"  sig_close={sc:.4f} ATR={a:.4f} limit={lim:.4f} -> fill={fil}")

print("\n=> If BOTH say no-fill (or both same fill date), the standalone backtester")
print("   is SCALE-INVARIANT and does NOT phantom-fill from re-download alone.")
print()
print("PERSISTENCE-WINDOW STRADDLE CHECK (the one place a single run CAN mix bases):")
print("  signal 6/9, holding_days=10 -> forward bars 6/10..6/24 INCLUDE the 6/15 ex-date.")
print("  In a current re-download: 6/10-6/12 lows are scaled by f (pre-ex),")
print("  6/15+ lows are NOT scaled. The limit (from 6/9 sig bar) is ALSO scaled by f.")
print("  So pre-ex forward bars stay scale-consistent with the limit (no phantom),")
print("  but post-ex forward bars are in a DIFFERENT basis than the scaled limit:")
# limit scaled by f; a post-ex bar is unscaled. Does an unscaled post-ex low
# spuriously clear a down-scaled limit? Down-scaling the limit makes it LOWER,
# i.e. HARDER to fill against an unscaled (higher) post-ex low -> conservative,
# not a phantom. Demonstrate:
atr_raw=0.82; k=0.5
limit_adj = (33.92*f) - k*(atr_raw*f)     # 6/9 limit in adj basis
postex_low_1516 = min(34.55, 34.23)        # 6/15,6/16 lows (unscaled)
print(f"  scaled 6/9 limit={limit_adj:.4f}; earliest post-ex low (unscaled)={postex_low_1516:.4f}; "
      f"fill={postex_low_1516<=limit_adj}")
print("  => post-ex bars are HIGHER than the down-scaled limit -> straddle is CONSERVATIVE, no phantom.")
