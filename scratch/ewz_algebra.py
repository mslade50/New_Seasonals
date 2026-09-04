"""Work the scale-invariance algebra and the mixed-basis live-vs-backtest case.

The two competing theories:
  (A) Pure re-download adjustment phantom-fills in a SINGLE consistent run.
  (B) Phantom fill only via MIXED basis (live-staged 33.51 vs re-adjusted bars).

Limit modes that matter for EWZ/OLV: "Limit Order -0.25 ATR (Persistent)".
  backtester.py 2088-2101:
     limit = sig_close - 0.25*ATR      (both from adjusted df at sig_idx)
     fill if  day_low <= limit         (day_low from same adjusted df, forward bar)
"""
import numpy as np

# ---- Observed 6/9 EWZ from the (pre-ex-div) cache that generated the live order
sig_close_live = 33.92      # 6/9 close, live basis
# Live staged limit was 33.51. Solve the implied ATR the live scan used:
# 33.51 = 33.92 - 0.25*ATR  -> ATR = (33.92-33.51)/0.25 = 1.64  (close-0.25ATR)
# But our recomputed ATR on the cache = 0.7086 -> limit 33.74, not 33.51.
# 33.51 from 0.5 ATR? 33.92 - 0.5*1.64=... let's just report both candidate mults.
for k, lbl in [(0.25, "0.25"), (0.5, "0.5"), (1.0, "1.0")]:
    implied_atr = (sig_close_live - 33.51) / k
    print(f"If live mult={lbl}: implied live ATR = {implied_atr:.4f}  (cache ATR ~0.71)")
print("-> live 33.51 implies ATR ~1.64 at 0.25 mult, i.e. a DIFFERENT ATR/close basis than the cache I can read.")
print()

print("=" * 70)
print("THEORY A: pure multiplicative back-adjustment, single consistent run")
print("=" * 70)
# Dividend D goes ex on date X. yfinance scales EVERY bar with date < X by
# factor f = 1 - D/Close_pre_ex(X-1)  (auto_adjust). f < 1.
# In a SINGLE re-run, sig_idx bars AND forward bars BOTH pre-ex => both * f.
f = 0.97  # example factor (3% div)
sig_close = 33.92
atr = 0.7086
limit = sig_close - 0.25 * atr
day_low = 33.58   # 6/9 low (the bar right after a 6/9 close signal would be 6/10)
print(f"UNADJUSTED:  limit={limit:.4f}  day_low={day_low:.4f}  fill={day_low<=limit}")
# Now apply f to BOTH the sig bar (close, atr) and the forward bar (low)
sig_close_a = sig_close * f
atr_a = atr * f            # ATR is a TR-based rolling mean of price diffs -> scales by f
limit_a = sig_close_a - 0.25 * atr_a
day_low_a = day_low * f
print(f"ADJUSTED:    limit={limit_a:.4f}  day_low={day_low_a:.4f}  fill={day_low_a<=limit_a}")
print(f"  limit_a = f*limit ? {np.isclose(limit_a, f*limit)}   day_low_a = f*day_low ? {np.isclose(day_low_a, f*day_low)}")
print("  => inequality (day_low<=limit) is SCALE-INVARIANT. Same fill verdict. THEORY A REFUTED for the")
print("     same-basis comparison: adjustment alone does NOT flip a no-fill into a fill in one run.")
print()

print("=" * 70)
print("THEORY B: MIXED BASIS — live-staged 33.51 (pre-ex) vs re-adjusted bars")
print("=" * 70)
print("This is NOT how backtester.py works internally (it recomputes the limit each run),")
print("so the standalone backtester does NOT phantom-fill from this. But the user's")
print("complaint ('the backtester thinks it filled at 33.51') maps to comparing a FIXED")
print("33.51 against the post-ex re-adjusted lows:")
live_limit = 33.51
# After ex-div, every pre-ex low is scaled DOWN by f, so an unfilled-live limit
# can now be >= the adjusted low.
for f in (0.99, 0.97, 0.95):
    low_pre = 33.58       # the live low that did NOT reach 33.51
    low_adj = low_pre * f
    print(f"  f={f}: live low {low_pre} -> adjusted low {low_adj:.4f}; "
          f"fixed 33.51 filled? {low_adj <= live_limit}")
print("  => A FIXED dollar limit vs re-adjusted lows DOES phantom-fill once f drops the low below 33.51.")
