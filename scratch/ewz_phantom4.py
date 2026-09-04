"""Forensic part 4: run the EXACT backtester limit_pers_025 fill check on the
FULLY-RE-ADJUSTED series (post 6/15 ex-div) vs RAW, for the 6/8 OLV signal.

This decides: is the phantom (a) scale-invariant self-consistent re-run, or
(b) ex-date discontinuity inside the persistence window, or
(c) live-raw-limit vs re-adjusted-cache mixed basis.
"""
import pandas as pd

pd.set_option("display.width", 200)

# Build the post-ex-div ADJUSTED series the nightly rebuild WILL produce.
# yfinance auto_adjust=True after 6/15 ex-div scales every pre-6/15 bar by f.
f = 0.99057
raw = pd.DataFrame({
    "Open":  {"2026-06-05": 34.29, "2026-06-08": 33.98, "2026-06-09": 33.99,
              "2026-06-10": 33.74, "2026-06-11": 34.00, "2026-06-12": 34.99},
    "High":  {"2026-06-05": 34.60, "2026-06-08": 34.10, "2026-06-09": 34.28,
              "2026-06-10": 34.07, "2026-06-11": 34.93, "2026-06-12": 35.22},
    "Low":   {"2026-06-05": 33.95, "2026-06-08": 33.53, "2026-06-09": 33.58,
              "2026-06-10": 33.63, "2026-06-11": 33.86, "2026-06-12": 34.84},
    "Close": {"2026-06-05": 34.01, "2026-06-08": 33.69, "2026-06-09": 33.92,
              "2026-06-10": 33.78, "2026-06-11": 34.81, "2026-06-12": 35.10},
})
raw.index = pd.to_datetime(raw.index)

# fully re-adjusted (all these bars are pre-6/15 ex, so all * f)
adj_post = (raw * f).round(6)

# ATR proxy on each basis: use the cache's current ATR ~0.754 for adj; for raw, /f.
atr_adj = 0.7540
atr_raw = atr_adj / f

def run_limit_pers_025(series, atr_at_sig, sig_date, hold=10, mult=0.25, direction="Long"):
    """Mirror backtester is_limit_pers_025: limit = sig_close - mult*ATR, scan forward lows."""
    idx = series.index.tolist()
    sig_i = idx.index(pd.to_datetime(sig_date))
    sig_close = series["Close"].iloc[sig_i]
    limit_price = sig_close - atr_at_sig * mult
    for wait_i in range(1, hold + 1):
        ci = sig_i + wait_i
        if ci >= len(series):
            break
        dl = series["Low"].iloc[ci]
        if dl <= limit_price:
            return True, idx[ci].date(), limit_price, dl
    return False, None, limit_price, None

print("Signal = OLV (Oversold Low Volume) Long, -0.25 ATR persistent, sig bar 2026-06-08")
print(f"f={f}  atr_adj={atr_adj}  atr_raw={atr_raw:.4f}\n")

# (a) RAW self-consistent (the live universe): limit from raw close, vs raw lows
r = run_limit_pers_025(raw, atr_raw, "2026-06-08")
print(f"(A) RAW self-consistent  -> filled={r[0]} on {r[1]}  limit={r[2]:.4f}  fill_low={r[3]}")

# (b) ADJ self-consistent (single re-run after ex-div): limit re-derived on adj, vs adj lows
a = run_limit_pers_025(adj_post, atr_adj, "2026-06-08")
print(f"(B) ADJ self-consistent  -> filled={a[0]} on {a[1]}  limit={a[2]:.4f}  fill_low={a[3]}")

# (c) MIXED: live raw limit 33.51 frozen, compared vs re-adjusted adj lows
LIVE_LIMIT = 33.51
idx = adj_post.index.tolist()
sig_i = idx.index(pd.to_datetime("2026-06-08"))
mixed_fill = None
for wait_i in range(1, 11):
    ci = sig_i + wait_i
    if ci >= len(adj_post):
        break
    dl = adj_post["Low"].iloc[ci]
    if dl <= LIVE_LIMIT:
        mixed_fill = (idx[ci].date(), dl)
        break
print(f"(C) MIXED live-raw-limit {LIVE_LIMIT} vs re-adj lows -> "
      f"{'FILLED ' + str(mixed_fill) if mixed_fill else 'no fill'}")

print("\nRe-adjusted (post-ex) lows around window:")
print(adj_post["Low"].round(4).to_string())
print(f"\nlimit on raw basis = 33.69 - 0.25*{atr_raw:.4f} = {33.69 - 0.25*atr_raw:.4f}")
print(f"limit on adj basis = {adj_post['Close'].loc['2026-06-08']:.4f} - 0.25*{atr_adj} = {adj_post['Close'].loc['2026-06-08'] - 0.25*atr_adj:.4f}")
