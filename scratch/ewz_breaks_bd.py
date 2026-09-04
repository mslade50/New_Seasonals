"""Test breaks (b) ex-date inside persistence window and (d) ATR computed
ACROSS the ex-date boundary. Show whether either is operative here.

Key fact: in the BACKTEST, every bar comes from ONE adjusted series.
yfinance adjustment is CONTINUOUS across the ex-date in the adjusted series
(the gap is removed). So an ATR computed on the adjusted series does NOT see
a dividend gap. The gap only appears in the RAW series.
"""
import pandas as pd, numpy as np

# raw closes incl across the 6/15 ex-div
raw = pd.DataFrame({
    "High":  [34.10, 34.28, 34.07, 34.93, 35.22, 35.62, 34.59, 35.16],
    "Low":   [33.53, 33.58, 33.63, 33.86, 34.84, 34.55, 34.23, 33.97],
    "Close": [33.69, 33.92, 33.78, 34.81, 35.10, 34.64, 34.41, 34.11],
}, index=pd.to_datetime(["2026-06-08","2026-06-09","2026-06-10","2026-06-11",
                          "2026-06-12","2026-06-15","2026-06-16","2026-06-17"]))
D, ex = 0.331, pd.Timestamp("2026-06-15")
f = (raw.loc["2026-06-12","Close"] - D)/raw.loc["2026-06-12","Close"]
adj = raw.copy().astype(float); adj.loc[adj.index<ex] *= f

def tr(df):
    hl = df["High"]-df["Low"]
    hc = (df["High"]-df["Close"].shift()).abs()
    lc = (df["Low"]-df["Close"].shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1)

print("True Range around 6/15 ex-date:")
out = pd.DataFrame({"TR_raw": tr(raw), "TR_adj": tr(adj)})
print(out.round(4).to_string())
print("\n(d) ATR-across-ex-date: the 6/15 TR_raw includes a spurious")
print("    dividend GAP (prev raw close 35.10 -> 6/15 low 34.55), but in the")
print("    ADJUSTED series the prior close is scaled so the gap shrinks.")
print(f"    TR_raw(6/15)={out.loc['2026-06-15','TR_raw']:.4f}  "
      f"TR_adj(6/15)={out.loc['2026-06-15','TR_adj']:.4f}")
print("    -> The backtester uses the ADJUSTED series everywhere, so its ATR")
print("       does NOT get the dividend-gap inflation. Break (d) is NOT active")
print("       in this single-series backtest. It WOULD bite a RAW-fed ATR.")

print("\n(b) persistence window straddling ex-date:")
print("    The EWZ fill (6/10) is BEFORE the 6/15 ex-date, so the relevant")
print("    forward bars are entirely pre-ex and ALL scaled by f. No straddle.")
print("    A straddle would only matter for a still-open GTC limit whose hold")
print("    window crosses 6/15: pre-ex bars scaled, post-ex bars NOT, while a")
print("    RAW-basis anchor compares against both -> mixed within the window.")
