"""DEFINITIVE proof: EWZ 33.51 phantom fill mechanism.

EWZ ex-div 2026-06-15, $0.331. yfinance scales ALL pre-6/15 bars by
f = 1 - D/C_{6/12}.  Live scan ran 6/9 on RAW basis (pre-ex). Limit 33.51
was a fixed dollar price. Backtester re-run reads CURRENT adjusted cache,
where 6/9 close + forward lows are scaled DOWN.

Three regimes tested:
  (1) fully RAW (live reality): limit vs raw bars -> did it fill?
  (2) fully ADJUSTED, self-consistent: re-derive limit from adj close+ATR,
      vs adj bars -> SCALE-INVARIANT, should match (1).
  (3) MIXED basis (the bug): fixed live 33.51 vs ADJ bars -> phantom?
"""
import pandas as pd, numpy as np
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

# Raw OHLC from yfinance pull (auto_adjust=False) 6/8..6/17
raw = pd.DataFrame({
    "Open":  [33.98, 33.99, 33.74, 34.00, 34.99, 35.46, 34.44, 34.76],
    "High":  [34.10, 34.28, 34.07, 34.93, 35.22, 35.62, 34.59, 35.16],
    "Low":   [33.53, 33.58, 33.63, 33.86, 34.84, 34.55, 34.23, 33.97],
    "Close": [33.69, 33.92, 33.78, 34.81, 35.10, 34.64, 34.41, 34.11],
}, index=pd.to_datetime(["2026-06-08","2026-06-09","2026-06-10","2026-06-11",
                          "2026-06-12","2026-06-15","2026-06-16","2026-06-17"]))

D = 0.331
ex = pd.Timestamp("2026-06-15")
c_prior = raw.loc["2026-06-12", "Close"]   # last close before ex
f = (c_prior - D) / c_prior
print(f"ex-div date={ex.date()}  D={D}  C_prior(6/12)={c_prior}  factor f={f:.6f}")
print(f"=> pre-6/15 bars multiplied by {f:.6f}; 6/15+ bars unchanged\n")

# Build adjusted series the way yfinance does: pre-ex *f, on/after-ex unchanged
adj = raw.copy().astype(float)
pre = adj.index < ex
adj.loc[pre, ["Open","High","Low","Close"]] *= f

print("RAW vs ADJUSTED OHLC:")
cmp = pd.concat([raw.add_suffix("_raw"), adj[["Low","Close"]].add_suffix("_adj")], axis=1)
print(cmp.round(4).to_string())

# --- ATR(14) we can't fully recompute from 8 bars; use the cache values.
# From master_prices (current adj cache): ATR14(6/9)=0.7086 (adjusted basis).
# Raw ATR = adj_ATR / f (uniform scaling of pre-ex true ranges).
atr_adj_69 = 0.7086
atr_raw_69 = atr_adj_69 / f
close_raw_69 = raw.loc["2026-06-09","Close"]      # 33.92
close_adj_69 = adj.loc["2026-06-09","Close"]      # ~33.60

print("\n6/9 anchor:")
print(f"  RAW : close={close_raw_69:.4f}  ATR={atr_raw_69:.4f}")
print(f"  ADJ : close={close_adj_69:.4f}  ATR={atr_adj_69:.4f}")

# which k off RAW close gives 33.51?
for k in (0.25, 0.5, 0.75):
    print(f"  k={k}:  RAW limit={close_raw_69-k*atr_raw_69:.4f}   "
          f"ADJ limit={close_adj_69-k*atr_adj_69:.4f}")

LIVE = 33.51
print(f"\nLIVE staged limit (fixed dollar) = {LIVE}")

# hold window: persistent limit, assume forward bars 6/10 onward
hold = adj.index[adj.index > pd.Timestamp("2026-06-09")]

def first_fill(limit, bars_df, low_col):
    for d in hold:
        if bars_df.loc[d, low_col] <= limit:
            return d, bars_df.loc[d, low_col]
    return None, None

print("\n" + "="*72)
print("REGIME 1 — fully RAW (what actually happened live)")
print("="*72)
# self-consistent raw limit at k=0.5 (closest match to 33.51 live)
for k in (0.25, 0.5, 0.75):
    lim = close_raw_69 - k*atr_raw_69
    d,l = first_fill(lim, raw, "Low")
    print(f"  raw k={k} limit={lim:.4f}: fill={d} (low {l})" if d is not None
          else f"  raw k={k} limit={lim:.4f}: NO FILL")
d,l = first_fill(LIVE, raw, "Low")
print(f"  fixed 33.51 vs RAW lows: {'FILL '+str(d)+' low '+str(l) if d else 'NO FILL'}")

print("\n" + "="*72)
print("REGIME 2 — fully ADJUSTED, self-consistent (scale-invariant check)")
print("="*72)
for k in (0.25, 0.5, 0.75):
    lim = close_adj_69 - k*atr_adj_69
    d,l = first_fill(lim, adj, "Low")
    print(f"  adj k={k} limit={lim:.4f}: fill={d} (low {l:.4f})" if d is not None
          else f"  adj k={k} limit={lim:.4f}: NO FILL")

print("\n" + "="*72)
print("REGIME 3 — MIXED basis (the bug): fixed LIVE 33.51 vs ADJ lows")
print("="*72)
d,l = first_fill(LIVE, adj, "Low")
print(f"  fixed 33.51 vs ADJUSTED lows: {'PHANTOM FILL '+str(d)+' adj low '+format(l,'.4f') if d else 'NO FILL'}")
print(f"\n  adjusted lows in hold window:")
print(adj.loc[hold, ["Low","Close"]].round(4).to_string())
