"""Q2: reconstruct rd2 fragility signal components from master_prices and
attribute the 63d composite on high-frag episode dates (esp. Feb-Mar 2026).

Caveat: current-vintage reconstruction (same caveat as the frozen parquet's
own history). Uses adjusted closes from master_prices; signal states are
point-in-time (rolling percentiles) but edge weights are full-sample.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
if not hasattr(st, "fragment"):  # old local streamlit; decorator only used by UI
    def _fragment(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    st.fragment = _fragment

from pages.risk_dashboard_v2 import (  # noqa: E402
    compute_da_signal, compute_vix_range_compression, compute_defensive_leadership,
    compute_fomc_signal, compute_low_ar_signal, compute_seasonal_divergence_signal,
    compute_dispersion_signal, load_horizon_stats, _signal_edge,
    _compute_calm_multiplier_series, compute_fragility_timeseries,
    SECTOR_ETFS, HORIZON_DAYS, HORIZON_DECAY_DD,
)
from abs_return_dispersion import SP500_TICKERS  # noqa: E402

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
mp["date"] = pd.to_datetime(mp["date"])

def get_ohlc(tkr):
    d = mp[mp.ticker == tkr].set_index("date").sort_index()
    return d[["Open", "High", "Low", "Close", "Volume"]]

def get_closes(tickers):
    d = mp[mp.ticker.isin(tickers)]
    return d.pivot_table(index="date", columns="ticker", values="Close").sort_index()

START = "2013-01-01"
spy_df = get_ohlc("SPY").loc[START:]
spy_close = spy_df["Close"]
vix_close = get_closes(["^VIX"]).loc[START:, "^VIX"].dropna()
sector_closes = get_closes(SECTOR_ETFS).loc[START:]
sector_returns = sector_closes.pct_change().dropna(how="all")
sp500_closes = get_closes(list(SP500_TICKERS)).loc[START:]
print(f"SPY {spy_close.index.min().date()}..{spy_close.index.max().date()}, "
      f"sp500 cols={sp500_closes.shape[1]}, sectors={sector_closes.shape[1]}")

da = compute_da_signal(spy_df)
vrc = compute_vix_range_compression(vix_close)
dl = compute_defensive_leadership(sp500_closes, spy_close)
fomc = compute_fomc_signal(spy_close)
ar = compute_low_ar_signal(sector_returns, spy_close)
srd = compute_seasonal_divergence_signal(spy_close)
disp = compute_dispersion_signal(sp500_closes, spy_df, spy_close)

signals_ordered = {
    'Distribution Dominance': da, 'VIX Range Compression': vrc,
    'Defensive Leadership': dl, 'Pre-FOMC Rally': fomc,
    'Low Absorption Ratio': ar, 'Seasonal Rank Divergence': srd,
    'Dispersion': disp,
}
horizon_stats = load_horizon_stats()

# --- validate reconstruction against frozen parquet ---
recon = compute_fragility_timeseries(signals_ordered, spy_close, horizon_stats)
frozen = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
both = recon["63d"].to_frame("recon").join(frozen["63d"].rename("frozen"), how="inner").dropna()
for per, lab in [("2016-07", "full"), ("2026-01", "2026")]:
    b = both.loc[per:]
    corr = b.recon.corr(b.frozen)
    mad = (b.recon - b.frozen).abs().mean()
    print(f"recon vs frozen 63d ({lab}, N={len(b)}): corr={corr:.3f}, MAD={mad:.2f}")

# --- per-signal contribution to the 63d composite (replicates the engine loop) ---
h = "63d"
h_days = HORIZON_DAYS[h]
edges = {n: _signal_edge(horizon_stats, n, h) for n in signals_ordered}
print("\n63d edge weights:", {k: round(v, 3) for k, v in edges.items()})

ret_12m = spy_close / spy_close.shift(252) - 1
sma_200 = spy_close.rolling(200).mean()
extension = spy_close / sma_200 - 1
high_52w = spy_close.rolling(252).max()
drawdown = spy_close / high_52w - 1
m = pd.Series(1.0, index=spy_close.index)
m = m + np.where(ret_12m > 0.25, 0.25, np.where(ret_12m > 0.15, 0.10, np.where(ret_12m < -0.05, -0.15, 0.0)))
m = m + np.where(extension > 0.10, 0.25, np.where(extension > 0.05, 0.10, np.where(extension < -0.02, -0.15, 0.0)))
m = m + np.where(drawdown > -0.02, 0.10, np.where(drawdown < -0.10, -0.20, 0.0))
regime_mult = m.clip(0.6, 1.8)
calm_mult = _compute_calm_multiplier_series(spy_close)
spy_pct_from_high = (-drawdown).clip(lower=0.0)

fires = {}
for name, sig in signals_ordered.items():
    hh = sig.get("signal_history")
    if hh is not None and not hh.empty:
        fires[name] = hh.astype(bool)
fire_df = pd.DataFrame(fires).reindex(spy_close.index).fillna(False).astype(bool)

contrib = {}
for name in signals_ordered:
    edge = edges[name]
    if name not in fire_df.columns or edge == 0.0:
        contrib[name] = pd.Series(0.0, index=spy_close.index)
        continue
    sig_on = fire_df[name]
    fire_int = sig_on.astype(int)
    group = fire_int.cumsum()
    days_since = group.groupby(group).cumcount()
    ever = group > 0
    days_since = days_since.where(ever, other=np.nan)
    remaining = ((h_days - days_since) / h_days).clip(0.0, 1.0)
    spy_factor = (1.0 - spy_pct_from_high / HORIZON_DECAY_DD[h]).clip(0.0, 1.0)
    w = np.where(sig_on, 1.0, np.where(ever & (remaining > 0), remaining * spy_factor, 0.0))
    contrib[name] = pd.Series(edge * w, index=spy_close.index)
contrib = pd.DataFrame(contrib)

fomc_w = contrib["Pre-FOMC Rally"] / max(edges["Pre-FOMC Rally"], 1e-9)
base_max = sum(e for n, e in edges.items() if n != "Pre-FOMC Rally")
max_w = base_max + np.where(fomc_w > 0, edges["Pre-FOMC Rally"], 0.0)
score = (contrib.sum(axis=1) / np.maximum(max_w, 1e-9)) * 80 * regime_mult * calm_mult

# --- fire dates per signal, 2025-07 onward (context into 2026) ---
print("\n=== signal fire dates 2025-10 .. 2026-06 (reconstruction) ===")
for name in fire_df.columns:
    f = fire_df.loc["2025-10":"2026-06", name]
    dts = f[f].index
    if len(dts):
        # compress to ranges
        s = dts.to_series()
        ep = (s.diff().dt.days.fillna(1) > 5).cumsum()
        rngs = [f"{g.min().date()}..{g.max().date()}({len(g)}d)" for _, g in s.groupby(ep)]
        print(f"  {name}: {', '.join(rngs)}")
    else:
        print(f"  {name}: none")

# --- attribution table on key 2026 episode dates ---
print("\n=== 63d composite attribution, Feb-Mar 2026 episode (reconstruction) ===")
dates = pd.to_datetime(["2026-02-13", "2026-02-20", "2026-02-27", "2026-03-04",
                        "2026-03-06", "2026-03-12", "2026-03-19", "2026-03-24"])
rows = []
for d in dates:
    if d not in contrib.index:
        d = contrib.index[contrib.index.get_indexer([d], method="nearest")[0]]
    row = {"date": d.date(), "score_raw": round(float(score.loc[d]), 1),
           "regime_m": round(float(regime_mult.loc[d]), 2),
           "calm_m": round(float(calm_mult.loc[d]), 2),
           "spy_dd%": round(float(drawdown.loc[d]) * 100, 1)}
    for name in contrib.columns:
        row[name[:14]] = round(float(contrib.loc[d, name]), 2)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# --- historical >=50 episodes in the FROZEN series and dominant signals then ---
frag_ma = frozen["63d"].dropna().rolling(10, min_periods=1).mean()
above = frag_ma[frag_ma >= 50]
s = above.index.to_series()
ep = (s.diff().dt.days.fillna(1) > 10).cumsum()
print("\n=== all >=50 episodes (frozen 63d 10dMA) + dominant reconstructed contributors ===")
for _, g in s.groupby(ep):
    d0, d1 = g.min(), g.max()
    peak = frag_ma.loc[d0:d1].max()
    win = contrib.loc[d0 - pd.Timedelta(days=90):d1]
    tot = win.sum().sort_values(ascending=False)
    top = ", ".join(f"{k} {v:.0f}" for k, v in tot.head(3).items() if v > 0)
    dd_ep = drawdown.loc[d0:d1]
    print(f"  {d0.date()}..{d1.date()} ({len(g)}d, peak {peak:.0f}) | SPY dd in-episode "
          f"{dd_ep.min()*100:.1f}%..{dd_ep.max()*100:.1f}% | top contributors (90d lead-in): {top}")

# SPY path around the 2026 episode
print("\nSPY context 2026-01..2026-07:")
spy26 = spy_close.loc["2026-01-01":]
print(f"  Jan 2 {spy26.iloc[0]:.0f} -> peak {spy26.max():.0f} ({spy26.idxmax().date()}) "
      f"-> trough {spy26.min():.0f} ({spy26.idxmin().date()}) -> now {spy26.iloc[-1]:.0f}")
dd26 = (spy26 / spy26.cummax() - 1)
print(f"  max 2026 drawdown from running high: {dd26.min()*100:.1f}% on {dd26.idxmin().date()}")
dd_full = drawdown.loc["2026-01-01":]
print(f"  drawdown vs trailing 52w high, worst: {dd_full.min()*100:.1f}% on {dd_full.idxmin().date()}")
