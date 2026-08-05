"""Rebuild data/atr_downside_stats.json — per-signal downside distribution tables
for the site risk tab's signal cards.

For each of the 7 fragility signals, over FULL history (master_prices, ~2000+),
computes the probability that SPY trades >= k ATR BELOW the fire-day close at some
point within each forward window (LOW-TOUCH: subsequent intraday low reaches the
level). ATR = Wilder-14 as of the fire day. Reported for ATR multiples [1,2,3,5]
and horizons [5,10,21,42,63], both episode-first (fresh trigger, overlap-free) and
day-level, with an all-market baseline.

Why a committed precompute (mirrors scripts/build_signal_horizon_stats.py): the
live risk pipeline (daily_risk_report.download_data) only fetches a 10-year
window, which leaves rare signals with too few events (SRD 22 vs 55 episodes,
Dispersion/Low-AR in the low teens total). This generator reconstructs the exact
production signal masks from the 25-year master_prices cache instead, so the
tables are stable. The dial-conditioned table is computed live in build_risk_json
(it depends on today's dial value).

Signal masks come from the PRODUCTION compute_* functions (pages/risk_dashboard_v2)
fed 25y inputs — no logic is reimplemented here. Event counts are printed against
the frozen signal_horizon_stats.json n_events as a reconstruction check.

Usage:  python scripts/build_atr_downside_stats.py [--out data/atr_downside_stats.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys


class _NoOpStreamlit:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


if "streamlit" not in sys.modules or not hasattr(sys.modules["streamlit"], "columns"):
    sys.modules["streamlit"] = _NoOpStreamlit()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pages"))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

HORIZONS = {"5d": 5, "10d": 10, "21d": 21, "42d": 42, "63d": 63}
MULTS = [1, 2, 3, 5]
ATR_N = 14
OUT_DEFAULT = os.path.join(_ROOT, "data", "atr_downside_stats.json")


def wilder_atr(H, L, C):
    n = len(C)
    prev_c = np.concatenate([[np.nan], C[:-1]])
    tr = np.maximum.reduce([H - L, np.abs(H - prev_c), np.abs(L - prev_c)])
    atr = np.full(n, np.nan)
    if n > ATR_N:
        atr[ATR_N] = np.nanmean(tr[1:ATR_N + 1])
        for i in range(ATR_N + 1, n):
            atr[i] = (atr[i - 1] * (ATR_N - 1) + tr[i]) / ATR_N
    return atr


def worst_low_drop(L, C, N):
    """max_j (Close_i - Low_{i+j}) / ATR_i over j=1..N; valid where window fits."""
    n = len(C)
    worst = np.zeros(n)
    valid = np.ones(n, bool)
    for j in range(1, N + 1):
        Ls = np.concatenate([L[j:], np.full(j, np.nan)])
        worst = np.fmax(worst, C - Ls)
    valid[n - N:] = False
    return worst, valid


def table_for_mask(mask, L, C, atr, atr_valid):
    """{horizon: {mult: pct, ...}, n_by_h: {...}} for the given day mask."""
    out, n_by_h = {}, {}
    for lab, N in HORIZONS.items():
        worst, wv = worst_low_drop(L, C, N)
        v = wv & atr_valid & mask
        dd = worst[v] / atr[v]
        out[lab] = {str(k): round(float(100 * np.mean(dd >= k)), 1) for k in MULTS} if v.sum() else {str(k): None for k in MULTS}
        n_by_h[lab] = int(v.sum())
    return out, n_by_h


def build_inputs_from_master():
    """spy_df (OHLC), closes (sector ETFs + ^VIX), sp500_closes — all 25y."""
    from risk_dashboard_v2 import SECTOR_ETFS
    try:
        from abs_return_dispersion import SP500_TICKERS
    except Exception:
        SP500_TICKERS = []

    mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"))
    mp["date"] = pd.to_datetime(mp["date"])

    spy_df = (mp[mp["ticker"] == "SPY"]
              .sort_values("date").set_index("date")[["Open", "High", "Low", "Close", "Volume"]].dropna())

    close_tickers = list(dict.fromkeys(list(SECTOR_ETFS) + ["^VIX", "^VIX3M", "^VVIX"]))
    cl = mp[mp["ticker"].isin(close_tickers)]
    closes = cl.pivot_table(index="date", columns="ticker", values="Close").sort_index()

    sp_tk = [t for t in SP500_TICKERS if t]
    sp = mp[mp["ticker"].isin(sp_tk)]
    sp500_closes = sp.pivot_table(index="date", columns="ticker", values="Close").sort_index()
    return spy_df, closes, sp500_closes


def compute_signal_masks(spy_df, closes, sp500_closes):
    """Replicates daily_risk_report.compute_all_signals lines 85-108 to get the
    per-signal signal_history masks WITHOUT the fragility bundle."""
    from risk_dashboard_v2 import (
        SECTOR_ETFS,
        compute_da_signal, compute_vix_range_compression,
        compute_defensive_leadership, compute_fomc_signal,
        compute_low_ar_signal, compute_seasonal_divergence_signal,
        compute_dispersion_signal, compute_pc_complacency_signal,
    )
    spy_close = spy_df["Close"]
    sector_cols = [c for c in SECTOR_ETFS if c in closes.columns]
    sector_returns = closes[sector_cols].dropna(axis=1, how="all").pct_change().dropna(how="all")
    vix_close = closes["^VIX"].dropna() if "^VIX" in closes.columns else pd.Series(dtype=float)

    return {
        "Distribution Dominance": compute_da_signal(spy_df),
        "VIX Range Compression": compute_vix_range_compression(vix_close),
        "Defensive Leadership": compute_defensive_leadership(sp500_closes, spy_close),
        "Pre-FOMC Rally": compute_fomc_signal(spy_close),
        "Low Absorption Ratio": compute_low_ar_signal(sector_returns, spy_close),
        "Seasonal Rank Divergence": compute_seasonal_divergence_signal(spy_close),
        "Dispersion": compute_dispersion_signal(sp500_closes, spy_df, spy_close),
        # 5d-only dial contributor (2026-08-05) — downside card renders on
        # the site whenever it fires, like every other signal.
        "Equity P/C Complacency": compute_pc_complacency_signal(spy_close),
    }


# Day-level event counts from the frozen signal_horizon_stats.json vintage.
# Only DA and SRD share this generator's day-level definition (both match
# exactly and anchor the reconstruction); the others counted deduped episodes
# (Low-AR, Dispersion) or triggered events (Pre-FOMC) so they are NOT expected
# to line up — shown for reference only.
_FROZEN_N = {
    "Distribution Dominance": 269, "VIX Range Compression": 107,
    "Defensive Leadership": 176, "Pre-FOMC Rally": 16,
    "Low Absorption Ratio": 17, "Seasonal Rank Divergence": 139, "Dispersion": 12,
}
_FROZEN_COMPARABLE = {"Distribution Dominance", "Seasonal Rank Divergence"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    print("building 25y inputs from master_prices ...")
    spy_df, closes, sp500_closes = build_inputs_from_master()
    print(f"  SPY {spy_df.index[0].date()} -> {spy_df.index[-1].date()} ({len(spy_df)} days); "
          f"sp500 cols {sp500_closes.shape[1]}; close cols {list(closes.columns)}")

    masks = compute_signal_masks(spy_df, closes, sp500_closes)

    # ATR + low arrays on the SPY calendar
    idx = spy_df.index
    H = spy_df["High"].to_numpy(float)
    L = spy_df["Low"].to_numpy(float)
    C = spy_df["Close"].to_numpy(float)
    atr = wilder_atr(H, L, C)
    atr_valid = np.isfinite(atr) & (atr > 0)
    n = len(idx)

    baseline, base_n = table_for_mask(np.ones(n, bool), L, C, atr, atr_valid)

    print("\nsignal                         recon_n  frozen_n  episodes  (frozen check)")
    sig_out = {}
    for name, sig in masks.items():
        hist = (sig or {}).get("signal_history")
        if hist is None or (hasattr(hist, "empty") and hist.empty):
            print(f"  {name:30s}  (no history, skipped)")
            continue
        hist = hist.dropna().astype(bool)
        mask_s = pd.Series(False, index=idx)
        common = hist.index.intersection(idx)
        mask_s.loc[common] = hist.loc[common].values.astype(bool)
        m = mask_s.to_numpy()
        ep = m & ~np.concatenate([[False], m[:-1]])

        ep_tbl, _ = table_for_mask(ep, L, C, atr, atr_valid)
        day_tbl, _ = table_for_mask(m, L, C, atr, atr_valid)
        sig_out[name] = {
            "n_events": int(m.sum()),
            "n_episodes": int(ep.sum()),
            "episode": ep_tbl,
            "day": day_tbl,
        }
        fz = _FROZEN_N.get(name, "?")
        chk = "OK" if (name in _FROZEN_COMPARABLE and abs(int(m.sum()) - fz) <= 2) else \
              ("MISMATCH" if name in _FROZEN_COMPARABLE else "n/a (diff defn)")
        print(f"  {name:30s}  {int(m.sum()):7d}  {fz:>8}  {int(ep.sum()):8d}  {chk}")

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=_ROOT, timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"

    payload = {
        "description": ("Low-touch downside distribution per fragility signal. value = "
                        "P(SPY intraday low reaches >= k*ATR below the fire-day close "
                        "within the horizon), %. ATR = Wilder-14 at fire day. "
                        "episode = first day of each activation run (overlap-free); "
                        "day = every active day; baseline = all market days."),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_through": spy_df.index[-1].strftime("%Y-%m-%d"),
        "data_from": spy_df.index[0].strftime("%Y-%m-%d"),
        "generator": "scripts/build_atr_downside_stats.py",
        "git_sha": sha,
        "measure": "low_touch",
        "atr_period": ATR_N,
        "mults": MULTS,
        "horizons": list(HORIZONS.keys()),
        "baseline": baseline,
        "baseline_n": base_n,
        "signals": sig_out,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out} ({len(sig_out)} signals, through {payload['data_through']})")


if __name__ == "__main__":
    main()
