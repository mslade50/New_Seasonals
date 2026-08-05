"""Rebuild data/signal_horizon_stats.json from the production signal pipeline.

The JSON weights the live fragility composite (risk_dashboard_v2._signal_edge
reads horizons[h].diff_mean), which in turn sizes live orders through
frag_risk_bands. Until now the file was a one-off vintage (2026-05-13) with no
generator; this script makes the calibration reproducible and stamped.

Method, per signal (same day-level semantics as the original file):
  - signal_mean / unconditional_mean: mean SPY forward return (pp) over
    signal-active days vs all valid days in the signal's history window
  - diff_mean = signal_mean - unconditional_mean (negative = downside edge)
  - hit_rate: % of active days with a POSITIVE forward return
  - p_value: Welch t-test, active vs inactive days. Overlapping forward
    windows inflate this (always have); episode fields are the honest check:
  - n_episodes / episode_mean / episode_t: distinct activation runs, mean
    forward return from each episode's FIRST day, one-sample t of those
    episode returns against the unconditional mean.

Usage:
  python scripts/build_signal_horizon_stats.py                 # write data/
  python scripts/build_signal_horizon_stats.py --out X.json    # candidate run
Prints a drift table against the existing JSON either way.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class _NoOpStreamlit:
    """Import stub so the risk pipeline runs headless (same trick as the PIT
    study, scratch/pit_extract_signals.py)."""

    def __getattr__(self, name):
        return self

    def __call__(self, *a, **k):
        return self

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cache_data(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    cache_resource = cache_data


if "streamlit" not in sys.modules or not hasattr(sys.modules["streamlit"], "columns"):
    sys.modules["streamlit"] = _NoOpStreamlit()

HORIZONS = {"5d": 5, "10d": 10, "21d": 21, "42d": 42, "63d": 63}
OUT_DEFAULT = os.path.join(_ROOT, "data", "signal_horizon_stats.json")

# Parameter descriptions mirror the constants in pages/risk_dashboard_v2.py
# compute_* functions as of this generator's vintage.
PARAMS = {
    "Distribution Dominance": "63d window, ratio > 3.75, SPY within 2% of 52w high, SPY > 50d SMA",
    "Distribution Dominance (Elevated)": "63d window, ratio > 6.0, SPY within 2% of 52w high",
    "VIX Range Compression": "21d range pctile < 15, VIX above floor and rising vs SMA",
    "Defensive Leadership": "50d risk-on minus risk-off spread < -10pp",
    "Pre-FOMC Rally": "3 sessions before scheduled FOMC decisions",
    "Low Absorption Ratio": "AR pctile < 10, SPY near 52w high",
    "Seasonal Rank Divergence": "risk-off minus risk-on seasonal spread > +10pp",
    "Dispersion": "composite dispersion pctile > 85",
    "Equity P/C Complacency": ("10d-MA CBOE equity put/call, trailing-252d "
                               "pctile < 10 (complacency tail)"),
}

# Horizons a signal is ALLOWED to carry in the JSON. The Equity P/C
# Complacency signal is 5d-only BY DESIGN (2026-08-05): its 63d dial
# candidacy was rejected (day-level edge is overlap inflation, episode t
# wrong-signed — scratch/putcall_dial_study.py), so a regeneration must
# never hand it 21d/63d edges and silently change the sizing column.
HORIZON_RESTRICT = {
    "Equity P/C Complacency": {"5d"},
}


def _episodes(hist: pd.Series) -> list[pd.Timestamp]:
    """First day of each contiguous True run."""
    h = hist.astype(bool)
    starts = h & ~h.shift(1, fill_value=False)
    return list(h.index[starts])


def signal_block(name: str, hist: pd.Series, spy_close: pd.Series) -> dict | None:
    hist = hist.dropna().astype(bool)
    hist = hist.reindex(spy_close.index).dropna() if hist.index.difference(
        spy_close.index).size else hist
    if hist.empty or not hist.any():
        return None

    episodes = _episodes(hist)
    horizons: dict = {}
    for label, days in HORIZONS.items():
        fwd = (spy_close.shift(-days) / spy_close - 1.0) * 100.0
        fwd = fwd.reindex(hist.index)
        valid = fwd.notna()
        active = hist & valid
        inactive = ~hist & valid
        if active.sum() < 3 or inactive.sum() < 3:
            continue
        sig_fwd = fwd[active]
        unc_mean = float(fwd[valid].mean())
        ep_fwd = fwd.loc[[d for d in episodes if d in fwd.index]].dropna()
        ep_t = np.nan
        if len(ep_fwd) > 2 and ep_fwd.std() > 0:
            ep_t = float((ep_fwd.mean() - unc_mean)
                         / (ep_fwd.std() / np.sqrt(len(ep_fwd))))
        horizons[label] = {
            "signal_mean": round(float(sig_fwd.mean()), 2),
            "unconditional_mean": round(unc_mean, 2),
            "diff_mean": round(float(sig_fwd.mean()) - unc_mean, 2),
            "hit_rate": round(float((sig_fwd > 0).mean() * 100), 1),
            "p_value": round(float(sps.ttest_ind(
                sig_fwd, fwd[inactive], equal_var=False).pvalue), 3),
            "episode_mean": round(float(ep_fwd.mean()), 2) if len(ep_fwd) else None,
            "episode_t": round(ep_t, 2) if np.isfinite(ep_t) else None,
        }
    if not horizons:
        return None
    return {
        "n_events": int(hist.sum()),
        "n_episodes": len(episodes),
        "pct_active": round(float(hist.mean() * 100), 1),
        "params": PARAMS.get(name, ""),
        "horizons": horizons,
    }


def drift_table(old: dict, new: dict) -> None:
    rows = []
    for name, sig in new["signals"].items():
        old_sig = old.get("signals", {}).get(name, {})
        for h, vals in sig["horizons"].items():
            ov = old_sig.get("horizons", {}).get(h, {})
            rows.append({
                "signal": name, "h": h,
                "diff_old": ov.get("diff_mean"), "diff_new": vals["diff_mean"],
                "hit_old": ov.get("hit_rate"), "hit_new": vals["hit_rate"],
                "ep_t": vals.get("episode_t"),
            })
    df = pd.DataFrame(rows)
    df["d_diff"] = df["diff_new"] - df["diff_old"]
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()

    from daily_risk_report import download_data, compute_all_signals

    print("downloading data (production pipeline) ...")
    spy_df, closes, sp500_closes = download_data()
    computed = compute_all_signals(spy_df, closes, sp500_closes)
    signals_ordered = computed["signals_ordered"]
    spy_close = computed["spy_close"].dropna()

    histories: dict[str, pd.Series] = {}
    for name, sig in signals_ordered.items():
        h = sig.get("signal_history")
        if h is not None and hasattr(h, "empty") and not h.empty:
            histories[name] = h
    da = signals_ordered.get("Distribution Dominance", {})
    if da.get("elevated_history") is not None:
        histories["Distribution Dominance (Elevated)"] = da["elevated_history"]

    blocks = {}
    for name, hist in histories.items():
        block = signal_block(name, hist, spy_close)
        if block:
            allowed = HORIZON_RESTRICT.get(name)
            if allowed:
                block["horizons"] = {h: v for h, v in block["horizons"].items()
                                     if h in allowed}
            blocks[name] = block
        else:
            print(f"  {name}: no usable history, skipped")

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=_ROOT,
                             timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"

    payload = {
        "description": ("Backtest statistics for risk dashboard signals. "
                        "diff_mean = signal_mean - unconditional_mean (negative = "
                        "downside risk). All return values in percentage points. "
                        "Day-level stats; episode_* fields are overlap-free checks."),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_through": spy_close.index[-1].strftime("%Y-%m-%d"),
        "generator": "scripts/build_signal_horizon_stats.py",
        "git_sha": sha,
        "signals": blocks,
    }

    if os.path.exists(OUT_DEFAULT):
        with open(OUT_DEFAULT) as f:
            old = json.load(f)
        print("\n=== drift vs existing data/signal_horizon_stats.json ===")
        drift_table(old, payload)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out} ({len(blocks)} signals, "
          f"data through {payload['data_through']})")


if __name__ == "__main__":
    main()
