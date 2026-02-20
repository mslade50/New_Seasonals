"""
Backtest forward SPY returns when N signals overlap.
Loads pre-computed signal fire history from parquet (run dashboard once first).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

# ---------------------------------------------------------------------------
# Load signal fire history from parquet
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SIGNAL_FIRE_HISTORY_PATH = os.path.join(DATA_DIR, "signal_fire_history.parquet")

if not os.path.exists(SIGNAL_FIRE_HISTORY_PATH):
    print(f"ERROR: {SIGNAL_FIRE_HISTORY_PATH} not found.")
    print("Run the Risk Dashboard V2 once first to generate the signal fire history.")
    sys.exit(1)

print("Loading signal fire history from parquet...")
combined = pd.read_parquet(SIGNAL_FIRE_HISTORY_PATH)
combined = combined.astype(bool)

# Download SPY close to compute forward returns
print("Downloading SPY data...")
START = combined.index.min().strftime("%Y-%m-%d")
spy_raw = yf.download("SPY", start=START, progress=False)
if isinstance(spy_raw.columns, pd.MultiIndex):
    spy_raw.columns = spy_raw.columns.get_level_values(0)
spy_raw.columns = [c.capitalize() for c in spy_raw.columns]
spy_close = spy_raw["Close"]

# Align combined to spy_close index
combined = combined.reindex(spy_close.index).fillna(False).astype(bool)
overlap_count = combined.sum(axis=1).astype(int)

HORIZONS = [5, 10, 21, 42, 63]

print(f"\nSignals loaded: {list(combined.columns)}")
print(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}")

print(f"\nOverlap distribution:")
for n in range(0, overlap_count.max() + 1):
    days = (overlap_count == n).sum()
    pct = days / len(overlap_count) * 100
    print(f"  {n} signals: {days:>5d} days ({pct:5.1f}%)")

# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------
fwd = {}
for h in HORIZONS:
    fwd[h] = spy_close.pct_change(h).shift(-h) * 100  # forward return in pct

# ---------------------------------------------------------------------------
# Backtest: for each overlap threshold >= N, compute stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("FORWARD SPY RETURNS BY SIGNAL OVERLAP COUNT")
print("=" * 90)

# Unconditional stats
print(f"\n{'Horizon':>8s}  {'Uncond Mean':>11s}  {'Uncond Med':>10s}  {'N':>6s}")
print("-" * 42)
for h in HORIZONS:
    valid = fwd[h].dropna()
    print(f"  {h:>3d}d     {float(valid.mean()):>9.2f}%   {float(valid.median()):>8.2f}%   {len(valid):>6d}")

# For each threshold
for threshold in [1, 2, 3, 4]:
    mask = overlap_count >= threshold
    n_days = int(mask.sum())
    pct_time = n_days / len(overlap_count) * 100

    print(f"\n{'-' * 90}")
    print(f"  >= {threshold} SIGNALS ACTIVE   ({n_days} days, {pct_time:.1f}% of time)")
    print(f"{'-' * 90}")
    print(f"  {'Horizon':>7s}  {'Signal Mean':>11s}  {'Uncond Mean':>11s}  {'Diff':>8s}  {'Signal Med':>10s}  {'Hit Rate':>8s}  {'t-stat':>7s}  {'p-value':>8s}  {'N':>5s}")
    print(f"  {'-' * 85}")

    for h in HORIZONS:
        valid_fwd = fwd[h].dropna()
        signal_fwd = valid_fwd[mask.reindex(valid_fwd.index, fill_value=False)]
        uncond_fwd = valid_fwd

        if len(signal_fwd) < 5:
            print(f"  {h:>4d}d    insufficient data (N={len(signal_fwd)})")
            continue

        sig_mean = float(signal_fwd.mean())
        unc_mean = float(uncond_fwd.mean())
        diff = sig_mean - unc_mean
        sig_med = float(signal_fwd.median())
        hit_rate = float((signal_fwd < 0).mean()) * 100  # % negative (downside)

        # Welch t-test: signal vs unconditional
        t_stat, p_val = stats.ttest_ind(signal_fwd, uncond_fwd, equal_var=False)

        print(f"  {h:>4d}d    {sig_mean:>9.2f}%   {unc_mean:>9.2f}%   {diff:>+7.2f}%   {sig_med:>8.2f}%   {hit_rate:>6.1f}%   {float(t_stat):>7.2f}  {float(p_val):>8.4f}  {len(signal_fwd):>5d}")

# ---------------------------------------------------------------------------
# Exact overlap count (== N, not >=N) for marginal analysis
# ---------------------------------------------------------------------------
print(f"\n{'=' * 90}")
print("FORWARD SPY RETURNS BY EXACT OVERLAP COUNT (== N)")
print(f"{'=' * 90}")

for exact_n in [0, 1, 2, 3, 4]:
    mask = overlap_count == exact_n
    n_days = int(mask.sum())
    if n_days < 10:
        continue
    pct_time = n_days / len(overlap_count) * 100

    print(f"\n  EXACTLY {exact_n} SIGNALS   ({n_days} days, {pct_time:.1f}%)")
    print(f"  {'Horizon':>7s}  {'Mean':>8s}  {'Median':>8s}  {'%Neg':>6s}  {'N':>5s}")
    print(f"  {'-' * 40}")

    for h in HORIZONS:
        valid_fwd = fwd[h].dropna()
        signal_fwd = valid_fwd[mask.reindex(valid_fwd.index, fill_value=False)]
        if len(signal_fwd) < 5:
            continue
        print(f"  {h:>4d}d    {float(signal_fwd.mean()):>+7.2f}%  {float(signal_fwd.median()):>+7.2f}%  {float((signal_fwd < 0).mean()) * 100:>5.1f}%  {len(signal_fwd):>5d}")

# ---------------------------------------------------------------------------
# Which signal combos appear most?
# ---------------------------------------------------------------------------
print(f"\n{'=' * 90}")
print("MOST COMMON SIGNAL COMBINATIONS (>= 2 signals)")
print(f"{'=' * 90}")

signal_names = list(combined.columns)
combo_counts = {}
for idx in combined.index:
    row = combined.loc[idx]
    active = tuple(sorted(n for n in signal_names if row.get(n, False)))
    if len(active) >= 2:
        combo_counts[active] = combo_counts.get(active, 0) + 1

sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:15]
print(f"\n  {'Combo':60s}  {'Days':>5s}")
print(f"  {'-' * 68}")
for combo, count in sorted_combos:
    label = " + ".join(combo)
    if len(label) > 58:
        label = label[:55] + "..."
    print(f"  {label:60s}  {count:>5d}")

# ---------------------------------------------------------------------------
# Deduped episode analysis for >= 2 overlap
# ---------------------------------------------------------------------------
print(f"\n{'=' * 90}")
print("DEDUPED EPISODE ANALYSIS (>= 2 signals, 21d gap = new episode)")
print(f"{'=' * 90}")

mask_2plus = overlap_count >= 2
fire_dates = mask_2plus[mask_2plus].index

episodes = []
if len(fire_dates) > 0:
    ep_start = fire_dates[0]
    ep_end = fire_dates[0]
    for d in fire_dates[1:]:
        if (d - ep_end).days > 30:  # ~21 trading days gap
            episodes.append((ep_start, ep_end))
            ep_start = d
        ep_end = d
    episodes.append((ep_start, ep_end))

print(f"\n  Total episodes: {len(episodes)}")
print(f"\n  {'#':>4s}  {'Start':>12s}  {'End':>12s}  {'Days':>5s}  {'5d Fwd':>7s}  {'21d Fwd':>8s}  {'63d Fwd':>8s}")
print(f"  {'-' * 65}")

ep_fwd = {h: [] for h in HORIZONS}
for i, (start, end) in enumerate(episodes):
    dur = (end - start).days
    row = f"  {i + 1:>4d}  {start.strftime('%Y-%m-%d'):>12s}  {end.strftime('%Y-%m-%d'):>12s}  {dur:>5d}"
    for h in [5, 21, 63]:
        val = fwd[h].get(start)
        if val is not None and pd.notna(val):
            row += f"  {float(val):>+6.1f}%"
            ep_fwd[h].append(float(val))
        else:
            row += f"  {'N/A':>7s}"
    print(row)

print(f"\n  Episode means (from episode start date):")
for h in HORIZONS:
    vals = ep_fwd.get(h, [])
    if vals:
        print(f"    {h:>3d}d: {np.mean(vals):>+.2f}% (N={len(vals)})")
