# Migration Guide: Wiring `indicators.py` Into the Codebase

## Overview

`indicators.py` is the **single source of truth** for `calculate_indicators()`. After this migration, the function will exist in exactly one place, and all four consumers will import it.

**Files to modify:**
1. `daily_scan.py`
2. `pages/backtester.py`
3. `pages/strat_backtester.py`
4. `pages/walk_forward.py` (if applicable)

**Files to add:**
- `indicators.py` (root of repo, same level as `daily_scan.py`)
- `tests/test_indicators.py`
- `tests/__init__.py` (empty file)

---

## Step-by-Step

### 1. Copy `indicators.py` to repo root

Place it alongside `daily_scan.py` and `strategy_config.py`.

### 2. Modify `daily_scan.py`

**Delete** the entire `calculate_indicators()` function definition.

**Delete** the `get_sznl_val_series()` function if it's defined locally (it's now in `indicators.py`).

**Add** at the top of the file, with the other imports:

```python
from indicators import calculate_indicators, get_sznl_val_series, apply_first_instance_filter
```

**No call-site changes needed.** The daily_scan signature already matches:
```python
# This existing call works unchanged:
calc_df = calculate_indicators(df.copy(), sznl_map, t_clean, market_series, vix_series, ref_ticker_ranks)
```

**One behavioral change to be aware of:** `min_periods` for perf ranks changes from 50 → 252. This means:
- Tickers with < 1 year of trading history will **no longer generate rank values** in production
- This is intentional — the old value of 50 was producing unreliable percentile ranks
- If any of your strategies specifically target very young tickers (< 1 year old), those tickers will now get `NaN` ranks and the rank filter will not fire. You may need to lower `min_age` to compensate, or we can add a config param for this.

### 3. Modify `pages/backtester.py`

**Delete** the local `calculate_indicators()`, `get_election_cycle()`, `get_age_bucket()`, `apply_first_instance_filter()`, and `get_sznl_val_series()` functions.

**Add** at the top:
```python
from indicators import (
    calculate_indicators,
    get_sznl_val_series,
    apply_first_instance_filter,
    get_election_cycle,
    get_age_bucket,
)
```

**Column name change:** The backtester previously used `Change_in_ATR` — this is now `today_return_atr`. Search for any reference to `Change_in_ATR` in backtester.py and replace with `today_return_atr`.

**Call-site change:** The backtester passes `market_sznl_series` as a positional arg. The new signature uses keyword args, so existing calls like:
```python
df = calculate_indicators(df_raw, sznl_map, ticker, market_series, vix_series, 
                          market_sznl_series, gap_window, req_custom_mas, acc_win, dist_win, ref_ticker_ranks)
```
...will still work positionally, but **convert to keyword args for safety:**
```python
df = calculate_indicators(
    df_raw, sznl_map, ticker,
    market_series=market_series,
    vix_series=vix_series,
    market_sznl_series=market_sznl_series,
    gap_window=gap_window,
    custom_sma_lengths=req_custom_mas,
    acc_window=acc_win,
    dist_window=dist_win,
    ref_ticker_ranks=ref_ticker_ranks,
)
```

### 4. Modify `pages/strat_backtester.py`

Same pattern: delete local function, add import. The strat_backtester had a **different parameter order** (gap, acc, dist, custom_sma_lengths vs the backtester's order). With the import, this is resolved — use keyword args:

```python
processed[t_clean] = calculate_indicators(
    df, _sznl_map, t_clean, market_series, _vix_series,
    gap_window=params['gap'],
    acc_window=params['acc'],
    dist_window=params['dist'],
    custom_sma_lengths=list(params['mas']),
)
```

### 5. Add test infrastructure

```bash
mkdir -p tests
touch tests/__init__.py
# Copy test_indicators.py into tests/
```

Add to your dev workflow:
```bash
# Run before any push:
pytest tests/ -v

# Or if pytest isn't installed:
python -m pytest tests/ -v
# Or:
pip install pytest && pytest tests/ -v
```

---

## Divergences Resolved by This Migration

| Issue | Before | After |
|-------|--------|-------|
| `min_periods` for perf rank | daily_scan=50, backtester=252 | **252 everywhere** |
| `fill_method` in pct_change | Inconsistent | Omitted (pandas default) |
| `is_ath` comparison | daily_scan `>=`, backtester `>` | **`>=` everywhere** |
| Column name for ATR return | `Change_in_ATR` vs `today_return_atr` | **`today_return_atr`** |
| `vol_ratio_10d_rank` min_periods | 50 vs 252 | **252** |
| Gap/Acc/Dist windows | Hardcoded vs configurable | **Both** (standard windows always present) |
| Market seasonal column | `Market_Sznl` vs `Mkt_Sznl_Ref` | **Both computed** |

---

## Verification Checklist

After wiring everything up:

- [ ] `pytest tests/ -v` → all 15 tests pass
- [ ] Run `daily_scan.py` locally and confirm signals match previous day's output
- [ ] Run a backtest in `backtester.py` on a known strategy and compare trade count
- [ ] Run `strat_backtester.py` and check portfolio metrics haven't shifted
- [ ] Grep for any remaining local `def calculate_indicators` — should find zero
