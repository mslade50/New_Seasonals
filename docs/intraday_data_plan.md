# Intraday Data — Exploration Notes (Deferred)

**Status:** On hold as of 2026-04-23. Daily-resolution MAE/MFE in `compute_trade_path_stats` (pages/backtester.py) is sufficient for current strategy refinement work. Revisit when intraday stop-loss simulation becomes the blocker.

## Why we explored this

The intra-trade path analysis added to the backtester uses daily H/L per trade. Three limitations identified:
1. Entry-day H/L pre-dates MOC/limit fills (mitigated with pessimistic entry-day rule — adverse side only)
2. Intraday ordering of H vs L unknown → can't detect "stopped intraday, recovered to close"
3. Exit-day peak after the exit fill may slightly overstate MFE

Intraday bars would fix all three. Daily already gets MAE/MFE **magnitudes** right; intraday refines the **path** story and unlocks accurate intraday stop simulation.

## Why we paused

- For current-era conclusions ("add a -2R stop", "exit early if no MFE by day N"), daily resolution is decisive. Intraday would shift stop levels by maybe 10-20 bps, not flip decisions.
- Until we're ready to wire `use_stop_loss=True` properly through `run_engine`, intraday-refined MAE is academic.
- Strategy-level improvements (filters, early-read exit rules) have higher ROI than data-layer improvements right now.

## Data provider comparison

| Source | 15-min history | Bulk pull friction | Cost | Notes |
|---|---|---|---|---|
| yfinance | **60 days** rolling | None | $0 | Dead for backtesting |
| IBKR | 2–6 yr per ticker | 12+ hours w/ pacing | ~$0-10/mo | Requires Gateway + 2FA + data sub; ticker coverage inconsistent |
| Polygon Starter | 5 yr | ~3 hr REST or flat files | **$29/mo** | Flat files included, unlimited API calls |
| Polygon Developer | 10 yr | ~3 hr REST or flat files | $79/mo | Adds Trades endpoint (tick-level); otherwise same as Starter |
| Databento | 20+ yr | Pay-per-query | **Pay-as-you-go + $125 free credit** | Exchange-sourced (higher quality); steeper API curve; `databento` pip pkg |

## Recommended approach when we return

### Step 1 — validity test (no storage commitment)
Fetch intraday data **on-demand** via API from the backtester's Streamlit page. ~519 trades × 1 call each ≈ 1–2 min total. Add a "🔬 Refine with Intraday" button under the MAE/MFE section that:
- Filters trades to the date range where intraday data is available
- Pulls 15-min bars per trade
- Recomputes MAE/MFE with the same pessimistic entry-day rule but at 15-min resolution
- Shows a side-by-side comparison (daily-res distribution vs intraday-res distribution) for the same trade subset
- Flags trades where the gap is large enough to change the conclusion (e.g., daily said MAE = -0.8R, intraday said -2.1R — stop-loss simulation would have fired)

~100 lines of code, no storage, no cron. If refinement is material, build the full pipeline. If not, we've saved the $29-79/mo.

### Step 2 — if validated, build the pipeline
- Pull 15-min bars (not 1-min) — 15× less storage, negligible resolution loss for daily-timeframe strategies
- Partitioned parquet in `data/intraday/symbol=XXX/year=YYYY/data.parquet` (~600 MB for 10 yr × 190 tickers)
- Daily incremental via cron/Actions
- Modify `compute_trade_path_stats` to prefer intraday parquet, fall back to daily when missing
- Wire `use_stop_loss` through `run_engine` with intraday trigger checks (the real payoff)

### Provider preference order
1. **Databento** — $125 free credit covers validity test with zero commitment. Pay-as-you-go scales naturally. Higher data quality. Use dataset `EQUS.MINI` for full US equities consolidated coverage (not `XNAS.ITCH` — misses NYSE names).
2. **Polygon Starter $29** — if Databento's API friction is too high. Flat files + unlimited API calls. 5-year limit enough for validity test.
3. **Polygon Developer $79** — only if we need 10-year intraday depth for stop-loss backtesting over multiple regimes. Skip unless validated Step 1.

### Do not pursue
- IBKR for historical intraday — too much ops overhead for offline backtesting. Reserve for the day we unify data+execution.
- 1-min bars unless we specifically need sub-15-min fill timing (we don't for current strategies).

## Key contextual facts to remember

- Backtester MAE/MFE lives in `compute_trade_path_stats()` in `pages/backtester.py`
- Entry-day rule is pessimistic: adverse side only, MOC entries skip the bar
- Current strategies mostly use time-exit (`use_stop_loss: False`) so intraday refinement is nice-to-have, not load-bearing
- The most recent Overbot Vol Spike backtest (2026-04-23) showed a clean monotonic path profile with 65.4% capture efficiency and a ≤-2R MAE bucket that loses ~95% of the time — the data told a decisive enough story at daily resolution
