# Intraday Data — Architecture & Maintenance

**Status:** Production. FMP for one-shot historical backfill, yfinance for ongoing maintenance.

## Architecture

Hybrid two-source design — each source plays to its strengths:

| Source | Role | Window | Why |
|---|---|---|---|
| **FMP** (`/stable/historical-chart/15min`) | One-shot historical backfill | 2003-09-10 → present | Has the deep history yfinance lacks. |
| **yfinance** (`yf.download(interval='15m')`) | Daily incremental maintenance | Rolling 60 days | Free, no API key, already wired everywhere else. Drift vs FMP is negligible (see below). |

The two sources have been validated to produce statistically equivalent OHLC: median close drift 0.001%, max ~0.07% on a 1,341-bar overlap. Volume diverges (different consolidated tapes) but only affects volume-conditional studies.

## Storage layout

```
data/intraday/{TICKER}_15min.parquet     # local cache (gitignored)
data/intraday/_meta.parquet              # ticker index (n_bars, first/last ts, n_days, size_kb)

R2: seasonals-cache/intraday/15min/{TICKER}.parquet
    seasonals-cache/intraday/15min/_meta.parquet
```

Schema per parquet: `ts (naive ET), open, high, low, close, volume`. Regular session only (09:30-15:45 ET). Unadjusted for splits/dividends — median drift vs auto_adjust=False yfinance ~0.02% on closes.

## Coverage

**Universe target:** `LIQUID_PLUS_COMMODITIES` (~197 tickers from `strategy_config.py`).

**Caveats:**
- yfinance intraday 60-day rolling means the cache must be refreshed **at least every ~50 days** or gaps open. The GHA workflow runs weekdays so this isn't a concern in practice — but the constraint should be documented for anyone restoring from an old R2 backup.
- Caret-prefixed tickers (`^GSPC`, `^NDX`) are not served by FMP `/historical-chart`. They're excluded from the backfill set. Add them via a separate yfinance one-shot if needed (60d only).

## Scripts

| Script | What it does |
|---|---|
| `intraday_data.py` | Runtime loader. `get_intraday(ticker, start, end)`, `available_tickers()`, lazy R2 refresh on stale (>18h) local copies. Used by `pages/backtester.py` Day Trade Limit modes. |
| `scripts/pull_intraday_validator.py` | FMP backfill driver. Walks `/historical-chart/15min` backwards in ~30-day chunks anchored on earliest returned ts, dedupes, writes parquet, validates aggregated daily vs yfinance. Supports `--tickers`, `--tickers-file`, `--skip-existing`. Self-throttles at 5 req/s. |
| `scripts/upload_intraday_to_r2.py` | Push local parquets to R2 + rebuild meta. When filtered with `--tickers`, meta is still rebuilt from *all* local files so partial uploads don't shrink the index. |
| `scripts/update_intraday_yfinance.py` | **The maintenance tool.** Reads existing parquets, fetches recent 15min bars from yfinance, converts UTC→ET, strips tz, dedupes, appends, rebuilds meta, optional `--upload` to R2. Smart-buffer: fetches max(7d, gap-since-last-bar+2d). |

## Automation

`.github/workflows/update_intraday_prices.yml` — weekdays 20:45 UTC (4:45 PM ET, 15 min after `update_master_prices.yml`):
1. Pulls `_meta.parquet` from R2 to discover what's in the cache
2. Pulls each ticker's parquet from R2 (~600 MB total when fully built)
3. Runs `update_intraday_yfinance.py --upload` — fetches recent bars, appends, dedupes, pushes back

If R2 is fresh (recent run), each ticker fetches ~26 bars (1 trading day) in ~0.3s. Full run ~1-2 minutes wall time.

## Validity testing & calibration

The original validity test (per the pre-FMP plan doc, since superseded) was:

> Filter trades to dates with intraday coverage. Pull 15-min bars per trade. Recompute MAE/MFE with the same pessimistic entry-day rule at 15-min resolution. Side-by-side comparison: did intraday change the conclusion?

This is still the right framing for any new strategy that wants intraday refinement. The relevant function is `compute_trade_path_stats()` in `pages/backtester.py`. Day Trade Limit entry modes (±0.5/0.75/1 ATR) now use 15min bars natively via the `use_intraday` toggle — those are the first strategies actually load-bearing on this data.

## Known costs

- **R2 storage:** ~3 MB per ticker, ~600 MB for the full LIQUID_PLUS_COMMODITIES set.
- **R2 egress:** ~600 MB per GHA run if it pulls the whole cache fresh. R2 egress is free; bandwidth caps are far higher than this.
- **FMP API quota:** ~190 calls per ticker × 197 tickers ≈ 37k calls for a full from-scratch backfill. One-time cost — not recurring.
- **yfinance:** free, no key.

## Future work

- **Sub-15min resolution** (5m or 1m) for sub-bar fill timing. yfinance offers 1m but only for 7d rolling — not enough for the validity-vs-storage trade. Would need FMP or Polygon for history.
- **Volume reconciliation** if any strategy ends up volume-conditional. FMP and yfinance use different consolidated tapes so neither is "right" — pick one and stick.
- **Index intraday** (^GSPC, ^NDX, ^VIX) via yfinance 60d rolling. Would need a separate maintenance path since they're not in FMP's universe.
