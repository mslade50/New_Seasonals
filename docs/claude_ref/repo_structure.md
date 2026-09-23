# Repo structure, critical rules, module boundaries, ticker constants

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## What This Project Is

A quantitative equity trading platform built on Streamlit. Three pillars:
1. **Strategy system** — backtesting, scanning, and order staging for directional equity strategies (1-63 day hold)
2. **Risk monitoring** — multi-layer market regime dashboard (volatility, internals, credit/macro)
3. **Dispersion analytics** — S&P 500 absolute return dispersion (Nomura methodology)

## Repo Structure

```
├── app.py                          # Main Streamlit entry point
├── strategy_config.py              # Strategy definitions (STRATEGY_BOOK)
├── daily_scan.py                   # Unified scanner — supports --scope=liquid|overflow|all (--moc-only flag retained for future use; no MOC strategies in book currently)
├── daily_risk_report.py            # Daily risk email (fragility dials + signals + forward returns)
├── daily_portfolio_report.py       # Daily portfolio health report (imports from strat_backtester)
├── weekly_market_rundown.py        # Weekly PDF rundown (tabloid landscape, 11 chart pages)
├── daily_pitch.py                  # Daily Pitch publisher (email + Pitch tab + journal) — see "Daily Pitch"
├── pitch_grammar.py                # Daily Pitch idea contract (vocabularies, sizing, order derivation)
├── pitch_journal.py                # Daily Pitch append-only journal (idea/killed/approval/outcome)
├── verify_fills.py                 # Post-close fill verification (updates Google Sheets)
├── indicators.py                   # Shared indicator library
├── earnings_filter.py              # Shared OVS earnings blackout helpers (load parquet, compute offset)
├── cache_io.py                     # Cloudflare R2 read/write wrapper (boto3) — graceful no-op without creds
├── abs_return_dispersion.py        # S&P 500 dispersion metric (~505 tickers)
├── local_overflow_scan.py          # DEPRECATED stub — forwards to `daily_scan.py --scope=overflow`
├── pages/                          # Streamlit pages (FLAT — no subfolders)
│   ├── risk_dashboard_v2.py        # Multi-layer regime monitor (standalone)
│   ├── backtester.py               # Strategy backtesting UI
│   ├── strat_backtester.py         # Extended backtester
│   ├── heatmaps.py                 # Market heatmap inspector
│   ├── correlation_heatmaps.py     # Correlation analysis
│   ├── macro_seasonality.py        # Macro seasonality (formerly sector_trends)
│   └── user_input.py               # User input page
├── .github/workflows/              # Dispatch-only backups + cloud-only site deploys
│   ├── daily_screener.yml          # 2x/day unified scan — pre-market (08:47 UTC) and post-close (22:00 UTC) bookends, both --scope=all
│   ├── build_earnings_calendar.yml # Nightly FMP refresh → R2
│   ├── update_master_prices.yml    # Nightly yfinance incremental → R2
│   ├── update_intraday_prices.yml  # Nightly 15min yfinance incremental → R2 (intraday cache)
│   ├── portfolio_report.yml        # Daily portfolio email
│   ├── bootstrap_caches.yml        # workflow_dispatch only — one-shot full master_prices rebuild
│   ├── risk_report.yml             # Daily risk dashboard email
│   ├── verify_fills.yml            # Post-close fill verification
│   ├── deploy_site.yml             # Private-site build + Pages deploy — reusable workflow (workflow_call) invoked by daily_screener's deploy-site job, same run (2x/day)
│   └── weekly_rundown.yml          # Sunday weekly PDF
├── scripts/                        # Task Scheduler PowerShell wrappers (most disabled post-Phase-2)
│   ├── upload_radar_recs.py        # Momentum radar recs -> R2 (local; the radar repo is private and CI has no cross-repo token)
│   ├── run_radar_sync.bat          # Mondays 8:50 AM ET — publish recs + reconcile trail stops (see Momentum Radar)
│   ├── register_radar_sync_task.ps1 # one-shot registration for the above
│   ├── run_earnings_calendar.ps1   # Weekdays 5:30 PM ET — local backup of GHA build (dual writers OK)
│   ├── build_earnings_calendar.py  # FMP earnings backfill (used by both local + GHA)
│   ├── update_master_prices.py     # yfinance incremental update (used by both local + GHA)
│   ├── build_master_prices.py      # One-shot full rebuild (used by bootstrap_caches.yml)
│   ├── build_trade_ledger.py       # Full-history trade ledger (data/backtest_trades_full.parquet)
│   ├── build_site.py               # Private-site JSON payloads + static assets -> dist/
│   ├── build_signal_charts.py      # Per-trade candlestick charts -> charts/ + R2 (lazy-served on the site)
│   ├── signal_chart_common.py      # Shared chart key + MAE/MFE helpers (build_signal_charts + build_site)
│   ├── build_risk_json.py          # Condensed risk summary for the site (best effort, exits 0)
│   ├── backtester_html_report.py   # Legacy single-file HTML view (reports/portfolio/)
│   ├── refresh_view.py             # Local one-command ledger + HTML refresh
│   └── (DISABLED locally: run_overflow_scan.ps1, run_daily_portfolio_report.ps1, run_master_prices_update.ps1)
├── site/                           # Private-site frontend (static HTML/CSS/JS, committed)
├── functions/                      # Cloudflare Pages Functions — chartimg/[[path]].js streams chart PNGs from R2,
│                                   #   radar-recs.js serves the momentum radar's weekly plans from R2
├── wrangler.toml                   # Pages config: pages_build_output_dir=dist + CHARTS R2 binding (TOML — action's wrangler 3.90.0 ignores .jsonc)
├── dist/                           # Site build output — gitignored, deployed to Cloudflare Pages
├── charts/                         # Per-trade chart PNGs — gitignored; R2 (charts/ prefix) is the source of truth
├── data/                           # Persistent cache (parquet files + radar digest) — gitignored
├── docs/                           # Documentation (private_site_setup.md = Cloudflare one-time setup)
└── tests/                          # Tests
```

## Critical Rules

### yfinance MultiIndex Bug
ALL multi-ticker yfinance downloads return MultiIndex columns `(Price, Ticker)`. You MUST handle this:
```python
# For multi-ticker downloads:
if isinstance(raw.columns, pd.MultiIndex):
    df = raw.xs(ticker, level='Ticker', axis=1)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.capitalize() for c in df.columns]
```
Skipping this causes silent crashes. Every data function must handle it.

### Dividend-Adjustment Basis (raw vs adjusted) — book-wide invariant
The rule, applied per surface:
- **Compare a FROZEN dollar level against RAW bars** (`auto_adjust=False`). A limit/stop/entry that was computed once and stored (sheet `Limit_Price`/`Entry`/`ATR`, a ledger entry, a live working order) lives in the as-traded basis it was minted in. Re-pulling ADJUSTED bars re-scales history down whenever a later dividend goes ex, dropping a past low below a limit that was never touched live (the EWZ 33.51 ex-div phantom fill, 2026-06). `verify_fills.py` pulls raw for exactly this reason.
- **RECOMPUTE a relative level each run → ADJUSTED bars are safe.** The backtest engines (`pages/backtester.py`, `pages/strat_backtester.py`) derive the limit from the same adjusted series each run (`Close ± k·ATR`) and compare to that series' forward bars. Both sides scale by the dividend factor `f`, so the fill decision is exactly scale-invariant — no phantom, and returns stay on the correct total-return basis. The engines do NOT round the limit (rounding is the one thing that could break invariance; `verify_fills` rounds, but it's moot there since it uses raw).
- **This holds only while every entry/exit level in the book is RELATIVE.** The moment an ABSOLUTE dollar level is added to the engine path (a hard limit price, a `$`-pivot, a fixed stop), scale-invariance breaks and that level must follow the frozen-level rule (raw bars), or move the cache to raw-OHLCV + read-time adjustment (the deferred "Tier 2" fix). Guard: `tests/test_verify_fills_exdiv.py`.
- **Cache note:** `master_prices.parquet` stores ADJUSTED OHLCV and `update_master_prices.py` re-adjusts a rolling window (`--max-lookback-days`, default 120 — capped above the 63-day max hold + ATR lookback so recent signals stay uniformly adjusted). Per-trade returns are unaffected by the cap; only buy-and-hold accounting past the cap drifts. Do NOT converge the engine basis (adjusted) with the `verify_fills` basis (raw).

### Pages Directory
The `pages/` directory must remain **FLAT** — no subfolders. Streamlit discovers pages by scanning this directory.

### Path Setup Pattern
All pages that import from the project root use:
```python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
```

### Caching Pattern
- `@st.cache_data(ttl=3600)` for data downloads (1-hour TTL)
- `@st.cache_resource` for static data (seasonal maps)
- Parquet files in `data/` for expensive computations (S&P 500 prices)

## Module Boundaries

**risk_dashboard_v2.py is STANDALONE.** It must never import from:
- `strategy_config.py`
- `strat_backtester.py`
- `daily_scan.py`
- `indicators.py`

It may optionally import `SP500_TICKERS` from `abs_return_dispersion.py` (with try/except fallback).

**Strategy modules** (`strat_backtester.py`, `daily_scan.py`, `daily_portfolio_report.py`) all depend on `strategy_config.py` for `STRATEGY_BOOK` and `ACCOUNT_VALUE`.

**daily_portfolio_report.py** imports backtesting logic from `strat_backtester.py`. Both must stay in sync with `daily_scan.py` for signal detection, sizing, and trade processing. `ACCOUNT_VALUE` from `strategy_config.py` is the single source of truth for portfolio sizing across all three. The pinned local-primary `postclose` pipeline runs it after pulling canonical R2 inputs; `.github/workflows/portfolio_report.yml` is receipt-gated backup only. Reports cover both liquid (LIQUID_PLUS_COMMODITIES) and overflow (CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES) universes — overflow-eligible strategies get a second deep-copied pass with `OVERFLOW_RISK_OVERRIDES` (only OLV 35→25 bps nominal remains; OVS uses path-1 nominal 40 bps for both tiers; all nominals scale by `GLOBAL_RISK_MULTIPLIER` — see "Sizing Conventions").

**daily_scan.py** is the single unified scanner (post-2026-04-30 merge with the retired `local_overflow_scan.py`). CLI flags:
- `--scope=liquid` (default) — scans every strategy against its native universe (typically LIQUID_PLUS_COMMODITIES)
- `--scope=overflow` — only the 6 overflow-eligible strategies, swapped to CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES with OLV bps override
- `--scope=all` — both passes concatenated, signals stamped with `Scan_Source='Liquid'` or `'Overflow'`
- `--moc-only` — restricts to strategies with `entry_type='Signal Close'`. Skips the overflow tier entirely (overflow doesn't MOC by convention). Currently a no-op since the strategy book has no MOC entries; the flag is retained for future use if a Signal Close strategy is added back.

Per-tier tab routing inside `save_staging_orders`: Liquid rows → `Order_Staging`, Overflow rows → `Overflow`. Both tabs are read by `order_staging.py` (which lives in `C:\Users\McKinley Slade\OneDrive\trading_ibkr\` — IBKR-bound, stays local).

## Ticker Constants

| Variable | Location | Count | Description |
|----------|----------|-------|-------------|
| `SP500_TICKERS` | `abs_return_dispersion.py` | ~505 | Full S&P 500 constituents |
| `LIQUID_PLUS_COMMODITIES` | `strategy_config.py` | ~190 | Liquid universe — daily_scan default scope |
| `CSV_UNIVERSE` | `strategy_config.py` | ~1060 | Full universe (liquid + overflow tier ~870) |
| `OVERFLOW_ELIGIBLE_STRATEGIES` | `daily_scan.py` | 6 | OVS, OLV, LT Trend ST OS, St OS Sznl, 52wh Breakout, ATR Extended Gap Up (no override — native 40 bps nominal on overflow) |
| `OVERFLOW_RISK_OVERRIDES` | `strategy_config.py` (imported by `daily_scan.py`, `daily_portfolio_report.py`, `strat_backtester.py`) | 1 | OLV: 35→25 bps nominal for overflow tier (52.5→37.5 effective); the engine applies it via `overflow_active=True` for tickers outside `LIQUID_PLUS_COMMODITIES` (wired 2026-08-12 — was a dead parameter) |
| `GLOBAL_RISK_MULTIPLIER` | `strategy_config.py` | 1.5 | Book-wide risk scaler applied at import — see "Sizing Conventions" |
| `SECTOR_ETFS` | `risk_dashboard_v2.py` | 11 | SPDR sector ETFs |
| `VOL_TICKERS` | `risk_dashboard_v2.py` | 4 | SPY, ^VIX, ^VIX3M, ^VVIX |
| `CROSS_ASSET_TICKERS` | `risk_dashboard_v2.py` | 7 | LQD, HYG, IEF, UUP, ^MOVE, ^TNX, ^IRX |
| `TAIL_RISK_TICKERS` | `risk_dashboard_v2.py` | 1 | ^SKEW |
| `SIGNAL_CACHE_PATH` | `risk_dashboard_v2.py` | — | `data/risk_dashboard_signal_state.json` |
