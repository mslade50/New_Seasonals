# CLAUDE.md — Project Guide for New_Seasonals

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
├── radar_weekly_summary.py         # Weekly radar digest (reads daily briefs, Claude distills best-of)
├── verify_fills.py                 # Post-close fill verification (updates Google Sheets)
├── indicators.py                   # Shared indicator library
├── earnings_filter.py              # Shared OVS earnings blackout helpers (load parquet, compute offset)
├── cache_io.py                     # Cloudflare R2 read/write wrapper (boto3) — graceful no-op without creds
├── abs_return_dispersion.py        # S&P 500 dispersion metric (~505 tickers)
├── local_overflow_scan.py          # DEPRECATED stub — forwards to `daily_scan.py --scope=overflow`
├── risk_dashboard_clean_sheet.md   # Risk Dashboard V2 design doc
├── pages/                          # Streamlit pages (FLAT — no subfolders)
│   ├── risk_dashboard_v2.py        # Multi-layer regime monitor (standalone)
│   ├── backtester.py               # Strategy backtesting UI
│   ├── strat_backtester.py         # Extended backtester
│   ├── heatmaps.py                 # Market heatmap inspector
│   ├── correlation_heatmaps.py     # Correlation analysis
│   ├── macro_seasonality.py        # Macro seasonality (formerly sector_trends)
│   └── user_input.py               # User input page
├── .github/workflows/              # GitHub Actions — see "Automated Pipeline" below
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
│   ├── run_radar_weekly.ps1        # Sundays 8:30 AM ET — runs radar digest, commits + pushes
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
├── functions/                      # Cloudflare Pages Functions — chartimg/[[path]].js streams chart PNGs from R2
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

**daily_portfolio_report.py** imports backtesting logic from `strat_backtester.py`. Both must stay in sync with `daily_scan.py` for signal detection, sizing, and trade processing. `ACCOUNT_VALUE` from `strategy_config.py` is the single source of truth for portfolio sizing across all three. Runs in **GitHub Actions** (weekdays 21:30 UTC = 5:30 PM ET) — pulls `data/master_prices.parquet` and `data/earnings_calendar.parquet` from Cloudflare R2 before running. Reports cover both liquid (LIQUID_PLUS_COMMODITIES) and overflow (CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES) universes — overflow-eligible strategies get a second deep-copied pass with `OVERFLOW_RISK_OVERRIDES` (only OLV 35→25 bps nominal remains; OVS uses path-1 nominal 40 bps for both tiers; all nominals scale by `GLOBAL_RISK_MULTIPLIER` — see "Sizing Conventions"). Workflow: `.github/workflows/portfolio_report.yml`.

**daily_scan.py** is the single unified scanner (post-2026-04-30 merge with the retired `local_overflow_scan.py`). CLI flags:
- `--scope=liquid` (default) — scans every strategy against its native universe (typically LIQUID_PLUS_COMMODITIES)
- `--scope=overflow` — only the 6 overflow-eligible strategies, swapped to CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES with OLV bps override
- `--scope=all` — both passes concatenated, signals stamped with `Scan_Source='Liquid'` or `'Overflow'`
- `--moc-only` — restricts to strategies with `entry_type='Signal Close'`. Skips the overflow tier entirely (overflow doesn't MOC by convention). Currently a no-op since the strategy book has no MOC entries; the flag is retained for future use if a Signal Close strategy is added back.

Per-tier tab routing inside `save_staging_orders`: Liquid rows → `Order_Staging`, Overflow rows → `Overflow`. Both tabs are read by `order_staging.py` (which lives in `C:\Users\mckin\OneDrive\trading_ibkr\` — IBKR-bound, stays local).

## Risk Dials / Fragility System (rewritten 2026-07-16)

The old "Risk Dashboard V2 Phases 1-2 / Layers 0-4 / Executive Summary"
description no longer matches the code. Current state:

**pages/risk_dashboard_v2.py** computes 7 fragility signals (Distribution
Dominance [+Elevated display tier], VIX Range Compression, Defensive
Leadership, Pre-FOMC Rally, Low Absorption Ratio, Seasonal Rank Divergence,
Dispersion) and a 0-100 composite dial at 3 horizons (5d/21d/63d), weighted
by diff_mean edges from `data/signal_horizon_stats.json` (reproducible via
`scripts/build_signal_horizon_stats.py`; the JSON's "(Elevated)" entry is
reference-only, NOT consumed by the composite). The page displays ONE dial
(63d + 10d MA + throttle state); 5d/21d are context chips (5d failed every
sizing test; 21d ~90% state-agreement with 63d — no "confirm" semantics
anywhere). `daily_risk_report.py` and `weekly_market_rundown.py` import the
page's compute functions (import surface: deleting page functions can crash
the GHA email that appends the sizing parquet — check both before removing
anything).

### The fragility-portfolio contract (B6, 2026-07-16)

- **The sizing statistic** is exactly: 10d MA of the 63d column of
  `data/rd2_fragility.parquet`, threshold 50. Nothing else sizes orders.
- **Vintage rule**: the parquet is APPEND-ONLY point-in-time since
  2026-07-02; earlier rows are a recompute vintage (drifted up to ~7 pts).
  Any backtest joining the dial must state which vintage it used.
  `rd2_fragility_ts.parquet` is a raw-basis full recompute for research/ML
  only — NEVER a sizing fallback (daily_scan's fallback removed 2026-07-16).
- **Staleness convention**: consumers fail OPEN to 1.0x sizing on readings
  older than 3 trading days (`daily_scan.FRAG_STALE_TD`,
  `exposure_leg.DIAL_STALE_TD`), and dial_filters entry gates fail CLOSED.
- **Schema**: columns 5d/21d/63d, 5d-smoothed basis, tz-naive normalized
  index; appends stamp `fragility_stats_sha256` (provenance of the weights
  vintage) plus basis/generated/last-date/frozen-through metadata.
  `tests/test_fragility_append.py` freezes the schema (new columns are
  DROPPED on append — shadow series get their own files).
- **Freeze policy (A2)**: five live thresholds calibrate to this series —
  frag_risk_bands 50, exposure_leg raw-21d 50 + ma10-63d 50, dial_filters
  30 (52wh Breakout) and 65 (St OS Sznl). Do NOT adopt a re-scored stats
  JSON (e.g. `scratch/signal_horizon_stats_candidate.json`) into the live
  path — it de-calibrates all five at once. Replacements go through a
  scratch/pit_reestimate.py-style PIT re-validation, full stop.
- **Pre-registration requirement**: any NEW dial-conditioned control needs a
  pre-registered protocol (gates, decision rule, sensitivity) BEFORE the
  study runs — the discipline that correctly killed the OVS tilt and the
  book-wide throttle. Live prereg docs:
  `scratch/ultracode_research/exposure_leg_replay_prereg_2026-07-16.md`,
  `scratch/ultracode_research/olv_frag_band_prereg_2026-07-16.md`.

### Consumers (change-impact map)

- `frag_risk_bands` (strategy_config -> daily_scan 2b -> strat_backtester
  3b3): FAMILY4 + 3x Bear Fade at [[50,999,0.25]]. Guard:
  `tests/test_frag_risk_bands.py` (includes the site serializer assertion).
- `exposure_leg.py` (25% NAV VOO/QQQ overlay in the AM scan email): kill
  rules raw-21d>50 and ma10-63d>50. The 1.25x boost was REMOVED 2026-07-16
  (mirrored the unanimously-killed per-trade boost). The raw-21d kill has a
  pre-registered replay pending — do not touch it before the replay runs.
- `dial_filters` entry gates, the daily risk email, the site risk tab
  (`sizing_state` reads the PIT parquet, never the deploy recompute), the
  portfolio page fragility adjuster (`fragility.json`), ML features.

### Simple-dial shadow (A6, accumulating since 2026-07-16)

`fragility_simple.py` -> `data/rd2_fragility_simple.parquet` (own file,
append-only, written by daily_risk_report): equal-weight 7-signal sum with
linear 63d decay — no edge weights, no regime/calm mults, no x80, fixed FOMC
denominator. Pre-registered threshold rule: percentile-match to the
incumbent gate's ON rate, NO scanning. Probes showed ~0.85 correlation /
~89% gate agreement with the incumbent, i.e. the fitted weights are mostly
cosmetic. Changes NOTHING until a PIT re-run gates a swap (~2027 earliest).
Guard: `tests/test_fragility_simple.py`.

### Negative results / triggers (institutional memory)

- Book-wide throttle/taper, dial-conditioned caps: dead (PIT t=-0.23; see
  Daily Risk Caps section). OVS tilt: dead (PIT gate 2026-07-03).
- Put hedges, VXX proxy, 21d "fast confirm" shadow, trend-sleeve gate,
  >1.0x hi-frag boosts, sub-50 sizing ramps: all rejected — reasoning
  preserved in scratch/ultracode_research/RISK_DIALS_2026-07-16.md section 4.
- 3x Bear Fade band re-exam TRIGGER: revisit at 2 new hi-frag episodes (its
  own hi bucket is flat, t=-0.05, N_hi=17; band kept by family analogy).
  Companion to the existing "re-examine FAMILY4 at +20 trades (~2029)".
- OLV mild-band candidacy: see prereg doc above; PIT re-bucket is THE gate.
- Exemptions CONFIRMED permanent pending new evidence: OVS, LT Trend ST OS,
  St OS Sznl, 3x Overbot Fade, 52wh Breakout, Sector BO, 3x Leader Gap Fade.

### Signal downside tables (site risk tab, 2026-07-22)

DISPLAY-ONLY conditional-downside tables on the risk tab (`payload["atr_downside"]`);
they size NOTHING. Measure = LOW-TOUCH: P(SPY intraday low reaches >= k*ATR BELOW
the fire/anchor close within a horizon), Wilder-14 ATR at the fire day, multiples
[1,2,3,5] x horizons [5,10,21,42,63], vs an all-market baseline. Two surfaces:
- **Per-signal card** (renders under a signal ONLY when it fires): episode-first
  (fresh-trigger, overlap-free) table. Full-history so rare signals aren't starved
  (SRD 55 episodes vs 22 on a 10y window; Dispersion/Low-AR are low-teens TOTAL).
- **Dial band** (under the sizing hero): days where the 10d-MA of the 63d dial
  closed within +-3 of its CURRENT value -> same low-touch table. Computed LIVE
  (depends on today's dial); dial history is 2016+ so this table is a decade deep.

Why a committed precompute for the per-signal tables: the live risk pipeline
(`daily_risk_report.download_data`) only fetches 10y. `scripts/build_atr_downside_stats.py`
reconstructs the EXACT production signal masks (`compute_all_signals` compute_*
functions) fed 25y of master_prices instead, and writes `data/atr_downside_stats.json`
(committed seed; regenerated fresh each deploy — best-effort step in `deploy_site.yml`
BEFORE build_risk_json, so the shipped tables track current data). The reconstruction
is validated by DA (268~269) and SRD (139=139) matching the frozen
`signal_horizon_stats.json` day-level counts exactly; the other signals' frozen
counts used deduped-episode / event definitions and are NOT comparable (not a bug).

Aligned sites -- change together:
- `scripts/build_atr_downside_stats.py` (generator; ATR + low-touch helpers are the
  single source of that math)
- `scripts/build_risk_json.py` `build_atr_downside()` (reads the committed stats for
  per-signal tables; computes the dial-band table live, IMPORTING the generator's
  helpers so the two are byte-identical)
- `site/assets/risk.js` `atrCellsHtml` / `atrSignalTableHtml` / `atrDialTableHtml`
  + `site/assets/style.css` `.atr-card` / `.atr-tbl`
- Guard: `tests/test_risk_site_js.py::test_atr_downside_tables_render` (dial table
  under hero, per-signal table under firing signals only, off-signals get none).

## Ticker Constants

| Variable | Location | Count | Description |
|----------|----------|-------|-------------|
| `SP500_TICKERS` | `abs_return_dispersion.py` | ~505 | Full S&P 500 constituents |
| `LIQUID_PLUS_COMMODITIES` | `strategy_config.py` | ~190 | Liquid universe — daily_scan default scope |
| `CSV_UNIVERSE` | `strategy_config.py` | ~1060 | Full universe (liquid + overflow tier ~870) |
| `OVERFLOW_ELIGIBLE_STRATEGIES` | `daily_scan.py` | 6 | OVS, OLV, LT Trend ST OS, St OS Sznl, 52wh Breakout, ATR Extended Gap Up (no override — native 40 bps nominal on overflow) |
| `OVERFLOW_RISK_OVERRIDES` | `daily_scan.py`, `daily_portfolio_report.py` | 1 | OLV: 35→25 bps nominal for overflow tier (52.5→37.5 effective) |
| `GLOBAL_RISK_MULTIPLIER` | `strategy_config.py` | 1.5 | Book-wide risk scaler applied at import — see "Sizing Conventions" |
| `SECTOR_ETFS` | `risk_dashboard_v2.py` | 11 | SPDR sector ETFs |
| `VOL_TICKERS` | `risk_dashboard_v2.py` | 4 | SPY, ^VIX, ^VIX3M, ^VVIX |
| `CROSS_ASSET_TICKERS` | `risk_dashboard_v2.py` | 7 | LQD, HYG, IEF, UUP, ^MOVE, ^TNX, ^IRX |
| `TAIL_RISK_TICKERS` | `risk_dashboard_v2.py` | 1 | ^SKEW |
| `SIGNAL_CACHE_PATH` | `risk_dashboard_v2.py` | — | `data/risk_dashboard_signal_state.json` |

## Sizing Conventions — GLOBAL_RISK_MULTIPLIER + overlays (2026-05-27)

`strategy_config.GLOBAL_RISK_MULTIPLIER` (currently **1.5**) scales the whole
book at import time: every execution `risk_bps`, OVS `path1_bps` / `path2_bps`
/ `path2_daily_cap_pct`, every `earnings_size_override.risk_bps`, and the
`OVERFLOW_RISK_OVERRIDES` in daily_scan / daily_portfolio_report. The dicts in
strategy_config SOURCE are nominal; everything downstream (scan, engines,
reports, staged `Risk_Amt`/`Risk_Bps`) sees SCALED values. **All bps in this
doc are nominal unless marked effective** — e.g. OLV liquid 35 nominal = 52.5
effective, overflow 25 = 37.5. (OLV additionally sizes by signal recency —
first iteration in a trailing 21td window 0.5x, second 0.7x — see "Ladder
Sizing"; a flat 35→18 cut shipped and was replaced by a ladder form the same
day, 2026-07-29, re-based to signal recency 2026-07-30.)

GRM evidence trail (2026-07-16, scratch/grm_replay_study.py — the constant
shipped 2026-05-27 with none): full-ledger replay at GRM 1.0/1.25/1.5/1.75
with caps FIXED at prod (250 per-strat, 500L/250S pooled, flat $750k).
Risk-adjusted metrics are nearly scale-invariant — Sharpe 1.89/1.87/1.85/1.83,
annPnL/maxDD ~1.66 flat, maxDD -8.9%/-10.8%/-12.6%/-14.4% NAV scaling
slightly sub-linearly (fixed caps clip the tail). No cliff anywhere in the
range: the setting is a clean risk-appetite dial, and 1.5 (~$157k/yr ann
flat PnL at -12.6% worst DD) is defensible. Results:
scratch/grm_replay_results.csv.

daily_scan per-signal sizing order (mirrored in strat_backtester step 3b):
base bps (tier x GRM) -> 2b fragility band -> 2c signal-recency ladder rung
(carrier: OLV {window_td: 21, mults: [0.5, 0.7, 1.0]} since 2026-07-30; the
old open-position-count ladder machinery survives dormant, carrier-less) ->
2c2 cycle-year mult -> 2d earnings size override (REPLACES the base but
COMPOSES with the 2c recency mult since 2026-07-30, itself
GRM-scaled; two carriers: OLV -10..0 TD -> 10 bps nominal / 15 effective;
St OS Sznl -5..-1 TD -> 6 bps nominal / 9 effective, added 2026-07-30 —
the no-stop 5d hold straddling an imminent print held every ledger tail
loser [-5..-1 cell N=9 avgR -0.50 vs +0.32 outside]; small-N appetite
haircut, guard tests/test_earnings_size_override.py, evidence
scratch/stos_earnings_proximity.py) -> shares -> ADV participation cap -> per-ticker notional cap
(OLV, 2026-07-20) -> 5c same-day signal de-rate (post-pass; 3x Bear fade —
see its section below).

## Daily Risk Caps (aligned 2026-07-10)

Two stacked pro-rata caps on STAGED (pre-fill) risk, applied after all
per-signal sizing. All values are EFFECTIVE (not GRM-scaled):
- **Per-strategy: 250 bps/day.** Each strategy's staged risk per signal date
  independently capped; rows scale by cap/total. Live:
  `order_staging.PER_STRAT_DAILY_CAP_BPS` (was 200 from inception —
  raised to 250 on 2026-07-10 to close a silent live-vs-ledger divergence:
  the engine always modeled 250, so live trimmed 200-250 bps days up to
  20% while the ledger booked full size; the drift came from the engine
  default being anchored to the POOLED 2.5% cap, not the per-strategy
  one). Engine: `process_signals_fast(cap_bps=...)`, default 250.
  `PER_STRAT_DAILY_CAP_DOLLARS` in order_staging holds per-strategy dollar
  overrides (currently empty).
- **Pooled per-direction caps: REMOVED 2026-07-16** (in place 2026-07-10 to
  2026-07-16 at long 500 / short 250 bps). The cap-impact study
  (`scratch/cap_impact_study.py` + `cap_impact_results.csv`) showed the
  pooled layer bound on the SAME net-positive cluster days as the
  per-strategy cap and cost ~$125k/23y with IDENTICAL maxDD and worst day —
  pure redundancy. Removed together: `order_staging` pooled stage (staged
  side totals still printed), `build_trade_ledger` POOLED_*_CAP_BPS = None,
  `daily_portfolio_report` call site, strat_backtester UI defaults (0 = off).
  The engine's `max_long_risk_bps`/`max_short_risk_bps` machinery is
  retained for counterfactuals (sequential-after-per-strategy semantics,
  fixed 2026-07-16; guard: `tests/test_pooled_cap_sequential.py`).
  Context: caps overall cost 25% of total return and 0.56 Sortino over 23y;
  the per-strategy 250 is kept because it alone bounds the worst single day
  (-$44k vs -$118k = -15.75% NAV uncapped, which by itself was the entire
  uncapped maxDD).

Aligned sites — change together: `order_staging.py` (OneDrive) constants,
`scripts/build_trade_ledger.py` POOLED_*_CAP_BPS, `daily_portfolio_report.py`
call site, `pages/strat_backtester.py` UI defaults + `cap_bps` fallback (250)
in `process_signals_fast`.

**Do NOT fragility-condition these caps** (negative result, codified
2026-07-16): dial-scaled pooled or per-strategy caps are the failed
book-wide throttle re-skinned — rest-of-book at dial >=50 shows no
significant degradation (p=.47 clustered), the aggregate PIT t was -0.23,
and the taper variant cost -11.4R — on the costliest possible surface (four
aligned sites incl. one out-of-repo, scalar-to-series engine change). The
book's only evidenced dial-sizing hook is per-strategy `frag_risk_bands`.
Evidence: scratch/ultracode_research/RISK_DIALS_2026-07-16.md.

## Ladder Sizing (OLV signal-recency form, 2026-07-30)

`execution['signal_recency_ladder']` has ONE carrier: OLV at
**{window_td: 21, mults: [0.5, 0.7, 1.0]}** — the rung is the count of that
ticker's OLV SIGNAL days (shared filter mask, fill-independent) in the
trailing 21 sessions before the signal day: 0 prior -> 0.5x, 1 prior ->
0.7x, 2+ -> full. The earnings size override COMPOSES with this mult (it
replaces the BASE bps only — a first-iteration pre-earnings signal is
10 x 0.5 bps nominal); every other overlay is still clobbered by the
override. A deliberate risk-appetite footprint trim aimed at OLV's weakest
legs (leg-1 avgR +0.56-0.82 across dial bands vs +1.1-1.4 for leg-3+;
OLV's open notional had doubled vs the 2018-2020 norm and carried 49% of
2026's intraday trough dollars), NOT a PnL-positive rule — it costs
expectancy by design.

It replaced the ONE-DAY-OLD open-position-count ladder [0.5, 1, 1]
(2026-07-29, itself the replacement for a same-day flat 35->18 cut). Why
the re-base: the open-count form reset to 0.5x whenever a chain had fully
exited (even a day later), was blind to still-unfilled working limits (a
day-2 signal before day-1's limit filled ALSO got 0.5x), and jumped
straight to full size on the second leg. Signal-recency counting fixes all
three and grades the second iteration at 0.7x.

Implementation — aligned sites (change together):
- `strategy_config.py` OLV execution `signal_recency_ladder` (source of
  truth; mults NOT GRM-scaled)
- `daily_scan.py` sizing step 2c: recomputes the fired ticker's mask
  (`filters.live_signal_mask`) and counts the trailing window
  (`filters.recency_prior_from_mask` — last bar excluded); mult carried
  into step 2d's earnings override
- `pages/strat_backtester.py`: candidate-recency pre-pass counts prior
  candidate df-positions per (strategy, ticker) from the RAW candidate list
  (candidates ARE mask days, so engine == scan; known bound: pre-cutoff
  signals invisible for the first window of a run) + `_recency_mult` in the
  earnings override
- order_staging needs nothing (takes scanner-staged sizes as-is)
- Guard: `tests/test_olv_stop_and_cap.py` (config invariants, consecutive
  grading, fill-independence, window expiry, override composition)

History: the ORIGINAL ladder (OLV-only 2026-04-22 to 2026-07-20, removed in
the stop/gate package) was the OPPOSITE bet — a mild 0.85 first-rung
discount graded UP — and flat 1.0x beat it ($654k vs $605k [0.85,1,1] vs
$627k [0.85,1,1.15] / 21y; evidence scratch/olv_package_sim.py). Do not
confuse the two: today's recency ladder is an appetite cut that accepts
that drag. The old open-position-count machinery
(`execution['ladder_multipliers']`, `daily_scan.load_open_position_counts`,
the engine's open-count rung) survives dormant with NO carriers.

## Cross-Strategy Overlap Clamp (2026-05-12)

`strategy_config.CROSS_STRATEGY_OVERLAP_OVERRIDES`: when the named strategies
fire on the SAME signal date and SAME tradeable ticker (compared after
`SPOT_TO_TRADEABLE` aliasing, ^GSPC->SPY ^NDX->QQQ), each side's risk is
clamped to `risk_bps_when_overlapping`. Currently one pair: Indices Oversold
Bounce + SPY QQQ MonFri Reversion -> 20 bps nominal each. Applied in
daily_scan step 5b and replayed in `pages/strat_backtester.py` (~line 2258).

## OVS Strategy — Earnings Blackout + 2-Path Sizing + Friday-only EOD-DD

Overbot Vol Spike has special-cased execution as of the 2026-04-30 merge.

### Earnings blackout (±10 trading days)
The OVS execution dict in `strategy_config.py` carries `earnings_blackout_td: 10`. Signals within ±10 trading days of an earnings announcement are dropped. Tickers with no earnings data in `data/earnings_calendar.parquet` (commodity ETFs, indices, futures, FX) **pass through** — NaN-as-True, mirroring the `Not Between` behavior in `pages/backtester.py`.

Implementation:
- `earnings_filter.py` — shared module with `load_earnings_dates_map()`, `signed_offset()`, `in_blackout(window=10)`. Loads `data/earnings_calendar.parquet`.
- `daily_scan.py` — applies the filter inline during the strategy loop (drops the signal before the dict is built).
- `pages/strat_backtester.py` — pre-pass that drops candidates from the chronological loop entirely, so the daily portfolio report's PnL reflects what live would do.

### Two-path execution (replaces the prior 30/20 bps + 1.3× ATR-sznl-5d sizer)
The OVS execution dict carries (nominal; ×GRM 1.5 at import → 60 / 12 / 1.125%):
- `path1_bps: 40` — full size on a decisive open gap
- `path2_bps: 8` — reduced size on a mild gap
- `path2_daily_cap_pct: 0.75` — 0.75% of ACCOUNT_VALUE aggregate cap on path-2 risk (was 1.0 pre-2026-05-01)

Decision happens in `order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) using IBKR's T+1 session open vs the signal's close + 0.25 ATR threshold. Same scheme for liquid AND overflow universes.

| T+1 open vs close | Path | Per-trade size |
|---|---|---|
| Open > Close + 0.25 ATR | **Path 1: Decisive** | 40 bps nominal / 60 effective (full) |
| Close < Open ≤ Close + 0.25 ATR | **Path 2: Mild** | 8 bps nominal / 12 effective, capped at the 0.75% nominal path-2 aggregate (pro-rata scale-down across all path-2 rows that day) |
| Open ≤ Close | **Skip** | 0 |

Scanner-side stamps `Path1_Bps`, `Path2_Bps`, `Path2_Daily_Cap_Pct` columns on every OVS staging row so order_staging can compute the multiplier without importing strategy_config.

### Scale-out (live 2026-06-17, engine-modeled 2026-07-16)
Every OVS P1/P2 primary-account row is split by order_staging into two
independent single-target brackets: **near = 40% of shares @ 1 ATR, far =
60% @ 2 ATR** (a tranche that rounds below 1 share = no split, single
full-size 2-ATR bracket; PA is never split). Deliberate short-book VARIANCE
SMOOTHING, not PnL-maximizing — the 2026-07-01 audit measured scale-outs as
-R vs full-size 2 ATR and McKinley accepted that trade-off explicitly
(2026-07-16). The engine books two tranche rows per fill (`Tranche` column:
near/far/'' ) with the same share split; EOD-DD days book as one row (both
live tranches exit at the same close); entry-day targets stay uncredited on
the near tranche (book convention). Aligned sites — change together:
- `strategy_config.py` OVS execution `scaleout_near_frac` /
  `scaleout_near_tgt_atr` (source of truth; NOT GRM-scaled)
- `order_staging.py` (OneDrive) `OVS_SCALEOUT_NEAR_FRAC` /
  `OVS_PROFIT_TAKER_ATR_MULT` + `_split_scaleout_for_primary`
- `pages/strat_backtester.py` tranche booking in `process_signals_fast`
- Guard: `tests/test_ovs_scaleout.py`

### Same-symbol precedence + P1-budget gate history (2026-07-16)
**ATR Extended Gap Up > OVS**: when both fire on the same symbol and the ATR
row passed its T+1 open gate, the OVS row is dropped (both short the same
blow-off; never double the slot). Live in order_staging since before 2026-07;
modeled in the engine pre-pass since 2026-07-16 (engine keys on ATR-Ext
candidates, whose mask already includes the T+1 gate — matching live's
Quantity > 0 condition). Guard: `tests/test_ovs_scaleout.py`.
**P1-budget gate REMOVED**: the engine-only rule "kill all P2 when the day's
P1 risk exceeds 60% of the per-strategy cap" fired on ~170 historical ledger
days but never existed live. Removed 2026-07-16 (decision: match live). The
P2 aggregate daily cap is live and stays.

### Entry-day drawdown stop (EOD-DD, Friday entries only)
The OVS execution dict carries `eod_dd_atr: 0.25` and `eod_dd_weekdays: [4]`. If a Friday-entered OVS trade is more than 0.25 ATR offside vs the entry-day fill by 15:58 ET, exit at the entry-day close. Mon-Thu entries skip the check entirely — those positions get the full hold window instead. Weekday list uses Python conventions (Mon=0..Fri=4); empty/missing = all weekdays.

Aligned across four systems — change `eod_dd_weekdays` in one place and they all move together:
- `strategy_config.py` — execution dict (single source of truth)
- `pages/strat_backtester.py` — reads `execution['eod_dd_weekdays']`, gates the EOD-DD block on `df.index[entry_idx].weekday() in [...]`. Drives both the backtester page and `daily_portfolio_report.py`.
- `pages/backtester.py` — UI multiselect lets you override per-run for exploration (separate from the prod-locked rule above).
- `order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) — hardcoded `weekday() == 4` gate on the STP-with-goodAfterTime=15:58 leg. Update both sides if you change the rule.
- Regression coverage: `tests/test_eod_dd.py` Cases C/D assert Fri fires + Tue skipped under `[4]`.

### Reference
- Trading-day arithmetic: `compute_signed_earnings_offsets()` in `pages/backtester.py` (np.busday_count + USFederalHolidayCalendar).
- Earnings parquet: `data/earnings_calendar.parquet` — 117k rows, 946 tickers, FMP-backfilled, includes forward dates.
- 2-path validation note (2026-04-29): 12 of 13 OVS signals on that date would have been killed by the blackout — only USO survived because no earnings data.

## Cycle-Year Risk Tilt (OVS, 2026-06-10)

OVS runs at 0.75x risk in midterm years (year%4==2). Evidence: all six
midterm years 2006-2026 underperform (avgR +0.19 vs +0.49 non-midterm),
leave-one-year-out stable, damage concentrated in P1 decisive-gap entries
(+0.63 -> +0.23 avgR). ~1.5 sigma after episode clustering -> shrunk-Kelly
0.75x, not the full-conviction 0.4x. Validated by LOYO, NOT by re-running
the backtest with the rule on (in-sample rules flatter themselves).

Three aligned sites -- change together:
- `strategy_config.py` OVS execution `cycle_risk_mults: {2: 0.75}` (source of truth)
- `pages/strat_backtester.py` sizing step 3b2 (generic: any strategy with the field)
- `daily_scan.py` sizing step 2c2 (stamps the mult into Sizing notes)
(order_staging needs nothing since 2026-06-11: the OVS P1 fixed-dollar
target and its `OVS_CYCLE_MULTS` were removed -- P1 takes the scanner's
staged size as-is, so the tilt flows through like every other overlay.)

The live-vs-backtest divergence found during this work (P1-only live with a
fixed $3,000 target, mild gaps dropped) was RESOLVED 2026-06-11:
order_staging trades both paths again, P1 at scanner qty x1.0 and P2 at
scanner qty x (Path2_Bps / Path1_Bps) from the row stamps plus the P2
aggregate daily cap, matching the 2-path scheme the ledger models
(P2 = 407 trades, +0.20 avgR, +82R/24y). The engine's `ovs_p1_only`
parameter remains for counterfactuals.

## Fragility Risk Bands (2026-07-02)

Per-strategy fragility sizing via `execution['frag_risk_bands']` =
`[[lo, hi, mult], ...]` on the 10d-MA 63d risk-dial score as of signal date
(first match wins, `lo <= score < hi`; missing/stale score or no bands = 1.0x).
REPLACED the retired book-wide ramp (1.25x boost -> 0.10x floor): the boost had
no edge case, and only specific pockets degrade at high fragility. Current
bands: the dip-buy FAMILY4 (Weak Close Decent Sznls, SPY QQQ MonFri Reversion,
Monday Dip, Indices Oversold Bounce) run `[[50, 999, 0.25]]`; the rest of the
book (including OVS) is 1.0x at all scores. Unlike the old ramp, the ENGINE
REPLAYS the bands point-in-time, so ledger and live agree (finding #26 closed
for this scheme). Evidence: scratch/ultracode_research/PORTFOLIO_RESEARCH_2026-07-02.md.

PIT edge-weight gate (roadmap step 5, run 2026-07-03, scratch/pit_reestimate.py
+ pit_extract_signals.py): the fragility composite's signal weights were
re-estimated on expanding windows (vintage Y-1 weights score year Y, 2018+)
to remove calibration lookahead. Results: PIT-vs-current series corr 0.94,
>=50 day agreement 92%. FAMILY4 throttle SURVIVED (hi -0.10 vs lo +0.63,
clustered t=-1.96 p=0.057; LOYO floor ~1.4-1.5 sigma; negative in 6 of 9
years) — stands, conviction one notch lower than the current-weights grading.
The OVS [21,44) 0.75x tilt FAILED (PIT t=-1.34; even current weights only
t=-0.63 on 2018+ — its z=-3.0 lived in untestable 2016-17) and was REMOVED
per the pre-agreed gate; OVS is fully exempt again. The aggregate book-wide
>=50 effect also fails PIT (t=-0.23), vindicating the family-only design.
Residual lookahead the PIT gate cannot cure: signal definitions/parameters
are today's code. Re-examine FAMILY4 at +20 high-frag family trades (~2029).

Three aligned sites -- change together (order_staging needs nothing: it takes
scanner-staged sizes as-is since 2026-06-11):
- `strategy_config.py` execution `frag_risk_bands` (source of truth)
- `pages/strat_backtester.py` sizing step 3b3 (`frag_band_mult_at`, reads
  data/rd2_fragility.parquet point-in-time; pre-2016 signals -> 1.0x)
- `daily_scan.py` sizing step 2b (`frag_band_mult`, today's score, stamps
  Sizing notes; scan-summary email shows active band tilts)
- Guard: `tests/test_frag_risk_bands.py` (config invariants + boundary
  behavior + engine/live parity). Replay parity vs the research cells:
  scratch/parity_check_frag_bands.py (FAMILY4 74@0.25x exact, OVS 226/230
  @0.75x exact with 4 cap-interaction deviations of 0.0004 on one day).

## 3x Bear ETF Overbot Fade + Same-Day Signal De-rate (2026-07-07)

The 13 bear-equity 3x names (`strategy_config.LEV3X_BEAR_EQ`) were carved out
of the generic 3x ETF Overbot Fade (now 29 tickers) into a looser bear-only
fade: short thresholds 85->80, 21d consec 3->1, SAME 126/252d < 65 leader
exclusion — that filter is LOAD-BEARING (it keeps the fade from shorting
sustained bear markets; dropping it collapsed avgR +0.66 -> +0.28 with
-14R/-16R years in 2020/2022). Universes are disjoint by construction so the
two fades can never fire the same ticker on the same day. 25 bps nominal
(vs parent 40) because every signal in the loosened config lands 2020+
(one-regime sample). Fading an overbought inverse ETF = buying a market
selloff, so the strategy carries the FAMILY4 `frag_risk_bands` [[50,999,0.25]].
Evidence: scratch/lev3x_fade_class_study.py + lev3x_fade_bear_episodes.py.

Position stacking (2026-07-28): BOTH 3x Overbot Fades run `max_one_pos: False`
— consecutive-day re-fires open additional full-size legs in the same ticker
(observed depth <= 3). This ALIGNED THE MODEL TO LIVE: daily_scan /
order_staging never enforced one-pos (eq_order_entry's dup guard keys on
staged date), so live always stacked; the ledger was the side that under-
counted (first live stack: SQQQ 2026-07-24 + 07-27 legs). Backtest impact of
adopting stacking: generic fade +12.7R/23y (maxDD unchanged), bear fade
+10.2R with ~1.5x wider maxDD/worst-5d — accepted; marginal-leg edge is
episode-concentrated (bear: mostly Apr 2024). Only the per-strategy 250
bps/day cap bounds a stack (it sees same-day staged risk, NOT open legs).
The 3x Leader Gap Fade keeps `max_one_pos: True` (guarded by its test).
Evidence: scratch/lev3x_fade_stacking_study.py + lev3x_fade_stacking_results.csv.

Same-day signal de-rate — new generic sizing overlay, currently bear-fade
only: `execution['same_day_signal_derate'] = 0.10` sizes each of the day's
signals at `max(floor, 1 - 0.10*(n-1))` where n = that strategy's SIGNAL
count that day (ex-ante staged count, NOT fills — only ~1/3 of signals fill,
but high signal count itself marks the violent-selloff days where per-trade
edge degrades). `same_day_derate_floor` = 0.30. Composes multiplicatively
with frag bands (April 2024: 5 signals x high fragility -> 0.15x). Evidence:
scratch/lev3x_fade_bear_sizing_rule.py (same totR, worst 2-day window
-6.2R -> -4.5R).

Aligned sites -- change together (order_staging needs nothing: takes staged
sizes as-is):
- `strategy_config.py` execution `same_day_signal_derate` /
  `same_day_derate_floor` + shared formula `same_day_derate_mult()`
- `pages/strat_backtester.py` sizing step 3b4 (counts staged candidates
  per (strategy, day) in a pre-loop pass, post earnings-blackout)
- `daily_scan.py` post-pass 5c (after the cross-strategy overlap clamp;
  runs post-loop because n is only known after the strategy's ticker loop;
  counts per (Strategy_Name, Scan_Source); rescales Shares/Risk_Amt/Notional
  and stamps Sizing notes)
- Guard: `tests/test_same_day_derate.py` (carve-out partition, filter
  invariants, formula boundaries, single-carrier assertion)

## Large-Gap-Up Size Derate (2026-07-21)

Per-strategy HALF-SIZE on a gap-up open, carried by the two liquid dip-buys
Monday Dip and SPY QQQ MonFri Reversion via
`execution['gap_size_derate'] = {threshold_atr: 0.25, mult: 0.5, dir: 'up'}`.
When the T+1 session open gaps more than `threshold_atr * ATR` ABOVE the signal
close, the dip-buy edge roughly halves (avgR ~0.45 -> ~0.23 in the ledger; the
bounce partly plays out at the open and the `Limit(Open-0.25ATR)` entry fills at
a worse price), so the trade is sized at `mult`. Deliberate risk-appetite
haircut, NOT a PnL win — the gap-up bucket is still net positive; the ledger
cost is ~-$16k (Monday Dip) / -$40k (SPY QQQ MonFri) flat over 23y, cutting
gap-up trades' Size_Mult ~0.97 -> ~0.48 while leaving all non-gap trades
byte-identical.

Key facts:
- **Sizing overlay, not a filter.** Composes multiplicatively with 3b3 frag
  bands (a high-fragility gap-up day = 0.25 x 0.5 = 0.125x). Distinct from SPY
  QQQ MonFri's `use_t1_gap_kill` (settings), a Friday-ONLY full DROP at 0.5 ATR
  enforced in `filters.get_historical_mask`. The kill runs first (removes the
  candidate); this derate then half-sizes whatever it leaves that still gaps
  > 0.25 ATR (non-Friday signals + Friday 0.25-0.5 ATR gaps). Both stay
  configured — neither replaces the other.
- **Only knowable at the open**, so live it is STAMPED by daily_scan and
  APPLIED by order_staging at the IBKR T+1 open (like MonGapKill / OVS 2-path);
  the scan itself never applies it. Fails OPEN (full size) on a missing
  open/ATR/signal-close — a haircut isn't worth dropping a valid fill.

Aligned sites — change together:
- `strategy_config.py` — execution `gap_size_derate` on both strats (source of
  truth; NOT GRM-scaled, it's a pure multiplier).
- `pages/strat_backtester.py` — `gap_derate_mult()` helper + sizing step 3b5
  (engine sees `entry_row['Open']` directly, so ledger == live). Drives the
  ledger + `daily_portfolio_report.py`.
- `daily_scan.py` — stamps `GapDerate_ATR` / `GapDerate_Mult` / `GapDerate_Dir`
  on every staging row (empty for strats without the field).
- `order_staging.py` (OneDrive) — reads the stamps, halves `Quantity` +
  sets `_GapMult` (so the daily caps see the reduced risk) + labels the row
  `DERATE_GAP`; gated on `path_label == ''` so a killed/gated/OVS row is never
  touched. Enforced at the live open right after the MonGap block.
- Guard: `tests/test_gap_size_derate.py` (config carriers, helper boundaries
  up/down, fail-open, frag-band composition, kill-coexistence).

## 3x Leader Gap Fade (pilot, 2026-07-10)

Capitulation fade on 3x ETFs whose UNDERLYING is spiking on fear. Universe =
`LEV3X_ALL` minus `LEV3X_BULL_EQ` (21 names: 13 bear-eq + TMF/TMV + 6 cmdty).
Filters: 2/5/10/21d rank > 80 (consec 1) AND 252d rank > 95 — the leader is
REQUIRED, the inverse of the other two 3x fades' <65 exclusion, so same-day
same-ticker cross-fire with them is impossible by construction (a ticker
cannot be <65 and >95 at once). Tape gate: T+1 open > close + 0.25 ATR,
resolved LIVE by order_staging's generic `T1_Open_Filters` gate (fail-closed;
scanner only stamps the JSON spec — it cannot see tomorrow's open). Entry:
Limit (Open + 0.75 ATR), OVS convention. 2-day time exit, NO STOP: stops
1.0-2.0 ATR (day-1 and day-2 armed) all destroyed the edge — adverse
excursion > 1 ATR is the normal path before the reversal (non-bull +23.7R ->
-39.8R at 1.0 ATR). The demanding entry IS the risk control (worst no-stop
trade -2.95R). Bull-eq exclusion is STRUCTURAL: every selectivity layer makes
bull-eq worse (strictest cell 0-for-7, avgR -1.28; losses span five bull
regimes 2018-2026) — do not re-add.

Sizing: 25 bps nominal (x GRM). Deliberately EXEMPT from frag_risk_bands and
same_day_signal_derate — the edge lives on exactly the high-fragility
multi-signal days those overlays would cut (Sept 2022, Apr 2025). Tail risk
is bounded instead by the per-strategy 250 bps daily cap (engine + live
aligned book-wide 2026-07-10 — see "Daily Risk Caps"; a 7-signal day at
37.5 eff = 262.5 bps trims ~5%). Validation
(2026-07-10): 31 trades / 15 episodes 2011-2025, avgR +0.80, PF 2.82,
episode-clustered t = 2.17, LOYO floor 1.55, drop-best-episode +9.4R @
t = 1.79, bootstrap P(<=0) = 2.1%. Pilot conviction — consider 40 bps only
after clean out-of-sample quarters.

Aligned sites — change together:
- `strategy_config.py` — the entry + `LEV3X_BULL_EQ` (source of truth)
- `pages/strat_backtester.py` / `daily_scan.py` — nothing bespoke; flows
  through generic paths (perf filters, T1_Open_Filters stamp, 0.75 ATR
  limit parse, max_one_pos, per-strategy daily cap)
- `order_staging.py` (OneDrive) — generic T1 gate enforces the gap at the
  IBKR T+1 open; per-strategy cap via the book-wide 250 bps default (a
  strategy-specific override existed for a few hours on 2026-07-10 and was
  removed when the book default aligned at 250)
- Guard: `tests/test_lev3x_leader_gap_fade.py`. Studies:
  `scratch/lev3x_fade_leader_*.py` (expansion, stops, entries, ovs_entry,
  class_split, bulleq_clusters, bulleq_strict, validation, capcheck,
  book_parity)

## Trend Sleeve (pilot, 2026-07-02)

`trend_sleeve.py` + `.github/workflows/trend_sleeve.yml`: monthly 12-ETF
trend-following ballast at 0.3x NAV (cut from 0.6x on 2026-07-17 to cap
overlap with the dial-gated SPY sleeve; combo = 12-1 momentum AND 10-month MA,
long/flat, inverse-vol slots capped 20%, cash otherwise). Universe = SPY QQQ
IWM EFA EEM FXI VNQ GLD SLV DBC TLT LQD — NO USO (roll decay) and NO UUP
(capital inefficiency: 20% slot for +0.00%/mo contribution + K-1; costs 2022
+0.5% -> -1.9%, accepted). Sector-ETF / intl-single expansion tested and
REJECTED (equity slots crowd out diversifiers, 2008/2022 flip negative);
exhaustion scale-down overlay REJECTED (Sharpe flat). Signals on the month's
last trading-day close, staged MOO (TIF=OPG) to the `Trend` Sheets tab for
next-session execution; held-share state in `trend_sleeve_state.json` (R2 —
the month-end run computes DELTAS against it; if staged orders were never
executed, clear the state or the next rebalance is wrong). The workflow runs
weekdays 21:35 UTC (AFTER update_master_prices' 21:10 PM cron — the script
hard-fails if today's close is missing) and no-ops except on the last trading
day; `Execute_On` (next ET trading day after the run) gates submission.
FULLY AUTOMATED end-to-end: order_staging.py (`load_trend_rows`) reads the
tab on Execute_On morning and emits naked-MOO rows (appended AFTER risk caps,
excluded from PA/execution_2); eq_order_entry.py places them as MKT/OPG
parent-only (Exit_Condition_Time='NONE' -> no exit legs — positions unwind
via future rebalance SELL rows). Ballast ONLY — it loses ~-0.4%/mo in
high-fragility months (frag_risk_bands handles that hole). Scale to 1.0x of
the fraction only after 2 clean quarters. Studies: scratch/tf_universe_study.py,
scratch/ultracode_research/trend-following.md + trend_prework_gates.md.

## OLV Vol-Confirmed Stop + Notional Cap (2026-07-20)

Package replacing OLV's resting 1.25 ATR STP, its `sector_loss_gate`
(live 2026-07-02 to 2026-07-20) and its `ladder_multipliers` in one change.
Evidence chain: scratch/ultracode_research/olv_stop_condition_2026-07-17.md
+ scratch/olv_package_sim.py (package $466k -> $654k flat / 21y, win 61->70%,
worst chain -$8.8k -> -$17.2k; every piece LOYO/episode-clustered).

**Stop (`stop_mode: 'vol_confirm_close'`, `stop_vol_mult: 1.5`)**: no
resting STP leg. Exit MOO at the NEXT open iff a session CLOSES <=
entry - 1.25 ATR AND its volume >= 1.5x the trailing 20d median (ex-that-
day). Quiet closes below the level are HELD — low-volume weakness is the
entry thesis; T+10 time exit still bounds everything. A volume-spike exit
and a fresh OLV signal (10d vol rank < 15) are near mutually exclusive, so
the old same-day stop+rebuy churn (39 events) is structurally gone.
`stop_atr` 1.25 still defines the sizing risk unit. Per-leg tails widen to
occasional -2..-3R; there is NO overnight stop (gaps evaluated at the next
close). Live flow (2026-07-30 — TRUE pre-market MOO): daily_scan
`stage_olv_vol_confirm_exits` (PM run evaluates today's settled close; AM
run re-evaluates with corrected data, risk-report-correction style; EVERY
open leg prints an explicit per-leg verdict line — CONFIRMED / no breach /
quiet breach / entry-day / stale — so stacked positions are auditable leg
by leg, and legs sharing (ticker, Time_Exit_Date) raise a warning because
downstream bracket matching can't distinguish them) -> `OLV_Exits` Sheets
tab (always cleared+rewritten; ONE ROW PER CONFIRMED LEG with
`Time_Exit_Date` bracket key + `Entry_Date` audit column; a STALE ticker
bar is never re-evaluated — its previously staged exits carry forward
PER LEG, keyed (Symbol, Time_Exit_Date)) -> `olv_exit_moo.py` (OneDrive
trading_ibkr; standalone Task Scheduler task 'IBKR OLV Pre-Market Exits',
weekdays 9:10 AM ET) reads the tab directly (Execute_On == today only) and
places SELL MKT **TIF=OPG** on BOTH accounts — a genuine market-on-open in
the opening auction, submitted before the 9:28 cutoff (past 9:25 it falls
back to MKT DAY, loudly). Same safety layers as the old path: matches the
leg's working OCA bracket by orderRef prefix + time-leg goodAfterTime
(nearest-date fallback for calendar desync), primary clamps qty to
min(staged, leg, held) while the PA sells min(leg, held) — the FULL
matched PA leg (staged qty is primary-basis and deliberately ignored),
cancels the bracket before selling, RE-ARMS a protective time-exit clone
on total placement failure, and journals placed exits
(olv_exit_placed.json) so re-runs are idempotent. It REUSES clientIds
99/98 on purpose — TWS binds persisted brackets to the placing clientId,
and it runs clear of the 9:31 chain. History: 2026-07-20..30 these rows
rode order_staging -> eq/pa_order_entry as "MOO" MKT DAY orders placed
~9:31+ — AFTER the open, never a real MOO (order_staging needs the live
open for the OVS gap check, so it can't run earlier). That staging path
was REMOVED 2026-07-30; the Is_Position_Exit handlers in eq/pa_order_entry
remain as dormant safety nets for hand-staged rows. Entry-day closes are
NEVER confirms (day-2 arming convention — the scan skips legs entered on
the evaluation session, matching the engine's entry_idx+1 loop). Every
layer fails SKIP/open and pipeline failures surface as OLV-EXIT warnings
in the daily scan email — a missed exit falls back to the T+10 time exit,
never a naked short.

**Notional cap (`ticker_notional_cap: {pct_nav: 0.50, exempt:
OLV_CAP_EXEMPT_ETFS}`)**: stacked OLV legs in ONE single-stock ticker may
not exceed 50% of NAV in entry notional; later legs scale down / skip. ETFs
exempt. Catastrophe insurance for the no-resting-stop world (~4% of OLV PnL
historically, every clipped leg a winner; balloon stacks are low-ATR names).
The engine binds the cap in FRACTION-OF-SIZING-EQUITY terms (each open
leg's notional / the equity it was sized against, `cap_equity` on
open_positions): flat pass == live's pct_nav x fixed ACCOUNT_VALUE, and
the compounded pass makes identical clip/skip decisions (either dollar
basis lets the passes diverge — NaN flat rows or 76 silently dropped
trades, both hit on 2026-07-20). KNOWN BOUND:
the cap counts FILLED positions only; with the T+3 fill window up to THREE
days' unfilled full-size limits are invisible, so worst-case concurrent
notional is ~3x one leg. Engine and live share the blindness (parity
holds); a working-order-aware check is the eventual fix.

**Sector gate removal**: the 20y drop list (-5.3R at ship) flipped to +10R
after the gate blocked the entire late-June-2026 oil recovery (OXY +2R x3,
USO winners) having saved only part of the decline. The generic gate +
`sector_gate_blocked` machinery survives dormant (keyed on the execution
field); `build_trade_ledger`'s nogate pass now SKIPS with a notice and the
site's gate_lab section quietly disappears. `data/sector_map.parquet` and
`scripts/build_sector_map.py` remain (other consumers). Ladder removal:
see "Ladder Sizing" above.

Aligned sites — change together:
- `strategy_config.py` OLV execution `stop_mode` / `stop_vol_mult` /
  `ticker_notional_cap` + `OLV_CAP_EXEMPT_ETFS` (source of truth)
- `pages/strat_backtester.py` — vol-confirm exit branch (target checked
  BEFORE the close-confirm; no confirm on the final hold day; next-open
  fill with stop slippage, no gap logic) + per-ticker notional cap replay
  (open_positions state, refund semantics mirror the net-exposure cap)
- `daily_scan.py` — Use_Stop stamped False for stop_mode strategies;
  `load_open_position_notionals` + sizing-step cap; `stage_olv_vol_confirm_exits`
- `olv_exit_moo.py` (OneDrive) — pre-market TIF=OPG exit runner for BOTH
  accounts + `run_olv_exit_moo.bat` / `register_olv_exit_task.ps1`
  (Task Scheduler 'IBKR OLV Pre-Market Exits', weekdays 9:10 AM ET);
  eq/pa_order_entry keep dormant `Is_Position_Exit` handlers;
  guard: `test_olv_exits.py` (OneDrive)
- Guard: `tests/test_olv_stop_and_cap.py` (engine + scan + config invariants)

Ledger SURVIVORSHIP CAVEAT (2026-07-16): the 23-year ledger trades only
tickers alive in today's universe files — 21 of 22 major 2020s delistings are
absent — which flatters long dip-buy stats and the ~870-name overflow tier
most. Treat overflow-tier historical avgR as an upper bound until the
dynamic-overflow work's point-in-time universe lands. Do not tune sizing off
overflow backtest stats alone.

Ledger provenance + integrity (2026-07-06, after a false TS/USO block): the
ledger is a FULL BACKTEST REBUILD, not a fill record -- marginal limit fills
flicker between vintages as yfinance revises recent bars, and the gate's -2.0R
threshold is a knife edge (the false block was -2.008R from a since-rebooked
trade in a weekend vintage of unknown origin). Mitigations, all in place:
- `build_trade_ledger.py` embeds provenance in the parquet schema metadata
  (`ledger_build_utc`, `ledger_source` = gha:<run_id> | local:<host>,
  `ledger_git_sha`, `ledger_rows`) and prints a vintage diff vs the prior
  ledger (new/gone/rebooked trades touching the last 15td) on every build.
- R2 upload of the prod ledger key is gated behind `--upload`; only
  `deploy_site.yml` passes it. Local runs (`refresh_view.py`) build but never
  overwrite the key that gates live orders.
- `daily_scan.py` prints the ledger's provenance at gate load and warns when
  the vintage is >4 days old or was built outside GHA (still fail-open).
- Blocked-signal notes name every contributing exit (ticker, date, R) so a
  block stays auditable after its vintage is overwritten.

## Stop-Arming Convention (book-wide, 2026-06-09)

Stop legs ARM AT THE NEXT SESSION, not at the fill. Decided after measuring
81 entry-day-stop episodes over 24y: booking -1R each vs arming on day 2 cost
-33R book-wide (dip-buy limit entries get stopped at max fear; a third of
MonFri's day-1 stop-outs went on to hit +2R targets).

Aligned across both sides -- change one, change both:
- `pages/strat_backtester.py` (`process_signals_fast`): entry-day stop check
  gated on `execution['stop_active_entry_day']`, **default False** (= day-2
  arming). Set True on a strategy to model a day-1-armed stop.
- `eq_order_entry.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`): STP
  child submitted with `goodAfterTime = next_session_gat()` (next trading day
  09:30, BDay-aware; holiday dates harmlessly defer to the next real session).
  Still in the OCA group, so a TARGET/TIME fill cancels the inactive stop.

Related conventions: entry-day TARGETS are never credited in the backtest
(intraday timing vs fill is ambiguous); OVS has `use_stop_loss=False` entirely
(its day-one valve is the Friday-only EOD-DD, see section above); OLV has NO
resting stop leg at all since 2026-07-20 (vol-confirmed next-open exit — see
"OLV Vol-Confirmed Stop + Notional Cap"; its rows stamp Use_Stop=False while
`use_stop_loss` stays True in config for the sizing risk unit).

## Stop-Fill Convention — gap-through + slippage (book-wide, 2026-06-27)

A stop the bar GAPS THROUGH fills at the OPEN, not the stop. The old engine
always booked the exit at exactly `stop_price`, which pinned every stop-out at
exactly -1R and understated the gap-down tail (the website showed OLV — and
every stop strategy — "never losing more than 1R"). The realized fill is now the
worse of the stop and that day's open, plus slippage.

`process_signals_fast` (`pages/strat_backtester.py`) — drives the full-history
ledger (`scripts/build_trade_ledger.py` -> site) AND `daily_portfolio_report.py`:
- `_stop_fill_price(direction, stop_price, day_open, gap_fill, slip_bps, gap_slip_bps)`
  is the single fill model. Long: `min(stop, open)`; Short: `max(stop, open)`.
- Slippage: `STOP_SLIP_BPS = 3.0` on EVERY stop fill, plus an ADDITIONAL
  `STOP_GAP_SLIP_BPS = 10.0` (so 13 bps total) when the bar gapped through.
  Always worsens the fill (long sells lower, short covers higher). Targets and
  time exits get NO slippage. OVS EOD-DD (close exit) is untouched.
- New kwargs `stop_gap_fill=True, stop_slip_bps=3.0, stop_gap_slip_bps=10.0`
  default to the prod behavior; pass `stop_gap_fill=False` to reproduce the
  legacy fill-at-stop for before/after measurement.
- Entry-day stop (off by default) gets slippage only — no gap-to-open, since the
  open precedes the intraday limit fill.
- Scale-invariant under the dividend-adjustment rule: the stop is relative and
  `Open` is on the same adjusted basis within a run, so both scale by the same
  factor (CLAUDE.md "Dividend-Adjustment Basis").

Impact (full book, 2003-2026, flat $750k, `scratch/stop_gap_slippage_impact.py`):
85 of 434 stop-outs (~20%) gapped through. Book TotR 605.9 -> 560.2 (-45.7R),
AvgR 0.525 -> 0.485, worst single trade -1.0R -> -4.56R, -$157.7k flat (~8% of
these strategies' PnL). OLV: 25/116 stops gapped, TotR 193.5 -> 182.9, worst
-1.0R -> -2.29R.

Live trading was already correct (IBKR STP -> market order fills at the gap
open); this only removes backtest/ledger/site optimism. `pages/backtester.py`
(interactive UI) is a separate engine: its persistent-limit path already does
`min(Open, stop)` (line ~2439); the simpler paths (~2312-2351) still fill at the
stop and would need the same treatment for full parity (deliberately separate
exploration surface, not yet aligned).

## OLV Entry-Order Live Window (T+3, 2026-06-24)

The OLV (Oversold Low Volume) persistent close-0.25 ATR limit is cancelled if
unfilled after **3 trading days** (T+1..T+3), not the full 10-day hold. A fill
inside the window is kept and unchanged (its hold is still reduced by wait time
off `hold_days`); a signal that hasn't filled by T+3 close is dropped.

Evidence (`scratch/olv_fill_window.py`, bucketing the full ledger by fill day):
89% of OLV fills land by T+3. The day 4-10 fills add ~0 total R (+211 -> +211 R
over 21y) while diluting per-trade edge: avgR +0.637 (T+3) vs +0.566 (T+10),
win 62.8% vs 60.6%, PF 2.90 vs 2.65. So total return is unchanged but
risk-adjusted quality improves, and capital isn't tied up in stale GTC orders
that mostly fill into names that kept bleeding for a week+.

Generic mechanism: `execution['fill_window_days']` caps the persistent-limit
fill search; **defaults to `hold_days` when absent**, so the other 5 persistent
strategies are untouched. Aligned sites (change together):
- `strategy_config.py` — OLV execution `fill_window_days: 3` (source of truth).
- `pages/strat_backtester.py` — `fill_window` bounds `search_end` in both
  persistent fill loops; the hold reduction still references `hold_days`.
  Drives the ledger + `daily_portfolio_report.py`.
- `daily_scan.py` — stamps `Fill_Window_Days` on every primary staging row.
- `order_staging.py` (in `OneDrive\trading_ibkr\`) — stamps `Entry_Expire_Time`
  = signal + `Fill_Window_Days` BDays = `Exit_Condition_Time` − (1 + hold − fill)
  BDays, into the execution CSV + `execution`/`execution_2` tabs. Defaults to
  `Exit_Condition_Time` (the 10-day-hold expiry) unless the row carries a valid
  `0 < Fill_Window_Days < Hold_Days`, so only OLV is affected.
- `eq_order_entry.py` + `pa_order_entry.py` (same dir) — the persistent GTC
  parent's `goodTillDate` reads `Entry_Expire_Time` (falls back to the time-exit
  `gat_time` when absent/blank). The TIME exit leg still uses `gat_time`, so a
  filled OLV position keeps its full reduced hold — only the unfilled entry order
  is cancelled early. The order is live T+1..T+3 (expires T+3 15:59).
- `pages/backtester.py` UI still uses `holding_days` as its fill window (an
  exploration surface, deliberately separate from the prod-locked rule).
- Regression coverage: `tests/test_olv_fill_window.py` (backtest engine);
  the live date math is validated by the entry-expire chain (daily_scan exit-date
  build ↔ order_staging back-computation, identical `CustomBusinessDay` calendar).

## Cloudflare R2 Cache + GHA Migration

As of 2026-04-30, the nightly pipeline runs entirely in GitHub Actions. The local Task Scheduler retains the radar tasks plus (as of 2026-05-13) two AM `workflow_dispatch` triggers that bypass GitHub's congested 8-9 UTC cron-queue lag. R2 is the persistence layer that lets cloud workflows share parquet caches.

### R2 secrets (in GHA repo settings)
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET=seasonals-cache`

### Bucket contents (key-value)
- `master_prices.parquet` — full ~2000 ticker × 25-yr OHLCV (~50-200 MB). Read by `daily_scan` (ALL scopes — **cache-first for every ticker**, incl. the liquid + 3x-ETF universes; yfinance is only a fallback for names the cache lacks, e.g. carets/delisted) and `daily_portfolio_report.py`. As of 2026-06-11 the 42 LEV3X names (DUST/JDST/TQQQ/…) were backfilled in so the liquid scan no longer depends on a live pre-market yfinance pull (that pull returned a stale bar on 2026-06-11 and silently zeroed the liquid tier). Written by `update_master_prices.yml` twice on weekdays (AM via local workflow_dispatch ~4:17 AM ET + PM via 20:30 UTC cron); its universe = whatever tickers already exist in the parquet, so backfilled names are auto-maintained. Pre-market runs pass `--exclude-today` so yfinance placeholder bars never enter the cache.
- `earnings_calendar.parquet` — FMP-backfilled (117k rows, 946 tickers). Read by `daily_scan` (any scope, OVS filter) and `daily_portfolio_report.py`. Written by `build_earnings_calendar.yml` weekdays at 21:30 UTC + the local belt-and-suspenders entry at the same slot.
- `intraday/15min/{TICKER}.parquet` + `intraday/15min/_meta.parquet` — 15min OHLCV cache. Historical depth backfilled from FMP (2003-present), ongoing maintenance via yfinance (60d rolling, no API key). Target universe is `LIQUID_PLUS_COMMODITIES` (~197 tickers, ~3 MB each, ~600 MB total). Read by `intraday_data.py` (lazy R2 refresh on stale local copies, 18h staleness window) which feeds Day Trade Limit modes in `pages/backtester.py`. Written by `update_intraday_prices.yml` weekdays at 20:45 UTC. Caret tickers (^GSPC, ^NDX) excluded — FMP doesn't serve them. Full architecture in `docs/intraday_data_plan.md`.

### `cache_io.py` API
```python
from cache_io import upload_from_local, download_to_local, is_configured

upload_from_local("data/foo.parquet", "foo.parquet")   # local → R2
download_to_local("foo.parquet", "data/foo.parquet")   # R2 → local
is_configured()                                          # bool: R2_* env vars set?
```
Both helpers no-op gracefully when R2 isn't configured (returns False, prints a notice). ASCII-only output to avoid Windows cp1252 crashes when running locally.

## Automated Pipeline

All five trading-day workflows now run in GHA. Order staging stays local (IBKR-bound).

| Workflow file | Schedule | What it does |
|---|---|---|
| `daily_screener.yml` | Weekdays 2x: AM via local workflow_dispatch at 4:47 AM ET (fallback GHA cron at 10:30 UTC, auto-skipped if dispatch succeeded today) + PM cron at 22:00 UTC | Unified scan, both runs `--scope=all` (full liquid + overflow, ~7-10 min). AM run also writes `data/exposure_state.json` and commits it back to main. Intraday MOC slots were retired when the strategy book lost its last Signal Close entry; restore them if MOC strategies are added back. |
| `build_earnings_calendar.yml` | Weekdays 21:30 UTC (5:30 PM ET) | FMP `/stable/earnings` pull → writes `data/earnings_calendar.parquet` → uploads to R2. Local `EarningsCalendarRefresh` Task Scheduler entry mirrors this for redundancy (last write wins). |
| `update_master_prices.yml` | Weekdays 2x: AM via local workflow_dispatch at 4:17 AM ET (fallback GHA cron at 9:30 UTC, auto-skipped if dispatch succeeded today) + PM cron at 20:30 UTC (4:30 PM ET) | Pulls `master_prices.parquet` from R2, fetches today's bars from yfinance for ~2000 tickers, appends, dedupes, writes back to R2. PM cron pulls today's close; every other trigger (AM dispatch, AM fallback cron, manual dispatch) passes `--exclude-today`. |
| `update_intraday_prices.yml` | Weekdays 20:45 UTC (4:45 PM ET) | Pulls per-ticker 15min parquets + meta from R2, runs `scripts/update_intraday_yfinance.py --upload` — fetches recent bars from yfinance for every ticker in meta, converts UTC→ET, appends, dedupes, writes back. yfinance has 60d rolling intraday history so this must run at least every ~50 days to avoid gaps; weekday cadence is fine in practice. |
| `portfolio_report.yml` | Weekdays 21:30 UTC (5:30 PM ET) | Pulls master_prices + earnings caches from R2, runs `daily_portfolio_report.py`, sends HTML email + writes Portfolio Sheets tab. |
| `bootstrap_caches.yml` | workflow_dispatch only | One-shot: builds `master_prices.parquet` from scratch via yfinance (~10-15 min for ~2000 tickers, 25-yr history) and uploads to R2. Used to seed the bucket (already run during Phase 2 setup). |
| `risk_report.yml` | Weekdays 2x: PM cron 21:15 UTC (5:15 PM ET, full run w/ email) + AM correction at 4:30 AM ET via local workflow_dispatch (fallback GHA cron 9:00 UTC), `--data-only --refresh-last` | Daily risk dashboard email (fragility dials + signals + forward returns). Writes `data/rd2_fragility.parquet` APPEND-ONLY (since 2026-07-02): history is frozen point-in-time, only new dates append. The PM run's newest row comes from a just-closed yfinance bar that can be provisional, so the AM correction (2026-07-17) refreshes ONLY the last session's row with settled prices before daily_scan sizes off it at ~4:47 AM — `merge_fragility_history(refresh_from=prev_bday)`; older rows never mutate. Same correction re-evaluates the dial-sleeve paper track's provisional day (dial_sleeve rollback) and the simple-dial shadow. Guards: `tests/test_fragility_append.py`, `tests/test_dial_sleeve.py`. This series sizes live orders (`frag_risk_bands`) — do not revert to full rewrites; recompute vintages drifted up to ~7 pts on the 63d dial. |
| `verify_fills.yml` | Weekdays 21:15 UTC | Post-close fill verification — updates Trade_Signals_Log. |
| `deploy_site.yml` | Reusable workflow (`workflow_call`), invoked by the `deploy-site` job at the tail of `daily_screener.yml` (`needs: run-scanner`) so it runs in the SAME run, right after the scan succeeds — 2x/trading day (after the ~4:47 AM ET dispatch scan and the PM bookend). Replaced the old best-effort `workflow_run` chain, which was silently not firing. A skipped (AM fallback) or failed scan skips the deploy and the prior deploy stays up. `workflow_dispatch` retained for manual rebuilds. | Builds + deploys the private analytics site to Cloudflare Pages (behind Cloudflare Access). Pipeline: R2 caches → `scripts/build_trade_ledger.py` (full-history ledger) → `scripts/build_signal_charts.py --all --upload --skip-existing` (renders only NEW per-trade charts to R2, best effort) → `daily_seasonal_ideas.py` (best effort) → `scripts/build_risk_json.py` (best effort) → `scripts/build_site.py` (JSON payloads + `site/` assets → `dist/`) → wrangler Pages deploy (config-driven via `wrangler.toml`; no positional dir, so the CHARTS R2 binding applies). Needs `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` secrets. One-time setup: `docs/private_site_setup.md`. Operational runbook (failure modes, decisions log,
trigger chain, out-of-repo file map): `docs/site_runbook.html`. |
| `weekly_rundown.yml` | Sundays 14:00 UTC (9 AM ET) | Tabloid PDF with all risk charts + radar digest body. |
| `trend_sleeve.yml` | Weekdays 21:35 UTC (no-ops except the month's last trading day) | Monthly trend-following ballast rebalance — writes MOO orders to the `Trend` Sheets tab + state to R2. See "Trend Sleeve" section. |
| `execution_report.yml` | Weekdays 20:30 AND 21:30 UTC — `daily_execution_report.py` gates on WHICH cron fired (`GHA_SCHEDULE` = `github.event.schedule`) + the date's DST regime, so exactly one sends at ~4:30 PM ET year-round even when GHA cron lag starts the run an hour+ late (the old hour==16 gate silently dropped the 2026-07-08/09 emails). Cron strings must match `EDT_CRON`/`EST_CRON` in the script; unknown cron fails open (sends). | Nightly email of LIVE primary-account positions (mirrors the site Execution tab). Pulls the execution-broker DO's `/book` snapshot (Bearer `STATUS_TOKEN`; URL via `EXEC_BROKER_URL` secret, defaults to the workers.dev URL), excludes OPT rows, and enriches each position from its own working exit legs: target = closing LMT `lmt`, stop = closing STP `aux` (NA if none), time stop = closing MKT leg's `goodAfterTime` date, strategy = 3rd pipe field of any leg's `orderRef` (requires `order_ref` in `book_snapshot.py` — added 2026-07-08 in `OneDrive\trading_ibkr`). No strategy-tagged legs → "Trend Sleeve" (symbol in `trend_sleeve_state.json` on R2) else "Discretionary". Recipients: repo variable `EXECUTION_REPORT_RECIPIENTS` (comma-separated; defaults to mckinleyslade@gmail.com). Guard: `tests/test_execution_report.py`. |

### Local Task Scheduler (post-Phase-2)

| Task | State | Notes |
|---|---|---|
| `EarningsCalendarRefresh` | Enabled | Belt-and-suspenders for the GHA equivalent. Both write to R2. |
| `Trigger Update Master Prices (GHA workflow_dispatch)` | Enabled | Weekdays 4:17 AM ET — fires `update_master_prices.yml` via the GitHub REST API to bypass shared-cron queue lag at 8-9 UTC. See "AM Trigger Architecture" below. |
| `Trigger Daily Screener (GHA workflow_dispatch)` | Enabled | Weekdays 4:47 AM ET, 30 min after the parquet trigger — fires `daily_screener.yml` via the GitHub REST API. Same mechanism. |
| `Trigger Risk Report AM Correction (GHA workflow_dispatch)` | Enabled | Weekdays 4:30 AM ET — fires `risk_report.yml` with `mode=data_only` so the fragility row daily_scan sizes off (~4:47 AM) reflects settled prices, not the provisional 5:15 PM bar. Scripts: `C:\Scripts\trigger_risk_report.ps1` + `_task.xml`. |
| `IBKR OLV Pre-Market Exits` | Enabled | Weekdays 9:10 AM ET — `run_olv_exit_moo.bat` -> `olv_exit_moo.py` (OneDrive trading_ibkr): reads the `OLV_Exits` tab and places TRUE market-on-open (TIF=OPG) SELLs for confirmed OLV stop legs on BOTH accounts before the 9:28 auction cutoff. Registered 2026-07-30 via `register_olv_exit_task.ps1`; must clear the 9:31 order chain (shares clientIds 99/98). |
| `RadarMorningBriefing` | Enabled | Lives in separate `last30days-radar` project — not yet migrated. |
| `RadarWeeklySummary` | Enabled | Sundays 8:30 AM ET — depends on radar briefs from above. Not yet migrated. |
| `DailyPortfolioReport` | Disabled | Replaced by `portfolio_report.yml`. Re-enable as fallback if GHA breaks. |
| `MasterPricesUpdate` | Disabled | Replaced by `update_master_prices.yml`. |
| `OverflowDailyScan` | Disabled | Replaced by the unified `daily_screener.yml --scope=all` post-close run. |

Order staging (`C:\Users\mckin\OneDrive\trading_ibkr\order_staging.py`) is a manual / scheduled local launch — talks to IBKR TWS on `127.0.0.1:7496`. Reads `Order_Staging` + `Overflow` Sheets tabs and submits orders pre-market.

### AM Trigger Architecture (added 2026-05-13)

GitHub's shared cron scheduler had 1-3h queue delays at 8:47 UTC, pushing the AM scan past pre-market staging deadlines. Fix: fire the AM runs from this machine via the GitHub REST API (`workflow_dispatch`), which has near-zero queue lag.

**Daily flow (weekdays):**
- **4:17 AM ET** local task → POST `…/update_master_prices.yml/dispatches` → GHA queues immediately, runs ~5 min
- **4:47 AM ET** local task → POST `…/daily_screener.yml/dispatches` → GHA queues immediately, runs ~7-10 min

**Fallback** (machine off / network outage): both workflows keep an early GHA cron (parquet 9:30 UTC, screener 10:30 UTC). Each workflow's first job (`check`) queries the GitHub API for today's `workflow_dispatch` runs and short-circuits if a successful one already exists; otherwise the main job runs. The fallback cron is subject to GHA's queue lag but still beats market open by ~3h in the worst case.

**Local artifacts:**
- Trigger scripts: `C:\Scripts\trigger_update_master_prices.ps1`, `C:\Scripts\trigger_daily_screener.ps1`
- Task XMLs: `C:\Scripts\*_task.xml` (S4U principal, WakeToRun, no AC required, restart-on-failure 5min × 3)
- Logs: `C:\Scripts\logs\trigger_*.log` (one line per dispatch attempt)
- PAT: `HKCU\Environment\GH_PAT_NEW_SEASONALS` (fine-grained, scoped to `mslade50/New_Seasonals`, permissions: Actions/Workflows/Contents — read+write, Metadata read). Rotate annually.

**Maintenance:** if the local task or PAT breaks, the fallback cron picks up the slack the same day. If both break, the PM cron at 20:30 / 22:00 UTC still runs (independent of any of this).

### Sunday Pipeline (two-step, still partially local)
1. **8:30 AM ET (local)**: `radar_weekly_summary.py` reads last 7 days of radar briefs from `C:\Users\mckin\projects\last30days-radar\output\briefs\`, pulls yfinance snapshots for all tickers, pipes to Claude Code subprocess with PM-style distillation framework (variant perception required, "who's on the other side" required). Output committed + pushed to `data/radar_weekly_summary.md`.
2. **9:00 AM ET (Actions)**: `weekly_market_rundown.py` generates tabloid (17x11") landscape PDF with all risk charts, reads the radar digest and includes it as styled HTML email body alongside the PDF attachment.

### Daily Risk Report — Forward Returns Table
Uses `compute_similar_reading_returns()` from `risk_dashboard_v2.py`. Forward returns at similar fragility readings include:
- Mean and Median conditional returns
- **Mean Z / Median Z** — z-scores vs unconditional sample (mean via z-test, median via bootstrap SE with 1000 resamples)
- % Negative and Baseline (unconditional mean)
- Mean column color follows Mean Z thresholds (green >= 0, yellow > -1, red <= -1)

### Radar Weekly Digest Framework
The Claude prompt enforces heavy filtration via two required gates:
- **Variant Perception**: Must articulate specific disagreement with market pricing. If thesis = consensus, idea is killed.
- **Who's on the other side**: Must identify why the opportunity exists (forced selling, informed disagreement, or neglect). If can't identify, idea is killed.

Supporting lenses (non-dogmatic, context-dependent): catalyst magnitude, valuation vs forward reality, trend/market structure, persistence across the week, crowd positioning.

Framework doc: `C:\Users\mckin\Documents\vault\trading\decisions\radar_weekly_digest_framework.md`

## Private Site (Cloudflare Pages)

Static, client-side analytics site deployed nightly by `deploy_site.yml` to
Cloudflare Pages project `seasonals-mslade`, locked behind Cloudflare Access
(email OTP, allowlist = mckinleyslade@gmail.com). One-time setup doc:
`docs/private_site_setup.md`.

- **Frontend** lives in `site/` (committed): `index.html` (portfolio app),
  `signals.html`, `charts.html` (per-trade chart gallery), `risk.html`,
  `montecarlo.html` + `assets/` (vanilla JS + Plotly CDN, no build step, no
  framework). `site/_headers` sets no-store on `/data/*`. Nav order
  (2026-07-28): Portfolio, Seasonal, Execution, Risk, Trade Log, then the
  rest, Monte Carlo last. The IDEAS TAB was REMOVED 2026-07-28 (page +
  ideas.js deleted); `ideas.json` is still built — the signals page's
  strategy-context block reads it.
- **Monte Carlo tab** (`montecarlo.html` + `assets/montecarlo.js`,
  2026-07-28): day/month/year outcome distributions for the current book —
  empirical daily stats (P(up), loss-threshold frequencies, VaR/CVaR, worst
  days) + stationary block bootstrap (10k sims, mean block 10td, seed 42) for
  21td/252td bands, within-horizon maxDD and P(>=1 down day < -1.5%).
  Payload: `build_monte_carlo()` in build_site (best effort) reads the DAILY
  pnl_flat parquet the ledger build wrote the same run; flat $750k basis.
  Also carries an INTRADAY drawdown-touch section (`build_intraday_touches`,
  needs the price map so the call lives in the priced block; --no-mtm dev
  builds ship the sim without it): per-day book trough from open positions'
  Low/High vs prior close / entry price — a pessimistic bound (per-ticker
  extremes not simultaneous; limit entries make entry days near-tight), close
  marks reconciled to booked fills. Renders as touch-frequency table +
  trough histogram + trough-vs-finish scatter. Drawups deliberately omitted:
  entry-day extremes can predate the fill, favorable side unknowable from
  daily bars. Studies: scratch/portfolio_monte_carlo.py +
  scratch/intraday_excursion_study.py.
- **Trade Log tab** (`tradelog.html` + `assets/tradelog.js`, 2026-07-24):
  actual IBKR executions for BOTH accounts (Primary TWS + PA Gateway).
  `book_snapshot.py` (OneDrive) appends today's fills (`ib.reqExecutions`)
  to each account's book push; the broker DO strips them from the stored
  book and folds them into per-day `fills:YYYY-MM-DD` storage keys —
  upsert by `exec_id` (commission reports lag a beat), 14d retention,
  500/day cap (DO 128 KiB per-value limit) — served at GET `/fills` and
  proxied by `functions/exec-fills.js`. IBKR only serves the CURRENT day's
  executions, so the DO ring IS the history: it accumulates from ship date
  and loses any day the agent never ran. Page aggregates per order
  (account+perm_id+side, VWAP) with a raw-fills toggle; strategy = 3rd pipe
  field of orderRef (same contract as `daily_execution_report.py`).
  Guard: `tests/test_tradelog_site.py`.
- **Payload contract** (written by `scripts/build_site.py` into `dist/data/`):
  `meta.json`, `trades.json` (columnar full ledger), `strategy_daily.json`
  (per `Strategy||Tier` daily MTM PnL on the FLAT $750k basis + book totals),
  `positions.json`, `exposure.json`, `correlation.json`, `charts.json`
  (per-trade chart manifest: stable image path + MAE/MFE), plus optional
  `ideas.json` / `signals.json` (Sheets snapshot) / `risk.json` / `fragility.json`
  (rd2 fragility dial series feeding the portfolio page's interactive sizing
  adjuster — per-trade what-ifs on dial/MA/threshold/floor/boost; forces the
  realized-at-exit curve basis while active) / `gate_lab.json` (sector-loss-gate
  counterfactual: blocked trades + gate-on/off realized curves, diffed from
  `data/backtest_trades_nogate.parquet` — a no-gate engine pass
  `build_trade_ledger.py` writes alongside the ledger; drives the portfolio
  page's gate-history section and its "All trades (+gate-blocked)" filter
  toggle, which also forces the realized-at-exit basis while on. DORMANT
  since 2026-07-20: no strategy carries the gate, the nogate pass skips,
  the payload stops being produced and the section auto-hides) /
  `ext_lab.json` (OVS hold-extension counterfactual — what-if lab, NOT a live
  rule: losing T+2 time exits rebooked to T+5 with the 2-ATR target live, a
  post-pass `build_trade_ledger.py` writes to
  `data/backtest_trades_ovsext.parquet`; drives the portfolio page's
  hold-extension section and its "OVS losers to T+5" filter toggle, which
  swaps the rebooked exits in by trade_id and forces the realized-at-exit
  basis while on. Evidence: scratch/ovs_hold_extension_*.py).
- **Trade charts** (the `charts.html` gallery): `scripts/build_signal_charts.py`
  renders a candlestick per trade (126 td before signal -> trade -> 63 td after
  exit; white/black candles, green/red volume, Signal/Entry/Exit verticals,
  dotted entry/stop/target, MAE/MFE stats box) and uploads to R2 under the
  `charts/` prefix. Keys are STABLE (`signals/<strategy>/<TICKER>_<YYYYMMDD>.png`,
  see `signal_chart_common.chart_relpath`) — not trade_id (reshuffles) or exit
  type (can flip). The site never bundles the PNGs (~360 MB); the
  `functions/chartimg/[[path]].js` Pages Function streams them from the `CHARTS`
  R2 binding on demand (route `/chartimg/*` -> R2 key `charts/*`; route differs
  from `/charts` so it doesn't shadow the gallery page). `deploy_site.yml`
  renders only NEW charts each run (`--all --upload --skip-existing`, best
  effort). Full backfill: `python scripts/build_signal_charts.py --all --upload`.
- **Sizing-basis rule**: client-side filtering recomputes everything on the
  flat $750k basis because per-trade dollars are additive. Strategy/tier/date
  filters get exact daily MTM curves (sum of per-strategy series); every
  OTHER selection (direction/ticker filters, gate + extension toggles,
  fragility multipliers) sums per-trade daily MTM vectors from
  `trade_mtm.json` (built by `build_trade_mtm` — ~21k marks book-wide,
  ~300 KB; mirrors `get_daily_mtm_series` conventions, each vector
  reconciles to the trade's booked PnL; includes vectors for gate-blocked
  rows keyed `Strategy|Tier|Ticker|SignalDate` and ext-rebooked rows by
  trade_id), so Sharpe/CAGR/vol stay on one basis everywhere. The
  realized-at-exit step curve survives only as a last-resort fallback when
  the payload is absent (old builds); the UI badges it. The compounded curve
  is shipped read-only — it cannot be decomposed per-filter (sizing depended
  on whole-book equity).
- **Local dev**: `python scripts/build_site.py --no-signals` then
  `python -m http.server 8123 --directory dist`. `--no-mtm` skips the slow
  payloads when iterating on frontend only.

## Google Sheets Integration

Tab layout in the `Trade_Signals_Log` workbook:
- `Order_Staging` — Liquid-tier signals (Limits, T+1 Open, Persistent GTC). Cleared + rewritten by every `daily_scan` run with `Scan_Source='Liquid'`.
- `Overflow` — Overflow-tier signals (same entry types, no MOC). Cleared + rewritten by `daily_scan --scope=overflow|all` with `Scan_Source='Overflow'`.
- Both staging tabs carry a `Manual_Limit` column (emitted empty by the scanner): type a price into it to pin that signal's entry — order_staging uses it verbatim as a LMT and anchors the bracket to it, skipping the gap clamp. Rows survive only until the next scan's clear+rewrite, so manual rows/pins must be added AFTER the ~4:47 AM ET scan and BEFORE order_staging runs (e.g. the 2026-07-06 TS/USO makeup rows via `scratch/stage_makeup_ts_uso.py`). Entry expiry is back-computed from `Time_Exit_Date` − (1 + `Hold_Days` − `Fill_Window_Days`) BDays, NOT from `Scan_Date`, so a makeup row can carry its true original schedule.
- `OLV_Exits` — vol-confirmed OLV stop exits (2026-07-20). Cleared+rewritten
  by BOTH bookend `daily_scan` runs (`stage_olv_vol_confirm_exits`); rows are
  per-LEG (stacked positions get one row per confirmed leg, keyed by
  `Time_Exit_Date`, with an `Entry_Date` audit column since 2026-07-30).
  Consumed by `olv_exit_moo.py` (OneDrive trading_ibkr) — the standalone
  pre-market task (weekdays 9:10 AM ET) that places rows with `Execute_On`
  == today as TRUE market-on-open SELLs (TIF=OPG, both accounts; PA sells
  the full matched PA leg). order_staging stopped consuming this tab
  2026-07-30 — its 9:31 post-open run could never deliver a real MOO.
- `moc_orders` — MOC entries from liquid tier only (`save_moc_orders` skips overflow rows). Currently vestigial: the strategy book has no Signal Close entries, so this tab is never written. Reactivates automatically if any strategy is set to `entry_type='Signal Close'`.
- `Seasonal` — tradeable seasonal-ideas tickets (longs + non-equity shorts). Written by `seasonal_order_staging.py` from `data/daily_seasonal_ideas.json`, `Scan_Source='Seasonal'`. Separate pipeline from the systematic book. Entry type per instrument (validated geography rule): US single stocks + US-session equity ETFs → `REL_OPEN` limit (0.25 ATR, DAY); everything that gaps overnight (intl/commodity/bond/FX ETFs, GLD/TLT) → `MOO` (market-on-open, `TIF=OPG`). Sizing: 20 bps/trade (13 bps in midterm years, `year%4==2`), 1% aggregate daily cap. order_staging must add `MOO` handling — see `docs/seasonal_order_staging_spec.md`.
- `sznl_nostage` — NOT auto-executed. Single-stock equity shorts (sized, tagged `[eq-short]`) + non-tradeable signals (futures/index/FX/crypto, `Quantity=0`, `Order_Type=NONE`, tagged `[need-proxy]` pending the proxy-ETF promotion). order_staging does not read this tab.
- `Trade_Signals_Log` (sheet1) — append-only signal history.
- `Portfolio` — open-positions snapshot from `daily_portfolio_report.py`.
- `execution`, `execution_2` — order_staging.py output for primary + small-account execution.

`daily_scan.py` writes both `Order_Staging` and `Overflow` via `save_staging_orders(..., tier_filter='Liquid'|'Overflow')`. The function clears+rewrites only the tier it's responsible for (so a `--scope=liquid` run never touches `Overflow`).

`order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) reads BOTH tabs and concatenates with `Scan_Source` distinguishing tier. Applies the OVS 2-path gap-tier sizer + path-2 daily aggregate cap + global 2.5% daily risk cap before submitting to IBKR.

`verify_fills.py` updates Trade_Signals_Log with fill status post-close.

Auth: `gspread` with GCP service account from Streamlit secrets / `GCP_JSON` env var (GHA) / `credentials.json` (local).
