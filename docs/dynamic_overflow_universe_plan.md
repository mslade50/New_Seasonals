# Dynamic Overflow Universe — Implementation Plan

**Status:** DRAFT (awaiting verification + sign-off — no code committed)
**Author:** Claude (Opus 4.8) + McKinley
**Date:** 2026-06-04
**Goal:** Replace the static `CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES` overflow tier (~864 names
frozen in `sznl_ranks.csv`) with a **dynamically rebuilt, liquidity- and volatility-screened
universe** that reaches deep into small/mid-caps — maximizing trade breadth in higher-opportunity
areas while keeping every name actually tradeable.

---

## 0. Revisions After Adversarial Review (2026-06-04)

Three reviewers (correctness/blast-radius, data/runtime feasibility, trading-risk) verified the draft.
All findings were valid; the design below is revised accordingly. **Defaults chosen here are
conservative and fully parameterized (config constants) — tune after backtest.**

**Correctness fixes**
- **R-C1 (was wrong):** `atr_sznl_filters` is non-empty for **3** of the 5 overflow-eligible
  strategies — **52wh Breakout** (`strategy_config.py:207`), **Overbot Vol Spike** (`:488`), **St OS
  Sznl** (`:695`). OLV (`:416`) and LT Trend ST OS (`:599`) have empty filters. A ticker absent from
  `atr_seasonal_ranks.parquet` **fail-closes to `False`** (`check_signal`, `daily_scan.py:826`) →
  zero signals on those 3. So the seasonal parquet MUST cover new names for 3 strategies, not 2.
- **R-C2:** `build_master_prices.py` unions `CSV_UNIVERSE ∪ sznl_ranks.csv ∪ seasonal_ranks.csv ∪
  INDICES_AND_ETFS`. The expanded universe must **preserve** `seasonal_ranks.csv` (don't drop it).

**Feasibility fixes**
- **R-F1:** `/stable/stock-list` is not the stable symbol endpoint (it's legacy v3); the stable
  `/stable/company-symbols-list` lacks `exchange`/`type`. → **Treat `/stable/company-screener` as
  authoritative; drop the broken fallback.**
- **R-F2:** FMP screener takes a **single** `exchange` per call and has a default result cap. →
  **Loop one call per exchange (NASDAQ/NYSE/AMEX), pass explicit `limit`, and assert the returned
  count is under the cap** (warn + tighten if hit).
- **R-F3 (material):** `load_master_prices_dict` does `pd.read_parquet(path)` with no pushdown,
  loading the whole file then filtering — at ~4k×20y (~20M rows) this risks **OOM on GHA's 7GB
  runner**. → **Add pyarrow predicate + column pushdown:** `read_parquet(path,
  filters=[('ticker','in',wanted)], columns=[...])`. (Draft §7.2 "no change needed" was wrong.)
- **R-F4:** Bootstrap of master_prices for ~4k×20y is memory-fragile and yfinance/GHA-throttle-prone.
  → **Run the one-time bootstrap LOCALLY**, with checkpointed per-chunk shards, higher retries, and
  jittered backoff. Do not rely on GHA for the initial backfill.
- **R-F5:** ProcessPool over the ticker loop must pickle large shared maps (`sznl_map`,
  `atr_sznl_map`, `xsec_rank_matrices`) → realistic gain ~1.5–2× on GHA 2-core, not Nx, and the loop
  mutates enclosing scope (`error_tickers`, `signals`, ticker reassignment for SPOT_TO_TRADEABLE /
  OVS qty cap). → **Prioritize pushdown (R-F3) + compute-only-needed-indicators first.** Parallelism
  is implemented as an opt-in `--workers` flag defaulting to **1** (behavior-preserving), via a pure
  `process_ticker(...)` function + pool `initializer` setting the maps as worker globals.

**Trading-risk fixes**
- **R-T1 (HIGH — survivorship/lookahead):** backtests must use **point-in-time** universe
  membership (rolling ADDV/ATR as-of each historical date), not today's screen. Small-cap
  mean-reversion is the family most inflated by survivorship. → `build_overflow_universe` computes
  the screen **per-date-capable** (no use of latest-only values); full point-in-time wiring into
  `strat_backtester` is a **required pre-promotion step** (flagged, larger change). Until then the
  rollout PnL number is treated as an optimistic ceiling; the trustworthy comparison metrics are
  **turnover / signal count / per-name concentration**.
- **R-T2 (HIGH — risk budget):** the binding aggregate control is the **2.5% daily cap inside the
  external `order_staging.py`**, not scanner-side. Uncapped breadth × fixed budget ⇒ dilution or
  arbitrary truncation. → Add a **per-strategy daily signal cap** (top-N by setup quality) as the
  budget-allocation lever, and **measure how often the budget binds** in dry-runs. Coordinate the
  2.5% policy with the external repo (out-of-repo, flagged).
- **R-T3 (HIGH — capacity/shortability):** $3MM ADDV is too thin for OVS **shorts** (borrow/locate +
  slippage). → **Tier the ADDV floor per strategy:** base **$3MM** for patient GTC-limit strategies
  (OLV, LT Trend ST OS, St OS Sznl); **$10MM** for OVS; **$5MM** for 52wh Breakout. Universe parquet
  carries `addv_63d`; the scan applies the per-strategy floor. Add an **OVS shortability note**
  (manual borrow check / restrict to the higher tier).
- **R-T4 (MED — ATR floor distorts OVS):** a 1.5% universe-wide ATR floor selects chronic-high-vol
  names and degrades OVS's spike-from-calm edge. → **Lower the universe ATR floor to 0.5%** (cut only
  dead names; each strategy keeps its own `min_atr_pct`). Make it a config constant; reserve any
  higher floor for a *specific* strategy after a backtest shows it helps.
- **R-T5 (MED — earnings on small-cap shorts):** OVS NaN-as-True trades small-caps *through*
  earnings on FMP-coverage gaps. → For the **overflow tier OVS only**, **SOFT drop**: still stage the
  signal but flag the missing coverage in `Sizing_Notes` ("⚠️ No earnings data — verify before fill")
  and an `Earnings_Cov='MISSING'` column, so it's eyeballed before fill rather than silently traded or
  silently killed. Keep silent NaN-as-True for the liquid tier (commodity/index names legitimately
  have no earnings). (Updated 2026-06-05 per McKinley: soft flag, not hard drop.)
- **R-T6 (MED — held-name drop monitoring):** weekly diff log must **flag dropped names currently
  held** (cross-ref the Portfolio snapshot) and the drop reason (liquidity vs freshness vs ATR).
- **R-T7 (LOW — quality gate):** make a **data-quality gate a membership criterion** (drop names with
  >X% NaN bars or split/price discontinuities), not just a logged metric. Ensure the weekly workflow
  rebuilds `atr_seasonal_ranks` + `earnings_calendar` **before** `overflow_universe` is consumed.
- **Sizing guard (kept, hardened):** the ADV participation cap (default **2%**) is enforced
  **scanner-side** (size reduced + `addv_63d` stamped on the row), not deferred to the external repo.
- **Activation gate (added 2026-06-05 per McKinley — "build the data but don't trade it yet"):**
  the dynamic universe is built + uploaded to R2 so it can be backtested/inspected, but the **live
  scan + portfolio report keep the legacy static tier** until the env var
  **`OVERFLOW_UNIVERSE_ACTIVE`** is truthy. `overflow_universe.load_overflow_universe/meta` return the
  fallback/{} when the gate is OFF (default), even if the parquet is present. Backtests/tooling pass
  `respect_active=False` to read the built universe regardless. **Promote = set
  `OVERFLOW_UNIVERSE_ACTIVE=1`** in `daily_screener.yml` + `portfolio_report.yml` env (and locally).
- **Validation tooling (added 2026-06-05):** `daily_scan.py --dry-run` runs the full scan and prints a
  signal summary with **zero side effects** (no Sheets/R2/email). Pair with `OVERFLOW_UNIVERSE_ACTIVE=1`
  to preview the dynamic universe before activating it live.
- **Backfill efficiency (added 2026-06-05):** `build_atr_seasonal_ranks.py` now seeds its price cache
  from `master_prices.parquet` (pushdown read) before falling back to yfinance — avoids re-downloading
  the ~3–4k overflow tickers a second time during the bootstrap.
- **ISOLATED staging path (added 2026-06-05 per McKinley — "data must not touch production"):** the
  trading gate isolates *trading*, not *data*; appending new names into production `master_prices`
  would make the daily `update_master_prices`/portfolio/earnings jobs ~3× heavier (R-F3/R-F4 in prod).
  So new-candidate prices go to a **separate `overflow_prices.parquet`** (R2 key `overflow_prices.parquet`),
  built by **`scripts/build_overflow_prices.py`** which *reads* master_prices (never writes it) and
  fetches only the genuinely-new names. `build_overflow_universe.py` screens `master_prices ∪
  overflow_prices`; `build_atr_seasonal_ranks.py` seeds from both. Production price/earnings caches and
  daily jobs are **completely untouched** until promote. **Promote-time step:** merge
  `overflow_prices` → `master_prices` (and have `load_master_prices_dict` cover it / the daily updater
  take over) at the same time the gate is flipped. FMP candidate pre-filter locked at **>300k
  shares/day → 2,176 candidates** (McKinley's choice; `--volume-more-than` overrides if widening later).
  Decisions chosen: candidate count 2,176 (300k filter). Earnings staging for new names deferred until
  `pages/backtester.py` wiring (the "backtest them" target) confirms what it needs.

---

## 1. Decisions (locked with user 2026-06-04)

| Decision | Choice |
|---|---|
| Broad candidate source | **FMP stock list / company-screener** (already pay for FMP; used for earnings) |
| Liquidity floor | **≥ $3,000,000** 63-trading-day average dollar volume (ADDV) |
| Volatility gate | **Min ATR% floor only** (e.g. 63d ATR% ≥ 1.5%) — ensure names move enough to signal |
| Refresh cadence | **Weekly rebuild** (Sunday, alongside the weekly pipeline) |
| Breadth vs runtime | **Uncapped** — admit every passing name; optimize the scan to absorb the count |

---

## 2. Current State (verified against the repo)

- `strategy_config.py:124-128` — `CSV_UNIVERSE = sorted(read_csv('sznl_ranks.csv')['ticker'].unique())`
  → ~1062 tickers. Fallback to `LIQUID_UNIVERSE` if the CSV is missing.
- `strategy_config.py:102` — `LIQUID_PLUS_COMMODITIES` = 190 hardcoded + 8 commodity ETFs = 198.
- Overflow tier is computed **in three places** as `sorted(set(CSV_UNIVERSE) - set(LIQUID_PLUS_COMMODITIES))`:
  - `daily_scan.py:106` (`build_effective_strategy_book`)
  - `daily_portfolio_report.py:68` (`OVERFLOW_TICKERS`)
  - `pages/strat_backtester.py:2741` and `scripts/build_indicator_cache.py:176`
    (these union with `sznl_map.keys()`)
- `OVERFLOW_ELIGIBLE_STRATEGIES` (5): Overbot Vol Spike, LT Trend ST OS, Oversold Low Volume,
  St OS Sznl, 52wh Breakout. Defined identically in `daily_scan.py:60`, `daily_portfolio_report.py:74`,
  `pages/strat_backtester.py:40`.
- `OVERFLOW_RISK_OVERRIDES = {"Oversold Low Volume": 25}` (vs liquid 35), scaled by
  `GLOBAL_RISK_MULTIPLIER` (1.5). `daily_scan.py:70`, `daily_portfolio_report.py:81`.
- Price cache: `data/master_prices.parquet` — **long format** `ticker, date, Open, High, Low,
  Close, Volume` (OHLC float32, Volume float64). Built from `CSV_UNIVERSE ∪ INDICES_AND_ETFS` by
  `scripts/build_master_prices.py`; incrementally updated by `scripts/update_master_prices.py`
  (reads its universe from the **existing parquet's tickers** — never adds new ones). Synced to R2
  key `master_prices.parquet`.
- `daily_scan.py:1901-1922` — for `scope=overflow|all`, overflow-tier prices are read from
  `master_prices.parquet` (`load_master_prices_dict`, lines 136-166), liquid from yfinance live.
- **Seasonal dependency (critical):** the overflow-eligible strategies **St OS Sznl** and
  **Overbot Vol Spike** gate on `atr_sznl_filters`, which read `atr_seasonal_ranks.parquet`
  (built by `build_atr_seasonal_ranks.py` over `CSV_UNIVERSE`). `daily_scan.py:2178` only applies
  the ATR-seasonal rank when `t_clean in atr_sznl_map`; a ticker absent from that parquet matches
  **nothing** on those filters → **zero signals**. So a new ticker must be in `atr_seasonal_ranks.parquet`
  to be eligible for the two seasonal strategies.
- **Earnings dependency:** OVS applies a ±10-TD earnings blackout via `earnings_calendar.parquet`.
  Tickers with **no** earnings rows pass through (NaN-as-True). For small caps this means trading
  *through* earnings unprotected unless they're in the calendar. `earnings_calendar.parquet` is
  built over `CSV_UNIVERSE` (`scripts/build_earnings_calendar.py`).
- Sizing is purely risk-bps / ATR-distance (`daily_scan.py:2325`). **No ADV-based notional cap exists today.**

**Implication:** the overflow universe is gated by *three* parquets, all currently keyed to
`CSV_UNIVERSE`: `master_prices`, `atr_seasonal_ranks`, `earnings_calendar`. Expanding breadth means
expanding all three, in R2, for the new names — exactly the "update all parquets" the user called out.

---

## 3. Target Architecture — Four Layers

Decouple "**data we maintain**" from "**names we trade this week**".

```
 Layer A  Candidate Symbol Master      symbol_master.parquet      (FMP, weekly)
            └─ all US common stock on NASDAQ/NYSE/AMEX, FMP-side pre-filtered
 Layer B  Maintained data caches       master_prices / atr_seasonal_ranks / earnings_calendar
            └─ expanded to cover  LIQUID ∪ Layer-A  (R2)
 Layer C  Active Overflow Universe      overflow_universe.parquet  (weekly screen of master_prices)
            └─ ADDV_63d ≥ $3MM  AND  ATR%_63d ≥ floor  AND  guards
 Layer D  Consumers                     daily_scan / portfolio_report / strat_backtester
            └─ overflow_tickers := Layer C   (replaces CSV_UNIVERSE − LIQUID)
```

Why two universes (A vs C):
- **Layer A** bounds how much price/earnings/seasonal data we maintain (FMP-side pre-filter on
  volume/price keeps it ~3–4k names, not the full ~6k+ listed set).
- **Layer C** applies the *precise, rolling* `$3MM / ATR%` screen computed from our own
  `master_prices` data — the single source of truth for what trades. Recomputing C is cheap
  (a groupby over a parquet we already hold); rebuilding A/B is the expensive weekly job.

---

## 4. Data Source — FMP

FMP conventions already in repo (`scripts/build_earnings_calendar.py`): `FMP_API_KEY` from env or
`.env`; `requests`; pace ~10/s; stable endpoints under `https://financialmodelingprep.com/stable/`.

**Layer A fetch** — FMP **company screener** (`/stable/company-screener`) with server-side filters
to keep the candidate set bounded:
- `exchange=NASDAQ,NYSE,AMEX`
- `isEtf=false`, `isFund=false`
- `isActivelyTrading=true`
- `priceMoreThan=3` (penny guard)
- `volumeMoreThan=300000` (shares/day; rough pre-filter — final $-volume screen is done in Layer C
  from our own data, this is just to cut the candidate count before we pull 25y of prices)
- `country=US`

Fallback if the screener field set differs by plan tier: `/stable/stock-list` (full symbol+exchange
list) then filter `type == 'stock'` and exchange locally. The build script will **try screener
first, fall back to stock-list**, and log which path it used.

Output of Layer A: `data/symbol_master.parquet` — `ticker, exchange, company_name, market_cap,
avg_volume, price, sector, as_of`. Upload to R2 key `symbol_master.parquet`.

Symbol normalization: uppercase, `.`→`-` (matches `load_master_prices_dict`’s `replace('.', '-')`);
drop symbols with `^`, `/`, ` `, or length > 6 that aren’t clean equity tickers; drop class/warrant/
unit suffixes that yfinance can’t resolve (configurable deny-regex, logged).

---

## 5. Screening Criteria (Layer C — the active universe)

Computed from `master_prices.parquet` (last ~70 rows per ticker is enough for a 63d window):

| Criterion | Rule | Rationale |
|---|---|---|
| Liquidity | `ADDV_63d = mean(Close * Volume, last 63 td) ≥ 3_000_000` | tradeable at our size |
| Volatility floor | `ATR%_63d = mean(ATR_14/Close, last 63 td)*100 ≥ ATR_PCT_FLOOR` (default **1.5**) | enough movement to signal |
| Price guard | `last_close ≥ 3.0` | avoid sub-$ microstructure |
| History | `n_bars ≥ 252` | 252d ranks / 52w high need a year |
| Freshness | `last_bar_date ≥ today − 5 td` | drop delisted/halted names |
| Exclusion | `ticker ∉ LIQUID_PLUS_COMMODITIES` | overflow is *additive* to liquid |

All thresholds are module-level constants in the build script (and mirrored as a small
`data/overflow_universe_config.json` written alongside the parquet, so the screen is auditable and
the same config drives backtests).

Output: `data/overflow_universe.parquet` — `ticker, addv_63d, atr_pct_63d, last_close, n_bars,
last_bar_date, as_of`. Upload to R2 key `overflow_universe.parquet`. Also write a plain
`overflow_universe.txt` (one ticker per line) for quick diffing / human review.

**Membership-churn handling (weekly):** the build emits a diff log (added / dropped vs prior parquet).
Names that drop out are *not* force-closed — open positions ride their existing exits (the scanner
just stops *opening* new ones). Document this explicitly; it matches current behavior (the scanner
never manages exits, `order_staging.py` + brackets do).

---

## 6. New Files

| File | Purpose |
|---|---|
| `scripts/build_symbol_master.py` | Layer A: FMP screener/stock-list → `symbol_master.parquet` → R2 |
| `scripts/build_overflow_universe.py` | Layer C: screen `master_prices` → `overflow_universe.parquet` (+ `.txt`, config json) → R2 |
| `overflow_universe.py` (project root) | Loader module: `load_overflow_universe()` → `list[str]`, with R2 pull + graceful fallback to `CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES`. Mirrors `cache_io` no-op semantics. |
| `.github/workflows/rebuild_overflow_universe.yml` | Sunday weekly: run symbol_master → expand master_prices/atr_seasonal/earnings for new names → build_overflow_universe → upload all to R2 |
| `tests/test_overflow_universe.py` | Unit tests: screen math (ADDV/ATR%), loader fallback, symbol normalization, schema |
| `docs/dynamic_overflow_universe_plan.md` | this document |

---

## 7. Modified Files (blast radius — keep the 3 definitions in lockstep)

1. **`strategy_config.py`**
   - Add `OVERFLOW_UNIVERSE = load_overflow_universe()` (import from new `overflow_universe.py`,
     wrapped in try/except → fallback `sorted(set(CSV_UNIVERSE) - set(LIQUID_PLUS_COMMODITIES))`).
   - Leave `CSV_UNIVERSE` as-is (it stays the seasonal-rank source for the *liquid* book and the
     historical sznl pipeline). The overflow tier no longer derives from it.

2. **`daily_scan.py`**
   - `build_effective_strategy_book`: `overflow_tickers = OVERFLOW_UNIVERSE` (import from
     strategy_config) instead of `set(CSV_UNIVERSE) - liquid_set`.
   - `load_master_prices_dict` already keys off `master_prices.parquet` — no change needed; the
     expanded parquet (Layer B) supplies the new names automatically.
   - Scan-runtime optimization (Section 9).

3. **`daily_portfolio_report.py`**
   - `OVERFLOW_TICKERS = OVERFLOW_UNIVERSE` (same import). Keep `OVERFLOW_ELIGIBLE` /
     `OVERFLOW_RISK_OVERRIDES` identical to daily_scan.

4. **`pages/strat_backtester.py`** and **`scripts/build_indicator_cache.py`**
   - Overflow swap uses `OVERFLOW_UNIVERSE` (with the existing `| sznl_map.keys()` union dropped or
     kept as a backtest-only superset — decide in review; default: use `OVERFLOW_UNIVERSE` directly
     so backtest membership == live membership).

5. **`scripts/build_master_prices.py`** and **`scripts/update_master_prices.py`**
   - Universe source becomes `LIQUID_PLUS_COMMODITIES ∪ symbol_master ∪ CSV_UNIVERSE ∪ INDICES_AND_ETFS`.
   - `update_master_prices.py` currently derives its universe from the *existing parquet* — add a
     step (or have the weekly workflow pass `--add-tickers @symbol_master`) so newly-screened names
     get backfilled the first time they appear. Backfill new tickers from a start date (e.g.
     2005-01-01) in their first run, then incremental thereafter.

6. **`build_atr_seasonal_ranks.py`** and **`scripts/build_earnings_calendar.py`**
   - Universe source extended to include `symbol_master` tickers so St OS Sznl / OVS seasonal gates
     and the OVS earnings blackout actually function on the new names. (These run weekly, not daily.)

7. **`CLAUDE.md`** — document the new universe layer, parquets, R2 keys, and the weekly workflow.

---

## 8. Parquet / R2 Changes (the "update all parquets" work)

| R2 key | Change | Producer |
|---|---|---|
| `symbol_master.parquet` | **NEW** | `build_symbol_master.py` (weekly) |
| `overflow_universe.parquet` | **NEW** | `build_overflow_universe.py` (weekly) |
| `master_prices.parquet` | **EXPAND** to ~3–4k tickers (one-time bootstrap, then incremental) | `build/update_master_prices.py` |
| `atr_seasonal_ranks.parquet` | **EXPAND** to new tickers | `build_atr_seasonal_ranks.py` (weekly) |
| `earnings_calendar.parquet` | **EXPAND** to new tickers | `build_earnings_calendar.py` (weekly) |

**One-time bootstrap sequence** (run manually / via `workflow_dispatch`, off-hours):
1. `build_symbol_master.py` → symbol_master.parquet (R2).
2. Expand `master_prices.parquet`: backfill all new symbols (chunked yfinance, ~2–4k names ×
   ~20y). Reuse `bootstrap_caches.yml` patterns; expect 30–60 min. Validate row counts + NaN rates.
3. `build_atr_seasonal_ranks.py` over the expanded universe.
4. `build_earnings_calendar.py` over the expanded universe (FMP, ~10/s → budget the call count).
5. `build_overflow_universe.py` → overflow_universe.parquet (R2).
6. Smoke-test `daily_scan.py --scope=overflow --dry-run` (Section 11) — **no Sheets write**.

---

## 9. Scan-Runtime Optimization (uncapped breadth)

Going from ~870 → potentially ~2–3k overflow names. The per-ticker indicator computation
(`calculate_indicators`) dominates. Levers, in priority order:

1. **Parallelize the overflow ticker loop** with a process pool (`concurrent.futures.ProcessPoolExecutor`,
   `os.cpu_count()` workers). GHA Linux runners have 2–4 cores; locally more. The loop is
   embarrassingly parallel per ticker (each computes its own `calc_df` + `check_signal`).
   Gather signals, then do the existing cross-strategy clamp / staging single-threaded.
2. **Compute only needed indicators** — pass the set of columns the active strategies actually read
   into `calculate_indicators` (it already accepts `custom_sma_lengths`, etc.). Skip pivots/weekly MA
   when no active filter needs them.
3. **Prefilter by data presence** — skip tickers with `< 252` bars before computing anything
   (the screen already guarantees this, but defend at scan time).
4. **Benchmark gate:** measure wall-clock at 1k / 2k / 3k names; ensure the GHA job stays well under
   the workflow timeout (set `timeout-minutes: 30` on the scan job; default GHA hard limit is 6h).

Keep optimization behind the same code path for liquid + overflow so behavior can't diverge; add a
`--workers N` flag (default = cpu_count, `1` reproduces today’s serial behavior for debugging).

---

## 10. Sizing Safety (recommended, optional)

At $3MM ADDV and ~25–40 bps risk on a $750k account (≈ $20–60k notional), liquidity is ample. Still,
add a **defensive ADV notional cap** for the long tail: `max_notional = min(computed_notional,
ADV_PARTICIPATION * addv_63d)` with `ADV_PARTICIPATION` default **0.02** (2% of a day's dollar
volume). Stamp the `addv_63d` onto staging rows so `order_staging.py` can enforce it too. This is a
guardrail, not a sizing change for normal names. Flag for review; default ON but conservative.

---

## 11. Verification Without Side Effects (no R2 write, no Sheets write, no order staging)

Add a `--dry-run` flag to `daily_scan.py` (if not present) that:
- runs the full scan,
- prints signal counts per strategy/tier and a sample,
- **skips** `save_staging_orders` / Sheets / R2 writes entirely.

Build scripts get a `--dry-run` that computes and prints the resulting universe + diff but **does not
write the parquet or upload to R2**. All new code is import-clean and unit-tested (Section 6) before
any live run. The weekly workflow is added **disabled** (or `workflow_dispatch`-only) until the
bootstrap is validated.

---

## 12. Rollout

1. Land code (this plan’s files) — **no workflow enabled, no parquet overwritten.**
2. Manual bootstrap (Section 8) into a *staging* R2 prefix or a copy, validated by dry-runs.
3. Compare overflow signal counts old-universe vs new-universe for a few historical dates via
   `strat_backtester` / `daily_portfolio_report --dry-run` to sanity-check PnL/turnover.
4. Promote: point loaders at the real R2 keys, enable the weekly workflow.
5. Update `CLAUDE.md`.

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Small-cap yfinance data gaps / bad ticks | `n_bars ≥ 252` + freshness guard + NaN-rate validation in bootstrap; price guard ≥ $3 |
| FMP screener field/plan differences | screener-then-stock-list fallback, logged; unit test the parser on a fixture |
| Seasonal/earnings parquets lag new names | weekly workflow rebuilds all three together; loader fallback keeps liquid unaffected |
| Scan timeout from bigger universe | process-pool parallelism + `timeout-minutes` + benchmark gate |
| Membership churn opening/closing names mid-position | scanner only *opens*; exits unchanged; weekly (not daily) cadence limits churn; diff log |
| order_staging.py (external) unaware of new ADV cap | stamp `addv_63d`; coordinate the external repo change separately (out of scope here) |
| Loader failure in prod | try/except → fallback to today's static `CSV_UNIVERSE − LIQUID`; never crash the scan |

---

## 14. Open Questions for Review

- Keep `ATR_PCT_FLOOR` at 1.5%, or calibrate from a backtest of signal yield vs win-rate?
- `symbol_master` size after FMP pre-filter — confirm it lands in the ~3–4k range (drives bootstrap cost).
- Should backtester/indicator-cache use live `OVERFLOW_UNIVERSE` or a historical point-in-time
  membership (to avoid lookahead in backtests)? Default here: live universe (simpler); note the
  survivorship caveat.
- ADV participation cap default (2%) and whether to enforce it scanner-side vs order_staging-side.
