# Private site (Cloudflare Pages)

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Private Site (Cloudflare Pages)

Static, client-side analytics site deployed nightly by `deploy_site.yml` to
Cloudflare Pages project `seasonals-mslade`, locked behind Cloudflare Access
(email OTP, allowlist = mckinleyslade@gmail.com). One-time setup doc:
`docs/private_site_setup.md`.

- **Frontend** lives in `site/` (committed): `index.html` (portfolio app),
  `signals.html`, `charts.html` (per-trade chart gallery), `risk.html`,
  `montecarlo.html` + `assets/` (vanilla JS + Plotly CDN, no build step, no
  framework). `site/_headers` sets no-store on `/data/*`. Nav order
  (2026-08-21): Portfolio, Seasonal, Execution, **Radar**, **Events**, Risk,
  Trade Log, then the rest, Monte Carlo last. The Events tab renders the
  event sleeve (see "Event Sleeve": status cards, open positions, realized
  history from the R2 journal) via `dist/data/event_sleeve.json`. The IDEAS TAB was REMOVED 2026-07-28 (page +
  ideas.js deleted); `ideas.json` is still built — the signals page's
  strategy-context block reads it.
- **Radar tab** (`radar.html` + `assets/radar.js`, 2026-08-18): the momentum
  radar's weekly plans, served LIVE from R2 by `functions/radar-recs.js` rather
  than baked into `dist/` (the radar runs on a weekend cadence independent of
  the 2x-daily deploy, same reasoning as `morning-orders.js`). Stage prefills
  the Execution ticket. Full section: "Momentum Radar — staging + trail".
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
- **Hedge panel (Exec tab, display-only)** (`assets/execution.js`, 2026-08-25):
  attributes each selected account's live stock positions to strategy-tagged
  working brackets, marks them, applies 63d or 252d SPY betas, nets counted
  equity-index futures, and shows MES/ES target arithmetic, SPY-shock scenarios,
  working entries, and the next 15 weekday roll-off dates. It reads the existing
  `/exec-book` snapshot, `assets/futures_specs.json`, and optional
  `data/betas.json`; it never sends, changes, schedules, or sizes an order and
  has no `data-mutation` controls. `scripts/build_betas.py` computes OLS slopes
  of adjusted daily close returns against SPY (minimum 20 paired observations),
  plus 63d residual volatility, from `master_prices.parquet`. The best-effort
  deploy step publishes `betas.json` in the immutable generated R2 bundle and
  `build_site.py` copies it into `dist/data/`; absent or missing-symbol betas
  deliberately degrade to 1.00 with an on-card assumption flag.
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
- **Shared Denali site risk tab** (`denali-seasonality`, 2026-09-18): the
  teammate site gets the MARKET REGIME and nothing about the book.
  `build_risk_json.redact_for_shared` deep-copies the same computed payload
  that writes `data/site_risk.json`, strips the `sizing_state` policy keys
  (`banded_strategies`, `throttled`, `threshold`, `throttle_on`,
  `gap_to_threshold`, `days_in_state`, `episodes`, `exposure`, `sleeve`) plus
  the "Book posture" nugget, and writes `data/site_risk_shared.json` — same
  run, same vintage, so the two sites can never disagree.
  `assert_shared_payload_clean` is the FAIL-CLOSED gate: it raises if any
  `STRATEGY_BOOK` name appears anywhere in the serialized payload or a banned
  key survives, and it runs three times (writer, R2 publish, shared builder +
  again on the bytes in `dist-shared`). Stable R2 key `shared/site_risk.json`,
  published best-effort by `site_r2_pipeline.py publish-shared` right after
  the private risk JSON is built; `deploy_shared_seasonals.yml` pulls it back
  to `data/site_risk_shared.json`. Staleness only DECLINES to copy — an `asof`
  older than 5 calendar days ships no `data/risk.json` and the tab renders its
  no-payload state rather than a month-old dial. The Seasonality page also
  carries a **Macro sub-tab** (`site/assets/macro_seasonal.js` ->
  `data/seasonality/macro.json` via `macro_site_data.export_macro_snapshot`,
  best effort off the `atr_seasonal_ranks.parquet` R2 pull) and moves
  Heatmaps/Correlations into a right-side "More" dropdown. Frontend:
  `shared_site/risk.html` + `site/assets/risk.js` shared-mode guards.
