# Fragility dial, risk report, downside tables

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Risk Dials / Fragility System (rewritten 2026-07-16)

The old "Risk Dashboard V2 Phases 1-2 / Layers 0-4 / Executive Summary"
description no longer matches the code. Current state:

**pages/risk_dashboard_v2.py** computes 8 fragility signals (Distribution
Dominance [+Elevated display tier], VIX Range Compression, Defensive
Leadership, Pre-FOMC Rally, Low Absorption Ratio, Seasonal Rank Divergence,
Dispersion, and — since 2026-08-05 — Equity P/C Complacency) and a 0-100
composite dial at 3 horizons (5d/21d/63d), weighted by diff_mean edges from
`data/signal_horizon_stats.json` (reproducible via
`scripts/build_signal_horizon_stats.py`; the JSON's "(Elevated)" entry is
reference-only, NOT consumed by the composite). The page displays ONE dial
(63d + 10d MA + throttle state); 5d/21d are context chips (5d failed every
sizing test; 21d ~90% state-agreement with 63d — no "confirm" semantics
anywhere). `daily_risk_report.py` and `weekly_market_rundown.py` import the
page's compute functions (import surface: deleting page functions can crash
the GHA email that appends the sizing parquet — check both before removing
anything).

**Equity P/C Complacency is a 5d-HORIZON-ONLY contributor** (like Pre-FOMC
in spirit, but a persistent signal, so it sits in the STATIC denominator):
10d-MA CBOE equity put/call trailing-252d pctile < 10
(`compute_pc_complacency_signal`, reads `data/cboe_putcall.parquet` via
`pc_fear.py`). Its stats JSON entry carries NO 21d/63d horizons and
`scripts/build_signal_horizon_stats.py HORIZON_RESTRICT` keeps regens that
way, so the sizing 63d column and the exposure-leg 21d input are unchanged —
only the display-only 5d column gets a new definitional vintage from
2026-08-05 (63d dial candidacy was REJECTED: day-level edge is overlap
inflation, episode t wrong-signed — scratch/putcall_dial_study.py). It is
EXCLUDED from the pre-registered simple-dial shadow (daily_risk_report
filters to `fragility_simple.SIMPLE_SIGNALS`, the registered 7) and from the
trade-console evidence fingerprint (build_risk_json filters to ABBR).
Present in all three compute_all_signals copies (page, daily_risk_report,
weekly_market_rundown) + build_atr_downside_stats. Guard:
`tests/test_pc_dial_signal.py` (21d/63d byte-invariance, 5d-only stats,
shadow pinning). Site note: the Pre-FOMC Rally card renders WITHOUT a
per-signal chart since 2026-08-05 (risk.js NO_CHART_SIGNALS — calendar
signal, chart earned its space poorly); its windows still shade the shared
overlay chart.

**NYSE Net Highs is a DISPLAY-ONLY warning layered on the main dial**
(`nyse_risk.py`, 2026-09-17; not in `ACTIVE_RISK_SIGNALS`, so no composite
weight of its own). It fires when NYSE net new highs are negative while SPY
sits within 3% of its 252-session closing high (severity 1.0 under 2%, 0.6
from 2% through 3%), borrows Low Absorption Ratio's 63d weight, fades over 63
sessions, and writes `main_score` into `data/rd2_fragility.parquet` as
`max(base, expanded)`. **Since 2026-09-18 the trigger series is a 5-period EMA
of `nyse_net`, not the raw daily print, for BOTH arming and the recovery
reset** (`smooth_nyse_net`, `ewm(span=5, adjust=False)`; a missing breadth
reading blanks the EMA for its whole trailing window, which is also the
warm-up). A single non-negative print no longer wipes the state and both
smoothing queues, which is the Aug-2026 flicker the change was made for.
McKinley's APPETITE call, evidence flat: the study
(`scratch/nyse_smoothing_study/`) cut 78 episodes to 34 and raised
P(5% drawdown in 63d) from 52% to 67%, but its own conclusion was not to
change the trigger, and against a placebo that just waits the same 4-session
lag the gain misses 1.8 sigma. The basis string moved v1 to
`nyse-reset-floor-v2-ema5`; the parquet is mixed-vintage on purpose and
`append_main_scores` freezes rows saved before `BASIS_V2_START` so the AM
`--refresh-last` correction can never rescore a v1 row. Guard:
`tests/test_nyse_risk.py`.

**Breadth collection is AUTOMATED twice a trading day since 2026-09-21**
(`scripts/collect_market_breadth.py`). It reads the public WSJ Markets Diary
JSON with a browser User-Agent (`marketsDiaryType=diaries`, Latest Close,
NYSE + NASDAQ) and imports through `scripts/maintain_market_breadth.py`, so
all existing validation and the digest-keyed revision logic apply unchanged.
Do NOT switch it to `marketsDiaryType=overview`: that set disagrees with the
diary on NASDAQ (2026-09-18: 72/244 vs 81/246) and timestamps its publication
rather than its session. `breadth_pm` runs in postclose AFTER
`master_prices_pm` and BEFORE `risk_pm`, which is the whole point: the evening
dial now carries the same day's NYSE floor instead of scoring unfloored until
the AM correction. `breadth_am` runs in premarket after `cboe_am` and before
`risk_am`, purely for overnight AMENDMENTS. Exit 2 (the diary has not
published the expected session) is declared non-blocking in the supervisor
catalog: the receipt records `health_status=degraded`, the pipeline continues
and the dial keeps its documented unfloored fallback. Both
`market_breadth.parquet` and `market_breadth.sqlite` are canonical R2 objects
so the pinned runtime bootstraps the store instead of starting empty. Manual
in-app-browser capture (`--observation`) is now the FALLBACK. Guards:
`tests/test_collect_market_breadth.py`, `tests/test_market_breadth_store.py`;
runbook: `docs/nyse_risk_dial_2026-09-17.md` "Breadth collection".

### The fragility-portfolio contract (B6, 2026-07-16)

- **The sizing statistic** is exactly: 10d MA of the 63d column of
  `data/rd2_fragility.parquet`, threshold 50. Nothing else sizes orders.
- **Vintage rule**: the parquet is APPEND-ONLY point-in-time since
  2026-07-02; earlier rows are a recompute vintage (drifted up to ~7 pts).
  Any backtest joining the dial must state which vintage it used.
  `rd2_fragility_ts.parquet` is a raw-basis full recompute for research
  only — NEVER a sizing fallback (daily_scan's fallback removed 2026-07-16).
- **Staleness convention**: daily_scan fails OPEN to 1.0x sizing on readings
  older than 3 trading days (`daily_scan.FRAG_STALE_TD`), and dial_filters
  entry gates fail CLOSED. EXCEPTION: exposure_leg
  (`exposure_leg.DIAL_STALE_TD`) SKIPS the whole leg on a stale dial —
  no targets, no orders, holdings unchanged — deliberate per its docstring
  (acting on a regime that may no longer hold is worse than skipping);
  doc-vs-code drift here corrected 2026-08-12. Additionally: an EXISTING but
  unreadable fragility parquet now fails the risk report loudly instead of
  falling back to a full frozen-history rewrite (2026-08-12) — only a
  genuinely missing cache bootstraps.
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
  3b3): FAMILY4 + 3x Bear Fade + Monthly Weak Close at [[50,999,0.25]]
  (OLV carried [[65,999,0.5]] for one session 2026-08-24 -> 25, retired). Guard:
  `tests/test_frag_risk_bands.py` (includes the site serializer assertion).
- `exposure_leg.py` (25% NAV VOO/QQQ overlay in the AM scan email): kill
  rules raw-21d>50 and ma10-63d>50. The 1.25x boost was REMOVED 2026-07-16
  (mirrored the unanimously-killed per-trade boost). The raw-21d kill has a
  pre-registered replay pending — do not touch it before the replay runs.
- `dial_filters` entry gates, the daily risk email, the site risk tab
  (`sizing_state` reads the PIT parquet, never the deploy recompute), the
  portfolio page fragility adjuster (`fragility.json`).

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

- **ML meta-labeling layer: DELETED 2026-08-07, and do not rebuild it on
  P(win).** The `ml/` package scored every staged signal with a calibrated
  win probability and mapped it to advisory SKIP/TRIM/FULL sizing. Two full
  walk-forward evaluations both said NO SHIP for the SAME reason, which is a
  structural fact about this book rather than a modelling failure: the model
  genuinely predicts wins (realized win rate climbs 44.7% -> 70.3% across
  calibrated p-deciles, Brier beats base rate) but mean R is FLAT across those
  deciles, because low win probability here comes with bigger winners. The
  bucket it wanted to SKIP averaged +0.60R at a 51% win rate; 8-ATR-target
  breakouts are the extreme case. Adding 8 features orthogonal to the entry
  rules (put/call, NAAIM, analyst-grade momentum, earnings distance) changed
  nothing: uplift -0.015R, bootstrap CI containing zero, 7/15 positive years.
  **Win rate and expectancy are decoupled in this book by design**, so any
  future proposal to improve results by being more selective has to clear
  that bar first. Ran advisory-only from 2026-06-10 with nothing consuming
  the output. The one line NOT closed by this was risk-targeted rather than
  expectancy-targeted (P(MAE >= 1R), AUC 0.606, calibrated). Code, tests,
  workflow, plan doc and the R2 model artifact are all gone; recover from git
  history at 45ee31c~ if ever needed.
- Book-wide throttle/taper, dial-conditioned caps: dead (PIT t=-0.23; see
  Daily Risk Caps section). OVS tilt: dead (PIT gate 2026-07-03).
- Put hedges, VXX proxy, 21d "fast confirm" shadow, trend-sleeve gate,
  >1.0x hi-frag boosts, sub-50 sizing ramps: all rejected — reasoning
  preserved in scratch/ultracode_research/RISK_DIALS_2026-07-16.md section 4.
- 3x Bear Fade band re-exam TRIGGER: revisit at 2 new hi-frag episodes (its
  own hi bucket is flat, t=-0.05, N_hi=17; band kept by family analogy).
  Companion to the existing "re-examine FAMILY4 at +20 trades (~2029)".
- OLV band: shipped 2026-08-24 as an explicit McKinley APPETITE decision
  (`[[65, 999, 0.5]]`, lifted in P/C fear) and RETIRED 2026-08-25 — he hedges
  the Aug-2026 episode manually instead. Not on evidence either way: 21 OLV
  trades at dial >= 70 ran +8R with zero stop-outs, OLV's worst drawdowns
  came at dial ~20 and were idiosyncratic. The 2026-07-16 prereg candidacy
  remains parked. See "Fragility Risk Bands".
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


### Daily Risk Report — Forward Returns Table
Uses `compute_similar_reading_returns()` from `risk_dashboard_v2.py`. Forward returns at similar fragility readings include:
- Mean and Median conditional returns
- **Mean Z / Median Z** — z-scores vs unconditional sample (mean via z-test, median via bootstrap SE with 1000 resamples)
- % Negative and Baseline (unconditional mean)
- Mean column color follows Mean Z thresholds (green >= 0, yellow > -1, red <= -1)
