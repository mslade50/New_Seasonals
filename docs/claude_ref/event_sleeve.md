# Event sleeve

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Event Sleeve (2026-08-06)

Four calendar-anchored index trades from `data/macro_events.csv`, prereg
frozen BEFORE go-live: scratch/ultracode_research/event_sleeve_prereg_2026-08-06.md
(evidence chain + kill rules there). Sizes are %NAV notional, NOT GRM-scaled,
no stops, no dial conditioning by design:
- **T1 FOMC_DRIFT** — long SPY 25% NAV, MOC 4 sessions before a scheduled
  FOMC decision -> MOO decision-day open. Non-midterm years only.
- **T2 FOMC_MIDTERM_SHORT** — short SPY 10%, same window, midterm years,
  ONLY when SPY 21d-return rank (252d, lag-1) < 50 — overbought tapes
  invert the edge and are excluded.
- **T3 SEP_POSTQUAD_SHORT** — short IWM 15%, Sep opex MOC -> Sep last
  session MOC; SKIPPED when IWM z10 (lag-1) < -1 (washouts bounce).
- **T4 DEC_POSTOPEX_LONG** — long IWM 25%, Dec opex MOC -> year-end MOC.
- **V2 NOVDEC_VOL** — long SVXY 5% (defined-risk short vol), first Nov
  session MOC -> year-end MOC, NON-midterm years only (both losing
  Nov-Dec years were midterms).
- **V4 POSTOPEX_VOL** — long SVXY 10%, every opex MOC -> +3 sessions MOC,
  EXCEPT September (crush inverts — T3's territory) and except while V2
  holds (no Nov/Dec doubling in non-midterm years).

Flow: `event_sleeve.py` runs in the daily_screener AM job (AM bookend only,
best-effort step) -> clears+rewrites the `Event` Sheets tab + state json
(`data/event_sleeve_state.json`, R2 round-trip) -> `event_moo.py` (OneDrive
trading_ibkr, Task Scheduler 'IBKR Event Sleeve Auction Orders' weekdays
9:05 AM ET, clientId 147, gated by `event_moo_enabled.flag`) validates the
due basket fail-closed (universe {SPY, IWM, SVXY}, max 3 rows, $350k notional
cap, OPG cutoff 9:25 / MOC cutoff 15:30) and places MKT+OPG / MKT+MOC
parent-only orders on the PRIMARY account (no exit legs — exits are
sleeve-scheduled). orderRef strategy field = trade id (execution-report
attribution). EXITS ARE STATE-DRIVEN, not calendar-driven: each entry
records `exit_on` + exit order type; any AM run at/past that date stages
the exit, so a failed run delays an exit by a session instead of dropping
it. Known bound (trend-sleeve convention): state marks positions open at
STAGING time — if a staged order never executed, clear it from the state
json.

**Sizing basis**: `nav_frac * ACCOUNT_VALUE` (the fixed $750k constant), NOT
live NLV — a 10% V4 position is ~$75k notional regardless of the account's
actual equity. Stated on every visibility surface.

**Broker MOC encoding (2026-08-21 incident, first live trade)**: rows keep
the MOO/OPG + MOC/MOC tab vocabulary, but the runners MUST place MOC as
NATIVE `orderType MOC` / `tif DAY`. `MKT` with `tif='MOC'` is IBKR-invalid
(error 10052) and TWS's order preset then coerces it into a WORKING MKT/DAY
while the API snapshot reports Cancelled — the V4 SVXY entry filled at the
9:30 OPEN while event_moo logged "broker rejected", and was hand-flattened
(-$96); the intended MOC entry was re-placed by hand the same afternoon.
Fixed in `event_moo.py` AND `pitch_moo.py` (same bug, OneDrive) plus a
verify-the-reject guard in both: a terminal reject status is checked
against `ib.openTrades()` for the orderRef and any survivor is cancelled —
a reject that leaves a live order is the worst failure mode. Guards:
`test_event_moo.py` / `test_pitch_moo.py` encoding tests (OneDrive).

**Visibility + journal (2026-08-21, after the first live trade — V4 SVXY —
surprised McKinley as "a random order")**: four surfaces now show the sleeve.
(1) The AM scan email's per-trade cards (status + rule + prereg evidence,
pre-existing) plus a NEW subject-line flag ("+ [EVENT] V4 SVXY") whenever a
card is staged today, so an auction order is never below the fold. (2) The
nightly execution report attributes untagged positions to "Event Sleeve (V4)"
via `event_sleeve_state.json` (event orders are parent-only — no exit legs, so
orderRef attribution can never see them; they previously rendered
"Discretionary") and reconciles state vs the live book (SHORTFALL-only alarm —
book overlap on SPY/IWM makes excess undecidable; entered-today skipped, the
MOC fill races the book push). (3) A site **Events tab** (`events.html` +
`assets/events.js`, nav after Radar) rendering `dist/data/event_sleeve.json`
(`build_site.build_event_sleeve`, best effort): cards, open positions marked
to the cache, realized history + per-trade realized-vs-prereg summary, the
FROZEN backtest evidence per trade (`BACKTEST_EVIDENCE` — verbatim prereg
transcriptions, NEVER recomputed; a bar-replay would silently diverge from
what was registered, above all V2/V4's synthetic -0.5x SVXY basis) and the
tested-and-not-shipped inventory (`REJECTED_STUDIES`, 7 entries — reviving
one needs a fresh prereg). NOTE: the production build runs in the data-free
assembler, so committed inputs under `data/` need a
`SOURCE_REFERENCE_REMAPS` entry in `stage_private_site_cloud_build.py` —
`macro_events.csv` is remapped to `reference/` (found 2026-08-21 when the
first deploy rendered one ERROR card; `macro_calendar.CSV_PATH` falls back).
(4) An APPEND-ONLY journal `data/event_sleeve_journal.jsonl` — R2-CANONICAL
(the writer is the GHA AM job whose checkout has no local copy; unlike the
pitch/posts journals it is NOT committed, a stale repo copy would clobber R2
on the sync-down-skip), one record per staged entry/exit, self-healing:
`backfill_entry_records` mints a missing entry record from the state on the
next run, which is also how the pre-journal V4 entry was seeded
(`scratch/seed_event_journal_20260821.py`). `realized_history` grades round
trips from master_prices bars — MOC legs at the Close, MOO legs at the Open —
modeled from bars, not fills, by design.

Aligned sites — change together:
- `event_sleeve.py` `EVENT_SLEEVE` dict (source of truth: tickers, sides,
  %NAV, rank/z thresholds) + `FOMC_ENTRY_TD_BEFORE`; journal helpers +
  `CARD_EXPLAINERS` (prereg rule/evidence strings) + `realized_history`
- `event_moo.py` (OneDrive) `EVENT_UNIVERSE` / `EVENT_TRADES` / caps
- `.github/workflows/daily_screener.yml` AM-gated step
- `daily_scan.py` event cards + `_staged_event` subject flag;
  `daily_execution_report.py` `event_symbol_map` / `reconcile_event`;
  `scripts/build_site.py` `build_event_sleeve`; `site/events.html` +
  `site/assets/events.js` + the `common.js` nav entry
- Guards: `tests/test_event_sleeve.py`, `tests/test_execution_report.py`,
  `tests/test_event_site.py` (repo), `test_event_moo.py` (OneDrive)
