# Momentum Radar: staging and trail

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Momentum Radar — staging + trail (2026-08-18)

Fourth agent product, and the only one that places orders. The radar itself is
NOT in this repo: it lives in `mslade50/radar-briefings` (a weekend GHA screen
plus an 11:00 UTC Claude cloud routine, `momentum-radar-weekly` /
`trig_0189pHRnCARusEDPN4PwGijH`). Its `scripts/book_update.py` mints plans and
steps trails into `data/recs/<sunday>.json`. **That engine is the source of
truth for every number.** Its own rendering rule ("R-A verbatim") applies on
this side too: nothing here recomputes, rounds, or infers a price, share count,
or date. A field this repo cannot express is a REFUSAL, never a derivation.

### The chain
```
radar clone -> scripts/upload_radar_recs.py -> R2 radar_recs.json
            -> functions/radar-recs.js (/radar-recs) -> site/radar.html
            -> Stage -> execution.html?stage=radar&... -> entry_bracket
            -> broker -> agent -> execute_order.py -> IBKR
```
Weekly upkeep: `scripts/run_radar_sync.bat` (Mondays 8:50 AM ET).

**Transport is LOCAL, not CI.** radar-briefings is private and this repo's
Actions have no cross-repo token, while the trading box has both the clone and
the R2 creds. Mirror image of `scripts/export_radar_pack.py`, which feeds the
radar in the other direction. Consequence: the tab goes stale if the box is
off, so the payload carries `date`/`age_days`, the tab banners a vintage over
10 days, and staging is blocked past that.

The payload copies a FIELD WHITELIST (`REC_FIELDS` / `POSITION_FIELDS`) so a new
engine field cannot reach the browser unreviewed.

### PRIMARY ONLY, and the sizing is NOT your NLV
The engine sizes against its own **$250,000 book** (`account_value` in the recs),
which is unrelated to either live account (primary NLV ~$632k, PA ~$66k as of
2026-08-18). A plan's stated 36 bps is 36 bps OF THAT $250k — about 14 bps of
live primary. The site stages the engine's share count unchanged, which is
exactly what the radar email has always shown; it does NOT rescale to the live
account, and it must not start doing so silently. The Radar tab states the
basis on every render.

Radar plans are a **primary-account sleeve**: `radar.js` pins `acct=primary`
into the stage link and `applyRadarPrefill` switches the tab to it, because one
plan's notional (BNY: $28.5k) nearly fills PA's entire $30k live cap (futures
exempt since 2026-09-21 via `LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS=pa`; the
stock cap the Radar stages against is unchanged).
`radar_trail_sync.py` defaults to the same account.

### Execution-bridge features this required
- **`STP_LMT` entry type** — the breakout plans are BUY_STOP_LIMIT. `entry` is
  the trigger, `entry_cap` the limit. The CAP is the worst acceptable fill and
  every gate reads it. See "Scale-out" + `docs/site_execution_schema.md`.
- **orderRef strategy tag** — `entry_bracket` gained `strategy` / `ref_date`,
  stamping `SYMBOL|ACTION|Strategy|Date` on the parent AND every child. Site
  orders were untagged before this and read as "Discretionary" everywhere.
  It is what lets the trail job find its own stops.
- **`scaleout: {frac, target}`** — two independent brackets (near takes T1,
  far runs behind the stop and time exit). Never one parent with a partial-size
  child; see the schema doc for the broker bug that forbids it.

### The trail is NOT ours to compute
`step_trails()` in the radar engine already does the chandelier (peak weekly
close - 2.5 ATR on the post-T1 remainder), the breakeven at entry + 2R, and the
never-lower rule, folding the result into `open_positions[].current_stop`
before the recs are written. `radar_trail_sync.py` (OneDrive/trading_ibkr) is
transport and reconciliation only: read that number, move the matching live STP
legs to it.

It routes through the broker (`/command` -> agent -> `_do_modify`) rather than
connecting to IBKR directly, because `_do_modify` already re-places an order AS
its owning clientId — IBKR only lets the placing client modify — and every
change then lands in the broker log and the site Activity panel.

Refusals, each independent of the radar's own logic:
- never lowers a stop, compared against the LIVE aux, not the engine's belief
- only touches legs whose orderRef strategy matches (`Momentum_Radar`). The
  primary account also runs the systematic book across ~1060 names; a
  symbol-only match would eventually move an OLV or OVS stop. Verified against
  the real book, which carries untagged STPs and a tagged OLV pair.
- scale-out aware: a near/far pair is trailed together and their quantities
  must SUM to `shares_remaining`; two stops sharing a tranche are refused as an
  unexplained duplicate
- an unfilled `STP LMT` entry parent is never mistaken for the protective STP
- dry-run by default; `--apply` plus `radar_trail_enabled.flag` to transmit

**`radar_trail_sync.py` does NOT git-pull.** It reads whatever the clone holds,
so it must run AFTER `upload_radar_recs.py` (which pulls). That ordering is the
entire reason `run_radar_sync.bat` exists — run standalone on a week-old clone
it would apply last week's stops, and because it never lowers, that
under-raises silently instead of failing.

### Known-open
- A native `STP LMT` does NOT die on a runaway gap, so the engine's `gap_rule`
  ("dead for the day if open > cap") is NOT enforced. The tab prints the rule
  and the caveat; honoring it literally needs an open-check at staging time
  (the OVS 2-path / `T1_Open_Filters` precedent).
- The engine tracks T1 fills from BARS, not from actual fills, so its
  `scaled` flag can diverge from the live book.

Aligned sites — change together:
- `scripts/upload_radar_recs.py` (whitelist + R2 key) <-> `functions/radar-recs.js`
- `site/assets/radar.js` `RADAR_STRATEGY` <-> `radar_trail_sync.py` RADAR_STRATEGY
  (a mismatch is a stop that never moves; pinned by a test)
- `radar.js` stage-link params <-> `execution.js` `radarStage` parser
- Guards: `tests/js/test_radar_tab.js` (blockers + verbatim round-trip),
  `tests/test_radar_transport.py`, `test_radar_trail_sync.py` +
  `test_stop_limit_entry.py` (OneDrive)
