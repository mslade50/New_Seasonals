# Daily Pitch

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Daily Pitch (2026-08-06)

Three novel trade ideas per trading morning, delivered as an email plus a
`Pitch` Sheets tab with a Y/N approval column. DELIBERATELY NOT SYSTEMATIC —
the strategy book is that layer, and an earlier "card library the code
activates" design was rejected. Every idea is a one-off judgement call,
invented from repo context and empirically falsified before delivery. Spec:
`daily_pitch_agent_spec_2026-08-06.html`; runbook: `docs/daily_pitch.md`.

Flow: 5:10 AM `scripts/run_daily_pitch.bat` -> grade yesterday
(`scripts/grade_pitch_journal.py`) -> assemble state
(`scripts/build_pitch_state.py` -> `data/pitch_state.json` +
`data/pitch_tape.json`) -> `claude -p "/daily-pitch"` (the skill runs stages
B-D: ideate 8-12 candidates across >= 4 novelty axes, fan out adversarial
verifiers that write real checks into `scratch/pitch_checks/<date>/`, compose
three) -> `daily_pitch.py` publishes -> McKinley types Y -> `pitch_moo.py`
(OneDrive, clientId 148) places at 9:05 (auction) and 9:32 (open-anchored
limits). NOTHING places without an exact `Y`, and the runner is
activation-flag gated (`pitch_moo_enabled.flag`).

Model/effort are PINNED in `run_daily_pitch.bat` (`--model opus --effort
xhigh`), not inherited from settings: an interactive `/model` change would
otherwise silently re-tier every following morning. Subagents inherit them,
so stage C's verifiers run at the composer's tier. Cut candidate count
before cutting tier or the falsification stage.

Hard requirements enforced in code, not prose (`pitch_grammar.py`):
- **1 to 3 ideas** (3 is the full slate), at most **one grade C**, every idea
  carries a **time stop**, entry/exit come only from the closed vocabularies
  (MOO | MOC | LIMIT(anchor, k ATR); time_td + optional target/stop ATR).
- **short slate** (2026-08-10): a COMPOSER publish under 3 ideas must carry a
  `short_slate` block accounting for the empty slots — 2 named kills per empty
  slot (`SHORT_SLATE_KILLS_PER_EMPTY_SLOT`, so the stand-down's 6 is the same
  line at 3 empties), plus the stand-down's own sweep floors (>= 8 candidates,
  >= 4 axes, >= 4 asset classes, >= 120-char reason, 1-3 near-misses each
  carrying the number it turned on). Nothing else relaxes: every shipped idea
  still needs evidence, `dev_script`, `survived` and the survey on disk. An
  ALL-directed publish is exempt (the count rule constrains the agent, not the
  human filter); one directed idea beside one composed one is not. Journaled
  as a `short_slate` record beside the idea records, and printed in the email
  under the cards. Added the day a run with 17 candidates and ONE survivor had
  no legal way to ship it: the publisher took 3 or a stand-down, both were
  false, the agent stopped to ask a headless prompt, and the morning delivered
  nothing. McKinley: "if there is one idea we want to see the idea."
- **fresh evidence** (summary, N, control, era note, and the path of the
  check script written that morning) plus a **`survived`** line naming one
  consideration that could have killed the idea and did not.
- **the survey has to be on disk** (2026-08-08, `validate_survey_evidence`):
  an ideas publish fails unless `scratch/pitch_checks/<asof>/` exists, holds
  `00_surface_map.md` and >= 1 `.py`, and every `evidence.script` /
  `evidence.dev_script` resolves INSIDE that folder (resolved-path
  containment, so `..` and Windows case cannot escape). A path into
  yesterday's folder fails as stale, which is the machine-checkable half of
  "computed fresh this morning". `dev_script` is required for every COMPOSED
  idea, making stage C round 3 structural. A DIRECTED-only publish is exempt
  from the folder/map check and from nothing else: the survey rule constrains
  the agent, not the human filter, but a directed idea still needs its own
  check written today. Closes the asymmetry where a stand-down had to prove
  it looked and three shipped ideas proved nothing, which made "three
  recall-generated ideas, no survey" the only unguarded path. `--checks-root`
  injects a fixture folder; `--validate-only` binds exactly like a real
  publish.
- **illegal-kill lint** (`lint_kill_reasons`, WARN ONLY): a kill reason that
  reads as sample size alone ("insufficient N", "not significant", "t below
  2") prints `KILL-LINT:` on every run mode and is tagged in the email's
  killed footer, on ideas publishes and stand-downs alike. Never blocks: the
  match is a heuristic over prose and a false positive must not cost a
  morning. Patterns are frozen constants (`KILL_SAMPLE_ONLY_PATTERNS`,
  `KILL_SUBSTANTIVE_MARKERS`); a substantive marker in the same reason
  suppresses the flag.
- **repetition control**: an idea whose structural fingerprint (legs + sides
  + entry type + horizon bucket) was pitched inside 10 td needs an explicit
  `changed_since`.
- grade/N coherence (A needs N>=50, B N>=15, C N<15).

**Stand-down (2026-08-07)**: a morning where nothing survives ships a NO
TRADES email, an EMPTY Pitch tab and a `stand_down` journal record instead of
three ideas. Added the day the first all-kill run (24 candidates, 50 check
scripts) had nowhere to put its verdict: the publisher offered three ideas or
a `PitchGrammarError`, so a real result was indistinguishable from a crashed
task and the journal recorded NOTHING (kills are journaled by the publisher,
which never ran). The path is deliberately MORE expensive than shipping so it
never becomes the way out of a hard morning — `validate_stand_down` enforces
>= 8 candidates over >= 4 distinct novelty axes, >= 4 distinct ASSET CLASSES,
>= 6 named kills with reasons, a >= 120-char verdict, a `checks_dir` holding
real `.py` files AND stage B1's `00_surface_map.md`, and 1-3 near-misses each
carrying **the number it turned on**.

**Coverage vs axis variety (2026-08-07, the same incident)**: that run hit
seven novelty axes and still ran EVERY calendar-anchored check on SPY, on an
August NFP in a midterm year, with the spec's own `event_fingerprint` example
being the midterm-August-NFP cross-asset table. Nothing was missing from the
data (the 217-name tape carries TLT/GLD/UUP/DX-Y.NYB/^TNX). The cause was
stage B's wording: "generate 8-12 candidates DRAWING ON at least four axes"
is a menu, one SPY idea ticked `event_fingerprint`, and no rule ever asked
which ASSETS had been examined. Its cross-asset work was all PRICE-STATE
anchored (bond floor, silver catch-up, GDX drawdown, natgas 52w low) and the
two search modes were never crossed. Fix is in SKILL.md stage B, rewritten
from "ideation" to **survey-then-select**: B1 writes
`scratch/pitch_checks/<date>/00_surface_map.md` enumerating every live
calendar event x 10 asset classes, every tape extreme by class, and every
live seasonal cell, giving each cell a verdict (dismissals must be reasoned,
never silently absent); B2 selects candidates from that map and must touch
>= 4 asset classes with >= 1 event-anchored and >= 1 price-state-anchored
candidate. Deliberately NOT solved by precomputing a grid into the state file
— McKinley's call, 2026-08-07: passing more data constrains the lens, the
lens itself had to widen.
`check_pitch_delivered` accepts 1-3 ideas OR a stand-down as delivery, and
checks the stand-down cases FIRST so the short-slate floor can never launder a
half-published run (ideas AND a stand-down) into a pass; the one legal mixture
stays a stand-down amended by ALL-directed ideas. Aligned sites:
`pitch_grammar.validate_stand_down` / `validate_short_slate` /
`short_slate_required` + the shared `_validate_sweep_scale` /
`_validate_named_kills` / `_validate_closest` floors,
`daily_pitch.render_stand_down` / `render_short_slate` / `publish_stand_down`
/ `stand_down_records` / `short_slate_records`, `pitch_journal.KINDS`,
`check_pitch_delivered --journal`, SKILL.md "When one or two survive" +
"When nothing survives". Guards: `tests/test_pitch_grammar.py`,
`tests/test_daily_pitch.py`, `tests/test_pitch_delivery_check.py`.

`render_scoreboard` formats every stat through `_fmt_r` / `_fmt_pct` because
the scoreboard is null until something is GRADED. Pitched-but-ungraded is the
normal early state and formatting those nulls raised TypeError on the whole
email — a second, independent reason 2026-08-10 could not have shipped.

Conventions that differ from the book on purpose:
- **ATR is Wilder-14** (spec section 4), matching
  `scripts/build_atr_downside_stats.wilder_atr`. The systematic book's ATR is
  a simple 14d mean of TR and sizes every scanner limit/stop. A pitch level is
  never a scanner level; do NOT converge them, and nothing in the book may
  import `pitch_grammar`.
- **Sanity bounds are ATR risk, never notional** (60 bps/idea, 150 bps/day,
  $15k in the runner's approved basket) per the book-wide no-notional-caps
  rule. The runner adds an ATR-percent-of-price band because a corrupted ATR
  inflates quantity while Risk_Amt still looks small.
- **Auto-placement follows the fill-price question**: only a CLOSE-anchored
  limit has a knowable fill price at 9:05, so only it gets a full bracket in
  the auction pass. MOO/MOC place with a time exit only; adding a price
  stop/target to them, or any futures leg, marks the row `Manual_Only`.
  OPEN-anchored limits route to the 9:32 pass, which fetches the true session
  open (order_staging's 1-min-bar method incl. its stale-session guard).
  Nothing ever fabricates a stop off a reference close.
- **The grader is pessimistic and grades DECLINED ideas too** (that is how
  the scoreboard measures the filter): a bar touching both stop and target
  books the stop, gapped stops fill at the open plus 13 bps, stops arm day 2.

Every idea and kill record is STAMPED with the `model`/`effort` that
produced it (from `PITCH_MODEL`/`PITCH_EFFORT`, exported by
run_daily_pitch.bat; a manual run with neither set records `unknown`
rather than guessing). `grade_pitch_journal` splits the scoreboard by
model, and the email footer prints the split once two models have
graded ideas. This is how the opus-vs-fable question gets settled on
realized R instead of on one adjudicated disagreement.

Journal (`pitch_journal.py` -> `data/pitch_journal.jsonl`, R2 mirror) is
APPEND-ONLY with four record kinds (idea / killed / approval / outcome);
`fold_ideas` merges them. Approvals have exactly one capture window: the next
morning's run reads the Approve cells BEFORE clearing the tab. A non-default
`--journal` path never touches R2, so tests and dev runs cannot pollute the
evidence trail. `data/pitch_negative_registry.md` is committed and GROWS: each
stage-C kill with a reusable lesson is appended the same morning.

### Approvals from fills (2026-09-23)

An idea staged by hand from the site Pitch tab counts as APPROVED once it
FILLS. `pitch_fills.py` reads `data/live_fills.parquet` (pulled fresh from R2
key `live_fills.parquet` on the default journal path only), keeps ENTRY fills
whose orderRef strategy matches `^Pitch-\d{4}-\d{2}-\d{2}-\d+$` (bracket exit
legs share the ref, so fill side must agree with the ref's action), and
appends one `approval` record per idea: `approve: "Y"`, `source: "fills"`,
plus a `fills` block (first fill session, accounts, total qty, VWAP overall
and per symbol). It runs inside `grade_pitch_journal.main` before the
journal is folded, so the same morning's scoreboard sees it; any failure
prints a WARNING and grading continues (`--no-fills-approvals` skips it). A
staged limit that never filled does not count, by decision. An idea is only
eligible once its tab answer has been captured (any tab approval record,
blank included) or it is 2+ sessions old, so a pitch_moo idea keeps its tab
`Y` as the record of provenance instead of being shadowed by a fills record;
a fill therefore lands on the scoreboard one morning later than the fill.
Fills tagged with an idea_id not in the journal are skipped with one loud
line. `fold_ideas` never lets a BLANK approval override an earlier non-blank
one (the tab capture journals `""` for every untouched idea), but a later
explicit answer such as `N` still wins. Guard:
`tests/test_pitch_fills_approval.py`.

Aligned sites — change together:
- `pitch_grammar.py` (the contract: vocabularies, validation, sizing, order
  derivation, placement routing) + `pitch_journal.py`
- `pitch_lab.py` (shared check library, 2026-08-08: price/event loading,
  lag-1 forward returns, declustering, controls, kill battery,
  `sign_test` — the N<15 statistic, t-stats never required; its p=0.5 path
  moved to exact Fraction arithmetic 2026-08-09 because the float form raised
  OverflowError above a few hundred n, which fires whenever a check script
  measures a CONTROL cell next to the small conditional one, values for every
  n that already worked unchanged — horizon scan,
  loser paths, watchlist I/O; consolidates the 08-07 ad hoc helpers; the
  book must never import it) + `data/pitch_watchlist.json` (parked
  near-misses with the number that turns them on; folded into the state,
  verdict owed in every B1 surface map, pruned after publish)
- `daily_pitch.py` (publisher: TAB_COLUMNS is the runner's schema)
- `.claude/skills/daily-pitch/SKILL.md` (stages B-D; the .gitignore carries a
  `!.claude/skills/` negation so this file is committed)
- `scripts/build_pitch_state.py`, `scripts/build_pitch_research_index.py`,
  `scripts/grade_pitch_journal.py`, `scripts/check_pitch_delivered.py`
- `pitch_moo.py` + `run_pitch_moo.bat` + `register_pitch_moo_task.ps1`
  (OneDrive trading_ibkr)
- Guards: `tests/test_pitch_grammar.py`, `tests/test_daily_pitch.py`,
  `tests/test_pitch_grader.py`, `tests/test_pitch_lab.py`,
  `test_pitch_moo.py` (OneDrive); `tests/conftest.py` holds the shared
  `survey` / `checks_root` fixtures both pitch modules build payloads from,
  so neither reads the repo's real scratch state

## Site Pitch tab (2026-09-23)

Stage a pitched leg into the Execution ticket after the fact, modeled on the
Radar tab. Chain: `daily_pitch.py` (after the journal reconciles, on an ideas
publish AND a stand-down) -> `data/pitch_today.json` -> R2 `pitch_today.json`
-> `functions/pitch-today.js` (`/pitch-today`, GET-only, no-store) ->
`site/pitch.html` + `site/assets/pitch.js` -> `execution.html?stage=pitch&...`
-> `execution.js` `pitchStage` / `applyPitchPrefill`. Served live from R2
because the pitch publishes ~05:30-05:40 ET, after the morning site deploy.
The upload is best effort and uses the delivery-receipt gate: a custom
`--journal` or explicit `--delivery-receipt` run writes
`<journal stem>.pitch_today.json` beside that journal and never touches R2.

- Whitelist: `SITE_IDEA_FIELDS`, evidence `summary`/`n`, and
  `SITE_ORDER_FIELDS` = `TAB_COLUMNS` minus `Approve`, plus `Multiplier`.
- PRIMARY only, strategy tag `Pitch-<idea_id>` (the string `pitch_moo.py`
  stamps), qty = `Quantity` verbatim (fixed `ACCOUNT_VALUE` basis).
- **Rule (McKinley, 2026-09-23): stage EXACTLY what the pitch says, or block.**
  No substitutes, no derived levels. The earlier MKT-after-the-auction
  substitute and the "off reference close" MOO/MOC stop/target were removed.
- Entry mapping: LIMIT/CLOSE -> `Order_Type`/`TIF` verbatim, Limit/Stop/Target
  prices verbatim, expiry = `Entry_Expire_Date` when GTD; LIMIT/OPEN -> user
  types the session open, priced exactly like `pitch_moo.price_open_row`
  (Python `round` ties-to-even mirrored by `pyRound2`), stageable only after
  09:30; MOO -> MOO only before 09:25 ET (`pitch_moo.OPG_CUTOFF`), then
  blocked; MOC -> MOC only before 15:30 ET (`pitch_moo.MOC_CUTOFF`), then
  blocked. A MOO/MOC entry's `entry` is `Ref_Close` as the ticket's risk
  reference only.
- Execution conventions ride the `entry_bracket` payload (executor contract in
  `docs/site_execution_schema.md`): `stop_arm: "next_session"` whenever there
  is a stop, `time_stop_at: "open"` when `Time_Exit_Order` is MOO, else
  `"close"`. Both appear in the ticket readout and confirm text. They are
  added only after a successful pitch prefill and only while the ticket holds
  the prefilled symbol (`pitchTicket` / `pitchExecFields`); Radar and every
  other ticket path send byte-identical payloads without them.
- Blockers (`stageBlockers`): stale date (`Execute_On` != today ET), time exit
  today/past, any `Manual_Only` row ("manual per pitch": FUT legs, MOO/MOC
  with a price stop/target), non-empty `Proxy_Ticker`, non-STK, no qty,
  stand-down, Order_Type/TIF not the vocabulary the row type implies, no
  pitch_moo pass, open-anchored with no open, and the clock gates below.
  Multi-leg ideas warn "stage every leg".
- **Double-placement gate**: pitch_moo's dedupe cannot see site-staged orders
  (shorts tag SELL vs SELL_SHORT, clientIds differ), so a leg is stageable
  only AFTER its pitch_moo pass has run: `Place_Pass` "auction" from 09:05 ET,
  "open" from 09:32 ET. An auction MOO is therefore stageable only 09:05-09:25.
  The card says: stage here OR approve `Y` in the Sheet, never both.
- **The pass wait applies only while `PITCH_MOO_ARMED` (pitch.js) is true.**
  It is `false` today: the pitch_moo tasks are unregistered and there is no
  `pitch_moo_enabled.flag`, so nothing can double and the wait would only
  delay staging. Unarmed, the card adds "pitch_moo runner is off — the Sheet Y
  places nothing" and every other gate stays (MOO < 09:25, MOC < 15:30,
  open-anchored from 09:30, stale date, Manual_Only, proxy, ...). Set it true
  in the same change that arms the runner. The constant lives only in
  pitch.js; the stage link carries `armed=0|1`, and execution.js skips the
  pass wait only on an explicit `armed=0` (absent or any other value enforces
  it, fail closed).
- Every gate re-runs at prefill (`pitchPrefillRefusal` in execution.js):
  `refdate` != today ET, pass not yet run (armed links only), MOO >= 09:25,
  MOC >= 15:30, open-anchored < 09:30. A refusal fills nothing and says why in
  the ticket message. The link carries `kind` (LIMIT_CLOSE / LIMIT_OPEN / MOO
  / MOC), `pass`, `armed` and `tsat`; a kind/type mismatch or missing `tsat` is
  not a pitch link. pitch.js re-renders every 60 s and on `visibilitychange`.
- `pitch_fills.ENTRY_SIDES` includes `("SELL_SHORT", "SLD")`: pitch_moo stamps
  SELL_SHORT in a short entry's orderRef.
- Guards: `tests/js/test_pitch_tab.js` (run by `tests/test_pitch_transport.py`),
  `tests/test_pitch_transport.py`, `tests/test_pitch_fills_approval.py`.
