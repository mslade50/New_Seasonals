# Daily Posts and the Market Context brief

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Daily Posts (the pseudonymous X account, 2026-08-10)

Third agent product: 3-6 daily X post drafts for the anon account, written
to a review queue. NOTHING auto-posts - McKinley posts by hand and marks
the queue, and the next run ingests the marks. Strategy evidence:
`research/fintwit_pillars_2026-08-10.md`; operating manual (voice,
disclosure, opsec, cadence, kill criteria): `content/playbook.md`; persona:
`content/persona.md`; pre-sanitized launch material: `content/backlog/`.

Hard rules enforced in code (`posts_grammar.py`, the product's contract):
- **Ideas specific, book vague**: standalone idea posts carry ticker +
  entry structure + horizon (closed vocabularies, pitch-style: MOO | MOC |
  LIMIT(anchor, k ATR), time_td, frozen Wilder-14 ATR + ref_close);
  `journal` (book-texture) posts may never name a ticker. Overflow-tier
  names are banned in EVERY post type - the one place an audience could
  move our fills, and the cleanest regulatory line.
- **No dollar sizes anywhere** (R and percentages only; prices are fine).
  Identity strings are linted (hard list blocks, soft list warns - Scott
  Bessent and McKinley tariffs are real collisions).
- Idea posts need evidence (summary + N), a digit in the text, and a
  repetition check (structural fingerprint inside 10 td needs
  `changed_since` - posted drafts only; the audience never saw the rest).

Flow: `scripts/build_posts_state.py` (ingests yesterday's queue marks into
the journal FIRST - the Pitch-tab one-window convention - then assembles
`data/posts_state.json`: tape through tonight's close via build_pitch_state
builders, calendar, risk, recent context-journal nuggets, pitch activity,
scoreboard, repetition state) -> `/daily-posts` skill drafts
`content/queue/<date>.md` (human, `Posted:` marks) + `.json`
(authoritative) -> `scripts/lint_posts.py --journal-drafts` (hard findings
block; journals every draft so unposted ideas are graded too) ->
`scripts/posts_scoreboard.py` (replay IMPORTS grade_pitch_journal.replay_leg
so post grading can never drift from pitch grading; posted vs unposted
split measures the filter; `--format-post` emits the weekly accountability
numbers). Journal: `posts_journal.py` -> `data/posts_journal.jsonl`
(append-only, R2 mirror, kinds draft/posted/outcome). Queue is gitignored
(publishing product); journal + scoreboard committed (evidence trail).

Boundaries: the book never imports posts machinery; posts machinery reads
the book's data one-way (context-engine precedent). Nothing here touches
the X API by design - the research was explicit that a new pseudonymous
account + API posting is the riskiest fingerprint, and the human filter is
part of the product. Guard: `tests/test_daily_posts.py`.

## Market Context brief (2026-08-09)

Second agent product in this repo, and the one that is NOT a pitch. One Slack
post per evening (Sun-Thu 18:30 ET, same webhook and channel as the denali
report card): 4 to 8 statistical nuggets about the session tomorrow and the
one that just closed. Spec of record:
`market_context_skill_design_2026-08-09.md`.

What keeps it separate from everything else here:

- **No trades and no advice.** Advisory verbs are lint-blocked at publish
  (`send_context_slack.py`, hard tier). The Daily Pitch is where an idea with
  legs belongs.
- **Position-blind.** It never reads the denali book, D1, HedgeFacts or
  `data.json`. That is what lets it say things the report card cannot.
- **No web.** Every claim is computed from local history, which is why the
  unattended run gets a scoped allowlist
  (`scripts/context_headless_settings.json`) rather than the pitch's
  `bypassPermissions`.
- **Module boundary**: nothing in the systematic book imports these modules,
  and they never write `data/pitch_*`. The dependency runs one way — the
  context engine reuses `build_pitch_state.build_calendar` / `_metrics_for`
  and `pitch_lab`, never the reverse.

The convention the whole product hangs on: **every cell anchors on today's
analogue, so h=1 is tomorrow.** An event on the next session anchors on the
session BEFORE it, which makes h=1 the event session's own move; a price
state anchors on the session it printed. Forward returns are lag=0
close-to-close, deliberately unlike the pitch's lag=1, because this is
context rather than an entry.

Three dates per run and only one names files: the RUN date (today) names the
cell-map folder, the brief and the delivery check; `meta.asof_session` is the
tape just read; `meta.next_session` is what the title previews. A Sunday run
has all three different.

Freshness is a hard gate. If the freshest core bar is older than the asof
session the ENTIRE price lane is suppressed and the brief ships the scheduled
lane with a stale banner. The runner pulls master_prices from R2 first
(`scripts/pull_context_prices.py`) and, on a still-stale bar, retries once
after 10 minutes (`--require-fresh` exits 2, state still written).

That retry is NOT for the cron-timing hazard the spec cites: it read CLAUDE.md
when the PM price cron was documented as 20:30 UTC, and the workflow had
already moved to 21:10 UTC (17:10 ET EDT / 16:10 ET EST), which clears 18:30
in both. What can still leave a stale bar at 18:30 is the PM job not having
run: GitHub sheds scheduled workflows under load and never backfills a missed
cron, the same silent-skip class the pitch state's `build_pipeline` check
exists for. Ten minutes buys one window for a late run; a genuine miss
degrades to the scheduled lane, which is the correct outcome.

The engine reads `data/master_prices.parquet` DIRECTLY rather than through
`data_provider.get_history()`, so `data_provider._refresh_from_r2_if_needed`
never fires for it. That is deliberate: its 18-hour mtime threshold is
exactly wrong for an evening run. A copy pulled at 08:00 is 10 hours old at
18:30, counts as fresh, and holds yesterday's close.

Aligned sites — change together:
- `scripts/build_context_state.py`: `CONTEXT_UNIVERSE`, `EVENT_LANE_SUBJECTS`,
  `PRICE_TRIGGERS` and the event sweep <-> the trigger inventory table in
  `.claude/skills/market-context/SKILL.md`. A trigger added to one and not the
  other is invisible to the cell map, which is the stage that decides what
  publishes.
- the brief markdown skeleton in SKILL.md <-> `ITEM_HEAD` / `LANE_SECTIONS` in
  `scripts/send_context_slack.py`. The `1. **Title** [tag]` head is a parser
  contract, not a style choice.
- the sidecar `.json` schema in SKILL.md <-> `advance_flag_state` +
  `append_journal` (they read `fingerprint` and `mean_pct` out of it).
- `data/context_flag_state.json` is advanced by the SENDER after a successful
  post, never by the engine: the engine runs before anything is chosen, and
  advancing on a run that never posted would block tomorrow from saying
  something it never said.
- `z10` is defined in the engine to match `build_pitch_state._metrics_for`
  (10d return over 21d vol scaled to 10d), NOT `pitch_lab.zscore`, whose
  docstring claims the same definition but computes something else. The tape
  block in the payload comes from `_metrics_for`, so the trigger has to agree
  with it. Pinned in `tests/test_context_engine.py`.
- **any two-sided price trigger MUST carry a `side_fn`** and a `{side}`
  placeholder in its cell name, enforced by
  `test_two_sided_triggers_declare_a_side`. Pooling the tails produces a
  number that describes neither state and then tags it. Found live on the
  first manual run: `P5:rank5_extreme` on ^NDX scored +0.24% at t=2.53 pooled,
  earned a `solid` tag and passed BH, while the whole effect was the bottom
  tail rebounding (+0.51%, t=3.21) and the top side that was actually live sat
  at -0.02% with a 50.0% hit since 2018.
- two dedup helpers that look interchangeable and are not:
  `_first_in_calendar_days` is a NOVELTY filter (the state must have been
  absent, so a four-month grind to new highs is one piece of news) and
  `_first_in_sessions` is a DECLUSTERING filter (pitch_lab's rule, right for
  a regime cross).
- Runtime: `scripts/pull_context_prices.py`, `scripts/run_market_context.bat`,
  `scripts/register_market_context_task.ps1`,
  `scripts/context_headless_settings.json`,
  `scripts/check_context_delivered.py`.
- Guards: `tests/test_context_engine.py` (anchor convention, units, tags,
  trigger masks, novelty), `tests/test_context_sender.py` (parser, the three
  publish gates, block assembly, flag state, journal).

**LIVE since 2026-08-10.** Scheduled task `Market Context Brief`, Sun-Thu
18:30 ET, Interactive/Limited, PT1H, no auto-restart. Registered after ONE
manual shakeout evening (2026-08-09) rather than the several the house rule
asks for, at McKinley's call. The agent stage is the unproven part;
`check_context_delivered.py` is what makes a bad night loud instead of
silent. Read `scripts/logs/market_context_<date>.log` for the first week.

The scoped allowlist's folder-prefix form,
`Bash(python scratch/context_checks/:*)`, was VERIFIED in a real headless run
on 2026-08-10 and matches. The spec's fallback to `bypassPermissions` is not
needed and should not be adopted. Drill scripts must be invoked as
`python scratch/context_checks/<date>/<name>.py` from the repo root; an
absolute or quoted path does not match the prefix.

Delivery is SLACK ONLY (reaffirmed 2026-08-10). A one-off HTML email renderer
exists for reading a brief outside Slack; it is deliberately not in `scripts/`
and nothing schedules it.

Committed vs local: the JOURNAL (`data/context_journal.jsonl`), the flag state
and the drill scripts under `scratch/context_checks/` are committed — they are
the audit trail the journal's `drill_script` field points at. The BRIEFS
(`data/context_briefs/`) and the three generated payloads are gitignored by
decision (McKinley 2026-08-09): the brief is a Slack product, not a repo
artifact. No claims scoreboard exists and none is planned; nothing replays the
journal.
