# Daily Pitch — runbook

Three novel trade ideas every trading morning, invented from this repo's
knowledge, interrogated against its data before delivery, approved one at a
time by typing Y in a spreadsheet.

Spec of record: `daily_pitch_agent_spec_2026-08-06.html` (repo root).
Process instructions for the agent: `.claude/skills/daily-pitch/SKILL.md`.

## The morning, end to end

| time (ET) | what runs | where |
|---|---|---|
| 4:17 / 4:30 / 4:47 | the existing chain refreshes prices, the risk dial and the scan | GHA + local triggers |
| 5:10 | `scripts/run_daily_pitch.bat` | this machine, Task Scheduler |
| 5:10 | grade yesterday's ideas, rebuild the scoreboard | `scripts/grade_pitch_journal.py` |
| 5:11 | assemble state | `scripts/build_pitch_state.py` |
| 5:12 to ~6:00 | invent, falsify, compose, publish | `claude -p "/daily-pitch"` |
| ~6:00 | email lands, `Pitch` tab rewritten | `daily_pitch.py` |
| any time before 9:05 | McKinley types Y next to the ideas he wants | Sheets |
| 9:05 | approved MOO / MOC / close-limit rows placed | `pitch_moo.py --pass auction` |
| 9:32 | approved open-anchored limits priced and placed | `pitch_moo.py --pass open` |
| next morning | yesterday's Approve cells captured before the tab is rewritten | `daily_pitch.py` |

## Files

| file | role |
|---|---|
| `pitch_grammar.py` | the contract: entry and exit vocabularies, validation, Wilder-14 ATR, sizing, order derivation, fingerprints |
| `pitch_journal.py` | append-only journal (idea / killed / approval / outcome), R2 mirror |
| `pitch_lab.py` | shared research library every check script imports: price/event loading, lag-aware forward returns, declustering, controls, kill battery, sign test, horizon scan, loser paths, watchlist I/O (added 2026-08-08; consolidates the helpers the 08-07 run built ad hoc) |
| `data/pitch_watchlist.json` | parked near-misses, each with the number that turns it on; folded into the state, verdict owed in every B1 surface map, maintained after publish |
| `scripts/build_pitch_state.py` | Stage A state assembly to `data/pitch_state.json` + `data/pitch_tape.json` |
| `scripts/build_pitch_research_index.py` | research-doc index + parsed negative registry |
| `data/pitch_negative_registry.md` | dead ends the pipeline must not re-pitch (committed, grows) |
| `.claude/skills/daily-pitch/SKILL.md` | stages B to D: ideation axes, falsification contract, prose rules, schema |
| `daily_pitch.py` | publisher: validate, capture approvals, email, `Pitch` tab, journal |
| `scripts/grade_pitch_journal.py` | replay every pitched idea, approved or not, write the scoreboard |
| `scripts/check_pitch_delivered.py` | non-zero exit when a morning delivered nothing |
| `pitch_moo.py` (OneDrive `trading_ibkr`) | the approval runner |
| `tests/test_pitch_grammar.py`, `tests/test_daily_pitch.py`, `tests/test_pitch_grader.py`, `tests/test_pitch_lab.py`, `test_pitch_moo.py` (OneDrive) | guards |

## Manual run

```bash
python scripts/build_pitch_state.py            # add --no-book to skip Sheets and broker
claude                                          # then: /daily-pitch
```

Or drive the publisher directly once `data/pitch_ideas.json` exists:

```bash
python daily_pitch.py --ideas data/pitch_ideas.json --validate-only
python daily_pitch.py --ideas data/pitch_ideas.json --dry-run --html-out preview.html
python daily_pitch.py --ideas data/pitch_ideas.json
```

`--journal PATH` writes somewhere other than the real journal, and a
non-default path never touches R2. Use it for anything experimental: the
journal is the product's evidence trail and fixture rows in it corrupt the
scoreboard.

## Arming the automation

Both scheduled pieces are inert until the operator registers them. House
convention applies: eyeball several days of manual output first.

```powershell
# the 5:10 AM agent run (writes files and sends email; places no orders)
powershell -ExecutionPolicy Bypass -File "C:\Users\McKinley Slade\dev\New_Seasonals\scripts\register_daily_pitch_task.ps1"

# the 9:05 and 9:32 approval runners (live money)
powershell -ExecutionPolicy Bypass -File "C:\Users\McKinley Slade\OneDrive\trading_ibkr\register_pitch_moo_task.ps1"
```

To disarm order placement without unregistering anything, delete
`OneDrive\trading_ibkr\pitch_moo_enabled.flag`. The runner then validates,
reports, and refuses to contact the broker.

Dry check of a morning's approved basket, no broker contact:

```
python pitch_moo.py --check
python pitch_moo.py --pass open --check
```

## The approval loop

`daily_pitch.py` writes one `Pitch` row per LEG with an empty `Approve`
column. Type `Y` to take an idea.

- Only an exact `Y` is a yes. Blank, `N`, `yes`, `maybe` and anything else
  place nothing. Non-Y answers are printed so a typo is visible.
- **Approve every leg of an idea or none.** A partly approved multi-leg idea
  raises and the whole basket is refused; a half-built spread is worse than
  no trade.
- Rows marked `Manual_Only` are never placed. Futures legs are manual in v1,
  as is any MOO/MOC idea carrying a price stop or target (the fill price is
  not knowable at 9:05, and nothing here fabricates a stop off a reference
  close). The email prints those specs for hand entry.
- The tab is cleared and rewritten every morning. Answers must go in before
  the next run, which is also the only window in which the pipeline can read
  them back into the journal.

## Publish gates (2026-08-08)

The publisher used to enforce proof-of-work asymmetrically. A stand-down had
to show its sweep on disk; a morning that shipped three ideas showed nothing,
because validation never touched the filesystem and `evidence.script` was
accepted as any non-empty string. Three recall-generated ideas with no survey
behind them was therefore the one unguarded path through the whole pipeline,
and it is exactly the lazy failure mode. Now an ideas publish also checks
`scratch/pitch_checks/<asof>/`: the folder exists, holds `00_surface_map.md`
and at least one `.py` check, and every `evidence.script` and
`evidence.dev_script` resolves to a file inside it. A path into yesterday's
folder fails as stale, which is the machine-checkable half of "computed fresh
this morning". `dev_script` is new and required for every composed idea, so
stage C round 3 (horizon scan, entry form, exits, loser paths) leaves a
script rather than a claim.

A directed-only publish skips the surface-map check and nothing else. Survey
enforcement exists to constrain the agent, and McKinley directing one idea ad
hoc should not cost a full morning sweep; that idea still needs its own check
written today. `--validate-only` binds exactly like a real publish, since
iterating on validate-only is precisely when these should bite.
`--checks-root DIR` points the whole check at a fixture folder for dev runs.

Kill reasons are linted separately. A reason that reads as sample size and
nothing else prints `KILL-LINT: ...` and is tagged in the email's killed
footer, because "insufficient N" and "t below 2" are illegal standalone kills
under the small-N doctrine. The lint never blocks a publish: the match is a
heuristic over prose and a false positive must not cost a morning.

## Placement routing

Entry shape decides which pass can place an idea, and the grammar computes it:

| entry | pass | legs attached |
|---|---|---|
| `LIMIT` anchored to `CLOSE` | auction (9:05) | target, stop, time exit |
| `MOO` / `MOC` with a time exit only | auction (9:05) | time exit |
| `MOO` / `MOC` with a price stop or target | manual | none |
| `LIMIT` anchored to `OPEN` | open (9:32) | target, stop, time exit |
| any futures leg | manual | none |

Exit legs go up as one OCA group. The stop carries `goodAfterTime` = the next
session open, the book-wide day-2 arming convention, and the grader replays it
the same way.

## Model and effort

Pinned in `scripts/run_daily_pitch.bat`, not inherited:

```
set "PITCH_MODEL=opus"
set "PITCH_EFFORT=xhigh"
```

Without those flags the morning run would take whatever
`~/.claude/settings.json` said at the time, so changing models in an
interactive session one afternoon would quietly change every following
morning's pitch with nothing in the email to show it. The log line
`[agent: model opus, effort xhigh]` records what actually ran.

Opus at xhigh is the right tier here: stage C writes and interprets real
empirical checks, and the spec is explicit that the falsification stage must
not be truncated to save tokens. Subagents inherit the session's model and
effort, so the verifier fan-out runs at the same tier as the composer, which
is the point of the fan-out.

A full run is 8 to 12 candidates, 2 to 3 verifier agents each writing and
running a check script, a red-team pass, and composition. That is a heavy
morning by design. If cost ever needs cutting, cut the candidate count in the
skill, not the model tier and not the falsification stage.

## The pipeline line

Every pitch email carries one line near the top:

```
Pipeline: 7/7 overnight jobs ran | prices 2026-08-06 | dial 2026-08-06 | P/C 2026-08-05 (1 bd) - all current
```

Green when every tracked workflow has a successful run dated on or after the
previous trading session AND no cache is behind that session; red, naming what
is missing or stale, otherwise.

It exists because of the 2026-08-06 GitHub Actions incident, which skipped an
entire evening of crons. A job that runs and FAILS is already loud. A job that
never STARTS leaves no trace at all, and missed crons are never backfilled, so
the whole PM chain went missing with nothing to show for it. The rule (last
success on or after the prior session) covers both the pre-market dispatches
and the prior evening's crons, and does not false-alarm on Mondays the way a
flat 24-hour window would.

Green days print too, deliberately: silence would otherwise be ambiguous
between "all good" and "the check itself broke". Needs `GH_PAT_NEW_SEASONALS`;
without it the line says the check was unavailable rather than implying health.

## Conventions worth knowing before you change anything

- **ATR is Wilder-14 here.** The systematic book uses a simple 14-day mean of
  true range for every limit, stop and size. A pitch level is never a scanner
  level, and the two must not be mixed. `pitch_grammar.wilder_atr` matches
  `scripts/build_atr_downside_stats.wilder_atr` exactly.
- **Sanity bounds are ATR risk, not notional** (60 bps per idea, 150 bps for
  the day, 15k of risk in the runner's approved basket). Notional-denominated
  caps are rejected book wide. The runner adds an ATR-percent-of-price band
  because a corrupted ATR is the one input that inflates quantity while the
  risk figure still looks small.
- **The grader is pessimistic on purpose.** A bar touching both stop and
  target books the stop; a gapped stop fills at the open plus 13 bps.
- **Declined ideas are graded too.** The approved-versus-declined line in the
  email footer measures the filter, not just the pipeline.
- **Repetition control**: an idea whose structural fingerprint was pitched
  inside 10 trading days needs a `changed_since` sentence or it is refused.

## Failure modes

| symptom | what happened | what to do |
|---|---|---|
| no email, task shows failure | `check_pitch_delivered.py` found fewer than three idea records for today | read `scripts/logs/daily_pitch_last_run.log`; the run does not retry, and a missed morning delivers nothing rather than stale ideas late |
| `PITCH VALIDATION FAILED` | the agent's ideas json broke the grammar | fix the idea, never the grammar |
| state warnings box in the email | a stale price cache, a missing dial, unreadable Sheets | treat the affected ideas' evidence as suspect; the header says which |
| email never arrives, run otherwise green | SMTP creds missing or the Gmail app password was revoked | `send_email` now says `NOT DELIVERED` or `EMAIL SEND FAILED` in the log rather than returning quietly. Creds resolve from the environment first, then the repo `.env`. Rotate the app password at myaccount.google.com and update BOTH `.env` and the GHA `EMAIL_PASS` secret |
| runner logs `activation marker absent` | the flag file is missing | expected when disarmed; recreate the flag to arm |
| runner logs `MISSED_OPG_CUTOFF` | the 9:05 task ran late, past 9:25 | nothing placed by design, there is no MKT/DAY fallback |
| runner logs `NO_SESSION_OPEN` | IBKR returned no printed open, or a stale session bar | nothing placed; place by hand if still wanted |
| scoreboard empty | nothing has reached its time exit yet | expected for the first week |

## Open questions for McKinley

The spec left five, and the build took the spec's own defaults for all of
them. Each is a one-line change:

1. Delivery at ~6:00 AM ET, from a 5:10 AM run (the spec's original 7:00
   slot was moved up); the binding constraint is the 4:47 scan chain
   finishing.
2. Approval by Sheets `Y`, not by email reply (a reply keyword needs an inbox
   poller, which is a much larger build).
3. Ideas propose new trades only; none of them adjust an existing position.
4. Single stocks come from the liquid universe by way of the state file's tape
   table; the full CSV universe is not surfaced.
5. One grade-C idea per day is the cap, enforced in `pitch_grammar`.
