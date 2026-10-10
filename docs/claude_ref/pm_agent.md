# PM Weekly (v1)

A **market-only weekly brief** written each Sunday by a Claude agent (`/pm-agent`). It covers what happened last week, what is most likely next week and why, and two **locked, falsifiable forecasts** that are graded against climatology. It also gives a read of the Risk Agent and a few questions as food for thought. It is delivered by email. It reads no book data, changes no rules and places no orders.

The design came out of a multi-agent design review (2026-10-09). The owner chose to ship the market-only brief first. Book check-ins (exposure, fills vs intent, realised vol, capital efficiency) are **phase 2**. They need the "wall" decision below first.

## Rules

- **Propose, never change.** The brief describes, forecasts and asks questions. The grammar (`pm_agent_grammar.BANNED_PHRASES`) refuses trading and rule-change language: resize, retune, raise or lower a cap, buy, sell, hedge or de-risk the book. Any idea about the book is a question. A change goes through a written prereg (CLAUDE.md "Pre-registration").
- **Market-only.** The R2 allowlist is `pm_agent_universe.MARKET_KEYS` plus the Risk Agent's published `risk_agent/today.json`. Book objects and Risk Agent internals are denied. Deny rules win, and anything not listed is denied. Check scripts run only through `scripts/pm_agent_run_check.py`. It refuses a script outside `PM_AGENT_HOME/checks/<date>/` or one whose source names a `FORBIDDEN_SOURCE_TOKENS` entry. The grammar refuses a cited evidence script for the same reasons.
- **Two claims, every week, both required:**

  | claim_type | fields | resolution |
  |---|---|---|
  | `spy_week_return` | p_up, q10_pct, q90_pct | SPY raw close on `resolves_on` / anchor close - 1, in percent |
  | `vix_week_change` | p_up, q10, q90 | ^VIX close on `resolves_on` - anchor close, in points |

  - The anchor is the state's `asof` close (raw, `master_prices`).
  - `resolves_on` is the last NYSE session of the ISO week after the anchor (`pm_agent_lab.target_week`).
  - `horizon_td` is the number of sessions between the two. Holiday weeks are shorter.
  - All of these come from the state, never from the agent.
- **Climatology is computed by code.** It is the trailing 10y distribution of `horizon_td`-session moves (`pm_agent_lab.climatology`). It is stored on each forecast at lock time and is the no-skill baseline.
- **Small N stays near the base rate.** If the evidence has n < 30, p_up may move at most 0.05 from climatology. Also enforced: 0.03 <= p_up <= 0.97, q10 < q90, |SPY q| <= 25%, and VIX q10 above -VIX.
- **Survey first.** The publisher refuses a brief unless `00_surface_map.md` is in the checks folder.
- **On time or unscored.** A brief published at or after 09:30 ET on the target week's first session is still delivered. Its forecasts are journaled `scored: false` and excluded from skill.
- **One brief per ISO week.** A second publish for the same `week` is refused.
- **Style.** ASCII only: no emoji, no em dashes. Number-dense.

## Independence (the Risk Agent stays blind)

- PM output lives in `PM_AGENT_HOME` (default `~/.pm_agent`, outside the repo) and in R2 `pm_agent/`. The Risk Agent's allowlist denies `pm_agent/` by default-deny, and `tests/test_pm_agent_universe.py` pins it. Nothing in the Risk Agent's code or skill mentions the PM, and a test pins that too.
- The PM never reads the Risk Agent before forecasting:
  - The state builder does not include the Risk Agent.
  - The headless settings deny reads of `risk_agent_*` files.
  - The publisher downloads `risk_agent/today.json` only after the forecasts are journaled, to a temp file it then deletes.
  - The pipeline test asserts the order (journal, then readout).
- **Known limit:** the Risk Agent still runs with `bypassPermissions` as the same Windows user. In principle it could read `~/.pm_agent` or this session's transcripts. Its skill confines it to its `data_catalog`. The hard version of this wall is the phase-2 decision.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Grade | `scripts/grade_pm_agent.py` | Resolves matured forecasts (raw closes). A bar missing 7 days after `resolves_on` gives `void`. Writes `scoreboard.json` (Brier and Brier skill vs climatology, q10/q90 coverage, pinball and pinball skill) and mirrors it to R2 |
| State | `scripts/build_pm_state.py` | Syncs `MARKET_KEYS` to `PM_AGENT_HOME/cache`, then writes `state.json`: target week, recap, daily path, vol, rates/FX, breadth, put/call, events, dashboard (context), climatology, anchors, scoreboard, recent briefs. It reuses the Risk Agent's pure block builders, pointed at the PM cache |
| Agent | `/pm-agent` skill (headless, scoped allowlist `scripts/pm_agent_headless_settings.json`, `--add-dir PM_AGENT_HOME`) | `checks/<asof>/00_surface_map.md`, check scripts, `brief.json` |
| Publish | `weekly_pm_agent.py` | Validates, journals the brief and the forecast records (sha256, climatology, anchor), pushes the journal, reads the Risk Agent readout, emails once behind a receipt, writes `today.json` and mirrors it to R2 `pm_agent/today.json` |
| Check | `scripts/check_pm_agent_delivered.py --require-r2` | Non-zero unless the week has exactly one brief or stand-down, the R2 journal agrees, and the receipt is `sent` |

Runner: `scripts/run_pm_agent.bat` then `scripts/invoke_pm_agent.ps1`. Model and effort are pinned at opus/xhigh, and the logs go to `PM_AGENT_HOME/logs`. There is no auto-retry, for the same reason as the Risk Agent.

**Schedule:** Sundays 16:00 ET on the trading desktop. Register it with `scripts/register_pm_agent_task.ps1` from `dev\New_Seasonals`. It runs after Friday's data and the Sunday 08:00 weekly rundown, and ends before Market Context (18:30) even at the 90-minute timeout.

## Journal (`PM_AGENT_HOME/journal.jsonl`, R2 `pm_agent/journal.jsonl`)

The journal is append-only and records are never edited.

| kind | Written by | Meaning |
|---|---|---|
| `brief` / `stand_down` | publish | Whole payload, week, model/effort, state sha256 |
| `forecast` | publish | One per claim: anchor, resolves_on, horizon, p_up/q10/q90, climatology at lock, evidence n/script, scored, sha256 |
| `resolution` | grader | `resolved` (value, up, below_q10, above_q90) or `void` (reason) |

## Phase 2 (not built; owner decisions needed)

- **The wall.** A separate Windows user or host for the PM, with its own Claude config/memory, R2 token and mailbox. This makes the Risk Agent's blindness enforced rather than instructed. It is required before the PM reads any book data.
- **Book check-in.**
  - Exception-only and code-only.
  - Starts with staleness per consumer (each with its own rule; never harmonized) and job/delivery health.
  - Fills-vs-intent (frozen RAW levels only) and ATR-risk vs cap come after their sources are proven.
- **Realised vol and capital efficiency.**
  - Descriptive only, with a basis tag on every number. The ledger is a rebuild on the flat $750k basis, not live NLV.
  - No vol band or target: a band is a vol target by another name, and that is a closed negative (`sizing.md`).
  - Live vol needs 20+ flow-adjusted NAV snapshots.

## Aligned sites, change together

- Claim vocabulary: `pm_agent_universe.CLAIMS`, `pm_agent_grammar._forecast`, `scripts/grade_pm_agent.py`, `.claude/skills/pm-agent/SKILL.md`, this doc.
- R2 boundary: `pm_agent_universe` allow/deny lists, `FORBIDDEN_SOURCE_TOKENS`, `scripts/pm_agent_headless_settings.json` read denies.
- Week mechanics: `pm_agent_lab.target_week` / `week_key`, used by the builder, the publisher deadline and the grader.

## Guard tests

`tests/test_pm_agent_universe.py`, `tests/test_pm_agent_pipeline.py`.
