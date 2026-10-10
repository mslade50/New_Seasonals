# PM layer: PM Weekly + daily check-in

A read-only "portfolio manager" beside the systematic book and the blind Risk Agent. It has two products:

- **PM Weekly** (`/pm-agent`, Sundays 16:00 ET, email):
  - What happened in markets last week, and what is most likely next week and why.
  - Two **locked, falsifiable forecasts** (SPY week return, VIX week change), graded against climatology.
  - A PM read of the book: live NLV, vol and exposure, fills, the ledger's vol, exposure and capital efficiency, sleeves, job health, and the week's check-in exceptions.
  - The Risk Agent readout.
  - Food for thought.
- **Daily check-in** (`scripts/pm_daily_check.py`, weekdays 08:30 ET, code only):
  - A fixed catalogue of exceptions.
  - Quiet by default: email only when an exception fires, but a journal record every session.

Neither product changes a rule, sizes anything or touches an order path.

History:
- 2026-10-09 design review (multi-agent).
- Owner decisions 2026-10-09: ship it; one Windows user and the current permissions (no separate-user "wall"); book reads allowed.

## Rules

- **Propose, never change.**
  - The grammar (`pm_agent_grammar.BANNED_PHRASES`) refuses trading and rule-change language: resize, retune, raise or lower a cap, buy, sell, hedge or de-risk the book, scale up or down, "vol target".
  - Ideas about the book are `questions`. A change goes through a written prereg (CLAUDE.md "Pre-registration").
- **No vol target.**
  - The book has none, and a book-level vol scaler is a closed negative (`sizing.md`).
  - Realised vol is reported only against the ledger's own 3-year distribution and against the ledger over the same days.
- **Basis on every book number.**
  - `live` means Primary NLV, positions and orders from the daily broker snapshot, plus the canonical fills store. It is not flow-adjusted: a day move over 5% is flagged as a suspected flow and excluded from vol.
  - `ledger` means the newest `site/builds/<run>/backtest_*.parquet`. It replays today's config on a flat $750k base, so pre-change notional understates live, and Overflow-tier numbers are survivorship upper bounds.
  - Each `book_notes` item must name its basis (grammar).
- **Read surface** (`pm_agent_universe`). Deny rules win, and everything else is denied.
  - Market: `MARKET_KEYS`.
  - Book: `BOOK_KEYS`, `BOOK_PREFIXES` (dated broker snapshots, automation receipts, agent delivery receipts) and the ledger pair under `site/builds/`.
  - Denied: the Risk Agent's journal, `morning_orders`, `trade_console`, the tagged-inventory seed and other agents' journals.
  - Check scripts run only through `scripts/pm_agent_run_check.py`. It refuses scripts outside `PM_AGENT_HOME/checks/<date>/` and any script naming the Risk Agent's working files or order-staging modules.
- **Two claims, every week, both required:**

  | claim_type | fields | resolution |
  |---|---|---|
  | `spy_week_return` | p_up, q10_pct, q90_pct | SPY raw close on `resolves_on` / anchor close - 1, in percent |
  | `vix_week_change` | p_up, q10, q90 | ^VIX close on `resolves_on` - anchor close, in points |

  - The anchor is the state's `asof` close.
  - `resolves_on` is the last NYSE session of the ISO week after the anchor (`pm_agent_lab.target_week`).
  - The climatology is a trailing 10-year baseline computed by code and stored at lock time.
- **Small N stays near the base rate.** If the evidence has n < 30, p_up may move at most 0.05 from climatology. Also enforced: 0.03 <= p_up <= 0.97, q10 < q90, |SPY q| <= 25%, and VIX q10 above -VIX.
- **Survey first.** The publisher refuses a brief without `00_surface_map.md`.
- **On time or unscored.** A brief published after 09:30 ET on the target week's first session is still delivered, but its forecasts are `scored: false`.
- **One brief per ISO week. One check-in per session.**
- **Style.** ASCII only: no emoji, no em dashes.

## Daily check-in catalogue

| kind | Fires when | Source |
|---|---|---|
| `job_failed` | An automation receipt has `status: failure` for the prior session or today. `degraded` alone does not fire | `automation/receipts/v1/<date>/<job>/latest.json` |
| `sleeve_task` | An ENABLED sleeve task is `Missing` or has a non-zero last result. Trend is skipped while `trend_moo_enabled` is false, and Legend while `legend_enabled` is false | `ops/sleeve_runtime_status.json` |
| `exit_missed` | Expected-exit obligations with `missed > 0` | `ops/expected_exit_status.json` |
| `fills_store` | A live_fills GAP, incomplete Primary coverage, or a last session behind the prior session | `live_fills_status.json` |
| `snapshot_missing` | No Primary broker snapshot for the prior session | `ops/olv_capacity/<date>.json` |
| `delivery_missing` | The Pitch or Seasonal (today), or the Risk Agent (prior-session asof), has no `sent` receipt | agent delivery receipts |
| `position_flip` | A Primary stock position is on the wrong side of its only strategy entry tag in 30 days (e.g. short after selling out a BUY-tagged entry). EXEC close tags are not entries | broker snapshot + `live_fills` |
| `nlv_move` | Primary NLV moved more than 3% in one session (a flow or a large P&L day) | broker snapshots |

- Each exception carries `days_running`, counted from prior check-ins.
- **Not here, deliberately:**
  - Stopless or unprotected position alerts. The owner said on 2026-10-09 that these are intentional.
  - Harmonised staleness. Each consumer keeps its own rule (CLAUDE.md).
  - Any sizing or dial proposal.
- First live findings on 2026-10-09 (dry run):
  - `macro_releases` had failed 4 days running.
  - Primary was short 700 BNS. The LT Trend ST OS exit sold 1,572 after two manual 700-share closes.

## Independence (the Risk Agent stays blind)

- PM output lives in `PM_AGENT_HOME` (default `~/.pm_agent`, outside the repo) and in R2 `pm_agent/`. The Risk Agent's allowlist denies `pm_agent/` by default-deny, and a test pins it. Nothing in the Risk Agent's code or skill mentions the PM (test).
- The PM never sees the Risk Agent's forecasts before locking its own:
  - The state has no Risk Agent block.
  - The headless settings deny reads of `risk_agent_*` files.
  - The publisher reads `risk_agent/today.json` only after journaling, to a temp file it deletes.
  - The pipeline test asserts the order.
- **Accepted limit (owner, 2026-10-09):** one Windows user. The Risk Agent runs `bypassPermissions` and could in principle read `~/.pm_agent`. Its skill confines it to its own `data_catalog`.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Grade | `scripts/grade_pm_agent.py` | Resolves matured forecasts on raw closes. A bar still missing 7 days after `resolves_on` gives `void`. Writes `scoreboard.json` (Brier and Brier skill vs climatology, q10/q90 coverage, pinball skill) and mirrors it to R2 |
| State | `scripts/build_pm_state.py` | Syncs market and book keys, then writes `state.json`: target week, recap, daily path, vol, rates/FX, breadth, put/call, events, dashboard, climatology, anchors, scoreboard, recent briefs, plus `book` (`pm_agent_book.build_book`) and `checkins`. `--no-book` gives a market-only state |
| Agent | `/pm-agent` skill (headless, scoped allowlist `scripts/pm_agent_headless_settings.json`, `--add-dir PM_AGENT_HOME`) | `checks/<asof>/00_surface_map.md`, check scripts, `brief.json` (incl. `book_notes`) |
| Publish | `weekly_pm_agent.py` | Validates; journals the brief and forecast records; pushes the journal; reads the Risk Agent readout; renders the email (code tables for tape, scoreboard, forecasts and book, plus the agent's prose); sends once behind a receipt; writes `today.json` to R2 `pm_agent/` |
| Check | `scripts/check_pm_agent_delivered.py --require-r2` | Non-zero unless the week has exactly one brief or stand-down, R2 agrees, and the receipt is `sent` |
| Daily | `scripts/pm_daily_check.py` via `scripts/run_pm_daily_check.bat` | A `check_in` journal record every session. Email (receipt in `PM_AGENT_HOME/checkin_receipts/`) only on exceptions. Writes `checkin_latest.json` to R2 `pm_agent/` |

Runners:
- Weekly: `scripts/run_pm_agent.bat` then `scripts/invoke_pm_agent.ps1`. Model and effort pinned at opus/xhigh, logs in `PM_AGENT_HOME/logs`, no auto-retry.
- Registration: `scripts/register_pm_agent_task.ps1` registers both tasks (`PM Weekly` Sun 16:00, `PM check-in` weekdays 08:30) from the trading desktop's `dev\New_Seasonals` checkout.

## Journal (`PM_AGENT_HOME/journal.jsonl`, R2 `pm_agent/journal.jsonl`)

The journal is append-only and records are never edited.

| kind | Written by | Meaning |
|---|---|---|
| `brief` / `stand_down` | publish | Whole payload, week, model/effort, state sha256 |
| `forecast` | publish | One per claim: anchor, resolves_on, horizon, p_up/q10/q90, climatology at lock, evidence n/script, scored, sha256 |
| `resolution` | grader | `resolved` (value, up, below_q10, above_q90) or `void` (reason) |
| `check_in` | daily check | Date, exceptions (with `days_running`), facts (NLV, exposure, fills health), emailed |

## Known limits

- Live NLV history starts 2026-09-16, when the broker snapshots begin. Live vol needs about 20 clean returns before it means much.
- NLV is not flow-adjusted. Days over 5% are only flagged, so a smaller deposit or withdrawal would read as P&L.
- Exposure is Primary only. PA is excluded.
- The ledger exposure series is modeled. It counts a position on its entry and exit days.

## Aligned sites, change together

- Claim vocabulary: `pm_agent_universe.CLAIMS`, `pm_agent_grammar._forecast`, `scripts/grade_pm_agent.py`, `.claude/skills/pm-agent/SKILL.md`, this doc.
- Read surface: `pm_agent_universe` allow/deny lists and `FORBIDDEN_SOURCE_TOKENS`, `pm_agent_book.dynamic_keys`, `scripts/pm_agent_headless_settings.json` read denies.
- Week mechanics: `pm_agent_lab.target_week` / `week_key`, used by the builder, the publisher deadline and the grader.
- Book basis labels: `pm_agent_book` docstring, `weekly_pm_agent.render_book`, `pm_agent_grammar.BASIS_RE`, the skill's Stage A item 6.
- Check-in catalogue: `scripts/pm_daily_check.py` docstring and this doc's table.

## Guard tests

`tests/test_pm_agent_universe.py`, `tests/test_pm_agent_pipeline.py`, `tests/test_pm_agent_book.py`.
