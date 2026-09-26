# Sleeve and hedge status

The existing `pipeline.html` page is labeled **Status** and opens with Event, Monthly Trend, Legend EMA, Open Breakout, paper SPY, and hedge status. Execution links directly to that section. This is an observability view: it does not place orders or enable trading strategies.

## Sources and freshness

`GET /sleeve-status` verifies Cloudflare Access before reading four fixed keys through the existing `CHARTS` R2 binding. Responses are not cached and contain only the projected status fields. The browser refreshes once a minute while visible, and offers a manual refresh.

| Key | Meaning |
|---|---|
| `event_sleeve_last_actions.json` | Producer actions/intended positions, not fills |
| `trend_sleeve_state.json` | Intended month-end allocation and cash gate |
| `dial_sleeve_paper.json` | Paper-only position/transition history |
| `ops/sleeve_runtime_status.json` | Read-only Windows task and gate observation |

Missing or malformed records are unavailable, never silently zero. One failed object does not suppress the other sleeves. Event/paper reports older than four calendar days and monthly Trend reports older than forty days are labeled aging; these are visibility thresholds, not exchange-session execution deadlines. Both payload date and object-upload time matter. A machine check expires after two hours. Losing the endpoint preserves prior cards with an explicit refresh-failed warning and leaves trade controls alone.

The Legend EMA and Open Breakout cards are built only from the machine check (see below), so they age with it. Neither card proves an order or fill succeeded. The hedge card is a dated scope review, not a live hedge observation; update it when D10 is delivered.

Hedge research exists in `scratch/ultracode_sizing_2026-09-02/dd_pit/pit_hedge_dd.md`; its findings call for a revised directional-overlay specification. `docs/plan_2026-09-04.md` D10 and `docs/running_list.md` O6 still list the specification and MES validation as open. The old dashboard hedge recommendation was retired July 16 (`scripts/build_risk_json.py:256`); it should not be represented as an existing protocol.

## Read-only machine collector

`scripts/publish_sleeve_runtime_status.py` publishes schema `sleeve-runtime.v2`. It reads five named Windows tasks (`IBKR Event Sleeve Auction Orders`, `IBKR Trend Sleeve MOO`, `IBKR Daily Order Chain`, `IBKR Legend EMA`, `IBKR Legend EMA Verify`) with state, last run, last result and next run. It also reads the Event, Trend and Legend enable markers in the executor directory (`legend_ema_enabled.flag` absent means the Legend runner is in dry run). It does not import execution code, connect to IBKR, start tasks, or change gates. A failed task inventory or missing executor directory refuses publication so prior evidence ages out.

Arguments: `--executor-root` (default `~/OneDrive/trading_ibkr`), `--breakout-runs` (default `artifacts/open_breakout_runs` in the checkout the script runs from), `--output` (local preview file, default under `artifacts/sleeve-status/`), `--print` (also echo the payload to stdout) and `--upload`. Without `--upload` nothing leaves the machine; with it the script writes only the fixed status key above. The old `--legend-runtime` argument is gone, along with the retired `NewSeasonals-LegendETF-*` tasks and their `runtime.env` fields.

**Legend EMA.** The collector reads `legend_ema_last_result.json` and `legend_ema_journal.jsonl` from the executor directory and reports the latest session date, the entries for that date (symbol, side, shares, status, outcome) and the skipped symbols with their reason. Only `Legend_EMA` records count; `Legend_EMA_TEST` records are ignored. The outcome comes from the exit fill legs first, then a `target_filled` record, then `entry_failed`, then the last verify result. The card headline comes from the runner task: missing, disabled, last run failed (with its exit code), verify failed, armed for live orders, or dry run. The card reads like "last: 2026-09-24 SPY long 1 sh, verify CLEAN · skipped QQQ no setup (...)", with the runner and verify task times listed underneath. The session ages after four calendar days.

**Open Breakout.** The collector scans `--breakout-runs` for folders named `<date>-shadow` or `<date>-live[-N]` and reads the newest shadow and newest live session's `runtime.sqlite`. It opens the database with a read-only URI and a two second timeout, so a running session keeps its writer lock and a missing database is never created. Only allow-listed meta fields are published: session, mode, pid, phase, heartbeat time and event count, finish time, last and feed errors, per-market risk bps and execution symbols, and the prior-range summary (status, ratio, skip, half, reason). The live acknowledgement, port, paths, source hashes and capital base are never copied, and any account-like string in an error is replaced with `[account]`. The card shows "No session today" when neither session is dated the current New York date, otherwise each mode's phase, heartbeat age and any markets skipped by the prior-range filter.

A session is stale when it has no terminal phase or finish time, its date is today or later (the shadow is often launched the evening before), it is before 16:05 ET on the session day, and its heartbeat was more than 120 seconds old. Heartbeat age is measured at the collector's `checked_at`, not at page view time: at a thirty minute cadence a view-time age would mark every running session stale. A true two-minute check needs the collector to run every minute or so on session days.

**Partial failures.** A missing journal, missing last-result file, missing or locked sqlite, or unreadable runs directory does not fail the publish. The affected card, or the affected mode within the Open Breakout card, reports "unavailable: <reason>" and the other cards publish normally.

**Schema change.** The Worker accepts only `sleeve-runtime.v2`. A previously uploaded v1 object is treated as an unavailable machine check, so every card shows "Machine check unavailable" until the first v2 upload. Deploy the Worker change and the collector together.

`scripts/register_sleeve_status_task.ps1` previews by default. After releasing to a stable checkout, invoke it with an existing Python interpreter containing the repo dependencies, a config directory containing the owner's normal R2 configuration, and the verified external executor directory. `-Install` registers **New Seasonals Sleeve Status** every thirty minutes under the current interactive owner, without waking the computer. Its PowerShell wrapper is hidden and propagates failure. The installer refuses to replace an existing task or install from an ephemeral task worktree. No existing trading task is edited.

As of September 26, 2026 the **New Seasonals Sleeve Status** task is not registered on this machine, so no machine check is being published.

## Release status, September 6, 2026

Implemented and locally verified; **not deployed**. No collector task registered, status object uploaded, alerts sent, or trading policy changed. The development preview used direct read-only canonical R2 observations and local machine metadata; it is not evidence of production freshness.

Release through the mandatory cloud-only site workflow from main. The existing workflow also runs `build_trade_ledger.py --upload`, replacing the canonical ledger read by live sizing. In addition, the active v8 scheduler dispatches site builds from its pinned older release; publishing this UI alone can be reverted by that subsequent build. Coordinate the production release and pinned scheduling policy, and validate the ledger/runtime implications before cutover. Do not silently change the trading runtime or use a local site build as a workaround.

## Verification

- `node tests/js/test_sleeve_status.mjs`: deployment modes; stale, future, invalid and missing data; fresh uploads of old reports; independent object failures; Access denial before reads; HTML escaping; refresh failure recovery.
  It also covers the Legend card (fresh, failed runner or verify, unavailable, aging) and the Open Breakout card (fresh, stale heartbeat, finished session, no session today, unreadable today, unavailable).
- `python -m pytest tests/test_publish_sleeve_runtime_status.py -q`: Legend and Open Breakout readers on temporary fixtures, including a missing journal, missing, corrupt and locked sqlite, newest-attempt selection, and account redaction.
- `node tests/js/test_access_fail_closed.mjs`: missing/malformed Access configuration and token rejection.
- Python compilation, actual read-only machine collection, and PowerShell installer preview.
- Browser development preview inspected against actual canonical report shapes at desktop and 390-pixel mobile widths; all five cards render and mobile content has no page overflow. Production deployment and live authenticated endpoint remain pending.
