# Sleeve and hedge status

The existing `pipeline.html` page is labeled **Status** and opens with Event, Monthly Trend, Legend EMA, paper SPY, and hedge status. Execution links directly to that section. This is an observability view: it does not place orders or enable trading strategies.

## Sources and freshness

`GET /sleeve-status` verifies Cloudflare Access before reading four fixed keys through the existing `CHARTS` R2 binding. Responses are not cached and contain only the projected status fields. The browser refreshes once a minute while visible, and offers a manual refresh.

| Key | Meaning |
|---|---|
| `event_sleeve_last_actions.json` | Producer actions/intended positions, not fills |
| `trend_sleeve_state.json` | Intended month-end allocation and cash gate |
| `dial_sleeve_paper.json` | Paper-only position/transition history |
| `ops/sleeve_runtime_status.json` | Read-only Windows task and gate observation |

Missing or malformed records are unavailable, never silently zero. One failed object does not suppress the other sleeves. Event/paper reports older than four calendar days and monthly Trend reports older than forty days are labeled aging; these are visibility thresholds, not exchange-session execution deadlines. Both payload date and object-upload time matter. A machine check expires after two hours. Losing the endpoint preserves prior cards with an explicit refresh-failed warning and leaves trade controls alone.

Legend currently has no session/fill evidence source in this view. Its machine status distinguishes absent, partial, shadow/unarmed, and same-day live configuration. Even a fully configured task set does not prove an order or fill succeeded. The hedge card is a dated scope review, not a live hedge observation; update it when D10 is delivered.

Hedge research exists in `scratch/ultracode_sizing_2026-09-02/dd_pit/pit_hedge_dd.md`; its findings call for a revised directional-overlay specification. `docs/plan_2026-09-04.md` D10 and `docs/running_list.md` O6 still list the specification and MES validation as open. The old dashboard hedge recommendation was retired July 16 (`scripts/build_risk_json.py:256`); it should not be represented as an existing protocol.

## Read-only machine collector

`scripts/publish_sleeve_runtime_status.py` reads six named Windows tasks, the Event/Trend enable markers, and only Legend's live-enabled/live-date environment fields. It does not import execution code, connect to IBKR, start tasks, or change gates. A failed inventory or missing executor directory refuses publication so prior evidence ages out. Without `--upload` it writes a local preview under `artifacts/sleeve-status/`; with that flag it writes only the fixed status key above.

`scripts/register_sleeve_status_task.ps1` previews by default. After releasing to a stable checkout, invoke it with an existing Python interpreter containing the repo dependencies, a config directory containing the owner's normal R2 configuration, and the verified external executor directory. `-Install` registers **New Seasonals Sleeve Status** every thirty minutes under the current interactive owner, without waking the computer. Its PowerShell wrapper is hidden and propagates failure. The installer refuses to replace an existing task or install from an ephemeral task worktree. No existing trading task is edited.

## Release status — September 6, 2026

Implemented and locally verified; **not deployed**. No collector task registered, status object uploaded, alerts sent, or trading policy changed. The development preview used direct read-only canonical R2 observations and local machine metadata; it is not evidence of production freshness.

Release through the mandatory cloud-only site workflow from main. The existing workflow also runs `build_trade_ledger.py --upload`, replacing the canonical ledger read by live sizing. In addition, the active v8 scheduler dispatches site builds from its pinned older release; publishing this UI alone can be reverted by that subsequent build. Coordinate the production release and pinned scheduling policy, and validate the ledger/runtime implications before cutover. Do not silently change the trading runtime or use a local site build as a workaround.

## Verification

- `node tests/js/test_sleeve_status.mjs`: deployment modes; stale, future, invalid and missing data; fresh uploads of old reports; independent object failures; Access denial before reads; HTML escaping; refresh failure recovery.
- `node tests/js/test_access_fail_closed.mjs`: missing/malformed Access configuration and token rejection.
- Python compilation, actual read-only machine collection, and PowerShell installer preview.
- Browser development preview inspected against actual canonical report shapes at desktop and 390-pixel mobile widths; all five cards render and mobile content has no page overflow. Production deployment and live authenticated endpoint remain pending.
