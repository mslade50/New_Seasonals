# Handoff: NQ/ES opening breakout, September 25 pilot

User's latest request: **test with one contract each if a valid setup occurs tomorrow**,
then hand this work to another agent. Tomorrow means **Friday, September 25, 2026,
America/New_York**. Prior explicit selections were shadow on live data, NQ 15bp /
ES 10bp effective risk, and **micros (MNQ/MES)**. Do not interpret “one contract” as
one full-size NQ/ES. This handoff did not enable orders or modify the running process.

## Current runtime — checked September 24 at 21:06 ET

- Local repository: `C:\Users\McKinley Slade\dev\New_Seasonals`.
- Shadow process **PID 14928**, healthy, `WAITING_FOR_PREOPEN`, over 75,000 events
  captured and all four signal/quote streams active. Recheck; this is dated evidence.
- Gateway **127.0.0.1:7496**, dedicated client **927480**, account ending **4234**.
  The exact account is in the ignored runtime config. Port **4001 has no CME data
  entitlement**. Both listeners are live-account Gateways; no paper listener was found.
- Config: `artifacts/open_breakout_runs/config-20260925-shadow.json`.
- State/logs/captures: `artifacts/open_breakout_runs/2026-09-25-shadow/`.
- This process records now, prepares inputs at **09:00 ET**, watches the cash open
  at **09:30**, and ends at **16:01**. It covers one session, not a recurring job.
- Both adapter and underlying IB client prohibit broker order transmission. Fills
  are simulated locally. Keep the PC awake and Gateway logged in.

Status command:

```powershell
python -m open_breakout status --state artifacts/open_breakout_runs/2026-09-25-shadow/runtime.sqlite
```

Graceful stop: create a file named `STOP` in that session directory. Preserve all
journals/captures. Do not run duplicate processes, reuse the active client ID, or
modify the running shadow config to enable orders. Python does not hot-reload edits.

## Frozen rules

- NQ/ES signals; same-expiry MNQ/MES execution. First 09:30 ET trade establishes open.
- Long/short breakout at open +/- **0.25 x prior full CME session true range**.
  Full session is 18:00-17:00 ET; true range includes the preceding session close.
- Initial stop: same quarter-TR distance from actual average fill, rounded outward.
  **No break-even, trailing stop, profit target or long filter.**
- Block shorts when prior cash-session **legacy 63d score SMA10 <20**. Do not switch
  to newer dashboard `main_score`. Equality 20 passes.
- Entries before **11:30 exclusive**; exit **15:55 ET**; one position per market;
  fresh recross after exit; current maximum **three submissions per market/day**.
- Shadow capital is existing `strategy_config.ACCOUNT_VALUE = $750,000`, not verified
  current broker NetLiquidation. Selected budgets: NQ $1,125 / ES $750; open cap
  25bp; submitted daily cap 75bp. Existing config permits up to 10 micros, so it
  **does not yet implement the requested one-contract pilot**.

## Verified inputs and contracts

December 18, 2026 contracts: NQ **563947726**, ES **515416632**, MNQ **815824267**,
MES **815824257**. Revalidate identities before any order.

Live NQ/ES trades, MNQ/MES bid/ask ticks, and NQ five-level depth were verified.
September 23/24 full-session IB minute history passed completeness checks. Cached
prior TR: NQ **457.5**, ES **76.25** points; runner fetches again at 09:00.

Local `data/rd2_fragility.parquet` was stale through September 23. Authoritative R2
was current through September 24 and downloaded into
`artifacts/open_breakout_build/activation_20260924/risk_authoritative.parquet`.
Legacy score **77.060357** allows both sides. The runner uses this frozen current
artifact. Do not silently fall back to the stale local cache.

## Next agent's task

1. Prepare a **one-MNQ / one-MES maximum** execution pilot that only acts on valid
   strategy signals and still passes risk, margin, freshness and ownership checks.
   Do not force a trade just to test connectivity.
2. Resolve the remaining execution scope: user said “test with 1 contract each”
   after selecting shadow, but did not explicitly specify **live versus IB paper**
   in this latest instruction. Also clarify whether the pilot should stop after
   **one entry per market**, or allow the existing three one-contract attempts.
3. Complete the broker acceptance work below before proposing live activation.
   Present the exact account, mode, instruments, quantity/attempt caps, possible
   exposure and stop/rollback procedure for the financial activation approval
   required by AGENTS.md. The current request to write a brief is not an instruction
   to place an immediate test order.
4. Retain a shadow-only fallback if execution cannot be validated before the open.

## Execution limitations that matter

**The live adapter is deliberately disabled in code**, even with `allow_live` and
`OPEN_BREAKOUT_LIVE_ACK`. `IBKR.connect()` rejects live mode; `send()` accepts only
paper; `standby.py` accepts shadow only. Do not merely remove these gates to meet
a date. There have been **no actual broker paper-order acceptance tests**.

Entries use marketable **IOC limits**, differing from the historical minute-path
research model. Each actual partial fill triggers a standalone stop update. Once
entry executions reconcile and the stop is acknowledged, a timed GTC market exit
joins the stop in an **OCA type-2 group**. There is a real **fill-to-stop-acknowledgement
protection gap**. Rejected/uncertain protection halts new entries but does not
provide a proven automatic emergency flatten. OCA partial-fill resizing and the
15:55 order's acceptance/survival through outages must be checked at IBKR.

Other blockers to verify: uncertain acknowledgements, partial/zero IOC fills,
stop rejection, disconnects, duplicate execution reports, orphan exits/reversals,
and timed exits. A stop-fill-during-ack race was reproduced and fixed offline.
Restarting a journal with orders intentionally halts for read-only reconciliation;
there is no automatic same-day resume. No unattended external alerting is installed.

The adapter requires exclusive ownership of the selected account/conId. Existing
stock/futures strategies must not share these contracts without a reviewed ownership
solution. Preflight found only a zero MES position, but **fresh positions and working
orders must be checked immediately before a broker pilot**. Cross-strategy/account
risk integration is not implemented.

## Source, evidence and workspace

- `open_breakout/`: config, pure strategy, inputs, SQLite journal, service, IBKR
  adapter, simulator, CLI and shadow standby runner.
- `tests/test_open_breakout.py`; focused suite including existing broker tests:
  **127 passed, 2 skipped**; latest workspace hygiene check passed.
- `docs/open_breakout_runbook.md`: setup, commands, assumptions and activation note.
- `artifacts/open_breakout_runs/2026-09-25-shadow/activation_verified.json`: startup
  receipt; `runtime.sqlite` records source hashes, heartbeat and phase.
- Read-only data probe and preflight evidence are under `artifacts/open_breakout_build/`.
- Frozen research candidate: `artifacts/research/qqq_open_breakout_20260923/current_candidate/`.
  The new service is not a new return/backtest certification.

Code is local/uncommitted (new package and tests are untracked). Preserve these and
all unrelated dirty changes. Do not create branches/worktrees without the user's
explicit request. Read AGENTS.md; use workspace hygiene before edits. No private-site,
stock scanner, OneDrive broker runtime, existing orders, or portfolio state was changed.
