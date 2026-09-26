# NQ / ES opening breakout: IBKR service

Status (2026-09-25): **Friday 2026-09-25 did not trade.** The 7496 Gateway restarted
around 07:00 ET and was not back before the open, so the shadow failed and the live
pilot was never launched; no orders were placed. **Monday 2026-09-28 is staged**
(shadow on client 927480 plus the live one-contract pilot, 1 MNQ / 1 MES, client
927481) with dated launchers; nothing is launched yet. See "Session 2026-09-25 outcome
and Monday 2026-09-28 staging" at the end. Live routing now exists
only behind layered guards: `mode: live` + `allow_live` + a non-DU account + the
`pilot` block + every `max_contracts == 1` + the code constant
`LIVE_PILOT_MAX_CONTRACTS = 1` + the exact session/account acknowledgement
`OPEN_BREAKOUT_LIVE_ACK`, and only through the `live-session` command after a
passing 09:25 ET preflight. `shadow-session` still refuses live configs. See the
dated sections below for both sessions.

## Frozen candidate

- Signals: NQ and ES mini futures; execution: same expiry NQ/MNQ and ES/MES.
- Open: first live Last print at 09:30 ET, arriving within two seconds by default.
- Entry thresholds: open +/- 25% of the previous full CME session true range.
  Range is raw same-contract 18:00-17:00 ET OHLC with the preceding session close.
- Initial stop: 25% of that true range from the actual average fill, rounded away
  from the position to a valid tick. Partial fills update the average and stop.
- No break-even, target, trailing stop or long filter. Short signals require the
  **previous cash-session legacy 63d risk score, smoothed over 10 observations,
  to be >=20**. The newer dashboard `main_score` is intentionally not selected.
- Entry cutoff 11:30 ET exclusive; 15:55 ET timed market exit; one position and at
  most three entry submissions per market/day; fresh recross after a closed trade.
- Example 15bp NQ / 5bp ES and $100,000 shadow equity are **illustrative**, not an
  approved allocation or actual account balance. All bps here are effective:
  the stock book's global multiplier is not applied. Configure the final budgets,
  fees, margin reserve and contract caps before a paper acceptance run.
- Config option (not part of the frozen research rules): `prior_range_filter`
  `{enabled, threshold, mode}` skips a market for the session when the previous full
  session's TR divided by its prior 20-session average is at or above the threshold,
  and fails closed when that average cannot be computed. Absent means off, with an
  unchanged config fingerprint. On in the live pilot config since 2026-09-25 (1.25,
  skip), off in the shadow config. See "Prior-range skip filter" at the end.

## Components

`open_breakout/config.py` validates explicit accounts, dedicated client IDs and
contract IDs. `inputs.py` prepares and validates point-in-time session inputs.
`strategy.py` implements signals and whole-contract risk sizing. `service.py`
owns the lifecycle and reconciliation checks. `store.py` provides SQLite WAL,
full synchronous writes and a single-process lock. `ibkr.py` is the optional
IBKR transport; `replay.py` is an offline quote-fill transport.

There is no dependency on ignored research scripts, the stock scanner, Google
Sheets, the private site, or the OneDrive broker installation. A chosen execution
contract must be exclusively owned by this strategy in its account: another
strategy's position or working order in that conId halts new entries. This release
does not integrate the stock book's account-wide risk limits or net multiple
strategies sharing a futures contract.

## Configure

Use a Python environment with the repository dependencies and, for an IBKR
connection, `python -m pip install -r open_breakout/requirements.txt`. This reuses
`ib_insync==0.9.86`, already used by repository broker tests. Offline replay imports
no IBKR library. Use the repository root as the working directory.

Copy `config/open_breakout.example.json` to a local path under `artifacts/` and
replace every placeholder. The template intentionally fails validation. Use
contract details from IBKR for conId and the exact YYYYMMDD expiry; signal contracts
must be NQ/ES and execution may use the matching mini or micro. Review the active
volume/front contract and rollover each day. There is no silent expiry-based roll
or continuous-futures order substitution.

Select the exact IBKR account, host, API socket port and an unused dedicated
client ID. Paper orders require a DU account, port 7497 (TWS) or 4002 (Gateway),
`mode: paper`, and `allow_live: false`. Shadow connects with API readonly enabled
and routes all modeled orders to the simulator, never to IBKR. Keep state and
captures outside OneDrive and preserve each session's journal.

Verified on 2026-09-24: live NQ/ES/MNQ/MES quotes, NQ/ES Last ticks, micro BidAsk
ticks, and NQ market depth on Gateway port 7496. Port 4001 did not have the CME
entitlement. Actual order acceptance, paper-account data sharing and outage behavior
remain unverified. TradingView entitlement does not establish IBKR API entitlement.

```powershell
python -m open_breakout validate --config artifacts/open_breakout/config.json
python -m open_breakout prepare --config artifacts/open_breakout/config.json --session YYYY-MM-DD --risk-parquet data/rd2_fragility.parquet --roll-verified --out artifacts/open_breakout/YYYY-MM-DD.inputs.json
python -m open_breakout run --config artifacts/open_breakout/config.json --inputs artifacts/open_breakout/YYYY-MM-DD.inputs.json --state artifacts/open_breakout/YYYY-MM-DD.sqlite --capture artifacts/open_breakout/YYYY-MM-DD.ticks.jsonl
python -m open_breakout status --state artifacts/open_breakout/YYYY-MM-DD.sqlite
```

`prepare` is read-only at IBKR and must run on the session date before 09:30 ET.
It requests seven days of raw 1-minute signal-contract TRADES and requires complete
prior sessions (the historical 16:15-16:30 halt may be absent). Missing bars, stale
risk data, cash holidays, early closes and intervening weekday holidays fail closed.
Holiday-adjacent trading is intentionally skipped pending separate CME calendar
validation. The reviewed `--roll-verified` flag is an operator assertion, not an
automatic volume-roll detector. Input creation never overwrites an existing file.

Start `run` before 09:30. A missed open skips the market. Delayed/frozen data,
stale quotes, a signal-stream gap longer than the configured tolerance, an unknown
position/order, rejected protection or an uncertain acknowledgement halts new
entries and records the reason. The process prints a halt notification; there is
no unattended email/SMS escalation installed. Keep a human watching paper runs.

## Execution and recovery

Entries are marketable IOC limit orders, with a configured maximum chase beyond
the current bid/ask. This bounds the entry price but can produce partial/no fills.
Sizing includes stop rounding, configured round-trip fees and an exit-slippage
reserve. Caps include total daily submitted risk and simultaneous open risk; unused
risk from an unfilled IOC stays consumed for that day. Paper sizing uses the
configured account's positive USD equity/excess liquidity and an IB what-if margin
check. Risk budgets are planning limits, not maximum losses through gaps.

Every intent is committed before transmission. Every execution ID is deduplicated.
An actual partial entry immediately creates/updates its standalone protective stop.
Once entry execution reports reconcile and the stop is acknowledged, a GTC market
exit with `goodAfterTime=15:55 America/New_York` is submitted in the same OCA group
(type 2: reduce remaining quantity with blocking). The process checks that the
position is flat after 15:56. It never issues a speculative second flatten order.

**There is a protection gap between an entry fill and broker acknowledgement of
its stop.** A crash, rejected stop or lost connection there requires prompt manual
handling in TWS. Resting stops/OCA/timed exits are broker requests, not a verified
outage guarantee. A halted process still handles protective stops for incoming
known entry executions, but never opens new positions or cancels existing protection.
This is a principal reason live activation is blocked pending broker testing.

A restarted journal containing any orders is halted for inspection. It does not
resubmit PREPARED intents, reconstruct positions from status messages, or assume a
missing order never reached IBKR. Run the read-only comparison:

```powershell
python -m open_breakout reconcile --config artifacts/open_breakout/config.json --state artifacts/open_breakout/YYYY-MM-DD.sqlite
```

This reports broker positions/open orders and available execution reports alongside
the journal. IB execution-history availability is limited; retain the journal and
broker statements. Inspect account/contract/client IDs in TWS, resolve uncertain
orders/positions manually, and retain the failed session's files. **No same-day
automatic resume or halt-reset command is provided.** After a clean reconciliation,
start a newly prepared session on the next trading day. Do not bypass a halt by
launching a second journal against a possibly open account.

## Offline replay

JSONL input contains timezone-aware events, in arrival order. Live captures also
record `received_at`; replay uses that timestamp to preserve observed data latency
and the exchange `time` to test quote/trade freshness. Handwritten fixtures may omit
`received_at` to model zero latency:

```json
{"kind":"quote","market":"NQ","time":"2026-09-24T09:30:00-04:00","bid":20000,"ask":20000.25}
{"kind":"trade","market":"NQ","time":"2026-09-24T09:30:00-04:00","price":20000.25}
```

```powershell
python -m open_breakout replay --config artifacts/open_breakout/config.json --inputs artifacts/open_breakout/YYYY-MM-DD.inputs.json --ticks artifacts/open_breakout/YYYY-MM-DD.ticks.jsonl --state artifacts/open_breakout/replay-YYYY-MM-DD.sqlite
python -m pytest tests/test_open_breakout.py -q -p no:cacheprovider
```

Replay and shadow fill immediately at quotes, with no order-book queue, market
impact, broker latency or real margin simulation. Their ledger is an operational
check, not a new strategy return estimate. A simulated stop uses executable bid/ask;
actual IB stop triggering must be observed in paper tests. Historical research used
minute paths and slippage; real tick crossings, IOC attempts, conservative risk
reservations, strict missing-data/calendar guards and explicit contract selection
can change trades. Return figures from the research backtest have not been
re-certified by this service.

## Acceptance before a live release

1. Confirm the exact account, API data entitlements, contract IDs/roll policy,
   effective NQ/ES allocations, fees, tick-subscription capacity and margin limits.
2. Observe full shadow sessions; compare opening print, prior range, legacy risk
   vintage, crossings and rejected opportunities against captured data/research.
3. Explicitly authorize and run paper orders: long/short, IOC partial and zero fill,
   stop partial fill, OCA quantity reduction, 15:55 activation, rejected stop,
   delayed acknowledgement, lost stream, TWS disconnect and process restart.
4. Verify no orphan exit/reversal and no duplicate entry on reconnect; demonstrate
   actual broker-held protection/time exit during an outage and document gaps.
5. Resolve recovery/protection findings and account-wide coexistence. Only then
   review a live-enabled release and obtain explicit live activation approval.

IBKR references: [bracket order transmission](https://www.interactivebrokers.com/docs/general/order-types/complex-orders/bracket-orders),
[API order fields including OCA and timing](https://www.interactivebrokers.com/docs/tws-api/ref/order),
[third-party data/API FAQ](https://www.interactivebrokers.com/docs/third-party-integrations/general-third-party-frequently-asked-questions).


## Shadow activation: 2026-09-25 session

User-selected settings: **shadow only**, NQ 15bp / ES 10bp effective risk,
December 2026 MNQ/MES simulated execution using December NQ/ES signals.
The capital base is the existing strategy-book `ACCOUNT_VALUE` of $750,000,
not a claim about current broker NetLiquidation. Initial per-trade budgets are
$1,125 / $750 before rounding down to whole contracts. Open-risk cap is 25bp;
submitted daily-risk cap is 75bp; maximum is 10 micro contracts per market.

Runtime configuration: `artifacts/open_breakout_runs/config-20260925-shadow.json`.
Session state/capture: `artifacts/open_breakout_runs/2026-09-25-shadow/`.
Account ends in 4234; dedicated API client ID 927480; localhost port 7496.
Source hashes, process ID, phase and live heartbeat are recorded in `runtime.sqlite`.
The initial process PID was 14928; always inspect the current heartbeat/process
rather than assuming this historical PID remains active.

The authoritative R2 risk parquet was downloaded to a new artifact and verified
through September 24; its legacy 63d SMA10 score was 77.060357. Both sides are
allowed by the selected short gate. Read-only IB history passed complete-session
checks for September 23/24. Prior full-session ranges were NQ 457.5 points and
ES 76.25 points. The runner fetches and validates history again at 09:00.

Launch command (already started; do not launch a duplicate):

```powershell
python -m open_breakout shadow-session --config artifacts/open_breakout_runs/config-20260925-shadow.json --session 2026-09-25 --risk-parquet artifacts/open_breakout_build/activation_20260924/risk_authoritative.parquet --state-dir artifacts/open_breakout_runs/2026-09-25-shadow --roll-verified
python -m open_breakout status --state artifacts/open_breakout_runs/2026-09-25-shadow/runtime.sqlite
```

The single-session process runs in a hidden window with stdout/stderr logs in that
folder. Keep this PC awake and IB Gateway logged in. It reconnects before arming;
a connection failure after arming halts the session for review. It stops at 16:01
ET on September 25 and does not automatically trade or repeat on later dates.
To stop gracefully, create a file named `STOP` in the session directory. Preserve
all existing state and captures when diagnosing or restarting.

Hourly `.jsonl.gz` captures preserve exchange and receipt timestamps and are
readable by the replay CLI. The runtime lock prevents a duplicate writer to this
session. In addition to the adapter's shadow gate, the underlying IB client order
transmission method is disabled inside this process. All modeled fills remain in
the local simulator and journal.


## Live pilot: 2026-09-25 session

Owner decision (2026-09-24): trade **one MNQ and one MES contract live** on Friday
2026-09-25 if, and only if, a valid strategy signal occurs. There is no paper Gateway
on this machine, so this one-lot live pilot is the broker acceptance test. The shadow
session above keeps running as the comparison baseline.

- Mode **live**; account ending **4234** (full account only in the ignored config);
  IB Gateway `127.0.0.1:7496`; dedicated API client ID **927481** (never 927480).
- Signals: NQ Dec-2026 conId 563947726, ES Dec-2026 conId 515416632.
  Execution: **MNQ Dec-2026 conId 815824267**, **MES Dec-2026 conId 815824257**
  (expiry 20261218, multipliers 2 / 5, tick 0.25; verified again at every connect).
- **One-contract hard cap**, enforced three ways: `LIVE_PILOT_MAX_CONTRACTS = 1` in
  `open_breakout/config.py` (sizing clamp after all existing sizing, and a refusal in
  `IBKR.send()` for any live order above it); the config's `max_contracts: 1` per
  market; the config's `pilot.max_contracts_per_market: 1`. A computed size of 0 is
  still no trade.
- Strategy rules unchanged: open +/- 0.25 x prior TR, quarter-TR stop from the fill,
  entries before 11:30, 15:55 timed exit, fresh recross after exit, up to **three entry
  submissions per market/day**. Risk budgets stay 15bp NQ / 10bp ES on the $750,000
  capital base (25bp open / 75bp daily caps). Margin checks use the account's actual
  IB NetLiquidation / ExcessLiquidity (limit: 20% of the smaller of the two).
- Worst case per market at the Sep 24 ranges (NQ TR 457.5, ES TR 76.25; the runner
  re-fetches TR at 09:00): MNQ stop 114.375 pts x $2 = **$228.75 per attempt, $686.25
  for three**; MES stop 19.0625 pts x $5 = **$95.31 per attempt, $285.94 for three**.
  With fees ($0.85/side) and the 4-tick exit reserve the sized risk is about $232 and
  $102 per attempt (about $1,000 for six losing attempts). Stops are market orders
  once triggered; gaps and fast markets can exceed these figures.
- Preflight (read-only) at connect and again at **09:25 ET** before arming: zero
  MNQ/MES position; no working MNQ/MES order from any API client or TWS
  (`reqAllOpenOrders`); contract identities; what-if margin for 1-lot BUY and SELL in
  each execution contract within the margin limit; all four data streams live and
  fresh. Any failure: no arming, session ends `HALTED_PREFLIGHT`. Until arming, an
  order gate below the adapter lets only what-if previews through.
- Protection: each entry fill gets a standalone STP (GTC) immediately, then a GTC MKT
  exit with `goodAfterTime` 15:55 America/New_York in the same OCA group (type 2). Every
  order carries the account and an `orderRef` of the form
  `MNQ|BUY|OpenBreakout|2026-09-25|NQ-1-STOP` (strategy is the 3rd pipe field, as the
  nightly execution report parses it). The timed exit is sent for every acknowledged
  stop, even if the session halted meanwhile (it can only reduce). After any exit fill
  takes the position to zero, the remaining sibling (stop or timed exit) is cancelled
  explicitly rather than trusting OCA, and a sibling still working 10 seconds later
  halts with an `ORPHANED_EXIT_ORDER` alert (cancel it by hand).
- **Emergency flatten (stop rejection).** Only an *explicit* rejection counts: a
  genuine `Inactive` status from TWS, or a stop whose last IB log entry carries error
  110, 200, 201, 203, 321 or 10147. ib_insync marks an order Cancelled for any
  non-warning error, so a Cancelled stop with any other code (or none) does **not**
  flatten: after one second, if the journal still holds the position, the session halts
  and alerts "FLATTEN BY HAND". On an explicit rejection with a position open the
  process halts, cancels the stop and timed exit, **waits up to 3 seconds for TWS to
  confirm both cancels** (a filled order or a 161/10148 answer is not a confirmation),
  takes a fresh position snapshot, and only if the broker still holds exactly the
  journal position sends **one** market flatten (DAY). Unconfirmed cancels or a position
  mismatch abort the flatten with a "FLATTEN BY HAND" alert. It never sends a second
  flatten. Warning codes (2104, 2106, 2108, 2109, 2158, 399, 404) never flatten. An
  uncertain acknowledgement halts new entries only.
- Other live safeguards: a stale or dislocated execution quote at entry time skips
  that one signal (no halt, no attempt consumed; a re-entry needs a fresh recross);
  the per-entry what-if is skipped in live (margin was checked at connect and 09:25);
  an entry execution arriving after the IOC looked finished is recorded, protected with
  a stop and timed exit, then the session halts. The watchdog halts on a broker/journal
  position or stop discrepancy only after **3 consecutive** stable checks, and on a
  signal stream silent for `watchdog_stale_seconds` (default 30, optional config key);
  the 3-second freshness still applies at entry. Market-data farm messages 2103/2105
  are warnings. From 15:56, any journal position or any working `OpenBreakout` order on
  MNQ/MES halts with an alert.
- Alerts: console always. Every new halt reason (and any halt while a position is open),
  preflight failure, arming, entry fill, flatten, orphaned exit and process exit also go
  to the repo's existing `SLACK_WEBHOOK_URL` (from `.env`) on background threads that
  never block the order path. The process waits up to 5 seconds for pending posts
  before it exits.
- **Attended session:** someone must watch TWS and Slack from **09:25 to 11:30 ET** (arming
  and all entries) and **15:50 to 16:01 ET** (timed exit and final checks).

Preflight evidence (client 927481, nothing placed):
`artifacts/open_breakout_runs/preflight-20260925-live.json` - all checks passed
(re-run after the review fixes; see the file's `checked_at`).

Launch (this is the live activation step; run once, before 09:25 ET on 2026-09-25):

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/open_breakout_runs/launch-20260925-live.ps1
python -m open_breakout status --state artifacts/open_breakout_runs/2026-09-25-live/runtime.sqlite
```

The launcher sets `OPEN_BREAKOUT_LIVE_ACK="LIVE 2026-09-25 <full account>"` for the
child process only, refuses if the state dir already has a journal or another
`live-session` process is running, and starts this command hidden with
`launch.stdout.log` / `launch.stderr.log` / `launcher.pid` in the state dir:

```powershell
python -m open_breakout live-session --config artifacts/open_breakout_runs/config-20260925-live.json --session 2026-09-25 --risk-parquet artifacts/open_breakout_build/activation_20260924/risk_authoritative.parquet --state-dir artifacts/open_breakout_runs/2026-09-25-live --roll-verified
```

**Relaunch** is allowed only when the earlier attempt placed **no orders** (for example,
the 09:00 history step failed or a preflight failed). Use a fresh state dir; never
reuse the old one:

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/open_breakout_runs/launch-20260925-live.ps1 -Attempt 2
```

That writes to `2026-09-25-live-2`. `live-session` refuses to start if any
`2026-09-25-live*` journal already contains orders. In that case reconcile instead.

Re-run the read-only preflight at any time (needs the same env ack; add `--out` to
save it). While the live process is running it owns client 927481, so pass a different
read-only client ID:

```powershell
python -m open_breakout preflight --config artifacts/open_breakout_runs/config-20260925-live.json --session 2026-09-25 --client-id 927482
```

**Graceful stop:** create a file named `STOP` in
`artifacts/open_breakout_runs/2026-09-25-live/`. The process exits within about a
second. It does not cancel or flatten anything: any broker-held stop and 15:55 exit
remain working.

**Manual rollback:** kill the process (PID in `launcher.pid` or the runtime heartbeat).
Broker-held protection stays: the STP and the 15:55 GTC MKT exit (same OCA group) are
at IBKR, not in the process. To be flat immediately, flatten MNQ/MES by hand in TWS
and cancel the remaining OpenBreakout orders there (`orderRef` 3rd field
`OpenBreakout`). A halted or restarted journal never resumes the same day; run
`python -m open_breakout reconcile --config artifacts/open_breakout_runs/config-20260925-live.json --state artifacts/open_breakout_runs/2026-09-25-live/trades.sqlite --session 2026-09-25 --client-id 927482`
(with the env ack) for a read-only comparison. `--client-id 927482` is required while
the live process is still alive (it owns 927481). The override must differ from the
configured ID.

Known limits for this pilot: the fill-to-stop-acknowledgement gap remains; if the stop
is never acknowledged, no timed exit is sent and the 15:56 check raises the alarm; a
stop refused locally before transmission halts without flattening; a Gateway
disconnect after arming ends the process (protection stays at IBKR).

Still unverified at the broker (to be observed during this pilot): acceptance and
survival of the GTC MKT order with `goodAfterTime` 15:55 America/New_York; OCA type 2
behaviour with STP plus a timed MKT on CME micros; and the real shape of a stop
rejection (Inactive vs ib_insync's synthesized Cancelled, and which error codes).

2026-09-25 what-if finding: Gateway server version 176 rejects `goodAfterTime`
strings ending in `US/Eastern` with error 337 (invalid date/time/time zone), while
`America/New_York` is accepted. The service used `US/Eastern` until this date, so
every timed exit would have been refused. Fixed in `service.py`; the what-if is in
`artifacts/open_breakout_build/mechanics_test.py --dry-run`. Survival of the
accepted order through the day is still to be observed.


## Session 2026-09-25 outcome and Monday 2026-09-28 staging

**What happened on 2026-09-25.** The IB Gateway on port 7496 restarted around 07:00 ET
and did not come back before 09:30. The shadow process (started Sep 24, PID 14928) kept
waiting for the connection and then failed with "Opening missed before connection
recovered" (`artifacts/open_breakout_runs/2026-09-25-shadow/launch.stdout.log`: "SHADOW
FAILED"). The live pilot was never launched. Nothing traded and no orders were placed.
All three pilot checks (GTC timed exit, OCA type 2, stop rejection shape) are still
unobserved and carry over to Monday.

**Gateway requirement.** The 7496 Gateway must be logged in and listening **before
08:45 ET** on the session day. Set its daily auto-restart outside 07:00 to 16:30 ET
(for example 23:45 ET), or enable auto-login so a restart comes back unattended.
Check it before launching anything: the preflight in step (a) below fails if it is not up.

**Risk refresh.** The short gate needs the legacy score for the prior cash session.
The one-shot 19:00 task on 2026-09-25 was killed at its time limit and would have been
too early anyway: R2 received Friday's row at 19:41 ET. Since 2026-09-26 the weekday
scheduled task `OpenBreakout_RiskRefresh_Nightly` runs
`artifacts/open_breakout_build/refresh_risk.py --session next` at 20:30 ET
(`refresh_risk_nightly.cmd`), with a 15 s connect / 60 s read timeout on the R2 call so
a stalled request cannot hang it. `next` resolves to the next XNYS session. Each run
writes a new `artifacts/open_breakout_build/risk_<UTC stamp>/` folder with
`risk_authoritative.parquet` and `refresh.json`, and appends to
`artifacts/open_breakout_build/refresh_nightly.log`. A good run shows the session
date, `risk_latest` equal to the prior cash session, a `legacy_score` and `short_gate`.
A `risk_error` means R2 was not updated yet; rerun the same command by hand later.
The launchers auto-select the newest `refresh.json` for their `-Session` that has no
`risk_error`. The 2026-09-28 refresh was run by hand on Saturday 2026-09-26 (score
76.78, gate OPEN); the two `risk_20260925_*` folders carry `risk_error` and are never
selected.

**Launchers.** Two dated-by-argument launchers replace the one-off Sep 25 script (kept
as history):

- `artifacts/open_breakout_runs/launch-shadow.ps1` (shadow-session, config
  `config-20260925-shadow.json`, client 927480, state dir `<Session>-shadow[-N]`)
- `artifacts/open_breakout_runs/launch-live.ps1` (live-session, config
  `config-20260925-live.json`, client 927481, state dir `<Session>-live[-N]`)

Both take `-Session YYYY-MM-DD` (required), `-Risk <parquet>` (optional), `-Attempt 1-9`
and `-DryRun`. Without `-Risk` they pick the newest `risk_*/refresh.json` whose
`session` matches and that has no `risk_error`, and print the chosen file with its
`risk_latest` and `legacy_score`; if none qualifies they stop with an error. `-DryRun`
prints the full python command, state dir, config fingerprint and risk file and starts
nothing. Both keep the old guards: config checks, refusal over an existing
`runtime.sqlite`, refusal if a process of the same kind is running, hidden window,
`launch.stdout.log` / `launch.stderr.log` / `launcher.pid` in the state dir. Only the
live launcher sets `OPEN_BREAKOUT_LIVE_ACK`, and only for the child process.
Expected fingerprints: shadow `50c1ca8c...48ae4`, live `b226b74f...d678f77` (live was
`c08a3222...bbbf38` until the prior-range filter was added on 2026-09-25; see the last section).

**Monday 2026-09-28 sequence.**

(a) Sunday night or Monday before 08:45 ET, with the Gateway up: read-only preflight.
It needs the live acknowledgement; set it for this one PowerShell window, using the
full account number from the live config, then clear it:

```powershell
$env:OPEN_BREAKOUT_LIVE_ACK = "LIVE 2026-09-28 <the account in the live config>"
python -m open_breakout preflight --config artifacts/open_breakout_runs/config-20260925-live.json --session 2026-09-28 --out artifacts/open_breakout_runs/preflight-20260928-live.json
Remove-Item Env:\OPEN_BREAKOUT_LIVE_ACK
```

`--out` refuses to overwrite an existing file; use a new file name for a second run.
Also confirm the dry runs pick the new refresh:
`powershell -File artifacts/open_breakout_runs/launch-live.ps1 -Session 2026-09-28 -DryRun`
(and the same for `launch-shadow.ps1`).

(b) Start the shadow (before 09:00 ET):

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/open_breakout_runs/launch-shadow.ps1 -Session 2026-09-28
```

(c) Live activation step, run once before 09:25 ET:

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/open_breakout_runs/launch-live.ps1 -Session 2026-09-28
```

A relaunch after an attempt that placed **no orders** uses `-Attempt 2` (state dir
`2026-09-28-live-2`); `live-session` refuses if any `2026-09-28-live*` journal holds orders.

(d) Status:

```powershell
python -m open_breakout status --state artifacts/open_breakout_runs/2026-09-28-shadow/runtime.sqlite
python -m open_breakout status --state artifacts/open_breakout_runs/2026-09-28-live/runtime.sqlite
```

(e) Attended windows: watch TWS and Slack **09:25 to 11:30 ET** and **15:50 to 16:01 ET**.
Graceful stop: create a file named `STOP` in the session state dir (it does not cancel
or flatten anything). Manual rollback is as in the Sep 25 section: kill the PID in
`launcher.pid`, flatten MNQ/MES by hand in TWS and cancel the remaining `OpenBreakout`
orders, then, with the env ack set as in step (a), run the read-only comparison:
`python -m open_breakout reconcile --config artifacts/open_breakout_runs/config-20260925-live.json --state artifacts/open_breakout_runs/2026-09-28-live/trades.sqlite --session 2026-09-28 --client-id 927482`.

## Prior-range skip filter (shipped 2026-09-25 for the live pilot)

Owner decision 2026-09-25: the rule from
`docs/prereg_open_breakout_range_filter_2026-09-25.md` ships in the live one-contract
pilot from Monday 2026-09-28. The shadow session stays unfiltered as the forward-tier
baseline. The prereg's own decision rule and forward tier are unchanged by this; the
shadow journal still supplies the skipped-day outcomes.

**Rule.** Per market, `ratio = prior_tr / atr20`. With the filter enabled in mode
`skip`, a market with `ratio >= threshold` is not armed that session (no opening
capture, no entries, both sides). The other market is unaffected.

**Definitions (match the research, `current_candidate/engine.py:make_sessions` and
`test_simple_filters.make_features`).**
- `prior_tr`: unchanged. Previous cash session's full CME session (18:00 to 17:00 ET),
  raw 1-minute same-contract OHLC from `session_range`,
  `max(high - low, |high - prev_close|, |low - prev_close|)`.
- `atr20`: mean of the 20 most recent valid full-session TRs strictly before the previous
  session (the previous session is excluded), point-in-time at the previous close. All
  CME trade dates count, including shortened holiday sessions such as Labor Day, as in
  the research.
- Roll handling: the research series is Databento's volume-front continuous contract.
  The service rebuilds it from the signal contract and the expiry before it
  (`includeExpired`): the front for UTC calendar date D is the contract with strictly
  greater volume on the second-most-recent CME trade date before D, so the switch lands
  at 00:00 UTC (19:00 or 20:00 ET) inside a session. As in the research there is no tie
  band: a 1% lead decides like a 50% lead, and a lead that moves back to the earlier
  expiry is followed. A session that spans two contracts, or whose contract differs from
  the prior session's, has no TR and is skipped when collecting the 20 (this covers
  both directions of a switch-back). This rule reproduced the research switch time on
  all 10 NQ/ES rolls from Sep 2025 to Jun 2026 (IB trade-date volumes), and the
  service `atr20` equals the research `atr20_before_prior` on every session compared
  across the Sep 2025, Dec 2025 and Mar 2026 rolls (324 NQ/ES sessions, 0 mismatches,
  scratch `roll_history_check.py`, 2026-09-25) and Jun 22 to Aug 28 2026 (parity
  below). For the Sep 2026 roll the volume lead changed on trade date 09-14, so 09-16
  and 09-17 are excluded for both markets.
- Data: IB `TRADES` 1-hour bars, `useRTH=False`, 2 months, both contracts
  (`IBKR.range_history`, four requests at the 09:00 prepare, about 1 to 4 seconds each).
  Each market is fetched separately (200 s cap per market, 450 s overall): one market's
  failure makes only that market UNAVAILABLE. An empty response is retried once after
  2 seconds (ib_insync returns an empty list on its own 45 s timeout). On a restart
  with a retained `inputs.json` nothing is re-fetched; the retained manifest is loaded.
  Hourly bars are used because the session edges (18:00, 17:00) and the roll switch
  (00:00 UTC) are all on the hour; the parity check below shows they reproduce the
  1-minute session TR exactly. The 7-day 1-minute request is unchanged and still sets
  `prior_tr`.
- Known differences from the research, all declared: hourly rather than 1-minute bars;
  IB volumes rather than Databento's for the roll decision, with the lag-2 rule fitted
  to Databento's observed switch times rather than taken from its documentation; only
  two expiries (the signal contract and the one before it), so a later expiry that took
  the volume lead while the service still signals on the earlier one is not seen (the
  reviewed contract roll prevents that); and completeness checks the research does not
  make (below). A CME trade date missing entirely from IB outside the XNYS calendar
  (for example a holiday session) is not detected; the research would have included it.

**Fail-closed.** `prior_range_status: "UNAVAILABLE"` (with `prior_range_reason`) when:
fewer than 20 valid prior TRs; a regular XNYS session (16:00 close) in the span missing,
or missing any expected hourly bar start (18:00 to 16:00 ET, 23 bars; the same
`missing_bars` rule as the 1-minute `session_range` check, whose 16:15 to 16:30 halt
allowance covers no whole hour); an ambiguous roll (a deciding trade date with equal or
zero volumes, or the start of history with no deciding date); an earlier expiry that
last traded inside the window returning no bars; the hourly same-contract TR of the
previous session differing from the 1-minute `prior_tr` by more than one tick; or the
history request failing. XNYS early-close days, holidays with a shortened CME session
(Labor Day, Good Friday, Juneteenth) and non-XNYS days are not completeness-checked and
count with the bars they have, as in the research. One missing hourly bar on a regular
session fails the market closed until that session leaves the span (about a month), so
a recurring `incomplete session` reason is worth checking against TWS. With the filter
enabled an unavailable market does not trade that session (`skip_prior_range: true`).
The service also refuses to arm a market when the filter is enabled and the manifest
has no `OK` decision for it.

**Manifest fields (additive; `prior_tr` and the hashes are unchanged).** Per market:
`prior_range_status`, `atr20`, `ratio`, `skip_prior_range`, `half_prior_range`,
`prior_range_reason`, `atr20_window` (first and last session), `roll_excluded`. Top
level: `prior_range_filter` (the config block or null) and `range_source`. The shadow
records `atr20` and `ratio` and never skips.

**Service behaviour.** A skipped market's state is `phase: SKIPPED`, note
`PRIOR_RANGE_SKIP: ratio ... >= threshold ...` (or the unavailable reason). A
`PRIOR_RANGE_SKIP` event with ratio, atr20, prior_tr and threshold is journaled when
the service is built (09:25 arm time live, 09:00 in shadow), and the process prints
`<market> not armed: ...`. The watchdog does not mark it `MISSED_0930_OPEN`, and a feed
gap on it does not halt the session. The `ARMED LIVE` alert names the skipped market
(`NOT ARMED {...}`); the 09:25 preflight runs as before. If both markets are skipped the
alert reads `ARMED LIVE: NO MARKET (all skipped; ...)`, no order can be sent, and the
run ends `SESSION_COMPLETE` at 16:01 without a halt.
`status` shows the SKIPPED state; the runtime `inputs` record carries a `prior_range`
summary. Protection paths, the 15:56 checks and the other market are unchanged. IOC
entry mode only (main has no bracket mode).

**Config.** Optional top-level key
`"prior_range_filter": {"enabled": bool, "threshold": 1.0 to 3.0, "mode": "skip" | "half"}`.
Absent means disabled and leaves the fingerprint unchanged. `half` (the prereg's A2:
half the computed contracts, floored) is rejected in live mode because half of the
one-contract pilot is zero.
- `config-20260925-live.json`: `{"enabled": true, "threshold": 1.25, "mode": "skip"}`.
  New live fingerprint `b226b74f63fa57874790fcb371c113b5824e1ec25dc7979162982ed26d678f77`
  (was `c08a3222...bbbf38`). `validate` passes; `launch-live.ps1 -DryRun` accepts it.
- `config-20260925-shadow.json`: untouched, no key, fingerprint still `50c1ca8c...48ae4`.

**Parity check (prereg gate 3): PASS.** `artifacts/open_breakout_build/prior_range_parity.py`
(read-only, client 927485) wrote `prior_range_parity_20260925.json`. Against the research
features files (`range_cuts/*_features.csv`):
- August 2026, non-roll sessions: NQ 20 of 20 and ES 20 of 20 equal to the research TR
  (difference 0.00) on both service paths, the hourly continuous series and
  `session_range` on 1-minute bars of the September contract.
- June 22 to July 31: 29 of 29 TRs equal for each market.
- `atr20` versus the research `atr20_before_prior`: 49 sessions per market (June 22 to
  August 28, including windows that span the June roll exclusion), maximum absolute
  difference 0.00.
- Overlapping recent week on the December contract (sessions 09-18 to 09-24): 1-minute
  `session_range` TR equals the hourly TR on 5 of 5 sessions for both markets.
- Re-run 2026-09-25 evening after the roll-rule review (strict volume leader, no 5% tie
  band, switch-backs followed, `missing_bars` completeness): PASS with the same counts;
  every session from June 22 to September 24 yields an `atr20` (none fail closed).

**Monday 2026-09-28 preview** (read-only, taken 17:00 ET Friday through
`build_manifest` with the live config; the runner recomputes at 09:00 ET Monday):

| Market | prior_tr (09-25) | atr20 (08-26 to 09-24) | ratio | Skip at 1.25 |
|---|---|---|---|---|
| NQ | 320.50 | 404.425 | 0.792 | no |
| ES | 66.25 | 68.0625 | 0.973 | no |

Both markets would be armed Monday. The roll sessions 09-16 and 09-17 are excluded from
both windows. Re-run after the roll-rule review: identical values. Hand check: the mean
of the 20 hourly TRs in the parity file for 08-26 to 09-24 (Labor Day 09-07 included,
09-16 and 09-17 excluded) is 404.425 (NQ) and 68.0625 (ES).

## Shared contracts with Legend EMA futures (2026-09-26)

Legend EMA is being cut over to also trade MES and MNQ in Primary (`legend_ema_fut.py`
in trading_ibkr, 1 contract cap per market at cutover, entry at 09:31, flat by the 10:30
time exit and the 10:32 residual check). Open Breakout reconciles the account's whole
MES/MNQ position against its own journal, so the two collide on a day Legend trades:
- The 09:25 preflight passes, because Legend has not entered yet.
- After Legend's 09:31 entry the broker position differs from Open Breakout's journal.
  The watchdog's position check trips after its 3 consecutive stable checks and halts
  new Open Breakout entries for the rest of the session.
- Protection is untouched: any Open Breakout stop and timed exit already working stay
  in place.
- Legend is unaffected, since it nets only executions carrying its own orderRefs
  (`MES|BUY|Legend_EMA|<date>` and its `|TARGET` / `|TIME` legs).

Accepted for the 2026-09-28 pilot. Follow-up: orderRef-scoped reconciliation, so the
watchdog compares Open Breakout's journal to executions tagged `OpenBreakout` rather
than to the account position.
