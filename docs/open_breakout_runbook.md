# NQ / ES opening breakout: IBKR service

Status (2026-09-24, evening): **one shadow session is running for 2026-09-25** on the
local IB Gateway at port 7496 (client 927480), and a **live one-contract pilot
(1 MNQ / 1 MES, client 927481) is prepared but not launched**. Live routing now exists
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
exit with `goodAfterTime=15:55 US/Eastern` is submitted in the same OCA group
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
  exit with `goodAfterTime` 15:55 US/Eastern in the same OCA group (type 2). Every
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
survival of the GTC MKT order with `goodAfterTime` 15:55 US/Eastern; OCA type 2
behaviour with STP plus a timed MKT on CME micros; and the real shape of a stop
rejection (Inactive vs ib_insync's synthesized Cancelled, and which error codes).
