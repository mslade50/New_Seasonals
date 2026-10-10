# NQ / ES opening breakout: IBKR service

**Entry amendment (2026-10-01, effective next launch / 2026-10-02): resting stop-limits.**
The owner requested broker-held entries after three five-MNQ software-triggered IOC
short orders cancelled without fills on October 1 (10:00:52, 10:08:43, 10:08:44 ET).
Live and shadow launcher configs now select `entry_order_type: "stop_limit"`.
Today's processes continue with their loaded IOC configuration; no restart or
additional live order was used to deploy this change. An absent setting or `"ioc"`
retains the legacy path for historical manifests and replay.

After the first 09:30 NQ/ES Last establishes the opening, the service parks eligible
BUY and SELL `STP LMT` entries on the corresponding MNQ/MES contract. Stops are the
tick-rounded opening +/- 25% of prior TR. Limits are the stop plus two ticks for
BUY, minus two ticks for SELL (`max_entry_slippage_ticks`, unchanged). IBKR triggers
on the execution contract's Last, not a software-observed mini crossing. Thus
mini/micro timing may differ even though level calculations retain the mini basis.
Entries are linked in a separate OCA type 1 group (cancel remaining, with block),
and `GTD` at **11:30 ET**, backed by explicit cutoff cancellation. A gap through the
limit leaves a triggered limit working until it fills or expires; it does not burn
repeated IOC retries. Resting means broker-held; exchange versus IBKR simulation
depends on IBKR's handling of the futures order.

Each arming cycle reserves the worse eligible side's risk (one side can fill),
including pending exposure in the pooled cap. At most three cycles per market/day
are submitted. After an exit, a fresh return inside the boundary on mini prices
and executable micro quotes is required before re-arming. Range/short filters,
capital basis, fees, sizing ceilings, margin day caps, protective-stop distance
from the actual fill, and the 15:55 exit are retained. Each partial fill is protected;
the unfilled parent remainder and opposite entry are cancelled. Halt and orderly
shutdown cancel entry orders only; actual-position exits remain. Unconfirmed entry
cancellation halts and alerts. Manual control still transfers the selected market
to Execution/TWS.

Verification: the regression demonstrated zero resting orders before this change.
Lifecycle tests cover pre-cross placement, both sides, quote-vs-Last triggering,
gap/no-fill persistence, partials, risk/margin/short gates, re-entry/attempt limits,
cutoff/halt/shutdown cancellation and uncertainty, and a complete mocked live day
through preflight, entries, protection and timed exits. At 10:45 ET on October 1,
read-only IBKR what-if previews accepted BUY/SELL MNQ and MES `STP LMT` + GTD +
OCA type 1 without warnings; execution-contract Last feeds were observed and the
working-order set was unchanged. Evidence: `artifacts/open_breakout_execution/broker_preview.json`.
This preview does not verify a live stop-limit fill. The next session's broker journal is
the acceptance check for real trigger, fill, OCA and expiry behavior.

**Live execution test (2026-10-01, owner-authorized outside the entry window).**
An isolated one-MNQ test reused the production fill, protective-stop and exit
lifecycle, with a supplied SELL trigger five points below live MNQ Last and an
exit five minutes after submission. At 11:56:56 ET, Last was 30,611; IBKR accepted
stop 30,606 / limit 30,605.50. It filled at 30,605.75 at 11:57:00, then created a
one-contract BUY stop at 30,704.25 and a BUY market exit held until 12:01:56, linked
by OCA type 2. The timed exit filled at 30,612.75 at 12:01:56, cancelling the stop.
Independent broker readback confirmed MNQ flat, both executions and no test orders.
Evidence: `artifacts/mnq_strategy_fill_test/receipt.json`. This verifies live
stop-limit filling, fill-created protection, broker-held timed closing and exit
sibling cancellation; normal opening-range arming and entry OCA remain next-session checks.

The test exposed an execution-before-status race: cancellation of the already
filled parent returned IBKR 10148 and unnecessarily halted the controller. Both
broker-held exits survived the disconnect and closed correctly. The adapter now
skips cancellation when the latest execution's cumulative quantity confirms a
full fill, and logs 10148 without halting only for a verified fully executed order.
Partial or unknown orders retain cancellation and error handling. Four regression
cases failed before the repair; the focused suites pass 240 tests with one skip.
The repair loads at the next strategy launch; existing session processes retain
their loaded code and configuration.

Shadow/replay stop-limits trigger from `execution_trade` captures and fill only
inside their limit. Micro Last uses normal streaming market-data updates to avoid
extra tick-by-tick subscriptions; the shadow is approximate and has no liquidity
or queue model. Existing captures without execution Last cannot establish native
stop-limit outcomes. Original October 1 configs are retained beside that session's
journals as `config.original.json` for reconciliation with the original fingerprints.

Status (2026-09-28 evening): **live at normal sizing from Tuesday 2026-09-29** (owner
decision 2026-09-28). Monday 2026-09-28 was the first live session at one contract per
market (1 MNQ / 1 MES, both shorts stopped, mechanics verified). From 2026-09-29 the live
session (client 927481, launched 08:12 ET by Task Scheduler) uses the regular risk sizing
(15 bp NQ / 10 bp ES of the $750,000 basis, 25 bp open / 75 bp daily caps) with **no
per-market contract cap** (owner decision, later the same evening): the configs carry
`max_contracts: 60` as a fat-finger ceiling under the code ceiling
`LIVE_HARD_MAX_CONTRACTS = 60`, and the 09:25 preflight margin-gates the planned sizes,
sizing DOWN proportionally (margin day cap) rather than refusing to arm. See "Live sizing
(from 2026-09-29)" and its amendments below. Live routing exists only behind layered
guards: `mode: live` + `allow_live` + a non-DU account + every `max_contracts <= 60` (and
the pilot ceiling, when present, in 1..60) + the exact session/account acknowledgement
`OPEN_BREAKOUT_LIVE_ACK`, and only through the `live-session` command after a passing
09:25 ET preflight. `shadow-session` still refuses live configs.

## Frozen candidate

- Level inputs: NQ and ES mini futures; execution: same expiry NQ/MNQ and ES/MES.
- Open: first live Last print at 09:30 ET, arriving within two seconds by default.
- Entry thresholds: open +/- 25% of the previous full CME session true range.
  Range is raw same-contract 18:00-17:00 ET OHLC with the preceding session close.
- Initial stop: 25% of that true range from the actual average fill, rounded away
  from the position to a valid tick. Partial fills update the average and stop.
- No break-even, target, trailing stop or long filter. Short signals require the
  **previous cash-session legacy 63d risk score, smoothed over 10 observations,
  to be >=20**. The newer dashboard `main_score` is intentionally not selected.
- Entry cutoff 11:30 ET exclusive; 15:55 ET timed market exit; one position and at
  most three entry cycles per market/day; fresh return inside before re-arming.
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
Sheets, the private site, or the OneDrive broker installation. Since 2026-09-27
ownership is the `orderRef`: an order or execution is Open Breakout's if and only if
its 3rd pipe field is `OpenBreakout`. Other strategies (Legend EMA futures) may hold
positions and working orders in the same MES/MNQ conIds in the same account; they are
logged as `other_book` and never halt, skip or cancel anything. See "Shared contracts:
orderRef ownership (2026-09-27)" at the end. This release does not integrate the stock
book's account-wide risk limits.

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

Entries are broker-held stop-limits (see the October 1 amendment above). The legacy
`entry_order_type: "ioc"` path uses marketable IOC limits beyond current bid/ask.
Both paths bound the entry price and can produce partial/no fills.
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


## Live sizing (from 2026-09-29)

Owner decision (2026-09-28 evening): from Tuesday 2026-09-29 the live book is sized
**normally**, no longer a one-contract pilot. Everything else in the live section below
(protection, flatten, preflight gates, alerts, attendance) is unchanged.

**Amendment, 2026-09-28 late evening (owner decision): no per-market contract cap.** The
10-contract cap is removed before the first normal-size session. Size is set by the risk
budgets (15 bp NQ / 10 bp ES of $750k), the 25 bp open-risk and 75 bp daily caps, and the
09:25 planned-size margin gate. What remains is a fat-finger ceiling:
`LIVE_HARD_MAX_CONTRACTS = 60` in `open_breakout/config.py`, and `max_contracts: 60` for both
markets in `config-20260925-live.json` (the `pilot` block is removed; it stays optional in
code, 1..60 when present). The shadow config also carries 60/60 so its simulated sizes follow
the live rule. `IBKR.send()` still refuses any live quantity outside 1..effective cap (60).
Launchers: `launch-live.ps1` and `launch-shadow.ps1` accept `max_contracts` 1..60 and print
them; `launch-live.ps1` no longer requires a pilot block.

- **Where 60 sits:** at the calmest prior ranges of the last 12 months (NQ TR ~100, ES
  ~20) the budget sizes 20 MNQ ($53.70 per contract) and 23 MES ($31.70), so 60 does not
  bind at today's index levels. In the full 2018-2026 replay it bound on 12 of 3,238 trades,
  all MNQ on 2018-2020 holiday-adjacent sessions (NQ prior TR 25.5-29, budget 61-67) when
  NQ traded at a third of today's level. MES never exceeded 44.
- **Connect-time what-if (~08:15 and daily-launch step 6):** it previews a fixed
  **reference size of 10 per market** (`standby.REFERENCE_WHATIF_CONTRACTS`, clamped to the
  effective cap), `what_if_basis: reference`, warnings `MARGIN_AT_REFERENCE:...` /
  `MARGIN_TOTAL_AT_REFERENCE:...`, never a failure. The prior TR is not available then (it
  comes from the 09:00 history pull, which can take minutes), and a what-if at 60 would mean
  nothing. The 09:25 planned-size gate is unchanged and is the real guard.
- **The 09:25 margin gate is now the only thing between a tight-range day and a large
  order** (live runs no per-entry what-if). A failure halts the whole session
  (`HALTED_PREFLIGHT`, both markets, no retry); it fails closed. At the 2026-09-28 evening
  per-contract margins (MNQ $6,728, MES $3,486 on the BUY side) and the $112,703 limit, 16
  MNQ alone is the most that fits. At NQ TR 100 / ES TR 20 the plan is 20 MNQ + 23 MES,
  about $134,567 + $80,187 = **$214,754, well over the limit: the session would not arm.**
  (Superseded the same night: the margin day cap below sizes such a day down to 10 / 12.)
  Replaying the last 12 months of prior ranges at those margins, about 10 of 237 sessions
  (4%) would have failed the gate (2026 to date: 1 of 161), mostly holiday-adjacent quiet
  days. Margins scale with price and volatility, so this is an estimate.
- **Tuesday 2026-09-29 is unchanged:** at NQ prior TR 565 / ES 79.75 the plan is 3 MNQ /
  7 MES (the 10 cap did not bind there), planned margin about $44,589 of $112,703.
- **Fingerprints from 2026-09-29 (after this amendment):** live
  `0d0876ae95114bee96b49973615d0780cd123bdca881e98eb54dce3be88fdce8` (was `44b60282...ce952`);
  shadow `fb28fcec5128e0b825866b868e4dcb2a2f707479f638afaa4b1529b4dd15c97a` (was
  `50c1ca8c...48ae4`; the shadow change is accepted, it only makes simulated sizes match live).
- **Research replay** (`scripts/build_intraday_replay.py`, cap 60, open/daily caps kept):
  2018-01..2026-08 sum $541,441 (was $435,263 at cap 10), day Sharpe 1.64 (1.65), max DD
  -$33,244 (-$33,221), 374.3 trades/yr, last 12 months $30,590 ($30,466). Average contracts
  12.9 MNQ / 13.0 MES; sizes p50/p90/max 10/26/60 MNQ and 11/23/44 MES. Most of the gain is
  2018-2020, when ranges in points were small; the last 12 months barely change.

- **Rollback:** set both `max_contracts` back to 10 (or 1) in `config-20260925-live.json`,
  or add `"pilot": {"max_contracts_per_market": N}`; either changes only the live fingerprint.

**Amendment, 2026-09-28 night (owner decision): margin day cap, size down instead of fail.**
The 09:25 planned-margin gate no longer halts the session when the planned sizes do not fit.
It sizes the session down. This supersedes "the session would not arm" in the bullets above.

- **Rule** (`standby.preflight` -> `apply_margin_day_cap` / `margin_day_caps`): after the
  planned-size what-ifs, if the summed worse-side initial margin exceeds the limit
  (`max_margin_fraction` x min(ExcessLiquidity, NetLiquidation)), each market gets a
  whole-contract `margin_day_cap`. The limit is split in proportion to each market's planned
  margin, so both scale by the same factor limit / total: cap = floor(planned x factor). A
  market that floors to 0 gets 1 if its one-contract what-if fits beside the other market's
  capped margin, else 0: it is not armed (`MARGIN_DAY_CAP_ZERO`) and the other market trades.
  The capped sizes are then what-iffed again and must fit. If they do not (margins are not
  linear), one contract comes off one market at a time, alternating, at most 10 times; after
  that, or when nothing is left to reduce, `MARGIN_TOTAL:<total>><limit>` fails closed as before.
  A single planned side over the limit is part of the same size-down (no longer a
  `MARGIN_PLANNED` failure); a warning, an unusable or a negative what-if still fails.
- **Entries:** `Service.enter` clamps each entry to the market's cap for the whole session,
  after the risk budget, the open/daily caps and the effective ceiling, before `send()` (which
  still refuses anything outside 1..60). It applies to both sides and all three attempts. The
  cap is per session: every arming recomputes it; no cap needed means `None` and no change.
- **Records:** the preflight report and runtime meta (`preflight_arm`) carry `margin_day_cap`
  (None or per symbol), `planned_qty`, `capped_qty`, `margin_total_planned`,
  `margin_total_capped` (and `margin_total`, the total at the sizes that will trade); runtime
  meta `margin_day_cap` holds the caps per market name. Journal events: `MARGIN_DAY_CAP` at
  arming (caps, sizes, totals, limit), `MARGIN_DAY_CAP_ZERO` for an unarmed market,
  `SIZED_DOWN_MARGIN` on each entry the cap reduces (market, side, attempt, planned, qty).
  The ARMED alert reads e.g. `planned 20 MNQ / 23 MES, margin-capped to 10 / 12 at today's
  ranges, max 60/60; margin $109,120 of limit $112,703 (uncapped $214,754)`; with no cap it
  is unchanged (`planned 3 MNQ / 7 MES ... planned margin $44,589 of limit $112,703`).
- **Calm-range example** (NQ TR 100 / ES TR 20, the 2026-09-28 evening BUY margins MNQ
  $6,728.37 / MES $3,486.38, limit $112,703.36): planned 20 MNQ + 23 MES = $134,567 + $80,187 =
  $214,754; factor 0.5248; caps floor(10.50) = **10 MNQ** and floor(12.07) = **12 MES**:
  $67,284 + $41,837 = **$109,120**, inside the limit. Both markets trade at about half size
  instead of the session not arming.
- **Tuesday 2026-09-29 is unaffected:** 3 MNQ / 7 MES plan about $44,589 of $112,703, so no
  cap. The connect-time reference what-if (10/10, warnings only) is unchanged and never
  computes a day cap; only the 09:25 arming (or a reconnect after 09:00) does.
- Tests: `test_margin_day_cap_*` and `test_live_session_arms_after_preflight_and_trades[margin_capped]`
  in `tests/test_open_breakout.py`.

The bullets below describe the 10-cap state of 2026-09-28 evening and are kept as history
where they conflict with the amendment above.

- **Sizing:** the regular whole-contract risk sizing, the same code path as the shadow:
  per-contract risk = |limit - stop| x multiplier + 2 x $0.85 fees + a 4-tick exit reserve
  (MNQ $2/pt, MES $5/pt, tick 0.25; the stop distance rounds outward to the tick), and
  qty = floor(budget / per-contract risk) with budget 15 bp (NQ) / 10 bp (ES) of the
  $750,000 basis. The 25 bp open-risk and 75 bp daily-reserved caps then shrink an
  entry to the whole contracts that still fit, using the same per-contract risk
  (event `SIZED_DOWN_RISK_CAP`), and skip it only when not one contract fits
  (`SKIP_RISK_CAP`). A computed 0 is still no trade. At 15/10 bp the two budgets sum
  to exactly the 25 bp open cap and three attempts each to the 75 bp daily cap, so
  neither cap binds with this config (changed 2026-09-28 night from skip-only).
- **Caps:** after that sizing, live clamps to the effective ceiling
  `min(market max_contracts, pilot.max_contracts_per_market if present, LIVE_HARD_MAX_CONTRACTS)`
  = min(10, 10, 20) = **10 per market**. `IBKR.send()` refuses any live order (entry,
  stop, timed exit, flatten) whose quantity is not a whole number in 1..effective cap.
  Config validation refuses a live config with any `max_contracts` above 20 or a pilot
  ceiling outside 1..20. The `pilot` block is optional in code; the live config keeps it
  (value 10) because `launch-live.ps1` requires it (1..20) and prints the effective caps.
- **Preflight margin** (revised 2026-09-28 night; the limit is 20% of
  min(ExcessLiquidity, NetLiquidation)):
  - Always: a one-contract BUY and SELL what-if per market (contract tradeable, no
    warning, within the limit). Failure `MARGIN:<sym>:<side>:...` blocks.
  - **09:25 arming preflight: the PLANNED sizes.** `standby.planned_sizes()` runs the
    real `size_order` for each market and side with the manifest's prior TR, the current
    mid quote, the configured risk bp, fees and exit reserve, clamped to the effective
    cap (a market the prior-range filter does not arm plans 0). Each planned side is
    previewed (`MARGIN_PLANNED:...` if unusable or over the limit) and the worse side of
    each market, summed, must fit the limit, else `MARGIN_TOTAL:<total>><limit>` and no
    arming. A missing quote fails closed (`PLANNED_SIZE:...`). The report carries
    `what_if_basis: planned`, `planned`, `planned_qty`, `what_if_qty`, `margin_total`,
    `margin_limit`; the ARMED alert reads e.g. `planned 3 MNQ / 7 MES at today's ranges,
    max 10/10; planned margin $44,589 of limit $112,703`. A reconnect after 09:00 (the
    manifest exists) uses the planned sizes too.
  - **Connect (~08:15) and the 08:12 daily-launch step 6, before the 09:00 manifest:**
    previews at the effective cap (10/10) for **information only**:
    `what_if_basis: cap`, and any over-limit result is a `warnings` entry
    (`MARGIN_TOTAL_AT_CAP:...`, `MARGIN_AT_CAP:...`), never a failure.
  - The per-entry what-if stays skipped in live.
- **Tuesday 2026-09-29 exposure at the prior ranges** (NQ prior TR 565.0, ES 79.75; the
  runner re-fetches TR at 09:00): NQ stop 0.25 x 565 = 141.25 pts, MNQ per-contract
  risk 282.50 + 1.70 + 2.00 = **$286.20**, budget $1,125 -> **3 MNQ** ($858.60). ES stop
  0.25 x 79.75 = 19.9375 pts, rounded outward to 20.00, MES per-contract risk 100.00 +
  1.70 + 5.00 = **$106.70**, budget $750 -> **7 MES** ($746.90). Both open at once:
  $1,605.50, inside the 25 bp open cap ($1,875). Three losing attempts each:
  $4,816.50, inside the 75 bp daily cap ($5,625). Neither market reaches the 10 cap at
  these ranges (for reference, Monday's shadow at TR 320.5 / 66.25 sized 6 MNQ / 8 MES).
  Stops are market orders once triggered; gaps can exceed these figures.
- **Preflight margin at max (dry run 2026-09-28 19:48 ET, client 927485, overnight
  margins):** 10 MNQ BUY $67,284 / SELL $61,167; 10 MES BUY $34,864 / SELL $28,606;
  summed worse sides **$102,147 against a limit of $112,703** (20% of ExcessLiquidity
  $563,517; NLV $613,580). That thin headroom is why the gate moved to the planned
  sizes: at Tuesday's 3 MNQ / 7 MES the same per-contract margins sum to about
  $44,589 (3 x $6,728 + 7 x $3,486), roughly 40% of the limit. The at-cap total is now
  only a connect-time warning.
- **Partial fills at size** (tests in `tests/test_open_breakout.py`): an IOC that fills
  4 of 6 gets a stop and a 15:55 timed exit for 4; a second partial on the same order
  re-averages the entry, moves the stop off the new average and resizes the same stop
  order in place; a zero fill consumes the attempt and keeps the daily reservation; an
  emergency flatten sizes to the journal quantity; a partial stop fill (OCA type 2
  reduces the timed exit at IB) reduces the journal, cancels nothing and reconciles;
  any quantity still held at 15:56 halts with an alert. Added 2026-09-28 night: a
  rejected in-place stop modify halts and alerts (explicit reject code: emergency
  flatten of the journal quantity; otherwise no flatten, loud hand-check alert); an
  entry execution reported after the 15:55 exit was sent resizes that exit in place and
  halts for review; a working timed exit larger than the journal position (OCA reduce
  missing) is a reconcile discrepancy that halts after three checks.
- Rollback to one contract: set `pilot.max_contracts_per_market` (and both
  `max_contracts`) back to 1 in `config-20260925-live.json`; that changes the live
  fingerprint and nothing else. Live fingerprint from 2026-09-29:
  `44b60282af425838dea8563c0b1fcf10bb55f9c6dd44790c54fcc011f10ce952` (was
  `86e8d186...4566`); shadow unchanged `50c1ca8c...48ae4`.

## Live pilot: 2026-09-25 session

The one-contract cap below applied 2026-09-25 to 2026-09-28; see "Live sizing (from
2026-09-29)" above for the current caps and preflight.

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
  Open Breakout position in MNQ/MES (from executions tagged `OpenBreakout`); no working
  MNQ/MES order carrying an `OpenBreakout` ref from any API client or TWS
  (`reqAllOpenOrders`). Since 2026-09-27 other strategies' MNQ/MES positions and working
  orders are reported under `other_book` and do not fail it. Also checked: contract identities; what-if margin for 1-lot (from 2026-09-29 also the planned sizes, summed, at 09:25; at-cap warning at connect; see Live sizing) BUY and SELL in
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
  takes a fresh snapshot, and only if Open Breakout's own executions (orderRef) still
  equal the journal position and the account ceiling holds (see the 2026-09-27 section)
  sends **one** market flatten (DAY), sized from the journal, never from the account
  position. Unconfirmed cancels or a position
  mismatch abort the flatten with a "FLATTEN BY HAND" alert. It never sends a second
  flatten. Warning codes (2104, 2106, 2108, 2109, 2158, 399, 404) never flatten. An
  uncertain acknowledgement halts new entries only.
- Other live safeguards: a stale or dislocated execution quote at entry time skips
  that one signal (no halt, no attempt consumed; a re-entry needs a fresh recross);
  the per-entry what-if is skipped in live (margin was checked at connect and at the planned sizes at 09:25);
  an entry execution arriving after the IOC looked finished is recorded, protected with
  a stop and timed exit, then the session halts. The watchdog halts on an own-execution/
  journal position, account-ceiling or own-stop discrepancy only after **3 consecutive**
  stable checks (orderRef-scoped since 2026-09-27), and on a
  signal stream silent for `watchdog_stale_seconds` (default 30, optional config key);
  the 3-second freshness still applies at entry. Market-data farm messages 2103/2105
  are warnings. From 15:56, any journal position or any working `OpenBreakout` order on
  MNQ/MES halts with an alert.
- Alerts: console only (the session's `launch.stdout.log` carries every `ALERT` line:
  new halt reasons, preflight failure, arming, entry fills, flatten, orphaned exit,
  process exit). Slack posting is OFF for this strategy by owner decision (2026-09-28,
  after the first live session posted four messages to the shared alerts channel). It
  is opt-in only: setting `OPEN_BREAKOUT_SLACK=1` in the process environment re-enables
  posting to the repo's `SLACK_WEBHOOK_URL` on background threads that never block the
  order path; the launcher does not set it.
- **Attended session:** someone must watch TWS from **09:25 to 11:30 ET** (arming
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
half the computed contracts, floored) is rejected in live mode (originally because half of
the one-contract pilot is zero; still rejected at normal sizing from 2026-09-29, since only
the skip variant is approved for live).
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

### 2026-09-28 amendment: skip only after a big win

Owner decision 2026-09-28 evening, live from session 2026-09-29. The prior-range skip now
applies only when the market's own previous session was also a big win.

**Rule.** With the filter enabled and `require_prior_big_win: true`, per market:
`skip_prior_range = (prior_range_status == OK and ratio >= threshold) and
(prior_day_r is not null and prior_day_r >= big_win_r)`. The other market is unaffected.

**Definitions.**
- `prior_day_r`: the market's summed R over the previous XNYS session's closed trades
  for the UNFILTERED strategy, read from the shadow journal (research row (v), see below).
  Per trade, `R = side x (avg exit fill - avg entry fill) / |avg entry fill - stop|` per
  contract, with fill prices (never marks) and the last journaled protective stop for that
  attempt. The service moves the stop only when a partial entry fill changes the average
  entry, so the last journaled stop is the stop for the whole filled entry. An open
  position at 15:55 exits through its TIME fill and counts like any other exit. Several
  attempts in a day are summed. Fees are not in this R.
- Source order (`prior_day_source`):
  1. The shadow journal, `<runs root>/<previous session>-shadow[-N]/trades.sqlite`, the
     highest attempt N that has a journal: `shadow_runtime_meta` (its `day_r` summary) or
     `shadow_journal` (recomputed from its fills and orders).
  2. Live only, when the shadow journal is missing, has no fills (halted or failed before
     trading) or is unreadable: the live journal, `<previous session>-live[-N]`, as
     `own_runtime_meta` or `own_journal`. The note records why the shadow was passed over.
  3. Neither usable: null.
  A shadow journal with fills is used even when the shadow halted mid-session; a market
  whose shadow position the journal never closed is null (no fallback for that market).
  The shadow session reads its own (shadow) journal only.
- Why the shadow first: a live session the filter skipped has no trades, so its own
  journal could never show the big win that arms the next day's skip. That is row (v-r),
  the weaker realized-prior variant. The owner chose row (v).
- Each session writes a `day_r` summary per market into its runtime meta when it
  finishes (SESSION_COMPLETE, halt or any exit). The write never raises, is skipped when
  the journal was never opened, and never replaces a recorded non-null R with a null.
  The next prepare prefers that summary and falls back to recomputing from the journal
  when it is absent (as for every session before 2026-09-29). Journals are opened with a
  read-only SQLite URI and a 1 s busy timeout; the lookup reads two local files and makes
  no network call.
- Big win: `prior_day_r >= big_win_r` (+2.0R).

**Fail open on a missing prior-day R.** No shadow or live journal, prior sessions that
halted or failed with no fills, no trades, a trade the journal never closed, a journal
for another day, an unreadable one, or an unknown previous session: `prior_day_r` is
null, `prior_big_win` is null, and the big-win leg is false. A missing prior-day R never
causes a skip.

**Fail-closed ratio path unchanged.** `prior_range_status: UNAVAILABLE` still skips the
market when the filter is enabled, whatever the prior-day R. That path is a data problem
in the range computation, not a big-win question.

**Manifest and runtime.** Per market, new fields `prior_day_r`, `prior_day_source`
(`shadow_runtime_meta`, `shadow_journal`, `own_runtime_meta`, `own_journal` or null;
`load_manifest` rejects any other value and a non-null R without a source),
`prior_day_note` (session dir and trade count) and `prior_big_win` (bool or null). `prior_range_reason` now names the deciding leg (range leg, big-win leg, or
fail open). The manifest's `prior_range_filter` copy carries the two new keys only when
the big-win leg is on. `load_manifest` checks that `prior_big_win` follows `prior_day_r`
and that the skip flag follows both legs. The shadow (no filter) records every field and
never skips. The `PRIOR_RANGE_SKIP` event carries ratio, prior_day_r, prior_day_source,
prior_big_win, threshold and big_win_r; the `ARMED LIVE` alert and the shadow armed line
print ratio, prior-day R and its source per market and the thresholds in force. `status` on a state dir with an
`inputs.json` prints a `prior_range` block; `prepare` prints the new fields and takes
`--runs-root` (default `artifacts/open_breakout_runs`) and `--client-id`.

**Config.** Optional keys in the `prior_range_filter` block:
`"require_prior_big_win": bool` (default false) and `"big_win_r": number in [0.5, 5]`
(default 2.0). Absent keys leave the rule and the fingerprint unchanged.
- `config-20260925-live.json`: `{"enabled": true, "threshold": 1.25, "mode": "skip",
  "require_prior_big_win": true, "big_win_r": 2.0}`. New live fingerprint
  `86e8d186c67e8b3944adb15863fed01488441648de845700199b50aefe5f4566` (was `b226b74f...8f77`).
  `validate` passes; `launch-live.ps1 -DryRun -Session 2026-09-29` accepts it.
- `config-20260925-shadow.json`: untouched, fingerprint still `50c1ca8c...48ae4`.

**Research row.** `artifacts/research/qqq_open_breakout_20260923/autocorr/lag1_rule/composite/overlap_walkforward/filter_vs_skip/RESULTS.md`,
variant (v) "range skip only after big win": +35.8R [+2.1, +65.4] against -3.6R for the
plain range skip, Sharpe 1.54, max DD 39.8R, 374 trades a year, positive in 8 of 9 years.
All in-sample. Row (v) uses the shadow prior (what the strategy did even on a skipped
day), and that is what the live runner implements by reading the shadow journal first.
Reading the live journal alone would be the realized prior, row (v-r) in the same table:
+25.1R [-6.1, +52.4], Sharpe 1.51, max DD 39.7R, 7 of 9 years. The difference: a live
session the filter skipped has no trades, so it can never be the big win that arms the
next day's skip. The live journal is only a fallback when the shadow journal is unusable.

**Tuesday 2026-09-29 preview** (read-only, client 927485, taken 2026-09-28 evening with
the shadow-first source order; the runner recomputes at 09:00 ET):

| Market | prior_tr (09-28) | atr20 (08-27 to 09-25) | ratio | prior_day_r (09-28 shadow; live) | Source | Skip under the amendment | Skip under the plain rule |
|---|---|---|---|---|---|---|---|
| NQ | 565.00 | 403.3625 | 1.401 | -1.00; -0.99 (short stopped in both) | shadow_journal | no | yes |
| ES | 79.75 | 69.40 | 1.149 | -1.00; -1.00 (short stopped in both) | shadow_journal | no | no |

The NQ values differ by the fill prices: live filled at 30637.75 and stopped at 30717.25
against a 30718.00 stop; the shadow filled at 30638.25 and stopped at 30718.50.

Both markets would be armed Tuesday. NQ is the first session where the amendment changes
the outcome: the prior session was wide but a loss.

## Shared contracts: orderRef ownership (2026-09-27)

Owner decision 2026-09-27. This replaces the 2026-09-26 note, which accepted that a
Legend EMA entry would halt Open Breakout for the rest of the session. Legend EMA
futures (`legend_ema_fut.py` in trading_ibkr; refs `MES|BUY|Legend_EMA|<date>` with its
`|TARGET` and `|TIME` legs; 1 contract per market, entry 09:31, flat by the 10:30 time
exit) trades the same MES/MNQ contracts in the same account. Open Breakout never halts,
skips, cancels or refuses because of another strategy's position or working orders.
Legend is unaffected; it already nets only its own refs.

**Rule.** An order or execution is Open Breakout's if and only if its orderRef 3rd pipe
field is `OpenBreakout` (`service.own_ref`, the same field `daily_execution_report.parse_ref`
reads). Nothing else in the account is Open Breakout's business, from any client.

**Own position.** `IBKR.own_position(conId)` is the signed sum of the day's executions
whose orderRef is `OpenBreakout` (`reqExecutions` with `ExecutionFilter(acctCode=account)`,
no client-id filter, so a previous process's executions with the same ref still count).
Only executions time-stamped since midnight New York count (TWS can return up to seven
days). It is cached for at most one second under the snapshot lock; an own fill clears
the cache, and an answer that was in flight when an own fill landed is used once and
never cached. An IB execution correction replaces the original by execId stem.
`reqPositions` stays in the snapshot for the `other_book` view and the account ceiling.
ib_insync suppresses the live fill event for an execId that a `reqExecutions` answer
delivered first; the transport therefore re-delivers any own-client execution it has
not yet handed to the journal (the journal dedups by execId). A fill caught this way
after the entry IOC looked terminal takes the late-entry path: stop sent, 15:55 exit
sent, session halted.

**Own position UNKNOWN.** The own position is trusted only when the executions request
succeeded (no exception, no timeout; 3 s inside the 5 s snapshot) and, once the journal
holds fills, the answer contains every journaled execId (by stem). Otherwise it is
UNKNOWN for that watchdog cycle: the own-mismatch and ceiling checks are skipped (the
mismatch counter neither grows nor resets), `foreign_base` is not captured, and the
stop/order checks still run. The first unknown cycle writes `OWN_POSITION_UNKNOWN_START`;
after 30 s of consecutive unknown cycles one loud alert and `OWN_POSITION_UNKNOWN` (not a
halt); recovery writes `OWN_POSITION_KNOWN`. The heartbeat carries `own_known`. The
emergency flatten never fires on UNKNOWN: it halts with
`EMERGENCY_FLATTEN_SKIPPED_OWN_POSITION_UNKNOWN` and alerts FLATTEN BY HAND. Preflight
retries the executions read 3 times; if it is still unreadable, preflight fails closed
with `OWN_POSITION_UNKNOWN:...` (flat cannot be verified; this is before any order).

**Account ceiling: alert-only by default.** A breach is the account holding less than
Open Breakout's own position in its direction. The account figure is taken net of other
strategies' attributed executions that day and of the non-Open Breakout position
recorded at the first watchdog where Open Breakout is verifiably flat in that contract
(live: the 09:25 arming check; `foreign_base` in the `trades.sqlite` meta). On 3
consecutive stable breaches the service writes `POSITION_CEILING_WARNING` (symbol, own,
account, foreign, foreign_base) and sends one loud alert per episode. It does not halt,
cancel or flatten, and it does not block an emergency flatten (which warns and proceeds).
Reason: if `reqExecutions` does not show Legend's executions to 927481, Legend long 1
against our short 1 looks exactly like a hand flatten. Config key `ceiling_halts`
(JSON boolean, default false; absent key keeps the fingerprint) restores the old
behaviour: the breach halts the watchdog after 3 checks and blocks the emergency flatten.

**What still halts.**
- Own mismatch: trusted `OpenBreakout` executions differ from the journal on 3
  consecutive stable watchdog checks.
- Own stray orders: a working order with an `OpenBreakout` ref on MNQ/MES that is not in
  this journal, or comes from another client id, halts at once and fails preflight.
- Own protective stop missing or different (only `OpenBreakout`-ref orders are read).
- From 15:56: a journal position, or any working `OpenBreakout`-ref order on MNQ/MES
  (matched on the 3rd ref field, so the `|NQ-1-TIME` suffix legs match).
- Restart with orders in the journal, the duplicate-process lock and the relaunch
  guard: unchanged.
- Emergency flatten: sent only if trusted own executions equal the journal AND the
  account ceiling holds. A ceiling breach at flatten time can mean the account no
  longer holds our position (a hand flatten), so a speculative flatten could reverse
  it; the service then halts with FLATTEN BY HAND instead of sending an order. This
  is deliberate and independent of `ceiling_halts`. The quantity, when sent, is the
  journal quantity, never the account position.

**What is now ignored (logged only).** Other strategies' MNQ/MES positions (account
minus own) and working orders. The watchdog prints one `Other book in execution
contracts (informational, not OpenBreakout)` line per change and stores the latest view
as `other_book` in the `trades.sqlite` meta. No alert, no journal event, no halt.
Preflight reports `own_positions`, `working_orders` (only `OpenBreakout` refs) and
`other_book` (`account_position`, `other_position`, `working_orders` per symbol); the live
runtime meta `preflight_connect` / `preflight_arm` carries `other_book` too. The shadow
records `other_book_connect` and `other_book_arm` in its `runtime.sqlite`. Preflight
field changes: `positions` became `own_positions`; the failure `NONZERO_POSITION` became
`OWN_POSITION`.

**Limits, unverified.**
- Whether `reqExecutions` returns other clients' executions to a non-master API client
  is not verified: the 2026-09-27 read-only check (a Sunday) saw no executions at all.
  The live process (client 927481) always sees its own. A read-only `preflight` or
  `reconcile` on another client id (927482/927485) may not see 927481's executions and
  would then report an own position of 0.
- If Legend's executions are not visible to 927481, Legend long 1 while Open Breakout is
  short 1 nets the account to 0 and raises a `POSITION_CEILING_WARNING` alert (no halt
  with the default `ceiling_halts: false`). **To confirm on the first shared day**
  (Monday 2026-09-28, after Legend's 09:31 entry): run read-only
  `reconcile --client-id 927482` (the 2026-09-28 command in the Monday staging section
  above); its `executions` list now
  shows `client_id`, `side`, `ref` and `time`, and `broker.foreign` the attributed other
  book. If Legend's `Legend_EMA` executions (its client id) and 927481's `OpenBreakout`
  executions both appear, cross-client visibility holds and the ceiling is accurate;
  record the answer here. If they do not, keep `ceiling_halts` false.
- An MNQ/MES trade without a strategy ref (manual TWS) that moves against an open Open
  Breakout position raises the ceiling warning after 3 checks (a halt only with
  `ceiling_halts: true`).
- Execution history covers the Gateway's window (since midnight by default). Open
  Breakout is flat by 15:55 by design, and the 15:56 check covers the same day.

**Parked `bracket-entry-mode` branch.** It still has the old exclusivity checks in its
bracket paths (whole-account position reconciliation and all-order ownership on the
execution conIds). It must adopt this orderRef scoping before it is merged.

Guard tests: `tests/test_open_breakout.py`, section "Shared contracts: orderRef
ownership".

## Session 2026-09-28 outcome (first live session)

Launched 08:12 ET by the assistant: shadow PID 1480 (client 927480), live PID 28108
(client 927481), both from this checkout at commit 0722f978 plus the untracked
config. Preflight passed at connect and at 09:25. Prior TR NQ 320.5 / ES 66.25,
ATR20 404.4 / 68.1, ratios 0.79 / 0.97, so the prior-range skip did not apply.
Legacy score 76.78, short gate open. Both sessions reached SESSION_COMPLETE; the
processes exited on their own after 16:01. No halts, no skips, no stray orders.

| Market | Open | Entry | Stop | Exit | Points | 1 contract | Shadow full size |
|---|---|---|---|---|---|---|---|
| NQ short 1 MNQ | 30719.25 | 30637.75 at 09:35:45 | 30718.00 | 30717.25 at 12:27 (stop) | -79.5 | -$160 | 6 MNQ at 30638.25, stop 30718.5, about -$960 |
| ES short 1 MES | 7772.50 | 7756.00 at 10:38:17 | 7772.75 | 7772.75 at 12:26 (stop) | -16.75 | -$84 | 8 MES at 7755.75, stop 7772.5, about -$670 |

One attempt per market; the entry window closed at 11:30 before either stop-out, so
no re-entry. Day P&L at one contract each about -$245 including four commissions
of $0.61. Shadow and live took the same two trades; live fills were a tick better
on NQ and within a tick on ES. A read-only reconcile from client 927482 after the
close: MNQ 0, MES 0, no working OpenBreakout orders, journal and broker agree.

Verified at the broker today: IOC marketable-limit entries fill in about 100 ms;
STP and GTC MKT goodAfterTime orders are accepted immediately and rest PreSubmitted;
a stop fill cancels the timed exit through OCA type 2 within 60 ms (code 202 on the
sibling), both times; the 09:25 preflight, the second-client reconcile and the
executions listing work, and `reqExecutions` from another client returns this
client's fills (cross-client visibility confirmed).

Still unverified: the real shape of an order rejection, and the 15:55 exit firing
and surviving on a day a position is still open into the close.

Two defects found, neither affecting orders, both fixed and committed the same day:
the own-position self-check reported UNKNOWN all session because the executions
answer carried three-segment execution ids while the journal held four
(`exec_key` now idempotent, 5186dd22); and Slack alerts posted to the shared
channel against the owner's wishes (Slack now opt-in only, c352c582).

Tuesday 2026-09-29 launches the same way with the same launchers, but at normal
sizing (owner decision 2026-09-28 evening; see "Live sizing (from 2026-09-29)"); the
runner picks up 5186dd22 and c352c582 automatically. The bracket-entry probe
remains parked pending the owner's go and the ownership port.

## Daily launch by Task Scheduler (2026-09-28)

The daily launch no longer needs an interactive session. Task **`OpenBreakout_DailyLaunch`**
(registered by the owner; weekdays **08:12 ET**, interactive logon, limited run level,
start-when-available, two-hour limit) runs
`artifacts/open_breakout_runs/daily_launch.ps1` from the repo root. The tracked source is
`scripts/run_open_breakout_daily.ps1`; deploy it byte-for-byte to the task path after qualification. It:

1. takes today's New York date as the session; a non-XNYS day logs
   `DAILY_LAUNCH SKIPPED` and exits 0 (`artifacts/open_breakout_build/is_session.py`);
2. refuses to launch at or after 09:20 ET (a late start-when-available run);
3. waits up to 10 minutes (30 s polls) for the Gateway port 7496 to listen;
4. refuses if any `open_breakout` shadow/live session process is running or any
   `<date>-shadow*` / `<date>-live*` dir already has `runtime.sqlite`;
5. finds the newest `risk_*/refresh.json` for the session without `risk_error` (same rule
   as the launchers); if none, runs `refresh_risk.py --session <date>` once and re-checks;
6. runs the read-only preflight with the live config and client **927485**, the env ack
   set only for that child, into `preflight-<date>-live.json`, and requires `"ok": true`.
   The startup helper permits at most three fresh child processes, with 2/5-second delays,
   only when every failure is explicitly retryable transport/recovery evidence. Each attempt
   writes a separate report without overwriting earlier evidence. Mixed account, position,
   working-order, margin, child-exit, timeout, or malformed-report failures stop immediately.
   Retries and successful completion are checked against the session date and 09:20 ET cutoff;
7. runs `launch-shadow.ps1 -Session <date>` then `launch-live.ps1 -Session <date>` (a
   shadow launcher failure does not block the live launch but fails step 8);
8. after 60 s (then polling up to 3 more minutes) requires both sessions alive, connected
   and with a fresh heartbeat, via `daily_status.ps1`;
9. monitors the live journal through the first healthy post-09:30 heartbeat. Task success
   requires an armed/running phase, connected/healthy transport, and an open order gate.
   The monitor is read-only and never connects to IBKR.

Exit codes: 0 ok or not a session day; 1 bad arguments or unexpected error; 2 Gateway port
not listening after 10 min; 3 a session process is running or a journal exists; 4 no valid
risk refresh; 5 preflight failed or wrote no report; 6 a launcher failed or the sessions are
not both running and connected; 7 late start. Every failure ends the log with one line
`DAILY_LAUNCH FAILED (<code>): <reason>`; success ends with `DAILY_LAUNCH OK`.

Log: `artifacts/open_breakout_runs/daily_launch_<date>.log` (appended; the account number
is masked; nothing goes to Slack). Check it right after 08:15 ET:
`Select-String -Path artifacts/open_breakout_runs/daily_launch_*.log -Pattern 'DAILY_LAUNCH (OK|FAILED|SKIPPED)'`.
On failure inspect the attempt reports and failure log first. Never restart a missed
session after the 09:20 ET cutoff or bypass a failed business/safety gate.

Status at any time: `powershell -File artifacts/open_breakout_runs/daily_status.ps1`
(`-Session YYYY-MM-DD`, default today in New York): pid, process alive, phase, heartbeat
age, connected for each state dir; exit 0 only while both are running and connected.

Rehearsal: `daily_launch.ps1 -DryRun [-Session YYYY-MM-DD]` runs steps 1 to 6 for real
(read-only; step 5 may add a `risk_*` folder), runs both launchers with `-DryRun`, and
creates no state dir. Its log is `daily_launch_<date>_dryrun.log` and its preflight file
`preflight-<date>-live-dryrun.json`. `-Session` is accepted only with `-DryRun`.
Dry run 2026-09-28 18:40 ET for 2026-09-29: all steps passed (risk refreshed by hand,
`risk_20260928_224025Z`, risk_latest 2026-09-28, score 76.60, short gate OPEN; preflight ok).
Re-run 19:48 ET after the normal-sizing change: all steps passed, preflight what-if at
10 MNQ / 10 MES ok (`preflight-2026-09-29-live-dryrun-2.json`), launcher printed
effective caps MNQ 10 / MES 10 and live fingerprint `44b60282...0ce952`.
Re-run 20:05 ET after the planned-size margin change: all steps passed; step 6 now logs
`preflight margin: basis=cap ...` (at-cap total $102,201 vs limit $112,697, no warning;
it would be a warning, never a failure). A read-only arm-style check on client 927485 with
Tuesday's staging TRs (565.0 / 79.75) planned 3 MNQ / 7 MES and summed planned margin
$44,619 (MNQ BUY $20,203 + MES BUY $24,416) against $112,697: ok.
Re-run 20:57 ET after the cap removal (60 fat-finger ceiling): all steps passed; step 6
logged `preflight margin: basis=reference qty={"MNQ":{"BUY":10,"SELL":10},"MES":{"BUY":10,"SELL":10}}
total=101915.2 limit=112697.44 warnings=` (`preflight-2026-09-29-live-dryrun-4.json`); both
launchers printed MNQ 60 / MES 60, live fingerprint `0d0876ae...dce8`, shadow `fb28fcec...c97a`.

Stop a running session: create a file named `STOP` in its state dir
(`artifacts/open_breakout_runs/<date>-live/STOP`, and `<date>-shadow/STOP`); it does not
cancel or flatten anything. Stop future launches:
`Disable-ScheduledTask -TaskName OpenBreakout_DailyLaunch` (re-enable with
`Enable-ScheduledTask`). The nightly `OpenBreakout_RiskRefresh_Nightly` task is separate.

The launched python processes are started by the launchers with `Start-Process` (hidden,
own console) and are meant to outlive the task, which now monitors through 09:30 ET. Confirm on
the first scheduled day that `daily_status.ps1` still shows both sessions alive after the
task shows Ready. The owner still attends **09:25 to 11:30 ET** and **15:50 to 16:01 ET**.


### Startup retry regression (2026-10-09)

The 08:12 task exited 5 after `RECOVERY_EVIDENCE_CHANGED`, even though the saved report
explicitly marked the failure retryable. Session-level retries were never reached.
The launcher now performs the bounded read-only retries above; it never treats the
failed proof as permission to trade. The exact callback that invalidated that morning's
proof was not retained, so its underlying trigger remains unconfirmed.

Recovery-token rule (2026-10-09 fix): the token is `(connection epoch, book revision)`. The revision
now bumps only on a MATERIAL change: an order or position change on an execution/signal contract for
this account, an execution in one of those contracts, or a reconnect (epoch). Routine
NetLiquidation/ExcessLiquidity ticks and fills in unrelated contracts no longer invalidate it (the
proof windows still compare the account-value stream revision directly). What-if previews emit no
openOrder events. A failing preflight now records `recovery_bumps` (which callbacks moved the
revision) and `recovery_now` in its report. The 08:12 trigger is still not proven (it did not
reproduce in 5 live read-only runs); the account-tick and foreign-fill paths were the only
candidates found in code. Tests: `test_recovery_token_*` in tests/test_open_breakout.py.

Qualification: `python -m pytest tests/test_open_breakout_startup_preflight.py
tests/test_open_breakout_launch_monitor.py -q` on Windows. These exercise the tracked
launcher's actual step 6 with broker/process/clock stubs, including exhausted retries,
mixed safety failures, missing/malformed reports, child failures, retained evidence,
clock jumps, and cutoff crossings. No broker connection is made by these tests.

Before installing, verify `OpenBreakout_DailyLaunch` is not running, retain the installed
launcher and exported task XML under `artifacts/open_breakout_runs/deployment_backups/`,
parse both PowerShell files, then copy the tracked launcher to the task's existing path.
Verify source/installed hashes match. Keep the task's trigger and enabled state unchanged;
deployment is not a manual session launch. The reviewed recovery adapter is a separately
installed release: preserve its verified receipt/hash and do not overwrite it from an
older Git checkout when deploying this launcher change.

08:12 ET is startup and broker qualification. Strategy preparation begins at 09:00,
arming preflight runs at 09:25, and the trading gate is 09:30. Starting the launcher at
08:12 does not itself authorize an 08:12 trade.
