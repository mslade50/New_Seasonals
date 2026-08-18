# Execution bridge — command schema + the two tabs + UX

Companion to `site_execution_plan.md`. The contract between the site, the
Cloudflare broker, and the local agent, plus the two-tab UI design. Grounded in
the real mechanics of `sznl_entry.py` (timed bracket) and `sznl_exit.py`
(attach ladder exits to an existing position) in `OneDrive/trading_ibkr`.

Design principle: **the site never builds an order.** It expresses *intent*; the
local agent constructs the concrete IBKR order against the live book (reusing the
existing `sznl_entry.build_orders` / `sznl_exit.build_orders` / `read_book`).

---

## Command envelope (every command)

```json
{ "id": "uuid",                       // idempotency key — agent dedups
  "type": "entry_bracket | scheduled_option | scheduled_option_cancel | exit_attach | close_only | flatten | cancel | modify",
  "account": "primary | pa",          // routes to TWS 7496 / PA Gateway 4001
  "dry_run": true,                    // agent validates + logs, transmits nothing
  "created_at": "iso", "expires_at": "iso",   // agent refuses if expired
  "sig": "hmac(payload, shared_secret)",
  "payload": { /* type-specific */ } }
```

### `entry_bracket`  (← sznl_entry.py)
```json
{ "symbol":"USO","sec_type":"STK|FUT|CASH","exchange":"SMART","currency":"USD","fut_expiry":"202608",
  "action":"BUY|SELL","quantity":692,"entry_type":"LMT|STP_LMT|MKT|MOO|MOC","entry":104.80,"entry_cap":null,"stop":103.29,"target":123.21,
  "use_target":true,"use_time_stop":true,"stop_tif":"GTC|GTD",
  "parent_fill_by":"2026-06-26T15:00-04:00",
  "exit_at":"2026-07-09T16:00-04:00","time_stop_at":"2026-07-09T15:00-04:00",
  "outside_rth":false,"stop_outside_rth":null,"timestop_outside_rth":null }
```
Validation (mirrors `validate_config`): BUY → `stop < entry` and
`worst fill < target`; SELL inverted; `time_stop < bracket_gtd`;
`parent_fill_by` in the future.

`entry` is the executable limit for `LMT`. For `MKT` and `MOC`, it is a required
risk/reference price used for notional, stop-distance, and confirmation math; it
is not sent to IBKR as price protection. `MOC` is initially stock-only. Entry
expiry applies to `LMT` and `STP_LMT`; `MKT` and `MOC` are DAY orders. The
executor builds true IBKR parent types (`LMT`, `STP LMT`, `MKT`, or `MOC`) and
attaches the same optional stop/target/time-exit children.

**`STP_LMT` — stop-limit entry (2026-08-18).** `entry` is the STOP TRIGGER and
`entry_cap` is the limit, so the parent goes out as IBKR `STP LMT` with
`auxPrice = entry` and `lmtPrice = entry_cap`. BUY needs `entry < entry_cap`,
SELL needs `entry_cap < entry`; `entry_cap` is required for this type and
rejected for every other. Stock-only initially (like `MOC`) — the FUT/CASH
tick-snap paths do not snap the cap, so relaxing that gate means snapping it in
both blocks.

The cap is the **worst acceptable fill**, and that is the number every gate
reads: notional, the per-account notional cap, stop-distance risk, the R:R
preview, and the `target` ordering check all use `entry_cap` rather than the
trigger. Sizing off the trigger would undercount both — on a 383.12 / 400.17
trigger-cap pair against a 355.84 stop, per-share risk is $44.33 at the cap
versus $27.28 at the trigger, a 62% understatement.

A native `STP LMT` does **not** die on a runaway gap. If the session opens above
the cap the trigger fires immediately and leaves a resting buy limit at the cap,
which fills on any pullback to it. Callers wanting "dead for the day if the open
gaps past the cap" must check the open before sending, the same way the OVS
2-path gate and `T1_Open_Filters` do.

**Stop optional (2026-07-27).** `stop: null` is a legal entry: the parent order
goes out with whatever subset of target/time legs is present (none at all =
naked entry, parent transmits itself). Ordering checks apply only to the legs
present. UNPROTECTED entries are risk-gated at the executor on
`2 × ATR(14) × notional` vs the account's live NetLiquidation: above
`RISK_ACK_BPS` (50) — or when the ATR/NLV estimate is unavailable — the command
is BOUNCED with `RISK_ACK_REQUIRED` in the detail and
`fill: {needs_risk_ack, est_risk, est_bps}`; the site re-prompts with the
machine's numbers (secondary approval) and resends the identical payload plus
`risk_ack: true`. This is a confirmation gate, deliberately NOT a hard cap.
Stopped entries keep the agent-side 5%-NLV stop-risk guard; stop-less entries
skip it (no stop distance) and rely on the ATR ack gate. `sec_type:"CASH"`
uses `symbol` as the base currency and `currency` as the quote currency
(`NZD` + `USD` = `NZD/USD`), routes to IBKR IDEALPRO, and sizes in whole
base-currency units. Initial FX support requires one leg to be USD so notional
and stop risk can be normalized to USD. CASH/FX entries have no hard notional
ceiling, but still require a protective stop and remain subject to the
account-NLV risk guard. CASH entry,
stop, and target prices are snapped to IBKR's live contract tick.

### `scheduled_option` (target-delta market buy resolved at trigger time)
```json
{ "symbol":"SPY", "right":"P", "target_delta":0.15,
  "delta_tolerance":0.03, "premium_budget":1000,
  "order_type":"MKT", "tif":"DAY",
  "execute_date":"2026-08-21", "execute_time":"15:45",
  "timezone":"America/New_York", "grace_minutes":5,
  "expiry_mode":"min_dte", "min_dte":30, "expiry":null }
```
`expiry_mode:"specific"` instead carries an exact ISO `expiry`. The command is
delivered and signature-validated immediately, then persisted on the local
trading machine. At the trigger the agent requires live underlying and option
quotes, resolves the exact listed expiration, chooses the same-right contract
nearest the absolute target delta inside `delta_tolerance`, and sizes whole
contracts as `floor(premium_budget / (live_ask * 100))`. It then submits one
SMART `MKT DAY` buy. The budget is an ask-based target, not a hard fill cap;
market slippage can make realized premium larger.

The instruction expires without transmission after its grace window. Frozen or
delayed data, missing Greeks/ask/conId, an unavailable specific expiration, no
expiration at or beyond minimum DTE, a delta miss outside tolerance, or a budget
below one ask-priced contract all reject fail-closed. The persisted state is set
to `executing` before the transmit subprocess starts; a process or connection
loss from that point becomes `unknown` and is never automatically retried.

### `scheduled_option_cancel`
```json
{ "schedule_id":"original-scheduled-option-command-uuid" }
```
Cancels only a still-pending locally persisted instruction. The original command
record is updated to `cancelled`; an already executing or terminal instruction
cannot be cancelled through this command.

### `exit_attach`  (IMPLEMENTED 2026-07-27 — single-rung; ladder rungs deferred)
```json
{ "symbol":"AAPL","sec_type":"STK","con_id":111,"expiry":null,"currency":"USD",
  "stop":190.0,"target":230.0,"time_stop":"2026-08-14","outside_rth":false }
```
Attach a standalone closing OCA group (any subset of price stop / limit target /
scheduled 15:59 ET MKT time stop; at least one required) to an existing position,
sized to the **full live held quantity** (agent + executor both detect side/qty —
the site never sends a size). Rejected while ANY order is already working on that
exact contract: closing exits → "cancel or modify instead"; same-direction
entries → over-coverage risk once they fill. Wrong-side prices vs the live mark
are rejected on both layers (browser book first, IBKR mark at the executor).
OPT positions rejected. Site entry points: the "attach exits" ticket type and
the amber `Protect…` button on any position row with nothing working against it.
The original sznl_exit-style multi-rung `targets` ladder remains a later phase.

### `flatten`  (quick Flatten/Trim buttons — cancel/resize exit orders)
```json
{ "symbol":"USO","sec_type":"STK","currency":"USD","con_id":123,"expiry":null,
  "fraction":1.0,            // or "qty": N (whole shares; REJECTED above held, never clamped)
  "order_type":"MKT|LMT","limit":null,"tif":"DAY|GTC",
  "outside_rth":false }
```
Semantics (2026-07-09): MKT is fill-gated (non-fill = FAILURE, exits restored at
full size). LMT is a RESTING close — success = working; `outside_rth:true`
requires LMT (IBKR ignores MKT outside RTH). A partial close (qty or fraction < 1)
captures the working exit legs before cancelling and re-attaches them sized to
`held − close_qty`, so a resting LMT plus the resized exits never exceed the held
quantity. A FULL LMT close cancels all exits and rests unprotected until it fills
(the preview + result say so).

### `close_only`  (Close… ticket — working orders are never touched)
```json
{ "symbol":"USO","sec_type":"STK","currency":"USD","con_id":123,"expiry":null,
  "action":"SELL",             // must be opposite the live position
  "fraction":0.4,               // or "qty": N; resolved qty cannot exceed held
  "order_type":"MKT|LMT","limit":null,"tif":"DAY|GTC",
  "outside_rth":false }
```
Closes the requested shares/units or arbitrary percentage without reading,
cancelling, resizing, or replacing any working order. The command is rejected
if its action would add to the live position or if its resolved quantity exceeds
the live holding. The executor re-reads both immediately before transmission.
Existing exits may therefore over-cover the remainder until manually modified;
the site confirmation and execution result state that explicitly.

### `cancel`
```json
{ "scope":"order|symbol", "order_id":123, "symbol":"USO" }
```

### `modify`  (← pa_resize_bracket.py)
```json
{ "order_id":123, "new_limit":null, "new_stop":null, "new_qty":null }
```

### Status returned by the agent (up the same socket)
```json
{ "id":"uuid","state":"queued|validated|rejected|working|filled|cancelled",
  "reason":"...", "ib_order_ids":[...], "fills":[...], "at":"iso" }
```

---

## The two tabs

### Tab A — STAGE (build & submit)
Two forms, **Entry Bracket** + **Exit Attach**, mirroring the scripts' CONFIG
blocks but as real UI.

```
Stage Order        Account [Primary ▾]      ●Entry bracket  ○Exit attach
──────────────────────────────────────────────────────────────────────────────
  Symbol [USO    ]  ●Stock ○Future ○FX        ← pre-filled from signal "USO 21d"
  Side  ●Buy ○Sell    Qty [692]   ·or·  Risk $ [1,045] → solves qty
  Entry [104.80]    Stop [103.29]    Target [123.21]    □ no target
  ┌ price ladder ───────────────────────────────┐
  │  Target 123.21  ▲ +17.6%   ███████████       │   R : R   12.2 : 1
  │  Entry  104.80  ●                            │   Risk    $1,045  (14 bps)
  │  Stop   103.29  ▼ −1.4%    ██                │   Reward  $12,750
  └──────────────────────────────────────────────┘
  Fill by [Jun 26 · 15:00]   Exit on [Jul 9 · 16:00]   Time-stop [Jul 9 · 15:00] □ off
  ▸ Advanced  (per-leg RTH, stop TIF GTC/GTD, exchange / expiry)
  ✓ valid                              [ Dry-run preview ]   [ Stage order ]
──────────────────────────────────────────────────────────────────────────────
  PENDING            USO  BUY 692 @104.80   ● working   (sent 09:31)
```

### Tab B — POSITIONS / ORDERS (TWS-like, click-to-trade)
Live book pushed up by the agent (it already has the snapshot in
`morning_order_summary.read_book()`).

```
Positions / Orders     [Primary ▾] [PA]              ● execution online · agent up
──────────────────────────────────────────────────────────────────────────────
 POSITIONS
  Sym   Side  Qty   Avg     Last    P&L $    P&L%   Exits          Actions
  USO   LONG  692  104.80  108.10  +2,284  +3.1%  ◎T123 ◎S103   [Flatten][Trim½][Exits▾]
  AAPL  LONG  108  210.30  208.90    −151  −0.7%  ⚠ no stop      [Flatten][Trim½][Exits▾]
  NG=F  SHORT   3    3.42    3.30    +360  +3.5%  ◎S3.55 ◎⏱7/22 [Flatten][Trim½][Exits▾]
──────────────────────────────────────────────────────────────────────────────
 OPEN ORDERS
  Sym   Leg          Side  Qty   Price    Status     Actions
  USO   target LMT   SELL  692  123.21   Working    [Cancel][Modify]
  USO   stop   STP   SELL  692  103.29   Working    [Cancel][Modify]
```
`[Flatten]` → confirm modal: *"Flatten USO — SELL 692 @ MKT (Primary), cancels
target+stop. P&L ≈ +$2,284."*  → `[Dry-run]` `[Confirm]`.

---

## UX principles ("intuitive + user friendly")
1. **Error-proof by default.** Every action goes through a confirm modal that
   shows the *exact* order chain + risk; **Dry-run is the default**; client AND
   agent validate (the agent is the hard gate).
2. **One-click common actions.** Flatten, Trim ½ are single buttons — the things
   you do most shouldn't need a form.
3. **Express intent, not share counts.** Prefer %/fraction and "Risk $ → solves
   qty"; pre-fill quantities from the live position or a signal so you rarely type
   a raw number.
4. **Live feedback.** Real-time P&L, order status pills, and an
   **execution-online heartbeat** so the UI never lies about whether the agent
   can act.
5. **See the structure.** The entry/stop/target **price ladder** and inline P&L
   colour make the trade legible at a glance, not just numbers.
6. **Progressive disclosure.** The common case (a stock entry bracket) is dead
   simple; sec-type/expiry, per-leg RTH, stop TIF live under "Advanced".
7. **Tie into the rest of the site.** A **"Stage this"** button on each seasonal
   signal card pre-fills the Entry form with the ticket's entry/stop/target/dates
   → see signal → review → submit, in three clicks.
8. **Consistent** with the existing card/table styling (no new visual language).

---

## Build order (unchanged)
P1 morning summary → site stands up the read-only half of this pipe and the Tab-B
table style. Then P2a heartbeat, P2b dry-run command loop, P2c tiny live (PA,
flatten-only), P2d expand.
