# Ledger provenance, stop conventions, live fills store

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

Ledger SURVIVORSHIP CAVEAT (2026-07-16): the 23-year ledger trades only
tickers alive in today's universe files — 21 of 22 major 2020s delistings are
absent — which flatters long dip-buy stats and the ~870-name overflow tier
most. Treat overflow-tier historical avgR as an upper bound until the
dynamic-overflow work's point-in-time universe lands. Do not tune sizing off
overflow backtest stats alone.

Ledger provenance + integrity (2026-07-06, after a false TS/USO block): the
ledger is a FULL BACKTEST REBUILD, not a fill record -- marginal limit fills
flicker between vintages as yfinance revises recent bars, and the gate's -2.0R
threshold is a knife edge (the false block was -2.008R from a since-rebooked
trade in a weekend vintage of unknown origin). Mitigations, all in place:
- `build_trade_ledger.py` embeds provenance in the parquet schema metadata
  (`ledger_build_utc`, `ledger_source` = gha:<run_id> | local:<host>,
  `ledger_git_sha`, `ledger_rows`) and prints a vintage diff vs the prior
  ledger (new/gone/rebooked trades touching the last 15td) on every build.
- R2 upload of the prod ledger key is gated behind `--upload`; only
  `deploy_site.yml` passes it. Local runs (`refresh_view.py`) build but never
  overwrite the key that gates live orders.
- `daily_scan.py` prints the ledger's provenance at gate load and warns when
  the vintage is >4 days old or was built outside GHA (still fail-open).
- Blocked-signal notes name every contributing exit (ticker, date, R) so a
  block stays auditable after its vintage is overwritten.

**The ledger replays TODAY's config over ALL history, so its SIZE column is
not what live traded before the most recent sizing change** (2026-09-03,
measured against the primary account's IBKR statement). Concrete instance:
every OLV position entered on or after 2026-07-29 matches live share-for-share
(ratio 1.000), and every earlier one is 1.15x-2.85x SMALLER in the ledger than
live was, because the ledger applies the signal-recency ladder (0.5/0.7/1.0,
shipped 2026-07-30) to a period that had no ladder at all; 1/0.5 = 2.0 is the
observed ratio on several. R is per-share-normalised so per-trade R is
unaffected, but **any ledger-based NOTIONAL or EXPOSURE claim about a period
before its rules shipped understates live** — which is why the June-2026 OLV
stack was bigger in reality than the ledger-based drawdown work showed.
Evidence: `scratch/live_vs_ledger_2026/`.


## Stop-Arming Convention (book-wide, 2026-06-09)

Stop legs ARM AT THE NEXT SESSION, not at the fill. Decided after measuring
81 entry-day-stop episodes over 24y: booking -1R each vs arming on day 2 cost
-33R book-wide (dip-buy limit entries get stopped at max fear; a third of
MonFri's day-1 stop-outs went on to hit +2R targets).

Aligned across both sides -- change one, change both:
- `pages/strat_backtester.py` (`process_signals_fast`): entry-day stop check
  gated on `execution['stop_active_entry_day']`, **default False** (= day-2
  arming). Set True on a strategy to model a day-1-armed stop.
- `eq_order_entry.py` (in `C:\Users\McKinley Slade\OneDrive\trading_ibkr\`): STP
  child submitted with `goodAfterTime = next_session_gat()` (next trading day
  09:30, BDay-aware; holiday dates harmlessly defer to the next real session).
  Still in the OCA group, so a TARGET/TIME fill cancels the inactive stop.

Related conventions: entry-day TARGETS are never credited in the backtest
(intraday timing vs fill is ambiguous); OVS has `use_stop_loss=False` entirely
(its day-one valve is the Friday-only EOD-DD, see section above); OLV has NO
resting stop leg at all since 2026-07-20 (vol-confirmed next-open exit — see
"OLV Vol-Confirmed Stop + Notional Cap"; its rows stamp Use_Stop=False while
`use_stop_loss` stays True in config for the sizing risk unit).

## Stop-Fill Convention — gap-through + slippage (book-wide, 2026-06-27)

A stop the bar GAPS THROUGH fills at the OPEN, not the stop. The old engine
always booked the exit at exactly `stop_price`, which pinned every stop-out at
exactly -1R and understated the gap-down tail (the website showed OLV — and
every stop strategy — "never losing more than 1R"). The realized fill is now the
worse of the stop and that day's open, plus slippage.

`process_signals_fast` (`pages/strat_backtester.py`) — drives the full-history
ledger (`scripts/build_trade_ledger.py` -> site) AND `daily_portfolio_report.py`:
- `_stop_fill_price(direction, stop_price, day_open, gap_fill, slip_bps, gap_slip_bps)`
  is the single fill model. Long: `min(stop, open)`; Short: `max(stop, open)`.
- Slippage: `STOP_SLIP_BPS = 3.0` on EVERY stop fill, plus an ADDITIONAL
  `STOP_GAP_SLIP_BPS = 10.0` (so 13 bps total) when the bar gapped through.
  Always worsens the fill (long sells lower, short covers higher). Targets and
  time exits get NO slippage. OVS EOD-DD (close exit) is untouched.
- New kwargs `stop_gap_fill=True, stop_slip_bps=3.0, stop_gap_slip_bps=10.0`
  default to the prod behavior; pass `stop_gap_fill=False` to reproduce the
  legacy fill-at-stop for before/after measurement.
- Entry-day stop (off by default) gets slippage only — no gap-to-open, since the
  open precedes the intraday limit fill.
- Scale-invariant under the dividend-adjustment rule: the stop is relative and
  `Open` is on the same adjusted basis within a run, so both scale by the same
  factor (CLAUDE.md "Dividend-Adjustment Basis").

Impact (full book, 2003-2026, flat $750k, `scratch/stop_gap_slippage_impact.py`):
85 of 434 stop-outs (~20%) gapped through. Book TotR 605.9 -> 560.2 (-45.7R),
AvgR 0.525 -> 0.485, worst single trade -1.0R -> -4.56R, -$157.7k flat (~8% of
these strategies' PnL). OLV: 25/116 stops gapped, TotR 193.5 -> 182.9, worst
-1.0R -> -2.29R.

Live trading was already correct (IBKR STP -> market order fills at the gap
open); this only removes backtest/ledger/site optimism. `pages/backtester.py`
(interactive UI) is a separate engine: its persistent-limit path already does
`min(Open, stop)` (line ~2439); the simpler paths (~2312-2351) still fill at the
stop and would need the same treatment for full parity (deliberately separate
exploration surface, not yet aligned).


### Live fills store (durable execution record, 2026-09-02)

`scripts/harvest_fills.py` -> `data/live_fills.parquet` (R2 key
`live_fills.parquet`, R2-CANONICAL, gitignored) + `live_fills_status.json`.
Runs as the `harvest_fills` job in the `postclose` pipeline, LOCAL-ONLY (no
GitHub workflow backup: the ring sits behind the broker's read token).

**Why it exists.** IBKR's API serves only the CURRENT session's executions.
`book_snapshot.py` pushes each account's fills with the book; the broker DO
folds them into per-day keys and DROPS them after `retention_days` (14). So
before this job the ring was the only accumulating copy of actual fills and a
row older than 14 days existed nowhere machine-readable — only in IBKR's own
Flex/activity statements. Found 2026-09-02 by the sizing due diligence.

**Execution is free, measured 2026-09-03** on the primary account's Jan-Sep
statement (`scratch/live_vs_ledger_2026/`): 69 matched positions from June
onward give live/ledger avgR 0.982, paired diff -0.004R (CI95 -0.046 to
+0.042), entry slippage 0.0 bps at the median, share counts exact. So
`live R = ledger R x 0.60` is NOT supported as an execution claim; any haircut
prices edge decay and discretionary override, not fills. That measures a
different thing from the ring-based 18-leg figure (ratio 0.72), which scored
the position's REALISED outcome including hand trims — the gap between the two
IS the discretion.

**The systematic book only went live in the primary account around June 2026.**
Ledger-position-to-live-entry match rate by month: Jan 3%, Feb 11%, Mar 0%,
Apr 4%, May 42%, Jun 77%, Jul 95%, Aug 100%, against a statement carrying
99-187 stock orders EVERY month. Jan-Apr activity in that account was not the
book. **No live-vs-ledger claim before June 2026 is meaningful**, and the 67
OVS ledger positions with no live counterpart sit there — they are not a
2-path gate divergence.

Order reference: IBKR exposes it on SINGLE-DAY activity flex queries only, so a
multi-month pull cannot carry it and statement matching is keyless on
symbol + session + side. No order placed before 2026-07-02 carries one anyway
(tagging entered `eq_order_entry.py` between the 07-01 and 07-02 backups).

Contract:
- **Upsert by `exec_id`, never append.** Commission reports lag the fill, and
  an MOC fill can miss the day's last book push and only appear in tomorrow's
  ring. The broker wins field-for-field EXCEPT `ENRICHMENT_COLUMNS`
  (commission, realized_pnl), where a null incoming value keeps what is
  stored — a later fetch must never erase a commission.
- **Set containment, not row count**, is the write guard: every `exec_id`
  already held must survive the merge or the run raises. An empty ring (a
  no-trade fortnight) is legitimate and never shrinks the store.
- **Gap detection is the point.** The ring's oldest session is compared to
  our newest stored session; a hole means rows aged out unseen and prints a
  loud GAP line (`--assert-no-gap` exits 3). Only an IBKR Flex pull recovers
  those.
- `session_date` is the EASTERN session, not the UTC date (00:30 UTC belongs
  to the prior session). `order_ref` is parsed into `ref_symbol` /
  `ref_action` / `strategy` / `ref_date` off the book's
  `SYMBOL|ACTION|Strategy|Date` contract — that is what makes the store
  joinable to `data/backtest_trades_full.parquet`. Untagged legs (pre-2026-07,
  discretionary) get empty strings, never a guess.
- Schema is FROZEN in `COLUMNS`; new broker fields are dropped until added
  there (the fragility-parquet convention).

Coverage starts 2026-08-20 (the oldest row in the ring on the day the job was
built). Earlier history is recoverable ONLY from an IBKR Flex Trades pull with
the Order Reference field, which is still a manual to-do. Guard:
`tests/test_fills_harvest.py`. The site Trade Log tab still reads the DO's 14
days directly; pointing it at this store is unbuilt.
