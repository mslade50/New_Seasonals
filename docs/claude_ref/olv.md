# OLV (Oversold Low Volume)

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## OLV Vol-Confirmed Stop + Notional Cap (2026-07-20)

Package replacing OLV's resting 1.25 ATR STP, its `sector_loss_gate`
(live 2026-07-02 to 2026-07-20) and its `ladder_multipliers` in one change.
Evidence chain: scratch/ultracode_research/olv_stop_condition_2026-07-17.md
+ scratch/olv_package_sim.py (package $466k -> $654k flat / 21y, win 61->70%,
worst chain -$8.8k -> -$17.2k; every piece LOYO/episode-clustered).

**Stop (`stop_mode: 'vol_confirm_close'`, `stop_vol_mult: 1.5`)**: no
resting STP leg. Exit MOO at the NEXT open iff a session CLOSES <=
entry - 1.25 ATR AND its volume >= 1.5x the trailing 20d median (ex-that-
day). Quiet closes below the level are HELD — low-volume weakness is the
entry thesis; T+10 time exit still bounds everything. A volume-spike exit
and a fresh OLV signal (10d vol rank < 15) are near mutually exclusive, so
the old same-day stop+rebuy churn (39 events) is structurally gone.
`stop_atr` 1.25 still defines the sizing risk unit. Per-leg tails widen to
occasional -2..-3R; there is NO overnight stop (gaps evaluated at the next
close). Live flow (2026-07-30 — TRUE pre-market MOO): daily_scan
`stage_olv_vol_confirm_exits` (PM run evaluates today's settled close; AM
run re-evaluates with corrected data, risk-report-correction style; EVERY
open leg prints an explicit per-leg verdict line — CONFIRMED / no breach /
quiet breach / entry-day / stale — so stacked positions are auditable leg
by leg, and legs sharing (ticker, Time_Exit_Date) raise a warning because
downstream bracket matching can't distinguish them) -> `OLV_Exits` Sheets
tab (always cleared+rewritten; ONE ROW PER CONFIRMED LEG with
`Time_Exit_Date` bracket key + `Entry_Date` audit column; a STALE ticker
bar is never re-evaluated — its previously staged exits carry forward
PER LEG, keyed (Symbol, Time_Exit_Date)) -> `olv_exit_moo.py` (OneDrive
trading_ibkr; standalone Task Scheduler task 'IBKR OLV Pre-Market Exits',
weekdays 9:10 AM ET) reads the tab directly (Execute_On == today only) and
places SELL MKT **TIF=OPG** on BOTH accounts — a genuine market-on-open in
the opening auction, submitted before the 9:28 cutoff (past 9:25 it falls
back to MKT DAY, loudly). Same safety layers as the old path: matches the
leg's working OCA bracket by orderRef prefix + time-leg goodAfterTime
(nearest-date fallback for calendar desync), primary clamps qty to
min(staged, leg, held) while the PA sells min(leg, held) — the FULL
matched PA leg (staged qty is primary-basis and deliberately ignored),
cancels the bracket before selling, RE-ARMS a protective time-exit clone
on total placement failure, and journals placed exits
(olv_exit_placed.json) so re-runs are idempotent. It REUSES clientIds
99/98 on purpose — TWS binds persisted brackets to the placing clientId,
and it runs clear of the 9:31 chain. History: 2026-07-20..30 these rows
rode order_staging -> eq/pa_order_entry as "MOO" MKT DAY orders placed
~9:31+ — AFTER the open, never a real MOO (order_staging needs the live
open for the OVS gap check, so it can't run earlier). That staging path
was REMOVED 2026-07-30; the Is_Position_Exit handlers in eq/pa_order_entry
remain as dormant safety nets for hand-staged rows. Entry-day closes are
NEVER confirms (day-2 arming convention — the scan skips legs entered on
the evaluation session, matching the engine's entry_idx+1 loop). Every
layer fails SKIP/open and pipeline failures surface as OLV-EXIT warnings
in the daily scan email — a missed exit falls back to the T+10 time exit,
never a naked short.

**Notional cap (`ticker_notional_cap: {pct_nav: 0.50, exempt:
OLV_CAP_EXEMPT_ETFS}`)**: stacked OLV legs in ONE single-stock ticker may
not exceed 50% of NAV in entry notional; later legs scale down / skip. ETFs
exempt. Catastrophe insurance for the no-resting-stop world (~4% of OLV PnL
historically, every clipped leg a winner; balloon stacks are low-ATR names).
The engine binds the cap in FRACTION-OF-SIZING-EQUITY terms (each open
leg's notional / the equity it was sized against, `cap_equity` on
open_positions): flat pass == live's pct_nav x fixed ACCOUNT_VALUE, and
the compounded pass makes identical clip/skip decisions (either dollar
basis lets the passes diverge — NaN flat rows or 76 silently dropped
trades, both hit on 2026-07-20). KNOWN BOUND:
the cap counts FILLED positions only; with the T+3 fill window up to THREE
days' unfilled full-size limits are invisible, so worst-case concurrent
notional is ~3x one leg. Engine and live share the blindness (parity
holds); a working-order-aware check is the eventual fix.

**Sector gate removal**: the 20y drop list (-5.3R at ship) flipped to +10R
after the gate blocked the entire late-June-2026 oil recovery (OXY +2R x3,
USO winners) having saved only part of the decline. The generic gate +
`sector_gate_blocked` machinery survives dormant (keyed on the execution
field); `build_trade_ledger`'s nogate pass now SKIPS with a notice and the
site's gate_lab section quietly disappears. `data/sector_map.parquet` and
`scripts/build_sector_map.py` remain (other consumers). Ladder removal:
see "Ladder Sizing" above.

Aligned sites — change together:
- `strategy_config.py` OLV execution `stop_mode` / `stop_vol_mult` /
  `ticker_notional_cap` + `OLV_CAP_EXEMPT_ETFS` (source of truth)
- `pages/strat_backtester.py` — vol-confirm exit branch (target checked
  BEFORE the close-confirm; no confirm on the final hold day; next-open
  fill with stop slippage, no gap logic) + per-ticker notional cap replay
  (open_positions state, refund semantics mirror the net-exposure cap)
- `daily_scan.py` — Use_Stop stamped False for stop_mode strategies;
  `load_open_position_notionals` + sizing-step cap; `stage_olv_vol_confirm_exits`
- `olv_exit_moo.py` (OneDrive) — pre-market TIF=OPG exit runner for BOTH
  accounts + `run_olv_exit_moo.bat` / `register_olv_exit_task.ps1`
  (Task Scheduler 'IBKR OLV Pre-Market Exits', weekdays 9:10 AM ET);
  eq/pa_order_entry keep dormant `Is_Position_Exit` handlers;
  guard: `test_olv_exits.py` (OneDrive)
- Guard: `tests/test_olv_stop_and_cap.py` (engine + scan + config invariants)

### OLV sizing fallback — complete capacity observation (2026-09-15)

The September 14 order-reference patch supplied held notional but left the
actual scanner cap, NAV and pending-order checks dependent on reconciled
inventory. It was not sufficient to restore the cap.

`olv_capacity.py` now supplies a separate sizing-only Capacity: held stock
market value, remaining OLV BUY-parent limit reservations, and actual Primary
NAV from one validated broker observation. The scanner gates on Capacity.known;
exit inventory retains its own status and is never promoted by this fallback.

Fallback holdings are conservative: all stock exposure in the candidate's same
ticker counts, including untagged or other-sleeve shares. This can tighten the
cap but avoids assuming that missing/old references prove no OLV ownership.
This is absolute broker net exposure, not verified gross OLV ownership;
offsetting shorts from other sleeves remain an attribution limitation.
Other tickers do not consume a candidate's per-ticker cap. Existing ETF
exemptions and the owner's fail-open policy when capacity is unavailable remain.

The existing 16:05 capture saves independent sizing evidence in
`ops/olv_capacity/`. It can succeed while reporting exit inventory unknown.
Bookend scans prefer the dated prior-close evidence until the next cash open;
a dedicated read-only broker query is the fallback. Its collection timestamps
and completed current-day executions reserve buys that occur between copied
holdings and pending orders. Stale, incomplete, nonfinite or
mismatched inputs are rejected. Capture success certifies sizing only.

Tests execute the real scanner setup and cap branch, independent closing
capture/next-morning use, partial pending fills, exemptions, invalid inputs,
and unchanged unknown exit-inventory state. See
`docs/olv_capacity_repair_2026-09-15.md` for release evidence.


## OLV Book Cap — EOD pro-rata trim to <= 100% NAV (2026-08-24, live-only) — DISABLED 2026-08-25

**Status: built, dry-run verified, never fired live.** Task Scheduler entry
'IBKR OLV Book Cap (EOD)' was DISABLED on 2026-08-25 (still registered;
`Enable-ScheduledTask` re-arms it) when McKinley opted to hedge the Aug-2026
episode manually as a one-off instead of standing machinery. Code + tests
remain in OneDrive trading_ibkr; `--dry-run` still works for inspection.
Design record follows.

McKinley: "I don't want it to get past 100% of NAV, ever." OLV has no
resting stop and nothing bounded the AGGREGATE book (the 50%-NAV ticker cap
bounds one name); on 2026-08-24 OLV carried ~78-84% of NAV across 10 names
with the dial at 89.5. `olv_book_cap.py` (OneDrive trading_ibkr; Task
Scheduler 'IBKR OLV Book Cap (EOD)', weekdays 3:40 PM ET; guard
`test_olv_book_cap.py`) runs on BOTH accounts:
1. Discovers the OLV book from the working brackets (SELL legs whose
   orderRef is `SYM|BUY|Oversold Low Volume|...`, grouped by OCA group); a
   bracket whose BUY parent is still working is an unfilled ENTRY, otherwise a
   filled LEG. Only bracket-attributable shares are touched, clamped to held.
2. Excludes legs whose time-exit leg fires today (goodAfterTime == today).
3. Marks the rest at the live session last and compares to the cap — flat
   `ACCOUNT_VALUE` ($750k) for the primary, live NetLiquidation for the PA.
4. Over the cap: every leg sells ceil(qty x (1 - cap/book)) as a NATIVE
   MOC/DAY (the 2026-08-21 encoding rule), AFTER its TARGET/TIME legs are
   modified down to the post-trim qty as the owning clientId (99/98). A
   rejected/vanished MOC restores the legs (or re-arms a time exit).
5. Working ENTRIES are shrunk into the remaining room (the T+3 fill window
   would otherwise refill the book overnight); cancelled below 1 share.
Idempotent via journal + `OLV_BOOK_CAP` orderRefs; past 15:48 ET the sell is
MKT DAY; past 15:58 nothing is placed. Emails only on action or failure.

**Engine parity: NOT modeled.** `process_signals_fast` caps at entry only
(net-exposure + ticker caps scale new legs) and has no partial-exit concept;
a "Trim" exit-type partial row is the eventual fix. Replay of the exact rule
on the flat-$750k ledger (MTM closes): binds ~10 days in 3 episodes since
2015 (Jul-2021 21%, Jan-2025 up to 41% on 26 legs, Jun/Jul-2026 13%),
~$58k foregone of $362k OLV PnL. Until modeled, ledger OLV stats overstate
live by roughly that on those episodes. Interplay: `olv_exit_moo` sells the
post-trim leg (it reads the live leg qty); `daily_scan`'s ticker-cap loader
sees the reduced nightly positions; the morning entry dedup keys on
orderRef identity, so a resized working entry is never re-placed.


## OLV Entry-Order Live Window (T+3, 2026-06-24)

The OLV (Oversold Low Volume) persistent close-0.25 ATR limit is cancelled if
unfilled after **3 trading days** (T+1..T+3), not the full 10-day hold. A fill
inside the window is kept and unchanged (its hold is still reduced by wait time
off `hold_days`); a signal that hasn't filled by T+3 close is dropped.

Evidence (`scratch/olv_fill_window.py`, bucketing the full ledger by fill day):
89% of OLV fills land by T+3. The day 4-10 fills add ~0 total R (+211 -> +211 R
over 21y) while diluting per-trade edge: avgR +0.637 (T+3) vs +0.566 (T+10),
win 62.8% vs 60.6%, PF 2.90 vs 2.65. So total return is unchanged but
risk-adjusted quality improves, and capital isn't tied up in stale GTC orders
that mostly fill into names that kept bleeding for a week+.

Generic mechanism: `execution['fill_window_days']` caps the persistent-limit
fill search; **defaults to `hold_days` when absent**, so the other 5 persistent
strategies are untouched. Aligned sites (change together):
- `strategy_config.py` — OLV execution `fill_window_days: 3` (source of truth).
- `pages/strat_backtester.py` — `fill_window` bounds `search_end` in both
  persistent fill loops; the hold reduction still references `hold_days`.
  Drives the ledger + `daily_portfolio_report.py`.
- `daily_scan.py` — stamps `Fill_Window_Days` on every primary staging row.
- `order_staging.py` (in `OneDrive\trading_ibkr\`) — stamps `Entry_Expire_Time`
  = signal + `Fill_Window_Days` BDays = `Exit_Condition_Time` − (1 + hold − fill)
  BDays, into the execution CSV + `execution`/`execution_2` tabs. Defaults to
  `Exit_Condition_Time` (the 10-day-hold expiry) unless the row carries a valid
  `0 < Fill_Window_Days < Hold_Days`, so only OLV is affected.
- `eq_order_entry.py` + `pa_order_entry.py` (same dir) — the persistent GTC
  parent's `goodTillDate` reads `Entry_Expire_Time` (falls back to the time-exit
  `gat_time` when absent/blank). The TIME exit leg still uses `gat_time`, so a
  filled OLV position keeps its full reduced hold — only the unfilled entry order
  is cancelled early. The order is live T+1..T+3 (expires T+3 15:59).
- `pages/backtester.py` UI still uses `holding_days` as its fill window (an
  exploration surface, deliberately separate from the prod-locked rule).
- Regression coverage: `tests/test_olv_fill_window.py` (backtest engine);
  the live date math is validated by the entry-expire chain (daily_scan exit-date
  build <-> order_staging back-computation, identical `CustomBusinessDay` calendar).
