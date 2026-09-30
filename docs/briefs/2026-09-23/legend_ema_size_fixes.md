# Legend EMA: fixes required before lifting the one-share cap (2026-09-23)

Owner decisions 2026-09-23: shorts stay OFF; sizing goes to 40/30 NAV
(`LEGEND_EMA_SPY_NAV_PCT=0.40`, `LEGEND_EMA_QQQ_NAV_PCT=0.30`). An adversarial
review at size (~316 SPY / ~245 QQQ, ~$429k) returned LIFT AFTER FIXES. This
brief is the fix list. File: `C:/Users/McKinley Slade/OneDrive/trading_ibkr/legend_ema.py`
(+ `test_legend_ema.py`, `LEGEND_EMA_RUNBOOK.md`, `register_legend_ema_task.ps1`,
`legend_ema.env`). Constraints: activation flag stays as is; no orders; no
registration run; pytest is firewalled from production files; a read-only
`--dry-run` against TWS is allowed. Line numbers from the reviewer's read.

## Blockers (silent-loss class at size)

- **F1 signed net.** `fill_net_qty` (:474) floors at 0, destroying the sign.
  An over-sold long (TIME fires after the target already filled) nets
  negative and reads as flat; `run_verify_only` (:2094/2099) and the 10:32
  verify (:2012) both call it CLEAN. Fix: return the SIGNED net; every
  residual handler trades the residual back to zero regardless of declared
  side (negative net on a BUY sig -> BUY back abs(net) at market). Tests:
  over-sold long buys back; both verify paths.
- **F3 restart loses the time leg.** Resumed symbol has `st.time_trade=None`
  (:1387); no `find_time_trade`; `cancel_time_leg` returns NO_TIME_LEG
  (:1324) which `_settle_target_fill` (:1407) treats as success -> real
  TIME MKT fires against a flat book. Fix: `find_time_trade` by ref
  `sig|TIME`/`|TIMERETRY`, orderType MKT, non-terminal; on resume use it;
  NO_TIME_LEG on a resumed symbol -> re-check openTrades; still absent ->
  alert `TIME_LEG_MISSING` (BAD_STATUSES). Tests both cases.
- **F4 ambiguous entry orphans the position.** `ENTRY_AMBIGUOUS` /
  `ENTRY_UNFILLED_CANCELLED` (:1111-1118) do not set `st.placed`, journal
  no `entry`, so no revisions/sweep/verify and `--kill` finds nothing. Fix:
  after any transmitted parent, journal an `entry` with the netted qty and
  set `st.placed=True`; at the next checkpoint, if exits are absent and
  net>0, place them sized to net. Tests both statuses.
- **F7 the cap must be a cap.** `legend_ema.env`: `LEGEND_EMA_MAX_SHARES=350`
  (covers 316/245 with headroom). Keep the >1 guard in code but key it on
  `LEGEND_EMA_SIZE_GUARD_LIFTED=1` (absent = guard on). Add an NLV
  plausibility band (refuse < $100k or > $5M, configurable) and log
  BuyingPower/ExcessLiquidity rows.

## Next tier (bites on an ordinary morning)

- **F2 partial target fill.** `order_is_filled` (:979) treats any
  filled_qty>0 as filled, so one share filling at 09:44 cancels the whole
  TIME leg and stops revisions for the remainder, which rides bare until
  the 10:32 verify. Fix: compare target filled qty to open qty; on a
  PARTIAL, cancel-and-replace the TIME leg at the remaining qty (await
  terminal on the cancel; if the ack fails, alert and do NOT place a second
  TIME leg), keep revising the target on the re-netted remainder, journal
  `target_partial`. Only a FULL fill cancels TIME. Same at the pre-exit
  sweep. Tests at each checkpoint.
- **F5 partial entry.** At the 10s timeout the parent remainder stays
  working. Cancel it (await terminal), re-net, size exits to the final net.
- **F6 stale quantities.** `revise_target`: re-net AFTER the cancel ack,
  before placing the replacement. 10:32 verify: `_cancel_refs` must await
  terminal per cancel, then re-net, then trade the residual (mirror
  `run_verify_only`).
- **F8 QQQ can miss the window.** Transmit both parents first, then await
  both fills, then place exits per symbol. Deadline applies to parents.
- **Alerting.** On any BAD status/alert, email via the folder's existing
  smtplib pattern (search `pa_nightly_report.py`; reuse its config, never
  hardcode a password); subject `[Legend EMA] <status> <date>`; best effort.
- **Docs.** Purge OCA leftovers (RUNBOOK 136-137, 239; register script line
  5; module docstring 24-26); rewrite the guard section for the env key;
  document signed-net and partial-target semantics.

## Verification

`python -m pytest -q test_legend_ema.py` green with the new tests.
`python legend_ema.py --dry-run` read-only (RESUMED branch expected today).
FIRST live `--verify-only`: only after the close, with no working Legend
orders and today's test fills netting 0; expected NOTHING_TO_EXIT on every
ref and exit 0. If any ref nets non-zero, stop before it trades and report.

## Timing recommendation

Tomorrow (2026-09-24) 09:29 stays at ONE share on the fixed code, plus a
forced one-share test at 11:00; lift to 350 for Thursday 09:29 if both are
clean. Owner may override.
