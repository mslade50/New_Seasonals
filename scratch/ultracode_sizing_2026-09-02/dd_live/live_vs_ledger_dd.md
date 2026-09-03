# GAP 4 due diligence: live fills vs the ledger (2026-09-02)

Scope: the review email's section-3 line "No live R series exists; the 40% haircut is
structural; the post-freeze OOS window is 108 trades with CI [-0.57, 1.22] on the
ratio." Everything below is read-only. Scripts and outputs live in this folder:
`fetch_do_fills.py` -> `do_fills.json` / `do_book.json`; `match_live_ledger.py` ->
`matched_legs_primary.csv`, `matched_legs_pa.csv`, `staged_window_three_graders.csv`,
`matched_stats.json`, `match_run.log`; `oos_reread.py` -> `oos_reread.json`.

## 1. Inventory of live-fill sources

| # | Source | Where | Date range | Rows | Fill price | Qty | orderRef `SYM\|ACTION\|Strategy\|Date` | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Execution-broker DO fills ring (`GET /fills`, STATUS_TOKEN from `trading_ibkr/exec_agent.env`) | Cloudflare DO; pulled to `do_fills.json` | 2026-08-20 -> 09-02 (9 trading days; 14-day retention, 08-20 rows drop ~09-03) | 182 fills (Primary 123, PA 59; STK 162, FUT 19, OPT 1) | yes (per exec, VWAP-able) + commission | yes | 105/182. Entries and tagged exit legs carry it; the manual sells and futures/options do not | THE ONLY BROKER-GRADE SOURCE. Nothing harvests it: the site reads it live, no file is ever written |
| 2 | `book_snapshot.py` -> DO `/book` | `do_book.json` | point-in-time (pushed 2026-09-02 evening) | 22 primary positions, 33 working orders | avg cost per POSITION (legs aggregated) | yes | on working legs (15 tagged refs) | positions, not fills |
| 3 | `Trade_Journal` Sheets tab (output of `trade_journal.py`) | `sheets_Trade_Journal.csv` | 2026-02-09 and 02-12 only | 27 (9 filled, 18 missed) | yes | yes | no (pre-tag era; `strategy` column present, retired strategies) | dead: last run Feb 2026; local `~/trading_journal/trade_journal.csv` DOES NOT EXIST (dir empty); reads only today's `ib.fills()` |
| 4 | `Manual_Journal` tab | `sheets_Manual_Journal.csv` | 2026-02-05 -> 02-11 | 37 | no | yes | no | hand notes, 2 graded |
| 5 | `Trade_Signals_Log` sheet1 (`verify_fills.py`) | `sheets_Signals.csv` | 2026-03-26 -> 08-31 | 881 (853 current-book strategies) | Fill_Price = MODELLED from yfinance daily bars (limit price or T+1 open), not a fill | Shares = STAGED qty | no; (Strategy_Name, Ticker, Date) is the key | yfinance-modelled. No T+1 open gate, so OVS rows grade FILLED that were never traded (8/8 in the ring window); 0-share P/C-zeroed rows also grade FILLED |
| 6 | `execution` / `execution_2` tabs, `staged_orders.csv` | `sheets_execution*.csv`, trading_ibkr | cleared daily; today holds 1 row (UNH 09-01) | 1 each | limit price only | staged qty | `Strategy_Ref` + Symbol + Staged_Date -> derivable | staging, not fills |
| 7 | `Portfolio` tab (`daily_portfolio_report.py`) | `sheets_Portfolio.csv` | snapshot 2026-09-01 | 16 | ENGINE entry price | engine shares | no | modelled, not live |
| 8 | Position Sheet emails (`daily_execution_report.py`, Gmail subject "Position Sheet") | Gmail | 2026-07-08 -> 09-01 | 40 emails, 1/trading day | Avg cost per position (legs aggregated: LUV 2,931 @ 41.70 across three legs) | yes | strategy tag + staged date, no per-leg key | entry evidence only, no exits; reconstructable by diffing consecutive days but lossy |
| 9 | Morning Orders emails (`morning_order_summary.py`) | Gmail | ~mid-April -> 09-02 | ~201 (2/day, primary + PA) | no (working/filled counts at 9:31) | staged qty | yes in body | staging record |
| 10 | `eq_placed_orders.json` / `pa_placed_orders.json` / `olv_exit_placed.json` | trading_ibkr | today only (rewritten) | 1 / 1 / 1 | no | yes | yes (`sig`) | placement dedup journals |
| 11 | `exec_agent_seen.jsonl` | trading_ibkr | 2026-06-30 -> 09-02 | 482 command receipts | no | no | no | site-order command log, no fills |
| 12 | `reconcile.py` (staged vs IBKR OHLC) | trading_ibkr | never run in this era | 0 | modelled | | | same class as verify_fills |
| 13 | IBKR Flex / Activity exports, TWS trade logs | none on disk (`C:\Jts` has no trade exports; `Downloads/fills*.csv` are 2022 Coinbase files; `OneDrive/Maestro_Trades*.csv` unrelated) | | | | | | NOT PULLED YET. This is the backfill source (see section 4) |
| 14 | `options_journal.jsonl` | trading_ibkr | 2026-08-06 | 1.5 KB | | | | options, not the book |

Ledger used throughout: `data/backtest_trades_full.parquet`, vintage `gha:33608560596`
built 2026-09-02 08:30 UTC, 4,696 rows, last signal 2026-08-28, last bar 2026-09-01.
The study's 05_oos JSON also reports last signal 08-28, so it ran on the same vintage.
16 post-freeze OLV rows are open positions booked as "Time" at the 09-01 close
(`max_exit_idx = min(entry_idx + hold_days, len(df) - 1)`), i.e. marks, not exits.

## 2. Matched set (Primary account, DO ring vs ledger)

Key: orderRef date = staged date = signal date + 1 bday; `signal_date = ref_date - 1 BDay`.
Live R uses the LEDGER risk-per-share (stop_atr x ATR, `Risk_flat_750k / Shares_flat`)
so the ratio isolates price and exit differences. Exits: tagged exit fills go to their
own leg; untagged sells (the manual trims) are allocated FIFO within the symbol; unsold
remainder is marked at the ledger's own last bar (09-01 close), exactly as the ledger
marks its open rows. Three entries (LUV 08-17, LUV 08-18, UNH 08-18) pre-date the ring
and use the Sheets limit price as the entry proxy (flagged `sheets_limit_proxy`); GTC
limit fills in the ring land AT the limit (median slip -0.09 bps) so the proxy is tight
except UNH 08-18, where the scan's limit (391.49) and the ledger's (386.62) differ by
126 bps (different close/ATR vintage on the signal day).

| Stat | Primary |
|---|---|
| Entry legs in ring | 20 (15 broker entries incl. GL LT Trend; 3 sheets-proxy; 2 unmatched) |
| Matched to ledger | 18 (17 OLV, 1 LT Trend ST OS) |
| Unmatched | SPG 08-27 and 08-28 (OLV): scan fired, live bought 477 + 710 and sold all 1,187 the same day untagged (manual), the ledger never has an OLV SPG trade |
| Live fully closed | 6; ledger closed 2 (ACHC target, GL time). 16 of 18 ledger rows are open marks |
| Live avgR / ledger avgR | -0.236 / -0.325 |
| Ratio (point) | 0.72 |
| Ratio CI95 (symbol-cluster bootstrap) | [-0.77, +2.35] (denominator near zero; meaningless at N=18) |
| Paired diff live - ledger | mean +0.089, median +0.006, CI95 [-0.15, +0.32] |
| Decomposition | ledger -0.325 -> live entry with ledger exit -0.330 (entry effect -0.005R) -> live exit -0.236 (exit/discretion effect +0.094R) |
| Entry slippage vs ledger entry | broker-only mean -9.0 bps (favorable), median -0.09; all 18: mean -0.5, median -0.09; 8 legs > 20 bps abs (ON -71, LUV -30, CMI -26, D -10, UNH +126 proxy) |
| Shares ratio live / ledger | 0.97 mean (D 08-24 leg 0.50: live 693 vs ledger 1,387) |
| Commission | 0.0017R per leg (mean), i.e. nothing |
| Closed-both subset (N=6) | live -0.446 vs ledger -0.606 |
| By strategy | OLV N=17 live -0.260 / ledger -0.353; LT Trend N=1 +0.170 / +0.152 |

Exit-type agreement (ledger rows x live outcome): ledger Target 1 (ACHC) -> live
manual sell the day BEFORE the target bar (+1.00R live vs +2.00R ledger); ledger Time
17 -> live: 11 still open at the same mark, 1 OLV vol-confirm exit (POWI 09-02, ledger
still holds), 2 tagged time legs (GL, LUV 08-18), 3 manual/untagged trims (LUV 08-17,
UNH x2). Manual trims were net POSITIVE vs the ledger (LUV 08-17 leg: -1.13R live vs
-2.88R ledger; UNH legs +0.13R and -0.03R) and negative on ACHC (-1.0R).

PA account (`matched_legs_pa.csv`): 16 legs, 15 matched, at ~0.11x primary size; the
whole book was flattened by hand on 08-27 (13 of 15 legs closed untagged), live avgR
+0.41 vs ledger -0.08. PA is a discretionary mirror, not a fill-quality sample.

Fill rate of staged limits, signals 2026-08-19..08-28 (fill windows inside the ring):

| Strategy | Staged | Sheets FILLED | Ledger fill | Broker fill |
|---|---|---|---|---|
| Oversold Low Volume | 26 | 16 | 14 | 16 (SPG x2 not in ledger) |
| Overbot Vol Spike | 53 | 8 | 0 | 0 |
| LT Trend ST OS | 2 | 1 | 1 | 1 |
| SPY QQQ MonFri (0-share rows) | 2 | 1 | 0 | 0 |
| Weak Close Decent Sznls (0-share) | 1 | 1 | 0 | 0 |
| Total | 84 | 27 | 15 | 17 |

Ledger-vs-broker agreement 97.6% (2 broker fills the ledger lacks, 0 ledger fills the
broker lacks). Sheets-vs-broker 88%: 10 Sheets FILLED rows never traded (8 OVS, 2
zero-share). The ledger is the better proxy for "did it fill"; the Sheets column is not.

What cannot be matched and why: (a) anything before 2026-08-20 has no broker record
on this machine (ring retention 14 days, trade_journal never ran, no Flex pull);
(b) manual sells carry no orderRef, so leg attribution inside a stacked symbol is FIFO
by construction; (c) Position Sheet emails give aggregated avg cost, no exits;
(d) the Feb-2026 Trade_Journal rows are retired strategies (Vol Spike LOC Add, Deep
Oversold) and pre-date the ledger's live window.

## 3. OOS window re-read (current vintage, freeze dates per strategy)

Post-freeze rows: 108, of which 16 are open OLV marks from the August stack. Pooled,
mix-matched (IS avgR at the OOS strategy mix):

| Cut | N | avgR | IS at mix | Ratio | CI trade | CI day-block | CI ticker-cluster | P(ratio < 0.5) day-block |
|---|---|---|---|---|---|---|---|---|
| All 108 (study's number) | 108 | 0.185 | 0.516 | 0.36 | [-0.08, 0.80] | [-0.57, 1.22] | [-0.26, 0.99] | 0.63 |
| Closed only | 92 | 0.305 | 0.462 | 0.66 | [0.16, 1.16] | [-0.55, 1.69] | [-0.04, 1.40] | |

Per strategy (post-freeze): OVS 62 rows avgR 0.226 vs IS 0.396; OLV 26 rows avgR 0.006
but 10 closed rows 0.816 (IS 0.828); LT Trend 8 at 0.253 (IS 0.322); Monday Dip 4 at
-0.02; 3x Bear Fade 3 at +2.09; ATR Ext 2 at -0.26; MonFri 2 at -1.02; 3x ETF Fade 1.
Seven strategies have ZERO post-freeze trades (52wh, IOB, WCDS, St OS Sznl, Sector BO,
3x Leader, Monthly Weak Close). Leave-one-strategy-out on the closed ratio: 0.53-0.78
(drop 3x Bear Fade -> 0.53; drop OVS -> 0.78 on N=30). Leave-one-ticker-out: 0.54-0.79
(FUN and PAYX carry +6.0R and +5.7R; FBIN and XBI -3.8R and -4.4R).

Does 2026 alone drive it? The post-freeze window IS 2026 by construction (freezes
2026-04 to 07). Within-2026 control: Jan-2026 -> freeze, N=240, avgR 0.193, ratio 0.37;
freeze -> Aug-28 closed, N=92, avgR 0.305, ratio 0.62. The pre-freeze half of 2026 is
WORSE than the post-freeze half, so the deficit is a 2026 effect, not a freeze effect.
2026 closed vs pre-2026: ratio 0.44. Midterm Jan-Aug controls at each year's mix: 2006
0.19, 2010 0.84, 2014 0.62, 2018 0.39, 2022 0.57; 2026's 0.44 sits inside that range.
2026 by month (N, avgR): Jan 51/+0.36, Feb 31/-0.14, Mar 47/+0.61, Apr 42/-0.17, May
33/-0.13, Jun 68/+0.16, Jul 58/+0.52, Aug 18/-0.33 (Aug = the open LUV/POWI stack).

Read: the 108/0.36/[-0.57, 1.22] line is technically right and practically the wrong
number to quote. 16 of the 108 are unfinished marks, the closed ratio is 0.66 with a
trade-level CI that excludes zero, and every control says "2026 midterm year" rather
than "rules stopped working after being frozen".

## 4. The 0.60 assumption

Verdict: 0.60 is not supported by any live measurement and is too harsh as a claim
about EXECUTION; it is about right as a claim about 2026 REALIZED EDGE, which is a
different thing and is already in the ledger. On the 18 matched legs the ledger and the
broker agree on fills (97.6%), on entry price (median -0.09 bps, broker mean -9 bps in
live's favor), on share count (0.97) and on commissions (0.002R). The live-ledger gap
in this window is entirely discretion: manual trims (net +0.09R per leg here, -1.0R on
ACHC) and one same-day manual unwind (SPG). Execution keep looks like ~1.0 on limit
entries; what is not measured yet is (i) OVS shorts, where the live T+1 open gate and
P2 sizing have no broker sample at all in the ring (0 fills of 53 staged), (ii) stop
and time exits under stress, and (iii) the discretionary overlay, which has no ledger
analogue and cuts both ways. A haircut that bundles those into one 0.60 hides the fact
that two of the three are ledger-modelled already (OVS gate, exits) and the third is a
McKinley decision, not a fill-quality fact. Keep a placeholder if one is wanted, but
label it "discretion + unmeasured OVS" and not "live R".

Fastest path to N >= 150: not a daily job. Pull an IBKR Flex Query (Trades section,
fields incl. Order Reference, Date/Time, Quantity, Price, Commission) for the primary
account 2026-03-26 -> today, once. orderRef tagging in `eq_order_entry.py` post-dates
the 2026-07-01 backup (0 hits) and predates 07-08, so Jul-Aug fills carry the key
(ledger: 43 + 18 signal keys, ~77% live-staged -> ~45 legs) and Mar-Jun fills (~130
ledger keys) match keylessly on (symbol, fill date = signal + 1..3 bdays, staged qty
from Trade_Signals_Log). That is ~100-150 matched trades on day one, closed. Then
schedule the DO harvest (`/fills` GET + append to a local parquet keyed by exec_id)
daily so the 14-day ring never drops a day again; `trade_journal.py` is the wrong tool
(today-only `ib.fills()`, Sheets-driven staging that no longer matches the tabs).

At the current rate, waiting instead: ledger signal keys run 1.4/td in the Sheets
window (1.25/td trailing 12m), live stages ~77% of them -> ~1.0-1.1 live entries per
trading day; the ring showed 17 entries in 9 days during an OLV stack. 150 matched
CLOSED fills from a harvest started today lands in ~140-150 trading days, i.e. late
March to April 2027; ~80 td (January) if the August fill rate held, which it will not.

## WORTH DISCUSSING

1. The ring is being lost right now. 14-day retention, no writer anywhere; the 08-20
   rows drop on 09-03. A 20-line harvester (GET /fills, upsert by exec_id to a local
   parquet, run from the postclose pipeline) preserves the only broker-grade record.
2. Discretion is the live-ledger gap, not execution. Every divergence in the matched
   set is a hand decision: LUV/UNH trims, the ACHC early sale one day before its 2R
   target, the SPG same-day unwind, the PA flatten. If the haircut is meant to price
   McKinley's overrides, say so and track them as their own line; if it is meant to
   price fills, the evidence says ~1.0.
3. Trade_Signals_Log Fill_Status should stop being cited even informally: 10 of 27
   FILLED rows in the window never traded, all OVS or zero-share, because
   `verify_fills.py` has no T+1 open gate and grades P/C-zeroed rows. The ledger is
   the better fill proxy (97.6% vs 88%).
4. Two scan-vs-ledger signal splits surfaced in one fortnight: SPG (scan fired twice,
   ledger never) and UNH 08-18 (limit 391.49 vs 386.62, 126 bps). Both are data-vintage
   differences at scan time, both cost real money, neither is a haircut question.
   Worth a small guard: the AM scan prints its close/ATR next to the R2 parquet's.
5. The "108 trades, ratio 0.36" line in the email should be replaced by "92 closed,
   ratio 0.66 [0.16, 1.16] trade-level, pre-freeze 2026 was worse (0.37), midterm
   control range 0.19-0.84". The strongest available statement is that 2026 is a
   normal bad midterm year, not that the rules degraded after freezing.
