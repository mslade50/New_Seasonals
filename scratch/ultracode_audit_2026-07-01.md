# Ultracode repo audit and research report, 2026-07-01

Produced by a 70-agent workflow: 10 subsystem finders, one adversarial verifier per finding
(instructed to refute; only findings that survived a line-by-line trace appear below),
5 improvement research tracks, and a completeness critic.

Raw findings: 58. Unique after cross-dimension dedupe: 54. Confirmed by verification: 49. Refuted: 5.

Severity counts: high: 11, medium: 23, low: 15

## Part 1 — Confirmed bugs

### Severity: HIGH

#### 1. Mode banner and confirm dialogs fail open to "dry-run" when the book is null or stale — a live order can transmit while the UI asserts nothing is transmitted

`site/assets/execution.js:130` (dimension: exec-frontend)

renderModeBanner() line 130: `const mode = (state.book && state.book.mode) || "dry-run"` — a missing book renders the green "DRY-RUN MODE — nothing is transmitted" card, and isLive()/actionLead() (lines 323-324) make every confirm dialog read "Dry-run: place/flatten/cancel ...". But live-vs-dry is decided solely by the agent's env (exec_agent.py `_live_eligible`, currently armed per exec_agent.env for BOTH accounts and all three types); the envelope's `dry_run: true` stamped by functions/exec-command.js line 33 is ignored by the agent, so `book.mode` is the only signal and the frontend defaults it to the safe-LOOKING value. Two concrete windows: (a) broker has no book yet (`{book:null}` from /exec-book) — the New Order ticket is still fully usable; (b) the broker Durable Object caches the last book forever (execution-broker/src/index.js /book reads storage, never cleared on agent disconnect), so right after arming (arm_live.bat + agent restart) the site keeps showing the PRE-arm book with mode "dry-run" until the armed agent's first successful book_snapshot push (~10-20s, or indefinitely if the read-only snapshot subprocess fails while the command socket stays up). The banner also ignores state.status.online and the book age it already computes for the conn bar.

**Failure scenario.** User runs arm_live.bat and reloads the Execution tab a few seconds later. Broker still serves yesterday's cached book with mode="dry-run" (or {book:null}). Banner: green "DRY-RUN MODE — nothing is transmitted". User fills the ticket and clicks Send to "preview"; confirm says "Dry-run: place BUY 692 USO @ 104.80 ...". They click OK. The armed agent's _live_eligible passes and execute_order.py transmits a REAL bracket to IBKR. The user only learns it was live from the Activity table afterwards.

**Suggested fix.** Treat unknown as dangerous: when state.book is null, book.at is older than ~2 poll cycles, or status.online is false while armed state is unknown, render an amber "MODE UNKNOWN — assume LIVE" banner and make actionLead() say "MODE UNKNOWN (may be LIVE)". Optionally have the broker /status echo the agent's armed flag from the heartbeat instead of only from the book.

**Verifier (high confidence).** Confirmed line-by-line. site/assets/execution.js:130 defaults the mode banner to "dry-run" when state.book is null or stale, and lines 323-324 drive every confirm dialog off the same test; poll() (line 113) nulls the book on any /exec-book miss and nothing gates the ticket on book freshness or status.online. functions/exec-command.js:33 does stamp dry_run:true, but exec_agent.py _handle_command (lines 322-371) never reads cmd["dry_run"] — live-vs-dry is decided solely by _live_eligible (lines 29

#### 2. Blank stop/target fields coerce to price 0 (`Number("")` === 0) and pass every gate — BUY with a cleared stop produces no UI warning and a 0.00 protective stop at IBKR

`site/assets/execution.js:434` (dimension: exec-frontend)

ticketPayload() lines 433-435 sends `stop: Number(val("f_stop"))`, `target: Number(val("f_target"))` — an emptied input becomes the real number 0, not null (the UI has no "no target" affordance even though the whole downstream stack supports target=None). The readout's ordering check (lines 401-402) passes for the dangerous cases: BUY with stop=0 satisfies `stop < entry && entry < target` so NO warning is shown; SELL with target=0 satisfies `target < entry && entry < stop`. sendTicket() (437-452) never blocks on readout warnings anyway. Agent-side gates also accept 0: exec_agent._validate checks `stop < entry` (0 < entry passes; the risk-vs-NLV check tops out at LIVE_MAX_NOTIONAL=$25k so it passes on the primary account), and execute_order._do_entry_bracket's post-tick-snap ordering check passes identically, then builds StopOrder(SELL, qty, 0.0).

**Failure scenario.** User clears the Stop field on a BUY stock bracket (or clears Target on a SELL, intending "no target"). Readout shows no warning; confirm shows "(stop 0, target 123.21)" which is easy to miss. Armed agent transmits the chain. If a time-stop leg is present it carries the transmit flag, so the parent goes live and fills while the 0.00 stop child is rejected by IBKR — a live position with no working protective stop. On the SELL/blank-target variant, a BUY LMT @ 0.00 child reaches IBKR (a VALID resting order on instruments that permit zero/negative prices, e.g. CL/NG futures).

**Suggested fix.** In ticketPayload send null for empty inputs (`val(id) === "" ? null : Number(val(id))`), require a non-null positive stop client-side, add an explicit "no target" checkbox, and make sendTicket refuse to submit while the readout warn string is non-empty.

**Verifier (high confidence).** Traced the full path end-to-end; every claim checks out.

Frontend (site/assets/execution.js): (1) ticketPayload lines 433-435 send `stop: Number(val("f_stop"))`, `target: Number(val("f_target"))` — a cleared input yields `Number("")` === 0, and line 435 shows the null-vs-0 treatment IS applied to time_stop/expiry (`val(...) || null`) but not to prices, so this is an oversight, not a convention. (2) Readout ordering checks at lines 401-402 pass exactly as claimed: BUY with stop=0 satisfies `stop

#### 3. Idempotency key is minted per-request, so a retry/double-click submits a duplicate live order

`functions/exec-command.js:30` (dimension: exec-api-broker)

The command envelope's `id` (the documented idempotency key — see site_execution_schema.md line 17 "id: uuid — agent dedups") is generated fresh with `crypto.randomUUID()` on every POST to /exec-command. The client (site/assets/execution.js sendCommand, line 458) sends only {type, account, payload} with no stable idempotency token. The agent dedups by `id`, but two submissions of the same intent get two different ids, so the agent sees two distinct commands and executes both. There is no dedup at the broker either (execution-broker/src/index.js /command just unshifts + pushes). The agent is armed (LIVE_TYPES includes entry_bracket/flatten per MEMORY), so this is a real-money duplicate.

**Failure scenario.** User clicks 'Send order' for an entry_bracket; the /exec-command fetch stalls or the broker is slow and the browser reports a timeout (sendCommand catch sets 'error: ...'), but the command actually reached the armed agent and a live bracket was placed. Seeing 'error', the user clicks Send again -> new UUID -> agent does not recognize it as a dup -> a SECOND live bracket is transmitted -> doubled position.

**Suggested fix.** Generate the idempotency id client-side per user intent (once per ticket/confirm) and pass it through unchanged; have the broker dedup by id in recent_commands as well, so a resubmit of the same intent is a no-op rather than a second order.

**Verifier (high confidence).** Finding confirmed on every leg. (1) exec-command.js:30 mints id=crypto.randomUUID() per POST; execution.js sendCommand (line 454) sends only {type, account, payload} with no client-side idempotency token, so a retry of the same intent gets a new id. (2) The agent's dedup (exec_agent.py:342, _SEEN) keys on id, so it only blocks redelivery of the identical envelope, not a resubmitted intent. (3) The broker /command handler (execution-broker/src/index.js:66-82) is a self-described dumb relay: unshi

#### 4. Flatten matches position and cancels working orders by SYMBOL only (sec_type/expiry ignored) and always cancels ALL exits even for a partial trim — wrong contract can be closed and remainders left naked

`C:/Users/McKinley Slade/OneDrive/trading_ibkr/execute_order.py:262` (dimension: exec-local-agent)

The site sends {symbol, sec_type, expiry, fraction} (execution.js:329) but _do_flatten drops sec_type/expiry: position lookup (line 262) is next() over ib.positions() matching symbol only, and _orders_for_symbol (lines 63-67) cancels every non-terminal order whose contract.symbol matches. Two consequences: (a) holding two contract months of the same futures root (roll week), or a stock and future sharing a root, flatten closes whichever position next() yields first — possibly not the row the user clicked — while cancelling BOTH contracts' protective stop/time legs; the not-closed contract is left naked with no stop and nothing re-arms it. (b) Trim½ (fraction=0.5, execution.js:179) also cancels ALL exits first, then closes half (line 290) — the remaining half has no stop, no target, and no time exit, so it never auto-closes at hold end; every backtest convention assumes exits stay attached.

**Failure scenario.** During an MES roll the book holds MES 202609 (long 2) and MES 202612 (long 1), each with its own stop. User clicks Flatten on the 202612 row: agent cancels all 4 MES working orders, ib.positions() yields 202609 first, execute_order market-sells the 202609 contracts and reports success — the 202612 long is still open with its stop cancelled, unprotected overnight.

**Suggested fix.** Match the position on (symbol, secType, lastTradeDateOrContractMonth) from the payload; restrict _orders_for_symbol to the same conId/expiry; for fraction<1 either reduce exit-leg quantities instead of cancelling, or re-attach exits to the remainder.

**Verifier (high confidence).** Confirmed by line-by-line trace. (a) execute_order.py:_do_flatten (line 262) matches ib.positions() by contract.symbol only via next(), and _orders_for_symbol (lines 63-67) cancels every non-terminal order matching the symbol — the payload's sec_type/expiry (sent by site/assets/execution.js:329) are never read anywhere in _do_flatten; the post-cancel position re-read (lines 281-282) repeats the same symbol-only lookup. Upstream, exec_agent.py's _find_position is also symbol-only, so no layer dis

#### 5. _execute_live's 30s timeout abandons but does NOT kill the execute_order subprocess — order still transmits while the UI reports ERROR, inviting a duplicate resubmission

`C:/Users/McKinley Slade/OneDrive/trading_ibkr/exec_agent.py:316` (dimension: exec-local-agent)

asyncio.wait_for(proc.communicate(), timeout=30) cancels the wait on timeout but leaves the child running (contrast _fetch_book, which proc.kill()s on timeout, lines 417-422). _do_flatten routinely needs >30s in the cancel-first path: reqAllOpenOrders sleep 1.5s + connect-as-owner up to 8s + 2.5s sleeps per owner + _confirm_cancelled up to 4x1.9s + fill poll up to 12s + the initial connect up to 8s. On timeout the agent replies state='error' ('execute subprocess error: TimeoutError') while the orphaned child completes the cancel+close (or places the entry bracket). The broker/site show ERROR with no fill info; _SEEN idempotency is (a) in-memory only and (b) keyed on command id, and every site retry mints a fresh uuid (exec-command.js:30), so a natural retry re-executes in full. For flatten the second run usually lands on 'already flat/no position' (self-limiting), but for entry_bracket a retry places a second identical live bracket — double position, double stops.

**Failure scenario.** TWS is sluggish; an entry_bracket takes 32s inside execute_order (slow connect + qualifyContracts). Agent replies ERROR at t=30s; the orphan transmits the bracket at t=32s. User sees ERROR in Activity, clicks Send order again; second bracket transmits. Account is now 2x the intended size with two independent OCA groups.

**Suggested fix.** On TimeoutError kill the subprocess before returning; raise the timeout above _do_flatten's worst case (or make it per-type); persist executed command ids to disk; better, have execute_order write a start-marker so an ambiguous timeout is reported as 'UNKNOWN — check TWS' rather than a clean error.

**Verifier (high confidence).** Confirmed by line-by-line trace. exec_agent.py:316 uses asyncio.wait_for(proc.communicate(), timeout=30) and the except at line 318 returns an error dict WITHOUT killing the child — asyncio.wait_for cancels only the communicate() wait, not the subprocess. The intended pattern exists 100 lines down: _fetch_book (lines 415-422) explicitly proc.kill()s on TimeoutError. execute_order.py never re-checks expires_at, so the orphan transmits whenever it finishes. _do_flatten's worst case genuinely excee

#### 6. Site equity curves book Stop/Target exits at the exit-day close, not the realized fill — verified +$265k (7.4%) optimism vs the ledger's own trades

`pages/strat_backtester.py:2216` (dimension: site-payload-contract)

get_daily_mtm_series values a trade over `all_dates >= entry_date & <= exit_date` using ffilled closes, so a trade's cumulative curve contribution telescopes to (exit-day Close − entry_price)·shares. The ledger's realized PnL is (exit FILL − entry)·shares, where the fill is the gap-aware stop/target price (the 2026-06-27 stop-fill convention). There is no terminal reconciliation, so for all 435 Stop + 539 Target trades the curve permanently absorbs (Close_exitday − fill)·shares. Measured on the current local payloads: sum(trades PnL_flat_750k) = $3,575,271 vs sum(backtest_daily_pnl.pnl_flat) = $3,840,549 → +$265,278 (+7.4%); compounded basis +$17.8M. This flows into strategy_daily.json via build_site.py build_strategy_daily (line 205) and total_flat/equity_compounded via build_trade_ledger.py (lines 223-229), and portfolio.js's curveExact() path presents these curves as 'exact', so the site's Total PnL / CAGR / Sharpe / MaxDD KPIs are inflated and disagree with the trade-log sum on the same page. This silently reintroduces, in the curve, the same optimism the documented stop gap-fill convention removed from the ledger (the convention doc covers only the per-trade fill, not the MTM series). daily_portfolio_report and the correlation payload use the same function.

**Failure scenario.** OLV long stopped intraday at min(stop, open) − 13 bps on a gap-down that bounces: fill near the low, close 1.5% higher. The daily curve books the exit day at the close, crediting the strategy 1.5%·notional that was never realized; over the full book the site's equity curve ends $265k (flat basis) above the sum of its own trade log, and Sharpe/CAGR shown to the user are computed off the inflated series.

**Suggested fix.** On the exit day, book (exit_fill − prior close)·shares instead of the close-to-close diff (exit fill = trade's Exit Price), or add a terminal correction of (trade PnL − curve-implied PnL) on the exit date. Then regenerate backtest_daily_pnl.parquet and strategy_daily.json.

**Verifier (high confidence).** Confirmed, not refuted. (1) Code trace: get_daily_mtm_series (pages/strat_backtester.py:2144-2226) books first-day PnL as (Close − entry_price)·shares and then close-to-close diffs through exit_date inclusive; the trade's contribution telescopes to (Close_exitday − entry)·shares with no reconciliation to Exit Price. Producers build_trade_ledger.py:223-229 and build_site.py:198-224 pass it straight through to backtest_daily_pnl.parquet and strategy_daily.json. (2) Empirics reproduced exactly: sum

#### 7. Single-day open-limit branch hardcodes 0.5 ATR offset — 'Limit (Open +/- 0.25 ATR)' strategies are backtested at twice the live offset

`pages/strat_backtester.py:1689` (dimension: engine-correctness)

`_limit_mult = 0.75 if '0.75' in entry_type else 0.5` has no 0.25 case. Three book strategies use 'Limit (Open +/- 0.25 ATR)' (Weak Close Decent Sznls, SPY QQQ MonFri Reversion, Monday Dip). Live, daily_scan.py:1299-1301 stamps Offset_ATR_Mult=0.25 and order_staging places the REL_OPEN limit at open−0.25·ATR; the UI backtester (pages/backtester.py:1320) also models 0.25 as a distinct mode. But this engine — which drives the full-history ledger (scripts/build_trade_ledger.py → site) and daily_portfolio_report.py — fills those trades only at open−0.5·ATR. Empirically confirmed with a synthetic run: with ATR=2 and T+1 open=100, a bar with low 99.2 (touches the true 99.5 limit) records NO fill; a bar with low 98.9 records a fill at 99.0 instead of 99.5. Result: the ledger misses a large fraction of real live fills (untracked live trades) and books every recorded fill 0.25 ATR better than live gets, inflating per-trade edge, MTM equity, and the compounding sizing path for those three strategies.

**Failure scenario.** Monday Dip long signal, ATR=2, T+1 open 100, day low 99.30. Live REL_OPEN limit 99.50 fills and the position exists at IBKR; the backtest/ledger/portfolio report shows no trade at all. Conversely on a deeper dip the ledger books entry at 99.00 vs the live fill at 99.50 — 0.25 ATR (~0.25R) of phantom edge on every recorded trade.

**Suggested fix.** Parse the multiplier like the persistent branch does (check '0.25' first, then '1 ATR'/'0.75'/'0.5'), or share one offset-parsing helper with daily_scan.get_entry_type_short ordering ('0.75' → '0.25' → '0.5' → '1 ATR'). Then rebuild the ledger and re-validate the three strategies' stats at the true 0.25 offset.

**Verifier (high confidence).** Confirmed line-by-line. (1) strategy_config.py lines 258/983/1162 give Weak Close Decent Sznls, SPY QQQ MonFri Reversion, and Monday Dip entry_type 'Limit (Open +/- 0.25 ATR)'. (2) In process_signals_fast (pages/strat_backtester.py), line 1460 sets is_limit_open_atr=True and line 1461 is_persistent=False for that string, routing to the single-day branch at 1688. Line 1689 (`_limit_mult = 0.75 if '0.75' in entry_type else 0.5`) has no 0.25 case, so the fill test at 1694-1698 uses open−0.5·ATR. (3

#### 8. PM master-prices cron runs 30 min BEFORE the close in winter, writing a partial intraday bar to R2 as the canonical daily close

`.github/workflows/update_master_prices.yml:19` (dimension: pipeline-gha)

The PM cron '30 20 * * 1-5' is commented as "4:30 PM ET, 30 min after close", but that is only true during EDT. During EST (early Nov to mid-Mar), 20:30 UTC = 3:30 PM ET, 30 min before the close. This is the ONE trigger path that deliberately omits --exclude-today (step 'Determine update args', line 100), so yfinance's in-progress daily bar (Close = last trade ~3:30 PM, High/Low/Volume incomplete) is appended to master_prices.parquet and uploaded to R2 as today's final bar. Same-evening consumers all trust it: portfolio_report (21:30 UTC = 4:30 PM EST), the daily_screener PM bookend (22:00 UTC = 5:00 PM EST) which clears+rewrites the Order_Staging and Overflow tabs with signals and Close±k*ATR limit prices computed off the 3:30 snapshot (daily_scan.py:2051-2054 sets expected_data_date = today for any post-open run, and the intraday-partial volume relaxation at line 2043 is OFF because it's past 16:00 ET), the deploy-site ledger rebuild, and ml_score at 22:25. The AM 4:17 refetch normally repairs the bar (keep='last' dedupe) and the 4:47 scan rewrites the tabs before order_staging reads them, so live orders are only wrong when the AM chain fails — but every winter weekday evening the R2 cache, the staged tabs, the portfolio email, the site, and the ML scores are silently built on a pre-close snapshot.

**Failure scenario.** Any EST-season weekday with a move in the last 30 min (e.g., FOMC-day 3:45 PM selloff): OVS fires (or fails to fire) off the 3:30 close, and a limit at 3:30-Close minus 0.25*ATR-with-partial-High/Low is staged to the Overflow/Order_Staging tabs. That night the local trigger machine is off and the 10:30 UTC fallback is queue-lagged past staging time -> order_staging.py submits the mispriced order to IBKR at pre-market.

**Suggested fix.** Move the PM cron to 21:30 UTC (4:30 PM EST / 5:30 PM EDT, post-close year-round) and shift the dependent 22:00 screener/20:45 intraday crons accordingly, or gate the no-exclude-today branch on an in-script check that now_ET >= 16:05; alternatively always fetch with --exclude-today and add a dedicated post-close ET-aware trigger.

**Verifier (high confidence).** Finding verified line-by-line. update_master_prices.yml:19 uses a fixed-UTC cron '30 20 * * 1-5'; during EST (Nov-Mar) that is 15:30 ET, 30 min BEFORE the close, contradicting the line-16 comment and CLAUDE.md's "pulls today's close" intent. Lines 100-104 confirm this is the only trigger path without --exclude-today, so scripts/update_master_prices.py (plain yf.download at line 47, dedupe keep='last' at 184, R2 upload at 205) writes yfinance's in-progress 15:30 bar to R2 as today's final bar. da

#### 9. FMP server errors are classified as 'ticker has no earnings', so a partial FMP outage silently deletes tickers from the earnings calendar and disables the OVS earnings blackout for them

`scripts/build_earnings_calendar.py:99` (dimension: pipeline-gha)

fetch_ticker returns [] (legit-empty) for any non-200/non-429 HTTP status (line 99) and for 200-with-dict error payloads such as quota-exceeded messages (line 94). Only full network exceptions after retries return None (counted as failure). build_calendar is a from-scratch full rebuild every weekday (21:30 UTC GHA + local task): it writes and uploads the parquet to R2 unconditionally as long as ANY rows came back (the only guard, line 203, aborts on zero rows). There is no row-count or ticker-coverage sanity check against the previous parquet. Downstream, the OVS blackout is NaN-as-True by design (CLAUDE.md): a ticker absent from earnings_calendar.parquet passes the +/-10-trading-day blackout in daily_scan and strat_backtester. So a degraded FMP day (5xx for a subset, or quota exhausted mid-run so the last N hundred tickers get dict responses) silently shrinks the calendar, and every OVS signal on a dropped ticker sails through the earnings filter the next morning.

**Failure scenario.** FMP quota exhausts after ticker 500 of ~1060 on Tuesday's 21:30 UTC build -> parquet written and uploaded with ~560 tickers missing -> Wednesday 4:47 AM scan: OVS fires on NVDA (vol spike) 5 trading days before its earnings; blackout lookup finds no NVDA rows -> pass-through -> 40 bps path-1 OVS order staged and submitted into earnings week — exactly the trade class the filter exists to block (the 2026-04-29 validation showed 12 of 13 OVS signals on one date were blackout kills).

**Suggested fix.** Treat non-200 statuses and dict responses as failures (return None); before writing/uploading, compare ticker coverage and row count to the existing parquet and abort (keep last good R2 copy) if coverage drops more than a small tolerance (e.g. >2%).

**Verifier (high confidence).** Traced end-to-end; the finding is accurate and the code behaves exactly as claimed.

1. Error-as-empty classification (scripts/build_earnings_calendar.py, fetch_ticker lines 89-105): a 200 response whose JSON is a dict (FMP error payload) returns [] at line 94; ANY other non-429 status (500/502/503/402/403) returns [] at line 99 with NO retry (single transient 5xx on attempt 1 immediately classifies the ticker as legit-empty). Only requests-level exceptions after 3 retries, or persistent 429s, r

#### 10. Live sizing consumes rd2_fragility.parquet with no staleness bound, despite comments claiming it 'fails closed if stale'

`daily_scan.py:2138` (dimension: risk-ml)

daily_scan sizes EVERY non-OVS order by a fragility multiplier (0.10x-1.25x, applied at line 2390-2391) taken as frag_series.iloc[-1] with zero check on the date of that last row. The dial-filter gate (daily_scan.py:1087) likewise uses reindex(method='ffill') unbounded, while its comment (line 1066) and strat_backtester.py:479 both claim the code 'fails closed if missing/stale' — staleness is not handled anywhere. The producer chain is fragile: risk_report.yml runs daily_risk_report.py (which sends the email FIRST) and only afterwards commits data/rd2_fragility.parquet with a bare git push (no pull/rebase, no retry, risk_report.yml:37-43), so a push race with any other bot commit, or any yfinance failure inside refresh_all_data (RuntimeError when SPY missing), leaves the committed file stale while the operator still receives a normal-looking risk email.

**Failure scenario.** risk_report.yml breaks (yfinance rate-limit or git push rejection) the week a selloff starts. The committed 63d score stays at ~20 (calm), so the next AM daily_scan applies ~1.05-1.25x size boost when the current dial would be ~90 (correct multiplier ~0.22x): every staged order reaches IBKR ~4.5-5x larger than the sizing policy intends, silently, for as long as the producer is broken.

**Suggested fix.** In daily_scan's fragility block: read the last index date and fail to a conservative default (e.g. 1.0x, or FRAG_MIN_MULT) with a loud email warning when it is older than N trading days; apply the same age bound in the dial-filter gate so 'fail closed on stale' matches the comment. In risk_report.yml, git pull --rebase before push and fail the job loudly if the commit doesn't land.

**Verifier (high confidence).** Confirmed by direct trace. daily_scan.py:2131-2146 takes frag_score = frag_series.iloc[-1] from data/rd2_fragility.parquet with no check on the last index date (guards exist only for missing file/column/empty/exception, all -> 1.0x); the multiplier is applied to every non-OVS order at 2390-2392. The dial-filter gate (1068-1097) uses reindex(method='ffill') unbounded at 1087, so arbitrarily stale data passes, while the comment at 1066-1067 (and pages/strat_backtester.py:478-479) claims it 'fails 

#### 11. SI spec is actually Micro Silver (1,000 oz) mislabeled as full-size — 5x risk understatement and ambiguous contract at transmit

`site/assets/futures_specs.json:88` (dimension: options-futures)

futures_specs.json lists SI as multiplier 1000 / tick_value $5 / tier "full". COMEX full-size Silver (SI) is 5,000 oz, $25/tick. The agent's authoritative copy (C:\Users\McKinley Slade\OneDrive\trading_ibkr\contract_reference.json) has the identical wrong row, and its captured trading_class field is "SIL" — proof the generator (contract_reference.py lines 96-98: reqContractDetails for symbol SI on COMEX returns BOTH the SI 5,000-oz and SIL 1,000-oz trading classes, then `details.sort(...)` + `details[0]` picks whichever expiry sorts first, here the micro). This wrong multiplier feeds: (a) futures_sizing.size_futures (risk_per_contract line 188), (b) exec_agent._validate risk/notional caps (line 243), (c) execute_order._do_entry_bracket's LIVE notional cap (line 220), and (d) the site ticket risk readout (execution.js line 408) and groupPnl best/worst. Compounding it, execute_order.py line 202 builds `Future(sym, fut_expiry, exch)` with no tradingClass, so an SI order is ambiguous between the 5,000-oz and 1,000-oz contracts (qualifyContracts' return is ignored at line 240); futures_front.py similarly unions expiries across both classes, so the auto-filled month may exist only for the micro.

**Failure scenario.** User sizes SI: entry 30.00, stop 29.50, risk $10,000 -> tool reports risk/contract $500 and returns 20 contracts. Real full-size SI risks $2,500/contract -> $50,000 at risk, 5x budget, with total notional ~$3.0M shown as ~$600k. The live gate's notional cap is computed with the same wrong 1000 multiplier so it does not catch it; if IBKR resolves the ambiguous SI contract to the full-size class, a 5x-oversized silver position is opened.

**Suggested fix.** In contract_reference.py, filter details to tradingClass == the requested symbol (or explicitly request each trading class) before taking [0]; regenerate contract_reference.json and site/assets/futures_specs.json (SI -> multiplier 5000, tick_value 25). In execute_order.py, set tradingClass on the Future and abort if qualifyContracts does not return exactly one contract.

**Verifier (high confidence).** CONFIRMED end to end; only the transmit magnitude in the narrated scenario is overstated given current env caps.

Verified facts:
1. site/assets/futures_specs.json lines 86-92: SI = multiplier 1000, min_tick 0.005, tick_value 5.0, tier "full". Real COMEX full-size SI is 5,000 oz ($25/tick); multiplier 1000/$5 tick is the 1,000-oz SIL contract. The file's own QI row (2,500 oz miNY, $31.25/tick) sits alongside, so the SI row cannot be the full-size.
2. Smoking gun in the authoritative source: C:/U

### Severity: MEDIUM

#### 12. Cancel button silently escalates to symbol-wide cancel when perm_id is 0 — one click can cancel every working order on the ticker, including an open position's protective stop

`site/assets/execution.js:333` (dimension: exec-frontend)

orderRow() line 213 emits `execCancel(${o.perm_id || 0}, ${o.order_id || 0}, ...)` and execCancel() line 333 sends `{scope:"symbol", symbol}` whenever permId is falsy — even though a nonzero order_id is in hand and the agent's _do_cancel (execute_order.py:333-341) supports order_id matching. book_snapshot.py stamps `perm_id: o.permId or 0`, and permId is 0 for an order the 20s snapshot catches before TWS assigns it (e.g. a bracket just staged from this very page). The agent's symbol branch then cancels ALL non-terminal orders for that symbol via _cancel_via_owners. Meanwhile the confirm dialog (line 332) always reads "cancel order {orderId} ({symbol})" — it never discloses that the actual command is symbol-scoped.

**Failure scenario.** USO has an open position protected by a working stop/target, plus a just-submitted add-on entry order whose book row carries perm_id=0. User clicks Cancel on the add-on leg; confirm says "LIVE — cancel order 7 (USO, primary)?". The agent receives {scope:"symbol", symbol:"USO"} and cancels the add-on AND the position's stop and target — the live position is left naked with no error shown.

**Suggested fix.** Prefer order scope whenever either id is nonzero: `permId || orderId ? {scope:"order", perm_id: permId||null, order_id: orderId} : {scope:"symbol",...}` (agent already falls through perm_id -> order_id). If symbol scope is ever used, the confirm must say "cancel ALL working orders for USO".

**Verifier (medium confidence).** Code trace confirms every mechanical claim. site/assets/execution.js:213 passes `o.perm_id || 0` and :333 sends `{scope:"symbol", symbol}` whenever permId is falsy, discarding the nonzero order_id it has in hand; the confirm at :332 always says "cancel order {orderId}" even when the command is symbol-scoped. book_snapshot.py:91 stamps `perm_id: o.permId or 0`. execute_order.py:_do_cancel (333-341) falls through perm_id -> order_id -> symbol; the symbol-scope payload carries no order_id, so it la

#### 13. groupPnl best/worst counts entry orders as exits and ignores leg action — wrong PnL dollars on the order summary rows

`site/assets/execution.js:250` (dimension: exec-frontend)

In the on-position branch, line 250 sets `exits = legs` (every order grouped under the symbol), so a working add-on/second entry LMT is summed into `best` as if it were a profit target, priced against the existing position's derived entry and with the POSITION's sign rather than the order's action (lines 261-270 never look at o.action). In the pending branch, line 252 `legs.find(parent_id===0)` picks only the FIRST parent; with two pending brackets on one ticker the second entry order is treated as an exit of the first, and both brackets' stops/targets are computed off the first bracket's entry price and side.

**Failure scenario.** Long 100 AAPL @ 200 with stop 190 / target 220 working, plus a pending add-on BUY 50 LMT @ 195 bracket (stop 188, target 215). All legs group under "On open positions"; best = (220-200)*100 + (195-200)*50 + (215-200)*50 = wrong (the BUY entry limit contributes -$250 as a fake exit and the add-on's target is based on the wrong entry), worst double-counts stops against a single basis. User trims or holds based on materially wrong best/worst dollars.

**Suggested fix.** Filter exits to child legs / orders whose action is opposite the position (or entry) side: in the on-position branch drop legs with `(o.parent_id||0)===0 && action matches position side`; in the pending branch compute per-parent groups (key legs by parent_id) instead of one basis per ticker.

**Verifier (high confidence).** Confirmed by line-by-line trace of site/assets/execution.js. (1) On-position branch: renderOrders (lines 309-311) buckets orders by symbol only, so an add-on entry bracket on a ticker with an open position lands in the on-pos group; groupPnl line 250 sets exits = legs with no parent/action filtering, and the loop at 261-269 never reads o.action — a BUY add-on entry LMT is summed into `best` as a fake profit target ((195-200)*50 = -$250 in the cited scenario), and the add-on's child stop/target a

#### 14. No endpoint independently verifies caller identity (Cf-Access-Jwt-Assertion) — a single Cloudflare Access misconfig exposes an armed live-order endpoint

`functions/exec-command.js:19` (dimension: exec-api-broker)

onRequestPost (and every exec-*.js Function, plus the Worker in execution-broker/src/index.js) authenticates nothing about the human caller. The Functions rely 100% on the Cloudflare Access wall being correctly configured in the dashboard; none of them validate the `Cf-Access-Jwt-Assertion` header against the Access public keys. The Worker's /command only checks `Authorization: Bearer STATUS_TOKEN`, and that token is held by the Pages Function itself — so anyone who can reach the Pages URL gets commands signed with STATUS_TOKEN and relayed to the armed agent for free. There is zero defense-in-depth at the code layer for a real-money order path.

**Failure scenario.** An Access policy is edited/removed, the app is deleted, a bypass/service-token rule is added, or the /exec-command route simply isn't covered by an Access application. The Function is now publicly reachable; an unauthenticated POST {type:'flatten', account:'primary', payload:{symbol,...}} is HMAC-signed server-side and forwarded to the armed agent, which transmits a live market order.

**Suggested fix.** Verify the Cf-Access-Jwt-Assertion JWT (aud + issuer against the team's /cdn-cgi/access/certs) inside each Function before forwarding, and reject requests lacking a valid identity even if the network edge lets them through.

**Verifier (high confidence).** Facts confirmed line by line. functions/exec-command.js:19-52 signs commands server-side with env.STATUS_TOKEN and never validates Cf-Access-Jwt-Assertion; no functions/exec-*.js does. execution-broker/src/index.js:67 authenticates /command only with Bearer STATUS_TOKEN, the same token the Pages Function holds. The agent (OneDrive/trading_ibkr/exec_agent.py) _verify (line 80) passes on any Function-minted signature, and _live_eligible (297-306) checks only LIVE_ENABLED/account/type — it never re

#### 15. Broker fans each command out to ALL connected agent sockets — >1 live socket means duplicate execution

`execution-broker/src/index.js:81` (dimension: exec-api-broker)

`for (const s of sockets) s.send(JSON.stringify({type:'command', signed, sig}))` pushes the identical signed command to every WebSocket returned by getWebSockets(). The design assumes a single agent (BROKER_NAME='main'), but nothing enforces one socket. webSocketClose (line 225) only records disconnected_at; it never trims the socket set, and with app-level heartbeats replacing WS ping/pong (ping_interval=None was set to stop flapping), a dead/zombie socket has no keepalive to reap it. Per-agent-process dedup does not protect across two distinct live processes.

**Failure scenario.** The ExecAgent scheduled task's RestartOnFailure / At-logon / 5AM triggers briefly leave two live agent processes each holding a socket (or a reconnect leaves a still-live prior socket). A single entry_bracket POST is delivered to both -> each verifies the sig, passes expiry, dedups only against its own in-memory set -> two live bracket orders -> doubled position.

**Suggested fix.** Keep exactly one active agent socket (close/replace the prior on a new /agent upgrade) and send the command to that single socket only; reject a second concurrent agent connection.

**Verifier (high confidence).** CONFIRMED with one correction to the failure scenario's likelihood. Code behaves exactly as claimed: execution-broker/src/index.js:81 fans each signed command to every socket from getWebSockets(); the /agent handler (lines 37-47) accepts new upgrades without closing prior sockets; webSocketClose (line 226) only records disconnected_at; no server-side ping is configured and the agent disables client WS pings (exec_agent.py:442-446, ping_interval=None), so a half-open socket can coexist with a fre

#### 16. OVS Path-2 live sizing (0.15x, no aggregate cap) diverges from the backtest (0.2x + 1% daily cap); scanner-stamped Path1_Bps/Path2_Bps/Path2_Daily_Cap_Pct are read by nothing

`C:/Users/McKinley Slade/OneDrive/trading_ibkr/order_staging.py:116` (dimension: exec-local-agent)

New angles on the known P2 item: P2 was RESTORED live 2026-06-12 (not retired as CLAUDE.md's 2026-06-10 note says), but with a hardcoded OVS_PATH2_QTY_MULT=0.15 (= 6 bps of the 40 bps P1-sized scanner qty), while the ledger/backtest models P2 as path2_bps/path1_bps = 8/40 = 0.2x (pages/strat_backtester.py:1534-1537) PLUS a path2_daily_cap_pct=1% aggregate pro-rata cap (strat_backtester.py:1247-1272, 1339). order_staging has no P2 aggregate cap at all — the comment at line 610 still references 'the path-2 aggregate cap pass that runs after the per-row loop', which no longer exists (only the per-strategy 200bps and direction caps remain, both looser than 1%). daily_scan.py:1378-1380 still stamps Path1_Bps/Path2_Bps/Path2_Daily_Cap_Pct onto every OVS row 'so order_staging can compute the multiplier' (CLAUDE.md), but order_staging never reads any of them. Editing path2_bps in strategy_config therefore changes the backtest and the stamps but silently does nothing live.

**Failure scenario.** A heavy OVS morning stages 20 P2 small-gap rows: live total P2 risk = 20 x 6 bps x cycle 0.75 = ~$6.8k with no P2-specific cap and each trade 25% smaller than modeled; the ledger driving the site/reports assumes 8 bps per trade capped at 1% aggregate — realized OVS P2 PnL and risk systematically diverge from every published backtest number.

**Suggested fix.** Have order_staging derive the P2 multiplier from the row's Path2_Bps/Path1_Bps stamps and reinstate the Path2_Daily_Cap_Pct pro-rata pass (the _PathLabel column already tags P2 rows); or update strategy_config/backtest to 6 bps/no-cap and re-measure.

**Verifier (high confidence).** Every material claim traced and confirmed. Live (order_staging.py:116, 751-758): P2 restored 2026-06-12 with hardcoded OVS_PATH2_QTY_MULT=0.15 applied to the P1-sized scanner qty (0.15 vs the modeled path2_bps/path1_bps=8/40=0.2x, i.e. each live P2 trade 25% smaller than the ledger models). No P2 aggregate cap exists anywhere in order_staging — the line 610 comment referencing 'the path-2 aggregate cap pass that runs after the per-row loop' is stale; only the precedence drop, per-strategy 200bps

#### 17. Re-running the staging+entry chain across the open re-prices REL_CLOSE/REL_OPEN limits, defeating the exact-match duplicate guard — two live entry brackets for one signal

`C:/Users/McKinley Slade/OneDrive/trading_ibkr/eq_order_entry.py:204` (dimension: exec-local-agent)

The duplicate guard skips only an EXACT fingerprint match (symbol, side, qty, type, limit-to-2dp; lines 46-66, 200-212), and nothing ever cancels a prior run's entries. order_staging recomputes limits per run: the REL_CLOSE close-gap clamp (order_staging.py:645-668) depends on whether the live open was available and how far it gapped, and per-strategy/direction cap scaling (lines 1105-1149) can change Quantity between runs. A pre-open run (open unavailable -> unclamped limit) followed by a post-open re-run (clamped limit, or scaled qty) produces a different fingerprint, so eq_order_entry places a SECOND full bracket while the first multi-day GTD parent (OLV lives T+1..T+3) is still working. pa_order_entry.py has the identical pattern (lines 64-84, 283-295).

**Failure scenario.** 9:00 AM: chain runs while TWS is briefly disconnected — REL_CLOSE OLV limit placed unclamped at $33.50, GTD 3 days. 9:40 AM: user re-runs the chain to pick up rows that failed; the stock gapped, the clamp moves the limit to $31.20, qty rescaled — new fingerprint, second bracket placed. A deep flush fills both -> 2x the intended OLV position with two OCA groups.

**Suggested fix.** Guard on signal identity, not price: fingerprint (symbol, side, Strategy_Ref, Staged_Date) for GTC/GTD parents and cancel-and-replace the prior entry when the price/qty changed, or refuse to place when ANY open entry exists for the same symbol+strategy staged today.

**Verifier (high confidence).** Confirmed line-by-line. eq_order_entry.py:46-66/200-212: duplicate guard is an exact tuple match (symbol, side, qty, type, limit@2dp) against openTrades() parents; the docstring explicitly lets any different-priced/sized order in the same ticker through, and grep confirms no cancelOrder or position check anywhere on the entry chain (eq_order_entry.py, pa_order_entry.py, order_staging.py). order_staging.py recomputes both price and qty per run: the REL_CLOSE gap guard (645-668, apply_close_gap_gu

#### 18. seasonal_cross_section silently carries the last-known rank forward with no staleness gate — guaranteed wrong-window signals starting 2027-01-01

`scripts/seasonal_edge.py:105` (dimension: ideas-seasonal)

seasonal_cross_section() takes each ticker's most recent rank row <= asof (idxmax) with no limit on how old that row is, and scan_seasonal_tickets() (line 864, rk = float(cs.loc[t, col])) never checks the stamped Date. atr_seasonal_ranks.parquet is built through DEFAULT_YEAR = 2026 (build_atr_seasonal_ranks.py line 49) and no workflow bumps the year (rebuild_overflow_universe.yml also builds --full up to 2026). On the first 2027 asof, every ticker's 'current' rank is its 2026-12-31 row — a late-December seasonal profile built with midterm-cycle (2026%4==2) weighting — and it stays that way for all of 2027 while the cycle-blend elsewhere in the pipeline switches to phase 3. The realized-window confirm gates only partially protect: they still let a name through whenever the actual calendar window happens to clear the 0.60/0.667 hit bars, with the extreme rank (the primary selection signal) being from the wrong window.

**Failure scenario.** asof 2027-03-15: cs.loc['GLD','atr_sznl_21d'] returns the 2026-12-31 rank (say 92, a year-end gold window). detect_cross_asset treats March GLD as an extreme seasonal long, mints a ticket, it passes the March realized-window confirm by coincidence, gets graded A, displayed on the site, and (when staging is on) staged as a MOO order — driven entirely by a December rank.

**Suggested fix.** In scan_seasonal_tickets, drop names whose cs Date is more than a few sessions older than asof (or assert max(cs.Date) is within N trading days of asof and abort loudly); add an annual-rebuild reminder or a scheduled `build_atr_seasonal_ranks.py --years <next_year> --merge --upload`.

**Verifier (high confidence).** Confirmed by line trace. seasonal_edge.py:105-107 takes each ticker's most recent rank row <= asof with no staleness bound, and scan_seasonal_tickets (line 843 cross-section, line 864 rk lookup) never reads the stamped Date. Producer build_atr_seasonal_ranks.py:49/476 hardcodes DEFAULT_YEAR=2026 (--full = 2001-2026); rebuild_overflow_universe.yml runs --full and is workflow_dispatch-only; append_atr_seasonal_ranks.py reuses the existing year range — nothing produces 2027 rows. Caller daily_seaso

#### 19. Forward log never syncs from R2 before appending: fresh checkouts clobber (or silently discard) the accumulated track record

`scripts/seasonal_ideas_ledger.py:56` (dimension: ideas-seasonal)

load_log() reads only the local data/seasonal_ideas_log.parquet; append_emitted() then writes local and uploads to R2 (upload=True default), overwriting R2 wholesale. On any machine without the local parquet — the exact cross-machine move docs/seasonal_session_handoff.md describes, or GHA — the 'append-only' log is rebuilt from a single day's rows. Today in deploy_site.yml the 'Build daily seasonal ideas' step (line 86-90) has no R2_* env, so is_configured() is False: the GHA-emitted rows are written to the ephemeral runner and evaporate every run — the automated pipeline is accumulating no track record at all, while printing '[ledger] logged N tickets' as if it worked. If R2 creds are ever added to that step (or a local run happens on a fresh clone with creds), the upload replaces the multi-week R2 log with one day of rows. score_seasonal_ideas.py OUT (outcomes parquet, line 53/113) has the identical local-only-read + R2-overwrite pattern.

**Failure scenario.** User clones the repo on a second machine with R2 creds in env (per the handoff doc), runs daily_seasonal_ideas.py: load_log() finds no local file, append_emitted writes a parquet containing only today's ~5 tickets and uploads it to R2:seasonal_ideas_log.parquet, destroying the accumulated out-of-sample record the whole tracking feature exists to build.

**Suggested fix.** In load_log(), when the local file is missing and cache_io.is_configured(), download_to_local(R2_KEY, LOG_PATH) first (same for the outcomes parquet); in deploy_site.yml either pass R2 env to the ideas step (after the sync fix) or stop pretending the hook persists anything on GHA.

**Verifier (high confidence).** Confirmed by line-by-line trace. load_log() (seasonal_ideas_ledger.py:56-62) reads only the local parquet, never R2. append_emitted() (65-90) merges new rows into whatever local state exists and, with default upload=True, uploads the result wholesale to R2:seasonal_ideas_log.parquet — so an empty local file yields a one-day log that replaces the accumulated record. daily_seasonal_ideas.py:337-338 invokes it with defaults on every run. In deploy_site.yml the seasonal-ideas step (lines 86-90) has 

#### 20. Chart gallery PNGs are frozen at first render: --skip-existing never re-renders after a trade's real exit, so nearly every chart shows a wrong exit

`scripts/build_signal_charts.py:277` (dimension: site-payload-contract)

deploy_site.yml (line 83) runs `--all --upload --skip-existing` twice per trading day, and the ledger includes open trades (marked Exit Type='Time' at the last bar). So a trade's chart is first rendered within hours of its entry fill — while it is still open — with the exit line/date at that day's last bar, MAE/MFE computed over the partial hold, and the 63-day post-window absent. The skip check (`r2_key in existing_r2`) keys only on existence of the stable (strategy, ticker, signal-date) key, so after the trade actually exits days later (Stop/Target, different date, different R) the PNG is never re-rendered. charts.json is rebuilt nightly with the correct exit_type/exit_date/mae_r, so the gallery caption contradicts the image it sits above. Only the manual full_chart_rebuild dispatch flag fixes it; nothing scheduled does. The docs describe the stable-key/skip-existing design but not this steady-state staleness for every in-flight trade.

**Failure scenario.** OVS short fills Monday; Monday-night deploy renders SPXL_20260629.png showing an open trade 'exited (Time)' at Monday's bar with MAE −0.1R. Thursday it stops out at −1.4R. Every subsequent deploy skips the key; the gallery forever shows the Monday snapshot while the caption says Stop / −1.4R — a user reviewing stop behavior from the charts sees systematically benign, truncated trades.

**Suggested fix.** Track render freshness: skip only if the R2 object's LastModified is later than (exit_date + 63 trading days) for the trade; else re-render. cache_io.list_keys_with_meta already returns LastModified, so the incremental sweep needs no extra API calls.

**Verifier (high confidence).** Confirmed by line-by-line trace. (1) Open trades land in data/backtest_trades_full.parquet with Exit Type='Time' at the last bar: pages/strat_backtester.py:1801-1805 clamps exit_idx to len(df)-1; scripts/build_site.py:150-153 (open_mask) documents this exact convention. (2) scripts/build_signal_charts.py main() iterates the full ledger with no open-trade filter, so a trade is first rendered on the entry-day PM deploy (deploy_site.yml runs --all --upload --skip-existing after each scan) with the 

#### 21. open_mask uses `Time Stop >= today`: on the post-close build, a trade that time-exited at today's close is published as an open position

`scripts/build_site.py:158` (dimension: site-payload-contract)

The PM deploy (22:00 UTC) runs after update_master_prices' 20:30 UTC close pull, so the ledger's engine has already exited any trade whose Time Stop is today (Exit Date = today, Exit Type = 'Time', PnL final). open_mask still classifies it open because `Time Stop >= today` is inclusive. It then appears in positions.json (build_positions) with live stop/target levels, Shares, and Mkt_Value, and is excluded from the trade log (portfolio.js filters `!t.Open`) until the next morning's build. The `>=` is only correct for the AM build, where the last bar is yesterday and the trade genuinely is still open; correctness depends on data as-of, not wall clock.

**Failure scenario.** A 10-day hold times out and is sold MOC today. That evening the user reconciles the site's Open Positions against IBKR: the site shows the position still open (with a stop price implying working orders) while the account is flat, and the closed trade is missing from the trade log — every time-exit does this for one evening.

**Suggested fix.** Compare Time Stop to the ledger's data as-of instead of wall-clock today: open iff `Time Stop > df['Exit Date'].max()` (last bar the engine saw), keeping the Exit Type=='Time' guard. AM builds (last bar = yesterday) still classify today's time-outs as open; PM builds close them.

**Verifier (high confidence).** Confirmed by line-by-line trace. Engine (pages/strat_backtester.py:1801-1805, 1995-1997) books a hold expiring on the data's last bar as Exit Date=today, Exit Type='Time', Time Stop=today with final PnL at today's close. The PM deploy (deploy_site.yml:52-69) pulls master_prices from R2 after the 20:30 UTC close pull, so today's bar is present; GHA runs UTC so pd.Timestamp.today().normalize() at ~22:15 UTC equals the session date. open_mask (scripts/build_site.py:158-160) then evaluates Time Stop

#### 22. Fill verification infers wrong ATR offsets: 0.75-ATR open limits verified at 0.25, and persistent 0.5-ATR limits parsed from the dollar price string

`verify_fills.py:157` (dimension: engine-correctness)

classify_order Priority 1 and build_strategy_map both only know offsets 0.25 and 0.5. (a) OVS and ATR Extended Gap Up ('Limit (Open +/- 0.75 ATR)', both Short) produce Entry_Type_Short 'Open ±0.75 ATR', which matches NO Priority-1 branch and falls to build_strategy_map (line 118-119), whose only bump is `if '0.5' in entry_type: offset = 0.5` — so offset=0.25 instead of 0.75. (b) Persistent strategies produce Entry_Type_Short like 'LMT $102.31 GTC'; line 157 does `offset = 0.5 if '0.5' in ets else 0.25`, i.e. the offset depends on whether the literal substring '0.5' appears in the DOLLAR PRICE — 52wh Breakout and Sector BO (-0.5 ATR persistent) get 0.25 unless the price digits coincidentally contain '0.5' (e.g. '$30.55'), and OLV (-0.25) flips to 0.5 whenever its price string contains '0.5'. check_fill then recomputes the limit from signal_close/T+1-open ± offset·ATR (ignoring the stamped Limit_Price for REL_CLOSE/REL_OPEN), so the fill-truth statuses written back to Trade_Signals_Log are systematically wrong.

**Failure scenario.** OVS short: signal close 100, ATR 2, T+1 open 101, day high 101.8. Live limit 101+0.75·2=102.50 never touches — no position. verify_fills computes 101+0.25·2=101.50, sees high ≥ 101.50, writes FILLED @101.50 to the log. Symmetrically, a 52wh Breakout long whose forward low reaches close−0.25·ATR but not close−0.5·ATR is marked FILLED though the live order never filled — the reconciliation layer reports positions that don't exist at the broker.

**Suggested fix.** For REL_CLOSE/REL_OPEN use the row's stamped Limit_Price / Offset_ATR_Mult when present instead of re-deriving; otherwise parse offsets with the same ordered check daily_scan uses ('0.75' before '0.25' before '0.5' before '1 ATR') applied to entry_type_raw, never to the formatted price string.

**Verifier (high confidence).** Confirmed by line-by-line trace. (a) OVS / ATR Extended Gap Up ("Limit (Open +/- 0.75 ATR)") stamp Entry_Type_Short "Open ±0.75 ATR" (daily_scan.py:1527-1528); this matches no Priority-1 branch in classify_order (verify_fills.py:151-165 — contains none of MOC/MOO/PERS/GTC/REL_OPEN/'ATR LMT'/LOC/LMT), falls to build_strategy_map whose only bump is `if '0.5' in entry_type` (verify_fills.py:118-119), and "0.75" does not contain "0.5" → offset 0.25. check_fill REL_OPEN (lines 314-321) recomputes t1_

#### 23. Persistent-limit hold reduction is one trading day short of the live time-exit leg

`pages/strat_backtester.py:1738` (dimension: engine-correctness)

Both persistent fill loops set `hold_days = max(1, execution['hold_days'] - (i - signal_idx))` (lines 1658/1664/1671/1677 and 1738/1744/1751/1757), so a fill at T+k time-exits at signal+hold (probe confirmed: persistent T+1 fill, hold=10 → exit T+10). Live, daily_scan.py:2491-2494 sets Time_Exit_Date = signal + (1+hold) trading days for every non-MOC entry, and per CLAUDE.md order_staging's Exit_Condition_Time chain (Entry_Expire = Exit_Condition − (1+hold−fill) BDays = signal+fill BDays) resolves to the same signal+1+hold. A plain 'T+1 Open' entry in this engine exits at entry+hold = signal+1+hold (probe: T+11) — matching live — so the persistent branch is the outlier: the reduction should be (i − signal_idx − 1). Affects 6 of 12 book strategies (all 'Persistent' entries: OLV, LT Trend ST OS, St OS Sznl, 52wh Breakout, Sector BO, Indices Oversold Bounce): the ledger, site, and daily_portfolio_report model one fewer session of exposure per time-exited trade than the live TIME leg actually holds.

**Failure scenario.** OLV long fills T+1 with hold_days=10. Backtest/ledger books the time exit at T+10's close and daily_portfolio_report shows the position closed; the live TIME exit order (Exit_Condition_Time = signal+11) keeps the position through T+11 — one extra day of unmodeled market exposure on every persistent time-exit, and a systematic PnL/exit-date mismatch between the log and the sheet-driven live book.

**Suggested fix.** Decide which side is the convention. If live (signal+1+hold) is truth, change the reduction to `execution['hold_days'] - (i - signal_idx - 1)` in both loops (and re-measure, since the T+3 OLV evidence was computed under the current arithmetic); if the backtest is truth, shift daily_scan's exit_date build and the order_staging back-computation by one BDay.

**Verifier (high confidence).** Confirmed by line-by-line trace. Engine: pages/strat_backtester.py lines 1650-1679 / 1730-1759 set hold_days = max(1, hold - (i - signal_idx)) at the fill bar i; with entry_idx = i (line 1798) and max_exit_idx = entry_idx + hold_days (line 1801), every in-window persistent fill time-exits at signal + hold. The T+1 Open path leaves hold_days untouched and exits at signal + 1 + hold. Live: daily_scan.py:2488-2494 stamps Time_Exit_Date = signal + 1TD + hold*TD for all non-Signal-Close entries; orde

#### 24. GTC fill window ignores OLV's T+3 entry-expire — fills after live cancellation are marked FILLED

`verify_fills.py:292` (dimension: engine-correctness)

check_fill sets `last_check = exit_dt` (the sheet's 'Time Exit' = full hold expiry, ~T+11) for every GTC order. The 2026-06-24 OLV change cancels the live GTC entry at T+3 15:59 (Entry_Expire_Time; fill_window_days=3 in strategy_config, honored by the backtest engine and order_staging), but verify_fills was never updated and the Trade_Signals_Log row doesn't carry Fill_Window_Days (daily_scan only stamps it on staging rows, line 1366). So the verifier keeps checking the limit for ~7 extra sessions after the live order no longer exists.

**Failure scenario.** OLV signal Monday; limit untouched T+1..T+3, live order expires T+3. Price then breaks down through the limit on T+7. verify_fills sees low ≤ limit within its window and writes FILLED with a T+7 fill date — the log asserts a position that was never opened at the broker, exactly the class of stale-GTC fill the T+3 window was built to avoid.

**Suggested fix.** Stamp Fill_Window_Days (or Entry_Expire_Date) onto the Trade_Signals_Log signal_dict in daily_scan, and in check_fill bound `last_check` for GTC orders at signal + fill_window trading days when present.

**Verifier (high confidence).** Confirmed by line-by-line trace. verify_fills.py lines 291-292 set last_check = exit_dt (the sheet's 'Time Exit') for every GTC limit order; the fill scan (lines 296-299, 344-351) marks FILLED on any low <= limit within T+1..last_check. OLV is entry_type "Limit Order -0.25 ATR (Persistent)" (strategy_config.py:407) with fill_window_days=3 (line 453); its Entry_Type_Short is "LMT $xx.xx GTC" (daily_scan.py:1537-1541) so classify_order (verify_fills.py:155-158) routes it to the GTC branch. The Tra

#### 25. Step B3 refreshes the wrong earnings artifact: overflow earnings coverage will silently go stale, defeating the OVS earnings blackout

`.github/workflows/rebuild_overflow_universe.yml:101` (dimension: scan-parity)

Step B3 runs `python scripts/build_earnings_calendar.py --with-symbol-master`, which writes the union (CSV_UNIVERSE + symbol_master, ~3k names) to data/earnings_calendar.parquet and uploads it to the PRODUCTION R2 key 'earnings_calendar.parquet' (scripts/build_earnings_calendar.py:263,280-291). Two problems: (1) the weekday production job (.github/workflows/build_earnings_calendar.yml:48) runs the script with NO flags, rebuilding CSV_UNIVERSE-only and re-uploading to the same key, so every Monday evening the overflow names are wiped again; (2) the file the scan actually relies on for overflow names — data/earnings_calendar_overflow.parquet, which earnings_filter.load_earnings_dates_map() unions in (earnings_filter.py:96-101) and daily_screener.yml pulls from R2 (line 105) — is NEVER refreshed by this workflow. The script has a dedicated `--overflow-staging` flag for exactly this (its help text warns 'the daily CSV_UNIVERSE rebuild can't wipe these'), and it is not used. The overflow staging parquet is frozen at its 2026-06-05 bootstrap content, whose FMP forward earnings dates run out after ~1-2 quarters.

**Failure scenario.** OVERFLOW_UNIVERSE_ACTIVE is flipped to 1 (planned promote) and this workflow's cron is enabled. By ~Oct 2026 the staging parquet's forward earnings dates for overflow small-caps are exhausted. An overflow OVS signal fires 3 days before ticker XYZ's earnings: signed_offset() only sees old historical dates (offset large, positive), in_blackout() returns False, and the Earnings_Cov='MISSING' soft-flag does NOT trigger because the array is non-empty (daily_scan.py:2315). A full-size 40bps short on a thin small-cap is staged and submitted into an earnings gap the blackout was built to avoid.

**Suggested fix.** Change step B3 to `python scripts/build_earnings_calendar.py --overflow-staging` (keeps production key untouched and refreshes earnings_calendar_overflow.parquet weekly). Optionally also flag stale overflow coverage: treat 'no future earnings date within N months' the same as missing coverage in daily_scan's soft-drop.

**Verifier (high confidence).** Every factual claim verified against the code. (1) rebuild_overflow_universe.yml:101 runs build_earnings_calendar.py --with-symbol-master, which per scripts/build_earnings_calendar.py:263-291 builds CSV_UNIVERSE ∪ symbol_master into data/earnings_calendar.parquet and uploads to the PRODUCTION R2 key (build_calendar line 233, default r2_key). (2) build_earnings_calendar.yml:48 runs the same script flagless on a weekday 21:30 UTC cron (line 22), rebuilding CSV_UNIVERSE-only to the same key — so th

#### 26. Fragility sizing multiplier (0.10x-1.25x) is applied to live staged orders but is absent from strat_backtester/daily_portfolio_report — systematic live-vs-model size divergence

`daily_scan.py:2390` (dimension: scan-parity)

daily_scan sizes every non-OVS signal by frag_mult derived from data/rd2_fragility.parquet (ramp: 1.25x at frag=0 down to 0.10x at frag=100; exactly 1.0 only at frag=25) at daily_scan.py:2140-2146,2390-2392, and stamps the result into Risk_Amt, which order_staging.py treats as the authoritative per-trade risk. process_signals_fast (pages/strat_backtester.py sizing block, lines 1544-1603) has NO fragility term at all — the only 'frag' usage in that file is the dial FILTER (line 478) and the optional exposure-leg overlay. Since daily_portfolio_report.run_12month_backtest and the full-history ledger/site both drive off process_signals_fast, every reported position size, Risk $, PnL, and equity curve models un-throttled/un-boosted sizes while live trades frag-adjusted ones.

**Failure scenario.** Fragility score = 60 -> live orders staged at ~0.44x nominal risk. The evening portfolio report shows the same trades at 1.0x: open-position shares, Risk $ and daily PnL are ~2.3x live reality, the Portfolio sheet snapshot disagrees with actual IBKR positions, and any decision made off the report/ledger (e.g. 'strategy X is sized fine') is based on sizes that were never traded. Conversely at frag=0 live trades 1.25x what the report shows.

**Suggested fix.** Either replay the frag multiplier historically in process_signals_fast (rd2_fragility.parquet has the history; reindex like the dial filter does) or explicitly document/accept the divergence and surface staged Risk_Amt vs modeled Risk $ in the report.

**Verifier (high confidence).** Confirmed end-to-end. daily_scan.py:2119-2155 builds frag_mult (1.25x at frag=0, 1.0 only at frag=25, floor 0.10x at frag=100) from data/rd2_fragility.parquet (committed to the repo by risk_report.yml:41, updated nightly, so present in every GHA scan run); lines 2386-2392 multiply it into every non-OVS signal's risk, which sets Shares (2449) and the staged Risk_Amt. order_staging.py (OneDrive\trading_ibkr, line 1077) declares "the scanner's Risk_Amt is the authoritative" and contains no fragilit

#### 27. ATR-seasonal rank filters fail OPEN (neutral 50.0) in the backtester but fail CLOSED in daily_scan — report shows trades live would never scan

`pages/strat_backtester.py:1094` (dimension: scan-parity)

precompute_all_indicators fills atr_sznl_* columns with 50.0 for any ticker absent from atr_seasonal_ranks.parquet (and ffills+fillna(50) gaps for present tickers) at lines 1087-1096. get_historical_mask then evaluates the filter against 50: OVS ('5d < 85'), 52wh Breakout ('> 15' x4), Monday Dip ('> 15'), all PASS at 50. daily_scan.check_signal does the opposite: missing column -> `return False` (daily_scan.py:876-878), and its merge uses exact-date reindex with no ffill/fill (daily_scan.py:2283-2286) so a missing date is NaN -> filter False. This is the exact 'in report but never scanned' drift class previously seen (memory: use_ath/52wh case), and it bites hardest on overflow names, where atr_seasonal_ranks coverage is incomplete (469 new names merged into the LOCAL file only per the in-flight project notes).

**Failure scenario.** Overflow ticker ABCD is missing from atr_seasonal_ranks.parquet. OVS's multi-horizon overbought condition fires on it. daily_portfolio_report (via process_signals_fast) books a short trade and shows it in the email/Portfolio tab; daily_scan rejected the same signal (fail-closed), so nothing was ever staged. The report tracks a phantom position and its PnL indefinitely diverges from live.

**Suggested fix.** In precompute_all_indicators, leave atr_sznl columns NaN when the ticker/date has no rank (NaN already combines to False in get_historical_mask), or explicitly mirror daily_scan's fail-closed semantics for strategies with atr_sznl_filters.

**Verifier (high confidence).** Confirmed line-by-line. Backtester fail-open: pages/strat_backtester.py:1087-1096 fills all atr_sznl_*d columns with 50.0 for map-missing tickers (and ffill+fillna(50) gaps for present ones); get_historical_mask:466-476 evaluates filters against 50 with no missing-data escape. Scan fail-closed: daily_scan.py:2281-2286 merges ranks only for tickers present in the map with exact-date reindex (no fill), and check_signal:875-878 returns False when the column is missing ('fail closed' per its own com

#### 28. Overflow pass in the report skips daily_scan's per-strategy ADDV floor and ADV participation cap

`daily_portfolio_report.py:113` (dimension: scan-parity)

build_full_strategy_book() swaps universe_tickers to OVERFLOW_TICKERS verbatim. daily_scan's overflow pass applies filter_by_addv(overflow_tickers, strategy, meta) per strategy (daily_scan.py:162 — OVS requires $10MM ADDV, 52wh $5MM per overflow_universe.PER_STRATEGY_MIN_ADDV) and caps shares at 2% of 63d ADDV at sizing time (daily_scan.py:2471-2482). Neither gate exists in the report's pass, and process_signals_fast has no equivalent. Both are no-ops today (gate OFF -> meta={}), but the moment OVERFLOW_UNIVERSE_ACTIVE=1 the report and the live scan trade different universes and different sizes. The strat_backtester UI overflow toggle (pages/strat_backtester.py:2895-2914) has the same gap and additionally REPLACES the liquid universe for the 6 eligible strategies instead of adding a second pass, and applies no OLV 25bps overflow override.

**Failure scenario.** Gate flipped ON. A $4MM-ADDV small-cap passes the base screen and fires OVS in the report's simulation (full 40bps short, uncapped shares); daily_scan excluded the ticker entirely via the $10MM OVS floor. The report's overflow-tier PnL and open positions systematically include thin names and larger-than-ADV-cap sizes that live never stages — precisely when validating the new tier's live behavior matters most.

**Suggested fix.** In build_full_strategy_book, apply filter_by_addv(OVERFLOW_TICKERS, s['name'], load_overflow_meta()) per strategy, and thread the ADV share cap into process_signals_fast (or post-scale shares) for overflow-source trades.

**Verifier (high confidence).** Confirmed line by line. daily_portfolio_report.py:113 swaps universe_tickers to OVERFLOW_TICKERS verbatim; the file contains zero references to filter_by_addv/load_overflow_meta/adv_share_cap, while daily_scan.py:162 applies filter_by_addv per strategy (OVS $10MM / 52wh $5MM / base $3MM per overflow_universe.py + docs/dynamic_overflow_universe_plan.md R-T3) and its sizing loop (~2467-2483) applies the 2% adv_share_cap to overflow rows. pages/strat_backtester.py has zero addv references (so proce

#### 29. One transient failed R2 download permanently evicts a ticker from the intraday cache via the meta rebuild round-trip

`.github/workflows/update_intraday_prices.yml:62` (dimension: pipeline-gha)

The pull step iterates meta['ticker'] and calls download_to_local(key, local) discarding the return value (line 62); cache_io prints the error and returns False, and the job continues. The ticker's parquet is then absent from disk, so scripts/update_intraday_yfinance.py skips it (_existing_files only lists on-disk files), _rebuild_meta (line 129-145) rebuilds _meta.parquet from on-disk files only — dropping the ticker — and --upload pushes the shrunken meta back to R2 (line 231). All future workflow runs derive their download list from that meta, so the ticker is never downloaded or updated again. Its R2 parquet still exists but goes stale; once yfinance's ~60-day rolling 15m window passes, the gap is unrecoverable from free sources. An R2 hiccup affecting N tickers evicts all N at once, silently (job stays green).

**Failure scenario.** 2026-12-03: R2 returns a 500 for SPY's parquet during the 20:45 UTC pull -> SPY absent locally -> new _meta.parquet without SPY uploaded -> every subsequent run skips SPY -> two months later SPY's intraday history has a permanent 60-day hole and every Day-Trade-Limit backtest in pages/backtester.py silently loses SPY coverage.

**Suggested fix.** Fail the step (or retry) when any download_to_local returns False for a ticker listed in meta; or make _rebuild_meta merge with the previous meta instead of rebuilding from disk, and upload per-ticker parquets only for tickers actually touched.

**Verifier (high confidence).** Confirmed by direct trace. (1) update_intraday_prices.yml line 62 discards download_to_local's return; cache_io.py lines 269-287 show download_to_local catches all exceptions and returns False without raising, so the step exits 0 and the ticker's parquet is simply absent from the ephemeral runner's disk. (2) scripts/update_intraday_yfinance.py runs with no --tickers filter: _existing_files (lines 46-60) enumerates on-disk parquets only, so the ticker is skipped; _rebuild_meta (lines 129-145) reb

#### 30. Fallback-cron gate keys on 'any successful workflow_dispatch today UTC', so a manual evening dispatch (>= 8 PM EDT / 7 PM EST) suppresses the next morning's pre-market fallback scan

`.github/workflows/daily_screener.yml:57` (dimension: pipeline-gha)

The check job (lines 56-64) counts successful workflow_dispatch runs created on the current UTC date. A manual dispatch after 00:00 UTC — i.e., after 8 PM EDT or 7 PM EST the previous evening — is 'today' from the perspective of the next morning's 10:30 UTC fallback. If the local 4:47 AM trigger machine is also off that morning, the fallback self-skips and NO pre-market scan runs: Order_Staging/Overflow keep the previous evening's rows, exposure_state.json is not refreshed, the morning-run data cutoff and the Overflow-OVS quantity-cap pass (daily_scan.py:2046-2094) never execute, and — in winter — the tabs still contain signals built on the partial 3:30 PM bar from finding 1. order_staging.py then submits from those stale tabs. The identical gate exists in update_master_prices.yml (line 45), where suppression additionally means the partial-bar repair refetch never happens. Note also the gate counts a run as 'successful' only if ALL jobs passed — a dispatch run whose scan succeeded but whose deploy-site job failed causes a redundant second scan instead (benign, but the asymmetry cuts the wrong way).

**Failure scenario.** Tuesday 9 PM EDT: manual dispatch to test a config change (run created 01:00 UTC Wed, succeeds). Wednesday the trigger machine is off (Windows update). 10:30 UTC fallback finds COUNT=1 and skips. order_staging submits Wednesday pre-market from Tuesday-evening tabs with no exposure refresh and no OVS cross-tier quantity cap.

**Suggested fix.** Constrain the gh run list query to runs created within the expected AM window (e.g. --created "$TODAY'T07:00Z..'$TODAY'T10:30Z"), or filter on the run-name/actor of the local trigger.

**Verifier (high confidence).** Trace confirms the finding line by line. daily_screener.yml:56-64 gates only the 10:30 UTC fallback cron and counts ALL successful workflow_dispatch runs created on the current UTC date (`--created="$TODAY"` with `date -u`), not just the 4:47 AM ET local trigger's dispatch. A manual dispatch after 00:00 UTC (>= 8 PM EDT / 7 PM EST the prior evening) is created 'today' from the fallback's perspective, sets COUNT>0, flips should_run=false (line 67), and skips run-scanner (line 76) plus the chained

#### 31. Weekly rundown keys the dispersion signal 'Dispersion Signal' but horizon stats/dial code expect 'Dispersion' — weekly fragility numbers silently exclude the signal

`weekly_market_rundown.py:141` (dimension: risk-ml)

signal_horizon_stats.json and every other caller (risk_dashboard_v2.py:3401, daily_risk_report.py:107) use the key 'Dispersion'. weekly_market_rundown.compute_all_signals uses 'Dispersion Signal', so inside compute_fragility_timeseries the edge lookup _signal_edge(stats, 'Dispersion Signal', h) returns 0.0 and the signal is dropped from BOTH numerator and denominator (risk_dashboard_v2.py:2165-2198), and inside compute_horizon_fragility signals_ordered.get('Dispersion') returns {} (numerator drops it while the denominator still includes its edge). The weekly PDF's cover dials, 63d fragility timeseries page, regime deep-dive table, and email subject all use this mis-normalized frag_df.

**Failure scenario.** 63d edges excluding FOMC sum to 15.53; excluding Dispersion the weekly denominator is 12.62, so with Dispersion OFF every weekly 63d score is ~1.23x the daily report's for the same date — the Friday daily email says 63d=55 (Neutral) while Sunday's PDF says ~68 (Elevated) and highlights the wrong 'current regime' column in the deep-dive table. When Dispersion IS active, its 2.91 edge is missing from the numerator and the weekly understates fragility instead.

**Suggested fix.** Rename the key to 'Dispersion' in weekly_market_rundown.compute_all_signals (the chart title string can stay separate).

**Verifier (high confidence).** Confirmed line by line. weekly_market_rundown.py:141 keys the signal 'Dispersion Signal' while data/signal_horizon_stats.json, risk_dashboard_v2.py:3401, and daily_risk_report.py:107 all use 'Dispersion'. In compute_fragility_timeseries (risk_dashboard_v2.py:2165-2198), _signal_edge returns 0.0 for the unknown key, so the signal is skipped in the numerator (edge==0 continue at 2174) and contributes 0 to base_max at 2198 — dropped from both sides. JSON 63d edges sum to 15.53 ex-FOMC (12.62 withou

#### 32. Pre-FOMC signal history is painted retroactively over the 8 days BEFORE the trigger is measurable — lookahead in the reconstructed fragility series

`pages/risk_dashboard_v2.py:700` (dimension: risk-ml)

compute_fomc_signal decides a fire at the FOMC date (5d trailing-return percentile > 75 evaluated AT the FOMC/snap date, lines 654-668) then marks signal_history True for all index dates in [fomc_date - 8d, fomc_date] (lines 697-701). Days t-8..t-1 are flagged using the return path through t. This history feeds compute_fragility_timeseries, where FOMC is ~42% of the 5d denominator when active (edge 1.28 vs 1.80 base) and ~4% of 21d, so the reconstructed frag 5d/21d values on pre-FOMC days are materially higher than what the live computation showed on those same days (live only fires once the percentile is already elevated 'today'). Contaminated consumers: the ML frag_5d/frag_21d training features (docs/ml_meta_layer_plan.md run-4 discloses only the in-window percentile-band caveat, not this forward backfill), the daily email's similar-reading forward-return tables, the fragility event study, and any 5d/21d dial-filter backtest. It also means stored rd2_fragility values for a given recent date change retroactively after each FOMC meeting — live-vs-backtest divergence by construction. The live 63d sizing path is NOT affected (FOMC 63d edge is null).

**Failure scenario.** A trade signal dated 6 trading days before a 2019 FOMC meeting gets frag_5d ~40 points higher in the ML dataset than the dashboard would have shown live that day; similar-reading forward-return stats for elevated 5d fragility are computed over episodes that were only identifiable in hindsight, overstating the signal's apparent forward-return edge in the daily email.

**Suggested fix.** Build the historical FOMC fire series causally: for each day t in a pre-FOMC window, fire iff pre_pctile.loc[t] > threshold (the same test the live path applies to 'today'), instead of backfilling from the FOMC-date reading.

**Verifier (high confidence).** Confirmed by line-by-line trace. pages/risk_dashboard_v2.py compute_fomc_signal: the live path fires on today's own percentile (line 652), but the historical signal_history is built by testing pre_pctile AT the FOMC snap date (lines 655-668) — a 5d trailing return through the FOMC close — and then backfilling True over the prior 8 calendar days (lines 697-701). Days t-8..t-1 are thus flagged using future returns; the reconstruction also misses fires that live would show (pctile >75 mid-window fa

#### 33. Front-month auto-resolve has no roll/last-trade buffer — pre-fills expiring or delivery-phase contracts into the live ticket

`C:\Users\McKinley Slade\OneDrive\trading_ibkr\futures_front.py:49` (dimension: options-futures)

pick_front keeps any contract with lastTradeDate >= today (`future = [r for r in rows if r[0] >= today] or rows`). There is no roll-date logic, no volume/OI check, and no buffer before last trade or first-notice: a contract on its final trading day (or a physically-delivered contract already inside its delivery/notice window) is still returned as "front" and auto-filled into the execution ticket's contract-month field (execution.js pollFront lines 553-558), from which execute_order transmits live. The `or rows` fallback would even return an already-expired month if IB ever returned only past expiries. The dimension's exact question — "does it ever resolve to an expired or illiquid contract near roll?" — is yes.

**Failure scenario.** 2026-07-20, user types CL in the FUT ticket: auto-resolve fills 202608 (CL Aug last trade 2026-07-21, market already rolled to Sep). A bracket with a time stop dated 2026-07-30 fills, then the contract stops trading the next day and the long runs into physical-delivery handling / forced liquidation. Equivalent index case: on quarterly expiration Friday (e.g. 2026-09-18) ES still resolves 202609 all day even though it stopped trading at 9:30 ET.

**Suggested fix.** Skip contracts within N days of last trade (e.g. 3 for financials, 8 for equity index quarterlies) and skip physically-delivered contracts once inside the month before first notice; simplest generic rule: require last_trade >= today + roll_buffer_days per asset class, returning the next month otherwise.

**Verifier (high confidence).** Code trace CONFIRMS the core claim; severity is overstated.

Confirmed: `futures_front.py` line 49 (`future = [r for r in rows if r[0] >= today] or rows`) keeps any contract whose lastTradeDateOrContractMonth is >= today, with no roll buffer, no first-notice logic, no volume/OI check. `resolve()` (lines 81-87) returns the nearest such expiry as `expiry`. The consumer chain is live and exactly as described: `exec_agent.py` lines 501-505 run the script on a `futures_front` query; `site/assets/exec

#### 34. Stale auto-filled contract month survives a symbol switch when front-resolve fails silently — order can go out in the previous symbol's month

`site/assets/execution.js:542` (dimension: options-futures)

Changing the FUT symbol (input listener line 368) does not clear the f_futexp field; it only schedules resolveFront. resolveFront returns silently on broker error (`if (!d.ok) return;` line 542) and swallows exceptions (line 546), and pollFront only overwrites the field on a successful, symbol-matching result. So if the agent is offline or the resolve times out, the field keeps the PREVIOUS symbol's auto-filled month while looking auto-populated. sendTicket only checks the field is non-empty (line 440), and 6-digit months are valid for most roots, so nothing downstream flags the mismatch.

**Failure scenario.** User types ES -> auto-fill sets 202609. User then changes symbol to CL while the agent is briefly offline: the resolve POST fails silently, the Contract field still shows 202609. The confirm dialog reads 'SELL 2 CL FUT 202609' and, if confirmed, a live order goes into the September CL contract instead of the August front — a materially different price and liquidity.

**Suggested fix.** Clear f_futexp (and set placeholder 'resolving…') whenever the symbol or sec_type changes and frontState.manual is false; on resolve failure leave the field EMPTY so the existing 'enter the contract month' guard blocks submission.

**Verifier (high confidence).** Traced and confirmed. site/assets/execution.js line 368: the symbol input listener only resets frontState.manual and schedules a resolve — it never clears f_futexp (the field is only recreated on sec_type change via renderFutRow, line 365). All three resolve-failure paths preserve the stale month: line 542 `if (!d.ok) return;`, line 546 silent catch, and pollFront timeout/mismatch (lines 550, 558) which only set the placeholder, never exp.value. Line 544 sets the "resolving…" hint only when the 

### Severity: LOW

#### 35. Time-stop and entry-expiry dates are never checked to be in the future — a past time-stop date market-closes the position immediately after the entry fills

`site/assets/execution.js:435` (dimension: exec-frontend)

ticketPayload passes `time_stop: val("f_timestop")` straight through; neither updateReadout nor sendTicket nor the agent (execute_order.py only validates the FORMAT: 8 digits) rejects a past date. build_bracket turns it into a MKT child with goodAfterTime=<past> 15:59:00 in an OCA group — active the moment the parent fills, so it closes the position at market and OCA-cancels the stop and target.

**Failure scenario.** User fat-fingers the date picker to last month (or a stale value) on a BUY bracket. Entry fills at 104.80; the time-stop MKT leg is instantly eligible, sells the position back at market seconds later (paying the spread twice), and cancels the protective legs — the trade the user thought they placed no longer exists.

**Suggested fix.** In updateReadout/sendTicket, warn and block when f_timestop or f_expiry < today (and when expiry > time_stop). The agent should apply the same future-date check as defence in depth.

**Verifier (high confidence).** Confirmed at every layer. Frontend (site/assets/execution.js): date inputs have no min attribute (lines 357-358); updateReadout (398-414) validates only price ordering/futures spec, merely displays the dates; ticketPayload line 435 passes time_stop/expiry through verbatim; sendTicket adds no date check. Agent gatekeeper (OneDrive/trading_ibkr/exec_agent.py _validate, line 199) checks account/qty/price-ordering/notional/risk but never the dates. Executor (OneDrive/trading_ibkr/execute_order.py li

#### 36. New Order ticket ships with realistic hardcoded live values (BUY 692 USO @ 104.80 / stop 103.29 / target 123.21) instead of empty fields

`site/assets/execution.js:352` (dimension: exec-frontend)

syncFields() lines 349-358 pre-fills qty 692 and real-looking prices as VALUES (not placeholders), and switching cmdType away and back silently resets any user edits to these defaults. With the agent currently armed for entry_bracket on both accounts, one reflexive Send + OK submits this exact stale order (mitigated only by the confirm text and execute_order's LIVE_MAX_NOTIONAL=$25k cap, which 692*104.80=$72.5k currently trips — but the cap is env-tunable and the pattern survives a raised cap).

**Failure scenario.** User toggles Type to flatten and back to entry bracket (wiping their typed ticket back to the USO defaults without noticing), clicks Send, and OKs the confirm out of habit — a BUY 692 USO limit bracket is queued to the armed agent with prices from a months-old example.

**Suggested fix.** Render empty inputs with placeholder= examples instead of value=, and preserve user-entered field values across cmdType toggles.

**Verifier (high confidence).** Confirmed line by line. execution.js line 339 renders value= (not placeholder=); lines 349-358 hardcode USO/692/104.80/103.29/123.21 as live field values; line 31 wires cmdType change to syncFields, which rebuilds cmdFields.innerHTML and thus wipes user edits back to the defaults on any Type toggle — exactly the claimed reset. sendTicket (437-451) sends those values after a single confirm(). The premise that the agent is armed is true: exec_agent.env has AGENT_LIVE_ENABLED=1, LIVE_ACCOUNTS=pa,pr

#### 37. Forced `dry_run:true` and the 'transmits nothing' header comment are false safety — the armed agent ignores the flag and transmits live

`functions/exec-command.js:33` (dimension: exec-api-broker)

The file header (lines 8-11) asserts 'dry_run is FORCED true here regardless of input... No order is constructed or transmitted anywhere on this path.' That was true in Phase 2b, but per MEMORY the agent is now armed (AGENT_LIVE_ENABLED=1, LIVE_TYPES=entry_bracket,flatten,cancel) and decides live-vs-dry-run from its own env, NOT from the command's dry_run field (the site's New Order ticket had its dry-run framing removed; sendDryRun was renamed sendCommand). So the hard-coded `dry_run:true` at line 33 is inert: a flatten/entry_bracket sent through this Function is executed live. The Pages layer therefore has NO ability to force dry-run, contradicting the code's own contract and providing false assurance to anyone reading it as a safety gate.

**Failure scenario.** Any POST to /exec-command with type='flatten' or 'entry_bracket' while the agent is armed -> exec-command.js stamps dry_run:true and forwards -> agent ignores the flag, all ramp gates pass -> a real IBKR order transmits, even though this code path claims it 'transmits nothing'. Also the /commands audit entry records dry_run:true (index.js line 78) for an order that actually filled live, masking it in the audit trail.

**Suggested fix.** Either honor the command's dry_run flag on the agent (so the server-side force is meaningful) or remove the dead flag + false comment and make the live/dry state explicit and server-authoritative; ensure the audit dry_run label reflects what the agent actually did.

**Verifier (high confidence).** Mechanics confirmed but severity grossly overstated. The trace is real: exec-command.js:33 stamps dry_run:true, and neither exec_agent.py (_handle_command lines 349-365 routes solely on _live_eligible, lines 297-306, env-driven AGENT_LIVE_ENABLED/LIVE_ACCOUNTS/LIVE_TYPES) nor execute_order.py (re-checks the same env gates, lines 359-366) ever reads cmd["dry_run"], so while armed a flatten/entry_bracket sent via this Function transmits live despite the stamped flag. However, this is the documente

#### 38. No cross-day dedupe: persistent seasonal ideas re-stage a fresh full-size order every session, stacking duplicate positions

`seasonal_order_staging.py:269` (dimension: ideas-seasonal)

build_seasonal_rows() stages every parseable TICKET in today's payload with no awareness of ideas staged on prior days or positions still open (holds are 5-21 trading days). The idea engine re-emits the same (ticker, direction, horizon) on consecutive sessions: the rank gate (<15/>85) and the blended-confirm gate are smooth, multi-week conditions, and the nadir filter (entry_offset_days==0) passes every day of a monotonically-rising expected path (argmin stays 0). The validated backtest explicitly deduped these re-emissions — scripts/backtest_seasonal_ideas.py dedup() line 89, 'one open position per (ticker, direction): if it's already on, it's done' — so live staging as spec'd diverges from the system the +1403R backtest validated. The 1% daily cap (_apply_daily_cap) is per-run only and does nothing across days, and docs/seasonal_order_staging_spec.md line 53 instructs order_staging to 'treat Quantity as final'.

**Failure scenario.** AAPL 21d long: rank stays >85 and the expected path bottoms next-session for 8 consecutive sessions. Each day seasonal_order_staging --write rewrites the Seasonal tab with a fresh full-size BUY; order_staging submits each morning. After 8 sessions the account holds 8 stacked AAPL positions = 1.6% risk (8x20bps) on one name whose backtest edge was measured with exactly one open position — an unintended 8x oversize reaching the broker.

**Suggested fix.** Before building rows, load the forward log (seasonal_ideas_log.parquet) or a staged-positions snapshot and drop any candidate whose (ticker, direction) has an unexpired prior ticket (asof + time_stop_days >= today) — mirroring backtest dedup(). Alternatively have order_staging skip Seasonal rows for symbols with an open seasonal position.

**Verifier (high confidence).** Code-level claims all verified: build_seasonal_rows (seasonal_order_staging.py:269-286) has no cross-day dedupe (only the per-run _apply_daily_cap, lines 246-257); the validated backtest dedupes by default (scripts/backtest_seasonal_ideas.py:89-100 dedup(), applied at 224-226 with do_dedup=True); re-emission is mechanically real (seasonal_edge.py:800-806 sets entry_offset_days=argmin of the expected path, so a rising path passes nadir_filter every session, and negative_filter only screens STRATE

#### 39. No conviction gate in staging: any TICKET in the payload is staged full-size, though the docstring/spec promise A-grade only

`seasonal_order_staging.py:279` (dimension: ideas-seasonal)

The module docstring (line 4-5, 'turns each tradeable A-grade ticket into an order_staging row') and CLAUDE.md describe A-grade staging, but build_seasonal_rows() never checks cand['conviction'] — it only copies it into Strategy_Ref as a label (line 187). Today this is masked because daily_seasonal_ideas.py defaults to --grades A, but the same JSON feeds the display-only site pages (seasonal.js renders all grades), so regenerating the digest with --grades ABC or 'all' for richer display silently widens execution too: C-grade tickets — explicitly including sign-conflict setups where all-years seasonality disagrees with the cycle (seasonal_edge._grade_2x2) — would be staged and submitted at the full 20/13 bps.

**Failure scenario.** User runs `python daily_seasonal_ideas.py --grades ABC` to populate the site's seasonal board, then the scheduled `seasonal_order_staging.py --write` runs on that JSON: every B and C ticket (e.g. a C-grade short whose all-years stat says long) reaches the Seasonal tab and gets submitted to IBKR at full size.

**Suggested fix.** In build_seasonal_rows, skip candidates with conviction not in an explicit STAGE_GRADES=('A',) constant, independent of what grades the digest was rendered with.

**Verifier (high confidence).** Core claim confirmed by trace: build_seasonal_rows (seasonal_order_staging.py:269-286) stages every candidate with a parseable evidence.TICKET; conviction is read at line 187 and used only as a Strategy_Ref label (line 209) — no grade gate, despite the docstring (lines 4-5) promising 'each tradeable A-grade ticket'. The only conviction filter in the whole path lives in the producer (daily_seasonal_ideas.py build(), lines 280-283), controlled by the display-oriented --grades CLI (default 'A', exp

#### 40. Per-tab Sheets fetch failure is serialized as an empty tab — signals page shows 'No staged orders' indistinguishably from a real empty scan

`scripts/build_site.py:528` (dimension: site-payload-contract)

fetch_signals catches any per-tab exception (gspread duplicate-header GSpreadException, quota/5xx, renamed tab) and writes `out['tabs'][tab] = []` with only a build-log print. signals.js (lines 57-63) renders an empty tab as the calm 'No staged orders on this tab.' caption, identical to a genuinely signal-free day, with a fresh `fetched_at` stamp lending it credibility. The payload carries no error flag for the frontend to surface.

**Failure scenario.** Order_Staging holds 8 staged rows but the Sheets read 429s during the 4:47 AM deploy. Pre-market, the user checks the Signals page, sees 'Liquid (0) — No staged orders on this tab', and skips reviewing order_staging's submissions — real orders go to IBKR that morning without the intended human pre-check.

**Suggested fix.** On per-tab failure store an error marker (e.g. out['tabs'][tab] = None and out.setdefault('errors', {})[tab] = str(e)); signals.js renders a red 'fetch FAILED — check the sheet directly' banner when the tab is null/errored instead of the empty-state caption.

**Verifier (high confidence).** Confirmed by line-level trace. scripts/build_site.py lines 520-528: the per-tab loop catches any Exception (quota 429, WorksheetNotFound, gspread duplicate-header errors from get_all_records) and writes out["tabs"][tab] = [] with only a build-log print; no error flag enters the payload, and fetched_at (line 518) is stamped fresh. main() lines 631-634 write signals.json whenever fetch_signals returns non-None, and a per-tab failure still returns a truthy dict, so the degraded payload deploys. Con

#### 41. risk.js reads ctx.regime / ctx.label but the payload key is regime_label — SPY regime sub-label always empty

`site/assets/risk.js:31` (dimension: site-payload-contract)

build_risk_json.py passes price_ctx through from daily_risk_report, whose key is 'regime_label' (used correctly by build_nuggets at build_risk_json.py:140 and daily_risk_report.py:478). risk.js's SPY KPI card reads `ctx.regime || ctx.label || ""` — neither key exists in the payload, so the regime line under the SPY price never renders. The raw value still appears lower in the generic 'Price context' kv dump, so the page silently degrades rather than erroring.

**Failure scenario.** Market enters 'Correction underway'; the risk page's headline SPY card shows the price with a blank regime line every day, and the at-a-glance regime state the card was designed for is never displayed.

**Suggested fix.** Change line 31 to `esc(ctx.regime_label || "")`.

**Verifier (high confidence).** Confirmed, not refuted. compute_price_context (pages/risk_dashboard_v2.py:1083-1093) returns the key 'regime_label' and never emits 'regime' or 'label'; build_risk_json.py:191/214 passes that dict through _clean (key-preserving) into payload['price_ctx'], and the same file uses ctx['regime_label'] correctly at lines 139 and 154 (as does daily_risk_report.py:478). site/assets/risk.js:31 reads `esc(ctx.regime || ctx.label || "")`, so the SPY KPI sub-label is always the empty string. The path is li

#### 42. exposure.json counts positions from Signal Date, not Entry Date — exposure shown 1-3 days before capital is deployed

`pages/strat_backtester.py:2319` (dimension: site-payload-contract)

build_site.py's page_shaped maps Date = Signal Date (line 128) and build_exposure (line 276) calls calculate_daily_exposure, which sets each trade's exposure start via searchsorted on sig_df['Date'] (signal date) rather than 'Entry Date'. Every strategy in the book enters T+1 or later (OLV persistent limits fill T+1..T+3), so the long/short/gross/net exposure series in exposure.json turns positions on before any fill exists, overstating gross on overlap days and mis-phasing the exposure chart against the equity curve (whose PnL correctly starts at Entry Date in get_daily_mtm_series).

**Failure scenario.** A cluster of 15 OVS signals fires Friday; Monday's opens are weak and order_staging skips most of them, but exposure.json already shows the full Friday-stamped gross for Friday-Monday. The user judging historical gross/net deployment off the site's exposure chart sees phantom exposure days and inflated overlap peaks.

**Suggested fix.** In calculate_daily_exposure use 'Entry Date' for start_idx when the column exists (page_shaped already ships it), falling back to 'Date'.

**Verifier (high confidence).** Confirmed the core mechanic by direct trace: scripts/build_site.py:127 maps 'Date' = Signal Date, build_exposure (line 276) calls calculate_daily_exposure, and pages/strat_backtester.py:2306-2319 starts each trade's exposure at sig_df['Date'] (signal date) via searchsorted, never reading 'Entry Date' (which page_shaped does ship at line 128). Meanwhile get_daily_mtm_series (lines 2157/2179) accrues PnL from 'Entry Date', so exposure.json leads the equity/MTM series by the signal-to-entry lag (1 

#### 43. OVS pre-pass P2 cap and P1-budget gate omit cycle_risk_mults — midterm years kill P2 days that per-trade sizing would have kept

`pages/strat_backtester.py:1323` (dimension: engine-correctness)

The pre-pass accumulates `_base_risk_p1 = starting_equity * _p1_bps / 10000 * _ovs_mult` (line 1323) and `_p2_cap_dollars` (line 1273) using only the user risk multiplier, while the main loop applies the cycle-year tilt (step 3b2, line 1584-1588; OVS midterm 0.75x) on top. In year%4==2 the gate/cap math therefore evaluates risk 33% higher than what the engine actually stages, so the P1-budget gate (line 1343) fires on days it shouldn't (zeroing all P2 trades) and the P2 pro-rata scale under-deploys vs path2_daily_cap_pct.

**Failure scenario.** Midterm year, equity $750k, cap_bps=250 → gate = 0.6×$18,750 = $11,250. Four decisive-gap OVS signals on one date: pre-pass counts 4×$3,000 = $12,000 > gate → every P2 trade that day is dropped from the ledger; the tilted per-trade risk the engine actually uses is 4×$2,250 = $9,000 < gate, so under the engine's own documented convention those P2 trades should exist. Ledger composition and OVS stats shift in exactly the years the tilt targets.

**Suggested fix.** Fold the per-date cycle mult into `_base_risk_p1` and `_p2_cap_dollars`/`_ovs_p1_gate_dollars` in the pre-pass (the signal date is available, so `_cyc.get(year%4,1.0)` can be applied there identically to step 3b2).

**Verifier (high confidence).** Confirmed by line-by-line trace. Pre-pass at pages/strat_backtester.py:1323 accumulates OVS P1 risk as starting_equity*p1_bps*_ovs_mult with no cycle_risk_mults, and lines 1273/1283 build the P2 cap and P1-budget gate the same way; the main loop applies the midterm 0.75x tilt at step 3b2 (lines 1584-1588) on top of _ovs_size_mult. Gate at line 1343 therefore compares an untilted P1 sum against the threshold, and line 1541-1542 drops every mild-gap OVS trade on gated days. The engine's own conven

#### 44. OVS morning size-match reads the Order_Staging tab for Scan_Source='Overflow' rows that can never exist there — dead safety check with a misleading comment

`daily_scan.py:2074` (dimension: scan-parity)

The morning-run block (daily_scan.py:2067-2094) builds overflow_ovs_quantities by scanning worksheet('Order_Staging') for rows with Strategy_Ref='Overbot Vol Spike' AND Scan_Source='Overflow'. But save_staging_orders writes with tier_filter routing: Liquid rows -> Order_Staging, Overflow rows -> the 'Overflow' tab (daily_scan.py:2702-2711, matching CLAUDE.md). Order_Staging therefore only ever contains Scan_Source='Liquid' rows, so the dict is always empty and the share cap at lines 2455-2465 never fires. The inline comment ('that [Overflow] tab is retired as of the merge', also repeated at line 1407-1410 'post-merge we stage everything to Order_Staging') contradicts the actual routing and CLAUDE.md. Impact today is nil because the liquid and overflow universes are disjoint (static tier = CSV−liquid; build_overflow_universe.py excludes liquid_set at line 152), but the control is silently inert and the stale comments invite a future edit that trusts them.

**Failure scenario.** A future change lets a ticker exist in both tiers (e.g. a liquid name added to sznl_ranks/symbol_master without updating the exclusion). OVS fires in both passes; the anti-doubling cap that appears to exist reads the wrong tab, finds nothing, and both full-size shorts are staged and submitted — double the intended exposure.

**Suggested fix.** Point the read at worksheet('Overflow') (any Scan_Source), or delete the block and the stale comments if the disjoint-universe invariant is considered guaranteed.

**Verifier (high confidence).** Confirmed by direct trace. daily_scan.py:2074-2080 reads worksheet('Order_Staging') filtering for Scan_Source=='Overflow' rows, but save_staging_orders (def at 1174, tier filter at 1191-1195) is the sole writer of that tab and is only ever called with tier_filter='Liquid' (line 2703-2706); Overflow rows are routed to the 'Overflow' tab (2707-2711), and the tab is cleared+rewritten every run (1196-1211), so no legacy Overflow rows can persist. Hence overflow_ovs_quantities is always empty and the

#### 45. risk_report's bare `git push` races the 21:30 UTC CBOE commit; a lost push leaves rd2_fragility.parquet stale, and daily_scan scales every order's size from it with no freshness check

`.github/workflows/risk_report.yml:43` (dimension: pipeline-gha)

risk_report (21:15 UTC) commits data/rd2_fragility.parquet + rd2_environment.json with a plain `git push` — no `git pull --rebase` and no retry (contrast update_cboe_putcall.yml:48-49 and daily_screener.yml:134-135, which at least rebase). update_cboe_putcall.yml pushes to main at ~21:32 UTC every weekday, inside risk_report's checkout->push window (checkout 21:15 + requirements install + full data download + report generation). When cboe wins, risk_report's push is rejected non-fast-forward and the step fails — but the email was already sent in the prior step, so the red run is easy to dismiss, and the repo's fragility parquet silently stays at yesterday's values. daily_scan.py:2121-2146 reads that checked-out parquet and derives frag_mult in [0.10, 1.25] applied to every staged order's risk, taking iloc[-1] of the series with NO check that the last row is recent; daily_seasonal_ideas and build_risk_json read the sibling rd2_environment.json.

**Failure scenario.** Vol regime breaks on Monday (fragility jumps 20 -> 70). Monday 21:32 UTC the cboe workflow pushes first; risk_report's push at 21:34 is rejected and the updated parquet is lost. Tuesday's 4:47 AM scan reads Friday-vintage fragility, computes frag_mult ~1.2x instead of ~0.4x, and stages every order ~3x larger than the sizing schedule intends; order_staging submits them.

**Suggested fix.** Add `git pull --rebase origin main` before the push (and a small retry loop) in risk_report.yml; in daily_scan, warn and fall back to 1.0x (or refuse the boost side) when the parquet's last index date is older than ~3 trading days.

**Verifier (high confidence).** Mechanics confirmed: risk_report.yml:43 bare `git push` (no rebase/retry, unlike update_cboe_putcall.yml:48-49 and daily_screener.yml:134-135), email sent before commit step, daily_scan.py:2131-2146 reads the checked-out parquet with iloc[-1] and no freshness check (daily_screener.yml pulls rd2_fragility only via git checkout, not R2). But the failure scenario is quantitatively impossible: daily_risk_report.py:710 writes the parquet 5d-smoothed and daily_scan.py:2136 applies a further 10d rollin

#### 46. ML scoring cron races the PM screener rewrite of the staging tabs, and a scoring failure leaves yesterday's decisions live in the ML_Scores sheet

`.github/workflows/ml_score.yml:11` (dimension: risk-ml)

ml_score.yml fires at 22:25 UTC assuming the 22:00 daily_screener PM run has finished rewriting Order_Staging/Overflow ('+ ~10 min runtime', header comment) — but there is no `needs`/same-run coordination, and this repo's own docs record 1-3h GHA cron queue lag (the deploy_site workflow was moved into the screener run for exactly this failure mode). If the screener starts late, _read_sheets scores YESTERDAY'S staging rows (or catches the clear-then-rewrite window and gets an empty tab -> 'no_signals' file). Separately, ml/score_daily.py's fail-safe exception path (lines 243-249) writes the pass-through CSV but never calls _write_sheets_tab even with --sheets-out, so after any scoring error the ML_Scores tab silently retains the PREVIOUS day's SKIP/TRIM rows for tickers no longer staged.

**Failure scenario.** GHA queues the 22:00 screener 40 minutes late. ml_score at 22:25 reads the not-yet-cleared tabs, scores yesterday's signals, and rewrites ML_Scores with today's date on the run while the actual staged book gets no scores; or ml_score throws (model artifact 404) and the human reviewing ML_Scores next morning sees stale SKIP decisions against today's freshly staged tickers. Advisory-only today, but it is the exact stale-score surface that would go live if sizing ever consumes the tab.

**Suggested fix.** Move scoring into daily_screener.yml as a `needs: run-scanner` job (mirroring deploy-site), or have _read_sheets verify the tabs' Scan_Date equals the expected trade date and bail to 'no_signals' otherwise; on the exception path, also rewrite the ML_Scores tab with the pass-through frame so stale decisions never persist.

**Verifier (high confidence).** Both claims are confirmed by line-by-line trace; the finding survives, but severity is overstated given the layer's explicitly advisory, unconsumed status.

Claim 1 (cron race) — CONFIRMED. `.github/workflows/ml_score.yml` line 11 fires `25 22 * * 1-5`; its own header (lines 1-2) encodes the fragile assumption: "runs after the PM scan has rewritten the staging tabs (daily_screener PM cron 22:00 UTC + ~10 min runtime)". `daily_screener.yml` PM cron is `0 22 * * 1-5` (line 29), a separate workflow

#### 47. Primary and fallback fragility files are on different bases (5d-smoothed vs raw) and the fallback is currently 8 weeks stale — consumers treat them as interchangeable

`daily_risk_report.py:710` (dimension: risk-ml)

daily_risk_report writes the 5d-rolling-MEAN series into data/rd2_fragility.parquet (line 710-711), while the Streamlit page writes the RAW timeseries into rd2_fragility_ts.parquet (risk_dashboard_v2.py:3416, only when someone opens the page). daily_scan (FRAG_CACHE -> FRAG_CACHE_TS, line 2131) and ml/ortho_features (FRAGILITY_PATHS, lines 39-40) fall back from one to the other as if equivalent. On disk right now: primary is fresh (2026-06-30) but the _ts fallback last updated 2026-05-07 and differs on the same date (63d 40.42 raw vs 44.06 smoothed). exposure_leg.py:25 even documents the primary's contents as 'Raw', showing a consumer already misreads the basis.

**Failure scenario.** risk_report.yml's commit step fails and a cleanup removes/corrupts rd2_fragility.parquet: the next daily_scan silently sizes off a two-month-old RAW series (double-smoothing it with its own 10d MA), printing a plausible-looking score with no warning; ML training/inference similarly switches feature basis (raw vs smoothed frag values) between runs with no flag.

**Suggested fix.** Write both files on the same basis (or store raw plus compute smoothing at read time), stamp a 'basis' + generation date in the parquet, and make fallback consumers log which file/basis they loaded and enforce a max age.

**Verifier (high confidence).** The factual core is verified: daily_risk_report.py:710-711 writes 5d-smoothed values to rd2_fragility.parquet while risk_dashboard_v2.py:3416 writes the raw series to rd2_fragility_ts.parquet; on disk the primary is fresh (2026-06-30) and _ts is stale (2026-05-07), differing 40.42 raw vs 44.06 smoothed on the same date; daily_scan.py:2131 and ml/ortho_features.py:39-40 both fall back silently with no age check. However the finding is overstated on three counts. (1) The failure scenario is not re

#### 48. 'What Changed' signal-state persistence is dead code: prev_state is loaded but compute_changes is never called, and the state file is overwritten on every rerun

`pages/risk_dashboard_v2.py:3663` (dimension: risk-ml)

main() loads prev_state (line 3663) and immediately overwrites data/risk_dashboard_signal_state.json with today's state on EVERY Streamlit rerun (line 3665), but compute_changes() has no callers anywhere — the activation/deactivation diff documented in CLAUDE.md ('What Changed line tracking signal activations... via JSON persistence') silently no longer exists. Even if re-wired, the design only works once per day: after the first render, load_previous_signal_state returns {} for same-day reads (line 1254), so every active signal would show as newly 'activated' on reruns. The file is also committed to git, so page views dirty the working tree. No race with daily_risk_report exists (it never touches this file).

**Failure scenario.** Operator relies on the documented What-Changed banner to notice a signal newly activating between sessions; the comparison never runs, so a new Defensive Leadership activation surfaces only if they manually diff the signal board against memory.

**Suggested fix.** Either delete load/save/compute_changes and the CLAUDE.md claim, or actually render compute_changes(current_state, prev_state) and only save when the stored date differs from today.

**Verifier (high confidence).** Every element of the finding verified by direct trace. (1) compute_changes (pages/risk_dashboard_v2.py:1287) has zero callers — repo-wide grep finds only the definition and a notes.md mention; no 'What Changed' UI exists anywhere. (2) prev_state at line 3663 is the sole occurrence of that name in the repo: assigned, never read (the '(still needed for dial computation)' comment at 3662 actually covers save_signal_fire_history/dial inputs, not the load). (3) save_current_signal_state at 3665 runs 

#### 49. spot = marketPrice() or close — NaN is truthy, producing invalid JSON that the broker DO silently drops

`C:\Users\McKinley Slade\OneDrive\trading_ibkr\option_quote.py:79` (dimension: options-futures)

ib_insync Ticker.marketPrice() returns nan when quotes haven't populated (common right after a delayed-data reqTickers). `nan or u.close` never falls back because nan is truthy in Python. The nan spot makes the strike-band filter (line 105) select zero strikes ('no 40/20 calls'), and `"spot": round(nan, 2)` is serialized by json.dumps as bare `NaN` — invalid JSON. exec_agent re-emits it over the websocket, and the Durable Object's JSON.parse (execution-broker/src/index.js line 166) throws, so the message is coerced to {type:"raw"} and the option result is never stored. The site polls for 80s then shows the misleading 'timed out — is the agent online?' even though the agent is fine.

**Failure scenario.** User quotes a thinly-quoted ticker pre-market: marketPrice() is nan -> agent produces {"spot": NaN, call/put errors} -> DO drops the frame on JSON.parse -> options page reports agent timeout; repeated retries all fail the same way with no indication the real problem is a missing quote.

**Suggested fix.** Use `spot = u.marketPrice(); spot = u.close if spot != spot else spot` (nan-check) and emit `None` for non-finite values (json.dumps(..., allow_nan=False) as a guard); return an explicit 'no spot price' error.

**Verifier (high confidence).** Finding CONFIRMED by full trace. (1) ib_insync 0.9.86's Ticker.marketPrice() (installed source verified) returns `last` with no bid/ask and has no close fallback, so it returns nan pre-market on thin names; `nan or u.close` at option_quote.py:79 stays nan because nan is truthy. (2) Line 105's `if spot and ...` passes (nan truthy) but nan comparisons are all False → empty strike band → both spreads error 'no 40/20 calls/puts'. (3) Line 119 `round(spot,2) if spot else None` → round(nan,2)=nan; jso

## Part 2 — Findings refuted during verification

Listed for transparency. Each was traced and killed by the verifier.

- **Read-only proxies swallow broker errors and return HTTP 200 with empty payloads, hiding delivery/agent failures** (`functions/exec-commands.js`, exec-api-broker). Why refuted: The finding's core claim — that a broker outage is invisible and indistinguishable from 'not delivered yet' — is contradicted by the actual UI wiring: site/assets/execution.js poll() (lines 106-117) fetches /exec-status in the same Promise.all as /exec-commands, and exec-status.js line 29 returns {o
- **Signed dry_run flag is never checked — every site command transmits live when the agent is env-armed, and the UI's DRY-RUN banner can be stale** (`C:/Users/McKinley Slade/OneDrive/trading_ibkr/exec_agent.py`, exec-local-agent). Why refuted: The finding's individual code facts are accurate: exec_agent.py never reads cmd['dry_run'] (_live_eligible lines 297-306 gates on env only), execute_order.py main() (lines 359-364) re-checks only env gates, functions/exec-command.js:33 stamps dry_run:true with a stale Phase-2b comment, and exec_agen
- **Staging reads daily_seasonal_ideas.json with no freshness guard — a failed ideas build silently re-stages yesterday's tickets** (`seasonal_order_staging.py`, ideas-seasonal). Why refuted: The failure scenario is blocked at three independent points. (1) The finding's key mechanism is wrong: daily_seasonal_ideas.py line 300 always writes meta.asof, so the today() fallback at seasonal_order_staging.py:264 never fires on producer output; with a stale Monday payload, Scan_Date is stamped 
- **Daily cap and sheet Risk_Amt use nominal risk, counting zero-share rows and over-reporting deployed risk to order_staging's global cap** (`seasonal_order_staging.py`, ideas-seasonal). Why refuted: The finder read the code correctly at the mechanical level (seasonal_order_staging.py:176 `shares = int(risk_amt / tk["risk_ps"])`, :217 nominal Risk_Amt, :249 cap sums nominal, :254 post-scale int floor), but the finding fails on three independent grounds.

(1) The failure scenario is numerically i
- **Sunday rundown cron (12:00 UTC) fires before the 8:30 AM ET local radar digest commit, so the weekly email always embeds LAST week's digest** (`.github/workflows/weekly_rundown.yml`, pipeline-gha). Why refuted: The finding's core factual premise — that the radar digest is committed at 8:30 AM ET, i.e. AFTER the 12:00 UTC rundown checkout — is contradicted by the repo's own git history.

1. The cron is not drift. Commit d32f7e9 (2026-04-05, "chore: move weekly rundown schedule to 8 AM ET (12 PM UTC)") delib

## Part 3 — Improvement research

### Execution platform hardening (pre-go-live)

*Track notes:* Ranked by risk reduction per unit effort. Confirmed-from-code findings (not speculative): (1) exec_agent.py never reads cmd.dry_run — when armed, every site click is live and preview is impossible (functions/exec-command.js forces dry_run:true into the signed envelope, agent ignores it); (2) flatten fraction is unclamped in both exec_agent._validate and execute_order._do_flatten — fraction>1 reverses a position at market; (3) _execute_live's asyncio.wait_for(30s) does not kill the child and a multi-owner flatten legitimately takes ~40s, so a filled live order can be reported as 'error'; (4) COMMAND_SECRET (golive step 1) appears nowhere in code or exec_agent.env — STATUS_TOKEN is read-proxy auth + broker auth + order-signing key simultaneously; (5) arm_live.bat arms both accounts/all types/$25k in one shot, contradicting the golive ramp; (6) audit = 50-item DO ring + an overwritten *_last_run.log; (7) broker sends commands to ALL sockets and agent dedup is in-memory per-process. Key file anchors: execution-broker/src/index.js (lines ~74-82 push-to-all, ~26 CMD_CAP), functions/exec-command.js (line 33 forced dry_run, line 36 60s expiry vs agent local clock), C:/Users/McKinley Slade/OneDrive/trading_ibkr/exec_agent.py (lines 309-319 subprocess timeout, 199-249 _validate, 52-54 env gates), execute_order.py (line ~290 fraction sizing), arm_live.bat, docs/site_execution_golive.md. Deliberate scope note: items 1-4, 6, 11, 12 are each roughly a day or less and would close every confirmed defect before arming; 5, 9, 10 are the durable-correctness layer worth building during the tiny-live phase. One additional small item folded into idea 8: _validate trusts a cached book of unbounded age — add a max-age gate. Do not block flatten/cancel on the daily-loss breaker (closing risk must always be allowed).

#### execution-hardening 1. Honor per-command dry_run and require preview-before-live (agent currently ignores the flag)

Effort: low. Category: guardrail / protocol

CONFIRMED gap: functions/exec-command.js signs `dry_run: true` into every envelope, but exec_agent.py never reads `cmd.dry_run` (grep: zero references) — live eligibility is purely env-based (_live_eligible). So the moment AGENT_LIVE_ENABLED=1, EVERY click on the site transmits live, including commands whose signed envelope says dry_run:true, and there is no way to preview an order while armed. This violates the schema contract ('dry_run: agent validates + logs, transmits nothing') and removes the single most useful safety behavior: rehearsing the exact order chain before sending it.

**Expected impact.** Eliminates the largest live-fire class: accidental one-click real orders while armed; restores the documented preview-then-commit flow.

**How to test / implement.** Implement: (1) exec_agent._handle_command gates _execute_live on `cmd.get('dry_run') is False` in addition to _live_eligible; (2) execute_order.py re-checks `cmd['dry_run'] is False` (defence in depth); (3) exec-command.js passes through `dry_run: body.dry_run !== false ? true : false` instead of forcing true; (4) site/assets/execution.js: two buttons — 'Preview' (dry_run:true) and 'Execute LIVE' (dry_run:false), with Execute disabled until a preview of the SAME payload hash returned ok. Test: arm on the PA/paper account, send a flatten with dry_run:true → assert state='dry_run' and no order appears in TWS; resend with dry_run:false → order appears. Unit-test the agent gate with LIVE_ENABLED monkeypatched true.

#### execution-hardening 2. Clamp flatten `fraction` to (0, 1] in both the agent validator and execute_order

Effort: low. Category: guardrail

CONFIRMED: neither exec_agent._validate nor execute_order._do_flatten bounds `fraction`. execute_order.py line ~290 computes `n = int(round(abs(pos.position) * fraction))` and places a market order for n — a payload with fraction=7 on a 100-share long SELLS 700, leaving you short 600 at market. The site only sends 1 or 0.5, but the API accepts any signed payload (buggy JS, replay of a mangled body, future scripted callers). Similarly `quantity` on entry_bracket should be re-bounded against the position-independent caps in _validate, not just in execute_order.

**Expected impact.** Removes a one-payload position-reversal footgun for ~20 lines of code; best risk-reduction-per-line in the review.

**How to test / implement.** Add to _validate: `frac = float(p.get('fraction', 1)); if not (0 < frac <= 1): reject`. Add the identical check in _do_flatten before sizing, plus assert the computed n <= abs(pos.position). Tests: pure unit tests on _validate/_do_flatten with fraction in {0, -1, 0.5, 1, 2, 7}; then a dry-run command from the site DevTools with fraction=2 → expect state='rejected' with the clamp reason.

#### execution-hardening 3. Remote kill switch: a halt flag in the Durable Object, enforced broker-side and agent-side

Effort: medium. Category: kill switch

The plan doc promised 'a flag in the DO the agent checks before every action' — it was never built. Today the ONLY kill switch is disarm_live.bat + restarting the ExecAgent task, which requires being at the trading box. If you're on your phone watching a runaway (or the machine is remote), you cannot stop execution from the site; you'd have to wait for the 21:00 window close. For a real-money system the kill path must be reachable from wherever you can see the damage.

**Expected impact.** Turns 'drive to the machine' into 'one authenticated click from anywhere'; also gives execute_order a local hard-disable independent of env restarts.

**How to test / implement.** Broker: add POST /halt and /halt state in ExecBroker storage; /command returns 409 'halted' when set and pushes {type:'halt'} down the socket. Pages: /exec-halt function behind Access with a typed-confirm UI button (big red HALT in the conn bar; un-halt requires the same). Agent: on {type:'halt'} set an in-memory+on-disk halted flag → refuse all commands, flip book.mode to 'halted' (banner turns red); execute_order.py refuses when the halt file exists. Test: send a dry-run command → ok; POST /exec-halt; resend → 409 at broker; kill the broker path and verify agent-side file check still refuses; un-halt and confirm recovery. Also chaos-test: halt while a command is mid-flight — the in-flight one completes, the next is refused.

#### execution-hardening 4. Fix the 30s execute-subprocess timeout: orphaned live order reported as an error

Effort: low. Category: chaos hardening

CONFIRMED chaos scenario: exec_agent._execute_live wraps execute_order.py in `asyncio.wait_for(..., timeout=30)` but never kills the child on timeout — the subprocess keeps running and can still transmit. A perfectly normal flatten whose bracket legs are owned by another clientId exceeds 30s (connect-as-owner 8s + sleeps 2.5s + _confirm_cancelled up to 7.6s + fill poll 12s + main connect 8s ≈ 40s+), so the agent replies state='error: TimeoutError' while the close FILLS seconds later. The site then shows a failed command for a trade that actually executed — the exact 'site-said vs IBKR-did' divergence you fear. Killing the child is worse (could kill between cancel and close, leaving the position naked); the fix is a longer timeout plus an honest terminal state.

**Expected impact.** Removes the silent-divergence failure mode where the operator retries a 'failed' flatten that actually filled — the classic double-execution setup.

**How to test / implement.** Raise the timeout to ~120s; on TimeoutError return `{ok: null, state: 'unknown', detail: 'transmit status unknown — VERIFY IN TWS'}` (a distinct amber badge in execution.js stateBadge, never green/red); have execute_order append phase lines (connected/cancelled/closing/done) to a per-command journal file so a follow-up sweep can resolve 'unknown' to the truth. Test: set an env-gated `time.sleep(45)` inside _do_flatten on paper, click Flatten → assert the site shows UNKNOWN (not error), the order still fills in TWS, and the journal records 'done'.

#### execution-hardening 5. Command acknowledgment + reconciliation loop (pushed → received → terminal, with an 'unresolved' sweep)

Effort: high. Category: protocol / reconciliation

The broker marks a command 'pushed' and only patches it if a result arrives on the socket (index.js webSocketMessage 'result'). If the agent crashes mid-command, the WS drops before the reply, or the process dies after execute_order transmitted, the command sits at 'pushed' forever and the audit lies. There is no delivery ack, no timeout, and no replay. The agent's dedup (_SEEN) is in-memory, so a crash-restart loses it too. For live money you need every command to reach a truthful terminal state: executed / rejected / unknown-verify-manually.

**Expected impact.** Every order attempt ends in a truthful, durable state; the site can never show a stale 'pushed' for an order that is live at IBKR.

**How to test / implement.** Agent: send {type:'received', id} immediately on delivery; keep a disk write-ahead journal (id, phase, ts) updated at received/validated/executing/terminal; on reconnect, replay unresolved journal entries as late results. Broker: store delivered_at on 'received'; set a DO alarm (ctx.storage.setAlarm) that sweeps recent_commands and flips anything non-terminal older than 120s to state='unresolved'; execution.js renders 'unresolved' amber with 'verify in TWS'. Reconciliation: after any 'executed' result, the next book snapshot should contain the returned order_ids — the agent cross-checks and emits a 'reconciled'/'MISSING' annotation. Chaos tests: (a) SIGKILL the agent between validation and execute (env-gated sleep), restart → command resolves via journal replay; (b) drop the network right after transmit → command flips to unresolved, then reconciles on reconnect; (c) duplicate replay of the journal → broker patch is idempotent by id.

#### execution-hardening 6. Implement the dedicated COMMAND_SECRET (golive step 1 — currently absent from all code)

Effort: low. Category: security

CONFIRMED: grep finds zero COMMAND_SECRET references in the broker, Pages functions, exec_agent.py, or execute_order.py, and exec_agent.env has no such key — despite docs/site_execution_golive.md listing it as go-live step 1. Today STATUS_TOKEN is triple-duty: bearer auth for every read proxy (/exec-status, /exec-book, /exec-commands, futures/option queries), broker /command auth, AND the HMAC signing key for live orders. Any leak from the lowest-privilege read path (a misconfigured Pages env, a logged header) mints valid live commands. Signing authority must be a separate secret held only by exec-command.js and the agent.

**Expected impact.** Compromise of the widely-shared read token no longer grants order-minting; matches the runbook's own stated precondition for arming.

**How to test / implement.** Generate COMMAND_SECRET; set on the Pages project + exec_agent.env. exec-command.js signs with env.COMMAND_SECRET (503 if unset — fail closed, no fallback to STATUS_TOKEN on the signing side). Agent _verify: use COMMAND_SECRET when set, else STATUS_TOKEN (temporary migration fallback, removed after cutover). Test: with the agent updated but Pages still signing with STATUS_TOKEN → 'bad signature' (proves enforcement); after Pages cutover → commands verify; tamper one byte of `signed` → rejected. Add 'COMMAND_SECRET set + fallback removed' as an explicit checkbox in the golive doc.

#### execution-hardening 7. Broker-side guardrails: rate limit, per-symbol in-flight lock, and an 'armed' gate at the DO

Effort: medium. Category: server-side guardrail

The DO is a deliberately dumb relay — every guard lives in agent env vars. But some guards belong server-side: a stuck browser retry loop (or a fat-fingered loop in DevTools) can POST dozens of flattens, each getting a fresh uuid so the agent's id-dedup never fires, and each executes. Nothing prevents two overlapping flattens on the same symbol (the second re-reads positions mid-cancel of the first). And the broker will happily relay commands even when nothing is armed, cluttering the audit.

**Expected impact.** Caps the blast radius of client-side bugs/compromise at the cloud layer, before anything reaches the trading box — the 'not just UI' enforcement the platform currently lacks.

**How to test / implement.** In ExecBroker /command: (1) sliding-window counters in storage — reject >6 commands/min and >40/day with 429 (echo/read queries exempt); (2) in-flight lock — reject a command whose (symbol, type) matches a recent_commands entry still in a non-terminal state ('one order per symbol at a time; wait or cancel'); (3) an `armed` storage flag the agent reports in its hello/book (mirrors AGENT_LIVE_ENABLED) so /command can annotate or refuse live-intent commands when the agent is dry-run. Tests: script 10 rapid POSTs → expect 429 after the 6th; send flatten USO twice in 2s → second rejected; verify counters reset by day (mock Date). All testable with wrangler dev + vitest against the DO, no IBKR needed.

#### execution-hardening 8. Daily circuit breakers in the agent: day-start NLV loss cutoff + cumulative daily notional cap

Effort: medium. Category: guardrail

All existing caps are per-command (MAX_NOTIONAL 250k, MAX_RISK_PCT 5%, LIVE_MAX_*). Nothing stops 20 sequential max-size entries in a day, or continuing to add risk into a -3% account day. The agent already has everything needed: the book loop delivers NLV every 20s, so it can anchor day-start NLV and accumulate per-day executed notional. This is the daily-loss guardrail the task calls out, enforced at the gatekeeper rather than the UI.

**Expected impact.** Bounds worst-case daily damage from any upstream failure (UI bug, token leak, own bad judgment on a tilted day) to a pre-committed number.

**How to test / implement.** Agent: persist {date, day_start_nlv, notional_today, commands_today} to disk (survives restarts); first book of each ET day sets the anchor. In _validate for entry_bracket: reject when nlv < anchor*(1-LOSS_CUTOFF_PCT, e.g. 2%) or notional_today + this > DAILY_NOTIONAL_CAP; increment the accumulator only on state='executed'. execute_order.py re-reads the same file and re-checks (defence in depth). Also refuse any command when the cached book is older than ~90s (stale-book guard — today _validate trusts a book that could be hours old if book_loop keeps erroring). Tests: unit tests with a fabricated anchor file (high anchor + low NLV → entry rejected, flatten still ALLOWED — closing risk must never be blocked); date rollover resets; stale book timestamp → reject with 'book stale'.

#### execution-hardening 9. Durable append-only audit trail (R2 JSONL from the broker + local append log at the agent)

Effort: medium. Category: audit durability

The entire audit is a 50-item ring in DO storage (CMD_CAP=50, index.js) — a busy day of echoes/queries/commands evicts real fill records, and there is no export. Locally, the agent writes exec_agent_last_run.log, which by the *_last_run convention is overwritten on every restart, so a crash destroys the record of exactly the session you need to reconstruct. The plan doc promised 'append-only audit… surfaced on the site'; live money needs a record that survives both a DO eviction and an agent restart.

**Expected impact.** Post-incident you can always answer 'what did the site tell the agent, and what did the agent do' — the difference between a reconstructable event and a shrug.

**How to test / implement.** Broker: on every /command accept and every result/close/error event, `ctx.waitUntil` an append to R2 (`audit/exec/YYYY-MM-DD.jsonl`; R2 has no append, so buffer events in DO storage and flush the day-object on an alarm every ~60s — acceptable loss window). Add the R2 binding to execution-broker/wrangler.toml. Agent: open exec_audit.jsonl in append mode, one line per command envelope + validation + result + arm-state; never truncated. Site: an 'export audit' link streaming the R2 day files via a Pages function (same pattern as chartimg). Test: send 60 mixed commands, assert the ring holds 50 but the R2 day file holds all 60 with terminal states; restart the agent mid-day and confirm the local JSONL retains pre-restart lines.

#### execution-hardening 10. Push alerting: agent offline during RTH, first live execution of the day, and bad-signature commands

Effort: medium. Category: alerting

Today 'execution offline' is only visible if the Execution tab is open. If the ExecAgent dies at 10:00 with positions on (exits are safe at IBKR, but you've lost the ability to act) you find out whenever you next look. Likewise a bad-signature command is the single loudest intrusion tell in the system and it currently only lands in an overwritten local log. The DO already tracks last_seen/disconnected_at — it just never tells anyone.

**Expected impact.** Cuts detection time for the two scariest silent states — execution unavailable with money on, and someone else knocking — from 'whenever you look' to minutes.

**How to test / implement.** DO: on webSocketClose and on a heartbeat alarm (setAlarm every 60s), if sockets==0 or heartbeat stale >120s during 09:25–16:05 ET Mon–Fri, send ONE alert per outage (flag in storage; clear on reconnect + send recovery notice). Delivery: reuse the existing email infra via a tiny authenticated webhook, or Cloudflare Email Workers (see cloudflare-email-service skill) — email is fine, the point is push. Also alert on: the first state='executed' result each day (confirms live activity is expected), and any 'bad signature' result (from the agent's reply relayed through webSocketMessage). Test: stop the ExecAgent task at 10:30 ET → email within ~3 min, exactly one; restart → recovery email; POST a garbage-sig command → intrusion alert. Verify NO alert outside RTH (the agent legitimately exits at 21:00).

#### execution-hardening 11. Single-agent enforcement + newest-socket-only delivery + persistent dedup

Effort: low. Category: chaos hardening

Duplicate-delivery paths exist: (1) broker /command sends to ALL open sockets (`for (const s of sockets) s.send(...)`) — a half-dead old socket plus a fresh reconnect means two deliveries; (2) if Task Scheduler relaunches ExecAgent while a hung instance survives, two PROCESSES each hold a socket and each executes (the _SEEN dedup is per-process memory); (3) an agent crash-restart wipes _SEEN, so a re-pushed command after restart is 'new'. Any of these turns one click into two live orders.

**Expected impact.** Closes every identified double-execution route (dup socket, dup process, restart-amnesia) with ~40 lines across the two sides.

**How to test / implement.** Agent: acquire an exclusive lock on a lockfile at startup (msvcrt.locking on Windows), exit immediately if held — guarantees one instance. Persist processed command ids+timestamps to disk (append to the audit JSONL and load the last 24h on boot) so dedup survives restarts. Broker: on a new /agent connect, close all previously-accepted sockets (iterate getWebSockets, close with code 4000 'superseded') and deliver commands only to the most recent. Tests: start exec_agent.py twice → second exits with 'already running'; wrangler-dev test: open two agent sockets, POST a command, assert only the newest receives it; restart the agent and replay a previously-processed signed command → 'duplicate command ignored'.

#### execution-hardening 12. Align arming tooling with the runbook ramp and surface the exact arm scope + caps in the UI

Effort: low. Category: go-live process / UX

CONFIRMED divergence: docs/site_execution_golive.md prescribes 'arm tiny — PA only, flatten only, LIVE_MAX_QTY=1' as step 3, but arm_live.bat (the actual tool) arms pa+primary, all three types, 2000 shares / $25k in one shot — the ramp exists only on paper. Meanwhile the UI banner says just 'LIVE ARMED' with no scope, and the ticket lets you compose orders that the gate will reject (>caps, unarmed types) with no upfront hint. The env file already contains LIVE_* keys, so verifying actual arm state before go-live matters. Two small gaps also belong on the checklist: clock-skew (the agent compares broker-signed expires_at to local time; >60s skew bricks or widens the replay window) and stale-env residue after disarm.

**Expected impact.** Makes the documented dry-run→tiny→full ramp the path of least resistance instead of a doc-only intention, and makes the current blast radius visible at a glance before every click.

**How to test / implement.** (1) Replace arm_live.bat with staged scripts matching runbook steps: arm_tiny.bat (pa/flatten/qty1/$500), arm_expand.bat (adds entry_bracket+cancel, still pa), arm_full.bat (adds primary + lifts caps) — each prints exactly what it armed. (2) Agent includes live_config {accounts, types, max_qty, max_notional, max_fut} in the book payload; execution.js renders it in the amber banner ('LIVE: pa · flatten · qty≤1 · ≤$500') and greys ticket types that aren't armed. (3) Golive checklist additions: run `w32tm /stripchart` (or compare heartbeat-ack server_now vs local, which the agent should log at startup) and require skew <15s; after disarm, assert `findstr LIVE_ exec_agent.env` is empty; verify the first watched fill matches its preview line-by-line and file the comparison in the audit. Test by running arm_tiny and confirming the banner text and that an entry_bracket is refused as 'type not armed'.

### Infrastructure and testing

*Track notes:* Ranked by expected value = (severity of failure prevented) x (observed likelihood in this repo's history). Ideas 1-3 target failure modes that have each already occurred at least once (filter drift: use_ath case plus the live ETF_ATR_EXEMPT divergence found during this review at daily_scan.py:676/732 vs strat_backtester.py:307-309; stale-bar order wipe: 2026-06-11). Idea 2 (CI) is ranked second despite being trivial because nothing currently executes the 19 existing test files, so every other testing idea is inert without it. Suggested sequencing: 2 -> 3 -> 6 -> 7 -> 1 -> 5 -> 4 -> 8 -> 9 -> 10 -> 11 -> 12; idea 12 (shared filter engine, listed last) should only start once idea 1's harness is green since it is the safety net for that refactor. Shared fixture (tests/fixtures/parity_prices.parquet) amortizes across ideas 1 and 5. Everything proposed uses deps already in the repo (pandas, boto3, gspread mocking via stubs); no pip installs needed beyond optionally pip-tools for idea 10.

#### infra-quality 1. Scanner/backtester filter parity harness (frozen-date signal diff)

Effort: medium. Category: testing / parity

check_signal (daily_scan.py:701) and get_historical_mask (pages/strat_backtester.py:275) reimplement the same ~25-filter DSL independently. This bug class has already fired once (the use_ath/52wh-Breakout case in memory), and a second live divergence exists right now: ETF_ATR_EXEMPT (daily_scan.py:676) exempts SPY/QQQ/IWM/DIA from min_atr_pct in the scanner only; the backtester applies it to every ticker. Divergence here means the ledger, the portfolio report, and the site model a different book than the one being traded.

**Expected impact.** Converts the worst recurring bug class (trade signals silently differing between live scan and the ledger/report that sizes and audits them) from prose-discipline to a failing test.

**How to test / implement.** Commit a small frozen fixture (tests/fixtures/parity_prices.parquet: ~25 tickers covering every filter type in STRATEGY_BOOK, 3-4 years of bars, plus frozen sznl/atr_sznl maps). New tests/test_scan_backtest_parity.py: for each strategy, run precompute_all_indicators + get_historical_mask over the fixture to get the backtester's signal-date set; then for each of the last ~500 sessions, slice df up to that date and call check_signal to get the scanner's set. Assert set equality per (strategy, ticker), printing the first divergent date and which filter disagreed (re-evaluate filters one at a time on the divergent row). Run it in CI (idea 2). Start by whitelisting the known ETF_ATR_EXEMPT divergence so the test lands green, then decide which side is correct and fix.

#### infra-quality 2. CI workflow that actually runs the test suite

Effort: low. Category: CI

tests/ has 19 files including regression guards the CLAUDE.md leans on (test_eod_dd, test_olv_fill_window, test_stop_gap_fill, test_verify_fills_exdiv), but no workflow in .github/workflows/ runs any of them. Every convention documented as 'guard: tests/...' is only a guard if something executes it. For a repo staging real orders, this is the cheapest risk reduction available.

**Expected impact.** Every existing and future regression test becomes enforced instead of advisory; a broken engine change can no longer ship to the nightly ledger unnoticed.

**How to test / implement.** Add .github/workflows/ci.yml: on push to main + workflow_dispatch, setup-python 3.10 (match prod workflows), pip install -r requirements.txt, then `python -m pytest tests/ -x -q --ignore=tests/backtest_put_hedge.py --ignore=tests/backtest_signal_overlap.py` plus `python tests/run_checks_staging.py` etc. for the script-style checks. Mark anything needing network/R2 with skipif on missing env. Add a badge or at least rely on GHA failure emails. Total runtime should be under 5 min since the pure tests use synthetic frames.

#### infra-quality 3. Stale-data gate before staging-tab wipes (sentinel bar-date assertion)

Effort: low. Category: pipeline safety

The 2026-06-11 incident: a stale last bar made every liquid strategy find zero signals, and save_staging_orders then cleared Order_Staging, wiping valid staged orders. The cache-first fix removed one cause, but daily_scan still has no explicit assertion that the evaluation date equals the expected trading session; a failed update_master_prices AM run would reproduce the same wipe from the cache side. The clear-on-zero-signals behavior (daily_scan.py:1197) is correct only when the data is known fresh.

**Expected impact.** Prevents recurrence of a real money-losing failure mode: silent stale scans that erase good staged orders. Turns quiet wrongness into a loud red run.

**How to test / implement.** In run_daily_scan after load_master_prices_dict: compute expected_session = previous NYSE trading day (pre-open) or today (post-close) via the same CustomBusinessDay calendar used elsewhere; assert SPY/QQQ/IWM last bar == expected_session. On failure: send the error email, exit nonzero (so the GHA run fails loudly and deploy-site is skipped), and critically do NOT call save_staging_orders, leaving the previous tabs intact. Add tests/test_scan_freshness_gate.py with a doctored dict whose last bar is two sessions old, asserting the gate raises before any Sheets write path is reached.

#### infra-quality 4. R2 data heartbeat monitor with alerting

Effort: medium. Category: monitoring

The pipeline has ~8 R2 artifacts with expected cadences (master_prices 2x/day, earnings_calendar nightly, intraday weekdays, atr_seasonal_ranks, overflow parquets) and several best-effort producers. Nothing checks that they actually updated; a dead local dispatch task plus a quietly failing fallback would leave everything running on aging data. cache_io.list_keys_with_meta (cache_io.py:332) already returns per-key last-modified epochs, so 90% of the plumbing exists.

**Expected impact.** Closes the 'many steps are best-effort and can silently stale' gap in one place, with a single dependency-free script and one small workflow.

**How to test / implement.** New scripts/check_data_heartbeats.py: a table of {r2_key: max_age_hours, trading_days_only} (master_prices: 20h, earnings_calendar: 30h, intraday/15min/_meta.parquet: 30h, atr_seasonal_ranks: 8d, ...). Pull last-modified via list_keys_with_meta, compare against now with a trading-calendar-aware allowance (weekends/holidays), print a status table, exit 1 with an email (reuse the send_email helper pattern from daily_scan) listing every stale key. New workflow heartbeats.yml: cron weekdays 11:30 UTC (after AM pipeline should be done) + 23:00 UTC (after PM). Also extend it to check the deployed site's /data/meta.json build timestamp via curl for end-to-end freshness.

#### infra-quality 5. Golden-file regression test for process_signals_fast

Effort: medium. Category: testing / regression

process_signals_fast (strat_backtester.py:1217) drives the ledger, the site, and the daily portfolio report, and its semantics have been deliberately changed several times (stop-arming, gap-through fills moved the book by -45.7R). Existing tests cover single conventions; nothing pins the whole-trade output, so an accidental interaction (e.g. a sizing tweak changing fill sequencing) can shift PnL attribution silently.

**Expected impact.** Any unintended change to fills, holds, stops, sizing, or exit priority fails CI with a trade-level diff instead of surfacing weeks later as a drifted ledger.

**How to test / implement.** Reuse the parity fixture (idea 1). tests/test_golden_ledger.py: run generate_candidates_fast + process_signals_fast over the fixture with flat sizing and fixed kwargs, then compare the resulting trades frame against a committed golden CSV (tests/fixtures/golden_trades.csv) on key columns: Strategy, Ticker, Signal/Entry/Exit dates, Entry/Exit prices (abs tol 1e-6), Exit_Type, R. On mismatch, print a row-level diff. Add a tiny scripts/regen_golden.py to regenerate the golden after an INTENTIONAL engine change, so the diff shows up in git review as an explicit artifact of the change.

#### infra-quality 6. STRATEGY_BOOK schema validation and executable convention checks

Effort: low. Category: config validation

strategy_config.py is maintained by manual copy-paste from the Backtester UI (its own header warns about this), strategy names are string-matched keys in daily_scan, and the book's conventions live as 'four aligned sites, change together' prose in CLAUDE.md. A typo'd key, a bad logic string, or fill_window_days >= hold_days would fail silently or misprice risk.

**Expected impact.** The copy-paste config workflow gets a machine gate; the prose 'critical warning for AI agents' at the top of strategy_config.py becomes enforced.

**How to test / implement.** tests/test_strategy_book_schema.py, plain assertions (no new deps): unique names/ids; settings keys drawn from a frozen allowed set (catches typos like 'perf_thres'); logic fields in {'<','>','Between','Not Between'}; thresh <= thresh_max; every perf/atr_sznl filter window in the supported set; execution dict fields typed (risk_bps > 0, 0 < fill_window_days < hold_days when present, eod_dd_weekdays subset of 0..4, cycle_risk_mults keys in 0..3); every strategy name referenced in daily_scan special-case code (OVERFLOW_ELIGIBLE_STRATEGIES, OVERFLOW_RISK_OVERRIDES, CROSS_STRATEGY_OVERLAP_OVERRIDES) exists in the book; universes resolve to nonempty ticker lists. This runs in the CI from idea 2 in under a second.

#### infra-quality 7. Sheets staging-tab column contract (frozen headers + write-time validation)

Effort: low. Category: contract testing

save_staging_orders builds row dicts whose keys ARE the interface to order_staging.py, which lives in a different directory (OneDrive/trading_ibkr) and submits real IBKR orders. Column renames or drops (e.g. Fill_Window_Days, Path1_Bps, Manual_Limit) would only be discovered when live staging misbehaves pre-market.

**Expected impact.** Column drift between the scanner and the IBKR-side consumer fails in CI or at write time instead of at order submission.

**How to test / implement.** Commit contracts/staging_columns.json listing required columns per tab (Order_Staging/Overflow, moc_orders, Seasonal) with types. (a) tests/test_staging_contract.py: call save_staging_orders with a mocked gspread client and 2-3 synthetic signals per entry type (LOC companion, persistent GTC, T+1 open, OVS with path columns), assert the written header row is a superset of the contract and that no contract column is empty-typed. (b) Runtime: in save_staging_orders, before worksheet.update, assert the frame contains all contract columns, else raise (which triggers the existing error email). The contract file doubles as documentation order_staging.py can be checked against (idea 12).

#### infra-quality 8. R2 upload sanity guard: refuse-to-shrink plus dated backups for master_prices

Effort: low. Category: data integrity

update_master_prices.py does an unguarded read-modify-write to the single most critical artifact, with two schedule triggers per day plus a local dispatch and (for earnings) an intentional dual local/GHA writer. A truncated yfinance response or a race could replace a good 25-year cache with a damaged one, and every downstream consumer (scan, ledger, report, site) would inherit it the same day.

**Expected impact.** A corrupted or truncated cache write becomes a failed workflow with a one-command restore path, instead of a same-day poisoning of every consumer.

**How to test / implement.** In scripts/update_master_prices.py before upload_from_local: download the current R2 object's row count and max date (or store them in a tiny sidecar master_prices.meta.json to avoid re-downloading 200MB); refuse to upload if new rows < old rows * 0.995 or new max date < old max date, exit 1 so the workflow fails loudly. Add a weekly backup step (Mondays, in the same workflow): server-side copy master_prices.parquet to backups/master_prices.YYYYMMDD.parquet via boto3 copy_object, plus an R2 lifecycle rule keeping ~8 weeks. Same guard pattern for earnings_calendar. Unit-test the guard function with synthetic old/new stats.

#### infra-quality 9. Best-effort step failure alerting and site staleness badges

Effort: low. Category: monitoring

deploy_site.yml deliberately continues past chart rendering, seasonal ideas, and risk JSON failures (continue-on-error, and build_risk_json always exits 0). That is the right availability tradeoff, but today a step can fail every run for a month with zero signal to you; the site just quietly serves aging ideas.json/risk.json.

**Expected impact.** Best-effort stays best-effort, but failures become visible within one run instead of being discovered by noticing the ideas page looks familiar.

**How to test / implement.** Two halves. (a) Workflow: give the best-effort steps ids, add a tail step `if: steps.charts.outcome == 'failure' || steps.ideas.outcome == 'failure'` that emits a ::warning:: and sends a short email (reuse EMAIL_USER/EMAIL_PASS secrets already present in the repo). (b) Site: build_site.py already writes meta.json; add per-payload generated-at timestamps (ideas_at, risk_at, charts_manifest_at) and have site/assets JS render a small amber/red badge in the header when a payload is older than 36h/72h. The badge logic is ~20 lines of vanilla JS matching the existing no-framework style.

#### infra-quality 10. Unify and pin dependencies across all workflows

Effort: low. Category: CI / reproducibility

daily_screener.yml installs `pandas pyarrow numpy yfinance gspread google-auth boto3` unpinned while deploy_site.yml uses requirements.txt; other workflows vary. A pandas or yfinance major release lands directly in the production scanner with no test in between (and yfinance breaks often). The scanner and the ledger builder can also silently run different pandas versions, which is its own parity hazard.

**Expected impact.** Eliminates dependency-drift breakage in scheduled prod runs, the second most likely silent-failure vector after data staleness.

**How to test / implement.** Create requirements-lock.txt via pip-compile (or a simple pinned requirements.txt with == pins for the ~10 prod deps). Point every workflow's install step at it. In the CI workflow (idea 2), run the suite against the same lock so upgrades are made by editing one file and watching CI. Add a monthly reminder (or Dependabot config scoped to pip) to refresh pins deliberately.

#### infra-quality 11. Cross-repo alignment test for order_staging.py (local-only, skip in CI)

Effort: medium. Category: contract testing

CLAUDE.md documents at least four conventions that must stay aligned with order_staging.py / eq_order_entry.py in OneDrive/trading_ibkr (OVS_CYCLE_MULTS vs cycle_risk_mults, the Friday EOD-DD weekday gate, Entry_Expire_Time math vs fill_window_days, stop goodAfterTime arming). All are enforced today by remembering. The live-vs-backtest OVS P2 divergence noted in CLAUDE.md is proof this drifts in practice.

**Expected impact.** The known-drifting boundary between this repo and the IBKR-side executor gets a mechanical check, catching exactly the class of divergence (OVS P2) already sitting unresolved.

**How to test / implement.** tests/test_ibkr_alignment.py with pytest.mark.skipif(not Path(IBKR_DIR).exists()). Where importable, sys.path the OneDrive dir and assert order_staging.OVS_CYCLE_MULTS == strategy_config OVS execution['cycle_risk_mults']; where import is unsafe (TWS connection at import), fall back to reading the source text and asserting anchored patterns (e.g. `weekday() == 4` present iff eod_dd_weekdays == [4], Fill_Window_Days handling present). Also assert order_staging reads every column in contracts/staging_columns.json (idea 7). Run it as part of a local pre-commit or a weekly Task Scheduler entry since GHA cannot see OneDrive; it skips cleanly in CI.

#### infra-quality 12. Single-source filter engine consumed by both scanner and backtester

Effort: high. Category: architecture

The structural fix behind idea 1: as long as check_signal and get_historical_mask are separate implementations, parity is a treadmill. One vectorized implementation can serve both, since a live scan is just the mask's last row. Do this only AFTER the parity harness exists, so the refactor is provably behavior-preserving (including deliberately preserving or fixing the ETF_ATR_EXEMPT divergence).

**Expected impact.** Permanently retires the scan-vs-backtest drift bug class instead of detecting it; every future filter is written once.

**How to test / implement.** New root module filter_engine.py: build_filter_mask(df, params, ctx) returning a boolean Series, extracted from get_historical_mask filter by filter (trend, liquidity gates, perf ranks, sznl, counts, 52w, ranges, dow/cycle/month). strat_backtester imports it; daily_scan.check_signal becomes `build_filter_mask(df, params, ctx).iloc[-1]` plus the genuinely scan-only pieces (ETF_ATR_EXEMPT policy, intraday-partial handling) expressed as explicit ctx flags rather than forks. Migrate incrementally, one filter family per commit, keeping the parity test (idea 1) and golden ledger (idea 5) green at each step; leave the interactive pages/backtester.py UI engine out of scope (it is a deliberately separate exploration surface).

### Strategy alpha research agenda

*Track notes:* Grounding: data/backtest_trades_full.parquet, 3033 trades 2003-01..2026-06, R_Multiple basis. Per-strategy: OVS n=1115 avgR 0.429 totR 479; OLV n=349/0.537/187R; 52wh n=256/0.434/111R; MonFri n=210/0.457/96R; Sector BO n=90/0.876/79R; Indices OSB n=307/0.256/79R; LT Trend n=246/0.315/78R; 3x Fade n=76/0.908/69R (PF 5.81); Weak Close n=186/0.324; ATR Gap Up n=72/0.782; Monday Dip n=68/0.461; St OS Sznl n=58/0.416. 2026 YTD is the worst year since 2015 (avgR 0.13, win 47.7%). MAE/MFE were computed on the fly from data/master_prices.parquet (adjusted daily bars, entry-day touches excluded per book convention) since the ledger itself lacks those columns.

NEGATIVE results from my verification cuts — do not re-litigate these without new data: (1) Tighter targets / scale-outs LOSE money everywhere: OVS all-exit-at-1R = -140R, half@1R = -70R, all@1.5R = -49R; same sign for Indices OSB, LT Trend, St OS Sznl. Current 2-ATR targets are on the right side of the frontier. (2) Disaster stops on the no-stop strategies destroy value: OVS trades with MAE>=2R realize -1.21 avg vs -2.0 booked (delta -0.79/trade, n=103); catastrophic for ATR Extended Gap Up (MAE>=1.5 trades actually realize +0.20). Only Indices OSB (+0.19 at 2R, n=14) and St OS Sznl (+0.24, n=8) show marginal positive deltas — too small to act on. Consistent with the book's day-2 stop-arming and use_stop_loss=False conventions. (3) MOC time exits beat MOO on the final day for every strategy except ATR Extended Gap Up (+0.04R there, immaterial) — the final session accrues +0.06 to +0.18R/trade, which is what motivates the hold-EXTENSION sweep (idea 5) rather than truncation. (4) OVS low-RV drag (0.31 vs 0.50+ avgR) failed LOYO — low<rest in only 9/18 years — so no vol-tiered OVS sizing was proposed.

Caveats: ledger reflects GLOBAL_RISK_MULTIPLIER=1.5 sizing stamps but R_Multiple is size-independent; ledger starts 2003 though configs cite 2000 backtests; overflow tier dominates OVS/52wh/LT Trend counts, so any accepted change should be checked per-tier before shipping. All experiment scripts should follow the repo pattern: scratch/*.py study -> LOYO/split-half validation -> config change + aligned sites + tests/, per the OLV T+3 and cycle-tilt precedents.

#### strategy-alpha 1. Sector BO stop-width redesign (1.0 ATR stop is choking a 4-8R winner engine)

Effort: low. Category: exit engineering / risk placement

Ledger: Sector BO has 90 trades, avgR 0.876 but win rate 26.7% because 65/90 (72%) stop out at -1.06R on a 1.0 ATR stop, while survivors are huge (11 targets at +8.0R, 14 time exits at +4.27R avg). A 63-day hold with a 1-ATR initial stop almost guarantees noise stop-outs; the sibling 52wh Breakout already runs 2.0 ATR. If even 15 of the 65 stopped trades survive to the winner distribution, totR gains are material. This is the single largest per-trade headroom in the book.

**Expected impact.** +15-40R equivalent over 23y on a 90-trade strategy (potentially +0.3-0.5 avgR-equivalent); high plausibility because the winner distribution is already proven

**How to test / implement.** Re-run process_signals_fast (via a scratch driver like scratch/stop_gap_slippage_impact.py) on SECTOR_INDEX_ETFS 2003-2026 with stop_atr swept over {1.0, 1.5, 2.0, 2.5, 3.0}, tgt 8 ATR and 63d hold frozen, stop gap-fill + 13bps slippage model on. Metric: dollar PnL at CONSTANT 25 bps per-trade risk (R renormalizes when the stop widens, so compare dollars, not raw R), plus PF and max drawdown. Also compute MAE of the 65 currently-stopped trades to see what fraction would survive each stop width. Confirm: >=15% dollar-PnL improvement with LOYO stability (improvement holds in >=70% of leave-one-year-out folds, not driven by 1-2 episodes). Kill: improvement concentrated in <3 years or 2016+ subsample flat. Multiple testing: 5-point pre-registered sweep with a monotonicity prior (wider stop -> fewer stop-outs); report the neighborhood average around the best point, never the best cell alone.

#### strategy-alpha 2. 52wh Breakout fill-window cap at T+10 (direct analog of the validated OLV T+3 study)

Effort: low. Category: entry refinement / fill-window tuning

The 52wh Breakout persistent -0.5 ATR limit stays live for the full 63-day hold. Ledger fill-day buckets: fills 11-63 BD after signal (n=32) average +0.172R vs ~+0.93R for day 2-10 fills (n=70) — a limit that fills a month later is filling into a failed breakout. Killing the stale tail costs only ~5.5R of 111R total while raising avgR and freeing capital, exactly the OLV T+3 pattern (there: totR flat, avgR +0.637 vs +0.566). Mechanism already exists generically (execution['fill_window_days']), so implementation is a one-line config change once validated. Bonus diagnostic: day-1 fills (n=154) also underperform at +0.233R — a breakout that retraces 0.5 ATR the very next day has weak follow-through.

**Expected impact.** +0.03-0.06 avgR on a 256-trade strategy, better capital velocity, near-zero totR cost; very high plausibility given the OLV precedent

**How to test / implement.** Re-run the engine with fill_window_days swept over {5, 10, 21, 63} for 52wh Breakout (and Sector BO as a secondary, though its buckets are non-monotone: days 4-10 = -8.9R over n=21 but days 11+ = +20.6R over n=14, so expect noise there). Metric: avgR, PF, totR, and capital-days tied up in unfilled GTC orders. Confirm: totR within -10R of baseline while avgR/PF improve, stable in split-half (2003-2015 vs 2016-2026). Kill: totR drops >15R or the effect only exists pre-2016. Multiple testing: single pre-registered hypothesis borrowed from an already-validated mechanism (OLV), 4-point sweep only; the day-1-fill finding is exploratory and must be re-derived on the candidate set (not the ledger) before acting.

#### strategy-alpha 3. Resolve the OVS Path-2 live-vs-backtest divergence with a pre-committed decision study

Effort: low. Category: live/backtest alignment + conditional sizing

CLAUDE.md flags this unresolved: live order_staging retired the OVS mild-gap Path 2, but the ledger still models it. My ledger split confirms P2 = 417 trades, avgR +0.195, +81R (vs P1 698 trades at +0.569). P2 is real but thin, and it is weakest exactly where you'd expect noise: low-RV regimes (P2 avgR 0.14 in bottom-tercile SPY 21d RV, n=157, vs 0.28/0.21 in mid/high). The book's largest strategy (1115 trades, 479R) currently has a ledger that doesn't match live — every downstream number (site, portfolio report) inherits the bias.

**Expected impact.** Either +81R of ledger honesty (P1-only) or recovered live P2 income; removes a known 407-trade divergence between the site/report numbers and what actually trades

**How to test / implement.** Dataset: full ledger P2 subset + engine re-run with ovs_p1_only toggled. Three pre-committed branches decided BEFORE looking: (a) compute P2 net economics at live 8 bps sizing including 2 bps slippage and commissions — if net expected dollars per year < ~$1.5k, flip the ledger to P1-only and close the divergence; (b) if positive, test ONE conditional refinement chosen a priori: P2 only in mid/high RV terciles (mechanism: mean-reversion fades need vol to harvest); (c) LOYO across 18 years — the RV-conditioned P2 must beat unconditional P2 in >=12/18 folds to justify the added rule. Metric: net dollars at live sizing, not R. Multiple testing: this is a decision study with thresholds fixed in advance, one conditional cut allowed, no mining.

#### strategy-alpha 4. New strategy: LT Downtrend ST Overbought short (bear-regime mirror of LT Trend ST OS)

Effort: medium. Category: new strategy / factor-exposure complement

The book has a structural regime hole: every long strategy is uptrend-gated (52wh Breakout and OLV have literally ZERO trades with SPY below its 200 SMA — gated by construction) and the shorts are overbought fades that need vol spikes. In a grinding bear market the book goes idle or bleeds: 2026 YTD is the worst year in the ledger since 2015 (avgR 0.13, win 47.7%, n=170; OLV -11.2R). A mirror of the proven LT Trend ST OS template — short-term overbought rallies (2/5/10/21d ranks all >85) in long-term LAGGARDS (252d rank <35, below 100/200 SMA for 20/50 consecutive days), close in top 15% of range, today's move >= +0.25 ATR, short via persistent limit at close +0.25 ATR, 1-2d hold, 2 ATR target, no stop, ±10 TD earnings blackout — would fire precisely when the rest of the book is dark.

**Expected impact.** Potential +5-15R/yr in regimes where the current book earns ~nothing, plus book-level drawdown smoothing; medium plausibility (short-side mean reversion is harder: borrow, squeeze tails)

**How to test / implement.** Build in pages/backtester.py on LIQUID_PLUS_COMMODITIES. Split: design and tune ONLY on 2000-2015; run 2016-2026 exactly once as untouched holdout. Metrics: OOS avgR, PF, and — the real point — R generated in months where SPY < 200 SMA (book-complementarity, measured as correlation of its daily PnL to book daily PnL). Confirm: OOS avgR >= 0.20, PF >= 1.5, negative-to-zero PnL correlation with the dip-buy complex. Kill: OOS avgR < 0.10, or edge exists only in 2008/2020 crash months (then it's just short-vol beta the OVS already owns). Multiple testing: the design is fixed a priori as the exact parameter mirror of LT Trend ST OS — no filter tuning allowed on the holdout; if the mirror fails as-specified, the idea dies rather than mutates.

#### strategy-alpha 5. Hold-length sweep for the short-hold time-exit strategies (extend, don't truncate)

Effort: medium. Category: hold-length optimization

Direction matters: my exit-day decomposition shows the FINAL session still accrues edge — LT Trend ST OS time exits realize +0.229R at the close vs +0.054R at the same day's open (+0.175R earned on exit day itself); same sign for OVS (+0.111 vs +0.045) and St OS Sznl. Meanwhile my simulated tighter-target variants all LOSE money (OVS all@1R: -140R; half@1R: -70R), so the drift is not exhausted at current exits — the cheap test is whether one MORE day of hold adds. Separately, St OS Sznl's 5-day time exits are outright negative (-0.457R, n=32, avg MAE 1.70R) — its hold may be too LONG. Per-strategy optimum is unknown and has never been swept systematically.

**Expected impact.** +0.05-0.15 avgR on 1-2 strategies (LT Trend at 246 trades and St OS Sznl); the negative St OS Sznl time-exit tail (-14.6R across 32 trades) alone is worth recovering

**How to test / implement.** Engine sweep of hold_days over {h-1, h, h+1, h+2} for the seven strategies with hold <= 5d (LT Trend 1d, OVS/Indices OSB/MonFri/Weak Close/Monday Dip 2d, St OS Sznl 5d), all other parameters frozen, target still checked intraday per engine convention. Metric: paired per-strategy avgR and totR deltas. Validation: split-half by odd/even years — require sign agreement in both halves AND effect > +0.05 avgR before changing anything. Multiple testing: ~7 strategies x 3 non-baseline holds = 21 comparisons; apply Benjamini-Hochberg FDR at 10% across the family, and treat any single-strategy result that fails FDR as descriptive only. Expect 1-2 real hits (LT Trend +1d and St OS Sznl -2d are the priors).

#### strategy-alpha 6. Validate the already-wired earnings-quality / analyst-grade filters (currently all use_*=False)

Effort: medium. Category: entry refinement / new data

strategy_config.py already carries a complete but dormant filter block (use_eps_surp_filter, use_rev_surp_filter, use_grades_filter) with source data present: data/earnings_calendar.parquet (117k rows with derived eps_surprise_pct / rev_surprise_pct) and data/analyst_grades.parquet. Nothing has ever validated thresholds. The natural first customers: 52wh Breakout (a volume-confirmed breakout backed by a positive EPS surprise is a fundamentally different animal from a squeeze) and OLV (a dip into a name whose last report MISSED is a value trap risk the low-volume filter can't see). Plumbing cost is zero — the runtime already treats the block as a no-op.

**Expected impact.** +0.1-0.2 avgR on the ~600 combined 52wh/OLV trades if the fundamental conditioning is real; also de-risks the overflow tier where name quality is lowest

**How to test / implement.** Join the 52wh Breakout and OLV candidate sets (pre-portfolio-constraint signals from the engine, not just taken trades) to last-reported eps_surprise_pct sign and net analyst-grade changes in the trailing 30d. Metric: avgR and PF by bucket (positive/negative/no-data, with no-data passing through per the NaN-as-True convention). Confirm: positive-surprise cohort beats negative by >= 0.15 avgR with n >= 50 per cell and the spread present in both 2010-2017 and 2018-2026 halves; then flip the wired flag on. Kill: spread < 0.10R or era-unstable. Multiple testing: pre-register exactly two strategies x two one-sided hypotheses (breakout better with beat; dip-buy worse after miss) = 4 tests, FDR-controlled; do NOT screen all 12 strategies against all 5 fundamental fields.

#### strategy-alpha 7. St OS Sznl retire-or-merge decision study (thin, overlapping, negative time exits)

Effort: low. Category: book hygiene / strategy consolidation

St OS Sznl is the weakest link on the evidence: n=58 over 22y (11 liquid), avgR 0.416 driven entirely by 26 target exits (+1.49) while its 32 time exits average -0.457R with 1.70R average MAE — the non-winners bleed for the full 5 days with no stop. Its filter set (2/5/10/21d ranks < 15) is nearly identical to LT Trend ST OS and overlaps OLV; it likely fires on the same tickers on the same dates and just holds the same signal longer and worse. Low n also makes every future study on it underpowered.

**Expected impact.** Removes a -14.6R time-exit bleed and one source of same-day risk stacking; frees a strategy slot and 40 bps of risk budget for ideas 4 or 9

**How to test / implement.** Overlap audit on the candidate sets: for each St OS Sznl signal, does LT Trend ST OS or OLV also fire same ticker within ±1 day? Then run the counterfactual: route St OS Sznl signals through the LT Trend exit template (1d hold, 2 ATR tgt) vs its own 5d/1.5 ATR, on the full history. Metrics: incremental book totR from keeping it as-is vs merged vs retired, plus daily PnL correlation to the rest of the dip-buy complex. Pre-committed decision rule (set BEFORE running): if unique, non-overlapping contribution < 10R over 20y, retire it and let the sibling strategies absorb the signals. Multiple testing: none to speak of — this is a portfolio decision with one counterfactual, not a search.

#### strategy-alpha 8. Same-day signal clustering: top-K selection + engine-vs-live daily risk cap parity

Effort: medium. Category: portfolio construction / risk budgeting

All five dip-buy strategies fire on the same market-wide down days, so book risk concentrates precisely when correlation goes to 1 (worst single-day ledger outcomes cluster: worstR -5.9 to -6.0 across OVS/Indices OSB in the same vol episodes). Live order_staging enforces a global 2.5% daily risk cap, but process_signals_fast does not model it — the ledger overstates what live can actually deploy on cluster days AND nobody has tested whether ranking the cluster beats pro-rata truncation. The engine's own docs admit same-day selection is alphabetical (docs/backtesting_logic.md 'Bias Warning'), which is pure noise as a tiebreaker.

**Expected impact.** Mostly risk-adjusted: 20-30% lower cluster-day drawdown at <15% R cost, plus fixes a known optimism gap between ledger and live capacity

**How to test / implement.** Dataset: ledger grouped by Entry Date. First measure: distribution of same-day book risk, and avgR of trades on days with 1-2 vs 3-5 vs 6+ simultaneous entries. Then simulate top-K selection under the 2.5% cap with two pre-specified rank keys: (a) signal-depth (21d rank distance below threshold for longs / above for shorts), (b) cross-sectional 252d rank. Metric: fraction of totR retained vs reduction in daily PnL vol and max drawdown at flat $750k sizing. Confirm: any ranking that retains >= 85% of totR while cutting cluster-day VaR >= 25%; also quantifies the ledger-vs-live gap for the site. Kill: if ranked selection retains no more R than random/pro-rata selection (then just model the cap and move on). Multiple testing: exactly two pre-registered rank keys; no key mining.

#### strategy-alpha 9. 3x ETF Overbot Fade family expansion (best edge in the book, starved for trades)

Effort: medium. Category: new strategy candidates / capacity

3x ETF Overbot Fade is the book's highest-quality signal: avgR 0.908, PF 5.81, 75% win rate, pure 2-day time exit — but only 76 trades in ~15 years (~5/yr) because it demands ALL of 2/5/10/21d > 85th with 21d 3-consecutive plus 126d AND 252d < 65th. RV conditioning shows it's robust everywhere (avgR 0.48 low-vol to 1.37 extreme). The scarcity is the only problem; adjacent parameterizations are untested, and the long-side mirror (multi-horizon oversold 3x ETFs, non-laggard) has never been tried — though leveraged-ETF decay drag makes the long side a genuinely different bet.

**Expected impact.** Doubling n at even 60% of current avgR adds ~+40R/15y from the book's cleanest signal; long mirror is a lottery ticket (low prior)

**How to test / implement.** Grid: thresholds {80, 85} x consec {1, 3} x LT-rank cap {65, 75} on LEV3X_ALL, 2010-2026 (3x ETFs barely exist earlier). Report the FULL grid, not the best cell; require the expansion cell to sit in a smooth neighborhood (adjacent cells within 0.15 avgR of each other) to accept. Separately, one long-mirror run with a priori mirrored parameters. Split: 2010-2019 build / 2020-2026 validate. Metric: n, avgR, PF; accept an expansion if n >= 150 with avgR >= 0.5 and OOS consistency. Kill: avgR collapses below 0.4 as filters loosen (edge lives in the extreme tail) or long mirror shows the decay drag eats the bounce. Multiple testing: 8-cell grid with neighborhood-smoothness acceptance criterion explicitly guards against cell-picking.

#### strategy-alpha 10. Trailing-stop test for the two 63-day breakout strategies (engine flags exist, never enabled)

Effort: medium. Category: exit engineering

52wh Breakout and Sector BO both carry use_trailing_stop: False, trail_atr: 2.0, trail_anchor: 'Peak High' in config — the machinery exists but has never been evaluated. Sector BO time exits average +4.27R and 52wh time exits +1.30R at day 63; the open question is giveback: how far below their peak MFE do these trades finish? Caution prior: my scale-out simulations on the short-hold strategies uniformly destroyed value (-5 to -140R), and momentum literature says tails pay for everything here — so this test is as likely to protect the current design from a future 'obvious improvement' as to find one. A measured negative result has real value for a book whose exits keep getting second-guessed.

**Expected impact.** Either +10-25R of giveback recovery on ~350 combined trades, or a documented negative result that locks in the current 63d fixed-hold design

**How to test / implement.** First a cheap ledger-side measurement: compute peak-MFE minus realized-R (giveback) for all 52wh/Sector BO target+time exits from master_prices bars. If median giveback < 1R, kill immediately without touching the engine. If giveback is large, run the engine with trailing stop variants {2.5, 3, 4} ATR off Peak High, armed only after +2R unrealized (protects the early noise phase). Metric: dollar PnL at constant risk, PF, and the 8-ATR-target hit count (the tail must survive). Confirm: PnL up AND >= 80% of the 8R winners retained, LOYO-stable. Kill: any loss of tail winners that isn't paid for 2:1 by saved giveback. Multiple testing: 3-point sweep, arm threshold fixed a priori, giveback pre-screen acts as a gate before any parameter search happens.

#### strategy-alpha 11. Sector BO calm-regime dial gate (mirror the 52wh Breakout 63d-dial < 30 filter)

Effort: low. Category: regime filter

52wh Breakout ships with a production-validated fragility gate (63d dial 10d-avg < 30) precisely to avoid buying breakouts into panic-vol chop; Sector BO — the same trade type on sector ETFs — has dial_filters: []. Ledger regime cuts hint the gate would bite: Sector BO avgR is 0.08 in extreme-RV (n=9) and 0.32 below the 200 SMA (n=9) vs 0.94 in a healthy tape, and 72% of all its entries stop out at 1 ATR, many plausibly in exactly these windows. Small n keeps this from being conclusive, which is why it's a test and not a change.

**Expected impact.** +0.1-0.2 avgR on 90 trades if regime is the stop-out driver; cheap because both the filter code and the precedent exist

**How to test / implement.** Re-run Sector BO through the engine with the identical dial filter added ({'dial': '63d', 'window': 10, 'logic': '<', 'thresh': 30.0}), 2003-2026. Metric: avgR/PF/totR delta and, specifically, the stop-out rate delta. Confirm: totR within -5R of baseline while stop-rate falls >= 8 points and avgR rises >= 0.10 — i.e., the gate removes mostly losers (same acceptance shape as the OLV window study). Kill: the gate removes winners proportionally (regime doesn't matter, the 1-ATR stop does — which feeds idea #1 instead). Multiple testing: single pre-registered filter transplanted from a sibling strategy with an articulated mechanism; threshold NOT tuned (30 is inherited, not searched).

#### strategy-alpha 12. OVS Monday-signal risk tilt (weakest weekday cell, mechanism-light — strictly LOYO-gated)

Effort: low. Category: conditional sizing / calendar

OVS Monday signals average +0.25R (n=171) vs +0.57/+0.56 for Wed/Fri (n=292/254) — the largest weekday spread in the book's largest strategy. Candidate mechanism is thin (Monday overbought readings may reflect weekend-gap continuation rather than exhaustion), which caps conviction; but the cycle-year tilt (0.75x in midterm years) established both the statistical bar (LOYO stability, ~1.5 sigma after clustering, shrunk-Kelly sizing) and the four-site implementation pattern (cycle_risk_mults), so testing and shipping a weekday analog is cheap.

**Expected impact.** ~+3-5R/decade of avoided drag if real; deliberately last-ranked because the mechanism is weak and the selection effect is the largest of any idea here

**How to test / implement.** Dataset: 1115 OVS ledger trades split by signal weekday. Test: Monday vs rest avgR gap with episode-clustering-adjusted SE (trades on the same date are one observation, exactly as the cycle-tilt study did), then leave-one-year-out: require the Monday deficit in >= 13/18 active years. If it passes, ship as weekday_risk_mults {0: 0.75} (shrunk, never 0x) mirroring cycle_risk_mults across the four aligned sites. Kill: LOYO shows the gap driven by <= 3 years, or clustering-adjusted significance < 1.5 sigma. Multiple testing: this is 1 cell selected from a 5-weekday x 12-strategy scan (60 implicit tests) — apply a Bonferroni-style discount by demanding ~2.5 sigma clustered OR an out-of-sample confirmation window (2024-2026 candidate signals only) before sizing down.

### Portfolio construction and sizing

*Track notes:* Measured diagnostics behind these ideas (data/backtest_trades_full.parquet, 3,033 trades 2003-2026, flat $750k basis; analysis scripts left in the session scratchpad as book_analysis.py / book_analysis2.py): (1) Strategy diversification is already excellent: avg pairwise monthly PnL correlation 0.018, max pair 0.27 (MonFri x Monday Dip); same-ticker same-day overlap across correlated pairs is near zero, so co-movement is temporal (market-wide oversold days), not doubled names. (2) Full-book daily-realized Sharpe 2.25; leave-one-out deltas: OVS +0.299, OLV +0.178, then a steep drop to four strategies adding <= 0.04 each (Weak Close +0.001). (3) Signal clustering is alpha, not risk: fwd10d book PnL rises monotonically with same-day entry count ($4.8k at 0 entries to $35.1k at 7+, 5.7% negative); the 2.5% daily cap bound on 34 days that averaged $32.8k fwd10d, i.e., the cap cuts the best days. Same-day variance is however elevated (worst-1% days average 2.4 entries). (4) Concentration: 76% of trades are ETF/index; SPY/QQQ/^GSPC/^NDX/DIA/IWM/SMH = 21.5% of trades = one factor; single-stock sectors are well spread and all profitable. Peak open at-risk $69k (9.2% of equity). (5) Fragility (2016-07+ only): long avgR decays monotonically with f63 (+0.657 / +0.399 / +0.368 / +0.257 / +0.059 across 0-20/20-40/40-60/60-80/80+), shorts do not (+0.524 fragile vs +0.491 calm); daily book Sharpe -0.24 at f63>=80 (N=29 days). Long-only downsize counterfactual raised both PnL and Sharpe (2.90 to 2.95); whole-book downsize was worse than long-only. (6) Drawdown deleveraging fails: fwd21 after DD<-2z is +$10.2k mean, 20% negative, p5 -$6.0k, the best tail of any DD state. (7) OVS P2: N=417, avgR +0.195, t=3.20, +0.036 book Sharpe, stable across cycle years, but only ~$3.2k/yr flat. Caveats: ledger bps embed GLOBAL_RISK_MULTIPLIER=1.5; fragility history is short and the score was partly designed on its own 2016-2026 episodes (ideas 9 and 10 exist to de-risk this); fwd-return conditioning on entry counts overlaps the book's own positions, so cluster-day results partly reflect the strategies working as designed. Note: scripts/backtest_signal_overlap.py from the task brief does not exist; the file is tests/backtest_signal_overlap.py (SPY forward returns by risk-dashboard signal overlap count, a different object from strategy-entry clustering). Relevant repo surfaces for implementation: strategy_config.py (execution dicts, CROSS_STRATEGY_OVERLAP_OVERRIDES pattern), pages/strat_backtester.py sizing step 3b2 (cycle_risk_mults is the template for frag_risk_mults), pages/fragility_sizing_lab.py (replay engine to reuse), order_staging.py in OneDrive/trading_ibkr (daily cap + OVS path logic), scripts/portfolio_analytics.py (daily PnL matrix construction).

#### portfolio-construction 1. Walk-forward risk budgeting: size strategies by shrunk edge, not ad-hoc bps

Effort: medium. Category: risk_budgeting

Current bps are nearly flat (35-60 pre-multiplier) and uncorrelated with realized edge. Measured on data/backtest_trades_full.parquet: OVS t-stat 11.7 and OLV t-stat 7.4 vs St OS Sznl t-stat 2.1 and Sector BO t-stat 2.4, yet St OS Sznl runs at the top 60 bps tier (with GLOBAL_RISK_MULTIPLIER 1.5) while OLV runs 41. Leave-one-out Sharpe deltas: OVS +0.299, OLV +0.178, Weak Close Decent Sznls +0.001, St OS Sznl +0.010. PnL-per-unit-annual-vol ranges 28.9 (OVS) down to 9.4 (St OS Sznl). Reallocating the same total daily risk toward high-t strategies should raise book Sharpe without changing gross risk.

**Expected impact.** Est. +0.1 to +0.3 book Sharpe from shifting budget out of the four strategies whose marginal Sharpe is under 0.05, with unchanged gross risk

**How to test / implement.** PnL is linear in bps on the flat $750k basis, so replay the ledger with rescaled per-strategy multipliers (same pattern as pages/fragility_sizing_lab.py replay_equity). Walk-forward: each Jan 1 from 2013, compute per-strategy shrunk weight w_s proportional to max(0, t-stat of trailing-10y avgR) shrunk toward equal weight (James-Stein style, shrink factor 0.5), renormalize so sum of (w_s x trades/yr x bps) equals the current book's expected daily risk, cap any strategy at 35% of total budget and floor at 0.5x current bps. Apply to next-year trades, compare full-period Sharpe, maxDD, worst day vs status quo. Gate with the block-bootstrap harness (idea 9): require P(Sharpe improves) > 70% across resamples. Ship by editing execution['risk_bps'] in strategy_config.py; no engine change needed.

#### portfolio-construction 2. Fragility-conditioned sizing on the LONG side only (not the whole book)

Effort: medium. Category: fragility_sizing

Joining trades to data/rd2_fragility.parquet (2016-07+): long avgR decays monotonically with 63d fragility, +0.657 (f63 0-20, N=916) to +0.399 (20-40) to +0.257 (60-80) to +0.059 (80+, N=25). Shorts do NOT decay: +0.524 fragile vs +0.491 calm, so whole-book downsizing throws away the hedge that works. Daily book Sharpe at f63 80+ is -0.24 (N=29 days) vs +3.08 below 40. Counterfactual already run: longs x0.5 at f63 60-80 and x0 at 80+ RAISED PnL by $13k AND Sharpe 2.90 to 2.95 since 2016. Whole-book version (x0.5/x0.25) gave only 2.92 and cost $28k. It is nearly free insurance against exactly the regime the risk dashboard was built to flag.

**Expected impact.** Small direct PnL gain (+$13k over 10y measured) but removes the worst-regime long tail; main value is crash-state protection at zero expected cost

**How to test / implement.** Implement as a generic execution field frag_risk_mults: {60: 0.5, 80: 0.0} applied only when trade_direction == Long, mirroring the existing cycle_risk_mults plumbing in strat_backtester step 3b2, daily_scan sizing, and order_staging. Validate the way the cycle tilt was validated: leave-one-episode-out (dedupe f63>=60 stretches with a 21-trading-day gap, expect N of 6-10 episodes), NOT by re-running with the rule on. Placebo test: shuffle the fragility series in 63d blocks 500x and confirm the real avgR monotonicity is above the 95th percentile of placebos. Caveat to check explicitly: the fragility score's regime multiplier was partly calibrated on 2016-2026 episodes, so also run on the reconstructed pre-2016 series from idea 10 before trusting the 80+ bucket (only 25 trades).

#### portfolio-construction 3. Symmetric calm-regime upsize: lever the book where the Sharpe actually lives

Effort: low. Category: fragility_sizing

The fat part of the sample is calm, not fragile: 916 of 1,821 fragility-matched trades landed at f63 < 20 with avgR +0.657 and 68.1% win rate, and daily book Sharpe below f63 40 is 3.08 vs 2.44-2.46 in the 40-80 band. exposure_leg.py already encodes this instinct for the buy-and-hold overlay (x1.25 when both dials < 5). Downsizing fragile states touches ~6% of trades; upsizing calm states touches ~70% and is where a sizing edge compounds. Combined with idea 2 this becomes a fragility tilt rather than only a brake.

**Expected impact.** If avgR holds, +10-20% book PnL at modest incremental risk taken in the statistically best regime; also makes idea 2 budget-neutral overall

**How to test / implement.** Counterfactual replay on the ledger (flat basis, linear in size): longs x1.25 when f63 < 20 AND f21 < 20, x1.0 otherwise, stacked with idea 2's downsize. Report Sharpe, maxDD, worst day, and the new daily-risk distribution vs the 2.5% cap (upsizing will push more days toward the cap; count the extra binding days). Then stress: apply the same rule to 2007-2008-shaped resamples via the bootstrap harness to confirm the calm gate does not simply proxy 'bull market so far'. Sensitivity grid over threshold {10, 20, 30} x mult {1.15, 1.25, 1.4}; require the edge to be flat across the grid, not a spike at one cell. Ship as the same frag_risk_mults field with a below-threshold entry.

#### portfolio-construction 4. Diversity-aware daily cap: stop the 2.5% cap from cutting the best days of the year

Effort: medium. Category: correlation_aware_caps

This inverts the usual intuition and the data is emphatic. Days with 7+ entries have fwd10d book PnL mean $35.1k, median $23.0k, only 5.7% negative, p5 +$532 (N=35) vs $5.9k mean unconditionally. The 34 days where staged risk exceeded the 2.5% cap ($18.75k flat) were followed by fwd10d mean $32.8k, 12% negative; the list is a who's who of capitulation lows (2018-02-09, 2018-10-11, 2020-era, 2022-03-08, 2025-01-08). Cross-checks: same-ticker same-day overlap across correlated strategies is essentially zero (0-4 instances per pair over 24y), so the clustering is temporal capitulation breadth, not doubled-up names. Pro-rata scale-down on those days is selling the book's single best conditional state.

**Expected impact.** Recovers most of the alpha currently truncated on ~1.5 capitulation days/yr; measured fwd10d on those days is 5-6x the unconditional mean

**How to test / implement.** Reconstruct every entry day's staged-risk set from the ledger and re-run the cap logic (mirror order_staging's pro-rata scale-down) under: (a) current flat 2.5%; (b) tiered cap 4.0% when entries span >=3 strategies AND >=4 distinct underlying complexes AND f63 < 60, else 2.5%; (c) uncapped control. Compare total PnL forgone by each cap, worst single day, maxDD, and the p1 of daily PnL. Key risk to quantify: same-day variance is elevated (worst-1% book days average 2.4 same-day entries vs 0.5 unconditionally), so report the same-day and fwd5d left tail specifically on the days the raised cap admits. Decision rule: adopt (b) if PnL recovered > 3x the widening of the p1 daily loss. Change lands in order_staging's global cap block plus a Cap_Tier stamp from daily_scan.

#### portfolio-construction 5. Concentration limits by underlying-complex bucket, not GICS sector

Effort: medium. Category: concentration_limits

Sector caps on single stocks would be solving a non-problem: single-stock trades are well spread (max sector = Technology at 158 trades over 24y, all sectors profitable). The real concentration is the index complex: 76% of trades are ETF/index, and SPY, QQQ, ^GSPC, ^NDX, DIA, IWM, SMH alone are 653 trades (21.5%) on what is one underlying factor; max same-day same-bucket entries hit 57 (2023-12-14). 3x ETF fades and sector ETF trades also collapse onto a few factors. Open long at-risk already reaches $69k (9.2% of flat equity) on stacked days, and the top-5% open-long days carry a p5 daily PnL of -$9.5k vs -$3.6k unconditionally.

**Expected impact.** Trims the one real single-factor tail (index complex stacking) with minimal PnL cost; likely 10-20% worst-day improvement on stacked-long days

**How to test / implement.** Build a ticker-to-bucket map (index-complex incl. levered aliases; one bucket per sector-ETF cluster from a 252d correlation dendrogram; single stocks by sector; commodities; rates). Replay the ledger chronologically tracking open at-risk per bucket; when a new entry would push a bucket above X% of equity, scale it down pro-rata. Grid X in {0.8%, 1.2%, 1.6%, 2.0%}. Report per-X: PnL forgone, maxDD change, worst-day change, binding frequency per bucket. Given idea 4's finding, exempt or loosen the cap when the breach comes from a diversified multi-strategy cluster day. Expect the answer to be a fairly loose cap (1.5-2%) that only binds the index complex; adopt only if maxDD improves more than 2x the PnL forgone. Implementation: bucket map in strategy_config.py, enforcement in order_staging next to the global cap.

#### portfolio-construction 6. Codify NO drawdown-based deleveraging; delever only on drawdown x fragility intersection

Effort: low. Category: drawdown_rules

Tested and the classic rule fails here: after book drawdowns deeper than -2z (z = DD dollars / 21d-scaled trailing 63d PnL vol), fwd21d book PnL is +$10.2k mean, 20.0% negative, p5 -$6.0k, which is BETTER than at equity highs (+$15.5k mean but p5 -$11.7k and 22% negative). A mean-reversion book's own drawdown is a capitulation state, the same state idea 4 shows is alpha. Mechanical deleveraging would systematically cut size at the bottom. The only state where forward book Sharpe measured negative is f63 >= 80 (-0.24, N=29 days), which is an exogenous market-state signal, not a book-PnL signal.

**Expected impact.** Primarily avoids a value-destroying rule (est. -5 to -15% PnL if naively adopted); the intersection variant may add a genuine crash brake if it survives the extended sample

**How to test / implement.** Two-arm backtest on daily flat PnL: (a) classic deleveraging (x0.5 when DD < -1.5z, restore at new high) vs (b) no rule vs (c) intersection rule: x0.5 only when DD < -1.5z AND f63 >= 70 simultaneously. Count the days each rule is active, PnL delta, maxDD delta, and time-to-recovery. Expect (a) to lose money outright (the -2z bucket has the best forward tail in the sample). For (c), N will be small since 2016; report it honestly and re-run on the idea 10 extended fragility series before adopting. Regardless of outcome, write the evidence table into CLAUDE.md as the documented decision so the rule is not re-litigated from intuition during the next drawdown.

#### portfolio-construction 7. OVS P1-only vs 2-path: re-enable P2 live at 8 bps, or flip the ledger, decided by one number

Effort: low. Category: ovs_decision

The unresolved live-vs-backtest divergence (live retired P2, ledger still models it) is a measured question now. P2 in the ledger: N=417, avgR +0.195, totR +81.4, t-stat 3.20, so the edge is statistically real, and it is stable where it matters: by cycle year avgR ranges +0.12 (midterm) to +0.26, rolling-5y avgR is +0.171 today and was only ever negative around 2005. Book Sharpe with P2 2.249 vs 2.213 without (+0.036). But dollar contribution is $73k over 23y flat, about $3.2k/yr ($4.8k at the 1.5x global multiplier), because the 8 bps size and 0.75-1% aggregate cap keep it tiny (avg risk $834/trade vs $2,520 for P1).

**Expected impact.** Either +$3-5k/yr of real edge restored live, or removal of a persistent optimistic bias from ledger, site, and portfolio report

**How to test / implement.** Decision procedure: (1) Run scripts/build_trade_ledger.py twice with the engine's ovs_p1_only flag on/off, diff Sharpe/PnL/maxDD to confirm the +0.036/+$73k numbers on current code. (2) Compute expected live $/yr at actual live sizing and compare against the operational cost of re-adding the mild-gap branch to order_staging (the P1 fixed-dollar-target logic must not clobber P2 rows, per the OVS_CYCLE_MULTS note). (3) Test one refinement before deciding: P2 gated to f21 < 40 and non-midterm years, since P2's weakness concentrates in midterms (avgR +0.12); if gated P2 avgR clears +0.30, re-enable gated; if not, re-enable plain (t=3.2 stands on its own) OR set ovs_p1_only=True permanently in build_trade_ledger so site/ledger/report stop displaying PnL live cannot earn. Either endpoint kills the divergence; leaving it unresolved is the only wrong answer.

#### portfolio-construction 8. Cluster-day tail overlay: finance index put spreads out of capitulation-day alpha

Effort: high. Category: tail_hedging

Ideas 4-6 all conclude 'stay big in clusters and drawdowns', which concentrates the residual risk in one scenario: a capitulation cluster that keeps falling (the 1-in-35 cluster day that breaks the pattern, unobserved in-sample). Same-day left tail on stacked days is real: top-5% open-long-risk days have p5 daily PnL -$9.5k and worst -$54.2k. The book already owns the machinery for cheap convex hedges: docs/fragility_alpha_playbook.md's put debit spread structures with measured theta recovery (97-99% premium recovery over 5-15 days if unneeded). A conditional overlay converts a little cluster alpha into protection for exactly the state where the book is maximally levered to further downside.

**Expected impact.** Caps the one unhedged scenario the other recommendations amplify; expected cost 2-4% of book PnL for a large improvement in the unobserved-tail p1

**How to test / implement.** Backtest on daily data: whenever same-day entries >= 5 AND open long at-risk > p90, buy a synthetic 21 DTE SPY put spread (strike near ATM, width equal to 1.5x the 21d expected move) sized so premium <= 15% of the measured fwd10d cluster alpha (~$5k notional premium per event, ~2-3 events/yr). Price the spread from the intraday/vol history or a Black-Scholes approximation off ^VIX; credit payoff against the book's realized fwd21d PnL path. Compare hedged vs unhedged: total PnL drag, p1/p5 of fwd21d PnL, maxDD. Accept if drag < 20% of cluster alpha while p5 improves > 30%. Execution would ride the existing Seasonal/manual ticket path, not the systematic staging pipeline.

#### portfolio-construction 9. Block-bootstrap policy harness: a gate every sizing change must pass

Effort: medium. Category: methodology

Every decision above rests on one historical path with clustered episodes and thin tails: 29 days at f63 >= 80, 34 cap-binding days, 35 seven-plus-entry days, 25 fragile-bucket trades. The repo's own culture already demands this (cycle tilt was LOYO-validated, not resimulated), but there is no reusable tool, so each study reinvents validation in scratch/. Point estimates on N ~ 30 states will flatter whichever rule was fit to them.

**Expected impact.** No direct PnL; converts every other idea from a point estimate into a probability statement and prevents shipping rules fit to 30 observations

**How to test / implement.** Write scripts/bootstrap_book.py: load the per-strategy daily PnL matrix (built from the ledger the way scripts/portfolio_analytics.py does), stationary block bootstrap with ~21-day blocks preserving cross-strategy same-day structure (resample days jointly, never per-strategy), 2,000 resamples. API: pass a policy function mapping (date, strategy, direction, fragility, entry-count, open-risk) to a size multiplier; returns the distribution of delta-Sharpe, delta-maxDD, delta-worst-day vs baseline, plus P(improvement). Unit test with a null policy (multiplier 1.0 everywhere must return zero deltas) and a known-negative policy (halve OVS, must show Sharpe loss). Then run ideas 1-6 through it and record pass/fail in the decision log.

#### portfolio-construction 10. Reconstruct the fragility score back to ~2004 to power every fragility-conditioned test

Effort: high. Category: data_infrastructure

All fragility conditioning currently starts 2016-07 (data/rd2_fragility.parquet, 2,511 days; fragility_63d_history.parquet is no deeper). That leaves 29 days at f63 >= 80 and misses the three richest stress laboratories in the ledger's span: 2008, 2011, 2015-16. The dashboard's inputs are mostly reconstructable to 2003-2004 from yfinance/FRED (^VIX, ^VIX3M from 2009 with VXV splice, sector ETFs for breadth/absorption from 1999, HYG/LQD from 2007, ^SKEW from 1990, MOVE via FRED). Tripling the fragile-state sample is worth more than any single rule refinement, and it also out-of-sample tests the score itself, since 2008 played no part in its design.

**Expected impact.** 3-4x the fragile-regime sample behind ideas 2, 3, and 6; determines whether the f63>=80 long-side collapse (avgR +0.06) is real or a 25-trade artifact

**How to test / implement.** Factor the signal computations out of pages/risk_dashboard_v2.py into a shared module (careful: the module-boundary rule says risk_dashboard_v2 must stay standalone, so extract downward into a new leaf module both can import, importing nothing from the strategy stack). Rebuild daily 21d/63d scores 2004-2016 with documented input substitutions and a graceful signal-count renormalization where an input is missing (the score is signal-count-driven, so fewer available signals rescales cleanly). Validation: the reconstructed 2016-2026 segment must correlate > 0.95 with the stored series; then check the reconstructed score flags 2008-09, 2011-08, 2015-08 with f63 > 80 without being told. Re-run ideas 2, 3, and 6c on the extended sample and re-report bucket avgRs.

### Research methodology

*Track notes:* Overall verdict on the research process: the repo has two cultures. The ml/ layer (config.py pre-registration, cv.py purged embargoed walk-forward, evaluate.py bootstrap CIs + ship/no-ship verdict + disclosed inherited biases) is genuinely strong methodology. The discretionary rules entering strategy_config live in a different world: hypotheses formed and cutoffs chosen from full-sample grids in scratch/ scripts, evidence recorded only as prose comments, no registry of alternatives tested, no decay reviews. Risk ranking for in-sample flattery / multiple testing: (1) seasonal midterm 13 bps — no study, and a live parameter whose own comment admits a 5x ambiguity (seasonal_order_staging.py:62-65); (2) OVS cycle tilt — 6 year-clusters, ~1.5 sigma pre-family-correction, post-hoc P1 slice, partial-2026 data likely inside the evidence set for a rule applied to the rest of 2026; (3) OLV T+3 — cutoff picked from an inspected grid, though downside is bounded (totR unchanged by construction, the claim is capital efficiency); (4) day-2 stop arming — N=81 episodes for a book-wide convention that changes tail shape, not just mean; (5) OVS blackout/2-path parameters — undocumented selection history, single-day anecdotal validation; (6) gap-fill slippage — conservative direction, lowest risk, but constants are assumptions not measurements. Cross-cutting integrity issue: the acknowledged live-vs-backtest OVS P2 divergence means backtest_trades_full.parquet (which every study and the ML dataset reads) models 407 trades live would never take — fix before further ledger studies. Files reviewed: C:\\Users\\McKinley Slade\\dev\\New_Seasonals\\scratch\\olv_fill_window.py, scratch\\stop_gap_slippage_impact.py, scratch\\olv_clusters.py, ml\\config.py, ml\\cv.py, ml\\evaluate.py, strategy_config.py:522-534, seasonal_order_staging.py:60-67.

#### methodology-critic 1. Adopt the ml/ pre-registration protocol for every strategy_config rule change

Effort: medium. Category: process / governance

ml/config.py already demonstrates the right discipline (pre-registered features, thresholds, ship criteria, 'changing after looking at test folds invalidates the evaluation'), but discretionary rules entering strategy_config (cycle tilt, T+3 window, blackout, 2-path params) bypass it entirely. Study scripts in scratch/ record results but never the hypothesis or the grid of alternatives considered, so multiple-testing accounting is impossible retroactively.

**Expected impact.** Eliminates silent forked-path selection; makes every future rule auditable and its multiple-testing burden explicit

**How to test / implement.** Create docs/rule_validation_protocol.md plus a research/ registry: before running a study, commit a short entry (rule, hypothesis, metric, exact cutoff grid to be scanned, minimum N, accept/reject criteria, decay-review date). Enforce mechanically: a test in tests/ that hashes each strategy's execution dict in strategy_config.py against a manifest; any changed key without a registry ID fails CI. Registry entry template mirrors ml/evaluate.py's ship_verdict structure.

#### methodology-critic 2. Time-box the OVS midterm 0.75x tilt with a pre-registered 2026 checkpoint and expiry

Effort: low. Category: re-validation / decay review

N_effective is 6 year-clusters at ~1.5 sigma, one partition of a large implicit family (4 cycle phases x ~10 strategies x several metrics), with the 'concentrated in P1' claim a post-hoc slice. The evidence window 2006-2026 apparently includes partial-2026 YTD data, and the rule was then applied to the remainder of 2026 — the current year is contaminated as an out-of-sample test. The shrunk, risk-reducing direction makes it acceptable insurance, but only if it cannot silently become permanent.

**Expected impact.** Converts a ~1-sigma calendar bet into a bounded, falsifiable experiment instead of a permanent book fixture

**How to test / implement.** Write the decision rule now, before year-end: (1) re-run the midterm study excluding all 2026 data to confirm the 5-cluster effect stands; (2) at 2026-12-31 compare live OVS results vs the untilted counterfactual (daily_scan already stamps the mult into Sizing notes, so the counterfactual is a column away); (3) add an expiry to the strategy_config comment — tilt auto-reverts to 1.0x before the 2030 midterm unless the 2026 checkpoint plus a fresh LOYO on 7 clusters re-passes.

#### methodology-critic 3. Resolve the seasonal 13 bps midterm ambiguity and demand strategy-native evidence or drop the tilt

Effort: low. Category: bug / unfounded rule

seasonal_order_staging.py:62-65 ships a live sizing parameter whose own comment admits a 5x ambiguity ('if you meant a 0.13x MULTIPLIER instead, set this to 2.6'). The tilt has zero direct evidence on seasonal-ideas trades — it is borrowed from OVS, a mean-reversion vol-spike strategy with a different signal universe. Borrowed effects across strategies are exactly how weak calendar noise metastasizes through a book.

**Expected impact.** Removes a live sizing parameter with a factor-of-5 ambiguity and no supporting study

**How to test / implement.** First, resolve the number explicitly and delete the hedging comment. Then run the cycle-phase split on the seasonal pipeline's own trades (data/seasonal_ideas_backtest.parquet and the enriched variant already exist): if midterm underperformance does not replicate at least directionally with a defensible N of season-year clusters, remove SEASONAL_MIDTERM_RISK_BPS entirely — an unfounded knob is worse than none.

#### methodology-critic 4. Fix the OVS P1-only live vs 2-path backtest divergence before running any further studies on the ledger

Effort: medium. Category: data integrity

CLAUDE.md acknowledges order_staging retired Path 2 live while the ledger still models 407 P2 trades (+82R/24y). Every downstream study reads backtest_trades_full.parquet — the OLV window study, the gap-slippage study, the ML dataset, and notably the cycle-tilt evidence (whose P1/P2 decomposition was load-bearing). Research conducted on a ledger that no longer matches live execution inherits a systematic bias in every OVS slice.

**Expected impact.** Stops a known live/backtest divergence from contaminating every future OVS-touching study

**How to test / implement.** Decide the open question (re-enable P2 live or flip the engine to ovs_p1_only) and rebuild the ledger. Then add provenance: scripts/build_trade_ledger.py stamps engine flags + build date into the parquet metadata, and every scratch/ study prints that vintage at the top of its output so stale-ledger studies are self-identifying.

#### methodology-critic 5. Split-half stability test for grid-selected cutoffs: OLV T+3, OVS 0.25 ATR gap threshold, ±10 earnings blackout

Effort: low. Category: multiple-testing control

olv_fill_window.py prints the full T+3/5/7/10 grid and the winner was chosen from it — a garden of forked paths. The avgR gain (0.637 vs 0.566) compares nested overlapping samples; the real question is whether the dropped day-4-10 set (avgR ~0, small N) is reliably worse, and small-N subsets of a grid are where noise wins. Same concern applies to the OVS 0.25 ATR threshold and the ±10 blackout window, whose selection histories are undocumented (the cited validation is one day's 13 signals).

**Expected impact.** Distinguishes real dose-response patterns from cutoff noise on three shipped parameters; establishes the habit for future grids

**How to test / implement.** Extend each study script with a split-half protocol: choose the cutoff on 2003-2014 only, verify the same cutoff (or a neighbor) wins on 2015-2026; report both halves' per-day/per-window profiles. For the blackout, add a ±5/±10/±15 sensitivity table with cluster-aware (per-earnings-event) resampling. Where the winner is unstable across halves, prefer the least restrictive setting or the round-number prior.

#### methodology-critic 6. Regime-split re-validation of the day-2 stop-arming convention

Effort: medium. Category: re-validation / tail risk

The -33R cost of day-1 arming was measured over just 81 episodes in 24 years, and the rule changes the loss distribution's shape, not just its mean: dip-buy limits fill at maximum fear precisely when day-1 gaps can be catastrophic. An average funded by benign periods can mask a sign flip in crisis regimes, and 81 episodes cannot resolve that tail. The economic logic is sound, but the position taken (book-wide default) exceeds what N=81 unconditionally supports.

**Expected impact.** Confirms or bounds the tail exposure of a book-wide convention adopted on 81 observations

**How to test / implement.** Re-run the episode study conditioned on VIX-at-entry terciles and on crisis windows (2008, 2011, 2018Q4, 2020, 2022): report day-1 vs day-2 R deltas with bootstrap CIs per bucket, plus worst-single-episode under each convention. If the high-VIX tercile flips sign materially, evaluate a vol-conditional arming rule; if not, document the accepted tail explicitly in the strategy_config comment.

#### methodology-critic 7. Calibrate the 3/13 bps stop-slippage constants and 20% gap-through rate against realized live fills

Effort: low. Category: model calibration

The gap-fill change is the healthiest rule reviewed (it worsens the backtest to match reality), but its constants are assumptions. verify_fills.py already reconciles live fills post-close, so the data to replace assumption with measurement exists and is accumulating for free.

**Expected impact.** Backtest/ledger/site PnL converges to measured execution reality instead of plausible guesses

**How to test / implement.** Extend verify_fills.py to log, for every live stop-out, modeled stop level vs realized fill (slippage in bps) and whether the bar gapped through. Quarterly script compares the realized slippage distribution and gap frequency to the 3/13 bps and 20% backtest figures; update constants only when live N >= ~30 stop-outs and the discrepancy exceeds the live sample's CI.

#### methodology-critic 8. Codify minimum-evidence standards: effective N in clusters, sigma thresholds scaled by rule direction, paper burn-in

Effort: low. Category: process / standards

Current effect sizes vary wildly against the positions taken: 6 clusters at 1.5 sigma bought a permanent sizing tilt; 81 episodes bought a book-wide convention; zero evidence bought the seasonal midterm downsize. A written standard prevents relitigating each case and makes the asymmetric-risk logic (risk-reducing shrinkage needs less proof than risk-adding leverage) explicit rather than ad hoc.

**Expected impact.** Every future rule faces a known evidentiary bar proportional to the risk it adds; weak effects get shadow-tracked instead of shipped

**How to test / implement.** Add a standards table to the protocol doc: (a) N_effective counted in independent clusters (years for calendar rules, episodes for event rules), minimum ~30 for return-enhancing rules; (b) >=2 sigma after cluster adjustment for return-enhancing rules, 1-1.5 sigma acceptable only for risk-reducing shrinkage capped at 25% downsize; (c) rules resting on <100 historical trades or <10 clusters require a 6-12 month shadow period where daily_portfolio_report tracks rule-on vs rule-off counterfactual PnL before the rule is graduated to permanent.

#### methodology-critic 9. Scheduled decay reviews for all calendar-conditioned and data-dependent rules

Effort: medium. Category: decay review / monitoring

Calendar effects decay and none of the shipped rules has a review date. The cycle tilt only generates evidence 1 year in 4, the OLV fill-day profile can drift with market microstructure, and the earnings blackout silently degrades as earnings_calendar.parquet coverage rots (NaN-as-True means missing data disables the filter without any signal).

**Expected impact.** Rules get falsified on schedule instead of persisting as fossilized comments; the blackout's silent data-coverage failure mode becomes visible

**How to test / implement.** Annual GHA job (mirroring risk_report.yml infra) that re-runs each rule's supporting statistic on the refreshed ledger and emails a one-page table: cycle-tilt midterm-vs-other avgR gap, OLV per-fill-day quality profile, blackout in-window vs out-of-window OVS avgR, blackout coverage rate (fraction of OVS signals with earnings data), seasonal cycle stats. Alert when a supporting stat's sign flips or magnitude halves; each alert links back to the rule's registry entry for a re-validation decision.

#### methodology-critic 10. Stop reusing the same OOS folds across ml/ feature-addition runs — reserve a final holdout

Effort: low. Category: multiple-testing control (ML layer)

ml/config.py RUN_NOTE shows four sequential runs (base, ortho, NAAIM/grades, fragility) evaluated against the identical 2012+ walk-forward folds with unchanged ship criteria. Each run peeks at the same test years, so the effective false-discovery rate compounds even though each individual run is honest — the one weakness in an otherwise exemplary pipeline.

**Expected impact.** Restores the validity of the pre-registered ship criteria that sequential test-set reuse has been quietly eroding

**How to test / implement.** Freeze the two most recent complete years as a never-inspected final gate: feature iteration (runs 1-N) uses walk-forward folds ending two years back via the existing last_test_year parameter in cv.walk_forward_splits; only the single candidate configuration selected from those runs gets scored once on the reserved years. Additionally require any run-5+ to show positive uplift on live trades accrued since run-4 (ml/monitor.py already collects them) before another full evaluation is run.

## Part 4 — Coverage gaps flagged by the completeness critic

- The entire live-execution stack in C:/Users/McKinley Slade/OneDrive/trading_ibkr (exec_agent.py, execute_order.py, eq_order_entry.py, order_staging.py, plus credentials.json and exec_agent.env) is not a git repository (verified: `git rev-parse` fails) — the code the audit's highest-severity findings live in has no version control, no rollback, no review path, and its only 'backup' is OneDrive sync that also replicates plaintext broker/Sheets secrets to Microsoft's cloud.
- Google Sheets is an unauthenticated command channel into IBKR that no dimension audited: order_staging.py auto-submits whatever rows sit in Order_Staging/Overflow, and the GCP service-account key that can write those tabs is sprawled across GHA secrets (GCP_JSON in 5+ workflows), Streamlit Cloud st.secrets, and a plaintext OneDrive credentials.json with no rotation policy — one leaked copy means arbitrary live orders.
- trading_ibkr contains a whole unaudited live order-modification subsystem — div_adjust.py (with DIVIDEND_ADJUSTMENT_DESIGN.md, div_adjust_pending.json, heartbeat/self-test artifacts) re-prices working orders on ex-div events, exactly the frozen-vs-adjusted basis area CLAUDE.md flags as the book's most dangerous invariant — plus intraday_store.py and book_snapshot.py, none touched by any dimension.
- No disaster-recovery story for the local machine: the AM workflow_dispatch triggers have GHA fallback crons, but order staging/entry (IBKR-bound), the ExecAgent scheduled task, the GH_PAT living only in HKCU registry, and the Sunday radar tasks all die silently with this box — new signals simply stop being submitted with no alert, and no finding or improvement track covers it.
- Confirmed sibling of the risk_report git-push race: .github/workflows/daily_screener.yml:135 and update_cboe_putcall.yml:49 both use `git push || true`, which is strictly worse — a lost push of data/exposure_state.json or the CBOE put/call parquet is swallowed silently instead of failing the job.
- Confirmed sibling of the FMP error-as-empty bug: scripts/build_analyst_grades.py:73 returns [] for any non-200/non-429 response, so an FMP 5xx classifies a ticker as 'no analyst grades' — the strategy-alpha idea to enable the analyst-grade filters would inherit the same silent-data-deletion failure mode.
- Probable sibling of the winter partial-bar finding: update_intraday_prices.yml runs at a fixed 20:45 UTC = 15:45 ET in winter (verified cron), capturing an in-progress final 15-minute bar into the R2 intraday cache; keep='last' dedupe self-heals next day, but same-evening Day Trade Limit backtests via intraday_data.py consume the partial bar.
- The Streamlit deployment itself is unaudited: app.py/cache_io.py show a Streamlit Cloud deployment (IS_CLOUD, st.secrets holding R2 + GCP keys) with unknown public/private status, and 13 of 16 pages/ (fx_sizer.py, fragility_sizing_lab.py, exposure_backtester.py, rotation_backtester.py, signal_backtester.py, seasonal_sigs.py, heatmaps, etc.) fell outside every audit dimension despite some feeding sizing decisions.
- Findings were filed against code that isn't deployed: .github/workflows/rebuild_overflow_universe.yml is untracked with its cron commented out (the dynamic-overflow project is uncommitted per memory), and the seasonal forward track record (data/seasonal_ideas_log.parquet and friends) plus presentation/ exist only on this disk — compounding the ledger-clobber finding with total-loss risk.
- The radar pipeline is unaudited end-to-end: radar_weekly_summary.py shells web-derived brief content into a Claude Code subprocess and commits/pushes the output to main (prompt-injection into a committed+emailed artifact), depends on the separate last30days-radar project under the stale 'mckin' profile path that CLAUDE.md still cites (actual profile is 'McKinley Slade'), and has no failure alerting.
- Notification-channel health is uncovered: daily_risk_report.py:646 silently skips sending when EMAIL_USER/EMAIL_PASS are unset (print-only), the same Gmail app-password pattern serves portfolio/rundown emails, and there is no inventory or rotation schedule for the secret sprawl (GH_PAT in registry, FMP key, R2 keys, CLOUDFLARE_API_TOKEN, EMAIL_PASS, GCP_JSON) beyond a single 'rotate annually' note for the PAT.
- Two scheduled workflows and their data products are absent from CLAUDE.md's pipeline table and every audit dimension: build_indicator_cache.yml (Mondays 07:00 UTC) and update_cboe_putcall.yml (weekdays 21:30 UTC), plus the committed root-level data files they and the filters depend on (atr_seasonal_ranks.parquet, sznl_ranks.csv, naaim.csv, market_dates.csv) whose staleness is checked by nothing.
