# Audit repair release — 6 September 2026

This is the current source-repair and rollout record for the whole-repository audit. The isolated branch is `codex/audit-fixes-integration-20260906`, based on `e53c478e232bc37016baed6c285fb8f3b6c6688b`. Original dirty work was preserved. No live broker, scheduler, email, R2 object, canonical dataset, or production site has been changed by these repairs.

## Numbered findings

All 47 numbered findings have source corrections in this branch or its hash-pinned external-executor candidate. Source correction is distinct from production promotion. Native broker behavior and historical data migration have explicit prerequisites below.

| Finding | Corrected behavior |
|---|---|
| E1 | Recovered protection covers remaining exact-contract inventory after confirmed fills and other working closes. |
| E2 | Flatten binds the selected account and conId throughout selection and rereads. |
| E3 | Event exits remain obligations until attributed executions resolve them. |
| E4 | Trend actual, desired, pending and expected quantities are separate; a skipped delta does not change actual holdings. |
| E5 | Working/partial flatten remains uncertain rather than a confirmed rejection. |
| E6 | Exit reporting uses exact account/contract identity, including derivative expiry. |
| E7 | Deadlines still terminate/reap an owned process tree when a child retains stdout after its parent exits. |
| E8 | Preview/dry-run intent is explicit and immutable from ticket through signed command. |
| E9 | Only supported single-long and same-expiry vertical option structures are executable; other analyses are identified as analysis-only. |
| DS1 | Annual Seasonal ranks exclude forward outcomes unknown at the annual cutoff. Production-rounded full-book impact is measured separately. |
| DS2 | An unsuccessful adjustment-basis repair preserves the prior complete ticker history. |
| DS3 | Earnings publication requires a readable baseline; partial failures preserve prior events. |
| DS4 | Annual/total returns and drawdowns include initial capital and first-day PnL. |
| DS5 | 3M/6M/12M use calendar months and explicit as-of dates. Existing entry-cohort semantics remain. |
| DS6 | Missing/nonfinite prices do not become observed zero returns or fabricated dispersion coverage. |
| DS7 | Rotation's next-session-close simulation is accurately labeled. A next-open model is not claimed. |
| DS8 | Singleton yfinance MultiIndex inputs normalize either column orientation. |
| DS9 | Compounding cannot finance an entry with profits realized later. |
| SITE-01 | Invalid account selections reject; they cannot silently become PA. |
| SITE-02 | Fundamental state preserves concurrent updates and rejects corrupt history, using conditional R2 writes. |
| SITE-03 | Retrying an uncertain intent retains its original identity after other tickets or a reload. |
| SITE-04 | Secondary risk acknowledgement retains the original account and execution mode. |
| SITE-05 | Missing/empty broker secrets fail authentication. Deployment checks require nonempty secrets. |
| SITE-06 | Durable queued/delivery-unknown states distinguish unsent or uncertain commands from successful delivery. Offline intents remain visible in Activity. |
| SITE-07 | Execution-family archive and pagination preserve fills beyond 500/day; capped legacy history and malformed source evidence cannot claim completeness. |
| SITE-08 | Site generator and assembler bind the same source and frozen, hashed input generation. |
| R1 | Frozen research tickets are graded against settled RAW prices. Legacy outcomes are corrected append-only on a future authorized grading run. |
| R2 | Corrupt JSONL blocks appends without truncation; concurrent local writers serialize and fsync complete records. |
| R3 | PASS/WATCH remove current reviews while retaining archived evidence. |
| R4 | Only newer qualifying evidence reopens research; DEEPEN completion identifies its exact request revision. |
| R5 | Underwrites validate currency, EV accounting, units, and claim-specific source freshness. |
| R6 | As-of datasets preserve each issuer's eligible archived vintage and statement periods. |
| R7 | EP SMTP acceptance is durable before QUIT; ambiguous delivery cannot silently resend. |
| R8 | Shared Access verification rejects broad/extra/bypass policies and checks all pages. |
| R9 | A failed Trade Log refresh retains the last successful data clock and displays the failure. |
| B1 | Indicator cache identity includes actual prices and all relevant dependencies. |
| B2 | Cross-sectional identity includes exact universe membership and prices. |
| B3 | Coverage accounts for requested missing names; unaffected work continues with explicit exceptions and preserved dated obligations. |
| B4 | Sheets replacements use one atomic cell batch plus readback, with prior-value checks. Concurrent writers still require leases. |
| B5 | Missing verification prices preserve prior status; a total outage fails without rewriting the table. Modeled fills remain distinct from broker fills. |
| B6 | Canonical fill-read failures cannot bootstrap empty replacement history. Immutable generations and conditional publication preserve history. |
| B7 | Effective fills use one latest execution revision per account/family. |
| B8 | Calendar-day retention and explicit source completeness drive gap reporting and automation outcomes. |
| B9 | Entry and batch failures propagate nonzero outcomes; a valid empty-order day is distinct. |
| B10 | Every report path honors no-send; stale/incomplete books cannot be presented as current. |
| B11 | Insufficient risk-price context emits nulls and an explicit insufficient-data label, with strict JSON. |
| B12 | Quote/book helper timeouts kill and reap their children. |

Detailed failure-path evidence: [data and strategy](data_strategy_fixes_2026-09-06.md), [execution and operations](execution_ops_fixes_2026-09-06.md), [research](research_audit_fixes_2026-09-06.md), [external executor](../broker_runtime/README.md).

## Daily workflow changes

- Navigation prioritizes Execution, Portfolio and Seasonal; occasional tools remain under More. The theoretical Portfolio remains the main view, with a separate Theo vs Actual view.
- The comparison displays recent model closes and broker-reported realized-PnL subtotals, explicitly marking missing PnL and coverage. Different sizing/accounting bases are disclosed; it is not an account-return or slippage comparison.
- Manual controls remain accessible with stale/offline snapshots. The executor validates current exact identity; unavailable mode is explicitly warned. Standalone modifications require entry/exit purpose and entry risk rather than guessing it.
- Actual Primary strategy-tagged inventory replaces theoretical Portfolio input for optional sizing overlays and OLV exits. Unknown inventory uses base sizing with an exception, never a fabricated zero. Missing optional dial data also permits the otherwise valid trade with an explicit fallback; known failing gates still apply.
- OLV frozen entry/ATR levels use RAW settled prices. Confirmed exit obligations survive stale feeds, recovered prices and missed auctions until actual fills resolve them. Handoffs identify account, conId, tranche, original entry tag and dates. The external runner rejects ambiguous matching and late market-order substitution.
- The expected-exit monitor checks five minutes after explicit deadlines, preserves unresolved obligations, deduplicates exceptions and records resolution. The 16:10 ET session summary and email claims are durable. A read-only source adapter and protected Execution status endpoint are included. [Monitor contract](expected_exit_monitor.md).
- Routine risk email is removed from local and backup job commands; risk calculation/publication remains. Other existing editorial emails retain their purpose.
- The discovery foundation now includes a dated native algorithm-family catalog. Fit excludes current holdings and distinguishes active algorithms from paper/planned references. This closes the missing family-overlap adapter, not the entire autonomous research process. [Exact boundaries](strategy_family_fit_mvp_2026-09-06.md).

## Verification

Final integrated verification and release-manifest identity are recorded at handoff. Tests use local artifacts, fake brokers/SMTP/R2/Sheets and an offline network guard. External source is parsed/compiled, never imported into a live session. Desktop/mobile UI checks use synthetic intercepted inputs; they are not production-freshness evidence.

The full Seasonal comparison uses all 21 strategy passes and 1,068 required symbols, with identical frozen input and base-cache hashes. Production-rounded results: 24,683 → 24,677 candidates; 3,500 → 3,497 distinct entries; 4,715 → 4,710 trade tranches. Flat-$750k modeled PnL changes by +$6,658. The much larger compounded difference is path-dependent and is documented with the complete results. This is a correction to historical simulation, not evidence to raise risk or a forecast. Canonical history has not been replaced.

## Concrete rollout and remaining prerequisites

1. Prepare the immutable local package with `scripts/prepare_audit_release.py` from the final clean commit. It verifies all external source hashes before producing a private candidate and records every changed source/candidate digest. Keep its candidate under ignored artifacts because it retains existing private configuration.
2. Reconcile a reviewed Primary opening-inventory seed and continuous execution coverage. Do not bootstrap from theoretical positions, an empty broker ring, or the first observed fill. Recover any old capped/lost history from reviewed broker records. Include original entry tags, raw ATR/entry prices and explicit deadlines. [Seed contract](tagged_inventory_adapter.md).
3. Exercise exact native bracket/owner-client behavior in paper/TWS verification before enabling repaired generic trading controls. Reconcile outstanding orders and unknown command claims. This task's fake-broker checks cannot prove IB's live order behavior.
4. Promote one source revision across the pinned local runtime, external candidate, broker Worker and cloud-only site workflow. Stage/register new tasks disabled first, capture old enabled state and runtime identities, and ensure dedicated auction routes are installed before retiring legacy routes. Carry persistent receipts/state forward. Do not run cleanup/prune commands as part of cutover.
5. Run the monitor in observation mode with its reviewed catalog and seed, using `scripts/run_expected_exit_monitor.py`. Its only broker calls read the existing book/fills endpoints. Add `--upload` for the fixed private status key and `--send` for exception delivery only after the operational review. Refresh canonical fills often enough to meet the 90-second monitor contract; old receipts must age out visibly. Scheduling is not currently registered by this work.
6. Build/deploy the private site only through GitHub Actions from authoritative R2 inputs. The cloud generator regenerates corrected Seasonal/ledger history with the reviewed source; local replay files are evidence only. Publish dated research outcome corrections on an authorized producer run.
7. Verify a complete daily cycle, no-order day, source outage, reconnect, partial fill, correction and recovery. Observe runtime/source revisions, broker outcomes, producer coverage and source dates rather than treating task success alone as proof.

Approval is required immediately before financially consequential activation under the workspace's AGENTS.md. Exposure includes changed staged sizes/exits and real Primary orders; a fixed maximum dollar exposure cannot be inferred from this review. Rollback restores the captured prior runtime/executor/site revisions and task states. It does not undo fills; outstanding orders must be reconciled before replaying any job.

Still separate from defect repair: X credentials/source selection and any API budget; SSRN intake; an empirical hypothesis-research runner with costs, robustness and marginal strategy-return/capital-use tests; and the conditional research write-up/delivery loop. These need implementation as well as source setup. No research engine, Legend strategy or automated hedge is claimed to be live by this release. Exact-tranche allocation remains required for ambiguous manual closes; no FIFO/proportional allocation is silently invented. Historical survivorship and calibration limitations remain research limitations.
