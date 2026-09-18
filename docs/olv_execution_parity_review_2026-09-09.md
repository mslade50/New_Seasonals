# OLV: model versus installed execution

> Note (2026-09-18): the task worktrees this review linked to were removed in the repo cleanup. The linked working notes are now tracked in this same `docs/` folder, and copies of the throwaway evidence live in `artifacts/evidence_archive/2026-09-09/`.

Reviewed September 9, 2026. Priority 3, second bounded strategy review. This
review records the pre-change findings. The owner subsequently approved both
target-anchor and pending-cap choices; those source changes and OLV rounding
are now implemented locally. See the
[implementation status](olv_implementation_2026-09-09.md).
The production inventory/stop cutover remains open. No scan, order, email,
scheduler change or production rebuild was performed.

## Recommendation

Restore the actual-inventory-to-volume-stop handoff first. It has a demonstrated operational gap, whereas most entry and sizing rules agree. Then align the model's gap-fill target and integer cap behavior with the installed orders. Decide whether pending entries should reserve the per-stock notional budget before changing that behavior. Preserve the accepted exclusion of entry-day target credit from the daily-bar model.

## Evidence and scope

- Reviewed production source `8952f835066f90570ad0d0f57a433ca9c5b6b6bb` in isolated worktree `codex/olv-parity-review-20260909`.
- Pinned runtime remains `90c30dd698244ef46ff9dc43a74d6bfb0ff5b7e3`. Its strategy configuration, scanner, strategy engine, actual-inventory adapter and pivot policy are byte-identical to the reviewed files.
- Inspected installed `OneDrive/trading_ibkr/order_staging.py`, `eq_order_entry.py`, `olv_exit_moo.py` and the OLV batch launcher without running them.
- Task Scheduler's `IBKR OLV Pre-Market Exits` is Ready and points to that installed batch/script at 09:10 ET. Its September 8 log says no exits were due and returns zero; that is not proof the upstream stop evaluation worked. The old OLV Book Cap task remains Disabled.
- Reproduction script and file hashes: not preserved. The `review.py` and `summary.json` this section used to link lived only in a throwaway task worktree under the gitignored `artifacts/`, were never committed on any ref, and were already gone before the 2026-09-18 cleanup. Fixtures execute the actual model and AST-extracted installed pure functions with inert inputs; no broker module is imported by the reproduction script.

## Rules that agree

| Area | Current executable behavior |
| --- | --- |
| Sizing | Liquid 35 nominal bps, overflow 25, multiplied by GRM 1.5. Recency multipliers 0.5/0.7/1.0 use prior signal days within 21 ticker sessions, including signals that did not fill. This is not the old open-position ladder. |
| Earnings | Signals -10 through 0 trading days relative to earnings use 10 nominal bps as the base, still multiplied by recency and GRM. Effective first-rung earnings risk is 7.5 bps. |
| Entry | Persistent signal-close limit, normally -0.25 ATR. The shared causal 40/40 closing-pivot policy deepens selected entries to -0.50/-0.75 ATR and rejects the specified >5-ATR resistance cases. Pivot source age is capped at 252 ticker sessions. |
| Entry lifetime | T+1 through T+3 inclusive. Installed GTD expiry and model fill window express the same full-session date rule. Late fills retain the original time-exit date, rather than restarting a ten-day clock. |
| Loss rule | No resting stop for current OLV entries. Starting after entry day, a close at/below entry minus 1.25 ATR with volume at least 1.5 times the preceding 20-session median confirms a next-open exit. Quiet breaches are held. Raw bars are required against actual frozen entry/ATR levels. |
| Other overlays | No current OLV fragility throttle or automated book-cap trim. The configured per-stock OLV cap excludes specified ETFs and uses fixed sizing capital, not live broker NLV. |

The signal-recency window's first 21 sessions at a backtest cutoff cannot see earlier candidates; this is an existing sample-boundary limitation. Daily bars also cannot reproduce submission timing, intraday entry/target order, partial fills or every auction outcome. The user's accepted same-day target convention remains appropriate to retain here.

## Findings, in priority order

### 1. Fresh volume-stop decisions are blocked, and the installed consumer is older than the producer contract

Neither the config checkout nor pinned runtime has `.local/tagged_inventory_seed.json`. No alternate `TAGGED_INVENTORY_SEED` is set in the process/user/machine environment or either of the scheduler's actual environment files. Calling the inventory adapter with the runtime's default seed path returns **unknown: reviewed starting inventory is not configured**.

This is supported by dated production evidence: [September 8 PM scan run 34301796391](https://github.com/mslade50/New_Seasonals/actions/runs/34301796391), at 22:08:50 ET, logged that actual inventory was unknown, the optional notional overlay was unavailable, and OLV inventory/exit metadata was unverified so prior staging was preserved. Thus a successful scan receipt does not establish successful OLV stop evaluation. Existing targets/time exits and any previously staged obligations are separate; this is not a claim that every OLV exit has stopped working or that a particular position missed an exit.

The repo includes a newer exact-tranche handoff, but the scheduled installed runner still:

- Selects only rows whose `Execute_On` equals today. Fixture: an unresolved September 8 exit disappears on September 9; the repo contract retains it.
- Matches fresh exits by symbol/strategy and time date, with a fallback to a sole or nearby bracket. Fixture: a requested September 17 bracket adopts the sole September 18 bracket.
- Treats staged quantity as model-based and can exit the full matched live leg instead of honoring the newer exact-tranche quantity contract.
- Processes both Primary and PA and permits a DAY market-order substitution after its cutoff; the newer prepared contract is Primary-only and preserves next-auction intent.

This requires a coordinated repair, not just copying one file. Reconstruct and review actual Primary OLV tranches from broker positions, order references and execution records, including frozen ATR and exit deadlines; make that seed and its continuous fill history available to each producer that can run; then install the matching consumer and verify confirmed, empty, stale-feed, overdue, ambiguous-bracket and partial-fill cases. Display exceptions through the existing status/expected-exit surfaces. Do not populate actual tranches from the theoretical Portfolio ledger or infer missing trade details.

The OVS approval for both accounts was specific to its shared sizing fallback. OLV repair remains Primary-first under the standing scope. The existing runner's PA behavior must be explicitly preserved or separately approved during design; overwriting it with a Primary-only candidate would silently remove its current PA route.

### 2. Gap-improved fills produce different profit targets in Portfolio and live orders

Installed `calculate_bracket_prices` anchors the target to the submitted entry limit. The engine anchors it to its simulated fill, which improves to the session open on a gap below a long limit. The installed entry executor places that precomputed target and does not re-anchor it to the fill.

Reproduced example: signal close 100, ATR 2, limit 99.50, fill 97. Portfolio targets 102; the live target remains 104.50. A subsequent high of 103 therefore closes the modeled trade but cannot reach that live target. This is distinct from the accepted entry-day target timing limitation.

The saved local ledger has 316 OLV entries from May 13, 2005 through September 4, 2026, with complete reconstruction inputs. Seventy-six have a modeled fill below the reconstructed submitted limit; median anchor difference is approximately 0.156 ATR. These are modeled entries in a hash-recorded local artifact, not actual fills or proof of current production freshness. No incremental PnL is claimed without replaying both exit definitions on the same inputs.

Recommended choice: retain the preplaced live target and make the model use its submitted-limit anchor for the OLV profit target. Keep the volume-loss threshold tied to actual entry as its present contract specifies. Moving live targets after every fill would add broker mutation and partial-fill handling; do that only if it is the intended strategy rule.

### 3. The stated per-stock cap does not reserve capacity for pending entries

The scanner counts filled OLV tranches when inventory is known; the model similarly counts positions already filled by the signal date. Neither reserves the notional of waiting entry orders. Missing actual inventory currently bypasses even the filled-position overlay, as separately reported in finding 1.

Fixture with $100,000 sizing capital and an intentionally reduced 20% cap: three signals have unfilled limits, then all fill together at 99.50. Each passes its signal-time check; combined entry notional reaches $59,700 against a $20,000 nominal cap. This establishes the mechanism, not an observed live breach. Production's threshold is 50% of configured sizing capital ($375,000 at $750,000); it is not a guaranteed 50%-of-current-NLV ceiling.

Recommended choice if that number is intended as a firm limit on possible OLV exposure: reserve filled plus remaining pending OLV entry notional by account/contract, releasing reservations on fills, cancellations and expiry, and mirror those reservations in the model. This can reduce new entries and needs an explicit behavior decision. ETF exemptions and the threshold need not change.

### 4. Fixed time exits do not follow early market closes

The installed stager always appends 15:59:00 to OLV time exits and entry expiries. Its calendar counts full-closure holidays but deliberately treats early-close dates as ordinary trading days; the executor uses the supplied activation time. Portfolio books its time exit at that day's daily close.

For example, NYSE lists November 27, 2026 as a 13:00 ET close, so a fixed 15:59 activation is not the intended near-close exit that day. [NYSE calendar](https://www.nyse.com/trade/hours-calendars). The native broker's eventual disposition/fill was not tested, and no after-hours outcome is assumed. Recommended repair: derive the existing one-minute-before-close deadline from the session's actual close, including entry expiry, rather than changing the strategy's hold count. This helper is shared beyond OLV, so its rollout requires a bounded review of those consumers.

### 5. Smaller deterministic share-rounding mismatch remains for OLV

The prior OVS repair intentionally preserved other strategies. Installed daily caps floor OLV shares; the model's generic cap rounds them and scales PnL separately. Reproduced forced-cap example: 203 shares times 0.625 yields 126 live versus 127 modeled. Recommended repair: floor OLV's final quantity and recompute PnL from final shares and modeled entry/exit prices. This does not establish the economic impact of the difference or justify a broad engine rewrite.

The scanner also values a new capped entry at signal close while the model uses its eventual entry price. They can clip different quantities even with the same existing holdings. A pending-order reservation contract should define one pre-fill price basis, rather than using future fill prices to size the modeled order.

## Decisions and next step

First repair the Primary stop pipeline, including inventory provenance and the deployed handoff. Before target/cap changes, confirm two behavioral choices: retain the submitted-limit profit-target anchor and adjust Portfolio to match, and count pending entries toward the existing per-stock budget. Both are recommended above; neither was changed during this review. Handle shared PA and early-close effects explicitly in the concrete rollout plan.

Verification: 101 targeted checks passed, one installed-module import check skipped. Coverage includes OLV stop behavior, T+3 fill boundaries, recency/earnings composition, causal pivot aging and scanner/model contracts, raw inventory input handling and prepared exact-tranche handoff failures. The separate AST/model reproduction script passed all assertions, including the installed loader/selector/target behavior, pending-order cap gap, and daily-cap rounding example. The prepared-handoff tests do not certify the old installed runner. No production code or settings changed.
