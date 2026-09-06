# Legend activation and dial hedge review — September 6, 2026

## Decision

Legend live execution is not ready to enable. The existing code tests pass, and the machine's futures cache has now been initialized from already-purchased data, with all three integrity checks passing. Primary TWS refused the configured local connection. The runtime account configuration, shared guard manifest, daily portfolio budget, and broker paper-proof artifacts are absent. No trading task, live date, order, or new paid-data permission has been enabled by this review.

Do not activate the automatic hedge on the evidence reviewed here. The original rolling-return-beta study reproduces, but it is not the proposed holdings-beta strategy. A direct implementation of the proposed sizing idea loses money across all three dial vintages in a new diagnostic replay, with little drawdown improvement. That is a reason to retain the manual calculator and reconsider the automatic rule, not tune parameters until a profitable backtest appears.

## Legend: work completed and remaining

- The five existing Legend test files pass: **88 tests**, one third-party calendar deprecation warning. Installed package versions exactly match `requirements-legend-etf.txt`.
- The full 2016–August 2026 replay passed on this source tree: **342 candidates across 2,660 complete sessions**, with no missing, extra, or duplicate candidates. Research, frozen reference, and production direction/contract choices agree for all 342. The largest ATR difference is floating-point noise (5.68e-14). Source and input hashes are retained in the parity evidence.
- Initialized the missing machine-global futures cache under `%LOCALAPPDATA%/NewSeasonals/legend_etf/futures_cache` from the purchased archive: ES 140,386 rows, NQ 140,366, RTY 137,747; each ends August 31 23:59 UTC. Each cache was read back through the integrity validator and matched exactly. No API data request or charge occurred. It still needs a current refresh before producing a new signal.
- A read-only metadata quote for the three-feed thirty-day refresh totaled **$0.307337641715**. Approval for a $0.50/trading-day ceiling has been requested and remains pending. The existing paid-data ceiling remains zero. Databento documents [cost quotes before data requests](https://databento.com/docs/api-reference-historical).
- `scripts/configure_legend_etf_shadow.py` now prepares Primary-only shadow configuration after connecting read-only to the configured Primary TWS port and verifying exactly one live-account identity. It refuses an existing configuration, keeps live flags/date/allowlist disabled, permits no paid data, and registers no tasks. Tests cover wrong-account rejection and disabled execution/spend.
- The current runbook's statement that executor support was not deployed is stale. The September 2 deployment receipt records that support deployed successfully. However, the shared guard activation marker is still absent and the deployment receipt explicitly did not arm Legend.
- Some current Execution actions are already hard-disabled: generic cancel/modify, Add, and Trim/Re-add. Those unconditional rejections are present in the deployed `execute_order.py`, not introduced by this review. Do not confuse enabled frontend controls with successful backend execution. Preserve the currently supported close/resize/flatten routes while resolving the remaining lifecycle work.

The remaining live-release requirements are substantive: observed complete shadows; broker paper proof of the IOC parent, partial fills, target revisions and 10:30 OCA/GAT exit; attestation of the final deployed runtime; and a real shared-capacity publisher. The budget validation/library exists, but this review found no scheduled publisher or live budget artifact. Do not manufacture a passing proof file or a capacity number to satisfy a validator.

IBKR documents that [OCA type 2 proportionately reduces siblings with block, and Good After Time governs activation](https://www.interactivebrokers.com/docs/tws-api/ref/order). That contract does not establish this application's exact timed-exit behavior. It must be observed with broker events. The [NYSE calendar](https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf) closes September 7; the next regular equity session is September 8.

## Hedge review: methods and results

The old study uses lagged dial thresholds 50 to enter and below 45 to release, with a 126-session rolling beta of book returns. I independently reproduced its $94,269 PIT overlay P&L and 2.908→3.055 Sharpe result over January 2018–September 1, 2026. It prices the new target over close-to-close returns and charges only on arming. A next-open implementation must carry the old hedge over the overnight gap and charge both entries and reductions/rebalances/exits. The new audit does that; synthetic tests prove a new hedge cannot earn the preceding overnight move and that all quantity changes/terminal liquidation incur cost.

| Diagnostic, $750,000 flat capital basis | Cumulative hedge P&L | Book Sharpe, before → after | Worst drawdown, before → after |
|---|---:|---:|---:|
| Original PIT return-beta proxy | +$94,269 | 2.908 → 3.055 | −7.34% → −7.29% |
| Same sizing, next-open trades and 2 bps on all turnover | +$91,209 | 2.908 → 3.045 | −7.34% → −7.29% |
| Same sizing, live-dial vintage and corrected execution | +$41,655 | 2.908 → 2.989 | −7.34% → −7.29% |
| PIT, 50% of positive holdings beta | **−$81,081** | 2.907 → 2.875 | −7.34% → −7.29% |
| PIT, 100% of positive holdings beta | **−$162,161** | 2.907 → 2.809 | −7.34% → −7.33% |

These are modeled overlay gains/losses, not actual account returns or annual results. The holdings replay uses a separately reconstructed, internally consistent current ledger (4,700 included trades); it is not spliced onto the old study's 4,696-trade book. Both runs end September 1. The half/full ratios are diagnostic comparisons, not newly approved sizing. The full holdings hedge also loses $130,908 on the current-weights dial and $104,869 on the live-vintage dial; the half version has half those losses. In the PIT full version, 2021 contributes approximately −$74,667.

Holdings are reconstructed after each close, excluding positions that exited that date. Per-name 126-session betas use information through that close, then exposure is shifted once and executed next open. No beta fallback was needed in the evaluated window. August holdings-beta values agree closely with the earlier diagnostic where its inventory convention agrees; the earlier diagnostic includes exit-day positions and lags the beta one additional day. The main daily-P&L reconstruction reuses the existing pure site/backtester functions without importing their application entry points. Source/input hashes were checked before and after both replays.

The likely economic explanation is that holdings-sized protection is largest when the mean-reversion book has bought weakness, so short-index exposure offsets the rebound it is trying to earn. The old return-beta hedge keeps a more constant short-market allocation, including when current algorithmic holdings have little exposure. This interpretation is an inference from the replay, not a proved causal result.

## Why no activation

The holdings method has no demonstrated after-cost benefit here and can demand excessive exposure: the uncapped full diagnostic reaches approximately 256% of the flat NAV in hedge notional. Neither replay establishes an executable MES strategy: they use fractional SPY exposure and assumed 2/5 bps costs, omit contract rounding, roll/basis, and broker-quoted margin, and cover the main strategy ledger rather than all active non-core sleeves. The [MES multiplier is $5 per index point](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.html); small contract counts materially change hedge granularity. No current account inventory, executable quote, or margin what-if has been verified because TWS is unavailable.

A replacement would need a distinct objective (directional short alpha versus limiting a specific exposure), capped sizing fixed before testing, all-sleeve attribution, existing-hedge offsets, next-open/resize/exit/roll rules, and broker lifecycle proof. Compare it with no hedge and the existing strategy-specific risk controls. A market hedge can fail when stock-specific losses coincide with a market rally; it also gives up upside when mean-reversion entries rebound. Do not reinterpret a proxy backtest or a futures margin estimate as a release-ready protocol.

## Evidence

- `artifacts/activation/readiness.json`: package versions, missing runtime artifacts, refused Primary port, metadata-only quote.
- `artifacts/activation/cache_seed.json`: verified free cache initialization.
- `artifacts/activation/candidate_parity_evidence.json`: full passing replay, completed September 6 at 17:16 UTC.
- `artifacts/activation/hedge/hedge_audit.json` and `hedge_daily_audit.csv`: original-model reproduction and execution/cost sensitivities.
- `artifacts/activation/holdings_replay.py`, `hedge/holdings_replay.json`, and `hedge/holdings_daily.csv`: holdings-based diagnostic and source hashes.
- Existing evidence: `scratch/ultracode_sizing_2026-09-02/dd_pit/pit_hedge_dd.md`; `docs/plan_2026-09-04.md` D10; `docs/running_list.md` O6; `docs/legend_etf_runbook.md`; September 2 external executor deployment receipt in the prior Legend worktree.

The owner's new request authorizes progressing toward activation and supersedes the older plan's calendar deferral. The remaining reasons for not activating are failed/missing evidence and the pending paid-data approval, not a request to reconfirm that overall intent.
