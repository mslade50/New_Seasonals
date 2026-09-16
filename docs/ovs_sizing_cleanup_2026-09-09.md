# OVS sizing cleanup — September 9, 2026

Status at 07:43 ET: broker correction installed with owner approval for both Primary and PA-derived orders; pinned producer runtime promoted and validated, including preservation of the approved Execution UI. Cloud Portfolio build is in progress; publication is not yet claimed.

## Result

The prepared OVS fallback uses the effective strategy configuration: P2/P1 = 20%, P2 daily cap = 1.125%. Valid stamped settings retain precedence. Missing, partial, non-positive, non-numeric, or non-finite path settings use the configured fallback and print an explicit warning. The first valid stamped daily cap remains authoritative; when none exists, its fallback is also reported. No scanner or broker connection runs during preparation.

The model now follows the live whole-share sequence for OVS: floor the scanner size, round the P2 multiplier, floor a binding P2 cap, floor a binding strategy cap against the total position, and split near/far. Optional research pooled caps also preserve position-level flooring. Unfilled staged orders still consume their risk budget. When a capped position can no longer split, it becomes a single far-target position; zero-share positions disappear. PnL is recomputed from final shares and existing modeled entry/exit prices. Internal grouping identifiers are removed before returning the ledger.

No other strategy's daily-cap rounding was changed. The engine still applies its daily-cap correction in the existing post-processing pass; this change does not redesign chronological portfolio equity/exposure accounting or certify every modeled-versus-live difference. Risk-dollar columns retain the existing staged-budget meaning rather than being redefined as shares times ATR.

## Preserved owner decisions

- Same-day targets remain active live and excluded from the daily-bar Portfolio model to avoid favorable assumptions about intraday price ordering.
- Friday's timed live loss stop remains unchanged; the daily model retains its closing-price approximation.
- Entry gates, configured risk rates, cycle tilt, brackets, scheduling, and manual execution controls are unchanged.
- No daily scan, trade, or email was run. The approved production-data build runs only through GitHub Actions from R2 inputs.

## Proof

Five new model regressions first failed on the original implementation. They now pass, including the reviewed example: 203 shares scaled by 0.625 becomes 126 shares split 50/76, instead of 127 split 51/76. Further cases cover both cap stages, one/zero shares, unfilled budget consumers, optional sequential caps, and preserved entry-day/Friday exit conventions.

The targeted model, OLV, fragility, earnings-sizing, ledger-provenance, and broker-fallback checks passed: 89 tests in the final combined run. The initial suite's one provenance skip was due to sandbox Git ownership; it passed with process-local safe-directory configuration. Running provenance alone encountered an existing local Streamlit/protobuf import incompatibility; the combined engine suite uses the existing Streamlit stub and passes. All four standalone EOD-DD cases also passed. One existing Plotly/NumPy deprecation warning remains.

Broker tests execute extracted pure candidate functions, not imported installed scripts. Actual-source AST comparison confirms only two fallback constants, the path multiplier function, and the daily-cap lookup in `pull_and_stage_orders` change, with one new pure cap helper. All other top-level executable content is preserved. Source drift fails preparation before any candidate files are written.

## Candidate and activation record

Generate with:

```powershell
python -m broker_runtime.prepare_ovs_sizing --source 'C:/Users/McKinley Slade/OneDrive/trading_ibkr' --output artifacts/ovs-sizing-candidate-v1
```

The output directory must be new. It contains local runtime configuration and must stay in ignored artifacts.

- Reviewed installed `order_staging.py` SHA-256: `cb6ae86474d17f5fe11404094facbcc6fe861c190cfafe8af9bb1c0bee52e779`.
- Prepared candidate SHA-256: `1305d5c55075ec4c28b3db74227b70e9f8de62a0553a9b4793e523721e1d1978`.
- Installed at 07:36 ET after idle checks and source/candidate hash verification. Only `OneDrive/trading_ibkr/order_staging.py` was replaced. Byte-verified backup: `OneDrive/trading_ibkr/.runtime_backups/ovs_fallback_20260909T073624/order_staging.py`. Restoring it rolls back future staging, not orders already submitted. The chain was not started or restarted.
- Normal valid stamped orders retain their current sizing. On the fallback path, 200 original shares yield 40 P2 shares instead of 30. The fallback P2 budget becomes $8,437.50 instead of $7,500 at the configured $750,000 sizing capital; the existing per-strategy cap remains. These are sizing budgets, not maximum-loss guarantees.
- This is a shared stager: its existing PA export is derived from the common staged frame. The owner explicitly approved applying the correction to both accounts before installation. No PA task or account configuration was changed.

PR #31 merged as `7633af144bddcf9ffeafb8971f6425e98b91f3bd`. The OVS-only commit was separately applied to the existing pinned runtime baseline as `7d7aafe8e97114abafbe9cfef0470953c98c8d92`, published under immutable tag `automation-runtime-2026-09-09.1`. Runtime v9 fast-forwarded to that commit at 07:39 ET; only the five reviewed OVS files changed. Its previous marker is preserved in `.local/runtime_promotions/ovs_20260909T073912/automation-runtime.json`. The existing runner's `-ValidateOnly` passed, and 64 model/related tests passed using that runtime's actual Python environment. Scheduler actions and cadence are unchanged. Source rollback can use a forward revert of the OVS commit and a matching new marker/tag rather than discarding runtime state.

The initial OVS-only tag was superseded before the next scheduled pipeline because site jobs also deploy from that pinned snapshot. Leaving its old site assets in place would regress the previously approved Execution controls. Commit `90c30dd698244ef46ff9dc43a74d6bfb0ff5b7e3`, tag `automation-runtime-2026-09-09.2`, adds only those already-deployed site assets, their matching tests, and the deployed workflow's explicit source-hash flag. At 07:43 ET runtime v9 fast-forwarded and passed `-ValidateOnly` again. The previous marker is retained in `.local/runtime_promotions/ovs_20260909T074347/automation-runtime.json`. Site assets, model, ledger/site generators and deployment workflow were compared against production source `7633af14` with no differences; 25 JavaScript suites and nine site/Execution contract tests passed using the promotion checkout.

The main-branch fallback controller pin is being aligned to final tag `automation-runtime-2026-09-09.2`. [Cloud build 34346385829](https://github.com/mslade50/New_Seasonals/actions/runs/34346385829) builds the merged model source from canonical R2 inputs. No local ledger is used as a production source. Publication and authenticated live checks remain pending at this checkpoint.

## Brief status

1. Answer-quality review: previously completed in its bounded scope; earlier rollout limitations still apply.
2. Execution simplification: deployed, including unified Close/Add/Re-add controls.
3. Modeled-versus-live reconciliation: OVS exit differences accepted; broker and pinned-runtime sizing corrections installed, cloud publication pending. OLV is the next bounded review after this step is accepted.
4. Strategy-discovery improvements: queued; not started by this change.
