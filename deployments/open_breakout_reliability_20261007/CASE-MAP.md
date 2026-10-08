The aggregate includes the unchanged existing strategy suite plus price,
coordination, native-boundary and order-reliability regressions. Native-shaped
fixtures are inert; they are not actual broker qualification. No cases are hidden
or skipped to obtain a passing aggregate.

| Requirement | Test evidence |
|---|---|
| Exact native acceptance; SDK status cannot certify it | real_SDK_decoder_hook_excludes_synthetic_error_status; duplicate_reordered_status_requires_echo_and_cancel_never_rearms |
| Callback before acceptance, failed journal commit/redelivery | actual_sdk_execution_first_binds_owner_and_redelivers_failed_commit; lifecycle_and_fill_journal_transaction_survive_crash_gap |
| Durable intents and no send/cancel retry after crash | crash_restart_never_authorizes_send_or_cancel_retransmit; immediate_fill_has_intents_before_send_and_no_duplicate_protector |
| Full owner identity/history/freshness proof | wrong_native_identity_is_rejected; partial_response_cycle_cannot_clear_account_block; snapshot_omission_never_proves_flat_even_after_cancel |
| Independent market waits with bounded risk | per_market_wait_releases_lock_but_unknown_exposure_blocks_NQ |
| Oct 7 consequence classification and first-cause persistence | Oct7_pending_cancel_and_OCA_sequence_preserves_first_cause; actual_adapter_quarantines_only_exact_owned_cancel_consequences |
| Oct 6 price pause and exact 600-second expiry | Oct6_price_gap_exact_grace_boundary_retains_protection_and_risk; reliability_and_price_pause_recover_only_price_latch |
| Held-position current protection and revisions | native_protector_disappearance_never_clears_account_block; partial_protector_revision_requires_current_echo; unexpected_current_native_echo_revokes_prior_acceptance |
| Ordinary and opposed-entry flat cleanup | normal_stop_to_flat_uses_durable_exit_cancel_and_deadline; opposed_OCA_net_zero_cleans_only_exact_owned_exits_after_full_proof |
| Late fills after prior exits; explicit manual limits | same_entry_late_partial_after_native_closed_exits_gets_new_protection; native_partial_exit_OCA_reduction_requires_manual_revision_proof; late_known_exit_reversal_is_journaled_once_and_loud_manual |
| Automatic flatten remains unqualified | rejected_protection_never_uses_unqualified_emergency_flatten |
| Independent rollout/rollback and interrupted installs | test_deployment.py; test_reliability_deployment.py, including recovery drift, exact transition and reparse guards |
| Reconnect stable acknowledgments, flaps, in-flight requests | execution-bridge/source/tests/test_execution_connection.py (19 inert cases) |

Local installed-payload evidence uses disposable mapped roots and verifies the
28 production prerequisites before and after. The two Legend source/root-pointer
fixtures map only their fixed startup path to a disposable root; evidence records
both mapped hashes and reviewed production hashes. The other 18 targets match
their rendered reviewed bytes exactly. Portable CI uses exact public
baseline modules, including daily_execution_report.parse_ref, and reviewed payload
overlays. It does not claim local installer certification. Bridge installer tests
are local-only; helper/lifecycle tests are portable and guarded. Evidence summaries
record test counts, zero broker/network attempts, timings and qualified-state flags.
