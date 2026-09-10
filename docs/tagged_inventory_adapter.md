# Reviewed tagged inventory adapter

`tagged_inventory.build_tagged_inventory(seed, fills, coverage, *, asof, algo_strategies, entry_metadata=None)` performs no I/O. It returns a `TaggedInventory` with `status` (`known` or `unknown`), reasons, `counts` and `notionals` keyed by `(symbol, strategy)`, exact open `tranches`, `exit_metadata_known`, and an explicit fallback description.

Unknown inventory permits the scanner's base sizing with a reported unavailable optional overlay. It must never be interpreted as flat inventory or used to create an inventory-derived exit. The adapter does not change trade controls. Existing working orders still need independent broker-side capacity protection.

A seed must have schema_version=1, account_key=primary, the exact broker_account, asof_utc, and a positions list. Its review object requires status=approved, reviewed_by, reviewed_at with timezone, and provenance identifying the reconciled broker evidence. A reviewed empty list can establish flat inventory. Neither missing rows nor theoretical portfolio targets can supply that review.

Each seed position identifies tranche_id (opaque, no pipe delimiter), account_key, account, con_id, symbol, sec_type=STK, currency=USD, strategy, ref_date, signed_qty, entry_price, and price_basis=raw. OLV exit staging additionally requires entry_order_ref copied exactly from original broker evidence. New tranches retain that original reference from their actual entry fill; it is never reconstructed from a theoretical row. For inventory-derived exits it also needs entry_date, raw frozen atr, exit_deadline_utc with timezone, and exit_protocol (MOO, MOC, TIME, or MANUAL_REVIEW). The adapter preserves these fields on partial exits. OLV exit checks must compare the frozen price/ATR to RAW bars, not the adjusted backtest cache.

Fills use the canonical harvest schema: exec_id, time_utc (or time), account/account_key, con_id, symbol, sec_type, currency, strategy, order_ref, qty, side (BOT/SLD or BUY/SELL), and price. The adapter uses only explicit algorithmic-strategy membership in Primary. A higher execution revision replaces its family before replay. Corrections older than the seed must be named in included_exec_ids as already incorporated, otherwise reconciliation is unknown and the seed needs review.

An order reference identifies SYMBOL|ACTION|STRATEGY|REF_DATE and optionally an explicit tranche id in the fifth field. Multiple matching tranches require an allocation mapping; the adapter does not guess FIFO. New entries can establish known quantity/notional from fills. Their ATR/deadline require trusted entry_metadata keyed by full order_ref before automated exit evaluation. One symbol mapped to multiple conIds is unknown, rather than conflating contracts.

Coverage must explicitly contain accounts.primary with complete=true, matching broker_account, continuous_from at/before the seed cutoff, and complete_through at/after the requested asof. Global truncation, unresolved legacy capped days, or merge errors make it unknown. A PA-only outage does not affect Primary. A fresh latest receipt or first observed fill is insufficient to establish continuous historical coverage. Current broker harvest receipts do not yet supply that entire continuity interval; operational bootstrap remains required. No production seed was created.

## Manual trades and reviewed assignments (2026-09-09)

Owner decision: untagged TWS trades remain discretionary unless explicitly
assigned. Sharing a symbol with an algorithm does not assign the fill to it.
The optional seed `execution_allocations` maps an exact execution id to a
`review` object (same approval/provenance fields as the seed) and an
`allocations` list of `{tranche_id, qty}`. Positive whole-share allocations
must sum to that execution's actual quantity. The fill supplies direction,
price, account and contract; the assignment cannot override them. Reductions
retain the entry/ATR/deadline; additions weight entry cost. Assignments cannot
reverse a tranche, reopen one already closed, or target another contract.
A corrected execution revision requires review of the replacement assignment.
These changes are source-tested; they are not yet promoted to runtime v9.

`scripts/reconcile_inventory_inputs.py` creates a review-only report from
captured `/book`, `/fills` and optional canonical parquet. It counts OCA exit
siblings once, excludes children of working entry parents, shows each exact
contract's net holding and residual, and distinguishes observed remaining
quantity from legacy total quantity. Exit claims and matching net quantities
are evidence for review, not an automatically approved seed. No network access,
inventory activation, order submission or canonical upload occurs.

The read-only snapshot producer now supplies remaining/filled quantities,
per-account order observation times, and completed current-session execution
query timestamps. That production change does not establish historical
continuity in the relay or canonical store. See `inventory_inputs_2026-09-09.md`.
