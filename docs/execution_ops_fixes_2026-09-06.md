# Execution and operations fixes — 2026-09-06

Source-only repairs for E1–E7 and B6/B7/B8/B9/B10/B12 are prepared in this worktree. No broker connection, order, task change, email, cloud write, deployment, deletion, or live OneDrive edit was performed.

- E1: recovery sizes protection from exact remaining inventory, deducting confirmed fills and working close commitments. All owner connections and identities are preflighted before resize/cancel mutations.
- E2/E5: flatten binds account plus conId at initial selection and every reread, passes the explicit account to the guard, and reports working/partial outcomes as unknown.
- E3: Event exit staging retains the obligation and stable entry identity. Attributed executions clear completed exits. Missed unsubmitted auctions remain eligible for the next auction; durable submitted/unknown claims require reconciliation.
- E4: Trend actual holdings, expected post-order holdings, pending orders, and model targets are separate. A skipped rebalance-band delta never changes actual shares; expected-zero positions remain visible to reconciliation. Legacy inventory cannot be bootstrapped from theoretical state.
- E6/B10: report exit attribution uses account/conId (explicit expiry for legacy derivative data), excludes pending entry children, refuses stale/incomplete Primary books, and honors no-send on errors.
- E7/B12: streaming deadlines also cover an exited parent whose child retains stdout; owned process-tree termination and child reaping are implemented. Quote/book helper timeouts kill and reap.
- B6/B7/B8: canonical read failures stop publication; immutable generations and ETag compare-and-swap protect prior history. Corrections supersede one account/execution family. Retention uses calendar days, explicit broker completeness, and scheduled assert-no-gap. Book snapshot emits explicit successful execution-request attestation per account.
- B9: entry preflight and row failures return nonzero; the batch wrapper preserves any stage/entry/summary failure.
- Event/Trend Sheets writes use the data agent's atomic replacement helper. Current producer coverage receipts annotate successful component receipts as degraded without duplicating completed mutations or blocking unrelated healthy work.

The sanitized external repair package is `broker_runtime/`; it creates a new candidate only after all source hashes match. Candidate outputs under ignored artifacts contain original private configuration and must not be committed. Root must integrate the broker completeness contract and frontend exact identity changes together with these sources. Native broker behavior still requires paper/TWS verification and explicit operational promotion. Legacy tagged inventory provenance is a separate bootstrap prerequisite.

Verification: targeted suite covers fills, reports, Event/Trend reconciliation, supervisor receipts, portable lifecycle helpers, source-hash candidate compilation, and AST-only external broker failure fixtures. All checks run with fake brokers/clients and local artifacts; no full external script imports or live actions.
