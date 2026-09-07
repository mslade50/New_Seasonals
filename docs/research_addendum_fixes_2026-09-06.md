# Research audit addendum closure

This closes bounded defects from the audit's unnumbered limitations, extending
the R1–R9 repairs rather than treating the numbered register as the whole scope.

## Changes and verification

- Planning and report health share dated endpoint readiness. Every required FMP
  endpoint must be current through the report cutoff; missing, stale or future
  snapshots cannot count as ready. Deep readiness also requires a current SEC
  package. Decision-ready review freshness was already fixed with R3.
- Dry-run refresh previews use the actual sector/size-balanced selector. The
  orchestrator passes that explicit proposed batch to the refresher, preventing
  the execution from independently choosing a different batch.
- Manifest source records are captured before building. Named current inputs,
  the symbol master and selected immutable FMP/SEC parts must be unchanged after
  report generation and validation. Files changing during hashing are rejected.
  A detected change prevents completed-manifest/transition publication. This is
  a before/after stability check, not a transactionally isolated filesystem
  snapshot; it does not claim to capture every transitive runtime dependency.
- Posts email and Context Slack use shared strict, serialized delivery claims.
  `SENDING` is persisted before transport; `SENT` is persisted after confirmed
  acceptance. Uncertain attempts block ordinary retries and changed-content
  bypasses. A proven connection/authentication failure before submission, or an
  explicit Slack rejection, is recorded `NOT_SENT` and may be retried. SMTP QUIT
  failure cannot reverse DATA acceptance. Partial recipient refusal remains
  ambiguous until provider reconciliation.
- Context no longer retries timeouts or server failures automatically. A retry
  after successful delivery can finish local novelty/journal updates without
  reposting or incrementing their counts twice. Corrupt journal/baseline bytes
  are preserved and reported. `--no-state` does not bypass delivery deduplication.

Delivery receipts are local-primary state at `data/posts_email_receipts.jsonl`
and `data/context_delivery_receipts.jsonl`. Neither receipt stores a password,
token or webhook URL. Existing content, recipient configuration and editorial
gates remain unchanged. This work did not send any real message.

Offline probes against original source reproduced three Slack POST attempts
after an uncertain acceptance and two Posts emails after DATA acceptance followed
by a QUIT disconnect. The repaired paths issue one fake transport call. A broad
relevant suite passed **672 tests**, followed by **13 addendum tests** including
full orchestrator rejection on changed input and actual balanced preview.
Existing context tests now require corrupt-baseline preservation. Evidence is
under `artifacts/research-fixes/research-addendum-*`.

## Remaining source work versus owner inputs

The [family-fit MVP](strategy_family_fit_mvp_2026-09-06.md) is useful offline
infrastructure. It does not fulfill the requested daily worthwhile-only research
email loop by itself. These are missing code pieces, not missing credentials:

| Source work | Bounded next implementation |
| --- | --- |
| X capture | Read-only collector with allowlists, pagination, window/cursor receipts, explicit request limits, and fake-provider tests. |
| SSRN intake | Publication/version capture and normalization into a reviewed research contract; V1 currently accepts X only. |
| Native exact duplicate catalog | Translate supported scanner rules faithfully into normalized discovery structures; report unsupported rules rather than invent a fingerprint. Family overlap is already available. |
| Empirical validation | Execute supported hypotheses against frozen data and record actual test results, costs, out-of-sample checks, bad-period overlap and marginal capital use against active algorithms. Verified artifact hashes alone are insufficient. |
| Useful-finding decision and write-up | Require that empirical evidence and incremental-fit evidence pass; write strategy, our validation, why it helps the algorithm book, and practical implementation. Missing evidence cannot become an email-ready item. |
| Delivery and scheduling | Build an outbox only for newly useful validated findings, deduplicate with durable receipts, and emit nothing on no-finding days. Source-collection failures require an explicit operational status rather than being called a quiet research day. Existing scheduling/email infrastructure can host this once implemented. |

Owner-supplied X credentials/access tier, a spending ceiling, and desired
accounts/lists are configuration inputs for that collector. SSRN access may be
public or restricted per source; no credential need should be invented before
the chosen source is known. None of those owner inputs is needed to implement
and test the interfaces with offline fixtures. No new paid service is required
for the present source repairs.

Other architectural limitations from the original audit remain explicit:
research journals/receipts use a local single-writer contract, not distributed
R2 compare-and-swap; provider request budgets/cost receipts and cross-workflow
refresh rotation are not yet shared; presentation statistics remain manually
maintained and need a dated generated export. Historical experimental sleeves
remain research references unless operating evidence establishes otherwise.
These larger changes should not be described as completed defect fixes or as
blocked solely on owner credentials.
