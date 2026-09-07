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

## Subsequent source-work implementation

The missing source-to-email layers listed in this addendum were implemented in
the follow-on [strategy research pipeline](strategy_research_pipeline.md):
official read-only X and Crossref/SSRN collection, two-phase cursors, native and
family catalogs, preregistered reproducible empirical work, active-algorithm
portfolio-fit gates, worthwhile-only writeups, at-most-once email delivery, and
an inert-until-registered daily scheduler.

Owner-supplied X credentials/access tier, a spending ceiling, and desired
accounts/lists remain configuration inputs. The public SSRN path requires no
paid service. X stays disabled until those inputs are supplied; this does not
block SSRN research.

Other architectural limitations from the original audit remain explicit:
research journals/receipts use a local single-writer contract, not distributed
R2 compare-and-swap; provider request budgets/cost receipts and cross-workflow
refresh rotation are not yet shared; presentation statistics remain manually
maintained and need a dated generated export. Historical experimental sleeves
remain research references unless operating evidence establishes otherwise.
These larger changes should not be described as completed defect fixes or as
blocked solely on owner credentials.
