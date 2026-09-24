# September 24 production cutover trial

## Prepared changes (not yet activated)

Economic releases use `scripts/refresh_macro_releases.py`: official 29-series
collection, source-specific freshness checks, unchanged populated history,
conditional R2 publication and digest-verified readback before local replacement.
Both the local macro job and dispatch-only backup use the same entry point.
New observations contain no consensus or surprises. Other historical FMP series
remain archived; this is not a claim of ongoing coverage for its entire catalog.
The date-only macro event calendar already uses official BLS/Fed sources and is
not replaced by the macro-history writer.

Today's independent comparison at 11:00 UTC matched 28 current reported values
and all eight upcoming release groups. FMP moved its GDP Q2 value from August 26
to September 30 between captures; the official BEA release retains August 26.
The earlier weekly-hours missing-unit diagnostic is resolved by its named Hours
measure; units for count series still require an explicit scale.

Artifact-only merge against a freshly downloaded canonical R2 history preserved
all 38,268 populated records. Source evidence and replay are under
`artifacts/fmp_cutover/20260924/` and
`artifacts/macro_shadow/20260924T110043101920Z/`.

Earnings trial preparation keeps the Alpha primary/FMP confirmation-and-fallback
architecture. This is a transitional cutover, not yet FMP independence. Daily
Alpha requests are coordinated by an atomic R2 snapshot claim shared by the
observer and producer. A failed attempt cannot trigger another Alpha request
that day. Today's authenticated capture is reused without another request.
The issuer-announced UEC September 29 date supersedes Alpha's stale September 23
expectation through an exact-period, expiring rule. Expected dates remain
expected until reported-date evidence arrives.

## Activation and rollback

Earnings activation requires immediate owner approval of the final policy replay,
under AGENTS.md's financially consequential change rule. Preserve the old
canonical object and runtime marker. A rollback restores the prior object with a
conditional write and the prior producer configuration/runtime, then verifies
both consumers. A config flip alone is insufficient for the earnings universe
transition. No broker order, strategy parameter, Focus job or grade collector is
part of the cutover.

Production status and exact release identities will be recorded here only after
actual promotion, producer success and remote digest verification.
