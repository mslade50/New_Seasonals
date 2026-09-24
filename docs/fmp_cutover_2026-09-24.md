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

## Final earnings rehearsal, September 24

Independent FMP reference: 1,552 requested names, zero failed requests, 68 empty
responses (retained as explicit unknown/noncorporate coverage). The initial
527-name additional universe history refresh is included in the rehearsal.
Candidate: 147,738 rows, UEC issuer correction applied; no second Alpha request.
NAVN's fresh FMP reference no longer contains the unsupported September 30 event.
ASTC is September 24 in both fresh feeds, still an unconfirmed expectation.

Six hypothetical policy flags change across two names:

| Name | Change today | Evidence/uncertainty |
| --- | --- | --- |
| BETA | OLV pre-earnings reduced sizing removed; OVS and LT Trend ST OS blackouts removed | FMP expects October 5; Alpha has no event. The issuer has no upcoming event announcement. Absence is not proof that FMP is wrong. |
| DAL | OLV reduced sizing begins; OVS and LT Trend ST OS blackouts begin | Alpha October 8 enters the ten-day window; FMP October 9 is just outside. Delta confirms an October 9 call, not an explicit separate publication time. |

This is a replay of policy flags, not orders or proof either name qualifies for a
strategy today. No strategy parameters or broker state change. Approval must
include these differences; the source does not silently reinterpret estimates
as confirmed events.

The initial trial still uses FMP for reported-date confirmation, newly covered
history and emergency fallback. Those are remaining expiry blockers, along with
any separately scheduled symbol-master/research enrichment dependencies. Do not
call the subscription migration complete after this first provider cutover.

Validation: the final 190-test source/publisher/scheduler/sizing suite passed
in both the development Python and the existing production Python environment. Macro's live no-upload capture passed all 29 series,
including current and next official CPI/PPI/NFP schedule checks. Forecast-only
records retain their full original observation when filled, and partial remote
publication is explicitly recorded before readback. The pinned runtime PDF
parser dependency is installed; runtime source/provider activation is pending.

Automatic approval review rejected the direct `origin main` push on September 24,
requiring explicit owner approval of that exact production-branch action. No push,
canonical publication, runtime-pin promotion or monitor change has occurred.
