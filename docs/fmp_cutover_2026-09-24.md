# September 24 production cutover trial

## Activated trial: official macro live; earnings selected FMP fallback

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

The owner explicitly approved the main push, runtime promotion and production trial
on September 24, including the final earnings policy replay and FMP fallback. Preserve the old
canonical object and runtime marker. A rollback restores the prior object with a
conditional write and the prior producer configuration/runtime, then verifies
both consumers. A config flip alone is insufficient for the earnings universe
transition. No broker order, strategy parameter, Focus job or grade collector is
part of the cutover.

Verified production status and exact release identities are recorded below.

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
parser dependency is installed. The same 190 tests also passed inside the promoted
production runtime before the first live producer runs.

The earlier automatic approval rejection was resolved by the owner's explicit
approval. Source, runtime promotion, canonical publication and monitor updates
were then completed and verified as described below.


## Observed production outcome, September 24 at 11:07 ET

Runtime and GitHub fallback tag: `automation-runtime-2026-09-24.alpha-official-macro`,
SHA `36cbf9c0234e26743c10b91af9a57647710f83c6`. Main provider releases are
`56591c4e` and `096daac9`; main `d37a9142` aligns the guarded fallback tag.
The old marker is preserved at `artifacts/fmp_cutover/20260924/activation-marker-before.json`.
The initial runs held the shared automation lock and ran only the two producers.
No scheduled component success was forged; the ordinary postclose run remains due.
Focus remains disabled and no analyst-grade command was restored.

**Economic releases: active and verified.** The 10:55 ET official producer
published 43,072 records covering 29 implemented series with zero FMP requests.
All 38,268 previously populated records remain unchanged. Today's initial and
continuing claims are official DOL observations. Canonical SHA-256:
`a76edba576f33756a398af4f3449505ce29d04a18ffb27d272ecd745d9fe5b22`.
The 11:01 ET independent comparison matched all 29 reported values and seven of
eight upcoming groups; FMP omitted the September 30 GDP release still announced
by BEA. This supersedes the earlier 28/29 morning comparison. The optional BLS
RSS feed returned 403; the required API and schedule checks passed.
Evidence: `artifacts/macro_shadow/20260924T150126329551Z/` and runtime
`artifacts/macro_provider/20260924T145549751027Z/`.

**Earnings: trial installed, Alpha cutover did not pass.** The producer selected
`fmp_fallback`, status `degraded`, publishing 147,928 rows with a verified remote
readback. Its exact failure was `Unconfirmed disappearance of today's earnings: RZLT`.
FMP's independently captured 07:08 ET reference expected September 23; the
10:58 ET production bootstrap moved RZLT to September 24. Alpha omitted the
corresponding event and no reported result confirmed it. The issuer listing did
not establish a release date, so its absence was not treated as disproof of FMP.
The guard remained intact. The fallback made zero policy-flag changes against
its refreshed production baseline; BETA/DAL therefore retain FMP treatment,
rather than adopting the approved Alpha rehearsal differences.

The runtime receipt is under `artifacts/earnings_provider/20260924T145606444430Z/`.
Independent post-publication verification is under MAIN
`artifacts/fmp_cutover/20260924/verified-publication/`: both canonical hashes match
producer and local copies, the actual earnings loader passes embedded freshness
and all-universe checks, UEC is September 29, and stale NAVN September 30 is absent.
Today's 06:32 authenticated Alpha snapshot was reused with origin
`existing_authenticated_observer`; no second live Alpha request was made.
RZLT evidence is under `artifacts/fmp_cutover/20260924/rzlt-adjudication/`.

## What establishes success

The existing heartbeat `compare-alpha-vantage-earnings-with-fmp` is active as
**Monitor earnings and economic cutover**, weekdays at 06:30 and 18:30 ET.
It now uses the promoted runtime observer, not the obsolete observer checkout.
The morning phase validates an independent FMP reference before comparing the
next ten NYSE trading days. The evening phase verifies scheduled producer
receipts, remote publication hashes, actual consumer freshness and source
coverage, and independently compares official economic observations against FMP.
Only the shared daily Alpha snapshot prefix may be written by the observer;
canonical data remains read-only. First meaningful failures or changes alert
in the existing task; unchanged discrepancies stay quiet. The monitor depends
on the local Codex app being able to execute and is not a minute-of-release SLA.
Its November 11 assessment/pause endpoint remains unchanged.

A successful earnings run must actually select `alpha`, report `status=ok`,
finish `remote_verified`, retain required history and coverage, and pass consumer
checks. Exit code zero alone is insufficient: today's explicitly degraded FMP
fallback is the concrete example. The first ordinary scheduled run has not yet
been observed. RZLT reconciliation and removal of FMP confirmation, bootstrap,
and fallback dependencies remain expiry blockers; the cutover trial is not a
claim of FMP independence. No financial safeguard was relaxed to make it pass.
