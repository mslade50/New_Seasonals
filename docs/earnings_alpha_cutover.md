# Alpha earnings cutover — prepared September 22, 2026

**September 24 trial activated; first run selected FMP fallback.** Production
runtime `36cbf9c0234e26743c10b91af9a57647710f83c6` requests Alpha but refused
an unconfirmed disappearance of RZLT's expected September 24 release. It published
a verified, explicitly degraded FMP fallback. Official macro releases are active.
See the [September 24 trial](fmp_cutover_2026-09-24.md) for exact evidence,
monitoring and remaining expiry blockers. September 22 findings below are dated
historical evidence, not a statement of current discrepancies.

## Prepared behavior

- `scripts/refresh_earnings_calendar.py` is the shared local/Actions entry point.
  The stable `earnings_and_grades` receipt ID is retained for dependent jobs;
  grade collection and its required output are removed from prepared source.
  The legacy PowerShell wrapper now uses the same entry point.
- Opt-in Alpha uses one bulk upcoming calendar fetch, normalizes share-class
  aliases, and replaces forward expectations while preserving historical rows
  and their existing financial values. Expected dates are explicitly labeled.
- Initial overflow history is refreshed from FMP rather than trusting the
  stale sidecar. New uncovered symbols get FMP history on later runs. Recent
  reported values/dates still come from bounded FMP confirmation requests.
  Exact-date actuals, including zero EPS, can confirm an elapsed expectation.
  Unconfirmed elapsed events, disappearance on release day, or disappearance
  of a near-term Alpha fiscal period stop Alpha publication and use the explicit
  FMP fallback. Legacy historical dates without actuals remain labeled unverified.
- Alpha failures use a full FMP fallback; fallback failure leaves the canonical
  object untouched. Input, schema, duplicate-period, coverage and freshness checks
  precede publication. The single R2 object is updated conditionally against its
  baseline ETag and read back for SHA256 verification before local replacement.
- The authoritative all-universe calendar suppresses the legacy overflow union
  in the scanner, backtester and discretionary Focus consumer. A fresh file mtime
  cannot disguise stale embedded producer timestamps. Live loading requires at
  least the previous NYSE session; historical replay permits older snapshots.
- `--no-upload` writes only new artifact directories. Provider overrides and
  historical replay cannot publish. The independent observer rejects an Alpha
  production calendar passed as its supposed FMP control.

The pinned runner reads the existing config-root `.env`, which already holds
`ALPHA_VANTAGE_API_KEY`. The same saved key was provisioned as an encrypted Actions
secret on `mslade50/New_Seasonals` after verifying signed-in owner/admin identity.
No secret value was displayed or committed.

## Dated validation

Artifacts: `artifacts/earnings_cutover/20260922/`.

- Reused the morning authenticated Alpha snapshot; no second Alpha request.
- Refreshed 527 current extra-universe names plus SA and UEC: 529 FMP requests,
  no hard failures, five empty responses. The present tracked universe is 1,552
  names, larger than the 1,541-name morning observation.
- Fresh comparison window: September 22 through October 6, inclusive (10 NYSE
  trading days). Raw feeds: 28/31 exact FMP events, two Alpha-only events, two
  FMP-only events and one date disagreement. An absent event in both feeds is
  unknown coverage, not agreement.
- `replay-04` builds 147,734 rows locally. It retains legacy historical evidence
  while refreshing the extra-universe baseline and excluding obsolete forward
  rows from refreshed FMP tickers. The candidate applies the source-backed SA
  exclusion described below.
- Policy replay identifies eight changed blackout/sizing flags across three
  tickers: ASTC, BETA and NAVN. It tests the configured policy windows on the
  full universe, not actual order generation or whether each ticker qualifies
  for each strategy today. No scanner, Sheets write or broker action was run.
- Unit/integration coverage includes historical preservation, period revisions,
  release-day disappearance, confirmations, stale metadata, overflow precedence,
  sizing changes, truncated responses, Alpha failure/FMP fallback, conditional
  publication, readback mismatch and observer-control independence. Publication
  tests use fake R2; no live deployment test has been performed.
- Final focused suite: **212 passed**, with one pre-existing dependency
  deprecation warning. `git diff --check` passed. Workspace hygiene identified
  an unrelated new `docs/briefs/2026-09-22/legend_ungated_live_override.md` from
  concurrent work; it and the pre-existing data/research changes were untouched.

## Event adjudication and activation gate

| Symbol | Finding | Disposition |
| --- | --- | --- |
| SA | Alpha repeats the June 30 period on September 23; [issuer 6-K](https://www.sec.gov/Archives/edgar/data/1231346/000106299326004389/form6k.htm) dates that quarter's release August 13. | Exact ticker/date/period exclusion, expiring September 30. Other SA events are untouched. |
| PENG | Both fresh FMP and Alpha now say October 6, matching [issuer announcement](https://ir.penguinsolutions.com/news/news-details/2026/Penguin-Solutions-Sets-Conference-Call-for-Fourth-Quarter-and-Fiscal-2026-Results/default.aspx). | Prior apparent omission was the stale comparison cache. |
| ASTC | Alpha September 24; fresh FMP has no upcoming event. [Issuer news](https://www.astrotechcorp.com/about/news) did not establish that date. | Unresolved; changes blackout and pre-earnings sizing. |
| BETA | Fresh FMP October 5; Alpha absent. [Issuer results](https://investors.beta.team/news-events/press-releases/detail/116/beta-technologies-inc-announces-second-quarter-2026-results) confirm August 12 Q2 results, not an October 5 date. | Unresolved forward estimate; changes blackout and sizing. Absence of an announcement is not proof of no event. |
| NAVN | Fresh FMP has both September 9 actuals and a September 30 estimate; Alpha has no September 30 event. [Issuer results](https://investors.navan.com/news-releases/news-release-details/navan-announces-strong-second-quarter-fiscal-year-2027-results) confirm September 9 for July 31 quarter. | September 30 appears to be a stale FMP estimate, but its period is not supplied. Changes a sizing flag; explicitly adjudicate before activation. |
| UEC | Alpha September 23 versus FMP September 24. [Issuer events](https://www.uraniumenergy.com/invest/events-and-webcasts/) did not confirm either 2026 year-end date. | Exact date unresolved; both feeds produce the same tested policy flags today. |

Activation remains gated on disposition of these differences, not a headline
percentage. FMP agreement alone is not ground truth. Repeat the current-date
replay immediately before promotion; this evidence is not evergreen.

## Remaining activation procedure

1. Resolve the remaining dates with issuer evidence, or explicitly agree an
   exception policy and test its effects. Do not silently convert estimates to
   confirmed dates. Rerun the full near-term blackout and sizing comparison.
2. Prepare an exact tested release containing these source changes, then request
   immediate approval for the financially consequential production change: Alpha
   primary with FMP confirmation/fallback, all-universe canonical cache, and
   grade collection retired. The exposure is changed earnings-based trade
   exclusions/sizing, not a new subscription. Preserve the previous runtime pin,
   canonical bytes/ETag and current generation before activation.
3. Promote the existing pinned runtime and matching GitHub fallback ref together;
   do not create worktrees/branches without a specific request. Set the versioned
   provider config to `alpha`. Ensure no old producer is in flight or able to
   overwrite the new object, and use the existing receipt/lease-controlled job.
4. Verify a real producer receipt, remote readback hash, both consumers' source
   precedence/freshness and next scheduled-run behavior. Do not claim that fake-R2
   tests are production verification. No website deployment is part of this task.
5. Update the observer's existing checkout/prompt to this tested source before
   activation. Once production is Alpha, obtain a genuinely independent FMP
   reference for daily comparison; never compare Alpha to its own canonical file.
   Keep FMP during this monitored transition.

Reference collection and replay (use a new output directory for each run):

```powershell
python scripts/refresh_earnings_calendar.py --reference-only --no-upload --output-dir artifacts/earnings_reference/NEW_RUN
python scripts/compare_earnings_shadow.py --fmp-baseline-dir artifacts/earnings_reference/NEW_RUN --alpha-csv SAVED_ALPHA_RAW --as-of YYYY-MM-DD
```

The reference command makes FMP requests but cannot publish. Reuse the day's Alpha
snapshot when already collected. A live observer call without `--alpha-csv` makes
one Alpha request, so coordinate it with the producer/heartbeat rather than
fetching twice unnecessarily.

Rollback requires restoring the archived pre-cutover canonical object with a
conditional write, restoring the previous runtime/fallback/config, and verifying
fresh FMP generation. A config flip alone is insufficient: the legacy builder's
coverage gate can reject shrinking an all-universe calendar back to CSV_UNIVERSE.
If readback fails after a successful remote write, investigate the published
generation first; do not blindly republish or assume nothing changed.

FMP cancellation is a separate migration: see [remaining dependencies](fmp_retirement_inventory.md).
