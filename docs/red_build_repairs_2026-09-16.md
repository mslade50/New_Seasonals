# Red-build repairs — September 16, 2026

These repairs are separate from the owner-approved whole-ticker capacity
activation, which must wait until 10:00 AM America/New_York today.

## Prepared source corrections

- Discretionary Focus accepts periodic financial reports on 20-F/40-F and
  amendments, as well as 10-Q/10-K. It still excludes generic 6-K/8-K current
  reports and preserves the 200-day freshness limit and SEC-host requirement.
  The live FMP lookup now resolves BLSH's March 10, 2026 20-F. This fixes the
  unsupported form; it does not certify future freshness or change research
  selection gates. SEC evidence:
  https://www.sec.gov/Archives/edgar/data/1872195/000143774926007417/0001437749-26-007417-index.htm
- Research collection now permits 250 records and four pages per SSRN source,
  with 500 total records and the existing 12-request ceiling. Crossref cursor
  queries retain their page size, and a truncated last page cannot claim
  completeness. A read-only live replay collected 152 in-window records in
  three requests and exhausted both source windows, including the backlog
  beginning September 14. Production cursors were not advanced by the probe.
- RadarPack computes a next-year projection in memory from cached prices no
  later than its as-of date, using the existing seasonal rank formula. It
  never writes those preliminary ranks into the canonical scanner cache.
  Export metadata identifies the projected years and training cutoff. Tickers
  without all 90 requested dates are explicitly missing. The real-data probe
  produced 51,570 rows for 573 complete tickers through January 26, 2027;
  345 universe tickers remain missing. The canonical rank file's hash stayed
  unchanged. Publishing also preserves dirty pack files and unrelated staged
  work instead of discarding or committing them.
- Health checking recognizes `.parquet.status.json` producer receipts as
  expected files; partial-write files still raise warnings.

## Other findings and operational changes

- Repaired `sim-health-check-weekly` to invoke the full quoted command through
  `cmd.exe`. Its old executable stopped at the space in the Windows username.
  Original task XML is retained under `artifacts/red-build-repair/`. The task
  was not run, so no health message was sent.
- Nightly PGA off-week guard is already on sims_process main, commit 3217d67,
  verified with 41 tests and read-only schedule checks.
- ReportCard and weekly-repo-check are blocked by Claude's monthly spending
  limit. Migration to Codex was offered; neither runner has been changed.
- The indicator fallback and execution report recovered after their cited
  failures. DeskReplies' flagged traceback predates its code repair and later
  successful scheduler runs. Do not rerun the live DeskReplies worker merely
  to clear an old log match.
- GolfShotRoundPublish's manifest contains PGA R2026060 rounds 1–4 published
  August 31, newer than the alert's R2026028 reference. Recent no-op runs also
  reject incomplete historical SG data. The quality gate remains intact.
- The red-build-watch job definition was not found in the available local
  automation directories; its stale-result rules have not been modified.

## Verification and rollout boundary

Focused tests cover foreign filings, source pagination and truncation,
cross-year export and training cutoff, and expected health sidecars. Replaying
the old implementation reproduces the foreign-form, year-boundary, and changing
cursor-page-size failures. Real-data probes write only under artifacts and do
not publish packs, change research state, send reports, or run trading scans.

The root checkout owns RadarPack and strategy research. The v9 pinned runtime
owns Discretionary Focus and health; the same research-only repairs must be
included when promoting the pinned runtime and guarded fallback at 10 AM.
