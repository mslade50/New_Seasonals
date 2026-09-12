# Scan audit — September 7–11, 2026

The OLV strategy cannot currently be certified healthy end to end. Friday's
AM and PM scans sent email, but inventory and fresh OLV exit staging were
unverified. A green scan receipt is not evidence those optional paths worked.

## Findings

- September 11 inventory-close failed before side effects while bridging the
  reviewed starting inventory to current executions. A read of canonical R2
  on September 12 verified its file/status hash and complete=true, but Primary
  and OLV continuous coverage began September 11 at 04:00 UTC. The reviewed
  seed is September 10 at 01:51:40.677 UTC. This does not prove the intervening
  interval. The latest saved coverage ended September 11 at 21:14:38.043 UTC.
  Repair requires verified intervening execution coverage or a newly reconciled,
  owner-reviewed inventory seed. No attribution was guessed or rewritten.
- September 11 AM and PM logs explicitly report unavailable optional OLV
  notional capacity and unverified OLV exit metadata; prior exit staging was
  preserved. The owner's existing fail-open entry policy remains unchanged.
- Execution reporting failed its live-book freshness validation during the
  week; September 11 Task Scheduler returned 1 for execution, inventory-close
  and health. The PM pipeline returned 0, demonstrating why process success
  must be separated from each trading path's health.
- September 8 AM freshness failure is documented in
  docs/incidents/2026-09-08_scheduler_recovery.md. September 8 PM ran in GitHub
  fallback run 34301796391 and sent its email, despite no local scan success
  receipt. GitHub's masking damages the legacy JSON line, limiting machine
  reconstruction. September 9–11 have local AM/PM success evidence.
- Logs also show earnings refresh exceptions/empty responses and failed fill
  harvest attempts. Empty earnings results alone do not prove a data defect;
  distinguish ETFs/no coverage from expected company coverage in follow-up.
- The existing broker migration fixture tests reference pre-install source.
  Against today's installed broker files, one source-hash guard fails and eight
  legacy patch tests error because the old function no longer exists. These
  are separate from the passing scanner/pivot/email tests; they prevent claiming
  the legacy fixture suite certifies today's broker installation.

## Tickers

Sixteen symbols appeared in surviving logs/coverage. Fifteen retired listings
have dated primary-source evidence in config/live_scan_exclusions.json:
ANSS, AXIA, CSGS, CTLP, CTRA, CUK, FFIC, FONR, GDEN, LEG, RSX, SEE, SEM, TGNA, WBS.
Exclusions take effect prospectively September 12; historical books and prices
remain unchanged. AXIA's OTC migration is not a symbol substitution.

DX-Y-NYB remains unresolved and is retained. It is a dollar-index/provider-symbol
issue, not established corporate delisting. A bounded September 12 provider retry
returned no bars for either DX-Y-NYB or DX-Y.NYB; no retirement is inferred.
Some cached last-bar dates postdate official trading cessation. Investigate the
provider/cache provenance separately before trusting those post-delisting bars
in historical analysis; no historical data was altered in this change.

Raw diagnostic outputs are in the task worktree's ignored
artifacts/olv-scan-update/, including weekly-audit.json, ticker-retry.json and
downloaded GitHub log evidence. Historical email bodies are not available here;
the audit uses the logs and coverage behind those emails. New scans preserve
the full exception list and sizing record for subsequent weekly reviews.

## Requested changes and verification

Policy v3 (olv_close_pivot_40_v3_20260912) extends the 0.50 ATR discount from
2 < distance <= 3 through 2 < distance <= 4. At exactly 4 the discount is 0.50;
above 4 through 5 it remains 0.75; above 5 remains skip. Other pivots/defaults,
shares, sizing ladder and caps are unchanged.

OLV email cards display the actual signal number, prior count in 21 ticker
sessions, multiplier, risk budget, and risk represented by the final scanner
quantity. The label explicitly precedes the broker daily cap. Signal number is
not fill count: raw qualifying signals count even when unfilled or pivot-skipped.

Targeted scanner, OLV sizing/caps/fill windows, email, archive, exclusions and
supervisor checks: 177 passed. The pivot regression first failed against the old
3–4 ATR policy, then passed after the requested change.
Weekly Friday 22:00 ET review is configured as the Codex heartbeat
weekly-scan-reliability-and-ticker-review, using docs/weekly_scan_audit.md.

Deployment evidence must be recorded separately after CI and runtime promotion;
these test results alone do not establish deployment or a future broker fill.
