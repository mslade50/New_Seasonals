# Macro seasonal rank coverage repair

The Denali Macro table displayed 54 priced symbols, but only 18 had seasonal
ranks. The canonical rank rebuild covers the 1,025 configured strategy symbols;
36 displayed Macro-only symbols were outside that universe. The shared build
checked file readability rather than per-row seasonal coverage.

## Change

The Macro exporter now computes absent Macro-only ranks from the full frozen
master-price history using the existing annual seasonal rank functions, annual
outcome cutoff, NYSE date mapping and one-decimal rounding. These supplemental
values exist only in the site's Macro JSON. Canonical strategy rank values and
the R2 rank object are read-only. Missing strategy ranks cannot be filled by
this calculation.

Each row carries its actual seasonal rank date and source. Both shared and
private site validators reject missing, stale, nonfinite or out-of-range rank
values in any of the six windows. The shared cloud build requires its canonical
rank download. The existing omission of caret indices without price history is
unchanged; missing rank values never cause rows to disappear.

## Verification and rollout

- Regression reproduced before the fix: the VIX 5-day rank was missing despite
  adequate history; the canonical calculation returned 32.7 in the fixture.
- 80 focused tests pass, including canonical parity, input immutability,
  current-year outcome exclusion, short-history failure, holiday/year rollover,
  shared-site redaction and frontend checks, and both final coverage gates.
- Read-only audit of canonical R2 inputs on September 22 produced complete
  ranks for all 54 displayed rows: 18 canonical rows unchanged, 36 supplemented.
  The rank object's SHA-256 remains
  `ceeadf4bed239a86c92892794a5444c4334dcdb517f0f7a12f6ee86a535eed04`.
- Production must build through `deploy_shared_seasonals.yml` on GitHub Actions
  using fresh R2 downloads, targeting the existing `denali-seasonality` project.
  Diagnostic artifacts under `artifacts/macro-rank-repair-20260922/` are never
  deployment inputs. No strategy configuration, runtime pin or risk signal is
  part of this repair.
- Rollback is a forward revert of this scoped source change followed by the
  same shared-site cloud workflow. No canonical data restoration is required.
