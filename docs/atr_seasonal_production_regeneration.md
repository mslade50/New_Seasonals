# Corrected ATR-seasonal production regeneration

Status: production migration completed and deployed on 2026-09-07.

The prior audit corrected the annual outcome cutoff in
`build_atr_seasonal_ranks.py`, but production R2 still held history generated
under the old formula. The 2026-09-07 cloud-only private-site run rebuilt and
promoted 5,660,705 rank rows for the 1,025 configured active tickers, archived
the exact predecessor, rebuilt the 4,711-row canonical trade ledger, and
deployed the bound bundle. The canonical rank SHA-256 is
`ceeadf4bed239a86c92892794a5444c4334dcdb517f0f7a12f6ee86a535eed04`;
the ledger SHA-256 is
`ab83b6e9c9335573f25d6fd021cf290491c898df47c1753d9b7cfdb6c2573144`.
Production evidence is retained in GitHub Actions run `34124625338` and the
R2 predecessor archive under `migrations/atr_seasonal_ranks/34124625338-1/`.

On every `deploy_site.yml` run, the isolated generator now:

1. downloads the canonical master-price and ATR-seasonal objects from R2;
2. validates the rank schema, version, values, duplicate keys, configured
   ticker coverage, and current-year coverage;
3. performs a full 2001-through-current-year deterministic rebuild only when
   the object is old or incomplete;
4. archives the exact predecessor parquet and a digest-bound migration receipt
   under immutable R2 keys before replacing the canonical object;
5. conditionally replaces `atr_seasonal_ranks.parquet` only if its R2 ETag
   still matches the object downloaded at the start of the run;
6. refreshes generator provenance, rebuilds the full trade ledger from the
   corrected ranks, freezes the generated bundle, and assembles/deploys from
   that exact R2-bound bundle.

Missing or short price history for any ticker in the current configured
strategy universe blocks the regeneration. Tickers that only exist in the
predecessor are retired during the rebuild; the predecessor is not a membership
authority. Concurrent R2 modification blocks replacement. A healthy
`annual-outcome-cutoff-v2` object produces a `CURRENT` receipt and performs no
R2 write. No local `data/` or `dist/` artifact is a production input.

Rollback restores the archived `predecessor.parquet` from that run's
`migrations/atr_seasonal_ranks/<run-id>/` prefix to the canonical
`atr_seasonal_ranks.parquet` key, then redeploys the prior application commit.

The migration changes historical ATR-seasonal filters and can therefore
change the reconstructed trade ledger, research statistics, and private-site
portfolio/seasonal data. It does not itself stage or place an order. Because
downstream research and live decisions use those datasets, the workflow should
be dispatched only after the exact commit is merged and the production change
is approved.
