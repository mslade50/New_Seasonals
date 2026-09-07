# Algorithm-family research fit: source implementation

The offline discovery foundation from audited overnight commit `f3e31dc2` is
imported into the repair branch. Its contracts, journal, pipeline, renderer,
examples and 160 tests are preserved. The CLI adds optional family-fit inputs;
the native discovery report and lifecycle contracts are unchanged.

This document describes the family-fit boundary delivered in the audit repair.
The subsequent active collector, empirical agent workflow, deterministic email
gate, and scheduler are documented in
[strategy_research_pipeline.md](strategy_research_pipeline.md).

## What runs

`scripts/build_algorithm_family_catalog.py` exports the source book, six Event
definitions, monthly Trend, and separate reference ideas. The registry supplies
conservative behavior/market classifications; core horizons come from configured
hold days. Event and Trend literals are read without importing their producers.
No current positions, live sleeve state, prices, or broker data are read.

The initial export contains 22 previously observed active algorithms and four
references. Trend stays in the baseline while cash-gated. Legend ETF, the hedge
candidate, paper dial SPY and Treasury research remain references. The prior
operating check, its evidence and its age are explicit; catalog creation does
not refresh the check. Unknown new strategies make the baseline incomplete.
Exact source file hashes and per-record configuration digests preserve lineage.

```powershell
python scripts/build_algorithm_family_catalog.py --as-of 2026-09-06T15:00:00Z
```

The command prints an immutable local path under
`artifacts/strategy_discovery/catalogs`. Supply that path with the existing
discovery inputs in [the runbook](strategy_discovery_runbook.md), adding
`--family-catalog <path> --candidate-families <path>`. Both are required together.
The catalog must be dated no later than the report cutoff.

Candidate annotations use this exact wrapper. The key is the discovery
candidate's structural fingerprint, not its title or a native config hash:

```json
{
  "schema_version": "candidate-family-profiles.v1",
  "candidate_profiles": {
    "<candidate fingerprint from the discovery report>": {
      "behavior": "MEAN_REVERSION",
      "markets": ["EQUITY_INDEX"],
      "horizon": "SHORT_TERM",
      "evidence_refs": ["Supplied SPY/QQQ three-session reversal rule specification"]
    }
  }
}
```

The taxonomy is in `family_fit.py`. Annotations are explicit source/spec
interpretations and require evidence references. The adapter checks identity
and holding-rule consistency; it cannot prove the annotator interpreted a paper
correctly. Missing classifications receive `NEEDS_CLASSIFICATION`.

Outputs distinguish `ACTIVE_FAMILY_OVERLAP`, `NO_ACTIVE_FAMILY_MATCH`, and
`INCOMPLETE_BASELINE`. Market, behavior and horizon overlap identify algorithm
peers despite different signal parameters. Opposite directions remain peers,
not an assumed hedge. Paper/planned peers are separate. A missing match is a
research lead, not measured diversification. A stale operating check (over 30
days), an unclassified active algorithm, or incomplete source coverage prevents
a conclusive fit label.

The CLI prints an immutable `family-fit-<digest>.json` companion binding the
report, catalog and annotation digests. It cannot promote a lifecycle or change
the journal. This broader family catalog is separate from the foundation's
exact-structure catalog: hashing scanner settings would not produce a valid
normalized proposal fingerprint. Running without the optional inputs explicitly
prints that family fit was not assessed.

## Integration completed after this MVP

- Official X v2 and Crossref/SSRN adapters now preserve cursors, raw text,
  social lineage, DOI/version provenance, request budgets, and complete versus
  partial coverage. X sources remain disabled until the owner supplies the
  approved list/query and bearer token.
- A native configured-strategy snapshot catches direct rule reuse; this broad
  family catalog remains the semantic overlap authority.
- The scheduled research skill preregisters candidates, builds costed and
  frozen empirical artifacts, tests active-book correlation/bad-period overlap,
  capacity and marginal portfolio improvement, and journals later validation.
- A deterministic finalizer applies the worthwhile gate and sends at most once.
  It writes a no-email decision without contacting SMTP when nothing qualifies.

No change here allocates capital, stages orders, changes activation, sends email,
uploads research, or modifies a scheduled task.

## Verification

The imported 160 tests plus new adapter tests cover actual core configuration,
all Event definitions, cash-gated Trend, separate references, index reversion
overlap, missing classifications, stale observations, corrupt digests, strict
input schemas, finite horizons, identity/horizon contradictions, immutable
companions, and the optional CLI without lifecycle promotion. Inputs were source
definitions and synthetic discovery fixtures. Live status, external coverage,
broker tradeability, empirical performance and delivery were not verified.
