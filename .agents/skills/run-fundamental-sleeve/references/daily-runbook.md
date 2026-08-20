# Daily Research Runbook

## 1. Initialize safely

1. Confirm the working directory and branch.
2. Read `AGENTS.md` and the owning Public Equity Investing skill instructions.
3. Inspect repository status and preserve unrelated changes.
4. Run workspace-hygiene start checks before editing a long-lived or dirty worktree.
5. Create disposable output locations with `python scripts/workspace_hygiene.py artifact-dir <category>`.
6. Record the as-of timestamp, policy/schema versions, intended source freeze times, and research-only posture.

Run the canonical planner first:

```powershell
python scripts/run_fundamental_sleeve.py --as-of YYYY-MM-DD
```

It is a true dry run by default. It performs no fetch, write, upload, deploy,
message, allocation, order, or portfolio mutation. For an authorized local
research build, use `--execute`; add `--refresh` only when the printed plan
shows missing or stale baselines. There is deliberately no `--upload` flag.

## 2. Load prior state first

Run:

```powershell
python scripts/pull_fundamental_inputs.py --site-state-only
```

Read `data/fundamental/current/site_state_latest.json` when present. Preserve it. Treat absence as an unavailable research-priority sync, not as unavailable fundamental research.

Load, when present:

- the prior candidate/current universe views;
- `underwrite_decisions_latest.json`;
- thesis/evidence/trigger state;
- company-map source records;
- the latest successful run manifest;
- a dated manual portfolio snapshot.

Canonical local projections are:

- controls: `data/fundamental/current/site_state_latest.json`;
- decisions: `data/fundamental/current/underwrite_decisions_latest.json`;
- triggers: `data/fundamental/current/triggers_latest.json`;
- thesis evidence: `data/fundamental/current/thesis_evidence_latest.json`;
- manual read-only portfolio context: `data/fundamental/current/portfolio_snapshot_latest.json`;
- latest run manifest: `data/fundamental/current/fundamental_run_manifest_latest.json`;
- append-only transitions: `data/fundamental/current/research_transitions.jsonl`.

Their contracts live under `fundamental/schema/`. Missing trigger, evidence,
manifest, or portfolio state fails closed and never means no events or zero
holdings.

Never infer that an absent portfolio snapshot means zero positions.

Apply controls:

- Prioritize `DEEPEN` for one bounded next-best diligence pass.
- Leave `WATCH` dormant until its trigger fires or evidence changes materially.
- Suppress `PASS` unless its stored reopening condition fires.
- Remove only the named override for `CLEAR`.

## 3. Check coverage and freshness

Inspect the broad universe, FMP endpoint coverage, SEC packages, estimates, price caches, source dates, parse failures, and specialist-lane coverage before fetching.

Maintain these distinctions:

- `discovered`: listed operating securities considered;
- `eligible`: securities passing listing, price, history, and liquidity rules;
- `baseline_ready`: eligible securities with current cheap research inputs;
- `deep_ready`: securities with the sources needed for a full underwrite;
- `decision_ready`: underwrites that pass every gate.

Refresh only missing, stale, or event-invalidated inputs. Use the repository CLIs and inspect `--help` before relying on unfamiliar flags. Keep batches bounded and resumable. Do not refetch the entire universe when coverage is adequate.

Use adjusted history for recomputed trend/return measures. Use the current as-traded price and current diluted capital structure for valuation. Preserve the repository dividend-basis invariant.

## 4. Maintain the broad view

Rebuild the eligible universe and baseline feature view when inputs changed. Keep standard companies, banks/lenders, insurers, REITs, commodity producers, utilities, pharma, and biotech in economically appropriate lanes.

Assign:

- one primary business-model lane;
- zero or more idea archetype hypotheses;
- structured coverage, rejection, and reopening reasons;
- price, estimate, filing, and capital-structure freshness;
- current trend state.

Avoid one universal cross-sector score. Use deterministic metrics to locate hypotheses and obvious risks, then use PM judgment to order research.

## 5. Build the event and work queue

Diff the current run against prior state. Arm work for:

- new SEC filing, earnings release, transcript, or IR presentation;
- material guidance, estimate, leadership, capital-allocation, financing, dilution, M&A, or regulatory change;
- thesis KPI confirm/warning/break threshold;
- underwritten valuation-band entry or exit;
- material market-relative price move or post-event gap;
- trend-state transition after hysteresis;
- imminent, delayed, or completed catalyst;
- critical source staleness or contradiction.

Order work:

1. Manually held names with warning or kill events.
2. Previously review-ready names with changed risk/reward.
3. Fired user-watch and thesis triggers.
4. User `DEEPEN` requests.
5. Hard catalysts inside the active window.
6. Active underwrites with the highest decision value.
7. New hypotheses with cheap, decisive rejection tests.
8. Routine coverage rotation.

Select no more than three companies for deep work.

## 6. Deep-underwrite selected names

For each name:

1. Verify issuer, security, current price, diluted shares, debt, cash, and enterprise-value bridge.
2. Tie reported financials and KPIs to primary sources.
3. Route through the selected business-model and idea-archetype framework.
4. Normalize earnings power, cash conversion, per-share value creation, leverage, dilution, and capital intensity.
5. Reconcile company guidance, consensus, estimate revisions, and the house view.
6. Quantify the current price's implied operating path.
7. Build bear/base/bull cases and an archetype-appropriate second valuation anchor when meaningful.
8. Map the variant wedge to KPIs, model lines, evidence timing, and the catalyst path.
9. Explain how money is lost before explaining how money is made.
10. Develop the strongest market-is-right case using disconfirming evidence.
11. Record exact proof, warning, break, and reopening triggers.
12. Assess trend, causal-driver overlap, liquidity, and conditional portfolio fit.

Route event-dependent work to the appropriate earnings or event workflow. Route ongoing pillars to `thesis-tracker`. Route conditional sizing only after a current portfolio snapshot exists.

## 7. Apply decisions and render

Apply [decision-gates-and-state.md](decision-gates-and-state.md). Allow the quantitative screen to propose research priority only. Require validated evidence for authoritative state changes.

Rebuild the candidate ranking and report after refreshed inputs or changed decisions. Update `reports/fundamental/fundamental_daily.html`, `reports/fundamental/company_maps.html`, private Fundamentals input payloads, and additive underwrites only when the underlying evidence changed.

Keep founder-CEO and product-circle maps separate from investment priority. Verify founder status against a current primary source and move completed leadership transitions to removals.

## 8. Verify and report

Run the narrow fundamental test suites first, then broader tests if shared code changed. Run frontend tests when the site payload or controls changed. Render changed HTML in a local headless browser and inspect the opening viewport plus decision-critical sections. Store screenshots and browser profiles under `artifacts/`.

After the real browser inspection, attest the exact report digest:

```powershell
python scripts/record_fundamental_visual_qa.py --status PASS --notes "What was inspected"
```

The immutable build manifest stays preserved; the latest projection links to
the separate QA attestation. A build remains `BUILT_AWAITING_VISUAL_QA` until
this step passes.

`completion_status: COMPLETE` means only that the bounded research workflow
and required QA finished successfully. It never means an issuer is
decision-ready. Read `investment_readiness` and `underwrite_contract`
separately; `NO_DECISION_READY` is a fully successful and common run outcome.

Return:

- `No action or review is needed` when nothing passes every gate; or
- no more than three `QUICK REVIEW` decisions.

Never commit, push, deploy, upload, message, allocate, stage, or trade during an automated research-only run unless McKinley explicitly expands scope.
