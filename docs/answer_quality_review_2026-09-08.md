# Priority 1: review of existing research and portfolio answers

Review date: 2026-09-08. Scope agreed in [the working plan](trading_desk_working_plan.md).

**Subsequent owner clarification, same day:** Daily Pitch must assess standalone
quality without considering holdings or the existing strategy mix. Its book,
staging, sleeve and exposure inputs and overlap criterion have therefore been
removed. The pitch input-date/overlap safeguards described below record the
earlier review; they are superseded by excluding those inputs altogether.
Historical corrections remain valid. Strategy-discovery fit rules are unchanged.

The sampled rejection decisions remain supported, but several explanations
overstated what their inputs established. This review corrects those conclusions
and the reusable input/instruction defects. It is not a certification of all
research, live execution, or future agent reasoning.

## Samples and disposition

| Sample | Evidence traced | Result |
| --- | --- | --- |
| September 8 pitch surface map and final stand-down payload | `scratch/pitch_checks/2026-09-08/00_surface_map.md`, `data/pitch_state.json`, `data/pitch_ideas.json`, publisher's `render_stand_down`, saved sent receipt | Unsupported actual-holdings/overlap conclusion and stale Trend observation. Corrected below; state contract and agent instructions changed. Receipt records a send, not independently verified inbox contents. |
| September SPY PPI/CPI candidate in that completed pitch | `c1b_september_subcell.py`, `pitch_lab.fwd_lag`, local price cache and macro event calendar | Main sample statistics reproduced; quoted permutation probability tests the wrong observed threshold. Corrected probability strengthens this specific rejection. |
| September 8 ETF-grid discovery decision | `data/strategy_research/latest_decision.json`, frozen research script, `results.json`, 5,763 deployment rows and 5,936 daily rows | Seventeen saved metrics recomputed to rounding tolerance; NO_EMAIL agrees with saved metrics. Interpretation, capacity and capital-budget claims need the qualifications below. |
| Theoretical portfolio email | `run_12month_backtest` → `generate_sizing_recommendations` → `send_portfolio_email` | Modeled positions/P&L were labeled generically. Email now explicitly identifies theoretical results and the last modeled session; numbers and strategy rules unchanged. |
| Portfolio / Theo vs Actual / Overlay Lab explanations | `site/index.html`, `site/comparison.html`, `site/assets/comparison.js`, `site/assets/portfolio.js`, existing regression checks | Main view is theoretical; actual figures are explicitly broker-reported subtotals on another basis. Unknown P&L stays unknown. Multi-overlay combinations disclose omitted interactions. Preserve these distinctions; no redesign or site changes warranted by this sample. |

## Corrected pitch conclusions

The saved pitch state is dated September 8, with September 4 as the preceding
NYSE session. It contains 15 configured algorithms and zero staged rows. It
deliberately contains no actual broker holdings. The saved exposure-leg setting
is zero as of September 4; the Trend CASH observation is **August 31**, not
September 8. There were no book warnings about that stale sleeve observation.

Consequently, neither the surface map's "holding nothing" claim nor the final
payload's "nothing in the book to hedge it" claim follows. Absence of current
signals also cannot establish absence of overlap with the configured strategies.
The final payload's "most fragile tape the dial has recorded" is unsupported by
a 99th-percentile description; that is not proof of a historical maximum.

Use this interpretation when reading the saved pitch:

> The price-state analysis uses September 4 observations. The saved fragility
> reading is 88.0; the saved exposure-leg setting is zero. Trend's available
> CASH observation is dated August 31 and cannot establish its current state.
> Zero staged rows says nothing about actual holdings and does not establish
> diversification. The candidate-specific tests, rather than an inferred flat
> account, are the evidence for the stand-down. Algorithmic overlap must be
> assessed against the configured strategy mix regardless of today's signals.

For the September SPY candidate, the original script reproduces 14 observations,
mean +0.844%, and 11 observations with FOMC outside the holding window averaging
+0.518%, 81.8% positive, sign-test result 0.0327. These are historical gross
window statistics from the repository cache, not new trade recommendations.

Its quoted permutation probability **0.1618** compares each shuffled maximum
month mean with the original **March maximum (+1.379%)**, not September's
+0.844%. Reusing exactly the same 5,000 permutations and seed, the probability
that any eligible shuffled month equals or exceeds September is **0.7354**.
The original claim about September's threshold was wrong; correcting it does
not rescue this candidate. This verifies the quoted test, not every candidate
or every research method in the completed pitch.

## Corrected grid-study conclusions

Original evidence is preserved in
`artifacts/strategy_discovery/validation_artifacts/2026-09-08-etfgrid/`.
The local price and ledger SHA-256 digests match those frozen by that study.

- **Relative versus absolute returns:** the grid averaged +505.35 bps per
  deployment in absolute terms versus +570.05 bps for the fixed initial-exposure
  control: an increment of **-64.69 bps**. Its -7,169.52 bps headline worst result
  is likewise relative to that control; its worst absolute grid return is
  -7,468.43 bps. Say "underperformed the control," not an unqualified "lost money."
- **The constant-weight control is fitted in sample:** its 42.6464% weight is
  the average realized exposure across the full evaluation sample. Holding that
  number constant removes window-specific fitting but does not make it
  hindsight-free. The +19.15 bps increment against it is a diagnostic, not
  validated executable incremental edge. The larger +431.80 bps result against
  the window-fitted control is also unsuitable as evidence of tradable alpha.
- **Capacity:** the saved $5.00m figure divides 1% of the least-liquid eligible
  dollar volume by a 10% tranche. Applying the same assumption to the 50%
  initial order yields **$1.00m**, before same-instrument concurrent orders,
  liquidation, spread or impact constraints. This still exceeds the existing
  $750k email threshold in isolation, so it does not reverse NO_EMAIL. Neither
  number certifies aggregate executable strategy capacity.
- **Capital budget:** the daily stream scales each deployment by $150k divided
  by full-sample mean concurrency (239.9). At peak concurrency of 315, implied
  committed capital reaches **$196,958**. The reported 8.98% average inventory
  occupancy is reproducible, but this is not a hard $150k capital-capped replay.
  Treat its incremental Sharpe (-0.0707) as a result of that normalization,
  not a verified prediction for an enforceable allocation policy.
- **Uncertainty:** present-day survivor selection has not established a
  direction or bound for bias in the strategy-minus-control result. The
  anchor-cluster bootstrap reproduces 0.9914, but grouping shared anchor dates
  does not by itself handle dependence across adjacent, overlapping 252-session
  deployments. Do not present its frequency as fully calibrated confidence.

The rejection remains supported by the negative increment under the stated
control and costs, weak neighboring results, and unfavorable saved portfolio-fit
diagnostics. No new allocation decision follows. This review reproduces saved
row arithmetic and reads the simulator; it does not independently reimplement
the full order model or resolve intraday fill-path ambiguity.

## Changes and verification

- `build_pitch_state.py` now states the configured-strategy overlap basis and
  explicit exclusion of actual holdings. Successfully read empty staging,
  unreadable staging and skipped staging have distinct source states.
- Each sleeve carries its own date status. Stale, future, undated and unreadable
  sources cannot silently appear as current observations. Old evidence is
  retained for dated interpretation rather than discarded.
- Daily Pitch instructions now require the correct portfolio meaning, dated
  claims and named permutation statistics. Strategy Research instructions cover
  return/control bases, fitted comparators, opening-order capacity, overlapping
  deployments and retrospective capital normalization. Both reference this
  correction before reusing the affected research.
- The portfolio email identifies the theoretical basis, uses "latest modeled
  session" instead of "today" for P&L, and labels modeled positions and configured
  equity. No calculation, broker action, allocation or trade gate changed.

Validation completed:

- **92 Python checks passed** across `test_answer_quality_contracts.py`,
  `test_pitch_automation_receipts.py`, `test_daily_pitch.py`,
  `test_strategy_research_email_gate.py` and `test_overlay_free_portfolio.py`.
  The new staging and sleeve tests first failed against the prior behavior.
  Coverage includes empty/failed/skipped staging, stale/future/missing/malformed
  sleeve dates, the Labor Day boundary, and actual MIME email formatting with a
  fake SMTP transport. No real message was sent.
- `node tests/js/test_portfolio_comparison.mjs` passed: excludes PA, untagged,
  open/future modeled trades and preserves unknown broker P&L.
- Seventeen grid-study metrics independently recomputed from saved rows within
  CSV rounding tolerance, including costed increment, median, bad-day co-loss,
  correlation, Sharpe change and anchor-cluster resampling frequency.
- Original September candidate script reran successfully; the corrected
  permutation threshold was recomputed with its original seed and sample.
- Offline book assembly using the saved files now identifies the August 31
  Trend state as stale. Event state has a generated timestamp but no explicit
  as-of date; it is marked undated rather than inventing one.
- Whitespace and workspace-hygiene checks passed; unrelated changes preserved.

The system Python has a pre-existing Streamlit/protobuf compatibility issue.
Tests used process-local `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`; no
dependency or runtime configuration was changed. The resulting 19 warnings are
upstream deprecations.

Read-only reproductions from the repo root:

```powershell
python artifacts/answer-quality/verify_saved_research.py
python artifacts/answer-quality/verify_pitch_september.py
```

Evidence is in `artifacts/answer-quality/`: `saved_research_verification_v2.json`,
`pitch_c1b_reproduction.txt`, `pitch_september_threshold_correction.txt` and
`observed_book_contract.json`. The verification JSON includes input digests.

## Limits and release boundary

Original pitch payloads, journals, delivery receipts and discovery artifacts
remain unchanged; this is the correction to use when interpreting them. Nothing
was resent. No new pitch, strategy search, daily scan, production data rebuild,
deployment, task registration or trading action was run.

Source corrections are validated in the development checkout. The local Daily
Pitch launcher reads that checkout; the separately pinned portfolio runtime will
need the normal reviewed promotion to receive the email wording changes. This
step does not certify a new unattended run. Priorities 2–4 remain queued.
