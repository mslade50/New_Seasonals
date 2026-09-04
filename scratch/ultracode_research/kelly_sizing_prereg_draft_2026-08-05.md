# Kelly Sizing Change Protocol — Pre-Registration Draft

**Draft date:** 2026-08-05  
**Evidence cutoff:** 2026-08-05  
**Current decision:** **NULL — retain every current sizing parameter**  
**Scope:** future static per-strategy and OVS path-ratio changes only  
**Not an authorization to ship:** this document defines reopening and decision gates; it changes no live file

## 1. Integrity statement

This draft was written after the Phase 3 results were known. It is therefore **not** a retroactive pre-registration of the analysis already completed. It prospectively locks the rules for any later attempt to convert the Kelly findings into a production change.

The Phase 3 null is final for the 2026-08-05 evidence vintage. The same history may be used to reproduce or debug the analysis, but not to search for another bps combination and call it new confirmation.

## 2. Registered null and alternatives

### Primary null, H0

The current per-strategy allocation is inside the practically usable noise band of fractional Kelly. Apparent differences in marginal growth are not sufficiently stable, transportable, and material to justify changing the current ratios.

### Global alternative, H1-G

A static, risk-budget-neutral rotation toward one or more seasoned strategies improves conservative log growth and exact-engine portfolio outcomes by enough to justify operational complexity and estimation risk.

### OVS alternative, H1-O

The current OVS Path-1 / Path-2 allocation is materially inferior to another static split that is supported in both the liquid and overflow evidence, survives conservative estimation, and improves exact-engine outcomes under the current Path-2 aggregate cap.

### Absolute-risk alternative, H1-A

A higher common Kelly fraction satisfies the drawdown constraint and is consistent with the separate GRM risk-appetite decision.

H1-A is recorded for completeness but is **not** an active sizing proposal. The owner has already set absolute appetite primarily through the GRM replay. This study cannot authorize a GRM change.

## 3. Locked 2026-08-05 findings

These results may not be reinterpreted as a passing proposal without new evidence:

- the full relative optimizer is a corner allocation dominated by OLV, ATR Extended Gap Up, and Sector BO;
- OLV is the only nonpilot upweight across every global sensitivity;
- the smallest liquid-supported OLV/WCDS rotation adds only **$551 per year** and **0.004 Sharpe** in exact replay;
- OVS full-history estimates favor P1, while liquid-only estimates favor P2;
- the current book is approximately 5%–6% of the fitted full-Kelly ray;
- quarter-Kelly has a **55.12%** estimated probability of a one-year drawdown worse than 20% of starting NAV;
- half-Kelly has a **96.74%** estimated probability of breaching that drawdown;
- the drawdown-constrained boundary is approximately **0.115 Kelly** under the registered model;
- all 25 verification checks passed.

Therefore, as of this cutoff:

| Parameter family | Registered action |
|---|---|
| Per-strategy nominal bps | No change |
| OVS P1 nominal bps | Keep 40 |
| OVS P2 nominal bps | Keep 8 |
| OVS P2 aggregate cap | Keep current 0.75% nominal / 1.125% effective |
| Global risk multiplier | No recommendation from this study |
| 250-bps per-strategy daily cap | No change |
| Pilots | Freeze at current size |
| Fragility, ladder, cycle, gap, scale-out, and other appetite overlays | Out of scope; no change |

“No change” includes not removing an appetite overlay merely because a fitted Kelly diagnostic would prefer more exposure. Appetite costs are deliberate policy choices and remain the owner’s decision.

## 4. Frozen research specification

Any future rerun intended to reopen the decision must start from this specification. Deviations must be written into a new pre-registration **before** results are computed.

### 4.1 Portfolio and engine controls

- flat NAV: **$750,000**;
- production book and engine as they exist at the future cutoff;
- production global risk multiplier as it exists at that cutoff;
- production per-strategy daily cap as it exists at that cutoff;
- no pooled caps unless they have returned to production before the new cutoff;
- liquid and overflow passes modeled separately before aggregation;
- current overlays replayed exactly;
- OVS scale-out rows collapsed to one original filled signal for R estimation;
- all exact counterfactuals rerun through the engine from candidates rather than scaling completed-trade P&L.

If the framework changes between cutoffs, both the incumbent and candidate must be rebuilt under the same new framework. A candidate cannot receive credit from a framework mismatch.

### 4.2 Estimation controls

- decision unit: one filled signal;
- return unit: realized R based on actual engine risk dollars;
- mean: risk-dollar weighted;
- primary conservative mean: minimum leave-one-year-out mean, then empirical-Bayes shrinkage;
- episode definition: gap greater than five trading days;
- OVS effective N: minimum of gap-episode and calendar-month effective samples;
- covariance: daily component P&L vectors with Ledoit-Wolf shrinkage;
- mandatory sensitivity vintages: full history, since 2018, crisis covariance, liquid only, and prior dispersion at 0.5x and 2.0x;
- pilots fixed at 1.0: 3x Bear ETF Overbot Fade, 3x Leader Gap Fade, Monthly Weak Close;
- nonnegative strategy multipliers;
- headline fraction: quarter-Kelly; half-Kelly shown only as reference;
- relative solution: preserve annualized filled-risk budget;
- drawdown model: 20,000 stationary-bootstrap paths, 252 trading days, mean block 10 days, seed **20260805**;
- drawdown gate: `P(max drawdown worse than 20% of starting NAV) < 5%`.

### 4.3 No-search rule

Only one primary bps candidate may be tested per reopening event. The candidate and its funding offset must be written down before exact replay. If it fails, the same data vintage is closed; no second carrier, offset, increment, cap, or path ratio may be substituted.

Exploratory optimizer output can identify a future hypothesis, but it cannot itself become a shipped allocation. Raw multipliers above practical engine capacity remain diagnostics.

## 5. Reopening triggers

No periodic rerun is required. Reopen only after one of the following occurs:

1. **Point-in-time overflow data:** a survivorship-controlled universe and membership history becomes available and can reconstruct the liquid/overflow opportunity set at each signal date.
2. **Material new OVS evidence:** at least **40 additional liquid filled signals in each path** after 2026-08-05, with no changes to path classification during the accumulation window.
3. **Material new carrier evidence:** at least **24 calendar months** and **30 new affected filled signals** for a named increase/funding pair after 2026-08-05.
4. **Book-framework change:** a production change to caps, path handling, universe, or a major overlay materially changes the allocation problem. This requires a new brief; it does not automatically favor a Kelly change.
5. **Owner-requested GRM review:** absolute appetite is explicitly reopened. This is separate from the relative-allocation protocol.

The first three triggers create permission to study, not a presumption to change.

## 6. Global relative-allocation gate

A future static cross-strategy rotation ships only if **every** gate below passes.

### 6.1 Eligibility

- the increased component is not a frozen pilot;
- its shrunk conservative R is positive;
- the funding component also retains positive shrunk conservative R, so the change is a rotation rather than an implicit retirement;
- the candidate does not alter fragility bands, ladders, cycle-year rules, earnings sizing, signal de-rates, scale-outs, stops, or caps;
- the first proposed step changes any strategy by no more than **5 nominal bps**;
- annualized filled risk is within **±1% of the incumbent filled-risk budget**.

The five-bps step is a production-candidate limit, not a bound on the diagnostic optimizer.

### 6.2 Directional robustness

- the proposed carrier is above 1.0 in the primary equal-risk solution;
- it remains above 1.0 under full, post-2018, crisis-covariance, liquid-only, and both prior-strength sensitivities;
- an increase may not depend on overflow-only evidence unless point-in-time overflow membership is available;
- the proposed funding component is below 1.0 in at least four of those six sensitivity families;
- the sign of the carrier-versus-funder shrunk expectancy difference is unchanged in every leave-one-year-out fit;
- dropping the single best calendar year leaves the counterfactual’s cumulative P&L delta positive.

### 6.3 Statistical and economic materiality

All figures come from the exact incumbent-versus-candidate engine replay:

- annualized P&L improvement at least **$7,500**, equal to 1% of starting NAV;
- Sharpe improvement at least **+0.05**;
- active-month paired, episode-aware t-statistic at least **+1.5**;
- cumulative delta positive in at least **60% of calendar years with affected trades**;
- minimum leave-one-year-out cumulative P&L delta greater than zero;
- no more than 25% of total improvement supplied by the single best year.

The $7,500 threshold is intentionally below the book’s $25,000–$30,000 new-sleeve materiality gate because a static sizing rotation adds little operational surface. It is still more than an order of magnitude above the $551 annual gain found in Phase 3.

### 6.4 Risk and concentration

- maximum drawdown may not worsen by more than **0.5% of NAV**, or $3,750 on the frozen basis;
- worst day may not worsen by more than **0.25% of NAV**, or $1,875;
- bootstrap probability of a drawdown worse than 20% NAV remains below **5%**;
- bootstrap 5th-percentile terminal P&L may not decline;
- no strategy’s annual filled-risk share may exceed **25% of total nonpilot filled risk** after the change;
- exact replay must show and explain every fill-count change caused by caps, notional limits, rounding, or state interactions.

Failure of any gate returns the decision to H0. Gates are not averaged into a score.

## 7. OVS path-split gate

OVS receives a separate gate because the current conflict is about transportability, not sample size alone.

### 7.1 Required evidence before a candidate exists

At least one must be true:

- a point-in-time overflow-universe reconstruction is available; or
- the post-cutoff liquid sample has added at least 40 filled P1 signals and 40 filled P2 signals.

Then all of the following must hold:

- the sign of `shrunk R(P1) - shrunk R(P2)` agrees in full, liquid-only, and overflow-only samples;
- that sign survives every leave-one-year-out exclusion and the 0.5x/2.0x prior-dispersion sensitivity;
- the path-difference clustered t-statistic is at least **1.5** in the full sample and at least **1.0** in liquid only;
- both paths retain positive shrunk conservative R;
- the fixed-OVS-risk optimizer points in the same direction under full, post-2018, crisis-covariance, and liquid-only specifications;
- no conclusion depends solely on current overflow survivors.

The 2026-08-05 evidence fails the first, third, and fifth conditions because the full and liquid rankings reverse.

### 7.2 Candidate construction

- neither path may be set to zero;
- the first candidate may move each path’s nominal bps by at most **20%** from the incumbent value;
- total annualized OVS filled risk must remain within **±1%** of incumbent OVS filled risk;
- the Path-2 aggregate cap remains fixed unless a separate cap study was pre-registered before results;
- path-classification threshold, fill rules, stops, hold logic, earnings blackout, cycle tilt, and tier eligibility remain fixed;
- the exact proposed bps pair is written before the engine replay.

### 7.3 OVS shipping gates

The global materiality and risk gates in Sections 6.3 and 6.4 apply, except that the annualized P&L threshold is **$5,000** because the comparison is internal to one strategy. In addition:

- annual OVS P&L improves in at least 60% of years with both paths represented;
- the worst OVS episode does not worsen;
- P1 and P2 each retain at least 20% of incumbent OVS filled risk, preventing an estimated corner from becoming de facto path retirement;
- the P2 aggregate-cap binding rate and the amount trimmed are reported before and after;
- liquid and overflow deltas are both nonnegative after dropping their respective best year.

Failure returns to the current 40/8 nominal split and current cap. There is no fallback ratio search on the same vintage.

## 8. Absolute-fraction gate

The Phase 3 reference fractions are registered as rejected under the current objective:

- quarter-Kelly: fails;
- half-Kelly: fails;
- full Kelly: fails.

The fitted 0.115-Kelly boundary is **not** a recommendation. It may be recomputed only if the owner explicitly reopens absolute risk and the engine is extended to apply common-scale changes with its real nonlinear caps and notional constraints.

Any future absolute-risk candidate must:

- be specified by the owner before replay;
- pass the same 5% drawdown-probability constraint;
- pass exact-engine replay rather than linear scaling;
- state the resulting effective per-strategy and pooled concentrations;
- remain separate from the relative-allocation decision so an attractive ratio result cannot smuggle in a higher GRM.

## 9. Reporting contract for a future rerun

The future decision memo must show, whether the candidate passes or fails:

1. provenance: commit, data cutoff, cache vintages, engine controls, and strategy count;
2. signal reconciliation before and after OVS tranche collapse;
3. full and liquid component estimates, effective N, LOYO floor, and shrinkage weight;
4. full, post-2018, crisis, liquid, and shrinkage-strength allocation tables;
5. OVS P1/P2 results by tier even when OVS is not the candidate;
6. full and crisis correlation matrices;
7. exact-engine incumbent/candidate P&L, volatility, Sharpe, Sortino, max drawdown, worst day, fill counts, and filled risk;
8. calendar-year, active-month, and episode-level candidate deltas;
9. stationary-bootstrap drawdown and terminal-P&L distributions;
10. a pass/fail table containing every registered gate, with no omitted failures;
11. scripts and outputs sufficient to reproduce every number;
12. an explicit final sentence choosing H0 or the named alternative.

## 10. Change-control requirements after a pass

Passing the research gate still does not authorize implementation. A separate change request must:

- enumerate every aligned live surface affected;
- include scan/engine parity tests and boundary tests for any changed bps or path cap;
- rebuild the full ledger and compare it to the registered counterfactual;
- verify staged `Risk_Bps`, `Risk_Amt`, shares, cap notes, and OVS path stamps on representative dates;
- preserve pilot freezes and every out-of-scope appetite overlay;
- define a one-line rollback to the prior bps values;
- receive the owner’s explicit approval before any financially consequential live action.

No implementation work is part of this research task.

## 11. Registered result for the current vintage

| Candidate | Statistical gate | Materiality gate | Transportability gate | Risk gate | Decision |
|---|---|---|---|---|---|
| Raw global Kelly allocation | Not decision-ready | Not engine-feasible | Fails tier/cap realism | Not literal at raw multipliers | Reject |
| OVS P1-heavy full-sample split | Positive full-sample edge | Not tested as a ship candidate | **Fails: liquid favors P2** | Not reached | Reject |
| OVS P2-heavy liquid split | Positive liquid edge | Not tested as a ship candidate | **Fails: full sample favors P1** | Not reached | Reject |
| OLV +5 bps liquid, WCDS offset | Active-month t ≈ 1.69 | **Fails: +$551/yr and +0.004 Sharpe** | Direction broadly robust | Passes max-DD / equal-risk checks | Reject |
| Quarter-Kelly absolute scale | Not the relative question | Large modeled growth | Not applicable | **Fails: 55.12% drawdown-breach probability** | Reject |
| Half-Kelly absolute scale | Not the relative question | Large modeled growth | Not applicable | **Fails: 96.74% drawdown-breach probability** | Reject |
| Current allocation | Baseline | Baseline | Avoids unsupported tier choice | Passes drawdown gate | **Retain** |

**Registered decision: H0. Change nothing.**
