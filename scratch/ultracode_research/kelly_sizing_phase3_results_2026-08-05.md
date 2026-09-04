# Kelly-Based Strategy-Book Sizing — Phase 3 Results

**Study date:** 2026-08-05  
**Status:** Research complete; no production change authorized or recommended  
**Headline conclusion:** **Change nothing.** The current strategy ratios are not the unconstrained Kelly ratios, but the apparent improvements are too corner-like, tier-dependent, and economically immaterial in exact replay to justify changing any per-strategy bps. OVS, the priority case, has positive estimated edge in both paths; its preferred path allocation reverses when overflow is removed, so the current 40/8 nominal split should remain unchanged.

## 1. Decision summary

| Question | Result | Decision |
|---|---|---|
| Are current per-strategy bps proportional to estimated Kelly? | No. Raw Kelly estimates are much larger and the correlated optimizer concentrates almost all nonpilot risk in OLV, ATR Extended Gap Up, and Sector BO. | Treat as a diagnostic, not a sizing instruction. |
| Is there a stable book-level reallocation? | No. The solution sits on nonnegative-weight corners and is sensitive to tier support and covariance sample. | No book-level ratio change. |
| Should OVS Path 1 / Path 2 be changed? | No. Full-universe data favors Path 1; liquid-only data favors Path 2. Both paths retain positive shrunk expectancy. | Keep 40/8 nominal and the current Path-2 cap. |
| Is the present book too small versus mathematical Kelly? | Yes, mechanically: current risk is about 5%–6% of the fitted full-Kelly ray. | No action. Absolute risk is outside the headline allocation question and the drawdown gate rejects quarter- and half-Kelly. |
| Does the strongest robust small rotation survive exact engine replay? | Barely in sign, not in magnitude: +$551 annual P&L and +0.004 Sharpe at essentially equal filled risk. | Reject as noise / immaterial. |

The null is not a statement that every current bps number is mathematically optimal. It is a statement that this evidence does not identify a different set of ratios with enough stability and economic importance to replace them.

## 2. Fresh current-framework replay

The supplied July ledger was not used as the primary estimation dataset because it predates the current strategy book, omits Monthly Weak Close, and expands OVS scale-out tranches into multiple rows. A fresh research-only replay was built from the current engine and current book in memory.

Replay controls:

- flat starting NAV: **$750,000**;
- current `GLOBAL_RISK_MULTIPLIER = 1.5`;
- current per-strategy daily cap: **250 effective bps**;
- pooled long/short caps: **off**, matching the current framework;
- all 15 current strategies, plus the six eligible overflow passes;
- OVS tranches collapsed to the original filled signal before estimating R distributions;
- no live code or configuration edited.

Baseline results:

| Metric | Fresh replay |
|---|---:|
| Candidate signals | 24,264 |
| Engine trade rows | 4,778 |
| Collapsed filled signals | 3,585 |
| Estimation components | 16, including OVS P1/P2 |
| History | 2003-01-01 through 2026-08-05 |
| Total P&L | $4,017,313 |
| Annualized P&L | $164,451 |
| Annualized volatility | $84,945 |
| Sharpe | 1.936 |
| Sortino | 3.328 |
| Maximum additive drawdown | -$91,099, or -12.15% of starting NAV |
| Worst day | -$44,244 |
| Annualized filled-risk budget | 50.15% of NAV |

“Filled-risk budget” is the annualized sum of risk dollars on filled trades divided by $750,000. It is a relative-allocation budget, not a forecast drawdown or a statement that half the account is simultaneously at risk.

## 3. Estimation and computation

### 3.1 Return observations

The primary observation is one filled signal, measured in realized R using the actual flat-NAV risk dollars carried by the engine. This preserves the effect of path sizing, caps, tier overrides, fragility bands, ladders, cycle overlays, and de-rates in the current implementation.

For OVS, scale-out tranches were recombined into one signal-level outcome. Path 1 and Path 2 remain separate allocation components.

### 3.2 Dependence-aware effective samples

Signals were grouped into episodes separated by more than five trading days. OVS used the more conservative effective sample obtained from gap-defined episodes and calendar-month clusters. Estimates therefore do not treat same-cluster signals as independent bets.

The primary conservative mean is:

1. risk-dollar-weighted expectancy;
2. replaced by the minimum leave-one-year-out estimate;
3. shrunk toward a cross-strategy empirical-Bayes prior using cluster uncertainty.

The fitted leave-one-year-out prior mean was **0.491 R**, with cross-strategy variance **0.134 R²**. Shrinkage was repeated at one-half and twice the fitted prior dispersion as a sensitivity.

### 3.3 Covariance and optimization

Daily component P&L vectors were used for covariance so same-day strategy clustering is retained. Ledoit-Wolf shrinkage was applied:

- full history: **4.4%** shrinkage;
- since 2018: **17.1%**;
- defined crisis windows: **54.3%**;
- liquid-only sample: **8.5%**.

The primary optimizer maximizes empirical log growth subject to nonnegative multipliers, with the three pilots fixed at 1.0. Its relative-allocation version preserves the current annualized filled-risk budget. No upper multiplier bound was imposed in the raw diagnostic, as agreed in Phase 2.

### 3.4 Drawdown computation

The drawdown gate uses 20,000 one-year stationary-bootstrap paths, a 10-trading-day mean block, and seed 20260805. Daily component vectors are resampled together, preserving observed cross-strategy co-movement. The constraint is:

> Probability that one-year additive drawdown exceeds 20% of starting NAV must be below 5%.

This is a flat-NAV research model, consistent with the sizing estimand. It is not a compounded production-account forecast.

## 4. Component estimates

The table below is deliberately diagnostic. “Standalone full Kelly” is the fraction of NAV risk implied by a one-component empirical log-growth problem; it is not a proposed position size. “Raw correlated full multiplier” is conditional on pilots being held at their current size. The equal-risk multiplier preserves the book’s current annual filled-risk budget.

| Component | Current effective bps, liquid | Ledger avg R | Effective N | Shrunk LOYO R | Standalone full Kelly | Raw correlated full multiplier | Equal-risk multiplier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3x Bear ETF Overbot Fade | 37.5 | 0.67 | 32 | 0.57 | 21.7% | 1.00* | 1.00* |
| 3x ETF Overbot Fade | 60.0 | 0.86 | 28 | 0.38 | 39.5% | 18.04 | 0.00 |
| 3x Leader Gap Fade | 37.5 | 1.16 | 11 | 0.50 | 36.0% | 1.00* | 1.00* |
| 52wh Breakout | 52.5 | 0.52 | 148 | 0.42 | 17.3% | 5.81 | 0.00 |
| ATR Extended Gap Up | 60.0 | 0.78 | 57 | 0.65 | 20.2% | 20.89 | 5.84 |
| Indices Oversold Bounce | 52.5 | 0.32 | 149 | 0.28 | 11.9% | 9.93 | 0.00 |
| LT Trend ST OS | 45.0 | 0.32 | 98 | 0.27 | 21.6% | 25.28 | 0.00 |
| Monday Dip | 45.0 | 0.38 | 65 | 0.38 | 31.1% | 10.83 | 0.00 |
| Monthly Weak Close | 45.0 | 1.81 | 12 | 1.63 | unbounded in sample | 1.00* | 1.00* |
| Overbot Vol Spike P1 | 60.0 | 0.50 | 134 | 0.29 | 16.3% | 19.68 | 0.00 |
| Overbot Vol Spike P2 | 12.0 | 0.23 | 156 | 0.17 | 15.1% | 51.84 | 0.00 |
| Oversold Low Volume | 52.5 | 0.70 | 99 | 0.69 | 26.2% | 27.33 | 11.52 |
| SPY QQQ MonFri Reversion | 52.5 | 0.40 | 186 | 0.43 | 18.9% | 37.07 | 0.00 |
| Sector BO | 37.5 | 0.93 | 56 | 0.56 | 13.6% | 3.21 | 1.48 |
| St OS Sznl | 60.0 | 0.32 | 37 | 0.41 | 7.5% | 12.25 | 0.00 |
| Weak Close Decent Sznls | 52.5 | 0.28 | 145 | 0.23 | 18.5% | 8.02 | 0.00 |

\* Pilot fixed at 1.0 by instruction; the displayed correlated multiplier is not its unconstrained optimum.

### What this table does and does not say

All 16 shrunk conservative expectancies are positive. Zero weights in the equal-risk optimizer therefore do **not** mean the omitted strategies have negative edge or should be retired. They mean that, in a linear regime far below fitted Kelly, a nonnegative optimizer sends the next unit of scarce risk to the components with the highest estimated marginal growth after covariance.

That creates a corner solution:

- OLV: **11.52x**;
- ATR Extended Gap Up: **5.84x**;
- Sector BO: **1.48x**;
- the three pilots: **1.00x**, fixed;
- every other nonpilot component: effectively **0x**.

The quadratic approximation produces the same qualitative corner. Crisis covariance, post-2018 means, liquid-only estimation, and shrinkage-strength sweeps move the exact numbers but not the basic concentration. OLV is the only nonpilot above 1.0 in every global sensitivity, with a broad approximate range of **4.5x to 21.7x**.

This apparent robustness is insufficient for implementation. It is exactly the pattern expected when:

- every mean is positive;
- present risk is only a small fraction of fitted Kelly;
- risk is constrained by a linear budget rather than realistic capacity functions;
- current caps and signal depletion become nonlinear well before 5x–20x multipliers;
- the available universe is selected using today’s surviving strategy definitions.

The optimizer is useful for identifying which evidence to challenge. It is not a credible literal allocation.

## 5. OVS priority case

OVS is the tightest overall sample, but the path split is not transportable across tiers.

| OVS component | Sample | Signals | Effective N | Full risk-weighted R | LOYO R | Shrunk LOYO R |
|---|---|---:|---:|---:|---:|---:|
| Path 1 | Full, liquid + overflow | 651 | 134 | 0.312 | 0.280 | 0.286 |
| Path 2 | Full, liquid + overflow | 599 | 156 | 0.185 | 0.159 | 0.166 |
| Path 1 | Liquid only | 164 | 63 | 0.142 | 0.087 | 0.123 |
| Path 2 | Liquid only | 140 | 69 | 0.290 | 0.235 | 0.256 |

At a fixed current OVS filled-risk budget:

| Scenario | P1 multiplier | P2 multiplier | Direction |
|---|---:|---:|---|
| Full-history conservative means | 1.259 | 0.000 | All OVS budget to P1 |
| Full-history unshrunk means | 1.259 | 0.000 | All OVS budget to P1 |
| Crisis covariance | 1.259 | 0.000 | All OVS budget to P1 |
| Since-2018 estimates | 1.250 | 0.000 | All OVS budget to P1 |
| Liquid-only estimates and covariance | 0.000 | 5.603 | All OVS budget to P2 |
| Full covariance with liquid-supported means | 0.000 | 4.863 | All OVS budget to P2 |

The current effective per-trade ratio is 60 bps for P1 versus 12 bps for P2, plus the fixed effective Path-2 aggregate cap of 1.125% of NAV. The full sample says this is not P1-heavy enough; the liquid sample says it is much too P1-heavy. The reversal is too large to average away as an inconsequential numerical difference.

The full-sample result is driven by overflow behavior that cannot presently be separated from survivor and universe-selection effects with a point-in-time membership dataset. Consequently:

1. neither path should be eliminated;
2. P2 should not be re-enabled, disabled, or resized on this analysis alone;
3. the current P1/P2 bps and P2 aggregate cap should remain fixed;
4. a future OVS split change requires a point-in-time overflow-universe audit or genuinely new out-of-sample liquid evidence.

## 6. Correlation and crisis behavior

The FAMILY4 average pairwise daily correlation rises from **0.148** over the full sample to **0.212** in the defined crisis windows. This confirms that their diversification benefit weakens when it matters, consistent with the existing fragility throttle.

OVS P1/P2 correlation is **0.267** full-history and **0.233** in crisis windows. The two paths are not independent sleeves, but their correlation is not the reason the split recommendation fails; the tier-dependent mean reversal is.

The largest visible crisis shift among frozen pilots is 3x Bear versus 3x Leader Gap, whose correlation rises from **0.257** to **0.641**. Since both are pilots and frozen by instruction, this is a monitoring observation rather than an allocation input.

## 7. Current book versus quarter- and half-Kelly

Projecting the current allocation onto the fitted correlated Kelly direction gives:

- risk-budget equivalent: **5.24% of full Kelly**;
- variance equivalent: **5.80% of full Kelly**;
- covariance-metric directional cosine: **0.846**.

So the book is directionally similar to the positive-edge Kelly vector but far smaller in absolute magnitude. That alone is not a reason to increase risk.

Bootstrap results along the fitted Kelly ray:

| Allocation scale, c | Probability of one-year drawdown worse than $150,000 | Drawdown gate |
|---|---:|---|
| Current book | 0.155% | Pass |
| 0.10 Kelly | 3.36% | Pass |
| 0.1148 Kelly | 5.00% | Boundary |
| **0.25 Kelly** | **55.12%** | **Fail** |
| **0.50 Kelly** | **96.74%** | **Fail** |
| 1.00 Kelly | approximately 100% | Fail |

For the current book, the bootstrap median one-year maximum drawdown is **-$45,558**, the 5th-percentile tail is **-$91,099**, and the 1st-percentile tail is **-$118,560**. Median terminal P&L is **$162,341**; the 5th percentile is **$40,255**; 1.51% of paths finish negative.

Quarter-Kelly remains the headline reference fraction and half-Kelly is shown for comparison, as requested. Neither passes the pre-agreed drawdown constraint. The largest passing scale is about **0.115 Kelly**, still an analytic tangent rather than a proposed absolute-risk change.

At large multipliers the linear component rescaling also ceases to reproduce the real engine because 250-bps daily caps, notional limits, share rounding, and fill-state interactions bind. If those nonlinearities are allowed to truncate the theoretical allocation, it is no longer literal quarter-Kelly. The drawdown result is therefore best read as a strong rejection of a broad absolute increase, not a precise recommended ceiling.

## 8. Exact engine replay of the strongest small rotation

To avoid rejecting the optimizer solely because its raw answer was extreme, the strongest stable direction was tested as a deliberately small, prespecified rotation:

- OLV liquid: **35 to 40 nominal bps**;
- OLV overflow: frozen at **25 nominal bps**;
- Weak Close Decent Sznls: **35 to 34.2499 nominal bps** as the risk-budget offset;
- pilots, OVS paths, overlays, and caps: unchanged.

This was the smallest liquid-supported implementation of the only nonpilot upweight that survived every global sensitivity. The production engine was rerun from candidates under both allocations; the proposal was not approximated by multiplying finished trade P&L.

| Metric | Baseline | Counterfactual | Change |
|---|---:|---:|---:|
| Engine rows | 4,778 | 4,776 | -2 |
| Total P&L, 2003–2026 | $4,017,313 | $4,030,772 | +$13,459 |
| Annualized P&L | $164,451 | $165,002 | **+$551** |
| Annualized volatility | $84,945 | $85,037 | +$93 |
| Sharpe | 1.9360 | 1.9404 | **+0.0044** |
| Sortino | 3.3277 | 3.3394 | +0.0117 |
| Maximum drawdown | -$91,099 | -$91,099 | $0 |
| Worst day | -$44,244 | -$43,777 | +$467 |
| Annual filled-risk fraction | 50.155% | 50.128% | -0.027 percentage point |

The sign is favorable, but the economic improvement is immaterial. The 24 calendar-year deltas are positive in 16 years and negative in 8, yet even dropping the best year leaves only **+$10,071 total** over the full history. The active-month paired statistic is about **t = 1.69** and is not enough to rescue a benefit of only $551 per year.

The exact replay also reveals nonlinear spillover: changing OLV liquid size altered two fills and slightly changed OLV overflow outcomes even though overflow bps were frozen. That is another reason not to infer implementable allocations from a linear Kelly table.

**Decision:** reject this rotation. A live change is not justified by a four-basis-point Sharpe increase and $551 of annual backtest P&L.

## 9. Why the null is the correct result

The evidence is internally consistent:

1. Every sleeve has positive estimated edge, so Kelly sees the entire current book as underbet in absolute terms.
2. In that low-risk region, relative optimization is dominated by small differences in estimated marginal edge and sends the risk budget to corners.
3. The most important actionable subproblem, OVS P1/P2, reverses direction across liquid and overflow support.
4. The only global upweight robust to every sensitivity is OLV, but a modest liquid-only implementation is economically negligible in exact replay.
5. Quarter- and half-Kelly fail the agreed drawdown constraint by a wide margin.

Changing bps would therefore replace a diversified, policy-shaped book with a statistically sharper-looking allocation whose advantage depends on selected history and disappears at practical size. The clean decision is to retain current ratios.

## 10. Limitations and interpretation boundaries

- **Strategy-selection lookahead:** every strategy definition is a survivor of prior research. Shrinking the mean cannot remove this selection bias.
- **Universe-selection lookahead:** the overflow universe is not reconstructed point in time, which is especially important for OVS and OLV tier comparisons.
- **Filled rather than staged risk budget:** the relative constraint uses filled engine risk. It is appropriate for return estimation but does not reproduce every operational staging-cap interaction.
- **Daily covariance:** daily vectors capture same-day clustering but not every overlapping-position state. The stationary block bootstrap partly retains serial dependence.
- **Large-scale nonlinearity:** raw 5x–50x multipliers are beyond the region where proportional rescaling matches caps, notional limits, signal depletion, or liquidity.
- **Empirical-Bayes model:** normal-normal shrinkage stabilizes the means but is not a structural return model. Prior-strength sensitivity was checked and did not cure the corner allocation.
- **LOYO is not out of sample:** leave-one-year-out minima are conservative reuse of the same history, not a prospective test.
- **Crisis sample is sparse:** pairwise crisis correlations can be noisy or unavailable for sleeves with little crisis overlap; the crisis Ledoit-Wolf estimate appropriately shrinks heavily.
- **Pilots are policy constraints:** 3x Bear, 3x Leader Gap, and Monthly Weak Close were fixed regardless of mathematical output. Their displayed portfolio multipliers are not unconstrained recommendations.
- **Drawdown is additive flat-NAV:** results answer the agreed research objective and should not be read as a compounded wealth-path forecast.

## 11. Reproducibility and verification

All calculations are confined to `scratch/` and `scratch/ultracode_research/`. Verification passed **25 of 25** checks, including signal/P&L reconciliation, exact baseline reproduction, budget equality, pilot locks, OVS sign reversal, drawdown gates, and engine-counterfactual reconciliation.

Primary artifacts:

- current replay builder: `scratch/kelly_build_current_replay_2026_08_05.py`;
- estimation and allocation: `scratch/kelly_estimate_allocate_2026_08_05.py`;
- exact counterfactual replay: `scratch/kelly_engine_replay_2026_08_05.py`;
- verification: `scratch/kelly_verify_outputs_2026_08_05.py`;
- component estimates: `scratch/kelly_estimates_2026-08-05.csv`;
- allocation diagnostics: `scratch/kelly_allocations_2026-08-05.csv`;
- scenario multipliers: `scratch/kelly_scenario_multipliers_2026-08-05.csv`;
- drawdown/growth curve: `scratch/kelly_growth_drawdown_curve_2026-08-05.csv` and `.png`;
- full and crisis correlations: `scratch/kelly_corr_full_2026-08-05.csv`, `scratch/kelly_corr_crisis_2026-08-05.csv`;
- exact engine-replay summary: `scratch/kelly_engine_replay_results_2026-08-05.json`;
- verification report: `scratch/kelly_verification_2026-08-05.json`.

### Research-process side effect

The first fresh replay called the production earnings loader, whose normal read-through behavior refreshed the stale local research caches from the configured R2 bucket:

- `data/earnings_calendar.parquet`;
- `data/earnings_calendar_overflow.parquet`.

No data was uploaded, no order or external financial action occurred, and no live code or configuration was changed. The refresh was an unintended local-cache side effect of using the current engine and should be kept in mind when comparing the workspace before and after this study.

## 12. Phase 3 conclusion

**No current per-strategy bps, OVS path bps, OVS Path-2 aggregate cap, overlay, daily cap, or pilot size should change as a result of this work.** Phase 4 records this null and pre-registers the evidence required to reopen the decision.
