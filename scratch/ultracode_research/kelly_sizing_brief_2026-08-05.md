# Kelly-Based Portfolio Sizing — Research Brief (2026-08-05)

Status: RESEARCH ONLY. Nothing in this study ships. Any proposed change goes
through the book's pre-registration discipline (see Guardrails) before a
single bps moves.

## Objective

Derive what a growth-optimal (Kelly) allocation across the 14-strategy book
would look like, compare it to the current judgment-tiered risk_bps
allocation, and produce a defensible answer to two questions:

1. **Relative allocation**: are the current per-strategy bps in roughly the
   right *ratios*? Which strategies are materially over- or under-sized
   relative to their (shrunk, correlation-aware) Kelly weight? OVS is the
   priority case: with N=2,429 it is the only strategy where the edge
   estimate is tight enough that Kelly arithmetic has real teeth.
2. **Absolute level**: where does the current book (GRM 1.5) sit on the
   Kelly spectrum, expressed as a fraction of full Kelly? The GRM replay
   already showed Sharpe is nearly flat across 1.0-1.75, which suggests we
   are deep in the linear regime, but nobody has computed the actual
   full-Kelly point or the growth/drawdown tradeoff curve.

The absolute level is ultimately a risk-appetite choice (that is what GRM
is for). The relative allocation is where a Kelly framework can genuinely
improve the book, so weight the effort there.

## Current sizing stack (read before touching math)

Per-signal sizing is per-trade risk in bps of a flat $750k account, NOT a
capital allocation. The pipeline (daily_scan steps 2b-5c, mirrored in
`pages/strat_backtester.py` steps 3b-3b5):

- base risk_bps per strategy (nominal, x GLOBAL_RISK_MULTIPLIER = 1.5 at
  import) -> fragility band mult -> signal-recency ladder (OLV) ->
  cycle-year mult (OVS 0.75x midterm) -> earnings size override ->
  ADV/notional caps -> same-day derate -> per-strategy 250 bps/day cap.

Current nominal base bps: OVS 40 (path1) / 8 (path2), OLV 35, 52wh Breakout
35, Weak Close Decent Sznls 35, LT Trend ST OS 30, St OS Sznl 40, 3x ETF
Overbot Fade 40, 3x Bear Fade 25, 3x Leader Gap Fade 25, Indices Oversold
Bounce 35, SPY QQQ MonFri Reversion 35, Sector BO 25, Monday Dip 30, ATR
Extended Gap Up 40, Monthly Weak Close 30. These were set by judgment and
sample-size conviction tiers, not by any optimization.

Ledger trade counts (data/backtest_trades_full.parquet, 4,694 rows):
OVS 2,429; MonFri 363; OLV 347; Indices OS 307; Weak Close 287; 52wh 254;
LT Trend 247; Monday Dip 115; Sector BO 75; ATR Ext 71; 3x Overbot 70;
St OS Sznl 57; 3x Bear 48; Leader Gap 24.

## Required reading (repo archaeology, phase 1)

Summarize each of these before doing any new math. The point is to not
re-derive what already exists and to inherit the negative results:

- CLAUDE.md sections: Sizing Conventions, Daily Risk Caps, Fragility Risk
  Bands, Ladder Sizing, plus each strategy's section for its overlays.
- `scratch/grm_replay_study.py` + `scratch/grm_replay_results.csv`: the
  book-level risk-appetite replay (Sharpe 1.89/1.87/1.85/1.83 at GRM
  1.0/1.25/1.5/1.75, maxDD -8.9% to -14.4%). This is the closest existing
  thing to a Kelly-curve probe; extend it rather than duplicating it.
- `scratch/cap_impact_study.py` + results: caps cost 25% of total return
  and 0.56 Sortino over 23y; the per-strategy 250 cap alone bounds the
  worst day. Any bps increase interacts with this cap.
- `scratch/ultracode_research/RISK_DIALS_2026-07-16.md` and
  `PORTFOLIO_RESEARCH_2026-07-02.md`: the graveyard of dial-conditioned
  sizing ideas and the pre-registration discipline that killed them.
- `pages/strat_backtester.py` `process_signals_fast`: the replay engine.
  `scripts/build_trade_ledger.py` shows how full-book passes are driven.
- Site payloads: `build_site.py` writes per-strategy daily MTM series
  (`strategy_daily.json` logic, flat $750k basis) and per-trade MTM
  vectors (`build_trade_mtm`). These give you daily return streams per
  strategy without running the engine.

## Why Kelly is not off-the-shelf here (the mapping problems)

Work through each of these explicitly in the math writeup. Hand-waving any
of them invalidates the result.

1. **The knob is per-trade risk, not a bankroll fraction.** Classic Kelly
   sizes a fraction of wealth per bet. Our decision variable is risk_bps
   per signal, strategies fire at wildly different frequencies (OVS ~100
   trades/yr vs Leader Gap ~1/yr), and multiple positions overlap. The
   clean reformulation: treat each strategy's DAILY flat-basis PnL stream
   (at current sizing) as a return stream, and solve for a vector of
   scalar multipliers m_i on current bps. Continuous-time multi-asset
   Kelly then gives m* proportional to Sigma^{-1} mu on those streams.
   Frequency, overlap, holding periods, and every live overlay are then
   automatically embedded in the streams.
2. **R is a sizing unit, not a loss bound.** Several strategies have no
   stop at all (OLV vol-confirm, Leader Gap, Monthly Weak Close, OVS) and
   stops gap through (worst booked trade -4.56R). Kelly needs the actual
   loss distribution, which the ledger provides empirically. Do NOT use
   win-rate/payoff-ratio Kelly formulas; use expected-log maximization on
   the empirical (or bootstrapped) distribution, and check the Gaussian
   mu/sigma^2 approximation against it. Report where they diverge (the
   short-vol strategies, presumably).
3. **Correlation is the whole game for the dip-buy cluster.** FAMILY4 plus
   Monday Dip plus Indices OS plus Monthly Weak Close all buy the same
   selloffs. Per-strategy standalone Kelly would badly overbet the
   cluster. The Sigma^{-1} mu form handles this, but Sigma estimated on
   daily overlap understates tail co-movement, so also compute
   crisis-window correlations (2020-03, 2022, 2024-08, 2025-04) and show
   the allocation under both.
4. **Estimation error dominates at small N.** Full Kelly with a noisy mu
   overbets with near certainty. Two required mitigations: (a) shrink each
   strategy's mean toward a common prior (empirical-Bayes across the book,
   shrinkage strength set by per-strategy effective N), and (b) report
   everything at fractional Kelly (1/2, 1/4) rather than the full-Kelly
   vertex. Effective N must be episode-clustered, not raw trade count:
   OVS's 2,429 trades cluster into far fewer vol episodes, and the 3x
   fades are almost entirely episode-driven. Reuse the episode-clustering
   conventions from the existing validation studies.
5. **The ledger is in-sample and survivorship-flattered.** Every strategy
   was designed on this data, so ledger avgR is an upper bound on true
   edge. The overflow tier is additionally survivorship-biased (21 of 22
   major 2020s delistings absent) and must not drive sizing conclusions
   (CLAUDE.md caveat, standing rule). Mitigations: LOYO means as the
   conservative mu estimate, and run the whole optimization on 2018+ only
   as a robustness pass.
6. **Caps and overlays make the response nonlinear.** Doubling OVS bps
   does not double OVS PnL: the 250 bps/day cap binds on cluster days,
   path-2 has its own aggregate cap, and derates/bands compose. So the
   analytic m* from stream algebra is a first-order answer only. Validate
   any proposed m vector with an actual engine replay (grm_replay_study
   pattern, per-strategy multipliers instead of one global scalar).
7. **OVS internal structure.** OVS is really two sub-strategies (path1
   decisive at 60 eff bps, path2 mild at 12 eff, N=407, avgR +0.20) plus
   scale-outs (accepted -R variance smoothing) plus the midterm 0.75x. A
   Kelly treatment of "OVS" should at minimum separate path1 and path2,
   since their edges and sizes already differ 5x.

## The math to work through (phase 2 deliverable)

A self-contained writeup, notation consistent throughout:

- Discrete Kelly for a single strategy with an empirical R distribution:
  f* = argmax E[log(1 + f R)], existence/uniqueness, behavior with fat
  left tails, and why f is bounded by the worst-loss constraint.
- The continuous approximation f* ~ mu/sigma^2, its error for skewed
  distributions, and the per-trade -> per-day frequency mapping (growth
  rate g(f) = sum over strategies of lambda_i * E[log(1+f_i R_i)] under
  independence, and why independence fails here).
- Multi-strategy correlated Kelly: m* = Sigma^{-1} mu on daily streams,
  fractional Kelly as both a shrinkage and a drawdown-control device, and
  the known result that half-Kelly gives ~75% of the growth at far less
  variance. Cover drawdown-constrained Kelly explicitly: the relationship
  between the Kelly fraction and drawdown distribution (for log-normal
  approximation, P(DD > d) ~ d^(2/f - 1) style results), since McKinley's
  binding constraint in the GRM decision was maxDD, not variance.
- Parameter uncertainty: Bayesian Kelly / shrunk-mean Kelly, and the
  N-dependent conviction tiers this implies (a principled version of what
  the current 25/30/35/40 bps tiers do by feel).

## Estimation and computation plan (phase 3)

1. Build per-strategy daily PnL streams on the flat $750k basis from the
   ledger MTM machinery (get_daily_mtm_series conventions; split OVS into
   path1/path2 if feasible from the ledger's Risk bps / Size_Mult columns).
   State the vintage of the ledger used.
2. Estimate mu (three ways: full-sample, LOYO-conservative, 2018+) and
   Sigma (full-sample daily, plus crisis-window). Shrink means via
   empirical Bayes with episode-clustered effective N.
3. Solve for m* = Sigma^{-1} mu multipliers; also solve the discrete
   expected-log problem per strategy on empirical R distributions as a
   cross-check. Normalize so the book's total risk matches current, and
   report the RELATIVE allocation first (m_i / m_book).
4. Compute the growth-vs-fraction curve g(c * m*) for c in [0, 1.5] and
   locate current GRM-1.5 sizing on it. Bootstrap (stationary block,
   reuse the Monte Carlo tab's block conventions, mean block 10td) the
   maxDD distribution at each c.
5. Sensitivity: LOYO on the final allocation, shrinkage-strength sweep,
   with/without crisis Sigma, with/without overflow tier, path1/path2
   split vs pooled OVS.
6. Engine replay of the single most promising discrete proposal (e.g.
   "OVS path1 to X bps, MonFri to Y, Leader Gap unchanged") with caps
   live, grm_replay_study pattern, to confirm the stream-algebra answer
   survives the nonlinearities.

## Deliverables

1. Phase 1 memo: current-framework summary + inherited negative results
   (short; mostly confirms the brief's own summary and flags anything the
   brief got wrong).
2. Phase 2: the math writeup.
3. Phase 3: results — a table per strategy: current eff bps, ledger avgR,
   episode-adjusted effective N, shrunk mu, standalone discrete Kelly
   fraction, correlated m*, and the ratio current/quarter-Kelly. Plus the
   growth/drawdown curve with GRM 1.5 marked on it.
4. A recommendation memo written as a PRE-REGISTRATION DRAFT: the specific
   proposed changes (if any), the decision gates that would have to pass
   before shipping (engine replay deltas, LOYO floors, clustered t), and
   the explicit null ("current allocation is inside the noise band of
   fractional Kelly; change nothing") treated as a fully acceptable
   outcome.

## Guardrails

- Do not modify strategy_config.py, daily_scan.py, the engine, or any
  live path. All work in scratch/ (scripts) and this directory (docs).
- Do not tune on overflow-tier stats alone (survivorship caveat).
- Respect the fragility freeze policy: no re-scoring of dial weights, no
  new dial-conditioned sizing (that family of ideas is dead, see
  RISK_DIALS doc). Kelly multipliers here are static per-strategy
  scalars, not regime-conditioned.
- Flat $750k basis for all PnL math (per-trade dollars are additive
  there; the compounded basis cannot be decomposed per strategy).
- Report expectancy-costing findings honestly. Several live overlays
  (OLV recency ladder, gap derate, scale-outs) are deliberate appetite
  cuts that cost EV; the Kelly answer will say "remove them." Note that,
  but the decision to hold appetite cuts is McKinley's and out of scope.
- Cite every number to a script + output file so results are reproducible.

## Open questions to resolve with McKinley before phase 3 locks

1. Objective: pure log growth, or log growth subject to a hard drawdown
   constraint (e.g. bootstrap P(maxDD > 20%) < 5%)? The GRM decision
   history suggests the latter.
2. Is the OVS path1/path2 split in scope for a sizing recommendation, or
   analysis-only?
3. Should pilot strategies (Leader Gap, Monthly Weak Close, 3x Bear) be
   frozen at current size regardless of what the math says (pilot
   convention), with Kelly applied only to the seasoned strategies?
4. Fraction convention for the headline recommendation: half-Kelly or
   quarter-Kelly?
