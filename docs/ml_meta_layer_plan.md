# ML Meta-Labeling Layer — Plan (v4, FINAL)

**Status:** Finalized after 4 rubric iterations (score trajectory 64 → 83 → 91 → 95; plateau).
**Prime directive:** Strictly additive. No existing file in this repo is modified. The rule-based
strategy book remains the primary signal generator; ML is a secondary, advisory layer.

---

## 1. Objective

Improve per-trade expectancy of the existing 12-strategy book by training a **meta-model** that
answers one question for every signal the scanner stages:

> *Given everything knowable at signal time, what is the probability this trade is a winner —
> and should it be taken at full size, trimmed, or skipped?*

This is meta-labeling (López de Prado): the existing strategies decide *what and when*; the ML
layer scores *how good this particular instance is*. It never generates trades, never touches
entries/stops/targets, and never upsizes above the book's configured risk.

**Why this application (and not alternatives considered):**
- *Price forecasting / new signal generation* — replaces rather than augments the process; rejected.
- *Portfolio optimization* — the book already has deliberate sizing conventions (bps risk,
  overlap clamps, 2-path OVS); rejected as invasive.
- *Regime classification only* — already partially covered by risk_dashboard_v2 fragility; a
  per-trade meta-model subsumes it by taking regime variables as features.
- *Meta-labeling* — additive by construction, directly measurable against the ledger baseline,
  and well-matched to the data available (3,362 labeled historical trades).

## 2. Training data

**Source:** `data/backtest_trades_full.parquet` (built nightly by `scripts/build_trade_ledger.py`).
3,362 trades, Signal Dates 2003-01-17 → present, 12 strategies × {Liquid, Overflow} tiers.

**Label:**
- `y_R` = `PnL_flat_750k / Risk_flat_750k` (risk-normalized return; falls back to `R_Multiple`
  where Risk is 0/NaN — 6 rows). Defined for every trade including no-stop/time-exit strategies.
- `y_win` = `y_R > 0` (primary classification target). Train-time winsorization of `y_R` at
  [-3, +8] for any regression diagnostics.

**Known biases — disclosed, with mitigations:**
1. The ledger is *modeled* execution (limit fills, stop-arming conventions), not live fills.
   The model inherits this; uplift estimates are relative to the same modeled baseline, which
   nets out most of the shared error.
2. `sznl_ranks.csv` / `atr_seasonal_ranks.parquet` construction may embed full-sample
   information (they are inputs the existing system already trusts). Mitigation: a **sensitivity
   run** trains a second model with all seasonal features removed and reports both; if uplift
   collapses without seasonal features, that is flagged in the report.
3. The strategies themselves were developed on this history (selection bias in the baseline).
   The ML layer measures *relative* uplift on walk-forward out-of-sample folds, which is the
   honest quantity available.
4. Uneven strategy counts (OVS ≈ 1,100 trades; St OS Sznl = 58). Mitigation: one pooled model
   with `Strategy` as a categorical feature; **per-strategy deployment gates** (§7) so thin
   strategies default to pass-through if their OOS evidence is weak.

## 3. Features — point-in-time discipline

All features are computable strictly from data available at the signal date close. Two sources,
one code path:

**A. Ticker-level** — computed by importing the existing `indicators.calculate_indicators()`
(read-only import; identical code path the backtester/scanner uses, guaranteeing train/inference
parity):
- Percentile ranks of returns: `rank_ret_{2,5,10,21,126,252}d`
- ATR-normalized rank: `rank_ret_atr_{5,21}d`
- `ATR_Pct`, `today_return_atr`, `range_in_atr`
- Derived in `ml/features.py` (cheap, additive): close-vs-SMA50/200 in ATR units, SMA50>SMA200
  flag, % off 52-week high, close-in-day's-range %, 21d dollar-volume rank
- Seasonal: `seasonal_rank` (sznl_ranks.csv), `atr_sznl_{5,21,63,252}d` (atr_seasonal_ranks.parquet)

**B. Market context** — built once from `data/master_prices.parquet` (verified to contain ^VIX,
^VIX3M, ^GSPC, all 11 sector ETFs, HYG/IEF):
- `vix_close`, `vix_5d_chg`, `vix_term` (VIX/VIX3M)
- `spx_vs_sma200_pct`, `spx_ret_21d_rank`, `spx_rv_21d` (realized vol)
- `breadth_pct_above_200d` (11 SPDR sectors)
- `hyg_ief_z63` (credit ratio z-score, 63d)

**C. Trade/meta** — `Strategy` (categorical), `Direction`, `Tier`, `hold_days_target`,
`stop_atr` (0 when no stop), `tgt_atr`, day-of-week, month.

~32 features total. **No-lookahead guarantee is unit-tested:** features for date *t* must be
bit-identical when all data after *t* is truncated (`tests/test_ml_no_lookahead.py`).

## 4. Model

- `sklearn.ensemble.HistGradientBoostingClassifier` — gradient-boosted trees, native categorical
  support, NaN-tolerant, already satisfiable by the repo's existing sklearn dependency.
  **Zero new dependencies** (no lightgbm/xgboost; nothing added to requirements.txt).
- Probability calibration: isotonic regression fit on out-of-fold train predictions.
- **Pre-registered hyperparameters** (no test-set tuning): `max_iter=300, learning_rate=0.05,
  max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, early_stopping on a 15%
  validation split of train, random_state=7`. A deliberately small model for N≈3.4k.
- Seed policy: all randomness seeded (7); artifacts reproducible from the same ledger.

## 5. Validation protocol — purged, embargoed walk-forward

- Expanding-window walk-forward by calendar year: first test year **2012** (≥9y initial train),
  then 2013 … 2026. Train always strictly precedes test.
- **Purge:** any train trade whose `Exit Date` ≥ test-window start is dropped (its label was
  resolved by prices inside the test period — label leakage otherwise; matters for the two 63d
  breakout strategies straddling year-ends).
- **Embargo:** additionally drop train trades exiting within 5 trading days before test start.
- Threshold selection (§6) and calibration use **training folds only**; test folds are scored once.
- Final-report hygiene: the walk-forward harness is run once with pre-registered settings. Any
  re-run after design changes is reported as such in the report header (run counter in metadata).

## 6. Decision policy (pre-registered)

Calibrated `p = P(win)` maps to a size multiplier — **never above 1.0** (an advisory layer must
not add risk):

| Condition | Decision | Multiplier |
|---|---|---|
| p < q20 of train-fold p-distribution | SKIP | 0.0 |
| q20 ≤ p < q50 | TRIM | 0.5 |
| p ≥ q50 | FULL | 1.0 |

Quantile thresholds are recomputed per fold from train predictions only, so the policy adapts
without touching test data. Defaults are quantile-based (not absolute p cutoffs) to stay robust
to base-rate drift across eras.

## 7. Evaluation — baseline, metrics, pre-registered success criteria

**Baseline:** all ledger trades at book sizing on the flat-$750k basis (exactly what
`build_trade_ledger.py` produces). **ML book:** same trades × multiplier from §6, computed only
on OOS test folds (2012–2026 pooled).

**Metrics (baseline vs ML, plus the SKIP bucket alone):** trade count & retention %, mean/median
R per trade, win rate, profit factor, total R, Sharpe of monthly aggregated R, max drawdown in R,
mean R of skipped trades (the single most direct evidence — skipped trades should underperform
taken ones), OOS Brier score vs constant-base-rate Brier, reliability (calibration) table,
mean R by p-decile, per-strategy and per-year breakdowns.

**Pre-registered ship/no-ship criteria** (all four must hold before the multiplier is ever used
for real sizing; otherwise the layer stays advisory-only):
1. Pooled OOS uplift in mean R per trade ≥ **+0.05R**, with bootstrap 95% CI of the uplift
   excluding 0 (10,000 resamples, trade-level).
2. OOS Brier < constant-base-rate Brier (the model has real skill, not just the base rate).
3. Retention ≥ 60% of trades overall, and every strategy retains ≥ 40% of its trades — any
   strategy failing this is excluded from gating (pass-through) rather than strangled.
4. Uplift positive in ≥ 8 of the ~14 test years (not concentrated in one regime).

**Failure is a valid result:** if criteria fail, the report says so plainly and the deliverable
remains an advisory score column with documented limitations.

## 8. Operations

- **Artifacts:** `data/ml/` (already inside gitignored `data/` — no .gitignore edit):
  `data/ml/dataset.parquet`, `data/ml/models/meta_model_<train-through-date>.joblib`,
  `data/ml/models/metadata_<date>.json` (feature list+hash, sklearn version, fold scores, run
  counter), `data/ml/reports/` (evaluation report .md + CSVs), `data/ml/scores/`.
- **Daily advisory scoring:** `python -m ml.score_daily` runs after the scan; reads
  `seasonal_screener_results.csv` (or `--input <csv>`) read-only, builds features from local/R2
  caches via the same code path, writes `data/ml/scores/ml_scores_<date>.csv` with
  `Ticker, Strategy, Date, p_win, decision, size_multiplier, drift_flags` and prints a console
  table. **Fallback:** any failure (missing model, stale caches) degrades to multiplier 1.0 with
  `model_unavailable` flagged, exit code 0 — the existing pipeline can never be blocked.
- **Retraining:** monthly or ad-hoc — `python -m ml.train` (rebuilds dataset from the current
  ledger, refits, writes a new dated artifact; old artifacts retained).
- **Monitoring:** `ml/monitor.py` — PSI per feature vs train distribution (warn > 0.2) and
  rolling-window calibration drift; flags surface in daily score output.
  **Kill criterion:** if trailing-12-month realized mean R of SKIP-bucket trades exceeds that of
  FULL-bucket trades, revert to pass-through and retrain before re-enabling.
- **Optional GHA:** `.github/workflows/ml_score.yml` with `on: workflow_dispatch` **only** (no
  cron) — inert until manually triggered, so repo behavior is unchanged. Wiring the multiplier
  into `order_staging.py` (separate repo, IBKR-bound) is explicitly **Phase 3, user-gated**.

## 9. Phasing

- **Phase 1 (this build):** dataset builder, model training, purged walk-forward evaluation,
  full report. Pure offline.
- **Phase 2 (this build):** daily advisory scoring CLI + tests + monitoring + inert GHA workflow.
- **Phase 3 (later, requires explicit user decision):** consume `size_multiplier` in
  `order_staging.py`, scheduled automation, Sheets `ML_Scores` tab, live-fill feedback loop,
  possible upsizing once live evidence accumulates.

## 10. New files (complete inventory — nothing else is touched)

```
ml/__init__.py              ml/config.py            ml/market_features.py
ml/features.py              ml/dataset.py           ml/cv.py
ml/modeling.py              ml/train.py             ml/evaluate.py
ml/score_daily.py           ml/monitor.py           ml/README.md
tests/test_ml_no_lookahead.py   tests/test_ml_cv.py   tests/test_ml_dataset.py
docs/ml_meta_layer_plan.md  (this file)
.github/workflows/ml_score.yml  (workflow_dispatch only — inert)
```

Existing modules are imported read-only: `indicators.py` (feature parity),
`data_provider.py`/`cache_io.py` (price access), `strategy_config.py` (strategy metadata).
Per CLAUDE.md module-boundary rules, nothing here imports from or into `risk_dashboard_v2.py`.

---

## Appendix A — Scoring rubric

| # | Criterion | Weight |
|---|---|---|
| 1 | **Grounding** — references actual repo assets (files, schemas, strategies), not generic ML | 10 |
| 2 | **Additivity** — zero modifications to existing files; explicit new-file inventory | 10 |
| 3 | **Statistical rigor** — leakage prevention (point-in-time, purge/embargo), overfitting control for N≈3.4k, pre-registered hyperparameters | 15 |
| 4 | **Label/objective alignment** — labels match the actual decision (take/trim/skip at signal time), valid for no-stop strategies | 10 |
| 5 | **Decision value** — explicit output→action mapping with economic uplift measurement | 10 |
| 6 | **Feasibility** — runs on local data/compute, no new dependencies, GHA-compatible | 10 |
| 7 | **Evaluation honesty & falsifiability** — pre-registered ship criteria, defined baseline, no-edge is a reportable outcome | 10 |
| 8 | **Operational integration** — daily flow, retrain cadence, monitoring, safe fallback | 10 |
| 9 | **Interpretability & trust** — calibration tables, importances, per-strategy diagnostics | 7.5 |
| 10 | **Phasing & scope discipline** — MVP-first, deferred items explicit, user-gated deployment | 7.5 |
| | **Total** | **100** |

## Appendix B — Iteration log

**v1 — score 64/100.** Sketch: binary R>0 classifier on ledger trades, generic walk-forward,
filter-only output, edits to requirements.txt/.gitignore contemplated.
Critique: label undefined for no-stop strategies; no purge/embargo spec (63d holds straddle fold
boundaries → label leakage); no calibration; no size mapping or retention floor; no pre-registered
success criteria; proposed dependency/file changes violating additivity; no fallback behavior.

**v2 — score 83/100.** Fixed: universal label `PnL_flat/Risk_flat`; purged+embargoed expanding
walk-forward; pooled model + Strategy categorical; isotonic calibration; skip/trim/full policy
capped at 1.0×; sklearn-only (no requirements.txt edit); artifacts under gitignored `data/ml/`
(no .gitignore edit); monthly retrain; pass-through fallback.
Critique: train/inference feature-parity risk (two code paths) — must reuse
`indicators.calculate_indicators`; seasonal-rank lookahead inheritance undisclosed; no
no-lookahead unit test; phasing mushy; baseline imprecisely defined; threshold selection
protocol could touch test data; interpretability thin.

**v3 — score 91/100.** Fixed: single shared feature code path + parity guarantee; bias
disclosure section with seasonal-sensitivity run; truncation-based no-lookahead test specified;
three phases with Phase 3 user-gated; baseline pinned to flat-$750k ledger; per-fold train-only
quantile thresholds; calibration/reliability + p-decile R tables added.
Critique: embargo length unjustified; hyperparameter protocol unstated (tuning = silent test-set
mining risk); no kill criterion for live drift; GHA workflow would auto-run on schedule
(behavior change — violates additivity in spirit); empty-scan-day handling unspecified.

**v4 — score 95/100.** Fixed: purge rule pinned to Exit-Date-vs-test-start + 5td embargo with
rationale; hyperparameters pre-registered, no tuning; PSI + calibration-drift monitoring with
explicit kill criterion; GHA workflow made `workflow_dispatch`-only (inert); graceful empty-day
and missing-model behavior; run-counter honesty mechanism in metadata.
Residual (accepted): permutation importances on OOS folds rather than SHAP (no new deps);
live-fill feedback loop deferred to Phase 3. Estimated gain from another iteration ≤ 1 point →
**plateau; v4 is final.**

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 7 | 8 | 8 | 6 | 6 | 8 | 5 | 5 | 5 | 6 | **64** |
| v2 | 9 | 10 | 12 | 8 | 8 | 9 | 8 | 7 | 6 | 6 | **83** |
| v3 | 9 | 10 | 14 | 9 | 9 | 9 | 9 | 8 | 7 | 7 | **91** |
| v4 | 9 | 10 | 14 | 9 | 9 | 10 | 10 | 9 | 7.5 | 7.5 | **95** |

## Appendix C — Execution outcome (2026-06-10)

Built and run as specified. All 13 tests pass, including the empirical
no-lookahead truncation tests on real repo data.

- **Run-1** (pre-registered): NO SHIP. c1 uplift FAIL (-0.01R), c2 Brier skill
  PASS, c3 retention PASS (88.7%), c4 year-consistency FAIL (7/15).
- **Run-2** (after a market-feature ffill plumbing fix — rolling-window NaN
  holes; design unchanged): identical verdict. The result is robust.

**Interpretation.** The model genuinely predicts win probability (realized win
rate climbs monotonically 44.7% -> 70.3% across calibrated p-deciles, Brier
beats base rate), but mean R is nearly FLAT across those deciles: in this book,
lower win probability is compensated by larger winners (8-ATR-target breakouts
are the extreme case — the SKIP bucket averaged +0.60R at a 51% win rate). The
strategies' own filters have already extracted the conditioning edge that the
observable features carry; win-probability gating therefore trims tails without
improving expectancy. Per the pre-registered criteria the layer is deployed
**advisory-only**: `p_win` ships as an informational column, `size_multiplier`
must not be wired into order staging (Phase 3 blocked at the c1/c4 gates).

Possible future runs (each would be a numbered, disclosed re-run): expected-R
objective instead of P(win) — though the flat decile-R table suggests little
headroom; per-strategy stop-hit probability as a *risk* (not expectancy) signal;
adding features orthogonal to the strategies' own filters (positioning,
liquidity, cross-asset states); live-fill labels once enough accumulate.

### Run-3 (2026-06-10) — orthogonal features

Added 8 features the strategy filters do not condition on (ml/ortho_features.py):
CBOE equity put/call level + 63d z (2024+ only — thin, disclosed), NAAIM level +
52w z (2006+, 86% coverage), analyst-grade momentum (net upgrades 21d, activity
63d; 2012+, 50% coverage), and earnings distance (days since/to, 71-76%
coverage). Evaluation design and ship criteria unchanged.

**Verdict: NO SHIP — identical failure mode.** Uplift -0.015R (bootstrap CI
[-0.047, +0.017], contains zero), 7/15 positive years, SKIP bucket still +0.60R
at a 47% win rate. Brier 0.2387 vs base-rate 0.2402 (modest real skill,
unchanged). Conclusion upgraded to high confidence: **per-trade expectancy in
this book is not gateable by P(win) on any feature set tried — the payoff
asymmetry is structural, not an artifact of feature overlap with the entry
rules.** The expectancy-gating line of inquiry is closed; remaining credible
directions are risk-targeted labels (adverse excursion) and live-fill labels.

### Adverse-excursion risk model (2026-06-10) — research-only

Re-targeted the harness at tail risk: P(MAE >= 1R) from entry-to-exit price
paths (ml/excursion.py, base rate 34.5%). Walk-forward OOS on 2,449 trades:
**AUC 0.606, Brier-skilled, well calibrated** — predicted-risk deciles span
realized MAE>=1R rates 18% -> 52% and stop-exit rates 1% -> 45%, while mean
trade R stays flat across deciles (the signal is orthogonal to expectancy —
the right shape for risk management).

Caveat that matters: pooled AUC is substantially strategy mix. Within-strategy
AUC: OVS 0.582 (n=877, real), LT Trend ST OS 0.572, MonFri 0.540, Weak Close
0.537 — but 52wh 0.494, OLV 0.506, Indices OSB 0.482 (nothing). Practical
implication: modest incremental tail-risk signal exists for OVS specifically,
where it could inform the EOD-DD valve / stop-arming conventions — but ONLY
via a dedicated policy backtest (e.g. re-run the 81-episode stop-arming study
conditioned on predicted risk). No policy change is made here.

### Run-4 (2026-06-10) — risk-dial fragility scores

Added frag_5d / frag_21d / frag_63d from `data/rd2_fragility.parquet`
(risk_dashboard_v2 composite, daily 2016-06+, ~100% coverage of the 1,921
trades since; reconstruction caveat: historical values computed by current
dashboard code with in-window percentile bands — inherited bias, disclosed).

**Findings (permutation importance on OOS folds 2018+, both targets):**
- Expectancy model: fragility is marginal — frag_63d ranks 16/49, frag_21d
  24/49, frag_5d noise. Verdict unchanged (NO SHIP, identical failure mode).
- Excursion model: **frag_63d ranks 6/49** — a genuinely used tail-risk input
  (behind pct_off_52w_high, rank_ret_atr_21d, spx_rv_21d, atr_sznl_252d,
  hyg_ief_z63). frag_5d/21d add nothing (too twitchy). Pooled AUC however is
  flat (0.599 vs 0.606 run-3): the 63d fragility information largely overlaps
  the vol/credit state features already present — the model substitutes
  rather than gains.

Net: fragility features RETAINED (model uses frag_63d; daily scoring now
carries current regime state; no headline degradation), but they do not
change any verdict. The 5d/21d windows are kept for completeness at zero
marginal cost; candidates for pruning in a future cleanup run.
