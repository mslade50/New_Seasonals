# Pre-registration: prior-range skip filter for the NQ/ES opening breakout

Date 2026-09-25. Author McKinley Slade (drafted by Claude). Status: APPROVED WITH OWNER OVERRIDE, 2026-09-25 afternoon. McKinley reviewed the draft and decided to ship the primary rule (skip at ratio >= 1.25) in the one-contract LIVE pilot from 2026-09-28, ahead of the forward tier, and to run the RTY cross-validation now. The forward tier still runs unchanged as a post-ship review: the shadow session stays unfiltered and remains the source of skipped-day outcomes, and a forward FAIL is grounds to turn the filter back off. The rest of this document is as drafted before that decision. RTY results: `artifacts/research/qqq_open_breakout_20260923/range_filter_rty/RESULTS.md`.

Candidate: `artifacts/research/qqq_open_breakout_20260923/current_candidate/` (break-even OFF, decision 2026-09-24). Live service: `open_breakout/`, runbook `docs/open_breakout_runbook.md`.

Repo rule (CLAUDE.md, "Pre-registration"): any new dial-conditioned control needs a written prereg (gates, decision rule, sensitivity) before the study runs. This filter is conditioned on a volatility state rather than the fragility dial, and it changes what trades and at what size, so it is held to the same rule. This file must be committed before any forward data is examined for this question and before the RTY run.

## Motivation (post hoc, stated plainly)

On 2026-09-25 a descriptive pass over the frozen candidate's trades (scratch `range_cuts/`, NQ and ES, 2018-01-02 to 2026-08-28) split sessions by the prior full-session true range divided by its 20-session average. The top quintile (ratio above about 1.28) was the weak cell: mean R per trade 0.03 on NQ and -0.06 on ES at base costs, against 0.20 to 0.32 (NQ) and 0.04 to 0.22 (ES) in the other four quintiles. The day-level top-vs-rest difference in summed R had 95% bootstrap intervals excluding zero in three of four market/cost cases, and the gap was wider in 2023-26 than in 2018-22. None of this was pre-specified. The same pass looked at 5 quintiles x 3 sides (all, long, short) x 2 markets x 2 cost cases (60 cells), a second bucketing by trigger distance in ATR units (48 cells, correlation 0.997 with the ratio, so a restatement of the same variable), and 9 trigger/stop variants per market. The top-quintile result is the most striking of roughly 120 looks and should be discounted accordingly. An older study in the same folder (`simple_filters`, NQ only, break-even ON engine) tested the opposite rule, trading only when the prior range was above its 20-session mean. That rule kept 730 of 2,303 trades, cut mean R from 0.17 to 0.08 and Sharpe from 1.63 to 0.57, and was rejected. It is often remembered as "the range filter hurt", but read as a partition it points the same way as today's observation: the removed below-average days averaged about 0.22R and the kept above-average days 0.08R. It is not independent evidence (same NQ sample, same variable), so it neither confirms nor refutes this hypothesis. All NQ and ES history in hand was used in discovery.

## Hypothesis

Variable: `ratio = prior_tr / atr20`, where
- `prior_tr` is the true range of the previous full CME session (18:00 to 17:00 ET), `max(high - low, |high - prev_close|, |low - prev_close|)`, from raw same-contract minute OHLC, exactly as `open_breakout/inputs.py:build_manifest` computes it for the session manifest;
- `atr20` is the mean of the 20 most recent valid full-session TRs strictly before the previous session (the previous session itself is excluded), computed point-in-time at the previous session close.

Roll handling follows the research: in `engine.make_sessions`, a session whose continuous-contract instrument differs from the prior session's, or that spans more than one instrument, has TR set to missing and is skipped when collecting the 20 valid TRs. The forward tier computes each TR same-contract (as the service does), so roll sessions produce a valid same-contract TR instead of a missing one. This is a known, declared difference; see Invalidation.

H1: on market-days with `ratio >= 1.25`, the frozen candidate's expectancy is materially below its expectancy on other days, low enough that skipping those days does not reduce, and likely raises, the strategy's risk-adjusted return.

H0: expectancy on `ratio >= 1.25` days is not materially lower; operationally, mean day-level R on those days is at or above +0.10R.

## Pre-registration (frozen; do not alter)

Primary rule: SKIP both sides for that market on any session with `ratio >= 1.25`. Applied per market (NQ and ES are classified separately). Nothing else about the candidate changes.

Why 1.25 and not the fitted 1.28: the in-sample quintile edge moved with the case (NQ base 1.28, ES base 1.29, ES stress 1.20), so its second decimal carries no information. 1.25 is a round number inside that range, chosen before any out-of-sample data is seen, and it is not tuned to a result. In sample it flags 21.4% of NQ traded days and 22.1% of ES traded days.

Pre-declared alternates, evaluated with the same metric and decision rule:
- A1: SKIP at `ratio >= 1.5`.
- A2: HALF-SIZE (not skip) at `ratio >= 1.25`. For whole-contract sizing, half the computed contracts, floored; a result of 0 contracts means no trade.

No other threshold, side split, cost case or combination may be reported as confirmatory. Nothing ships unless the primary passes. If the primary passes, the owner may choose the primary, A1 or A2 for implementation and records the choice in writing; an alternate that passes while the primary fails ships nothing.

## Out-of-sample plan

All NQ and ES minute history in the repo (2017-01 onward, discovery window 2018-01-02 to 2026-08-28) was used in discovery. There is no clean NQ/ES holdout. Three tiers, in order of weight.

(a) RTY, untouched for this question. RTY minute data (`es_rty/rty_1m_snapshot.parquet`, 2017-07-09 to 2026-08-31) has the columns the engine reads (`ts, open, high, low, close, volume, instrument_id`, the same schema as the ES file), and the short gate (`risk20/daily_gate.csv`, date-level, 2011-01-03 to 2026-09-23) covers it. Honest caveat: RTY is not untouched for the strategy. The `es_rty` study ran the older break-even ON engine on RTY and found almost no edge (mean R 0.03 base, -0.10 stress). No range-conditioned result has been computed on RTY; today's check counted RTY ratio frequencies only (below). The current break-even OFF engine has not been run on RTY. For RTY to count, all must hold:
1. Run `current_candidate/engine.py` unchanged with RTY settings (tick 0.10, multiplier 5, base slip 1 tick, stress 4 ticks, fee 0.475, same short gate, same window), after first confirming the same harness reproduces the frozen ES ledger exactly.
2. The RTY session audit matches `es_rty/RTY_session_audit.csv` (same excluded roll and incomplete sessions).
3. At least 150 RTY traded days with `ratio >= 1.25`.
The unfiltered RTY result is reported first, whatever it is. Because the claim is relative (qualifying days versus the rest), a weak RTY baseline does not disqualify the test, but it is stated beside the result.

(b) Forward tracking, the deciding tier. From 2026-09-28 the `open_breakout` service runs a shadow session (full unfiltered candidate, client 927480) and a one-contract live pilot (client 927481). Each session manifest records `prior_tr` per market. The manifest does not record `atr20`; it is computed by a separate read-only script that applies the service's `session_range` function to the 20 preceding valid sessions from IB same-contract history. Skipped-day outcomes come from the shadow journal (primary); live pilot fills are a cross-check on the shadow's fills. A qualifying day is a market-session with `ratio >= 1.25` on which the unfiltered shadow made at least one entry. Qualifying sessions with no entry are counted and reported but carry no R.

Required sample: 40 qualifying market-days across NQ and ES. Expected pace from the discovery sample: 525 of 2,059 NQ sessions (25.5%) and 530 of 2,059 ES sessions (25.7%) had `ratio >= 1.25`, and qualifying days that also traded ran at 296 (NQ) plus 290 (ES) over 2,059 sessions, about 0.285 per session across both markets. 40 qualifying market-days therefore needs about 140 sessions, roughly 6.5 to 7 months: a decision near mid-April 2027 if the service runs every session from 2026-09-28. The one allowed extension of 20 more adds about 70 sessions (to about mid-July 2027). The markets cluster: in 2023-26, 211 NQ and 217 ES qualifying sessions fell on 258 distinct dates, 170 of them shared, so 40 market-days is about 25 independent dates. The bootstrap resamples by date for this reason.

Power, stated before the fact: in the discovery sample the standard deviation of day-level summed R on qualifying days was about 1.32R (NQ and ES alike). The 90% interval half-width at n=40 is therefore about 0.34R (0.28R at n=60). A PASS needs an observed mean near -0.24R or lower at n=40 (about -0.18R at n=60). The in-sample qualifying-day mean was close to zero, not strongly negative, so if the in-sample effect is real but no larger, the likely outcome is INCONCLUSIVE, and nothing ships. The owner accepts this: the rule is designed to ship only on a clear forward result.

Sessions where the shadow halted, failed or never ran (for example 2026-09-25, Gateway outage) are excluded from the primary. A secondary count adds those sessions replayed with the research engine on IB same-contract bars, flagged as replays.

(c) In-sample era split, supporting only. The 2018-22 versus 2023-26 split of the discovery data is reported with the results and is never decisive.

Descriptive feasibility counts (sessions with a valid ratio, `ratio >= 1.25`, no returns computed):

| Year | NQ sessions | NQ >= 1.25 | ES sessions | ES >= 1.25 | RTY sessions | RTY >= 1.25 |
|---|---|---|---|---|---|---|
| 2018 | 236 | 72 | 236 | 65 | 250 | 60 |
| 2019 | 237 | 56 | 237 | 56 | 250 | 52 |
| 2020 | 239 | 65 | 239 | 59 | 251 | 56 |
| 2021 | 239 | 67 | 239 | 71 | 251 | 61 |
| 2022 | 238 | 54 | 238 | 61 | 250 | 56 |
| 2023 | 236 | 50 | 236 | 56 | 250 | 46 |
| 2024 | 238 | 63 | 238 | 65 | 251 | 57 |
| 2025 | 236 | 59 | 236 | 56 | 250 | 56 |
| 2026 (to Aug) | 160 | 39 | 160 | 41 | 166 | 44 |
| Total | 2,059 | 525 (25.5%) | 2,059 | 530 (25.7%) | 2,169 | 488 (22.5%) |

NQ and ES rows are cash-day sessions from the research features files; RTY rows are all valid full sessions from `es_rty/RTY_full_session_ranges.csv` (72 of 2,364 RTY sessions have a missing roll/incomplete TR), so the RTY session count runs higher. Share of traded days qualifying: NQ 296 of 1,386 (21.4%), ES 290 of 1,314 (22.1%). At 1.5 the totals are NQ 293 and ES 325; at 2.0, NQ 108 and ES 130.

## Primary metric and decision rule

Forward tier (decides). For each qualifying market-day, the day's summed net R across all of the unfiltered shadow's entries (up to three), where R is per-contract P&L divided by the stop distance in dollars per contract, net of the configured fees and actual or simulated slippage. Also reported in points per contract.
- PASS: summed R over qualifying days is negative AND the upper bound of the 90% bootstrap interval of mean day-level R is below +0.10R.
- FAIL: mean day-level R on qualifying days is at or above +0.10R.
- INCONCLUSIVE: anything else. Extend once by 20 more qualifying market-days, apply the same rule to the full 60, then stop. A second INCONCLUSIVE is final and counts as not passing.

Bootstrap: 10,000 draws, resampling calendar dates (an NQ and ES pair on the same date moves together), percentile interval, fixed seed 20260925.

Reported beside the decision, not deciding: a one-sided sign test on month-level paired differences (mean day-level R on qualifying days minus mean day-level R on non-qualifying traded days in the same month), because trade-level wins mislead on a payoff with about a 40% win rate and large winners; the day-level sign count on qualifying days; and a plain t-stat on qualifying-day day-level R.

RTY tier (corroborates or blocks). With `top = ratio >= 1.25` (fixed, not a quintile), compute the difference in mean day-level summed R, top minus rest, at base costs, with the same date bootstrap and 90% interval.
- SUPPORTS: the interval's upper bound is below 0.
- CONTRADICTS: the point estimate is at or above 0.
- NEUTRAL: otherwise.
Shipping requires a forward PASS and an RTY result that is not CONTRADICTS. RTY alone can never ship the filter.

## Gates before the study runs

1. This prereg is committed to main before any forward data is examined for this question and before the RTY run. The commit hash is recorded in the results file.
2. Forward tier only: the candidate's live pilot has at least 10 sessions of journal (sessions that ran to 16:01 ET without halting), so the shadow and live fills can be compared before any skipped-day R is counted.
3. Forward tier only: the ratio script uses the same TR the service uses (same 18:00 to 17:00 ET session, same-contract raw OHLC, same `session_range` function). Parity check before first use: for at least 10 non-roll sessions in August 2026, the script's TR equals the research TR (`current_candidate` ranges) within one tick for both NQ and ES. Failure stops the forward tier until fixed.
4. RTY tier: gate 1 plus the three conditions listed under tier (a).

## Sensitivity (pre-declared, reported, never used to change the primary)

- Thresholds 1.0, 1.25, 1.5, 2.0 (skip), with counts at each.
- Skip versus half-size at each threshold.
- Base versus stress costs (RTY and the in-sample context; the forward tier reports shadow and live fill bases separately).
- Longs versus shorts on qualifying days.
- Eras 2018-22 versus 2023-26 (in-sample and RTY).
- NQ versus ES separately in the forward tier.
- Forward primary with and without replayed sessions.

## What ships

If the primary PASSES and RTY does not CONTRADICT: a per-market skip in the `open_breakout` inputs step. `build_manifest` computes `ratio` next to `prior_tr`, writes both to the manifest, and marks the market `skip_prior_range: true` when the ratio is at or above the threshold. Fail-closed: if `atr20` cannot be computed (fewer than 20 valid prior TRs, incomplete history, roll ambiguity), that market does not trade that session. One config flag (for example `prior_range_filter: {enabled, threshold, mode}`) in the service config and the mirrored field in `current_candidate/config.json`, default OFF. It is turned on only by a second written approval from McKinley after the results file is reviewed. The service places live orders, so the change follows the live-money change process in the runbook.

If it FAILS or ends INCONCLUSIVE: nothing ships. The result is recorded in the research folder and this question is closed; it is not re-tested on the same forward data with a different threshold.

## Stop conditions and invalidation

- Any change to the candidate's frozen rules (trigger or stop fraction, 11:30 cutoff, three attempts, 15:55 exit, short gate definition, or reinstating the break-even) invalidates qualifying days accumulated before the change. They are reported separately and the count restarts. The ratio correlates 0.997 with trigger distance in ATR terms, so a trigger change alters what this filter selects.
- A change to roll handling in the service or the ratio script, or a vendor restatement of minute bars that changes any qualifying classification, triggers a recount of every affected day. If more than 10% of qualifying classifications flip, the forward tier restarts.
- An entry-mode change from marketable IOC limits to bracket orders does not invalidate the study. The filter acts before any order exists: it only decides whether the market is armed that session. Entry mode changes fill prices on qualifying and non-qualifying days alike. The date of such a change is recorded and a before/after split is added to the sensitivity list.
- A sizing change (bps budget, contract cap) does not invalidate, since the metric is in R and points per contract.
- The study stops early only if the candidate itself is retired or suspended. Early stopping for a favorable interim result is not allowed; interim looks are not taken.

## Appendix: what we already know (IN-SAMPLE, not pre-specified)

Source: scratch `range_cuts/` (`sim.py` re-simulation with exact parity to the frozen ledgers, `analyze.py`, `q5check.py`), 2018-01-02 to 2026-08-28. Quintile edges are on traded days.

Mean R per trade by ratio quintile, all sides:

| Case | Q1 | Q2 | Q3 | Q4 | Q5 (top) | Q5 edge |
|---|---|---|---|---|---|---|
| NQ base | 0.319 | 0.309 | 0.200 | 0.250 | 0.032 | 1.28 |
| NQ stress | 0.113 | 0.233 | 0.158 | 0.218 | 0.014 | 1.28 |
| ES base | 0.148 | 0.105 | 0.043 | 0.219 | -0.056 | 1.29 |
| ES stress | -0.234 | -0.157 | -0.159 | 0.092 | -0.169 | 1.20 |

Top quintile versus rest, day-level summed R, difference in means with 95% bootstrap interval (5,000 draws), and mean R per trade by era:

| Case | Diff top-rest | 95% interval | Days top/rest | 2018-22 rest / top | 2023-26 rest / top |
|---|---|---|---|---|---|
| NQ base | -0.32 | [-0.53, -0.12] | 277 / 1,109 | 0.302 / 0.126 | 0.245 / -0.104 |
| NQ stress | -0.22 | [-0.42, -0.02] | 277 / 1,109 | 0.216 / 0.100 | 0.133 / -0.114 |
| ES base | -0.22 | [-0.42, -0.02] | 263 / 1,051 | 0.136 / 0.023 | 0.119 / -0.156 |
| ES stress | -0.01 | [-0.23, 0.21] | 243 / 972 | -0.177 / -0.074 | -0.075 / -0.415 |

Earlier `simple_filters` range_expansion study (NQ, break-even ON engine, trade only when prior TR > its 20-session mean): base 730 trades, mean R 0.08, Sharpe 0.57, CAGR 6.4% versus unfiltered 2,303 trades, mean R 0.17, Sharpe 1.63, CAGR 52.1%; paired monthly bootstrap annualized return difference -39% [-56%, -23%] base, -22% [-39%, -6%] stress. Implied mean R on the removed below-average days is about 0.22.
