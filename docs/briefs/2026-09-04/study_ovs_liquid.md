# Brief: study_ovs_liquid (pre-registered study: has the liquid-tier OVS edge decayed?)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0 hard rules, D9, section 6 report format). Type: RESEARCH. Read-only on the repo; you write only under `scratch/ultracode_research/ovs_liquid_2026-09-04/`.

## Decision and why
The 2026-09-03 book review found Overbot Vol Spike (OVS) liquid-tier positions at avgR +0.50 from 2003 to 2023 (N 250) and -0.03 from 2024 onward (N 66, Welch t 3.3 unclustered), with 2026 liquid at -0.31 on 29 and six names (MU, DE, XLK, GLW, INTC, IBM) carrying -$40k, while the overflow tier is in line with its history. OVS is the book's largest slot-earner and is cap-bound on 97% of its signal days, so any OVS change moves real money. The mind pre-registers here, before you run anything, the question, the statistic, the robustness cuts and the closed decision set. You compute; the mind decides; nothing in config changes.

## Files you own
`scratch/ultracode_research/ovs_liquid_2026-09-04/` only.

## Hard rules
Section 0 of the plan. Do not rebuild the ledger. Overflow-tier historical stats are an upper bound (static universe survivorship, CLAUDE.md "Ledger SURVIVORSHIP CAVEAT"); the comparison in this study is liquid against its own history, never liquid against overflow.

## Pre-registration (frozen; do not alter)
Population: OVS rows in `data/backtest_trades_full.parquet`, near/far tranches collapsed to positions (R = sum PnL / sum risk), tier from the ledger's tier column (liquid = ticker in `strategy_config.LIQUID_PLUS_COMMODITIES`).
H1: liquid-tier OVS expectancy from 2024-01-01 is below its 2010-01-01..2023-12-31 level.
Primary statistic: difference in avgR (2024+ minus 2010-2023), t-statistic clustered by signal date, plus a monthly block bootstrap (10,000 draws) 95% interval on the difference.
Peek acknowledged: the split was noticed on the 2024-2026 data. There is no clean holdout, so the robustness cuts below are the discipline, and every cut is reported whether or not it helps H1.
Robustness cuts (all mandatory):
(i) drop the six named worst names, recompute the primary;
(ii) path split: P1 (decisive gap) vs P2 (mild gap) tranches, each era;
(iii) exclude 2026 (midterm; the 0.75x cycle multiplier already applies) and recompute;
(iv) bottom-extremity share: fraction of liquid signals with mean(rank_2d, rank_5d, rank_10d, rank_21d) < 94 in each era, and the primary recomputed on the top cell only. If the deficit disappears in the top cell, plan item D3.5 already addresses it;
(v) sector and theme concentration of 2024+ liquid signals (semis / mega-cap tech share) vs 2010-2023;
(vi) signal supply: liquid OVS positions per year 2010-2026 and per-year avgR, so a rising count with falling edge is visible;
(vii) controls: the same era split on overflow OVS and on the 3x ETF Overbot Fade (short-side control);
(viii) exit-type mix by era (target, time exit, EOD-DD) and the time-exit avgR by era.
Closed decision set (the mind applies it; you report the inputs): recommend "liquid OVS 0.5x" only if ALL hold: primary clustered t <= -2.0; survives (i) at t <= -1.5; the deficit persists in the top-extremity cell under (iv) at t <= -1.5; and 2024, 2025 and 2026 are each individually below the 2010-2023 mean. If (iv) alone explains it: no action beyond D3.5. Otherwise: no action, re-examine at +40 liquid positions. No other multiplier or filter may be proposed from this study.

## Recon first
`scratch/ultracode_research/ovs_liquid_2026-09-04/00_plan.md`: columns you found for tier, path, tranche, exit type, ranks (if the ranks are not in the ledger, compute them from `data/master_prices.parquet` with `indicators.py`'s definitions and say so), and your clustering and bootstrap method.

## Verification
`scratch/ultracode_research/ovs_liquid_2026-09-04/checks.json` from your script:
`{"n_liquid_2010_2023": int, "n_liquid_2024p": int, "avgR_2010_2023": float, "avgR_2024p": float, "diff": float, "t_clustered": float, "boot_ci95": [float, float], "cut_i_t": float, "cut_iii_t": float, "cut_iv_top_cell_t": float, "cut_iv_bottom_share_2010_2023": float, "cut_iv_bottom_share_2024p": float, "years_below_mean": ["2024", ...], "overflow_diff_t": float, "lev3x_fade_diff_t": float, "decision_inputs_all_hold": bool, "explained_by_extremity": bool}`.
No screenshots.

## Report
Section 6 format. Findings: the primary, then each cut with its number. Handoff: state which of the closed decision set's conditions hold; do not propose anything outside it.
