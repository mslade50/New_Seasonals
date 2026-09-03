# Gaps 5 and 10: PIT dial through 2026-09-01, hedge re-run on the extended series

Folder: `scratch/ultracode_sizing_2026-09-02/dd_pit/` (nothing in the repo, its data caches, `rd2_fragility.parquet` or `signal_horizon_stats.json` was touched; the risk caches were redirected into this folder).

Scripts, in order: `01_extract_fires.py` (re-runs the production signal pipeline, writes `pit_signals_extended.pkl`), `02_pit_dial_extended.py` (`pit_dial_extended.parquet`, `pit_vintages_extended.json`), `03_strategy_daily.py` (`strategy_daily_extended.parquet`), `04_hedge.py` (`hedge_results.json`, `hedge_aug2026_daily.csv`), `05_aug_holdings_beta.py` (`hedge_aug2026_holdings.csv`). Logs sit beside each script.

## 1. Gap 5: why the PIT series "ends 2026-05-07", and the extension

**Diagnosis.** 2026-05-07 is a misattribution. Neither PIT dial in the study stops there: `cross_strategy_regime_pit_dial.parquet` and `cycle_macro_pit_dial.parquet` both run to **2026-07-02**, which is the last row of `scratch/pit_signals.pkl`, the fire-history pickle `scratch/pit_extract_signals.py` wrote once on 2026-07-03 for the July PIT gate and nobody refreshed. So the limit is a stale input, not a signal ending and not a deliberate cut. The 05-07 date entered the evidence pack from two look-alikes: `data/signal_fire_history.parquet` (a page-side artifact whose last row is 2026-05-07) and the `"last": "2026-05-07"` field of one cycle-macro cell, which is the last SIGNAL DATE of an ATR Extended Gap Up trade, not a dial date. The substantive point survives either way: the Jul-Sep 2026 episode was never scored on vintage weights.

**Extension.** `01_extract_fires.py` re-ran `daily_risk_report.compute_all_signals` on a fresh 10-year pull pinned to the pickle's start (2016-07-05) so both histories share a burn-in. Over the 2513-day overlap the seven composite signals agree exactly on five (DA, VRC, Pre-FOMC bar one day, Low-AR, SRD); Defensive Leadership differs on 12 days (2017-2021) and Dispersion on 12 (2021-2024), both S&P-constituent-list effects (503 vs 505 names in the fresh pull). SPY closes agree to 7e-7.

`02_pit_dial_extended.py` is `cross_strategy_regime_0_pit_dial.py` unchanged: expanding-window diff_mean edges at each year-end, vintage Y-1 scores year Y, **2026 scored on weights fit through 2025-12-31**, production `compute_fragility_timeseries`, live basis raw -> rolling(5) -> rolling(10). Vintage-2025 63d edges: DA 0.98, VRC 0.98, DL 5.64, FOMC 0.42, Low-AR 4.35, SRD 3.28, Dispersion 3.15 (the shipped frozen JSON the live parquet uses: 2.42 / 1.50 / 4.00 / 0 / 3.50 / 1.20 / 2.91).

Validation against the study's own PIT over 2018-01-02..2026-07-02: corr 0.9993, mean abs diff 0.35 pts, >=50 agreement 99.5%. One divergence of up to 14 pts for two weeks in late Jan 2020 (the DL reconstitution difference landing on a live episode).

### The three series over the Aug-2026 episode (10d-MA of the 63d dial)

| date | PIT (ext.) | current-weights recompute | live parquet |
|---|---|---|---|
| 2026-07-02 | 18.0 | 14.3 | 20.2 |
| 2026-07-10 | 23.1 | 19.7 | 24.0 |
| 2026-07-17 | 39.3 | 35.9 | 37.8 |
| 2026-07-22 | 48.1 | 44.3 | 45.0 |
| 2026-07-23 | **50.4** | 46.6 | 46.6 |
| 2026-07-27 | 53.7 | **50.1** | 48.5 |
| 2026-07-30 | 56.7 | 53.7 | **50.3** |
| 2026-08-07 | 68.8 | 66.6 | 60.2 |
| 2026-08-14 | 89.4 | 86.5 | 80.8 |
| 2026-08-21 | 98.6 (peak) | 95.5 (peak) | 89.5 (peak) |
| 2026-08-28 | 96.4 | 92.9 | 87.6 |
| 2026-09-01 | 95.6 | 91.8 | 87.5 |

Arm (>= 50) / release (< 45) dates, 2026, dial-date basis (the hedge acts lag-1, so the first hedged session is the next one):

| vintage | Feb-Mar 2026 episode | Jul-2026 arm | release | 2026 peak |
|---|---|---|---|---|
| PIT extended | arm 02-18, release 03-26 | **2026-07-23** | none yet | 98.6 on 08-21 |
| current-weights recompute | arm 02-18, release 03-25 | **2026-07-27** | none yet | 95.5 on 08-21 |
| live parquet (append-only PIT on frozen weights) | arm 02-13, release 03-30 | **2026-07-30** | none yet | 89.5 on 08-21 |

Agreement: over the append-only rows (2026-07-02+) PIT vs live corr 0.998 with PIT running +5.3 pts hotter; current-weights vs live corr 0.997, +2.2 pts. Day-level >=50 agreement on those rows is 88.4%, all of it the 4-5 session lag at the arm. All three vintages agree on the shape and on the fact that the dial is at or near its all-time high; they disagree by a week on when the hedge would have gone on. Whole-history PIT vs live: corr 0.892, >=50 agreement 88.3% (rows before 2026-07-02 in the live file are the recompute vintage).

## 2. Gap 10 input: the per-strategy daily payload through 2026-09-01

`03_strategy_daily.py` rebuilt the `Strategy||Tier` daily MTM series from `data/backtest_trades_full.parquet` (4,696 trades, signals to 2026-08-28, exits to 2026-09-01) with the site's own engine (`get_daily_mtm_series` on `build_site.page_shaped` frames, flat $750k, prices straight from `master_prices.parquet`). The book series sums to the ledger's PnL_flat exactly ($4,137,131). Against `dist/data/strategy_daily.json` (built 2026-08-07) the overlap correlates 0.994; the residual is ledger vintage, not method (2018 -$38.6k, 2026 +$25.4k, 2021 +$15.9k of re-booked trades), which also moves the full-sample maxDD from the plan's -8.32% to -7.34%. That vintage difference, plus the 12+12 fire-day differences above, is the whole gap between the plan's PIT hedge row and the same window re-run below (+$110k/t 1.67 -> +$98k/t 1.55; 13 -> 12 episodes because the two May-2024 arms merge under the 21-session gap rule).

## 3. Gap 10: the hedge on the extended PIT dial

Shipped spec: arm >= 50 / release < 45 on the lag-1 10d-MA 63d dial, whole book, SPY proxy, 126d rolling beta (lag-1, clipped [-1, 2]), ratio 1.0x, 2 bps x |beta| per arm event. Window 2018-01-02..2026-09-01 for the PIT comparison; the live parquet also on its full 2016-07-20+ window.

### Headline

| run | armed days | episodes (pos) | hedge PnL | t (clustered) | Sharpe unh -> hedged | maxDD unh -> hedged | worst-21d unh -> hedged | armed-day Sharpe unh -> hedged | armed-day maxDD | armed-day worst-21d | beta_hat armed | realised beta armed (up / down) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **PIT ext, 2018+** | 567 | 13 (11) | **+$94.3k** | **1.47** | 2.91 -> 3.05 | -7.34 -> -7.29% | -6.30 -> -4.64% | 1.68 -> 2.29 | -7.79 -> -6.53% | -7.18 -> -5.81% | 0.41 | 0.33 (0.46 / 0.20) |
| current-weights, 2018+ | 467 | 10 (8) | +$62.6k | 1.24 | 2.91 -> 3.02 | -7.34 -> -7.29% | -6.30 -> -5.01% | 1.66 -> 2.15 | -8.82 -> -6.50% | -6.41 -> -4.47% | 0.42 | 0.38 (0.56 / 0.22) |
| live parquet, 2018+ | 389 | 9 (6) | +$49.6k | 0.89 | 2.91 -> 3.00 | -7.34 -> -7.29% | -6.30 -> -5.01% | 1.57 -> 2.06 | -11.20 -> -10.09% | -10.42 -> -9.22% | 0.43 | 0.37 (0.50 / 0.24) |
| live parquet, 2016-07+ | 409 | 10 (6) | +$36.0k | 0.61 | 2.76 -> 2.83 | -7.34 -> -7.29% | -6.30 -> -5.01% | 1.78 -> 2.18 | -11.20 -> -10.09% | -10.42 -> -9.22% | 0.45 | 0.38 (0.48 / 0.24) |
| PIT, study window (..2026-07-02) | 539 | 12 (11) | +$98.3k | 1.55 | 2.91 -> 3.07 | -7.34 -> -7.29% | -6.30 -> -4.64% | 1.84 -> 2.54 | -7.79 -> -6.53% | -7.18 -> -5.81% | 0.42 | 0.31 (0.40 / 0.20) |
| PIT ext, **ex Aug-2026 episode** | 539 | 12 (11) | +$98.3k | 1.55 | 2.91 -> 3.06 | same | same | 1.84 -> 2.54 | same | same | 0.42 | 0.31 (0.40 / 0.20) |
| PIT ext, **ex all of 2026** | 513 | 11 (10) | +$86.8k | 1.37 | 2.91 -> 3.06 | same | same | 1.71 -> 2.36 | same | same | 0.43 | 0.30 (0.38 / 0.19) |
| current-weights ex Aug-2026 ep | 441 | 9 (8) | +$66.5k | 1.34 | 2.91 -> 3.02 | same | same | 1.82 -> 2.40 | same | same | 0.44 | 0.36 (0.51 / 0.21) |
| live 2016-07+ ex Aug-2026 ep | 386 | 9 (6) | +$39.2k | 0.67 | 2.76 -> 2.83 | same | same | 1.92 -> 2.37 | same | same | 0.46 | 0.38 (0.50 / 0.25) |

LOYO years with arming / hedged Sharpe not worse / hedge PnL positive: PIT 9/7/8, current 9/7/7, live 8/6/6 (2018+) and 9/6/6 (2016+). Full-sample maxDD is untouched by every variant (-7.34 -> -7.29%): the book's worst drawdown is not in an armed window.

Episode bootstrap and drop-best: PIT P(total <= 0) 0.064, drop-best +$60.6k at t 1.05 (best = Jan-Mar 2020, +$33.7k); current 0.103 / +$34.1k t 0.77; live 2018+ 0.170 / +$21.3k t 0.43; live 2016+ 0.256 / +$7.7k t 0.15; PIT ex Aug-2026 0.055 / +$64.6k t 1.13.

63d-beta sensitivity (the study's original primary): PIT +$105.1k t 1.78, current +$82.4k t 1.88, live 2018+ +$55.5k t 1.42, live 2016+ +$37.8k t 0.83; Sharpes within 0.02 of the 126d rows. Hedged maxDD is slightly worse with 63d beta on armed days (PIT -7.37% vs -6.53%).

### PIT episode table (extended)

| start | end | armed d | hedge $ | book $ | hedged $ | SPY % | beta_hat | VIX at arm | dial max |
|---|---|---|---|---|---|---|---|---|---|
| 2018-07-27 | 2018-10-26 | 65 | +19,019 | -3,897 | +15,122 | -5.9 | 0.44 | 12.1 | 103 |
| 2019-06-26 | 2019-08-07 | 30 | +8,861 | +40,492 | +49,353 | -1.0 | 0.60 | 16.3 | 59 |
| 2019-09-20 | 2019-10-03 | 10 | +10,445 | -34,922 | -24,477 | -3.1 | 0.46 | 14.1 | 53 |
| 2020-01-14 | 2020-03-05 | 36 | +33,671 | +28,684 | +62,355 | -7.8 | 0.46 | 12.3 | 89 |
| 2021-04-22 | 2021-10-05 | 116 | **-36,901** | +157,015 | +120,114 | +4.8 | 0.74 | 17.5 | 125 |
| 2021-11-16 | 2022-01-26 | 39 | +33,315 | +4,712 | +38,027 | -7.0 | 0.54 | 16.5 | 72 |
| 2023-06-22 | 2023-08-25 | 46 | +160 | +42,852 | +43,011 | +1.2 | 0.26 | 13.2 | 72 |
| 2024-03-25 | 2024-05-30 | 37 | +3,710 | -11,116 | -7,406 | +0.3 | 0.25 | 13.1 | 89 |
| 2024-07-10 | 2024-12-18 | 104 | +781 | +52,452 | +53,233 | +5.8 | 0.19 | 12.5 | 100 |
| 2025-02-26 | 2025-03-10 | 9 | +12,903 | +8,381 | +21,284 | -5.7 | 0.30 | 19.4 | 57 |
| 2025-10-09 | 2025-11-06 | 21 | +854 | +27,190 | +28,044 | -0.4 | 0.16 | 16.3 | 52 |
| 2026-02-19 | 2026-03-26 | 26 | +11,458 | +46,403 | +57,861 | -6.0 | 0.25 | 19.6 | 75 |
| **2026-07-24** | **open** | 28 | **-4,007** | +3,852 | -154 | **+3.2** | **0.19** | 18.7 | 98.6 |

By year (PIT): 2018 +19.0k, 2019 +19.3k, 2020 +33.7k, 2021 -37.3k, 2022 +33.7k, 2023 +0.2k, 2024 +4.5k, 2025 +13.8k, 2026 +7.5k. Current-weights and live episode tables are in `04_hedge.log`; they lose the 2019 and Oct-2025 episodes and the live series lost most of 2024 (its armed windows there are 12 and 81 days on a book that rallied).

### The Aug-2026 episode day by day (PIT arms the 07-24 session; current 07-28; live 07-31)

Book on the PIT-armed days +$3,852, hedge -$4,007, hedged -$154; SPY +3.20% over those 28 sessions. Current-weights: book +$4,399, hedge -$3,949, hedged +$450 (26 days, SPY +3.07%). Live: book -$9,283, hedge -$3,262, hedged -$12,544 (23 days, SPY +2.71%). The hedge cost about the same under every vintage; the book number differs because the 07-28..07-30 sessions (-$27k then +$41k) fall inside or outside the window. Cumulative hedge path (PIT): -$1.7k by 07-31, -$6.6k on 08-07 after three SPY-up sessions of +1.4/+1.8/+0.6%, back to -$4.0k on 09-01 as SPY gave back -0.7% on the last day. Full table: `hedge_aug2026_daily.csv`.

What the 126d beta missed: the book's realised OLS beta on the armed days was **1.12** against a beta_hat of 0.19-0.25. `05_aug_holdings_beta.py` rebuilt the ledger's open positions day by day: the flat-basis book was 105-127% NAV gross and net LONG 85-92% NAV on 07-21..07-23 (holdings beta 1.35-1.68, unhedged because the dial was still 42-48), flipped to net SHORT 22-62% NAV on 07-28..07-31 (OVS/bear-fade shorts; holdings beta 0.09-0.73, hedge_at_holdings_beta would have LOST on the 07-30 +1.7% day), sat at 6-23% NAV and near-zero beta through 08-04..08-19, then OLV rebuilt to 61% (08-20), 83% (08-24), 101% (08-26) and 115% NAV (08-31) with a holdings beta of 1.1-1.2. A hedge sized on lag-1 holdings beta would have cost -$1.9k over the same days instead of -$4.0k and would currently be short ~$840k of SPY notional, not $188k.

### The asymmetric-beta reading

Armed-day realised betas, PIT 2018+: up-days 0.46, down-days 0.20 (current 0.56 / 0.22, live 0.50 / 0.24; the email's 0.50 / 0.27 was the live series with 63d beta). Unarmed: 0.27 / 0.07. Per armed day, the book earns 31.6 bps on SPY-up days and loses 16.6 bps on SPY-down days; the hedge gives back 23.8 bps on up days and pays 29.8 bps on down days. Both signs of the question are true at once:

- On up days beta_hat (0.41) is close to the book's up-beta (0.46): the hedge gives back almost all of the rally participation (up-day book +$687k vs hedge -$518k over 290 armed up days).
- On down days beta_hat is double the book's down-beta (0.41 vs 0.20): the hedge pays roughly twice what the book loses (275 down days: book -$343k, hedge +$614k). The hedged book has a NEGATIVE net beta on selloffs.

So the hedge is not under-hedging selloffs; it is over-paying on them, and the excess is a directional short-SPY bet that happens to be right because SPY's armed-day drift is negative (-13.6% annualised on PIT armed days, up-day mean +59 bps vs down-day mean -74 bps). Variance reduction is 6% of armed-day sd (75 -> 71 bps). The plan's own words ("never described as variance reduction") stand; what this adds is that the convexity the plan wants to keep (participate in rallies, cushion selloffs) is exactly what a symmetric beta hedge removes, and at 0.41 it removes the good side nearly in full. The state-dependent-beta warning cuts the other way in Aug-2026: there the realised beta was ~1.1-1.4 on both sides and the hedge was 1/6 the size of the exposure.

Caveat on the split: sign-conditioned OLS on the regressor is a biased estimator of a "true" up/down beta (truncation), which is why the per-day dollar attribution is reported next to it; the dollars carry the same message.

### Futures margin line

Source chain: IBKR's own margin page returned 403 to the fetch; AMP Futures' live margin table (CME rates +10% "heightened risk" markup) shows ES maintenance $28,628 and MES $2,863, and a Barchart-sourced search result gives ES maintenance $26,055, consistent with AMP's figure net of the markup (~$26,025). CME speculator initial = 110% of maintenance, which is also what IBKR's overnight initial tracks, so **ES $28,660 / MES $2,866 per contract initial** is used; confirm in TWS (View > Account > Margin Requirements, or a what-if on MESZ6/ESZ6) before sizing anything. SPX 7,631 on 2026-09-01: ES notional $381.6k, MES $38.2k.

| beta case | hedge notional | ES / MES | initial margin | % of $750k NAV |
|---|---|---|---|---|
| today's beta_hat 0.25 | $188k | 0.49 / 4.9 | $14.1k | 1.9% |
| armed-day mean 0.41 | $309k | 0.81 / 8.1 | $23.2k | 3.1% |
| armed-day 95th pct 0.80 | $599k | 1.57 / 15.7 | $45.0k | 6.0% |
| clip ceiling 2.0 | $1.50m | 3.93 / 39.3 | $112.7k | 15.0% |
| holdings beta today 1.12 (from step 5) | $840k | 2.2 / 22 | $63.1k | 8.4% |

Additive to the equity legs' Reg-T/portfolio-margin requirement (WP1 guard); at MES granularity ($38k) the ratio is 1.0x only to within +-5% of NAV, which for a 0.19 beta is a +-25% sizing error on the hedge itself. Note the arithmetic on live NLV (~$632k) is 19% larger in % terms than on the $750k basis.

## 4. Sanity: does the result depend on the 2026 episode?

No. Removing the open Aug-2026 episode IMPROVES every PIT headline (+$94.3k -> +$98.3k, t 1.47 -> 1.55, armed Sharpe 2.29 -> 2.54, P(boot <= 0) 0.064 -> 0.055) because that episode is so far a small loser (-$4.0k on a +3.2% SPY window). Removing all of 2026 (two episodes, +$7.5k combined) leaves +$86.8k at t 1.37, Sharpe 2.91 -> 3.06, LOYO 8/6/7. The dependence that matters is the other way: the whole result leans on Jan-Mar 2020 (+$33.7k, drop-best t 1.05) and on the SPY drift on armed days; the current and live vintages keep the sign but lose the t (1.24 and 0.61-0.89) and the live-vintage bootstrap P(<= 0) is 0.17-0.26.

## 5. WORTH DISCUSSING

1. **The Aug-2026 episode is, so far, a third false alarm in the plan's own terms** (SPY +3.2% over 28 armed sessions; the others: Apr-Oct 2021 -$36.9k, Jul-Dec 2024 +$0.8k). The plan's "paper-track one episode, then ship" gate has its episode now, it is live and open, and it is a small loser under all three dial vintages. The plan should say what closes it (a release below 45) and what the rule counts as a fail.
2. **The beta estimator is the wrong object.** The book's realised armed-day beta in this episode was ~1.1-1.4 against a 126d beta_hat of 0.19-0.25; the holdings-based beta (open positions x ticker betas, which the Exec-tab hedge panel already computes) tracked it from -0.07 (net short late July) to 1.2 (OLV at 115% NAV on 08-31) day by day. A rolling return-beta on a book whose exposure turns over inside 10 sessions has no predictive content exactly when armed (the study found the same for 63d). Either the hedge sizes on holdings beta, or ratio 1.0x is a number on a beta that is not the book's. This also changes the margin line: 8.4% of NAV today, not 1.9%.
3. **Over-paying on selloffs is the return source, and it is a bet.** Down-day realised beta 0.20 vs hedge 0.41; the hedged book is net short on selloffs. The plan's stated purpose (keep convexity, hedge tail) is not what the replay shows the hedge doing; what it shows is a short-SPY position gated by the dial with a 6% sd reduction. If that is the intended product, say so and size it as a directional overlay with its own prereg (the exposure-leg precedent), not as "beta-neutralising".
4. **The email's gap-5 sentence needs correcting** ("ends 2026-05-07" -> "ended 2026-07-02, the date of the one-off `pit_signals.pkl` extraction; extended through 2026-09-01 in dd_pit"), and the study's PIT numbers in plan section 8 are one ledger vintage old (maxDD -8.32% belongs to the 2026-08-07 payload; today's ledger gives -7.34% unhedged on the same window). A `pit_signals` refresh belongs in whatever runbook re-scores the dial, otherwise this gap reopens with every new episode.
