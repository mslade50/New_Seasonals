# Factor-Exposure Sleeve — Prototype Study

Run date: 2026-07-02. Track: `factor-sleeves`.
Scripts (all under `scratch/ultracode_research/`):
- `factor_etf_inventory.py` — what master_prices has
- `fetch_factor_etfs.py` — yfinance pull of factor ETF history (network available; saved to `factor_etf_prices.parquet`)
- `factor_sleeve_backtest.py` — main prototype backtest (static + fragility-conditional)
- `factor_timing_robustness.py` — episode attribution, LOYO, significance
- `combined_portfolio.py` — marginal effect on the combined book

**Bottom line: not worth it.** Static ETF factor tilts are diluted beta (none beat SPY's
Sharpe on the common 2013-08+ window except SPHQ marginally), they LOSE money in
exactly the high-fragility months the book needs help in (USMV -0.66%/mo, MQUV
-1.19%/mo in dial>=50 months), and adding any of them at 0.5x NAV lowers the combined
book Sharpe (2.79 → 2.63-2.70). Fragility-timed rotation to USMV adds ~nothing
(+4.8% cumulative over 10y, episode-clustered p=0.28). The one variant that looks good
— rotate SPY→cash at dial>=50, Sharpe 1.35 vs 0.99 — is not a factor result at all:
it is the existing sizing throttle re-expressed as a market-timing overlay, 73% of its
gain is the COVID quarter, it is insignificant after episode clustering (p=0.117), and
it carries the dial's calibration-lookahead caveat at full strength.

---

## 1. Data available

`data/master_prices.parquet` (long format: ticker/date/OHLCV, adjusted, 1114 tickers)
has **none** of the dedicated factor ETFs — MTUM, QUAL, USMV, VLUE, SIZE, SPLV, SPHQ,
RSP, VTV/VUG/IWD/IWF, COWZ, NOBL, DBMF/BTAL are all missing. It does have SPY/QQQ/IWM,
EFA/EEM, TLT/IEF/LQD/HYG, GLD, and the SPDR sectors (XLP/XLU/XLV) from 2000-2003.

yfinance works from this machine, so the prototype pulled adjusted closes directly
(`fetch_factor_etfs.py`): USMV from 2011-10, SPLV 2011-05, MTUM/VLUE 2013-04, QUAL
2013-07, SPHQ 2005-12, RSP 2003-05, BIL 2007-05. Run-date partial bar dropped.
Sanity check: master_prices SPY vs yfinance SPY track to 8bps relative drift over 3y.

**Gap for production:** if any of this were adopted, the factor ETFs would need to be
backfilled into `master_prices.parquet` (same path as the 2026-06-11 LEV3X backfill;
`update_master_prices.yml` auto-maintains whatever is in the parquet).

## 2. Rules of the prototype (`factor_sleeve_backtest.py`)

- Monthly rebalance. Signals read at month-end close, executed same close
  (one-overnight optimism, negligible at this turnover).
- Costs: 5 bps per side on every dollar traded (a full switch = 10 bps).
- Fragility dial = `rd2_fragility.parquet` 63d column, 10d MA (live sizing basis),
  ffill limit 5, read as-of month-end. History starts 2016-07; conditional tests run
  2016-09+ (119 months).
- Sharpe uses rf=0 (BIL CAGR 1.36% shown for context).
- Book series = ledger `PnL_flat_750k` summed by exit month / $750k.

## 3. Static tilts — gross performance (max available history)

| sleeve | window | CAGR% | Vol% | Sharpe | MaxDD% |
|---|---|---|---|---|---|
| SPY (b&h) | 2000-02+ | 8.42 | 15.2 | 0.61 | -50.8 |
| USMV | 2011-11+ | 11.41 | 11.1 | 1.03 | -19.1 |
| QUAL | 2013-08+ | 13.63 | 14.6 | 0.95 | -27.8 |
| MTUM | 2013-05+ | 16.30 | 16.3 | 1.01 | -30.2 |
| VLUE | 2013-05+ | 13.53 | 17.7 | 0.81 | -29.0 |
| SPLV | 2011-06+ | 9.92 | 11.7 | 0.87 | -21.4 |
| SPHQ | 2006-01+ | 10.36 | 15.4 | 0.72 | -53.9 |
| RSP | 2003-06+ | 10.86 | 16.4 | 0.71 | -55.6 |
| MQUV blend (25% each) | 2013-08+ | 13.50 | 13.9 | 0.98 | -23.9 |
| EW XLP/XLU/XLV (2003+ proxy) | 2000-02+ | 8.22 | 11.6 | 0.74 | -33.6 |

**Common window 2013-08+ (the honest comparison):** SPY Sharpe 0.99 / CAGR 14.0.
USMV 0.93/10.5, QUAL 0.95/13.6, MTUM 1.00/16.2, VLUE 0.78/13.0, MQUV 0.98/13.5,
SPHQ 1.06/14.5, EW-defensive-sectors 0.89/10.1. Only SPHQ nudges past SPY, and by an
amount well inside noise for 13 years of a single spread. There is no static-tilt
free lunch here: the last decade was a bad regime for min-vol and value, and a
factor sleeve initiated today buys the same beta with tracking error.

Per-year table for the main variants is printed by `factor_sleeve_backtest.py`
(2000-2026 for SPY/EW-defensive, 2011+/2013+ for the factor ETFs). Notable: USMV lags
SPY in 8 of the last 10 calendar years; the EW XLP/XLU/XLV proxy shows the 2000-02
and 2008 defensive value (+12.8% in 2000, -22% vs -37% in 2008) — the regimes the
factor ETFs never lived through.

## 4. Fragility-conditional rotation (2016-09+, 119 months)

Rule: hold base; when month-end dial >= thr, hold the defensive asset next month.
References over the identical window: SPY Sharpe 0.99 (CAGR 15.1, maxDD -23.9),
USMV 0.83, MQUV 0.96.

| variant | CAGR% | Sharpe | MaxDD% | % months defensive |
|---|---|---|---|---|
| SPY→USMV thr50 | 15.6 | 1.03 | -24.6 | 16% |
| SPY→TLT thr50 | 18.2 | 1.22 | -25.2 | 16% |
| SPY→BIL thr50 | 18.7 | 1.35 | -20.5 | 16% |
| SPY→BIL thr44 | 19.0 | 1.39 | -20.5 | 21% |
| MQUV→BIL thr44 | 18.4 | 1.42 | -20.7 | 21% |
| daily hysteresis 50/45 SPY→USMV | 16.0 | 1.06 | -24.1 | 15% |

Threshold sweep (40/44/50/55) is flat-ish for the BIL variants (Sharpe 1.20-1.39),
so the result is not a single-threshold artifact. But:

### Robustness (`factor_timing_robustness.py`) — this is where it dies

19 defensive months in 8 contiguous episodes. Active return attribution, SPY→BIL thr50
(after 10 bps per switch):

| episode | n_mo | SPY total % | active % |
|---|---|---|---|
| 2017-10 | 1 | +2.4 | -2.4 |
| 2018-09..10 | 2 | -6.4 | +6.5 |
| **2020-01..03** | 3 | **-19.5** | **+21.8** |
| 2021-05..09 | 5 | +3.5 | -3.9 |
| 2021-12..2022-01 | 2 | -0.9 | +0.3 |
| 2024-09..12 | 4 | +4.6 | -3.4 |
| 2025-03 | 1 | -5.6 | +5.8 |
| 2026-03 | 1 | -4.9 | +5.1 |

- Total active ~+30% over ~10y; **drop the COVID episode and +8.0% remains** (~0.8%/yr).
- Hit rate 10 avoided-loss months (avg -5.1%) vs 9 missed-gain months (avg +2.7%).
- Significance: monthly t=+1.41 p=0.175; **episode-clustered t=+1.79 p=0.117, N=8
  episodes. Not significant.**
- LOYO Sharpe uplift survives every drop-year (worst: drop 2020 → 1.22 vs 1.06) but
  the active total collapses from +36% to +9.5% ex-2020.
- SPY→TLT is pure COVID (drop 2020 → active -14.3%; 2024 rate selloff cost -12.2%). Dead.
- **SPY→USMV — the actual factor-timing question — adds +4.8% total over 10y,
  episode-clustered p=0.28, and goes negative dropping 2025.** Factor timing via the
  dial adds nothing over static allocation; whatever the dial knows is expressed by
  being OUT of equity, not by which equity flavor you hold.

### The vintage caveat cuts hardest exactly where the gain is

The fragility history is a current-vintage reconstruction: component states are
point-in-time, but the composite edge weights come from a full-sample event study
(`data/signal_horizon_stats.json`). COVID is in that calibration sample. The dial
reading 50+ in Jan-Feb 2020 — the episode carrying 73% of the overlay's gain — is
partially a product of weights fitted knowing COVID happened. The honest ex-2020
number (+8% / 10y, p>0.1) is the right one to reason from, and it does not clear
the bar.

## 5. High-fragility months (month-mean dial >= 50, 2016-09+, N=16)

Months: 2018-09/10, 2020-01/02, 2021-05/06/07/09/12, 2022-01, 2024-09..12, 2026-02/03.

| series | avg %/mo in high-frag | avg %/mo other |
|---|---|---|
| SPY | -0.95 | +1.62 |
| USMV | **-0.66** | +1.08 |
| MQUV | **-1.19** | +1.57 |
| EW XLP/XLU/XLV | -0.41 | +0.99 |
| SPY→BIL thr50 | +0.06 | +1.74 |
| book (flat 750k) | **+1.93** | +2.73 |

The requirement was a sleeve that WORKS in high-fragility months. Every long-only
factor tilt loses money there — min-vol and quality are still ~0.8-0.9 beta and the
factor premium does not overcome the market's -1%/mo in those windows. The only thing
that "works" is cash, which is not a sleeve. Meanwhile the book itself, at the monthly
PnL level, is still +1.93%/mo in those months (its per-trade edge degrades to
+0.17-0.19 avgR, but positive trade flow keeps monthly PnL above water) — the book
needs less size there, not a long-beta hedge that bleeds alongside the market.

## 6. Correlation to the book & combined-portfolio marginal effect

Book monthly return (PnL_flat_750k by exit month / 750k), 2016-09+: avg +2.63%/mo,
Sharpe 2.79 (gross, survivorship-flattered — treat as an upper bound), maxDD -7.2%.

Correlations of sleeve monthly returns to book monthly return: SPY +0.20 (282 mo
overlap) / +0.18 (2016-09+); USMV +0.15/+0.14; MQUV +0.22/+0.19; EW-defensive
+0.12/+0.10; the rotation variants +0.17-0.18. Low — but the low correlation is a
property of the book being idiosyncratic, not of the factor tilt: SPY itself
correlates +0.18. The timing overlay's ACTIVE return correlates -0.04 with the book
(genuinely orthogonal, but tiny and insignificant).

Adding a sleeve at 0.5x NAV on top of the book (`combined_portfolio.py`, 2016-09+):

| portfolio | avg %/mo | Sharpe | maxDD | sleeve drag in high-frag months |
|---|---|---|---|---|
| book alone | +2.63 | 2.79 | -7.2% | — |
| book + 0.5x SPY | +3.26 | 2.65 | -8.8% | -0.48%/mo × 16 |
| book + 0.5x USMV | +3.05 | 2.70 | -8.7% | -0.33%/mo × 16 |
| book + 0.5x MQUV | +3.23 | 2.63 | -9.4% | -0.60%/mo × 16 |

Every variant lowers the combined Sharpe and deepens maxDD, and the USMV tilt is
barely distinguishable from plain SPY at the portfolio level. The book's Sharpe is
inflated (gross ledger, single-stock survivorship), which weakens the exact Sharpe
comparison but not the sign: a 0.15-0.2-correlated sleeve with Sharpe ~1 cannot
improve a portfolio whose existing engine is several times better risk-adjusted,
and it adds drag concentrated in the book's worst months.

## 7. Execution fit (if it were adopted anyway)

- Instruments: US-listed ETFs (USMV/QUAL/MTUM/VLUE/BIL), $5-40B AUM, penny-wide
  spreads. Capacity at $750k: trivial (a full sleeve is <$400k, <0.01% of ADV).
- Turnover: monthly-conditional variant = 16-26 switches per decade; static blend
  rebalance turnover ~1-2%/mo. Order types: MOC or limit-at-close on rebalance day
  fits the existing staged-order pattern; these are US-session ETFs → `REL_OPEN`
  per the Seasonal-tab geography rule would also work.
- Plumbing needed: backfill the ETFs into master_prices (auto-maintained after),
  a small monthly staging script, one more Sheets tab or reuse of `Seasonal`.
  All cheap — cost is not the objection; expected value is.

## 8. Sub-question 2 — single-stock factor screens (feasibility memo, no backtest)

**What the repo has:** prices only (adjusted OHLCV, ~1114 tickers, today's-membership
universe) plus an FMP key currently used for `/stable/earnings`.

**FMP endpoints needed for factor screens** (same key, higher-tier plan likely
required for bulk/history):
- `/stable/key-metrics` + `/stable/ratios` (quarterly history): ROE/ROIC, gross
  profitability, accruals, FCF yield, EV/EBITDA, debt ratios — covers quality + value.
- `/stable/income-statement`, `/balance-sheet-statement`, `/cash-flow-statement`
  (as-reported, with `acceptedDate`/`fillingDate` fields — these enable point-in-time
  alignment, which is the one thing that makes a fundamentals backtest honest).
- `/stable/delisted-companies` for the dead-name list; historical index constituents
  (S&P membership history) is on FMP's higher tiers only.
- Momentum and low-vol screens need no new data (prices suffice) but inherit the
  universe survivorship.

**Why not to backtest it now:** `master_prices` is today's membership. A quality or
value screen backtested on survivors overstates returns by construction (the
classic several-%/yr small-cap value survivorship inflation), and unlike the
book's short-horizon signals (days-scale, where survivorship mostly inflates the
long tail modestly), a 6-12 month factor hold compounds the bias directly. An honest
single-stock factor backtest needs point-in-time universe + fundamentals — either
FMP's higher tier plus constituent history, or a purchased PIT dataset. Given
Section 3-6 show even the *clean* ETF implementation doesn't help this book,
spending on PIT fundamentals to re-ask the same question with more idiosyncratic
risk and more turnover is not justified. If a fundamentals screen is ever wanted,
implement it as a tilt INSIDE the ETF wrapper (e.g. SPHQ/COWZ) rather than 30-50
single names at $750k with a solo operator.

## 9. Bias inventory

1. **Dial vintage / calibration lookahead** — composite edge weights fitted on the
   full sample including the very episodes (COVID, 2022) the overlay profits from.
   Point-in-time component states mitigate but do not remove this. Affects every
   conditional result; worst for SPY→BIL 2020.
2. **Episode concentration** — 8 defensive episodes, 1 carries 73% of the gain.
   Episode-clustered p=0.117. Nothing here is significant.
3. **Same-close execution** — signal read and executed at the same month-end close;
   real fill is next open. At 16-26 switches/decade the effect is small (~bps) but
   directionally flattering.
4. **Survivorship** — none for the ETF results (all live funds, full histories);
   severe for any single-stock extension (Section 8). EW XLP/XLU/XLV proxy is clean.
5. **Adjusted-price basis** — total-return closes, correct for return math per the
   project's dividend rule (relative levels recomputed per run); no frozen dollar
   levels anywhere in the prototype.
6. **Costs** — 5 bps/side applied on all trades; no borrow, no market impact
   (irrelevant at this size). BIL used as cash proxy (earns T-bill yield honestly).
7. **Factor-ETF era** — 2011/2013+ only; the regimes where defensive factors shine
   (2000-02, 2008) are visible only in the sector-proxy series, which did add value
   then. A 2003+ claim about USMV is not possible from live data; the common-window
   comparison is the honest one.
8. **Book Sharpe inflation** — combined-portfolio comparison uses the gross,
   survivorship-flattered ledger; the sign of the conclusion (sleeve dilutes) is
   robust to this, the magnitudes are not.

## 10. Verdict and recommendation

Static ETF factor tilts: diluted beta, no Sharpe edge on the common window, lose
money in high-fragility months, dilute the combined book. Fragility-timed factor
rotation (SPY→USMV): +4.8% per decade, p~0.3, sign-unstable — nothing. Fragility-timed
de-risking to cash: the only live wire, but it is insignificant (p=0.117), 73% COVID,
maximally exposed to the dial's calibration lookahead, and strategically redundant —
the book already de-risks at dial 50 via the sizing throttle, which is the same trade
expressed where the book actually holds risk.

**Recommendation: do not add a factor sleeve.** Redirect the appetite for "something
that works at dial 55+" into the already-pending throttle change (kill the 1.25x
boost, 1.0x through 50, taper to 0.5x by 60) and, if idle cash should earn something,
hold it in BIL/T-bills as cash management rather than as a timed equity sleeve. Revisit
factor exposure only if the goal changes from "help the book in fragile months" to
"deploy long-term passive capital," which is a different mandate than this book's.

## Adversarial verification (2026-07-02)

Independent recompute in `verify_factor-sleeves.py` (+ `verify_fetch_fs.py`,
`verify_probe_fs.py`). Fresh yfinance pull (`verify_fs_prices.parquet`) — matches the
researcher's parquet to <=0.24% relative (dividend-adjustment vintage noise), and
master_prices SPY to 0.26% over 3y. Own monthly-return, dial, episode, and t-stat
machinery; nothing reused from the study scripts. All six decisive claims CONFIRMED.

| claim | theirs | mine | verdict |
|---|---|---|---|
| Common-window 2013-08+ Sharpe | SPY 0.99, USMV 0.93, QUAL 0.95, MTUM 1.00, VLUE 0.78, MQUV 0.98, SPHQ 1.06 | SPY 0.99, USMV 0.93, QUAL 0.96, MTUM 1.03, VLUE 0.79, MQUV 1.00, SPHQ 1.07 (N=155 mo each) | CONFIRMED* |
| High-frag months (dial>=50 month-mean) | N=16, USMV -0.66, MQUV -1.19, SPY -0.95, book +1.93 %/mo | Exact month list reproduced; USMV -0.66, MQUV -1.19, SPY -0.95, book +1.93 | CONFIRMED |
| Book + 0.5x sleeve dilutes | book 2.79 vs +SPY 2.65 / +USMV 2.70 / +MQUV 2.63 | On clean 2016-09..2026-06 window: book 2.90 vs 2.73 / 2.78 / 2.73 — dilution slightly LARGER; every sleeve deepens maxDD (-5.7% -> -8.7..-9.4%) | CONFIRMED** |
| SPY->USMV thr50 timing | +4.8% active/10y, p=0.28, negative ex-2025 | +3.9% active, episode-sum t=+0.39 p=0.71 (mean-monthly variant p~0.35), ex-2025 = -0.8% (sign flips) | CONFIRMED (even weaker than claimed) |
| SPY->BIL thr50 | Sharpe 1.35 vs 0.99; COVID +21.8 of +30; ex-2020 +8%; t=1.79 p=0.117 | Sharpe 1.36 vs 1.00; COVID episode +20.7 of +28.5 (73%); LOYO ex-2020 +7.8%; episode-SUM t=+1.25 p=0.25; episode mean-monthly t=1.71 p=0.13 | CONFIRMED*** |
| Correlations | sleeve-book +0.14..0.22, SPY +0.18, active -0.04 | USMV +0.14, MQUV +0.19, SPY +0.18 (2016-09+) / +0.20 (2003+), BIL-rotation active -0.09 | CONFIRMED |

Notes:
- \* In my (fresher) data MTUM Sharpe is 1.03 vs SPY 0.99, so "no factor ETF beats SPY
  except SPHQ" is marginally overstated — MTUM also nudges past. Both gaps are well
  inside noise for 13y of a single spread; the substantive claim (no static-tilt edge)
  stands.
- \** Their book Sharpe 2.79 / +2.63%/mo reproduces ONLY when the partial July-2026
  month (2 trading days) is included as a full month (their "119 months"). Clean
  118-month window gives 2.90 / +2.69%/mo. The blemish is conservative — it makes the
  book look worse, i.e. understates the dilution they report.
- \*** Their t=1.79 is not reproducible on episode SUMS (t=1.25 on their own printed
  episode table); it matches a t-test on episode MEAN monthly actives (t=1.71 in my
  data). Both conventions agree the overlay is insignificant (p 0.13-0.25), so the
  conclusion is unaffected; if anything the sums-based test is weaker than reported.
- Episode attribution table reproduced within 0.1-1.1pp per episode (8 episodes,
  19 defensive months, identical episode boundaries).
- The calibration-lookahead caveat on the dial is real and inherited here unchanged;
  it further weakens (never strengthens) the only marginally-positive variant (BIL
  rotation), consistent with the study's framing.

**Verifier verdict: the recommendation stands.** No claim was refuted; two decisive
claims (USMV timing, BIL-rotation significance) are even weaker on independent
recompute than as reported.
