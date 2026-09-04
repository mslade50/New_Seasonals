# Cross-Sectional Momentum Rotation — Prototype & Verdict

Run date: 2026-07-02. Track: `momentum-rotation`.
Scripts: `scratch/ultracode_research/mom_rotation.py` (main prototype),
`mom_followup.py` (sub-period alpha, absolute-momentum overlay, book-worst-month exhibit),
`mom_inspect.py` / `mom_inspect2.py` (data coverage checks).
Saved series: `scratch/ultracode_research/mom_monthly_series.parquet` (all variants + SPY + book monthly PnL%).

## Verdict up front

**Do not productionize any variant.** Sector rotation is SPY beta with turnover (net Sharpe 0.85
vs SPY 0.84, corr 0.86, alpha +1.9%/yr with t=1.25 full-sample, decaying to +0.85%/yr t=0.41
since 2016). Country rotation is strictly worse than SPY. The single-stock 12-1 book shows
+4-5%/yr alpha but on a near-totally survivorship-biased universe, so it cannot be trusted.
Decisively: **every variant LOSES money in high-fragility months** (avg -0.4% to -1.8%/month
when the 63d MA10 fragility month-mean >= 50) while the existing book averages +1.9%/month in
those same months. The sleeve fails the one mandate that motivated it.

## Rules (as implemented)

- Monthly rebalance at month-end close (last trading day); signal on month-end adjusted closes,
  position held over the following month. No intramonth trading.
- **A1 sectors**: 11 SPDRs (XLB/XLC/XLE/XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY; XLRE enters 2016,
  XLC 2019 after 13-month history gate), momentum = mean(6m, 12m total return), top 3 equal weight.
- **A2 sectors**: same universe, classic 12-1 momentum (t-12 to t-1, skip last month), top 3 EW.
- **B countries**: all 10 country/regional ETFs in the cache (EWJ EWW EWZ EWT EWY FXI INDA EEM
  EFA RSX), blend momentum, top 3 EW. RSX hard-dropped after 2022-03-31 (trading halted; the
  cache carries phantom flat prices to 2026 that would otherwise rank it top-3 on 0% "momentum").
- **C stocks**: 12-1 momentum, top 20 EW, universe = `LIQUID_UNIVERSE` minus ETFs/indices
  (~162 current mega/large caps).
- Eligibility: >= 13 monthly closes at signal time.
- Costs: charged per side on the L1 weight change vs drifted prior weights. ETFs 5 bps/side,
  stocks 15 bps/side.
- Sample: 2003-02 .. 2026-06 (281 months; the partial 2026-07-01 bar is dropped in all headline
  stats — `mom_followup.py` re-check).

## (2) Gross/net performance 2003-2026 (net of costs)

| Variant | CAGR | Vol | Sharpe | MaxDD | Avg 1-way turnover/mo | Cost drag/yr |
|---|---|---|---|---|---|---|
| A1 sector top3 blend(6,12) | 11.45% | 13.9% | 0.85 | -38.0% | 25% | 0.30% |
| A2 sector top3 12-1 | 11.20% | 14.6% | 0.80 | -41.3% | 23% | 0.28% |
| B country top3 blend | 10.62% | 21.4% | 0.58 | **-66.8%** | 24% | 0.29% |
| C stock top20 12-1 (BIASED) | 15.78% | 17.7% | 0.92 | -50.3% | 30% | 1.08% |
| SPY buy-hold | 11.68% | 14.5% | 0.84 | -50.8% | — | — |
| A1 + absolute filter (neg-mom slot to cash) | 10.50% | 12.9% | 0.84 | -22.3% | ~25% | ~0.3% |

Sharpe uses rf=0 (comparison-only; all rows treated identically).

### Beta / alpha vs SPY (monthly OLS, 281 months)

| Variant | beta | corr | alpha/yr | naive t |
|---|---|---|---|---|
| A1 sectors | 0.82 | 0.86 | +1.85% | +1.25 |
| A2 sectors | 0.87 | 0.86 | +1.10% | +0.72 |
| B countries | 1.07 | 0.73 | -0.79% | -0.26 |
| C stocks | 1.00 | 0.82 | +4.03% | +1.92 |
| A1+absfilter | 0.69 | — | +2.45% | +1.47 |

Sub-period alpha (the decay exhibit):

| Variant | 2003-2012 | 2013-2026 | 2016-07..2026-06 |
|---|---|---|---|
| A1 sectors | +3.05%/yr (t=1.17) | +1.05% (t=0.60) | +0.85% (t=0.41) |
| A2 sectors | +1.37% (t=0.52) | +1.15% (t=0.64) | +0.98% (t=0.43) |
| C stocks | +2.68% (t=0.77) | **+5.43% (t=2.09)** | +4.98% (t=1.51) |

Sector-rotation alpha decays toward zero in the modern sample and is never significant. The
stock book's alpha being LARGER in the recent period is exactly the signature of
universe-selection bias: the liquid universe was hand-picked recently from names that are big
today (NVDA, META, AVGO...), so recent-period momentum on it is mechanically flattered.

### Per-year returns (net)

| Year | A1 sectors | A2 sectors | B countries | C stocks | SPY |
|---|---|---|---|---|---|
| 2003 | +26.3% | +32.2% | +39.8% | +32.5% | +31.4% |
| 2004 | +13.9% | +13.9% | +19.9% | +15.1% | +10.7% |
| 2005 | +16.5% | +18.3% | +42.8% | +29.5% | +4.8% |
| 2006 | +13.8% | +6.3% | +43.2% | +7.5% | +15.8% |
| 2007 | +11.2% | +9.8% | +47.5% | +9.8% | +5.1% |
| 2008 | -26.0% | -29.8% | **-52.6%** | -39.7% | -36.8% |
| 2009 | +22.4% | +19.5% | +32.1% | +15.7% | +26.4% |
| 2010 | +17.1% | +13.8% | +16.5% | +20.1% | +15.1% |
| 2011 | -1.8% | -1.4% | -25.6% | -1.4% | +1.9% |
| 2012 | +6.7% | +7.2% | +9.2% | +29.3% | +16.0% |
| 2013 | +40.6% | +38.2% | +6.4% | +40.3% | +32.3% |
| 2014 | +5.5% | +10.5% | +2.8% | +22.5% | +13.5% |
| 2015 | -1.8% | +1.6% | -11.4% | +7.8% | +1.2% |
| 2016 | +7.4% | +5.4% | +22.2% | +15.5% | +12.0% |
| 2017 | +17.4% | +15.8% | +18.9% | +30.7% | +21.7% |
| 2018 | -4.3% | -7.6% | -20.4% | -2.5% | -4.6% |
| 2019 | +19.6% | +21.6% | +19.6% | +26.2% | +31.2% |
| 2020 | +20.9% | +20.8% | +12.2% | +24.8% | +18.3% |
| 2021 | +12.6% | +19.6% | +4.8% | +20.4% | +28.7% |
| 2022 | +7.5% | +9.2% | -6.2% | +1.6% | -18.2% |
| 2023 | +16.5% | +13.8% | +9.8% | +10.8% | +26.2% |
| 2024 | +19.7% | +17.6% | +1.1% | +28.7% | +24.9% |
| 2025 | +15.6% | +15.4% | +34.2% | +10.4% | +17.7% |
| 2026 YTD | +6.7% | +8.9% | +53.9% | +44.7% | +9.7% |

(2026 rows from the main run include the partial July bar; headline stats exclude it. Country
2026 +54% is EWZ/EWW-driven and on a 9-name universe — noise. Stock 2026 +45% is the bias
signature again.)

### Momentum crash windows

- **2009**: the classic momentum crash (Mar-Jun junk rally) shows up as underperformance, not
  disaster, because these are long-only top-N books: A1 captured +4.1/+4.8/+2.2% in Mar-May 09
  vs SPY +8.3/+9.9/+5.8% — lag, not blowup. C stocks was flat Apr 09 (-0.4%) vs SPY +9.9%.
- **2020**: Feb-Mar drawdowns essentially match SPY (A1 -8.5%/-10.4% vs SPY -7.9%/-12.5%);
  the Nov-2020 vaccine rotation cost nothing visible at monthly granularity.
- Worst single months: A1 -12.4% (2008-10); B -27.5% (2008-10), -16.2% (2011-09); C -14.3%
  (2008-10). Long-only rotation carries full equity-crash tails.

## (3) Correlation to the existing book

Book series = ledger `PnL_flat_750k` summed by exit month / 750k (281 overlapping months, 2003-2026):

| Series | corr w/ book monthly PnL |
|---|---|
| A1 sectors | +0.195 |
| A2 sectors | +0.197 |
| B countries | +0.158 |
| C stocks | +0.239 |
| SPY buy-hold | +0.194 |

Correlations are low — but no lower than plain SPY. The sleeve adds nothing that a passive
beta allocation wouldn't. And in the book's 12 worst months since 2016-07 (bottom decile,
book avg about -2.1%), every variant averaged about -1.0% to -1.2% (SPY -1.3%). It is
pro-cyclical to the book's pain, not a diversifier where it counts.

## (4) High-fragility months (63d MA10, 2016-07..2026-06)

High month = calendar month whose mean of the live basis (63d fragility, 10d MA) >= 50.
16 such months: 2018-09/10, 2020-01/02, 2021-05/06/07/09/12, 2022-01, 2024-09..12, 2026-02/03.

| Series | high-frag avg/mo (N=16) | rest avg/mo (N=105) |
|---|---|---|
| A1 sectors | **-0.58%** | +1.35% |
| A2 sectors | -0.44% | +1.39% |
| B countries | **-1.78%** | +1.56% |
| C stocks | -0.78% | +2.01% |
| A1+absfilter | -0.58% | +1.25% |
| SPY buy-hold | -0.97% | +1.63% |
| **BOOK (flat 750k PnL%)** | **+1.93%** | +2.69% |

Looser cut (month-max >= 50, N=27) shows the same sign pattern (variants -0.1% to -1.0%, book
+2.15%). This is the kill shot: fragility-elevated months are down-beta months, and long-only
momentum rotation is 0.8-1.1 beta. Even the absolute-momentum overlay doesn't help because
6-12 month momentum turns far slower than the fragility dial — in Sep-Dec 2024 and Feb 2026 the
book of "winners" was still fully invested. The book itself already outperforms the candidate
sleeve in exactly the regime the sleeve was supposed to patch.

## (5) Execution fit

- **Instruments**: A1/A2 = 3 SPDR sector ETFs; trivially liquid, $150-250k sleeve is noise vs
  ADV. B = single-country ETFs, fine at this size but the investable local universe is only
  9-10 names (top-3-of-9 is barely a cross-section). C = 20 large caps, fine at this size.
- **Order types**: 12 rebalances/yr, ~0.75 name changes/month for sectors (25% 1-way turnover).
  MOC or next-open orders once a month; fits the existing staged-order workflow with near-zero
  operator load. No new data needed for sectors/countries (master_prices covers them).
- **Data the repo lacks**: a survivorship-clean single-stock universe with point-in-time
  membership (e.g. historical S&P 500 constituents + delisted price histories) — required
  before variant C could ever be evaluated honestly. Also more country ETFs (only 10 exist in
  the cache; a real country sleeve wants 20-45).
- **Capacity**: unconstrained at $750k for all variants.

## (6) Bias inventory (honest list)

1. **Survivorship (fatal for variant C)**: master_prices holds 1,114 tickers of which only 8
   have last-dates before 2026-06 — the universe is essentially today's members only. Worse,
   `LIQUID_UNIVERSE` was hand-curated recently from names that are large NOW (universe-selection
   lookahead on top of delisting bias). The +4-5%/yr "alpha" of variant C, concentrated in the
   recent period, is consistent with bias rather than edge. Treat C's numbers as an upper bound
   with unknown but large inflation. ETF variants (A, B) are mostly clean.
2. **Adjusted-price basis**: all levels here are relative and recomputed per run, so the
   dividend-adjustment invariant holds (CLAUDE.md rule); total-return basis is what you want
   for rotation. No frozen dollar levels involved.
3. **Transaction costs**: modeled at 5 bps/side ETFs, 15 bps/side stocks on L1 weight change;
   no market impact (irrelevant at this size), no borrow (long-only). Cost drag is small
   (28-30 bps/yr ETFs) — costs are not what kills this; lack of alpha is.
4. **RSX phantom prices**: cache carries flat post-halt prices to 2026; dropped after
   2022-03-31, which also means variant B dodges part of the realized Russia -100% (it eats
   Jan-Feb 2022 only if selected; it was not top-3 then). Real country rotation in 2022 could
   have been worse than shown.
5. **Fragility vintage**: rd2_fragility is a current-vintage reconstruction with the known
   composite-weight calibration lookahead (established caveat); the high-frag month labels
   inherit it.
6. **Book series is itself a backtest** (ledger PnL), so the correlation estimates share the
   engine's assumptions.
7. **Monthly granularity** hides intramonth momentum-crash dynamics (the Mar-2009 daily
   reversal was sharper than monthly bars show).
8. No significance claim survives clustering: sector alpha t-stats of 0.4-1.3 on monthly
   (already-clustered) returns are plainly indistinguishable from zero.

## Recommendation

Reject the sleeve. Cross-sectional long-only momentum rotation at this account is SPY beta
plus 25%/month turnover: no reliable net alpha in sectors (t <= 1.25 and decaying), negative
value in countries, and untrustworthy numbers in single stocks without a point-in-time
universe. Most importantly it is anti-correlated with the mandate: it averages -0.4% to -1.8%
per month in the 63d-MA10 >= 50 fragility months where the book needs help, while the book
itself averages +1.9% there. If a slow sleeve is still wanted, the evidence points away from
cross-sectional selection and toward defensiveness that reacts on the fragility dial's
timescale (the pending >= 50 sizing taper, and/or a hedge overlay per
tests/backtest_put_hedge.py prior art) — not this. If single-stock momentum is ever revisited,
first acquire survivorship-clean point-in-time constituent data; nothing in the current repo
supports an honest test.

## Adversarial verification

Verified 2026-07-02 by an independent recompute
(`scratch/ultracode_research/verify_momentum-rotation.py` + `verify_mr_extra.py` — fresh code,
only the rule spec taken from this report). Every decisive number reproduced within tolerance.

| Claim | Reported | Recomputed | Verdict |
|---|---|---|---|
| A1 sector net CAGR / Sharpe / corr vs SPY | 11.45% / 0.85 / 0.86 vs SPY 11.68% / 0.84 | 11.45% / 0.85 / 0.86 vs 11.68% / 0.84 (N=281, maxDD -38.0% vs -50.8%) | CONFIRMED |
| A1 alpha full then 2016-07+ | +1.85%/yr t=1.25 -> +0.85% t=0.41 | +1.92%/yr t=+1.25 -> +0.85% t=+0.39 | CONFIRMED |
| Hi-frag months (63d MA10 month-mean >= 50) | N=16; A1 -0.58%, B -1.78%, C -0.78%, book +1.93% | N=16, identical month list; A1 -0.58%, A2 -0.44%, B -1.78%, C -0.78%, SPY -0.97%, book +1.93% (rest +2.76%) | CONFIRMED |
| Country rotation Sharpe / maxDD | 0.58 / -66.8% | 0.58 / -66.8% (CAGR 10.62%) | CONFIRMED |
| Stock alpha 2013-2026 + survivorship count | +5.43%/yr t=2.09; 8 of 1,114 delisted | +5.43%/yr t=+1.99; exactly 8 of 1,114 tickers with last-date before 2026-06 | CONFIRMED |
| Corr to book monthly PnL | +0.16..+0.24, SPY +0.19 | A1 +0.193, A2 +0.196, B +0.152, C +0.236, SPY +0.194 (N=281) | CONFIRMED (B is 0.15, hair below the stated range low — immaterial) |
| Book's 12 worst months since 2016-07 | sleeves -1.0..-1.2%, SPY -1.3% | A1 -1.16%, A2 -1.51%, B -1.07%, C -1.01%, SPY -1.34% (book avg -2.06%) | CONFIRMED (A2 lands at -1.51%, worse than the stated range — strengthens the conclusion) |
| Cost drag / turnover, A1 | ~30 bps/yr at 25% 1-way/mo | 0.30%/yr at 25.3% 1-way/mo | CONFIRMED |

Adversarial probes that did NOT overturn anything:

1. **Is the book's hi-frag +1.93% an OVS artifact?** No. Ex-OVS the book still averages
   +1.61%/mo in the 16 hi-frag months (OVS contributes +0.32%). The book-vs-sleeve contrast
   is real, not the exempt strategy.
2. **Concentration of the sleeve's hi-frag loss.** A1's -0.58% average is driven by exactly
   two months (2018-10 -8.3%, 2020-02 -8.5%); the other 14 hi-frag months average +0.58%
   gross. So "loses money in high-fragility months" is really "carries full crash beta into
   the two crash months the dial flagged" — which IS the mandate failure, but a
   significance test on 16 clustered months (~6 episodes) would not clear any bar. The
   report never claims significance here, so this stands as descriptive.
3. **Looser cut (month-max >= 50, N=27)** reproduced: variants -0.07% to -1.00%, book +2.15%.
   Same sign pattern.
4. **Lookahead scan**: signal = month-end close, held next month — clean. RSX post-halt drop
   uses hindsight (report flags it; makes variant B look BETTER than realizable, and B still
   fails). 13-month eligibility gate handles XLRE/XLC entry. No cherry-picked window found:
   sample = full cache history minus the 13-month warmup, partial July bar excluded as stated.
5. **Known shared caveats stand** (flagged in the report): fragility labels are
   current-vintage with composite-weight calibration lookahead; book series is itself a
   backtest; variant C is unratable due to survivorship + universe-selection bias (the 8/1,114
   delisting count verifies — the cache is effectively today's members only).

Verdict: all eight decisive claims CONFIRMED; the reject recommendation survives adversarial
review. Minor nits only: full-sample A1 alpha is +1.92% not +1.85% (rounding/estimator
variant), 2016-07+ t=0.39 not 0.41, C 2013-2026 t=1.99 not 2.09 — all immaterial to the
conclusion, and none of the alphas clears significance anyway.
