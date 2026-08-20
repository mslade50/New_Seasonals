# Fundamental Research Foundations v2

## Purpose and evidence standard

This document defines the evidence base for the fundamental sleeve. Its job is
not to manufacture recommendations. Broad factor research supplies **base-rate
priors** for deciding where to investigate; it does not prove that an individual
security is mispriced, establish a target price, or authorize a position.

An investable stock thesis still requires company-specific primary evidence,
price-implied expectations, valuation support, a credible earnings path,
observable proof and kill conditions, financing and dilution analysis, and
portfolio fit. Missing material evidence is a failed gate, not a neutral score.

The academic literature is useful but easy to overstate. McLean and Pontiff
found that published return predictors were 26% weaker out of sample and 58%
weaker after publication. Hou, Xue, and Zhang found that most of 452 reported
anomalies failed common replication hurdles after controlling for microcap
effects and multiple testing. Jensen, Kelly, and Pedersen reached a less
pessimistic conclusion when signals were grouped into economically coherent
themes and tested internationally. The practical reconciliation is that broad
themes can be informative while precise formulas and backtested magnitudes are
fragile. See [Does Academic Research Destroy Stock Return Predictability?](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12365),
[Replicating Anomalies](https://academic.oup.com/rfs/article/33/5/2019/5236964),
and [Is There a Replication Crisis in Finance?](https://www.nber.org/papers/w28432).

## Durable signal families

### Profitability and economically defined quality

Profitability is the strongest fundamental starting point. Robert Novy-Marx
found that gross profitability had roughly the same cross-sectional predictive
power as book-to-market and materially improved value strategies, including
among large, liquid stocks. Later work found cash-based operating profitability
more predictive than accounting-profit measures and capable of absorbing much
of the accrual signal. See [The Other Side of Value](https://www.nber.org/papers/w15940)
and [Accruals, Cash Flows, and Operating Profitability](https://www.sciencedirect.com/science/article/pii/S0304405X16300307).

For this project, quality means evidence of:

- cash profitability and cash conversion, not adjusted EPS alone;
- durable or improving returns on incremental invested capital;
- a reinvestment runway with evidence that new capital earns attractive returns;
- balance-sheet resilience and manageable refinancing needs;
- disciplined investment, acquisition, payout, and dilution decisions; and
- accounting that reconciles earnings to cash and does not rely on serial
  exclusions.

Quality is not a generic stability score. A broad review of practitioner quality
definitions found support for profitability, accounting quality,
payout/dilution, and investment discipline, but much less evidence for several
popular proxies such as earnings stability or simple growth in profitability.
See [What Is Quality?](https://www.tandfonline.com/doi/full/10.1080/0015198X.2019.1567194).
AQR's [Quality Minus Junk](https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk)
supports quality as a diversified return theme; it does not imply that every
high-quality company is attractive at every price.

### Valuation, conditioned on quality and accounting reality

Valuation remains necessary because the price paid determines the return
available, but simple cheapness is not sufficient. Traditional U.S. book value
has been especially weak in recent decades. The project should prefer multiple
economically relevant lenses:

- enterprise value to normalized operating earnings or cash flow;
- owner-earnings and free-cash-flow yield after required reinvestment;
- normalized earnings power for cyclical companies;
- asset value only when assets are economically realizable;
- sector-specific methods for financials, REITs, and biotech; and
- a reverse DCF or other price-implied-expectations analysis.

The central question is not whether a multiple is below history. It is what
revenue growth, margins, reinvestment returns, duration, capital intensity, and
terminal economics the current price already discounts, and whether primary
evidence supports a better path.

Value should normally be paired with quality, revisions, or a catalyst. A low
multiple combined with falling earnings, heavy issuance, aggressive investment,
or deteriorating economics is more likely to be a value trap than a bargain.
Asset growth has predicted lower subsequent returns in broad samples, including
large companies, and changes in shares outstanding contain information about
future returns. See [Asset Growth and the Cross-Section of Stock Returns](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2008.01370.x)
and the Federal Reserve's [Why Does the Change in Shares Predict Stock Returns?](https://www.federalreserve.gov/econres/feds/why-does-the-change-in-shares-predict-stock-returns.htm).

### Earnings revisions and post-event underreaction

Earnings information is a realization signal rather than a substitute for an
underwrite. Classic evidence found that past price momentum and past earnings
news independently predicted subsequent drift and that analyst estimates
adjusted gradually. See [Momentum Strategies](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1996.tb05222.x).
The phenomenon remains well documented, though its magnitude varies with
liquidity, attention, trading costs, and implementation; see the modern
[review of post-earnings-announcement drift](https://www.sciencedirect.com/science/article/pii/S2214635020303750).

The system should therefore track:

- 30- and 90-day changes in forward revenue, EBITDA, EPS, and FCF estimates;
- revision breadth, not just the consensus mean;
- guidance changes relative to the expectation before the release;
- whether revenue, margins, cash flow, and operating KPIs confirm one another;
- the initial price response and subsequent relative strength; and
- contradictions between the headline surprise and the forward estimate path.

Analyst recommendations and consensus price targets are not primary signals.
They are useful for mapping expectations and disagreement, but evidence shows
that post-recommendation return drift weakened materially in the modern era.
See [Can Analysts Pick Stocks for the Long Run?](https://www.sciencedirect.com/science/article/pii/S0304405X15001713).

### Price momentum and trend

Recent price trends are among the most consistently important predictors in
modern cross-sectional machine-learning research. See
[Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276).
Time-series momentum also has evidence across equity indices, bonds, currencies,
and commodities and across unusually long histories. See
[Time Series Momentum](https://research-api.cbs.dk/ws/portalfiles/portal/58851003/time_series_momentum_lasse_heje.pdf)
and [A Century of Evidence on Trend-Following Investing](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing).

That evidence does not prove that an exact 200-day moving-average crossing is a
standalone alpha rule for individual stocks. Data-snooping-aware work has found
no reliable moving-average timing rule that consistently beats buy-and-hold at
standard significance thresholds. See
[Tactical Asset Allocation on Technical Trading Rules and Data Snooping](https://www.sciencedirect.com/science/article/pii/S0927538X18300775).

Use the 200-day average as a slow risk and timing overlay:

- **Green:** price above the 200-day average and the average has a positive
  20-trading-day slope.
- **Amber:** price is within 5% of the average, or price and slope disagree.
- **Red:** price is below a falling 200-day average.

Do not react to one crossover. Full research candidates should normally be
Green. Amber can support a conditional candidate when primary evidence and a
hard catalyst are strong. Red normally means wait for proof. A fundamental kill
condition overrides a Green trend; Red plus falling estimates is the strongest
deterioration combination.

The 200-week average is too slow to be an entry trigger for this sleeve. It is a
structural-damage flag only. A company below a falling 200-week average must be
handled as a turnaround or bounded special situation with a hard catalyst and
specialist underwriting; otherwise it should be passed.

## What has decayed or become less reliable

The last 10-20 years changed the implementation more than the underlying logic:

1. **Simple anomalies became crowded and faster.** Cheap public data and
   systematic capital reduced the payoff from obvious one-variable rules.
2. **Book accounting became less representative.** R&D, software, brand, and
   customer-acquisition investment are commonly expensed, distorting book value,
   reported profitability, and investment. Intangible-adjusted characteristics
   improve recent factor models; see
   [Intangible Capital in Factor Models](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.01261).
3. **Headline earnings became easier to manage.** Stock-based compensation,
   acquisition adjustments, restructuring exclusions, and buybacks require
   reconciliation to cash flow and diluted share count.
4. **Information is incorporated faster.** The useful revision signal is the
   forward path after a release, not merely the reported beat.
5. **Individual-stock outcomes remain extremely skewed.** A small fraction of
   stocks creates most long-run wealth, making premature profit-taking and
   concentrated false positives both dangerous. See
   [Do Stocks Outperform Treasury Bills?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2900447).

Signals that should not promote a company on their own include low P/E or P/B,
high historical growth, a founder CEO, brand familiarity, an announced buyback,
a single insider purchase, analyst target-price upside, a one-day moving-average
cross, or a generic moat label.

## The v2 decision model

The process should require two independent keys:

### Key 1: fundamental edge

- source-backed variant perception;
- durable economics or a measurable improvement path;
- conservative valuation support under at least two relevant methods;
- explicit downside mechanism and stress value; and
- acceptable balance-sheet, accounting, financing, and dilution risk.

### Key 2: realization edge

At least two of:

- improving or inflecting forward estimates;
- a dated or observable catalyst/evidence window; and
- Green trend confirmation.

High-quality compounders may use recurring earnings delivery and estimate
progression as the realization path. Deep value, cyclicals, and turnarounds need
a harder catalyst. A screen score may prioritize diligence, but only a complete,
sourced underwrite can turn both keys and produce a `QUICK REVIEW`.

Every review must answer, in order:

1. What is mispriced?
2. What is already priced in?
3. Which operating and estimate path makes the stock work?
4. What proves it, and by when?
5. How is money lost?
6. What kills the thesis?
7. What evidence is still missing?

## Practitioner practices worth emulating

Copy process elements, not portfolios or reputation.

- **Warren Buffett and Berkshire Hathaway:** business-owner economics,
  capital-allocation discipline, patience, and concentrated action when the odds
  are favorable. Berkshire reports a 19.7% compounded annual gain in per-share
  market value from 1965-2025 versus 10.5% for the S&P 500, but insurance float,
  operating subsidiaries, acquisitions, and tax structure make the vehicle
  impossible to copy directly. See the
  [2025 Berkshire shareholder letter](https://www.berkshirehathaway.com/letters/2025ltr.pdf).
- **Nick Sleep and Qais Zakaria:** destination analysis, scale economies shared
  with customers, long holding periods, and explicit learning from mistakes.
  Their final Nomad letter reported 20.8% annualized before performance fees
  versus 6.5% for MSCI World and clearly labeled the results unaudited. See the
  [Nomad letters](https://igyfoundation.org.uk/wp-content/uploads/2021/03/Full_Collection_Nomad_Letters.pdf).
- **Terry Smith and Fundsmith:** high operating returns, cash conversion,
  durable advantages, modest leverage, reinvestment, valuation discipline, and
  low turnover. The useful lesson is quality plus price, including the risk of
  overpaying for quality. See the
  [2025 Fundsmith semiannual letter](https://www.fundsmith.co.uk/media/bvgden5v/2025-fef-semi-annual-letter-to-shareholders.pdf).
- **Bill Ackman and Pershing Square:** concentrated underwriting, catalyst
  creation, scenario work, active governance, and public mistake review. The
  uneven public-vehicle record is also a warning about path risk and long
  drawdowns in concentrated catalyst strategies. See the
  [2024 Pershing Square annual report](https://assets.pershingsquareholdings.com/2025/03/14150847/Pershing-Square-Holdings-Ltd.-2024-Annual-Report.pdf).
- **Cliff Asness and AQR:** systematic falsification, explicit factor
  definitions, and complementary value and momentum evidence. Emulate the
  measurement discipline, not a proprietary factor portfolio. See
  [Value and Momentum Everywhere](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021).

## Anti-patterns the skill must prevent

- Treating a screen rank as a recommendation.
- Applying one generic score to standard companies, financials, REITs, and
  biotech.
- Calling a stock cheap from one historical multiple.
- Using current cyclical earnings as normalized earnings.
- Treating management guidance, adjusted EPS, or a buyback authorization as
  verified economic improvement.
- Ignoring leases, stock compensation, dilution, pensions, acquisition spending,
  or debt maturities.
- Giving a turnaround the same no-catalyst patience as a proven compounder.
- Using precise DCF targets without a reverse-DCF expectations test and scenario
  range.
- Retuning the model after a few recent misses or accepting a backtest without
  point-in-time data, delistings, liquidity, and transaction costs.
- Producing a long queue for the reader instead of automatically passing weak or
  unproven ideas.
- Filling a ten-position capacity target when no company clears the bar.

The durable operating principle is: **search broadly, reject automatically,
underwrite by archetype, require both fundamental and realization evidence, and
surface only changed decisions.**
