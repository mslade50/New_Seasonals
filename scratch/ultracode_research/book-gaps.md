# Book Gap Map — what any new sleeve must complement

Track: `book-gaps` (integration analyst). Run date 2026-07-02.
Code: `scratch/ultracode_research/book_gap_map.py`, `scratch/ultracode_research/book_gap_followups.py`.
Shared artifact for the other tracks: `scratch/ultracode_research/book_monthly_series.csv`
(index = exit month, cols: `pnl`, `ret_pct`, `R`, `n_exits`, `frag_avg`, `frag_max`,
`spy_ret_pct`, `spy_dd_pct`). Correlate your sleeve's monthly returns against `ret_pct` (or `R`).

Data: `data/backtest_trades_full.parquet` (3,286 trades, 2003-01..2026-07, flat $750k basis,
GROSS of the live fragility multiplier), `data/rd2_fragility.parquet` (63d col, 10d MA = live
basis, 2016-07+), `data/master_prices.parquet` (SPY, adjusted).

## Headline: the fragility dial does not flag the book's worst months

The brief's premise was that the book's weakness is the top fragility decile. That is true
**per-trade** (established: non-OVS avgR +0.17 at band 55+) but false **per-month in dollars**:

- High-frag months (monthly avg frag ≥ 50, N=16 of 119 since 2016-07): book avg **+$14,462/mo,
  75% positive**. Non-OVS alone: +$12,069/mo, 81% positive. The book is fine there — fewer
  good trades, but OVS and the survivors carry the month.
- The book's worst-decile months (ret ≤ −1.20%, N=29 all-time, 13 since 2016-07) have
  **median frag_avg = 19** (base-rate median 20.6). Of the 10 worst months ever, only one
  (2024-04, frag 44.8) was even near the throttle zone; 2026-07, 2019-10 sat at frag 16-21.
- Monthly fragility *change* also doesn't discriminate: frag-rising-≥10 months avg +$21.4k
  (t=+0.34 vs rest).

So a sleeve pitched purely as "makes money when frag ≥ 55" solves a per-trade sizing problem
the pending multiplier rec already addresses. The **dollar** gap lives elsewhere — see §1.

## 1. Monthly series and the 10 worst months

Book monthly stats (283 months, flat $750k, by exit month, gross):

| metric | value |
|---|---|
| avg monthly PnL | $13,163 (+1.76% of 750k) → ~$158k/yr |
| monthly vol | 2.81% → 9.7% ann. |
| Sharpe (ann., 0% hurdle, flat basis) | **2.16** full-sample; 2.77 since 2016-07 |
| max drawdown (flat cumsum) | −$65,604 (−8.75%) |
| % months positive | 76.0% |
| total | $3.73M, +1,399R |

10 worst months with regime context:

| month | PnL | R | frag_avg | SPY mo ret | SPY DD (from ATH, adj) | character |
|---|---|---|---|---|---|---|
| 2015-08 | −$59,403 | −12.3 | n/a | −6.1% | −11.9% | flash-crash from calm; Indices OS Bounce −$44k |
| 2008-01 | −$48,306 | −13.1 | n/a | −6.1% | −16.0% | bear onset; St OS Sznl −$20k |
| 2012-05 | −$40,944 | −12.9 | n/a | −6.0% | −8.7% | calm→correction; St OS Sznl −$27k |
| 2026-07* | −$32,887 | −9.8 | 20.7 | −0.1% | −1.6% | *1 trading day, partial.* OLV single-stock cluster −$30k |
| 2024-04 | −$31,587 | −7.8 | 44.8 | −4.0% | −5.4% | mid-frag correction; St OS Sznl −$24k |
| 2004-03 | −$28,244 | −7.8 | n/a | −1.3% | −24.6%† | chop |
| 2019-10 | −$25,664 | −1.2 | 15.8 | +2.2% | −4.2% | SPY UP month; Weak Close −$31k, idiosyncratic |
| 2012-07 | −$22,739 | −6.1 | n/a | +1.2% | −5.5% | chop |
| 2006-05 | −$20,391 | −7.2 | n/a | −3.0% | −10.6% | Indices OS Bounce −$25k |
| 2026-06 | −$19,383 | −9.6 | 30.1 | −1.0% | −4.5% | OLV single-stock cluster −$38k (OVS +$24k saved it) |

† DD measured vs adjusted all-time high — 2004 was still under the 2000 peak; read as "post-bear recovery", not a live 24% drawdown.

Strategy attribution across the 10 worst months (totals): Indices Oversold Bounce −$82k,
OLV −$75k, St OS Sznl −$71k, 52wh Breakout −$41k, Weak Close −$40k. **OVS was net +$14k in
the worst months** — it is already the book's internal hedge. The losers are the dip-buyers
catching knives in fast 4-6% corrections, plus (new in 2026) OLV overflow single-stock
clusters: OLV lost **−$67.8k across Jun 1–Jul 1 2026** at frag 20-30.

Book beta to market: monthly corr to SPY **+0.196**, beta +0.13, downside-only corr (SPY≤0
months) **+0.085**. In the 31 SPY ≤ −4% months the book averaged **+0.04%** (61% positive) —
big bear months (2008-09/10, 2020-03, 2022) were mostly fine or great. The damage profile is
narrower: **the first fast leg down out of calm** (2015-08, 2012-05, 2008-01, 2024-04) and
**idiosyncratic dip-buyer clusters** (2019-10, 2026-06/07).

## 2. Exposure cadence — the book piles in as fragility rises

Daily open-trade count / gross notional / open risk (% of 750k), joined to frag MA10, 2016-07+:

| frag band | days | avg open trades | avg gross % | p90 gross % | avg open risk % | new sigs/day | non-OVS open |
|---|---|---|---|---|---|---|---|
| 0-25 | 1447 | 3.75 | 49.7 | 125.6 | 1.47 | 0.75 | 2.68 |
| 25-44 | 606 | 3.88 | 62.4 | 142.1 | 1.58 | 0.76 | 3.02 |
| 44-50 | 136 | 4.27 | 67.3 | 144.0 | 1.74 | 0.73 | 3.65 |
| 50-55 | 104 | 5.44 | **88.4** | **173.2** | 2.18 | 0.88 | 4.71 |
| 55+ | 253 | 3.26 | 55.3 | 123.3 | 1.34 | 0.85 | 2.71 |

The unthrottled book **maxes gross exactly in the 50-55 band** (88% avg gross, 173% p90 —
1.8x the calm-band exposure) where per-trade R is already degrading (+0.36 established), then
thins out at 55+ only because signals exit fast. It does NOT naturally de-gross; the live
multiplier is doing real work, and a new sleeve must not add notional in the 44-55 band.
Capacity context: full-sample avg gross 52%, p90 123%, **max 458%** (OVS spike days, 86
concurrent trades in 2020). 17% of all days have zero open trades — capital idles in calm.

## 3. Seasonality

By calendar month (avg monthly PnL, 23-24 obs each): strong Nov–Mar ($15.2–18.0k), soft
**Jul–Sep ($8.6–11.0k avg, $9.7k pooled vs $14.3k for the other nine months)**. June also
soft ($10.7k). By presidential cycle year:

| cycle | avg mo PnL | pos frac | n months |
|---|---|---|---|
| post-election | $18,170 | 0.82 | 72 |
| election | $12,742 | 0.69 | 72 |
| pre-election | $11,618 | 0.86 | 72 |
| **midterm** | **$9,896** | 0.66 | 67 |

Midterm years (2026 is one) run at ~54% of post-election run-rate — consistent with the OVS
cycle tilt already shipped. A sleeve that carries Jul–Sep and midterm years fills a real,
recurring hole.

## 4. Concentration

- By strategy: OVS 19.3% of PnL, MonFri 13.6%, OLV 11.7%, 52wh BO 10.4% — no single point of
  failure, but ALL twelve are short-horizon US-equity strategies.
- By ticker family: single stocks 52.2% of PnL (709 tickers — survivorship-inflated, see §6),
  SPY/QQQ/index-core 26.1%, sector ETF 12.2%, 3x levered 6.0%, **commodity/bond/FX 2.6%,
  intl 1.0%**. Top-4 tickers (SPY/QQQ/^NDX/^GSPC) = 22.9% of PnL; top-10 = 32.3%; ticker HHI
  0.021 (effective N ≈ 47). Long 68% of PnL, short 32% (mostly OVS fades).
- **Horizon is the true concentration**: median hold 2 business days, p90 = 8, only 5.7% of
  trades held >15bd (they carry 24.2% of PnL). Round-trip notional ≈ $40M/yr = 53x account.
  There is genuinely no slow sleeve, no non-equity risk, no overnight-session diversification.

## 5. The spec any new sleeve must satisfy (scoring rubric for the other four tracks)

Marginal-Sharpe math: adding a sleeve at small size improves book Sharpe iff
S_sleeve > ρ × S_book. With S_book ≈ 2.16 (flat, gross), that is brutal: ρ=0.2 already
demands S_sleeve > 0.43; ρ=0.3 demands > 0.65. From the combined grid (sleeve vol = f × book
vol): a Sharpe-0.5 sleeve at ρ=0, f=0.25 lifts 2.16 → 2.22; at f=0.50 it is breakeven; at
ρ=0.3 it dilutes at any size. **Conclusion: almost no realistic slow sleeve improves the
book's Sharpe ratio.** The honest justification channels are (a) dollars on idle capital
(avg open risk is only 1.5% of NAV; avg gross 52%), (b) hedging the specific worst-month
profile, (c) filling Jul–Sep / midterm dead zones. Score proposals on:

1. **Worst-month profile (weight 30%).** Avg PnL must be ≥ 0 in the book's 29 worst-decile
   months and in the 31 SPY ≤ −4% months (book: +0.04%). Bonus if positive in the four
   knife-catch archetypes (2015-08, 2012-05, 2024-04, 2019-10 analog). "Works at frag ≥ 55"
   earns LITTLE credit by itself — the book already survives those months (+$14.5k/mo avg)
   and the pending multiplier fix handles per-trade sizing.
2. **Standalone quality after costs (25%).** Monthly Sharpe ≥ 0.5 net of ≥5 bps/side ETF /
   more on stocks, 2003+ (or max history), with LOYO stability and monthly-clustered stats.
   Below 0.5 the operator attention cost isn't paid.
3. **Correlation (20%).** |ρ| ≤ 0.25 vs book monthly `ret_pct` (2003+ where possible; report
   2016+ separately). Also require the sleeve's own SPY monthly beta ≤ ~0.3 — the book is
   nearly market-neutral (corr 0.20) and a beta-1 sleeve would dominate combined vol.
4. **Capacity & execution fit (15%).** Livable inside ~30-40% average gross notional at
   $750k without competing for margin on OVS spike days (book p90 gross 123%, max 458%);
   monthly rebalance or slower; IBKR-stageable order types (limit/MOO/MOC); instruments the
   repo can price daily (master_prices covers ~2000 US tickers — intl/futures need new data).
5. **Dead-zone fill (10%).** Positive expectancy in Jul–Sep and in midterm years.
6. **Materiality gate (pass/fail).** Expected ≥ ~$25-30k/yr net (≥2% of NAV, ≈15-20% of book
   run-rate) at the allowed risk. Below that, "not worth it" is the correct verdict for a
   solo operator.

Capital share guidance: size the sleeve to 25-40% of book monthly vol (≈0.7-1.1%/mo of NAV,
$5-8k monthly vol) — the grid shows f > 0.5 only helps if the sleeve is genuinely ρ ≤ 0 and
Sharpe ≥ 0.8, which should be presumed false until proven.

## 6. Biases and caveats in this baseline

- **Survivorship**: 709 single-stock tickers = today's universe membership; the 52% single-
  stock PnL share (esp. Overflow tier, $1.56M) is overstated to an unknown degree. The
  index/ETF share (~47%) is the reliable core. Any sleeve comparison should apply the same
  skepticism to its own stock-level results.
- **Fragility reconstruction**: current-vintage; composite edge weights have calibration
  lookahead (established caveat). Pre-2016 worst months have no frag reading at all — 5 of
  the 10 worst are unscored, so the "worst months are low-frag" claim rests on the 2016+
  half (5 of 5 scored worst months at frag ≤ 45) plus the worst-decile median (19 vs base 20.6).
- **Flat basis, gross**: ledger is flat $750k, no fragility multiplier, no financing cost on
  the >100% gross days, 0% cash hurdle in the Sharpe. Stop fills include the 3/13 bps slippage
  model (2026-06-27 convention) but entries/targets carry no cost. Live/backtest divergence:
  OVS P2 retired live but modeled here.
- **Exit-month attribution** lumps a multi-week trade's whole PnL into its exit month;
  fine at median hold 2bd, slightly smears the 5.7% of long holds.
- **2026-07 is one trading day** (21 exits, −$32.9k); it will move. The 2026 OLV cluster
  (−$68k in 5 weeks) is live out-of-sample pain, worth its own review outside this track.
- Monthly Sharpe of 2.16 on a flat notional base is not comparable to a fund Sharpe — the
  denominator is realized PnL vol on a fixed $750k, with avg gross only 52% and 17% of days
  flat. Treat it as an internal yardstick only; it is still the correct hurdle for the
  marginal-Sharpe condition because sleeve candidates are measured on the same basis.

## Adversarial verification (2026-07-02, independent recompute)

Verifier script: `scratch/ultracode_research/verify_book-gaps.py` (fresh implementation,
nothing reused from `book_gap_map.py`; supplementary `verify_bg_cbfx.py`). Verdict:
**all ten decisive claims reproduce**; two cosmetic label errors noted below.

| claim | their number | my recompute | verdict |
|---|---|---|---|
| Monthly stats | 283mo, $13,163/mo, 2.81% vol, Sharpe 2.16 (2.77 since 2016), maxDD -$65,604 (-8.75%), 76% pos | 282mo, $13,210/mo, 2.81%, 2.17 (2.77), -$65,604 (-8.75%), 76.2% | CONFIRMED |
| High-frag months not the gap | N=16, +$14,462, 75% pos; non-OVS +$12,069, 81% | exact match (16 / $14,462 / 75.0%; $12,069 / 81.2%) | CONFIRMED |
| Worst-decile months low-frag | median 19 vs base 20.6; 1 of 10 worst >44 | thr -1.20%, N=29 (13 scored), median 19.0 vs 20.6; only 2024-04 (44.8) >44 | CONFIRMED |
| Market-neutral monthly | corr .196, downside .085, +0.04% in 31 SPY<=-4% months | .196 / .085 / +0.04% (61.3% pos), beta 0.13 — exact | CONFIRMED |
| Worst-10 attribution | IndOSB -$82k, OLV -$75k, StOSSznl -$71k, OVS +$14k | -$82.5k / -$75.1k / -$71.0k / +$14.4k | CONFIRMED |
| Gross piles into 50-55 band | 88.4% avg / 173.2% p90 vs 49.7% / 125.6% at 0-25 | 89.7% / 175.1% vs 50.2% / 126.1% (day counts differ ~3% — calendar choice); pattern survives excl-exit-day sensitivity (65.5% vs 39.7%) | CONFIRMED |
| Seasonal dead zones | Jul-Sep $9.7k vs $14.3k; midterm $9.9k vs post-elec $18.2k | $9,814 vs $14,310; midterm $10,046 (N=66; their $9,896/N=67 implies one zero-exit month padded in — same substance); post-elec $18,170 exact | CONFIRMED |
| Horizon concentration | med 2bd, 5.7% >15bd (24.2% PnL), 53x/yr, CBFX 2.6% | med 2 / p90 8 / 5.7% / 24.2% / 53x exact; CBFX 2.7% narrow, 3.4% incl. miner ETFs (classification-fuzzy but "tiny" holds) | CONFIRMED |
| Marginal-Sharpe hurdle | S > rho x 2.16; rho=.3 -> .65; Sharpe-.5 at rho=0,f=.25 -> 2.22 | standard result; grid: 2.23 at f=.25, breakeven-to-dilutive at f=.5 (2.165 vs 2.17), 2.08 at rho=.3 — matches | CONFIRMED |
| Materiality gate | >= $25-30k/yr "(2% of NAV)", 25-40% of book vol, rho<=.25, beta<=.3 | vol fraction math checks (2.81% x .25-.40 = 0.70-1.12%/mo); **"2% of NAV" is a mislabel — $25-30k on $750k is 3.3-4%**; the 15-20%-of-run-rate anchor is correct. Normative thresholds otherwise unfalsifiable | CONFIRMED (with label fix) |

Notes and residual weaknesses (none overturn the conclusions):
- **The frag-scoring of the worst months rests on a thin scored half.** 6 of the 10 worst
  months are pre-2016 (unscored), not 5 as the caveat text says; 4 of 4 scored are <=44.8.
  The worst-decile median (13 scored months) is the stronger leg and reproduces exactly.
- 2026-07 (1 trading day, -$32.9k) is inside the 282/283-month sample, the worst-10, and
  the worst decile. Dropping it: avg $13,374/mo, Sharpe 2.21, decile thr -1.17% — no
  conclusion changes, but the month will move.
- N=16 high-frag months is small; a 75% hit rate has a wide CI. The claim as stated
  ("high-frag months are not the dollar gap") is directional and survives, especially
  next to the established 2026 inversion.
- Exposure-band day counts (mine 2461 vs their 2546) differ by calendar construction
  (I used only days with a frag print); band means/p90s agree to ~1pt, ordering identical.
- All inherited caveats stand: flat/gross basis, current-vintage fragility, exit-month
  attribution, survivorship in the 52% single-stock share, entries/targets uncosted.
