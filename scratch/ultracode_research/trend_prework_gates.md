# Trend-Following Pilot — Step 7 Pre-Work Gates

Run date: 2026-07-02. Code: `scratch/ultracode_research/gate_ab.py` (both gates +
re-verification, fresh engine reimplemented from the §1 rules of
`trend-following.md`), `gate_b_followup.py` (year-clustered book checks, combined
effect), `gate_inspect.py` (data format checks). Saved series:
`gate_ab_series.parquet` (ex-bonds close/open, full-16 close, book %/R monthly).

Data: `data/master_prices.parquet` (long format, ticker/date/OHLCV, adjusted;
all 12 ex-bonds ETFs + ^IRX verified present, Open column 0% NaN, data through
2026-07-01), `data/backtest_trades_full.parquet` (3,286 trades),
`data/rd2_fragility.parquet` (63d dial, 2016-07-05+).

All results below exclude the partial 2026-07 month (one trading day). Note: the
prototype's headline "N=303" **includes** that partial July row
(`tf_monthly_series.parquet` ends 2026-07-31); impact is trivial — my N=302
recompute matches the prototype to the second decimal — but it should not be
called 303 full months.

---

## GATE A — next-open execution (the roadmap's formal go-live gate)

Setup: identical rules to the prototype (combo = 12-1 momentum sign AND price >
10-month MA, long/flat, slot inverse-vol capped 20%, cash at ^IRX/12, 5 bps per
side on target-weight turnover). Only the execution leg changes: signals are
still computed on the month-end close, but the position transitions at the
**next trading day's open**; asset period returns become open-to-open at the
first trading day of each month (the old weights correctly earn the
month-end-close → next-open overnight gap). Same cost model on both legs.

### Headline (net of 5 bps/side), 2001-05..2026-06, N=302 months

| Spec | Exec | CAGR | Vol | Sharpe | MaxDD | excess t |
|---|---|---|---|---|---|---|
| **EX-BONDS 12-ETF (pilot spec)** | same-close | 6.79% | 6.33% | **0.80** | -10.4% | 4.04 |
| **EX-BONDS 12-ETF (pilot spec)** | next-open | 6.52% | 6.52% | **0.74** | -10.5% | 3.74 |
| Full-16 (primary spec) | same-close | 5.24% | 4.04% | 0.86 | -4.5% | 4.35 |
| Full-16 (primary spec) | next-open | 5.10% | 4.12% | 0.81 | -4.8% | 4.11 |

Same-close recompute matches the prototype (they reported ex-bonds
6.75/6.32/0.79/-10.4 and full-16 5.22/4.03/0.86/-4.5 including the partial July),
so the comparison is apples-to-apples.

- Paired monthly difference (open − close), ex-bonds: **-2.0 bps/mo**, paired
  t = -0.82 (N=302, not significant); series correlation 0.975. Full-16:
  -1.2 bps/mo, t = -0.72, corr 0.972.
- Sharpe delta: ex-bonds **0.80 → 0.74 (-0.06)**; full-16 0.86 → 0.81 (-0.05).
- The drop decomposes as a small negative timing drift (~2 bps/mo, statistically
  zero) plus slightly higher vol (open-to-open returns are noisier). No single
  year dominates: worst per-year slippage -1.8pp (2008, 2010), best +1.3pp
  (2025); 11 of 26 years the next-open leg is *better*.

### Per-year net returns, ex-bonds (same-close | next-open)

| Year | Close | Open | | Year | Close | Open |
|---|---|---|---|---|---|---|
| 2001* | +2.0% | +1.6% | | 2014 | +3.1% | +3.3% |
| 2002 | +1.1% | +1.1% | | 2015 | +1.2% | -0.3% |
| 2003 | +16.0% | +16.5% | | 2016 | +0.3% | +0.8% |
| 2004 | +10.0% | +9.4% | | 2017 | +11.8% | +11.7% |
| 2005 | +7.2% | +8.1% | | 2018 | +0.9% | +0.2% |
| 2006 | +23.9% | +23.5% | | 2019 | +5.8% | +5.5% |
| 2007 | +16.6% | +16.4% | | 2020 | +4.3% | +4.4% |
| 2008 | +2.1% | +0.3% | | 2021 | +9.8% | +9.3% |
| 2009 | +7.5% | +8.2% | | 2022 | +0.5% | +0.6% |
| 2010 | +4.6% | +2.7% | | 2023 | +1.5% | +1.3% |
| 2011 | -1.9% | -2.2% | | 2024 | +10.2% | +10.4% |
| 2012 | +2.3% | +3.0% | | 2025 | +17.2% | +18.5% |
| 2013 | +8.8% | +7.3% | | 2026 YTD | +8.7% | +7.1% |

*2001 = May-Dec. Note the ex-bonds variant has ONE negative year (2011, -1.9%),
unlike the full-16's clean sheet, and 2008 crisis alpha shrinks to +2.1% close /
+0.3% open (the +8.0% in the proposal's §3 table was the full-16 spec — the
bonds did most of the 2008 work). 2022 holds at +0.5/+0.6%.

### Verdict: **GATE A PASSES.**

Next-open Sharpe 0.74 vs same-close 0.80 — a delta of 0.06, inside the ~0.1
gate, and the underlying monthly slippage is statistically indistinguishable
from zero. The pass is not marginal: full-16 shows the same ~0.05 delta, and the
0.55 full-month-delay bound from the proposal correctly bracketed it from far
below. Next-open (MOO / TIF=OPG) execution is viable; use **Sharpe ~0.74,
CAGR ~6.5%, vol ~6.5%, maxDD ~-10.5%** as the honest planning numbers for the
ex-bonds pilot, not the same-close 0.80/6.8%.

---

## GATE B — dead-zone fill (Jul-Sep months, midterm years)

Sleeve = monthly net returns from the recompute (window 2001-05..2026-06,
N=302; the dead-zone masks below are evaluated on that window for the sleeve and
on 2003-01..2026-06, N=282 complete months, for the book). Book = ledger
`PnL_flat_750k` summed by exit month / 750k, and separately summed `R_Multiple`
by exit month.

### The book's dead zones, quantified (N=282 months)

| Window | N | avg %NAV/mo | hit | avg R/mo | R vs other | Welch t (monthly) |
|---|---|---|---|---|---|---|
| Overall | 282 | +1.78% | 76% | +5.00R | — | — |
| Jul-Sep | 69 | +1.37% | 78% | +3.53R | vs +5.47R | t=-2.02, p=0.044 |
| Midterm years | 66 | +1.41% | 67% | +3.13R | vs +5.56R | t=-2.36, p=0.019 |
| Jul-Sep × midterm | 15 | **+0.53%** | 67% | **+1.46R** | vs +5.19R | t=-4.30, p<0.001 |

Year-clustered (the honest unit for the midterm claim): midterm years average
+34.5R/yr vs +66.8R/yr in other years (N=6/18, Welch t=-2.17, **p=0.042**).
Jul-Sep quarters average +10.6R vs +16.2R for an average other quarter (paired
by year, N=23, t=-2.03). So the dead zones are real, though the book is still
*positive* in all of them — these are soft patches, not losses.

### Does the sleeve fill them? (ex-bonds pilot spec)

| Series | Overall avg/mo (hit) | Jul-Sep | Midterm yrs | Jul-Sep × midterm |
|---|---|---|---|---|
| Ex-bonds, same-close (N=302) | +0.57% (70%) | +0.40% (71%), N=75 | +0.53% (72%), N=78 | +0.21% (78%), N=18 |
| Ex-bonds, next-open (N=302) | +0.55% (69%) | +0.35% (65%), N=75 | +0.48% (73%), N=78 | +0.17% (72%), N=18 |
| Full-16, same-close (N=302) | +0.43% (73%) | +0.45% (73%), N=75 | +0.36% (77%), N=78 | +0.31% (89%), N=18 |

None of the sleeve's in-window vs out-of-window differences are significant
(Welch t between -1.45 and +0.15, all p > 0.15). Year-level: sleeve midterm
years average +6.1%/yr vs +7.0%/yr other (N=7/19, p=0.80) — the sleeve does
**not** share the book's midterm-year weakness in any measurable way, and its
2022 (+0.5%) and 2026 YTD (+8.7%) midterm rows are fine.

Combined effect at 1.0x NAV (next-open sleeve + book, % of 750k, 2003+ window):

| Window | Book alone | + sleeve | Sleeve adds | corr(sleeve, book) in window |
|---|---|---|---|---|
| Jul-Sep (N=69) | +1.37%/mo (78% hit) | +1.74%/mo (80%) | +0.37% | +0.10 |
| Midterm (N=66) | +1.41%/mo (67%) | +1.96%/mo (71%) | +0.55% | +0.14 |
| Intersection (N=15) | +0.53%/mo (67%) | +0.70%/mo (73%) | +0.17% | +0.17 |

### Verdict: **GATE B is a PARTIAL PASS — positive carry, not a counter-cyclical fill.**

Honest summary: the sleeve stays positive with ~65-78% hit rates in every dead
window and is uncorrelated to the book there (+0.10 to +0.17), so every dollar
it adds is additive. But it is not counter-cyclical: it earns *below* its own
average in exactly the deepest hole — in Jul-Sep × midterm it delivers +0.17%/mo
(next-open) vs +0.55% overall, filling only **~13%** of the book's ~$9.9k/mo
intersection shortfall at 1.0x NAV (~$1.3k/mo). At the roadmap's 0.5x pilot,
halve that. Note the irony: the **full-16** variant (with bonds) is the better
dead-zone filler (+0.45%/mo in Jul-Sep, +0.31% with an 89% hit in the
intersection, N=18) because bonds carry late summer — the ex-bonds choice trades
away some dead-zone fill for the better post-2020 standalone Sharpe. N=15-18
intersection months means none of these window numbers deserve a second decimal
of confidence.

---

## Re-verification of the two integration numbers

**1. Correlation of sleeve monthly returns to book monthly returns.**
Recomputed on 2003-01..2026-06, N=282 complete months (book by exit month /
750k, zero-filled empty months):

- Full-16 same-close: **+0.117** — matches the reported +0.117 exactly. CONFIRMED.
- Caveat that matters for the pilot: the headline belongs to the *full-16*
  spec. The **ex-bonds pilot spec is +0.166** same-close, **+0.157** next-open —
  still comfortably under the rubric's |rho| <= 0.25 bound and the
  marginal-Sharpe hurdle (0.74 > 0.157 × 2.16 = 0.34), but the proposal quotes
  the friendlier number next to the ex-bonds recommendation.

**2. Sleeve loses money in high-fragility months** (63d dial `.rolling(10).mean()`,
month-mean >= 50, `rd2_fragility.parquet`, coverage 2016-07+; same 16 months
identified: 2018-09/10, 2020-01/02, 2021-05/06/07/09/12, 2022-01, 2024-09/10/11/12,
2026-02/03):

- Full-16 same-close: **-0.23%/mo, 44% hit, vs +0.52%/mo in other 2016-07+ months
  (N=16/104, Welch t=-2.16, p=0.045)** — matches the reported -0.23/+0.51/44%.
  CONFIRMED.
- The ex-bonds pilot spec is **worse**: -0.40%/mo same-close (44% hit, t=-2.38,
  p=0.029); -0.39%/mo next-open (38% hit, t=-2.31). The bonds were cushioning
  the fragility months; dropping them deepens the known frag-month loss.
  Inherits the usual caveat: the fragility series is a current-vintage
  reconstruction (calibration lookahead), N=16 months / 7 contiguous episodes.

## Prototype audit (lookahead scan)

Read `tf_backtest.py` line by line. **No lookahead found**: mom12-1 uses
`shift(1)/shift(12)`; ma10 and the 63d vol use data through the signal close;
weights are applied with `shift(1)` to the following month; rf is shifted;
turnover cost is charged in the transition month. Minor findings: (a) headline
N=303 includes the partial 2026-07 row (trivial); (b) dead code in the
equal-weight branch of `base_weights` (harmless); (c) costs are charged on
*target*-weight turnover, ignoring intramonth drift — mildly understates
turnover, immaterial at 5 bps; (d) the same-close convention itself was the
only real gap, and Gate A closes it.

---

## Recommendation: **PROCEED to the 0.5x pilot, with modified expectations.**

- **Gate A passes cleanly** (0.74 vs 0.80, delta 0.06 < 0.1; slippage -2 bps/mo,
  t=-0.82). Implement with next-open execution (MOO/`TIF=OPG` basket on the
  first session of the month, signal from the prior month-end close) — it is
  operationally simpler than a last-session MOC with a 15:50 signal estimate,
  and it costs essentially nothing. Plan on **Sharpe ~0.74 / CAGR ~6.5% /
  maxDD ~-10.5%** for the ex-bonds spec; quote those, not the same-close numbers.
- **Gate B: adopt-as-ballast survives, but do not claim dead-zone hedging.** The
  sleeve adds uncorrelated positive carry in the dead zones (+0.17 to +0.55%/mo
  at 1.0x, hit 65-78%) but fills only ~13% of the deepest (Jul-Sep × midterm)
  shortfall and earns below its own average there. It also does not share the
  book's midterm weakness (p=0.80 year-clustered), which is the useful half of
  the story.
- **Two modifications to carry into implementation:** (1) update the pilot's
  stated integration stats to the ex-bonds spec — corr +0.157-0.166 (not
  +0.117) and high-frag -0.39 to -0.40%/mo (not -0.23%) — both still inside the
  rubric bounds but the ex-bonds numbers are uniformly less flattering than the
  full-16 headline; (2) reconsider whether the pilot horse should be full-16
  rather than ex-bonds if dead-zone/fragility behavior is weighted at all —
  full-16 is better in Jul-Sep (+0.45%/mo), better in the intersection (+0.31%,
  89% hit), better in high-frag months (-0.23 vs -0.40), and had the real 2008
  (+8.0% vs +2.1%); ex-bonds wins only on post-2020 standalone Sharpe and on
  not owning the duration regime risk. Either passes both gates; the choice is
  a portfolio-construction preference, and it should be made explicitly rather
  than inherited from the proposal's §4 framing.
