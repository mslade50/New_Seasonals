# Multi-Asset Trend Following on Liquid ETFs — Prototype Study

Run date: 2026-07-02. Track: `trend-following`.
Code: `scratch/ultracode_research/tf_backtest.py` (main prototype),
`tf_followup.py` (robustness, attribution, fragility timing),
`tf_inspect.py` / `tf_inspect2.py` (data checks).
Saved series: `tf_monthly_series.parquet` (sleeve net/gross, book, SPY monthly).

## Verdict in one paragraph

The classic spec works and is a genuine diversifier: combo (12-1 momentum sign AND
price>10-month MA), long/flat, inverse-vol slot weights on 16 liquid ETFs delivers
**5.2% CAGR at 4.0% vol, Sharpe 0.86, max drawdown -4.5%** net of 5 bps/side over
2001-05..2026-06 (N=303 months, excess-return t=4.33), with **+8.0% in 2008** and
**+0.8% in 2022** against SPY -36.8% / -18.2%, and only **+0.12 correlation** to the
book's monthly R. Sub-period Sharpe is stable and, critically, the ex-bonds variant is
its *best* self post-2020 (Sharpe 0.90), so the edge does not depend on the dead bond
tailwind. **But it fails the specific brief**: in concurrent high-fragility months
(63d MA10 >= 50, 2016+) the sleeve averages **-0.23%/mo vs +0.51%/mo elsewhere**
(N=16, hit rate 44%), and it is still -0.23%/mo at t+1. Trend monetizes *extended*
bears (2008, 2022), not the 1-2-month fragility-peak whipsaws (2018-10, 2020-02,
2021 chop, 2024 Q4) that hurt the book. Adopt it as a low-cost strategic ballast
sleeve if desired; do NOT adopt it as the high-fragility hedge — that problem is
better addressed by the already-pending sizing taper.

## 1. Rules (exact, as implemented)

- **Universe (16 ETFs, dynamic inception, all verified in `master_prices.parquet`):**
  - Equities/RE (7): SPY, QQQ, IWM, EFA, EEM, FXI, VNQ
  - Bonds/credit (4): TLT, IEF, LQD, HYG
  - Commodities (4): GLD, SLV, DBC, USO
  - Dollar (1): UUP
  - Coverage: SPY/QQQ from 2000; TLT/IEF/LQD 2002-07; EEM 2003-04; GLD 2004-11;
    DBC/USO/SLV 2006; HYG/UUP 2007. Assets enter when they have 13 monthly closes.
    Backtest starts 2001-05 (>=3 eligible); avg 13.8 eligible, 7.6 held.
- **Signals** (month-end adjusted closes): `mom12_1` = sign of P[t-1m]/P[t-12m]-1;
  `ma10` = P[t] > SMA(10 monthly closes); `combo` = both ON. Monthly rebalance.
- **Weights:** slot-based inverse-vol — each eligible asset gets
  (1/vol63d_ann)/Σ(1/vol) over ALL eligible assets, capped at 20% (excess to cash).
  OFF slot sits in T-bills. Avg net exposure 55%.
- **Cash:** flat sleeve earns ^IRX/100/12 (13-week bill yield, in master_prices).
- **Costs:** 5 bps per side on target-weight turnover. Avg turnover 16.7%/mo →
  ~1.7 bps/mo drag (gross Sharpe 0.88 → net 0.86).
- **Convention:** signal computed on month-end close, traded at that close (live =
  MOC on last session; the repo already has MOC plumbing). See timing caveat, §7.

## 2. Full-sample results (net of costs), 2001-05..2026-06, N=303 months

| Spec | CAGR | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| mom12-1 L/F inv-vol | 5.45% | 4.93% | 0.75 | -8.2% |
| ma10 L/F inv-vol | 5.84% | 4.65% | 0.88 | -5.2% |
| **combo L/F inv-vol (primary)** | **5.22%** | **4.03%** | **0.86** | **-4.5%** |
| combo L/S inv-vol | 3.95% | 7.69% | 0.32 | -18.3% |
| combo L/F equal-weight | 6.05% | 5.39% | 0.80 | -8.3% |
| combo L/F inv-vol EX-BONDS | 6.75% | 6.32% | 0.79 | -10.4% |
| mom12-1 L/F EX-BONDS | 7.18% | 7.62% | 0.72 | -16.2% |
| SPY buy & hold (same window) | 9.21% | 14.90% | 0.55 | -50.8% |

- Monthly excess return +0.288%/mo, t = 4.33 (months are the cluster unit, so this
  is honest — no per-trade clustering issue).
- **Long/short is dead** (Sharpe 0.32, maxDD -18%): the short leg of ETF trend has
  not paid since ~2010. Skip it; also avoids borrow cost/availability noise at IBKR.
- Spec choice is not fragile: all three long/flat signal variants land Sharpe
  0.75-0.88; equal weight vs inverse-vol changes little.

## 3. Per-year net returns (primary spec)

| Year | Sleeve | SPY | | Year | Sleeve | SPY |
|---|---|---|---|---|---|---|
| 2001* | +2.0% | -8.9% | | 2014 | +3.6% | +13.5% |
| 2002 | +1.1% | -21.6% | | 2015 | +0.5% | +1.2% |
| 2003 | +11.5% | +28.2% | | 2016 | +0.7% | +12.0% |
| 2004 | +6.8% | +10.7% | | 2017 | +8.3% | +21.7% |
| 2005 | +3.6% | +4.8% | | 2018 | +0.4% | -4.6% |
| 2006 | +11.5% | +15.9% | | 2019 | +8.3% | +31.2% |
| 2007 | +11.8% | +5.2% | | 2020 | +3.1% | +18.3% |
| 2008 | **+8.0%** | **-36.8%** | | 2021 | +5.0% | +28.7% |
| 2009 | +3.1% | +26.4% | | 2022 | **+0.8%** | **-18.2%** |
| 2010 | +6.3% | +15.1% | | 2023 | +2.0% | +26.2% |
| 2011 | +2.0% | +1.9% | | 2024 | +7.3% | +24.9% |
| 2012 | +4.4% | +16.0% | | 2025 | +12.2% | +17.7% |
| 2013 | +4.4% | +32.3% | | 2026 YTD | +4.8% | +9.7% |

*2001 = May-Dec, few assets eligible. No negative year in 26; worst stretch is the
2015-2016 trend winter (+0.5%, +0.7%) — the well-known mid-sample decay shows up as
low absolute return, not drawdown, because the sleeve de-risks to bills.

## 4. Sub-period stability and the bond question

| Period | Full 16 | | EX-BONDS (12 ETFs) | |
|---|---|---|---|---|
| | Sharpe | CAGR | Sharpe | CAGR |
| 2003-2012 | 1.06 | 6.85% | 0.84 | 8.57% |
| 2013-2019 | 0.88 | 3.69% | 0.89 | 4.46% |
| 2020+ | 0.64 | 5.29% | **0.90** | **7.75%** |

The full-universe 2020+ dip is entirely the bond sleeve (bond contribution
+0.102%/mo full-sample → +0.047%/mo 2020+; commodities went the other way, +0.070 →
+0.118). **Ex-bonds Sharpe is 0.84 / 0.89 / 0.90 across the three sub-periods** —
the strategy does not need the 2003-2021 duration tailwind. Attribution full-sample:
equities +0.21%/mo, bonds +0.10, commodities +0.07, UUP +0.00 (dead weight; keep or
drop, immaterial).

## 5. Crisis alpha (primary spec, net)

- **2008: +8.0% vs SPY -36.8%.** Never worse than -1.0% in any month; +0.8/+3.5/+1.6%
  in Oct/Nov/Dec while SPY did -16.5/-7.0/+1.0.
- **2022: +0.8% vs SPY -18.2%.** Flat-to-up all year while both stocks and bonds bled.
- **2020 Feb-Mar: -3.2% cumulative vs SPY -19.4%.** Partial protection only — the
  crash was faster than a monthly trend clock; the sleeve was long risk in February.
  This is the honest preview of the fragility result below.

## 6. Fit with the existing book

Book series = ledger `PnL_flat_750k` summed by exit month / 750k
(`data/backtest_trades_full.parquet`, 3,286 trades). Clean window 2003-01..2026-06
(N=282 complete months; 2026-07 partial month excluded).

- **Correlation: +0.117.** In the 66 months the book lost money: corr +0.174, sleeve
  averaged **+0.21%/mo with a 65% hit rate**. In the 12 worst book months the sleeve
  was positive 6 times, and never worse than -2.1%.
- Additive on the flat $750k basis: book alone mean +1.78%/mo, ann vol 9.66%,
  Sharpe ~2.21 → **book + 1x sleeve: +2.23%/mo, Sharpe ~2.44** (worst month -7.9% →
  -9.2%, both driven by 2015-08 where both were down). The sleeve is small but free
  diversification; at 2x the marginal Sharpe gain is already gone (~2.39).

### The high-fragility question (the actual brief) — NEGATIVE result

High-frag month = calendar month whose mean of frag 63d `.rolling(10).mean()` >= 50,
2016-07+ (N=16; the any-day>=50 and month-start>=50 definitions give N=27/N=18 and
the same sign). Fragility history is a current-vintage reconstruction (calibration
lookahead caveat from the established findings applies).

| Timing | N | Sleeve avg/mo | vs other months | Sleeve hit | Book avg/mo | SPY avg/mo |
|---|---|---|---|---|---|---|
| t (concurrent) | 16 | **-0.23%** | +0.51% | 44% | +1.93% | -0.97% |
| t+1 | 16 | -0.23% | +0.51% | 50% | +3.34% | -0.51% |
| t+2 | 16 | +0.19% | +0.45% | 62% | +3.55% | +0.92% |
| t+3 | 16 | +0.35% | +0.42% | 75% | +3.56% | +1.20% |
| t..t+3 union | 34 | +0.09% | — | — | +2.62% | +0.07% |

The sleeve *loses* money at and immediately after fragility peaks and only normalizes
by t+2/t+3. Month detail shows why: 2018-10 (-2.0%), 2020-02 (-1.8%), 2021-09
(-1.1%), 2022-01 (-1.3%), 2024-10/-12 (-1.5/-1.8%) — every post-2016 fragility
episode resolved as a fast dip + V-recovery or chop, exactly the regime that whipsaws
a monthly trend clock. Trend's crisis alpha needs a *persistent* downtrend (2008,
2022); by the time trend flips defensive in a fragility episode, the book's
short-horizon mean-reversion entries are already being paid. Note the book itself
averaged +1.93%/mo in these months — the frag>=50 damage documented in the
established findings is per-trade avgR dilution (+0.17R), not monthly book losses,
which further weakens the case for hedging it with this sleeve.

## 7. Execution fit

- **Instruments:** 16 US-listed ETFs, all among the most liquid in existence
  (SPY/QQQ/IWM/TLT/GLD each trade $1-30B/day). At avg 55% of $750k deployed
  (~$410k) and 16.7%/mo turnover (~$125k/mo traded, ~2-4 tickets), capacity is a
  non-issue by 3+ orders of magnitude.
- **Order types:** monthly MOC basket on the last trading session (signal computable
  from the intraday price at 15:50 ET with negligible error), or MOO next session.
  The repo's vestigial `moc_orders` tab / Signal Close plumbing fits naturally;
  alternatively a fourth Sheets tab (`Trend`) with `TIF=OPG` orders like the
  Seasonal pipeline. ~1.8 position flips/month plus small inverse-vol rebalances
  (a 1%-of-NAV rebalance band would cut ticket count further).
- **Data:** zero new requirements — all 16 tickers plus ^IRX are already in
  `master_prices.parquet` and maintained by the nightly pipeline. (No SHY/BIL/AGG
  in the cache; ^IRX handles the cash leg. If ever wanted: add BIL to the universe
  file and backfill, same as the LEV3X backfill.)
- **Margin:** long-only, unlevered at 1x; coexists with the equity book under IBKR
  margin easily since book positions are episodic.
- **Timing sensitivity (caveat that matters):** delaying execution by one FULL month
  drops Sharpe 0.86 → 0.55. A 1-day slip is far smaller than a month but this says
  the signal decays — execute on schedule, don't let staged orders go stale.

## 8. Honest bias inventory

1. **Universe hindsight:** the 16 ETFs were picked in 2026 knowing asset-class
   history. Mitigation: it's the standard textbook TAA menu (Faber 2007-era), losers
   (USO +0.005%/mo, UUP +0.002%/mo, DBC, SLV) were kept, and no asset dominates
   (best single contributor SPY at +0.043%/mo of the +0.42%/mo total). ETF
   survivorship itself is negligible for broad-class funds. Single-stock
   survivorship: not applicable (no single stocks).
2. **Adjusted-price basis:** signals are relative levels recomputed per run —
   scale-invariant and safe per the repo's dividend-adjustment invariant. Monthly
   returns are total-return (dividends reinvested), which is what an IBKR account
   approximately realizes with DRIP off + periodic rebalance; drift is bps/yr.
3. **Execution convention:** trade at the signal close is the standard monthly
   backtest convention but is technically same-bar. Bounds: full-month delay Sharpe
   0.55. True next-open execution sits very close to the base case (overnight gap on
   a diversified 16-ETF basket, 12x/yr); not simulated at daily granularity here.
4. **Costs:** 5 bps/side on turnover ≈ 2 bps/mo. Actual spreads on these ETFs are
   0.5-3 bps; no impact modeling needed at this size. Conservative but only mildly.
5. **Cash model:** ^IRX/100/12 approximates bill yield; ignores compounding and
   IBKR's below-benchmark cash interest tiers (first $10k earns 0). Real-world cash
   drag could shave ~10-30 bps/yr off the flat sleeve unless idle cash is swept to
   BIL/box spreads.
6. **Fragility reconstruction:** current-vintage composite with calibration
   lookahead (per established findings) — the high-frag month classification is the
   same series used in the sizing studies, so the negative fragility result is at
   least consistent with them, but neither is point-in-time pure.
7. **Sample breadth:** pre-2004 the universe is 5-8 assets (mostly equities+bonds);
   the 2001-2003 rows are less diversified than the headline spec.
8. **No regime fitting:** all specs are classic 1990s-2000s literature parameters
   (12-1, 10m MA, inverse vol, monthly) — nothing was tuned on this dataset, and the
   spec table (§2) shows results are insensitive across the tested variants.

## 9. Recommendation

**Adopt as strategic ballast only if idle-cash yield pickup is wanted; reject as the
high-fragility fix.** Concretely: if adopted, run `combo L/F inv-vol` (or the
ex-bonds 12-ETF variant, which is the better horse post-2020: Sharpe 0.90, CAGR
7.75% since 2020) at ~0.5-1.0x of NAV as a monthly MOC basket — it adds ~+0.45%/mo
at +0.12 correlation and lifts combined Sharpe ~2.21 → ~2.44 with trivial
operational load (2-4 orders/month, zero new data). But the stated goal — something
that works at fragility 55+ — is not met: -0.23%/mo concurrent and t+1, 44% hit
rate, because post-2016 fragility episodes are fast whipsaws, not trends. Keep the
pending fragility sizing taper (kill 1.25x boost, taper to 0.5x by 60) as the
frag-hole treatment, and evaluate a convexity instrument (e.g. the put-hedge track /
`tests/backtest_put_hedge.py` prior art) if a true frag-peak hedge is still wanted.

## Adversarial verification (independent recompute, 2026-07-02)

Verifier: fresh reimplementation from the §1 rules only, no code reuse
(`scratch/ultracode_research/verify_trend-following.py`, episode check in
`verify_tf_episodes.py`). Same data sources (`master_prices.parquet` Close,
`rd2_fragility.parquet` 63d MA10, ledger by exit month).

| Claim | Verdict | Recompute |
|---|---|---|
| Primary spec full sample | **CONFIRMED** | CAGR 5.24% / vol 4.03% / Sharpe 0.87 / maxDD -4.5% / t=4.36, N=301 (vs 5.22/4.03/0.86/-4.5/4.33, N=303 — N differs by start-month bookkeeping only) |
| Crisis alpha | **CONFIRMED** | 2008 +8.0% vs SPY -36.8%; 2022 +0.8% vs -18.2%; 2020 Feb-Mar -3.1% vs -19.4% (exact) |
| Book correlation / losing months | **CONFIRMED** | corr +0.117 N=282; 66 losing book months: sleeve +0.21%/mo, 65% hit; 12 worst: positive 6/12, worst -2.1% (exact) |
| High-fragility negative | **CONFIRMED, and stronger than reported** | Same 16 months; t+0 -0.21%/mo (44% hit) vs +0.52% other; t+1 -0.21%; t+2 +0.19%; t+3 +0.36%. Adversarial episode-clustered check (16 months = 7 contiguous episodes): episode-level Welch t=-3.55, p=0.007, 4/7 episodes negative — the negative result survives the clustering objection |
| Sub-period stability / ex-bonds | **CONFIRMED** | full16 Sharpe 1.07/0.90/0.67; ex-bonds 0.84/0.90/0.93 across 2003-12/2013-19/2020+ (vs 1.06/0.88/0.64 and 0.84/0.89/0.90); ex-bonds full sample 0.81 vs their 0.79 |
| Long/short dead | **CONFIRMED directionally; exact number not reproduced** | My short-leg convention (short every eligible OFF asset) gives Sharpe 0.15, maxDD -19.5% vs their 0.32/-18.3% — the exact figure is convention-dependent, but every convention lands in "dead"; the reject conclusion holds a fortiori |
| Portfolio effect | **CONFIRMED** | book Sharpe 2.21 → 2.45 at 1x, 2.40 at 2x (vs 2.44/2.39) |
| Execution + delay | **CONFIRMED** | 7.6 avg positions, 16.9%/mo turnover (~$127k), 1.8 flips/mo, full-month delay Sharpe 0.55 (exact) |

Residual caveats the verifier endorses rather than adds: universe hindsight (§8.1) is
the largest unremovable bias — Sharpe 0.86 on a 2026-chosen menu is an upper bound;
same-close execution is unverified at daily granularity (the 0.55 delay bound brackets
it but the true next-open number was not computed by either side); N=16 frag months /
7 episodes is small even though it clusters cleanly. No lookahead, cost omission, or
cherry-picked window found. The recommendation (reject as frag hedge, optional small
ballast, keep the sizing taper) follows from numbers that all reproduce.
