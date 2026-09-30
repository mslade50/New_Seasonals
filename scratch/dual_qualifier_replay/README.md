# Dual-qualifier replay: dial >= 50 AND SPY within 2% of its 252-session high

Research only. Nothing here is imported by production. Run:

```
python scratch/dual_qualifier_replay/replay.py             # engine + analysis
python scratch/dual_qualifier_replay/replay.py --analysis-only
```

Built 2026-09-18 against ledger vintage `data/backtest_trades_full.parquet`
(2026-09-18 05:10), `data/rd2_fragility.parquet` (through 2026-09-17),
`data/cboe_putcall.parquet` (through 2026-09-17), `data/master_prices.parquet`
(through 2026-09-15).

---

## Protocol

**Schemes measured.** Carriers are the six `frag_risk_bands` strategies:
Weak Close Decent Sznls, SPY QQQ MonFri Reversion, Monday Dip, Indices
Oversold Bounce, 3x Bear ETF Overbot Fade, Monthly Weak Close.

- **Incumbent.** Band table selected by the lag-1 P/C fear state through
  `pc_fear.fear_state_asof` / `pc_fear.select_bands`: fear ON (252d pctile of
  the 10d-MA equity P/C > 85) gives `[[0,50,1.25],[50,999,1.0]]`, fear OFF
  gives `[[0,50,1.0],[50,999,0.0]]`, stale (> 3 bd) or missing or pre-2007-11
  falls closed to the plain `frag_risk_bands` `[[50,999,0.25]]`. First-match
  band lookup on the PIT dial. No dial for the signal date gives 1.0x.
- **Variant A (STACK).** Incumbent, then zero any signal with dial >= 50 AND
  near-high, whatever the fear state.
- **Variant B (NARROW).** Below 50 unchanged. At dial >= 50 and near-high the
  incumbent treatment stands. At dial >= 50 and NOT near-high the signal trades
  at 1.0x and the fear tables are ignored there.
- **No-band baseline.** 1.0x everywhere for all six carriers.

**How the schemes were replayed.** Not by re-weighting the shipped ledger. The
pcfear shadow parquet is a 2026-08-07 vintage that predates the 2026-09-04
base-bps tilt and the retired WCDS seasonal size tiers, and its last signal is
2026-07-29, so it cannot see the August and September window at all. Instead
`replay.py` loads the six carriers through `build_full_strategy_book()`, runs
`precompute_all_indicators` / `generate_candidates_fast` once, and then runs
`process_signals_fast` four times on the same candidate list with
`pages.strat_backtester.frag_band_mult_at` monkeypatched to each scheme. Every
other production knob is untouched: `cap_bps=250`, `overflow_active=True`,
`flat_sizing=True` ($750k), pooled caps None, cross-strategy overlap clamp, gap
derate, same-day derate, stop-fill convention. Cap re-allocation and dropped
zero-share rows are therefore handled by the engine rather than by arithmetic.

The qualifier depends only on the signal DATE (dial, SPY near-high, fear
state), never on the ticker, so a patched `frag_band_mult_at` expresses all
four schemes exactly.

**Provenance check.** The replayed incumbent reproduces the shipped ledger's
carrier rows exactly on the 2016+ window: 504 trades and $995,183 flat PnL on
both sides. Cross-scheme differences below are engine output, not estimates.

**Inputs and definitions.**

1. **Universe.** The six carriers' signals with `Signal Date >= 2016-01-01`
   (the dial parquet begins 2016-07-05; Jan to Jun 2016 signals have no dial
   and therefore size 1.0x in every scheme, including the incumbent).
2. **Dial.** `pages.strat_backtester._frag_score_series()`, the exact
   production sizing statistic: `nyse_risk.main_dial_from_frame` over
   `data/rd2_fragility.parquet` (`main_score` where written, else the
   10-session mean of the `63d` column), normalized index, daily grid,
   `ffill(limit=5)`. Read point-in-time at the signal date. Vintage caveat
   stands: rows before 2026-07-02 are a recompute vintage that drifted up to
   about 7 points.
3. **Near-high.** SPY adjusted close from `master_prices.parquet`.
   `near_high = close >= 0.98 * close.rolling(252, min_periods=60).max()`,
   evaluated on the signal date close, which is what the scan sees. Zero
   baseline trades had an unresolvable flag.
4. **Cell table** is built on the no-band baseline pass so every signal carries
   its full-size outcome, including the 88 the incumbent currently zeroes.
5. **Curves.** True daily MTM through `get_daily_mtm_series`, not exit-date
   booking. maxDD is on the cumulative daily PnL curve, worst-21d is the
   minimum rolling 21-session sum. Flat $750k basis throughout.
6. **Episodes.** Signals in the same cell within 10 SPY sessions of each other
   count as one episode. Tests are on episode-mean R. Sign test is the exact
   two-sided binomial on episode means.
7. **Second dial vintage.** `rd2_fragility_ts.parquet` (research recompute)
   covers 2016-05 to 2026-05-07 only, so it cannot carry the scheme comparison
   or the August window. It is reported as a cell-table sensitivity, where it
   agrees with the live parquet on the dial >= 50 call for 97.2% of the 529
   covered trades.

**Known bounds.** Ledger survivorship (CLAUDE.md) applies. The ledger replays
today's config over all history, so pre-2026 sizing is not what live traded.
The qualifier is evaluated on the carriers only; nothing here says anything
about the rest of the book.

---

## 1. Universe, 2016+

| Strategy | signals | filled trades | signals dial>=50 | signals dial>=50 & near-high |
|---|---|---|---|---|
| 3x Bear ETF Overbot Fade | 163 | 76 | 48 | 13 |
| Indices Oversold Bounce | 208 | 147 | 31 | 9 |
| Monday Dip | 77 | 51 | 17 | 14 |
| Monthly Weak Close | 11 | 8 | 2 | 0 |
| SPY QQQ MonFri Reversion | 207 | 147 | 34 | 21 |
| Weak Close Decent Sznls | 267 | 163 | 33 | 30 |
| **TOTAL** | **933** | **592** | **165** | **87** |

## 2. Cell table (baseline 1.0x sizing, filled trades, 2016+)

| dial>=50 | near-high | fear | N | avgR | medR | hit | totR | avg $ | tot $ |
|---|---|---|---|---|---|---|---|---|---|
| no | no | off | 214 | 0.545 | 0.734 | 69.2% | 116.72 | 1,810 | 387,293 |
| no | no | on | 90 | 0.624 | 0.713 | 74.4% | 56.11 | 2,006 | 180,573 |
| no | yes | off | 178 | 0.557 | 0.681 | 69.7% | 99.20 | 1,880 | 334,625 |
| no | yes | on | 3 | 1.292 | 1.129 | 100% | 3.88 | 3,164 | 9,493 |
| **yes** | **no** | **off** | **35** | **-0.058** | -0.584 | 40.0% | -2.04 | -114 | **-3,977** |
| yes | no | on | 10 | 0.492 | 0.735 | 90.0% | 4.92 | 2,005 | 20,053 |
| **yes** | **yes** | **off** | **53** | **-0.275** | -0.620 | 34.0% | -14.56 | -1,096 | **-58,072** |
| **yes** | **yes** | **on** | **9** | **1.255** | 1.855 | 88.9% | 11.29 | 3,232 | **29,090** |

Collapsed over fear state:

| dial>=50 | near-high | N | avgR | medR | hit | totR | tot $ |
|---|---|---|---|---|---|---|---|
| no | no | 304 | 0.569 | 0.723 | 70.7% | 172.84 | 567,866 |
| no | yes | 181 | 0.570 | 0.687 | 70.2% | 103.07 | 344,118 |
| yes | no | 45 | 0.064 | 0.102 | 51.1% | 2.89 | 16,076 |
| yes | yes | 62 | -0.053 | -0.379 | 41.9% | -3.27 | -28,982 |

The two cells in bold above are the ones the incumbent already zeroes (both
fear-OFF dial>=50 cells, 88 trades, -$62,049 at full size). The third bold cell
(dial>=50, near-high, fear ON, +$29,090) is the one Variant A would kill.

**dial>=50 & near-high cell by strategy**

| Strategy | N | avgR | medR | hit | totR | tot $ |
|---|---|---|---|---|---|---|
| 3x Bear ETF Overbot Fade | 9 | 1.309 | 1.689 | 88.9% | 11.78 | 33,142 |
| Indices Oversold Bounce | 8 | 0.073 | -0.257 | 25.0% | 0.59 | 366 |
| Monday Dip | 9 | -0.533 | -0.656 | 33.3% | -4.80 | -17,351 |
| SPY QQQ MonFri Reversion | 17 | -0.137 | -1.015 | 41.2% | -2.32 | -20,000 |
| Weak Close Decent Sznls | 19 | -0.448 | -1.012 | 31.6% | -8.51 | -25,139 |

**dial>=50 & near-high cell by year**

| year | N | avgR | hit | totR | tot $ | share of N |
|---|---|---|---|---|---|---|
| 2018 | 3 | 0.246 | 33.3% | 0.74 | 2,177 | 4.8% |
| 2019 | 1 | 0.977 | 100% | 0.98 | 2,884 | 1.6% |
| 2021 | 35 | -0.034 | 48.6% | -1.17 | -14,346 | **56.5%** |
| 2022 | 2 | -0.257 | 0% | -0.51 | -2,023 | 3.2% |
| 2024 | 13 | -0.509 | 23.1% | -6.61 | -25,215 | 21.0% |
| 2026 | 8 | 0.415 | 50.0% | 3.32 | 7,541 | 12.9% |

Max single-year share of the cell's N is 56.5% (2021).

## 3. Scheme comparison, full 2016+ carrier set, flat $750k, daily MTM

| scheme | trades | total PnL $ | avgR | R dollar-wtd | totR | maxDD $ | maxDD %NAV | worst 21d $ | worst day $ | vs incumbent $ |
|---|---|---|---|---|---|---|---|---|---|---|
| incumbent | 504 | 995,183 | 0.579 | 0.607 | 292.0 | -43,576 | -5.81% | -28,862 | -24,747 | 0 |
| variant A (stack) | 495 | 966,093 | 0.567 | 0.600 | 280.7 | -43,576 | -5.81% | -28,862 | -24,747 | **-29,090** |
| variant B (narrow) | 539 | 991,206 | 0.538 | 0.567 | 290.0 | -44,113 | -5.88% | -35,123 | -24,873 | **-3,977** |
| no-band baseline | 592 | 899,078 | 0.465 | 0.480 | 275.5 | -67,003 | -8.93% | -44,255 | -53,575 | -96,105 |

Variant A's drawdown and worst-21d are byte-identical to the incumbent's: the
nine trades it removes are all winners sitting outside the drawdown path, so it
buys no tail protection at all. Variant B is slightly worse on both risk
measures than the incumbent while also being slightly worse on PnL.

By strategy:

| Strategy | incumbent | variant A | variant B | no-band |
|---|---|---|---|---|
| 3x Bear ETF Overbot Fade | 101,403 | 93,756 | 118,384 | 136,020 |
| Indices Oversold Bounce | 237,395 | 229,016 | 223,881 | 212,222 |
| Monday Dip | 56,434 | 56,434 | 56,120 | 36,262 |
| Monthly Weak Close | 44,872 | 44,872 | 51,616 | 49,930 |
| SPY QQQ MonFri Reversion | 316,534 | 300,483 | 308,660 | 258,735 |
| Weak Close Decent Sznls | 238,545 | 241,532 | 232,545 | 205,909 |

## 4. Episode clustering

| cell | trades | episodes | avgR per trade | avgR per episode |
|---|---|---|---|---|
| dial>=50 & near-high | 62 | 15 | -0.053 | +0.210 |
| dial>=50 & NOT near-high | 45 | 12 | +0.064 | +0.220 |

Episode-clustered Welch t of the zeroed cell against the rest of dial>=50:
**t = -0.027, p = 0.979**. The zeroed cell's own episode means against zero:
t = +1.015, p = 0.328, sign test 10+/5-, p = 0.302. The trade-level minus sign
is an artefact of the 2021 cluster carrying 35 of the 62 trades; equal-weighted
by episode the cell is positive.

Split inside each fear state, which is what matters because the incumbent
already governs the dial>=50 zone by fear:

| fear | N near | ep near | avgR near | N not-near | ep not-near | avgR not-near | Welch t (ep) | p |
|---|---|---|---|---|---|---|---|---|
| off | 53 | 14 | -0.275 | 35 | 10 | -0.058 | -0.354 | 0.728 |
| on | 9 | 3 | +1.255 | 10 | 3 | +0.492 | +0.521 | 0.631 |

The qualifier points in opposite directions in the two states. Inside fear OFF,
near-high is the worse half. Inside fear ON, near-high is the better half.

The cells each variant actually moves relative to the incumbent:

| cell | N | episodes | avgR | totR | PnL at 1.0x | t (ep vs 0) | sign | effect |
|---|---|---|---|---|---|---|---|---|
| A zeroes: dial>=50 & near & fear ON | 9 | 3 | +1.255 | 11.29 | +29,090 | +1.641 | 2+/1- | lose |
| B restores: dial>=50 & NOT near & fear OFF | 35 | 10 | -0.058 | -2.04 | -3,977 | +0.731 | 6+/4- | gain |

**LOYO on the zeroed cell.** Dropping any single year leaves the episode mean
positive: +0.075 (drop 2021) to +0.361 (drop 2024), t between +0.33 and +1.60,
never negative. Dropping the largest-|PnL| episode leaves 52 trades over 14
episodes at +0.304 episode-mean R, t = +1.545.

## 5. Sensitivity

| dial thresh | near thresh | N zerocell | episodes | avgR zerocell | tot $ zerocell | N rest | avgR rest | Welch t (ep) |
|---|---|---|---|---|---|---|---|---|
| 50 | 1% | 45 | 12 | **-0.481** | -73,549 | 62 | +0.343 | -1.407 |
| 50 | 2% | 62 | 15 | -0.053 | -28,982 | 45 | +0.064 | -0.027 |
| 50 | 3% | 79 | 18 | -0.054 | -28,042 | 28 | +0.139 | -0.536 |
| 65 | 1% | 10 | 5 | -0.422 | -14,026 | 28 | +0.232 | +0.103 |
| 65 | 2% | 18 | 6 | +0.064 | -2,043 | 20 | +0.056 | +0.799 |
| 65 | 3% | 25 | 8 | +0.145 | +6,334 | 13 | -0.103 | +0.351 |

The 2% threshold McKinley named is the flat spot, not the edge. A 1% threshold
does isolate a worse cell, but the whole difference between 1% and 2% is 17
trades, and the 2021-07-16 cluster (five trades averaging about +2R) plus
2026-09-02 SOXS at +2.86R account for most of it (see
`results/near_band_1to2pct.csv`). At the 65 dial threshold the sign flips at
2% and 3%. That is threshold-shopping territory, not a stable effect.

**Era split** (baseline 1.0x):

| era | cell | N | avgR | hit | tot $ |
|---|---|---|---|---|---|
| pre-2020 | dial>=50 & near | 4 | +0.428 | 50.0% | 5,061 |
| pre-2020 | dial>=50 & NOT near | 2 | -0.239 | 50.0% | -1,880 |
| pre-2020 | dial<50 | 150 | +0.313 | 63.3% | 213,036 |
| 2020+ | dial>=50 & near | 58 | -0.086 | 41.4% | -34,043 |
| 2020+ | dial>=50 & NOT near | 43 | +0.078 | 51.2% | 17,956 |
| 2020+ | dial<50 | 335 | +0.683 | 73.7% | 698,948 |

Pre-2020 the dial>=50 cells hold 6 trades in total, so the whole question is a
2020+ question.

**Second dial vintage** (`rd2_fragility_ts.parquet`, research recompute,
covers through 2026-05-07, 97.2% agreement on the dial>=50 call over 529
covered trades):

| dial>=50 (ts) | near-high | N | avgR | hit | tot $ |
|---|---|---|---|---|---|
| no | no | 276 | 0.581 | 70.7% | 515,374 |
| no | yes | 162 | 0.646 | 72.8% | 349,371 |
| yes | no | 35 | 0.121 | 60.0% | 17,191 |
| yes | yes | 56 | -0.180 | 37.5% | -47,995 |

Same ordering as the live parquet, so the cell structure is not a vintage
artefact. The scheme comparison itself was run on the live parquet only,
because the ts vintage stops five months short of the window in question.

## 6. The August to September 2026 window

Every carrier signal since 2026-08-01, with its full-size (1.0x) outcome:

| Signal Date | Strategy | Ticker | dial | near-high | SPY dd % | fear | incumbent | A | B | R at 1.0x | $ at 1.0x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-10 | Monday Dip | SMH | 63.7 | yes | 0.03 | off | 0.0 | 0.0 | 0.0 | +0.680 | 1,148 |
| 2026-08-10 | SPY QQQ MonFri Reversion | QQQ | 63.7 | yes | 0.03 | off | 0.0 | 0.0 | 0.0 | +0.866 | 4,431 |
| 2026-08-17 | Indices Oversold Bounce | ^GSPC | 84.8 | yes | 0.67 | off | 0.0 | 0.0 | 0.0 | -0.733 | -1,649 |
| 2026-08-17 | SPY QQQ MonFri Reversion | SPY | 84.8 | yes | 0.67 | off | 0.0 | 0.0 | 0.0 | no fill | |
| 2026-08-17 | SPY QQQ MonFri Reversion | QQQ | 84.8 | yes | 0.67 | off | 0.0 | 0.0 | 0.0 | -0.499 | -2,554 |
| 2026-08-28 | SPY QQQ MonFri Reversion | SPY | 87.6 | yes | 1.10 | off | 0.0 | 0.0 | 0.0 | -1.045 | -5,347 |
| 2026-08-28 | SPY QQQ MonFri Reversion | QQQ | 87.6 | yes | 1.10 | off | 0.0 | 0.0 | 0.0 | no fill | |
| 2026-08-28 | Weak Close Decent Sznls | XBI | 87.6 | yes | 1.10 | off | 0.0 | 0.0 | 0.0 | +1.238 | 3,657 |
| 2026-09-02 | 3x Bear ETF Overbot Fade | SOXS | 87.9 | yes | 1.64 | off | 0.0 | 0.0 | 0.0 | +2.861 | 8,047 |
| 2026-09-08 | Indices Oversold Bounce | ^GSPC | 87.9 | yes | 1.53 | off | 0.0 | 0.0 | 0.0 | -0.049 | -192 |
| 2026-09-10 | 3x Bear ETF Overbot Fade | SDOW | 87.0 | **no** | 2.58 | off | 0.0 | 0.0 | **1.0** | no fill | |
| 2026-09-10 | 3x Bear ETF Overbot Fade | TZA | 87.0 | **no** | 2.58 | off | 0.0 | 0.0 | **1.0** | -1.310 | -3,315 |
| 2026-09-15 | 3x Bear ETF Overbot Fade | SQQQ | 84.1 | **no** | 2.63 | off | 0.0 | 0.0 | **1.0** | +1.705 | 4,315 |
| 2026-09-15 | 3x Bear ETF Overbot Fade | SDOW | 84.1 | **no** | 2.63 | off | 0.0 | 0.0 | **1.0** | -0.351 | -888 |

Subtotals on filled signals, flat basis:

| window | filled | totR at 1.0x | $ at 1.0x | incumbent $ | variant A $ | variant B $ |
|---|---|---|---|---|---|---|
| Aug 2026 | 6 | +0.507 | -314 | 0 | 0 | 0 |
| Sep 2026 | 5 | +2.856 | 7,967 | 0 | 0 | 112 |
| Aug-Sep 2026 | 11 | +3.363 | 7,653 | 0 | 0 | 112 |

Fear has been OFF the whole window, so **Variant A changes nothing at all
here**: every dial>=50 signal since 2026-08-01 is already zeroed by the
incumbent. Variant B restores the four not-near 3x Bear Fade signals of
September 10 and 15, three of which filled, for +$112 net. Note that this
window's zeroed set now books +0.507R for August (the 2026-09-04 CLAUDE.md note
recorded +2.21R / +$8.5k for six signals; the ledger is a full rebuild and
recent marginal fills and exits do flicker between vintages).

---

## Conclusion

1. Variant A costs $29,090 over 2016+ and buys nothing: it only bites in the
   dial>=50, near-high, fear-ON cell, which is 9 trades at +1.25R avgR, and the
   incumbent's drawdown and worst-21-day figures do not move by a single dollar.
2. Variant B costs $3,977 over 2016+, restores 35 flat trades at -0.058R avgR,
   and makes the worst 21-day window worse, from -$28,862 to -$35,123.
3. Near-high does NOT isolate a worse cell than dial>=50 alone: pooled, the
   episode-clustered Welch t is -0.027 (p=0.98), and the near-high half is
   worse only inside fear OFF (-0.275 vs -0.058, t=-0.354) while being better
   inside fear ON (+1.255 vs +0.492), so it mostly re-labels what the P/C fear
   state already separates.
4. Neither variant clears the book's bar for a dial-conditioned control:
   nothing reaches clustered t <= -2, the zeroed cell's own episode means are
   POSITIVE (+0.21, t=+1.02, sign 10+/5-), LOYO never turns it negative, and
   56.5% of its trades are one year (2021).
5. The 2% threshold is the flat spot in the sensitivity grid. Only 1% looks
   punitive (avgR -0.48, t=-1.41) and that gap is 17 trades, dominated by one
   July 2021 cluster and one September 2026 print, while at a dial of 65 the
   sign flips at 2% and 3%.
6. In the live August to September 2026 window A is a no-op because fear has
   been OFF throughout, and B would have restored four September 3x Bear Fade
   signals for +$112, so neither variant is a way to act on the current
   episode.

---

## 7. What the CURRENT rule removes, by distance to the 252-session high

Added 2026-09-20. "Removed" is the six carriers' dial >= 50 signals that the
incumbent sizes at 0.0x (fear OFF) or 0.25x (stale P/C), 2016+, measured on the
cached no-band 1.0x engine pass so every removed signal carries its full-size
outcome. Buckets are SPY's distance below its trailing 252-session closing high
on the signal date: `<2%` is `[0, 2%)`, `2-3%` is `[2%, 3%)`, `3-5%` is
`[3%, 5%)`, `>5%` is `[5%, inf)`. Full grid in
`results/removed_by_distance.csv`, pooled test in
`results/removed_near_vs_far.csv`.

**There are zero stale-P/C rows in the 2016+ window**, so the removed set is
entirely the fear-OFF 0.0x cell.

### REMOVED (dial>=50, incumbent 0.0x/0.25x)

| bucket | N signals | N trades | avgR | medR | hit | totR | tot $ at 1.0x | max year share |
|---|---|---|---|---|---|---|---|---|
| <2% | 75 | 53 | -0.275 | -0.620 | 34.0% | -14.56 | -58,072 | 49.1% (2021) |
| 2-3% | 23 | 14 | -0.369 | -1.021 | 28.6% | -5.17 | -14,173 | 35.7% (2024) |
| 3-5% | 33 | 17 | **+0.473** | +0.256 | 52.9% | +8.04 | **+28,809** | 58.8% (2024) |
| >5% | 4 | 4 | -1.226 | -0.832 | 25.0% | -4.91 | -18,613 | 75.0% (2020) |
| ALL | 135 | 88 | -0.189 | -0.593 | 36.4% | -16.60 | -62,049 | 36.4% (2021) |

### KEPT at full size (dial>=50, fear ON)

| bucket | N signals | N trades | avgR | medR | hit | totR | tot $ at 1.0x | max year share |
|---|---|---|---|---|---|---|---|---|
| <2% | 12 | 9 | +1.255 | +1.855 | 88.9% | +11.29 | +29,090 | 100% (2021) |
| 2-3% | 7 | 3 | +1.388 | +1.142 | 100% | +4.17 | +15,113 | 66.7% (2021) |
| 3-5% | 6 | 5 | +0.046 | +0.727 | 80.0% | +0.23 | +2,227 | 80.0% (2026) |
| >5% | 5 | 2 | +0.265 | +0.265 | 100% | +0.53 | +2,713 | 100% (2022) |
| ALL | 30 | 19 | +0.854 | +0.977 | 89.5% | +16.22 | +49,143 | 57.9% (2021) |

### ALL dial>=50, regardless of fear state (one line per bucket)

| bucket | N signals | N trades | avgR | medR | hit | totR | tot $ at 1.0x | max year share |
|---|---|---|---|---|---|---|---|---|
| <2% | 87 | 62 | -0.053 | -0.379 | 41.9% | -3.27 | -28,982 | 56.5% (2021) |
| 2-3% | 30 | 17 | -0.059 | -1.013 | 41.2% | -1.01 | +940 | 29.4% (2024) |
| 3-5% | 39 | 22 | +0.376 | +0.578 | 59.1% | +8.27 | +31,036 | 45.5% (2024) |
| >5% | 9 | 6 | -0.729 | -0.260 | 50.0% | -4.38 | -15,900 | 50.0% (2020) |
| ALL | 165 | 107 | -0.004 | -0.227 | 45.8% | -0.38 | -12,906 | 40.2% (2021) |

### dial<50 (all carrier trades, the healthy book)

| bucket | N signals | N trades | avgR | medR | hit | totR | tot $ at 1.0x | max year share |
|---|---|---|---|---|---|---|---|---|
| <2% | 297 | 181 | +0.570 | +0.687 | 70.2% | +103.07 | +344,118 | 16.0% (2025) |
| 2-3% | 87 | 55 | +0.559 | +0.787 | 67.3% | +30.75 | +112,330 | 23.6% (2026) |
| 3-5% | 85 | 48 | +0.264 | +0.509 | 64.6% | +12.65 | +54,624 | 35.4% (2024) |
| >5% | 299 | 201 | +0.644 | +0.728 | 73.1% | +129.43 | +400,912 | 31.3% (2022) |
| ALL | 768 | 485 | +0.569 | +0.706 | 70.5% | +275.91 | +911,984 | 13.2% (2020) |

### Within the removed set: <2% vs >=2% from the high

| cell | trades | episodes | avgR per trade | avgR per episode | tot $ | t (ep vs 0) | sign | p |
|---|---|---|---|---|---|---|---|---|
| removed, <2% from high | 53 | 14 | -0.275 | +0.114 | -58,072 | +0.520 | 9+/5- | 0.424 |
| removed, >=2% from high | 35 | 10 | -0.058 | +0.263 | -3,977 | +0.731 | 6+/4- | 0.754 |

Pooled episode-clustered Welch t of `<2%` against `>=2%`: **t = -0.354,
p = 0.728**.

### Answer

The near-high bucket is the worse half of the removed set on trade-level avgR,
-0.275 against -0.058, and it carries almost all the dollars, -$58,072 of the
-$62,049. But that does not survive being read properly:

- The gap is not significant once episodes are respected. Episode-clustered
  Welch t = -0.354, p = 0.73. Both halves have POSITIVE episode means (+0.114
  and +0.263), so the trade-level minus signs come from cluster weight, not
  from a per-episode edge.
- The relationship is not monotone in distance. Going out from the high the
  buckets read -0.275, -0.369, **+0.473**, -1.226. The 3-5% bucket is the best
  one in the removed set and is the only one that is positive in dollars
  (+$28,809), and the worst bucket by far is `>5%`, which is four trades and
  75% of them one year.
- Every bucket is dominated by one year: 49.1% 2021 in `<2%`, 58.8% 2024 in
  `3-5%`, 75.0% 2020 in `>5%`.
- The same non-ordering shows up in the neighbours. In the fear-ON cell the
  book keeps at full size, `<2%` and `2-3%` are the BEST buckets (+1.255,
  +1.388) and `3-5%` the worst (+0.046). In the healthy dial<50 book the
  weakest bucket is `3-5%` (+0.264) while `<2%` and `>5%` are both around
  +0.6. Distance to the high orders nothing anywhere in the grid.

So distance to the 252-session high adds nothing beyond what dial >= 50 plus
fear OFF already removes. The two qualifiers overlap because a high dial is
usually printed near a high, which is why 75 of the removed set's 135 signals
sit in the `<2%` bucket to begin with, but conditioning on the distance does
not find a cleaner slice of what is being cut.
