# Refutation ledger: composition, tails, completeness (2026-09-02)

Independent adversarial pass on `sizing_optimal_plan_2026-09-02.html` / the implementation
brief, closing review-email gap 1 (no independent refuters) and gap 2 (shipped forms never
replayed) from the INTERACTIONS / COMPOSITION / TAILS lens, plus the completeness critique.

Method: per-trade multiplier table for the package AS THE BRIEF WRITES IT (GRM 1.25x relative
with overflow longs excluded, WP6 tilt clip 0.7-1.3, OLV `max(recency, depth)` ladder with
depth = filled + later-filled working entries, WCDS/LT 0.8/1.2, OVS extremity 0.7, OVS P2 cap
0.75 -> 1.0, flow 1.2x with the dip_buy dial<50 gate on the brief's family membership, cap
relief 375 max-not-product, OLV composite clip on the ABSOLUTE product tilt x ladder x
pullback x flow <= 1.5, IOB clone 0.5x, clamp extensions at 20 bps nominal x GRM, guard proxy
= (open requirement + 1.10 x staged FILLED entries) / $632k > 60% turns flow + relief off).
Each trade's daily MTM (rebuilt from master_prices, reconciled to booked PnL, same helper the
practitioner used) is scaled by its composed ratio; the per-strategy cap is re-applied per
(strategy, signal date). The practitioner's package C is re-run through the same machinery as a
control (replica at GRM 1.5: 2016+ maxDD -7.84% vs his -7.5%; same episode ordering within
0.4 pt). Ledger gha:33608560596 through 2026-09-01, flat $750k. Scripts and JSON in this folder:
`dd_tail_01_package_tails.py`, `dd_tail_02_variants.py`, `dd_tail_results*.json`,
`june2026_olv_legs.csv`, `interaction_matrix.csv`, `per_trade_ratios.csv`.

Same bounds as the practitioner replay: fills scale linearly, OLV ticker cap and the row-level
P2 cap are not re-simulated, dial before 2026-07-02 is the recompute vintage, the ledger holds
only FILLED entries (so staged-at-full-size margin projections are lower bounds).

---

## (a) Tails of the package as it would ship

| Config (GRM in name) | Ann PnL % | Sharpe | maxDD 2005+ | Worst day | Worst 21d 2005+ | maxDD 2016+ (trough) | Worst 21d 2016+ | Risk x | 2016+ worst episode |
|---|---|---|---|---|---|---|---|---|---|
| (i) today, GRM 1.5 | 24.6 | 2.00 | -12.38 | -5.90 | -9.52 | -7.34 (2024-04-19) | -6.30 | 1.000 | Apr-2024 St OS Sznl / 3x Bear |
| today, pure GRM 1.875 | 29.7 | 1.97 | -15.48 | -7.37 | -11.89 | -9.17 (2024-04-19) | -7.72 | 1.223 | Apr-2024 |
| (ii) study-form package C, 1.875 | 33.0 | 2.15 | -12.50 | -7.33 | -9.51 | **-9.72 (2026-07-01)** | -7.87 | 1.234 | **Jun-2026 OLV -$84k** |
| (iii) SHIPPED form, 1.875 | 30.2 | 2.14 | -11.67 | -6.82 | -9.72 | **-10.53 (2026-07-01)** | -7.06 | 1.160 | **Jun-2026 OLV -$87k** |
| shipped form, GRM-1.5-equivalent | 26.0 | 2.15 | -10.06 | -5.56 | -8.54 | **-9.44 (2026-07-01)** | -6.19 | 0.988 | **Jun-2026 OLV -$78k** |
| (iv) shipped, OLV clip REMOVED, 1.875 | 30.3 | 2.14 | -11.67 | -6.82 | -9.72 | -10.99 (2026-07-01) | -7.28 | 1.161 | Jun-2026 OLV -$91k |
| (v) shipped, flow up-size removed, 1.875 | 29.2 | 2.14 | -11.14 | -6.15 | -9.44 | -9.20 (2026-07-01) | -6.22 | 1.127 | Jun-2026 OLV -$77k |
| shipped, no flow and no relief, 1.875 | 28.5 | 2.12 | -11.14 | -6.15 | -9.44 | -9.08 (2026-07-01) | -6.18 | 1.109 | Jun-2026 |
| shipped but clip on the RATIO to today (practitioner's code), 1.875 | 29.5 | 2.13 | -11.67 | -6.82 | -9.72 | -8.29 (2021-01-27) | -6.03 | 1.145 | Jan-2021 OVS; Jun-2026 #2 at -8.05 |
| same, GRM-1.5-equivalent | 25.4 | 2.14 | -10.06 | -5.56 | -8.54 | -7.13 (2026-07-01) | -5.53 | 0.974 | Jun-2026 still #1 |
| shipped, absolute clip 1.0 instead of 1.5, 1.875 | 29.4 | 2.13 | -11.67 | -6.82 | -9.72 | -8.29 (2021-01-27) | -6.05 | 1.145 | Jan-2021; passes gate |
| same, GRM-1.5-equivalent | 25.3 | 2.14 | -10.06 | -5.56 | -8.54 | -6.83 (2021-01-27) | -5.53 | 0.974 | passes gate |
| shipped, no depth rung, 1.875 | 29.4 | 2.12 | -11.67 | -6.82 | -9.72 | -8.29 (2021-01-27) | -6.05 | 1.142 | Jan-2021 |
| shipped, depth rung capped 0.7, 1.875 | 29.7 | 2.13 | -11.67 | -6.82 | -9.72 | -8.90 (2026-07-01) | -6.15 | 1.150 | Jun-2026 |
| shipped, all OLV levers off, 1.875 | 28.7 | 2.11 | -11.67 | -6.82 | -9.72 | -8.29 (2021-01-27) | -6.10 | 1.129 | Jan-2021 |

**The clip claim is refuted for the clip as written.** The brief's `OLV_COMPOSITE_CLIP = 1.5`
caps the absolute product tilt x ladder x pullback x flow. On a depth-3+ leg that product is
1.17 x 1.0 x 1.15 x 1.2 = 1.61, so the clip trims 7% on the 104 pullback rows and nothing
elsewhere. The thing that made June 2026 the worst episode is the depth rung taking a 0.5x
recency leg to 1.0x (a 2.0x ratio) BEFORE the product ever reaches 1.5. The practitioner's
script clips the RATIO to today's size at 1.5x (`m[o].clip(upper=1.5)` on `new/old x tilt x
flow x pullback`), which is a different and much tighter rule: a 0.5-rung leg can reach 0.75x
absolute, not 1.5x. That is the rule that produced the plan's "package C returns to April 2024"
number, and even that rule only holds at GRM 1.5 by 0.4 pt (replica: Jun-2026 -7.84 vs Apr-2024
-7.47); at the GRM the plan ships, both the study form and the shipped form have June 2026 as
the worst 2016+ episode.

Per-leg, June 2026 (`june2026_olv_legs.csv`): the shipped form sizes WLK 06-05 at 3.00x today,
GLNG/BP/AKAM 06-18 at 2.81x, OXY 06-18 at 3.51x, LYB 06-05 at 2.14x; OLV ratio distribution
under ship: median 1.50x, p95 2.95x, max 3.75x, 28% of OLV legs above 2x today's risk (study
form: uniform ~2.0x, max 2.10x). Window MTM 06-12..07-01: today OLV -$36.5k, ship -$79.2k,
no-clip -$82.3k, ratio-clip -$62.5k, no-flow -$69.6k. Note the same legs booked +$37k (today) /
+$67k (ship) AT EXIT: the episode is an MTM trough on eventual winners (the OXY/USO oil stack),
which is exactly what the disabled 100%-NAV OLV book cap and a margin call would have turned
into a realised loss.

**Gate 1.11 as written: the shipped package fails it.** Condition 1 passes (+5.6 pts/yr at
1.875). Condition 2 (maxDD no worse by >1 pt at GRM-1.5-equivalent) passes on 2005+ (-10.06 vs
-12.38) and FAILS on 2016-07+ (-9.44 vs -7.34, +2.1 pts). Condition 3 (worst-21d no worse by
>10%) passes on 2005+ (+2%) and FAILS on 2016+ at 1.875 (-7.06 vs -6.30, +12%). Condition 4
(2016+ worst episode not Jun-2026 OLV) FAILS at both GRMs. The gate does not say which window;
the brief names both. Fixes that pass all four: absolute clip 1.0 (OLV raises may offset ladder
cuts, never exceed base) or dropping the depth rung; each costs ~0.8 pt of annual PnL vs ship.

Top-10 drawdown episodes 2005+ (shipped, 1.875): 2014-10..12 -11.67 (IOB/MonFri/52wh);
2007-12..2008-01 -10.57 (MonFri/StOS/Sector BO); **2026-06-12..07-01 -10.53 (OLV -87k)**;
2006-01..06 -10.43; 2014-01..02 -9.95 (52wh -50.7k, down from -71.6k today); 2015-03..05 -8.69;
2009-09..10 -8.48 (WCDS); 2021-01-21..27 -8.29 (OVS -54k); 2024-04 -7.79; **2026-07-22..29
-7.52 (3x Bear -19.8k, OLV -16.1k, MonFri -10.4k)**. Today's list has ONE 2026 episode outside
the top 10; ship puts two 2026 episodes in the top 10 and a third (2026-08-26..09-01, OLV
-45.6k, -6.08%, still open) at #6 of the 2016+ list.

## (b) Lever stacking on the worst days

- Rows with 3+ multipliers above 1.0 composed on the SAME trade (grm, tilt, ladder, adds, flow,
  pullback, p2cap, fear boost): 553 of 4,696 (12%); 4+: 78. Ratio to today on the 3+ rows:
  median 1.76x, p90 2.24x, max 3.75x. By strategy: OLV 152, OVS 139 (grm+flow+p2cap), MonFri 81
  (grm+tilt 1.30+flow 1.2 = 1.95x), LT Trend 73, WCDS 41, 3x ETF Fade 36.
- Composed pre-GRM product across the book: p90 1.35, p99 2.70, max 3.23; 391 rows above 1.5x
  pre-GRM, 230 of them NON-OLV, i.e. outside the only clip that exists.
- Strategy-days whose pre-cap staged risk exceeds 250 bps: 32 today -> 59 ship; above 375: 28.
  The relief covers 760 of the 1,244 cap-bound rows. Book-level staged risk on the top 3+-lever
  days: 2021-07-16 531 bps, 2020-07-13 512, 2024-12-18 491 (17 LT Trend rows 152 -> 232 bps),
  2013-05-31 491, 2025-01-07 420 (15 OLV rows 250 -> 375).
- **Max-not-product holds nowhere except two places**: `max(recency, depth)` inside the OLV
  ladder, and the relief max, which is vacuous because the standing OVS 375 was dropped and the
  flow relief is the only relief left. Everything else is a product: GRM x tilt x P/C fear boost
  x flow x adds/ladder x pullback x (earnings override composed with all of the above). A MonFri
  fear-ON hi-flow leg is 1.25 x 1.30 x 1.2 x 1.875 = 3.66x nominal (1.875x today); historically
  0 such rows, structurally allowed.
- Worst 20 shipped days: 2025-01-10 (-$33k vs -$17k today, 22 of 23 open rows carry 3+ levers,
  mean ratio 1.82), 2026-07-01 (-$25k vs -$13k, 12 of 22), 2026-08-28 (-$24k vs -$11k, 11 of
  15, mean ratio 2.35, dial 87.6), 2026-03-20 (-$22k vs -$12k, dial 55.9). Seven days are new to
  the worst-20 vs today: 2020-06-08, 2023-02-07, 2024-02-15, 2026-03-20, 2026-06-26, 2026-07-01,
  2026-08-28. 2026 goes from 1 to 5 of the worst 20.

## (c) Correlated tail

- Worst 20 SPY days since 2010: book sum today -3.19% of NAV, pure GRM 1.875 -4.93%, shipped
  -0.91% (the short book at 1.875 and the 3x fade tilt pay on 2025-04-04 and 2020-06-11); mean
  net exposure on those days 0.21 NAV both before and after; 2018-02-05 is the one that worsens
  (-3.77 -> -4.84%, net long 1.33 -> 1.68 NAV). SPY crashes are not where the package's tail is.
- Conditional beta (book/NAV on SPY, 2016-07+, lag-1 live dial): today dial>=50 beta 0.387 /
  <50 0.211; shipped 0.480 / 0.223; study form 0.534 / 0.252. PIT dial 2018+: today 0.225 /
  0.212, shipped 0.294 / 0.222. **The package raises conditional beta at dial>=50 by 24-31%
  and below 50 by ~5%.** Net long exposure at dial>=50: mean 0.30 -> 0.36 NAV, p95 0.97 ->
  1.57 NAV. The dial-armed hedge on the shipped book: arms 0 days in the June 2026 episode (dial
  ~20) and 5 days in the Aug 2026 one (+$1.5k against -$32.8k of book); hedged 2016+ maxDD is
  unchanged at -10.53. The hedge is irrelevant to the package's new worst episodes, which are
  OLV-idiosyncratic at dial 20 (consistent with the 2026-08-25 SPY-hedge study in CLAUDE.md).
- Worst-20 book-day migration (2016+): share at dial>=50 0% today -> 10% ship (2026-08-28,
  2026-03-20) -> 15% study form; mean dial 20.7 -> 27.1.

## (d) Completeness: live overlays x new levers (rows carrying both, ledger)

| Live overlay | rows | GRM step | tilt | OLV ladder up | adds | OVS ext | flow up | relief day | pullback | P2 relax | clamp ext | Plan addressed? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frag band, FAMILY4+ at dial>=50 (present rows are fear-ON 1.0x; zeroed rows absent) | 17 | 17 | 12 | 0 | 1 | 0 | 0 | 8 | 0 | 0 | 0 | Yes (tables unchanged, hedged re-score prereg) |
| P/C fear boost 1.25x | 48 | 48 | 39 | 0 | 3 | 0 | 1 | 6 | 0 | 0 | 3 | **No**: 1.25 x tilt 1.30 x flow 1.2 x GRM allowed on MonFri (3.66x nominal); 16 MonFri boost rows |
| OLV recency ladder 0.5/0.7 | 264 | 72 | 264 | 142 | 0 | 0 | 96 | 114 | 104 | 0 | 0 | Yes (re-keyed), but depth 3+ and hi-flow are the SAME state: P(depth>=3 given hi-flow) 0.83 vs 0.14; 111 of 137 depth-3+ legs are hi-flow, so flow 1.2 and the regrade double-count |
| earnings size override (OLV 10 bps, St OS 6) | 68 | 14 | 68 | 26 | 0 | 0 | 21 | 30 | 16 | 0 | 0 | **No**: brief tilts the override and composes ladder/flow/pullback; 59 OLV rows go 10.5 -> 20.6 eff bps mean, max 28.1, 28 rows >2x; a "small-N appetite haircut" doubled |
| OVS cycle-year 0.75 | 525 | 525 | 0 | 0 | 0 | 172 | 72 | 138 | 0 | 295 | 0 | Partly: ext x cycle = 0.525x on 172 rows (fine); flow x P2 cap not modelled |
| same-day derate (3x Bear) | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | Yes (bear_etf_fade excluded from flow) |
| gap-size derate (Monday Dip, MonFri) | 109 | 109 | 109 | 0 | 0 | 0 | 27 | 32 | 0 | 0 | 3 | Ignored; benign (a cut) |
| overlap clamp IOB+MonFri | 188 | 188 | 188 | 0 | 0 | 0 | 99 | 101 | 0 | 0 | 4 | Yes (extended, 38 new rows); IOB clone 0.5x also lands on 50 clamped rows (double cut, variance only) |
| OLV ticker notional cap | 0 in ledger | | | | | | | | | | | Ignored; the depth regrade fills stacks faster; cap counts FILLED legs only |
| ADV participation cap | n/a | | | | | | | | | | | Re-expressed: 203 rows over the 1%/0.4% rule (LT 50, OVS 46, WCDS 35; 152 overflow), $242k of PnL in those rows, mean haircut 0.51 if trimmed; 21 rows over the 5% refusal ($11k) |
| per-strategy 250 cap bound | 1,244 (26.5%) | 1,207 | 141 | 22 | 75 | 295 | 656 | 760 | 23 | 206 | 4 | Yes (relief), but relief is 0.7 pt/yr of the 5.6 and the guard basis is fills-only here |
| OVS 2-path P2 rows / P2 cap-scaled | 1,183 / 495 | | | | | 391 | 212 | 280 | | 495 | | **Partly**: flow 1.2 on 212 P2 rows pushes into the fixed P2 aggregate cap on exactly the cluster days; not re-simulated (plan admits) |
| OVS scale-out tranches | 2,426 | 2,426 | 0 | 0 | 0 | 748 | 672 | 822 | 0 | 478 | 0 | Ignored; benign (split is a fraction) |
| WCDS legacy seasonal Size_Mult 1.5/0.66 | 153 | 153 | 153 | 0 | 153 | 0 | 36 | 45 | 0 | 0 | 8 | Flagged as a basis mismatch only; 1.5 x 1.2 adds x 0.75 tilt = 1.35 pre-GRM on 120 rows |
| trend sleeve (0.3 NAV MOO) | not in ledger | | | | | | | | | | | **Ignored**: ~2.4% NAV of requirement at 8%, additive to WP1's Req_proj |
| event sleeve (SPY 25%, IWM 15-25%, SVXY 5-10% NAV) | not in ledger | | | | | | | | | | | **Ignored**: 3-8% NAV of requirement (SVXY margin is the unknown), additive |
| exposure leg (25% NAV VOO/QQQ, dial-killed) | not in ledger | | | | | | | | | | | **Ignored**: 2% NAV requirement when on; its raw-21d kill fires at the same dial the hedge arms |

Also unaddressed: flow family membership. Thresholds 6/7/104 were fit with St OS Sznl in
oversold_hold, 3x Bear in dip_buy and 3x Leader in short_fade (`flow_conditional_lib.FAMILY`);
the brief moves St OS to dip_buy and carves out bear_etf_fade. 86 rows flip hi-flow state
(3x Bear 25, IOB 14, St OS 14, MonFri 12); tail effect nil, but the thresholds are now
defined on a different variable than the one whose terciles they were.

## (e) Not considered by the plan

1. **The disabled OLV book cap vs the package's OLV footprint.** Since 2016-07 OLV open
   notional exceeds 100% of NAV on 15 days today, 28 at pure GRM 1.875, **60 under the shipped
   package (max 4.1x NAV on 2025-01-13, p99 1.64x)**. Plan section 13 closes "OLV sleeve caps
   and the retired 100%-NAV book cap" while shipping a rule set that quadruples the days over
   the line McKinley said he never wanted crossed (2026-08-24).
2. **Guard projection basis.** WP1 projects staged entries at full size; OVS stages ~16x its
   fills. This replay (fills only, $632k NLV) has the guard above 60% on 53 days and above 70%
   on 25 (1.1/yr, not 0.5), with a negligible trim cost ($6k over 23y; 15 trim days, mean
   factor 0.32). Live, with unfilled OVS/OLV limits in the projection, the 60% line will be
   crossed far more often, and on exactly the short_fade hi-flow days: WP1 and WP8 can cancel
   each other by construction. On the $750k base the counts are 14 / 2.
3. **Depth counted on working entries = signal count.** 5 of 137 depth-3+ legs reach the rung
   only through working limits, so the ledger effect is small, but the semantics turn the T+3
   blind spot (three unfilled full-size limits) into an up-size trigger for the fourth.
4. **The regraded legs' 2026.** The 61 legs the ladder takes from 0.5x to 1.0x average +0.86R
   over the sample, +0.66R since 2024, **-0.03R on the 20 legs in 2026** (PnL +$0.7k). The
   package's largest per-leg step (2x) lands on the cell the review email calls "unexplained".
   LT Trend (raised 1.04 x 1.2 adds x 1.2 flow) is +0.01R on 23 trades in 2026.
5. **Year shape.** Shipped vs pure-GRM 1.875: better in 13 of 22 years; 2013 -$101k (52wh cut
   0.70 in its best year), 2008 -$28k, 2017 -$28k; 2015 +$71k, 2016 +$47k, 2026 +$44k. Half the
   package's gain is 2015-2016 and 2020-2021 and 2026.
6. **PA account.** Reg-T, own cap, mirrors staged rows: every raised row is clipped harder there,
   so the two books diverge more. Not modelled anywhere (plan admits; no number).
7. **Dial vintage.** 7 dip_buy hi-flow rows flip the dial<50 gate between PIT and live dial;
   the hedge and the LT Trend cell stop at PIT 2026-05-07 so the Aug-2026 -6.08% open episode
   (dial 85-89, OLV -45.6k) is current-weights only. Small.
8. **Tax/wash.** Stacked same-ticker OLV legs on 10-day holds and OVS near/far tranches are the
   wash-sale-prone structures; the package raises their size and count. Not quantifiable from
   the ledger; zero mention in the plan.
9. The guard's own cost is measurable now (item 2): the email's "trims the book's best days" is
   not borne out on filled entries ($6k).

---

## Refutation ledger (composition / tail lens)

| # | Change | Verdict | The number |
|---|---|---|---|
| 1.1 | GRM 1.5 -> 1.875 | STANDS-WEAKENED | Linear as claimed (maxDD -12.4 -> -15.5, worst day -5.9 -> -7.4% = -$55k, w21 -9.5 -> -11.9). Guard >70% on 25 fills-only days on $632k (1.1/yr vs the plan's 0.5); trim cost ~$6k/23y. Live projection with unfilled staged rows is a lower-bounded unknown. |
| 1.2 | Overflow longs excluded, participation rule | STANDS / UNTESTABLE | Overflow ratio 1.0 works; the 1%/0.4% rule binds 203 rows ($242k PnL, mean haircut 0.51), 5% refusal 21 rows ($11k). Fills at 1.875x never modelled. |
| 1.3 | Keep-adjusted tilt | STANDS-WEAKENED | Tilt alone at 1.5-eq: 2005+ maxDD -12.05 -> -10.06 (helps) but 2016+ -8.34 -> -9.44 (hurts), because OLV 1.17 and MonFri 1.30 feed the 2026 stacks. 13/22 years better vs pure GRM; 2013 -$101k. |
| 1.4 | OLV ladder re-key + composite clip 1.5 | **REFUTED as written** | Clip on the absolute product bites 7% on 104 rows; June 2026 is the worst 2016+ episode at -10.53% (1.875) and -9.44% (1.5-eq) vs -7.34% today; OLV legs p95 2.95x, max 3.75x today's risk. The number the plan quotes came from a RATIO clip (1.5x today), which passes at 1.5 by 0.4 pt and fails at 1.875 (-8.29 with Jun-2026 at -8.05). Depth 3+ and hi-flow are one state (0.83 vs 0.14): double-count. Passing forms: absolute clip 1.0, or no depth rung (each ~-0.8 pt/yr vs ship). |
| 1.5 | WCDS / LT Trend 0.8 / 1.2 | STANDS-WEAKENED | Composes with tilt, flow and (WCDS) the legacy 1.5 seasonal mult: LT legs at 1.04 x 1.2 x 1.2 x 1.25 = 1.87x on 17 rows 2024-12-18 (152 -> 232 bps); LT 2026 avgR +0.01 (N 23) while its band prereg waits in Phase 3. |
| 1.6 | OVS extremity 0.7 | STANDS | A cut; x cycle 0.75 = 0.525 on 172 rows; x flow 1.2 = 0.84. No tail interaction found. |
| 1.7 | OVS P2 cap 0.75 -> 1.0 | STANDS / UNTESTABLE | 495 cap-scaled P2 rows relaxed by <= 4/3; 212 P2 rows also get flow 1.2, which the fixed P2 aggregate then neutralises on cluster days (not re-simulated, plan admits). |
| 1.8 | Index clone clamps | STANDS | 152 IOB clone rows, 38 new clamp rows; 50 rows carry the live clamp AND the clone cut. Variance only. |
| 1.9 | Flow cap relief 375, max-not-product | STANDS-WEAKENED | Worth +0.7 pt/yr (28.5 -> 29.2); strategy-days >250 pre-cap 32 -> 59, >375 28. Max-not-product is vacuous (one relief). Guard basis (staged full-size incl. unfilled, OVS ~16x fills) can switch it off on the days it targets. Family membership differs from the fit (86 rows flip). |
| 1.10 | Flow up-size 1.2 | STANDS-WEAKENED | Worth +1.0 pt/yr; costs 2016+ maxDD -9.20 -> -10.53 and w21 -6.22 -> -7.06 (+12%, over the gate's 10%); 2005+ worst day -6.15 -> -6.82. On OLV it is the depth regrade again. |
| 1.11 | Engine gate | STANDS, and the shipped package FAILS it | Cond 1 pass (+5.6); cond 2 fails on 2016+ (+2.1 pt at 1.5-eq); cond 3 fails on 2016+ (+12%); cond 4 fails at both GRMs. The gate must name its window. |
| 2.1 | Dial-armed beta hedge | STANDS-WEAKENED | Arms 0 days in Jun-2026, 5 in Aug-2026 (+$1.5k vs -$32.8k); package raises dial>=50 beta 0.39 -> 0.48 (PIT 0.225 -> 0.294) and p95 net long at >=50 from 0.97 to 1.57 NAV; a 1.0x-capped, 126d-lagged hedge sees that late. |
| 2.2 | OLV pullback 1.15 | STANDS-WEAKENED | 104 of 315 legs; inside a clip that barely binds; 16 of the 59 earnings-override rows; in Jun-2026 it sits on the OXY/USO legs that won. Harmless alone, one more factor in the product. |
| S.3 | 250 cap fixed | STANDS | Alone bounds the worst day; the relief and the guard interact with it, not the cap itself. |

## WORTH DISCUSSING

1. **Which clip did you approve?** The plan's composition numbers come from a ratio clip (OLV
   leg <= 1.5x today's risk); the brief ships an absolute clip (product <= 1.5) that does almost
   nothing. Under the brief the June 2026 stack is the worst episode since 2016 at both GRMs, and
   the package fails three of its own four gate conditions. Absolute clip 1.0 or no depth rung
   passes everything at ~0.8 pt/yr; decide which, and write the gate's window.
2. **June 2026 is an MTM trough on winners** (+$67k at exit on the shipped sizing). Insuring
   against it is really insuring against forced liquidation with the OLV book cap disabled, on
   a sub-book that the package puts above 100% of NAV on 60 days since 2016 (max 4.1x). Either
   re-arm the EOD book cap (built, dry-run verified) or accept that the guard is the only thing
   between a 4x-NAV OLV stack and IBKR's 15:45 liquidation, on a $632k NLV.
3. **Depth and flow are one variable on OLV** (83% overlap). Ship one of them, not both, or the
   2x regrade times 1.2 lands on the 61 legs whose 2026 realisation is -0.03R.
4. **The guard and the flow rules cancel by construction.** Req_proj counts staged entries at
   full size; OVS stages ~16x its fills; the 60% line will trip on the short_fade hi-flow days
   and switch the relief and up-size off. Decide whether staged rows enter Req_proj at expected
   fill rate (measurable from WP4) or accept that WP8 is mostly off for OVS.
5. **The package moves risk into the dial>=50 zone** (conditional beta +24-31%, p95 net long 1.57
   NAV) while the only dial control is a lagged, 1.0x-capped hedge that did not arm in either
   2026 episode. The sleeves' margin (trend 0.3 NAV, event sleeve up to ~50% NAV of index plus
   SVXY, exposure leg 25%) is not in WP1's projection.
6. **The earnings override is silently doubled** (59 OLV rows 10.5 -> 20.6 eff bps mean, 28 rows
   >2x): the brief says "tilt the override too" and composes the ladder, flow and pullback on it.
   If it is still an appetite haircut, freeze it in effective bps.
