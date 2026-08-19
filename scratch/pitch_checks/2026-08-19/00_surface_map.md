# Surface map, 2026-08-19 (Wednesday, midterm year, tdom 13)

Tape read: `00_tape_survey.py` -> `00_tape_survey.txt`. Live-state numbers:
`00b_state_quantify.py`. Freshest bar 2026-08-18, pipeline all green, no
warnings. Fragility ma10-63d **86.8** (exposure leg killed, mult 0.0x), P/C
fear OFF at the 54.8th pctile, one dial signal on (Low Absorption Ratio, 7th
pctile - a low-commonality/rotation regime reading, which the day below is
the tape version of).

## The one fact that defines today

2026-08-18 was a **maximal healthcare-over-technology rotation day**. XLV
+1.60% closing AT a 52-week high, XLE +1.76% also AT a 52-week high, against
XLK -2.47% and SMH -4.09%, with SPY only -0.68%.

- XLV minus XLK on the day is **+4.07pp, the 99.3rd percentile of 6,729
  sessions** since 2000 and the 97.2nd of the trailing year.
- Max-minus-min sector spread 4.23pp, 88.5th percentile.
- It is NOT a breadth day: 95 of 211 liquid names closed up (45.0%). So the
  correct description is rotation between sectors, not the average stock
  beating the index.
- Single names: MU -7.02, GLW -7.68, INTC -6.57, AMD -4.27, AMAT -3.92,
  TXN -3.77, ADI -3.50 (prints tonight), AVGO -3.17 against JNJ +3.33,
  LLY +3.60, GILD +3.25, KO +2.12, XOM +2.54, CVX +1.50.

That state is not in the registry in any form and is where most of today's
candidate budget goes.

## 1. Calendar events x asset class

Seven events in the [-5, +15] td window. Two are already past (CPI 08-12,
PPI 08-13) and two sit past the 10 td horizon cap (NFP 09-04 at +12,
PPI 09-10 at +15), so they cannot be traded from today and are dismissed on
that alone. That leaves three anchors, and **the registry has closed all
three**, which is the honest headline of this lane.

| anchor | td | status |
|---|---|---|
| vix_expiry 08-19 | 0 | **CLOSED.** VIX-expiry-week drift is mid-month position plus noise (within-month paired excess +0.065%, t=0.67; 2018+ negative) and the settle day itself is the worst day. Pre-expiry short-vol carry separately dead on a gate-matched control. 2026-08-17 then showed vix_expiry and opex are ONE anchor sharing 189 of 307 days. |
| opex 08-21 | +2 | **CLOSED, four ways.** The run into opex (+0.342% vs SPY's +0.374% unconditional), the pre-opex week entered the Friday before, IWM at a 52w high into opex (the gate is an inverter, -0.250% vs +0.373%), and the post-opex vol window which IS the book (event sleeve V4, long SVXY opex MOC -> +3 sessions, ex-September). Anything post-opex short-vol is a book re-run by definition. |
| jackson_hole 08-28 | +7 | **CLOSED as of yesterday, on all five classes.** Rates, gold, FX, small caps and now large caps. The offset ladder is 9-for-9 flat (true offset ranks 8 of 16 at h=10), the unconditional late-August window beats the anchor (+0.234% over 286 starts vs +0.102% over 26), and midterm years invert it to -1.485%. This is the one I would otherwise have spent the morning on: a Fed-communication event 7 sessions out with the dollar at a 21d rank of 14 is exactly the cross I would build. It was built yesterday, on FX among others, and it is empty. |

Crossed against the ten classes, the cells break down as:

- **us_large / us_small x all three anchors** - dismissed, directly killed
  (JH large caps 08-18, JH IWM and opex IWM 08-17, pre-opex week 08-14,
  run-into-opex 08-07).
- **rates, gold, dollar_fx x jackson_hole** - dismissed, killed 08-13
  (rates), 08-17/08-18 (gold, FX) in the five-class sweep.
- **vol x vix_expiry / opex** - dismissed, three separate kills plus book
  overlap with V4.
- **credit, energy, metals, intl x all three anchors** - not examined on
  these anchors, and dismissed on the anchor rather than the class: with the
  offset ladder flat on five classes for JH and the vix/opex pair shown to be
  one contaminated anchor, spending a check on HYG-into-opex is buying a
  ninth draw from a distribution that has produced eight blanks. Recorded so
  the dismissal is visible rather than absent.
- **The one event anchor NOT closed: the earnings calendar.** NVDA prints
  2026-08-26 (+5 td) with the semis complex at a 63-day rank of 4.4, WMT and
  DE tomorrow, TGT/LOW/TJX/ADI tonight. The washed-out-retailer-into-its-own
  print cell died 08-14 on a placebo ladder and the big-box complex version
  died 08-17 on the intact-trend inverter, so retail is closed. Semis into
  the August NVDA print was measured on 08-14 as a LONG and killed with a
  number that points the other way (-0.322% for August prints against SMH's
  +0.424% unconditional h=7, and -1.339% for 2020+ August prints). Taking the
  short is a post-hoc sign flip out of a kill report, which the 08-18 XLU
  finding says do not do - so it goes to a checker with the anti-rescue rule
  attached and the search charged, not as a free candidate. **C4.**

## 2. Tape extremes by class

| class | what is extreme | verdict |
|---|---|---|
| us_large | SPY -1.34% off its 52w high, 21d rank 79 / 63d rank 38. QQQ -3.73% off, SMH -14.82% off. The index is holding while its heaviest sector is not. | **C3, C5** |
| sectors | XLV and XLE both AT 52w highs; XLV 63d rank 98.8, XLK 6.24% off its high. The 4.07pp one-day gap is the 99.3rd pctile. | **C1, C2** |
| semis / tech | SMH 63d rank 4.4, CSCO 0.4, ADI 1.2, GOOG 2.4, INTC 2.8. Whole complex at the bottom of its own 63-day distribution while the index sits near a high. | **C4, C5** |
| rates | TLT +0.38% off its 52w LOW, IEF +0.83%, LQD +0.48% - the tight three-way rung is live for a seventh straight session. ^TNX 21d +2.35% at a 68.7 rank. | watchlist 6/14 own this; both fail their own gates (freshness, distance-from-low). New cross below: **C6** |
| dollar_fx | DXY 21d rank **14.3**, UUP 16.1, both near the bottom of the trailing year, while yields rise. Dollar is the most one-sided macro reading on the tape. | **C6, C7** |
| gold / miners | GLD 21d +8.42% (rank 77), GDX 21d +25.74% (rank 98.8), both down hard on the day (-1.71 / -3.20). NEM 21d +30%, z10 1.94. | ratio reversion dead 08-18; divergence trigger not met (watchlist 4). **C7** takes the macro angle only |
| metals | SLV -3.58% on the day, 21d +12.67 but 63d rank 17.1 and -45.61% off its 52w high - a wrecked chart with a hot month. | dismissed: the silver catch-up cell was examined 08-07 and the 08-18 miner/metal work covers the complex's reversion direction |
| energy | XLE AT a 52w high, 5d rank 91.7, z10 1.79, while USO's 63d rank is **6.0**. Producers at highs, barrel at a 63-day floor. | spread form killed 08-14 on a knife edge (sign flips at 18pp, live spread 18.36pp). Outright not measured: **C9** |
| credit | HYG 0.33% off its 52w high, LQD 0.48% off its low - the joint extreme is live and unchanged. | watchlist 2, episode count. PASS |
| intl | EEM -2.94% on the day, 63d rank 7.1; EWZ z10 -1.51 (lowest on the tape); FXI 5d rank 29. | EWZ killed 08-17 at today's exact depth; FXI on watchlist 10 and its gates fail; EEM's break is the same family. Dismissed. |
| vol | VIX +4.28% to 15.8 on a -0.68% SPY day, 21d rank 17.1. ^SKEW 5d rank **94.8** (0.2 short of watchlist 7's gate), z10 1.35. | "fear without damage" killed 08-18 (the no-damage leg is an inverter). Watchlist 15 needed a >=5% pop and got 4.28%. Dismissed. |
| single names | META 21d rank 4.0 / 5d rank 5.2, 3.5% off a 52w low, -30% on the year, while SPY sits 1.3% from its high. | **C10** |
| staples / food | CPB 63d rank 98.8 while -29.75% off its 52w high; GIS 98.8 at -22.21% off; CAG 96.0 at -20.52% off. A left-for-dead complex thrusting out of a base. | **C8** |

## 3. Seasonal and cycle cells

Midterm year (year%4==2), August, trading day 13. The board's own read is
de-risk: book win 56.4% vs 64.9% all-years on 1,099 midterm trades, and every
sleeve tilt it prints is a fade. Midterm is treated as a CONDITIONER on the
candidates below, not as an idea. Two specific midterm facts inherited from
this week: the JH anchor inverts to -1.485% in midterms, and the short-SPY
late-August midterm cell was killed twice (08-07, re-found and re-killed
08-17 on concentration and sign instability at h=10 vs h=21). Month-end is
2026-08-31, **8 sessions out**; the parked month-end duration anchor wants
minus-9, which was yesterday, and its distance-from-the-low gate fails
anyway.

## 4. Watchlist verdicts (15 active, every one answered)

| # | entry | today's number | verdict |
|---|---|---|---|
| 1 | TLT from the NFP close, long end at its floor | next NFP 2026-09-04 = +12 td (past the horizon cap) and midterm | PASS, structurally unreachable until 2027-01 |
| 2 | LQD vs HYG at joint 52w extremes | state live (HYG -0.33% off high, LQD +0.48% off low) but still the cluster begun 2026-07-22; 4 declustered episodes against the 8 required | PASS |
| 3 | SVXY overnight into CPI | next CPI 2026-09-11 = +17 td | PASS, re-measure owed at the 2026-09-10 run |
| 4 | GLD on a miner-led thrust the metal has not joined | GDX 5d rank 34.1, GLD 33.3 - needs GDX >=95 while GLD <95 | PASS, divergence absent |
| 5 | XLE on a crude one-day pop of 5-6% and >=1.50 ATR | USO 1d **+0.28%** | PASS |
| 6 | TLT with the IG complex pinned at 52w lows | price rung IS live (0.38 / 0.83 / 0.48 against 0.5 / 1.0 / 1.0) but freshness fails again - this is session 7 of the cluster against the >=10-session gap required, and the 08-18 distance gradient now argues against the rung itself | PASS |
| 7 | SPY on a skew spike alone | ^SKEW 5d rank **94.8** against >=95; SPY now -1.34% off its high, which CLEARS the >1% leg for the first time; year is midterm, which fails | PASS, closest it has been - one leg cleared, one missed by 0.2, one structural |
| 8 | Fade a crude thrust out of a deep base | USO 5d rank 65.1 against >=90 (the 63d rank 6.0 leg clears) | PASS |
| 9 | IHI medical-device thrust | 21d rank **99.2** against the 100 required, and the reference-class gate cannot move in a session | PASS |
| 10 | FXI break inside an intact thrust | 5d rank 29.0 (needs <=20), 21d rank 57.1 (needs >=80), EEM 5d -0.14% (needs positive) - all three fail | PASS |
| 11 | Industry breadth washout, trend BROKEN | no coherent industry has >=70% of names at a 5d rank <=20 with a median 63d rank below 70; semis are washed on 63d but their 5d ranks span 4-64 | PASS |
| 12 | TLT on the November month-position effect | parks to trading days 4-12 of November 2026 | PASS |
| 13 | Short SPY at a 52w high with TLT at a 52w low | TLT leg clears (+0.38%); SPY leg fails, -1.34% off its high against the <=0.5% required | PASS |
| 14 | TLT into the month-end close, ungated | TLT +0.38% above its 52w low against the >3% the trigger needs, and today is month-end-minus-8 rather than minus-9 | PASS |
| 15 | SPY on a vol pop inside calm tape | calm leg clears (VIX 21d rank 17.1 <=25) and the spot leg clears (-0.68% < 0.75%), but the pop is **+4.28%** against the >=5% required; the arm condition is a statistical increment (Welch t >=2.0) that no single session moves | PASS, nearest miss on the tape |

Nothing on the watchlist fires. Two entries (7, 15) came within a rounding
error of their tape legs and still fail on a second leg each.

## 5. Axis scoreboard read

4 graded ideas lifetime (event_fingerprint 2 at +0.622R, interaction_cell 1
at +0.146R, relative_value 1 at +0.099R, all four positive). That is a
handful, not a signal, so it does not steer selection today beyond noting
that no axis has bled.

## 6. Candidates selected (10)

Asset classes touched: sectors, us_large, semis/tech, rates, dollar_fx,
gold, energy, staples, single-name us_large. Axes: relative_value,
inversion, interaction_cell, event_fingerprint, historical_analogue.
Event-anchored: C4. Price-state anchored: C1, C2, C3, C5, C6, C7, C8, C9,
C10.

| id | candidate | axis | class |
|---|---|---|---|
| C1 | Long XLV against short XLK after a 99th-percentile one-day healthcare-over-tech rotation, continuation | relative_value | sectors |
| C2 | The same trigger taken the other way, long XLK against short XLV, i.e. the rotation snaps back | inversion | sectors |
| C3 | What a maximal one-day sector-rotation print says about the INDEX, long or short SPY | interaction_cell | us_large |
| C4 | Short the semis complex into the August NVDA print with SMH at a 63-day rank floor | inversion + event_fingerprint | semis |
| C5 | A megacap-growth complex breaking hard while the index holds within 1.5% of its high, both directions | interaction_cell | us_large / tech |
| C6 | Yields rising while the dollar sits at the bottom of its trailing year, a rate rise the currency does not confirm | interaction_cell | rates x dollar_fx |
| C7 | Long gold on the same dollar-at-a-floor-with-yields-up state, the real-rates read | interaction_cell | gold |
| C8 | Names at a 63-day rank >=95 while still deep below their 52w high, the base-breakout cross-section | historical_analogue | staples / cross-section |
| C9 | Long XLE at a fresh 52-week high while crude's 63-day rank sits at 6, the outright rather than the dead spread | interaction_cell | energy |
| C10 | A megacap at a 21-day rank <=5 while the index is within 2% of its 52w high, both directions | price-state | single-name us_large |
