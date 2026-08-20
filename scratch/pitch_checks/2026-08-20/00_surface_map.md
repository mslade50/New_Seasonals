# Surface map, 2026-08-20 (Thursday, midterm year, tdom 14 of August)

Freshest bar 2026-08-19. Pipeline 7/7 green, no state warnings. Entry convention
for everything below: signal on the 08-19 close, entry MOC **today** (08-20),
which is **opex-1**. Horizon cap 10 td, so the last reachable close is 09-03.

Tape sort: `00_tape_sort.py` / `.txt`. Recon: `01_recon.py`, `02_recon2.py`,
`03_recon_events.py`. Watchlist arithmetic: `04_watchlist_verdicts.py`.

## Where the tape actually is

One sentence, because everything below hangs off it: **the index did nothing
while the cross-section detonated**. SPY +0.21% on a day MRK printed +12.6%,
GDX +9.4%, IBB +6.6%, XBI +5.9%, NUE -5.9%, GE -5.0%, AVGO -4.6%. Cross-sectional
sd of daily returns 2.455%, the 89.3rd percentile of full history; sd divided by
|SPY| is 11.7, the 86.9th. The risk dials agree from their own side: Dispersion ON
at the 86th percentile (component RV 38.3% against SPY RV 13.3%), Low Absorption
Ratio ON at the 1st, and the fragility dial's 10d-MA 63d sits at **88.8**, up from
42.9 twenty-one sessions ago. Healthcare is the destination (XLV 63d rank 100.0 at
a 52w high, IBB 100.0, XBI 98.8, IHI 97.6), tech and semis are the source (XLK 5d
rank 14.7, SMH 63d rank 3.2), and the dollar is on the floor (DXY 21d rank 0.8).

Book posture that every idea below has to be read against: at dial 88.8 with P/C
fear OFF, `PC_FEAR_BANDS` **zeroes** the six band carriers, and `exposure_leg` is
at mult 0.0 on the raw-21d rule. The systematic book is already as defensive as
its rules allow. Anything short here is an addition to that, not a hedge for it.

## 1. Live calendar events x asset class

Seven events sit in the [-5, +15] td window. Only **two are reachable** inside the
10 td horizon cap: opex at +1 and Jackson Hole at +6. NFP (+11), PPI (+14) and CPI
(+15) cannot be held to; they are dismissed on the horizon, not on their merits,
and the two past anchors (PPI -5, VIX expiry -1) get verdicts as trailing anchors.

Grid measured, not asserted: `03_recon_events.py` crosses both live anchors with
15 vehicles spanning all ten classes at h=1/2/3/5/10. Verdicts below cite it.

| class | vehicle | OPEX-1 (pooled, N=319) | OPEX-1 (August, N=26) | JH-6 (N=26) | verdict |
|---|---|---|---|---|---|
| us_large | SPY | h1 -0.07% ex -0.1, h5 +0.11% ex -0.0 | h3 -0.02% at a 69% hit, h5 +0.42% ex +0.2 | h5 -0.20% ex -0.3, h10 +0.64% ex +0.2 | JH **closed** by registry across 5 classes incl. large caps, ladder 9-for-9 flat. Pooled opex excess is -0.1pp at h1/h2 and gone by h3: ~7 bp on a 2 bp round trip, not a trade. August SPY has a 69% hit at h=3 on a ZERO mean, i.e. fat left tail, so **C3 takes the small-cap leg, not this one** |
| us_small | IWM | h5 +0.27% ex +0.0, h10 +0.67% ex +0.2 | **h5 +0.74% (65%), h10 +1.60% (69%) ex +1.1pp** | h10 +1.61% ex +1.1 | **CANDIDATE C3.** Best cell in the grid. Owes the offset placebo ladder that closed JH, the midterm split that killed the JH-IWM version, and a year histogram |
| rates | TLT | h10 +0.46% ex +0.3 | h10 +0.72% at a 75% hit ex +0.5 | h8 +0.79% ex +0.6 | dismissed as an EVENT cell: registry killed JH-rates outright and the opex numbers sit inside noise on 26 August anchors. Rates gets a price-state candidate instead (**C2**) |
| credit | HYG | h10 +0.45% (64%) ex +0.2 | h10 +0.64% at an **89% hit** ex +0.4 | h10 +0.67% at an 84% hit ex +0.4 | dismissed: the hit rate is HYG's unconditional character (0.24% ATR, the lowest in the tape), the excess is +0.4pp of a 10-day hold on an instrument whose spread eats a chunk of it, and both anchors give the same number, which is the signature of a calendar-agnostic drift |
| gold | GLD | h3 +0.26% ex +0.1 | h10 +1.14% (76%) ex +0.6 | h10 +1.20% (76%) ex +0.7 | dismissed: registry closed JH-gold explicitly (10-11, 92% two episodes, midterm -1.213% at 1-4). The August opex number is the same late-August seasonal wearing a different anchor |
| metals | SLV | h5 +0.45% ex +0.2 | h10 +2.12% ex +1.6 | h8 +1.51%, h10 +1.91% ex +1.4 | dismissed on the vehicle and the family: silver thrust-from-drawdown is registry-dead (2026-08-10), miner-vs-metal is dead (08-18), and a +1.6pp excess on 26 anchors in a 3.11%-ATR instrument is one episode wide |
| energy | USO | h10 +0.52% ex +0.5 | **h5 +0.95% (65%), h10 +1.47% (65%) ex +1.4** | **h6 +1.78% (70%) ex +1.7, h10 +2.15% ex +2.1** | **CANDIDATE C8.** Energy is the one class the JH sweep never reached (registry covered rates, gold, FX, small caps, large caps). Registry's own prior applies: the anchor is probably decoration on a late-August seasonal, so the offset ladder IS the check |
| energy_eq | XLE | h5 +0.47% ex +0.2 | h5 +0.90% ex +0.6 | h6 +0.92% ex +0.6 | folded into C8 as the vehicle question rather than a separate candidate; XLE's crude beta of 0.479 is registry-established (2026-08-11) so it cannot be pitched as producer alpha |
| dollar_fx | DX-Y.NYB | flat at every horizon, h10 -0.00% | h5 +0.12% ex +0.1 | h10 -0.02% ex -0.0 | dismissed: registry closed JH-FX (13-13, drop-best flips the sign) and the opex grid is a flat line to three decimals. FX gets a price-state candidate instead (**C6**) |
| intl | EEM | h10 +0.78% (62%) ex +0.3 | h10 +1.20% (65%) ex +0.7 | h10 +1.21% ex +0.7 | dismissed as an event cell for the same reason as gold: an identical number under two unrelated anchors is a seasonal, not an event. International gets a cross-asset candidate instead (**C5**) |
| vol | ^VIX / SVXY | **SVXY h3 +1.25%, h5 +1.43% ex +0.8pp at a 60-63% hit** | **SVXY h3 -0.73%, h5 -1.27%, h10 -2.01% ex -3.2pp; VIX h2 +1.90% ex +1.4** | SVXY h8 -1.61% ex -2.6 | **CANDIDATE C4, and the most consequential thing in the grid.** The pooled cell IS the event sleeve's live V4, which stages at tomorrow's opex MOC. August is the second month after September where it inverts. Must respect the 2018-02-28 SVXY leverage break (registry 08-14, 08-17) |
| tech | XLK | h1 -0.19% ex -0.2, h3 -0.16% ex -0.3 | h5 +0.70% ex +0.4 | h3 -0.36% ex -0.5 | dismissed: sector-vs-index pairs on a leadership trigger are a dead family (three registry entries), and the standalone excesses here are 0.2-0.5pp on 26 anchors |
| defensive | XLV | flat, h3 -0.05% ex -0.1 | h5 +0.30% ex +0.1 | h3 -0.41% ex -0.5 | dismissed, same family kill; see also the healthcare price-state dismissal in section 2 |

Trailing anchors, verdicts owed:

- **PPI, 2026-08-13, -5 td.** Registry (2026-08-10) swept PPI on equities across 323
  events and on the curve, and found the effect is exactly one session wide. A -5 td
  anchor is five sessions past the only session that carried anything. Dismissed.
- **VIX expiry, 2026-08-19, -1 td.** Registry (2026-08-17) established that VIX expiry
  and opex are ONE anchor sharing 189 of 307 days, and any grid crossing both
  double-counts a single calendar fact. This month's spacing is the common +2 td case.
  Folded into the opex row; not counted twice.
- **NFP 09-04 (+11), PPI 09-10 (+14), CPI 09-11 (+15).** Outside the 10 td cap.
  Nothing entered today can be held to them without breaking the grammar. Dismissed
  on horizon. The watchlist's NFP-rates entry and CPI-SVXY entry park to those dates.

## 2. Tape extremes, by class

Every class gets a line whether or not it produced a candidate.

- **us_large.** SPY 1.13% off its 52w high, 63d rank 49.2, ATR 0.92%. The extreme is
  not the level, it is the DISPERSION around it (see the opening paragraph).
  Gives **CANDIDATE C1**, the index-quiet / components-wild cell, 131 declustered
  episodes, SPY excess -0.083 / -0.240 / -0.158 / -0.633pp at h=1/3/5/10 and VIX
  +6.5% at h=10 (`02_recon2.py`).
- **us_small.** IWM 1.10% off its high, 63d rank 68.3, leading SPY by 5.7pp over 63d.
  No price-state extreme; taken through the calendar instead (**C3**).
- **rates.** TLT +1.67% yesterday, its biggest session in a while (volume 1.86x its
  63d norm), from 2.05% above a 52-week low. IEF 1.31% off its low, LQD 1.17%. The
  whole IG complex is still pinned near the bottom of its year. Gives **CANDIDATE
  C2** (18 declustered episodes; the forward is decisively NEGATIVE, h=2 -0.638% at
  a 24% hit, so this is a short and the sign came out of the recon, which the check
  must charge for). Curve: 10y-5y +0.300, the 20th percentile of the trailing year;
  ^TNX 21-session change +0.025pt, which is what kills watchlist W16.
- **credit.** HYG 0.10% off its 52w high with the lowest ATR in the 218-name tape
  (0.24%), LQD 3.15% off its high and 1.17% off its low. Credit is priced for
  nothing to happen while equity dispersion sits at the 89th percentile and the
  fragility dial reads 88.8. Tempting and dismissed: the HYG-high / LQD-low joint
  state is watchlist W2 and still has **4 declustered episodes, three of them in
  2018** (`04_watchlist_verdicts.py`), and a dial-conditioned credit short is the
  registry's dead "dial as a direction" wearing a credit ticker.
- **gold / miners.** GDX +9.42% on the day, 21d rank 100.0, +31.19% over 21 sessions;
  GLD 21d rank 87.7. Dismissed as a closed family: GDX's maximal 21d thrust died to
  its reference class (08-17), miner-versus-metal ratio reversion died (08-18), and
  the premise correction filed on 08-19 still holds today, since GLD's 63d rank is
  34.9 and it is 16.55% below its 52-week high, so this is a bounce inside a
  drawdown rather than a gold bull market.
- **metals, other.** SLV 43.17% below its 52w high but +73.69% over a year, 63d rank
  28.2, the laggard inside the metals thrust. Dismissed: "silver thrust from deep
  inside a drawdown" is registry-dead (08-10) and "adding a second metals leg beside
  a live one" is dead (08-11). XME at a 21d rank of 84.1 while 11.80% below its high
  is the base-breakout shape that died on 08-19 at P(max name mean >= observed)=1.000.
- **energy.** XLE at its 52-week high (-0.16%), z10 +2.22, 5d rank 88.5, while USO's
  63d rank is 4.4 on roll decay. Dismissed as a price-state cell, since that exact
  divergence was killed on 08-19 (energy at a fresh high with crude at a 63d floor)
  and on 08-14 (producers against the barrel, all three readings). Energy survives
  only through the calendar door (**C8**).
- **dollar / FX.** DXY 21d rank 0.8, UUP 0.4, 2.71% above a 52-week low. The loudest
  rank in the tape. Gives **CANDIDATE C6**, and the recon has already half-killed it:
  the rank bought a 21-day move of **-2.32%, which is -1.05 sd and the 13.4th
  percentile of full history**, and the rank form (+0.509pp at h=10) and the
  magnitude form (-0.015pp) disagree in sign (`02_recon2.py`).
- **international.** EEM 7.16% off its high with a 63d rank of 13.5, FXI +1.77%,
  EWJ 5d rank 7.9. The live hook is not a country, it is the dollar. Gives
  **CANDIDATE C5** (KWEB on the dollar washout, +1.209pp excess at h=5 over 18
  episodes, and the only positive of seven risk assets on the same trigger, which is
  the thing to attack).
- **volatility.** VIX 15.1 and -6.0% on the session, 21d rank 22.2; VIX3M -3.63%;
  MOVE -4.96% and -16.48% over 63d; SVXY at its 52-week high; ^SKEW 5d rank 89.3,
  63d rank 86.5. Macro vol is dead while micro vol is at the 89th percentile, which
  is the same fact as C1 read from the other side. Skew-plus-low-vol is explicitly
  registry-dead ("the filter subtracts", 08-12; both poles closed 08-14), and
  watchlist W7 fails on the 5d rank (89.3 against 95) and on the structural midterm
  leg. Volatility's candidate is the calendar one (**C4**).
- **sectors.** XLV 63d rank 100.0 at a 52w high with IBB 100.0, XBI 98.8 and IHI 97.6
  alongside it. Counted before designing anything, per the registry rule: the 4-way
  joint state has **24 days and 4 declustered episodes in 26 years** (`01_recon.py`),
  which is unmeasurable, and long-IBB-on-healthcare-leadership is separately dead
  (08-11, sign inverts, bootstrap P(mean<=0) 0.985). Dismissed. The tradeable residue
  inside sectors is the BANKS split: **72.7% of the 11-name bank complex sits at a 5d
  rank <= 20 while its median 63d rank is 82.5**, with KRE at a 5d rank of 8.7 against
  XLF's 63d rank of 97.2. Gives **CANDIDATE C7**.
- **single names.** MRK +12.6% in one session on 3.08x normal volume, the largest
  megacap print on the tape. Dismissed: a single-name repricing owes a reference
  class before it owes anything else (08-13), the book's ATR Extended Gap Up and
  Overbot Vol Spike both live on exactly this shape from the short side, and the
  last two weeks of single-name cross-sections have died to the alphabetical placebo.

## 3. Seasonal and cycle cells

- **Month position.** Trading day 14 of August; the month's last session is 08-31,
  which puts today's close at month-end **minus 7**. The month-end TLT anchor
  (watchlist W14) is defined at minus-9 and additionally needs TLT more than 3%
  above its 52-week low, against 2.05% today. Both legs fail.
- **Late August.** Reachable, and it is where C3, C4 and C8 all live. Registry has
  already closed the run INTO August opex (2026-08-07) and midterm mid-August
  seasonality (N=6, carried by 2002). What is NOT closed is the run OUT of it, which
  is what the grid above measures for the first time.
- **Cycle year.** Midterm (year mod 4 = 2). This is a conditioner on every candidate
  rather than a candidate itself: the seasonal board reads book win% 56.4 against
  64.9 all-years on 1,099 midterm trades, and the midterm split has independently
  killed the JH-IWM, JH-gold, TLT-NFP, DX-NFP and UUP-NFP cells. **Every
  event-anchored candidate below owes its midterm split.**
- **Seasonal board.** 0 A/B-grade setups across 2 channels as of its 08-05 asof, and
  its only non-regime line is a P/C complacency read that the live risk state
  contradicts (equity P/C at the 51st percentile today, fear OFF). Nothing to take.

## 4. Axis scoreboard read

4 graded ideas lifetime, all opus: avg +0.372R, 4 for 4. By axis: event_fingerprint
2 at +0.622R, interaction_cell 1 at +0.146R, relative_value 1 at +0.099R. The graded
count is a handful, so this steers nothing yet. Recorded so a future morning can see
it was read rather than skipped.

## 5. Watchlist, 18 active entries, every one with today's number

| # | entry | today | verdict |
|---|---|---|---|
| W1 | TLT on the NFP rates floor | next NFP 09-04 is +11 td, past the horizon cap, and midterm | PASS, structurally unreachable until 2027-01 |
| W2 | LQD vs HYG at joint 52w extremes | HYG -0.10% off its high, LQD +1.17% off its low, so the state is LIVE. Declustered episodes **4**, years [2018, 2026], against the >=8-over->=3-years-ex-2018 arm | PASS, count unmoved |
| W3 | SVXY overnight into CPI | next CPI 09-11, +15 td; the re-measure is owed at the 09-10 run | PASS |
| W4 | GLD on a miner-led thrust | GDX 5d rank 80.2 (needs >=95), GLD 65.5 (needs <95). Both legs fail, and today is the two rallying together, which is the +0.239% cell | PASS |
| W5 | XLE on a crude pop in [5%,6%) | USO 1d +0.19% | PASS |
| W6 | TLT, tight IG rung plus freshness | TLT is now **2.05%** off its 52w low against the <=0.5% rung. The rung itself has broken for the first time in eight sessions | PASS, further from arming than at any point this month |
| W7 | SPY on a skew spike alone | ^SKEW 5d rank **89.3** against >=95; the SPY leg clears (1.13% off its high, needs >1%); midterm fails structurally | PASS |
| W8 | Fade a crude thrust from a deep base | USO 5d rank 68.7 against >=90; the 63d rank 4.4 leg clears; post-2020 episode count still 4 of 8 | PASS |
| W9 | IHI at a 21d rank of 100 | **99.6**, a miss of 0.4, and the reference-class arm (Cochran Q p < 0.05) is untested since | PASS |
| W10 | FXI breaking inside an intact thrust | FXI 5d rank 66.7 (needs <=20), 21d 74.6 (needs >=80), EEM 5d -0.53% (needs positive). All three fail | PASS |
| W11 | Industry breadth washout, trend BROKEN | **First half-fire.** Banks sit at **72.7% of names with a 5d rank <= 20**, clearing the >=70% breadth leg for the first time since the entry was written, but the median 63d rank is **82.5** against the <70 the entry requires, i.e. exactly the intact-trend half the entry says pays -0.789%. Industrials are the mirror: median 63d rank 65.5 clears, breadth 52.4% does not | CHECK, does not arm, and the failing leg is the inverter. Feeds **C7**, which takes the state the entry calls the LOSING half and therefore trades it SHORT rather than long |
| W12 | TLT, November month-position | parks to trading days 4-12 of November 2026 | PASS |
| W13 | Short SPY at a 52w high with TLT at a 52w low | SPY 1.13% off its high (needs <=0.5%) and TLT 2.05% off its low (needs <=1%). Both legs have now failed; yesterday only one had | PASS |
| W14 | TLT into the month-end close, minus 9 | today is month-end minus **7**, and TLT is 2.05% above its 52w low against the >3% the trigger needs | PASS |
| W15 | SPY on a vol pop inside a calm tape | calm leg clears (VIX 21d rank 22.2 <= 25), spot leg clears (SPY +0.21%), and the pop **fails hard**: VIX -6.00% against the >=+5% required | PASS |
| W16 | Gold on an unconfirmed rate rise | dollar leg clears at a DX 21d rank of **0.8**; the yield leg fails at a 21-session ^TNX change of **+0.025pt** against +0.20pt, an eighth of what it needs | PASS |
| W17 | Tech against healthcare after a rotation gap | **STATE FIRES.** Gap 4.57pp (needs >=3.0), SPY 1.13% off its high (needs within 3%), SPY ATR 0.92% (needs <1.2%). The arm does not: drop-best **+0.468%** against +0.50%, **N=21** against >=24, record **12-9** against >=15-9, and today is the **eighth 2026 episode** where the arm explicitly requires three new winners OUTSIDE that cluster. Ex-2026 is 7-7 at -0.599% | CHECK, does not arm. Do not take the naked-XLK reading either; it is the same cluster |
| W18 | Short the dollar on an unconfirmed rate rise | ^TNX 21d rank 46.8 against >=65 | PASS |

Nothing expired. Two entries (W11, W17) went to CHECK and both came back short of
their arms; the rest PASS with today's number recorded above.

## 6. Candidates selected, with coverage accounting

| # | candidate | axis | class | anchor mode |
|---|---|---|---|---|
| C1 | Short SPY on the index-quiet / components-wild state | interaction_cell | us_large | price-state |
| C2 | Short TLT on a big up day from inside the 52-week low zone | inversion | rates | price-state |
| C3 | Long IWM out of August opex | event_fingerprint | us_small | event |
| C4 | The August inversion of the post-opex short-vol cell, i.e. the live V4 window | inversion | volatility | event |
| C5 | Long KWEB on a dollar washout | interaction_cell | international | price-state x cross-asset |
| C6 | Long the dollar after a 21-day rank washout | price-state | dollar_fx | price-state |
| C7 | Short the bank complex on a breadth washout inside an intact trend | relative_value | credit / financials | price-state |
| C8 | Long crude through Jackson Hole, the one class the JH sweep never reached | event_fingerprint | energy | event |
| C9 | Long TIP against IEF: breakevens with the dollar on the floor and gold bid | instrument_translation | rates / inflation | price-state x cross-asset |
| C10 | The opex overnight, i.e. does the post-opex drift accrue overnight or intraday | flow_mechanics | us_large | event |

Coverage check against the requirements:

- **Asset classes touched: 8** (us_large, us_small, rates, volatility, international,
  dollar_fx, credit/financials, energy), against a floor of 4.
- **Novelty axes: 7** (interaction_cell, inversion, event_fingerprint,
  relative_value, price-state, flow_mechanics, instrument_translation), against a
  floor of 4.
- **Event-anchored: C3, C4, C8, C10. Price-state anchored: C1, C2, C5, C6, C7, C9.**
  Both modes present, and C4, C5 and C9 are the two modes CROSSED rather than run in
  parallel, which is the failure this section was rewritten to prevent.
- **Registry collisions declared up front:** C4 sits beside the dead "post-CPI vol
  crush" and the live V4 sleeve trade; C6 sits beside "UUP is dead on drag" and the
  rank-vs-magnitude method trap, which the recon has already fired; C7 sits beside
  the dead insurance-breadth cell and inherits its intact-trend inverter as the
  REASON for the direction; C8 sits beside a Jackson Hole anchor closed in five other
  classes, and the offset placebo ladder that closed them is the check it owes; C1
  sits beside "the fragility dial is not a direction" and must show it is the
  dispersion component and not the composite.
