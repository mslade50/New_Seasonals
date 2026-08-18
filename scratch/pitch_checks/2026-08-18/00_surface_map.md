# Surface map, 2026-08-18 (Tuesday, midterm year, August tdom 12)

Freshest bar 2026-08-17. No state warnings. Dial ma10-63d **84.8** (raw 63d 92.9,
raw 21d 66.3), which zeroes the book's dip-buy family (P/C fear OFF at the 42.9th
pctile) and puts the exposure leg at 0.0x. No fragility signals on. Pipeline: the
scan's last success is 2026-08-14, prices/dial/put-call all dated 2026-08-17.

Recon behind every number below: `01_live_state_recon.py` (+ `.txt`), computed on
each instrument's OWN series, never a panel column. Tape sort: `00_tape_sort.py`.

**Scoreboard read before selecting.** 4 graded ideas lifetime (event_fingerprint 2
at +0.622R, interaction_cell 1 at +0.146R, relative_value 1 at +0.099R). That is a
handful, not a split worth steering by. No axis is penalised or favoured today.

**Standing context that shapes the whole map.** The last three mornings
(08-13, 08-14, 08-17) were stand-downs and between them they closed most of the
obvious surface here: the JH anchor on rates/gold/FX/small caps, the August TLT
month-position seasonal, the IG-52w-low rung, SPY-high x TLT-low both directions,
the energy thrust into a 52w high, GDX's maximal 21d thrust, XRT into the big-box
cluster, the term-structure carry cell, and the vix_expiry/opex anchor pair. Those
are dismissals below with citations, not fresh looks.

---

## 1. Calendar events x asset classes

Six events sit in the [-5, +15] td window. Two are behind us, one is beyond the
10 td horizon cap. Live anchors reachable by a pitch: vix_expiry (+1), opex (+3),
jackson_hole (+8). Month-end (2026-08-31) is +8 sessions and is not in the events
file but is a real flow date, so it is enumerated as a seventh row.

| event | date | td | status |
|---|---|---|---|
| cpi | 2026-08-12 | -4 | behind |
| ppi | 2026-08-13 | -3 | behind |
| vix_expiry | 2026-08-19 | +1 | live |
| opex | 2026-08-21 | +3 | live |
| jackson_hole | 2026-08-28 | +8 | live |
| nfp | 2026-09-04 | +13 | beyond the 10 td horizon |
| (month-end) | 2026-08-31 | +8 | live, flow date |

### vix_expiry (+1) x 10 classes

Blanket dismissal with one exception. 2026-08-17 established that **vix_expiry and
opex are ONE anchor**, sharing 189 of 307 days, so any cell crossing both
double-counts a single calendar fact (v2_c8_expiry_pair.py). The lane is already
examined three ways and empty: VIX-expiry-week drift (2026-08-07 d4), pre-expiry
short-vol carry (2026-08-07 c12), and the post-expiry vol cells swept in the
2026-08-06 event sweep. The settle session itself is the worst day of its own week.

- us_large / us_small / rates / credit / gold_miners / other_metals / energy /
  dollar_fx / international: **not examined.** The anchor is dead on its own
  instrument (vol); there is no reason to expect it to carry cross-asset
  information it does not carry on the thing it settles.
- volatility: **dismissed on precedent**, above. Additionally the vehicle leg
  inverts once the 2018-02-28 SVXY leverage break is respected, and 92% of opex-4
  anchors already sit inside a live V4 sleeve window.

### opex (+3) x 10 classes

- us_large: **dismissed.** The opex window is now examined from the NFP close over
  10 td (2026-08-07 d2), from the Friday before over 5 td (2026-08-14 c6, anchor
  14th of 21 offsets, 2018+ -0.033%), and as VIX-expiry week. Empty in all three.
- us_small: **dismissed.** IWM at a 52w high into opex week was killed 2026-08-17
  (the opex gate is an INVERTER: -0.250% with it against +0.373% without). IWM is
  0.34% off its 52w high today, i.e. the same state.
- volatility: **dismissed**, and it is also the book's own trade. Event sleeve V4
  goes long SVXY at the opex MOC for 3 sessions, every month except September. A
  post-opex vol idea is a re-run of a live sleeve position.
- rates / credit / gold_miners / other_metals / energy / dollar_fx /
  international: **not examined.** Opex is an equity-and-vol calendar fact; the
  repo's only measured cross-asset opex claim (IWM/SPY) was concentration noise
  (top-2 episodes 103% of total). No mechanism proposes a metals or FX response to
  equity option expiry.
- flow_mechanics sub-lane: **structurally unfalsifiable here.** option_surface_history
  holds 1 row and option_positioning_history 90, all dated 2026-08-05. Any dealer
  gamma claim would be unverified prose. Stated as a data limit, per the honesty rule.

### jackson_hole (+8) x 10 classes

2026-08-13 examined this anchor on rates, gold and FX and killed all three; the
decisive finding was that the JH offset ladder is a **plateau, not a spike**, i.e.
the anchor is August month position wearing an event label (the unconditional Aug
6-16 window pays +1.025% at t=6.90 over 189 starts, against the JH anchor's +1.162%
over 24). 2026-08-11 killed the IWM leg.

- rates: **dismissed**, killed 2026-08-13 (b1_c4_jh.py).
- gold_miners / other_metals: **dismissed**, killed 2026-08-13 (10-11 at +0.577%,
  92% of it two episodes; midterm -1.213% at 1-4).
- dollar_fx: **dismissed**, killed 2026-08-13 (13-13, drop-best flips the sign).
- us_small: **dismissed**, killed 2026-08-11 (wrong-signed in midterms).
- us_large: **CHECK (C9).** The one class never examined on this anchor, and the
  class the symposium's headlines are actually about. Prior is strongly negative
  given the plateau finding, so the check leads with the offset ladder and the
  August month-position control rather than with the conditional mean. Cheap, and
  the map cannot dismiss a live event's largest class on a prior alone.
- volatility: **not examined.** Would be the fourth vehicle-translation attempt on
  a vol ETP in eight sessions; the SVXY/UVXY roll-drag kill (2026-08-13 correction)
  applies to any direction, and the anchor is already a plateau on four classes.
- credit / energy / international: **not examined.** No mechanism connects a Fed
  policy symposium to these more tightly than to rates and the dollar, both of
  which are dead on this anchor.

### nfp (+13), cpi (-4), ppi (-3)

- **nfp: out of reach.** 13 td exceeds the 10 td horizon cap, so no pitch can hold
  to it. This is also watchlist W1's structural blocker.
- **cpi / ppi: behind us.** Post-print windows from 08-12 and 08-13 have already
  expired at h<=3 and expire this week at h=5. The whole CPI/PPI lane was swept on
  2026-08-10 and 2026-08-12 across equities, curve, credit, vol, energy and FX.

### month-end 2026-08-31 (+8 sessions) x 10 classes

Not an events-file entry, and the only genuinely unexamined calendar object today.

- rates + us_large: **CHECK (C1).** Stock/bond divergence over the month so far is
  **+7.32pp** (SPY 21d +3.95%, TLT 21d -3.36%), the **89.2nd percentile** of that
  statistic since 2000. The 60-40 rebalance mechanism predicts month-end demand for
  the loser and supply of the winner, sized by the divergence. Note the trap this
  must clear: "turn-of-month" is registry-dead as a bare calendar cell, and the
  bare August TLT window was killed on 2026-08-17 as a bond-bull fossil. The check
  is worth spending precisely because the CONDITIONAL form (divergence-sized) has
  never been measured here and the unconditional forms are known dead, which makes
  gate attribution the whole test rather than an afterthought.
- credit: **not examined**, folded into the rates leg. LQD/HYG would be a
  lower-duration expression of the same flow and 2026-08-12 showed the PPI curve
  edge was proportional to duration and nothing else.
- gold_miners / other_metals / energy / dollar_fx / international / volatility /
  us_small: **not examined.** The rebalance mechanism is specific to the two legs
  of a 60-40 portfolio. Extending it to metals or FX would be a calendar cell with
  no flow story, which is the shape the registry's "famous calendar cells" entry
  already covers.

---

## 2. Tape extremes by class

Sorted across all 218 names (`00_tape_sort.txt`), not looked up by recall.

**us_large.** SPY -0.67% off its 52w high, 21d rank 82.5, 63d rank 44.0. QQQ -2.08%
off, 21d +4.97%. Nothing extreme at the index level; SPY's own 1d was -0.47%.
Verdict: no standalone index price-state candidate. The index appears as the
control leg or the short leg of C1/C4/C6/C9.

**us_small_breadth.** IWM -0.34% off its 52w high, 63d +9.79% (leading SPY's
+4.80%). Verdict: **dismissed**, this exact state into opex was killed 2026-08-17.

**rates.** The loudest cell on the board. TLT is **AT its 52w low (0.00%)**, IEF
0.73% off, LQD 0.35% off, ^TNX 21d rank 86.5. Bond vol moved: **^MOVE +8.70% in one
session, the 96.7th percentile of its daily moves since 2002**, though its LEVEL is
only the 43.2nd percentile of full history and the 64.7th of the trailing year.

- The 52w-low rung itself: **dismissed**, W6 below, and the rung was separately
  killed as a gate on 2026-08-17.
- The MOVE **spike**: **CHECK (C2).** Every prior MOVE entry in the registry is
  about MOVE at a FLOOR (2026-08-10 c3, 2026-08-14 c8); a one-day spike is a
  different object and is untested. The level-vs-rank trap is handled by quoting
  both numbers up front, which is what the 2026-08-14 SKEW entry demands.

**credit.** HYG 0.23% off its 52w high while LQD sits 0.35% off its 52w LOW.
Verdict: **dismissed**, this is watchlist W2 and the state is unchanged from the
2026-07-22 cluster (see W2). The 2026-08-10 kill also showed the LQD leg's residual
against IEF is +0.000pp, so there is no credit component at a pitch horizon.

**gold_miners.** GDX 21d rank **100.0** (+28.84%), NEM 21d rank 100.0 (+34.15%),
GLD 21d rank 86.1 (+10.06%), all three with 63d ranks in the 30s. The miners
outran the metal by **+17.06pp over 21 days, the 97.9th percentile** of that spread
since 2006 (ratio rank252 99.6).

- GDX outright thrust: **dismissed**, closed by the reference class on 2026-08-17
  (dispersion ratio 0.97, P(max >= GDX) = 0.582) and by the magnitude band.
- The miner-vs-metal **ratio** at an extreme: **CHECK (C5).** A different object
  from the outright: it is a spread with a structural driver (miner operating
  leverage on the gold price), the reference-class objection does not apply to a
  named pair, and it has never been measured in this repo.

**other_metals.** SLV 21d +17.31% but -43.59% off its 52w high and 63d -13.72%.
Verdict: **dismissed.** The silver thrust-from-a-drawdown cell was killed
2026-08-10 (the drawdown conditioner inverts; nudging the thrust 8%->10% flips h=5
to -4.229%), and the 2026-08-11 basket kill closes adding a second metals leg.

**energy.** XLE at a literal **52w high (0.00%)**, 5d rank 85.7, while USO's 63d
rank is 6.3 and its 63d return is -12.10%. USO +2.91% on the day (0.76 ATR).
Verdict: **dismissed on two independent 2026-08-17 kills** — the 5-day complex
thrust into a 52w high (magnitude form negative at every horizon 1-10) and
producers-against-the-barrel on a 63d divergence (sign flips at 18pp against a
live +18.85pp, era-reversed, bear-tape over-selection). W5 and W8 below cover the
crude-pop forms and both fail their triggers today.

**dollar_fx.** UUP 21d rank 17.9, DX-Y.NYB 18.3, both ~1.8-2.0% off 52w highs.
Verdict: **not extreme, no candidate.** The dollar is mid-range on every statistic
in the tape; the 2026-08-10 kill also showed a nested pullback subset there is a
threshold artifact. Nothing to check today.

**international.** EFA **0.22% off its 52w high**, EWJ 0.30% off with a 21d rank of
95.6, both beating SPY over 63d (+8.30% / +8.38% against +4.80%) while SPY is
0.67% off its own high. EEM 5d rank 81.7 but -5.46% off. FXI 5d rank 9.9 (W10).

- Verdict: **CHECK (C6).** The country family the registry closed (EWZ twice, FXI,
  SMH/QQQ) is *one market BREAKING inside an intact thrust*, i.e. the weakness
  direction. Sustained international LEADERSHIP into a 52w high is the opposite
  construction and untested. The check must lead with beta-neutralisation, because
  the standing kill for pairs here is that the trigger selects tape both legs share.

**volatility.** ^VIX +6.60% on the day to 15.19 while SPY fell only 0.47%; ^VIX 21d
rank 12.7 and -51.08% off its 52w high; ^SKEW 5d rank 84.9 at a level of 142.91;
^VIX3M/^VIX ratio elevated. The **VIX-up-hard-while-spot-barely-moves** state has
433 instances in 6698 sessions (6.46%).

- Term structure as an entry: **dismissed**, killed twice (2026-08-13, re-confirmed
  2026-08-17: offset -10 pays +5.433% against the true anchor's +1.672%, a lagging
  marker).
- SKEW: **dismissed.** Both poles are dead (2026-08-14) and the surviving spike
  cell is W7, whose three legs all fail today.
- The VIX/spot divergence: **CHECK (C4).** Untested, and distinct from every dead
  vol cell here because it conditions on the RELATIONSHIP between the two rather
  than on a level or a term-structure percentile. Gate attribution against a plain
  small-down-day and a plain VIX-pop day is the whole check.

**sectors and single names.** The 5-day cross-section is sharply bifurcated:
utilities sweep the top (CMS 95.2, DUK 94.4, DTE 92.1, EIX 92.1, AEP 87.7, PPL
87.3, XLU 88.1) while megacap tech and consumer sweep the bottom (ROST 1.6, CSCO
2.8, TJX 4.8, HON 5.6, AVGO 7.5, NKE 7.5, AMZN 8.3). Extension extremes: **MU sits
81.8% above its 200d, the 97.7th percentile of its own history**, +775% off its 52w
low; SMH +28.3% above its 200d (93.3rd pctile). Capitulation extremes: **NKE closed
AT a 52w low, 49.27% off its high and 25.56% below its 200d**; CSCO's 63d rank is
0.4. XRT 5d rank 9.1 inside an intact 63d trend (86.1).

- XRT into the retail cluster: **dismissed**, killed 2026-08-17, and the
  intact-trend gate is an inverter (the live state pays -0.443%).
- Utilities long, in any washout form: **dismissed**, dead in six expressions.
- Utilities STRONG while their rate driver goes the other way: **CHECK (C10).**
  This is not any of the six dead expressions, all of which were utilities washed
  out or paired against XLV/XLP. It is a divergence-resolution short, and the map
  should not dismiss the only untested direction of a class whose entire 5-day
  cross-section is at the top of the board.
- Parabolic extension: **CHECK (C7)**, in its cross-sectional form rather than as
  "fade MU". The 2026-08-13 reference-class lesson says a single-name claim must be
  priced against its peer group, so the candidate IS the peer group.
- Deep capitulation at a new 52w low: **CHECK (C8)**, same design rule, plus the
  alphabetical placebo the 2026-08-14 entry demands for any basket cut to <=4 legs.
- The bifurcation itself (defensives bid, megacap sold): **not examined as its own
  cell.** It is the Defensive Leadership fragility signal, which is OFF (50d spread
  +1pp), and the dial-as-direction kill (2026-08-07, re-confirmed 2026-08-13 on the
  rate of change) covers reading a fragility component as a direction.

---

## 3. Seasonal and cycle cells

- **Midterm year (year%4==2), August, tdom 12.** The seasonal board's live read is
  regime context, not a setup: 0 A+B-grade tickets, book win 56.4% vs 64.9%
  all-years in midterms, OVS 55.4% vs 67.6%. **Used as a conditioner on every
  candidate below, never as an idea.** Midterm mid-August as its own seasonal is
  registry-dead (N=6, carried by 2002, drop-two-best negative).
- **August month-position on duration:** killed 2026-08-17 as a bond-bull fossil
  (2018-2025 -0.013% at 4 of 8 years). This is exactly why C1 is designed as a
  divergence-conditional cell with a month control, not an August cell.
- **November duration month-position:** alive but parks to November (W12).
- **Turn-of-month:** registry-dead as a bare cell; C1 must beat it, not lean on it.
- No other live seasonal cell in the state file.

---

## 4. Watchlist verdicts (13 active, all with today's number)

| # | entry | verdict | today's number |
|---|---|---|---|
| W1 | TLT from the NFP close at the 52w rates floor | **PASS** | next NFP 2026-09-04 is +13 td, past the 10 td horizon cap, and midterm. Structurally unreachable until 2027-01. |
| W2 | LQD vs HYG at joint 52w extremes | **PASS** | state live (HYG -0.23% off high, LQD +0.35% off low) and still the same cluster begun 2026-07-22; count stays 4 declustered episodes, three of them 2018, against the >= 8 over >= 3 non-2018 years required. |
| W3 | SVXY overnight into CPI | **PASS** | next CPI 2026-09-11, +17 td. Re-measure is owed at the 2026-09-10 run. |
| W4 | GLD on a miner-led thrust GLD has not joined | **PASS** | needs GDX 5d rank >= 95 with GLD < 95; today GDX 5d rank **47.6**, GLD **46.0**. The 21d divergence is extreme but this entry is defined on the 5-day, and C5 is a different (ratio-reversion) construction, noted so the two are not confused. |
| W5 | XLE on a crude 1-day pop in the [5,6)% band | **PASS** | USO 1d **+2.91% = 0.76 ATR**, against [5,6)% and >= 1.50 ATR. |
| W6 | TLT with the IG complex pinned at 52w lows | **PASS** | price rung IS live (TLT 0.00%, IEF 0.73%, LQD 0.35%) and the freshness leg fails hard: **1 session** since the prior trigger, cluster **depth 6**, against >= 10 required. Pooled depth>1 entries pay -0.629% at a 37.3% hit (N=59). The 2026-08-17 kill of the rung as a gate raised this bar further. |
| W7 | SPY on a skew spike alone | **PASS** | all three legs fail: ^SKEW 5d rank **84.9** (needs >= 95), SPY **-0.67%** off its high (needs < -1.0%), and the year is midterm (needs non-midterm). |
| W8 | Fade a crude thrust out of a deep base | **PASS** | USO 5d rank **71.8** (needs >= 90); the 63d rank 6.3 leg does clear. Post-2020 episode count still 4 against 8. |
| W9 | IHI at a 21d rank of 100 | **PASS** | IHI 21d rank **99.2** (needs 100), and the reference-class gate (Cochran Q p 0.544) cannot move in a session. |
| W10 | FXI's 5-day break inside an intact thrust | **PASS** | FXI 5d rank **9.9** clears and EEM 5d +3.30% clears, but FXI's 21d rank is **73.0** against the >= 80 that defines the intact thrust. Family separately closed 2026-08-17 on EWZ. |
| W11 | Industry-wide breadth washout with the trend BROKEN | **PASS** | no coherent industry universe is at >= 70% of names at a 5d rank <= 20 with a median 63d rank below 70. The washed-out cluster today is megacap tech and retail, whose 63d ranks are mid-to-high (XRT 86.1), i.e. the INTACT half this entry parks against. The untested long-side selection rule remains outstanding. |
| W12 | TLT on the NOVEMBER month-position effect | **PASS** | parks to a date: trading days 4-12 of November 2026, roughly 2026-11-05 to 11-17. Owes a rate-regime check and a re-run through October first. |
| W13 | Short SPY at a 52w high while the long end sits at a 52w low | **PASS** | the SPY rung fails today: **-0.67%** off its high against the <= 0.5% required (TLT's leg does clear at 0.00%). The de-concentrated 5x-cost trigger is unchanged in any case. |

No entry fires. Nothing expires today; the file is rewritten after publish.

---

## 5. Selected candidates (11, six axes, eight asset classes)

Coverage check: event-anchored candidates present (C1 month-end flow, C9 Jackson
Hole); price-state anchored present (C2, C3, C4, C5, C6, C7, C8, C10, C11). Asset
classes touched: rates, us_large, volatility, gold_miners, international, sectors,
single-name equity, credit (as C1's dismissed sibling). Axes: flow_mechanics,
interaction_cell, relative_value, inversion, historical_analogue, event_fingerprint.

| id | candidate | axis | class |
|---|---|---|---|
| C1 | Month-end rebalance flow sized by the stock/bond divergence (+7.32pp, 89.2nd pctile), long TLT into 2026-08-31 | flow_mechanics | rates x us_large |
| C2 | ^MOVE one-day spike (+8.70%, 96.7th pctile of daily moves) as a rates-direction signal | interaction_cell | rates |
| C3 | The same MOVE spike as a cross-asset equity signal (rate vol leading equity vol) | event_fingerprint | us_large |
| C4 | VIX +6.60% on a day SPY fell 0.47%: fear without damage, 433 instances | interaction_cell | volatility x us_large |
| C5 | GDX/GLD 21d spread at its 97.9th percentile, ratio reversion (long GLD, short GDX) | relative_value | gold_miners |
| C6 | International leadership into a 52w high, EFA/EWJ against SPY, beta-neutral | relative_value | international |
| C7 | Parabolic extension above the 200d, cross-sectional form (MU 97.7th pctile) | inversion | sectors / single-name |
| C8 | A new 52w low >= 20% below the 200d, cross-sectional form (NKE live) | historical_analogue | single-name equity |
| C9 | Jackson Hole x US large, the one unexamined class on a live anchor | event_fingerprint | us_large |
| C10 | Utilities strong while duration prints a 52w low: divergence resolution, short XLU | inversion | sectors x rates |
| C11 | The 63d-rank laggard cross-section (CSCO 0.4, USO 6.3, SMH 8.3) as a reversion set | price-state | mixed |

C11 is included as a cheap sweep of the deepest 63d ranks so the map covers the
laggard lane rather than dismissing it by analogy with the dead SMH/QQQ pair;
prior is strongly negative and it will be measured, not assumed.
