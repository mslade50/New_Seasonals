# Surface map — 2026-09-02 (Wednesday, trading day 2 of September, midterm year)

Recon backing this map: `00_recon.py` (all 33 watchlist triggers read off
master_prices directly, plus the live geometry of every class) and
`00b_anchor_map.py` (count-first on every cell today's tape suggests that no
watchlist entry covers). Raw output in `_out_recon.txt` / `_out_anchor.txt`.

**One correction applied before selection.** The first draft of `00_recon.py`
called `pct_rank(ret, n=252)` on an already-differenced return series, which
double-differences and reported XLK's 5-day rank as 100.0 against a true 3.2,
XLI's as 93.7 against a true 4.0 and ^TNX's 21-day rank as 15.1 against a true
68.3. `pct_rank` takes the PRICE series and does its own `pct_change`. Fixed
and rerun; every number below is from the corrected run and agrees with
`pitch_tape.json` to the decimal.

## Standing state

- Prices fresh through **2026-09-01**; pipeline 7/7 green, receipts all
  collected. One stale tape ticker (LEG), used nowhere below.
- **Fragility dial ma10-63d 87.5** (raw 63d 85.6, raw 21d 62.8), against 51.6
  twenty-one sessions ago. Exposure leg 0.0x, trend sleeve in CASH. Same
  standing consequence as yesterday: most of the parked inventory has a
  historical trigger-day dial maximum in the 66-75 range, so **the watchlist is
  broadly out of sample on this tape**. That argues for lower grades, not for a
  dial trade — the dial is closed as a directional signal in the registry.
- P/C fear OFF (52nd pctile, data 2026-09-01). One fragility signal on:
  **VIX Range Compression** (21-day range at the 1st percentile, VIX 16.3).
- Book has **0 staged signals** (the state excludes live positions by design).
- Repetition: one fingerprint inside the 10 td window, `749b2073856902b3`
  from 2026-08-27, "Buy the first flush inside a parabolic gold-miner run".
  Any LONG gold/miner flush idea today is that fingerprint and needs
  `changed_since`; the short side is a different fingerprint but inherits the
  scrutiny.
- Scoreboard: 5 pitched, **4 graded**. Too few to read the per-axis split as
  a signal, so the axis feedback loop is noted and not acted on. For the
  record: event_fingerprint 2 graded at +0.622R, interaction_cell 1 at
  +0.146R, relative_value 1 at +0.099R, all opus.

## 1. Every live calendar event x every asset class

Eight events in the [-5,+15] td window. **Nothing is new inside the 10 td cap
today** — the FOMC and the VIX expiry both crossed in yesterday, at which
point the whole pre-FOMC window was swept on fifteen asset classes and closed.

| event | date | td | status |
|---|---|---|---|
| jackson_hole | 2026-08-28 | -3 | CLOSED. Eight classes pre-speech, ten post (08-27/28). One cell parked to a non-midterm year (watchlist 27). |
| nfp | 2026-09-04 | **+2** | Plain ladder CLOSED (08-26) on four vehicles; September prints -0.038%, September-in-a-midterm -0.676% on 3-3. The Labor-Day-eve interaction was dismissed on count 09-01 (14 of 17 September NFPs are holiday eves, both parents dead). **What is NOT closed is the crossing with today's vol state** — see C9 — and the NFP sits inside C1's hold, which C1's checker must price. |
| ppi | 2026-09-10 | +5 | CLOSED 08-27 as a containment object on equities, rates and gold. Live role today is as an EXCLUSION: it sits outside a 3-session hold, which is the leg watchlist 4 needs. |
| cpi | 2026-09-11 | +6 | Same closure. Live role is the conditioner in C8, where the hold is long enough to contain it. |
| fomc_decision | 2026-09-16 | +9 | **Swept and closed yesterday**, 15 asset classes, Cochran Q 9.26 on 14 df at p 0.8138, I-squared 0.0%, permutation max-of-15 P 0.2447. The midterm inversion is sampling noise; the one exception (energy inverting the other way) was killed on a placebo ladder and 157%-of-total concentration in two 2026 episodes. Walking down the offset ladder from -10 td to -9 td to re-open it is the exact placebo-ladder trap that killed the NVDA cell. **Not re-examined.** |
| vix_expiry | 2026-09-16 | +9 | CLOSED yesterday: 41 of 42 coincident decisions fall in Mar/Jun/Sep/Dec at trading-day-of-month 11-16, a mid-month confound, and the 14:00 placebo falsified the settle story. |
| opex | 2026-09-18 | +11 | Beyond the 10 td horizon cap. September post-opex is the event sleeve's T3 and the registry records September inverting the post-opex vol crush. Out of range. |
| quad_witching | 2026-09-18 | +11 | Same. |

### The event x class cross, stated as verdicts

The FOMC row above is the ten-class cross, run yesterday and closed, so
repeating the grid today would buy nothing. The two events that remain live
and unclosed at the CLASS level are NFP (+2) and CPI (+6), crossed here:

| class | NFP +2 td | CPI +6 td |
|---|---|---|
| us_large | ladder closed 08-26 | containment closed 08-27 |
| us_small | ladder closed 08-26 | closed, same |
| rates | midterm-parked to 2027-01 (watchlist 1) | closed 08-27 |
| credit | not examined: no credit cell survived the 08-26/08-31 sweeps and today's HYG/LQD geometry is watchlist 26's, at an episode count of one | not examined, same |
| gold / miners | not examined: repetition-blocked (fingerprint above) and GLD is -20.0% off its high | not examined, same |
| other metals | the long side closed yesterday; the short side is C3, anchored on the price break rather than the print | not examined |
| **energy** | **CHECK as a conditioner inside C1**, not as an anchor: the armed watchlist-4 hold contains the print and the entry's event clause names only CPI/PPI | outside C1's 3-session hold |
| **commodities (broad)** | not examined separately from energy | **CHECK — C8**, 33 of 105 DBC-at-a-high episodes carry a CPI in a 6-session hold |
| dollar / fx | washout cells midterm-parked; DX 21d rank 43.7 is mid-range, no state to anchor | same |
| international | no live event cell; EEM/EFA/FXI carry no event conditioner this repo has opened | same |
| **volatility** | **CHECK — C9**, the crossing of the print with today's compressed-range state, which is the one thing about this NFP that is not already closed | the SVXY overnight is watchlist 3, 5 sessions early |

## 2. Tape extremes by class

Full 218-name sort in `scratch/_tape_sort_0902.py`, verified against
master_prices in `00_recon.py`.

- **us_large**: SPY -2.07% off its 52w high, r5 27.4, r63 12.7, z10 -0.39.
  QQQ r63 3.2, ^NDX 3.2. Nothing extreme outright. The near-high-with-a-weak-
  63d form was killed 08-28. SPY's live role today is as the SECOND leg of C7
  (it fell only 0.69% on a day VIX rose 9.52%).
- **us_small**: IWM z10 -1.18, r5 6.7, -4.76% off its high. The IWM/SPY pair
  and the HYG-substitution form were both closed 08-31. Nothing new.
- **rates**: **^TNX closed AT its trailing-252 maximum again** (0.00% off),
  r5 96.8; 21-session change +11.0 bp, 252-session +62.0 bp. TLT +1.02% above
  its 252-low, IEF +0.28%, **LQD AT its 252-low (0.00%)**. This is the loudest
  level extreme on the tape and it is also the most thoroughly closed lane in
  the repo: four rates kills yesterday alone (the flattener on multiplicity
  plus the wrong magnitude half, the pre-FOMC duration cell on calendar-leg
  gate attribution, the no-thrust yield maximum on 24 indistinguishable
  comparisons, the bond-vol/equity-vol divergence on a false premise). The
  three live arms all fail by a stated number: watchlist 5 needs TLT within
  0.5% of its low against 1.02%, watchlist 18 needs a 252-session thrust of
  +78 bp against +62.0, watchlist 26 needs HYG within 0.25% of its high
  against -0.80%. **Dismissed: swept, closed, and no arm met.**
- **credit**: HYG -0.80% off its 252d high while LQD sits AT its 252d low.
  The joint state has **6 declustered episodes in two years** (2018 x1, 2026
  x5) and watchlist 26 already measured its residual: LQD = 0.485*IEF - 0.001pp
  at h=10, i.e. duration wearing a credit label. **Dismissed on the standing
  closure, not on count alone.**
- **gold**: GLD -2.86% on the day, -7.32% over five, r5 4.4, and -19.99% below
  its 52-week high after a +6.74% 21-day run. **Repetition-blocked on the long
  side.**
- **other metals**: **the whole complex broke together on 2026-09-01** — SLV
  -3.68%, GLD -2.86%, GDX -3.90%, NEM -2.72%, CEF -3.56% — while GDX's 21-day
  rank is still 94.4 and NEM's 95.2. That conjunction after a parabolic miner
  run has **12 declustered episodes across 8 years** and is LIVE for the first
  time since 2026-08-28. **CHECK — C3 (short continuation) and C4 (the
  miner/metal pair).** Watchlist 29's own arm is NOT met: its depth bucket
  needs an SLV break of -4% or worse against a live -3.68%, and its lag-profile
  debt is unpaid regardless.
- **energy**: XLE, XOP, COP, CVX, VLO and DBC all closed AT a 52-week high;
  **USO +5.46% on the day**, +11.77% over five. That one-day pop lands in the
  [5%,6%) band at **1.74 ATR** with no CPI or PPI inside a three-session hold,
  which is **every leg of watchlist 4's stated arm**. The only fully armed
  entry on the list. **CHECK — C1, C2.** The narrow-thrust entry (watchlist 19)
  does NOT fire: the count of the 11-name complex at z10 >= 2.0 is **0**, top
  reading SLB +0.73. The at-a-high-on-a-down-index cell (watchlist 32) IS live
  but its standing blocker is explicitly not waivable and it was killed
  yesterday. Breadth is not extreme either: 5 of 10 names within 0.5% of a
  252d high is the median rung of a 144-episode population, not a tail.
- **commodities (broad)**: **DBC closed AT a fresh 252-day high**, r5 93.3,
  z10 +1.16, +10.56% over 21 sessions. 105 declustered episodes since 2007,
  33 of them with a CPI inside a 6-session hold. **CHECK — C8.**
- **dollar / fx**: UUP r5 84.5 but -1.36% off its high and r21 43.7; DX-Y.NYB
  r21 43.7, +0.24% on the day. Mid-range on every axis. The bare washout is
  midterm-parked (watchlist 23), the confirming-rate-rise forms are closed
  (08-19 both ways) and UUP is separately cost-dead as a vehicle. **Nothing
  live; not examined.**
- **international**: **EWZ carries the tape's highest z10 at +2.03** and is
  +1.50% on a session SPY fell, inside an EM complex whose 63-day rank is 0.4
  (EEM). FXI r5 82.1 fails watchlist 10's <=20 leg; EFA r5 9.1 is a mild
  laggard with no cell attached. The EWZ-against-EEM divergence is the one
  international object with a live extreme. **CHECK — C10.**
- **volatility**: **^VIX +9.52% to 16.34 while its own 21-day range sits at
  the 8.3rd percentile of the year** and SPY fell only 0.69%. The compressed-
  range-then-pop object has **40 declustered episodes over 20 years** at the
  bottom-15% rung (the bottom-5% rung has 11 and is NOT live at 8.3).
  ^VIX3M +4.56% to 18.33; ^MOVE +3.40%, level at the 79th percentile — so
  today is equity vol catching UP to bond vol, the mirror of the divergence
  killed yesterday on a false premise, and no cleaner for it. **CHECK — C7,
  C9.** SVXY's plain pre-FOMC leg is registry-closed; the CPI overnight is
  watchlist 3 and is 5 sessions early.
- **sectors**: **XLI at r5 4.0, r21 5.6 AND r63 6.3 simultaneously**, z10
  -2.29, -7.39% off its high but +1.63% above its 200d. The triple floor with
  the index within 3% of its high has **11 declustered episodes over 7 years**
  and 85 pooled across the nine SPDRs, so a reference class exists — which is
  mandatory here, because the homogeneous-family shape is exactly what killed
  watchlist 21 (XLI, Cochran Q p 0.789) and watchlist 25 (SMH, family-wise p
  0.8805). **CHECK — C5.** Separately the **defense complex washed out
  together**: ITA z10 -2.55, RTX -1.97, GD -1.88, LMT -1.70, NOC -1.65, four
  of five at -1.5 or worse with the index near its high, 8 episodes over 5
  years. **CHECK — C6**, with the concentration flagged up front (3 of the 8
  are 2026). XLRE r5 4.8 within 4.28% of its high fires watchlist 21's literal
  rung but XLRE is outside the nine-SPDR family that entry is written on, and
  the dial blocker (episode max 68.6 against 87.5) stands either way.
  XLU r21 10.3 with two idiosyncratic blowups inside it (EIX -21.4%, PCG
  -23.4% over five sessions, both bouncing hard today) — the two-name utility
  shock was closed 08-31 and the sector reading is that shock, not a sector
  state. Dismissed.
- **growth vs defensive**: XLK r63 3.2, SMH 0.4, QQQ 3.2 against XLV 98.0,
  IBB 99.6 — an XLV-minus-XLK 63-day rank spread of **+94.8 points, the
  98.8th percentile of its own year**. Count-first kills it before selection:
  the spread has been >= 90 on **exactly one day in history, today**.
  **Dismissed on count**, and widening to reach a sample is the move the
  2026-08-07 lesson forbids.
- **single names**: CRM +38.81% over 21 days, TJX at its 52-week low with the
  tape's worst z10 (-2.62), ORCL -56.6% off its high, MU +685% over 252 days.
  Single-name cells are closed as a class on a 205-name reference sweep at
  family-wise p 1.0000 (08-27) and the largest-winner-at-the-turn form was
  closed 08-31. **Dismissed as a class.**
- **natgas**: UNG -37.40% off its high, r5 72.6, z10 +0.88. Closed yesterday:
  negative in absolute terms at this drawdown depth inside a -0.887%/10td
  structural bleed. Not re-examined.

## 3. Live seasonal and cycle cells

- **Midterm year (year%4==2)** conditions everything above, and it is the
  reason four of the five seasonal-board candidates are "de-risk" context
  lines (book win 56.4% against 64.9% all-years, OVS 55.4% against 67.6%,
  LT Trend ST OS 53.7% against 67.6%, Indices Oversold Bounce 59.0% against
  64.5%). The board's fifth candidate is a P/C complacency read stamped
  2026-08-05 and contradicted by today's live 52nd-percentile reading.
- The board carries **0 A+B-grade setups**, so there is no seasonal ticket to
  cross with anything.
- **Month position**: trading day 2 of September, the month the whole cycle
  literature treats as the worst. Crossed with the midterm year it becomes
  **C11**, checked as a cheap count rather than assumed.

## 4. Watchlist — every active entry gets a verdict

33 active, 0 expired. Full readings in `_out_recon.txt`. **One entry is
fully armed.**

| # | entry | verdict | today's number |
|---|---|---|---|
| 0 | TLT from the NFP close, long end at a 52w floor | PASS | midterm; parks to 2027-01 |
| 1 | LQD against HYG at joint 52w extremes | PASS | 4 declustered episodes against the 8 required; state live, still uncountable |
| 2 | SVXY overnight into CPI | PASS | CPI is +6 td; the entry is 5 sessions away |
| 3 | GLD on a miner-led thrust | PASS | GDX r5 18.3 against >=95 — the miners BROKE, they did not thrust; GLD -19.99% off its high against the added within-10% leg |
| **4** | **XLE on a crude [5,6)% thrust** | **ARMED** | **USO +5.46% (band), 1.74 ATR (>=1.50), PPI +5 td and CPI +6 td both outside a 3-session hold** |
| 5 | TLT with the IG complex at 52w lows | PASS | TLT +1.02% above its low against the <=0.5% rung (IEF +0.28, LQD +0.00 both pass) |
| 6 | SPY on a skew spike alone | PASS | ^SKEW r5 68.7 against >=95; midterm block stands |
| 7 | Fade a crude thrust out of a deep base | PASS | USO r63 53.2 against the <=20 leg |
| 8 | IHI at a 21d rank of 100 | PASS | IHI r21 44.8 |
| 9 | FXI break inside an intact thrust | PASS | FXI r5 82.1 against <=20 |
| 10 | TLT November month-position | PASS | parks to trading days 4-12 of November |
| 11 | Short SPY at a 52w high with TLT at a low | PASS | SPY -2.07% off its high against <=0.5% |
| 12 | SPY on a vol pop inside a calm tape | PASS | pop leg LIVE (+9.52%) and the SPY leg LIVE (-0.69%), but the calm-tape leg fails hard: VIX's LEVEL sits at the 95.2nd percentile of its trailing 21 sessions and its 21-day return rank is 56.7, against a <=25 rung. C7 is the DIFFERENT object — range compression, not a calm level |
| 13 | GLD on an unconfirmed rate rise | PASS | DX r21 43.7 against <=15; 21-session yield change +11.0 bp against +20 |
| 14 | XLK against XLV after a rotation gap | PASS | one-day gap +2.20pp against >=+3.0pp |
| 15 | Short the dollar on an unconfirmed rate rise | PASS | ^TNX r21 68.3 clears >=65, but DX r21 43.7 fails <=20 |
| 16 | Short TLT after a big up day at the low | PASS | TLT 1d -0.41% against >=+1.5% |
| 17 | KRE against XLF on a breadth washout | PASS | arm is an ex-crisis cost threshold no new episode moves |
| 18 | The duration-neutral flattener | PASS | 252-session yield change +62.0 bp against the +78 bp arm, and ^TNX is AT its 252d max so the proximity leg is not the binding one |
| 19 | The NARROW energy thrust cluster | PASS | count of 11-name complex at z10>=2.0 is **0** against the [2,3] arm |
| 20 | Cross-sectional new-high breadth | PASS | SPY -2.07% off its high against >2.0%; raw-21d fragility 62.8 against <=50 |
| 21 | Sector washout into a 52w high, as a family | PASS | XLI r5 4.0 clears the washout leg but is -7.39% off its high against within-5%; XLRE clears both and is outside the nine-SPDR family; dial 87.5 against an episode max of 68.6 |
| 22 | Utilities washout with the long end hit alongside | PASS | XLU r21 10.3 against <=5, and TLT r21 57.1 against <25 — rates leg wrong-signed a fifth session |
| 23 | The bare dollar washout | PASS | parks to a non-midterm year; DX r21 43.7 in any case |
| 24 | HY at a fresh high while the index has not | PASS | SPY -2.07% off its high against >=2.0%; dial 87.5 against <50 |
| 25 | SMH at a 63d rank floor in a top-decile year | PASS | state live (r63 0.4) but r5 23.8 against the <15 arm, and the 23-ETF Cochran Q is unmoved |
| 26 | IG at lows while HY prints a high | PASS | HYG -0.80% off its high against the <=0.25% rung |
| 27 | IEF out of the Jackson Hole close | PASS | midterm; parks to 2027-09 |
| 28 | The pooled laggard still falling | PASS | **no** holder of r21>=90 AND r63<=10 in the 29-name pool |
| 29 | Short silver after a complex break | PASS on the arm | the three-name break is LIVE for the first time since 08-28, but the depth bucket needs SLV <= -4% against -3.68%, and the lag-profile debt is unpaid. C3 is the fresh-anchor form and inherits that debt as a mandatory round-2 item |
| 30 | Duration at a yield high with bond vol mid-range | PASS | ^MOVE trailing-252 LEVEL percentile 79.0 against the [40,50) band |
| 31 | The December month-end overnight | PASS | parks to a date |
| 32 | Energy at a fresh high on a down-index session | PASS | state IS live (XLE at its 252d max, SPY -0.69%) but the standing blocker is explicitly not a number and may not be waived, and the cell was killed yesterday |

## 5. Selected candidates

Eleven, from the map rather than from recall. Six axes, seven asset classes.
At least one event-anchored (C8, C9) and at least one price-state-anchored
(C1, C3, C5, C7) as required.

| # | candidate | axis | class | population |
|---|---|---|---|---|
| C1 | Long XLE on the crude one-day thrust in the [5,6)% band at 1.74 ATR, no CPI/PPI in the hold | interaction_cell | energy | watchlist 4's parked cell, +1.109% excess at 73.7% and sign p 0.0025 as parked; must be re-derived today and the NFP-in-hold conditioner priced |
| C2 | The same thrust taken in crude itself rather than the producers | instrument_translation | energy | the parked entry's own caveat says XLE is 0.479-beta crude with a +0.291% residual at a 49.3% hit |
| C3 | Short silver on the three-name metals break out of a parabolic miner run | inversion | metals | 12 declustered episodes, 8 years; inherits watchlist 29's lag-profile debt |
| C4 | Long GDX against short GLD on the same break (miners overshoot the metal) | relative_value | metals | same 12 episodes, pair form never measured |
| C5 | Long XLI at a triple 5/21/63-day rank floor while the index is near its high | interaction_cell | sectors | 11 episodes, 7 years; 85 pooled across nine SPDRs — reference class MANDATORY |
| C6 | Long the defense complex on a coordinated five-name z10 washout | interaction_cell | sectors | 8 episodes, 5 years, 3 of them 2026 |
| C7 | Long SPY on a violent VIX pop out of a compressed 21-day range, index barely down | flow_mechanics | volatility x us_large | 40 episodes, 20 years at the bottom-15% rung |
| C8 | Broad commodities at a fresh 252-day high with a CPI print inside the hold | event_fingerprint | commodities | 105 episodes, 33 carrying a CPI |
| C9 | The volatility complex into the payrolls print out of a dead range | event_fingerprint | volatility x event | the one NFP crossing not already closed |
| C10 | Long EWZ at the tape's z10 extreme inside an EM complex at a 63-day rank floor | relative_value | international | z10 +2.03 against EEM r63 0.4 |
| C11 | The index over the first week of September in a midterm year | interaction_cell | us_large | cheap count; assumed by everyone, measured by nobody here |

Dismissed with reasons above and NOT sent to a checker: the whole rates lane
(swept, four kills yesterday, three arms unmet by a stated number), credit
(standing residual closure), the growth/defensive spread (one day in history),
gold long (repetition-blocked), single names (family-wise p 1.0000), natgas
(closed yesterday), the dollar (no extreme), utilities (a two-name shock),
energy breadth (median rung), and every FOMC/VIX-expiry/opex form (closed
yesterday or out of range).
