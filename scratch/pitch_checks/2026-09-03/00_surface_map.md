# Surface map, 2026-09-03 (Thursday, midterm year, September)

State: `data/pitch_state.json` + `data/pitch_tape.json`, freshest bar **2026-09-02**
(prior session, clean). One tape warning: LEG stale, irrelevant to everything below.
Recon numbers: `00_recon.py`, `00b_watchlist_recon.py`.

Regime, quoted once so every cell below inherits it:
SPY 765.16, **-1.64% off its 52-week high**, +7.9% above its 200d, 105 sessions
since a 5% pullback. Fragility dial **ma10(63d) 87.9** (raw 63d 85.1, raw 21d 61.4),
up from 52.9 twenty-one sessions ago; exposure leg OFF; trend sleeve in CASH.
Signals on: **VIX Range Compression** (21d range 1st percentile) and Low Absorption
Ratio (2nd percentile). P/C fear OFF at the 54th percentile. VIX 15.20, VIX3M 17.73,
contango +16.6%. **^TNX closed AT its trailing-252 maximum (4.796)** and **^MOVE's
level sits at the 83.9th percentile with a 5-day return rank of 93.3**.

The one-line reading of the tape: equity volatility is as dead as it has been in a
year while bond volatility is bid and the ten-year is at a one-year yield high, with
a payrolls print tomorrow morning. Everything in section 1 is organised around that.

---

## 1. Every live calendar event x every asset class

Eight events sit in the [-5, +15] td window. Cells are numbered E1..E8 by event.
A dismissal here always carries its reason; nothing is silently absent.

| event | date | td | status |
|---|---|---|---|
| jackson_hole | 2026-08-28 | -4 | passed |
| **nfp** | **2026-09-04** | **+1** | **tomorrow, the dominant live anchor** |
| ppi | 2026-09-10 | +4 | |
| cpi | 2026-09-11 | +5 | |
| fomc_decision | 2026-09-16 | +8 | |
| vix_expiry | 2026-09-16 | +8 | |
| opex | 2026-09-18 | +10 | |
| quad_witching | 2026-09-18 | +10 | |

### E2. NFP, 2026-09-04 (+1 td) — the only event whose entry session is today

Today's close is the **last session before the print**, i.e. the k=-1 entry off a
k=-2 anchor dated 2026-09-02. `anchor_positions` cannot see 2026-09-04 because the
price index ends 2026-09-02, so the live instance is identified by hand and verified
in `00_recon.py`: the last three constructed k=-2 anchors are 2026-06-03, 2026-06-30
and 2026-08-05, exactly two sessions before the June, July and August prints.

| class | verdict |
|---|---|
| volatility | **CHECK — C1.** Watchlist 33 arms on a DATE and the date is today. Gate: VIX 21d range / 21d mean, trailing-252 percentile **3.57** (abs-range form 1.98) against a <= 15.0 rung. Three debts attached, all unwaivable. |
| rates | **CHECK — C3.** The pre-print anchor with the ten-year AT a 252-day yield high and IEF's 5-day rank at 6.0. Distinct anchor from watchlist 0, which is the POST-print long and is midterm-dead. |
| rates x volatility | **CHECK — C2.** ^MOVE level 83.9 / 5d rank 93.3 while the VIX 21d range is at the 4th percentile. Two vol markets disagreeing into a labour print; whichever is wrong pays. |
| us_large | **CHECK — C12.** The direction leg of the same cross, kept separate from C1 because the registry (2026-08-07) already swept post-NFP equity DIRECTION and found it empty; the PRE-print session out of a dead range was not what was swept. |
| gold and miners | **CHECK — C6.** GLD -4.40% over five sessions (r5 9.1) into the print while GDX/NEM hold 21-day ranks of 94.4. A metal washout with the miners still bid, crossed with a labour number that moves real rates. |
| energy | **CHECK — C7.** XLE, XOP, DBC, COP, CVX and VLO all closed AT 52-week highs. Registry 2026-09-02 killed the crude-thrust band four ways and recorded that a payrolls print inside the hold is NOT the CPI/PPI containment effect (+1.973% n=7 vs +1.088%, Welch t +0.64), so this is examined as the at-a-high state rather than the thrust. |
| credit | **CHECK — C8.** LQD closed **+0.12% off its 252-day low** while HYG is only -0.79% off its 252-day HIGH. The joint state is watchlist 1 and 26 territory (both fail their rungs), but neither was ever crossed with an event anchor. |
| international | **CHECK — C9.** EWZ +4.16% on the day at a 5-day rank of 96.0. z10 is +2.23 on the tape's definition and **+1.96 on `pitch_lab.zscore`** — both printed per the 2026-09-02 registry entry, which was about EWZ specifically. |
| US small and breadth | Not examined. IWM's r5 is 17.9 and it is -3.63% off its high, i.e. nothing distinguishes it from SPY today (5-day returns -1.65% against -0.12%), and the small-cap-into-payrolls cell has no live conditioner to hang on. Its September opex form belongs to the event sleeve's T3, not here. |
| dollar and FX | Not examined as a standalone. DX 21-day rank 41.7 and UUP 42.9 are mid-range in every lookback, so the two live dollar entries (watchlist 15, 23) both fail on the dollar leg and there is no dollar state to anchor a payrolls cell on. The dollar appears inside C6 as the transmission channel, not as its own cell. |
| other metals | Folded into C6. SLV bounced +1.99% today out of the three-name complex break, and watchlist 29's short-silver arm is explicitly unmet (needs a break of -4.00% or worse; 2026-08-28's was -3.68%). No separate NFP x silver cell: the mechanism is the same real-rate channel as gold and running both would be one position twice. |

### E3/E4. PPI (+4 td) and CPI (+5 td)

| class | verdict |
|---|---|
| volatility | Watchlist 2 (long SVXY overnight, print eve to print open) parks to a DATE and that date is 2026-09-10, five sessions away. PASS today. **But the PPI/CPI pair matters to C1 as a mechanism probe**: in the 2026-09-02 reference-class table, PPI is the one anchor that INVERTS (SVXY -1.023%) while NFP, CPI and FOMC all pay, and PPI is structurally the print that is immediately followed by another print. C1 owes that test. |
| all others | Not examined at a 4-to-5 session distance. Every price-state cell below would have its entry today and its hold ending before the print, so the print is neither an anchor nor a tail risk for a 1-to-2 day idea. Recorded so it is visible: the only 10-day idea in the candidate set is C4, and its window does contain both prints. |

### E5/E6. FOMC decision and VIX expiry (+8 td)

| class | verdict |
|---|---|
| us_large | **Dismissed on ownership, not on evidence.** The event sleeve's T2 FOMC_MIDTERM_SHORT stages MOC four sessions before the decision, i.e. around 2026-09-10, in exactly this cell in exactly this cycle year. Pitching a pre-FOMC index trade would be a second copy of a live sleeve position. |
| volatility | Pre-FOMC vol is not fireable today (the fragility page's Pre-FOMC Rally signal reads "14d away, outside window") and the post-FOMC vol cell is in the registry as swept and empty (2026-08-06 event sweep). |
| all others | Not examined at eight sessions. Same reason as E3/E4. |

### E7/E8. Opex and quad witching (+10 td)

| class | verdict |
|---|---|
| us_small | **Dismissed on ownership.** Event sleeve T3 SEP_POSTQUAD_SHORT is the September opex-to-month-end short IWM and it is live this month. |
| volatility | Event sleeve V4 POSTOPEX_VOL explicitly EXCLUDES September, and the registry records that opex ex-September is the only post-event vol cell that survived the 2026-08-06 sweep. Nothing to add. |
| all others | Not examined at ten sessions, same reason. |

### E1. Jackson Hole (-4 td, passed)

Watchlist 27 (long IEF one session out of the close) is non-midterm-only and 2026 is
a midterm year; the conference is also four sessions gone. PASS, parks to 2027-09.
No other class examined: the anchor is behind us and its whole 210-cell grid already
fails a family-wise permutation at P 0.065.

---

## 2. Tape extremes by class (218 names sorted; `00_recon.py`, `_tape_sort_0903.py`)

| class | what is extreme | verdict |
|---|---|---|
| **volatility** | VIX 15.20, -6.98% on the day, **21d range at the 1st percentile**; ^SKEW level 144.12 at a **21-day rank of 99.6**; SVXY closed exactly AT its own-series 252-day max; ^MOVE 5d +14.79%, level pctile 83.9; UVXY -73.9% off its high | C1, C2, **C10** (skew bid while ATM vol is dead), C11 (which vehicle) |
| **rates** | ^TNX AT its 252-day max, r5 92.5; IEF r5 **6.0**, +0.36% above its 252d low; TLT +1.12% above its low, -8.41% off its high | C3, C2 |
| **credit** | LQD **+0.12% above its 252-day low**, r63 9.9; HYG -0.79% off its 252-day HIGH | C8 |
| **energy** | XLE / XOP / DBC / COP / CVX / VLO all AT 52-week highs; USO +21.9% over 21 sessions, r5 91.3; OIH r5 95.2 | C7 |
| **sectors** | **XLI r5 0.8 / r21 2.8 / r63 7.1, z10 -1.97** (tape -2.29) with NSC 0.4, UNP 0.4, CSX 0.8, DOV 1.2, ITW 1.6, PH 1.6, MMM 2.4 — a twelve-name industrial and rail rank floor; XLF r63 99.2; XLV r63 97.6; SMH r63 0.8; XLRE r5 4.8 | **C4** (family form), and see the dismissals below |
| **utilities, single names** | **PCG -26.84% and EIX -25.93% over five sessions**, both at r5 <= 0.8 and ~-30% off their highs, while XLU is only -1.93% over five and no other utility is below r5 22. A two-name California event, not a sector washout | **C5** |
| **gold and miners** | GLD -4.40% over 5 (r5 9.1) but +7.65% over 21 and -18.78% off its high; GDX +25.30% over 21 at r21 94.4 yet -15.72% off its high; NEM +28.07% over 21 | C6 |
| **other metals** | SLV +1.99% today after the 2026-08-28 three-name break; -44.06% off its 52-week high | folded into C6 |
| **international** | EWZ +4.16% on the day, r5 96.0; EEM r63 1.6; FXI r63 69.8 | C9 |
| **us_large** | Nothing extreme. SPY r5 40.1, QQQ r63 4.8 (the one soft spot), DIA r21 9.5 | C12 uses SPY as the event vehicle, not as a price state |
| **us_small** | IWM r5 17.9, -3.63% off its high. Not extreme | dismissed above |
| **dollar and FX** | DX 21d rank 41.7, UUP 42.9. Mid-range in every lookback | dismissed above |
| **single names, non-sector** | CRM +24.9% over five sessions at r5 99.6; DE +10.1% with z10 2.6 at a 52-week high; MU +706% off its 52-week low | **Dismissed.** Both CRM and DE are post-earnings continuation states, which is exactly what the book's ATR Extended Gap Up and Overbot Vol Spike are pointed at, and a single-name PEAD long has no novelty axis available that the scanner does not already occupy. MU's distance-to-low is a denominator artifact of a 2025 collapse, the 2026-08-13 IHI lesson. |

---

## 3. Seasonal and cycle cells

- **September, midterm year.** Closed hard on 2026-09-02: anchored at trading day 1,
  ^GSPC September pays -0.042% at h=3 on 15-11, an excess of -0.080pp over all other
  months, and the midterm crossing is POSITIVE at +1.344% on 4-2. Over the 48-cell
  month x cycle grid it ranks 43 of 48 from the negative end with P(min-of-48) = 1.0000.
  **No September-weakness candidate this morning**, and any that appears in a later
  run owes the grid permutation before the anecdote.
- **Midterm as a conditioner, not an idea.** The seasonal board (asof 2026-08-05, five
  sessions stale, flagged) carries the book-level midterm read: win 56.4% against 64.9%
  all-years over 1099 trades, +0.24R against +0.43R. It conditions C1 (the exact live
  cell — a September print in a midterm year with the gate on — has N=0), C3 and C6.
  It kills watchlist 0, 23 and 27 outright.
- **Trading day of month 3.** No live month-position cell. The one in the watchlist
  (31, the small-cap month-end overnight) parks to December.

---

## 4. Watchlist verdicts — all 35 active entries

CHECK = trigger moved, worth today's compute. PASS = trigger unchanged, today's value cited.

| # | cell | verdict |
|---|---|---|
| 0 | nfp x rates | PASS. Midterm; parks to the first non-midterm NFP, 2027-01. NFP is +1 td and the cell is still dead in this cycle year. |
| 1 | credit price-state | PASS on the episode count, 4 against the 8 required. State is live and TIGHTER than yesterday: LQD +0.12% above its 252d low, HYG -0.79% off its high. Feeds C8's framing but does not arm. |
| 2 | cpi x volatility | PASS. The overnight entry is the 2026-09-10 close, five sessions out. |
| 3 | gold price-state | PASS. GDX r5 21.4 against the >= 95 leg; GLD -18.78% off its high against the added within-10% leg. Both legs fail. |
| 4 | energy price-state | PASS. The arm fired and the cell died on 2026-09-02; USO's 1-day move today is +0.11%, nowhere near the [5,6)% band. |
| 5 | rates price-state | PASS, and it is the closest of the rates entries: TLT +1.12% above its 252d low against the <= 0.5% rung, IEF +0.36% and LQD +0.12% both clear. Same single failing leg as 2026-08-12. |
| 6 | vol x us_large | PASS. ^SKEW r5 57.1 against the >= 95 leg (its 21-day rank of 99.6 is a different lookback and is C10's object, not this entry's). Midterm block also stands. |
| 7 | energy x event | PASS. USO r63 49.6 against the <= 20 deep-base leg. |
| 8 | sectors price-state | PASS. IHI r21 61.9 against the rank-100 rung. |
| 9 | international | PASS. FXI r5 48.4 against the <= 20 break leg. |
| 10 | rates seasonal | PASS. Parks to trading days 4-12 of November 2026. |
| 11 | rates x us_large | PASS. SPY -1.64% off its high against <= 0.5%; TLT +1.12% above its low against <= 1%. Both legs fail. |
| 12 | vol x us_large | PASS. VIX 21-day return rank 32.5 against <= 25, and the pop leg is wrong-signed anyway (VIX -6.98% today against a >= +5% rung). |
| 13 | rates x fx -> gold | PASS. DX r21 41.7 against <= 15; the 21-session yield rise clears its half. |
| 14 | sectors rotation | PASS. One-day XLV minus XLK gap is +0.77pp against the >= +3.0pp rung. |
| 15 | rates x fx | PASS. ^TNX r21 83.7 clears >= 65 for a second session; DX r21 41.7 fails <= 20. |
| 16 | rates price-state | PASS. TLT 1-day +0.10% against the >= +1.5% thrust rung. |
| 17 | financials breadth | PASS. The arm is an ex-crisis cost threshold that no single new episode moves. |
| 18 | rates curve | PASS. 252-session yield change **+51.9 bp** against the +78 bp arm (^TNX closed at its 252d max again, so the proximity leg is not the binding one — the magnitude floor is, exactly as the 2026-09-01 re-arm intended). Note the number FELL from yesterday's +62.0 bp, because the trailing reference bar rolled. |
| 19 | energy breadth | PASS. Count of the 11-name complex at z10 >= 2.0 is **0** (max SLB +0.97) against the [2,3] arm. |
| 20 | us_large breadth | PASS. SPY -1.64% off its high against > 2.0%; raw-21d fragility 61.4 against <= 50. |
| 21 | sectors family | PASS. XLI r5 0.8 clears the washout leg but is -7.36% off its high against within-5%. C4 tests a deliberately DIFFERENT form (see its note). |
| 22 | utilities x rates | PASS. XLU r21 18.3 against <= 5; TLT's 21-day rank is 45.6 against < 25, wrong-signed a sixth straight session. |
| 23 | dollar_fx | PASS. Midterm-blocked, and DX r21 41.7 is not a washout in any case. |
| 24 | credit x us_large | PASS. HYG -0.79% off its high against the <= 0.05% touch; dial 87.9 against the < 50 requirement. |
| 25 | semis price-state | PASS. SMH r63 0.8 so the floor is live, but r5 31.0 against the < 15 still-falling arm, and the 23-ETF Cochran Q is unmoved by one session. |
| 26 | rates x credit | PASS. HYG -0.79% off its 252d high against the <= 0.25% rung, even though LQD is now +0.12% above its low. Episode count still one. |
| 27 | jackson_hole x rates | PASS. Midterm-blocked and the anchor is four sessions gone. |
| 28 | cross-asset laggard | PASS. **No holder** of r21 >= 90 AND r63 <= 10 anywhere in the 218-name tape. |
| 29 | metals | PASS. SLV BOUNCED +1.99% today; the depth arm needs a break of -4.00% or worse and 2026-08-28's was -3.68%. The parabolic route is closed as of yesterday. |
| 30 | rates x bond vol | PASS, and it moved further away: ^MOVE trailing-252 LEVEL percentile **83.9** against the [40,50) band. That reading is C2's object from the other side. |
| 31 | us_small seasonal | PASS. Parks to December. |
| 32 | energy x us_large | PASS. XLE closed at its 252d max again, but on a session SPY ROSE +0.44%, and the entry requires a down-index session. The standing blocker (inverted dose response on SPY's own move) is not a number and may not be waived. |
| 33 | **nfp x volatility** | **CHECK — this is C1.** The arm is a DATE and today is it: today's close is the last session before the 2026-09-04 print, and the rel-range percentile closed at **3.57** against the <= 15.0 rung (abs-range 1.98). All three stated debts travel with it. |
| 34 | sectors pooled | PASS. XLI holds the triple rank floor (0.8 / 2.8 / 7.1) and eight other names do too, but the arm is "a reason to exist beside the book" and the registry finding that the bare pooled floor IS the book stands. C4 is the attempt to find that reason; if C4 dies, this entry stays parked. |

Pruning after publish: none expire today. Entry 3 expires 2026-11-11 and is the next one due.

---

## 5. Scoreboard read (required before selection)

Lifetime graded ideas: **4**. That is a handful, not a signal — no axis or grade split
is actionable yet. For the record, the four sit at B avgR +0.448 (n=3) and C +0.146
(n=1), with event_fingerprint at +0.622 on two graded. Nothing here earns or loses an
axis a slot this morning; noted and moved past, per the skill's instruction.

---

## 6. Selected candidates

Twelve candidates, **eight asset classes** (volatility, rates, us_large, sectors,
credit, energy, gold and miners, international), **six novelty axes**, eight
event-anchored and four price-state-anchored.

| # | candidate | axis | class |
|---|---|---|---|
| C1 | Long SVXY MOC into the payrolls print out of a dead 21-day VIX range (watchlist 33, ARMED) | event_fingerprint | volatility |
| C2 | Bond volatility bid while equity volatility is dead, into the labour print | interaction_cell | rates x volatility |
| C3 | Long duration into the payrolls print with the ten-year AT a 252-day yield high | event_fingerprint | rates |
| C4 | The twelve-name industrial and rail rank floor as a FAMILY, not an XLI call | relative_value | sectors |
| C5 | The two-name California utility catastrophe after a 26% five-day collapse | historical_analogue | sectors / utilities |
| C6 | Long gold into the print after a five-day metal washout with the miners still bid | interaction_cell | gold and miners |
| C7 | The energy complex AT 52-week highs into a payrolls print | event_fingerprint | energy |
| C8 | Investment grade at its 252-day low while high yield holds its high, into the print | interaction_cell | credit |
| C9 | The Brazil thrust into a payrolls print with the dollar mid-range | event_fingerprint | international |
| C10 | Tail hedges bid while at-the-money vol is dead: ^SKEW r21 99.6 against a 1st-percentile range | inversion | volatility / us_large |
| C11 | Which vehicle expresses C1: SVXY against short UVXY against spot | instrument_translation | volatility |
| C12 | The equity DIRECTION leg of the same pre-print cross | event_fingerprint | us_large |

Negative-registry collisions declared up front, each owed a difference in the write-up:
- C1 vs "post-NFP, post-FOMC and post-VIX-expiry vol cells, swept and empty" (2026-08-06):
  that sweep was POST-event. C1 enters before the print and exits at its close.
- C1 vs "'the range has been dead, then it broke' is monotone in the WRONG direction"
  (2026-09-02): that cell requires a >= 8% VIX POP to have already happened and is long
  SPY afterwards. C1 requires no pop and is short vol before an event. Day-level overlap
  must be measured, not asserted.
- C10 vs "a skew spike with a low-vol filter attached: the filter subtracts" (2026-08-12).
- C4 vs the three XLI kills (2026-08-24 pair, 2026-08-26 intact-trend washout, 2026-09-02
  triple floor) and against "the index-near-a-high gate is a bull-tape selector", confirmed
  three times.
- C7 vs the whole 2026-09-02 energy section, and against "a commodity index at a fresh
  52-week high into a CPI print" whose placebo anchor ladder is four-for-four as a killer.
- C12 vs "post-NFP equity DIRECTION, separately swept and empty" (2026-08-07).
- C6 vs the 2026-08-21 GLD teardown and the 2026-09-02 miner-versus-metal kills.
