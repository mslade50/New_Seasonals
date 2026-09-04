# Surface map — 2026-08-14 (Friday)

Stage B1. Written before any candidate was generated. Facts cited here come
from `00_map_facts.py` (run first) plus `data/pitch_state.json` /
`data/pitch_tape.json`. Every cell gets a verdict; dismissals carry a reason.

**Session state.** Anchor = 2026-08-13 close (freshest bar, no staleness
warning, pipeline 7/7 green). Entry MOC = Friday 2026-08-14. A 10 td hold
exits 2026-08-28. Midterm year (2026 % 4 == 2). SPY at a 52w high (0.00% off),
z10 +1.77, breadth 75.2% above the 200d. Fragility ma10-63d 76.8 (raw 63d
94.9, up from 32.1 twenty-one sessions ago), exposure leg 0.0x, P/C fear OFF
at the 42nd percentile, one signal on (Low Absorption Ratio, 1st pctile).

**The structural fact about today's surface**: the last five mornings swept
the macro-event calendar in this window nearly to the floor (NFP, CPI, PPI,
Jackson Hole, VIX expiry, opex, the macro vacuum between them). What is left
that is genuinely unexamined is (a) the vol-surface state, which is at a
different pole than anything checked so far, (b) equity micro-structure —
industry-level and earnings-anchored cells the product has never opened, and
(c) the commodity-vs-producer split. The map is weighted accordingly.

---

## 1. Calendar events x asset class

Seven events in the [-5, +15] td window. Three are behind the entry
(unusable as anchors, listed for completeness), four are ahead.

| event | date | td | reachable from a 14 Aug MOC entry? |
|---|---|---|---|
| nfp | 08-07 | -5 | no, behind entry |
| cpi | 08-12 | -2 | no, behind entry |
| ppi | 08-13 | -1 | no, behind entry (it IS the anchor session) |
| vix_expiry | 08-19 | +3 | yes, inside any h>=3 hold |
| opex | 08-21 | +5 | yes, inside any h>=5 hold |
| jackson_hole | 08-28 | +10 | yes, exactly at h=10 |
| nfp | 09-04 | +15 | **NO — unreachable.** Max legal horizon is 10 td, exiting 08-28. Any September-NFP cell is out of scope today by construction, which retires the whole nfp x class row. |

### The ten classes, crossed with the four reachable events

| class | vix_expiry (+3) | opex (+5) | jackson_hole (+10) |
|---|---|---|---|
| us_large | **NOT EXAMINED — dead.** Registry 2026-08-07: VIX-expiry-week drift raw +0.175% (N=319) is mid-month position, within-month paired excess +0.065% (t 0.67), 2018+ paired excess negative, and the settle day itself is the worst day. | **CHECK — C6.** Registry killed "the run INTO August opex" measured from the NFP close over 10 td (+0.342% vs SPY's +0.374% unconditional). It did NOT measure the pre-opex WEEK itself from the Friday before, which is the window a 14 Aug entry actually buys, and it is the one calendar cell in this window with an untested definition. Worth one check to close the row. | dead. Swept 2026-08-13 on rates, gold, FX and (08-11) small caps; the anchor turned out to be decoration on an unconditional Aug 6-16 seasonal (+1.025%, t 6.90, N=189, no event involved). Equity leg inherits the same control problem and the entry sits at h=10 exactly, the worst-resolution end of the horizon scan. |
| us_small | dead, same cell as us_large; IWM adds noise not information. | folded into C6 as the second leg if the SPY cell shows anything. | dead (2026-08-11 sweep). |
| rates | not examined: no mechanism links a VIX SOQ to duration, and ^MOVE's LEVEL is at the 32.9th pctile so there is no bond-vol extreme to hang it on. | not examined, same reason. | dead, killed in detail 2026-08-13 (+1.162% over 24 events ranks 14th of 127 placebo offsets; the Aug 6-16 seasonal explains it). |
| credit | not examined. HYG's ATR is 0.25%/day — an event edge would have to be enormous to clear a two-leg round trip, and none of these events has a credit mechanism. | not examined, same. | dead. |
| gold_miners | not examined; no mechanism. | not examined. | dead: GLD across the JH run is 10-11 at +0.577% with 92% of it in two episodes, midterm -1.213% at 1-4, and an independent Aug 6-16 midterm control agrees at -0.859% (t -2.53). |
| other_metals | not examined; SLV has no event channel to any of these three. | not examined. | not examined — the gold leg already failed and silver is the higher-vol expression of the same trade. |
| energy | not examined; no mechanism. | not examined. | not examined. |
| dollar_fx | not examined. DX is quiet (rank5 48.8, rank21 35.7, rank63 54.4, 1.62% off its 52w high) so there is no state to condition on, and the event that DOES move it, NFP, is unreachable at +15 td. | same. | dead: 13-13 at +0.090%, drop-best flips the sign, the midterm cell is entirely 2022's +3.00%, and 9 bps is 4.5x a DX round trip against the 5x bar. |
| international | not examined; no mechanism from a US vol settle to EFA/EEM/FXI. | not examined. | not examined. |
| volatility | **CHECK — folded into C6.** The distinct thing about this week is that BOTH vol events sit inside a 5 td hold. Registry killed pre-expiry short-vol carry (long SVXY into VIX expiry) and every post-event vol cell except post-opex. Post-opex IS the book's live V4 sleeve trade (long SVXY, opex MOC to +3 sessions, fires 08-21), so a short-vol idea here would be re-running the book — dead on arrival by the anti-rip-off rule. What is left is the pre-opex week direction, which C6 covers. | as left. | dead. |

**Earnings is the event lane that is NOT swept.** 13 liquid prints inside 10
td and the product has never anchored on one. Two rows below (C3, C4) open it.

---

## 2. Tape extremes, by class

Sorted from the full 218-name tape (`00_tape_sort.py` output in the run log),
not from names I already had a view on.

| class | the outlier(s) | verdict |
|---|---|---|
| us_large | SPY at its 52w high, z10 +1.77 (7th highest in the tape), rank21 74.6. QQQ -1.78% off its high but rank63 27.4. | context for everything below rather than a cell of its own. The "stretched high" long is the book's own territory and the weekend-discount short is a registry-certified pre-2013 fossil (2013+ -0.010%, and the weekday placebo has Tue/Wed/Thu significantly positive). |
| us_small | IWM at its 52w high, rank63 49.6, nothing extreme. | dismissed, no extreme. |
| rates | TLT/IEF/LQD all within 1.25% of 52w lows; ^TNX rank21 67.1. | **watchlist W6 governs — PASS, see section 4.** The tight rung is now OFF (TLT 0.82% vs the 0.5% it needs, IEF 1.23% vs 1.0%, LQD 1.16% vs 1.0%): Thursday's +0.58% TLT bounce took the state out. Nothing to check. |
| credit | HYG exactly at its 52w high (0.00%) while LQD sits 1.16% off its 52w low. | **watchlist W2 governs — PASS.** State live, episode count still 4 with three in 2018, and the live cluster began 2026-07-22 so today is a mid-cluster entry. Unmeasurable is a kill. |
| gold_miners | GDX rank21 89.7 / rank63 24.6; GLD rank21 74.6 / rank63 24.2. Both -20% to -24% off 52w highs. | **watchlist W4 governs — PASS, trigger not live**: the cell needs GDX rank5 >= 95 while GLD < 95 and today is 70.2 / 68.7, i.e. no divergence at all. GLD/GDX correlation now +0.831. |
| other_metals | SLV rank63 3.6 (-26.7% over 63d) yet rank21 69.0 (+11.4%) — the sharpest washout-plus-thrust in the tape, -44.9% off its 52w high. | dismissed into the dead family. This is the exact "deep laggard snapping back" shape killed three times (SMH/QQQ 2026-08-07, EWZ 2026-08-12, FXI 2026-08-13) and once as an outright (SMH 2026-08-12). Re-opening it needs a new mechanism and silver does not supply one. |
| energy | **the split is the story**: XLE rank5 95.6 / rank21 79.8 / ret63 +6.7% while USO rank63 6.3 / ret63 -12.0% and DBC rank63 6.7. Producers have beaten the barrel by ~19pp over 63 sessions. VLO 99.2, XOP 97.6, COP 94.8 all in the top of the 5d rank. | **CHECK — C5.** Distinct from watchlist W5, which needs a crude one-day POP (today is -1.78%, no pop). This is the standing 63-day divergence between the commodity and its producers, which no morning has opened. |
| dollar_fx | nothing: DX rank5 48.8, rank21 35.7, rank63 54.4, ATR 0.43%. | dismissed, no extreme and no reachable anchor. |
| international | EEM rank63 2.8 with rank5 77.0; FXI rank5 15.5; EWZ rank5 3.6 and z10 -1.69, the tape's worst. | dismissed. EEM is the dead laggard-snapback family again. EWZ's washout long was killed on its own numbers 2026-08-12 (top-2 episodes +60.6pp of a +85.8pp total; tightening rank5 3 -> 1 flips the sign). FXI is watchlist W10, PASS: rank5 15.5 clears but rank21 61.9 misses the >= 80 leg. |
| volatility | **the double extreme.** ^SKEW LEVEL at the **1.6th percentile** of its trailing year AND its 21d return at the 2.0 rank (-9.52%). ^VIX LEVEL at the 4.8th pctile, ^VIX3M 15.1st. SVXY 0.25% off its 52w high. | **CHECK — C1, and it is the freshest thing on the board.** Every prior skew check in this repo measured a skew SPIKE (rank5 >= 95, long SPY, parked to W7). The bottom pole has never been touched, and today is not a marginal reading of it. Note the 2026-08-10 MOVE lesson and quote the LEVEL percentile, not only the rank. |
| sectors (not in the B1 table, but where the cross-section actually is) | XLF rank63 **100.0** at a 52w high with BAC/JPM also at 100.0 and KRE 90.9 — financials at maximum leadership. Inside that sector the **insurance complex broke together**: AFL rank5 1.19, HIG 3.97, ALL 4.37, TRV 5.56, AIG 6.35, CB 9.52, MET 15.08, PGR 15.87 — 8 of 10 in the bottom sixth of their own 5-day distributions, down 2.5% to 5.3% on the week, while their 21d and 63d ranks stay high (ALL 94.8, TRV 96.4, LNC 100.0, PRU 95.6, MET 92.9). | **CHECK — C2.** A whole industry breaking on the same week inside an intact uptrend, while its parent sector prints a rank of 100. Never examined here. Collides with the dead "break inside an intact thrust" family and the check must confront that head on — the difference claimed is that this is a synchronized BREADTH event across 8 names rather than one index decoupling. |
| semis | ADI rank63 **0.4** (the tape's worst), INTC 1.6, MU 2.4, SMH 2.4 — the deepest 63d laggard cluster anywhere, with NVDA printing 08-26 (+8 td). | **CHECK — C4, on the earnings anchor only.** The price-state form is registry-dead twice over (the SMH/QQQ pair and the outright). What is new is the event: a pre-print window is not a laggard-snapback claim. |
| retail | TJX rank5 3.97 (-4.79% on the week) printing 08-19; ROST rank5 5.56 (-3.68%) printing 08-20; DG rank5 16.7; against TGT +6.54% and WMT +3.26%. | **CHECK — C3.** Washed-out retailers days before their own prints, with the whole sub-industry reporting the same week. Earnings-anchored, exit before the print. |
| utilities | EIX rank21 3.2, SRE 4.0, CNP 4.4, ETR 5.6, PNW 7.1, DTE 9.9 — loud again. | **dismissed without a check, on standing orders.** Utilities are dead in SIX expressions (outright washout, XLP pair, SPY spread, rates channel, rank21 form, XLU/XLV dispersion). The registry's own line applies: a sector being the loudest thing in the cross-section is a reason to look, not evidence of an edge. No new mechanism today. |
| mega-cap tech | GOOG rank63 0.4 and rank5 13.1, AAPL rank21 6.3 / z10 -1.24, META rank21 8.3, while QQQ sits 1.78% off its high. | dismissed into the dead family (one name decoupling from an intact thrust), and the single-ticker version additionally owes the 2026-08-13 reference-class charge, which is what killed IHI. |
| extension tail | MU +721% off its 52w low and +73.1% above its 200d; INTC +370.6%; RHI +55.3%. | dismissed. A short here is a momentum fade on the strongest names in the tape, which is the book's 3x-fade family territory measured on 1x names — killed explicitly 2026-08-13 (IHI form, -0.953% at 5-15). |

---

## 3. Live seasonal and cycle cells

- **Midterm year.** Conditions everything rather than being a cell. The board
  itself reads "de-risk" (book win 56.4% midterm vs 64.9% all-years, n=1099).
  Registry: midterm mid-August seasonality as a standalone is dead (N=6,
  carried entirely by 2002's +8.68%, drop-two-best negative, and the midterm
  restriction anti-works at 21 td). **Not a candidate; a conditioner every
  check must cross.**
- **Aug 6-16 window.** The 2026-08-13 sweep established this is a real
  unconditional seasonal on TLT (+1.025%, t 6.90, N=189) and it is the control
  that ate the Jackson Hole cell. Entry today is at the tail end of it, so any
  rates idea today would be buying the last two sessions of a window whose
  edge is already spent. Reinforces the rates dismissal above.
- **Seasonal board**: 0 A+B-grade setups flagged. Nothing live to inherit.
- **Trading day of month 10 of 21.** Mid-month. The 2026-08-10 lesson (a tdom
  control is mandatory on any rates cell) is moot today since no rates cell
  survives to a check, but C6 owes a month-position control because "pre-opex
  week" and "third week of the month" are the same days.

---

## 4. Watchlist verdicts (all 10 active entries, every one cited with today's number)

| # | entry | verdict |
|---|---|---|
| W1 | TLT from the NFP close at the 52w rates floor | **PASS.** Trigger unchanged and structurally unreachable: it needs a non-midterm NFP (first is 2027-01) and the next NFP is 2026-09-04, both midterm and 15 td out, beyond any legal horizon. |
| W2 | Long LQD / short HYG at joint 52w extremes | **PASS.** State live (HYG 0.00% off its 52w high, LQD 1.16% off its low) but the count is still 4 declustered episodes with three in 2018 and the fourth being the live cluster from 2026-07-22, so today is a mid-cluster entry. Needs >= 8 episodes across >= 3 non-2018 years. |
| W3 | SVXY overnight into the CPI print | **PASS, and the owed re-measure is deferred with cause.** CPI printed 2026-08-12 so the re-measure is due, but the next CPI is 2026-09-11, 20 td out and unreachable from any legal horizon. Owed at the 2026-09-10 run, as the entry itself says. |
| W4 | Long GLD on a miner-led thrust the metal has not joined | **PASS, trigger not live.** Needs GDX rank5 >= 95 while GLD < 95; today is GDX 70.2 and GLD 68.7, no divergence in either direction. |
| W5 | Long XLE on a crude one-day pop in the 5-6% band | **PASS, trigger not live.** USO's one-day move is -1.78%. Not a pop. (The standing 63d divergence between XLE and USO is a different object and is candidate C5.) |
| W6 | Long TLT with the whole IG complex at 52w lows, on a FRESH trigger | **PASS, and the price rung has now switched OFF.** TLT 0.82% off its 52w low against the <= 0.5% the tight rung needs, IEF 1.23% (needs <= 1.0), LQD 1.16% (needs <= 1.0). Thursday's +0.58% TLT session ended the state. The freshness leg was already failing (episode began 2026-08-03, 4 trigger days). |
| W7 | Long SPY on a skew SPIKE alone | **PASS, trigger not live, and today is its mirror image.** ^SKEW rank5 is 47.6 against the >= 95 required; SPY is 0.00% off its 52w high against the > 1% below required; and 2026 is a midterm year, which the entry's second arming leg excludes. Today's skew state is the opposite pole and is candidate C1 — a distinct cell, not this one firing. |
| W8 | Fade a crude thrust out of a deep base with a macro print inside | **PASS.** Still 4 post-2020 episodes against the 8 required, and there is no thrust to fade (USO 1d -1.78%). |
| W9 | Long IHI on a 21d-rank-100 thrust | **PASS, and it cannot fire on a price move.** Its trigger is a reference-class condition (Cochran Q p < 0.05 across 27 sector ETFs, measured 0.544) plus episode-first freshness. A structural gate does not flip in one session. |
| W10 | FXI's five-day break inside an intact thrust | **PASS.** FXI rank5 15.5 clears the <= 20 leg but rank21 is 61.9 against the >= 80 required, so the "intact thrust" half is absent. EEM's 5d return is +2.55%, which does clear. Two of three legs fail. |

No entry has expired and none fired. Nothing to prune before publish beyond
whatever today adds.

---

## 5. Axis and grade read from the scoreboard

4 pitched, 2 graded, avg +0.468R, both winners. By axis: event_fingerprint
1 graded at +0.837R, relative_value 1 graded at +0.099R, interaction_cell
ungraded. **The graded count is a handful, so there is nothing to read yet**
and no axis is penalised or favoured on this evidence. Revisit once the
graded count reaches double digits.

---

## 6. Selected candidates (stage B2)

Eleven, from the cells marked CHECK above plus the coverage the map owes.

| # | candidate | axis | class | anchor mode |
|---|---|---|---|---|
| C1 | Skew at a 1.6th-percentile LEVEL with VIX also bottom-decile and SPY at a 52w high — direction to be decided by the data, not the folk story | inversion | volatility / us_large | price-state |
| C1b | The same state expressed as short vol rather than long equity (SVXY), because "everything is cheap" has two readings | instrument_translation | volatility | price-state |
| C2 | The insurance complex breaking together inside an intact uptrend, while XLF prints a 63d rank of 100 | interaction_cell | sectors / us_large | price-state |
| C2b | The same, expressed relative to XLF, so the check prices the legs before the spread | relative_value | sectors | price-state |
| C3 | A washed-out retailer into its own print days later, exiting before the print | event_fingerprint | consumer / us_large | event (earnings) |
| C4 | The semis complex at a 63d-rank floor into the NVDA print | event_fingerprint | sectors | event (earnings) |
| C5 | Producers against the barrel: XLE has beaten USO by ~19pp over 63 sessions | relative_value | energy | price-state |
| C6 | The pre-opex WEEK itself from the Friday before, with both vol events inside the hold — the one calendar definition in this window that has never been measured | flow_mechanics | us_large / volatility | event |
| C7 | Nearest-neighbour tapes to today by vol level, skew level, extension and breadth, and what the following week did | historical_analogue | us_large | price-state |
| C8 | Bond vol compressed into a quiet macro window, long duration | interaction_cell | rates | price-state |
| C9 | Long the insurance breadth event's single strongest name rather than the basket, to test whether the basket is doing any work | relative_value | sectors | price-state |

Coverage check against the B1 requirements:
- **asset classes touched: 6** (volatility, us_large, sectors, consumer,
  energy, rates) against a floor of 4.
- **novelty axes: 6** (inversion, instrument_translation, interaction_cell,
  relative_value, event_fingerprint, flow_mechanics, historical_analogue —
  seven, in fact) against a floor of 4.
- **event-anchored candidates: C3, C4, C6.** **Price-state anchored: C1, C1b,
  C2, C2b, C5, C7, C8, C9.** Both modes present, and C4/C3 explicitly cross
  the two by putting an event anchor on a price-state cell, which is the
  crossing the 2026-08-07 post-mortem said was missing.
- registry collisions declared up front: **C2/C2b vs the "break inside an
  intact thrust" family; C4 vs the twice-dead SMH laggard; C6 vs the dead run
  into August opex; C8 vs the 2026-08-10 MOVE level-vs-rank trap and the
  mandatory tdom control; C1b vs the dead pre-expiry short-vol carry.** Each
  check must state what is different or kill it.
