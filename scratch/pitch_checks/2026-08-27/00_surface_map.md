# Surface map — 2026-08-27 (Thursday, midterm year, ME-2, Jackson Hole tomorrow)

State files: `data/pitch_state.json` (382 KB), `data/pitch_tape.json` (218 names).
Freshest bar **2026-08-26** = the prior session. `warnings` is EMPTY, `pipeline.ok`
is true, all 7 pipelines green. No staleness caveat is owed on any claim today.

Recon written this morning: `00_tape_sort.py` (+ `.txt`), `01_recon.py`,
`01b_recon_xsec.py`, `01c_watchlist_verdicts.py`.

## 0. The two facts that condition everything below

**(a) The fragility dial is at the top of its own history.** ma10(63d) = **88.6**
(raw 63d 90.4, raw 21d 71.4). Only **23 days in the entire 2016+ series** have
printed ma10(63d) >= 85, and they sit in three episodes: 2021 (9 days), 2022 (7),
2026 (7). The practical consequence for a price-state pitch is not that the dial
predicts anything — the registry closed the dial as a directional signal
(2026-08-07, `d1_dial_spike_calm_surface.py`) and as a sizing rule (RISK_DIALS
2026-07-16) — it is that **today's reading sits outside the support of almost every
cell this repo has measured.** Watchlist 23 records its trigger population's dial
max at 80.6; watchlist 25 records 68.6; watchlist 28 records 68.0 with zero days
at or above 70. Any cell shipped today is being extrapolated to a regime it has
never been measured in, and the write-up owes that sentence.

**(b) The macro calendar is exhausted, for a FOURTH consecutive session, except
for two anchors that came into range this week.** The registry's own 2026-08-26
calendar finding named the next new anchor as "the September CPI/PPI pair on
2026-09-10/11, which enters the horizon around 2026-08-28". It is +9 and +10 td
today, i.e. inside the cap for the first time, so it is checked below rather than
dismissed. Jackson Hole is TOMORROW and has been swept to seven classes; the two
that were never covered are volatility and sectors, and volatility is checked.

## 1. Every live calendar anchor x every asset class

Anchors inside or adjacent to the 10 td horizon. `opex` at -4 td is included
because post-opex windows are still live; FOMC (+13), VIX expiry (+13), the
September opex/quad (+15) and the election (+47) are **beyond the 10 td horizon
cap and are dismissed on that alone** — no hold started today reaches them.

Anchors: **JH** jackson_hole 2026-08-28 (+1) | **NFP** 2026-09-04 (+6) |
**PPI** 2026-09-10 (+9) | **CPI** 2026-09-11 (+10) | **OPX** opex 2026-08-21
(-4) | **ME** month end 2026-08-31 (today = ME-2).

| class | JH (+1) | NFP (+6) | PPI (+9) / CPI (+10) | OPX (-4) | ME (-2) |
|---|---|---|---|---|---|
| us_large | CLOSED 2026-08-18: ladder 8 of 16 at h=10, plateau, the unconditional August window beats the anchor, midterm inverts to -1.485% | CLOSED 2026-08-07 + 2026-08-26: post-NFP direction swept, ladder closed, September-midterm cell -0.676% | **CHECK (C3)** — a hold entered today CONTAINS both prints for the first time. The parent print cells are dead (SPY on PPI day -0.009% against a +0.039% same-span drift) but the both-prints-inside-one-hold object has never been run | CLOSED both directions 2026-08-20 | CLOSED 2026-08-26 in both the month-position and month-of-year senses (`c9b`, `c9c`) |
| us_small | CLOSED 2026-08-11 (JH-13 IWM, midterm wrong-signed) | see us_large | folded into C3 | CLOSED | CLOSED 2026-08-26: the ME-3 to ME-2 session is the whole effect, it is a 16x12 grid result at a scan-charged P of 0.063, and **midterm pays -0.366% on a 3-3 record**. The entry was yesterday's close in any case |
| rates | CLOSED 2026-08-13: the anchor is decoration on an August seasonal and the mechanism loses inside its own window; the August TLT seasonal itself closed 2026-08-17 as a bond-bull fossil | Watchlist 0: midterm-dead (+0.071%, N=12, 58% hit) against non-midterm +0.978%. PASS, parks to 2027-01 | PPI-on-the-curve is real but **exactly one session wide** (2026-08-10) and arms only on the print eve, 8 sessions away, so it is not reachable today. CPI-TLT closed 2026-08-10 as a trading-day-of-month artifact | CLOSED | SUSPENDED (watchlist 12): the stated blocker is literally "NOT August", and the ME-1 to ME-0 session decayed to +3.99 bp at t=0.37 in 2020-2026. PASS |
| credit | CLOSED 2026-08-21: HYG's August-tdom excess of +0.345pp sits BELOW SPY's +0.700pp, so credit subtracts from an equity leg already closed | not examined — the NFP anchor is closed on its two strongest classes and credit is the weakest of the ten here (HYG's ATR is 0.23% of price, so an 8 bp edge cannot pay 4 bp twice at this horizon) | Long HYG into CPI CLOSED 2026-08-11 (era sign flip, +17.8 bp pre-2018 to -2.8 bp after) | CLOSED | the ME-1 to ME-0 decay was replicated on LQD (+23.38 to +3.56 bp) and AGG. Dismissed as measured and decayed |
| gold_miners | CLOSED 2026-08-13: 10-11 at +0.577%, 92% of it two episodes, **midterm -1.213% at 1-4**, with an independent Aug 6-16 midterm control agreeing at -0.859%, t=-2.53 | not examined: gold's event lane is the print itself, and the 2026-08-07 sweep showed pre-event windows on an event's own instrument underperform that instrument's unconditional drift | GLD pre-CPI CLOSED 2026-08-07 (+0.040% against a +0.092% h=2 drift). GDX pre-CPI closed 2026-08-10 | CLOSED 2026-08-17 (the GDX/GLD ratio cell) | not examined — the metals story today is a price state (C6), not a calendar one, and the month-turn commodity cell is taken up as C2 |
| other_metals | folded into gold_miners; SLV's JH cell has no mechanism separable from gold's | not examined, same reason as gold_miners | not examined | CLOSED 2026-08-21 (`b1c_c3_slv_teardown.py`) | **CHECK (C2)** — the month-end anchor is closed on equities, suspended on rates and closed on FX. **Commodities and metals is the one class never swept at the month turn.** Weak prior, but it is the last unswept cell of a live anchor |
| energy | Watchlist 17 (USO at JH-6): PASS — the JH-6 anchor was 2026-08-20 and today is JH-1, and XLE at -2.07% off its 52w high sits inside the 5% band the entry forbids | not examined: energy's release lane is EIA and OPEC, neither of which is in `macro_events.csv` | Energy's washout-into-CPI CLOSED 2026-08-10 (the CPI anchor SUBTRACTS: +0.464% alone, -0.441% pre-CPI). Short energy across a PPI print CLOSED 2026-08-12 on the placebo ladder | CLOSED | folded into C2 |
| dollar_fx | CLOSED 2026-08-13: 13-13 at +0.090%, drop-best flips the sign, and the midterm cell is entirely 2022's +3.00% | the registered one-day weak-NFP-close DX cell exists but does NOT transfer to any 21-day parent (watchlist 27), and NFP is +6 td, so nothing is enterable today | DX into CPI is -3.5 bp, which pays neither the futures nor the ETF round trip (2026-08-07) | CLOSED | CLOSED 2026-08-26 — FX was the month-end anchor's last unswept class |
| international | CLOSED 2026-08-21 alongside credit (six vehicles by ten horizons; every SPY-beta-hedged residual inside +/-0.12pp except FXI's -0.738pp) | not examined: the anchor's own class sweep is closed, and international adds a currency leg to an already-null event | not examined, same | CLOSED (the FXI ladder ranks 2 of 17 at h=10 and 9 of 17 at h=5) | not examined: there is no international flow mechanism at a US month turn separable from the equity cell already closed |
| volatility | **CHECK (C1)** — the eighth class, never swept. The seven closed classes were all DIRECTIONAL; a vol cell asks a different question (is the speech sold as event premium) and the event is tomorrow | post-NFP vol CLOSED (registry, event_seasonality_sweep) | post-CPI vol crush CLOSED (it died after 2018); short vol across a PPI print CLOSED 2026-08-12 on an eight-sessions-later placebo | opex ex-September is the ONE surviving post-event vol cell and it IS the event sleeve's V4, so a pitch here duplicates the book | not examined: no month-turn vol mechanism distinct from the equity cell |

Two anchors that are not in `calendar.events` were considered and dismissed:
**single-name earnings** (8 liquid prints inside 10 td — ADSK and HRL today, MDT
+3, AVGO +4, CPB +5, ORCL +7, ADBE +9, KR +10) was tried and died on 2026-08-26,
and the book runs an OVS earnings blackout precisely because these windows are
unmodellable at pitch size; **quarterly Treasury refunding** closed 2026-08-10
with the mechanism falsified inside its own window.

## 2. Tape extremes by asset class

Sorted whole (`00_tape_sort.txt`), not looked up. 218 names.

- **us_large** — SPY 766.08, -1.52% off its 52w high, r5 32.1 / r21 77.0 /
  **r63 21.0**; QQQ -4.56% off, r63 11.5; DIA r63 55.6. The index sits near a high
  with a 63-day return rank in the bottom quartile, which is the rotation signature
  below. z10 is mildly negative across the board (-0.32 to -0.41). Nothing extreme
  on the index itself.
- **us_small** — IWM -2.02% off its high, r5 25.8 / r21 40.5 / r63 16.3.
  Unremarkable; the only live small-cap cell (watchlist 29) is midterm-dead and its
  entry was yesterday's close.
- **rates** — TLT is +2.40% above its 52-week LOW, IEF +1.24%, LQD +1.29%: the whole
  investment-grade complex is pinned near a one-year floor, but TLT rallied +2.22%
  over five sessions and is now outside watchlist 5's <=0.5% rung. ^TNX 4.664 =
  **98.29% of its trailing-252 high** (watchlist 21 needs 99.75). ^MOVE 69.44,
  r63 61.5, 24.5% above its low — bond vol is mid-range and NOT at a floor, so the
  2026-08-10 level-versus-rank trap is explicitly avoided here.
- **credit** — **HYG is AT its 52-week high (-0.03%)** while LQD sits 1.29% above
  its 52-week low. Credit is priced for no stress while investment grade sits at a
  floor. That joint state is checked as **C8**; the LQD-vs-HYG spread FORM of it is
  watchlist 1 and is episode-count blocked at 4 of a required 8.
- **gold_miners** — the loudest thing on the tape. GDX 21d **+38.01%**, PIT
  percentile 99.6, **full-history percentile 99.4**; NEM 21d +43.79% at a
  full-history 99.7; GLD +14.06% at 99.1. All three fell hard on 08-26
  (GDX -2.94%, NEM -2.62%, GLD -1.58%). A genuine post-parabolic state, taken as **C6**.
- **other_metals** — SLV 21d +19.13% (full-history 96.6) yet still **-41.68% off
  its 52-week high**; FCX +14.34% in five sessions, r5 98.0, at its 52-week high;
  XME 21d +18.60%. The metals complex is thrusting broadly. The BREADTH-COUNT form
  of this was pitched and killed on 2026-08-26 (116 of 121 count>=4 days are
  already GDX-rank>=95 days), so a count is not a way around C6 and is not
  re-opened.
- **energy** — the complex is pausing inside a thrust: XLE r5 19.8 / r21 78.6,
  -2.07% off its high; XOP r5 27.4 / r21 82.5; OIH r5 15.1 / r63 15.9;
  USO r5 26.6. **The OIH-minus-XOP 63d spread is at PIT percentile 0.40**
  (-18.32pp), the deepest reading in a year and deeper than the 1.19 watchlist 24
  was written at — but that entry's trigger is a RECORD (28-23, needs 32-23), which
  a new state day cannot move. The energy count at z10 >= 2.0 is **0 of 11**
  (watchlist 22).
- **dollar_fx** — DX-Y.NYB r21 **4.8** with r5 65.1; UUP r21 2.4 with r5 68.7. A
  21-day washout that has begun bouncing. The bare washout parent is midterm-parked
  (watchlist 27); the washout-THEN-bounce conditioner is a different object and is
  taken as **C9**.
- **international** — EEM **r63 2.4** with r21 82.5, the sharpest laggard-to-leader
  reversal on the tape. That is exactly the registry's closed laggard-snapback shape
  (SMH/QQQ form: flat at h=5, N=57, +0.27% at t=0.80, and the trigger over-selects
  bear tape by +29pp), so it is dismissed on the reference rather than re-run.
  FXI r5 42.9 (watchlist 9 needs <=20). EFA -0.50% off its high. EWZ r5 87.7 with
  z10 1.19, closed 2026-08-11 (`d3_ewz_decoupler.py`).
- **volatility** — VIX 15.21 at a **12.4** PIT level percentile;
  **VIX3M 17.99 at a 4.0 PIT level percentile and only +1.52% above its 52-week
  low**; the VIX/VIX3M ratio is 0.8455 at a 35.5 percentile, so this is a LEVEL
  story and not a term-structure story (which matters: the extreme-contango carry
  cell was killed on 2026-08-13). SVXY is at its 52-week high, UVXY at its
  52-week low. Three-month implied vol at a one-year floor while the repo's own
  fragility composite sits at 88.6 is the sharpest variant-perception setup on the
  board, taken as **C4**.
- **sectors** — XLF r63 98.8, XLV r63 97.6 and IBB r63 99.2 against
  **SMH r63 0.8** (the single lowest 63-day rank in the tape), XLK 18.7, XLY 17.9,
  XLC 22.2, XLU 20.2. The XLK-minus-XLV 63d spread sits at **PIT percentile 0.0**,
  its lowest reading in a year; SMH-XLV is -23.82pp at 1.2. Taken as **C5**.
  Cross-sectional 63d-rank DISPERSION is NOT extreme (spread PIT 31.5, sd PIT
  64.9), so the dispersion framing is dismissed with its number and only the pair
  is carried forward.
- **single names / consumer** — TJX is the most extreme name on the tape
  (r21 0.4, r63 0.4, z10 -2.72, 1.06% above its 52-week low, -11.4% below its
  200d), with WMT beside it (-8.50% in five sessions, z10 -2.17). NKE is **at** its
  52-week low. But the COMPLEX is not washed out: only 3 of 14 consumer names sit
  at r21 <= 10 and XRT's own r21 is 20.2, so a breadth form here would be the
  2026-08-26 redundancy trap in a new costume. The single-name form, with a
  218-name reference class built FIRST, is **C10**.
- **cross-section** — only 2 of 218 names sit within 1% of a 52-week low (NKE,
  UVXY) and 9 within 2%, five of those being rates or utility proxies. 22 names sit
  within 1% of a 52-week high. Neither tail is a breadth event, and watchlist 23's
  new-high breadth cell is separately blocked on both of its legs.

## 3. Live seasonal and cycle cells

- **Cycle: midterm (year %% 4 == 2).** This is a conditioner on everything above and
  it has come back wrong-signed in this repo six independent times: the Jackson Hole
  inversion in 4 of 6 vehicles, the NFP-TLT cell, the IWM JH-13 cell, the August
  ME-3 session at -0.366%, the bare dollar washout at -0.479%, and the
  September-midterm NFP cell at -0.676%. Any candidate whose sample is
  cycle-splittable owes the split.
- **Seasonal board (asof 2026-08-05, `board_candidates`)**: 0 A/B-grade setups
  across 2 channels. The four live entries are all `direction: context` regime
  tilts — book win 56.4% against 64.9% in midterms, OVS 55.4% against 67.6%,
  LT Trend ST OS 53.7% against 67.6%, Indices Oversold Bounce 59.0% against 64.5%.
  All four say the same thing, which is de-risk in midterm years. None is a
  tradeable ticket; they are recorded here as the cycle conditioner above rather
  than dismissed silently.
- **The board's fifth entry** (P/C depressed, conviction B) is stale: it reads asof
  2026-08-04 at a 4th-percentile total P/C, while today's live `pc_fear` state is
  **off at the 52.8th percentile**. The stale board entry is discarded in favour of
  the live state.
- **Month position**: today is **ME-2** (August's last trading day is 2026-08-31).
  Every equity form of the month-turn anchor was closed on 2026-08-26; rates is
  suspended with an explicit "NOT August"; FX closed. Commodities is C2.
- **Month of year**: late August. The August-specific duration seasonal closed
  2026-08-17 as a bond-bull fossil, and the unconditional August trading-day window
  beating the Jackson Hole anchor outright on large caps is what closed that anchor.

## 4. Watchlist — verdict on every active entry (30 active, 0 expired)

**No entry fires today.** Numbers are this morning's, from the recon scripts above.

| # | cell | verdict | today's number |
|---|---|---|---|
| 0 | nfp x rates (TLT at the 52w floor) | PASS | still midterm, parks to 2027-01. NFP is +6 td |
| 1 | credit price-state (LQD vs HYG at joint 52w extremes) | PASS | the state IS live (HYG -0.03% off its high, LQD +1.29% above its low) but the trigger is episode COUNT, still 4 declustered since 2007 against the 8 required. A live state cannot move a count |
| 2 | cpi x volatility (SVXY overnight) | PASS | CPI is 2026-09-11, +10 td, so the OVERNIGHT entry is 9 sessions away and unenterable today |
| 3 | gold price-state (GLD on a miner-led thrust) | PASS | GDX r5 **68.3** against the >=95 leg, so the trigger does not fire at all today, before the fourth condition even applies (GLD is -15.04% off its 52-week high against the >-10% rung) |
| 4 | energy price-state (XLE on a crude pop) | PASS | USO's one-day move was **+0.95%**, nowhere near the [5%,6%) band |
| 5 | rates price-state (IG complex at 52w lows) | PASS | TLT is **+2.40%** above its low against the <=0.5% rung after a +2.22% five-session rally; IEF +1.24% and LQD +1.29% are also outside theirs |
| 6 | vol x us_large (SPY on a skew spike) | PASS | ^SKEW r5 **48.0** against the >=95 leg, and the midterm block stands |
| 7 | energy x event (crude thrust fade) | PASS | USO r5 **26.6** against the >=90 leg |
| 8 | sectors (IHI medtech thrust) | PASS | IHI r21 **90.5**, not 100, and the family-wise p 0.933 reference-class blocker is unchanged |
| 9 | international (FXI break inside a thrust) | PASS | FXI r5 **42.9** against the <=20 trigger |
| 10 | rates seasonal (November TLT) | PASS | parks to a date: trading days 4-12 of November 2026 |
| 11 | rates x us_large (short SPY at a high, TLT at a low) | PASS | SPY **-1.52%** off its high against the <=0.5% rung; TLT +2.40% above its low against <=1% |
| 12 | rates x month-end flow (TLT into ME) | PASS | today IS ME-2, inside the flat ME-3..ME-7 band, but the entry's own blocker is "NOT August" and its ME-1 to ME-0 hit-rate condition is unmoved |
| 13 | volatility x us_large (SPY on a vol pop in calm tape) | PASS | VIX r21 15.5 clears the calm-tape leg, but the day's VIX move was **-1.55%** against the >=+5.0% pop rung |
| 14 | rates x fx into gold | PASS | the dollar leg fires (DX r21 4.8) but the 21-session yield change is far under the +0.20pt floor (^TNX 21d return +1.30% on a 4.66 handle) |
| 15 | sectors (one-day XLV-XLK rotation gap) | PASS | today's one-day gap is **-1.61pp** (XLV -1.00, XLK +0.61) against the >=+3.0pp rung. NOTE: the 63-DAY form of the same pair sits at a PIT floor and is taken up as C5 — a different object at a different horizon, and the write-up must say so |
| 16 | rates x dollar_fx (short the dollar) | PASS | ^TNX r21 **53.2** against the >=65 leg |
| 17 | energy x jackson_hole (USO at JH-6) | PASS | the JH-6 anchor was 2026-08-20 and today is JH-1. XLE at -2.07% off its high also sits inside the forbidden 5% band |
| 18 | rates price-state (short TLT after a thrust) | PASS | TLT's one-day move was **-0.20%** against the >=+1.5% thrust rung, and it has drifted to +2.40% above its low against "within 4%" |
| 19 | financials (short KRE against XLF) | PASS | KRE r5 **36.5**, so the breadth washout has resolved; the trigger is an ex-crisis cost threshold no new episode can move in any case |
| 20 | jackson_hole x credit | PASS | JH-5 was 2026-08-21, six sessions gone, and the anchor is closed on credit |
| 21 | rates curve (IEF vs 0.523 TLT at a yield high) | PASS, moved closer | ^TNX at **98.29%** of its trailing-252 high, up from 97.77% yesterday, against the 99.75 rung. The cost turn-on is unmoved |
| 22 | energy breadth count (z10 >= 2.0) | PASS | the count is **0 of 11** |
| 23 | us_large breadth x index distance | PASS | SPY **-1.52%** off its high against the >2.0% requirement, and raw-21d fragility **71.4** against the <=50 requirement |
| 24 | energy dispersion (long OIH at an OIH-XOP 63d extreme) | PASS, state deeper than ever | the spread is **-18.32pp at PIT percentile 0.40**, deeper than the 1.19 the entry was written at. The trigger is a RECORD (28-23, needs 32-23) and a state day cannot move a record. Filed as the closest thing on the board to a live trigger |
| 25 | sectors (washout into a 52w high, family form) | PASS | no sector sits at a 5-day rank <= 5; the lowest is XLE at 19.8. The dial maximum on any historical episode is 68.6 against today's 88.6 |
| 26 | utilities x rates (XLU washout with the long end hit) | PASS | XLU r21 **8.7** (the rung is <=5) and TLT RALLIED — r5 63.9, r21 43.7 — so the rates leg is the wrong sign for a third session |
| 27 | dollar_fx (bare dollar washout) | PASS | parks to the first trigger in a non-midterm year, 2027 at the earliest. The state is live (DX r21 4.8), which is why the BOUNCE-conditioned variant is checked as C9 rather than the parent |
| 28 | credit x us_large (HYG at a high, SPY not) | PASS | SPY **-1.52%** off its high against the >=2.0% requirement, and dial ma10(63d) **88.6** against the <50 requirement, in a cell whose observed dial maximum is 68.0 |
| 29 | us_small (the ME-3 to ME-2 session) | PASS | midterm-blocked (-0.366% on a 3-3 record), and the entry was yesterday's close in any case |

## 5. Axis scoreboard read

`scoreboard.lifetime` carries **4 graded ideas** (avg +0.372R, 4 for 4). The
per-axis split (event_fingerprint 2 at +0.622R, interaction_cell 1 at +0.146R,
relative_value 1 at +0.099R) is a handful of observations and is not read as a
signal about which axis earns. Recorded and moved past, per the skill's own rule.

## 6. Candidates selected from this map

Ten. Four axes minimum: satisfied at six. Four asset classes minimum: satisfied at
ten. At least one event-anchored and one price-state-anchored: satisfied.

| id | candidate | axis | class | source cell |
|---|---|---|---|---|
| C1 | Long SVXY (short vol) from the JH-1 close through the speech and +3 | event_fingerprint | volatility | JH x volatility, the eighth and last unswept JH class |
| C2 | The month turn on the metals and commodity complex, ME-2 entry | flow_mechanics | other_metals | ME anchor, the one class never swept |
| C3 | A 10 td hold entered today that contains BOTH the Sep 10 PPI and the Sep 11 CPI | event_fingerprint | us_large | the anchor the registry named as next-reachable |
| C4 | VIX3M level at a trailing-252 floor, forward SPY and SVXY | interaction_cell | volatility | VIX3M PIT 4.0, +1.52% above its 52-week low, dial 88.6 |
| C5 | XLK and SMH minus XLV 63d spread at a PIT floor, long the laggard against the leader | relative_value | sectors | XLK-XLV PIT 0.0, SMH-XLV 1.2 |
| C6 | GDX after a 99th-percentile 21d thrust and a >=2% down day, both directions | inversion | gold_miners | GDX 21d +38.01%, full-history percentile 99.4 |
| C7 | SMH at a 63d-rank floor while its 252d return is top-decile, against a 218-name reference class | historical_analogue | us_large | SMH r63 0.8, 252d +89.68%, -16.91% off its high |
| C8 | IEF and LQD within 1.5% of 52-week lows while HYG is AT a 52-week high | interaction_cell | rates + credit | a pure-rates repricing with zero credit stress |
| C9 | DX 21d rank <= 5 THEN a 5d rank >= 60, the washout confirmed by a bounce | price-state | dollar_fx | DX r21 4.8 / r5 65.1 |
| C10 | The TJX-class washout (z10 <= -2, r21 <= 2) judged against a 218-name reference class built first | historical_analogue | consumer / single names | TJX r21 0.4, z10 -2.72, 1.06% above its 52-week low |

Negative-registry collisions each candidate must address, checked before dispatch:
C1 against "post-event vol cells swept and empty" (opex ex-September is the only
survivor and it IS the book's V4); C4 against "fragility dial as a directional
signal" and the SVXY beta-translation trap; C5 against "sector-vs-index pairs on a
crowding or leadership trigger" and "laggard-snapback continuation (SMH/QQQ form)"
— C5 is sector-versus-SECTOR at a 63-day spread rank and it owes both legs priced
separately; C6 against the closed GDX thrust cell and the 2026-08-26 metals count;
C7 against that same laggard-snapback entry, which is the nearest thing in the
registry and which C7 must distinguish itself from on the CONDITIONER (a 63d-rank
floor inside a top-decile 252d, not a snapback already in progress); C10 against
the IHI reference-class kill, which is exactly why the reference class is built
before the cell rather than after.
