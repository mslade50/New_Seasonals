# Surface map — 2026-09-01 (Tuesday, first session of September, midterm year)

Recon scripts backing this map: `00_recon.py` (every watchlist trigger read on
today's tape), `00b_anchor_map.py` (anchor reachability plus the count-first
pass on the two anchors that are new today), `00c_state_recon.py` (count-first
on the interaction cells today's tape suggests, plus two data-integrity
checks).

**CORRECTIONS applied after checker A's round 2, and both were mine.**
(1) `00_recon.py` and `00c_state_recon.py` converted ^TNX point changes to
basis points by multiplying by 10 rather than 100, so every yield change in
the first draft of this map was **10x too low**. The true readings are
**+1.3 bp over 21 sessions, +28.3 bp over 63 and +55.1 bp over 252**, on a
252-session range width of 80.5 bp. Both scripts are patched and rerun; the
sections below carry the corrected numbers. The framing the wrong units
produced — "a one-year high at the top of a year of nothing" — was false and
is withdrawn: the year did +55 bp. This changes nothing about which trigger
fired, and it is the whole reason A3 was selected, so A3's premise was
damaged before it was ever measured.
(2) Section 2 originally said ^MOVE popped "while ^VIX fell". **^VIX ROSE
+3.40% on 2026-08-31**, which `00_recon.py` printed correctly and the map
copied wrong. Corrected below. A4's live divergence therefore did not exist.

## Standing state

- Prices fresh through **2026-08-31**; pipeline 7/7 green. One stale tape
  ticker (LEG), not used anywhere below.
- **Fragility dial ma10-63d 87.1** (raw 63d 88.0, raw 21d 66.2), up from 50.9
  twenty-one sessions ago. Exposure leg 0.0x, trend sleeve in CASH.
  This matters for candidate selection rather than as an idea: the dial is
  closed as a directional signal (registry), but a large share of the parked
  inventory is blocked precisely because its historical trigger-day dial
  maximum is 66-69 against 87 today. **Most of the watchlist is out of sample
  on this tape**, which argues for lower grades, not for a dial trade.
- P/C fear OFF (49th pctile). Signals on: Defensive Leadership, Low Absorption
  Ratio. Dispersion at the 87th pctile (component RV 33.0% against SPY RV
  10.3%).
- Book has 3 staged signals: OLV UNH long, OVS SLB short (liquid), OVS NOW
  short (overflow). SLB is the tape's second-best 5-day mover and sits AT a
  52-week high, so anything long energy today must disclose the overlap with a
  live staged short in the same complex.

## 1. Every live calendar event x every asset class

Eight events in the [-5,+15] td window. Two of them are **new inside the 10 td
cap today**, which the registry has been predicting for six consecutive
sessions.

| event | date | td | status |
|---|---|---|---|
| jackson_hole | 2026-08-28 | -2 | CLOSED. Eight classes pre-speech, ten post (2026-08-27/28). One live cell parked to a non-midterm year. |
| nfp | 2026-09-04 | +3 | CLOSED on the ladder (2026-08-26): plateau on four vehicles, September prints -0.038%, September-in-a-midterm -0.676% on 3-3. |
| ppi | 2026-09-10 | +6 | CLOSED (2026-08-27): the PPI/CPI containment gate is on 39.7% of days and agrees with "CPI in the hold" on 92.3%. |
| cpi | 2026-09-11 | +7 | CLOSED, same entry. |
| **fomc_decision** | **2026-09-16** | **+10** | **NEW TODAY, at the cap.** A MOC entry today held 10 td exits on the decision close. |
| **vix_expiry** | **2026-09-16** | **+10** | **NEW TODAY.** Coincides with the decision; 42 of 212 historical decisions do. |
| opex | 2026-09-18 | +12 | Beyond the 10 td cap. |
| quad_witching | 2026-09-18 | +12 | Beyond the cap. |

### FOMC x ten asset classes (the cross the 2026-08-07 lesson demands)

The repo's own note says the pre-FOMC drift is "spoken for by the event
sleeve's T1/T2". **Both are OFF today**: T1 is non-midterm only, and T2 needs
SPY's 21-day rank under 50 against a live 67.5. So the sleeve places nothing
into this decision, and the pre-FOMC window has never been swept on any class
but SPY in this repo. Coarse map (`00b`, entry MOC at decision-10td, hold to
the decision close, 204-212 anchors; MAP ONLY, anything selected here is
charged for the 15x2 grid):

| class | vehicle | full-sample edge vs own drift | midterm edge | verdict |
|---|---|---|---|---|
| us_large | SPY | +0.064pp | **-0.820pp** (N=53) | CHECK — the documented midterm inversion, at a window the sleeve does not trade |
| us_small | IWM | +0.149pp | -0.497pp (N=53) | CHECK, inside the family sweep |
| rates | TLT | +0.130pp | -0.289pp (N=49) | CHECK, and crossed with today's yield state |
| rates belly | IEF | +0.085pp | -0.206pp | CHECK, inside the family sweep |
| credit | HYG | +0.078pp | -0.344pp (N=37) | CHECK, inside the family sweep |
| gold | GLD | -0.073pp | -0.292pp | CHECK, inside the family sweep |
| miners | GDX | -0.290pp | -0.908pp | dismissed separately: gold-miner ideas are repetition-blocked today (section 4) |
| metals | SLV | +0.024pp | +0.114pp | CHECK, inside the family sweep |
| **energy** | **USO / XLE** | **-0.457pp / +0.168pp** | **+0.820pp / +0.364pp** | **CHECK — the only class that inverts the OTHER way.** Crude is the worst class into an FOMC on the full sample and the best in midterms |
| dollar / fx | UUP | -0.256pp | -0.003pp | CHECK, inside the family sweep. UUP is separately cost-dead as a vehicle (registry); DX futures are the expression if anything survives |
| intl dev | EFA | +0.355pp | -0.295pp | CHECK, inside the family sweep |
| intl em | EEM | +0.297pp | -0.151pp | CHECK, inside the family sweep |
| volatility | SVXY | +1.284pp | **-2.484pp** (N=29) | CHECK on the coincidence form only; SVXY as a plain pre-FOMC leg is already closed in the registry |
| tech | XLK | -0.097pp | -1.339pp | CHECK, inside the family sweep |

The single strongest read on the map is that the **midterm inversion is
near-universal and energy is the exception**, which is an `inversion`-axis
object rather than a cell of a scan. It still owes the full family charge.

### VIX expiry x classes

42 of 212 decisions land ON a VIX expiry, and 2026-09-16 is one. Never swept
in this repo as a coincidence: the registry closed post-VIX-expiry vol cells
and VIX-expiry-week drift, both on the expiry ALONE. A settle that is also a
policy print is a different object. CHECK (vol complex plus SPY).

### NFP x classes, and the holiday interaction

NFP is +3 td and lands on the Friday before Labor Day. Count-first (`00c`):
**14 September NFPs since 2000 are Monday-holiday eves**, out of 17 such NFPs
in total. Both parents are already dead — the NFP run-up ladder is closed on
four vehicles and the pre-holiday drift is a documented pre-2013 fossil — so
the interaction is a join of two corpses at N=14. **Dismissed on count and on
both parents, not examined further.** The NFP x rates cell is separately
midterm-parked to 2027-01 (watchlist 1).

### PPI / CPI x classes

Closed 2026-08-27 as a containment object on equities, rates and gold, with
the September sub-cell shown to be the equity month turn in disguise (7 of its
9 dates are month-turn dates). **Not re-examined.**

### Opex / quad witching

+12 td, beyond the horizon cap. September post-opex is separately the event
sleeve's T3 territory and the registry records that September INVERTS the
post-opex vol crush. **Out of range, not examined.**

## 2. Tape extremes by asset class

Sorted the full 218-name tape (`scratch/_tape_sort_0901.py`).

- **us_large**: SPY -1.39% off its 52w high, r5 52.4, r63 15.5, z10 -0.35.
  QQQ r63 6.7. Nothing extreme. The "index near a high with a bottom-quartile
  63-day rank" form was killed 2026-08-28.
- **us_small**: IWM z10 -1.16, r5 20.6, -3.66% off its high, lagging SPY.
  CLOSED twice this week: the ME-0 IWM/SPY pair (08-31) and the HYG-high x
  IWM-depth substitution (08-31, "same kill, different ticker").
- **rates**: **^TNX closed AT its trailing-252 maximum (100.00% under BOTH
  percentile conventions)** — the only clean level extreme in the tape. But
  the 21-session yield change is **+1.3 bp** while the 252-session change is
  **+55.1 bp** on a 80.5 bp range (units corrected — see the note at the top;
  the "year of nothing" framing was an artifact of the bad conversion). TLT +1.44%
  above its 252-low, IEF +0.61%, LQD +0.75%. **CHECK** — this is both the
  armed watchlist trigger and a never-measured object in its own right, since
  every prior yield-high cell in this repo carried a thrust.
- **credit**: HYG -0.14% off its 52w high, LQD +0.75% above its 252-low. The
  IG-at-lows / HY-at-highs conjunction is watchlist 26 at an episode count of
  ONE, and the HYG-high forms are closed on SPY (08-26) and IWM (08-31).
  Dismissed.
- **gold**: GLD r5 8.7 after +9.93% in 21 days; GDX r21 98.0 with r5 19.8;
  NEM r21 97.6, r5 21.0. A pullback inside a thrust. **Repetition-blocked**
  (section 4).
- **other metals**: **SLV -43.06% below its 52-week high while +69.28% over
  252 days.** Verified real (`00c`): silver printed 105.60 on 2026-01-28 and
  trades 60.13. A post-parabolic drawdown inside a still-huge year is a state
  this repo has never examined from the long side. **CHECK.**
  XME r21 87.3 / r63 9.9 is the nearest holder of the pooled-laggard state and
  it fails the r21>=90 leg (watchlist 29).
- **energy**: the loudest thing on the tape. XLE closed **AT** a 52-week high
  (+2.04% on a session SPY fell -0.30%), XOP -0.31% off, SLB and VLO at highs,
  USO +3.08% on the day, OIH r5 81.3, whole complex up 1.6-2.6%. **CHECK**
  (the divergence at a high, and crossed with the FOMC anchor). Note the
  narrow-thrust watchlist entry does NOT fire: the count of the 11-name
  complex at z10 >= 2.0 is **0**.
- **dollar / fx**: UUP r5 70.2, r21 38.5, -1.68% off its high; DX-Y r21 41.7.
  Mid-range, no extreme. The dollar washout cells are midterm-parked and the
  confirming inversion was closed 08-31. Nothing live.
- **international**: EEM r5 61.1 / r63 0.4, EFA r5 29.4, FXI -13.72% off its
  high with r5 42.5 (fails watchlist 10's <=20 leg), EWZ z10 +1.49. EEM's r63
  of 0.4 with r21 59.1 is a laggard shape, but it fails watchlist 29's r21>=90
  leg and the pooled parent is dead on its own terms. Dismissed.
- **volatility**: ^VIX 14.92, -51.95% off its 52w high, r21 34.5. ^VIX3M r21
  21.8 (its LEVEL-floor form was closed 08-27). SVXY at a fresh 52-week high
  (closed 08-28). ^SKEW r21 88.5 / r63 81.3 with VIX at 15 — the ratio form
  was closed 08-31 at a day-level overlap of 1.000 and the SKEW-alone form is
  midterm-blocked (watchlist 7, and its r5 is 68.7 against a >=95 rung).
  ^MOVE **+6.13% on the day** — but ^VIX **also rose, +3.40%** (the first
  draft of this map said it fell, which was wrong). So the divergence A4 was
  selected on does not exist on today's tape. Left in the candidate list
  because it was already dispatched to a checker; the bare bond-vol spike is
  closed twice (08-18) and only survives as an interaction anyway.
- **sectors**: XLK r5 80.6 against XLV 8.7, XLP 7.1, XLRE 6.0, XLI 11.1,
  XLU 13.5 — **five of eleven sectors in the bottom 15% of their 5-day rank
  at once.** Count-first (`00c`): 239 of 1805 sessions carry a count this high
  or higher, so the state is at the 13th percentile of rarity, not the 1st.
  **CHECK cheaply** as a count object; the single-sector and pair forms are
  closed (08-25 XLI, 08-27 the 132-pair family, 08-28 the defensive complex).
  XLU r21 4.8 fires watchlist 23's utilities leg but its rates leg fails
  (TLT r21 68.3 against a <25 rung) — see section 4.
- **single names**: EIX -27.02% and PCG -26.73% in five sessions (a two-member
  idiosyncratic utility shock, closed 08-31). CRM +23.19% in five days and
  +39.95% in 21 (the largest-winner-at-the-turn form was closed 08-31).
  ORCL -54.22% off its high. Single-name cells are closed on a 205-name
  reference class at family-wise p 1.0000 (08-27). Dismissed as a class.
- **natgas**: UNG z10 +1.34, r5 76.2, -37.63% off its 52-week high, +3.84%
  over five sessions, entering the September shoulder season. The registry
  kills UNG LONG AT A 52-WEEK LOW on structural bleed (-0.90% per 10 td),
  which is not this state. **CHECK** with the bleed charged explicitly.

## 3. Live seasonal and cycle cells

- **Midterm year (year%4==2)** conditions everything above and is the reason
  the FOMC family sweep is run midterm-split rather than pooled.
- The seasonal board carries **0 A+B-grade setups**. Its five candidates are
  all regime-context "midterm de-risk" lines (book win 56.4% against 64.9%,
  OVS 55.4% against 67.6%, LT Trend ST OS 53.7% against 67.6%, Indices
  Oversold Bounce 59.0% against 64.5%) plus a P/C complacency read that is
  stale (2026-08-04) and contradicted by today's live 49th-percentile reading.
  No tradeable seasonal cell. Dismissed.
- **First trading day of September**: measured in `00b`. SPY from the first
  September session to +10 td is +0.421% on N=26 at a 61.5% hit against an
  all-days +0.376%. **+0.045pp of edge on 26 observations. Dead, dismissed.**
- Month-end is now behind us and the anchor is closed on six forms across five
  asset classes (08-31). Not re-examined.
- September month-of-year on duration was closed 08-31 (TLT's third-worst
  month, cost 0.53x at h=5) and on the dollar the same day.

## 4. Watchlist verdicts (all 32 active entries)

Read on today's tape by `00_recon.py`. **Exactly one fires.**

| # | entry | today's reading | verdict |
|---|---|---|---|
| 1 | TLT from the NFP close, long end at its floor | midterm; parks to 2027-01 | PASS |
| 2 | LQD against HYG at joint 52w extremes | trigger is an episode COUNT (4 of 8 needed) | PASS |
| 3 | SVXY overnight into CPI | CPI is +7 td, the overnight entry is 6 sessions away | PASS |
| 4 | GLD on a miner-led thrust | GDX r5 **19.8** against a >=95 rung | PASS |
| 5 | XLE on a crude 1d thrust in [5%,6%) | USO 1d **+3.08%**, outside the band | PASS |
| 6 | TLT with the IG complex pinned at 52w lows | TLT **+1.44%** above its low against a <=0.5% rung | PASS |
| 7 | SPY on a skew spike alone | ^SKEW r5 **68.7** against >=95; midterm block also stands | PASS |
| 8 | Fade a crude thrust out of a deep base | USO r5 53.6 against a >=90 rung | PASS |
| 9 | IHI medical-device thrust | IHI r21 **87.7**, not 100 | PASS |
| 10 | FXI 5d break inside an intact thrust | FXI r5 **42.5** against <=20 | PASS |
| 11 | TLT on the November month-position effect | parks to a date: November trading days 4-12 | PASS |
| 12 | Short SPY at a 52w high with TLT at a 52w low | SPY **-1.39%** off, TLT **+1.44%** above its low | PASS |
| 13 | SPY on a vol pop inside a calm tape | VIX r21 **34.5** (needs <=25), VIX 1d **+3.40%** (needs >=5%) | PASS |
| 14 | Gold on an unconfirmed rate rise, both dials at force | 21-session yield change **+0.1 bp** against a +20 bp floor | PASS |
| 15 | XLK against XLV after a 1-day rotation gap | gap **-0.80pp** against >=+3.0pp | PASS |
| 16 | Short the dollar on an unconfirmed rate rise | ^TNX r21 **40.5** against >=65 | PASS |
| 17 | Short TLT after a big up day from the low zone | TLT 1d **-0.43%** against >=+1.5% | PASS |
| 18 | Short KRE against XLF on a breadth washout | KRE r5 **22.6**; arm is an ex-crisis cost bar no new episode can move | PASS |
| 19 | **Duration-neutral flattener, long IEF / short 0.523 TLT, ^TNX at its 52w high** | **^TNX at 100.00% of its trailing-252 max under BOTH conventions (rung 99.75).** The second blocker — episodes whose hold spans Jackson Hole are 0-for-6 — **now clears**, because JH was 2026-08-28 and a hold starting today does not span it | **FIRES. Sole live arm. CHECK.** Remaining blocker is the stated cost bar |
| 20 | Narrow energy thrust cluster (2-3 names at z10>=2) | count is **0 of 11** | PASS |
| 21 | Cross-sectional new-high breadth | SPY -1.39% off (needs >2.0%), raw-21d fragility 66.2 (needs <=50) | PASS |
| 22 | Sector washout into a 52w high, family form | lowest sector r5 is XLRE at **6.0** against a <=5 rung; dial 87.1 against a cell maximum of 68.6 | PASS |
| 23 | Utilities washout with the long end hit ALONGSIDE | XLU r21 **4.8** fires, but TLT r21 **68.3** against a <25 rung — the rates leg is the wrong sign for a fourth straight session | PASS |
| 24 | Bare dollar washout, long the dollar | parks to a non-midterm year | PASS |
| 25 | HY at a 52w high while the index has not | SPY -1.39% off (needs >=2.0%), dial 87.1 against a <50 requirement | PASS |
| 26 | Rates repricing with zero credit stress | episode count of ONE, and it is the live one | PASS |
| 27 | SMH at a 63d rank floor in a top-decile year | SMH r63 **0.4**, r252 +86.8%: the state is live but the entry's own round 2 killed it on a 29-name reference class and it is parked on two numbers, neither of which has moved | PASS |
| 28 | IEF one session out of the Jackson Hole close | midterm; parks to 2027-09 | PASS |
| 29 | Pooled laggard STILL FALLING (r21>=90, r63<=10, r5<15) | **no holder** in the 29-name pool; nearest is XME at r21 87.3 / r63 9.9 | PASS |
| 30 | Short silver after a complex break | SLV 1d **+0.18%**, GLD -0.11%, GDX -1.14% — no complex break, and the arm additionally requires a lag-profile fix | PASS |
| 31 | Long duration, yield high x bond vol MID-RANGE [40,50) | ^MOVE level pctile **66.3** | PASS |
| 32 | December small-cap month-end overnight | parks to a date | PASS |

Pruning: nothing expired, nothing fired except 19, so the file is rewritten
unchanged apart from today's verdict notes.

## 5. Scoreboard read (required before selection)

Five ideas pitched lifetime, four graded: avg +0.372R, 4-for-4, fill rate 80%.
By axis: event_fingerprint 2 graded at +0.622R, interaction_cell 1 at +0.146R,
relative_value 1 at +0.099R, inversion 1 ungraded. By grade: B 3 at +0.448R,
C 1 at +0.146R. **The graded count is a handful and no axis has bled**, so the
splits carry no information yet and nothing is being tilted on them. Recorded
so the read is on the record.

## 6. Candidates selected (12, across 9 asset classes, 6 novelty axes)

Event-anchored: A2, A4, B1, B2, B3, B4. Price-state-anchored: A1, A3, C1, C2,
C3, C4. Asset classes touched: rates, energy, us_large, volatility, metals,
sectors, international, gold, us_small.

**A — rates and the armed arm**
- **A1** `relative_value` / `instrument_translation` — the parked
  duration-neutral flattener, long IEF against short 0.523 TLT, MOC entry with
  ^TNX at its trailing-252 maximum. Sole live watchlist arm. The whole
  question is the stated cost bar plus whether a level high on a +5.5 bp year
  is the same conditioning state the cell was measured in.
- **A2** `event_fingerprint` — duration into the September FOMC with the
  ten-year pinned at a one-year high. Count-first already says the tight
  interaction is **N=6 of 204 anchors** (two midterm), so the check is the
  loosened magnitude form or nothing.
- **A3** `interaction_cell` — the never-measured object: a trailing-252 yield
  MAXIMUM reached on a 21-session change under 5 bp. NOTE: selected on the
  10x-low units. On corrected units the 21-session change is +1.3 bp, which
  still clears a 5 bp cut, but the "no thrust over the year" half of the
  premise is gone.
- **A4** `interaction_cell` — ^MOVE up 6.13% on a session ^VIX ALSO rose
  +3.40%. Selected on a misread of the map's own recon; the divergence is not
  live. The bare bond-vol spike is closed twice.

**B — the FOMC family, new anchor**
- **B1** `event_fingerprint` — the full 15-class x 2-sign pre-FOMC midterm
  sweep with a family multiplicity charge. Reports a survivor or closes the
  anchor cross-asset in one pass.
- **B2** `inversion` — long crude and energy equity into a midterm-year FOMC:
  the one class whose sign flips POSITIVE where equities flip negative.
- **B3** `event_fingerprint` — short SPY at FOMC-10td in a midterm year, the
  exact window and gate state the event sleeve's T2 declines to trade. The
  overlap with T2 must be named.
- **B4** `flow_mechanics` — the 42 decisions that land ON a VIX expiry, and
  what a settle that is also a policy print does to the vol complex.

**C — price state, non-rates**
- **C1** `historical_analogue` — long silver deep inside a post-parabolic
  drawdown while still up 69% on the year.
- **C2** `interaction_cell` — energy closing AT a fresh 52-week high on a
  session the index fell, crossed with the new FOMC anchor.
- **C3** `interaction_cell` — the count of sectors simultaneously in the
  bottom 15% of their 5-day rank, as a breadth object rather than a call on
  any one sector. Cheap check; the count is only at the 13th percentile of
  rarity.
- **C4** `interaction_cell` — natural gas thrusting into the September
  shoulder season, with the structural bleed charged explicitly.

Dismissed with reasons above and not spent a check on: NFP x anything (ladder
closed, September-midterm -0.676%), the NFP x Labor-Day interaction (N=14,
both parents dead), PPI/CPI containment (closed), opex and quad witching (out
of range), the first-September-session cell (+0.045pp on N=26), gold and
miners (repetition-blocked, fingerprint 749b2073856902b3 pitched 2026-08-27
and inside the 10 td window), single names (205-name reference class at
family-wise p 1.0000), the dollar (no extreme, cells midterm-parked), credit
(closed twice this week), IWM (closed twice this week), SVXY at its high
(closed 08-28), the SKEW forms (closed 08-12 and 08-31), and the seasonal
board (0 A+B setups).
