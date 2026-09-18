# Surface map — 2026-09-10 (Thursday, midterm year)

Stage B1. Written BEFORE any candidate was generated. Every cell below carries a
verdict; a dismissal names its reason. Readings are the 2026-09-09 close (the
freshest bar) unless stated. Today 2026-09-10 is a live session and the entry
convention is lag=1, so a signal read on the 09-09 close is entered MOC today.

Pipeline check: 4 of 4 receipt components green, prices bar 2026-09-09, dial
as-of 2026-09-09, P/C data 2026-09-09 at 1 business day. One tape warning: LEG
is stale. LEG is not used anywhere below.

**Data correction found while reconciling this map (`00c_recon.py`), and it
changed a framing before it reached a candidate.** The tape reports `^VIX` up
**+8.29% over five sessions**. On SPY's own calendar the five-day change is
**+0.73%** — essentially unchanged. The tape differenced `^VIX` on `^VIX`'s
calendar, which carries a 2026-09-07 Labor Day bar the equity complex lacks;
that is the exact defect the 2026-09-09 registry entry names, reproducing one
session after it was filed. There is NO week-long volatility bid on this tape,
and nothing below is allowed to assume one. The one-day move (+4.71%) is
unaffected because 09-08 and 09-09 are both equity sessions.

## 0. What is distinctive about today, stated before the search

Three things, and every one of them is a level rather than a change:

1. **The whole investment-grade complex sits at 52-week lows while `^TNX` prints
   a 252-day yield HIGH.** IEF +0.06% above its 52-week low, LQD +0.09%, TLT
   +0.85%; `^TNX` 4.837 at dist 0.00% from its own 252-day maximum, z10 +1.76.
   And credit is untouched: HYG is 0.63% from a 52-week HIGH.
2. **The commodity complex prints 52-week highs into a PPI/CPI pair.** DBC at
   dist 0.00%, USO +19.10% over 21 days at z10 +2.78, XLE and XOP both at 0.00%.
3. **The `^VIX` 21-day range sits at a compression extreme** while the forward
   six sessions carry PPI, CPI, an FOMC decision, a VIX settlement, opex and
   quad witching — the densest calendar of the quarter.
   **AMENDED after the arm pass, and the amendment matters.** Two different
   statistics are in play and they disagree. The book's risk dashboard reports
   the 21-day range at the **2nd percentile** (it fires below the 15th). The
   watchlist's own definition — the RELATIVE range, (21d max − 21d min) / 21d
   mean, trailing-252 rank — reads **9.92** on the same bar, having run
   3.57 → 3.97 → 4.37 → 4.37 → 9.92. Anything conditioned on compression must
   name WHICH statistic it is using, because a cell built on the dashboard's
   number and a cell built on the watchlist's number are not the same cell.
   The consequence: watchlist entry #33's band is (5, 15] and 9.92 is INSIDE it,
   so that entry armed for the first time since it was parked. See section 4.

The fragility dial's 10-day MA of the 63-day score reads 87.7, roughly the 99th
percentile of its own 2016+ series. That is a CONTEXT reading about the tape's
own fragility. It is not a portfolio observation, not a sizing instruction, and
this repo's own record is that the dial's level has no directional content
(aggregate point-in-time t = -0.23, book-wide throttle registry-dead).

Standing structural note, and it changes what is eligible today: the owner's
2026-09-08 decision makes Daily Pitch judge STANDALONE quality only. Book,
staging and sleeve overlap are no longer rejection reasons. That VOIDS the
overlap leg of any parked arm — see the pooled-sector-floor entry at section 4.

## 1. Every live calendar event crossed with every asset class

Seven events sit inside the [-5, +15] td window. `election` (td +38) is outside
it and is not enumerated. Sixty-plus cells; the table records which deserve a
check today and why the rest do not.

| event | td | verdict across the ten classes |
|---|---|---|
| nfp 2026-09-04 | -3 | **PASSED, all ten classes.** The anchor is three sessions behind us and the post-NFP window is exhausted. Swept forward on 2026-09-03 and 2026-09-04 across equity, rates, metals, credit, energy and FX; every cell killed. Post-NFP duration conditioned on the prior print's surprise is parked (watchlist, band arm). No live cell. |
| ppi 2026-09-10 | **0** | **US large: PASS.** Both orderings of a one-session-apart PPI/CPI pair are measured and the PPI-then-CPI side is the negative one: -0.1135% on N=127, the gate worth -0.115pp where its complement is +0.024pp, placebo rank 8 of 11 (SPY) and 9 of 11 (IWM). **Rates: PASS.** The pair gate subtracts on duration too, and September is the worst and wrong-signed cell in that table at TLT -1.120pp on a 25.0% hit over 12 observations. **Commodities: PASS**, killed 2026-09-08, the cell does not exist outside 2007/2008/2021/2022 which hold 26 of 53 episodes and more than 100% of the total. **Energy equity: PASS**, XOP's edge is positive at 0 of 10 horizons and September-and-midterm episodes number 0 of 84. **Vol: PASS**, "a print on the very next session" is set-identical to runway == 1, which is the measured dead half (SVXY +0.185%, t 0.66). **Credit, gold, dollar, international, small caps: PASS**, release-anchored versions of all five were swept 09-08 and 09-09; the standing gap-share finding is that duration SELLS OFF on the print (-28% of the h=3 hold on IEF, -55% on TLT) and makes it back afterwards, which is the opposite of a print-anchored long. |
| cpi 2026-09-11 | +1 | **All ten classes PASS as a standalone anchor**, for the same reason: with PPI today, every CPI-anchored hold sits inside the pair whose gate is measured negative on equity and on duration. The one CPI cell with any life is the parked overnight SVXY rung (MOC the eve to MOO the print), and it is parked on concentration. **Retained as a CONDITIONER rather than an anchor** — every candidate below reports its behaviour with a CPI inside the hold. |
| fomc_decision 2026-09-16 | +4 | **US large: PASS**, and this is the closest call of the day. Today is exactly the run-in anchor and 2026 is midterm, and the 2026-09-09 work found the collision (a VIX settlement landing ON the decision date) AMPLIFIES the midterm inversion: all-FOMC midterm gap -1.084pp against the collision's -3.257pp, run-in midterm -1.753% on a 4-6 record. It PASSES on its own numbers, not on overlap: n=10 with a 40% hit and a cell that reduces to the pre-FOMC drift's midterm split, which was itself swept across fifteen asset classes on 2026-09-01 and came back empty. **Vol: PASS**, the settle-session SVXY rung is parked on an era break that lands on SVXY's Feb-2018 re-levering, and the settle session is 09-16 rather than today. **Rates, credit, gold, energy, dollar, international, small caps: PASS**, all inside the 2026-09-01 fifteen-class pre-FOMC sweep. |
| vix_expiry 2026-09-16 | +4 | **Vol: PASS**, see above; the collision belongs to the FOMC (placebo rank 1 of 11 on both the run-in and the settle). **Other nine classes: not examined as a standalone anchor**, and the reason is that a VIX settlement has no measured cross-asset content in this repo and the one session it governs is 09-16, outside a hold entered today at h <= 3. Enumerated so the dismissal is visible. |
| opex 2026-09-18 | +6 | **Small caps: PASS**, "Long IWM into September quad witching" killed 2026-09-07 and "The small-cap laggard into September quad witching" killed 2026-09-04. **US large: CHECKED**, not as an opex cell but as the far edge of the event-density window — see candidate C1. **Rates, credit, commodities, metals, energy, dollar, international, vol: not examined as opex cells.** Reason: an opex anchor six sessions out cannot be entered today except by holding through PPI, CPI, the FOMC and the settlement, which makes it an event-density trade rather than an opex trade, and that is exactly what C1 tests. |
| quad_witching 2026-09-18 | +6 | Same disposition as opex; they are the same date. Counted separately only in the density count at C1. |

**Coverage note.** Six of the seven live events are dismissed on MEASURED
grounds from the last nine mornings rather than on judgement, which is the
honest state of this surface: the print-and-FOMC lane has been swept four times
in eight sessions. The one thing nobody has measured is the events as a SET,
which is why C1 exists.

## 2. Every tape extreme, by asset class

218-name tape, sorted on rank_5d, rank_21d, rank_63d, z10, distance from the
52-week high and low, distance from the 200-day SMA, 21d/63d/252d return, ATR%,
vol-vs-63d and 21-day realized vol (`00_tape_sort.py`, `00b_facts.py`).

**US large.** SPY -1.99% from its high, r21 15.9, z10 -0.28, 21d realized vol
8.2% annualised. QQQ -3.89%, r21 31.3. Breadth is the split: 61.9% of the tape
is above its 200-day SMA but only 27.1% has a 21-day rank above 50.
**Verdict: CHECK** as the vehicle for C1, not as a price state of its own.

**US small.** IWM -4.74% from its high, r21 11.5, r63 13.1, z10 -1.14 — the
weakest major index on both windows. **Verdict: PASS.** The small-cap laggard
was pitched into quad witching on 09-04 and into the month turn on 08-31 and
killed both times; nothing about the reading has changed except that it got
slightly worse.

**Rates.** `^TNX` 4.837 AT its 252-day high, z10 +1.76, r63 85.7. TLT +0.85%
above its 52-week low, IEF +0.06%, LQD +0.09%. `^MOVE` 76.74 at a 21-day rank of
60.3 and a level percentile of roughly 63.
**Verdict: CHECK, but only as a CONDITIONER.** The duration LONG at this exact
state has been pitched and killed on 08-31, 09-01, 09-03 and 09-09, and the
release-anchored version dies on gap share. What has never been run is the yield
extreme as a conditioner on EQUITY cross-sections, which is A1 and B2.

**Credit.** HYG -0.63% from a 52-week high while IG sits at 52-week lows.
**Verdict: PASS, dismissed on a structural finding rather than on this reading.**
The credit-specific residual has failed SIX consecutive times here; the standing
decomposition is HYG = -0.001% + 0.189·IEF + 0.395·SPY at R-squared 0.477, so a
long-credit-on-a-credit-state idea is dead on arrival and the honest move is to
pitch the equity leg or nothing. A HYG-versus-LQD quality pair at joint 52-week
extremes is the seventh instance of the same object and is dismissed without a
check for that reason.

**Gold and miners.** GLD 403.35, z10 -1.07, r21 38.5, -18.66% from its high;
GDX +9.92% over 21d and +5.07% over 5d, r5 69.0.
**Verdict: CHECK** — the metal is washed out on a 10-day basis while the higher-
beta expressions run. See B3, and see the harshness clause attached to it.

**Other metals.** SLV +4.83% over 5 days, +63.31% above its 52-week low and
still -42.50% below its high. **Verdict: CHECK**, as the short leg of B3. Not
examined standalone: "Long silver deep inside a post-parabolic drawdown" was
killed 09-01 and "Short silver after the complex break" is parked on a lag
profile that has not moved.

**Energy.** XLE and XOP at 0.00% from their 52-week highs, z10 +1.74 and +2.05;
USO +19.10% over 21d at z10 +2.78; VLO r63 99.6 and z10 +2.93; CVX and COP at
their highs. **Verdict: CHECK, but only as the SHORT leg of a cross-sector
pair.** Long energy at a 52-week high with a print in the hold is measured dead
(XOP positive at 0 of 10 horizons, era-inverted, 0 of 84 September-and-midterm
episodes) and short energy as a crude fade dies on USO's roll at 1.8x cost. The
untested object is energy against the sector it is taxing — A2.

**Natural gas.** UNG 10.09, -3.54% on the day, +4.78% above its 52-week low,
-13.68% below its 200-day SMA. **Verdict: CHECK** — the September shoulder-season
trough is a real mechanical seasonal and the only cell today in a class the last
nine mornings barely touched. The 09-01 kill was a natgas THRUST, the opposite
state. Roll drag is the pre-flagged threat; C2 prices it first.

**Broad commodities.** DBC at 0.00% from its 52-week high, z10 +2.50, r21 85.3.
**Verdict: PASS**, killed 09-08 and 09-09 on both the long and the rates
expression; the parked short-IEF rung is arm-blocked on a placebo ladder.

**Dollar and FX.** DX-Y.NYB 98.77, r5 12.7, r21 25.0, r63 15.1, 5d -0.79%; UUP
r5 10.3, r63 13.1. The currency is at a three-month rank floor on the same
sessions the 10-year prints a one-year yield high.
**Verdict: CHECK, and it is the highest-priority cell of the morning.** The SHORT
side of this exact state was tested and killed on 2026-09-09 at 0 wins in 6
episodes and 0-for-11 at day level with a bootstrap of 1.000. Nobody pitched the
LONG side, which is what that record describes. See B1.

**International.** EFA r21 12.7 and -2.07% from its high; EWJ -1.49% from its
high with r5 73.4 and r21 29.8; FXI -5.32% over 21d, r21 16.3; EEM r21 61.9;
EWZ +8.18% over 21d, r5 85.3.
**Verdict: CHECK on Japan, PASS on everything else.** EM was swept 09-09 (EWZ vs
FXI is one leg, and EWZ ranks 9 of 9 in its own reference class at a max-of-K P
of 1.0000); the country-decoupling family is closed at P 0.477. Japan is the one
developed market this repo has never run a cell on, and today it is the strongest
member of a lagging EAFE. See B2.

**Volatility.** `^VIX` 16.46 with its 21-day range at the 2nd percentile of the
trailing year; `^VIX3M` 18.87, ratio 0.8723; `^SKEW` 149.25 at a 21-day rank of
94.0; SVXY -1.73% from its 52-week high; UVXY -73.27% from its.
**Verdict on SKEW: PASS.** The 21-day skew rank is measured as the DILUTED tail
of the parked 5-day form (excess +0.021pp at the pitched threshold; the midterm
block steepens with the threshold to -0.387pp at 98-plus) and it has no
cross-sectional content. **Verdict on compression: CHECK, and it moved during
this morning's own arm pass.** C3 was specified against the DASHBOARD's 2nd
percentile, on the theory that the measured-dead (0,5] bucket might flip sign on
a dense calendar. On the watchlist's relative-range definition the reading is
**9.92**, not sub-5, which puts today inside the (5,15] band that is the LIVE
POSITIVE half of that same bimodal gate. So the premise of C3 as written is
wrong and the real object is watchlist entry #33 — sent to a fourth checker as
D1. C3 still runs, because whichever definition is right, one of the two cells
is being tested on a false premise and it is worth knowing which.

**Cross-sectional breadth.** Eight names closed at fresh 52-week lows (AON, LOW,
MCD, NKE, SYK, TJX, VFC, VMC) with the index 2% off its high; thirteen of 218
sit at z10 <= -2 and they are overwhelmingly quality-industrial-staples (MMM
-3.01, PSA -2.53, ITW -2.37, TJX -2.32, IP -2.31, SYK -2.23).
**Verdict: DISMISSED WITHOUT A CHECK, and the reason is the direction of the
survivorship bias.** The tape is a FIXED set of today's members, so names that
made 52-week lows in the past and then delisted or were acquired are absent, and
past new-low counts are systematically UNDERSTATED while today's is not. The bias
therefore INFLATES today's apparent extremity — it runs against the idea rather
than for it. This repo quantified the same gap on 2026-09-09 at 0 (11-name SPDR
universe) against 84.5 (218-name tape) for the rank-floor version. The SPDR-level
check is empty in any case: zero sectors sit at a 52-week low today.

## 3. Every live seasonal and cycle cell

- **Cycle year: MIDTERM** (2026 mod 4 == 2). This is a conditioner on everything
  above and it is a documented one here: the dollar goes wrong-signed in midterm
  years on the payrolls anchor (seven reproductions on record), the pre-FOMC
  drift inverts in midterm years, and the 21-day skew block steepens. Every
  candidate below reports its midterm split.
- **Month: September, trading day 7.** The turn into September was killed on
  08-26 (the anchor does not name the one scanned session that carries it) and
  the September PPI/CPI subcell was killed on 09-08 and then CORRECTED on
  2026-09-08's answer-quality review: its quoted permutation of 0.1618 tested
  MARCH's maximum, and the probability that any eligible shuffled month equals or
  exceeds SEPTEMBER's +0.844% is **0.7354**. That cell is dead and the correction
  is the reason. **Verdict: PASS.**
- **The seasonality board in the state file is dated 2026-08-05 and is five weeks
  stale.** Its CBOE put/call reading of 0.69 at the 4th percentile is a MONTH-OLD
  observation of the TOTAL series and it disagrees with today's live EQUITY
  series at the 43.25th percentile with fear OFF. **Not used.** Its board
  candidates are regime context on the systematic book, which the standalone rule
  puts out of scope for selection anyway.
- **Natural gas injection season** is the one live seasonal with a mechanism
  rather than a count, and it is checked at C2.

## 4. Every active watchlist entry

46 active, 0 expired. Full numbered verdicts are in
`01_watchlist_verdicts.md`, produced by a dedicated arm-computation pass with
`01_watchlist_arms.py`, which computed every mechanically-checkable arm rather
than eyeballing it. Nine came back CHECK and **two armed outright**:

- **#33, "Long SVXY into a scheduled print out of a 21-day VIX range in the
  (5,15] band, with a clear calendar behind it" — ARMED IN FULL**, for the first
  time since it was parked on 2026-09-03. Relative range 9.92, inside the band.
  The live anchor is CPI 2026-09-11, whose k=−2 session is 2026-09-09 at a runway
  of 3 sessions to the FOMC, which puts the entry MOC at **today's close**.
  PPI 2026-09-10 is disqualified by the entry's own text at runway 1. Sent to a
  fourth checker as **D1** for a from-scratch re-derivation plus rounds 2 and 3.
- **#18, the duration-neutral flattener (long IEF / short 0.523 TLT) with the
  ten-year at a 252-day maximum — BOTH LEGS ARMED**, also a first. But the arm
  pass found it armed on a **denominator roll rather than on repricing**: the
  year-ago `^TNX` reference fell 16.1 bp over six sessions while the live yield
  rose 7.9 bp, so the required close dropped from 4.987 to 4.826 and now clears
  by **1.1 bp**. Sent to the same checker as **D2**, with that knife edge and the
  entry's unpaid multiplicity debt as the first two things to attack.

Seven more came back CHECK and are dismissed here with their numbers: #43 (today
IS the PPI release session, but the statistical arm is a correlated-family
permutation that has not moved), #5 (three of four legs clear; TLT must close at
or below 81.44 and sits 0.35% above it), #2 (today IS the entry session but the
LOYO floor is 20.1 bp against a 40-50 bp bar, and it is one position with #33 at
corr 0.626), #20 (SPY is 1.990% off its high against an arm of more than 2.000%,
missing by a single basis point, and its dial leg fails by 15 points), #19 (the
energy count at z10 >= 2.0 is **4**, one past the [2,3] band and exactly at the
ladder's zero crossing), #12 (`^VIX` +4.71% against an arm of >= +5%, 29 bp
short, and the 21-day rank leg fails at 65.1 against <= 25), #32 (the state
re-armed but the arm is a permutation and it has not moved).

The structurally important ones among the other 37:

- **The pooled sector triple rank floor** was parked on two legs, and one of them
  is now VOID. Its arm read "A REASON TO EXIST BESIDE THE BOOK", and the owner's
  2026-09-08 standalone-quality decision retires book overlap as a criterion. The
  entry's SECOND leg still binds and it is statistical: the near-high gate that
  would make it novel SUBTRACTS 0.733pp, and the nine-SPDR fixed-effect common
  excess of the gated form is negative at -0.381pp. **Verdict: PASS on the
  statistics, and the reading has collapsed anyway** — no sector holds the 5, 21
  and 63-day floor jointly today (XLI is at 3.2 / 3.2 on the 21- and 63-day but
  33.3 on the 5-day).
- Every calendar-parked entry (November TLT, the non-midterm NFP TLT floor, the
  non-midterm dollar washout, the December small-cap month-end, the post-Jackson-
  Hole IEF session) parks to a DATE that is not today. **PASS.**
- The dial-conditioned entries (SPY-vs-IWM in the [56,70) band, the pre-print
  compression direction leg) need the dial BELOW 50 or back inside [56,70). It is
  87.7. **PASS.**

## 5. Scoreboard read before selecting

Five graded ideas lifetime, avg R +0.174, hit rate 80%, all opus. By grade: B
n=3 at +0.448 avg R, C n=2 at -0.237. By axis: event_fingerprint n=2 at +0.622,
interaction_cell n=1 at +0.146, relative_value n=1 at +0.099, inversion n=1 at
-0.620. **The graded count is five, which is a handful and not a signal.** No
axis is up- or down-weighted on it. Recorded so the read is visible.

## 6. Candidates selected from this map

Nine candidates, three checkers, each candidate handed over with a PRE-SPECIFIED
direction, vehicle and mechanism so it is not charged as a search (the
2026-09-09 method finding).

| # | candidate | axis | asset class | anchored on |
|---|---|---|---|---|
| A1 | Long XLRE / short XLF when real estate sits at a 63d rank floor and `^TNX` prints a 252d yield high | interaction_cell | real_estate + rates + financials | price state |
| A2 | Long XLY / short XLE at a 21d energy-minus-discretionary spread extreme | relative_value | us_sectors + energy | price state |
| A3 | Long the ETF whose 63d rank is bottom-decile while its 5d rank is top-quartile, pooled over 29 | inversion | us index and industry ETFs | price state |
| B1 | Long UUP with the dollar at a 63d rank floor while `^TNX` prints a 252d high | inversion | dollar_fx | price state |
| B2 | Long EWJ / short EFA on a `^TNX` 252-day yield high | interaction_cell | international | price state |
| B3 | Long GLD / short SLV when silver outruns a washed-out gold | relative_value | metals | price state |
| C1 | Short SPY into a forward six sessions carrying four or more scheduled events | event_fingerprint | us_large | **calendar** |
| C2 | Long UNG in September within 6% of its 252-day low | interaction_cell | energy / natgas | calendar x price state |
| C3 | Long SVXY at a sub-5th-percentile 21d VIX range into a dense calendar | interaction_cell | volatility | **calendar** x price state |
| D1 | Watchlist #33: long SVXY, (5,15] relative-range band, k=-2 print anchor, clear calendar — ARMED | interaction_cell | volatility | **calendar** x price state |
| D2 | Watchlist #18: long IEF / short 0.523 TLT with the ten-year at a 252d maximum — ARMED | relative_value | rates | price state |

Eleven candidates once the arm pass added D1 and D2.

Axes: interaction_cell, relative_value, inversion, event_fingerprint — four,
against a floor of four. Asset classes: real_estate, financials, us_sectors,
us_large, energy, natgas, metals, dollar_fx, international, volatility — ten,
against a floor of four. Event-anchored: C1, C3 (and C2 on the month). Price-
state-anchored: A1, A2, A3, B1, B2, B3. Both search modes are present and they
are CROSSED in A1, B1, B2 and C3, which is the 2026-08-07 failure the two-mode
requirement exists to prevent.

Dismissed at selection, with reasons, so the map records the choice:
- cross-sectional new-52-week-low breadth — survivorship bias runs against it (section 2)
- HYG versus LQD at joint 52-week extremes — the credit residual has failed six times (section 2)
- the pooled sector triple rank floor — nothing armed, and the gate subtracts (section 4)
- the pre-FOMC midterm run-in — swept across fifteen classes on 09-01, n=10 at a 40% hit (section 1)
- the PPI/CPI pair on equity or duration — both orderings measured, September wrong-signed (section 1)
- commodities at a 252-day high into the print — does not exist outside four shock years (section 1)
- `^SKEW` at a 21-day rank of 94 — the diluted tail of the parked 5-day form (section 2)
- energy leadership with a print in the hold — 0 of 10 horizons, 0 of 84 midterm-September episodes (section 2)
- long duration at the 52-week-low complex — pitched and killed four times in eight sessions (section 2)
