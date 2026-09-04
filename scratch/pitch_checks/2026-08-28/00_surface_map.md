# Surface map — 2026-08-28 (Friday, Jackson Hole day, ME-1, midterm year)

Freshest bar 2026-08-27. `warnings` empty. Dial ma10-63d **88.1** (raw 21d 69.3,
raw 63d 89.5), exposure leg 0.0x, P/C fear OFF at the 52.8th percentile, one
fragility signal on (Low Absorption Ratio). SPY 771.10, 0.87% off its 52-week
high, 9.07% above its 200d, 101 sessions since a 5% pullback.

Book staged today: OLV long SPG, OVS shorts MSTR and NEO (overflow). No event or
trend sleeve positions open.

Scoreboard read before selecting: 5 lifetime ideas, 4 graded, avg +0.372R.
`event_fingerprint` leads at +0.622R on 2 graded, `interaction_cell` +0.146R on
1, `relative_value` +0.099R on 1, `inversion` 0 graded. That is a handful, not a
signal, so no axis is up- or down-weighted today. Recorded and moving on.

---

## 1. Every live calendar event x every asset class

Nine events sit in the [-5, +15] td window. The registry has now filed **four
consecutive** calendar findings saying the late-August midterm anchor set is
exhausted, most recently 2026-08-27: "every macro anchor reachable from late
August of a midterm year has now been swept on every asset class it has a
vehicle for," with the next genuinely new anchor being the September FOMC
entering the horizon around 2026-09-02. Today adds nothing to the inventory —
no anchor crossed inside the 10 td cap that was outside it yesterday. The one
thing that DID change is that Jackson Hole moved from JH-1 to **JH+0**, which
turns the anchor from a pre-event window into a post-event one, and that is the
only cell in this section that gets a check.

| event | date | td | verdict |
|---|---|---|---|
| opex | 2026-08-21 | -5 | DISMISS. Post-opex closed in both directions on equities and across ten non-equity vehicles by ten horizons (2026-08-20). At +5 td the window is also spent. |
| **jackson_hole** | **2026-08-28** | **0** | **CHECK (C6).** Eight classes are closed PRE-speech and the offset ladder is 9-for-9 against the anchor. The POST-speech anchor has never been swept as a cross-asset object: the only post-conference number in the registry is the TLT one (2026-08-13, entering after the conference -0.204%, +1 to +3 sessions -0.607% to -0.848%), i.e. one class of ten. Cheap to close, and it is the last live direction of this anchor. |
| nfp | 2026-09-04 | 5 | DISMISS. Closed 2026-08-26 on the placebo offset ladder (SPY k=-2 pays +0.618% against the true anchor's +0.537%) plus a September-midterm cell of -0.676%. The one live rates cell is watchlist 0, midterm-parked to 2027-01. |
| ppi | 2026-09-10 | 8 | DISMISS. Closed 2026-08-27 as a containment object: the PPI/CPI containment gate is ON for 39.7% of all days and sits near the bottom of its own placebo ladder. Short commodities / short vol across a PPI session died on the ladder 2026-08-12. |
| cpi | 2026-09-11 | 9 | DISMISS as an entry today; the one live cell (watchlist 2, SVXY MOC-eve to MOO-print) parks to the eve, 2026-09-10. Long HYG into CPI and the SVXY 3-day hold are both closed. |
| fomc_decision | 2026-09-16 | 12 | DISMISS, out of horizon. Enters ~2026-09-02, and the pre-FOMC drift is already the event sleeve's T1/T2 (midterm years take the T2 short, gated on SPY 21d rank < 50; today 91.3, so even the sleeve rule is off). |
| vix_expiry | 2026-09-16 | 12 | DISMISS, out of horizon. Also: vix_expiry and opex are ONE anchor sharing 189 of 307 days (2026-08-17), and pre-expiry short-vol carry died 2026-08-07. |
| opex | 2026-09-18 | 14 | DISMISS, out of horizon. |
| quad_witching | 2026-09-18 | 14 | DISMISS, out of horizon. |

Non-macro calendar position: today is **ME-1** (August's last session is Monday
2026-08-31) and **opex+5**. The month-turn anchor closed its fourth and last
class on 2026-08-27, and the ME-3 -> ME-2 small-cap session (watchlist 29) is
both spent and midterm-blocked. What has NOT been swept is the month turn
crossed with a live SECTOR price state rather than an index — that crossing is
candidate **C11**, and it is the one calendar cell today that is not a re-run.

### The event x class grid

Ten classes x the six events that are inside any tradeable horizon. Rather than
sixty separate lines, the grid collapses on two facts, both cited above: the
pre-speech Jackson Hole anchor is closed on all eight classes it has a vehicle
for, and every other event is either spent (opex), out of horizon (FOMC, VIX
expiry, September opex, quad witching) or closed on its own ladder (NFP, PPI,
CPI). The classes therefore inherit the event's verdict, and the only cell where
the class identity could still matter is post-JH, which is why C6 sweeps all ten
classes at once instead of picking one.

| class | proxy used in C6 |
|---|---|
| US large | SPY, QQQ |
| US small | IWM |
| rates | TLT, IEF, ^TNX |
| credit | HYG, LQD |
| gold and miners | GLD, GDX |
| other metals | SLV, XME |
| energy | USO, XLE |
| dollar and FX | UUP, DX-Y.NYB |
| international | EFA, EEM, FXI |
| volatility | ^VIX, SVXY |

---

## 2. Tape extremes by class

The whole 218-name tape was sorted (`00_tape_sort.py`); state recon in
`02_state_recon.py`. What is actually extreme today:

**us_large** — the tape's defining feature is a 21-day/63-day rank inversion
running through every index: SPY 91.3 / 23.4, QQQ 87.7 / 13.9, IWM 61.9 / 14.7,
XLK 88.5 / 24.2, SMH 79.0 / 1.6, EEM 94.0 / 3.2, EWJ 93.3 / 23.8. The market
round-tripped a 63-day drawdown in 21 sessions and is back on its high. SPY
inside 1% of its 52-week high with a 63-day return rank at or under 25 is
**138 days of 6,389** (2.16%), against 2,286 near-high days and 1,787
low-63d-rank days taken separately. Rare, live, and never checked. **C1.**

**us_small** — IWM 1.73% off its high, 21d rank 61.9, 63d rank 14.7. The same
shape as C1 but less extreme; it rides in C1's reference class rather than
getting its own slot. Month-position cells on IWM are closed (watchlist 29).

**volatility** — VIX 14.51 at a 21-day rank of 6.0 after a 29.77% 21-day
collapse; VIX3M 17.56 sitting **exactly on** its 52-week low; **SVXY closed
exactly at its 52-week high** (dist +0.000%), 21d rank 98.0. Term-structure
percentile as an entry is closed in both directions and re-confirmed twice on
the placebo ladder (2026-08-13, 2026-08-17), and "3-month IV at a one-year
floor, forward SPY and SVXY" was killed yesterday, so the IV-level lane is shut.
The SVXY PRICE state is a different object and has never been examined. **C2.**

**rates** — TLT 83.13, 2.19% off its 52-week low, 5d rank 78.6 (rallying);
IEF 1.14% off its low; LQD 1.24% off its low; ^TNX 4.67, 1.54% off its high,
63d rank 74.6. The IG complex is pinned at the floor while the long end bounces.
Both forms of this are already parked with their numbers (watchlist 5 on
freshness, 31 on an episode count of one) and neither trigger moved. DISMISS.

**credit** — HYG 0.06% off its 52-week high, 21d rank 91.3, and it is the
tightest thing in the tape. The two shapes this supports are both closed:
HY-at-a-high-while-the-index-is-not died on a depth-matched split (watchlist 28,
and today SPY is only 0.87% off, so the depth leg is not even there), and
synchronized SPY/EFA/HYG highs add +0.036pp over SPY alone (2026-08-10).
DISMISS.

**gold and miners** — GDX +40.94% over 21 days at rank 100.0, NEM +44.83% at
rank 100.0, GLD +13.88% at rank 95.2 and STILL 14.78% under its own 52-week
high. Yesterday's published idea is the first flush inside this run
(fingerprint 749b2073856902b3, inside the 10 td repetition window), the miner
fade is closed twice, and the miner/metal ratio is wrong-signed at all ten
horizons. The one un-run object here is the JOINT state with equities: GLD and
SPY both in the top decile of their 21-day returns, 88 days of 5,205 against 75
expected under independence. **C10.**

**other metals** — FCX z10 +1.81, the largest thrust in the tape, 5d +10.11%,
21d rank 99.6, 1.86% off its high; XME 5d +7.24%, 21d rank 99.6 but 63d rank
20.2. The copper thrust was pitched and killed on 2026-08-24 for a false premise
(FCX ran while copper itself did not), and nothing about that diagnosis has
changed in three sessions. DISMISS as an outright; XME rides in C4's family.

**energy** — the complex is pulling back inside a thrust: XLE 5d -2.29% (rank
19.4) against 21d +6.21% (rank 68.3), 2.29% off its 52-week high; COP -3.98/
+10.44; EOG -5.05/-0.99; OIH -0.27/+12.78 with a +2.49% last session while E&P
fell. The thrust-into-a-high form is closed (2026-08-17) and the washout form
needs a 5d rank <= 5 that no sector has (watchlist 25, lowest is XLU at 26.6).
The PULLBACK-inside-a-thrust rung between them is unexamined. **C8.**
Services-vs-E&P is live at a PIT of 3.17 but its trigger is a RECORD standing at
28 of the 32 wins it needs (watchlist 24). PASS.

**dollar and FX** — DX-Y.NYB 99.16, 21d rank 9.5, z10 -0.85, 3.06% off its
52-week low. Four dollar cells are parked and every one fails today (watchlist
14 misses the yield leg at +0.050pt against +0.20pt, 16 misses the TNX rank at
52.0 against 65, 27 misses the PIT rank at 9.5 against 2 and is midterm-parked).
The translation channel — a dollar washout expressed through UNHEDGED developed
international rather than through the dollar or through EM funding — has not
been tested. **C9.**

**international** — EEM 21d rank 94.0 against a 63d rank of 3.2, the sharpest
turn in the tape (the literal joint state is 6 days ever, so it is a family
question, not an EEM call — that is C4). EWZ 5d rank 91.3 while sitting 13.49%
below its 52-week high, which is a thrust from inside a drawdown and the
inversion of the closed "one market BREAKING inside an intact thrust" family
(EWZ twice, FXI, SMH/QQQ). **C3.** FXI fails watchlist 9 at 32.1 / 37.7.

**sectors** — XLF 63d rank 98.0 and 0.74% off its high; XLV 63d rank 90.1;
IBB 21d AND 63d rank both 96.8 with z10 +1.65, 1.00% off its 52-week high, XBI
0.78% off its own on a +87.80% year. Biotech is the one industry group in this
tape at a sustained double-rank extreme and the class has never been swept here.
**C5.** On the other side, all three rate-sensitive defensives sit at 21-day
rank floors together (XLP 20.2, XLU 11.9, XLRE 9.9) while the index is 0.87%
off its high — 16 days in history. **C7.** Tech-vs-healthcare died yesterday on
reference-class indistinguishability across 132 ordered pairs; SMH's 63d floor
is parked and its conditioner (5d rank < 15) reads 50.4 today; KRE-vs-XLF is
parked on cost. DISMISS all three.

**single names** — CRM +22.69% in five sessions, the largest liquid large-cap
move on the tape, and TJX/HRL at the other pole (z10 -2.90 and -1.74, TJX at its
52-week low). Both poles are book territory: the registry records that 52wh
Breakout is substantially an earnings-reaction strategy (148 of 250 signals
within one session of a print), and the pre-print washout lane including TJX by
name was closed on 2026-08-14 and again on 2026-08-27. DISMISS.

---

## 3. Seasonal and cycle cells

The seasonality board carries 0 A/B-grade setups and five context rows, four of
which say the same thing: **midterm years are the unfavourable cycle for this
book** (book win 56.4% vs 64.9%, +0.24R vs +0.43R on 1,099 midterm trades), with
OVS, LT Trend ST OS and Indices Oversold Bounce all flagged to fade. That is a
conditioner on everything below rather than an idea, and it is why every
candidate below owes a midterm split. The fifth row is the depressed CBOE
put/call reading, which is a fragility input rather than a cell and reads at the
52.8th percentile as of 2026-08-26, i.e. fear OFF but nowhere near complacent.

Month-of-year: the turn into September from the last August session is closed
(2026-08-26, 93% of the three-day window sits in the ME-3 to ME-2 session alone,
max-of-12 permutation P 0.238). The September-specific inversions the repo knows
about are already the event sleeve's T3 and V4 carve-out.

---

## 4. Watchlist verdicts

All 32 active entries were priced against today's tape in
`01_watchlist_verdicts.py`, which prints the number for each. **32 PASS, 0
CHECK.** Three have a LIVE STATE whose trigger is a count or a record rather
than a level, so they stay parked without re-testing: #1 LQD/HYG at joint 52w
extremes (state live, needs >= 8 declustered episodes ex-2018, still 4), #24
OIH at a services-vs-E&P extreme (state firing at PIT 3.17, needs 32 of 51 wins,
stands at 28), #31 the IG complex at 52w lows against an HY high (state live,
needs more than one declustered episode and this is still the same 2026 one).
Nothing was pruned; the file is rewritten after publish.

---

## 5. Selected candidates

Eleven, over seven novelty axes and nine asset classes. Two search modes are
crossed rather than run in parallel: C11 is a calendar anchor applied to a live
price state, and C6 is the event anchor swept across every class the tape
carries.

| # | candidate | axis | class |
|---|---|---|---|
| C1 | SPY inside 1% of its 52-week high while its 63-day return rank is bottom-quartile — the round-trip breakout | interaction_cell | us_large |
| C2 | SVXY closing at a fresh 52-week high, post-2018-03 vehicle only | instrument_translation | volatility |
| C3 | EWZ thrusting (5d rank >= 90) from more than 10% below its own 52-week high | inversion | international |
| C4 | The V that turned, as a family: 21-day rank >= 90 with 63-day rank <= 10 across every index and industry ETF | interaction_cell | international / us_large / sectors |
| C5 | IBB with 21-day and 63-day ranks both >= 95 inside 1% of its 52-week high | interaction_cell | sectors (biotech) |
| C6 | The POST-Jackson-Hole anchor, JH close to +1..+10, swept on ten classes | event_fingerprint | all |
| C7 | XLP, XLU and XLRE all at bottom-quintile 21-day ranks while SPY sits inside 1% of its high | interaction_cell | sectors (defensives) |
| C8 | XLE 5-day pullback (rank <= 20) inside a 21-day thrust (rank >= 65) within 3% of its 52-week high | interaction_cell | energy |
| C9 | Long EFA against SPY on a dollar washout — the translation channel, not the EM funding one | relative_value | dollar_fx x international |
| C10 | Gold and the S&P both in the top decile of their 21-day returns | historical_analogue | gold x us_large |
| C11 | The month turn on the washed-out defensive sectors, ME-1 close to ME+3 | flow_mechanics | sectors x calendar |

Coverage check: event-anchored C6 and C11; price-state-anchored C1-C5, C7-C10.
Classes touched by candidates: us_large, us_small (inside C1/C4), volatility,
international, sectors, energy, dollar_fx, gold, and rates/credit by verdict
only — both of those are dismissed with the parked numbers above rather than
silently absent.

Negative-registry collisions declared up front: C1 is adjacent to the closed
"cross-sectional new-high breadth" family and to the 2026-08-14 low-VIX
near-high cell that dies on its local control, so it owes CTRL-c. C2 must use
the post-2018-02-28 vehicle only. C4 is adjacent to the parked SMH cell and must
be run as a reference class, not as one name. C5 is the IHI shape and owes a
Cochran Q. C7 and C10 are joint states and owe a "does the join do any work"
attribution. C8 must beat the bare pullback with no near-high clause. C9 owes a
beta attribution before any of it is called translation.
