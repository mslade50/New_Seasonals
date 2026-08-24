# Surface map, 2026-08-24 (Monday, midterm year)

Tape bar 2026-08-21. Freshness OK: `pipeline.ok = true`, 7/7 green, dial 2026-08-21,
P/C 2026-08-21, no `warnings`. Every number below comes from `01_recon.py` or
`02_watchlist_verdicts.py` in this folder, not from recall.

## Where the calendar actually is

| anchor | date | offset from today |
|---|---|---|
| vix_expiry | 2026-08-19 | -3 td |
| opex | 2026-08-21 | -1 td (today is opex+1) |
| **jackson_hole** | 2026-08-28 | **+4 td (today is JH-4)** |
| **August month-end close** | 2026-08-31 | **+5 td (today is ME-5)** |
| nfp | 2026-09-04 | +9 td |
| ppi | 2026-09-10 | +12 td (outside the 10 td horizon cap) |
| cpi | 2026-09-11 | +13 td (outside) |
| fomc_decision | 2026-09-16 | +16 td (outside) |
| quad_witching / opex | 2026-09-18 | +18 td (outside) |

**The most important calendar fact this morning: every live anchor inside the horizon
is already closed in this repo.** The JH anchor is examined and empty on rates, gold,
FX, small caps, large caps, credit and international (2026-08-13, -08-18, -08-21), with
a pre-speech class mean at h<=4 of **+0.010pp** and a ladder that is a plateau from -12
to +3. The opex anchor is closed in both directions on US equities (2026-08-07,
-08-20) and across ten non-equity vehicles by ten horizons (2026-08-21: grid excess sd
0.132pp, 0 of 100 cells clear 1.0pp). The two are also frequently the SAME anchor: the
August opex close is JH-5 in 21 of 26 years. That leaves **month-end** as the only
anchor in the window never swept on equities, which is where C1 and C2 go.

## 1. Every live event x ten asset classes

Rows are the four anchors inside the horizon. CHECK = a candidate was cut for it.

| class | opex+1 (-1td) | jackson_hole (+4td) | month-end (+5td) | nfp (+9td) |
|---|---|---|---|---|
| us_large | closed both directions 2026-08-07 / -08-20; the overnight decomposition is 0.24x cost | closed 2026-08-18, ladder 8 of 16 at h=10, midterm inverts to -1.485% | **CHECK C2** - never measured on equities in this repo | post-NFP equity direction swept empty 2026-08-07, and +9 td is the horizon edge |
| us_small | closed 2026-08-20 (IWM run out of opex; the live near-high state inverts it) | closed 2026-08-11 (IWM JH-13 wrong-signed in midterms) | folded into C2 as the vehicle question | not examined; an NFP anchor entry is ~2026-09-03, not today |
| rates | closed 2026-08-21 (TLT -0.105pp, -5.5x cost) | closed 2026-08-13, the anchor is decoration on an August seasonal that is itself a bond-bull fossil | **CHECK C1** - the parked ME-9 cell at the entry offset that is live today | midterm-dead, parked to 2027-01 (W0) |
| credit | closed 2026-08-21 (HYG +6.0 bps, 1.5x) | closed 2026-08-21; credit SUBTRACTS 0.355pp from the SPY leg | not examined; C10 covers the credit complex on a price state instead | not examined |
| gold_miners | closed 2026-08-21 (grid) | closed 2026-08-13 (10-11, 92% two episodes, midterm -1.213%) | not examined - dismissed: the complex is at a 21d rank of 100 and four separate 2026-08 kills already cover thrust states in it | not examined |
| other_metals | silver closed 2026-08-21 (ladder rank 1 of 17 and it still died: drop-two +0.363% against an August base of +0.584%) | not examined - dismissed under "treat the JH anchor as closed" plus the +0.010pp class mean | not examined | not examined |
| energy | crude JH-6 killed 2026-08-20 on concentration; XLE blocked by its own 52w-high condition | as left | not examined | not examined |
| dollar_fx | closed 2026-08-21 (-0.050pp, -0.8x) | closed 2026-08-13 (13-13, drop-best flips the sign) | not examined | not examined |
| international | closed 2026-08-21 (the FXI gate is worth +0.220pp; ladder 2 of 17 at h=10 but 9 of 17 at h=5) | closed 2026-08-21 (residuals inside +/-0.12pp) | not examined | not examined |
| volatility | the opex and vix_expiry anchors are ONE anchor (189 of 307 shared days); V4_POSTOPEX_VOL is live and holds SVXY through 2026-08-26 | not examined - dismissed: closed anchor, and a JH-anchored vol leg would sit on top of the live V4 position | not examined | post-NFP vol swept and empty (event sleeve prereg) |

Four cells read "not examined" without a check, and the reason is the same in all four,
stated once: the anchor they hang on has been closed on five to seven other classes with
a ladder that is a plateau rather than a spike, which is a statement about the ANCHOR
rather than about the class. Opening an eighth class on a dead anchor is the cheapest
available way to manufacture a false positive.

## 2. Tape extremes by class

218 names. **20 sit within 0.25% of a 52-week high** (AMGN BDX COP EOG FCX HYG KO MRK
NEM NSC RHI SCHW SVXY TGT UNP V XLB XLE XOP ^TNX) and **6 within 2% of a 52-week low**
(CMS IEF LQD PEG TLT UVXY). Not one of the twenty is a technology name.

| class | the extreme | verdict |
|---|---|---|
| us_large | SPY -1.56% off its high, r21 rank 81.7 against an r5 rank of 15.9; QQQ -4.28% off, r63 rank 17.5 against SPY's 30.2 | **CHECK C6** (breadth), **CHECK C8** (the index pair) |
| us_small | IWM -1.68% off its high, r63 rank 40.5, z10 -0.17 | nothing extreme; covered inside C6 and C8 as a vehicle question |
| rates | **^TNX AT a 52-week high** (-0.15%) while its 21d RETURN rank is only 49.2; TLT +0.86%, IEF +0.70%, LQD +0.56% off their 52w lows | **CHECK C5** (the level-at-a-high trigger, which is not the return-rank trigger the registry has killed), **CHECK C10** (the IG rung that does fire) |
| credit | HYG -0.23% off its 52w HIGH while LQD is +0.56% off its 52w LOW | W1 PASS: the trigger is episode count and it is still 4 declustered since 2007. Also answered on the merits - the LQD leg's residual against IEF is +0.000pp at h=5 |
| gold_miners | NEM +38.91% over 21d AT a 52w high; GDX r21 rank 100.0 at +37.07%; GLD r21 rank 96.8 | dismissed on four independent 2026-08 kills (GDX maximal thrust, miner/metal ratio, GLD miner-led thrust, silver/gold drawdown) plus book overlap - the scanner staged SHORT NEM, AGI, AU and CGAU this morning. **C9 takes the one angle none of them covers**: gold strong with yields at a 52-week high |
| other_metals | **FCX +15.30% in 5 sessions to a fresh 52w high on 2.0x volume**, +118.72% off its 52w low; XME +15.67% over 21d; XLB at a 52w high | **CHECK C3.** Copper has never been examined in this repo. Silver and gold both have |
| energy | XLE -0.17% and XOP 0.00% off 52w highs with **five names carrying z10 > 2** (VLO 2.56, COP 2.35, XLE 2.18, XOP 2.10, CVX 2.04) | **CHECK C7** on the z10 CLUSTER, a breadth-of-thrust statistic distinct from the four killed single-instrument energy cells |
| dollar_fx | DX-Y.NYB r21 rank **0.4**, the most extreme single reading on the tape, +2.68% off its 52w low | dismissed 2026-08-20, and that kill is specific to today's cell: the positive sign lives only in "rank-extreme, magnitude-ORDINARY", today's -2.59% against a trigger median of -4.19%, and that slice is 105% top-2 episodes at h=5 with a mechanism running backwards |
| international | EEM r63 rank 11.1 while EFA r21 rank 89.3 and FXI r5 rank 87.7 | dismissed: the country-decoupling family is closed on five members and the standing bar is P(max-of-K) < 0.05 on the residual before a sixth is worth a check |
| volatility | **SVXY AT a 52w high, UVXY AT a 52w low**, VIX 15.13 with a 21d return rank of 12.7 | dismissed on the lagging-marker kill (2026-08-13, -08-17): the gate identifies carry already harvested, and SVXY's trailing 21d is +9.28% against the +10.46% trigger median. Live book overlap too - V4 holds SVXY to 08-26 |
| sectors | XLV r63 rank **99.6** (+18.39%) and -0.60% off its high, against XLK r5 rank 12.3 and r63 rank 28.2; **XLI r5 rank 2.4**; XLU r21 rank 0.8 | **CHECK C4** on the cyclical split (XLI washed while XLB and XLE print highs). Utilities dismissed - dead in SEVEN expressions. XLV against the index dismissed - the sector-vs-index pair family is closed and the one-day rotation gap was closed 2026-08-19 |

## 3. Seasonal and cycle cells

- **Midterm year (2026 % 4 == 2).** Not an idea, a conditioner on every cell above, and
  the repo's own evidence says it is usually a negative one: the JH inversion reproduces
  in six vehicles, the skew-dip cell is negative at every horizon in midterms, the NFP
  rates cell is midterm-dead, and the seasonal board's book stats read midterm win 56.4%
  against 64.9% all-years. Every check below splits on it.
- **Late August, trading days 17-21.** The August window measured here is tdom 6-16 (SPY
  +0.234% over 286 starts; TLT +0.990% and a fossil). The last week of August has not
  been, and it is the same object as the month-end anchor, so it is folded into C2 rather
  than run twice.
- **Mid-August midterm seasonality**: dead (N=6, carried by 2002, drop-two negative).
- **Turn-of-month**: the classic last-1 / first-3 cell is in the registry as arbitraged
  away post-2013. C1 and C2 are the ME-5 to ME-0 span, a different window and the one the
  parked TLT entry sits on.
- **September**: the whole month is +18 td away at its opex. Nothing entered today reaches it.

## 4. Watchlist, all 21 active entries

Verdicts from `02_watchlist_verdicts.py`. Every one is a PASS today, and each reason is a
number rather than an absence.

| # | entry | today's number | verdict |
|---|---|---|---|
| W0 | TLT from the NFP close | 2026 is midterm; the trigger is the cycle year | PASS, arms 2027-01 |
| W1 | LQD against HYG at joint 52w extremes | the state IS live (HYG -0.23%, LQD +0.56%) but the trigger is >=8 declustered episodes and it is still 4 since 2007 | PASS |
| W2 | SVXY overnight into CPI | the next CPI is +13 td, outside the horizon cap | PASS |
| W3 | GLD on a miner-led thrust | GDX r5 98.4 and GLD r5 94.0 both fire; the fourth condition added on 08-21 does not - GLD is **-14.63%** off its 52w high against the >-10% rung | PASS |
| W4 | XLE on a crude one-day thrust in [5,6)% | USO 1d **+0.07%** | PASS |
| W5 | TLT with the IG complex pinned, tight rung | IEF +0.70% and LQD +0.56% both clear; **TLT +0.86% against the <=0.5% rung** | PASS, the closest it has been. C10 asks the translation question instead |
| W6 | SPY on a skew spike | SKEW r5 rank 82.1 against >=95, and midterm blocks it regardless | PASS, arms 2027-01 |
| W7 | crude thrust from a deep base | USO r5 79.4 (needs >=90), r63 29.4 (needs <=20) | PASS |
| W8 | IHI at a 21d rank of 100 out of a drawdown | r21 rank **99.6**, and the reference-class blocker (family-wise p 0.933) stands | PASS |
| W9 | FXI five-day break inside an intact thrust | FXI r5 rank **87.7**, the opposite of the <=20 trigger | PASS |
| W10 | TLT November month-position | parks to a date, roughly 2026-11-05 | PASS |
| W11 | short SPY at a 52w high with TLT at a 52w low | SPY -1.56% off its high against a <=0.5% rung | PASS |
| W12 | TLT month-end ME-9, ungated | **August's ME-9 was 2026-08-18 and is gone; today is ME-5** - and TLT is +0.86% off its low against the >3% trigger | PASS. C1 asks the entry-offset question the entry never ran |
| W13 | SPY on a vol pop inside a calm tape | VIX 21d rank 12.7 clears; VIX 1d **-5.50%** against the >=+5% pop | PASS |
| W14 | gold on an unconfirmed rate rise | DX 21d rank 0.4 clears easily; the 21-session yield rise is **+0.035pt** against the +0.20pt floor | PASS. C9 asks the LEVEL question this rank-and-magnitude entry does not encode |
| W15 | tech against healthcare on a rotation gap | the one-day XLV-XLK gap is **+1.18pp** against >=+3.0 | PASS |
| W16 | short the dollar on a rate rise | TNX 21d RETURN rank 49.2 against >=65 | PASS |
| W17 | crude through Jackson Hole at JH-6 | JH-6 was 2026-08-20 and is gone; XLE is -0.17% off its high, which the entry's own condition forbids | PASS |
| W18 | short TLT after a big up day at the low | TLT 1d **-0.35%** against >=+1.5% | PASS |
| W19 | KRE against XLF on a bank breadth washout | KRE r5 rank 6.0, so the state is live; the trigger is a cost threshold on history that one episode cannot move | PASS |
| W20 | HYG across Jackson Hole at JH-5 | JH-5 was 2026-08-21 and is gone; the anchor was closed on credit the same day | PASS |
| exp | industry breadth with the trend BROKEN | closed by the 2026-08-21 check; prune on rewrite | EXPIRED |

## 5. Scoreboard read before selecting

Four graded ideas lifetime (avg +0.372R, all four positive): event_fingerprint 2 at
+0.622R, interaction_cell 1 at +0.146R, relative_value 1 at +0.099R. That is a handful
and not a signal, so no axis is up- or down-weighted on it. Recorded so the read is on
file.

## 6. Candidates selected

Ten, over six axes and seven asset classes. Two are event-anchored (C1, C2) and eight are
price-state anchored, which is the crossing the 2026-08-07 failure was about.

| id | candidate | axis | class |
|---|---|---|---|
| C1 | Long TLT MOC at ME-5 into the August month-end close | event_fingerprint | rates |
| C2 | Long SPY MOC at ME-5 into the month-end close, i.e. the last week of August | event_fingerprint | us_large |
| C3 | Long copper on a five-day thrust to a fresh 52-week high (FCX, COPX) | interaction_cell | other_metals |
| C4 | Long XLI against the leading cyclical on XLI's r5 rank <= 5 while the peer prints a 52w high | relative_value | sectors |
| C5 | The 10-year yield AT a 52-week high as a LEVEL trigger, traded on the curve | instrument_translation | rates |
| C6 | Cross-sectional new-high breadth while the index itself is off its high | flow_mechanics | us_large |
| C7 | The energy z10 >= 2 cluster, five names thrusting at once | interaction_cell | energy |
| C8 | Tech absent from the entire new-high list: SPY against QQQ | inversion | us_large |
| C9 | Gold at a 21-day rank >= 95 while the 10-year yield prints a 52-week high | inversion | gold_miners |
| C10 | Long IEF rather than TLT, with the IG complex pinned at the rung that actually fires | instrument_translation | rates / credit |
