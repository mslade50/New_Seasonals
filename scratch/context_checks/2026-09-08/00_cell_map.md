# Cell map — run 2026-09-08 (asof session 2026-09-08 Tue, next session 2026-09-09 Wed)

Sweep: 1217 cells scanned, 103 fired (72 event / 31 price). BH crit p 0.0006, 3 pass.
Prices FRESH (core bar 2026-09-08). Both lanes live.
Cycle: midterm year. Next session is the 6th trading day of September, 15 td from month end.

Tape summary so I am not writing from memory: ^TNX 4.806 CLOSED AT ITS 52-WEEK HIGH,
^FVX likewise, TLT 1.43% off its 52w low, IEF 0.34% off, LQD 0.25% off. SPY -0.55,
^GSPC -0.58, ^DJI -1.18, QQQ -0.08, IWM -0.45. CL=F +3.03 (21d +14.8, z10 +2.33).
HG=F at a 52w high. GC=F -1.71. USDJPY -1.42 (63d rank 0.4). VIX 15.72 +2.75,
VIX3M 18.39 +4.43, ^MOVE +4.16. KC=F -10.58.

## Calendar inside the next five sessions

| when | event | verdict |
|---|---|---|
| Wed 2026-09-09 (next session) | nothing scheduled | n/a — the next session is a blank calendar day, so the tomorrow lane has to come from position-in-the-week cells and the run-up to Thu/Fri |
| Thu 2026-09-10 (+2 td) | PPI | DRILL — the k2 anchor is TODAY, so h=1 is tomorrow and h=2 is the print. CL=F is the only subject with |t|>2 |
| Fri 2026-09-11 (+3 td) | CPI | DRILL — top tier. Also a CPI landing on a FRIDAY, which is unusual enough to be its own cell |
| Wed 2026-09-16 (+6 td) | FOMC decision + VIX expiry | SKIP(outside the 5-session window; k1..k3 anchors have not fired yet). Calendar section only |
| Fri 2026-09-18 (+8 td) | opex + quad witching | SKIP(outside window). Calendar section only |
| Tue 2026-11-03 (+40 td) | midterm election | SKIP(far). Not mentioned |

## Event lane groups

### E:ppi — session 2 td before a PPI, n_anchors 318
- **CL=F** n=310 h1 +0.321% hit 55.8 t=2.37 edge +0.327 sign p 0.023 era-stable — **DRILL**.
  The only subject in the group clearing |t|=2, and crude is simultaneously the most
  stretched thing on the tape (z10 +2.33, 21d +14.8%). Needs a control and a check of
  whether the cell survives conditioning on crude already being extended, because the
  live setup is not the average anchor.
- EURUSD=X n=269 t=1.93 sign p 0.056 — SKIP(edge +0.063% is inside any plausible FX
  spread and the cell has no mechanism; a 6bp mean is not a sentence Scott needs).
- HG=F t=1.52 era_stable False — SKIP(era-unstable and a 0.09% edge).
- ^VIX n=318 h1 +0.457% t=1.30 but record 153-163 DOWN — SKIP(mean and median disagree,
  the mean is carried by right-tail spike days; a cell whose sign flips between the two
  statistics is not publishable as a directional claim).
- IWM, SI=F, DX-Y.NYB, ^TNX, TLT, IEF, HYG, ^GSPC, GC=F, SPY, QQQ, EEM, JPY=X, NG=F —
  SKIP(all |t| < 1.0 and edges under 0.12%; nothing to say).

### E:cpi — session 3 td before a CPI, n_anchors 318, TOP TIER
- Whole-group read: nothing clears |t| = 2. Best are DX-Y.NYB t=-1.65 and GC=F t=1.43.
- **GC=F** n=310 h1 +0.079% hit 57.1 record 177-131 up sign p 0.0072 era-stable — **DRILL**.
  The mean is trivial but the RECORD is not: 57% up over 310 anchors with an exact sign
  p under 0.01. That is a hit-rate cell rather than a magnitude cell and it should be
  written as one or dropped. Also gold fell 1.71% today, so the live state is off-anchor.
- **^GSPC / SPY / QQQ** h1 t ~0.6 but records 179-139 / 178-139 / 179-135 up, sign p
  0.014-0.019 — **DRILL** as one cell, not three. Same underlying question as gold: a
  drift-sized mean with a lopsided record over 318 anchors. Check whether the k3 run-up
  is distinguishable from the unconditional up-day base rate (~53-54%), because if it is
  not, this is the classic "equities go up" non-finding and it dies.
- DX-Y.NYB t=-1.65 era_stable False — SKIP(era-unstable, 4bp).
- SI=F, NG=F, JPY=X, EURUSD=X, CL=F, HG=F, ^VIX, ^TNX, EEM, TLT, IWM, IEF, HYG —
  SKIP(|t| < 1.2, no record worth quoting).

### E:weekday_month — Wednesdays in September, n=110
Pre-specified? NO. This is a bare 12x5 grid the engine fires every day, so it is a swept
family and everything here owes multiplicity. bh_pass is False for all 18 subjects.
- **CL=F** n=110 h1 +0.569% hit 57.3 t=2.48 edge +0.575 era-stable — **DRILL**, jointly
  with the E:ppi CL=F cell. Two structurally different anchors pointing the same way on
  the same instrument is the only reason this is worth any ink; alone it is a grid cell.
- **^VIX** n=110 h1 -1.215% hit 37.3 record 41-69 down sign p 0.0049 t=-2.16 era-stable
  — **DRILL**. Strongest sign-test in the group. Needs the honest framing that spot VIX
  drifts down on most days, so the claim has to be an EDGE over the all-Wednesday and
  all-September base rates, not a level.
- HG=F t=1.78 — SKIP(0.21% edge, no supporting cell).
- NG=F h5 +2.94% t(h1)=1.24 record 52-57 down — DRILL(folded into the seasonal_doy natgas
  question below, not on its own).
- SPY 58.2% hit sign p 0.052, EEM, IEF, TLT, ^GSPC, IWM, SI=F, JPY=X, GC=F, HYG, QQQ,
  DX-Y.NYB, ^TNX, EURUSD=X — SKIP(|t| < 1.2; the SPY record is suggestive but it is the
  same "equities up" cell as E:cpi and I am only paying for that question once).

### E:seasonal_doy — same trading day of year +/-2, Sep 09
- **TLT** — **SKIP(repeat_blocked: published 2026-09-03, twice, number unmoved).** This is
  the one hard novelty block tonight and it is respected.
- **NG=F** all-years n=25 mean +1.351% median +0.862% record 18-7 up sign p 0.0216;
  midterm n=6 mean +1.974% 5-1 up — **DRILL**. Natgas is 61% below its 52-week high, the
  seasonal cell and the September-Wednesday cell (h5 +2.94%) point the same way, and the
  mechanism (injection season ending, first cold-risk pricing) is nameable. N=25 caps this
  at suggestive and the midterm split at n=6 is an anecdote at best.
- ^TNX h5 midterm 5-1 up mean +2.619% n=6 — SKIP(n=6 anecdote AND it duplicates the bond
  story I am already drilling from the price side with far better N).
- HYG 13-6 up sign p 0.084 n=19 — SKIP(anecdote-tier N, edge unremarkable).
- SPY/^GSPC h5 17-9 up sign p 0.084 — SKIP(same equity-drift question again).
- QQQ, IWM, IEF, GC=F, SI=F, HG=F, CL=F, DX-Y.NYB, EURUSD=X, JPY=X, ^VIX, EEM —
  SKIP(sign p > 0.15 on n<=26, nothing separable from noise).

## Price lane groups

### P1 / P1b:new_52w_high — AUDNZD=X only
SKIP(AUDNZD is a cross nobody in this audience carries, n=22/14, era_stable False, and the
edge is 0.12-0.18%). Noted, dismissed. Same for the AUDNZD entries dropped by the cap.

### P4:z10_extreme, stretched down — KC=F, CHFJPY, NZDJPY, EURJPY, GBPJPY
- KC=F n=90 +0.314% 54-36 up sign p 0.036 era-stable — DRILL(folded into the coffee
  question below; the interesting thing about coffee tonight is the -10.58% session, not
  the z10 cell).
- The four yen crosses — **SKIP(published 2026-09-07 as the seven-cross breadth nugget,
  rank 4, and 2026-09-06 as the yields-up/yen-up cell).** The state is unchanged and
  re-telling it with tonight's slightly deeper percentile is exactly the countdown failure
  the novelty rule bans. Not blocked by fingerprint because last night's was a custom
  composite, so I am blocking it by hand and saying so.

### P4:z10_extreme, stretched up — ^BVSP, CL=F, ZS=F
- CL=F n=171 h1 -0.041% t=-0.21 — DEAD as a directional cell, but it is the right CONTROL
  for the crude drill: it says extended crude has no next-day edge on its own, which is
  what makes the calendar-conditioned version worth testing rather than assuming.
- ^BVSP t=0.90, ZS=F t=-0.15 — SKIP(no edge).

### P5:rank5_extreme, bottom 5% — JPY=X, KC=F, EURJPY, CT=F, CHFJPY, GBPJPY, NZDJPY
- JPY=X n=328 t=2.12 sign p 0.0005 **bh_pass TRUE**, EURJPY n=288 sign p 0.0004 bh_pass
  TRUE — SKIP(yen, published 09-07, see above). Recorded here because these are two of the
  only three BH survivors in tonight's sweep and dismissing them silently would misrepresent
  the map. They are strong and they are stale.
- KC=F n=315 +0.239% t=1.89 but record 161-150, sign p 0.37 — SKIP as its own cell(mean
  without a record is a tail artifact). Feeds the coffee drill as background.
- CT=F t=-1.09 — SKIP.

### P5:rank5_extreme, top 5% — EWZ
SKIP(t=0.52, h5 -0.335%, era_stable False; Brazil ripping is a real tape fact but the cell
says nothing).

### P5b:rank21_extreme, top 5% — ZC=F, ZW=F, ZS=F
- ZC=F n=426 +0.196% t=2.12 era-stable — SKIP(grains momentum is a genuine cell but corn is
  outside anything this reader touches and the edge is 0.16%; three grains at 52w highs is
  worth one clause in the crude/copper reflation line, not a nugget).
- ZW=F, ZS=F — SKIP.

### P5b:rank21_extreme, bottom 5% — JPY=X + four crosses
SKIP(yen, published 09-07). JPY=X bh_pass TRUE noted.

### P6:two_atr_day, down — FXI, KC=F, USDCNY=X
- **FXI** n=88 h1 +0.568% hit 56.8 t=1.76 edge +0.527 era-stable, session -2.45% — **DRILL**.
  Half-decent N, era-stable, a large edge, and China is the one non-yen risk asset that
  broke today. Check concentration (2015 and 2022 could carry the whole mean).
- **KC=F** n=54 h1 +0.348% 27-25 sign p 0.554 — **DRILL** but the drill is about the
  magnitude of today's -10.58% session, not this cell, which is a coin flip.
- USDCNY=X n=112 record 49-48, sign p 0.92 — DEAD.

### P6:two_atr_day, up — CT=F
SKIP(t=0.36, 45-47, era_stable False).

### P7b:down_streak — KC=F
SKIP(n=186, edge -0.007%, record 90-96 down; the streak adds nothing over the crash itself).

## Cells I am building by hand, not from the sweep

These are NOT in cells_index. I selected them off the tape extremes, so they are swept in
spirit and owe the same multiplicity honesty as the grid cells. Saying so here is the point.

- **^TNX closing at a 52-week high** — DRILL. The single most relevant fact about tomorrow:
  the whole curve (^IRX rank 92.5, ^FVX and ^TNX at 252d maxima) is at the year's highs with
  a CPI on Friday. No P1 trigger fired for ^TNX because the trigger requires the first new
  high in 30+ days and yields have been grinding up. Highest priority drill of the night.
- **TLT / IEF / LQD within 1.5% of 52-week lows simultaneously** — DRILL, as the same
  question from the price side. Breadth version: how often is the whole US duration complex
  pinned at the lows, and what follows.
- **^DJI -1.18% while QQQ -0.08%** — DRILL. A 1.1pp Dow-minus-Nasdaq gap on a down day.
  ^DJI 21d rank is 9.1 against QQQ's 30.6, so this is a persistent divergence, not one bar.
- **VIX3M/VIX ratio** — DRILL. VIX3M +4.43% and +4.91% over 5d while spot VIX is -3.79%
  over 5d. Term structure steepening into a CPI with spot vol at 15.7. Distinct from the
  09-07 Labor Day VIX3M nugget (that was a calendar cell about the holiday, this is a
  ratio-level cell).
- **HG=F at a 52-week high with GC=F -1.71%** — SKIP unless the bond drill needs it. The
  copper-gold-yields reflation triangle is a nice story and I have no room for a fourth
  version of "yields are up".

## Drill queue

1. `01_tnx_52w_high.py` — ^TNX at a 252d max: forward SPY/TLT/^TNX, era split, concentration
2. `02_duration_at_lows.py` — TLT+IEF+LQD all within 1.5% of 52w lows, and the CPI crossing
3. `03_cpi_friday.py` — CPI landing on a Friday: how rare, and the k3 run-up cell honestly
4. `04_crude_ppi_wednesday.py` — CL=F September-Wednesday and PPI-k2 cells, controls, overlap
5. `05_dow_nasdaq_split.py` — ^DJI down 1%+ with QQQ flat, forward behaviour
6. `06_vix_term_steepen.py` — VIX3M/VIX ratio rising with spot VIX low, into the print
7. `07_natgas_september.py` — NG=F seasonal doy + September Wednesday, one honest read
8. `08_fxi_2atr.py` — FXI 2-ATR down day concentration check
9. `09_coffee_crash.py` — how historic is a -10.58% KC=F session

## LATE FINDING (11_roll_seam_check.py): tonight's commodity outliers are ROLL SEAMS

The 2026-09-06 brief excluded grains, softs and metals for exactly this reason on the
09-04 bar. It recurs tonight, worse. Today's volume against each contract's trailing
20-day median:

    KC=F  -10.58%   24,202 vs      180   =   134x
    CT=F   +4.89%   18,621 vs       24   =   776x
    GC=F   -1.71%  214,258 vs      666   =   321x
    SI=F   -0.56%   49,930 vs       35   =  1427x
    HG=F   +1.41%   74,697 vs      819   =    91x
    ZS=F   +1.78%  126,269 vs    8,488   =    15x
    CC=F   -2.55%   26,676 vs    1,967   =    14x

A trailing median of 180 contracts in coffee and 35 in silver is a DYING contract. The
continuous series switched front month today, so the close-to-close "move" is the
calendar spread, not a price. Consequences, all applied:

- **09_coffee_crash.py is VOID.** It scored -10.58% as the 3rd worst coffee session in
  26 years. It is not a session at all. KILLED, not published. The same script's -8%
  history is likely contaminated at other roll dates too and I am not repairing it
  tonight.
- **HG=F "at a 52-week high" and GC=F "-1.71%" are both VOID** as tape facts. Neither
  goes in the brief, including as background colour in another nugget.
- The engine's P4/P5/P6/P7b cells on KC=F, and the P5b grain cells on ZS=F, inherit the
  contamination. They were already SKIPped above on their own weakness; this is a
  second and stronger reason.
- **CL=F is CLEAN**: 403,832 vs a 248,310 median = 1.63x, the bar contains the prior
  close, and no roll inside the trailing 21 sessions. The crude drill stands.
- **NG=F is CLEAN**: 1.58x. ZW=F 2.69x and ZC=F 1.74x are clean enough to mention.
- Nothing in the equity, rates, FX or vol lanes is affected.

This is why the coffee line does not appear in tonight's brief despite being the largest
number on the tape. It goes in the footnote instead.

## Additional novelty blocks found by reading the last two briefs

- **^TNX at a 52-week high: SKIP(published 2026-09-06, nugget 3).** That brief already
  ran the S&P's 21-session deficit against its neighbourhood off this exact anchor.
  01_tnx_52w_high.py and 02_yields_high_gold.py were written before I checked, and their
  results are recorded here as kills rather than published: gold's h21 cell looks strong
  full-sample (26-11 up, +1.99%, sign p 0.0100) but its 2018+ edge over the SAME-ERA
  all-days mean is +0.06pp, so it is gold's own bull market and not a yield effect;
  IEF's h5 bounce is 78.1% pre-2018 against 54.1% after; TLT's is 68.8% against 48.6%.
  All three DEAD post-2018.
- **The yen complex: SKIP(published 2026-09-07 as the seven-cross breadth nugget, and
  2026-09-06 as the yields-up/yen-up reflex).** Tonight it holds two of the sweep's three
  BH survivors and it is still not eligible.
- **09_duration_at_lows.py: rare state, no signal.** TLT+IEF+LQD all within 2% of 52-week
  lows has occurred in six calendar years since 2003 (2006, 2013, 2018, 2022, 2023, 2026;
  219 sessions, 30 of them this year). Every forward edge is negative or inside noise:
  ^GSPC h21 +0.34% against a +1.20% neighbourhood, TLT h21 8-11. Publishable only as an
  explicit null, and only if the brief has room.

## Drill results and final verdicts

**01/02 ^TNX at a 252d max — KILLED and also already published 09-06.** Gold h21 26-11
up (+1.99%, sign p 0.0100) full sample, but the 2018+ edge over the SAME-ERA all-days
mean is +0.06pp. IEF h5 78.1% pre-2018 / 54.1% after. TLT h5 68.8% / 48.6%. Nothing here.

**03/03b/03c CPI — PUBLISH, this is the headline.** The run-up is where the drift lives,
not the print. ^GSPC from the k3 anchor (today) to h2: non-Friday CPI n=246, 145-101 up
(58.9%), sign p 0.0030, mean +0.124% against an unconditional 2-session base of 55.6% /
+0.065%; 2018+ n=92, 58-34 (63.0%), sign p 0.0080, +0.265%. The print session itself in
2018+ is 52-40 (56.5%), sign p 0.126, nothing. Base rates: h1 53.7%, h2 55.6%, h3 56.8%.
SEPARATE NUGGET: the Friday cut. 74 Friday CPIs since 2000, this is the 75th. Friday
run-up 40-33 (sign p 0.241); the pre-2013 Friday print-day negative (-0.443%, n=49) is
80% carried by four sessions and dies after: 2013-2017 exactly 0.00%, 2018+ 6-4 up on
ten prints. So the Friday split is a caution with no number behind it, and saying that
is the honest nugget. PRE-SPECIFICATION: the pre-CPI drift is a famous pre-specified
hypothesis and does not owe the sweep a correction; the WEEKDAY split is mine and was
chosen because Friday is what is scheduled, not by scanning five weekdays for the best.

**04 crude — SKIP the edge, PUBLISH the deflation.** PPI-k2 reproduces exactly (n=310,
+0.321%, 55.8%, t=2.37, 173-135, sign p 0.0233, all-days control -0.006%, top-2 only 19%
of total). Then every condition that is TRUE TOMORROW removes it: September anchors n=26
go 12-14 DOWN at +0.223%; midterm years n=81 +0.139% t=0.54 against non-midterm +0.385%
t=2.43; crude at z10 >= 2 (tonight +2.33) n=9, -0.484%, 3-6. Also era-lopsided:
pre-2018 t=0.88, 2018+ t=2.91.
The September-Wednesday companion is a HARD KILL and therefore gives no independent
support: it ranks #4 of the 60 weekday x month cells, nominal two-sided p 0.0131 x 60 =
0.79, its sign test misses 5% (63-45, p 0.0755) while its t is 2.48, the same grid
produces September TUESDAYS at -0.559% t=-2.24, and the live state inverts it (z10 >= 1
at the anchor: n=15, -0.474%, 6-9). Overlap between the two cells is only 7 anchors, so
they are near-disjoint samples, but they call the same instrument on the same session and
quoting both would read as two supports for one claim.

**05 Dow vs Nasdaq — PUBLISH as a null.** Cell fires today (^DJI -1.18%, QQQ -0.08%,
spread +1.09pp). 78 raw days since 2000, 57 episodes at 10td, crisis-loaded (24 in
2000-2002, 13 in 2020). The forward SPREAD is nothing: h1 +0.156pp (27-29, p 0.656),
h5 -0.113pp (26-30, p 0.748), h10 +0.029pp, against an unconditional h5 of +0.085pp and
a local control of +0.091pp. The mirror cell (Dow beating a falling Nasdaq by 1pp+, 238
episodes) is +0.026pp at h5. Zero in both directions. The index-LEVEL bounce that looks
good is a down-day bounce, not a divergence effect: any ^DJI <= -1% episode gives +0.157%
h1 on 338 episodes, so the split condition adds +0.124pp on n=56. The 2018+ slice
(n=22, h5 +1.244%, 17-5, sign p 0.008) is an era split with no mechanism and is PARKED,
not published.

**06 VIX term structure — SKIP, and my premise was wrong.** The state file's ^VIX
ret_1d of +2.75% is computed across a PHANTOM Labor Day bar (see below). The curve
FLATTENED today: ratio 1.2120 -> 1.1698, a 1d change at the 11.5th percentile. The
ratio level sits at the 55.6 trailing-252d rank, mid-range, so Cell A (top decile +
VIX < 17) DOES NOT FIRE tonight. For the record its ^VIX leg is strong and era-stable
(h21 55-16, sign p < 0.00001, both eras t > 3.3) and is simply not live. Nothing about
the vol term structure is unusual tonight.

**07 natural gas — PUBLISH at h1 only.** Sep-09 +/-2 doy: n=25, mean +1.351%, median
+0.862%, 18-7 up (72.0%), sign p 0.0216, t=1.69, against all-September +0.436% (n=533),
the Sept 4-14 window +0.466% (n=192) and all-days +0.066%. Era-stable (pre-2018 +1.147%
on 17, 2018+ +1.783% on 8 with 7 of 8 up). CAVEATS THAT MUST SHIP: the MEAN is two years
(2009 +11.4% and 2022 +10.0% = 63% of the +33.8pp total), so the RECORD and MEDIAN are
the claim and the mean is not; dropping the biggest leaves 17-7 (p 0.032), two leaves
16-7 (p 0.047). h=5 is DEAD (+1.570% against its own month's +2.535%) and is not
published. Midterm n=6, 5-1, sign p 0.1094, anecdote tier. Live haircut: gas is 61.1%
below its 52-week high and the analogue years that broken are n=4, 2-2, while the 21
years above -60% are 16-5 (sign p 0.013).
BONUS KILL: NG's September-Wednesday h5 of +2.940% (t=3.22, 68-42, p 0.0078) is the
MONTH wearing a weekday label. September non-Wednesdays are +2.420% (n=422, t=5.71) and
the top four of the 60 grid cells at h5 are all September. Weekday framing dropped.

**08 FXI — SKIP.** Reproduces exactly (n=88, +0.568%, t=1.762) and then dies. The record
is 50-38, not the engine's 50-37 (the engine dropped a loss), exact sign p 0.120. Top-2
episodes are 2008-09-17 and 2008-09-29 = 50% of total return; 2008 + 2020 = 85%.
ex-2008 n=83, t=1.21, p 0.190. "Era-stable" holds in SIGN only: 2018+ n=35, +0.233%,
t=0.53. The killing control is that a cruder threshold beats it: any FXI session <= -3%
gives +0.793% at 58.9% on n=248, t=3.06, so ATR-normalising REMOVES information. And
declustered at 10td the cell is +0.078% (39-35, t=0.25), i.e. one 2008 fortnight counted
repeatedly. The honest version is not live anyway: FXI fell 2.45% today, not 3%.

**09 duration at 52-week lows — PUBLISH as a null.** See above.

## SECOND DATA FINDING: ^VIX carries a phantom Labor Day bar

master_prices holds a 2026-09-07 bar for ^VIX (15.30) on a day the US market was CLOSED.
Every sibling US cash index and ETF correctly has no bar: ^GSPC, ^DJI, ^NDX, ^IXIC,
^RUT, ^NYA, ^TNX, ^FVX, ^IRX, ^MOVE, ^SKEW, and crucially ^VIX3M and ^VVIX. Of the 49
universe tickers holding a 09-07 bar, 48 are legitimately open that day (FX, crypto,
futures, foreign cash indices). ^VIX is the only wrong one.

Consequence: the state file reports ^VIX ret_1d +2.75% and ret_5d -3.79%. Both are
computed across the phantom bar. The true move from the last real session (09-04 close
14.53) to today is **+8.19%**, and 5d is +5.36%. Nothing in the brief may quote the
engine's ^VIX 1d or 5d figures tonight. Historical ^VIX cells are barely affected (one
extra row in n=110 to n=318 samples) but the tape block is wrong and it is what a writer
reads first. Footnoted.
