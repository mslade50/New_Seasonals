# Surface map — 2026-09-07 (Labor Day, NYSE closed)

Freshest bar **2026-09-04**. Next session **2026-09-08**. Cycle year **midterm**
(year % 4 == 2). Pipeline green, 7 of 7 components fresh; the only state warnings
are "2026-09-07 is not an NYSE session" and one stale name (LEG).

**The defining fact of this morning: today is not a session.** Everything below
is anchored on the 2026-09-04 close and enters on Tuesday 2026-09-08, the first
session after a four-calendar-day closure. That closure is itself a live event
and it is NOT in `data/macro_events.csv`, so it gets its own lane.

Operational note found while reading the plumbing: `pitch_grammar.build_orders`
stamped `Execute_On` with the asof date verbatim, so a holiday morning would
have produced rows dated 2026-09-07 — a date with no bar. The runner never
places those and `grade_pitch_journal.replay_leg` returns `no_session` forever,
so the whole morning would have been both unplaceable and ungradeable. Fixed
(roll +0 CustomBusinessDay, a no-op on a real session) with two guard tests in
`tests/test_pitch_grammar.py`. First market holiday since the product launched
on 2026-08-06, which is why nothing caught it earlier.

## Three state-file corrections, found by cross-checking the raw bars

1. **`vol_vs_63d` is a VOLUME ratio, not a realised-volatility ratio.**
   `build_pitch_state._metrics_for` computes it off Volume. SPY's 0.69 is
   turnover, not vol. The true SPY 21-day realised-vol ratio against its own
   63-day average is **0.60** (8.2% annualised). Anything reading that field as
   vol compression is reading the wrong series.
2. **^VIX closed 14.53 on 2026-09-04, not 15.0.** The 15.0 in the risk block is
   the signal summary's rounded display.
3. **The production `VIX Range Compression` signal reads OFF**, despite the
   21-day range sitting at the 1st percentile. Its definition
   (`pages/risk_dashboard_v2.py:495-500`) is `pctile < 15 AND vix > 13 AND vix >
   its 20d SMA`, and the failing leg is the last one. No candidate may claim
   that signal is firing.

---

## 0. Regime, and what the systematic book is already doing

| dial | reading |
|---|---|
| fragility 10d-MA 63d (the sizing statistic) | **88.0** as of 2026-09-04, vs 57.2 twenty-one sessions ago |
| raw 21d / raw 63d | 67.2 / 87.9 |
| P/C fear state | **OFF** (equity P/C 10d-MA at the 52nd percentile) |
| exposure leg | **0.0x** — killed by rule 1 (raw 21d 67.2 > 50) |
| trend sleeve | **CASH** (10d-MA 63d 87.21 > 50) |
| fragility signals on | Low Absorption Ratio only (AR 0.344, 9th pctile, near a high) |
| staged scanner signals | **none** |

The book is close to flat by construction and the dial is at the **99th
percentile of its entire 2016+ series** — only about 21 days ever at or above
85, in two episodes (Dec 2021 and Aug-Sep 2026). That has a consequence the
whole morning has to respect: "out of sample on the dial" is available as a
kill against essentially any candidate today. It is a legitimate statement
about SUPPORT and says nothing about the EFFECT, so the honest treatment is
the 2026-08-18 distance-from-extreme gradient, not an empty-bucket argument,
and a cell that is otherwise clean but dial-out-of-sample may ship as a grade C
with the dial named as an unmodelled risk.

Two sleeve trades land inside any horizon a pitch would use:

- **T2 FOMC_MIDTERM_SHORT** — short SPY 10% of NAV ($75k), entered **MOC
  2026-09-10**, exit MOO 2026-09-16. Gate is SPY 21d return rank (252d, lag-1)
  < 50; SPY sits at 31.3, so it should fire.
- **T3 SEP_POSTQUAD_SHORT** — short IWM 15% of NAV ($112.5k), Sep opex MOC
  (2026-09-18) to the Sep last session, skipped if IWM z10 < -1.

Live broker positions are deliberately absent from the pitch state; McKinley
applies his own book when he reads the cards.

---

## 1. Calendar events x asset class

Seven events sit in the [-5, +15] td window. The closure is an eighth the
calendar file does not carry.

| event | date | td from 2026-09-08 |
|---|---|---|
| **market closure (Labor Day)** | 2026-09-05..07 | the entry session IS k=0 |
| nfp | 2026-09-04 | behind us (was the last session) |
| ppi | 2026-09-10 | +2 |
| cpi | 2026-09-11 | +3 |
| fomc_decision | 2026-09-16 | +6 |
| vix_expiry | 2026-09-16 | +6 |
| opex | 2026-09-18 | +8 |
| quad_witching | 2026-09-18 | +8 |

**Structural fact that constrains every candidate: any hold of 3 sessions or
more from 2026-09-08 contains BOTH the PPI and the CPI.** The registry closed
that object on 2026-08-27 — the exact PPI-then-CPI configuration pays +0.406%
against a same-span +0.458%, and the neither-print cell (+0.538%) beats the
both-prints cell outright. Every candidate below owes a print-containment split.

### 1a. The scheduled-event grid — 192 cells run (`02_event_class_*.py`)

Entry MOC on the session k trading days before the event (k = the live
distance, so the entry session is always 2026-09-08), exit either the session
before the event or on it. Controls: own drift, full history, and local
+/-126td ex-anchor. Sign tests are against each instrument's OWN up-rate, not a
coin, because a drifting ETF beats a coin for free.

**Pipeline calibration first:** the Event Sleeve's own T1/T2 window reproduces
exactly — T1 non-midterm +0.432% at 107-52, p 0.0000; T2 midterm -0.349%; T2
midterm with the 21d-rank gate -0.634% at 13-15. The house sign flip is there,
so the rest of the grid is trustworthy.

| class | verdict | the number |
|---|---|---|
| us_large SPY | **EMPTY** | best of 12 cells is cpi/thru at p_dir 0.094 with an edge of **+2 bps** over SPY's own 3-day drift against a 15 bp cost gate. The largest apparent edge (quad/thru, -26 bps) has a +0.657% median — pure tail. |
| us_small IWM | **EMPTY as an event cell** | the quad-witching run-up looks alive Sep-only (20-6, +0.675%, p_coin 0.0047) but the **anchor placebo ranks it 10 of 20** — offsets -10 to -8 score as well or better, so it is "first half of September", not "into quad witching". Sep x midterm is 5-1 at a mean of **-0.318%**. The placebo scan independently reproduces the book's own T3 (offsets +4..+8 run -1.0% to -2.0%). |
| rates TLT, IEF | **EMPTY** | best of 24 cells is ppi/thru/IEF at p_dir 0.052 with an edge of **+2 bps**. TLT's largest edge (quad/pre, -21 bps) has a +0.125% median on a 50-46 record. Pre-FOMC TLT by cycle: midterm 28-21 / +0.03%, non-midterm 73-70 / -0.07%. No cycle structure. |
| credit HYG, LQD | **EMPTY on the scheduled calendar** | best of 24 is vix_expiry/thru/LQD at p_dir 0.158 with an edge of **+0.0 bps**, literally identical to LQD's own drift. (Credit's pulse this morning is on the CLOSURE anchor, section 1b, not on any scheduled print.) |
| gold GLD | **EMPTY** | quad/thru flags at -0.37% (p_dir 0.087) but the median is -0.168% against a +0.372% control on a 42-45 record. Pre-FOMC by cycle: midterm 21-24 / -0.125%, non-midterm 67-61 / +0.211%, p_base 0.78. |
| miners GDX | **EMPTY** | best of 12 is fomc/pre at p_dir 0.120, edge -34 bps, record 74-87, median -0.755%. Its largest edge (quad/pre, -49 bps) has a **+0.455% median**. |
| metals SLV | **PLACEBO-KILLED** | Sep-only into the VIX expiry is 14-6 at +1.26% (p_coin 0.058) and both eras positive, but the **anchor placebo ranks it 8 of 20** — the expiry date does no work, this is first-half-September metals drift, 50% of it in 2008+2009. |
| energy USO, XLE | **ERA-KILLED** | ppi/pre/USO was the best non-vol cell (+0.196%, 137-106, p_dir 0.059, edge +20 bps) and **the entire effect is post-2018**: pre-2018 n=140 mean -0.041% on 71-69; 2018+ n=103 +0.519% on 66-37. Placebo rank 4/20, September-only 11-9, 3.9x cost. XLE/cpi/pre carries an edge of only +10 bps. |
| dollar UUP, DX | **ERA-KILLED** | short UUP into mid-Sep expiry is 105-129 overall (p_dir 0.07) and September-only 6-13, but pre-2018 is 3-8 and **2018+ is 3-5 at a mean of +0.016%**. The whole thing is 2008-2013. |
| international EFA, EEM | **UNDER COST** | cpi/thru/EEM is the cleanest sign test in the entire grid (168-110, p 0.017) and its edge over EEM's own 3-day drift is **+8 bps**. |
| volatility SVXY | **the grid's only pulse** | long SVXY 2026-09-08 MOC into the Sep VIX settle, 13-1 over 14 instances, p_coin 0.0009; anchor placebo gives it the best record in the scan with offsets +5..+8 flipping to -1.7%/-3.3%. **Sent to stage C** (`07_svxy_sep_expiry.py`). |

**Grid charge stated plainly:** 192 grid cells + 78 cycle sub-cells = 270
candidate-generating measurements, and 9 passed a one-sided 0.10 screen where
about 19 would be expected by luck. The screen did not over-produce, and the
cells are not independent (PPI and CPI windows overlap, 14 of 28 September VIX
expiries ARE FOMC days, quad witching is a subset of opex, within-class proxies
correlate above 0.9). Effective independent tests are more like 30-50.

### 1b. The closure anchor — 697 cells run (`01_closure_lane_*.py`)

The one live event the calendar file does not carry, and the class the repo has
never swept it on. Prior closure work (2026-09-04) covered only ^VIX, SPY, IWM
and SVXY. Closure = a consecutive-session gap of >= 3 calendar days; gap == 3
is the ordinary weekend CONTROL (N=1211), gap >= 4 the holiday cell (N=180).

**Nothing in the survey clears its own grid.** Family-wise |t| for p=0.05 is
about 3.6 at 170 cells and 4.0 at 697; one cell clears 170 on its raw t and
none clears it on the control-differenced t, which is the honest null.

| cell | number | verdict |
|---|---|---|
| LQD MOC h=3 | N=163, +0.200%, 63.2% hit, t 3.85; weekend control +0.057%, **excess t 2.36**; 3.3x cost | screen hit. Clears a 170-cell Bonferroni on the ZERO null, which is the wrong null for a coupon-bearing ETF; against the weekend control it clears nothing. Also 0.78-0.91 correlated with IEF/TLT — the four rates+credit rows are about one and a half independent cells. |
| **HYG MOC h=5** | N=130, +0.275%, **66.2% hit**, t 3.26; weekend +0.094%, excess t 1.78; 4.6x cost; era-stable; survives month-turn; **0.38 correlated with LQD, ~0.00 with IEF/TLT** | **sent to stage C** (`08_hyg_closure.py`). The only pulse whose Labor-Day slice agrees (15-4). |
| GLD MOO h=5 | N=149, +0.612%, 62.4% hit, t 3.38, excess t 2.35, **12.2x cost** — the best economics in the survey | **DEAD IN TODAY'S SLOT.** By holiday slot, **September is the worst of the ten: n=21, -0.241%, 42.9% hit, record 9-12, worst -7.44%.** The effect is a December/February/January phenomenon. |
| SVXY MOC h=3 | all-history N=101 +1.076%; **post-2018 basis n=58 +1.231%, 62.1% hit, t 3.03, sign p 0.043**, excess +1.160%; 6.0x cost | survives the vehicle break, which is the check that was expected to kill it, but 0.51 correlated with the HYG cell and sd 5% with a -21.3% worst observation. Folded into the SVXY stage-C brief rather than run twice. |
| long volatility after the closure | ^VIX MOO h=3 is -1.830% (t -2.91) but **the ordinary weekend is -1.712% — edge -0.118%, t vs weekend -0.17** | **EMPTY.** The "vol bleeds after a closure" effect IS the weekend effect; the extra calendar day adds nothing. |
| the runway conditioner | **0 of 170 runway contrasts reach \|t\| >= 2.5 and only 2 reach 2.0, against ~8 expected by chance** | **runway does not condition this lane.** Watchlist 38's arm (runway >= 4) is directionally consistent on vol (^VIX MOO h=2 long-runway -2.20% at t -3.18 vs short-runway -0.35% at t -0.38) but the contrast t is 1.10-1.60, so it is not established. Today is the short half and it is flat, not adverse. |
| IWM | max \|t vs weekend\| across its 10 cells = **0.80**; Labor Day 14-12, 13-13, 15-11 | EMPTY |
| EFA / EEM | max \|t vs weekend\| **0.96** and **0.97** | EMPTY |
| XLE | max \|t vs weekend\| **1.53**, and its best-looking cell is NEGATIVE; Labor Day 13-13 / 14-12 with means inside +/-0.25% | EMPTY |
| SPY | MOC h=1 +0.205% (t 2.45) but weekend +0.077%, **edge t 1.41**; the anchor session itself is a coin flip intraday (MOO h=1 -0.039%, 48.3%); Labor Day exactly **13-13** | EMPTY |
| UUP | Labor Day MOC h=1 and MOO h=2 both print **3-16 at sign p 0.0022** on n=19 — and in a 170-cell grid the expected smallest p is ~0.006, so that is one draw of noise. Its own full-holiday cells at those coordinates are +0.012% and +0.038% | **the clearest search artifact in the survey** |
| GDX / SLV h=10 | control differences -1.65% and -1.49% (t -2.43, -2.40) but the cells' own t is only -1.50 and -1.39 on ~7% sd — the edge is that the weekend control drifts UP | EMPTY at a 3-day pitch horizon |
| USO MOO h=3/h=5 | -0.572% (t -2.05) and -0.679% (t -1.85), 7-8x cost, t vs weekend -2.13/-2.31 | near-miss, parked not pitched: structural roll decay, and 138 anchors at 2% sd is below the survey's own noise floor after the grid charge |

**The gap itself** (Open[k0] / Close[k0-1]): only ^VIX (+5.21% vs +3.01%
weekend, t 4.93) and GLD (+0.218% vs +0.015%, t 2.19) separate. Closure gaps
are only about 1.2x wider in sd than weekend gaps. Neither entry form captures
that leg — a decision made ON the holiday has no pre-closure entry available —
and the gap is uncorrelated with what follows (corr between -0.12 and +0.07 on
all six pulse cells), so there is no gap-momentum story either.

**Labor-Day-only, as a slice:** 25 of 170 cells reach sign p <= 0.10 in either
direction against ~34 expected by chance. **The Labor-Day-specific slice is,
as a whole, quieter than a coin.** No candidate may lean on it.

---

### 1c. The price-state lane — about 750 cells run (`03_pricestate_s*.py`)

Six live states, each measured against own-drift, all-days and local +/-126td
controls, declustered. **Every number below is uncharged for a ~750-cell grid.**

| state | verdict | the number |
|---|---|---|
| **S1 miners high / metal below its 200d** | **EMPTY** | the beta-hedged long-GDX/short-GLD pair pays **-0.111% at h=5 on a 37-47 record, sign p 0.885** across 84 episodes, and is positive at no horizon on any threshold from rank 70 to 90. GDX outright has an edge of **-0.053pp**; its one positive horizon (h=10, +0.966pp) is 29-23 with **66% of the total in two episodes**. The metal gate does discriminate (+0.215 gated vs -0.156 with GLD above its 200d) — it discriminates into a zero. |
| **S2 IG at the floor, HY at the ceiling** | **one artefact, one pulse** | the STRICT rung (LQD at its floor AND HYG at its ceiling) is 41 days in **only two calendar years**; its TLT h=10 7-0 at +1.003% is an overlap artefact (3 independent runs; drop-2026 leaves 4-0, drop-2018 leaves 3-0). The LQD/HYG convergence pair is dead: **h=5 +0.017%, 4-7, sign p 0.887**, under half its own cost. The BROAD rung is P3 below. |
| **S3 vol compression + SKEW bid** | **the conjunction is EMPTY and the compression gate is NEGATIVE** | today sits in the conjunction, which is the SKEW parent's weakest subset. SKEW rank21 >= 95 ALONE pays SPY +0.333% (108-58, p 0.0001) and SVXY +1.374% (59-36, p 0.0117); the conjunction pays SPY +0.215% on 13-11 and SVXY +0.521% on **8-9**. Conjunction minus parent: SPY **-0.075pp**, SVXY **-0.886pp** at h=5. Compression alone carries an SVXY edge of **-1.065pp**. It subtracts at every horizon on both vehicles. |
| **S4 sector washout under an index high** | **pooled family EMPTY** | 9-SPDR fixed effects are negative at ALL five horizons against own drift (-0.074 to -0.305pp) and against SPY (-0.015 to -0.162pp), on 256-963 pooled episodes. Gate attribution finishes it: **without** the SPY-near-high gate the same washout is strongly positive (h=5 +0.222pp, t 2.13, N=1879). XLI's own cell is -0.002% at h=5 (edge -0.211pp, 17-14) — it does not differ from the family, it IS the family. Real estate the same (-0.16 to -0.19pp). One subsector is the exception: P2 below. |
| **S5 energy thrust at a 52-week high** | **COMPREHENSIVELY EMPTY, and the repo's prior kills replicate** | every vehicle is negative against its own baseline: XLE h=10 edge **-1.275pp**, XOP **-1.300pp**, XLE-minus-SPY -0.005pp at h=5. Only USO is nominally positive (h=5 +0.447%) on a **36-44 record with a -0.292% median**. Not one of 124 cells clears cost with a record. |
| **S6 the dial at an extreme** | **direction without a record; the CROSS-SECTION is the live object** | SPY outright at the top decile has an h=10 edge of -1.280pp on an **18-20** record (live conjunction 8-8) — the book's own PIT t of -0.23 reproducing. The damage concentrates in **IWM** (h=10 edge -2.137pp, short side 25-14, sign p 0.054). That asymmetry is P1 below. |

**The four pulses this lane produced**, all live today:

- **P1, long SPY / short IWM at the dial's top decile** — h=5 71 episodes **+0.517%, 66.2% hit, 47-24, sign p 0.0043**, edges +0.538pp / +0.578pp; h=10 +0.998% on 26-13. Survives every overlap form (RUN-1st 9-4, **YEAR-mean 8-0 at sign p 0.0039**), LOYO-stable, monotone in the threshold, top-3 episodes **-14%** of total. **Sent to stage C** (`10_dial_spy_iwm.py`). Nobody in this repo has ever tested the dial cross-sectionally; every prior test was direction or sizing.
- **P2, long ITA at a 21d rank <= 10 with SPY near its high** — h=10 29 episodes +1.223%, bootstrap 0.010, and it gets STRONGER under stricter declustering (gap=63 +2.343% on 12-3; RUN-1st +1.254% on 30-14, p 0.011; YEAR-mean 9-1). 24.5x cost outright. ITA is the deepest reading on the tape at rank21 2.8. **Not checked — it is one subsector plucked from a scan whose parent family (S4) is empty, and the reference-class test that would settle it did not fit in the morning.** Goes to the watchlist.
- **P3, long TLT at a 2% floor with HYG within 1% of its ceiling** — h=5 36 episodes +0.208%, **72.2% hit, 26-10, sign p 0.0057**, and the credit gate does all the work (TLT-at-floor alone is negative at every horizon). It **inverts under regime-scale declustering** (gap=63 h=5 -0.500%) and the top-3 episodes are -120% of total P&L at h=5, so the biggest moves are the losers. Only the h=10 RUN-1st form holds. Adjacent to C1 and folded into that verdict.
- **P4, the SKEW rank21 parent** — dead for this cycle year: the registry killed the r21 form on 2026-09-03 with **midterm -1.106% against +0.536%**, and today sits in the parent's weakest subset anyway.

Correction this lane produced: the **VIX 21-day range percentile recomputes to 2.4 on a trailing-252 basis and 5.0 expanding**, not the 1st percentile the risk block displays. Still bottom decile, different denominator. Also **JNK, XES and RSP are absent from `master_prices.parquet`**, so the high-yield cross-check, the oil-services leg and the equal-weight leg were dropped rather than proxied.

---

## 2. Tape extremes, by class

Measured on 2026-09-04. Whole tape sorted, not a lookup of names I walked in
with.

**Only 14 of 218 names are within 1% of a 52-week high:** BNY, HPQ, STT, VLO
(all at 0.00), DBC -0.19, EWJ -0.19, ^TNX -0.25, SVXY -0.27, HYG -0.41,
EFA -0.42, DE -0.69, XLF -0.79, VZ -0.95, SPY -0.99.

**13 names within 3% of a 52-week low:** MCD 0.00, LQD 0.25, IEF 0.44, TJX 0.59,
UVXY 0.69, CMS 0.74, NKE 0.96, ^VIX3M 1.09, TLT 1.44, PEG 1.50, LOW 2.31,
VMC 2.39, VFC 2.78.

**73 of 218 below their 200d SMA**, including GLD, SLV, UNG, TLT, IEF, LQD,
META, AVGO, ORCL, COST, HD, WMT and the whole utilities/staples block.

| class | the extreme |
|---|---|
| us_large | SPY -0.99% off its 52w high, +8.41% above its 200d, 21d rank 31.3, realised vol 8.2% ann = 0.60x its own 63d average. QQQ -3.54%, DIA 21d rank 15.1. |
| us_small | IWM -2.98% off its high, 21d rank 21.4, z10 -0.53. |
| rates | **^TNX 4.784%, 0.25% off its 52-week HIGH**, 21d rank 69, +9.74% above its 200d. TLT +1.44% above its 52w LOW, IEF +0.44%, both below their 200d. |
| credit | **HYG -0.41% off its 52-week HIGH while LQD sits +0.25% above its 52-week LOW.** Duration at the floor, spread product at the ceiling. |
| gold/miners | GDX +18.28%/21d, +25.90%/63d, +10.54% above its 200d; NEM 21d rank 90.5. **GLD is 2.09% BELOW its 200d and 17.97% off its high.** The equity leads and the metal does not confirm. |
| other metals | SLV 43.35% below its 52w high and 8.42% below its 200d, but +7.11% over 21d. XME 21d rank 55.2. |
| energy | **USO +19.42%/21d, 5d rank 87.7. DBC at a 52-week high. XOP 21d rank 91.7, XLE 83.3 and 1.60% off its high, +17.58% above its 200d.** The leading complex. |
| dollar/FX | UUP 21d rank 32.1, DX-Y.NYB 31.0 and flat on its 200d. Quiet. |
| international | EWZ 5d rank 95.2 with z10 +1.60, EWJ -0.19% off its high, EEM 21d rank 66.3, FXI 12.47% off its high. |
| volatility | **^VIX 14.53, 53% below its 52w high, 21-day RANGE at the 1st percentile. ^VIX3M only 1.09% above its 52-week LOW. ^SKEW 21d RETURN rank 98.0. SVXY 0.27% off its 52w high, z10 +1.18.** |

**Sectors and industries.** XLE 21d rank 83.3 at the top; XLI **7.1** and ITA
**2.8** at the bottom, with XLRE/IYR/VNQ near 16 and XLY 27.4. Biotech runs
underneath (IBB 63d rank 96.8, +25.83%; XBI 86.1). SMH has a 63d rank of 5.2
and is still 18.8% above its 200d.

**Breadth: 66.5% of the live 218-name tape above its 200d SMA but only 37.2%
with a 21-day rank above 50** (68.6% on a fixed, survivorship-controlled
191-name subset). A tape broadly intact on a one-year view and broadly stalled
on a one-month view, one percent from a high.

**Verdict on the two most eye-catching vol readings, both dismissed with a
number:**
- ^SKEW's 98.0 is a 21-day RETURN rank, i.e. a rebound off a low, not a level.
  The registry killed that exact form on 2026-09-03: skew r21 >= 95 alone pays
  +0.333% over 166 episodes at t 2.29, adding range compression discards 140 of
  166 to leave +0.094% at an edge of -0.097pp, and **midterm pays -1.106%
  against +0.536%**. Dead for this cycle year.
- The 1st-percentile VIX 21-day range sits BELOW every band the compression
  work has found tradeable: watchlist 34's bimodal result puts (0,5] at
  **-0.096% over 25 anchors at 13-11** while (5,10] pays +1.465% and (10,15]
  +2.034%. Today is in the dead half.

---

## 3. Flow mechanics — what is actually measurable here

Honesty line the skill requires. The repo's positioning data is thinner than
CLAUDE.md implies:

- `data/option_surface_history.parquet` — **1 row**, dated 2026-08-05.
- `data/option_positioning_history.parquet` — **90 rows, all 2026-08-05.**
- `data/iv_history.parquet` — ends **2026-08-11**, nearly a month stale.

"Accruing since 2026-08-05" is not what happened: those files captured one day
and stopped. **Dealer gamma, term structure and IV history are NOT available
for any check this morning.** A flow idea is measurable only through the expiry
calendar and CBOE put/call, and any flow thesis needing more must say so.

CBOE put/call at 2026-09-04 (10d-MA, trailing-252 percentile): index 0.951 at
the **2.4th**, spx 1.099 at 7.6, etp 1.029 at 27.9, total 0.833 at 23.5,
equity 0.588 at 51.4. Index hedging demand at a trailing-year low with
single-name mid-range — **that exact cell was killed 2026-08-26** (the
index-P/C-low gate is 289 days on its own and adding the mid-range equity leg
discards 184 of them, filtering in the wrong direction). Not re-opened.

---

## 4. Seasonal and cycle cells (`04_seasonal_cycle_cells.py`)

Recomputed from raw bars; the state file's seasonal board is stale (asof
2026-08-05) with no live tickets, so it is not used. 26 Labor-Day anchors,
2000-2025.

- **Post-Labor-Day session, SPY:** h=1 -0.141% (42.3% hit), h=2 -0.351%,
  against an all-days +0.040% / +0.078%; h=3 turns +0.185%; h=10 -0.555%
  against all-days +0.376%. No sign-test p below 0.16 at any horizon. Empty.
- **IWM** the same shape, slightly worse (h=10 -0.575% at 46.2%).
- **HYG** is the grid's one exception: h=1 +0.207% at a **73.7% hit (14 of 19),
  sign p 0.0318** against HYG's own +0.021%; h=3 +0.220% at the same 73.7%.
  Grid charge: 11 tickers x 5 horizons = 55 cells, ~1.6 hits expected at
  p<=0.03 and two found. A screen hit, and it fed the stage-C HYG brief only
  because two other independent constructions agreed.
- **Metals are the negative pole:** GDX h=3 -1.538% at a 35% hit, SLV h=3
  -1.153%, both against positive drift, but no sign-test p under 0.74.
- **September month-position (Tuesday is Sep session #6):** SPY h=5 +0.109% at
  65.4% (sign p 0.084) over 26 years, h=10 -0.765% at 46.2%. The midterm slice
  is n=6 at -1.102% / -1.943% — six observations, reported not traded.
- **Full September by cycle:** SPY midterm mean -1.477% on n=6 at a 50% hit;
  IWM midterm -2.141% at 33.3%. The famous September cell is real in the mean
  with no sign-test support at this N. Regime context for the book's own T2/T3,
  not a pitch.

---

## 5. Watchlist — all 39 active entries triaged (`00_watchlist_triage*.py`)

**CHECK 0 | PASS 33 | DATE-PARK 5 | UNCOMPUTABLE 1. No parked cell fires
today.** Full table with each entry's live number is in the triage scripts'
output. The three closest, and why they still fail:

1. **Entry 5, TLT + IEF + LQD at 52-week lows.** The leg that has blocked it
   since 2026-08-12 — freshness — **now clears for the first time**: the last
   tight-rung firing was 2026-08-18, 13 sessions ago against the >= 10 arm. Only
   the price rung is missing: TLT closed 82.21 against a 252d low of 81.04, so
   the 0.5% rung needs **81.44**, which is -0.93% or 1.27 Wilder-14 ATR. IEF
   (+0.44%) and LQD (+0.25%) already clear. **Sent to stage C** as a limit-entry
   construction (`06_tlt_floor_limit.py`), because a resting order at that level
   IS the trigger.
2. **Entry 33, SVXY into a print out of a (5,15] VIX relative range.** The band
   leg is 0.63 percentile points below the lower edge and has risen three
   sessions running (3.57 -> 3.97 -> 4.37). Today is not an anchor; the next
   qualifying one is the CPI k=-2 session on 2026-09-09.
3. **Entry 18, IEF against 0.523 TLT.** Its status changed: the ^TNX
   trailing-252 max touch **lapsed** on the 09-04 bar (4.784 against a 4.796
   max set 2026-09-01), so proximity is now a second binding leg beside the
   magnitude arm, which needs a 4.956 close (+17.2 bp).

Two corrections the triage produced: the 2026-09-04 watchlist notes were
written on the **2026-09-03** bar (that morning's run preceded the close),
which explains every apparent drift; and entry 1's episode count is **5, not
the 3 or 4 its own text claims** — the 2026 raw cluster has split in two.

---

## 6. Selection, and what was checked

Candidates that earned an adversarial checker, with the axis and the lane each
came from. Coverage: **6 asset classes touched (rates, credit, volatility,
gold, energy, us_large), 4 novelty axes, both search modes** — event-anchored
(SVXY into the Sep settle, HYG out of the closure) and price-state-anchored
(TLT at the IG floor, GLD/XLE from the analogue).

| # | candidate | axis | lane | script |
|---|---|---|---|---|
| C1 | Long TLT on a limit at the level that arms the three-way IG floor | `interaction_cell` | watchlist 5 | `06_tlt_floor_limit.py` |
| C2 | Long SVXY 09-08 MOC into the Sep VIX settle | `event_fingerprint` | event grid | `07_svxy_sep_expiry.py` |
| C3 | Long HYG at the first close back from the long weekend | `event_fingerprint` | closure lane | `08_hyg_closure.py` |
| C4 | Short GLD, h=5 | `historical_analogue` | analogue | `09_analogue_gld_xle.py` |
| C5 | Short XLE, h=10 | `historical_analogue` | analogue | `09_analogue_gld_xle.py` |
| C6 | Long SPY / short IWM at the dial's top decile | `interaction_cell` | price-state | `10_dial_spy_iwm.py` |

**All six were killed, and the morning shipped a stand-down.** Twenty candidates,
about 1,700 screen cells across four survey lanes, six adversarial checkers,
sixteen named kills. Verdicts:

| # | verdict | the kill |
|---|---|---|
| C1 TLT floor limit | **KILL** | the cell reproduces (+0.385pp, 83.3% on 18 episodes, sign p 0.0038) but it is lag-1 and its whole edge is session +2; the session a limit fill adds pays **-0.302% at a 38.9% hit**, and getting filled selects against you (armed fills -0.564% at D+1, reversals +0.957%). |
| C2 SVXY Sep settle | **KILL** | it IS the 2026-08-07 pre-expiry corpse (`P(corpse\|this) = 1.0000`, 14 of 14 anchors) and the settle is worse than the calendar slot it sits in: **6 of 8 post-2018 Septembers lose against their own tdom-matched neighbours**, paired -0.216pp. The pass-through ratio and the FOMC-coincidence split both PASSED. |
| C3 HYG post-closure | **KILL** | the closure gate filters nothing in the live state: **+0.064% vs +0.075% for an ordinary weekend**, excess -0.012% at t -0.12, while the whole +0.267% excess lives in the off-the-high half. +0.196% of the headline is IEF+SPY beta. |
| C4 GLD analogue short | **KILL** | 53 independent episodes pay **0.8 bps, 0.2x cost, 25-28**; the drawdown parent it actually rides LOSES (-0.050% over 109 episodes) with a U-shaped dose response. |
| C5 XLE analogue short | **KILL** | the short **loses (-0.262%, 25-28, -5.2x cost)**, and on the 9 episodes matching today's energy leadership it pays **-1.299%** — falsified exactly where it is named. |
| C6 dial SPY/IWM pair | **KILL** | the strongest statistic of the morning, killed on LOCATION not sample size: the whole +0.517% lives in the dial's **[56,70) band (49 episodes, 35-14, p 0.0019)** while [70,80) is 14 episodes at +0.071% on 6-8, and the return-on-dial slope is -0.0053pp per point. Today's 87.96 is the one part of the mask with no content. |

Three cells went to the watchlist with the number that would turn each on: the
dial pair (dial back inside [56,70)), watchlist 5 (TLT closing at or below
81.44, and NOT as a limit), and ITA (a 13-name subsector reference class at
max-of-13 permutation P <= 0.10). The reusable lessons, including the finding
that "negative edge in all six analogue constructions" is a property of
conditioning on a calm bull tape rather than a signal, are in
`data/pitch_negative_registry.md` under 2026-09-07.
