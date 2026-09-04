# Cell map — run 2026-08-10 (Mon), asof session 2026-08-10, next session 2026-08-11 (Tue)

Midterm year. August, trading day 7 of 21. Prices FRESH (core bar 2026-08-10).
Sweep: 1,231 cells scanned, 89 fired (72 event / 17 price), BH crit p 0.0, 1 pass.
Novelty: `delta_suppressed: true` (no flag state, the 08-09 shakeout never posted).
No NEW / first-time claims tonight, and I am hand-checking against the 2026-08-09
brief so nothing repeats it.

## The tape I am writing against

Equities at the top of the range and long bonds at the bottom of theirs, with a
commodity complex running hard underneath:

- SPY -0.03% and 0.03% off its 52w high; ^GSPC -0.06%, 0.06% off. ^NYA closed
  AT a 52w high (+0.30%) while SPY, QQQ (-0.30%, 3.28% off its high, 21d rank
  30.2) and IWM (-0.52%) all closed lower.
- TLT -0.85%, sitting 0.17% above its 52w LOW, 63d rank 8.3. IEF 0.65% off its
  low, LQD 0.60%. ^TNX +0.84% to 4.699, 63d rank 89.7.
- GC=F +2.49%, 5d +10.29% (rank 99.6). SI=F +4.02%, 5d +14.24%. PA=F 5d +11.0%,
  PL=F 5d +9.03%. CL=F +5.27%, 21d +15.25% (rank 86.1). ZC=F +5.24%. NG=F +4.36%.
- DX-Y.NYB z10 -1.72, 21d rank 17.9. ^VIX +3.76% to 15.46 on a flat tape,
  ^MOVE +4.76%, ^SKEW +3.44%.
- ^FCHI and ^GDAXI both closed AT 52w highs (63d ranks 99.2 / 95.2).
- Breadth 66.7% above the 200d, 67.8% 21 sessions ago.

CPI lands Wednesday. That is the session the whole week hangs on and it is the
lane I want the headline in.

---

## Event lane

### Calendar entries inside the next 5 sessions

| entry | verdict |
|---|---|
| CPI Wed 2026-08-12 08:30 ET (2 td) | **DRILL** — top tier, next-session-adjacent, and the base cells are all weak. The specificity has to come from conditioning on the state above, not from the bare cell |
| PPI Thu 2026-08-13 08:30 ET (3 td) | **DRILL** for NG=F only (see below); SKIP for everything else |

Outside 5 sessions, listed in the brief's calendar block only, no cells:
VIX expiry 08-19 (7 td), opex 08-21 (9 td), Jackson Hole 08-28 (14 td),
next FOMC 09-16 (26 td), next NFP 09-04 (19 td).

### E:cpi — 18 subjects, anchor = 2 td before a CPI

- **^GSPC / SPY / QQQ / IWM: SKIP(no edge in the bare cell).** SPY h1 +0.061%
  against a +0.039% all-days control, t=0.78, edge +0.022%. The August CPI
  month cell is n=26, -0.042%, 46.2% hit. Not era stable (pre-2018 -0.035,
  post-2018 +0.264). Nothing here Scott does not already assume.
- **SPY h5: SKIP(known and generic).** 191-126 up, sign p 0.0002, but the
  all-days control is 58.1% hit and the edge is +0.052% over a week. This is
  drift, not a CPI effect.
- **^VIX: DRILL.** h1 +0.754% (t=1.66) into the print and h5 -1.107% after it.
  The bare version is the most famous fact in the product's universe. Worth
  one drill only because VIX is 50.2% below its 52w high and rose 3.76% today
  on a flat tape, which is a genuinely different starting point.
- **^TNX: DRILL(folded into the CPI conditioning script).** h1 +0.306%,
  t=1.58, era stable, edge +0.283%. Yields firming into a CPI is the cell that
  most directly meets today's bond tape.
- **CL=F / NG=F / GC=F / SI=F / HG=F: SKIP(weak).** Every one is inside
  +/-0.31% with |t| < 1.7 and no sign-test support.
- **EURUSD=X / DX-Y.NYB / JPY=X: SKIP(noise).** |mean| <= 0.04%, |t| <= 1.18.
- **TLT / IEF / HYG / EEM: SKIP(flat).** |edge| <= 0.035%.
- **The conditioned version is where the value is: DRILL.** Crude has run
  15.25% in 21 sessions into this print and the index is pinned at a 52w high.
  Both are computable conditions on the same 317 anchors.

### E:ppi — 18 subjects, anchor = 3 td before a PPI

- **NG=F: DRILL.** The single strongest event cell in the sweep and the only
  `tag_hint: solid` one. n=308, h1 +0.452%, hit 57.5%, t=2.50, 177-130,
  sign p 0.0051, era stable, edge +0.386% over control. It also flatly
  contradicts the Aug-11 seasonal cell below (NG down 19 of 25). I do not
  believe a natgas-before-PPI mechanism, so the drill is a debunk attempt: PPI
  lands mid-month, so this anchor is close to a fixed day-of-month, and a
  day-of-month control is the obvious killer. Publishable either way.
- **SI=F: SKIP(swept).** 172-134, sign p 0.023, but mean +0.163%, t=1.38, and
  it is one of 18 subjects x 3 horizons. Sugar-grade multiplicity.
- **EURUSD=X: SKIP(mechanism-free).** t=-1.90 on -0.066% is a rounding error
  in FX terms.
- **SPY / QQQ / IWM / ^GSPC / ^VIX / everything else: SKIP(flat).** All inside
  +/-0.15% with |t| <= 1.3, and PPI at 3 td is not the next session anyway.

### E:weekday_month — Tuesdays in August, 18 subjects

- **JPY=X: SKIP(era-unstable and swept).** t=-2.36 but n=112, sign p 0.11,
  `era_stable: false`. The only cell in the group with a real t and it fails on
  both the era and the multiplicity test.
- **SPY / ^GSPC / QQQ / IWM: DEAD.** SPY 58-57, mean +0.008%, t=0.08. ^GSPC
  58-58. This is the null cell and it is fine that it is.
- **Everything else in the group: SKIP(no edge).** Largest |t| among the
  remaining 13 is HG=F at -0.94.

### E:seasonal_doy — same trading day of year (+/-2), Aug 11, 18 subjects

- **SPY: DRILL.** 18-8 up, mean +0.525%, median +0.205%, sign p 0.0378, n=26.
  ^GSPC identical (18-8, +0.508%). The best-supported seasonal cell available
  for tomorrow. Needs an era split, an August-drift control and a look at what
  carries the mean before it can be published. Midterm subset is 3-3 (n=6), so
  the cycle conditioning kills rather than helps here and the brief has to say
  so.
- **NG=F: DRILL(with the PPI cell).** 19-6 down, mean -1.008%, sign p 0.0073,
  n=25. Same script.
- **HG=F midterm: SKIP(too cute).** 0-for-6 down, sign p 0.0156, mean -1.063%.
  A 6-observation cell selected out of 18 subjects x 4 splits. That is 72
  cells; one 0-for-6 is the expected yield. Noting it because the record is
  clean, publishing it would be dishonest.
- **CL=F: SKIP(subsumed).** 16-9 down all years, sign p 0.115. Weaker than NG
  and I am already spending a nugget on energy seasonality.
- **^VIX: SKIP(unstable).** 17-9 down all years, but the midterm subset is 3-3
  and h5 midterm is -8.4% on n=6. The magnitude is carried by 2 episodes.
- **QQQ / IWM / TLT / IEF / ^TNX / HYG / GC=F / SI=F / DX-Y.NYB / EURUSD=X /
  JPY=X / EEM: SKIP(no record).** All sign p > 0.10 on the all-years cell.

---

## Price lane (prices fresh, so the lane is live)

The fired price cells are almost entirely in instruments that are not macro
subjects, which is the honest summary of tonight's price lane:

| cell | verdict |
|---|---|
| `P5:rank5_extreme` HE=F bottom 5%, `P6:two_atr_day` HE=F down, `P8:sma200_cross` HE=F down | **SKIP(out of scope).** Lean hogs fell 12.46% today and fired three triggers on its own. Not a macro subject under the universe rule |
| `P4:z10_extreme` USDTRY=X | **SKIP(degenerate).** n=575, 397-177 up, t=3.14, and it is the ONE cell that passed BH. It is also a managed depreciating currency where "stretched up" is the permanent state. The statistic is real and describes nothing |
| `P4:z10_extreme` SB=F, `P5/P5b` SB=F, `P7:up_streak` SB=F | **SKIP(not macro).** Sugar fired four separate triggers. Same universe rule |
| `P5:rank5_extreme` GC=F top 5% | **SKIP(published 2026-08-09).** The exact cell ran in last night's brief with its number, and the number has not moved in a way that changes the claim. Gold's 5d MAGNITUDE is a different cell and is drilled instead |
| `P5:rank5_extreme` CADJPY=X, `P5b` CHFJPY=X bottom | **SKIP(cross-rate noise).** CHFJPY 186-147, sign p 0.019, but h1 mean is +0.061% and yen crosses are not a subject Scott reads for |
| `P5b:rank21_extreme` FXI top 5% | **SKIP(h1 dead).** 21d rank 96.8 is a real state, but h1 is -0.002% on n=351, 172-177. h5 +0.538% is not era stable. Nothing to say tomorrow |
| `P6:two_atr_day` ZC=F up | **SKIP(no edge).** 77-97 down after, mean +0.108%, t=0.87. Corn +5.24% today is tape colour, not a cell |
| `P6:two_atr_day` USDCNY=X down | **SKIP(managed).** Same objection as USDTRY |
| `P7:up_streak` ^FCHI | **DRILL.** n=205, mean -0.158%, hit 43.9%, t=-2.03, 90-115, sign p 0.0467, era stable. The CAC closed AT a 52w high on a 5-session streak, and ^GDAXI is at its high too. A real macro subject with a real cell |
| `P7:up_streak` PA=F | **SKIP(swept).** 129-110, sign p 0.136, t=1.57 |
| `P7b:down_streak` ^BVSP | **SKIP(marginal and remote).** 85-57, t=1.96, era stable, but Brazil is a peripheral subject and the state is a 0.19% down day |

### Triggers that did NOT fire but whose state is live

Recording these because my headline comes from one of them and the map has to
show it was reasoned rather than recalled.

- **`P9` family (stocks and bonds together): DRILL, off-sweep.** The P9 cells
  key on joint MOVES, and today's moves were small (SPY -0.03%, TLT -0.85%),
  so nothing fired. The joint LEVEL is the striking thing: SPY 0.03% from a
  52w high while TLT sits 0.17% off a 52w low. That is a state the trigger
  inventory does not have and it is the most specific description of this
  tape. Computing it fresh.
- **`P1/P1b` (first 52w high in 30+ days): correctly silent.** SPY has been
  making highs continuously, so there is no novelty event. ^NYA printing one
  today while SPY, QQQ and IWM closed lower is folded into the same script as
  a secondary check.
- **`P10` (VIX term structure, VIX +10%): correctly silent.** VIX +3.76% is
  below the 10% trigger and 15.46 / 18.98 is a normal upward term structure.
  The vol bid is small in absolute terms and I am not inflating it.
- **`P11` (breadth crossing 80% / 20%): correctly silent.** 66.7% above the
  200d, mid-range, and barely moved over 21 sessions.
- **`P3` (reversal after a 52w extreme), `P2` (52w lows), `P8` on macro names,
  `P12` (conditioned on today's print): all correctly silent.** No US releases
  today, so P12 had nothing to condition on.

---

## Selected for drilling

| # | script | lane | why |
|---|---|---|---|
| 1 | `01_stocks_high_bonds_low.py` | today | SPY at a 52w high with TLT at a 52w low. Off-sweep, most specific description of the tape, headline candidate |
| 2 | `02_cpi_conditioned.py` | tomorrow | CPI k2 conditioned on the 21d crude run and on the index sitting at its high. Also the ^VIX and ^TNX legs |
| 3 | `03_gold_5d_magnitude.py` | today | 10%+ 5-session gold moves as a magnitude cell, distinct from the rank cell published last night |
| 4 | `04_natgas_ppi_vs_seasonal.py` | tomorrow | Resolve the contradiction between the sweep's only `solid` event cell and the Aug-11 seasonal, with a day-of-month control |
| 5 | `05_seasonal_aug11.py` | tomorrow | SPY 18-8: era split, August-drift control, concentration |
| 6 | `06_europe_at_highs.py` | today | ^FCHI up-streak at a 52w high, with ^GDAXI as the confirmation and a US-tape control |

Tag budget note: at most two `[anecdote]`, and an anecdote may not headline.
The Aug-11 seasonal (n=26) and the CPI conditioning (n will drop well below 317)
are the cells most at risk of drifting into anecdote territory, so their N is
the thing to watch when composing.

---

## Drill outcomes (written after the scripts ran)

A methodology note that cost two rewrites and changed published numbers:
`close_panel` builds a UNION index across the tickers you ask for, so a
252-bar rolling max for SPY includes rows where SPY did not trade. Adding
^RUT to a panel moved the at-a-52w-high anchor count from 76 to 90 on the same
cell. Every number below is recomputed in `09_native_index_recheck.py` /
`10_final_numbers.py` from `load_prices`, one ticker at a time, on native
sessions. Those are the numbers in the brief.

### PUBLISHED

1. **^VIX, CPI eve, SPY at a 52w high** (`09`, `10`). n=91, +2.666%, median
   +1.318%, 68.1% hit, t=3.25, 62-29, sign p 0.0004. Survives everything I
   threw at it: all 314 CPI eves are 159-155, CPI eves NOT at a high are
   97-126, SPY at a high on a non-CPI-eve session is 852-767, so neither
   marginal produces it. Ex the two largest moves 60-29. Strengthens by decade
   (56.5% / 69.8% / 76.0%). Stable from 0.2% to 3.0% off the high (68/68/64/
   60/58), so it is not a threshold-tuned edge. CPI-specific: the same
   conditioning gives 33-36 on NFP eves and 28-26 on FOMC eves. `[solid]`, and
   the brief says it came from a conditioned search.
2. **TLT over the CPI session, crude 21d >= +10%** (`09`, `10`). n=49, +0.394%,
   71.4% hit, t=2.35, 35-14, sign p 0.0019. Control kills the crude story on
   its own: crude hot on any session gives TLT -0.024% over two days, 53.1% on
   n=1025. All CPI days 55.5%, CPI days with crude cold 52.1%. ^TNX -1.03%,
   19-30. Hit rate is the same in both eras (71.9% / 70.6%) but the 2010s carry
   the strength (88.2%) and the 2020s are 8-5. The losses are the tell and the
   brief says so: 2022-02, 2022-03, 2022-06, 2023-08 and 2026-03, i.e. the
   prints where inflation was genuinely accelerating. `[suggestive]` (n<50).
3. **NG=F, 3 td before a PPI** (`04`, `10`). n=307, +0.513%, 58.0%, t=2.83,
   178-129, sign p 0.0030. NOT a day-of-month artifact, which was the
   hypothesis: sessions at the same trading-day-of-month band (6-8) that are
   not PPI anchors run 362-374 at -0.020%. It is a modern effect only:
   2000-2009 -0.261% / 46.8%, 2010-2017 +0.539% / 61.5%, 2018-2026 +1.315% /
   66.7%. `[suggestive]`, single-era.
4. **The precious complex thrust** (`03`, `10`). Gold +5%, silver +8%, platinum
   +5% and palladium +5% over the same five sessions: 17 declustered episodes
   since 2000. SI=F over the next five sessions -5.37%, 25.0% hit, 4-12,
   sign p 0.038, t=-2.79. Gold's own version is weak (7-9). Concentration is
   real and published: the top two episodes carry 47% and both are 2026.
   `[suggestive]`.
5. **SPY at a 52w high with TLT at a 52w low** (`01`). 54 raw sessions, 15
   declustered episodes, six distinct years in TLT's whole history. SPY next
   session 8-7 at -0.05%, TLT over the next week 5-10. The rarity and the
   absence of a follow-on are the fact. `[suggestive]`.
6. **The Aug-11 seasonal is an anchor artifact** (`05`, `05b`). The sweep's
   SPY cell is 18-8 up, sign p 0.0378. Walking the engine's own anchor back
   through the eight preceding sessions gives 13-13, 15-11, 10-16, 13-12,
   11-15, 14-12, 9-15, 13-13, with means from -0.57% to +0.52% and a median of
   -0.09%. One of the nine anchors reaches sign p < 0.10 and it is the one the
   calendar happened to land on. NG=F's Aug-11 cell (6-19 down, sign p 0.0073)
   fails the same test, with neighbours at 14-11, 14-11, 14-11, 10-15, 16-6,
   8-17, 12-13, 8-17. This is what resolves the natgas contradiction in item 3:
   the PPI cell survives its control and the seasonal does not. `[suggestive]`.

### DRILLED AND DROPPED

- **^NYA at a 52w high while SPY, QQQ and IWM all closed lower** (`01`, `07`).
  Looked like the night's best find at 11-18 with IWM at -0.38%. It is two
  episodes: 2022-01-04 and 2024-01-30 carry 78-83% of the total, and removing
  them leaves SPY 10-11 at -0.047% and IWM 10-11 at -0.055%. Dead.
- **IWM after ^NYA prints a 52w high** (`07` control, `08`). Raw 638 sessions
  gave -0.133%, 44.5%, t=-3.76, which is overlap inflation and nothing else:
  5td-declustered t=-1.90, 21td-declustered t=-0.29 on 94 anchors. The
  IWM-minus-SPY version survives 5td (41.4%, t=-2.35) and dies at 21td.
  Not publishable.
- **^FCHI 5+ up closes** (`06`). The sweep's cell reports t=-2.03 on n=205.
  My raw reconstruction is 198-206 at t=-0.97 and the 10td-declustered version
  flips sign to 98-87 UP. It does not survive declustering. Separately, the
  joint state I wanted (streak AND at a 52w high, with ^GDAXI also at its
  high) has one prior instance ever, in 2017.
- **Gold's own 5d magnitude cells** (`03`). 5d >= +6% is 32 episodes but the
  h5 mean flips across 2018 (+1.73% to -0.82%). 5d >= +10% is five episodes.
  The complex version above is the only one worth publishing.
- **CPI conditioned on August, on midterm years, and on August-midterm** (`02`).
  August CPI is 13-13 on the print. Midterm CPI is 45-34 at -0.00%. The
  August-midterm intersection is six observations. All null, all dismissed.
- **CPI eve VIX with VIX itself starting low** (`02`). 69-68 on n=137. The
  at-a-52w-high conditioning is the one that works; low vol on its own is not.
