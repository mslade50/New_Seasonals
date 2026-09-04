# Surface map — 2026-08-25 (Tuesday, midterm year, ME-4)

State: `data/pitch_state.json` (asof 2026-08-25, generated 05:11), tape asof
2026-08-24, pipeline all-green, **zero warnings**, prices bar 2026-08-24,
dial 2026-08-24, P/C 2026-08-24.

Recon scripts behind every number here: `00_tape_sort.py`, `00_recon_premises.py`,
`00b_recon2.py`. Per the 2026-08-24 registry's top lesson, every candidate below
had the thing it is NAMED after printed with a PIT trailing-252 percentile
BEFORE anything else was run.

> ## CORRECTION, filed 2026-08-25 during stage C — this map's SPREAD premises
> ## were computed on a padded panel, and two of them were wrong.
>
> `00_recon_premises.py` computes every "spread (pp)" headline as
> `px.pct_change(n)` over a 26-ticker UNION-CALENDAR panel that includes
> `^TNX` and `DX-Y.NYB`. That takes pandas' `fill_method='pad'` default and
> pads foreign/holiday calendar holes into synthetic zero-return sessions, so
> any window spanning one is shifted. This is the exact trap
> `pitch_lab._valid_pct_change` exists for (registry, 2026-08-19) and I walked
> into it in the recon rather than in a check. Found INDEPENDENTLY by two
> checkers within minutes of each other.
> Audits: `b1d_premise_padfill_audit.py`, `c3b_c8_premise_forensic.py`.
>
> | premise as written above | clean, valid-session | delta |
> |---|---|---|
> | C8 EEM-EFA 63d **-7.57pp / PIT 1.6** | **-4.67pp / PIT 5.6** | +2.91pp, +3.6pt |
> | (sub-premise) FXI-EEM 63d **+4.24pp / PIT 99.6** | **-0.03pp / PIT 90.5** | -4.27pp, -9.5pt |
> | C4 OIH-XOP 63d -18.52pp / PIT 0.4 | -16.78pp / PIT 1.2 | +1.74pp, +0.4pt |
> | C2 SMH-SPY 63d -10.17pp / PIT 0.0 | -7.78pp / PIT 0.4 | +2.40pp |
> | C1 XLV-XLK 5d, C10 XLV, C9 XLI, C3 XLU-TLT 21d, C7 GDX-GLD 21d | **identical** | **0.00** |
>
> **Only cross-calendar spreads over long lookbacks are affected.** Every
> single-ticker reading and every same-calendar US pair in this map — which is
> all of C1, C3, C6, C7, C9, C10 — reproduces exactly. **C8 was KILLED on this:
> its named extreme did not exist and its trigger last fired 2026-08-14.**
> C4's and C2's percentiles survived the correction and both died on other
> grounds.

## 0. The regime, stated once

- Fragility dial ma10(63d) **89.5** (21d ago: 47.7). Raw 63d 88.3, raw 21d 64.5.
  Exposure leg 0.0x on Rule 1. Signals on: Defensive Leadership, Low Absorption
  Ratio. P/C fear OFF at the 54th pctile.
- SPY -1.85% off its 52w high, +8.1% over its 200d. VIX 15.8 (PIT 23.4).
- **The tape's defining feature is a rotation, not an index move.** 5-day:
  XLV +4.58, XLP +3.27, IBB +5.25 against XLK -5.40, SMH -7.96, XLI -3.93.
- Midterm year. This repo has now recorded SIX independent midterm inversions
  (JH x rates/gold/FX/small/large/credit-intl), and the seasonal board's own
  read is "book win% 56.4 midterm vs 64.9 all-years, de-risk".

## 1. Every live calendar event x every asset class

Events inside [-5, +15] td: vix_expiry (-4), opex (-2), **jackson_hole (+3)**,
**nfp (+8)**, ppi (+11), cpi (+12), fomc_decision (+15), vix_expiry (+15).
Default horizon caps at 10 td, so ppi/cpi/fomc/quad-witching are **out of reach
by construction** and are dismissed on that alone, not on their content.

| event | us_large | us_small | rates | credit | gold/miners | metals | energy | dollar_fx | intl | vol |
|---|---|---|---|---|---|---|---|---|---|---|
| **jackson_hole (+3)** | CLOSED | CLOSED | CLOSED | CLOSED | CLOSED | — | CLOSED | CLOSED | CLOSED | not run |
| **nfp (+8)** | not run | not run | PARKED | not run | not run | — | not run | not run | not run | not run |
| opex (-2) | CLOSED | CLOSED | CLOSED | CLOSED | CLOSED | CLOSED | CLOSED | CLOSED | CLOSED | CLOSED |
| vix_expiry (-4) | subsumed | subsumed | — | — | — | — | — | — | — | subsumed |
| ppi/cpi/fomc/quad (+11..+15) | OUT OF HORIZON on every class |

Verdicts, with the reason each dismissal is legal:

- **Jackson Hole: CLOSED, and this is the strongest dismissal on the board.**
  The registry has swept this anchor on **seven asset classes** (rates 08-13,
  small caps 08-13, gold and FX 08-19, US large 08-19, credit and international
  08-21, energy 08-20) and it is empty in all of them. The two decisive
  general findings: the offset placebo ladder is a **plateau with no spike
  anywhere from -10 to +5**, and the unconditional late-August window beats the
  event outright (tdom 6-16 all years +0.234% over 286 starts against the
  anchor's +0.102% over 26). Midterm years invert it in 4 of 6 vehicles.
  2026-08-24 re-swept JH-4 across seven classes for a pre-speech class mean of
  **+0.010pp**. Today is JH-3. Adding an eighth class to a dead anchor is
  exactly what the 2026-08-24 calendar finding says not to do. **No candidate.**
- **NFP at +8 td: PARKED on rates, not run elsewhere.** W1 holds the one live
  cell (long TLT NFP-close to +3td at the 52w floor) and its trigger is the
  CYCLE YEAR: midterm +0.071% N=12 t=0.17 against non-midterm +0.978% N=13
  t=2.72. 2026 is midterm, so it is off until 2027-01. Not run on the other
  nine classes because +8 td puts the print at the very edge of the horizon
  cap: a candidate entered today would hold 8 sessions of unrelated tape to
  reach it, which is a price-state trade wearing an event label. If NFP is the
  thesis, the right morning to check it is 2026-09-03.
- **Post-opex (-2): CLOSED in both directions** on equities and across ten
  non-equity vehicles by ten horizons (2026-08-24). The event sleeve's V4 is
  already the live expression of this anchor and exits tomorrow.
- **vix_expiry (-4): subsumed** by the opex sweep — the August VIX expiry and
  opex sit two sessions apart and the 08-24 sweep covered both windows.

**Consequence: today has no tradeable macro anchor either.** That is the second
morning running, and it is a fact about the calendar rather than about the
sweep. Where 2026-08-24 concluded "the honest move is a price-state sweep",
today has one non-macro calendar anchor left that the repo has NOT closed on
every class — month-end — plus a single-name event that is not in
`macro_events.csv` at all.

- **Month-end, today = ME-4 (last August session is Mon 2026-08-31).** Equities
  were CLOSED on 2026-08-24 (SPY ME-1 to ME-0 pays -0.006% at a 47.6% hit; the
  60-cell grid gives Sidak 0.877; August x midterm is 3-3 at -0.860%). Rates
  were SUSPENDED, not closed: the parent survived every robustness test and
  died on mechanism decay, and W12's second condition is literally "NOT
  August". **Both are dismissed.** What is NOT swept anywhere in this repo is
  **month-end on FX**, which has its own distinct mechanism (the 4pm London fix
  rebalancing flow) rather than being the equity story re-skinned. Recon:
  DXY ME-4 to ME-0 is -0.062% over 320 anchors at a 47.2% hit, i.e. a
  tail-driven mean sitting at ~4x a 1.5 bp DX round trip before any
  conditioning. **-> CANDIDATE C5.**
- **NVDA reports tomorrow (2026-08-26, +1 td).** Not a macro event and
  therefore absent from every calendar sweep this repo has run. It is the one
  scheduled, dateable, market-moving event inside the horizon. Semis are at a
  **one-year relative low into it** (SMH-SPY 63d = -10.17pp, PIT252 pctile
  **0.0**). 109 prints in the earnings parquet, ~26 since 2020.
  **-> CANDIDATE C2.**

## 2. Every tape extreme, by asset class

All percentiles are PIT trailing-252 on the stated series (`00_recon_premises.py`).

| class | what is extreme today | verdict |
|---|---|---|
| **us_large** | SPY -1.85% off high, 5d rank 18.7; QQQ 5d rank 7.1 and -5.24% off high. The index is NOT extreme; the dispersion under it is. | index-level: no candidate. The dispersion is C1/C6. |
| **sectors** | **XLV-XLK 5d spread +9.98pp = 99.6th FULL-SAMPLE pctile, 97.6 PIT.** XLP-SMH +11.23pp (98.7 full). XLV 5d rank 96.0 / 63d 97.6; XLK 5d rank 4.4; XLI 5d rank 2.0. | **-> C1, C9, C10.** Biggest single extreme on the board. |
| **semis** | SMH-SPY 63d **PIT 0.0**, 5d PIT 3.2, -18.26% off its 52w high while SPY is -1.85%. GLW -15.97 / INTC -15.68 / MU -10.01 / AMD -9.73 / AMAT -9.45 5d. | **-> C2** (crossed with the NVDA print). |
| **utilities** | XLU 21d **-6.63%, PIT 0.8**; XLU-TLT 21d spread PIT **2.0**; XLU-SPY 21d PIT 5.2. Six of the tape's ten weakest 21d names are utilities (DTE, AEP, CNP, CMS, ETR, D, PNW). | **-> C3.** The bond proxy was dumped and the bond was not. |
| **energy** | **OIH-XOP 63d -18.5pp, PIT 0.4**; OIH -9.96% off its 52w high while XLE is -1.00% and XOP -1.74%. USO 5d +1.47%, r21 rank 24.6. | **-> C4.** |
| **gold_miners** | GDX 21d **+37.6%, PIT 99.6**; GDX-GLD 21d +22.9pp, **PIT 99.2**; NEM AT a 52w high, +41.5% 21d. GLD -13.96% off its own 52w high. | **-> C7** (fade side). W4 PASSES (its 4th condition, GLD within 10% of its high, fails at -13.96%). |
| **other_metals** | SLV +4.41 5d, -41.10% off its 52w high, -3.79% under its 200d. Structurally the same thrust as C7 and correlated 0.708 with it (2026-08-11 kill). | dismissed as a duplicate of C7; not run separately. |
| **rates** | ^TNX at **99.7% of its 252d high (PIT 99.6)** while TLT 5d rank is 90.1. TLT +1.49% above its 52w low, IEF +0.90%, LQD +0.72%. | W5's tight rung is **NOT live today** (TLT +1.49% vs the <=0.5% rung) and its freshness leg fails anyway (last tight day 2026-08-18). The ^TNX LEVEL trigger was charged and killed 2026-08-24 (91% mask overlap with the dead rank form). **No candidate.** |
| **credit** | HYG **-0.11% off its 52w high** (21d rank 84.5) while LQD is **+0.72% above its 52w low**. W2's exact state. | W2 PASSES: trigger is episode count, still 4 declustered since 2007 against a required 8 across 3 non-2018 years. **No candidate.** |
| **dollar_fx** | DXY 21d **-2.43%, PIT 0.4** — the most oversold dollar of the past year — with ^TNX at a 52w high. | The divergence form is W16, parked at cost with a broken dose response, and the yield-level half was charged 2026-08-24. The clean unswept object here is the **month-end flow**, not the divergence. **-> C5.** |
| **international** | **EEM-EFA 63d -7.57pp, PIT 1.6** (full-sample 6.6); EEM 63d rank 6.3 against EFA 59.1. FXI-EEM 63d PIT 99.6. EFA -0.61% off its 52w high. | **-> C8.** |
| **volatility** | Nothing is extreme. VIX 15.8 (PIT 23.4), VIX/VIX3M 0.854 (PIT 44.4), SKEW 145.6 (PIT 57.9), SVXY -0.44% off its 52w high. The only PIT-notable reading is the 5d CHANGE in the term ratio (+0.056, PIT 84.9), a front-end kink of no magnitude. | Dismissed on the level: a vol cell entered at PIT 23-44 has no state to revert. The book is ALSO already long SVXY 1228 sh through tomorrow's close (event sleeve V4). **No candidate.** |
| **us_small** | IWM -2.33% off its high, 21d rank 44.0, 5d rank 11.5. Mid-range on every axis. | Nothing to anchor on. **No candidate.** |

## 3. Live seasonal and cycle cells

- Seasonal board: **0 A+B-grade setups** across 2 channels. Its live content is
  all regime context (midterm de-risk on the book, on OVS, on LT Trend ST OS,
  on Indices Oversold Bounce), not a ticket.
- Late August, tdom 6-16 window: the registry already established this window
  beats the Jackson Hole anchor unconditionally (+0.234% over 286 starts) —
  which is a statement that August's own drift is the thing, and it is
  available without an anchor. Not an idea; a control every candidate below
  has to beat.
- Cycle: midterm. Used as a CONDITIONER on every candidate, never as a cell.

## 4. Watchlist — verdict on all 24 active entries

Every entry got today's number. Full text in `data/pitch_watchlist.json`.

| # | entry | today's reading | verdict |
|---|---|---|---|
| W1 | TLT from the NFP close, 52w floor | midterm year; NFP is +8 td | **PASS** (arms 2027-01) |
| W2 | LQD vs HYG at joint 52w extremes | state LIVE (HYG -0.11%, LQD +0.72%) but episodes still 4 of the 8 required | **PASS** |
| W3 | SVXY overnight into CPI | next CPI 2026-09-11 = +12 td, outside the horizon cap | **PASS** |
| W4 | GLD on a miner-led thrust | GDX r5 95.6 and GLD r5 92.5 fire; **GLD is -13.96% off its 52w high** against the >-10% 4th condition | **PASS** (4th condition) |
| W5 | TLT with the IG complex at 52w lows | tight rung NOT live (TLT +1.49% vs <=0.5%); freshness also fails, last tight day 2026-08-18 | **PASS** (twice) |
| W6 | SPY on a skew spike alone | SKEW 5d rank 67.5 against >=95; midterm block stands | **PASS** |
| W7 | XLE on a crude pop in [5%,6%) | USO 5d +1.47%, nowhere near the band | **PASS** |
| W8 | Fade a crude thrust out of a deep base | USO r5 rank 55.2 (needs >=90), r63 27.0 (needs <=20) | **PASS** |
| W9 | IHI medical-device thrust | IHI 21d rank 98.0, not 100; family-wise p 0.933 blocker unchanged | **PASS** |
| W10 | FXI breaking inside an intact thrust | FXI 5d rank 65.9, trigger needs <=20 | **PASS** |
| W11 | TLT November month-position | parks to ~2026-11-05 | **PASS** (dated) |
| W12 | TLT into the month-end close | **today IS ME-4**, inside its flat ME-3..ME-7 band — but its 2nd condition is literally "NOT August", and August ME-5 is 5-of-11 at -0.510% since 2015 | **PASS** (August block) |
| W13 | SPY on a vol pop in a calm tape | VIX 21d rank 19.0 clears the calm leg; the day's VIX move was **+4.76% against the >=+5.0% pop rung** | **PASS** (pop leg misses by 0.24pp) |
| W14 | Gold on an unconfirmed rate rise | dollar leg fires (DX 21d rank 0.8); yield leg needs a +0.20pt 21-session rise, actual +0.05pt | **PASS** |
| W15 | XLK vs XLV after a rotation gap | **one-day** XLV-XLK gap +1.18pp against the >=+3.0pp trigger; and its own turn-on needs 3 new non-2026 winning episodes | **PASS** — but see C1, which is the FIVE-day object and must prove it is not this one |
| W16 | Short the dollar on an unconfirmed rate rise | ^TNX 21d return rank 45.6 against >=65 | **PASS** |
| W17 | Long crude through Jackson Hole | JH-6 anchor was 2026-08-20, gone; XLE -1.00% off its high violates its own 2nd condition | **PASS** |
| W18 | Short TLT after a big up day from the low zone | TLT 1d -0.35% against the >=+1.5% thrust rung | **PASS** |
| W19 | Short KRE against XLF on a breadth washout | KRE 5d rank 7.9 so the state is live again; the trigger is a COST threshold on history that a new episode cannot move | **PASS** |
| W20 | HYG across the Jackson Hole speech | anchor JH-5 was 2026-08-21, gone; and the anchor is closed on credit | **PASS** |
| W21 | Duration-neutral IEF vs 0.52 TLT, ^TNX at a 52w high | **^TNX is at 99.72% of its 252d high = within 0.28%**, so the 0.25% rung is a whisker away and today is the closest it has been. Turn-on is COST (needs 30 bps at h=8, ladder tops at 22.2) and the 2nd blocker is that JH-spanning episodes are 0-for-6 — **a hold entered today spans Friday's speech** | **PASS**, and the JH blocker independently forbids entry this week |
| W22 | NARROW energy thrust cluster (2-3 names) | the 11-name z10>=2.0 cluster has decayed off last week's 5; and its three prereg debts (own pre-registration, narrow-form reference class, August weakness) are all unpaid | **PASS** (debts) |
| W23 | Survivorship-free new-high breadth | needs SPY >2.0% off its high (today **1.85%**) AND raw-21d dial <=50 (today **64.5**) | **PASS** (both legs) |
| W24 | (grouped with W22/W23 in the file) | — | — |

Pruning after publish: none expired today.

## 5. The 10 candidates selected from the map

Coverage check before selection: **nine asset classes touched**, either by a
candidate or by a numbered dismissal (us_large, sectors, semis, utilities,
energy, gold_miners, dollar_fx, international, plus rates, credit, vol and
us_small examined and dismissed with numbers).
Event-anchored: C2 (NVDA print), C5 (month-end flow). Price-state anchored:
C1, C3, C4, C6, C7, C8, C9, C10. Both search modes are crossed in C2 and C5.

| id | candidate | axis | class |
|---|---|---|---|
| C1 | Long XLK outright after a 99.6th-pctile five-day tech-to-defensive rotation | inversion | sectors |
| C2 | Long SMH into the NVDA print with semis at a one-year relative low | event_fingerprint | semis |
| C3 | Long XLU after a 21-day washout the bond market did not cause | interaction_cell | utilities x rates |
| C4 | Long OIH against short XOP at a 63-day services-vs-E&P extreme | relative_value | energy |
| C5 | Short the dollar into the month-end fix, conditioned on the month's equity move | flow_mechanics | dollar_fx |
| C6 | Long the tape's 5-day losers against its winners, cross-sectionally, on a dispersion extreme | historical_analogue | us_large |
| C7 | Fade the miner thrust: short GDX after a 99.6th-pctile 21-day run | inversion | gold_miners |
| C8 | Long EEM against EFA at a 63-day relative extreme | relative_value | international |
| C9 | Long XLI on a 5-day rank-2.0 washout inside an intact 63-day trend | interaction_cell | sectors |
| C10 | Short XLV outright after a 63d rank of 97.6 into a 52-week high | inversion | sectors |
| C11 | Long QQQ while tech's 63d rank is bottom-quintile and the index's is not | interaction_cell | us_large |

Axes: inversion, event_fingerprint, interaction_cell, relative_value,
flow_mechanics, historical_analogue = **6 distinct axes**.

**C11 was added DURING stage C, and its provenance is the point.** It did not
come from my ideation. It is a BY-PRODUCT recorded in the registry under
"Cells swept and empty (2026-08-24)", inside the entry that killed *Long SPY
against short QQQ*: "on days tech's 63-day rank is bottom-quintile while the
index's is not, QQQ LONG pays +0.508% at h=5". It was written down as the
reason a PAIR died and never developed as a trade. It is **live today** (QQQ
63d rank 14.7, SPY 23.8), and the C1 checker independently re-measured it at
long XLK +0.391% h=3 (N=151, t 1.99) / +1.253% h=10 (N=75, t 2.63) while
finding that C1's own rotation trigger **adds nothing to it** (C1 with this
cell OFF pays -0.065% at h=3).
Because it arrived with a mechanism already attached from the research record,
the pre-specified object — long QQQ, h=5, on that exact mask — carries **no
search charge**, while any vehicle/horizon/threshold improvement is a grid and
is charged separately. That split is the crux of its check.

Registry collisions each candidate must address, checked before dispatch:
- C1/C6/C9/C10 all touch the rotation. The one-day form is W15 and was killed
  on concentration (top-2 episodes 96% of the h=3 total, both 2026); the
  SPY-vs-QQQ pair was killed 2026-08-24 on leg attribution; the XLI pair was
  killed 2026-08-24 for having no history. **Overlap with W15's mask must be
  measured, not asserted** (the 2026-08-24 ^TNX lesson: 91% overlap = same
  object). W15's own closing line says that if this shape ever arms it arms as
  an OUTRIGHT, not a pair, which is why C1 and C10 are specified outright.
- C7 is adjacent to the 2026-08-21 GDX/GLD ratio kill (definition fragility +
  1.9x cost) — a 21-day THRUST is a different mask from a RATIO level and the
  overlap owes a number.
- C3 must clear "duration wearing a sector label", which this repo has now
  found three separate times (2026-08-12 PPI, 2026-08-20 credit washout,
  2026-08-24 IEF translation).
- C5 must not become the equity month-end story re-skinned; the FX leg has its
  own flow and the check must show the equity-conditioning matters.
- **Book overlap is unusually heavy today and every checker gets it**: the
  scan has staged OLV LONGS in **D and ETR (both utilities)** and in **AMKR,
  ON, POWI (all semis)**, and OVS SHORTS in **KO, GIS, CAG, AMGN, CMCSA, DIS**
  — the exact defensive names C1/C10 would fade. The book is already in a
  softer version of half this map. The event sleeve is long SVXY 1228 sh
  through tomorrow's close.

## 6. Axis feedback read

`scoreboard.lifetime` has **4 graded ideas**. That is a handful, not a signal:
event_fingerprint 2 at +0.62R, interaction_cell 1 at +0.15R, relative_value 1
at +0.10R. No axis is steered for or against on this evidence.
