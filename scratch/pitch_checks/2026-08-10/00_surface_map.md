# Surface map — 2026-08-10 (Monday)

State: `data/pitch_state.json`, tape freshest bar **2026-08-07**, no warnings,
pipeline 7/7 green. Cycle year **midterm** (2026 % 4 == 2). Fragility ma10-63d
**60.2** (was 21.9 twenty-one sessions ago; raw 21d 73.3, raw 63d 82.5),
exposure leg **0.0x**, P/C fear **off** (51st pctile). One fragility signal on:
Low Absorption Ratio (7th pctile, at a high).

Tape sorted by `00_tape_sort.py` across ret_5d / ret_21d / ret_63d / z10 /
dist_52w_high / dist_sma200 / rank_63d, both tails, all 218 names.

---

## 1. Calendar events x asset classes

Six events inside the [-5, +15] td window. Ten classes. Sixty cells; each row
below states which cells earn a check and why the rest do not.

| event | date | td | status |
|---|---|---|---|
| nfp | 2026-08-07 | -1 | just passed, Friday's session |
| **cpi** | 2026-08-12 | **+2** | inside every candidate horizon |
| **ppi** | 2026-08-13 | **+3** | inside every candidate horizon |
| vix_expiry | 2026-08-19 | +7 | inside a 7-10 td horizon |
| opex | 2026-08-21 | +9 | inside a 9-10 td horizon |
| jackson_hole | 2026-08-28 | +14 | beyond the 1-10 td default |

### nfp (2026-08-07, -1 td) — the whole row is DISMISSED

Friday's run swept exactly this event across rates, FX, credit, utilities,
small caps and US large and killed twelve candidates on it (state
`history.recent_kills`). The event is now behind us; anchoring on a print
that already happened is a stale-trigger, and the one survivor of that sweep
(long DX, fingerprint `e409803df080ad9e`) is inside the 10 td repetition
window. No nfp cell is examined today.

- nfp x us_large: dismissed, killed 2026-08-07 (+0.129% vs +0.221% all-days).
- nfp x rates: dismissed, killed and parked to the watchlist (2027-01).
- nfp x dollar_fx: dismissed, pitched Friday, inside the repetition window.
- nfp x credit / us_small / volatility / intl / energy / metals: dismissed,
  swept Friday, all killed on their own numbers.

### cpi (2026-08-12, +2 td)

| class | verdict |
|---|---|
| us_large | CHECK as part of the CPI+PPI cluster (see ppi row); the plain "SPY into CPI" cell is the dead `nfp -> cpi` run, killed 2026-08-07. |
| us_small | dismissed. IWM's NFP-anchored cell was -0.014pp vs its own drift Friday; nothing about the CPI anchor changes the objection, and IWM at a 52w high is the same stretched tape. |
| **rates** | **CHECK**. ^MOVE is at the 7.5th 5d rank and -37.4% from its 52w high while TLT sits 1.03% off its 52w LOW. Bond vol at a floor into a print, with the instrument at the floor, is an unexamined interaction. C3. |
| credit | CHECK-lite, count first. HYG is AT its 52w high while LQD is +1.16% off its 52w LOW. Friday killed the same divergence on NFP for N=2; CPI has 323 events so the count may exist. Registry method trap says count before measuring. C10-adjacent, folded into C3's checker. |
| **gold_miners** | **CHECK**. Registry kills GLD into CPI (underperforms its own drift, and conditioning on gold already rallying selects the crash tail). GDX is a different instrument with miner leverage and is at rank5d 100. Whether the registry kill transfers is the question. C4. |
| other_metals | dismissed as an event cell. SLV's interest today is the drawdown-thrust price state (C7), not the print. |
| **energy** | **CHECK**. EOG rank5d 0.4, XOP 3.6, CVX 4.4, USO 5.2, XLE 10.7 — energy is the week's washout, and it lands two sessions before a CPI whose largest swing component is energy. Unexamined. C5. |
| dollar_fx | dismissed. Registry: "DX into CPI is -3.5 bps, which pays neither the 1.5 bp futures nor the 6 bp UUP round trip", and the conditional cell it was tested in has occurred zero times in 318 CPI events. Today's DX state (rank21 13.5 inside rank63 68.5) is a different cell but sits on top of a base rate that does not pay costs. The FX look today is the price state, C11. |
| international | dismissed. EFA/EWJ at 52w highs is the same synchronized-high state as C9, which is the better-specified version; a CPI anchor on a foreign index adds a translation layer without a mechanism. |
| volatility | dismissed. Registry: post-CPI vol crush "died after 2018; the era check kills it". |

### ppi (2026-08-13, +3 td)

**PPI has never been swept in this repo.** `event_seasonality_sweep_2026-08-06.md`
covers nfp, cpi, fomc, minutes, opex, quad witching, vix expiry, Jackson Hole
and elections. The word PPI does not appear in it. The calendar carries 323 PPI
events back to 2000. This is the single largest unexamined surface today.

| class | verdict |
|---|---|
| **us_large** | **CHECK**. Two questions: does PPI day itself have a fingerprint on SPY, and does the back-to-back CPI-then-PPI pair (this week's exact shape) behave differently from a lone print? C1. |
| **rates** | **CHECK**. PPI is a rates event before it is an equity event, and TLT at a 52w low is the state it lands on. C2. |
| credit | dismissed for today, deferred to the C3 checker's count-first pass; one unexamined event does not justify two thin cells. |
| us_small / gold_miners / other_metals / energy / dollar_fx / international / volatility | dismissed. PPI is being opened for the first time today; the honest move is to test it on the two classes with a mechanism (the index and the curve) rather than spray eight cells across an event with no prior and then report the best one. That is exactly the search a multiplicity correction would have to charge for. If C1/C2 show a pulse, the other classes become tomorrow's map. |

### vix_expiry (2026-08-19, +7 td) — the whole row is DISMISSED

Registry, three separate entries: VIX-expiry-week drift is "mid-month position
plus noise" with the mechanism falsified inside its own window; pre-expiry
short-vol carry (long SVXY into VIX expiry) died on a gate-matched control;
post-NFP/post-FOMC/post-VIX-expiry vol cells were swept and empty. No class
rescues an event whose own drift is dead.

### opex (2026-08-21, +9 td) — DISMISSED except one live-book note

- opex x us_large: dead. "The run into August opex" is +0.342% over 26 years
  against SPY's +0.374% unconditional h=10 drift, and 2010+ is -0.514%.
- opex x volatility: this is the event sleeve's live **V4** trade (long SVXY
  post-opex MOC to +3 sessions). Pitching it would be re-running the book.
- opex x every other class: dismissed. The anchor's own equity and vol cells
  are dead or already traded; a sector or FX cross keyed on it inherits a dead
  anchor.

### jackson_hole (2026-08-28, +14 td) — DISMISSED for today, on horizon

Swept 2026-08-06: into-keynote +59 bps, t 1.6, N 26, "fades to nothing 2013+.
Not tradeable, N tiny." Separately it is 14 td out, beyond the 1-10 td default,
so even a live version would be pitched around 2026-08-20, not today.

---

## 2. Tape extremes by class

| class | what is extreme | verdict |
|---|---|---|
| us_large | SPY and ^GSPC at exactly their 52w high, +3.51% / 5d (rank5d 94.8), z10 1.66, 10.3% above the 200d. VIX 14.9. | The stretched-high cell itself is registry-dead twice over (weekend-risk discount at a stretched high is a pre-2013 fossil; the fragility dial as a directional read at a 52w high is dead at 5 episodes). What is NOT dead is the company it is keeping: **C9**, the synchronized high across equity, international and credit. |
| us_small | IWM -0.05% from its 52w high, rank5d 89.3 but rank21 38.5. | Dismissed as its own idea. IWM is confirming, not diverging, and the small-cap calendar cells (T3/T4) are event sleeve property. |
| rates | TLT 1.03% off its 52w LOW, rank21 23.0; IEF -3.69% from its high; ^TNX rank21 74.2 / rank63 82.1 but rank5d 15.9 — yields FELL last week. ^MOVE -13.2% / 5d, rank5d 7.5, -37.4% from its 52w high. | **CHECK** twice: the bond-vol floor into CPI (C3) and PPI's first-ever look (C2). The plain "TLT at the 52w floor" long is on the watchlist, parked to a non-midterm NFP. |
| credit | HYG AT its 52w high (rank5d 92.5), LQD -3.16% from its high and +1.16% off its low. Credit is confirming equity while duration lags. | The divergence pair is a count-first question (folded into C3's checker). The confirming half is the interesting half: **C9**. |
| gold_miners | GDX +21.31% / 5d (rank5d **100.0**), z10 1.92, but still -22.4% from its 52w high and only +2.88% above its 200d. NEM +20.56%, XME +14.99%. | **CHECK** twice: the miner-over-metal thrust (C6) and the CPI cross (C4). This is the loudest thing on the tape and it would be indefensible to leave it unexamined. |
| other_metals | SLV +9.82% / 5d yet -45.55% from its 52w high, -9.82% below its 200d, rank63 13.1. GLD +7.25% / 5d, -19.65% from its high. CEF rank5d 96.4 with ret63d -12.33. | **CHECK**. A violent thrust from deep inside a drawdown is a distinct state from a thrust at a high, and the repo has never separated them. C7. |
| energy | EOG **rank5d 0.4** (the tape's single most extreme reading), XOP 3.6, CVX 4.4, VLO 7.5, USO rank5d 5.2 with rank63 5.2, XLE rank5d 10.7 against rank21 66.7. | **CHECK**, event-crossed (C5). The outright long is DOA on its own: a 5d washout inside 21d strength is materially the book's **LT Trend ST OS** setup, which is exactly why Friday killed the USO version before spending a check. |
| utilities (equity, rate-proxy) | The tape's deepest coherent washout: SRE z10 -2.78 (rank21 0.8, rank63 1.6), CNP -2.10, EIX -2.06, AEP -2.03, NEE -2.00, DTE -1.90, PEG -1.80, ETR -1.76, D -1.61; XLU z10 -2.17, rank21 12.7. | **CHECK, but only in a new form** (C10). Friday killed the outright XLU long (episodes -0.123% vs +0.207% own drift, and the SPY-near-high gate that fires today HURTS) and the XLU/SPY pair (t -0.65, bootstrap 0.774). The one thing Friday did not condition on: utilities fell all week while ^TNX rank5d sat at 15.9, i.e. their driver moved the helpful way. Decoupling is a different trigger from washout. If that conditioner does not change the numbers, this dies for the third time. |
| dollar_fx | DX-Y z10 -1.92, rank21 13.5, but rank63 68.5 and only -1.98% from its 52w high. UUP z10 -1.57. | **CHECK** as a price state, not an event cell (C11), and in futures rather than UUP per the registry cost kill. Named risk up front: the long-DX fingerprint from Friday is inside the repetition window, so any surviving form owes a `changed_since`. |
| international | EFA at its 52w high, EWJ -0.07% off, FXI rank21 96.8 but -11.77% from its high, EEM rank63 **1.6** (the laggard) with rank5d 74.6, EWZ rank5d 10.7. | Folded into **C9**. A standalone EEM-laggard-snapback is registry-dead: "long the deep 63d laggard that is snapping back does not continue" (SMH/QQQ form, episode N=57, +0.27%, t=0.80). EEM is the same trade with a different ticker. |
| volatility | VIX 14.9 (-52.0% from its 52w high, -19.7% below its 200d), VIX3M 18.72, UVXY at its 52w low, **^SKEW rank5d 8.3 / rank21 5.6** and -8.4% below its 200d, ^MOVE at a floor. | **CHECK**. Crushed skew at a 52w equity high is the one volatility state today that is neither an event cell nor in the registry: SKEW appears in the risk dashboard's ticker list and in no study. C8. The tradeable expression is SPY/SVXY direction, not options (no options in v1). |
| single-name oddities | GLW +19.84% / 5d with rank21 7.1; IBM -19.65% / 21d with rank5d 84.9; QCOM +13.72% / 5d with rank21 17.9; ORCL +13.21% / 5d yet -54.87% from its high; WHR +15.41% with d200 -27.84; REGN z10 3.05; MSFT z10 2.60 with rank21 99.6. | Dismissed as a cohort. This is the laggard-snapback family the registry already killed at index level, and the single-name version adds idiosyncratic and earnings risk on top of a dead base rate (CSCO prints in 2 td, AMAT in 3, and six more names inside 10 td). Not worth a check ahead of C6/C7, which are the same snapback question asked where a mechanism exists. |
| semis | SMH rank21 13.1 / rank63 7.5 but rank5d 90.5 and +27.8% above its 200d; MU d200 +64.1, AMD +51.9, INTC +48.2 with rank63 1.2. | Dismissed. Explicitly the registry's SMH/QQQ laggard-snapback kill, re-verified on this exact pair 2026-08-07. |

---

## 3. Live seasonal and cycle cells

From `seasonality.board_candidates` (board asof 2026-08-05) and the cycle state.

- **Midterm de-risk, book level** (conviction A): book win 56.4% vs 64.9%
  all-years over 1099 midterm trades, +0.24R vs +0.43R. Treated as a
  **conditioner on every candidate**, not an idea. Every checker is told to
  split midterm vs non-midterm, because Friday's sweep died on exactly that
  split four separate times.
- **Midterm sleeve tilts** (OVS -12.2pp, LT Trend ST OS -14.0pp, Indices
  Oversold Bounce -5.5pp): these grade the systematic book's own strategies.
  Pitching against them would be re-running the book. Dismissed as ideas,
  retained as the reason C5's energy cell must not be an outright dip-buy.
- **CBOE put/call depressed** (total 0.69, 4th pctile; equity 0.46, 6th
  pctile), conviction B, FDR borderline. Dismissed as an idea: this is the
  live `Equity P/C Complacency` fragility signal and the `PC_FEAR_BANDS`
  sizing input. It is book machinery. Retained as context — it is the same
  complacency family as C8's crushed skew, and the C8 checker is told to
  control for P/C so the two are not one bet wearing two hats.
- **Mid-August midterm seasonality**: registry-dead. N=6, carried entirely by
  2002 (+8.68%), drop-two-best negative, and the midterm restriction
  anti-works at 21 td. No calendar-position idea today.
- **A+B-grade seasonal setups**: zero flagged across both channels.

---

## 4. Watchlist

One active entry, one verdict owed.

| entry | trigger | today's value | verdict |
|---|---|---|---|
| Long TLT from the NFP close to +3 td with the long end at its 52w floor (added 2026-08-07, expires 2027-02-15) | Cycle year. Midterm-dead (+0.071%, N=12, 58% hit, t=0.17), alive outside midterms (+0.978%, N=13, 92% hit, t=2.72, bootstrap 0.021). Turns on at the first non-midterm NFP, 2027-01. | The price half is still armed: TLT closed 1.03% off its 52w low. The calendar half is not: the next NFP is 2026-09-04, which is **still a midterm-year print**, and the entry anchors on an NFP close in any case. | **PASS**, trigger unchanged. Not expired, stays parked. |

No entries in `expired`. Nothing to prune.

---

## 5. Axis feedback

`scoreboard` carries 1 lifetime pitched idea and **0 graded**. There is no
per-axis or per-grade signal to read yet, and pretending otherwise would be
fitting to a sample of one. Noted and moved on; this section becomes real once
outcomes accrue.

---

## 6. Selected candidates

Eleven, from the map above. Seven axes, nine asset classes, five
event-anchored and six price-state-anchored.

| id | candidate | axis | class | anchor |
|---|---|---|---|---|
| C1 | PPI day fingerprint on SPY, and the back-to-back CPI-then-PPI pair | `event_fingerprint` | us_large | event |
| C2 | PPI on the curve — TLT/IEF, with the long end at its 52w floor | `event_fingerprint` | rates | event |
| C3 | Long duration into CPI while bond vol (^MOVE) sits in its bottom decile | `interaction_cell` | rates + volatility | event |
| C4 | GDX into CPI on a miner thrust (the instrument the GLD kill did not test) | `event_fingerprint` | gold_miners | event |
| C5 | Energy's 5d washout into a CPI print, in a form that is not the book's dip-buy | `interaction_cell` | energy | event |
| C6 | Miner-over-metal: GDX against GLD after an extreme 5d thrust spread | `relative_value` | gold_miners + gold | price |
| C7 | Silver thrust from deep inside a drawdown, distinct from a thrust at a high | `historical_analogue` | other_metals | price |
| C8 | Crushed skew at a 52w equity high (^SKEW bottom decile), controlled for P/C | `flow_mechanics` | us_large + volatility | price |
| C9 | Synchronized 52w high across SPY, EFA and HYG — credit confirming equity | `interaction_cell` | intl + credit + us_large | price |
| C10 | Utilities falling while yields fell — decoupling, not washout | `inversion` | utilities/rates | price |
| C11 | Dollar pullback inside an uptrend, expressed in DX futures not UUP | `instrument_translation` | dollar_fx | price |

Standing instruction to every checker: lag=1 MOC-tomorrow entry, episode-level
statistics or nothing, control against the instrument's own drift over the same
horizon, and split midterm vs non-midterm. Kill on substance and name which
kill; sample size alone is not a verdict.
