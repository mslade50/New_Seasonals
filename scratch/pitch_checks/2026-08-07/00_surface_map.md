# Surface map — 2026-08-07 (Friday, midterm year)

Stage B1 artifact. Built from a SORT of all 217 tape names
(`00_survey_sort.py`), not from recall. This is the second run of the day: the
5:10 AM scheduled run swept 24 candidates and killed all of them, so every
cell below carries one of four verdicts:

- **CHECK** — open, worth a script this run
- **DEAD-AM** — already killed this morning, script named, not re-pitched
- **PASS** — examined here and dismissed, reason given
- **N/A** — event/class pairing has no mechanism worth the look, reason given

## Today's state, one paragraph

NFP prints **today** (td_ahead 0), so every "into the print" trade is already
gone and the only live NFP structure is the reaction. SPY is 0.36% off its 52w
high, rank5 96, z10 +1.46, +9.7% over its 200d: an extended, quiet tape. Dial
ma10-63d 57.2 (above the 50 sizing line), P/C fear OFF at the 69th pctile, one
fragility signal on (Low Absorption Ratio, 10th pctile). Cycle year is midterm.

The single loudest cross-sectional fact in the sort: **utilities are being
liquidated**. Ten of the twelve lowest z10 readings in the entire 217-name tape
are utilities (SRE -2.76, EIX -2.38, XLU -2.31, D -2.06, PEG -2.06, NEE -2.03,
AEP -1.97, CNP -1.90, ETR -1.89, DTE -1.59), with PEG at a 52w low. Alongside
it: TLT 0.73% off its 52w low, IEF 0.85% off, LQD 0.98% off, ^TNX rank63 87.7
at 4.67. One story, two expressions — the long end is at the floor and its
equity proxy is being dumped into a jobs print.

---

## 1. Calendar events x asset classes

Six live events. Ten classes. Sixty cells. Verdict on every one.

### nfp — 2026-08-07, td_ahead 0 (TODAY, pre-open)

| class | verdict | note |
|---|---|---|
| us_large | DEAD-AM | `c9_nfp_into_cpi.py`. +0.129% vs +0.221% all-days control. Midterm cell -0.248% at +5td, 42% hit; rank5>=90 cell -0.368% at +3td. Today is both. |
| us_small | **CHECK** | IWM never tested against NFP this morning. Rate-sensitive, small-cap, and the print is a rates event. Distinct from the SPY cell. |
| rates | **CHECK** | **The cell the AM run missed.** TLT/IEF sitting 0.73%/0.85% off 52w lows into a jobs print. Never examined in any form. Highest-value open cell on the board. |
| credit | **CHECK** | HYG 0.11% off its 52w HIGH while LQD is 0.98% off its 52w LOW. That divergence is pure duration, and NFP is what repriecs duration. |
| gold_miners | PASS | GLD/GDX were examined this morning but anchored on CPI (`d5_gold_pre_cpi.py`) and on drawdown state (`c8`). The NFP anchor is open in principle, but GDX +9.3% in 5d means today enters after the move, not before it; the reaction cell is contaminated by the run-up. Deferred, not dismissed on merit. |
| other_metals | PASS | SLV is the same trade as GDX with more noise (atr% 3.44, rank63 11.1). No separate mechanism vs the print. |
| energy | N/A | Oil's NFP sensitivity runs through the dollar, which is tested directly below. USO -6.75% in 5d is its own supply story, unrelated to payrolls. |
| dollar_fx | **CHECK** | DX rank63 90.5 but z10 -1.54: strong on the quarter, soft into the week. NFP is the single biggest scheduled dollar event. AM run only tested dollar-into-CPI (`d6`, N=0 cell). |
| international | PASS | EEM rank63 0.4 (dead last of 217) is a dollar expression, already covered by the dollar_fx cell. FXI rank21 96.0 is a China-policy story with no payroll mechanism. |
| volatility | DEAD-AM | `c12_svxy_pre_expiry.py` + `c12b`. Gate-matched control ate it, 2018+ t=0.18, one 8td window -24.8%. SVXY is also at its exact 52w high today, the worst possible entry for a short-vol carry trade. |

### cpi — 2026-08-12, td_ahead 3

| class | verdict | note |
|---|---|---|
| us_large | DEAD-AM | `c9_nfp_into_cpi.py` is exactly this window (NFP close -> pre-CPI close). Killed above. |
| us_small | PASS | Same window as the SPY cell, same era problem; IWM adds noise, not a mechanism. Covered by the NFP x us_small check instead, which is the cleaner anchor. |
| rates | PASS | Folded into the NFP x rates check: with CPI 3 td out, the NFP hold window and the pre-CPI window are the same window. Tested there as a CPI-in-window split rather than as a separate cell. |
| credit | PASS | Same folding as rates. |
| gold_miners | DEAD-AM | `d5_gold_pre_cpi.py`. GLD pre-CPI +0.040% loses to its own +0.092% h=2 drift; conditioning on gold already rallying selects the crash tail. Gold IS already rallying today. |
| other_metals | N/A | No independent CPI mechanism beyond gold's. |
| energy | N/A | Oil into CPI is a component-of-the-print circularity, not a tradeable anticipation. |
| dollar_fx | DEAD-AM | `d6_dollar_pre_cpi.py`. The described cell (rank21<20 inside rank63>90) has occurred ZERO times in 318 CPI events since 2000. Today is close to that state, which is exactly why it has no sample. |
| international | N/A | Second-order to the dollar cell. |
| volatility | PASS | Vol into CPI is the generic pre-event compression already covered by the dead VIX-expiry work; VIX at 15.15 leaves little to compress. |

### ppi — 2026-08-13, td_ahead 4

| class | verdict | note |
|---|---|---|
| all ten | PASS | PPI lands one session after CPI. Its window is fully nested inside the CPI window for every class, so any PPI cell is the CPI cell with a worse signal-to-noise ratio. Recorded rather than examined, deliberately. |

### vix_expiry — 2026-08-19, td_ahead 8

| class | verdict | note |
|---|---|---|
| volatility | DEAD-AM | `d4_vix_expiry_week.py` + `d4b`. Within-month paired excess +0.065% (t=0.67); mechanism falsified inside its own window since the settle day is the worst day. |
| us_large | DEAD-AM | Same scripts, equity leg of the same cell. |
| other eight | N/A | VIX settlement has no transmission to rates, credit, metals, energy, FX or international that is not routed through equity vol, which is dead above. |

### opex — 2026-08-21, td_ahead 10

| class | verdict | note |
|---|---|---|
| us_large | DEAD-AM | `d2_nfp_to_opex_run.py`. +0.342% over 26 non-overlapping years vs SPY's +0.374% unconditional h=10 drift. Worse than a random 10-day long; effect lives in 2000-2004. |
| volatility | DEAD-AM | `c12_svxy_pre_expiry.py`, the post-opex vol carry. Also the event sleeve's V4 territory. |
| other eight | PASS | Opex gamma is an equity-index mechanic. Single-stock and cross-asset opex effects exist but at 10 td out the window overlaps CPI and PPI, so nothing measured there would be attributable. |

### jackson_hole — 2026-08-28, td_ahead 15

| class | verdict | note |
|---|---|---|
| all ten | PASS | 15 td out is beyond the product's 1-10 td horizon. Any position taken now would be graded on CPI and opex, not on Powell. Revisit ~2026-08-21. |

---

## 2. Tape extremes, by class (all 217 sorted)

| class | extreme | verdict |
|---|---|---|
| us_large | SPY rank5 96.0, 0.36% off 52wh, z10 +1.46 | Context for every gate below, not a trade. Extension is a conditioner. |
| us_large single | MSFT z10 +2.61 rank21 100.0; NVDA +12.28% 5d | PASS. Single-name momentum extremes are the book's `52wh Breakout` and `ATR Extended Gap Up` territory. Anti-rip-off rule. |
| us_small | IWM rank5 70.6, rank63 20.6, 12.1% over 200d | Feeds the NFP x us_small check. |
| rates | TLT 0.73% off 52w LOW; IEF 0.85%; ^TNX rank63 87.7 | **CHECK** — the headline setup. |
| credit | HYG -0.11% off 52w HIGH; LQD 0.98% off 52w LOW | **CHECK** — duration divergence. |
| gold_miners | GDX +9.30% 5d, rank5 91.3, but -27.6% off 52wh | DEAD-AM (`c8_gdx_drawdown_restart.py`: declustering flipped h=10 from +4.41% to -2.80%). |
| other_metals | SLV -47.1% off 52wh, rank63 11.1 | PASS, folded into gold. |
| energy | UNG at its exact 52w LOW (0.00), -17.0% 21d | DEAD-AM (`c7_natgas_floor.py`: -0.90%/10td structural bleed swamps a 0.25pp edge). |
| energy | USO -6.75% 5d, rank5 10.3, +79.6% off 52w low | PASS. Two-sided: oversold on the week, extended on the year. No clean thesis, and the AM run's oil work was the UNG floor rather than USO. Recorded as genuinely unexamined. |
| dollar_fx | DX rank63 90.5, z10 -1.54 | **CHECK** via NFP. |
| international | EEM rank63 0.4 — LAST of all 217 names | PASS. A dollar expression; the dollar cell is the cleaner leg. |
| volatility | SVXY at its exact 52w HIGH; UVXY at its 52w LOW; VIX 15.15 | DEAD-AM and structurally hostile to new short-vol entry today. |
| sectors | XLU z10 -2.31, rank5 6.3; 10 of 12 lowest z10 names are utilities | DEAD-AM as an outright (`c1_xlu_washout_outright.py`: episode -0.123% vs +0.207% own drift, and the SPY-near-high gate HURTS: +0.605% ungated vs -0.123% gated). Its rates-channel cousin is alive and is the NFP x rates check. |
| sectors | SMH rank63 2.0 but rank5 85.3 | DEAD-AM (`c10_smh_qqq_laggard.py` / `r1`: lag-0 entry bug, corrected t=1.39). |

## 3. Seasonal and cycle cells

| cell | verdict |
|---|---|
| midterm mid-August | DEAD-AM. `d3_midterm_mid_august.py`: N=6 carried entirely by 2002 (+8.68%), drop-two-best negative, and the midterm restriction anti-works at 21td. |
| August opex run | DEAD-AM. `d2`, above. |
| Friday-to-Monday weekend effect | DEAD-AM. `e4`/`e5`: pre-2013 fossil, 2018+ is -0.007%, and the weekday placebo is decisive. |
| midterm as a CONDITIONER | Live. Applied inside every check below as an era/cycle split rather than as a standalone idea. |

---

## Coverage summary

- Events with a verdict on every class: **6 of 6** (nfp, cpi, ppi, vix_expiry, opex, jackson_hole)
- Cells marked CHECK this run: **5** — nfp x rates, nfp x credit, nfp x dollar_fx, nfp x us_small, plus the rates/utilities transmission leg
- Asset classes touched by candidates: **rates, credit, dollar_fx, us_small, us_large** (5, floor is 4)
- Both search modes present: event-anchored (all five CHECK cells) and price-state anchored (TLT/LQD at 52w lows, HYG at 52w high)

The AM run's blind spot is now explicit. It ran every calendar-anchored check
on SPY and every cross-asset check on a chart level. The intersection —
a jobs print hitting the long end while the long end sits at its 52w floor —
was never opened.
