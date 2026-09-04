# Surface map — 2026-08-11 (Tuesday, midterm year)

> **AMENDMENT, written after stage C (do not read section 2 without it).**
> The C4/C8/C11 checker found a real bug in `02_price_state_recon.py`:
> `pitch_lab.pct_rank` takes a PRICE series and differences it internally, and
> the recon passed it a RETURN series, so every rank-based state ranked
> `pct_change(n)` OF `pct_change(n)`. States 1, 3, 4, 5, 7 and 8 were measuring
> a second difference nobody meant to compute. States 2 (52w extremes) and 6
> (`USO 1d >= +5%`, which never went through pct_rank) are unaffected, so the
> energy candidates C3 and C5 stand on correct numbers. The script is fixed and
> re-run; corrected readings are in section 2b below, and where they change a
> verdict the verdict is rewritten rather than quietly kept. Two of section 2's
> dismissals — utilities and semis — were NOT earned by the numbers I quoted
> for them, and section 2b says so.


Written BEFORE any candidate was selected. Every cell below carries a verdict.
Numbers come from `01_event_class_recon.py` (event x class, tdom-matched
control, lag=1 MOC entry, declustered 5 td) and `02_price_state_recon.py`
(tape extremes vs each instrument's own drift). Raw table:
`01_event_class_recon.csv`.

State: pipeline green, no warnings, freshest bar 2026-08-10 = the prior
session. Fragility dial ma10(63d) 63.7 (high), exposure leg 0.0x, P/C fear
OFF at the 49th pctile, Low Absorption Ratio the only signal on.

**The structural fact of the morning:** today IS the session before CPI. An
MOC entry tonight is a pre-print position, so h=1 is the CPI session itself
and h=2 adds PPI. The lag=1 convention lines up exactly with the anchor
"2 sessions before a CPI", which is what every CPI row below measures.

---

## 1. Every live calendar event x every asset class

Events in the [-5, +15] td window: nfp 2026-08-07 (past, -2), **cpi 2026-08-12
(+1)**, **ppi 2026-08-13 (+2)**, vix_expiry 2026-08-19 (+6), opex 2026-08-21
(+8), jackson_hole 2026-08-28 (+13).

### CPI (the live one — today is its eve). Excess = mean minus tdom-matched control.

| class | proxy | h=1 excess | h=3 excess | verdict |
|---|---|---|---|---|
| volatility | SVXY | **+0.280** (hit 64.8, p 0.0001) | **+1.130** (hit 71.0, p 0.0000, N=176) | **CHECK — the largest cell on the board. Collides with the registry's "post-CPI vol crush died after 2018", which anchors AFTER the print; this one holds THROUGH it. Distinctness and the Feb-2018 -1x/-0.5x leverage break are the checks.** |
| us_large | QQQ | +0.072 (hit 57.4, p 0.005) | +0.206 (hit 60.6, p 0.0001, N=312) | **CHECK as a spread against SPY, whose h=3 excess is only +0.059 and whose h=2 is negative. Beta-neutrality is the kill risk (2026-08-10 GDX/GLD lesson).** |
| us_large | SPY | -0.015 | +0.059 | dismissed outright: worse than QQQ at every horizon, and h=2/h=5/h=10 excess are all negative. It survives only as the short leg of the spread. |
| us_small | IWM | -0.026 | -0.064 | DISMISS. Negative excess at h=1,2,3,5,10 without exception. The worst equity cell on the CPI row. |
| rates | TLT | +0.004 | +0.052 | DISMISS. 2026-08-10 already killed long TLT into CPI against a tdom control (+6.7 bps, 49.7% hit); this run reproduces it at +0.4 bps on h=1. The rates edge is on the PPI session, not CPI — see watchlist. |
| rates | IEF | +0.025 | +0.017 | DISMISS, same cell with less duration and therefore less of nothing. |
| credit | LQD | +0.012 | +0.036 (hit 64.0, p 0.0000) | DISMISS on cost. The hit rate is real and the edge is 3.6 bps, under an LQD round trip. Recorded because the high hit rate looks tradeable and is not. |
| credit | HYG | +0.008 | +0.074 (hit 62.4, p 0.0001) | CHECK, marginal. Bigger than LQD but still ~7 bps against ~5 bps of cost, so it needs the hit rate to carry it. Low priority. |
| gold_miners | GLD | +0.100 | +0.042 | DISMISS. Registry already holds "GLD into CPI underperforms its own drift"; this run agrees once the horizon passes h=1. |
| gold_miners | GDX | +0.308 (h=1) | +0.014 | DISMISS on repetition, not on numbers: long GDX into this exact print was pitched 2026-08-10 and is live to 2026-08-17. |
| other_metals | SLV | +0.109 | +0.116 | CHECK, but as the price-state thrust cell (below), not the event cell — the CPI anchor adds ~11 bps and the thrust adds ~120. |
| energy | USO | +0.131 | +0.277 (N=241) | CHECK as the interaction with today's 6.7% one-day pop, which is the cross the 2026-08-07 post-mortem said never gets done. |
| energy | DBC | +0.075 | +0.048 | DISMISS standalone; folded into the energy check as a vehicle alternative. |
| energy | UNG | +0.086 | -0.174 | CHECK the SHORT side: h=5 is -0.461 excess at a 43.4% hit. Registry already documents UNG's -0.90%/10td structural bleed, which is tailwind for a short and a squeeze risk at a 52w low +5.3%. |
| dollar_fx | UUP | -0.060 | +0.024 | DISMISS. Wrong-signed at h=1, and the registry already retired UUP on cost (6 bps cannot pay its drag). |
| dollar_fx | DX-Y.NYB | -0.038 | -0.002 | DISMISS. Flat to negative at every horizon, and a DX idea was killed yesterday on definition fragility. |
| international | EEM | +0.097 (h=1) | +0.036 | DISMISS. Positive only at h=1; h=2, h=5 and h=10 excess are all negative (h=10 -0.354). |
| international | EFA | +0.016 | +0.006 | DISMISS, flat by inspection. |
| international | FXI | +0.153 (h=1) | +0.046 | DISMISS. h=1 only, 50.8% hit, p 0.43. |

### PPI (2026-08-13, +2 td)

Not today's trade. The watchlist entry (below) establishes the effect is
exactly ONE session wide — the print session, entered at the close before —
so it turns on with tomorrow's run, not this one. Cells scanned anyway at
today's offset: GDX h=10 -0.764 excess, EEM h=10 -0.557, SLV h=10 -0.522, SVXY
h=2 -0.502. All negative, all at long horizons off a 3-session-early anchor,
i.e. this is the "your anchor is stale" row. DISMISS the whole row for today.

### VIX expiry (2026-08-19, +6 td)

GDX h=5 +0.528 excess (hit 50.4, p 0.47 — mean without a record, so tail),
SVXY h=2/h=3 NEGATIVE excess (-0.361, -0.394). DISMISS. Registry already holds
"post-VIX-expiry vol cells swept and empty" and "VIX-expiry-week drift" as
mid-month position plus noise; entering 7 sessions early does not fix that.

### Opex (2026-08-21, +8 td)

SVXY h=10 +0.589 excess and GDX h=3 +0.485 look alive, but the anchor is 9
sessions early, which makes these month-position cells wearing an opex label —
the exact trap `d4b_vix_week_vs_monthpos.py` documented. The event sleeve's V4
already owns the real opex-anchored short-vol window (entry AT the opex close),
and it fires 2026-08-21 on its own. DISMISS: do not front-run a sleeve trade
with a worse-anchored version of it.

### Jackson Hole (2026-08-28, +13 td)

IWM h=1 -0.420 excess at a 26.9% hit (N=26, sign p for the short 0.005), SPY
h=5 +0.595, TLT h=10 +0.581, FXI h=1 -0.829. CHECK the IWM short, with the
explicit suspicion that a 13-sessions-early anchor is mid-August seasonality
and not a Jackson Hole effect; the registry's "midterm mid-August seasonality"
kill (N=6, carried by 2002) is adjacent but is a different, midterm-only cell.
Everything else on this row is N~20-26 at a meaningless offset: DISMISS.

### NFP (2026-08-07, -2 td, past)

No forward cell. The one live artifact is the TLT watchlist entry, verdict
below.

---

## 2. Tape extremes, by class

Sorted from `00_tape_sort.py` over all 218 names, not from the handful I
arrived with.

- **gold / miners / metals — the loudest state on the tape.** NEM +22.9% 5d,
  GDX +19.0 (rank5 99.6, z10 1.93), XME +14.0, SLV +13.2, CEF +11.9, FCX
  +10.8, GLD +8.3 (rank5 97.6). Recon: on a GDX 5d rank >= 95 trigger (N=81),
  SLV pays +1.51% at h=5 for +1.26 excess at a 65.0% hit (p 0.005) and GLD
  +0.80 for +0.57 (65.0%, p 0.005). CHECK SLV as the instrument translation.
  Standing problem: **the book is already long this** via the 2026-08-10 GDX
  pitch running to 2026-08-17, so basket correlation is the live kill risk.
- **volatility — VIX 15.46 (63d rank 28.6), SVXY AT its 52w high, UVXY 0.3%
  off its 52w low.** Short vol is already crowded and already working, which
  is the honest counterweight to the CPI cell above. CHECK inside C1.
- **rates and credit at the floor.** TLT 0.17% off its 52w low, LQD 0.60%,
  IEF 0.65%, PEG 0.0%; ^TNX 63d rank 89.7 and 0.97% off its 52w high. The
  duration-divergence cross (SPY at a 52w high WHILE TLT is at a 52w low) has
  **2 declustered episodes in 25 years**. DISMISS on episode count — the
  "count occurrences before measuring edge" rule.
- **utilities and bond proxies washed out.** XLU 21d rank 2.8 and z10 -2.08;
  PNW, AEP, CNP, DTE, EIX, ETR, SRE, CMS, PEG, NEE, SO, DUK, D all at the
  bottom of the 218. Recon on XLU 21d rank <= 5 (N=126): excess is NEGATIVE at
  h=1, 2, 3, 5 and 10 for XLU itself (h=5 -0.40) and worse for VNQ (h=5
  -0.65); XLP is +0.00 at h=5. DISMISS, and this is the third independent kill
  of the utilities-washout family (2026-08-07 outright and paired,
  2026-08-10 on yield direction).
- **energy thrust.** USO +6.7% in one session (+15.8% 21d), XLE +4.7%, OIH
  rank5 95.6, SLB +7.9%, VLO at a 52w high. Recon on USO 1d >= +5% (N=68):
  XLE pays +1.02% at h=3 for +0.88 excess at a **67.2% hit (p 0.003)**, DBC
  +0.72 excess at h=2. CHECK — this is the strongest price-state cell of the
  morning and it is not the one anybody would have guessed from the metals
  headline.
- **healthcare leadership.** XLV 63d rank 100 and AT its 52w high; ABT, AMGN,
  MDT, SYK, IHI all at 100; IBB at a 52w high, XBI +7.3% 5d, LLY +9.9%, PFE
  +8.1%, REGN z10 3.24. Recon on XLV 63d rank >= 98 (N=109): XLV's own excess
  is -0.15 at h=5 and +0.03 at h=1, i.e. the leader does not continue; IBB
  +0.21 at h=2 (p 0.042). CHECK IBB at low priority — the sector is loud and
  the cell is quiet, which is worth recording.
- **semis are the 63d laggard.** SMH 63d rank 0.8, MU 1.2, INTC 0.4, AMAT 21d
  rank 6.3 — while MU sits +60% above its 200d and INTC +41.6%. Recon on SMH
  63d rank <= 2 (N=83): excess NEGATIVE at h=2, 3 and 5. DISMISS, and the
  registry already holds the SMH/QQQ laggard-snapback as dead.
- **EM split.** EEM 63d rank 0.4 (the single worst 63d rank in the universe)
  while FXI 21d rank is 96.8. Joint cell N=15 with incoherent signs (+0.61 at
  h=1 on a 40% hit, -1.08 at h=3). DISMISS as noise.
- **dollar.** DX-Y z10 -1.72, 21d rank 17.9 inside a 63d rank of 69.4. Recon
  on the washout-inside-uptrend cell (N=50): +0.05 to +0.11 excess, 50-56%
  hit. DISMISS — and yesterday killed the same cell on definition fragility.
- **single-name washouts** ROK -9.5% 5d, CVS -9.2% (21d rank 4.0). DISMISS:
  single-name mean reversion in the liquid universe is materially the book's
  LT Trend ST OS / Oversold Low Volume territory, which is the anti-rip-off
  rule.
- **extension extremes** MU +60.0% above its 200d, RHI +52.7, AMD +47.0.
  DISMISS: fading extension in single names is the 3x Overbot Fade thesis
  without the leverage, and the book already runs it.

---

## 2b. Corrected price-state readings (post-bugfix)

Re-run of `02_price_state_recon.py` with `pct_rank` fed prices. Excess is over
each instrument's own drift at the same horizon, lag=1, declustered 5 td.

| state | corrected reading | verdict now |
|---|---|---|
| utilities washout (XLU 21d rank <= 5, N=112) | XLU h=1 **+0.22** excess (59.5% hit, p 0.029), h=5 **+0.28** (59.5%, p 0.029), h=10 **+0.34** (63.1%, p 0.004); XLP h=10 **+0.45** (64.9%, p 0.001); VNQ h=10 -0.35 | **My section-2 dismissal was wrong.** The cell is mildly POSITIVE, not negative. It is NOT dismissed on these numbers and I am not claiming it is. What stands instead: the family has two independent, correct-method kills inside three sessions on adjacent triggers (2026-08-07 outright XLU, episodes -0.123% vs +0.207% own drift, with the SPY-near-high gate that fires today HURTING; 2026-08-10 the yield-direction form, which needs -15 bps of yield move where today offers -8.5). Those used a z10 washout rather than a 21d rank, so they are adjacent and not identical. Honest status: **not carried to a verdict this morning**, deprioritised behind two recent kills rather than refuted. Parked for a future run. |
| semis laggard (SMH 63d rank <= 2, N=108) | SMH h=3 **+0.44** excess (58.9%, p 0.041); XLK h=3 **+0.62** (64.5%, p 0.002), h=10 +0.26 (61.3%, p 0.013) | **My section-2 dismissal was wrong** for the same reason. Registry support is real but on a different construction (the dead entry is the SMH/QQQ PAIR at h=5, +0.27% on 57 episodes at t=0.80, with the trigger over-selecting bear tape by +29pp). The outright is not the pair. Honest status: **not carried to a verdict this morning**. Parked. |
| metals thrust (GDX 5d rank >= 95) | correct trigger is 272 days / 120 episodes, not the 105/80 the buggy statistic selected — 7.6% overlap. Re-derived by the checker: SLV h=5 **+0.217% against its own +0.250% drift = -0.033 excess** (welch t -0.09) | DISMISS, and now on correct numbers. The buggy cell's "+1.26 excess" does not exist. |
| healthcare leadership (XLV 63d rank >= 98) | today's real XLV 63d rank is 100.0 while the buggy statistic read 15.5, i.e. today was not even in the measured population. Correct trigger, IBB excess: h=1 -0.267, h=2 -0.533, h=3 -0.393, h=5 -0.715, h=10 -0.959, bootstrap P(mean<=0)=0.985 | DISMISS, sign inverted under the correct definition. |
| EM split (EEM 63d rank <= 5 & FXI 21d rank >= 90) | N drops 15 -> **7**, signs still incoherent (FXI h=2 -1.49, h=10 +3.59, hit 33%) | DISMISS, unchanged and stronger. |
| dollar washout (DX z10 <= -1.5 & 63d rank >= 60) | N drops 50 -> **24**; excess -0.13, -0.17, -0.10, +0.07, -0.06 across h=1..10 | DISMISS, unchanged. |
| energy thrust (USO 1d >= +5%) | **UNAFFECTED** — never used pct_rank. XLE h=3 +0.88 excess, 67.2% hit, p 0.003, N=68 | CHECK, and it is the candidate that carried the morning. |
| SPY 52wh & TLT 52wl | **UNAFFECTED** — 2 declustered episodes | DISMISS on episode count. |

## 3. Live seasonal and cycle cells

- **Midterm year (year%4==2).** A conditioner on everything above, never an
  idea. The seasonal board's own read is de-risk: book win 56.4% vs 64.9%
  all-years (n=1099), OVS 55.4 vs 67.6, LT Trend ST OS 53.7 vs 67.6. Registry
  also holds "midterm mid-August seasonality" dead (N=6, carried by 2002) and
  "midterm-year pre-FOMC drift, ungated" dead. Applied as a required era/cycle
  cut inside every check below, not pitched on its own.
- **Seasonal board: 0 A+B-grade setups across 2 channels.** Nothing to select.
- **Mid-August calendar position.** Feeds the IWM candidate only, and its
  suspicion is written into the candidate itself.
- **P/C complacency channel** (board conviction B, total P/C pctile 4): the
  board's own note caps it at B on a thin history and the live pc_fear state
  is OFF at the 49th pctile of the equity series. The 2026-08-10 run already
  killed the crushed-skew version of this. DISMISS.

---

## 4. Watchlist verdicts (all three active entries)

1. **Long TLT from the NFP close to +3td at the 52w rates floor** (added
   2026-08-07). Trigger is the cycle year: alive outside midterms (+0.978%,
   N=13, 92% hit), midterm-dead (+0.071%, t=0.17). **PASS.** The price half is
   armed today — TLT closed 0.17% off its 52w low, tighter than on 08-10 — but
   the next NFP, 2026-09-04, is still a midterm print. Turns on 2027-01.
2. **Long TLT across the PPI print session** (added 2026-08-10). Trigger is
   the entry date: it turns on when the run date is the session immediately
   before a PPI release. PPI is 2026-08-13, so that session is **2026-08-12,
   which is tomorrow's run, not today's**. **PASS, one session early.** Today's
   run reproduces why: at today's 3-sessions-early anchor the whole PPI row is
   negative, and the watchlist entry itself states every pre-print session is
   dead 2018+. Entering tonight would buy exactly the dead sessions.
3. **Credit-quality divergence, long LQD against short HYG at joint 52w
   extremes** (added 2026-08-10). Trigger is >= 8 declustered episodes over
   >= 3 distinct years excluding 2018. **PASS, unchanged.** The joint state is
   still live on today's tape (HYG 0.16% off its 52w high, LQD 0.60% off its
   52w low) but it is the SAME cluster that began 2026-07-22, so the episode
   count is still 4 and three of them are 2018. Nothing has moved.

---

## 5. Scoreboard read

2 ideas pitched lifetime, **0 graded**. The per-axis and per-grade splits are
empty, so there is no feedback to read yet and none is claimed. Revisit once a
handful have graded.

---

## 6. Selected candidates (stage B2)

Eleven candidates, 8 asset classes, 7 novelty axes, at least one event-anchored
and at least one price-state-anchored, plus the cross of the two.

| # | candidate | class | axis |
|---|---|---|---|
| C1 | Long SVXY held through the CPI print, exit +3 td | volatility | event_fingerprint |
| C2 | Long QQQ against short SPY into the CPI print, beta-neutral | us_large | relative_value |
| C3 | Long XLE on the crude one-day thrust | energy | flow_mechanics |
| C4 | Long SLV on the miner-led metals thrust | other_metals | instrument_translation |
| C5 | Crude thrust CROSSED with the CPI print (does the event add to C3?) | energy | interaction_cell |
| C6 | Short UNG through the CPI window | energy | inversion |
| C7 | Short IWM in the back half of August | us_small | event_fingerprint |
| C8 | Long IBB on healthcare 63d leadership at a 52w high | us_large | interaction_cell |
| C9 | Long HYG into the CPI print | credit | event_fingerprint |
| C10 | Nearest-neighbour tapes to today (52w high + high dial + metals thrust + long end at its floor) and what followed | us_large | historical_analogue |
| C11 | Long GLD as the lower-vol expression of the metals thrust | gold_miners | instrument_translation |

Pre-spent kills, recorded above rather than checked: utilities washout,
semis laggard, dollar washout, SPY/TLT duration divergence, EM split, TLT and
IEF into CPI, LQD into CPI on cost, GLD into CPI on the registry, UUP and DX
into CPI, opex and vix-expiry rows, single-name washouts, extension fades.
