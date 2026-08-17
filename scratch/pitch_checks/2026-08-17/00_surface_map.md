# Surface map — 2026-08-17 (Monday)

State: `data/pitch_state.json` generated 05:11, `warnings: []`, pipeline 7/7 green,
freshest bar **2026-08-14** (Friday, the prior session — fresh). Dial as of 08-14,
P/C data 08-14. Nothing here is stale, so no evidence line needs a staleness caveat.

One state defect worth naming: `book.staged_signals` carries two `Order_Staging`
rows with `strategy: null` and `ticker: null` (quantities 8027 and 5363, both SELL
REL_OPEN, scan date today). The tickers did not survive the state builder, so
book-overlap disclosure today leans on the ledger and the sleeve state rather than
on today's staged names. Flagged, not fixed inside a pitch run.

## Regime

| | |
|---|---|
| cycle | **midterm** (2026 % 4 == 2) |
| calendar | Aug 17, tdom 11 |
| SPY | 776.34, **-0.20% off its 52w high**, +10.4% over its 200d, 92 sessions since a 5% pullback |
| fragility | ma10-63d **80.9** — the **99.2nd percentile** of the whole 2016+ series; raw 63d 94.1, raw 21d 68.9 |
| exposure leg | **0.0x**, killed by raw-21d 68.9 > 50 |
| P/C fear | **OFF** (38.9th pctile), so the 6 family-band carriers are ZEROED at dial >= 50 |
| fragility signals on | Low Absorption Ratio only (0.321, 3rd pctile, near a high) |
| vol | VIX 14.25, VIX3M 18.46, **ratio 0.772 = the 1.2nd percentile of its trailing year** |
| skew | ^SKEW 138.36 — 5d rank 86.1 but level only the 9.1st percentile of the year |

The book is already maximally defensive by its own rules: overlay at zero, family
carriers zeroed. That is context for sizing, not an idea.

## 1. Every live calendar event x every asset class

Events in the `[-5, +15]` td window: cpi 08-12 (-3), ppi 08-13 (-2), **vix_expiry
08-19 (+2)**, **opex 08-21 (+4)**, **jackson_hole 08-28 (+9)**, nfp 09-04 (+14).

CPI and PPI are BEHIND us and no legal entry reaches them; both are dismissed as
anchors, though "what the tape did into them" is live history the price-state
lane can use. NFP at +14 td is beyond the 10 td maximum pitch horizon and is
dismissed for reach, the same dismissal watchlist entry 1 already carries.
FOMC 09-16 (+21), CPI 09-11 (+18), quad witching 09-18 (+23) — all out of reach.

The grid below is `02_event_surface.py`: today's analogue for each event (the
session sitting at the same td offset), excess over the instrument's own
unconditional drift, lag-1 entry. **It is a search**, so anything read out of it
is multiplicity-charged and needs a mechanism before it earns a check.

| class | vix_expiry -2 (N=307) | opex -4 (N=307) | jackson_hole -9 (N=25) | verdict |
|---|---|---|---|---|
| us_large | SPY h5 -0.365, QQQ h5 -0.347 | SPY h5 -0.260, QQQ h5 -0.351 | SPY h5 **-0.931**, QQQ h5 -0.802 | **CHECK** — three anchors agree on a negative equity drift, and midterm doubles it (SPY h5 -0.794 / -0.585, QQQ h10 -0.910 / -0.884). C2. |
| us_small | IWM h5 -0.191 | IWM h5 -0.203 | IWM h5 **-1.035** | folded into C2/C9; IWM is the same macro bet as SPY here |
| rates | TLT h10 +0.291 | TLT h10 **+0.365**, monotone from h1 | TLT h10 **+1.204** | **CHECK** — the single most consistent cell on the grid. C1. |
| credit | LQD h10 +0.165, HYG flat | LQD h10 +0.179 | HYG h5 -0.259, LQD +0.061 | dismissed as an anchor: LQD is the low-beta shadow of the TLT cell (registry 08-10 already found the LQD/IEF residual is +0.000pp), and the credit price state is watchlist 2, mid-cluster |
| gold_miners | GLD flat, GDX h5 -0.384 | GDX h5 -0.064 | GLD h10 +0.558, GDX +0.727 all-years but **midterm -2.920** | dismissed as an anchor: the sign flips between all-years and midterm on N=6, which is a coin, and the price-state version is C10 |
| other_metals | SLV h1 +0.146 | SLV h8 +0.279 | SLV h10 +1.738 all-years, **midterm -2.081** | same dismissal as gold; the metals event cells are cycle-unstable and the live story is 21d price state, not the calendar |
| energy | USO h8 +0.323, XLE h5 -0.316 | USO h8 +0.336, XLE h5 -0.052 | USO h5 -0.931, XLE h5 -0.898 | dismissed as an anchor (nothing coherent across the three), but the PRICE STATE is the loudest thing on the tape. C5. |
| dollar_fx | UUP flat, midterm h10 +0.446 | UUP h10 +0.055, midterm +0.527 | UUP h10 -0.273, midterm +0.593 | **dismissed.** The midterm-positive dollar reading is real but the tape offers nothing: UUP 21d rank 17.9, DX 21d rank 21.8, DX 1.9% off its high and 3.6% off its low, i.e. mid-range. Both recent DX cells are already on the kill list (08-07 shipped, 08-10 killed as a re-skin). No trigger, no new angle. |
| international | EFA h5 -0.243, EEM -0.277, FXI -0.221 | EEM h5 -0.131 | EFA h5 -0.843, EEM -1.029, FXI -1.119 | dismissed as an anchor (it is the equity cell with more beta); the live divergence is EWZ. C6. |
| volatility | ^VIX h10 +1.538, SVXY h10 -0.523 | ^VIX h3 -1.377 and h8 -1.788, SVXY h5 **+0.705** | ^VIX h5 **+6.847**, SVXY h5 -3.310 | **CHECK** — opex and Jackson Hole point opposite ways on vol, which is the interesting part. C3, C8. |

The one clean read across the whole grid: from mid-August into late August, equity
excess is negative in every class and every anchor, duration excess is positive in
every anchor, and vol excess is positive at the annual anchor. That is one macro
bet with three faces, which the red team has to price before more than one of them
ships.

## 2. Tape extremes, by class

Whole tape sorted in `00_tape_survey.py` (218 names). Extremes by class:

- **us_large** — SPY -0.20% off its 52w high, z10 +1.44; QQQ -1.91%, 63d rank 23.8.
  Nothing extreme except the proximity to the high itself. Dismissed on its own;
  it is the conditioner on everything else.
- **us_small** — IWM **at its 52w high, 0.00% off**, z10 +1.59, 63d rank 49.6. Small
  caps leading large into a double expiry. C9.
- **rates** — TLT **0.15% off its 52w low**, IEF 0.95%, LQD 0.75%: the tight rung of
  watchlist 6 is LIVE again for the first time since 08-12. ^TNX 4.70, 21d rank
  75.8. **CHECK**, and the freshness leg is the whole question. C1b.
- **credit** — HYG **0.10% off its 52w high** while LQD is 0.75% off its 52w low.
  Watchlist 2, mid-cluster since 2026-07-22, count still 4 episodes with three in
  2018. PASS.
- **gold_miners** — GDX **21d rank 100.0** (+26.01%) yet 63d rank 31.7 and -22.33%
  off its 52w high. NEM 21d rank 100.0, z10 +2.78, +29.65% in 21 days. A violent
  snapback deep inside a drawdown. C10.
- **other_metals** — SLV **63d rank 8.7** (-22.55%) against 21d rank 76.6 (+16.05%),
  and -44.62% off its 52w high. The same V as the miners, more extreme. Collides
  with the 08-10 kill "Silver thrust from deep inside a drawdown"; carried as the
  comparison leg of C10 rather than its own candidate.
- **energy** — the loudest cell on the board. **XLE 5d rank 100.0** (+7.67%) and
  -0.33% off its 52w high, XOP 99.6, OIH 97.2, VLO 5d rank 99.6 (+14.54%), CVX
  98.8, COP 98.4, WMB 97.6, HAL +4.81% Friday, SLB +3.28%. Crude joined late (USO
  +7.31% 5d) but is still 63d rank 8.3. C5.
- **dollar_fx** — nothing. See the grid row above for the dismissal.
- **international** — **EWZ z10 -1.68** (the tape's lowest), 5d rank 9.1, 63d rank
  12.7, while EEM's 5d is +1.48% and EEM's 63d rank is 2.8. FXI 5d rank 11.5 but
  its 21d rank is 60.7, so watchlist 10's intact-thrust leg fails. C6.
- **volatility** — VIX 14.25 at a 21d rank of 17.1 and -23% under its 200d; **the
  VIX/VIX3M ratio at 0.772 is the 1.2nd percentile of its trailing year**; SVXY at
  a 52w high; UVXY at a 52w low; ^SKEW rising fast (5d rank 86.1) off a low level
  (9.1st pctile). C3.
- **sectors** — **SMH 63d rank 1.6** with a 252d of +95.71% and +27.3% over its
  200d, after AVGO -5.94% and AMAT -5.12% on Friday: semis are at the very bottom
  of their own 3-month distribution while the index sits at a high. C4.
  XLF 63d rank 99.6, KRE at a 52w high, BAC/SCHW 63d rank 100 — financials are the
  leadership. Healthcare/devices 63d rank 94-100 (XLV, BDX, MDT, ABT, IHI).
  Defense at highs (ITA 0.00% off). Deep-value staples bouncing (GIS, CPB 63d rank
  100). XRT 5d rank 18.7 into its own earnings week. C7.
  The financials, healthcare, defense and staples cells are all **dismissed as
  candidates**: each is a 63d leadership reading with no fresh trigger this week
  and no event inside the horizon, i.e. momentum I would be buying at day 60 of 63.
  Named so they are visibly dismissed, not silently absent.
- **crypto_adjacent** — not in the 218-name tape at all (no BTC/COIN/MSTR series in
  the pitch tape). Dismissed for want of data, not for want of interest.

## 3. Live seasonal and cycle cells

`03_state_scan.py` section B, entry on the analogue of Aug 15-19, declustered:

| | all-years | midterm |
|---|---|---|
| SPY h=10 | N=26, -0.126%, excess **-0.504** | N=6, -1.366%, excess **-1.744** |
| QQQ h=10 | N=26, +0.100%, excess -0.350 | N=6, -2.011%, excess **-2.461** |
| TLT h=10 | N=24, +1.109%, excess **+0.944** | N=6, +1.039%, excess +0.875 |

Late August is the seasonal spine of today's whole surface, and it is the same
macro bet the event grid produced from a different direction. The midterm
conditioner amplifies the equity leg rather than inverting it, on N=6, which is a
coin flip's worth of years and grades accordingly.

Board candidates from `seasonality` (asof 2026-08-05, **12 days stale** — read as
context, not as a live number): midterm book win 56.4% vs 64.9% all-years, and
the same de-rating for OVS, LT Trend ST OS and Indices Oversold Bounce. Zero A/B
seasonal tickets flagged. Consistent with the above, adds nothing tradeable.

Fragility dial as a cell: ma10-63d >= 78 has 31 days and only **4 declustered
episodes** across 2021, 2022 and 2026, one of which is live. Any dial-conditioned
directional idea is therefore a 3-episode sample with a mid-cluster entry, on top
of the codified negative result that the aggregate book-wide dial effect fails PIT
at t=-0.23. **Dismissed before spending a check.**

## 4. Watchlist verdicts

Full output in `01_watchlist_verdicts.py`, one block per entry with today's number.

| # | entry | verdict |
|---|---|---|
| 1 | TLT from the NFP close at the rates floor | PASS — needs a non-midterm NFP (2027-01); next NFP is midterm and 14 td out |
| 2 | LQD vs HYG at joint 52w extremes | PASS — state live (HYG -0.10%, LQD +0.75%) but still 4 episodes, three in 2018, mid-cluster |
| 3 | SVXY overnight into CPI | PASS, deferred with cause — next CPI 09-11, 18 td out; re-measure owed at the 09-10 run |
| 4 | GLD on a miner-led thrust | PASS — GDX 5d rank 40.5 (needs >= 95), GLD 46.4; the thrust is a 21d event, not the 5d shape |
| 5 | XLE on a crude 1d pop in [5,6)% | PASS — USO 1d +1.26%, +0.31 ATR; no pop. The live shape is a 5-day complex thrust, checked separately as C5 |
| 6 | TLT with the IG complex at 52w lows | **CHECK** — price rung LIVE again (TLT 0.15 / IEF 0.95 / LQD 0.75) but the last trigger day was 08-12, a **2-session gap against the >= 10 the freshness leg requires**. Feeds C1b |
| 7 | SPY on a skew spike alone | PASS — skew 5d rank 86.1 (needs 95), SPY 0.20% off its high (needs > 1%), midterm year (needs non-midterm) |
| 8 | Fade a crude thrust out of a deep base | PASS — USO 5d rank 82.9 (needs >= 90); episode count still 4 post-2020 |
| 9 | IHI medical-device thrust | PASS — 21d rank 97.6 (needs 100), and the reference-class gate (Cochran Q p 0.544) cannot move in two sessions |
| 10 | FXI break inside an intact thrust | PASS — 5d rank 11.5 fires and EEM holds, but the 21d rank is 60.7 against the >= 80 that made it more than EM beta |
| 11 | Industry breadth washout, trend broken | PASS — the selection-rule leg has never been tested, so it cannot fire on tape alone |

## 5. Scoreboard read

4 pitched, 3 graded, avg +0.361R, 3-for-3. Per axis: event_fingerprint 1 graded
(+0.837R), interaction_cell 1 (+0.146R), relative_value 1 (+0.099R). By grade:
B +0.468R on 2, C +0.146R on 1. **The graded count is a handful**, so there is no
axis to favour or penalise yet; the split is noted and not acted on. Two
consecutive stand-downs (08-13, 08-14) precede today, which is a reason to survey
harder, not a reason to lower the bar.

## 6. Candidates selected for stage C

Asset classes touched: rates, us_large, volatility, sectors, energy,
international, gold_miners, us_small. Axes: event_fingerprint, interaction_cell,
relative_value, inversion, historical_analogue, flow_mechanics. At least one
event-anchored (C1, C2, C8) and at least one price-state-anchored (C4, C5, C6,
C10) as required.

| id | candidate | axis | class | source cell |
|---|---|---|---|---|
| C1 | Long TLT into the late-August window, h ~8-10 | event_fingerprint | rates | grid row "rates" + seasonal table |
| C1b | The same trade gated on the IG 52w-low rung | interaction_cell | rates | watchlist 6, mid-episode |
| C2 | Short SPY or QQQ into late August in a midterm year | interaction_cell | us_large | grid + seasonal, N=6 midterm |
| C3 | The VIX/VIX3M ratio at its 1.2nd percentile — which way | historical_analogue | volatility | tape extreme |
| C4 | SMH at a 63d rank of 1.6 against QQQ | relative_value | sectors | tape extreme |
| C5 | The energy complex thrusting to a 52w high, XLE long | inversion | energy | tape extreme (book is systematically short these) |
| C6 | EWZ washout against EEM strength | relative_value | international | tape extreme, lowest z10 on the board |
| C7 | XRT into the retail earnings cluster | interaction_cell | sectors | earnings block + XRT 5d rank 18.7 |
| C8 | The vol complex into the VIX-expiry / opex pair | event_fingerprint | volatility | grid, opex and JH disagreeing |
| C9 | IWM at a 52w high into August opex | flow_mechanics | us_small | tape extreme + calendar |
| C10 | GDX 21d rank 100 deep inside a 22% drawdown | historical_analogue | gold_miners | tape extreme |
| C12 | Yields at a 21d rank of 75.8 while SPY sits at a 52w high | interaction_cell | rates x us_large | cross-class state |

11 candidates, 6 axes, 8 asset classes.

Registry collisions to carry into the checks: C1/C1b inherit the 08-10 TLT kill
("a control that does not control" — **tdom-matched control is mandatory**) and
the 08-12 freshness finding. C5 inherits "Energy's 5d washout into the CPI print"
and the 08-14 note that the XLE-over-USO 63d divergence is a bear-tape selector.
C10 inherits both 08-10 metals kills (miner-over-metal is beta; the silver
drawdown filter does not filter). C3/C8 inherit the 08-10 "Crushed skew at a 52w
equity high" kill and the SVXY pre/post-2018-02-28 instrument break. C9 has no
gamma data in the repo and must say so.
