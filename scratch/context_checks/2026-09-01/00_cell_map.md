# Cell map — run 2026-09-01 (Tue), asof session 2026-09-01, next session 2026-09-02 (Wed)

Prices FRESH through 2026-09-01 (core SPY/^GSPC/QQQ/TLT all print). Both lanes live.
Sweep: 1217 cells scanned, 114 fired (72 event / 42 price), 19 cleared BH at crit p 0.0196.
Cycle: midterm year. Next session is td 2 of September (21 sessions in the month).
Stale tape: LBS=F, ^AXJO, ^HSI, ^KS11, ^N225, ^SKEW carry no 2026-09-01 bar.
Cap: P5:rank5_extreme dropped ^TNX / TLT / IEF (per-trigger cap 8). Those three are the
rates complex and they are the heart of tonight's tape, so they are recomputed by hand in
drill 02 rather than treated as absent.

## The session, in one paragraph

The whole US fixed-income complex printed 52-week lows on the same session. IEF closed AT
its trailing-252 minimum (dist_52w_low 0.00%), LQD AT its minimum (0.00%), TLT 0.64% above
its own, HYG 2.92% above. On the other side ^TNX and ^FVX both closed AT 52-week yield
highs, with 5d return ranks of 96.8 and 98.0 and ^IRX at 97.6. SPY fell 0.69% and sits
2.07% under its 52-week high; ^VIX rose 9.52% to 16.34. Payrolls land Friday 2026-09-04,
3 sessions out. Last night's brief closed the ^TNX 52-week-high item with "duration without
credit." Tonight credit joined, which is the novelty.

## Event lane

| trigger | verdict |
|---|---|
| `E:nfp` (payrolls 2026-09-04, k=3) | **DRILL**. Top-tier event, the lead. SPY n=316 +0.175% at t 2.87, 188-127, sign p 0.0004, BH pass, era-stable — but this is a PRE-SPECIFIED famous cell (pre-payrolls drift) and it owes the sweep no correction. h1 here is Wednesday, 2 td before the print, not the print itself. Drill 04 crosses it with the state that is actually live: yields at a 52-week high going in. Drill 06 does the same for IWM (n=312, 177-135, p 0.0101, BH pass) against its own 5d rank of 6.7. |
| `E:nfp` ^VIX arm | **SKIP(subsumed)**. n=316, -0.826% at t -2.43, BH pass, 137-178. Real, but it is the mechanical mirror of the SPY arm and spending a nugget on both says one fact twice. Folded into drill 04 as a supporting line if it survives the midterm/rates conditioning. |
| `E:nfp` commodity + FX arms (GC, SI, HG, CL, NG, DX-Y, EURUSD, JPY) | **SKIP(no edge)**. Every one is inside +/-0.13% at abs(t) < 1.6 with sign p > 0.12, and HG/CL/EURUSD/TLT/IEF are era-unstable on top of it. Nothing to condition. |
| `E:nfp` EEM arm | **SKIP(repeat)**. n=278, +0.205% at t 2.04, but EEM was last night's headline subject on the turn-of-month cell. A second EEM lead 24 hours later is the countdown failure mode wearing a different trigger id. |
| `E:turn_of_month` (next session is td 2) | **SKIP(repeat)**. This trigger group WAS last night's headline (EEM vs SPY, 329-229 across 558 anchors) and the absolute SPY/QQQ/IWM/EEM arms were the material it was built from. The numbers have not moved. Genuinely strong (SPY 375-264, EEM t 4.01) and genuinely already told. |
| `E:weekday_month` (Wednesdays in September) | **SKIP(near-duplicate)**. ^VIX n=109, -1.162% at t -2.06, 41-68, BH pass, is the best thing in the group. Last night published "^VIX — Tuesdays in September" from the identical trigger with the identical construction. Publishing the Wednesday version tonight is the same cell with the weekday rotated, which is exactly what the novelty rule exists to stop. CL=F (+0.566%, t 2.44, 62-45) fails BH and is a lone commodity arm with sign p 0.09. |
| `E:seasonal_doy` (Sep 02 +/-2, one pick per year) | **DRILL(folded)**. Nothing here stands alone: SPY 26 anchors at -0.061%, 14-12, sign p 0.42, and the midterm arms are n=6. The one arm with a record is TLT h5 in midterm years, 5 of 5 down, mean -1.93%, sign p 0.031 — an anecdote by construction, last published 2026-08-17 (11 td, not blocked). It is not a headline; drill 03 checks whether it survives beside the down-streak cell or is 5 coincidences. |
| `E:holiday_pre` / `E:holiday_post` | **N/A**. The engine fires these only when the NEXT session is the holiday-adjacent one. Labor Day is 2026-09-07 and the pre-holiday session is Friday 2026-09-04, so this trigger belongs to Thursday night's brief, not tonight's. Noted so its absence is not read as a miss. |
| `E:fomc_*`, `E:cpi`, `E:ppi`, `E:opex`, `E:quad_witching`, `E:vix_expiry`, `E:jackson_hole`, `E:election`, `E:month_end` | **N/A**. Outside their anchor windows (PPI 6 td, CPI 7 td, FOMC + VIX expiry 10 td, quad witching 12 td). They belong in the Calendar block only. Countdown re-tellings are banned. |

## Price lane

| trigger | verdict |
|---|---|
| `P2:new_52w_low` / `P2b:new_52w_low_90` — LQD | **DRILL**. LQD's first 52-week low in 90+ days. The engine's own-forward cell is n=8/n=10 and dead on its own (-1.19% h1, 3-5). The content is not LQD's own path, it is the JOINT state: investment-grade credit and 7-10y Treasuries at 52-week lows on the same session while the S&P is 2% off its high. Drill 02 builds that cell from scratch. |
| `P2` / `P2b` — IEF | **DRILL(same cell)**. n=12/n=8, +0.09% h1, 6-2. Same treatment, same drill. Two arms of one state. |
| `P7b:down_streak` — TLT | **DRILL**. The strongest single cell on the board: n=93, +0.303% at t 3.41, 62-31, sign p 0.0009, era-stable, BH pass, and h5 +0.496%. Drill 03 asks the two questions the base cell cannot: does it hold when the streak ends at a 52-week low rather than mid-range, and do the top two episodes carry the mean. |
| `P7b:down_streak` — IEF, LQD | **DRILL(folded)**. IEF n=100 +0.077% t 1.90 (fails BH), LQD n=91 +0.021% t 0.29 and era-UNSTABLE. Weak on their own; they matter only as evidence the TLT cell is a duration effect and not a TLT artifact. Into drill 03. |
| `P7b:down_streak` — ^MXX, ^NYA, EFA | **SKIP(off-subject)**. ^MXX is the best-tagged cell in the group (n=172, +0.346%, t 2.59, BH pass) and ^NYA passes BH too. But a Mexican-index mean-reversion stat on the night the US bond market printed 52-week lows across four instruments is a strong number about nothing Scott needs. EFA fails outright (t 0.40, 65-60). |
| `P4:z10_extreme` — ZC=F | **DEAD(repeat_blocked)**. `novelty.flags` blocks it: published 2026-08-27, 3 td ago, number unmoved. |
| `P4:z10_extreme` — ZW=F, ZS=F, EWZ, CT=F, ^BVSP | **SKIP(roll artifact suspected)**. ZW BH-passes at 74-112 but every grain arm here is era-UNSTABLE, and last night's brief established that the 2026-08-31 grain and softs bars were 94-98% overnight gap, i.e. continuous-contract rolls. Drill 01 verifies tonight's bars before anything is written; if they are rolls again the whole grain complex is an artifact, not a tape. |
| `P5:rank5_extreme` (top) — ^IRX, ZW, ^FVX, ZS, ZC | **SKIP(pooled/artifact)**. ^IRX n=359 +4.649% at t 1.77 with sign p 0.9925 is the pooled-tail pathology in plain sight: 157-153 with a mean driven by 2008-2009 rate collapse magnitudes. The grain arms go with drill 01. The rates content is taken up properly in drill 02/04 as a level story, not a rank one. |
| `P5:rank5_extreme` (bottom) — KC=F, HYG, LQD | **SKIP**. KC=F is drill 01's problem (-10.34% today after -9.21% yesterday, roll suspected, and the roll story already ran two nights). HYG n=239 t -0.35 and LQD n=293 t -0.12 are both flat and era-unstable; LQD's real content is the 52-week low, not the 5d rank. |
| `P5b:rank21_extreme` — BTC-USD | **SKIP(off-subject)**. n=301, +0.730% at t 2.84, BH pass, era-stable, tagged solid. A real cell. But BTC fell 1.79% today, the state is 21d-rank momentum rather than anything tonight printed, and it competes for space against a four-instrument 52-week-low event in the asset class Scott's screen is actually red in. Parked, not rejected. |
| `P5b:rank21_extreme` — ETH, ZC, ZW, ZS, SB | **SKIP(no edge / artifact)**. ETH t 1.32 era-unstable; the grains and sugar are drill 01's. |
| `P6:two_atr_day` (down) — KC=F, HYG, LE=F, LQD | **SKIP(no edge)**. Best arm is KC=F at t 0.94, 27-25. HYG 36-32, LQD 22-22 era-unstable. The 2-ATR days themselves are informative context for the bond nugget and are used there as description, not as a cell. |
| `P6:two_atr_day` (up) — ZS=F, ZC=F | **SKIP**. Drill 01, and both are flat anyway. |
| `P7:up_streak` — ^FVX, ^TNX, ^BVSP, JPY=X | **SKIP(no edge)**. The yield arms are the interesting subject and the numbers refuse: ^TNX n=125 at -0.143%, 53-71, sign p 0.076, ^FVX 74-79 at t -0.53. Worth stating in the map that the "yields have run too far" reflex has no support in this sweep. |
| `P9b:stocks_bonds_down` — SPY, TLT | **SKIP(no edge)**. SPY and TLT both down 50bp+ is exactly today, and the cell is empty: n=270, SPY +0.126% at t 0.97, era-UNSTABLE, 146-121; TLT +0.017% at t 0.29. The honest verdict on the day's most quotable framing is that it predicts nothing, and drill 02 tests whether the sharper joint condition does better. |
| `P1/P1b` (52w highs), `P3*` (reversal after an extreme), `P8` (200d cross), `P9/P9c-f` (dollar-gold, VIX-up-on-up-day, curve), `P10*` (VIX term structure, VIX +10%), `P11*` (breadth), `P12*` (release vs consensus) | **N/A — did not fire**. Two near-misses worth recording: ^VIX rose 9.52% against P10c's 10% threshold, and there were no US prints today so the P12 family had nothing to condition on. The VIX near-miss is taken up in drill 05 as its own hand-built cell, since a 9.5% vol jump on a 0.69% equity decline is a disproportion the trigger inventory has no line for. |

## Drills queued

- `01_roll_gap_check.py` — are tonight's KC/ZC/ZS/ZW bars gap or trade? Gates every commodity verdict above.
- `02_bonds_credit_52w_low.py` — IEF and LQD at trailing-252 lows on one session with SPY within 3% of its high. Rarity, forward SPY/LQD/TLT, era split, control.
- `03_tlt_down_streak.py` — TLT 5+ down closes: current streak length, the near-52w-low arm, concentration, era split, IEF/LQD companions, and the seasonal_doy midterm h5 arm beside it.
- `04_nfp_rates_cross.py` — pre-payrolls drift crossed with ^TNX at a 52-week high going in; plus midterm and September arms; plus the ^VIX mirror.
- `05_vix_jump_small_decline.py` — VIX +8% or more on a session SPY fell less than 1%. Forward SPY and VIX, era split.
- `06_iwm_into_payrolls.py` — the IWM payrolls arm conditioned on IWM's own 5d rank being in the bottom decile.

Tag-hint notes: no cell is upgraded past its hint tonight. `E:nfp|SPY|k3` arrives hinted
`solid` and is PRE-SPECIFIED, so its BH pass is not load-bearing. Everything reached by
drill 02, 03's conditioning, 05 and 06 is post-selection conditioning on my part and is
capped at `suggestive` regardless of what the t comes back as.

## Drill results, written after the scripts ran

- **01 (roll gaps): CONFIRMS the SKIP verdicts.** Volume is the tell. ZS=F traded 7,638
  lots on 2026-08-31 against a 20-session median of 8,488, then 175,695 tonight; ZW=F
  10,298 then 95,862; KC=F 16 then 22,286. The old contract went quiet and the new one
  took over, so tonight is the far side of the roll seam: KC=F -10.34% is 86% overnight
  gap, ZC=F +5.92% is 73%, ZW=F +3.24% is 74%, LE=F -3.78% is 99%. CT=F printed a zero
  open. The three grains' "52-week closing highs" are seam artifacts and must not be
  written as a market fact. Third session of the same story, published twice already,
  so it is a non-publication, not a nugget.
- **02 + 02b: PUBLISH, as a census.** IEF and LQD have closed at a 252d low together on
  30 sessions since 2002-07-30. SPY's median distance from its own 52-week high on those
  sessions is -7.63%, mean -11.06%, and 21 of the 30 are 2022. Tightening to SPY within
  3% leaves 3 sessions and 2 episodes: 2006-04-13 and tonight. The forward cell is
  therefore n=1 and is NOT the claim. The bond-low-only control (9 episodes) has SPY at
  -0.480% h1, 3-5, which is the company the state normally keeps.
- **03 + 03b: PUBLISH, base cell plus its exception.** TLT's 5+ down streak reproduces
  (n=93, +0.303% h1, 62-31, sign p 0.00086, t 3.41; era +0.344/+0.252 both positive; top
  two episodes only 20% of total). Splitting on the live condition: 87 not-near-low
  sessions go +0.317% h1 at t 3.85 and +0.647% h5 at t 3.83, while the 7 near-a-252d-low
  sessions go 1-5 at h5 and 0-6 at h21, mean -4.03%, sign p 0.0156. Those 7 are three
  episodes and all of them sit in 2021-02, 2022-04 and 2022-10, which is one rate-shock
  regime and has to be said. Companion arm: early-September TLT in midterm years is 0-5
  at h5 (2006, 2010, 2014, 2018, 2022), mean -1.42%, sign p 0.0312. Current streak is
  exactly 5 sessions, and IEF and LQD are both on 5 as well.
- **04: PUBLISH, and it is the lead.** First pass was WRONG and is recorded in the
  script header: `anchor_positions` returns kept EVENT dates, not anchor dates, so
  forward returns were measured from the print itself and the cell read -0.093%. Fixed,
  it reproduces the engine (n=318 vs 316, +0.176% vs +0.175%, 190-128 vs 188-127, t 2.90
  vs 2.87). The new content is the decomposition: the T-2 leg is 190-128 at t 2.90 and
  sign p 0.0003, the T-1 leg is 175-144 at -0.029%, and the print session itself is
  179-140 at +0.054%. Tomorrow is the T-2 leg. Era-stable (+0.180 pre / +0.165 post),
  and the top two episodes SUBTRACT from the total, so there is no concentration to
  disclose. Midterm arm 49-32. The ^TNX-near-a-high arm is n=7, 6-1, same sign, quoted
  as an anecdote only. ^VIX mirror 138-180.
- **05: PUBLISH as a null.** VIX +8% or more with SPY down under 1% is 210 sessions and
  174 episodes. SPY h5 +0.287% at 110-63 looks like something until the right control
  runs: the 1,922 SPY-down-under-1% sessions WITHOUT a vol jump give +0.232% at
  1133-787, so the jump is worth 0.055pp. h10 +0.554% vs +0.360%. Local +/-126td control
  +0.219%. Era h5 +0.378 pre / +0.157 post. The publishable fact is that the vol pop
  carries no information the small down day did not already carry.
- **06: PUBLISH, paired with 04.** IWM's own 5d rank was 6.7 tonight. On the 28
  pre-payrolls anchors with IWM in its bottom rank decile: h1 +0.500%, 20-8, sign p
  0.0178, era-stable (+0.478 pre / +0.548 post) but 40% of the total in the top two
  episodes. It then unwinds: h3 -1.386% at 10-18, and IWM minus SPY is -0.819% at 7-21
  by h5, sign p 0.0063. Capped at `suggestive`: the decile was chosen after seeing
  tonight's rank.

## Final slate (6 nuggets, 2 lanes)

Tomorrow's tape: 04 SPY (solid), 06 IWM (suggestive).
Today in context: 02 bonds+credit census (suggestive, headline), 03 TLT (solid, exception
arm labelled), 05 VIX null (suggestive), 03b September-midterm TLT (anecdote, folded into
the TLT item rather than given a slot).
Anecdote budget: no nugget is tagged `anecdote`; the two anecdote-grade arms (the 2006
analogue, the 3-episode near-low exception) live inside `suggestive` and `solid` items
with their N and their concentration stated in the same sentence.
