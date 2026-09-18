# Surface map — 2026-09-11 (Friday, midterm year)

State: `data/pitch_state.json` / `data/pitch_tape.json`, generated 05:11:07.
Freshest bar **2026-09-10** = the prior session. Pipeline 4/4 green. One
warning: LEG stale, irrelevant to everything below.
Live-arm probe for every number quoted here: `01_live_arms.py`.

**Entry convention for this morning.** The state is observed on the
2026-09-10 close; orders place today. So a `MOC` idea enters at today's
close, which is **after** the 08:30 CPI print. lag=1 in `pitch_lab` terms.

## 0. The tape in one paragraph

Yields are the whole story. `^TNX` closed 4.944 **at a 252-day high**, +2.21%
on the session, 252-session change **+87.0 bp**, and TLT, IEF and LQD all
closed **exactly at 252-day lows** (0.000% above, all three). Against that,
the commodity complex is at the other extreme: DBC and USO both **at 252-day
highs**, USO +5.61% on the session and +24.1% over 21 days. Credit refuses to
join — HYG is only 1.09% below its own 252-day high. Equities are mid-range
and soft (SPY -1.65% / 21d, rank21 13.1; IWM the laggard at -4.42%, rank21
6.7). VIX 17.84, and the term structure is still in contango at
VIX/VIX3M **0.9042**. Fragility dial ma10(63d) **87.0**, the highest reading
any candidate this month has been scored against. P/C fear OFF at the 41st
percentile.

**Data correction, second consecutive session.** The tape reports `^VIX`
+24.58% over five sessions. On SPY's calendar it is **+17.37%**. The tape
differences `^VIX` on `^VIX`'s own calendar, which carries the 2026-09-07
Labor Day bar. The 2026-09-10 registry entry predicted this would reproduce
and it did. Every `^VIX` number below is on the equity calendar.

## 1. Calendar x asset class

Eight events in the [-5, +15] td window. Crossed with the ten classes, sixty
cells; each gets a line or is covered by a block dismissal.

| event | date | td | verdict |
|---|---|---|---|
| nfp | 2026-09-04 | -4 | past, and the post-NFP lane is registry-empty on direction, vol and duration. All ten classes DISMISSED. |
| ppi | 2026-09-10 | -1 | past. The POST-release anchor is watchlist 43, killed 2026-09-09 on a correlated-family permutation. DISMISSED, all classes. |
| **cpi** | **2026-09-11** | **0** | **LIVE, and it is the k=0 session.** Three classes CHECKED (below); seven dismissed. |
| **fomc_decision** | **2026-09-16** | **+3** | **LIVE.** Two classes CHECKED; midterm inverts the Lucca-Moench drift (registry), so the ungated long is dead by construction. |
| vix_expiry | 2026-09-16 | +3 | collides with FOMC. Watchlist 45 owns this cell and its anchor is the **settle session**, not today. PASS, re-check on 09-16. |
| **opex / quad_witching** | **2026-09-18** | **+5** | **LIVE.** us_small CHECKED. Vol classes dismissed: "September post-opex vol crush" and "September INVERTS the crush" are both registry-closed. |
| nfp | 2026-10-02 | +15 | outside every horizon this product trades. DISMISSED. |

Per-class detail on the two live events:

| class | CPI k=0 | FOMC +3 |
|---|---|---|
| us large | **CHECK (A4)** — the CPI close crossed with `^TNX` at a 252-day high has never been run. Density-short is registry-dead; density-long is sub-cost. | covered by A4's window; midterm short is the book's own T2 and the ungated long is registry-dead. |
| us small | dismissed — no CPI x IWM object exists that is not the SPY cell with more noise. | **CHECK (A3)** via the quad-witching run, which contains the decision. |
| rates | **CHECK (A1 entry-day interaction)** — A1 enters on a print session, so the print-day subset is a mandatory probe rather than a candidate of its own. | exit is +1 td, FOMC is +3. Outside. |
| credit | dismissed — the credit-specific residual has failed **six consecutive times** (registry 2026-09-09) and nothing about today is different. |
| gold/miners | dismissed — GLD pre-CPI, CPI-day gold and miners-vs-metal are all separately registry-dead. |
| other metals | dismissed — no event object; silver is examined as a price state (A9). |
| energy | dismissed — "long commodities at a 252-day high into an inflation print" and "energy equity leadership with a print in the hold" are both registry-dead, and the exact live configuration was measured at 13 episodes / -0.027%. |
| dollar/FX | dismissed — DX into CPI is -3.5 bp, and the dollar-at-a-yield-high cell died twice (2026-09-09, corrected 2026-09-10). Nothing in FX is extreme: DX 63d rank 19.8, z10 -0.07. |
| international | dismissed on the event axis; EWZ examined as a price state (A10). |
| volatility | dismissed on the CPI axis (post-CPI crush died after 2018; watchlist 2's overnight is 2023-concentrated and its anchor was last night). **CHECK (A5)** on the price-state axis instead. |

## 2. Tape extremes by class

| class | extreme | verdict |
|---|---|---|
| rates | TLT / IEF / LQD **all at 252-day lows simultaneously**; `^TNX` at a 252-day high, +87.0 bp on the year | **CHECK — A1**, and it is an armed watchlist entry, not a fresh idea |
| energy | USO and DBC at 252-day highs, USO +5.61% on the day, XOP at a high, XLE -0.58% off | **CHECK — A7** (the joint commodity-high / IG-low state). The outright energy cells are registry-dead and watchlist 4 and 19 are both PASS (section 3). |
| us large | SPY rank21 13.1, -2.58% off the high, z10 -0.64 | **CHECK — A4, A5** |
| us small | IWM rank21 6.7, rank63 13.5, z10 -1.47, weakest index | **CHECK — A3, A6** |
| credit | HYG 1.09% off a 252-day high while IG prints lows | DISMISSED — six consecutive credit-residual failures |
| gold / miners | GLD 4.68% **below** its 200d and 20.1% off its high; GDX 6.52% **above** its 200d, +30.1% / 63d | DISMISSED — "miners leading with the metal below its 200d" is registry-dead at -0.111% on 37-47 |
| other metals | SLV -5.30% on the session, 45.6% off its 52-week high but +54.7% / 252d | **CHECK — A9** |
| dollar / FX | nothing at an extreme | DISMISSED, stated above |
| international | EWZ +13.5% / 21d, rank21 90.1, z10 +1.84; FXI -3.35% / 5d | **CHECK — A10** |
| volatility | VIX +17.4% / 5d (equity calendar) with VIX/VIX3M still 0.9042; `^MOVE` at the 90.9th percentile | **CHECK — A5** |
| sectors, downside | XLV r5 **0.8**, IBB r5 **0.4** while IBB r63 is **91.3**; IHI r5 0.4; XLB r5 3.2 | **CHECK — A2**, the flush-inside-strength diagonal |
| sectors, washout | XLI r21 2.0 and z10 -2.04, with MMM -3.55, SNA -2.67, ITW -2.60, GD -2.36, PH -2.32, DOV -2.19 | **CHECK — A8**, the count object mirrored onto the downside |
| sectors, rates-driven | XLRE and IYR both r63 1.2 | DISMISSED — XLRE at a yield extreme was killed 2026-09-10 as a **double** anti-filter, 0-for-13 in exactly today's configuration (a CPI inside the hold) |
| staples subgroup | CPB -11.9%, GIS -11.4%, CAG -9.2%, KMB -8.8%, HRL -5.5% while XLP is only -2.9% | DISMISSED — "a subgroup flush inside an intact sector is the short-term reversal factor wearing a sector label" (registry 2026-09-09) |

## 3. Watchlist — a verdict on every active entry

47 active, 0 expired. Numbers from `01_live_arms.py` unless noted.

**ARMED**

- **[5] Long TLT, IG complex pinned at 52-week lows.** TLT 0.000% / IEF 0.000%
  / LQD 0.000% above their 252-day lows, all three inside the tight rung. Last
  prior trigger 2026-08-18, **gap 16 td** against a >= 10 arm. **Every leg of
  the arm clears for the first time since parking.** → candidate **A1**.

**PASS, with today's number**

- [0] TLT from the NFP close — midterm blocks it; parks to 2027-01. Today is
  not an NFP session anyway.
- [1] LQD vs HYG at joint 52-week extremes — needs >= 8 declustered episodes
  outside 2018; the count has not moved.
- [2] SVXY overnight into CPI — the anchor was **last night's** close. The
  order could not be placed this morning, and the cell is 2023-concentrated
  (1512 of 3247 bp from 12 of 100 events).
- [3] GLD on a miner-led thrust — GDX 5d rank **33.3**, arm needs >= 95.
- [4] XLE on a crude thrust in [5,6)% — the price leg **fires again** (USO 1d
  **+5.608%**), but the arm is the crude-beta residual clearing at sign p
  <= 0.10 / hit > 65% on >= 20 band episodes plus the [4,5)% bucket ceasing to
  be negative. Those are properties of history, measured 2026-09-02 and
  failing; nothing in three weeks of new data can have moved them. PASS.
- [6] SPY on a skew spike — `^SKEW` 5d rank **68.7**, arm needs >= 95. The 21d
  form is separately registry-dead (diluted tail, +0.021pp at the pitched
  threshold, and midterm steepens the block).
- [7] Crude thrust out of a deep base — USO 63d rank 71.4, arm needs <= 20.
- [8] IHI at a 21d rank of 100 — IHI r21 is **5.2**, the opposite end.
- [9] FXI inside an intact thrust — FXI 5d rank 13.9 but 21d rank 25.4, arm
  needs >= 80.
- [10] TLT November month-position — parks to a date, November.
- [11] Short SPY at a 52-week high with TLT at a low — SPY **-2.578%** off its
  high, arm needs within 0.5%. The rates leg is the only half live.
- [12] SPY on a vol pop inside calm tape — VIX 21-day **level** rank **100.0**,
  arm needs <= 25. This is the opposite tape.
- [13] Gold on an unconfirmed rate rise — needs DX 21d rank <= 15; DX 21d rank
  is 29.8 and the dollar is not washed out.
- [14] Tech vs healthcare after a rotation gap — needs SPY within 3% of its
  52-week high; SPY is 2.58% off, so that leg clears, but the one-day
  XLV-minus-XLK gap is nowhere near +3.0pp (XLV -0.55%, XLK -1.41%, i.e. XLV
  **out**performed by 0.86pp, wrong sign).
- [15] Short the dollar on an unconfirmed rate rise — same DX leg as [13].
- [16] Short TLT after a big up day from the low zone — TLT 1d **-1.16%**, arm
  needs >= +1.5%.
- [17] Short KRE vs XLF on a bank washout — arm is 5x cost ex-crisis, a
  historical property that has not changed; KRE r5 37.7 is not a washout.
- [18] Duration-neutral flattener — `^TNX` at a 252-day max with a 252-session
  change of **+87.0 bp** against the +78 bp arm, so the **label** clears. The
  re-arm of 2026-09-10 is **the dose**, and this is (a) no longer a first
  crossing, yesterday was, and (b) a clearance of **9.0 bp**, inside the
  <= 10 bp band that pays +11.3 bps on n=7 at 1.32x cost. PASS on its own
  terms, one day after it fired and failed.
- [19] Narrow energy thrust count — **and this entry exposes a convention
  defect worth filing.** Under `pitch_lab.zscore` the count of the 11-name
  complex at z10 >= 2.0 is **1** (USO 2.00 alone). Under the tape's
  `build_pitch_state._metrics_for` convention it is **2** (USO 3.23, VLO 2.21),
  which is inside the [2,3] arm. CLAUDE.md already records that these two
  functions compute different objects. The entry's parked numbers were
  produced in a check script, i.e. under `pitch_lab`, so `pitch_lab` is the
  binding convention and the count is 1. **PASS**, and the ambiguity is filed.
  Three standing debts (own pre-registration, reference class on the narrow
  form, forward re-derivation) make it expensive even when it does arm, and
  energy-with-a-print is registry-dead.
- [20] Survivorship-free new-high breadth — needs two numbers to move; neither
  has.
- [21] Sector washout into a 52-week high — the washouts are live (XLV r5 0.8,
  IBB r5 0.4) but **none is within 5% of its 52-week high**: XLV -5.70%, IBB
  -6.60%, XBI -7.51%, IHI -20.79%. The price gate fails. PASS — and the
  neighbouring diagonal it does not cover becomes **A2**.
- [22] Utilities washout with the long end hit alongside — XLU r21 **26.2**,
  arm needs <= 5. The rates half is live, the utilities half is not.
- [23] Bare dollar washout — parks to a non-midterm year.
- [24] HYG at a fresh high while the index has not — HYG is **1.09%** below its
  high, so there is no fresh high, and SPY's depth is 2.58% against the
  entry's own >= 2.0% requirement. One leg of two.
- [25] SMH at a 63-day rank floor inside a top-decile year — SMH r63 **4.8**,
  so the floor leg is live, but the entry's second number is the one the
  original pitch got backwards and it has not moved. Related: the pooled form
  of this was killed outright 2026-09-10 at a fixed-effect **-0.381% / t
  -4.57**. PASS.
- [26] IG at lows while HYG prints a high — IEF and LQD clear at 0.000%; HYG
  is **-1.088%** against a >= -0.25% arm. One leg short, and the entry's real
  problem is an episode count of one.
- [27] IEF out of Jackson Hole — parks to 2027, non-midterm.
- [28] Pooled laggard still falling — **zero live instances** across the
  29-ETF pool. And the cell was killed outright on 2026-09-10.
- [29] Short silver after a complex break — the complex **did** break together
  (SLV -5.30%, GDX -3.46%, GLD -1.73%). The arm is a **lag profile**, an
  unmet mechanism test, not a level, and the short direction is the one the
  2026-09-10 registry says is wrong-signed on the neighbouring cell. PASS as
  written; the reversal side becomes **A9**.
- [30] TLT with bond vol MID-RANGE — `^MOVE` trailing-252 level percentile
  **90.9**, arm band is [40,50). Decisively outside, and in the direction that
  says bond vol is bid rather than compressed.
- [31] IWM month-end overnight in December — parks to December.
- [32] XLE at a fresh high on a down-SPY session at h=21 — XLE is 0.58% off
  its high, so there is no fresh high today.
- [33] SVXY into a print out of a compressed 21-day VIX range — the range is
  no longer compressed: VIX 21-day **level** rank is 100.0 and the index is
  +17.4% over five sessions. Killed outright yesterday on the SPY-residual
  rule regardless.
- [34] Pooled sector triple rank floor — the arm is "a reason to exist beside
  the book", which no number can supply.
- [35] SPY into a print out of a dead VIX range — **blocked on the dial**: the
  arm needs a moderate fragility reading and ma10(63d) is **87.0**, the
  highest this quarter. The range is also no longer dead.
- [36] Risk premium across an extended closure — no closure inside any horizon.
- [37] Post-NFP duration on a moderate prior miss — not an NFP session.
- [38] SVXY at the first close after a closure — no closure.
- [39] SPY vs IWM on the dial in [56,70] — dial is **87.0**, outside the band.
  PASS. The rates-conditioned version of the same cross-section, which this
  entry does not cover, becomes **A6**.
- [40] HYG at the first close back from a closure — no closure.
- [41] Short IEF with commodities at a 252-day high and a print inside the
  hold — DBC **is** at a 252-day high, and that is the only leg that clears.
  The arm is the true print anchor outranking its own k=-5..+5 placebo ladder,
  a historical property that failed at rank 5 of 11 and cannot have moved. The
  print is also on the **entry** session rather than inside the hold. PASS.
  The joint commodity-high / IG-low state the entry gestures at, which it does
  not test, becomes **A7**.
- [42] SPY across a September PPI-then-CPI pair entered two sessions before
  the first print — that entry was 2026-09-08. The window is already running
  and the anchor is past.
- [43] TLT from the PPI release close — the release close was **yesterday**.
  Anchor past.
- [44] SPY with HYG at a 252-day high on a `^TNX` 252-day-high session — the
  `^TNX` leg clears exactly; HYG at **-1.088%** fails a >= -0.5% arm.
- [45] SVXY on a VIX-expiry / FOMC-decision collision — **the collision is
  real and it is 2026-09-16**, three sessions out. The anchor is the settle
  session itself, so nothing can be entered today. PASS, re-check 09-16.
- [46] September natural gas on the front contract — the arm demands a
  non-roll expression and a mechanism surviving the month ladder; September
  ranks 5 of 12 and neither has been supplied. UNG is registry-banned as a
  vehicle at -35.57 bp per session in September.

Nothing to prune: no entry is listed under `expired`, and the one entry that
armed (5) is being pitched rather than retired.

## 4. Seasonality and cycle

September, trading day 11 of the month, **midterm year** (2026 % 4 == 2),
Friday. The seasonal board carries no A/B setups and its content is regime
context: midterm book win rate 56.4% against 64.9% all-years. Midterm is used
below as a **conditioner on every candidate**, never as an idea of its own —
the standalone midterm-August cell died on 2026-08-07 at N=6.

## 5. Scoreboard read

Five graded ideas lifetime: avg **+0.174R**, hit 80%, B-grades +0.448R on
three and C-grades -0.237R on two. By axis: event_fingerprint 2/2 at +0.622R,
inversion 1/1 at -0.62R, interaction_cell and relative_value one each. **That
is a handful, not a signal**, so no axis is up- or down-weighted today. The
number that does matter for calibration is the run history: **21 stand-downs
against 5 shipped ideas, and the last thirteen consecutive sessions have all
stood down.** The doctrine that small N is not a kill and that false
positives are McKinley's to filter is being applied against that record today.

## 6. Candidates selected

Twelve, over nine asset classes and **all seven** novelty axes, with four
event-anchored and eight price-state-anchored. A11 and A12 were added to the
sweep specifically because `historical_analogue` and `flow_mechanics` were the
two axes the first ten never touched, and the axis table is a search-mode
inventory rather than a menu.

| id | candidate | class | axis |
|---|---|---|---|
| A1 | Long TLT h=1 MOC with TLT, IEF and LQD all at 252-day lows on a fresh trigger (armed watchlist 5) | rates | interaction_cell |
| A2 | Long the five-day flush inside a top-decile 63-day trend, pooled over industry ETFs, live on IBB | sectors / healthcare | interaction_cell |
| A3 | Short IWM into September quad witching in a midterm year | us small | event_fingerprint |
| A4 | SPY forward from a CPI print session with `^TNX` at a 252-day high | us large x rates | interaction_cell |
| A5 | Long SPY on a five-session VIX thrust that leaves VIX/VIX3M in contango | volatility x us large | interaction_cell |
| A6 | Short IWM against long SPY with `^TNX` at a 252-day high | us small x rates | relative_value |
| A7 | The joint inflation repricing: DBC at a 252-day high on a session the IG complex prints 252-day lows | energy x rates | interaction_cell |
| A8 | The industrial-complex washout count, the energy thrust-count object mirrored onto the downside | sectors | inversion |
| A9 | SLV reversal after a >= 4% single-session break inside a deep drawdown but a strong year | other metals | inversion |
| A10 | EWZ's residual against EEM on a 21-day rank thrust | international | relative_value |
| A11 | Nearest-neighbour tapes to the 2026-09-10 state on a pre-specified five-feature vector, forward on SPY / TLT / GLD / DBC | cross-asset | historical_analogue |
| A12 | The pre-collision window when an FOMC decision and a `^VIX` expiry land on the SAME date, on the index rather than on SVXY | us large x volatility | flow_mechanics |

**The instrument_translation axis** is carried inside A1 (whether the IG-lows
cell belongs in TLT or in a shorter-duration vehicle is an entry-form question
handed to that checker) and inside A9 (whether the silver object belongs in
SLV or in GLD, which the reference-class pass answers). Neither earned a
standalone slot because every translation on today's tape has a registry
blocker already attached to the vehicle: UNG is banned outright at -35.57 bp
per September session, UUP died as an inception artifact on 2026-09-10, and
SVXY has failed the mandatory SPY-residual charge twice in three sessions.
