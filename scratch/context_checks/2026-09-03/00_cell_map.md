# Cell map — run 2026-09-03 (Thu), asof session 2026-09-03, previewing 2026-09-04 (Fri)

Sweep: 1217 cells scanned, 105 fired (72 event / 33 price), BH crit p 0.0189, 18 pass.
Prices FRESH (core bar 2026-09-03). Both lanes live. Cycle year: midterm.

Tomorrow is a double: **September payrolls AND the session before Labor Day**.
That intersection is the obvious place to spend the evening, because each
trigger on its own is a cell Scott can already recite.

Stale tape: LBS=F, ^AXJO, ^HSI, ^KS11, ^N225, ^SKEW (no 2026-09-03 bar).
Foreign cash indices are event-lane-excluded by construction.
`^GSPTSE` missing from master_prices.

Today's tape in one line: risk-on across almost everything. SPY +1.05, QQQ +1.19,
^DJI +1.18, EFA +1.07, EWJ +1.94, VIX -5.79 to 14.32, the whole curve down
(^TNX -0.71 to 4.762, ^FVX -0.94, ^IRX -0.85) off a 52w yield high, gold +3.53,
silver +4.42, palladium +6.60, BTC +5.38, dollar -0.57 and the yen up 2.75 vs USD.

---

## Event lane

| trigger | subject(s) | verdict |
|---|---|---|
| `E:nfp` | ^VIX (n318, -1.14%, 106-209 down, t -2.61, BH pass) | **DRILL** — the only `solid` hint in the group, but "VIX falls into payrolls" is close to "VIX falls on any scheduled-event resolution". Needs a control against other top-tier prints before it earns a line. |
| `E:nfp` | ^GSPC / SPY (n318, +0.04%/+0.05%, 179-139 up, sign p 0.014, BH pass) | **DRILL** — the mean is a rounding error and the t is 0.62. The record is the claim, not the mean, and the record is what wants conditioning (September, midterm, pre-holiday). |
| `E:nfp` | EEM (n278, +0.19%, 56.8% hit, t 2.04, BH pass) | SKIP(subject relevance) — real, but EM into US payrolls is a second-order subject on a night where the primary cross is sitting right there. Parked. |
| `E:nfp` | ^TNX (n318, +0.33%, t 1.82, sign p 0.27) | **DRILL** — yields drift UP into/through payrolls on average, and the curve just came off a 52-week high with TLT 1.3% off its 52-week low. Condition on the bond state. |
| `E:nfp` | TLT, IEF, CL=F, NG=F, HG=F, SI=F, GC=F, QQQ, HYG, DX-Y.NYB, IWM, JPY=X, EURUSD=X | SKIP(no edge) — all \|t\| < 1.4 with sign p > 0.15 or era-unstable. Nothing here beats an unconditional day. |
| `E:holiday_pre` | CL=F (n241, +0.58%, 63.5% hit, t 3.48, BH pass, era-stable) | **DRILL** — strongest single cell in the whole sweep and it is pre-specified-adjacent (pre-holiday drift is a famous hypothesis, so BH is not owed). Wants a control and a Labor-Day-specific split. |
| `E:holiday_pre` | ^VIX (n246, -1.05%, 95-148 down, t -2.96, BH pass) | SKIP(duplicate mechanism) — same "vol bleeds into a known date" story as the NFP ^VIX cell. One vol nugget maximum, and it collides. |
| `E:holiday_pre` | EEM (n216, +0.28%, 63.0%, t 3.34, BH), HYG (n179, +0.13%, 66.5%, t 2.53, BH) | SKIP(subject relevance) — genuinely strong, but both are the same pre-holiday risk-on drift as the index cell, on subjects Scott will read as filler. |
| `E:holiday_pre` | ^GSPC / SPY (n246, +0.11%/+0.10%, 60.6%/58.9%, t 1.77/1.71, BH pass) | **DRILL** — this is the base rate the NFP∩pre-holiday cross has to beat. Compute both together or the cross means nothing. |
| `E:holiday_pre` | IWM (n242, +0.14%, 58.3%, sign p 0.006), GC=F (t 2.37) | SKIP(covered) — same drift, subsumed by the index drill. |
| `E:holiday_pre` | ^TNX, SI=F, HG=F, TLT, JPY=X, NG=F, IEF, QQQ, DX-Y.NYB, EURUSD=X | SKIP(no edge or era-unstable). |
| `E:weekday_month` | Fridays in September, all 18 subjects | SKIP(dominated) — best cell is CL=F -0.36% t -1.84 and SPY is 55-56 at t -1.24. A bare weekday×month cell has nothing to say on a night that carries two real events, and it partly double-counts the pre-holiday and NFP anchors. |
| `E:seasonal_doy` | TLT midterm (n5, -1.23%, **0-5 down**, sign p 0.031; h5 also 0-5, -1.72%) and ^TNX midterm (n6, +1.89%, 5-1 up; h5 **6-0 up**, +3.58%) | **DRILL** — N is anecdote-tier and the two are the same fact seen twice, but the record is perfect in both directions and it lands the night before payrolls with the long end already at 52-week lows. Verify it is not one overlapping episode, then publish honestly or kill it. |
| `E:seasonal_doy` | SPY/QQQ/IWM/^GSPC (n26 all-years, ~12-14, sign p 0.42) | SKIP(no edge) — coin flip. Midterm cells are 3-3 on N=6. |
| `E:seasonal_doy` | GC=F, SI=F, HG=F, NG=F, CL=F, EEM, HYG, FX | SKIP(no edge) — nothing clears sign p 0.10 on the all-years cell. |
| calendar: `ppi` 2026-09-10 (4 td), `cpi` 2026-09-11 (5 td) | | SKIP(too far, countdown ban) — not next-session, and a countdown re-telling is banned outright. |
| calendar: `fomc_decision` + `vix_expiry` 2026-09-16 (8 td) | | SKIP(outside window) — belongs in the Calendar block only. |
| calendar: `opex` + `quad_witching` 2026-09-18 (10 td) | | SKIP(outside window) — Calendar block only. |
| calendar: `jackson_hole` 2026-08-28 (-4 td) | | SKIP(past). |

## Price lane

| trigger | subject(s) | verdict |
|---|---|---|
| `P6:two_atr_day` down | JPY=X (n42, +0.17%, 29-13 up, sign p 0.0098, BH pass), EURJPY=X (n34, +0.48%, 24-10 up, t 2.99, BH pass), CHFJPY (n21), NZDJPY (n30) | **DRILL** — four correlated subjects firing on the same yen surge is one event, not four. Worth one nugget IF it survives declustering and an era split; USDJPY -2.75% is a big session and it happened the night before payrolls. |
| `P6:two_atr_day` down | ^MOVE (n299, -0.38%, 116-177 down, sign p 0.0009, BH pass) | **DRILL** — bond vol collapsed 6.3% on the eve of payrolls, which is the counterintuitive part. Check it is not just mean reversion mechanically baked into a vol index. |
| `P6:two_atr_day` down | KC=F (n54, coffee -10.2%) | SKIP(out of scope) — softs are not a macro subject for this reader. |
| `P6:two_atr_day` up | PA=F (n239, t 0.69), PL=F (n206, t 0.02) | SKIP(no edge) — both coin flips despite the +6.6% / +4.0% sessions. |
| `P5b:rank21_extreme` top | BTC-USD (n303, +0.74%, 56.8%, t 2.91, era-stable, BH pass) | **DRILL** — the one `solid` price cell in the sweep, +5.4% today with the 21d return at the 100th percentile of its year. Momentum continuation in crypto is plausible but heavily concentrated by construction; check `cluster_note` before it goes anywhere near the brief. |
| `P5b:rank21_extreme` top | ETH-USD (n189, t 1.39), ZC=F, ZW=F, ZS=F, SB=F | SKIP(no edge / out of scope) — grains and softs are not macro subjects; ETH is the same crypto fact at half the significance. |
| `P5:rank5_extreme` bottom | EURJPY (n286, 60.1% hit, sign p 0.0004, BH pass), CHFJPY, AUDJPY, NZDJPY, KC=F | SKIP(duplicate) — the yen-cross story again on a different clock. Folded into the yen drill, not published twice. |
| `P5:rank5_extreme` top | ^IRX (n361, +4.62% mean on a 3.7% yield level, 43.5% hit, sign p 0.994) | SKIP(degenerate) — a mean that huge with a sub-50% hit rate on a rate LEVEL series is an arithmetic artifact of low absolute yields in the sample, not a claim. |
| `P5:rank5_extreme` top | AUDNZD (t -1.09), EWZ (era-unstable) | SKIP(no edge). |
| `P4:z10_extreme` up | ZC=F, EWZ, ZS=F, ^BVSP | SKIP(no edge + out of scope) — all \|t\| < 1.4, and grains/Brazil are not macro subjects here. |
| `P3:drop50_after_high` | ^TNX (n70, +0.06%, 32-37, t 0.25) | SKIP(no edge on its own) — but the STATE it describes (10y yield made a 52-week high on 09-02 and fell 71bp relative today) is real and feeds the `E:nfp`×bond-state drill. Not a standalone nugget. |
| `P3:drop50_after_high` / `P3b` | SB=F | SKIP(out of scope). |
| `P2` / `P2b:new_52w_low` | EURAUD=X (n24 / n14) | SKIP(no edge) — 15-9 at t 1.64, and a EUR/AUD cross is not a subject for this reader. |
| `P7:up_streak` | AUDNZD=X (n193, t -0.14) | DEAD — exactly 50-50. |
| `P8:sma200_cross` down | NZDJPY=X (n22, 11-11, t 0.01) | DEAD — coin flip, and it is the yen event a third time. |
| `P9:stocks_bonds_up` and the rest of P9 | — | did NOT fire. SPY +1.05% but TLT only +0.15%, under the 50bp co-move floor. Worth stating in the brief as the shape of the day: stocks and gold ran, the long bond did not follow. |
| `P9c:dollar_gold_up` | — | did NOT fire, correctly: gold +3.53% with DXY -0.57%. |
| `P9d:vix_up_on_spx_up`, `P10c:vix_spike` | — | did NOT fire (VIX -5.79%). |
| `P9e/f:curve_steepen/flatten` | — | did NOT fire; the 10y-5y step was inside 2sd. |
| `P10/P10b:vix_inversion` | — | did NOT fire; ^VIX 14.32 vs ^VIX3M 17.42, no crossing. |
| `P11/P11b:breadth` | — | did NOT fire; no 80%/20% crossing. |
| `P12:<event>_<label>` | — | did NOT fire; zero US prints today. |
| `P1/P1b:new_52w_high`, `P3c:pop50_after_low`, `P7b:down_streak` | — | did NOT fire anywhere in the universe. |

## Cap check

`sweep.dropped_by_cap` truncated `P5:rank5_extreme` (^BVSP) and `P6:two_atr_day`
(CADJPY, AUDJPY, USDCNY). The three dropped yen crosses are the same event as
the four that were kept and are covered by the yen drill; ^BVSP and USDCNY are
out of scope. Nothing lost.

## Hints I am not inheriting

- `tag_hint` **downgrades taken**: `E:nfp|^VIX` and `E:holiday_pre|^VIX`/`CL=F`/
  `EEM`/`HYG` all arrive `solid`; none of them will be published at `solid`
  unless the drill reproduces the number on a declustered, era-split basis.
- `bh_pass` **not owed** by: `E:holiday_pre` (pre-holiday drift is a famous
  pre-specified hypothesis), `E:nfp` (top-tier scheduled print, pre-specified).
  BH **is** owed by every price-lane cell, all of which came out of the search:
  the yen `P6` cells, `P5b|BTC-USD` and `P6|^MOVE` each pass at crit p 0.0189
  and their tags still cap at what the drill supports.

## Drill list

1. `01_nfp_preholiday.py` — payrolls that are ALSO the session before a market holiday, vs each parent cell.
2. `02_nfp_bond_state.py` — payrolls with the long end already at/near a 52-week low.
3. `03_yen_spike.py` — 2 ATR USDJPY down day, declustered, era-split, and crossed with "next session is a top-tier print".
4. `04_sep_labor_day.py` — the pre-Labor-Day session specifically, and September payrolls / midterm September payrolls.
5. `05_move_collapse.py` — ^MOVE 2 ATR down day, controlled against its own base rate.
6. `06_btc_momentum.py` — 21d return at the top of its year, concentration and era.
7. `07_seasonal_doy_bonds.py` — the midterm Sep-04 bond cell, checked for episode overlap.

---

## Drill outcomes (written after the drills ran)

Nine scripts, five nuggets shipped. What happened to the rest:

| drill | outcome |
|---|---|
| `01` + `01b` + `01c` NFP x pre-holiday | **SHIPPED as the headline.** All 25 NFP-x-pre-holiday dates are July 4th or Labor Day eves, so `01`'s "all pre-holidays" control was the wrong one (mostly Thanksgiving and Christmas). `01b` retried against summer eves, `01c` tightened to the two holidays that can actually collide with payrolls. Result held and got stronger: 22-6 up (78.6%, t 2.78, sign p 0.0019, top-2 episodes = 1% of total, 78.9%/77.8% across the era cut) without payrolls, 13-12 with. Confirmed on IWM (20-8 / 11-14) and crude (19-8 / 10-15). |
| `02` NFP x bond state | **SHIPPED.** 19 payroll eves with IEF inside 1% of its 52-week low across 8 distinct years: ^GSPC 14-5, sign p 0.032, both eras positive, top-2 = -2% of total. h5 bond bounce (TLT 15-4, sign p 0.0096) came out of the same cell so it ships inside the same nugget, not as a sixth. |
| `03` + `03b` yen | **SHIPPED, reframed.** The engine's four yen-cross cells are one event; `P6\|EURJPY=X\|down` is `repeat_blocked` (published 2026-09-02, number unmoved). Shipped from the un-blocked `P6\|JPY=X\|down` fingerprint and ONLY because it adds specificity yesterday did not have: a 0.20th-percentile session, and the snapback being pre-2018 only (21-6 then, 4-6 since), which points the other way from yesterday's four-crosses version. `03b` killed the sharper-looking TLT leg (9-23 down, sign p 0.0100) as GFC-dependent: ex-2008/09 it is 8-15 at sign p 0.105 and the 2018+ half is 4-6. |
| `05` ^MOVE collapse | **KILLED.** The engine's cell (116-177 down, BH pass) does not survive its own control. MOVE's next-session hit rate is 45.4% unconditionally, 43.9% after any down day, 39.8% after any -5% day and 35.8% after a 2 ATR down day. The 2 ATR cut adds ~4pp over "any -5% day" on an index that mean-reverts by construction. Also yesterday's second "today" nugget, so it was doubly out. |
| `06` BTC momentum | **KILLED, and this is the one worth remembering.** The episode-level cell looks excellent (53 episodes, 33-20, +1.26% vs a +0.20% control, era-stable across 2021). But the last episode START was 2026-08-30, so tonight is mid-episode and the episode statistic does not describe tonight's state. The statistic that DOES is the day-level one, and in the modern era that is 52.3% (n=128) against BTC's 52.4% unconditional hit rate. Indistinguishable from the base rate. The `solid` tag_hint and the BH pass both came from a day-level t inflated by overlapping windows. |
| `07` seasonal doy bonds | **SHIPPED as the anecdote.** My reconstruction differs from the engine's (n=6 vs 5, one pick per midterm year 2002-2022, no overlap). TLT 0-6 down at h5, sign p 0.0156, and the 26-observation all-years cell agrees at 8-16, so it is not purely a six-episode fluke. Ships with the conflict against `02`'s h5 leg stated in the open. |
| `08` stocks and gold | **SHIPPED.** Built by hand because no trigger covers it: `P9:stocks_bonds_up` needs TLT +50bp and TLT managed +15bp. 21 sessions since 2000, four of them in 2026 against a prior best of three in 2010. No forward information (13-7, sign p 0.13 vs a 53.3% control), which is the honest content. |
| `09` verification | Every quoted number re-derived: sample starts, the USDJPY rank (14 of 6,928, 0.20th percentile), the 4-leg per-year counts, the IEF percentile (3.1st of 5,813 readings since 2003), the 19 pinned-bond eves across 8 distinct years. |
| ^VIX (both parents) | **DROPPED for repetition, not for weakness.** The payroll-eve VIX cross is real (7-18 down, sign p 0.0216, more negative than either parent). Yesterday's brief already led with VIX into this payrolls print at higher N. Publishing a second conditioner on the same fact one session later is the countdown re-telling the rules ban. |

Anecdote budget: 1 of 2 used, and it is not the headline. Lanes: 3 tomorrow, 2 today.
