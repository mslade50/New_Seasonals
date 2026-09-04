# Cell map — run 2026-08-13 (Thu), asof session 2026-08-13, next session 2026-08-14 (Fri)

Midterm year. Prices FRESH (core bar 2026-08-13), both lanes live.
Sweep: 1167 scanned, 63 fired (36 event / 27 price), BH crit p 0.0001, 4 pass.
`dropped_by_cap` is empty, so the map covers the whole fired surface.

Not quiet: price triggers fired, so the 4-8 nugget contract applies.

## Repetition state (read BEFORE selecting)

The journal's fingerprints are all drill-specific, so every engine base
fingerprint reads `is_new`. That is an artefact, not a licence. What actually
published in the last 5 sessions and constrains tonight:

| published | fingerprint | constraint tonight |
|---|---|---|
| 08-10 (3 td) | `P9:level_divergence\|SPY_TLT\|high_low` | SPY at a 52w high with TLT on the floor is BLOCKED. Same state is still live (SPY 0.0%, TLT +0.82% off its low) and the number has not moved. Any high/low cross-asset nugget tonight must be a different cell, not this one re-counted. |
| 08-11 (2 td) | `P5:rank5_extreme\|^SKEW\|vix_divergence` | SKEW published at the 98.4th percentile as a record TAIL BID. Tonight's SKEW cell is the opposite state (21d return 2nd percentile). Opposite sign, new number, different fingerprint: not a repeat, but the brief must say it is the same instrument two sessions later or it reads as amnesia. |
| 08-12 (1 td) | `E:ppi\|TLT\|k1\|*`, `P10:move_crush`, `P1:new_52w_high\|ZC=F` | PPI is spent (printed today). No countdown re-tellings. MOVE fell another 3.97% today, but a second MOVE nugget one session later is exactly the banned re-telling. |
| 08-12 (1 td) | `P4:z10_extreme\|^GSPC\|thrust_stall_at_high` | SPY/GSPC z10 stretch at a high was covered last night (z10 2.16, now 1.76). Not again. |

## Engine hints I am not inheriting

- `bh_pass` on `E:weekday_month|^VIX` prices the SWEEP. The August-Friday VIX
  cell is not a pre-specified famous hypothesis in the "FOMC drift" sense, so
  it does owe the correction, and it passes it. Keeping the pass, but the real
  risk is not multiplicity, it is that the Friday weekend effect is a
  well-known confound the cell does not separate. That is a DRILL, not a tag.
- `tag_hint: solid` on `P5b:rank21_extreme|^SKEW` describes SKEW mean-reverting
  in SKEW, which is mechanical and uninteresting. Downgrading the subject: the
  publishable question is what the collapse says about SPY and VIX, and that is
  a different cell with its own N.
- `tag_hint: solid` on `P6:two_atr_day|HE=F` is real and irrelevant. Lean hogs
  are not macro. Out of scope by universe rule, not by statistics.

## Data defect found tonight (excluded, not a nugget)

`tape.breadth.pct_above_sma200` = 64.4 with `n_members` 98.
`breadth_series` in `build_context_state.py:1103` is handed the FULL subject
dict, so the denominator includes 29 FX crosses and 20 futures. Its own
docstring says "equity-index + sector panel". A dollar-yen cross above its
200d mean is being counted as market breadth. The number is uninterpretable as
breadth and cannot support the divergence nugget it superficially invites
(64.4% now vs 69.0% 21d ago with the index at a high). SKIP, and flag to
McKinley outside the brief.

## Event lane

| trigger | cell | verdict |
|---|---|---|
| `E:weekday_month` | Fridays in August, `^VIX` n=117, -0.921%, 33-83 down, sign p 0.0000, bh_pass | **DRILL** — 71.6% down is the strongest thing on the board for tomorrow. Entire question is whether August adds anything to the plain Friday weekend effect. Control against all Fridays, all August non-Fridays, and condition on VIX entering low (14.63, 63d rank 20.6). -> `01` |
| `E:weekday_month` | Fridays in August, `TLT` 64-44 up sign p 0.0335, `IEF` 65-43 sign p 0.0214 | **DRILL** — two bond legs agreeing, and TLT sits 0.82% off a 52-week low, which makes the cell worth more than its t of 1.81. Same Friday-effect control problem. -> `02` |
| `E:weekday_month` | Fridays in August, `EURUSD=X` 35-58 down sign p 0.0198, `DX-Y.NYB` 52-62 | **DRILL** — pairs with the Aug-14 doy dollar cell below; two roughly independent calendar cuts pointing the same way is the only reason either is worth print. -> `03` |
| `E:weekday_month` | Fridays in August, `GC=F` t=2.30 65-48 up | SKIP(gold published 08-11 in its CPI form; t=2.30 with sign p 0.066 is the weakest of the three FX/metal legs and it would be the third calendar nugget in a row) |
| `E:weekday_month` | Fridays in August, `SPY` t=0.23 / `^GSPC` t=0.20 / `QQQ` t=-0.75 | SKIP(empty — this is the honest reason the headline is not an equity calendar claim) |
| `E:weekday_month` | Fridays in August, `NG=F` `SI=F` `HG=F` `CL=F` `JPY=X` `HYG` `EEM` `^TNX` `IWM` | SKIP(none clears sign p 0.05 with an era-stable sign; `NG=F` at p 0.066 is era-unstable, `^TNX` p 0.069 is the mirror of the TLT leg already drilled) |
| `E:seasonal_doy` | Aug 14, `EURUSD=X` all years 16-6 down p 0.0262, midterm 5-0 down p 0.0312 | **DRILL** — feeds `03`. A 5-0 midterm record is n=5 and cannot carry itself; it is corroboration or it is nothing. |
| `E:seasonal_doy` | Aug 14, `TLT` h5 17-6 up p 0.0173, `IEF` h5 17-6 up p 0.0173 | **DRILL** — feeds `02`. Note this is the h5 leg, a different horizon from the Friday h1 leg, so combining them is a claim about the week, not the day. |
| `E:seasonal_doy` | Aug 14, `^VIX` h1 19-7 down p 0.0145 | **DRILL** — feeds `01`, same direction as the August-Friday VIX cell. Two calendar cuts of the same weekend. |
| `E:seasonal_doy` | Aug 14, `SPY` `^GSPC` `IWM` `QQQ` | SKIP(h1 all-years 14-12, 14-12, 12-13, 18-8; the QQQ 18-8 at p 0.0378 is the exact cell the 08-10 brief killed with an anchor walk — neighbouring anchors run 13-13, 15-11, 10-16. Killed again by prior work, not re-drilled) |
| `E:seasonal_doy` | Aug 14, `HG=F` midterm 5-0 up p 0.109, `SI=F` midterm 5-1 down p 0.109 | DEAD(n=6 midterm cells at p>0.10, and the anchor-walk precedent says single-anchor seasonals of this size are noise) |
| `E:seasonal_doy` | Aug 14, `GC=F` `CL=F` `NG=F` `DX-Y.NYB` `JPY=X` `HYG` `EEM` `^TNX` | SKIP(nothing below p 0.10 that is not already covered) |

Calendar entries inside the next 5 sessions: **VIX expiry Wed 2026-08-19 (4 td)**
is the only one. Opex 8/21 is 6 td and Jackson Hole 8/28 is 11 td, both outside
the window. Verdict on the VIX expiry: SKIP tonight(4 td out, no trigger fired
on it, and leading with it would be the countdown re-telling the novelty rule
bans. It belongs in the Calendar block, not as a nugget).

## Price lane

| trigger | cell | verdict |
|---|---|---|
| — | **SPY, ^GSPC, IWM, ^RUT, ^NYA and HYG all closed AT a 52-week high on the same session; QQQ did not (-1.78%)** | **DRILL** — no trigger fired on this because P1 requires a FIRST high in 30+ days and these have been printing highs. It is the most specific thing on tonight's tape and it is a different cell from the blocked 08-10 SPY/TLT divergence: this one is about equity and credit breadth agreeing while the growth index lags. -> `04` |
| `P5b:rank21_extreme` | `^SKEW` bottom 5%, n=350, +0.981%, t=6.26, bh_pass | **DRILL** — downgrade the subject per above. SKEW at 134.37, 21d return -9.52%, 2nd percentile of its year, two sessions after publishing at the 98th. Real question is SPY and VIX forward, plus how rare the round trip is. -> `05` |
| `P7:up_streak` | `^NYA` n=239, -0.106%, 109-130 down, sign p 0.0978, era-stable | **DRILL** — weak alone, but the NYSE composite on a 5-day run AND at a 52-week high AND at a 21d rank of 90.5 is the same object as `04`. Folding in rather than publishing the bare cell. -> `04` |
| `P9:stocks_bonds_up` | `SPY` 176-137 up sign p 0.0183 but edge -0.001%; `TLT` edge +0.012% | SKIP(the hit rate is entirely the equity drift baseline — an edge of one ten-thousandth of a percent is the definition of a cell that says nothing. Today's post-PPI joint rally is real and means nothing forward) |
| `P5:rank5_extreme` | `EWZ` bottom 5%, `^BVSP` bottom 5% | SKIP(published 08-12 as `P7b:down_streak\|EWZ\|into_a_us_high`, one session ago, and the state has barely moved. Blocked) |
| `P7b:down_streak` | `^BVSP` 85-60 up p 0.0229, `EWZ`, `USDMXN=X` | SKIP(same Brazil object as the line above, published 08-12) |
| `P4:z10_extreme` | `USDTRY=X` stretched up, n=578, t=3.15, bh_pass, 400-177 | DEAD(degenerate. A pegged-then-devaluing carry pair drifts up on 69% of all sessions by construction; the "edge" is the depreciation trend. Not a market fact) |
| `P7:up_streak` | `USDTRY=X` 294-121 up, bh_pass | DEAD(same degeneracy, era-unstable on top of it) |
| `P4:z10_extreme` | `SB=F` stretched up t=-0.05 | SKIP(empty) |
| `P4:z10_extreme` | `USDCNY=X` stretched down, t=-1.26 | SKIP(empty, and a managed currency's 2 sigma is not the same object as a floating one's) |
| `P5:rank5_extreme` | `CADJPY=X` `GBPCHF=X` top 5% | SKIP(t=-0.54 and -0.72, both empty) |
| `P5:rank5_extreme` | `ZC=F` top 5% | SKIP(corn published 08-12 as the second largest session on record. Blocked, and the follow-on cell is empty at t=0.40) |
| `P5b:rank21_extreme` | `SB=F` top 5%, t=0.40 | SKIP(empty) |
| `P5b:rank21_extreme` | `CHFJPY=X` bottom 5%, 187-148 p 0.0189, t=0.79 | SKIP(hit rate without magnitude, mean +0.06% on a cross whose daily sigma is ~0.5%. Nothing to say) |
| `P6:two_atr_day` | `HE=F` down, n=115, t=-2.77 tag solid | SKIP(out of universe scope — lean hogs are not macro context. The statistic is fine and irrelevant) |
| `P6:two_atr_day` | `KC=F` `PA=F` `LE=F` `USDCNY=X` down | SKIP(coffee/palladium/cattle out of scope as macro subjects; none clears p 0.10 anyway) |
| `P8:sma200_cross` | `HE=F` crossing down, n=25 | DEAD(out of scope and t=-0.44) |
| `P7:up_streak` | `CADJPY=X` `AUDJPY=X` | SKIP(yen-cross carry streaks, t=1.75 and 1.61, means under 0.08%. Below the specificity bar) |

## Drill queue

1. `01_august_friday_vix.py` — is the August-Friday VIX cell just the weekend effect
2. `02_august_friday_duration.py` — TLT/IEF Friday leg + the Aug-14 h5 week leg, with TLT at a 52w low
3. `03_doy_dollar.py` — EURUSD Aug-14 anchor walk plus the August-Friday dollar leg
4. `04_broad_52w_high_cluster.py` — SPY+IWM+HYG+^NYA joint 52w highs with QQQ lagging
5. `05_skew_collapse.py` — SKEW 21d bottom 5%, forward SPY/VIX, and the 98th-to-2nd round trip
