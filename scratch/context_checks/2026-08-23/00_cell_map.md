# Cell map — run 2026-08-23 (Sunday)

asof session 2026-08-21 (Fri) | next session 2026-08-24 (Mon) | midterm year
prices_fresh = True, core bar 2026-08-21. Both lanes live.
sweep: 1195 cells scanned, 69 fired (36 event / 33 price), BH crit p 0.0109, 7 pass.

Next session position: td 16 of 21 in August, 5 sessions from month end.
Monthly opex was FRIDAY 2026-08-21, so Monday is the first post-opex session.

## Engine hints I am not inheriting blind

- `tag_hint` floor only. Everything below is graded on the drilled numbers.
- Pre-specified vs swept. `E:weekday_month` (day-of-week x month), `E:seasonal_doy`
  and the post-opex window are FAMOUS pre-specified hypotheses, not sweep finds,
  so they do not owe BH a correction. Every P-lane cell below WAS found by the
  sweep and does owe it.
- `dropped_by_cap`: P5b:rank21_extreme truncated at 8, dropping DX-Y.NYB,
  EURUSD=X, UUP, USDSGD=X. The dollar drop matters (DXY 21d rank 0.4, the
  lowest reading on the whole tape), so it is recomputed by hand in drill 05
  rather than treated as absent.

## Event lane

| trigger | verdict |
|---|---|
| `E:weekday_month` Mondays in August | **DRILL** (01). QQQ 76-41 up, n=117, sign p 0.0008, BH pass, era stable, and the mean is only +0.14% at t=1.12. A hit-rate cell, not a magnitude cell, and Monday is the next session. ^VIX +2.39% t=2.33 same cell needs a plain-Monday control before it means anything (weekend decay is the obvious confound). |
| `E:seasonal_doy` Aug 24 +/-2 | **DRILL** (02). QQQ 19-7 up all years (p 0.0145); IWM midterm 5-1 up; NG=F midterm h5 6 of 6 down at -5.27% mean (p 0.0156). Crosses directly with the Monday cell. |
| `E:seasonal_doy|TLT` | **SKIP(repeat_blocked)** — published 2026-08-17 and the number has not moved. |
| `E:opex` | not in `cells_index`: opex was Friday, so the k=1..3 anchor window is behind us. The forward side of it (the post-opex week) is NOT swept by the engine at all. **DRILL** (03), and flag the trigger-inventory gap. |
| `E:jackson_hole` 2026-08-28 | not fired: k=5, outside the engine's k in 1..3 anchor. Inside the next five sessions, so it goes in the Calendar block. **DRILL** (06) only if a week-into-JH cell survives its control; relevance to Monday itself is weak. |
| `E:month_end` / `E:turn_of_month` | **SKIP(not yet)** — the last-3 window opens Thu 2026-08-27. Calendar block only. |
| `E:cpi` `E:nfp` `E:ppi` `E:fomc_*` `E:quad_witching` `E:vix_expiry` `E:election` | **SKIP(out of window)** — nearest is NFP at +10 td. Calendar block only. |
| `E:holiday_pre` / `E:holiday_post` | **SKIP(no closure nearby)** — next is Labor Day, 2026-09-07. |

## Price lane

| trigger | subjects | verdict |
|---|---|---|
| `P4:z10_extreme` up | BTC-USD (z10 2.98, t=3.14, BH pass), ETH-USD, USDTRY=X, ZC=F | **DRILL** (04) for BTC only, and the drill has to answer the calendar problem: bitcoin prints seven bars a week, so its h=1 is Saturday 2026-08-22, which is behind us by the time this posts. Any bitcoin nugget has to move to a horizon that is still ahead. USDTRY **SKIP(degenerate)** — a managed depreciation trending one way is not a forward-return claim. ZC=F **SKIP(covered)** — corn published 2026-08-17. |
| `P5:rank5_extreme` top | BTC, ETH, CT=F, ZS=F, GC=F, NZDUSD=X, ZC=F | **SKIP(same state)** — for BTC/ETH this is the z10 cell wearing a second hat; drilled once in 04. GC=F t=1.13 and the grains are inside their own noise. |
| `P5:rank5_extreme` bottom | HE=F | **DEAD(roll artifact)** — lean hogs continuous contracts gap on roll and the trigger cannot tell a roll from a move. Same reason LE=F's -2.44% session is not a nugget. |
| `P5b:rank21_extreme` top | BTC, ETH, GC=F, SB=F | **SKIP** for BTC/ETH (folded into 04). GC=F n=418 t=-0.56, nothing there. Gold's own thrust was published 2026-08-19, so a second gold-momentum nugget four days later fails novelty on its own terms. |
| `P5b:rank21_extreme` bottom | USDZAR=X (t=2.39 BH pass), JPY=X (t=2.08 BH pass), HE=F, CAD=X | **DRILL** (05) but as the DOLLAR cell, not four separate FX cells: DXY 21d rank 0.4 was capped out of this very trigger, and USDZAR / JPY / CAD / USDSGD / EURUSD all firing the same tail at once IS the dollar. HE=F dead as above. |
| `P6:two_atr_day` down | KC=F (-10.6%, the biggest move on the tape), USDCNY=X | **DRILL** (07) for coffee. USDCNY **SKIP(magnitude)** — h1 mean +0.096% on a managed currency, and the record is 48-48. |
| `P6:two_atr_day` up | ETH, BTC, ZC=F | **SKIP** — folded into 04 / already covered. |
| `P7:up_streak` | BTC, ETH, EURJPY=X, CT=F | **SKIP** — BTC folded into 04; EURJPY t=-2.11 is a cross-rate mean reversion of -0.095% and Scott does not trade EURJPY; CT=F t=0.20. |
| `P7b:down_streak` | USDHKD=X | **DEAD(magnitude)** — 64.2% up next at a mean of +0.01%. That is the peg band, not a market. |
| `P1:new_52w_high` / `P1b` | CT=F | **SKIP(contradictory and thin)** — the 30-day gate says +0.096% and the 90-day gate says -0.333% on n=15. Nothing survives that. |
| NOT FIRED but on the tape | HYG -0.23% from its 52w high while TLT sits +0.86% off its 52w LOW and ^TNX is -0.15% from its 52w high | **DRILL** (08). No single trigger sees this because it is a cross-asset state and the engine's P9 family does not carry a credit-versus-duration pair. Flag as a second trigger-inventory gap. |

## Inventory gaps found tonight

1. No forward-side opex trigger. `E:opex` anchors k=1..3 BEFORE expiration and
   goes dark the moment it passes, so the post-expiration week, which is the
   single most-studied window in the equity calendar, is invisible to the sweep.
2. No credit-versus-duration pair in the P9 family (HYG at a 52w high against
   TLT at a 52w low is the live example).

Both are engine-side changes, noted here, not made tonight.
