# Cell map — run 2026-09-07 (Mon, Labor Day)

asof session 2026-09-04 (Fri, NFP day) -> next session 2026-09-08 (Tue).
Monday 2026-09-07 is Labor Day, market closed. Three dates all differ.
prices_fresh = True (core bar 2026-09-04). cells_scanned 1287, fired 102,
BH crit p 0.0182, 16 pass.

Cycle: midterm year (2026 % 4 == 2). Next session is td 5 of September.

## Session summary (what has to be explained)

Friday was NFP. The whole yen complex ripped: JPY=X -2.05% (dollar down vs yen),
GBPJPY -1.72, EURJPY -1.69, CADJPY -1.67, AUDJPY -1.53. Six crosses in the 5d
rank bottom 5% of their own year on one session. Simultaneously ^TNX sits 0.25%
off its 52w HIGH and TLT/IEF/LQD sit 1.4/0.4/0.25% off 52w LOWS. Yen up while US
yields are at the highs is the anomaly worth drilling. Asia/EM rallied hard
(EEM +1.82, ^HSI +1.74, ^KS11 +1.64, FXI +1.53, ^N225 +1.26) while ^FCHI sits in
the bottom 5% of its 21d range. ^VIX +1.47 into a three-day weekend. ^SKEW 21d
rank 98.0. BTC 21d rank 99.6 but printed -1.97 on the day.

## Event lane

| trigger | verdict |
|---|---|
| `E:holiday_post` | **DRILL** — the flagship. Next session is the one after Labor Day. Engine pooled ALL holidays: ^VIX +3.83% n=246 hit 69.5 t=7.10 BH-pass, GC=F +0.26 t=3.19 BH, SI=F +0.44 t=3.05 BH, HYG -0.098 t=-2.43, and SPY/^GSPC dead flat (-0.003 / -0.019, abs t < 0.3). Pooling Thanksgiving, July 4 and New Year with Labor Day is exactly the over-broad cell this product exists to sharpen. Drill 01: Labor Day ONLY. |
| `E:weekday_month` | **SKIP(subsumed)** — "Tuesdays in September" fires every day by construction and its ^VIX +1.87 t=2.43 is the same post-holiday vol effect leaking in (the first September Tuesday is usually the post-Labor-Day session). HG=F 41-68 down (sign p 0.011, BH pass) is a swept day-of-week cell with no mechanism; declining to launder it into a nugget. Drill 01 controls for the overlap explicitly. |
| `E:ppi` | **SKIP(no mechanism, k=3)** — PPI is 3 sessions out on 09-10. The whole 18-subject group is anticipation noise: only NG=F clears BH (+0.447 t=2.48 n=309) and natural gas three days before a PPI print has no transmission channel worth Scott's attention. The equity cells are +0.05 to +0.09 with t < 0.9. CPI at 4 td is the event that matters and it is outside the k<=3 anchor window. |
| `E:seasonal_doy` | **SKIP(weak)** — Sep 08 +/-2: SPY h1 all-years -0.228% on 14-12 down (sign p 0.42), midterm 4-2 down on n=6. Nothing separates from zero. The h5 midterm SPY +0.637 (5-1 up) is n=6 and would be an anecdote headline, which the tag budget forbids. |

Calendar entries inside the next five sessions, each with a verdict:

- **2026-09-08 Tue, post-Labor-Day** — DRILL (01). This is the next session itself.
- **2026-09-10 Thu, PPI (3 td)** — SKIP, see above.
- **2026-09-11 Fri, CPI (4 td)** — NOTE ONLY in the calendar block. Outside the
  engine's k<=3 event anchor, so no cell exists and I am not going to invent one
  from recall. It is the week's real event and gets one calendar line.
- **2026-09-16 Wed, FOMC + VIX expiry (7 td)** — calendar line only, too far out.
- **2026-09-18 Fri, opex + quad witching (9 td)** — calendar line only.

## Price lane

| trigger | verdict |
|---|---|
| `P5:rank5_extreme` (JPY complex) | **DRILL** — 6 of the 8 kept subjects are yen crosses, plus JPY=X itself (n=326, +0.121% h1, 59.2 hit, sign p 0.0005, BH pass) and EURJPY (n=286, 60.1 hit, sign p 0.0004, BH pass). The engine scores each cross in isolation; the actual event is that they all fired at once. Drill 02 builds the cross-sectional breadth condition the engine cannot express. |
| `P6:two_atr_day` (JPY) | **DRILL** — folded into 02. EURJPY -1.69 was a >=2 ATR down day: n=34, +0.48% h1, 70.6 hit, t=2.99, BH pass. JPY=X n=42, 69.0 hit, sign p 0.0098, BH pass. Same underlying event as P5; publishing both separately would be one fact twice. |
| `P7b:down_streak` (NZDJPY, CHFJPY) | **SKIP(same fact)** — third view of the yen move. NZDJPY 5+ down closes n=140 +0.109 t=1.34. Rolled into 02 as corroboration, not published separately. |
| `P8:sma200_cross` (NZDJPY down) | **SKIP(dead)** — n=22, h1 +0.002%, 11-11, t=0.01. Degenerate. |
| `P5b:rank21_extreme` (^SKEW top 5%) | **DRILL** — engine computed SKEW's own mean reversion (n=361, -1.25%, t=-7.48, BH). SKEW reverting tells Scott nothing; what he wants is whether a 98th-percentile crash-hedge bid says anything about SPY. Drill 03 does the transfer with a real control. |
| `P5b:rank21_extreme` (BTC/ETH top 5%) | **DRILL** — BTC 21d rank 99.6, n=304 +0.733 t=2.88 BH pass, but Friday printed -1.97%. Stretched-and-still-rising is a different cell from stretched-and-rolling-over. Drill 04 splits it. ETH (n=190, t=1.36, era-unstable) rides along as corroboration only. |
| `P5b:rank21_extreme` (^FCHI bottom 5%) | **SKIP(era)** — n=400 +0.174 t=1.41, BH pass on sign but `era_stable: false`. An era-unstable European mean-reversion cell is not worth one of 4-8 slots when the yen and the bond complex are both live. |
| `P2 / P2b:new_52w_low` (EURAUD) | **SKIP(thin)** — n=24 / n=14, t=1.64 / 0.07. EURAUD is also not a subject Scott has any use for. |
| `P3c:pop50_after_low` (^VIX3M) | **SKIP(mechanical)** — n=64 t=1.77. VIX3M popping off a 52w low the session before a three-day weekend is the holiday-decay artifact drill 01 already owns. |
| `P4:z10_extreme` (EURAUD down, AUDNZD up, ^BVSP up) | **DEAD/SKIP** — EURAUD -0.017% t=-0.37, AUDNZD -0.04 t=-1.41, ^BVSP +0.064 t=0.83. All indistinguishable from zero. |
| `P7:up_streak` (CL=F, USDTRY) | **SKIP** — CL=F n=201 t=-0.56, nothing. USDTRY 5+ up closes n=421 70.8 hit BH pass is a pure carry/devaluation drift artifact, not a market observation. |
| `P5:rank5_extreme` (ZW=F bottom 5%) | **SKIP(weak)** — wheat -2.72% was the day's worst tape print but the cell is n=307 t=1.64 sign p 0.15. No edge over control. |
| `P5b` (ZC=F, ZS=F top 5%) | **SKIP(weak)** — corn t=2.03 on n=425 with sign p 0.35, beans t=0.59. Grain complex is stretched but the forward cells are noise. |
| capped: `P5:rank5_extreme` dropped CT=F, EWZ, USDTRY, AUDNZD, ^BVSP | **SKIP** — none is a macro subject worth a slot; the cap did not hide anything. Cotton and Brazil are outside what this brief covers. |

## Not in the sweep, drilled anyway

**The bond/yen dislocation.** Nothing in `PRICE_TRIGGERS` fires on "^TNX near a
52w high while TLT/IEF/LQD sit at 52w lows and the yen rallies 2%", because the
engine has no cross-asset divergence trigger for it (P9 covers stocks+bonds and
dollar+gold, not rates+yen). This is the most unusual thing on the tape and it
gets drill 05. Flagged as an inventory gap for the aligned-pair table.

## Engine hints I am not inheriting

- `tag_hint` downgrades taken: the entire `E:ppi` group arrives `suggestive` on
  n~310 and is being dismissed outright, not published at a lower tag.
- `bh_pass` exemptions: `E:holiday_post` is a PRE-SPECIFIED cell (post-holiday
  drift is a named calendar hypothesis, it was not found by this sweep) so it
  does not owe the BH correction a nod. It passes anyway. Everything in the
  PRICE lane WAS found by the search and is held to `bh_pass` before any
  `[solid]` tag. ^SKEW and BTC both carry it; the drill-03 SPY transfer does not
  inherit SKEW's BH pass because it is a different cell and is scored on its own.
- `E:weekday_month` HG=F and `P7:up_streak` USDTRY both carry `bh_pass: true`
  and are still being skipped. BH controls the false-discovery rate; it does not
  supply a mechanism.

## Novelty

`delta_suppressed: false`, so new claims are allowed tonight. `E:ppi|NG=F|k3` is
the only fingerprint with history (published 2026-08-10, 19 td ago, number has
not moved) and it is not `repeat_blocked` — being skipped on merit instead.
Everything else in the fired set is `is_new: true`.
