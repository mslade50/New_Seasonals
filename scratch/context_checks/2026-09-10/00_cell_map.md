# Cell map — run date 2026-09-10

asof session 2026-09-10 (Thu, PPI) | next session 2026-09-11 (Fri, CPI) | midterm year
prices_fresh = true, core bar 2026-09-10 | 1199 cells scanned, 96 fired, BH crit p 0.0092, 10 pass

## Gate that shaped the whole evening: continuous-contract roll seams

Drill 01 re-ran yesterday's volume test on every commodity in the tape. Seven of tonight's
loudest bars are contract changes, not trading. Session volume against the trailing 20-day
median: KC=F 534x, SI=F 1079x, HG=F 137x, PA=F 3295x, CT=F 628x, GC=F 338x, PL=F (prior
median 0). Coffee's entire -9.16% is the gap. Cotton's bar prints a -100% open. Every cell
whose ENTRY STATE is one of those bars is dead tonight regardless of its statistics.
Clean by the same test: CL=F 1.69x, SB=F 1.43x, ZC=F 1.89x, NG=F 1.11x, and all ETFs/indices.

## Second carried fault, and it moved a number

09-08 recorded that ^VIX holds bars on 2026 market closures. Drill 10 confirms four phantom
bars (05-25, 06-19, 07-03, 09-07) and that 09-07 is inside tonight's window. The engine's
P7:up_streak fired at "5+ consecutive up closes"; on the SPY calendar the real count is FOUR
(09-04 14.53, 09-08 15.72, 09-09 16.46, 09-10 17.84). The state did not fire as reported.
No CPI eve/print pair touches a phantom bar, so the event lane is unaffected.

## Event lane

| trigger | verdict |
|---|---|
| `E:cpi` (18 subjects, top tier, next session) | **DRILL -> PUBLISH x3.** Drills 02/03/04/07/09. See per-subject notes below. |
| `E:weekday_month` Fridays in September (18) | SKIP(weak and pre-empted). Best equity leg ^GSPC -0.136% t -1.41. The only BH pass is EURUSD=X at +0.048% on n=96, a mean too small to be worth a line, and the CPI owns the same session anyway. |
| `E:seasonal_doy` Sep 11 (18) | DEAD(n=0). Every subject returns n=0 under the midterm cycle-phase filter. TLT and NG=F are also `repeat_blocked` from 09-03 and 09-08. |

Per-subject inside `E:cpi`:
- `^VIX` (n=318, -0.86%, 120-195, t -2.15, BH pass) -> **PUBLISH**, split by how vol entered. Drill 04.
- `^GSPC` / `SPY` (n=318, +0.02/0.03%, edge -0.01%) -> SKIP(no edge pooled; the CPI equity legs published 09-08 as the k3 run-up and the Friday-print cell). Re-enters only through the risk-shape split.
- `QQQ` (n=318, +0.117%, t 1.29, BH pass) -> **PUBLISH** conditioned on the eve's risk shape. Drill 07/10.
- `^TNX` / `TLT` / `IEF` -> **PUBLISH** the 10y-at-a-252d-high conditioning as an anecdote. Drill 09.
- `GC=F` `SI=F` `HG=F` (all BH-relevant, gold t 2.32) -> SKIP(roll seam). The cells are fine historically but tonight's entry state is a contract change, so there is nothing true to say about walking into the print.
- `CL=F` -> DRILL then SKIP. Crude +5% CPI eves are n=6 (drill 07), and the pooled leg is t 0.63. The crude story moves to the price lane as rarity, not as a CPI conditioning.
- `EEM` (BH pass, +0.139%, 160-119) -> SKIP(slot). Real but ordinary, and EEM published 09-07.
- `EURUSD=X` `JPY=X` `DX-Y.NYB` -> SKIP(repetition). The yen complex published 09-02, 09-03, 09-06 and 09-07. Four briefs is enough.
- `NG=F` `IWM` `HYG` `HG=F` -> SKIP(t under 1, no cell).

## Price lane

| trigger | verdict |
|---|---|
| `P2:new_52w_low` IEF (n=12) | DRILL -> folded into the bond nugget. Drill 05/09. |
| `P2b:new_52w_low_90` IEF (n=8) | same, folded. |
| `P3:drop50_after_high` HG=F | SKIP(roll seam). Copper's "52-week high then reversal" is the contract change. |
| `P3b:drop100_after_high` HG=F | SKIP(roll seam), same bar. |
| `P4:z10_extreme` up: ^IRX ^TNX ^FVX CL=F ^BVSP | SKIP(repetition). The bill-yield repricing published 09-09 and the 10y-near-its-high cell 09-06. Crude handled separately. |
| `P4:z10_extreme` down: AUDJPY NZDJPY EURJPY | SKIP(repetition), yen complex. |
| `P5:rank5_extreme` bottom: KC YM JPY IEF | KC DEAD(roll seam); JPY SKIP(repetition); IEF folded into the bond nugget; YM=F SKIP(t 0.98). |
| `P5:rank5_extreme` top: ^VVIX ^IRX ^FVX ^TNX | **^VVIX DRILL -> PUBLISH** (drill 08/09). Rates legs SKIP(repetition). |
| `P5b:rank21_extreme` top: ^IRX ZC ZW ZS SB | SKIP(seam-built). Yesterday established corn, soybeans and sugar 252-day highs as roll artefacts; today's ZC bar is clean but its 21-day rank is measured through the seam. |
| `P5b:rank21_extreme` bottom: ^FCHI NZDJPY JPY | SKIP. CAC published 09-09, yen repetition. |
| `P6:two_atr_day` down: KC PA PL SI | DEAD(all four roll seams). |
| `P6:two_atr_day` up: CL ^MOVE SB CT | **CL=F DRILL -> PUBLISH as rarity** (drill 06). ^MOVE published 09-02. CT roll seam. SB n=49, t -0.29. |
| `P7:up_streak` ^VIX (n=87) | SKIP(state did not fire). Phantom 09-07 bar; the real streak is 4. The correction itself goes in the footnote. |
| `P7b:down_streak` ^FTSE | SKIP(t -0.46, and ^FTSE has no 09-10 bar). |
| `P8:sma200_cross` up ^VVIX (n=39) | DRILL -> folded into the VVIX nugget as a cross. The below-200d arm is n=9. |
| `P8:sma200_cross` down AUDJPY (n=22) | SKIP(repetition, t -0.40). |
| `P9b:stocks_bonds_down` SPY TLT (n=270) | SKIP(no edge). SPY +0.126% t 0.97, TLT +0.017% t 0.29, both inside baseline. Superseded by the sharper all-three-at-lows cell. |

## Calendar inside five sessions

- Fri 09-11 CPI, top tier, next session -> leads the brief.
- Wed 09-16 FOMC decision (4 td) -> SKIP. The pre-FOMC drift published 09-09 and TODAY'S close
  was its anchor. Re-telling it tomorrow is the banned countdown.
- Wed 09-16 VIX expiry (4 td) -> SKIP(no cell, not the next session).
- Fri 09-18 opex + quad witching (6 td) -> outside the window, calendar line only.

## Hints not inherited

- `tag_hint` on `P5:rank5_extreme|^VVIX` is **solid**, earned on 239 overlapping sessions.
  Declustered at 5 td it is 106 episodes and the S&P transfer collapses (h5 +0.248% against a
  +0.203% baseline and a +0.210% local control). The VVIX's own reversion survives and keeps
  solid; the index leg is published as the null it is.
- `bh_pass`: CPI is a pre-specified subject, not a swept discovery, so the event lane owes the
  sweep no correction. The two splits I built on it are constructions and each quotes its own
  control instead. The bond-lows and crude cells are rarity statements, not p-values.

## Near-misses kept on file

- CPI eves with crude up 5% or more: n=6, ^GSPC -0.853%. Turns on at n>=15, roughly a decade away.
- TLT+IEF+LQD at lows, h5: -0.206% against a local control of -0.205%. Publishes as a null.
