# Cell map — run 2026-08-19 (Wed), asof session 2026-08-19, next session 2026-08-20 (Thu)

midterm year, August, td 14 of 21, 7 td from month end.
prices_fresh = True (core bar 2026-08-19). Price lane is live.
sweep: 1213 cells scanned, 93 fired (54 event / 39 price), BH crit p 0.0045, 4 pass.
capped: P5b:rank21_extreme dropped JPY=X SB=F USDSGD=X; P6:two_atr_day dropped
LE=F ZS=F TLT CHF=X GBPCHF=X UUP EURCHF=X DX-Y.NYB USDCNY=X. TLT and DX-Y.NYB
are the two that matter tonight and both get recomputed in a drill rather than
inherited.

## The session that just closed, in one paragraph

Equities did essentially nothing (SPY +0.21, QQQ -0.20, IWM +0.50, all inside
half an ATR). Everything else moved: DXY -0.85% to 98.80, its 21d return at the
0.4 percentile of its own year and its first close below the 200d mean in 63+
sessions; gold +4.92%, silver +4.99%, platinum +6.00%, palladium +3.86%; corn
+7.56% at a 52w high, wheat +4.82%, cotton +4.26%; TLT +1.67% (2.1 ATR) with
the 10y-5y curve flattening 2sd; ETH +18.91%, BTC +7.64%; VIX -6.0% to 14.89
on its own expiry day, VVIX -6.83%, MOVE -4.96%. Coffee -9.70%.

## Calendar inside the next five sessions

| date | event | td ahead | verdict |
|---|---|---|---|
| 2026-08-21 | monthly opex | 2 | DRILL, but only on a subject the last two briefs did not use. The 08-18 brief published the three-session bond run into this same expiry; a second telling is the banned countdown. NG=F is the one k2 subject with a real number (n=311, 129-181 down, sign p 0.0023, BH pass, era-stable) and has never been a subject here. |
| 2026-08-28 | Jackson Hole | 7 | SKIP(outside the anchor window; k runs 1..3 and no cell fired. Calendar line only). |
| 2026-09-04 | NFP | 12 | SKIP(too far; nothing anchored). |

VIX expiry was TODAY (td_ahead 0), so it is a today-lane fact, not a
tomorrow-lane cell. The 08-16 and 08-17 briefs both led on VIX-expiry drift and
the 08-13 brief on August Fridays for VIX; the vol-calendar seam is exhausted
for now and is deliberately not revisited.

## Event lane

| trigger | verdict |
|---|---|
| `E:opex` (k2, 18 subjects) | DRILL for NG=F only. The index leg is dead on arrival: SPY -0.028% at t -0.43, ^GSPC -0.012%, QQQ -0.006%, all inside noise. ^VIX 138-181 down at sign p 0.0093 is real but is the third vol-calendar item in five briefs. SKIP(recently covered) for the rest. |
| `E:weekday_month` (Thursdays in August) | DRILL. Two BH passes in one direction: TLT 68-39 up (sign p 0.0045) and ^TNX 43-74 down (sign p 0.0037), with IEF 64-42 at p 0.0335 agreeing. Both flagged era_stable=False, so the drill has to be an era test before it is anything else. TLT alone is off limits (the 08-13 brief published the August-TLT month cell and the 08-17 brief the late-August TLT seasonal); the rates cross is a different claim and only publishes if the era split survives. |
| `E:seasonal_doy` (Aug 20 +/-2) | DRILL as the corroborating leg of the same rates question: ^TNX 7-19 down p 0.0145, IEF 17-6 up p 0.0173, TLT 16-7 p 0.0466. `E:seasonal_doy|TLT` is repeat_blocked (published 08-17) and stays blocked. Equity legs SKIP(SPY 14-12, QQQ 15-11, IWM 12-13, no cell). GC=F h5 17-8 +0.605% SKIP(the 08-16 brief already published gold's mid-August seasonal). |

## Price lane

| trigger | subjects | verdict |
|---|---|---|
| `P8:sma200_cross` down | DX-Y.NYB, CAD=X | DRILL, and this is tonight's spine. The dollar's first 200d break in 63+ sessions arriving on the same session as a 2-ATR down day and a 0.4-percentile 21d return is three states at once; the engine priced only the weakest of them (n=21, mean +0.05%, 11-10). Recompute jointly. CAD=X SKIP(same dollar fact in one cross). |
| `P8:sma200_cross` up | EURUSD=X, ETH-USD, BTC-USD | SKIP(EURUSD is the mirror image of the dollar break, so publishing both would be one fact told twice; the crypto legs are n=8 each and DEAD). |
| `P5b:rank21_extreme` bottom | DX-Y.NYB, UUP, USDSEK=X, HE=F | DRILL(DX-Y.NYB) into the same joint state. UUP is the same instrument SKIP(duplicate). USDSEK=X 172-143 and HE=F SKIP(no cell, and hogs are outside anything Scott reads). |
| `P6:two_atr_day` up | GC=F SI=F PL=F ZC=F CT=F BTC ETH (+TLT, DX-Y.NYB dropped by cap) | DRILL. Gold's +4.92% is 2.6 ATR and the engine's generic 2-ATR gold cell (n=138, +0.034%, t 0.34) is far too coarse for a move this size. The right cell is the magnitude tail, not the 2-ATR threshold. TLT was dropped by the cap and gets its own drill. |
| `P6:two_atr_day` down | KC=F | SKIP(coffee -9.7% is the day's largest single move but n=54, mean +0.348% at t 0.94 is nothing, and softs are outside the macro brief's remit). |
| `P9f:curve_flatten` | TLT, SPY | DRILL as the second leg of the TLT drill. Alone it is weak (TLT +0.159% on 67-54, t 1.31) and would not publish; crossed with a 2-ATR TLT session on a flat equity tape it is a different cell. |
| `P5b:rank21_extreme` top | GC=F, ETH-USD, ZC=F, EURUSD=X | DRILL(GC=F) as the other half of the dollar/gold joint state. EURUSD SKIP(duplicate of the dollar). ZC=F SKIP(the 08-12 and 08-17 briefs both published the corn breakout). |
| `P5:rank5_extreme` top | BTC-USD, ETH-USD, ZS=F, ZC=F | DRILL(ETH-USD, BTC-USD) once, cheaply. ETH +18.91% with a 5d return at the 100th percentile of its year is the largest number on the tape and crypto is explicitly in-universe as context. BTC-USD's engine cell is the sweep's only equity-like `solid` hint (n=224, +0.716% at t 2.53). Grains SKIP(covered, see above). |
| `P5:rank5_extreme` bottom | HE=F, GBPCHF=X, CHF=X, LE=F | SKIP(all four are the dollar-strength mirror in thin crosses; no cell survives its own control and none is a subject Scott reads). |
| `P4:z10_extreme` up | USDTRY=X, SB=F | SKIP(USDTRY n=583 at t 3.16 is a carry-currency drift artefact, not news; sugar has no cell at t 0.09). |
| `P4:z10_extreme` down | CAD=X | SKIP(n=108, +0.049% at t 0.92, no cell). |
| `P7b:down_streak` | ^FCHI, USDSEK=X, CAD=X | SKIP(France 78-51 at t 0.71 is the only one with a record and its mean is +0.13%; the 08-18 brief just published a down-streak item on ^BVSP and a second one is repetition of form). |
| `P1/P1b:new_52w_high` | ZC=F | SKIP(published 2026-08-12 and again 2026-08-17; repeat). |
| `P2 P3 P7 P9a-e P10 P11 P12` | — | did not fire. P12 in particular could not: zero US prints today. |

## Hints I am not inheriting

- `tag_hint` on the `E:opex` and `E:weekday_month` groups reads `suggestive`
  across the board on n in the hundreds. That is a floor. TLT's August-Thursday
  cell carries era_stable=False and cannot be tagged above `suggestive` no
  matter what its sign p says.
- `bh_pass` prices the sweep and matters for the four cells that carry it
  (`E:opex|NG=F`, `E:weekday_month|TLT`, `E:weekday_month|^TNX`,
  `P4:z10_extreme|USDTRY=X`). None of tonight's cells is a pre-specified famous
  hypothesis, so none is exempt: the opex-week and August-weekday cells were
  found by this sweep and the BH flag is the correction they owe. The dollar and
  gold items are state descriptions of what actually printed today rather than
  searched p-values, and are labelled as such in the brief.

## Drill list

1. `01_gold_thrust.py` — gold's +4.92% against its own magnitude tail, forward paths, era, concentration, control.
2. `02_dollar_break.py` — DXY 200d break + 2-ATR down day + bottom-percentile 21d, jointly.
3. `03_tlt_flat_tape.py` — TLT 2-ATR up on a flat equity session, plus the curve leg.
4. `04_natgas_opex.py` — the NG=F opex cell: is it opex, or is it the third week?
5. `05_august_thursday_rates.py` — ^TNX/IEF/TLT August Thursdays, era split, weekday cross, doy corroboration.
6. `06_crypto_thrust.py` — ETH +18.9% and the BTC 5d-rank cell.

## Post-drill verdicts (written after stage C, before composing)

Two DRILL verdicts turned into drops, and one skip turned into a drill. Recorded
here so the map matches what actually shipped.

- **`E:weekday_month` + `E:seasonal_doy` rates cross: DROPPED, not published.**
  `05_august_thursday_rates.py` killed it on the era test, which is the reason
  the engine set era_stable=False on both BH-passing cells. TLT on August
  Wednesday anchors is 52-19 up at +0.353% (t 2.97) pre-2018 and 15-22 DOWN at
  -0.272% after; ^TNX flips the same way, -0.467% then +0.854%, with the top-two
  episodes at 287% of the total. The honesty contract allows publishing a
  sign-flip as its own nugget, but the 2026-08-13 brief already published
  exactly that finding ("TLT, the best month duration has, and it stopped in
  2018") for August TLT, so a finer-grained retelling is repetition of
  substance even though the fingerprint differs. Worth knowing for future
  sweeps: the late-August duration seasonal will keep firing this sweep from
  three unrelated cell definitions and it is dead in all of them post-2018.
- **`E:jackson_hole` run-in: DRILLED (07), then DROPPED.** TLT gains +0.573%
  over the seven sessions into the opening on 18-6, t 2.24, sign p 0.0113
  against a +0.117% base, which looks publishable until you notice it is the
  same late-August duration seasonal as above wearing a different hat. Also the
  2026 anchor the script computed is spurious: the price history ends
  2026-08-19, so "the session 7 td before 2026-08-28" resolved to 2026-08-11
  rather than a real anchor. Calendar line only, as originally verdicted.
- **`P6:two_atr_day|GC=F`: the engine's cell was replaced, not inherited.** The
  2-ATR threshold is far too loose for a 2.6-ATR move; the drill priced the 4%
  magnitude tail instead (17 declustered episodes) and reports its 65%
  two-episode concentration in the brief.
- **`2026-01-28` verified, not discarded.** That bar shows SPY -0.01% and VIX
  +0.00%, which reads like a stale print. `07_consolidate.py` pulls the raw
  OHLCV: VIX closed 16.35 on both 01-27 and 01-28 and SPY traded 61.2m shares.
  It is a real flat session and stays in the ten-episode gold+silver set.

Shipped: 6 nuggets, 1 tomorrow-lane and 5 today-lane, 2 anecdotes (the tag
budget), headline on the gold magnitude. The tomorrow lane is thin because the
calendar genuinely is: opex on Friday and then nothing until Jackson Hole seven
sessions out, and the two cells that could have filled it were the repeats
above.
