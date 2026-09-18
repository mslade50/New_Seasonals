# Cell map — run date 2026-09-13 (Sunday run)

asof session 2026-09-11 (Fri, CPI printed) | next session 2026-09-14 (Mon) | FOMC decision Wed 2026-09-16 | midterm year
prices_fresh = true, core bar 2026-09-11 | 1217 cells scanned, 112 fired (72 event / 40 price), BH crit p 0.0094, 11 pass

Friday's tape (drill 01): ^GSPC +0.86%, ^VIX -11.21% to 15.84, ^VVIX -11.09%, ^VIX3M -5.73%,
^FVX 4.791% (+5.8bp, 252d high, 5-session +28.2bp = largest 5d rise in its year), ^TNX 4.975% (252d high),
^IRX 3.913% (+1.77%), TLT +0.11%, IEF -0.19%, ^MOVE flat.

## Gates carried in

1. Continuous-contract roll seams (drill 01). Session volume vs trailing 20d median: GC=F 443x,
   KC=F 390x (gap -8.34%, i.e. the whole -9.85% is the gap), CT=F 756x with a -100% open,
   ZS=F 29x, PA=F prior median 0. Every price cell whose ENTRY STATE is one of those bars is dead.
   ZC=F 4.0x and its 21d window spans last week's roll (09-10 brief). Clean: CL=F 1.52x, NG=F 0.82x, all ETFs.
2. ^VIX phantom bars on 2026 closures (05-25, 06-19, 07-03, 09-07) re-confirmed. Every VIX drill
   here reindexes to the SPY calendar.
3. Repetition: pre-FOMC S&P drift (09-09), yen complex (09-02/03/06/07), CAC (09-09), bond three-way
   low (09-10), ^IRX front-end repricing (09-09), VVIX spike (09-10), crude (09-10).

## Event lane

| trigger | verdict |
|---|---|
| `E:fomc_decision` k3 (18 subjects, top tier) | see per-subject |
| `E:vix_expiry` k3 (18) | SKIP(overlap). Same date as the FOMC decision; SPY/QQQ/^GSPC BH passes are hit-rate only on a ~0.05-0.09% edge, generic up-drift. ^VIX +1.06% is the same Monday lift as below. Decision-on-expiry split checked in drill 05/06, parked (median and mean disagree). |
| `E:weekday_month` Mondays in September (18) | see per-subject |
| `E:seasonal_doy` Sep 14 (18) | SKIP. Midterm legs n=5-6: SPY h5 -0.88%, QQQ -1.15%, ^VIX h5 +8.8% 5 of 6. The midterm September pre-FOMC weakness published 09-09 and these six windows are the same weeks. NG=F `repeat_blocked` (09-08); TLT published 09-03. All-years SPY/IWM h1 19-7 / 19-6 are pre-specified-ish but carry a negative SPY mean (-0.09%); skip on slot. |

Per-subject inside `E:fomc_decision` k3 (h1 = Monday for a Wednesday decision):
- `^VIX` (n=212, +1.76%, t 3.57, BH pass) -> **DRILL -> PUBLISH** (drill 02/06). Against all days this is mostly the weekend lift: every Fri->Mon +1.86%, 58.8% up (n=1211). FOMC Mondays 105 of 151 up, +2.81%, vs other Mondays 57.3%, sign p 0.0013 on that base, edge +1.09pp t 1.69, both eras, top-2 12%. Friday->decision close only 80 of 151 up. Tag suggestive (edge t < 2.5). Construction, quotes its own control.
- `^TNX` / `IEF` / `TLT` (t ~1) -> **DRILL -> PUBLISH** as the yield-surge conditioning (drill 03/06/07). 5y 5-session bp-change rank >= 90 at the anchor: n=24, 5y higher h5 15 of 24 (+2.0bp) vs 82 of 188 (-2.1bp), p 0.049; ^GSPC to decision close 9 of 24 vs 114 of 188, p 0.019, but mean -0.04% vs +0.36% t -0.94; TLT 8 of 21, IEF 10 of 21. Non-FOMC control (195 declustered surges) h5 flat. Top 5% threshold n=9 same direction. Not the midterm confound: 4 of 14 up outside midterm years. Tag suggestive.
- `CL=F` (-0.53%, t -2.74, not BH) -> DRILL -> SKIP (drill 06). Crude Mondays run -0.26% anyway; FOMC Mondays -0.59%; trimmed -0.45%; swept p, fails BH, no mechanism.
- `NG=F` (sign p 0.006 BH, mean t -0.81, era unstable) -> SKIP(hit-rate only, era flips).
- `QQQ` (BH pass on 129-80 hit, mean +0.03%) -> SKIP(no magnitude; the drift itself published 09-09).
- `SPY` `^GSPC` `IWM` -> SKIP(pre-FOMC drift published 09-09).
- `HG=F` `SI=F` `GC=F` -> SKIP(roll seam in GC; t < 1).
- `HYG` `EEM` `EURUSD=X` `DX-Y.NYB` `JPY=X` -> SKIP(t < 1.5, no cell; yen repetition).

Per-subject inside `E:weekday_month` (Mondays in September):
- `^VIX` (+2.99%, t 3.36, BH pass, tag_hint solid) -> **DRILL -> SKIP as a nugget, footnote**. Weekend artifact: Sept Mondays +2.53% vs other Mondays +1.72%, diff t 0.93; February is higher. Downgraded from solid to dead-as-stated.
- `NG=F` (+1.31%, t 2.8, BH pass) -> **DRILL -> PUBLISH** (drill 06/07). Clean early-September Mondays (before the 22nd, so the Oct->Nov roll is excluded; no volume seam on the big ones): 42 of 63 up, +1.16%, vs 49.2% / +0.00% in other months, sign p 0.0038, t 2.42; Sept's other sessions +0.06%; 2018+ 13 of 19 but mean +0.46%; top-2 34%. Tomorrow is the 14th, not a roll day. Tag suggestive. One of 12 months, so multiplicity is real; the engine cell cleared BH.
- `TLT` `IEF` `GC=F` (hit 60%, sign p 0.03-0.05) -> SKIP(swept p, no BH, t < 1.6).
- equities / FX / crude / metals -> SKIP(t < 1.5, era unstable).

## Price lane

| trigger | verdict |
|---|---|
| `P3/P3b` ZS=F drop after 52w high | DEAD(roll seam, 29x volume) |
| `P4` stretched up: ^IRX ^FVX ^TNX | SKIP(no forward edge; the rates surge is carried by the FOMC conditioning above, non-FOMC control h5 flat) |
| `P4` stretched up: ^BVSP | SKIP(t 0.8; BVSP streak published 09-02) |
| `P4` stretched up: CL=F | SKIP(h1 +0.003%; crude published 09-10) |
| `P4` stretched down: CHFJPY CADJPY AUDJPY (+ dropped NZDJPY EURJPY GBPJPY IEF) | SKIP(yen repetition, four briefs; IEF covered by bonds 09-10) |
| `P5` bottom 5%: KC=F | DEAD(roll seam, 390x) |
| `P5` bottom 5%: CHFJPY NZDJPY | SKIP(yen) |
| `P5` top 5%: ^IRX ^FVX ^TNX | SKIP(rates, see above; ^IRX sign p 0.99 is a level-near-zero artifact) |
| `P5` top 5%: GBPCHF EURCHF | SKIP(t < 1, no mechanism) |
| `P5b` top 5%: ^IRX ^FVX | SKIP(same) |
| `P5b` top 5%: ZC=F ZS=F | DEAD(roll windows) |
| `P5b` bottom 5%: ^FCHI (BH pass) | SKIP(repetition, CAC published 09-09; 57% hit on +0.17%) |
| `P5b` bottom 5%: AUDJPY (BH pass) CHFJPY CADJPY | SKIP(yen) |
| `P6` down: ^VVIX | DRILL-adjacent -> SKIP(h1 46-54, era unstable; VVIX published 09-10) |
| `P6` down: KC=F | DEAD(roll seam) |
| `P6` down: USDCNY=X | SKIP(managed rate, ATR not meaningful) |
| `P6` up: ^IRX | SKIP(rates) |
| (not in the sweep) ^VIX -11.21% Friday | **DRILL -> PUBLISH** (drill 02/07). Friday VIX <= -10% -> Monday: pre-2018 20 of 34 up vs 60.4% base (ordinary); 2018+ 13 of 34 vs 55.5%, median -1.22%, p 0.033, monotone in drop size (6%: 51.6%, 8%: 41.4%). Single-era -> suggestive. FOMC-week intersection since 2018 splits 2-2. |
| (not in the sweep) CPI-day VIX crush durability | DRILL -> SKIP (drill 05). -8%: h5 lower 22 of 36 vs 56% non-event, t -1.06, top-2 70%. |
| (not in the sweep) S&P rally with 5y +5bp | DRILL -> SKIP (drill 04). h21 lag vs yields-down rallies is pre-2018 only (2018+ +1.91% vs +1.88%). |
| `P7` up streak: ^IRX ^FVX ^TNX | SKIP(rates) |
| `P7` up streak: USDTRY=X | DEAD(degenerate: structural depreciation) |
| `P7b` down streak: IEF HYG NZDUSD | SKIP(IEF bonds 09-10; HYG h1 flat era unstable; NZDUSD null) |
| `P8` 200d cross down: AUDJPY | SKIP(yen) |
| `P9f` curve flatten: TLT SPY | SKIP(t 1.2 / null; folded into the rates context) |

## Calendar, next five sessions

| date | entry | verdict |
|---|---|---|
| Mon 09-14 | Monday in September; 2 sessions before FOMC | PUBLISH (VIX FOMC Monday, NG September Monday) |
| Tue 09-15 | nothing on the tracked calendar | SKIP |
| Wed 09-16 | FOMC decision 14:00 ET + VIX expiry at the open | PUBLISH via the yield-surge nugget; decision-on-expiry split parked (quarterly on-expiry decision-day VIX lower 30 of 41, median -3.91%, but mean -1.50% vs -2.31% for other quarterly meetings) |
| Thu 09-17 | nothing tracked | SKIP |
| Fri 09-18 | opex + quad witching | SKIP(5 td ahead, outside k 1..3; calendar line only) |

## Selection

Tomorrow: ^VIX FOMC Monday (headline), ^FVX yield surge into the decision, NG=F early-September Mondays.
Today: ^VIX Friday collapse -> Monday, 2018+.
No anecdotes. Pre-specified: none of the four is a raw sweep cell; each is a construction quoting its own control.
