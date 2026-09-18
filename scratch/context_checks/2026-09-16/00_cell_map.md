# Cell map — run 2026-09-16 (Wed), asof 2026-09-16, next session Thu 2026-09-17

Sweep: 1203 cells scanned, 111 fired (72 event / 39 price), BH 8 pass at crit p 0.0084. prices_fresh True
(core bar 2026-09-16). Midterm year. Stale: LBS=F, ^AXJO, ^HSI, ^KS11, ^MXX, ^N225, ^SKEW (no 09-16 bar).
Capped: P5 dropped ^FVX AUDUSD DXY UUP CAD MXN SGD ^IRX ^TNX IEF; P5b dropped ZC AUDNZD ^IRX CHFJPY NZDJPY.
The dropped dollar and rates names are covered by drills 03 and 04 rather than the cap.

## Tape verification (01_tape_verify.py)
- Roll seams, DEAD for any claim: HE=F -11.60% (gap -12.42%, intraday +0.94%), KC=F -6.64% (516x volume,
  gap -5.27%), CT=F +4.20% (opens at zero, 542x), SB=F +5.46% (gap +5.18%, intraday +0.26%).
- ES=F +0.43% / NQ=F +1.07% are the Dec contract roll (gaps +0.98% / +1.07%); cash is ^GSPC -0.45%,
  ^NDX +0.02%. Use cash only.
- ^TNX 5.006%, first close >= 5.00% since 2007-07-19; ^FVX 4.859% highest since 2023-10-25; ^IRX 3.97%
  highest since 2025-09-04. DXY +0.66% to 100.307, fifth straight up close. UUP +0.64%, GLD -0.61%.
- Timing (03): Yahoo FX pairs lag the US session. corr(JPY=X t+1, UUP t) +0.32 vs same-day +0.03; CHF=X
  +0.47 vs +0.21; EURUSD=X -0.49 vs -0.26. GC=F t+1 vs GLD t +0.12 (COMEX settles 13:30 ET). So any FX-pair
  or GC=F "day after a 14:00 event" record is a timing artifact, and today's FX-pair bars were pulled
  mid-bar at 17:10 ET.

## Calendar, next 5 sessions
| entry | verdict |
|---|---|
| fomc_decision 2026-09-16 (td 0, today) | DRILL -> 02, 03, 05, 08. The session after is a pre-specified cell (post-decision reversal / continuation), not a swept one. |
| vix_expiry 2026-09-16 (td 0) | SKIP(settled at today's open; no forward cell beyond the VIX drill 05) |
| opex 2026-09-18 (td 2) | see E:opex below |
| quad_witching 2026-09-18 (td 2) | see E:quad_witching below; Friday itself is owed to Thursday's brief when it is the next session |
| nfp 2026-10-02 (td 12) | SKIP(outside 5 sessions) |

## Event lane
| trigger | subject(s) | verdict |
|---|---|---|
| E:opex k2 | NG=F (129-182 down, sign p 0.0019, BH pass, era stable; published 08-19, 19 td, not blocked) | DRILL -> 06. Swept cell. Tomorrow IS the third Thursday. |
| E:opex k2 | ^VIX (139-181 down, not BH, era unstable) | SKIP(not BH, era-unstable, VIX covered by 05) |
| E:opex k2 | SI, ^TNX, EEM, IWM, SPY, HG, QQQ, ^GSPC, GC, TLT, JPY, EURUSD, HYG, DXY, IEF, CL | SKIP(|t| < 1.2 everywhere, hit rates 47-53%) |
| E:quad_witching k2 | ^VIX 45-61 down, h5 -2.0% | SKIP(sign p 0.072, not BH; h5 lands after Friday, owed to Thursday's brief) |
| E:quad_witching k2 | CL, QQQ, NG, HYG, EEM, HG, ^TNX, GC, EURUSD, IEF, SI, DXY, SPY, ^GSPC, IWM, JPY, TLT | SKIP(no cell above |t| 1, hit 47-56%) |
| E:weekday_month (Sep Thursdays) | JPY=X 71-40 up, BH pass | SKIP(FX pair bar lags the US session per 03; the weekday label is not the session it claims) |
| E:weekday_month | ^VIX 48-64 down sign p 0.078; all others | SKIP(not BH; bare weekday x month cell, pattern-only, rest |t| < 1.5) |
| E:seasonal_doy (Sep 17) | TLT midterm 5 of 5 down | SKIP(published 09-03 and before; N=5 cycle-year slice, anecdote budget better spent elsewhere) |
| E:seasonal_doy | SPY/^GSPC 18 of 26 up (sign p 0.038), HYG 14 of 19 | SKIP(mean +0.02-0.05% on a median-only record, swept, not BH) |
| E:seasonal_doy | QQQ, IWM, IEF, ^TNX, GC, SI, HG, CL, NG, DXY, EURUSD, JPY, ^VIX, EEM | SKIP(no all-years sign p below 0.05; midterm slices N=5-6) |

## Price lane
| trigger | subject(s) | verdict |
|---|---|---|
| P1 first 52w high | CHF=X (3-14 down, t -4.16, BH pass) | SKIP(today's CHF=X bar was pulled mid-bar and Yahoo FX bars lag the US session per 03; the 52w-high print itself may revise) |
| P2 / P2b first 52w low | HE=F | DEAD(roll seam, 01) |
| P4 z10 down | KC=F | DEAD(roll seam) |
| P4 z10 down | CHFJPY, NZDJPY | SKIP(yen crosses held back for repetition five briefs running; FX-bar timing) |
| P4 z10 up | ^IRX (132-92 up), ^FVX, AUDNZD | SKIP(rates surge published 09-13 and 09-15; drill 04 found the 10y-at-252d-high decision-day cell N=6 with IEF next day 1 of 6 up, too thin; AUDNZD flat) |
| P5 5d bottom 5% | HE=F, KC=F | DEAD(roll seams) |
| P5 5d bottom 5% | HG=F, EURUSD, NZDUSD | SKIP(|t| <= 1, FX-bar timing on the pairs) |
| P5 5d top 5% | JPY=X (150-198 down, BH pass), USDSEK, CHF=X | SKIP(FX pair bars in progress at pull and lagged per 03; the dollar state is carried by UUP in drill 03) |
| P5b 21d bottom 5% | KC=F | DEAD(roll seam) |
| P5b 21d bottom 5% | EURJPY (BH pass), AUDJPY (BH pass), GBPJPY, CADJPY | SKIP(yen crosses held back for repetition, five briefs) |
| P5b 21d bottom 5% | IWM (220-180 up, sign p 0.026), ^RUT | DRILL -> 08D. Result: with the 10-year up 25bp+ over the month, 12 declustered episodes, IWM h1 4 of 12 up, h21 6 of 11; nothing -> SKIP |
| P5b 21d top 5% | ^FVX | SKIP(5-year surge published 09-13) |
| P6 two-ATR day | HE=F (tag solid), KC=F | DEAD(roll seams; the HE "solid" tag is a seam artifact) |
| P6 two-ATR day | USDCNY | SKIP(49-48, nothing) |
| P7 up streak | USDSEK (51-86 down, t -2.63, solid, BH pass) | SKIP(today's +0.93% FX bar was pulled mid-bar, so the fifth up close is not settled; the dollar item carries the state on a 16:00 vehicle) |
| P7 up streak | USDTRY (BH pass) | DEAD(degenerate: structural lira depreciation drift) |
| P7 up streak | ^IRX, CAD, UUP, DXY, CHF | SKIP(DXY 84-83 and UUP 50-55 carry nothing; ^IRX rates repetition) |
| P7b down streak | IEF (60-46 up), PL, KC | SKIP(IEF bond lows published 09-10; PL flat; KC seam) |
| P8 200d cross down | GBPUSD | SKIP(10-11, nothing) |

## Drills beyond the base cells
- 02 post-decision session, 14 subjects: S&P-down x DXY >= +0.5% (17 decisions). JPY=X 16/17, CHF=X 17/17,
  GC=F 16/17 lower looked strong and are timing artifacts (03). DXY 15/17 up and synchronous with UUP (corr 0.94).
- 03 re-run on 16:00 vehicles: UUP 13 of 15 up (+0.30%, sign p 0.0037), GLD 12 of 16 down (-1.24%, sign p 0.038).
  Same state on non-decision days: UUP 157/338, GLD 195/376 up. -> PUBLISH (drill construction; splits
  run: S&P sign, DXY >= 0.5%, the conjunction, DXY up any; UUP thresholds 0.4/0.5/0.6% in 08 all hold).
- 02/08 S&P session after a down decision day: pre-2018 37/62 up, 2018+ 17/39 (median -0.13%) vs 239/447
  Wednesday-to-Thursday control. Pre-specified reversal hypothesis. -> PUBLISH as an era item.
- 05/08 VIX after a decision-day rise with S&P down < 1%: 30 of 44 lower, 13 of 17 since 2018; control
  161 of 270. Also scores Monday's construction: now 20 of 23. -> PUBLISH, framed as mostly ordinary mean
  reversion.
- 06 NG third Thursday: 179 of 305 down vs 527 of 1016 other Thursdays; 2022+ 28 of 53 down at -0.03%.
  -> PUBLISH with the fade stated.
- 04 stocks and bonds down together over 21d (SPY and IEF 21d rank <= 10): 15 declustered episodes, SPY h21
  10 of 14 up but 2018+ flat with 2022's three losers. -> SKIP(era flip, 2022-concentrated)
- 07 Dow <= -1% with NDX >= 0: 29 declustered, NDX h1 20 of 29 up but seven 2020 episodes; Dow is
  price-weighted and XLF -1.62% led. -> SKIP(2020 concentration, single-name risk in a price-weighted index)

## Multiplicity
The NG third-Thursday cell is swept and cleared BH. The post-decision S&P cell is the pre-specified reversal
hypothesis. The UUP/GLD and VIX items are drill constructions and quote their own controls; neither is a
swept p-value, and none is tagged solid.
