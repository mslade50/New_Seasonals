# Cell map — run 2026-09-15 (Tuesday)

asof session 2026-09-15 (Tue) -> next session 2026-09-16 (Wed). Midterm year, September.
prices_fresh = True (core bars 2026-09-15). Sweep: 1253 scanned, 145 fired (108 event / 37 price),
BH pass 5 at crit p 0.0038 (VIX k1, EEM k1, QQQ k1, HYG k1 on the decision; USDTRY up-streak).
Caps dropped NZDJPY (P4) and CADJPY/AUDJPY/EURJPY/GBPJPY/AUDNZD/IEF/NZDJPY (P5b): yen complex and IEF,
both already represented by kept subjects, nothing unique lost.
Warnings: 6 tape tickers without a 09-15 bar (LBS=F, ^AXJO, ^HSI, ^KS11, ^N225, ^SKEW); none is a
nugget subject tonight.

Tape tonight: S&P -0.45%, SPY 21d rank 7.1; QQQ -0.65%; IWM -0.96%, 21d -6.54% (rank 2.4). VIX +0.58% to
17.20 after a +2.75% open (the eve of last night's headline cell). 10y 4.996%, highest close since
2007-07-19 (above 2023-10-19's 4.988); 5y 4.826% (+31.9bp in 10); ^IRX 3.960% (+22.8bp in 10, rank 99.6).
IEF/TLT/LQD at 252d lows; HYG seven straight down closes, z10 -2.27. Crude 105.48 (+4.03%, +24.8% in 21).
EEM -0.35% after Monday's -2.73%, 63d rank 0.4.

Tape verification (01_tape_verify.py): coffee 818x volume with a -6.46% gap, cotton opened at 0 with 952x
volume, soybeans 31x, wheat 13x (Monday bar flat on zero volume), corn 5.6x with a +4.15% gap the session
after September grain expiry, hogs -12.59% of which -10.93% is the gap on 1.45x volume. All treated as
contract-roll seams: every KC/CT/ZC/ZS/ZW/HE cell is DEAD tonight. SPY/QQQ/IWM/bonds/HYG bars are clean.
^VIX phantom bars persist (2026-05-25, 06-19, 07-03, 09-07): VIX drills run on the NYSE calendar.

Repetition ledger (journal + last six briefs): pre-FOMC k4 drift by cycle year (09-09), 5y top-decile surge
into the decision with the S&P decision-close record from k3 (09-13), VIX Monday-before (09-13), VIX k2 7%
lift to the decision close (09-14 headline), TLT eve bid (09-14), SPY VIX-expiry eve (09-14), EEM/Korea
(09-14), yen crosses (five briefs), bond three-way low (09-10), 10y at a 52w high (09-06, 09-10), crude +8%
session (09-10), small caps washed out with the index near highs (09-09).

## Calendar, next 5 sessions
| date | entry | verdict |
|---|---|---|
| Wed 09-16 | FOMC decision 14:00 ET (top tier, k1 tonight), SEP asterisk in macro_events | DRILL (02 EEM, 03 HYG/IWM, 05 bills + S&P, 06 S&P era + SEP split) |
| Wed 09-16 | VIX expiry, 09:30 settlement (k1) | DRILL folded into 04 (VIX follow-on on the decision-day close); the expiry-eve SPY cell is SKIP(published 09-14) |
| Thu 09-17 | nothing scheduled | calendar line only |
| Fri 09-18 | monthly opex + quad witching (k3) | SKIP(base cells null for equities, |t| < 1.4; the post-opex September week belongs to next week's briefs) |
| Mon 09-21 | nothing scheduled | calendar line only |
| Tue 09-22 | nothing scheduled | calendar line only |

## Event lane
| trigger | subjects / cell | verdict |
|---|---|---|
| E:fomc_decision k1 | ^GSPC +0.222% t 2.65, SPY +0.228% t 2.74 (solid hints, engine era_stable by sign) | DRILL 05/06. PRE-SPECIFIED famous hypothesis (announcement-day drift), exempt from BH. The worthless base number; the product is the era split: 82/144 up before 2018, 29/68 since. |
| E:fomc_decision k1 | EEM +0.533% t 3.97, 117-68, BH pass, era-stable | DRILL 02: SWEPT cell that cleared BH, so [solid] is available. Mechanism check vs the dollar and conditioning on tonight's 10y week and EEM's weak week. |
| E:fomc_decision k1 | HYG +0.26% t 4.00, 94-59, BH pass | DRILL 03: concentration (Dec 2008 / 2020) and conditioning on HYG's z10; SWEPT, cleared BH. |
| E:fomc_decision k1 | IWM +0.298% t 2.60 | DRILL 03: 84/140 up before 2018, 33/68 since, decision-day bid gone; fold into the footnote, small caps vs index published 09-09. |
| E:fomc_decision k1 | QQQ +0.295% t 2.56 BH pass | fold into 06 as the S&P item's contrast (38/68 up since 2018). |
| E:fomc_decision k1 | ^VIX -1.49%, 75-133 down, BH pass, era-unstable | DRILL 04 as the escalation re-telling of the 09-14 headline with new specificity (eve direction split). Base cell alone SKIP(era-unstable mean, Dec 2024 +74%). |
| E:fomc_decision k1 | DX-Y.NYB -0.12% t -3.13 (solid hint) | fold into 02 as EEM's mechanism (DX fell on 109 of 186 EEM decisions); alone SKIP(swept, fails BH at sign p 0.032, 12bp move). |
| E:fomc_decision k1 | EURUSD +0.10% t 2.18 era-unstable; TLT, IEF, ^TNX, CL, NG, SI, HG, JPY, GC | SKIP(no |t| above 2.2 on a BH-failing swept cell; TLT/IEF eve published 09-14; bills/10y through the decision checked in 05 and dropped) |
| E:vix_expiry k1 | SI=F 177-135 sign p 0.010, GC=F t 2.07, EURUSD t 2.31, SPY -0.10%, VIX | SKIP(metals carry roll seams all week; every expiry anchor sits on the FOMC k1 session tonight, so equity/VIX cells duplicate the decision cells; none passes BH) |
| E:opex k3 | TLT +0.147% t 2.55 (solid hint), SI=F sign p 0.010, ^TNX -0.24% | SKIP(swept, fails BH; TLT's k3-opex session is the FOMC eve on FOMC-opex weeks and the eve bid was published 09-14; silver roll seams) |
| E:quad_witching k3 | ^VIX 43-62 sign p 0.049, h5 -2.30%; QQQ -0.21% t -1.35 | SKIP(swept, fails BH, VIX era-unstable and the quarterly FOMC-opex overlap makes it the decision-week vol crush again) |
| E:weekday_month | Wednesdays in September: ^VIX 42-69 down sign p 0.0066, CL=F +0.593% t 2.59 (solid hint), NG h5 +2.9% | SKIP(bare weekday x month; September Wednesdays are FOMC and VIX-expiry days so the VIX leg is the decision crush in disguise; crude sign p 0.064 fails BH and crude was published 09-08/09-10) |
| E:seasonal_doy | EURUSD midterm h1 5-0 up (sign p 0.031), DX h5 18-8 (0.038), EEM h5 16-7 (0.047), HG h1 17-7 (0.054) | SKIP(one of ~72 swept day-of-year records each, none near BH, no mechanism; FX DOY records flipped under this lens on 09-14) |
| E:seasonal_doy | TLT (published 09-03), NG=F (published 09-08) | SKIP(novelty flags: repeats with no new specificity) |
| E:seasonal_doy | SPY/QQQ/IWM/^GSPC midterm h5 -0.7% to -1.3%, 2-4 of 6 | DEAD(n 6, records 4-2 at best) |

## Price lane
| trigger | subject | verdict |
|---|---|---|
| P1 first 52w high 30+d | USDSEK n 23, 7-16 down, sign p 0.047 | SKIP(swept FX minor, fails BH, no mechanism, and it is the same dollar week as the rates move) |
| P2/P2b first 52w low | HE=F | DEAD(roll seam: -10.93% gap on 1.45x volume) |
| P4 z10 stretched up | ^IRX n 242, ^FVX n 187 | SKIP(percent-of-yield basis is degenerate in ZIRP years; checked in bp in 05: bills +20bp in 10 into a decision, S&P decision day 5 of 10, 2008 and 2022 are the sample) |
| P4 z10 stretched up | CL=F n 176, h1 +0.03% | SKIP(null cell; crude published 09-10) |
| P4 z10 stretched down | HYG n 100, +0.33% t 2.46, 60-37 | DRILL 03 into the decision: z10 <= -2 5 of 5, <= -1.5 12 of 14. Anecdote tier. |
| P4 z10 stretched down | CHFJPY, AUDJPY, EURJPY, GBPJPY | SKIP(yen complex, five briefs; no pair above t 1.3) |
| P5 5d bottom 5% | HE=F, KC=F | DEAD(roll seams) |
| P5 5d bottom 5% | HYG (null, era-unstable), IEF +0.063% t 2.07 | SKIP(HYG covered by the z10 drill; IEF swept fails BH and the bond lows were published 09-10) |
| P5 5d top 5% | ^IRX, ^FVX, ^TNX | SKIP(yield percent basis; the 5y surge was 09-13's item) |
| P5 5d top 5% | USDMXN n 325, -0.02% | SKIP(null) |
| P5b 21d bottom 5% | IWM/^RUT 220-179 up sign p 0.023, +0.12% t 0.87 | SKIP(hit rate without a mean; small caps washed out published 09-09; decision-day IWM checked in 03) |
| P5b 21d bottom 5% | HE=F | DEAD(roll seam) |
| P5b 21d bottom 5% | CHFJPY 192-153 sign p 0.020 | SKIP(yen repetition, overlapping-day hit rate) |
| P5b 21d top 5% | ^IRX, ^FVX | SKIP(yield percent basis) |
| P5b 21d top 5% | ZC=F +0.195% t 2.12, ZS=F | DEAD(roll seams on the session after September expiry) |
| P6 >= 2 ATR down | HE=F n 116 -0.48% t -2.83 (solid hint), KC=F | DEAD(both roll seams; the solid hint is a seam artifact) |
| P6 >= 2 ATR up | ZC=F, CT=F | DEAD(roll seams; cotton opened at zero) |
| P7 5+ up closes | ^IRX (yield basis), CHF=X null | SKIP |
| P7 5+ up closes | USDTRY 300-124 up, BH pass, era-unstable | SKIP(managed-depreciation drift, the up-streak is the currency's standing trend; macro-irrelevant to tomorrow) |
| P7b 5+ down closes | IEF 60-45, LQD 55-35 sign p 0.038 | SKIP(swept, fails BH; bond lows published 09-10) |
| P7b 5+ down closes | HYG 60-64 null | folded into the HYG z10 drill (seven straight down closes is the state description) |

## Selection plan
Tomorrow: ^GSPC decision-day era split (headline, pre-specified, [suggestive]); EEM decision day and the
dollar ([solid], swept + BH pass); ^VIX eve-direction follow-on ([anecdote], escalation re-telling).
Today: HYG seventh down close and z10 into the decision ([anecdote]), carrying the 10y 19-year-high fact.
Anecdotes: 2 (budget 2), neither the headline.
