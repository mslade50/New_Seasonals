# Cell map — run 2026-08-30 (Sunday)

Run date 2026-08-30. Asof session 2026-08-28 (Friday). Next session
2026-08-31 (Monday). All three differ; the run date names this folder.

Setup: Monday is trading day 21 of 21, the LAST session of August, a Monday,
in a midterm year. `prices_fresh=True`, core bar 2026-08-28. No top-tier event
on the next session; NFP is 5 td out. 1255 cells scanned, 88 fired, BH crit
p 0.0101 with 10 passing. No cells dropped by the cap.

Standing constraint from the journal: the last four briefs (08-24, 08-25,
08-26, 08-27) all led on an August equity calendar slot with a midterm split.
A fifth would be the countdown failure mode in everything but name, even
though the slot itself is new. The August-last-session equity cell has to earn
its place below a different lane, or not publish.

## Event lane

### E:month_end — final 3 sessions of the month (n_anchors 960)
The engine's cell is the last THREE sessions. Monday is the last ONE. Every
verdict here is contingent on re-cutting to the final session.

| subject | engine h1 | verdict |
|---|---|---|
| IEF | +0.076%, 58.8% hit, t 5.55, BH pass, era stable | **DRILL** — strongest cell in the whole sweep by t. Re-cut to the final session; cross with IEF sitting 0.73% off its 52w low |
| TLT | +0.123%, 57.1%, t 4.38, BH pass, era stable | **DRILL** — same re-cut, same cross (TLT 1.88% off its 52w low) |
| ^TNX | -0.291%, 41.7%, t -4.37, BH pass, era stable | **DRILL** — the yield side of the same claim, keeps the mechanism honest |
| HYG | +0.064%, 56.4%, t 2.54, BH pass | DRILL as a supporting leg only. Credit is not the index-extension story and HYG sits 0.23% off its 52w HIGH, the opposite corner from IEF/TLT |
| NG=F | +0.560%, t 3.13, era stable | SKIP(mechanism absent, and the seasonal_doy cell has Aug 31 nat gas at -1.07% over 25 years, which contradicts it. Two cells disagreeing at the same anchor is not publishable as either) |
| ^VIX | +0.579%, t 2.35, 48.3% hit | SKIP(mean up while the record is 464-489 down. Right-skew artifact, not a claim) |
| CL=F | +0.146%, t 1.73 | SKIP(sub-2 t, no cross) |
| EEM | +0.101%, t 1.78, era UNSTABLE | SKIP(era) |
| SPY / ^GSPC | +0.055% / +0.049%, t 1.54 / 1.36 | SKIP(no edge over control, 0.016pp both. Also the fifth-August-slot problem above) |
| QQQ | +0.014%, t 0.28, era unstable | SKIP(no mean) |
| IWM | +0.074%, t 1.61, era unstable | SKIP(era) |
| HG=F, GC=F, SI=F | t 0.02 / 0.38 / 0.40 | SKIP(no effect) |
| DX-Y.NYB, EURUSD=X, JPY=X | abs(t) <= 0.61 | SKIP(no effect) |

Pre-specification note: month-end bond extension is a documented,
pre-specified hypothesis (index duration extension at the month boundary),
not something this sweep discovered. It does not owe the BH correction, and
it passes anyway.

### E:weekday_month — Mondays in August (n 118)
| subject | engine h1 | verdict |
|---|---|---|
| QQQ | +0.130%, 64.4% hit, 76-42, sign p 0.0011, BH pass, t 1.05 | **DRILL** — a hit-rate cell with no mean is a specific and unusual shape, but it is confounded with month-end (some August Mondays are month-end) and with the August-slot fatigue above. Decompose or drop |
| ^VIX | +2.409%, t 2.37, era stable | **DRILL** — VIX up 2.4% on August Mondays while VIX sits 53.5% below its 52w high and VIX3M is AT a 52w low. Cross the calendar cell with the vol-floor state |
| ^TNX | -0.289%, t -1.51 | SKIP(same sign as the month-end cell and weaker; the month-end drill owns this claim) |
| everything else | abs(t) <= 1.56 | SKIP(no effect) |

### E:seasonal_doy — same trading day of year (+/-2), Aug 31
| subject | verdict |
|---|---|
| SPY / ^GSPC / QQQ / IWM midterm (n 6, -0.72 to -0.91%, 4-2 down, sign p 0.34) | SKIP(n 6 at sign p 0.34 is an anecdote with no record, it would be the fifth straight August-equity-midterm nugget, and the doy +/-2 window is a worse cut of the same tape than a clean last-session-of-August cut. If the equity slot publishes at all it publishes from drill 02, not from here) |
| SI=F all years (n 25, +0.367%, 18-7 up, sign p 0.0216) | **DRILL** — silver printed -3.37% Friday and sits 41.7% below its 52w high. Cheap to check whether the doy cell survives a clean last-session-of-August cut |
| EURUSD=X midterm (n 5, 5-0 up, sign p 0.0312) | SKIP(n 5, and a perfect record at n 5 with no mechanism is the exact shape the anecdote budget exists to ration. Not worth one of two slots) |
| NG=F (n 25, -1.067%, 15-10) | SKIP(sign p 0.21, contradicts the month-end cell, see above) |
| TLT | SKIP(published 2026-08-17, 9 td ago; the month-end drill is a different cell on the same subject and must not restate the doy number) |
| GC=F, HG=F, CL=F, HYG, ^TNX, ^VIX, EEM, DX-Y, JPY | SKIP(all sign p >= 0.10 in both the all-years and midterm cuts) |

## Price lane

### Ags complex — the roll trap first
ZW=F +5.49% to a 52w high, ZC=F +5.10% to a 52w high, ZS=F +2.49%, CC=F
+7.81%, PA=F +8.10%, KC=F -8.57%. Friday's brief ran a roll-signature check on
exactly this complex and tagged corn's bar as plumbing. **Nothing in the ags
or metals price lane gets a verdict until drill 04 re-runs that check on
Friday's bar.** Provisional verdicts, all conditional:

| cell | verdict |
|---|---|
| P4:z10_extreme ZW=F (n 189, h1 -0.104%, 74-112 down, sign p 0.0066, BH pass) | **DRILL** — the only signed, BH-passing mean-reversion cell in the ags. Publishable only if the bar is real |
| P7:up_streak ZW=F (n 122, h5 -0.937%) | **DRILL** — same drill, as the crossing (stretched AND on a streak AND at a 52w high) |
| P4:z10_extreme ZC=F | DEAD(repeat_blocked, published 2026-08-27, 1 td ago, and Friday's drill already called its bar a roll gap) |
| P5/P5b ZC=F, ZS=F, CT=F, SB=F | SKIP(all abs(t) <= 2.13 with sign p >= 0.24; and same roll suspicion) |
| P6:two_atr_day PA=F, ZC=F | SKIP(t <= 0.78) |
| P3/P3b SB=F (n 83 / 60) | SKIP(t -0.32 / -0.24, no effect either horizon) |
| P5/P6 KC=F (n 315 / 54) | SKIP(t 1.89 / 0.94. An 8.57% single-session drop is a striking number with no forward claim attached, and "coffee fell a lot" is not a nugget) |
| P5 LE=F, EURAUD=X | SKIP(t 0.14 / 1.75) |

### Crypto
| cell | verdict |
|---|---|
| P4:z10_extreme BTC-USD (n 298, +0.772%, t 3.12, BH pass, era stable) and P5b:rank21 BTC-USD (n 297, +0.740%, t 2.85, BH pass) | **DRILL** — two independent BH-passing continuation cells on the same subject, and the anchor session itself was -3.63%. The crossing (stretched up, but the anchor day sold off hard) is the question the base cell cannot answer |
| P5b:rank21 ETH-USD (n 183, t 1.34) | SKIP(t 1.34; folds into the BTC drill as a companion if it survives) |

### Rates and cross-asset
| cell | verdict |
|---|---|
| P1:new_52w_high ^FVX (n 24, t -0.55, era unstable) | **DRILL** as STATE not as cell. The forward stat is dead at n 24, but the 5y yield printing a 52w high going into the month-end bond bid is the cross that makes drill 01 specific |
| P9f:curve_flatten TLT (n 122, +0.151%, t 1.25) | **DRILL** — same drill, as the second conditioning variable |
| P9f:curve_flatten SPY (n 131, t -0.17) | SKIP(no effect) |
| P6:two_atr_day ^IRX up (n 157, -1.473%, 53-89, sign p 0.0551) | SKIP(^IRX percentage moves off a near-zero base are not a meaningful unit; the sign test is real but the magnitude is uninterpretable) |

### FX
| cell | verdict |
|---|---|
| P4 + P7 USDTRY=X (n 589 / 421, both BH pass, 69-71% hit) | SKIP(structural carry decay in a managed currency. Statistically the strongest thing in the sweep and completely uninformative) |
| P7 GBPCHF=X, JPY=X, P5 AUDNZD=X | SKIP(abs(t) <= 1.09) |
| P7:up_streak ^BVSP | SKIP(t -0.04) |

## Calendar entries inside the next 5 sessions
- **opex 2026-08-21** (td -5, past). SKIP(the post-expiration window was the 08-24 headline; the five-session window it named ends Friday and nothing new attaches)
- **jackson_hole 2026-08-28** (td 0, the session just closed). SKIP(led both the 08-26 and 08-27 briefs. The symposium-session outcome is today-lane material at best, and re-telling it a third time is the banned countdown shape)
- **nfp 2026-09-04** (td +5). SKIP for a nugget, PUBLISH in the calendar block. At 5 td out the anchor is not tonight's session, so an NFP cell would be a countdown, which is exactly what the novelty rule bans

## Selected for drilling
1. `01_month_end_bonds.py` — month-end bond bid, re-cut to the FINAL session, crossed with the 52w-low / 5y-high / curve-flattening state. Headline candidate
2. `02_aug_last_session.py` — last session of August for equities, clean cut, all years and midterm, with the September-first-session follow-on
3. `03_aug_monday_qqq.py` — decompose QQQ's 64.4% August-Monday hit rate against month-end overlap and against its own mean
4. `04_ags_roll_check.py` — is Friday's grain bar a roll gap. Gates every ags verdict
5. `05_btc_stretched_downday.py` — BTC continuation cells conditioned on the anchor session itself being down hard
6. `06_vix_august_monday.py` — VIX on August Mondays, crossed with the current vol floor
7. `07_silver_aug31.py` — silver's Aug 31 doy cell on a clean last-session cut

## Resolutions after the drills

A bug found and fixed in drills 01 and 02 before anything was read: both
initially treated August 2026 as a completed month, so Friday became a fake
"final session of the month" observation and landed in exactly the
low-proximity bucket the brief wanted to quote. Months now count only once a
later month has printed.

| drill | outcome |
|---|---|
| 01 | **PUBLISH x2.** IEF final session n=289, 180-109, +0.111%, t 4.84 vs a +0.014% baseline; grades 160-128 / 170-118 / 180-109 across the last three sessions; ^TNX 119-200 lower; SPY 152-167 in the same slot. Era stable, every 5y block positive, top-2 episodes 0% of total. Second nugget from the qualifier: August is the flat month (14-10, +0.045%) and the within-3%-of-52w-low bucket is the weakest (27-19, +0.036%), which is Monday's state. TLT is NOT the vehicle, t 4.10 pre-2018 to 0.78 after |
| 02 | **August equity slot DEAD as predicted.** ^GSPC last session of August 14-12, midterm 3-3, SPY midterm 2-4, concentration 106% in top two. No fifth August-slot nugget. The drill's live finding is the turn, handed to 02b |
| 02b | **PUBLISH.** Turn-of-month into September 13 of 26 up, -0.260%, against 186-107 and +0.223% for the other 293 turns; Welch t -1.73; both eras exactly 50%; the first September session is also worse than the rest of September (256-244) |
| 03 | **NOT WRITTEN, folded into 06.** QQQ's August-Monday cell was answered inside drill 06 Q1/Q2 rather than in its own file: 64.4% hit decays to 56.4% post-2018, which is exactly the other-month Monday control (56.4%), and the month-end-Monday sub-cell is n=3. SKIP(era-decayed to its own control) |
| 04 | **ALL AGS AND METALS KILLED, and the kill is the nugget.** ZC/ZW/ZS/SI/GC each carried identical volume on 08-26 and 08-27, then 1.9x to 1280x the 63d median on 08-28 with a gap. Corn's move is 82% gap, coffee's 96%, cotton gapped -30.65% on 1030x volume. Wheat's and corn's 52-week highs are the deferred contract's price. Publishes as one [anecdote] extending Thursday's corn finding to the complex, on the ZW fingerprint (not the repeat-blocked ZC one) |
| 05 | **PUBLISH the base cell, with the sub-arm reported unresolved.** BTC z10>=2: 170 of 298 up, +0.772%, t 3.12, era stable, top-2 20%. The arm matching Friday (anchor closed <= -3%) is n=15 at 8-7 and flat, declustered 6-6, while ETH's equivalent n=17 runs 12-5 the OTHER way with 65% concentration. The split does not carry signal at that size and the brief says so |
| 06 | **PUBLISH the VIX cell, SKIP the QQQ cell.** ^VIX August Mondays entered from the bottom third of its 52w range: n=84, 52-32, median +1.58%, t 3.19, sign p 0.019. Caveats carried into the brief: the unconditioned cell (t 2.37) did not clear tonight's BH line, the middle third is opposite-signed at 6-15, and 2018+ is only 19-17. The August-Monday-that-is-also-month-end sub-cell is n=3, DEAD by the n<5 convention, and is not quoted anywhere |
| 07 | **SILVER KILLED on two counts.** The Aug-final cell is 17-9 at +0.418% but the sign flips across 2018 (pre +0.639%/77.8%, post -0.080%/37.5%) and the top two episodes carry 60% of the total. Independently, Friday's SI=F bar is a roll (vol x1280, duplicated 104/104 volume before it), so the -3.37% is measured against a stale close |

Final slate: 6 nuggets, 1 anecdote, headline [solid]. Lanes are Tomorrow's
tape (4) and Friday in context (2), the asof session being Friday.
