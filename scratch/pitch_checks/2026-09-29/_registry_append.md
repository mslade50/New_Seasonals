
## 2026-09-29: the metals break on the rates-shock tape, eight candidates, all empty

Eight candidates over five novelty axes and seven asset classes, three checkers,
stand-down. The 09-28 tape: GLD -3.94%, SLV -5.49%, GDX -5.36% (the first
complex-wide break, watchlist 28 out-of-sample firing 1 of 10), ^TNX +1.08% to a 252
high of 5.24%, UUP at its 252 high, SPY -0.74%, ^VIX +8.07% to 16.1, ^MOVE +6.06%,
HYG -0.41% on 2.78x volume a day before quarter-end, TLT -0.88% on 2.0x into a fresh
252 low. All scripts are in scratch/pitch_checks/2026-09-29/.

- **Long gold after a one-day crash of >= 3.5%, with yields and the dollar at highs.**
  The 2013 inversion shows up in its own window: with ^TNX near its 252 high the crash
  day pays -1.429% on 1-4 at h=5 (GC=F 1-3), against +0.876% on 22-16 bare. The 2 ATR
  form is flat (33-33 at h=5). Below the 200d and >= 15% off the high, the crash day
  loses to the same drawdown without one (+0.214% on 5-7 against +0.468% on 84-73);
  86% is re-anchoring (reanchor p 0.43). Joins the gold-at-a-yield-high family (lines
  1885, 4612, 4845). (kA_c1_gold_crash.py, kA_c1b)
- **Long gold from a first complex break into the payrolls close.** The crash strips
  gold's ungated NFP run-in (+0.237% on 151-109) to -0.300% on 8-7 for breaks 2-4
  sessions before the print, 0.17pp under the same trigger with no print. The placebo
  ladder by distance to NFP is flat (-0.136% to -0.382%). (kA_c7_gold_nfp.py)
- **Short SLV against beta-GLD after a first complex break.** It is W28's GLD-beta
  residual in pair clothing: the silver leg is 150% of the h=1 pair (+0.412% on 61-40)
  and the gold leg loses. When silver fell LESS than its beta on the break day (the
  09-28 case, 85th pctile), h=3 pays +0.136% against +1.113%: read the break-day
  residual before scoring any W28 firing. (kA_c2_slv_gld_pair.py, kA_c2b)
- **Long HYG after a >= 2.5x-volume down day, at the month turn or any date.** At
  ME-3..-1 the spike subtracts (+0.441% on 3-1 against +1.004% on 12-3 for the 5d
  washout alone). The any-date parent (+0.800% on 37-16 at h=5) is 2018+ -0.443% on
  6-8 with 2008 at 44% of the total, and the duration-driven form (IEF r5 <= 20) pays
  -1.249% over 10. HYG has no pre-quarter-end volume bulge (0.97-1.00x at QE-3..-1);
  its month-end bump lands on the ME close itself. (kC_c3_hyg_volume.py, kC_c3b, kC_c3c)
- **Long SPY from the midterm Q3-end with the index within 2% of its high.** The
  famous rally comes off a midterm low: near-high midterm Octobers +0.05% on 4-4 (1950+
  monthly, French market series) against +4.74% on 9-2 for those > 2% off the high;
  daily 2000+ near-high midterms 1-2 at -3.26% (QE-1, h=10); 2013+ midterm Q4 -0.97%
  on 2-1. The live cycle split is odd against even years (+1.84% on 11-2 against -1.43%
  on 6-7), the same finding as the 09-28 vol kill. ^GSPC in the cache starts
  2000-01-03. (kC_c6_midterm_q4.py)
- **EWZ into and across the Brazilian first round (uncertainty resolution).** The
  record is strong (^BVSP run-in 6-0, EWZ-EEM across +7.41% on 5-0) and the mechanism
  is wrong: the run-in is positive, the six runoffs show nothing (pair +0.004%, reaction
  2-4 at -1.89%), and municipal years run in 1-4 at -2.5%. The big Mondays (2014, 2018,
  2022) are first rounds where the market-favoured challenger beat the polls; 2002 was
  -6.0%. Any Brazil-vote trade needs a poll-miss side, which the repo lacks.
  (kB_c4_ewz_vote.py, kB_c4b)
- **Long USDJPY after the Japanese half-year book close (Mar+Sep).** +0.305% on 33-20
  QE-1..QE+5 against the post-QE dollar parent +0.228% (t 0.35); September alone
  +0.198%; USDJPY RISES into the closes (+0.235% on 35-18), the opposite of
  repatriation. Third dead Japanese book-close cell after EWJ (line 4937).
  (kB_c5_usdjpy_qe.py, kB_c5b)
- **Long crude into payrolls on a 63d thrust (USO r63 >= 75).** The gate adds +0.27pp
  at t 0.40 (+0.482% on 35-31 against +0.210% without a print); 70/80/90 thresholds all
  under t 0.9. The 21d neighbour is real and parked on the watchlist, but it dies when
  a month-end falls inside the hold (-0.397% on 8-6). (kB_c8_uso_nfp.py, kB_c8b-d)
- **Method note: two z10 conventions.** HYG z10 on 09-28 is -2.08 on
  `pitch_lab.zscore` and -1.63 on the tape builder's convention. Any watchlist arm
  written on "z10" must name which one it used.
