
## 2026-09-25: the second leg of the rates shock, eight unopened cells, all empty

Eight candidates over five novelty axes and seven asset classes, three checkers,
stand-down. The 09-24 tape: ^TNX +4.8 bp to 5.162% at its 252 high, TLT/IEF/LQD/TIP
all exactly at 252 lows (TLT -1.29% on 1.94x), ^MOVE +9.57% (97.4th pctile of daily
moves, level at the 98.4th pctile of its year) with ^VIX 15.67 (19th), UUP at its 252
high, USDMXN +1.48%, copper 1.25% under its high while gold is -7.6% in 21d, UNG
+6.26% on 5.55x two sessions after the 09-23 pitch's rule fired. All scripts in
scratch/pitch_checks/2026-09-25/.

### Method traps

- **A ratio that moves on its denominator is not the signal its numerator names.**
  Copper/gold at its 96.8th level pctile read as a growth-and-inflation
  confirmation of the yield breakout, but 81% of the 21d move was gold falling
  (HG +1.86%, GC -7.94%), and 12 of the 13 historical episodes were copper-led.
  Before using any ratio as a conditioner, split its move into the two legs and
  check the live split matches the sample's. Same family as the 2026-08-10 MOVE/VIX
  denominator trap. (kB_c1_cugold_tlt.py, _b)
- **A pitched rule's RE-FIRE is a different population from its first firing.**
  The 09-23 UNG volume-thrust rule's 26 episodes split into 16 first firings (+4.26%,
  14-2) and 10 re-fires within five sessions (+0.84%, 6-4), and inside the re-fire
  state the volume gate no longer separates. When a pitched rule fires again inside
  its own hold, score the re-fire population, never the headline. (kA_n1_ung_refire.py, _b)

### Kills

- **Long UNG h=2 on a re-fire of the >= +5% / >= 3x volume thrust.** Re-fires +0.84%
  on 6-4 (sign p 0.377); +5% days without the volume inside the re-fire state +1.41%;
  drop-best-1 -0.51%; h=3..5 -0.92% to -1.12%; re-fires inside the prior hold -7.02%
  at h=10 (2-5), an exhaustion signature. The no-Thursday slice (6-1, +3.66%) is a
  post-hoc split of a post-hoc split, parked as a scored out-of-sample entry only.
  (kA_n1_ung_refire.py, _b)
- **Long equity vol (short SVXY residual, short SPY) with ^MOVE level >= 95th pctile
  and ^VIX <= 30th.** The VIX FALLS afterwards (-2.26% at h=5 against +3.66% for calm
  VIX alone), short SPY 2-7 at h=5 (-0.714%). Third independent confirmation that a
  bond-vol extreme is followed by calmer equity vol. The long-SPY flip belongs to the
  MOVE-level parent (+0.812% at h=5, 33-17), and 47 of that parent's 50 episodes had
  VIX above 30: a crisis-rebound effect, not a calm-tape one. (kA_v1_move_level_calm_vix.py)
- **Long SVXY residual across the quarter turn, QE-3 to QE+1.** The first test of the
  month/quarter-end anchor on volatility, and it closes the anchor on the last class:
  34 quarter turns since 2018-03, residual +0.081% on 18-16 (1.2x cost); VIX/VIX3M does
  not soften across the turn (+0.0023 vs -0.0001); September QEs -0.747% on 3-5; pre-
  2018 synthetic -0.5x wrong-signed; 6 of 45 on the placebo ladder, 9 of 45 positive.
  Ordinary month-ends run the residual -0.745% (28-40) because SPY rallies and SVXY
  lags. **The month-end anchor is now closed on equities, rates, FX, commodities and
  volatility.** (kA_q1_svxy_qturn.py, _b)
- **Short TLT when copper/gold (21d rank >= 90) confirms a ^TNX 252 high.** 13
  episodes 2003-2022; h=5 3-10, h=10 +0.681% with top two 86%; over the TLT-at-a-low
  parent the ratio gate adds +0.10pp; the ratio as a trade (long HG / 0.534 GC) is
  pre-2018 +1.78% against 2018+ -0.64% (1-4). (kB_c1_cugold_tlt.py, _b)
- **Long copper miners against beta-HG after a >= 8pp 21d lag with copper within 3% of
  its high.** First miner-vs-metal cell outside gold. FCX: the lag with copper NOT near
  its high +1.292% at h=5 (170) against the cell's +0.640%, cell h=10 -0.609% (17-21),
  residual net of HG and SPY -1.082%; -6/-10/-12pp neighbours negative. COPX has 4
  episodes, all in the 2025-26 COMEX tariff-premium era. SCCO, TECK, XME and GDX/GLD
  carry nothing: the label adds nothing. Byproduct, not parked: FCX lag at ANY copper
  level +1.106% at h=5 (187, p 0.029) is -0.032% on COPX, the tradeable vehicle.
  (kB_m1_copx_hg.py, kB_zz_byproducts.py)
- **Long MXN after a carry-unwind session (USDMXN >= +1.25% with MOVE or VIX up).**
  First carry-currency cell in the repo. 122 episodes, h=2 +0.130% on 62-60, top two
  (2020-03-23, 2008-11-19) 61%; the edge is flat from h=1 to h=10 (+0.03 to +0.19pp)
  when carry reassertion needs it to build; the vol gate subtracts at h=5; BRL, AUDJPY
  and CADJPY wrong-signed under the same rule. NFP inside an h=5 hold -0.336% (26).
  The >= +1.50% rung at h=2 is parked (charged for ~48 cells). (kC_x1_mxn_carry.py, _b, _c)
- **Long FXI from QE-3 across Golden Week (the flip of the 09-18 short).** Raw +1.098%
  on 12-9 vs own drift +0.323%; EEM residual ex-2024 +0.31% (10-10), 2015+ ex-2024
  +0.01% (5-5) where the southbound-suspension story needs 2015+ stronger; placebo
  rank 10 of 21; March and December quarter-ends pay the same shape. Golden Week is
  closed in both signs. (kC_i1_fxi_gw.py, _b)
- **Long a turned 63d laggard (r63 <= 5, r5 >= 70) into its print, k=-3 (live MU,
  AMAT).** 162 liquid names, +0.056% SPY-hedged on 57-50 against the ungated print
  premium +0.164% (637-538) and the same state with no print +0.017%. The already-
  turned leg takes 0.35-0.40pp off the r63 floor alone (+0.402%), which is the closed
  laggard lane. Semis in the state 2-7; MU in the state 1-2. Note: the earnings
  calendar's timing column is 99% empty, so print-day exits cannot distinguish BMO
  from AMC. (kC_e1_panel.py, kC_e1_mu_print.py)
