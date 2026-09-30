
## 2026-09-28: the rates shock at the quarter turn, twelve candidates in two waves, all empty

Twelve candidates over six novelty axes and ten asset classes, four checkers,
stand-down. The 09-25 tape: ^TNX at its 252 high (63d rank 100), TLT at a fresh 252
low on 2.0x volume on a flat day, ^MOVE +38% in 21d after a +21.5% day on 09-23 and
-8.2% on 09-25, ^VIX 14.9 with its 21d range compressed 22 sessions, SPY 0.59% under
its high and QQQ 0.40%, IWM 63d rank 0.4 (QQQ minus IWM +10.2pp over 21d), UUP z10
2.00 near its high, GLD 20.7% and SLV 44.9% under their highs, USO +16.5% over 21d
with a third -3% day in eight sessions. All scripts are in scratch/pitch_checks/2026-09-28/.

- **Long IWM against short QQQ after a record 21d size spread (trailing-252 rank >= 99.5).**
  h=5 +0.062% on 26-22 is 1.0x a 6 bp pair; 2018+ excess -0.045pp (7-5) at h=5 and
  -0.105pp (4-5) at h=10. The QQQ-within-1%-of-its-high half pays the other side, 4-10
  at -0.491%. The absolute >= 10pp neighbour is a high-vol-tape cell (2001, 2002, 2008,
  2020). Beating "QQQ strong" alone proves nothing, since that control is momentum.
  (a1_size_spread.py, a1b)
- **Long IWM across the quarter turn gated on IWM r63 <= 5.** The gate subtracts:
  +0.063% at h=4 against +0.363% ungated over 105 quarters, the same gate at ordinary
  month-ends beats quarter-ends, and the IWM-minus-SPY residual is negative at every
  horizon, so there is no small-cap reversal. The h=5 number is 2001-09 and 2015-09
  (drop-best-2 -0.98%). (a8_q4_turn.py)
- **Nearest-neighbour analogue on seven cross-asset features (fifth dead analogue lane).**
  Largest edge t 0.97 over 10 proxies by 2 horizons, random-date P 0.982, and a
  150-anchor reference class beats today's grid 96% of the time. Dropping one feature
  keeps only 6-9 of 20 neighbours. (a7_knn.py, a7b, a7c)
- **NFP run-in (k=-4) short DX and long GLD with DX 21d rank >= 85 and ^TNX at its 252
  high.** The crowded-dollar leg is wrong-signed alone (short DX -0.083% on 24-26,
  midterm -0.545% on 5-10) and strips GLD's ungated NFP run-in (+0.290% on 150-111 to
  -0.082%). The TNX-high leg alone is parked (see the watchlist). (b2_nfp_runin.py, b2b)
- **Short equity vol with ^MOVE's 21d rank >= 95 while VIX is calm (21d-return form).**
  SVB 2023-03 and 2018-01 are the trade; without them the MOVE gate is worth -1.00pp on
  the SVXY residual and -1.59pp on ^VIX. This is the fourth form (level, 5d rank, spike,
  21d rank) to show that a bond-vol extreme is followed by calmer equity vol.
  (b3_move_vix.py, b3b)
- **Credit not confirming an index high (HYG 5d rank <= 5, LQD at its 252 low, SPY
  within 1% of its high).** One precedent (2026-09-25 itself). Without the LQD leg it is
  kC_c1, and the duration-driven half pays short SPY 0.00% on 2-5 at h=5.
  (b4_credit_nonconfirm.py, b4b)
- **Long SLV against beta-GLD after a >= 35% silver drawdown with the ratio at a 252
  extreme.** The ratio gate subtracts (-0.007% on 20-26 against +0.267% for the
  drawdown alone), and 45 of 46 episodes are pre-2018. (c5_slv_gld.py)
- **Long GLD >= 15% under its 252 high with DX 21d rank >= 85.** It passes round 1 (55
  episodes, +1.132% at h=5, 33-22), but the dollar gate does not filter (-0.43pp, 72%
  re-anchoring, reanchor p 0.015). Below GLD's 200d, the live regime, 2018+ is 7-7 at
  +0.013pp. The 7-0 above-200d half is post-hoc. (c5b_gld_dx.py, c5c, c5d)
- **Short crude after a -3% day inside a 21d thrust.** Wrong-signed in all 28 cells (USO
  6-18 at -1.413%, CL=F 7-17 at -1.749% at h=3), and the thrust gate makes it worse than
  a plain -3% day (+0.276% short). This is the second confirmation after the 09-23
  round-trip kill. (c6_crude_crack.py)
- **Long XLE against short USO on a 21d crude-over-equity gap.** The dose runs backwards
  (>= 20pp -1.560%), and the 63d-state intersection pays -1.176% on 5-10. With SPY above
  its 200d it is 10-17. It is a lookback neighbour of b5_xle_uso_divergence (44% shared
  trigger days). (c9_xle_uso21.py)
- **Long TLT/IEF on the first >= 5% ^MOVE fall inside five sessions of a top-3% MOVE
  rise (the vol-targeting re-lever story, written down before the check).** All 8
  vehicle-horizon cells trail drift. Entering on the crush filters -0.149pp against
  entering on the spike day, which itself pays +0.207% at h=5. Top-1% spikes pay -0.550%
  on 14-22. (d10_move_crush.py)
- **Long equity vol from the September QE-2 close into the midterm October.** Presidential
  years pay more (^VIX h=10 +27.1% against +20.8%), so the real split is election year
  against odd year, which is closed (kC_c7). 2014 and 2018 carry 97% of the total, and
  the midterm-minus-other gap ranks 12 of 12 months at h=5. (d11_midterm_oct_vol.py, d11b, d11c)
- **Long XLU against ex-ante beta TLT and SPY on a trailing-252 residual low.** The tenth
  dead utilities expression and the first residual pair: h=5 -0.346% on 30-31, 2018+ 8-14.
  The residual gate filters -0.771pp against the plain washout, and the SPY hedge leg
  subtracts -0.44pp. (d12_xlu_resid.py)
- **Method note: the fragility column used for history.** `main_score` in
  rd2_fragility.parquet is filled only from 2026-09-17. A 10-day mean of the `63d` column
  reads 76.8 on 09-25 against the state file's 80.9, so any cell gated on "ma10(63d)" has
  to name the construction it used.
