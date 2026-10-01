# Daily Seasonal negative-results registry

Dead ends the Daily Seasonal must not re-pitch. Stage C checks every candidate
against this file AND against `data/pitch_negative_registry.md` (read-only for
this product). An entry here means the obvious form of the idea was tested and
failed, so a candidate that collides with one must either be dropped or state
exactly what is different about its construction.

Format, one dead end per bullet under the kill rule that killed it, parsed by
`scripts/build_pitch_research_index.py`: a dash, the short key in bold, an em
dash, then why it is dead and the check script that killed it (the same bullet
form as the pitch registry).

The five headings are the seasonal kill rules from
`docs/seasonal_agent_design_2026-09-30.md` (stage C). The registry GROWS: every
stage-C kill with a reusable lesson is appended the same morning.

## 1. Entry-anchored windows only

The historical window starts at the close the order would fill at, never the
day-of-year close. A cell that only works from the day-of-year anchor is dead.

- **DGX midterm October 10d short (2026-10-01)**: the rank file's "midterm 5/6
  lower" is 3-3 from the first October close, and drop-two-best flips to -0.80%
  1-3. The DGX+LH lab short is 2002 and 2008 (99% of the 10d total), 2018+ -0.08%
  4-4. scratch/seasonal_checks/2026-10-01/d11_labs.py

## 2. Index and sector residual is mandatory

Report the return net of SPY and net of the sector ETF, with its own hit rate
and sign test. A seasonal whose residual is zero is an index seasonal and must
be pitched as one or killed.

- **XLU October 21d/63d and the 63d utility rank cluster (2026-10-01)**: XLU
  minus SPY 21d +0.04% 13-13, 63d -1.24% 11-15; a cluster of utilities at 63d
  rank 96-98 is XLU plus SPY. After a ten-year spike (TNX 21d rank 95+) XLU minus
  SPY runs -0.57% over 21d. scratch/seasonal_checks/2026-10-01/a2_xlu.py
- **SLV October 10d board ticket (2026-10-01)**: SLV minus beta-weighted GLD
  -0.36% 10-9; the silver ticket is gold's October. With the ten-year rising
  over 21d SLV minus GLD runs -1.89% 5-6.
  scratch/seasonal_checks/2026-10-01/c7_slv_oct_r1.py
- **Regional bank names at 63d rank 95-99 (2026-10-01)**: the nine-name basket
  minus KRE over 63d is +0.17% 9-11; test KRE, not the names.
  scratch/seasonal_checks/2026-10-01/a3_banks.py

## 3. Cycle cells: drop-best, drop-two-best, and the non-cycle cohort

A cycle conditioner that adds nothing over the non-cycle cohort, or that dies
when its best one or two years are dropped, is dead.

- **SPY midterm fourth quarter from the first October close (2026-10-01)**:
  midterm 63d +2.92% 5-1 trails non-midterm +4.90% 17-3; drop-two-best +0.09%.
  The edge is generic Q4. A TNX shock at entry favoured the long (4-0 near-high),
  so the 2018 path is not identifiable at entry.
  scratch/seasonal_checks/2026-10-01/b9_spy_midterm_q4.py
- **KRE October 63d (2026-10-01)**: KRE minus SPY 12-8 with 71% of the sum in the
  2016 and 2020 presidential Novembers; midterm KRE flips on drop-best (-1.63%).
  The XLF minus SPY near-high 63d 10-1 residue holds only at exactly h63 (7-4 at
  h42, 6-4 at start +/-3). scratch/seasonal_checks/2026-10-01/a3_banks_b.py
- **Crude Q4 short, midterm (2026-10-01)**: CL=F h63 midterm +13.1% 4-2 flips to
  -0.19% on drop-two-best. scratch/seasonal_checks/2026-10-01/c5_crude_q4_r1.py

## 4. Regime branch

Split the history by SPY within 2% of its 52-week high at entry and by the
fragility dial where the PIT history allows. A pattern whose edge lives only
in the branch today is NOT in is dead for today.

- **P&C insurer basket into cat-season end, 21d (2026-10-01)**: equal-weight
  TRV CB HIG AIG ALL PGR SIGI RLI near-high +0.44% 5-6, minus XLF -0.65% 6-5;
  off-high 12-3. Flat at every near-high cut from 1% to 5%. October ranks 5th of
  12 months for basket minus XLF, so there is no cat-season decay to own; the
  basket repeats the 2026-09-30 TRV kill.
  scratch/seasonal_checks/2026-10-01/a4_insurers_b.py
- **Short UNG through Q4, 63d (2026-10-01)**: 16-3 overall, but near-high +4.21%
  5-3 against +5.51% drift while off-high is 11-0; stops up to 3 ATR gut it
  (2018 adverse 26.9 ATR). The NG=F winter rally is roll seams (+7.23pp of
  +7.32% at h21). scratch/seasonal_checks/2026-10-01/c6_natgas_q4_r1.py
- **XLB minus SPY October 63d (2026-10-01, by-product)**: 21-5 all years (p
  0.001) but -0.09% in the near-high branch.
  scratch/seasonal_checks/2026-10-01/a3_banks_b_sectors.py

## 5. Recency

The last 10 years individually, with max adverse excursion in ATR. A 25-year
record with a coin-flip last decade is grade C at best.

- **Crude Q4 short from the first October close (2026-10-01)**: USO h63 last ten
  years 4-6 at -0.14%; after a 30%+ 63-day run the short is 4-8 at -5.40% over
  21d. scratch/seasonal_checks/2026-10-01/c5_crude_q4_r1.py
- **SMH early-October 10d short (2026-10-01)**: 10-16 overall, last ten 3-7.
  scratch/seasonal_checks/2026-10-01/d12_semis.py

## Battery (round 1 controls and gates)

- **TLT long after a 21d bond washout (2026-10-01)**: washouts do not revert at
  10-21 sessions; h21 -0.12% on 30-32 episodes against +0.34% drift.
  scratch/seasonal_checks/2026-10-01/b8_tlt_oct.py
- **IWM vs SPY October, laggard gate (2026-10-01)**: pair h21 9-17 from the
  first October close; the 63d laggard gate adds +0.09pp; the tax-loss short is
  wrong-signed after a lag (4-2 to the long).
  scratch/seasonal_checks/2026-10-01/b10_iwm_spy_oct.py
- **Single-name semis picked by prior-October record (2026-10-01)**: no PIT
  skill; top three minus group -0.14% 7-11, Spearman +0.004. Apply this test to
  any single-name rank outlier. scratch/seasonal_checks/2026-10-01/d12d_pit_gate.py
- **Calendar cells inside 10 sessions across ten classes (2026-10-01)**: 492
  rows (m=984), 23 of 426 tradeable rows under p 0.05 (chance); CPI run-up equity
  is drift (SPY p 0.60 against its base rate); the NFP-day ^VIX/^MOVE crush is
  untradeable (SVXY p 0.12). scratch/seasonal_checks/2026-10-01/d13_sweep.py
