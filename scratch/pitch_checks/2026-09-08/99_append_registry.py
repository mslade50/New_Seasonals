"""Append 2026-09-08's reusable kills to the negative registry."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "data/pitch_negative_registry.md"

SECTION = """
## 2026-09-08 — the second stand-down in two sessions, on the same tape

Twelve candidates, four adversarial checkers, 28 check scripts, all twelve
killed on substantive grounds. The structural fact: **2026-09-05..07 was the
Labor Day closure, so this morning had NO NEW BARS.** The tape was identical to
the one yesterday's stand-down swept over ~1,700 cells, and yesterday was
already targeting this same entry session. Only the calendar moved. That is why
the morning went almost entirely into the event lane, and why the price-state
lane was re-verification by construction.

### Method traps

- **`align()` must NEVER be applied to a forward-return series, and the damage
  lands exactly on today's trigger.** Two checkers found this independently in
  `scratch/pitch_checks/2026-09-07/_survey_lib.py`, whose union-reindex-then-
  ffill carries the last resolvable value into the trailing `lag+h` rows that
  are NaN by construction. Measured cost: on SVXY it smeared one h=10 return
  from 2026-08-20 across **11 sessions, 2026-08-21..2026-09-04, INCLUDING the
  live anchor, 3 of them inside the live mask**, entering the cell as fabricated
  episodes. On ITA it minted a single phantom episode dated **2026-09-02 whose
  booked -4.942% at h=10 is exactly the "worst episode -4.94%" watchlist 40
  quoted**, and it moved the published cell from N=42/+0.629% to N=43/+0.543% at
  h=5 and N=28/+1.443% to N=29/+1.223% at h=10. Any other 2026-09-07 cell built
  on that helper with a LIVE trigger carries the same phantom. `pitch_lab` is
  unaffected; this was a day-local helper, which is the argument for the
  standing rule that reusable machinery gets promoted into `pitch_lab` with a
  test rather than rebuilt each morning.
  (c5c_align_bug_and_residual.py, c12b_repro_discrepancy.py)
- **A number carried out of a kill report is RAW until proven otherwise.** The
  2026-09-07 line "SKEW's 21-day rank alone pays SPY +0.333% and SVXY +1.374%"
  is raw. Against an all-days control the excess is **+0.142pp on SPY and
  +0.738pp on SVXY at h=5, and -0.111pp and -0.016pp at h=10.** State the
  control basis when quoting a control leg, because the next morning will read
  it as an edge.
- **A gate that discards 91% of a parent while the complement KEEPS the parent's
  edge is a lucky subset, not a filter.** The cleanest instance yet: a live
  three-way short-TLT cell at h=8 reads **5-for-5, +1.670%, worst +0.357%, sign
  p 0.0312, tdom-matched +1.686pp, placebo rank 1 of 11** — the only offset with
  a perfect record — and dies anyway. It drops **49 of 54 episodes and the
  complement still pays +0.379% against the parent's +0.443%**, so nothing may
  be attributed to the leg; P(a random 5-subset of its own parent beats it) is
  **0.0631** before any search charge, and the 16 cells searched give a
  family-wise P of **0.648**. Complement-retains-the-edge is a faster test than
  a permutation and should be run first. (c8c_threeway_kill.py, c8e)
- **Gap-share is the decisive test for a "the release moves it" mechanism.**
  Long duration across two 08:30 prints accrues only **7.9% of its hold in the
  two release gaps** (+0.0118% against an unconditional two-gap baseline of
  +0.0066%, a gap excess of +0.0052pp = ~6% of the claimed tdom excess). 92%
  arrives after the news is public, so the cell is not about the release. Cheap,
  and it falsifies inside the window rather than around it. (c2)
- **A left-open threshold's dose response, checked yet again, ran BACKWARDS on
  two separate cells this morning.** SKEW r21 excess by band at h=5: [85,90)
  **+0.188pp**, [90,95) +0.057, [95,98) +0.082, [98,101) +0.114 — the band BELOW
  the threshold is the best one and cum>=85 beats cum>=90. Energy leadership by
  21d rank: +0.205 / +0.218 / +0.087 / **-0.006** / **-0.390** / -0.358%, more
  leadership paying less, with both live readings in the bad end. Third and
  fourth consecutive mornings the trap has fired. (c4, c9)

### Cells swept and empty

- **^SKEW's 21-day return rank is the DILUTED TAIL of the parked 5-day form, not
  its parent.** Excess at h=5, th=95, decays monotonically by lookback: r5
  **+0.327**, r10 +0.174, r15 +0.076, r21 +0.142, r42 -0.031, r63 -0.151pp, and
  BOTH level-percentile conventions are negative (trailing-252 -0.193, full
  history -0.029). At the pitched r21>=90 the excess is **+0.021pp**. Grid
  walked: 6 lookbacks x 3 thresholds x 6 horizons + 2 level bases = 144 cells,
  of which 12 of 24 h=5 cells are positive at a median +0.011pp. **The midterm
  block that parks watchlist 6 reproduces on the parent and STEEPENS with the
  threshold** — >=90 -0.079pp, >=95 -0.143pp, **>=98 -0.387pp on 12-13** at h=5,
  and -0.756pp on 13-11 at h=10. And it is dip-buying wearing a skew label: plain
  SPY 5d<=-1% pays +0.220pp on N=512, skew AND dip +0.711pp on N=57, **skew and
  NOT dip -0.070pp on N=253**. (c4_skew_r21_spy.py, c4b_live_intersection.py)
- **SVXY on the same skew state has no volatility-specific residual, which was
  the whole instrument-translation premise.** SVXY = -0.293% + 1.62*SPY (R^2
  0.648); on the cell's own episodes **SPY's own excess is -0.107pp**, so the
  vehicle is levered exposure to an equity leg below its drift. The one
  surviving cell is **70% one year** (2023 supplies 7 of 26 episodes and +24.66pp
  of a +35.01pp total); drop it and +0.663pp becomes **-0.139pp**. The pooled
  headline also blends the retired -1.0x product with the tradeable -0.5x one
  (pre-break excess +1.329pp). (c5, c5b, c5c)
- **A skew spike has NO cross-sectional content: the IWM leg pays MORE than the
  SPY leg** (+0.149 vs +0.142pp at h=5), so long SPY / short IWM is -0.007pp on
  86-80, and every threshold x horizon lands inside +/-0.10pp. The dial's known
  [56,70) cross-sectional edge does not reach it: **Jaccard 0.043** (17 days),
  and Jaccard with dial>=80 is 0.016 on 5 days, none with a resolvable return.
  TESTED AND REJECTED, recorded so it is not rediscovered: the full live nested
  ladder (band 98+ / midterm / non-dip / near-high) reads **+0.907pp on n=12,
  9-3, sign p 0.073 at h=10** while the IDENTICAL 12 episodes read **6-7,
  +0.083pp at h=5**. Sign instability across horizons on the same episodes at
  the end of a 5-layer nest built off a mask that is 50-50 at layer 0.
  (c6_skew_r21_spy_iwm.py)
- **BOTH orderings of a one-session-apart CPI/PPI pair are now measured, and the
  PPI-then-CPI side is the negative one.** It reproduces the 2026-08-10 line at
  **-0.1135% on N=127** (registry -0.071% on N=133). The pair GATE selects its
  parent's worse half: SPY over the pair window pays +0.001% against its own
  same-span drift of +0.112%, while **PPI with no CPI next session pays
  +0.140%** — the gate is worth **-0.115pp where its complement is +0.024pp**.
  IWM is -0.193pp on a 50.4% hit. Placebo rank 8 of 11 (SPY), 9 of 11 (IWM),
  with k=+1 paying +0.339%. Do not re-open the annual cell as novel.
  (c1_ppi_cpi_pair_equity.py)
- **The pair gate SUBTRACTS on duration too.** tdom-matched h=3 excess: pair
  +0.088pp (TLT) / +0.030pp (IEF) against **+0.120 / +0.044 for PPI-with-no-CPI**
  and +0.106 / +0.038 all-PPI. **September, the live month, is the worst cell in
  the table and wrong-signed: TLT -1.120pp at a 25.0% hit** on 12 observations.
  Cost 2.9x tdom-matched and **1.5x under month x tdom**. (c2)
- **"A print on the very next session" is SET-IDENTICAL to runway == 1: N=139 vs
  N=139.** A back-to-back print pair therefore cannot be anything but the dead
  half of watchlist 33's runway conditioner, and the live ambiguity is closed by
  definition rather than by measurement. Reproduced alongside: SVXY runway<=1
  +0.185% (t 0.66) / short ^VIX -0.392%, against runway>=3 +0.336% (t 2.12) /
  +1.004% (t 3.34). The pre-pair hold is separately wrong-signed on the
  unlevered vehicle — **short ^VIX cumulates -0.52 / -1.36 / -0.67 / -0.45 /
  +0.92 through it**, so vol RISES into and across the pair and **76% of the
  return lands only after the calendar clears**. (c3, c3b, c3c)
- **Long commodities at a 252-day high into an inflation print: the cell does
  not exist outside the inflation shocks.** 2007, 2008, 2021 and 2022 hold 26 of
  53 episodes and **MORE than 100% of the total** (+55.06pp of a +50.72pp DBC
  total); ex those years DBC -0.161% and USO -0.225%, both at a 44.4% hit. The
  print anchor is decoration: placebo rank **7 of 11 (DBC) and 9 of 11 (USO)**,
  with k=-4 paying +0.494 / +0.792% against the true +0.147 / +0.116%, and at
  h=10 the gate catches **55.8% of all days**. The exact live configuration,
  both prints inside the window, is **13 episodes at -0.027%**. Roll priced:
  **USO CAGR -5.07%/yr vs CL=F front +2.32%/yr over 19.7y = -8.8 bps per
  3-session hold**, larger than the entire h=3 episode mean of +11.6 bps; DBC's
  drag is benign at +0.26%/yr. (c7, c7b, c8b)
- **Energy equity leadership with a print in the hold is negative against every
  control and inverts across eras.** XLE +0.013% vs all-days +0.147% = -0.134pp
  at 0.4x cost; XOP -0.169% vs +0.123% = -0.292pp at NEGATIVE cost, 44-40. XLE's
  edge is positive at 2 of 10 horizons, **XOP at 0 of 10**. XOP pre-2018 +0.511%
  against **2018+ -0.954%**. Beta-neutralising does not save it (XLE-SPY 47-61).
  **September-and-midterm episodes number 0 of 108 (XLE) and 0 of 84 (XOP)**,
  which reconciles with rather than contradicts the 2026-09-04 pre-holiday
  r21>=80 finding. (c9)
- **A SUBGROUP flush inside an intact sector is the short-term reversal factor
  wearing a group label, and its short side is wrong-signed at every horizon.**
  Paired on the SAME 100 common episodes against an equal-weight basket of the
  equally-flushed names from an 80-name NON-staples universe, the food basket
  pays +0.940% against **+0.899%** — a difference of **+0.041pp at t +0.11 on a
  48-52 record**. The broad universe's own version pays +0.537% over **1,024
  episodes at t +4.26**, so the factor is real, generic and already known. The
  pair-vs-XLP form is dead on its own numbers (56-50, sign p 0.314; the
  ALL-member basket vs XLP is NEGATIVE at -0.162%), a sixth confirmation of
  "price the legs before the spread". Nine deterministic sector subgroups under
  their own SPDR gate give **P(max-of-10 >= FOOD) = 0.6911** and 500 random
  10-name subsets of the 41-name Consumer Defensive pool give **P = 0.1380**;
  drop-best-5 takes the excess from +0.671pp to +0.098pp. **The SHORT side is
  positive at every horizon 1 through 10** (h=5 +0.931%, t +2.45, 69-37 gated;
  +0.428%, t +2.26 ungated), which is the correction owed to the prior: the
  2026-09-07 kill was a SECTOR-level washout under an index at its high, and a
  MEMBER-level flush inside a sector runs the other way. (c10, c10b, c10c)
- **Watchlist 40 (ITA) is RETIRED: its arm ran and returned P = 0.6892.** The
  13-name subsector reference class (ITA IHI IBB XBI ITB XHB XRT XME XOP OIH KRE
  SMH IYR) puts ITA at **rank 2 of 13 on h=10 excess with max-of-13 P 0.6892**,
  and 0.9771 / 0.9889 / 1.0000 on the other three bases; **on all four ITA's
  observed excess sits BELOW the null's median best-of-13 draw**. The class is
  homogeneous (Cochran Q 7.80-14.21 on 12 df, I-squared 0.0-15.6%) at a
  fixed-effect common excess of +0.020pp (h=10) and +0.127pp (h=5), and IHI
  beats ITA outright at h=10 while XRT beats it at h=5. Same shape that closed
  the country-decoupling family at P 0.477. Revival needs P below 0.05.
  (c12_ita_refclass.py)

### Calendar finding, filed because it is a structural property of the product

**A market closure means the next morning's run has NO NEW INFORMATION, and the
pitch product has no rule for that.** 2026-09-07 and 2026-09-08 read the
IDENTICAL tape (bars 2026-09-04 on 215 of 218 names; only `^VIX` and
`DX-Y.NYB` carry holiday stubs) and targeted the IDENTICAL entry session, so
every price-state cell killed on the first morning was already killed on the
second before it started. Both stood down. That is the correct outcome and it
was reached twice at full cost. The lesson for the next long weekend: on a
no-new-bars morning the price-state lane is re-verification, the marginal value
is entirely in the axis that MOVED (here the calendar advancing one session into
a PPI/CPI pair and a pre-FOMC window), and the watchlist arms are the cheapest
work available because their checks already exist. Budget the morning that way
from stage B1 rather than discovering it in stage C.
"""

txt = P.read_text(encoding="utf-8")
if "## 2026-09-08 — the second stand-down" in txt:
    raise SystemExit("already appended")
P.write_text(txt.rstrip("\n") + "\n" + SECTION, encoding="utf-8")
print(f"appended, registry now {len(P.read_text(encoding='utf-8').splitlines())} lines")
