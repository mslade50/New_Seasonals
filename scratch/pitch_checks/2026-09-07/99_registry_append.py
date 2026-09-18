"""Append the 2026-09-07 stand-down's reusable lessons to the negative registry."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / "data" / "pitch_negative_registry.md"

BLOCK = """
## 2026-09-07 — the Labor Day stand-down

Twenty candidates, roughly 1,700 screen cells over four lanes, six adversarial
checkers, nothing shipped. Entry session would have been 2026-09-08, the first
session after a four-calendar-day closure, in a midterm year with the fragility
dial at 87.96 (99.15th percentile of its own 2016+ series).

- **The extended-closure anchor adds NOTHING to an ordinary weekend, on any
  class.** 697 cells: 17 proxies x 5 horizons x 2 entry forms, plus gap
  comparisons, Labor-Day-only slices and runway halves. Closure (session gap
  >= 4 calendar days, N=180) against the 3-day-weekend control (N=1211): the
  headline vol result inverts into nothing — **^VIX MOO h=3 is -1.830% at
  t -2.91 but the weekend control is -1.712%, an edge of -0.118% at Welch
  t -0.17**. IWM's largest control-differenced statistic across ten cells is
  |t| = **0.80**; EFA 0.96, EEM 0.97, XLE 1.53. SPY MOC h=1 is +0.205% raw and
  +0.128% over the weekend at t 1.41, and Labor Day SPY MOC h=1 is exactly
  **13-13**. Family-wise |t| for p=0.05 is ~3.6 at 170 cells and ~4.0 at 697;
  **nothing clears 697 and nothing clears 170 on the control-differenced t.**
  (01_closure_lane_drift.py, 01_closure_lane_gap.py, 01_closure_lane_confounds.py)
- **The runway conditioner does not exist in the closure lane.** 0 of 170
  runway contrasts reach |t| >= 2.5 and only 2 reach 2.0, against ~8 expected
  by chance. Watchlist 38's arm (long vol after a closure needs runway >= 4) is
  directionally consistent (^VIX MOO h=2 -2.20% at t -3.18 long-runway vs
  -0.35% at t -0.38 short) but the contrast t is 1.10-1.60. The short-runway
  half that was live is flat, not adverse. (01_closure_lane_runway.py)
- **The Labor-Day-only slice is quieter than a coin, so no cell may lean on
  it.** 25 of 170 cells reach sign p <= 0.10 in either direction against ~34
  expected by chance, and the smallest p in the grid (UUP 3-16 at 0.0022 on
  n=19) is roughly one draw of noise against an E[min] of ~0.006. UUP's own
  full-holiday cells at the same coordinates are +0.012% and +0.038%.
  (01_closure_lane_labor_day.py)
- **Do NOT express a lag-1 cell as a resting limit at its own trigger level.**
  Watchlist 5 (TLT with the IG complex at 52-week lows) reproduces cleanly
  (+0.385pp excess, 83.3% hit, 18 episodes, sign p 0.0038, 13.4x cost, zero
  book overlap), and its entire edge is **session +2 (+0.402% at 83.3%)** while
  the session a limit fill adds — close D to close D+1 — pays **-0.302% at a
  38.9% hit**. Every limit variant measured (2 populations x 3 exits x 2 fill
  rules x 11 horizons x 6 overlays, 51 cells) is negative or inside cost. And
  the fill selects against you: the 7 fills where the state actually armed pay
  **-0.564%** at D+1 while the 4 that reversed out pay **+0.957%**, which is
  unselectable at order time. "The order IS the trigger" is exactly backwards
  when the trigger is measured lag-1. (06_tlt_floor_limit.py, _dev.py)
- **An IG-leg contribution can be an ANCHOR SWAP rather than a filter.** In the
  same cell the IEF leg deletes 4 of 80 days at day level and moves the mean
  +0.025pp, but at EPISODE level it moves it **+0.321pp of a +0.385pp
  headline** — purely by re-anchoring the Sept-2022 episode from 2022-09-06
  (-1.03%) onto 2022-09-19 (+1.68%). Check day-level and episode-level gate
  attribution separately before calling a join load-bearing. (LQD, on the same
  cell, does filter: 44 days / 8 episodes averaging -0.72%.)
- **"Negative edge in all six neighbour constructions" is not a finding, it is
  a property of conditioning on a calm bull tape.** Across a 25-instrument x
  4-horizon reference class (100 cells), a k-NN analogue built on today's state
  vector produces **negative edge in 70-82% of cells** (median -0.20 to
  -0.37pp), with twelve unanimous-negative across all six sets. Conditioning on
  a calm tape strips each instrument's fat right tail out of the conditional
  mean while leaving it in the all-days drift. GLD h=5 ranked 12 of 100 and XLE
  h=10 ranked 5 of 100 — both were killed. Any future analogue lane must rank
  its agreements against this reference class before calling one a candidate.
  (09_analogue_gld_xle.py)
- **Short GLD on a tape analogue — killed.** Declustered to 53 independent
  episodes the short pays **0.8 bps, 0.2x a 4 bp round trip, 25-28**, bootstrap
  P(mean<=0) 0.495; the effect decays monotonically with k (+1.194% at k=10 ->
  +0.096% at k=100). Its only content is a drawdown conditioner neither method
  names, and that parent LOSES: plain "GLD >10% below its 252d high" pays the
  short **-0.050% over 109 episodes**, with a non-monotone dose response
  reading -0.988% one band deeper than the live -17.97% and -0.576% near the
  high — the 2026-08-10 silver U-shape reproduced on gold. Six of seven anchor
  shifts make the short negative. (09_analogue_gld_xle.py)
- **Short XLE on a tape analogue — killed, and the mechanism is refuted on its
  own sub-state.** 53 episodes pay **-0.262% (25-28), -5.2x cost**; positive
  only in the N=10 and N=20 sets that are ~45% year-2018. Splitting by XLE's
  own 21-day rank, the **9 episodes matching the live energy leadership pay the
  short -1.299%** against -0.050% on the 44 that do not. Reference-class
  permutation for max-of-9 sectors gives **P 0.765**. Book overlap is additive:
  12 of 12 energy ledger signals around these episodes were SHORT at avgR
  +0.626. (09_analogue_gld_xle.py)
- **XLE's crude beta: BOTH published numbers are right and they measure
  different proxies.** XLE's daily beta on **USO is 0.506** (corr 0.628,
  n=5133) and on **CL=F is 0.112** (corr 0.300, n=6546) — the ETF proxy carries
  about 4.5x the beta of the futures proxy. The 2026-08-11 (0.479) and
  2026-08-28 (0.112) entries are not in conflict. Always state which crude.
- **Long SVXY into the September VIX settle — killed on the control, not the
  mechanism, and it IS the 2026-08-07 pre-expiry corpse.** `P(corpse mask |
  this mask) = 1.0000`, 14 of 14 anchors, with September then picked **rank 2
  of 12** in that parent's own month scan. Paired against its own non-anchor
  neighbours at matched trading-day-of-month over the same 8 post-2018
  Septembers it **loses 6 of 8 at -0.216pp**, and the pooled +1.641pp is 92%
  one year (2016) on the retired -1.0x security. Drop-best-2 on the tradeable
  -0.5x vehicle leaves +0.372% (1.9x cost) with 2020 at 57% of the total, and
  holding PAST the settle pays more (h=7 +2.178% vs h=5 +1.335%). Two attacks
  PASSED and should not be cited as the kill: the pass-through ratio is
  **1.09x against a 0.68x baseline** over 1,169 horizon-matched down-VIX
  windows, and the FOMC-coincidence objection reproduces in the parent
  (-0.261% vs +1.610%) but NOT inside September, where the coincident half is
  **10-0 at +3.504%**. (07_svxy_sep_expiry.py, 07_svxy_sep_expiry_r2.py)
- **Long HYG out of a closure — killed by the live state, and the fifth failure
  of this family to produce a credit-specific residual.** The bare cell is real
  (130 anchors, +0.275%, 66.2% hit, t 3.26, era-stable, month-turn-proof) and
  three independent constructions found it the same morning. But with HYG
  within 1% of its 252d high AND in the calm realised-vol tercile it pays
  **+0.064% (n=40) against +0.075% for an ordinary weekend in the same state**
  — closure excess **-0.012% at Welch t -0.12** — while the whole +0.267%
  excess (t 1.93) lives in the off-the-high half. HYG = -0.014% + 0.189*IEF +
  0.446*SPY, so **+0.196% of the +0.275% is beta** and the residual is 68-62 at
  sign p 0.331; raw SPY pays **+0.370%** on the identical anchors. The accrual
  mechanism is falsified inside its own window: the first session back is
  **-0.122% at t -2.62** against an unconditional +0.021%, and 45% of the
  5-day total arrives on hold day 2. Reversal, not carry.
  (08_hyg_closure.py, 08_hyg_closure_dev.py)
- **The fragility dial DOES have cross-sectional content, and it is NOT at the
  extreme.** First cross-sectional test of a dial this repo has only ever
  tested for direction and sizing. Long SPY / short IWM at the dial's top
  decile pays **+0.517% at h=5 over 71 episodes (47-24, sign p 0.0043)**,
  survives every overlap form (YEAR-mean 8-0 at sign p 0.0039), is LOYO-stable,
  monotone in the threshold, and its top-3 episodes are **minus 14%** of total.
  It is not a proxy (best substitute, days-since-a-5%-drawdown, +0.377% at
  43-24, Jaccard 0.20; dial-only residual +0.502% at 31-16) and not the static
  large-over-small tilt (unconditional dial-era drift +0.075% = 14.6% of it).
  **But the entire edge lives in [56,70)** — 49 episodes, +0.594%, 35-14, sign
  p 0.0019 — while **[70,80) is 14 episodes at +0.071% on a 6-8 record**, and
  the return-on-dial slope is **-0.0053pp per point (t -0.33, R2 0.002)**. The
  complacency gradient the idea is sold on does not exist, so a 99th-percentile
  reading is the one part of the mask with no content. Parked with the arm "the
  10d-MA 63d dial closes back inside [56,70)". (10_dial_spy_iwm.py)
- **A washed-out sector under an index at its high is the gate running
  backwards, pooled and by name.** Nine SPDRs with sector fixed effects:
  negative at **all five** horizons against own drift (-0.074 to -0.305pp) and
  against SPY (-0.015 to -0.162pp) over 256-963 episodes, while the SAME
  washout **without** the index-near-high gate is strongly positive (h=5
  +0.222pp, t 2.13, N=1879). Broad selloffs mean-revert; an idiosyncratic
  sector washout under a strong index does not. XLI at a 21d rank of 7.1 does
  not differ from the family (h=5 -0.002%, edge -0.211pp, XLI-minus-SPY 15-15).
  A fourth confirmation that "while the index holds near its high" starts from
  a negative prior here. (03_pricestate_s4_sector_washout.py)
- **Miners leading with the metal below its 200d: the gate discriminates into a
  zero.** Beta-hedged at the live PIT beta of 1.49, long GDX / short GLD pays
  **-0.111% at h=5 on 37-47, sign p 0.885** over 84 episodes, positive at no
  horizon on any GDX rank threshold from 70 to 90. The metal gate genuinely
  separates (+0.215% gated vs -0.156% with GLD above its 200d vs -0.020%
  ungated) — it just separates into nothing.
  (03_pricestate_s1_miners_metal.py)
- **The VIX-range-compression gate is a NEGATIVE conditioner on the SKEW
  parent, not a neutral one.** SKEW's 21-day rank alone pays SPY +0.333%
  (108-58, sign p 0.0001) and SVXY +1.374% (59-36, p 0.0117); the conjunction
  with a bottom-decile 21-day VIX range pays SPY +0.215% on 13-11 and SVXY
  +0.521% on **8-9**, i.e. the conjunction **subtracts 0.075pp on SPY and
  0.886pp on SVXY** at h=5 and -0.410pp / -0.327pp at h=10. Compression alone
  carries an SVXY edge of -1.065pp.
  (03_pricestate_s3_vol_compression_skew.py)

### Three data corrections filed the same morning

- **`build_pitch_state._metrics_for`'s `vol_vs_63d` is a VOLUME ratio, not a
  realised-volatility ratio.** SPY's 0.69 on 2026-09-04 is turnover; the true
  21d realised-vol ratio against its own 63d average is **0.60**. Two lanes
  independently read it as vol compression before the raw bars corrected it.
- **The VIX 21-day relative-range percentile depends on its denominator.** It
  recomputes to **2.4 trailing-252** and **5.0 expanding**, against the "1st
  percentile" the risk block displays. Still bottom decile; quote the basis.
  Separately, the production `VIX Range Compression` signal reads **OFF**
  despite that reading, because its definition
  (`pages/risk_dashboard_v2.py:495-500`) also requires VIX above its own 20-day
  SMA and VIX is below it. No candidate may claim that signal is firing.
- **`master_prices.parquet` does not carry JNK, XES or RSP**, so a high-yield
  cross-check, an oil-services leg and an equal-weight leg were dropped rather
  than proxied.
"""

text = REG.read_text(encoding="utf-8")
if "## 2026-09-07 — the Labor Day stand-down" in text:
    raise SystemExit("already appended")
REG.write_text(text.rstrip() + "\n" + BLOCK, encoding="utf-8")
print(f"appended {len(BLOCK.splitlines())} lines to {REG}")
