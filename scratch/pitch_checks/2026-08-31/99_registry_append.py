"""Append the 2026-08-31 sweep's reusable lessons to the negative registry."""
from pathlib import Path

BLOCK = """

## Method traps (2026-08-31, from a 22-candidate sweep that killed all 22)

- **The LAG PROFILE settles whether an effect has a shape, and it costs one
  line.** Run lag=0 / lag=1 / lag=2 at the same horizon before crediting any
  mechanism. The short-silver-after-a-complex-break cell reads **+0.039% /
  +0.516% / +0.035%** at h=1: one session wide, and it starts a session LATE.
  No forced-deleveraging continuation predicts that, and the direction of the
  anomaly is itself the tell -- the registry's standing worry is that the
  untradeable lag=0 look FLATTERS a cell, and here the tradeable lag=1 is
  **13x LARGER** than lag=0, which is backwards from every other cell measured
  in this repo. Nothing else caught it: gate attribution passed decisively
  (conjunction +0.531% against +0.031% for the single-name parent and -0.278%
  for the anti-cell), the local +/-126td control cleared at welch t +2.21, the
  record was 67-52 at sign p 0.019 scored against silver's own down-rate,
  declustering was stable at gap 5/10/21, and the six-family reference class
  CONFIRMED rather than killed. Era, decluster and reference-class work would
  all have shipped this. (b4c_c19_slv_short_teardown.py)
- **A continuation cell must be split by the ENTRY-DAY move, and the split is
  a mechanism test rather than a robustness test.** The same silver cell pays
  **+0.867% on 28-15 when silver BOUNCES more than 1% on the entry session**
  and +0.573% on 25-25 when it keeps falling, with an entry-day correlation of
  **-0.033**. "The highest-beta member keeps bleeding" is falsified by its own
  data. Any story about flow persisting into the next session owes this split.
  (b4c3_c19_entryday_and_2026.py)
- **`pitch_lab.cluster_note` ranks the top-k by ABSOLUTE value, so it NETS a
  large winner against a large loser and can report a concentrated cell as
  clean.** The silver cell's "top-2 episodes = 3% of a +41.66pp total" is a
  +16.86% and a -15.56% cancelling; ranked by VALUE on the side actually being
  traded, top-3 is **103% of total** and drop-best-3 is negative. Report
  concentration by value on the traded side, not by magnitude.
- **A percentile is two different statistics and this repo uses both.**
  `rolling(252).rank(pct=True)` (inclusive of the current observation) and the
  `w[:-1] <= w[-1]` form used by the morning recon (exclusive) differ by about
  0.4pp on a 252-day window. On the oil-services spread cell that is the whole
  result: the identical rung on identical data gives **+0.934% on 28-23 at
  15.6x cost** under one convention and **+0.005% on 28-29 at 0.08x** under the
  other, and today's live bar straddled the gate at **3.98 excl-self against
  4.37 rank**. Every parked cell must record which convention minted it.
  (b4_c6_oih_xop_ladder.py)
- **Charge the grid you SCANNED, not the axis you found it on.** The
  bond-vol band cell cleared a band-only permutation at P 0.029 and reads
  **P 0.857** once charged for the bands x horizons x vehicles it was actually
  walked over -- a below-median draw from the best-cell-under-no-effect
  distribution. The candidate spec itself named four sign and vehicle
  combinations, which is the disclosure that makes the wider charge mandatory.
  (b2c_c2_fullgrid_and_dose.py)
- **An inverted-U conditioner ladder falsifies a directional mechanism even
  when the live bucket is the maximum.** The tails are where the mechanism
  makes its strongest prediction, and for "an orderly repricing trends" the
  most compressed bond-vol bucket [0,20) is the **worst** long-duration bucket
  at -0.809%. A monotone ladder supports a dose response; a hump means the
  chosen band is mid-range wearing an extremity label.
- **A DEPTH BAND is instrument-specific and cannot be transplanted.** The
  2026-08-26 credit kill established that the index-distance gate is worth
  +0.615pp beyond 2% and -0.042pp in the 1-2% band -- measured on SPY.
  Substituting IWM because it sat 3.06% off its high assumed SPY's ladder;
  measured on IWM, the 2.0-5.0% band is IWM's **dead** band at -0.119% and
  **-6.0x cost**, and the open-ended >=2% form is positive only because it
  pools that dead band with the >5% tail. Same kill, different ticker.
  (b2f_c4_hyg_high_iwm_depth.py)
- **Print the distance-from-extreme across a calendar anchor's history before
  running any statistics.** One `print` of 27 August month-end anchors' yield
  distances ended the September duration candidate in a line: **^TNX has never
  been within 2% of its trailing-252 high at an August month end**, distances
  running -1.54% (today) to -62.48%, so the interaction cell has exactly one
  observation and it is the live one. This is the 2026-08-07 count-first rule
  applied to a conditioner rather than to a joint state.
  (b2e_c8_empty_cell_and_fallback.py)
- **The placebo offset ladder is now 12-for-12**, adding a second single-name
  earnings failure (the pre-print anchor ranks 5 of 16, with four neighbouring
  offsets beating it) after the 2026-08-25 case.
- **`data/earnings_calendar.parquet` is not usable as an anchor calendar before
  ~1993.** 82-87% of 1985-1992 rows land exactly on a quarter END and 1988-91
  carries up to 61% weekend dates: those are fiscal period ends masquerading as
  announcement dates. Restrict to 1996+ before any earnings anchor (prices
  bound it at 1999 anyway). First use of this file as a pitch anchor.
  (b3_c3_recon.py)
- **^GSPC is not a usable OVERNIGHT instrument before ~2013.** Yahoo's
  synthetic open gives a pre-2013 overnight series with a **median of exactly
  0.000 at a 25.0% up-rate** and an sd of 0.159%. A dividend-contamination
  hypothesis raised against the month-end overnight cell was WRONG for the
  same reason and is recorded here so it is not raised again: the RAW
  unadjusted overnight excess is LARGER than the adjusted one (SPY +9.92
  against +7.38 bp), so adjustment does not manufacture overnight returns.
  (b1_c1_me0_overnight.py, b1f_c1_single_roundtrip.py)

## Cells swept and empty (2026-08-31)

- **The month-end anchor's OVERNIGHT return, which is a genuinely new return
  object in this repo and closes the anchor's sixth form.** All five prior
  month-end closures measured close-to-close; nobody had measured
  `Open[ME+1]/Close[ME-0]`. The headline is real -- SPY +10.48 bp against a
  +2.98 bp unconditional overnight, 206-113, **sign p 0.0004** against SPY's
  own 55.1% overnight base rate; IWM +16.80 bp, sign p 0.0004 -- and the
  mechanism is false. The reversal regression that the auction story predicts
  **runs backwards on the one session that has the auction**: slope +0.194
  (t +2.03) on SPY's ME-0 sessions against **-0.131 (t -6.56)** on all
  sessions, with IWM +0.081 against -0.079 and QQQ +0.057 against -0.277. The
  15-vehicle reference class then names it: **EEM (+21.2 bp) and EFA (+15.0 bp)
  rank first and second**, two markets that are SHUT during the US closing
  auction and reopen overnight in Asia and Europe, and the family is
  homogeneous (Cochran Q p 0.6875, I-squared 0.0%) at a common excess of
  **+8.26 bp (t +7.24)** with SPY 9 of 15. One market-wide overnight drift
  wearing fifteen labels. August is the WORST of the twelve months on every
  vehicle (SPY **-9.87 bp** at a 46.2% hit, DIA -13.24 at 38.5%); the cell is
  Dec plus Oct-Nov, and December fails its own max-of-12 scan at P 0.476-0.931.
  The ladder does not isolate (ME-4 beats ME-0 on all five vehicles, true
  anchor **rank 2 of 9**), era decay is monotone (SPY 10.34 -> 4.78 -> 3.91 bp),
  cost never reaches 5x, and August-in-a-midterm is **3-3 and negative on all
  five vehicles**. (b1_c1_me0_overnight.py, b1d_round2_refclass.py)
- **The intraday shape of the month-end session, and the first use of the
  15-minute cache by a pitch check.** Data is deep and fine (SPY/IWM
  2003-09-10 onward, 5,708 sessions, 265 ME-0). The finding is genuine: the
  ME-0 last hour IS distinguishable on SPY (**-0.065% against +0.004%, welch
  t -2.52**) and IWM (**-0.128% against +0.009%, t -4.35**) on a last-hour
  volume share of 30.0% against 23.7%, and the offset ladder isolates the true
  anchor at **rank 1 of 9** on both, which is rare. As a trade it dies on era
  sign instability: SPY runs **+14.46 bp pre-2013 -> +1.75 (2013+) -> -2.03
  (2018+) -> -4.82 bp (2020+)**, wrong-signed in the modern era, and IWM
  decays to 0.43x cost. QQQ has the volume signature (25.1% against 21.2%) and
  **no return signature at all** (+0.003%, t +0.20), which is absorption rather
  than impact and is the cleanest single argument that the flow does not move
  price. The one-round-trip join (buy 15:00 ME-0, sell MOO ME+1) is worse than
  either leg at 0.67x / 0.22x / 1.12x cost, because the last hour eats half the
  overnight. (b1b_c15_intraday_shape.py, b1e_c15_lasthour_standalone.py)
- **Short silver after the whole metals complex breaks together.** See the lag
  profile and entry-day entries above; parked with two arming numbers. Filed
  here because of what did NOT kill it, so nobody re-runs them: the
  state-matched and depth-matched splits are **not distinguishable** (worst
  welch t -1.59 for "all four of today's states") and the multiplicity charge
  on those refinements is **p 0.822 over a 56-cell grid**, so the
  unconditional cell is the honest estimate and the state-matched h=3/h=5 forms
  (+1.355%, +1.334%) are not real. The GLD-beta residual also survives
  (+0.322pp edge, 61.1% hit), so this is NOT the closed "second metals leg is
  size, not diversification" objection. Per-leg attribution on the same
  trigger: short gold is **-2.6x cost** and short the miners **-1.1x**, both
  with top-2 concentration at 202% and -233% of total.
- **Fading the miners' 21-day outperformance of the metal at a 98th-percentile
  spread.** Wrong-signed at every horizon (-0.186% at h=1 to **-1.718% at h=10
  on 24-41**) with the percentile ladder monotone in the wrong direction, so
  the more extreme the spread the worse the fade. The beta-neutral form is also
  negative and the long-metal leg subtracts. The seven-pair miner-metal
  reference class is homogeneous with a **negative** common excess (-0.021% at
  h=1, -0.165% at h=3) and no family support for the fade in any name, while
  the CONTINUATION side pays **+1.718% at h=10 on 41-24, sign p 0.045**,
  independently re-confirming the 2026-08-27 finding. (b4d_c18_gdx_gld_fade.py)
- **Industrial metals thrusting while precious metals flush**, which is inside
  the sector-pair family closed on 2026-08-27 and reproduces that closure on a
  fresh enumeration: 131 ordered pairs from a 12-name pool give Cochran Q
  p 0.936, I-squared 0.0%, permutation max-of-131 **P 0.723**, with this pair
  ranking **39 of 131**; the pair form is worse still (common excess -0.144%,
  P 0.996). The short leg is the wrong side, since silver RISES after this
  trigger (**-0.474% at h=5 on 10-19** for the short), so the pair is strictly
  worse than the outright everywhere, and drop-best-2 at h=5 is -0.721% with
  the top two episodes at 356% of total. The trigger has also been ON since
  2026-08-11 with nothing happening. (b4b_c6_c10_pair_refclass.py)
- **Long oil services at a services-versus-exploration 63-day spread extreme**,
  the parked watchlist entry, now CLOSED rather than re-parked. Beyond the
  percentile-convention finding above: the ladder is non-monotone and inverted
  at the tight end (pit<=0.5 **-1.404% on 10-14**, <=1.0 -0.655%, <=2.5
  +0.934%, <=3.0 +0.005%, <=20 +0.875% on 90-63), so extremity is not what the
  cell keys on; the headline exists only at a declustering gap of exactly 10
  (gap 5 +0.306%, gap 21 +0.091%, gap 42 **-1.682%**); drop-best-3 is -0.126%;
  midterm is **-2.233% on 6-11**; and the 12-pair reference class is
  homogeneous with a **negative** common excess of -0.121% at a permutation
  max-of-12 P of 0.806, where this pair is not even the family maximum. The
  entry's "four wins away" arm was arithmetically wrong -- it converted losses
  to wins rather than adding episodes; the real answer at the 4.0 rung (31-32)
  is that no number under 15 consecutive wins arms it.
- **The tail-premium-to-at-the-money ratio (SKEW over VIX3M) at a trailing-year
  extreme.** A total re-skin, and the overlap statistic is the whole finding:
  `P(inside the closed VIX3M-floor OR SKEW-rank cells | ratio at a 95th
  percentile)` is **1.000 at day level (590 of 590) and 1.000 at episode level
  (95 of 95)**, and tightening the ratio rung makes it MORE redundant (0.989 at
  the 98th). The ratio moves on its denominator: corr(dlog ratio, -dlog VIX3M)
  **0.895** against corr(dlog ratio, dlog SKEW) 0.549, with VIX3M's daily sd
  1.9x SKEW's. The conjunction subtracts (SKEW alone +0.268pp of excess at
  sign p 0.035, the ratio -0.011pp, the VIX3M leg alone -0.090pp) and the
  ladder inverts exactly as the 2026-08-27 VIX3M floor did. Mechanism
  falsified in-window in BOTH directions: across the hold ^SKEW falls -0.251%
  but **^VIX RISES 3.816%** and ^VIX3M +1.956%, so the ratio reverts because
  the denominator rises, not because tail premium decays. Also out of sample:
  the live reading is the 98.4th trailing-252 percentile but only the **80.7th
  of full history**, the 2026-08-14 SKEW-median-drift trap live, and on the
  full-history basis the mechanism needs, today does not trigger at any rung.
  (b3_c5_overlap.py, b3_c5_battery.py, b3b_c5_livecell.py)
- **The dollar CONFIRMING a rate rise, the untested inversion of the parked
  unconfirmed form.** The premise is false on the live tape and that is the
  kill: the "rate rise" is **+5.7 bp over 21 sessions, the 6.9th percentile of
  1,001 trigger days** against a trigger-day median of +24.5 bp, and the dollar
  leg reads a 21-day rank of 42.9 against the rule's 65 floor. The ten-year is
  at a 52-week high **by level only**. Where the rule does fire the
  confirmation leg adds **+0.016pp at t +0.21** over the dollar alone, 45% of
  the total sits in two late-2008 episodes, the record is 136-145 at sign
  p 0.725, and the grid charge over 320 cells gives P 0.807. Restated to the
  state that actually fires today the sign INVERTS to **-0.536% at t -2.20**
  (32.4% hit, bootstrap 0.991), with short EURUSD, long USDJPY and the dollar
  ETF all agreeing. The offset ladder makes it a lagging label: **k=-5 pays
  +1.371% at a 100.0% hit, t 12.40** against the true anchor's +0.055%.
  No threshold arms it, so nothing is parked. (b3_c14_r1.py, b3b_c14_r2.py)
- **Pre-print drift in a deeply lagging mega-cap, the first use of the earnings
  calendar as a pitch anchor.** The pitched conditioner is what kills it: the
  "deeply lagging" gate is worth **-1.867pp on the very name pitched**, taking
  its pre-print session from +0.513% on 33-22 ungated to **-1.354% gated**, and
  the gate ladder is monotone against the pitch (<=5 -1.354% / <=10 -0.407% /
  <=25 -0.441% / **>25 +0.779%**), with today's reading on the worst rung. The
  horizon ladder falsifies the mechanism outright: holds beyond two sessions
  run -0.56pp at 3 td, -2.01pp at 5 td and **-4.52pp at 9 td**, so the lagging
  name keeps FALLING into its print and there is no run-up, only a 1-2 session
  tail. Reference class closes both forms -- gated, the name cannot even enter
  the 535-name class; ungated it ranks **116 of 934 at a permutation P of
  1.0000** against a near-homogeneous class (common excess +0.145pp, I-squared
  16.5%). Era flips pooled (+0.190pp at t 3.31 pre-2018 to **+0.002pp at t
  0.04** from 2018) at 1.49x cost, and liquid large caps specifically are
  **0.44x cost**. Half the raw number is beta (pooled beta 1.057, and the gate
  over-selects up-tape). Survivorship note: 97.7% of the calendar's tickers
  still report in 2026 and the price cache holds today's universe only, so a
  cell selecting names that just fell 17% in 63 days is exactly where the
  missing delistings sit -- the common excess is an UPPER BOUND.
  (b3_c3_preprint_r1.py, b3b_c3_preprint_r2.py, b3b_c3_ungated_refclass.py)
- **High yield at a 52-week high while the SMALL-CAP index sits below its
  own**, the depth-substituted re-ask of the 2026-08-26 credit kill. Dead as a
  re-skin (see the depth-band entry above) and independently on its reference
  class: six indices in the depth slot give a fixed-effect common excess of
  +0.143pp with **Cochran Q 4.32 on 5 df, I-squared 0.0%**, a cross-sectional
  sd of 0.152pp against a mean sampling SE of 0.162pp (**ratio 0.94**, so the
  whole spread is sampling noise), IWM ranking **5 of 6**, and a random-date
  max-of-6 **P of 0.954** -- the left tail of a null, not the right tail. The
  dial split finishes it: the entire edge is in dial [0,30) (+0.311 / +0.658 /
  +1.570%) while [50,70) pays **-0.453 / -0.794 / -1.749%** and the live
  dial>=80 slice is 2 episodes. It is a calm-tape effect and this tape is the
  opposite. (b2f_c4_hyg_high_iwm_depth.py, b2g_c4_freshhigh_repro_and_dial.py)
- **The small-cap laggard into the month turn, long IWM against short SPY.**
  A join that subtracts at every rung, on a parent that is already negative:
  the beta-neutral pair at ME-0 pays -0.038 / -0.108 / -0.170 / -0.293% at
  h=1/3/5/10 against its own all-days drift of -0.007 / -0.020 / -0.032 /
  -0.063%, so the anchor makes the pair WORSE at every horizon and the ladder
  ranks the true anchor **8 of 9**. The short leg is the better leg (SPY
  outright beats IWM outright at every horizon; IWM at ME-0 is below its own
  all-days drift), and short-leg attribution is negative at every rung and
  horizon (-0.116 to -0.284pp). Threshold-mined around the live reading: the
  h=5 gate ladder runs r5<20 **-0.200% on 9-18**, r5<30 +0.021%, r5<40 -0.039%,
  and the complement (IWM LEADING, r5>70) beats it. Confirmed NOT a re-skin of
  the closed index-pair cell (residual correlation +0.019 to +0.052), so it is
  a genuinely new object that dies on its own. Per the 2026-08-28 rule, a
  negative interaction is not parkable. (b1c_c12_iwm_spy_me0.py)
- **Long duration with the ten-year at a 52-week yield high and bond vol
  compressed.** The pitched conjunction beats neither parent; the surviving
  mid-range band is parked with an episode count. See the grid-charge and
  inverted-U entries above.
- **Short duration into September with the ten-year at a 52-week yield high.**
  The interaction cell is empty (one observation, the live one). The bare
  September parent fails on cost once short carry of ~1.79 bp/session is
  charged (**0.53x at h=5, 2.27x at h=10, -0.91x at h=21**) and on its own
  twelve-month scan (August ranks 4 of 12, **max-of-12 P 0.997**), with the top
  three episodes at 107% of total, 2021+ at -0.872% on a 40% hit, and the
  bond-bull objection INVERTED (falling-yield years +0.764% against
  rising-yield +0.551%). Reproduced the registry's month-of-year table exactly,
  with one correction: September is TLT's **third**-worst month on those
  numbers, not the second as the 2026-08-13 entry states (Oct -0.432%, Apr
  -0.240%, Sep -0.220%). (b2d_c8_september_duration.py)

### Calendar finding, filed because it is now six consecutive sessions

- **A sixth straight empty anchor set, and the month turn arrived and closed
  its sixth form the day it became reachable.** Jackson Hole moved to JH-1 and
  is closed on eight classes pre-speech and ten post; NFP at +4 td, PPI at +7
  and CPI at +8 are all closed on their own ladders; FOMC and VIX expiry at
  +11 and opex and quad witching at +13 remain beyond the 10 td cap. The month
  turn was the only live anchor, and its one never-measured return -- the
  overnight -- was measured today on daily bars AND at 15-minute resolution,
  and closed both ways. **The month-end anchor is now closed on six forms
  across five asset classes.** The next genuinely new anchor is still the
  September FOMC on 2026-09-16, which enters the horizon around 2026-09-02 and
  is spoken for by the event sleeve's T1/T2 -- and the midterm T2 short is
  gated on SPY's 21-day rank being under 50, which reads 91.3, so the sleeve's
  own rule remains off. The practical consequence is unchanged and now six
  sessions old: a price-state sweep is the only honest search mode. Today it
  produced twenty-two candidates across ten asset classes and no survivor, and
  the closest of them died on a lag profile that no other test in the battery
  would have caught.
"""

p = Path("data/pitch_negative_registry.md")
text = p.read_text(encoding="utf-8")
if "Method traps (2026-08-31" in text:
    print("already appended, no change")
else:
    p.write_text(text.rstrip() + BLOCK, encoding="utf-8")
    print("appended", len(BLOCK), "chars ->", p)
