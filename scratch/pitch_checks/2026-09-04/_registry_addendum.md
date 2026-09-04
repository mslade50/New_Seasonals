
## 2026-09-04 (a 12-candidate sweep, all 7 checked candidates killed, stand-down)

- **A left-open threshold (`<= X`) is a CLAIM THAT THE EFFECT IS MONOTONE UP TO
  X, and the band the LIVE reading falls in has to be quoted on its own.** This
  is the second consecutive morning the trap fired and it is now the house
  rule. Post-NFP TLT conditioned on the PRIOR print's surprise looked like a
  survivor at `<= -50k` (h=3 +0.207%, n=31, gate attribution clean at parent
  -0.013% / gated +0.207% / discarded complement -0.066%). Decomposed: the
  moderate half `(-100,-50]` pays **+0.813% on 12** and the half today actually
  sits in, `<= -100k`, pays **-0.175% on 19 at 10-9**. The dose response runs
  BACKWARDS to the stated mechanism, so a bigger miss is a worse trade.
  Yesterday's bimodal VIX-range finding said the same thing about a `<= 15`
  arm. Decompose before believing, and never quote a threshold cell as today's
  expectation without locating today inside it.
  (b1_c3_nfp_prior_surprise_duration.py, b1b_c3_live_band_and_concentration.py)
- **`data/macro_release_history.parquet` exists, is 42,489 US rows of
  actual/consensus/surprise from 2013, and had never been opened by this
  product. Check the newest row PER SERIES, not per file.** The file is frozen
  at 2026-08-07. Monthly series survive that: NFP's last print (2026-08-07,
  -23k against +80k, surprise **-103k**) predates the cutoff and is readable
  today. Weekly series do not: **CFTC COT speculative net positions are five
  releases and 24 sessions stale**, and the staleness was priced rather than
  asserted, since the 104-week percentile moves a median 11.5 points (gold)
  over five releases at a p90 of 40.7, and the `>= 90` state flipped across
  such a gap **20 times in 39 hi readings**. A conditioner that cannot be read
  on the morning of the trade is a kill, and an unstable one is a kill twice.
  (b2_c4_cot_metals_positioning.py)
- **COT positioning is not a flow instrument on this data: both tails are
  long-positive, which means it is reading drift.** Gold above its 90th
  positioning percentile moves GLD h=5 by three basis points (parent +0.378%
  n=294, gated +0.521% n=32, discarded complement +0.491% n=203), and on GDX
  the gate is WORSE than its complement (+0.591% against +0.879%). The
  forced-unwind mechanism needs extreme length to be bearish; instead the
  crowded-SHORT tail pays as much or more (GDX at pctile `<= 15` gives
  +1.744%). The 32 episodes are one episode: 28 consecutive weeks from
  2024-04-05 to 2024-11-05, and dropping 2024 leaves four. The skill's honesty
  note about this repo lacking positioning data should now read "positioning
  data exists, is stale weekly, and did not survive gate attribution."
  (b2_c4_cot_metals_positioning.py)
- **The Dispersion signal does NOT invert into a short-correlation trade, and
  the book's fragility reading of it is the correct one.** Measured on the
  unlevered instrument, short ^VIX from the trigger pays **-6.62% at h=10 over
  104 episodes against ^VIX's own -1.82% drift, an excess of -4.81pp at Welch
  t -2.23**, same sign at h=5 and h=21. Index vol gets MORE expensive to
  realize after extreme dispersion, not less. On SVXY the cell is below the
  vehicle's own drift at every horizon and degrades monotonically toward
  today's reading (>85 +0.293%, >90 +0.290%, >92 -0.576%). Today's exact
  conjunction is the table's worst cell: dispersion with the dial `>= 50` pays
  **-4.17% at h=10 over 62 episodes at a 29.0% hit** against +1.78% with the
  dial below 50. The seven-signal reference class is homogeneous (Cochran Q
  1.49 on 7 df, I-squared 0.0%) with Dispersion 6 of 8. Definition fragility is
  severe: under a 252-day-lookback composite the signal does not fire at all
  today, reading 79.6. (c5_dispersion_short_vol.py, c5b_blockers.py)
- **A market HOLIDAY is a real anchor and its sign is the opposite of the folk
  trade.** First use of a closure anchor in this repo, derived from
  `master_prices` index gaps because there is no holiday list here. ^VIX RISES
  **+4.80% across a >= 3 calendar-day closure over 180 gaps at 136-44, sign p
  0.0000**, against +2.19% across an ordinary weekend and -0.28% on a plain
  overnight, monotone in the extra calendar day. The short-vol-into-the-long-
  weekend trade is therefore wrong-signed by about three VIX points, and SVXY
  across the closure pays -0.272% against +0.291% across a weekend. The
  mark-down is taken INTRADAY on the eve (short ^VIX on the eve session
  +1.257%, n=143, t 3.07) and reverses across the gap, so an eve MOC is one
  session late. (a1_svxy_closure.py, a1b_svxy_gap_decomp.py)
- **The Labor Day long is 0-for-8 since 2018 and the post-Labor-Day short is a
  fixed calendar date wearing a holiday label.** Long SPY entered MOC on the
  eve pays -0.290% at a 34.6% hit over 26 years, the placebo ladder ranks k=0
  **14 of 17**, gate-off across all 154 closures gives -0.076% so the Labor Day
  gate selects the parent's WORSE half, and 2018+ is **0-for-8 at -0.932%**
  (IWM 0-for-8 at -1.402%, t -4.34). Short from the first post-holiday close:
  SPY h=7 -0.551% against a FIXED September trading-day-4 anchor at -0.528%,
  IWM -0.617% against -0.617%, identical to three decimals, and the whole
  number is 2001's 9/11 week at +10.11% on SPY. The folk claim that September
  weakness begins after Labor Day fails its own calendar test on IWM, where
  forward-10 after the holiday is +0.713% against -1.452% before.
  (a2_labor_day_index.py, a2b_gate_attrib_and_inversion.py)
- **September quad witching is an FOMC anchor in costume.** The ungated run-in
  splits on whether an FOMC decision lands inside the window: **+2.382% over 11
  years at a 90.9% hit and t 3.50 with one, -0.834% over 15 at 46.7% without.**
  Every year has a quad and only 42% carry an FOMC. The laggard gate on the
  IWM/SPY pair is worth **+0.006pp** (+0.446% to +0.452%), and the same 63-day
  floor applied year-round pays -0.266% over 160 episodes at a 45.0% hit, which
  reproduces the 2026-08-31 kill exactly. The reference class over 16 index and
  industry ETFs is homogeneous (Q 13.50 on 15 df, I-squared 0.0%) with IWM 6 of
  16. Book overlap was checked and does NOT stick: 152 of 4,701 ledger rows
  in-window is 3.23% against a 3.49% calendar share, 0.93x.
  (c6_iwm_into_sep_quad.py, c6b_gate_vs_anchor.py)
- **The Labor Day driving-season boundary does nothing to energy, and the
  seasonal cannot be separated from the momentum state it arrives with.** Edge
  against each vehicle's own drift at h=10: USO -0.085pp, XLE -0.382pp, XOP
  +0.033pp, VLO -1.352pp, with sign tests undecided in BOTH directions on both
  crude vehicles. The placebo ladder ranks the true anchor 5, 6, 5 and 11 of
  17. The reference class over 7 energy plus 5 non-energy vehicles is
  homogeneous (Q 7.64 on 11 df, I-squared 0.0%, pooled +0.075%) and its only
  positive member is registry-dead UNG. Crossed with the live state, a
  pre-holiday 21-day rank at or above 80, XOP has **zero prior observations**
  against today's 94.8, XLE two averaging -1.42% and VLO four averaging
  -2.74%. (c10_labor_day_energy.py)

### Method finding, filed because it is the reason the morning shipped nothing

The strongest statistic produced all morning was a **corpse-recovered sign
flip** and it was not shipped. Long ^VIX and short the index ACROSS an extended
closure is coherent, monotone in the extra calendar day, era-stable and
strongest in the era we trade (short IWM 2018+ +0.424% at 40-19, sign p 0.0043,
14.1x cost; Labor Day 2018+ 8-for-8 on both vehicles). It surfaced INSIDE the
blockers run against two candidates it is the exact opposite of, so it carries
sign, era and horizon multiple comparisons before any threshold grid was
walked. That is the route the 2026-08-07 entry closed by name, after two such
inversions both died on re-examination. It is parked on the watchlist with a
forward arm instead. **A morning is allowed to end empty while holding a number
it likes; that is what "designed forward, not recovered from a corpse" costs,
and paying it is the point.**
