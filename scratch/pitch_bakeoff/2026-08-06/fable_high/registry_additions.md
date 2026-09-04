# Registry-worthy kills — fable_high bake-off run, 2026-08-06

NOT appended to data/pitch_negative_registry.md per the bake-off overrides.
If this variant's run is adopted, these four belong there.

## Calendar and factor seasonality

- **TLT August seasonality (and any "bonds rally over the August NFP week"
  framing)** — the Aug-NFP TLT week (+50 bps, hit 0.75, N=24) is legacy
  August bond seasonality in disguise: all-August TLT days ran +16.8 bps/day
  2002-2012, +11.1 2013-2019, then FLIPPED to -6.9 bps/day 2020+ with four
  straight losing Augusts 2020-2023. Era check kills both the seasonal and
  the event-week cell riding on it.
  (scratch/pitch_bakeoff/2026-08-06/fable_high/checks/xlu_era_tlt_aug.py)

- **Aug-NFP hot-tape interaction** — conditioning the August payroll-day
  short on a hot 5d run into the print adds nothing: the all-NFP hot cell is
  -13 bps t -0.49, and Aug+hot is N=2. The plain August cell (-36 bps, hit
  35%, N=26) is the entire story; do not spend future checks re-opening the
  interaction.
  (scratch/pitch_bakeoff/2026-08-06/fable_high/checks/spy_nfp_hot_tape.py)

## Strategy-structure dead ends

- **Buying a utilities washout while the index is near highs** — wrong-signed.
  XLU 21d rank < 10 with SPY within 2% of its 252d high: outright XLU -45 bps
  fwd 5d, XLU-SPY spread -112 bps (hit 24%, N=25, 2005-2026). The washout
  CONTINUES; the evidenced side is the relative short, softer but still
  negative 2018+ (-71 bps, hit 40%).
  (scratch/pitch_bakeoff/2026-08-06/fable_high/checks/xlu_washout.py)

- **AAPL vs QQQ extreme 5d-divergence snapback** — no RV edge: the
  tradeable-N cell (spread at its 252d 2nd pctile, N=61 declustered) returns
  +28 bps fwd 5d against an unconditional AAPL-QQQ spread drift of +37 bps.
  The extreme-rank cell (AAPL rank < 10, QQQ > 85) has N=4 in 26 years.
  Extreme single-name-vs-index divergence in a mega-cap is not a signal.
  (scratch/pitch_bakeoff/2026-08-06/fable_high/checks/aapl_qqq_rv.py)
