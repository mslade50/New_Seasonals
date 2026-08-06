# Event Sleeve pre-registration (2026-08-06)

Four calendar-anchored index trades from the macro event calendar work
(evidence chain: event_seasonality_sweep_2026-08-06.md + the studies listed
per trade). McKinley approved trading all four on 2026-08-06; this doc
freezes the specs BEFORE live staging. Any change to a threshold, window,
or ticker after this date is a new study, not a tweak.

Shared conventions

- All entries and exits are auction orders: MOC for close legs, MOO
  (TIF=OPG) for open legs, staged pre-market the same day by
  `event_sleeve.py` into the `Event` Sheets tab.
- Every filter uses LAG-1 data (the prior session's close), because staging
  happens pre-market. The numbers below are the lag-1 variants
  (scratch/event_sleeve_pit_variants.py), not the best-looking cells.
- No stops inside windows. Windows are 3 to 8 sessions on index ETFs; the
  window IS the risk control. Sizing is %NAV notional, GRM does NOT apply.
- Sleeve state in `data/event_sleeve_state.json` (held shares per trade,
  entry marks); re-runs are idempotent.
- Trading calendar: NYSE sessions (federal holidays minus Columbus and
  Veterans days, plus Good Friday).

## T1 FOMC_DRIFT — long SPY into non-midterm decisions

- Window: buy MOC 4 sessions before a scheduled FOMC decision (td-4 close),
  sell MOO on decision day (td0 open). Holds ~3.5 sessions, through the
  final overnight (the strongest leg: +19 bps t 3.7 alone).
- Filter: none. Years with year%4==2 are skipped entirely. The 5d-rank
  exclusion tested well on SPY but inverted on QQQ, so it was dropped
  (not robust).
- Evidence: +38.2 bps/window t 2.51 N 150 hit 0.67 (SPY 2000+); QQQ +46.5
  t 2.50. Prior: Lucca-Moench (2011) documented the pre-announcement drift
  on 1994-2011 data, so half our sample is out-of-sample for the claim.
- Sizing: 25% NAV notional. Expected ~+75 bps NAV/yr gross (8 windows).
- Worst historical window at this basis: -959 bps x 25% = ~-2.4% NAV
  (March 2020). Accepted.
- Kill / review: pause if cumulative sleeve PnL hits -2.0% NAV; scheduled
  review after 16 windows (~2 years). First live window: January 2027
  (2026 is a midterm year; T1 idles until then).

## T2 FOMC_MIDTERM_SHORT — short SPY into midterm decisions, weak tape only

- Window: sell short MOC at td-4 close, cover MOO at td0 open (same shape
  as T1, opposite side). The decision day itself drifts UP even in midterm
  years, so the short never holds through the announcement.
- Filter: SPY 21d-return percentile rank (252d window, lag-1) < 50.
  Overbought tapes are EXCLUDED: rank21>70 flips the short to a loser
  (SPY -51 bps, QQQ -123 at rank5>70). The midterm edge is weak-tape trend
  continuation, not overbought reversal. Threshold 50 is the median split,
  chosen without scanning (40 tested similar; not tuned).
- Evidence (lag-1): SPY +63.4 bps t 1.53 N 28 hit 0.54; QQQ +97.3 t 1.67.
  This is the thinnest of the four. Pilot conviction only.
- Sizing: 10% NAV notional short. ~3-5 windows per midterm year.
- Kill / review: kill on 4 consecutive losing windows or cumulative
  -1.0% NAV; review after the 2026 cycle ends (Dec 2026).
- First live window: Sep 16 2026 meeting -> entry Thu Sep 10 if SPY rank21
  at the Sep 9 close < 50. Remaining 2026 windows: Oct 28, Dec 9.

## T3 SEP_POSTQUAD_SHORT — short IWM, Sep opex to month-end

- Window: sell short MOC on September opex day, cover MOC on September's
  last session (~7 sessions).
- Filter (washout exception): skip when IWM z10 (10-session return over
  vol21*sqrt(10), lag-1) < -1. Washed-out tapes into quad expiry BOUNCE
  (the opex_reversal_study finding); the short is for calm/mild tapes
  where the post-quad drag lives. Lag-1 exception would have skipped 2001.
- Evidence (lag-1): IWM +185.0 bps t 2.27 N 24 hit 0.67; SPY variant
  +131.2 t 2.30. First-week-of-September baseline is flat, so this is the
  expiry-anchored back half, not generic September.
- Sizing: 15% NAV notional short.
- Kill / review: one window per year; kill on cumulative -2.0% NAV or 3
  losses in any 4 consecutive years; review at N+5 (2031).
- First live window: Sep 18 2026 opex -> cover Wed Sep 30 2026.
- Optional TLT long leg (+95 bps t 1.75) is NOT included: 2022 showed the
  duration regime risk and the equity leg carries the edge.

## T4 DEC_POSTOPEX_LONG — long IWM, Dec opex to year-end

- Window: buy MOC on December opex day, sell MOC on the year's last
  session (~7 sessions).
- Filter: none.
- Evidence: IWM +85.3 bps t 2.30 N 26 hit 0.65 (close basis); SPY +65.7
  t 2.19. QQQ does not confirm (-17.7), consistent with a small/value
  year-end rotation; IWM is the carrier.
- Sizing: 25% NAV notional.
- Kill / review: kill on cumulative -2.0% NAV or 3 losses in 4 years;
  review at N+5.
- First live window: Dec 18 2026 -> Dec 31 2026.

## V2 NOVDEC_VOL — long SVXY, first November session to year-end (added
## 2026-08-06 PM, McKinley approved)

- Window: buy MOC on the first November session, sell MOC on the year's
  last session. NON-MIDTERM YEARS ONLY: midterm Novembers went 1-of-3
  with both sample losers (2014 -5.4%, 2018 -13.4%); non-midterm 10 of
  11 up, +11.1% avg on the -0.5x basis. Same ex-midterm doctrine as T1.
- Instrument: SVXY (-0.5x VIX futures ETP) — short vol with loss bounded
  at the position. No stop; the bound IS the stop.
- Sizing: 5% NAV notional (2018-style repeat ~ -65 bps NAV; that year is
  now excluded by the midterm filter, kept as the sizing yardstick).
- Kill / review: kill on 2 consecutive losing windows or cumulative
  -1.5% NAV; review at N+5 non-midterm windows (~2032).
- First live window: Nov 1 2027 (2026 is midterm).
- Evidence: scratch/svxy_defined_risk_study.py + uvxy_event_study.py
  (UVXY short side t 4.5).

## V4 POSTOPEX_VOL — long SVXY, opex close to +3 sessions, ex-September
## (added 2026-08-06 PM, McKinley approved)

- Window: buy MOC on every monthly opex day, sell MOC 3 sessions later.
  TWO exclusions, both structural: September (the crush INVERTS: -65
  bps, 21% hit — that stress is T3's equity-short trade), and any opex
  while V2 already holds the Nov-Dec position (no doubling; in midterm
  years V2 is idle so V4 trades Nov/Dec normally).
- Evidence (synthetic -0.5x validated 0.9967 vs real): +108 bps/window
  t 3.55 N 164 full sample; +71 t 2.2 in the real -0.5x era (2018+);
  +134 t 3.75 hit 72% since 2021-06. ~11 windows/yr.
- Sizing: 10% NAV notional. Worst window (Aug 2015 -21.5%) ~ -2.2% NAV;
  worst year (2018, -20.6% cumulative) ~ -2.1% NAV.
- Kill / review: pause on cumulative -2.5% NAV; review after 22 windows
  (~2 years).
- First live window: Aug 21 2026 -> exit Aug 26 2026.
- Evidence: scratch/svxy_postevent_grid.py. FOMC/NFP/VIX-expiry post-
  event cells and post-CPI (faded after 2018) were tested and REJECTED
  in the same grid — do not add them later without a fresh prereg.

## Interactions and overlap audit

- T2 and T3 cannot overlap: the Sep FOMC decision precedes Sep opex every
  year in the calendar (2026: Sep 16 vs Sep 18).
- T1 and T4 are sequential in December (decision precedes opex).
- T1 runs alongside exposure_leg (both long index). Combined worst-case
  long overlay: 25% (T1) + 25% (exposure_leg) = 50% NAV for up to 4
  sessions, 8x in non-midterm years. Accepted; exposure_leg's dial kill
  rules do NOT gate T1 (no dial conditioning anywhere in this sleeve, by
  design; adding it later is a new prereg).
- T2/T3 shorts coexist with the book's short strategies (OVS etc.) under
  no shared cap. The staged-risk daily caps do not see %NAV sleeves.
  Accepted for pilot sizes; revisit if sleeve sizes grow.
- The opex washout bounce (long side) is deliberately NOT a sleeve trade:
  it overlaps the dip-buy family. Parked for a future dip-buy sizing study.

## Go-live checklist (out-of-repo steps need McKinley)

1. `event_sleeve.py` runs pre-market after the 4:17 AM parquet update and
   before order_staging (piggyback the daily_screener AM job with a step,
   or a third local trigger at ~4:40 AM). Writes the `Event` tab.
2. `order_staging.py` (OneDrive): add `load_event_rows()` mirroring
   `load_trend_rows()` — naked auction rows, appended after risk caps,
   excluded from PA. MOC rows submit as MKT + TIF=MOC; MOO rows reuse the
   OPG path. SELL_SHORT action maps to SELL with negative position intent.
3. eq_order_entry: confirm the dormant MOC path places MKT/MOC parent-only
   orders; no exit legs (the sleeve schedules its own exits).
4. Eyeball the first staged row of each trade type before letting it
   submit (house convention).
5. Sleeve PnL lands in the execution report via the orderRef strategy
   field (`EVT|<trade_id>`); no ledger integration in the pilot.

## Fixed decisions log

- 2026-08-06: windows, tickers, thresholds (rank21<50, z10<-1), sizes
  (25/10/15/25 %NAV), kill rules frozen as above. Basis: studies
  prefomc_exit_variants.py, prefomc_midterm_short_study.py,
  event_sleeve_pit_variants.py, opex_reversal_study.py,
  event_sweep_drilldown.py.
