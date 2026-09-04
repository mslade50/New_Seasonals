# Negative-registry additions owed from 2026-08-06 (bake-off variant opus_xhigh)

Bake-off run, so `data/pitch_negative_registry.md` was NOT touched. These are
the kills from today that carry a reusable lesson and would be appended on a
real morning, written in the registry's own `section / key / why_dead` shape.

---

### Hedging and volatility vehicles

**key:** SVXY on VIX/VIX3M contango plus SPY-at-highs carry

**why_dead:** the trigger is a 25%-of-tape cell that *underperforms* SVXY's own
unconditional drift at every horizon, and is negative outright in the -0.5x era
that actually exists (5-session mean -0.25% against a +0.28% baseline; 41
declustered episodes, t = -1.69). All 15 cells of the ratio x near-high
threshold grid are negative post-2018, and all of SVXY's post-2018 drift lives
in the complement of the cell (t = 4.11). The trigger was continuously ON
through 2018-01-22..26, ten sessions before the -1x product lost 90%: steep
contango at index highs is the precondition for the vol event, not protection
from it. Companion to the existing "SVXY as a pre-FOMC leg" entry, killed on a
worse ground than correlation.
(scratch/pitch_bakeoff/2026-08-06/opus_xhigh/checks/c1_svxy_carry.py)

---

### Calendar and factor seasonality

**key:** pre-CPI drift in equities, bonds, gold or the dollar

**why_dead:** 24 pre-specified cells (4 assets x 3 entry offsets x 2 exits).
Best |t| = 2.44 (SPY entering 4 sessions before the print, exiting on the eve),
which a random event calendar beats 47% of the time on both a random-position
and a circular-shift null. That cell is a knife edge on the entry offset
(t = 0.96 / 2.44 / 1.25 at 3 / 4 / 5 sessions), loses ~40% of its size to a
trading-day-of-month-matched control, and its payroll-contaminated subset
(24% of history) carries a third of the edge with no significance. TLT is
negative in the 2019+ era, GLD's gap-21 episodes are wrong-signed at t = -2.00,
and DX is flat in all six of its cells. The registry's existing "post-CPI vol
crush" entry now has a pre-CPI sibling.
(scratch/pitch_bakeoff/2026-08-06/opus_xhigh/checks/f1_pre_cpi.py)

**key:** macro-print density as an equity signal

**why_dead:** "index within 1% of its 52-week high AND >= 3 tier-1 prints in the
next 5 sessions" has a signal-day t of 0.04 to 0.82 before any attack. The
trigger's trading-day-of-month distribution is concentrated on days 2-5 (mean
3.2 versus 11.0 for near-high days generally), so it is a calendar-position cell
in a macro costume; against a day-of-month-matched control the density adds
-0.22 to +0.29 in t. The print ladder is not monotone where it matters (within
near-high days, density 2 gives t = 3.80 and density 3 gives t = 0.04), which is
not what a real "more prints, more drift" mechanism looks like.
(scratch/pitch_bakeoff/2026-08-06/opus_xhigh/checks/e3_macro_density.py)

---

### Risk-dial conditioning

**key:** fragility-dial VELOCITY (as opposed to level)

**why_dead:** the book conditions on dial level only, and velocity adds nothing
over it. Cell "10d-MA-63d rose >= 25 points over 21 sessions AND SPY within 1%
of its 252d high": 102 signal days collapse to 11 episodes, 10 of them graded.
One four-day episode (the 2020-02-14 COVID top) carries the entire effect;
dropping it flips 5, 10 and 21 sessions from negative to positive. A +-7 point
threshold move, which is exactly the documented recompute-vintage drift, flips
the sign (threshold 18 gives t = +1.69, threshold 32 gives t = -2.67) and halves
N. HAC-OLS on near-high days puts all significance on level (t -2.7 to -3.0) and
none on velocity (t -0.79 to -1.27). Also note `data/rd2_fragility_ts.parquet`
ENDS 2026-05-07 and cannot serve a live-dated study; `rd2_fragility.parquet` is
the only series that reaches today.
(scratch/pitch_bakeoff/2026-08-06/opus_xhigh/checks/k6_dial_velocity.py)

**key:** equity put/call VELOCITY (the fear-to-complacency "flip")

**why_dead:** requiring the 10-day MA percentile to stay elevated while the daily
reading collapses into the bottom decile adds nothing to the plain bottom-decile
cell, and at 5 and 10 sessions the flip half is the WEAKER half (flip episode
t 0.41 / 0.34 versus no-flip 2.57 / 3.04). A one-business-day shift in the feed
lag moves the 3-session t from -0.15 to +1.31. Companion to the existing
"Equity P/C Complacency 63d dial candidacy rejected" entry: the level is
display-only for a reason and the velocity is not better.
(scratch/pitch_bakeoff/2026-08-06/opus_xhigh/checks/e2_pc_flip.py)

---

### Strategy-structure dead ends

**key:** EEM violent surge off a dead base (5d rank >= 95 + 63d rank <= 15 + above 200d)

**why_dead:** the "dead base" leg is destructive rather than decorative, taking
the surge cell from +0.16% to -0.56% at 5 sessions and +0.35% to -1.26% at 10.
N = 18 days / 9 episodes in 23 years, excess versus SPY -0.48% at 5 sessions on
a 31% hit rate, 14 of 16 gradeable observations pre-2012, LOYO floor t = 0.02,
and EFA and FXI do not confirm. Day-2+ entries, which is where a next-morning
MOO lands, run -1.62% at 5 sessions and -3.11% at 10, t = -2.04.
(scratch/pitch_bakeoff/2026-08-06/opus_xhigh/checks/d1_eem_turn.py)

**key:** trading a defensive washout against the index (long XLU / short SPY)

**why_dead:** the short-XLU leg is a measured drag, not a hedge. Inside the
washout cell XLU beats its OWN unconditional drift at every horizon, so plain
long SPY returns more than double the pair at identical volatility (10-session
+1.09% versus +0.48%, hit 68.5% versus 57.1%). Episode t never clears 1.4 at any
horizon, two of four regimes are negative, and the effect is largely a rates
trade: in the rising-rate subset the episode edge is +0.05% to +0.14%. The
outright long-XLU version fails separately: declustered on an executable entry
the lift over XLU's own drift is +0.097% at 21 sessions (Welch t 0.18) and
negative at 3, 5 and 10.
(checks/k1_xlu_spy.py, checks/g1_xlu_outright.py)

---

### Sizing and accounting methodology

**key:** quoting a signal-day t on overlapping windows

**why_dead:** eleven of nineteen candidates on 2026-08-06 died at exactly this
step, and two arrived with headline numbers that were pure overlap inflation.
Working rule established that morning: assume the episode t is roughly
`t / sqrt(days per episode)` until proven otherwise, and always report TWO
decluster gaps, because on one cell that day gap 10 and gap 21 disagreed by 5.5
t-units on the same 27 observations, and the single-gap version would have read
as a grade A.

**key:** quoting a close-to-close edge for a next-morning entry

**why_dead:** a pitch written pre-market enters at the NEXT OPEN, so the
overnight gap after the signal is not capturable. Measured the same morning:
EEM's 3-session cell went from +0.81% close-basis to +0.23% on the executable
basis (three quarters of the edge was the gap); SLV/GLD's 3-session t fell from
2.09 to 1.58 and its 5-session t from 0.94 to 0.25; XLE gave up 22% / 17% / 10%
at 3 / 5 / 10 sessions. Any candidate quoted close-to-close must be re-run on a
next-open entry before it is graded.
