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

## 2. Index and sector residual is mandatory

Report the return net of SPY and net of the sector ETF, with its own hit rate
and sign test. A seasonal whose residual is zero is an index seasonal and must
be pitched as one or killed.

## 3. Cycle cells: drop-best, drop-two-best, and the non-cycle cohort

A cycle conditioner that adds nothing over the non-cycle cohort, or that dies
when its best one or two years are dropped, is dead.

## 4. Regime branch

Split the history by SPY within 2% of its 52-week high at entry and by the
fragility dial where the PIT history allows. A pattern whose edge lives only
in the branch today is NOT in is dead for today.

## 5. Recency

The last 10 years individually, with max adverse excursion in ATR. A 25-year
record with a coin-flip last decade is grade C at best.
