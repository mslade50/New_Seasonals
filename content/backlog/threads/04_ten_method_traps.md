# Thread: ten method traps that killed 44 ideas in four days

Status: unposted
Source: data/pitch_negative_registry.md (method-trap sections)
Voice: teacher. Can also run as a 10-part daily series (one trap per post);
if run as a series, post trap 1 and trap 4 first, they are the strongest.
Tickers swapped for asset-class descriptions where the registry named them.

---

1/
Our idea pipeline killed 44 candidate trades in four days last week. Most
died the same ten ways. Every one of these traps produced a result that
LOOKED tradeable. Thread:

2/
Trap 1: lag-0 forward returns on a close-entry idea. If the order enters
at the NEXT close, measuring from the signal close credits you a session
you cannot trade. Fixing the lag on one pair took the t-stat from 2.30 to
1.39. The entire "edge" lived in the untradeable day.

3/
Trap 2: day-level t-stats on overlapping triggers. A state that persists
for days is one observation, not five. Declustering flipped one commodity
ETF cell from +1.02% to -0.64%. Another from +4.41% to -2.80%. If your N
counts days, your t-stat is fiction.

4/
Trap 3: the control is never zero. The instrument's own unconditional
drift over the same horizon is the bar. "Gold up 4 bps into the print"
sounds like an edge until you notice gold's baseline over any 2 days is
+9 bps. We killed an August seasonal that underperformed a RANDOM 10-day
long in the same window.

5/
Trap 4: build the trading-day-of-month control first. A bond ETF's own
unconditional 3-day return swings from -0.20% (t -2.2) early-month to
+0.22% (t +2.6) mid-month with NO event anywhere. Every macro release has
a fixed day-of-month footprint. We watched a whole "CPI works, NFP
inverts" pattern dissolve into that calendar profile with event labels on.

6/
Trap 5: beta-neutralize before crediting a spread. Miners-minus-metal on
a thrust trigger paid +0.79% per episode equal-dollar. At the measured
beta of 1.8, the same episodes pay 0.00%. Report the regression beta, not
the correlation.

7/
Trap 6: a nested subset that reverses its parent's sign is a partition of
noise. Cell pays -0.23%, deeper cut -0.23%, deepest cut +0.67%. The
deepest is a subset of the second, so the sliver between them must run
about -1.2%. When "today's reading" lands exactly in the only positive
slice, you found the mining artifact, not the edge.

8/
Trap 7: check N before checking edge. One conditional cell had occurred
ZERO times in 318 events since 2000. Another twice in 24 years.
Unmeasurable is a kill, not a pass.

9/
Trap 8: an era cut that isolates one macro episode. "2018+" sounded like
an era split and was actually a fence around 2021-22: 8 of 12 episodes in
those two years, t = 0.16 without them, and the same trigger LOST before
2018.

10/
Trap 9: a rank gate is not a magnitude gate. A yield-move trigger at the
85th percentile of its own history bought a move at the 3.8th percentile
of the winning episodes' magnitude distribution. Percentile of occurrence
and size of move are different axes. Gate on the one that pays.

11/
Trap 10: post-hoc sign flips. A result found while hunting its OPPOSITE
carries sign x era x horizon comparisons before any threshold grid.
Nominal t = 2.18. Priced for the implicit search: p roughly 0.5. A coin.

12/
All ten of these came out of a pipeline whose job is to kill its own
ideas before breakfast. The kill file is append-only and the pipeline is
banned from re-pitching anything in it without stating what changed.

Most of the edge in research is subtraction.
