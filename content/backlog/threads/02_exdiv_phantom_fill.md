# Thread: the ex-div phantom fill

Status: unposted
Source: CLAUDE.md dividend-adjustment invariant (EWZ 33.51, 2026-06)
Voice: dry practitioner. Highest "check your own code" energy; good early thread.

---

1/
A dividend went ex on an ETF we trade and our verification layer booked a
fill that never happened.

No bug in the order code. No bug in the data. Just two correct systems
using two different definitions of "price." This one costs people real
money. Thread.

2/
Setup: we had a resting limit order at a frozen dollar level. Never
touched. Weeks later, an automated check re-pulled history to verify old
fills and found a past bar whose low was below our limit.

Phantom fill. The system now believed we owned something we never bought.

3/
What happened: the re-pull used dividend-ADJUSTED bars. When a dividend
goes ex, the entire adjusted history rescales downward. A low that printed
above your limit in June can sit below it in July, retroactively, because
the series itself moved.

The order lived in as-traded dollars. The check lived in adjusted dollars.

4/
The rule we froze into the codebase afterward:

a FROZEN dollar level (a stored limit, a live working order, a ledger
entry) must only ever be compared against RAW bars.

a RELATIVE level recomputed each run (close minus k ATR off the same
series) is safe on adjusted bars, because both sides rescale together and
the comparison is exactly scale-invariant.

5/
The subtle trap inside the safe case: rounding. The moment you round the
recomputed level to cents, the two sides no longer scale identically and
the invariance breaks. Our engines deliberately do not round. It took a
while to be sure that was the ONLY thing that could break it.

6/
It took about 15 forensic scripts to pin this down, including one that
just proved the algebra. Worth it: the invariant now holds book-wide, one
regression test guards it, and every backtest engine in the repo states
which basis it reads.

If your backtest compares stored prices to yfinance history, go check
which basis you're on. Today.
