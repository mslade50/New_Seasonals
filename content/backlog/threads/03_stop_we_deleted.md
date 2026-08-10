# Thread: the stop we deleted (and the year it lost money)

Status: unposted
Source: scratch/ultracode_research/olv_stop_condition_2026-07-17.md + CLAUDE.md
Voice: dry practitioner, teacher-mode middle. EDGE-sanitized: no thresholds,
no volume multiple, strategy described by shape only.

---

1/
One of our mean-reversion strategies kept doing something stupid: stop out
at -1R intraday, then re-buy the same dip the same day.

We audited 21 years of it, deleted the stop, and replaced it with a
condition. The full arc, including the part where the new rule loses money
this year:

2/
Forensics first, 329 trades. The stop fired 118 times. 35 of those were
same-day stop-and-rebuy. Re-entry after a stop won 64% of the time at
+0.58R average.

Read that again: the thing the stop "saved" us from was the exact
condition the strategy exists to buy.

3/
Variant table, 21 years:
- production intraday stop: +0.58 avgR, worst trade -2.3R
- no stop at all: +0.73 avgR, worst trade -4.8R, ugly worst-chain
- the winner: exit only when a session CLOSES through the level on a
  volume spike vs its recent norm. +0.69 avgR, churn events 39 -> 7.

4/
Why the volume condition works structurally, not statistically: the entry
requires QUIET tape. A volume-spike exit and a fresh entry signal are
near mutually exclusive. The stop-and-rebuy churn isn't tuned away, it's
impossible by construction.

Decomposition: ~2/3 of the gain from confirming on the close instead of
the intraday touch, ~1/3 from the volume filter.

5/
Robustness before shipping: leave-one-year-out on the difference, minimum
year still +22R. Episode-clustered t = 2.4 across 83 ticker-chains.
Drop-the-best-chain still +32R. Positive in 13 of 20 years. Both
liquidity tiers gain independently.

6/
Now the honest part. 2026, the year that motivated the study? Every
loosened variant LOSES this year. And the specific frustrating positions
that started the whole investigation got 1-2R worse under the new rule.

The frustration episodes are not evidence for the rule. The other 20
years are.

7/
Postscript: we then pre-registered a widening of the exit condition
(catch grinding declines, not just single-day spikes). Fixed the adoption
threshold before running it. Result: 4 extra exits in 21 years, net
-0.6R, motivating year got worse. Rejected, as specified.

Most rules should die. The process is the edge.
