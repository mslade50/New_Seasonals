# Thread: the deleted ML model

Status: unposted
Source: CLAUDE.md negative-results section (ML meta-labeling layer, deleted 2026-08-07)
Voice: dry practitioner. The flagship thread; post it early but not first week.

---

1/
We built an ML model that predicts our trades' win rate. It works. Realized
win rate climbs from 45% to 70% across its probability deciles. Calibration
beats base rate.

We deleted it last week. Here's why.

2/
The model's job was meta-labeling: score every signal the system stages,
skip the ones likely to lose. Standard playbook. Two full walk-forward
evaluations, months apart, different feature sets.

Both said the same thing: no ship.

3/
The reason is the interesting part. Win probability and expectancy are
decoupled in this book. Low win probability comes bundled with bigger
winners. The bucket the model wanted to skip averaged +0.60R at a 51% win
rate.

It wanted to skip our breakout trades at the bottom of their range.

4/
We tried to rescue it. Added 8 features orthogonal to the entry rules:
put/call, sentiment surveys, analyst momentum, earnings distance.

Uplift: -0.015R. Bootstrap CI containing zero. 7 of 15 years positive.
Nothing.

5/
The lesson isn't "ML doesn't work in trading." The model was genuinely
good at its assigned task. The lesson is that the task was wrong: win rate
is not the thing. Any filter that makes you more selective has to prove it
improves EXPECTANCY, and in a book where losers are small by construction
and winners are fat, selectivity mostly trims winners.

6/
So it ran advisory-only for two months with nothing consuming its output,
and then we deleted it. Code, tests, pipeline, the trained artifact. All of
it. A model nobody acts on is a maintenance bill with a dashboard.

The one thread we kept open: predicting RISK (P of a 1R adverse excursion)
instead of wins. That one had signal and an actual use.

7/
If you take one thing: before you build the win-rate predictor, check
whether win rate and expectancy even point the same direction in your
system. Ours don't. By design. The whole book is built on taking trades
that look bad and pay well.
