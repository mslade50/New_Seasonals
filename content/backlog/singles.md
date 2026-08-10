# Singles and short threads

One block per post. `---` separates parts of a multi-part short. Mark the
Status line with the date when something goes out. All numbers verified
against the repo at write time (2026-08-10); re-check anything stale before
posting. Dollar figures deliberately absent: R and percentages only.

## Discipline / negative results

### S1. The kill file
Status: unposted

Our research pipeline keeps an append-only file of every idea it killed,
with the reason. 305 lines and growing. The pipeline is banned from
re-pitching anything in it without stating what's structurally different.

Best week so far: 28 candidates in, 28 dead. The file is the product.

### S2. Small N is a grade, not a kill
Status: unposted

"Not significant, N too small" is a banned kill reason in our shop. Markets
produce small samples by construction: a cycle-year cell yields one
observation every four years. Demanding N=50 selects for stale regimes, not
safe ones.

Small N with a mechanism = tradeable, at C-grade size.

### S3. The one-survivor morning
Status: unposted

Our idea pipeline once produced 17 candidates and killed 16. The publishing
rules at the time allowed exactly two outputs: a full slate or nothing. A
real result had no legal way to ship.

We rewrote the rules the same day: partial slates allowed, but every empty
slot must name two kills that earned it. If there is one idea, we want to
see the idea.

### S4. Making "nothing today" expensive
Status: unposted
---
Part 1:
A research pipeline that can say "no trades today" will eventually say it
every day. It's the easiest output.

So we made standing down MORE expensive than shipping: a no-trades verdict
requires 8+ candidates examined across 4+ asset classes, 6+ named kills
with reasons, and every near-miss documented with the number that would
turn it on.
---
Part 2:
First all-kill morning under the new rules produced 50 check scripts to
justify shipping nothing.

That sounds insane until you realize the alternative: a quiet morning and a
broken pipeline look identical from the outside. Silence has to be
distinguishable from failure, and proof of work is the only way.

### S5. We shipped a rule we know loses money
Status: unposted

Three rules in our book are documented as expectancy-NEGATIVE on the full
backtest: a first-signal size discount, a gap-up half-sizing, and a
scale-out split. Each bought something else: smaller footprint, smaller
tail, smoother short-book variance.

"It costs expectancy by design" is a sentence most quant content will never
say. Risk appetite is a real input. Pretending every rule is a PnL
optimization is how you end up lying to yourself about why it exists.

### S6. In-sample rules flatter themselves
Status: unposted

How NOT to validate a new sizing rule: re-run the backtest with the rule on
and admire the improvement. The rule was fit on exactly that history. Of
course it improves it.

Our standard instead: leave-one-year-out stability plus episode clustering
(events, not days). A rule that only helps because of one year, one
cluster, or one regime fails loudly there. Several of ours have.

### S7. The graduation that didn't trade
Status: unposted
---
Part 1:
Two calendar anomalies ran the same frozen validation gate this summer.

One passed everything: 288 events, +0.33% per event, t = 3.5, positive in
20 of 25 years, bootstrap P(mean<=0) = 0.0001. It trades.

The other ALSO passed statistically. We rejected it anyway.
---
Part 2:
Reason: 58% of its executable events overlapped a position an existing
strategy would already have on. Same exposure, same days, second name.

A portfolio is not a museum of significant results. Redundancy is a kill
reason even when the t-stat is fine. That check has to run against your
actual book, which is why nobody selling signals will ever run it for you.

### S8. When the correction does NOT apply
Status: unposted
---
Part 1:
An automated checker once killed one of our pre-specified hypotheses by
building a 47-cell grid AROUND it, scoring our one cell against the
best-of-47 null, and reporting family-wise p = 0.90 against our
pre-specified p = 0.011.

We overruled the checker. It was wrong, and the reason matters.
---
Part 2:
A multiplicity correction prices the cost of a SEARCH. It applies to cells
a search found. It does not apply to a hypothesis specified before any
grid existed, scored against a grid someone else built later.

Correct for the searches you ran. All of them. And only them.

### S9. The parked trade
Status: unposted

Found this summer: a rates long off a macro-print close, conditioned on the
instrument sitting near its 52w low. N=25, 76% hit, +0.54% vs +0.05% own
drift, every leave-one-year-out t positive, declustering IMPROVED it,
dropping the two best years improved it.

Then the cycle-year split: all of the edge lives in non-midterm years.
This is a midterm year. Parked, with the number that unparks it written
down. The process that found it killed it, and that's the system working.

## Bugs and data integrity

### S10. The strategy that never lost more than 1R
Status: unposted
---
Part 1:
For years our site showed a strategy that never lost more than 1R. Clean
riskless-looking exit distribution. It was a bug.

The engine booked every stop exit at exactly the stop price. Gaps through
the stop didn't exist. Every stop-out pinned at precisely -1.0R.
---
Part 2:
Fixed fill model: worse of stop and open, plus slippage, plus extra when
the bar gapped through. ~20% of 434 stop-outs had gapped. Book total
dropped 46R. Worst trade went from -1.0R to -4.6R.

Live trading was already correct the whole time (a broker stop becomes a
market order at the gap). The fix removed nothing but optimism.

### S11. Three ways a price cache lies
Status: unposted
---
Part 1:
Three real incidents from one price pipeline:

1. A free API returned a stale pre-market bar and silently zeroed an
entire scan tier. No error. Just no signals that morning.
---
Part 2:
2. The nightly close-pull cron was set in UTC. Half the year that landed
BEFORE the US close, appending the in-progress bar as the canonical daily
close. Correct all summer, wrong after the clocks changed.

3. A halted ETF sat in the cache with flat phantom prices for four years,
which would have ranked it a top momentum name on 0% "returns."

Cache-first, exclude-today on morning pulls, and hard-drop dead listings.
Every one of these was silent until hunted.

### S12. Green checkmarks over a dead output
Status: unposted
---
Part 1:
A weekly report of ours shipped on time, every week, for three weeks,
containing a section frozen at the same date. The job that refreshed that
section had been disabled. The job that SENT the report was green.

Success signals must be about the artifact, not the process.
---
Part 2:
Same family: a deploy chain triggered "after" another workflow that
silently never fired. An email gated on hour == 16 that dropped two days
of sends when the scheduler ran an hour late.

Now: senders verify content freshness, deploys run in the same job as the
thing they deploy, and time gates key on which cron fired, failing OPEN.
The checklist that matters is "did the thing exist," not "did the job run."

### S13. Survivorship, quantified on our own book
Status: unposted

Our 23-year backtest only trades tickers that exist in TODAY'S universe
file. 21 of 22 major 2020s delistings are simply absent from history.

We can't fully fix it yet, so it's written at the top of the ledger doc as
a standing caveat: small-cap dip-buy stats are an upper bound, and nothing
gets sized off them alone. Knowing which of your numbers are inflated beats
pretending none are.

### S14. The 7-point drift
Status: unposted

Our risk dial rebuilt its whole history from scratch on every run. Data
revisions drifted recomputed vintages up to 7 points on a 0-100 scale
whose live threshold is a hard cut.

Now the series is append-only: history frozen point-in-time, only new dates
append, every append stamped with the weights-vintage hash. Any backtest
that joins it must state which vintage it used. If your signal's history
can quietly change under you, you don't have a signal, you have a feed.

### S15. The -2.008R false block
Status: unposted

A live entry gate of ours keys on a round-number R threshold read from a
rebuilt ledger. One weekend a marginal fill flickered in a rebuild and a
strategy got blocked by 0.008R of vintage noise.

The fix wasn't moving the threshold. It was provenance: every ledger build
now embeds its source, git hash and a diff vs the prior vintage, and local
builds physically cannot overwrite the file that gates live orders.

## Sizing and risk

### S16. No cliff anywhere
Status: unposted

We replayed the whole book at 4 global risk multipliers before trusting
the one we'd already shipped. Sharpe: 1.89 / 1.87 / 1.85 / 1.83. Return
over maxDD: flat. Drawdown scaled slightly SUB-linearly (fixed caps clip
the tail).

That's what a well-behaved parameter looks like: a clean appetite dial, no
optimization pretense. If your backtest has a performance cliff at a
specific sizing knob value, the knob is fit, not chosen.

### S17. Day-2 stops
Status: unposted

We measured 81 episodes where a dip-buy would have hit its stop on entry
day: booking those cost -33R over 24 years vs arming the stop the NEXT
session. A third of one strategy's day-1 stop-outs went on to hit +2R
targets.

Dip-buy limits fill at maximum fear by construction. A stop that's live in
that same hour is a donation. Ours arm at the next open, book-wide.

### S18. Stops that destroy the trade
Status: unposted

One of our strategies fades capitulation spikes in leveraged ETFs. Adding
ANY stop, at any distance we tested, flips it from solidly positive to
deeply negative. Adverse excursion beyond 1 ATR is the normal path before
the reversal. The demanding entry is the risk control; worst no-stop trade
in 15 years: -2.9R.

"Always use stops" is portfolio advice pretending to be trade advice. The
right stop for some structures is none, sized accordingly.

### S19. The cap that only clips winners
Status: unposted

Our per-ticker concentration cap bound 3-8 times in 21 years. Every single
clipped position was a winner. Cost: ~4% of that strategy's PnL.

We keep it anyway, and the doc says why in one word: catastrophe. The cap
isn't there for the observed sample, it's there for the stack that hasn't
happened yet, in a strategy that deliberately trades without resting
stops. Insurance that never pays in-sample is not evidence insurance is
wrong.

### S20. The redundant cap
Status: unposted

We ran two stacked daily risk caps for a while. A replay showed the second
one bound on the SAME days as the first, cost 23 years of return, and
improved max drawdown by exactly nothing. Removed.

The surviving cap earns its keep in one number: it alone bounds the worst
single day at about a third of its uncapped size, and the uncapped worst
day WAS the entire max drawdown. One cap that binds beats two that
overlap.

### S21. Model follows live, for once
Status: unposted

Audit finding: our live order path never enforced one-position-per-ticker
on consecutive-day signals. It had always stacked. The MODEL was the side
enforcing a limit that didn't exist, quietly under-counting for years.

We aligned the model to live (adds R, widens one strategy's drawdown,
accepted). The lesson runs backwards from the usual one: sometimes the
backtest is the thing drifting from reality.

## Infrastructure

### S22. The 9:31 "market on open"
Status: unposted
---
Part 1:
For three months, some of our exits labeled "market-on-open" were placed
at 9:31. After the open. Every day. The staging script needed the live
opening print for a different check, so it couldn't run before the bell,
so the "MOO" orders chased the open they were supposed to make.
---
Part 2:
Fix: a standalone 9:10 task that stages true opening-auction orders
(TIF=OPG) before the exchange cutoff, with a loud fallback if it misses.
The label on an order is not the mechanics of the order. If you automate
execution, verify the fill TIME distribution, not just the fills.

### S23. Fired from a laptop
Status: unposted

GitHub's shared cron ran our pre-market jobs 1-3 hours late in the busy
UTC window. Deadline-critical morning pipeline, best-effort scheduler.

Fix: a local scheduled task fires the cloud workflows via API dispatch
(near-zero queue lag), and each workflow keeps a fallback cron that
checks whether the dispatch already succeeded and stands down. Belt,
suspenders, and a written failure ladder for when both break. Free-tier
infra is fine; UNMONITORED free-tier infra is not.

### S24. Aligned sites
Status: unposted

Every rule in our book doc ends the same way: a list of the exact files
that must change together (config, engine, scanner, order layer, guard
test). Fifteen-plus sections, each with its list.

It exists because a rule that lives in four places drifts in four places.
We once found live trimming size 20% below what the ledger modeled, for
weeks. The heuristic: if you can't enumerate the aligned sites, you don't
understand the change yet.

### S25. Memory with an expiry date
Status: unposted

Our negative results carry re-examination triggers, not just verdicts:
"revisit at 20 more trades in this cell (~2029)." "Re-examine at 2 new
episodes." One arrives ~2039 at current signal rates.

Dead ideas keep accruing evidence in shadow files while switched off, so
the revisit is a comparison, not an argument. Institutional memory that
can't say WHEN it should be questioned is just dogma with receipts.

### S26. The fitted weights were cosmetic
Status: unposted

We shadow our fitted risk composite with a dumb version: equal weights,
same signals, no calibration. Correlation with production: 0.85. Gate
agreement: 89%.

That shadow accrues silently toward a swap decision years out. If a naive
version of your model does 90% of the work, the sophistication is mostly
decoration, and it's better to learn that from your own shadow file than
from a drawdown.

### S27. The placebo that killed a premise
Status: unposted

Tested: "8:30am macro releases resolve overnight, so the overnight session
should carry the event premium. The 2pm Fed print should show none."

Result: the LARGEST overnight premium in the study was Fed day. And the
8:30 prints raise overnight variance while the 2pm print raises INTRADAY
variance, exactly as mechanics demand. The "overnight event premium" was a
session-of-day effect wearing an event costume. One placebo, whole premise
dead. Cheapest test we ran all week.

### S28. Cluster depth cuts both ways
Status: unposted

Same week, two lessons that point opposite directions: one signal's edge
at streak-depth 4 (that day's actual state) was -1.5% at 1-for-9 while
depth-1 paid +0.4%. A different signal paid +1.6% at depth >2 vs +0.3%
shallow.

Rule we wrote down: measure the depth bucket you are actually IN and
quote it. Staleness is an empirical property, not an assumption, and it
doesn't have a consistent sign.
