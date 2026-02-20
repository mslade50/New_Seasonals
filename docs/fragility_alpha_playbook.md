# Fragility Alpha Playbook

Signal-driven directional trades using the risk dashboard's fragility score.
Two strategies: 21-day (intermediate) and 63-day (long-term).

Based on backtested persistence studies and episode analysis (N=14 episodes, 2016-2026).

---

## Strategy 1: 21-Day Fragility Alpha

### Edge Profile

| Metric | Value | Source |
|--------|-------|--------|
| Threshold | >= 70 | Event study |
| Mean fwd 21d at threshold cross | -3.45% | 4 completed events |
| Edge vs baseline | -4.78% | |
| Hit rate (negative 21d) | 75% | |
| Persistence amplification | Days 4-10: -5.08% edge, 67% hit rate | Persistence study, N=12 |
| Early signal (days 1-3) | -3.27% edge, 54% hit rate | Persistence study, N=13 |

### Entry

**Trigger:** 21d fragility crosses above 70.

**Scaling plan:**

| Timing | Action | Risk Budget | Rationale |
|--------|--------|-------------|-----------|
| Day 1 (cross above 70) | Enter starter position | 1% of portfolio | 54% hit rate — accept uncertainty |
| Day 4-5 (still >= 70) | Add to position | +0.5-1% of portfolio | Persistence confirmed, 67% hit rate |
| **Total max risk** | | **1.5-2%** | |

**Do NOT enter if:**
- 21d fragility is already decaying from a prior spike (you missed the window)
- SPY is already -3%+ from recent highs (the move may have happened)

### Structure

**Core:** Put debit spread, ~$15-20 wide, ~30 DTE (monthly expiry).
- Buy near-ATM put (~40 delta)
- Sell put $15-20 lower

The spread width should capture the mean expected move (-3.5% of SPY = ~$24 at current levels).
Tighter spreads have better risk/reward ratios but cap out sooner.

**Optional tail kicker (0.25-0.5% risk):** Far OTM puts (~5 delta), same expiry. Only if VIX is low and they're cheap.

### Holding Rules (Days 1-10)

**Do nothing on signal noise.** The episode analysis showed signals flicker on/off during every episode, including real warnings. Individual signal count changes are NOT exit triggers.

Specifically ignore:
- Signal count dropping by 1-2
- Fragility dipping below 70 for 1 day then recovering
- SPY rallying 1-2% (happened in every real warning before the correction)

### Exit Rules

| Priority | Trigger | Action | Rationale |
|----------|---------|--------|-----------|
| 1 | SPY -3% to -4% from entry | Close spread | Mean outcome achieved |
| 2 | Day 10, SPY flat/up (within -1%) | Close spread | Edge window closed per persistence data |
| 3 | Day 10, SPY -1% to -2% | Extend hold to day 15 max | Move started but hasn't reached target |
| 4 | Day 15 (hard stop) | Close regardless | No further holding justified |
| 5 | Fragility < 70 for 3+ consecutive days within days 1-3 | Close starter | Signal died before confirmation |

**Time value recovery:** A 30 DTE put spread loses minimal theta in the first 10 days. At day 10 with SPY flat, you recover ~90% of premium by selling. This is the built-in stop loss.

### What NOT to do

- Don't exit because a single signal turns off (noise in every episode)
- Don't hold past day 15 hoping for a late break (the 21d persistence data shows the edge is days 4-10; beyond that you're paying theta for nothing)
- Don't use 21d fragility < 70 as a mechanical exit if SPY has already dropped 2%+ (that's the signal working — the regime multiplier compresses the score as SPY falls)

---

## Strategy 2: 63-Day Fragility Alpha

### Edge Profile

| Metric | Value | Source |
|--------|-------|--------|
| Threshold | >= 70 | Event study |
| Mean fwd 63d at threshold cross | -3.28% | 5 completed events |
| Edge vs baseline | -7.16% | |
| Hit rate (negative 63d) | 60% | |
| At 90+ fragility | -4.46% mean, -8.3% edge, 68% hit rate | N=31 daily observations |
| Days 1-5 persistence | -6.74% edge, 59% hit rate | N=65 |
| Days 6-15 persistence | -5.93% edge, 51% hit rate | N=61 |
| Days 16+ persistence | -0.12% edge, 14% hit rate | N=7 (signal dead) |

### Entry

**Trigger:** 63d fragility crosses above 90 AND Distribution Dominance actively firing.

The 70-88 range includes false alarms (Apr 2021, Aug 2024 patterns). The 90+ level has been a real warning in 5/5 completed historical episodes. D/A actively firing is the strongest differentiator: 3/3 completed D/A episodes were real warnings.

**Additional entry filter:** SPY within 2% of 52w high. You want to enter while the market is still complacent. If SPY is already -5%, the regime multiplier is compressing the score and the easy money in the put spread is gone.

| Timing | Action | Risk Budget | Rationale |
|--------|--------|-------------|-----------|
| Day 1 (cross above 90 + D/A active) | Full entry | 1.5% of portfolio | High-conviction entry, 68% hit rate at 90+ |
| Same day | Tail kicker | 0.5% of portfolio | Crash convexity |
| **Total max risk** | | **2%** | |

No scaling — the 63d edge is frontloaded (days 1-5 strongest). Waiting costs edge.

**Do NOT enter if:**
- D/A is not actively firing (DL + Low AR alone = false alarm pattern)
- Fragility peaked at 90+ but has settled to 70-88 for 5+ days (the "muddled middle")
- SPY is already -3%+ from highs

### Structure

**Core (1.5% risk):** Put debit spread, ~$30 wide, ~3 month expiry (quarterly).
- Buy near-ATM put (~40 delta)
- Sell put $30 lower
- Max payoff ~$30/share (1.8x on ~$11 cost)

**Tail kicker (0.5% risk):** Far OTM puts (~5 delta), same expiry.
- At $240/contract, buys significant convexity
- Only pays in a -20%+ crash scenario
- Treat as sunk cost / lottery ticket

### Holding Rules (Days 1-15)

Same principle as 21d: **do nothing on signal noise.**

The episode analysis showed that in EVERY real warning (Q4 2018, Jan 2020, Dec 2021):
- Signals flickered on/off while SPY was still flat/up
- SPY often rallied 1-3% in the first 5-10 days before correcting
- Fragility sometimes dipped below 70 for a day then recovered
- The correction came 2-4 weeks after entry

Signal count changes during the holding period are NOT actionable. The signal is a one-shot entry trigger with a time-limited edge.

### Exit Rules

| Priority | Trigger | Action | Rationale |
|----------|---------|--------|-----------|
| 1 | SPY -4% to -5% from entry | Close spread | Mean backtested outcome achieved |
| 2 | SPY -8%+ from entry | Close spread, sell half tail kicker | Spread near $30 cap; tail kicker is now the trade |
| 3 | Day 15, SPY flat/up (within -1%) | Close spread | Edge exhausted: 14% hit rate after day 16 |
| 4 | Day 15, SPY -1% to -3% | Hold to day 20 max | Move started, give it 5 more days |
| 5 | Day 20 (hard stop) | Close spread regardless | No further holding justified |

**Tail kicker exits:**
- Hold through ALL spread exits. The $3k is sunk.
- Only sell if SPY crashes (monetize half, ride half)
- Otherwise let expire worthless
- Tail kicker payoff is uncorrelated with the fragility signal — it's pure crash insurance

### Time Value Recovery

3-month put spreads decay slowly in the first 15 trading days:

| Exit day | Spread recovery (SPY flat) |
|----------|---------------------------|
| Day 5 | ~99% of entry premium |
| Day 10 | ~98% |
| Day 15 | ~97% |
| Day 30 | ~91% |
| Expiry | 0% (if OTM) |

This means the time stop at day 15 costs almost nothing if the signal fails. Your real max loss in the false alarm scenario is the tail kicker premium (~0.5% of portfolio).

---

## Episode Taxonomy (Historical Reference)

### Real Warnings (preceded -10% to -15% drawdowns)

| Episode | Start | Peak Frag | Signals | Fwd 63d | Pattern |
|---------|-------|-----------|---------|---------|---------|
| Q4 2018 | 2018-09-18 | 93.7 | D/A + DL + SRD | -11.9% | Accelerated past 90, SPY flat for 10 days then broke |
| Jan 2020 | 2020-01-22 | 96.0 | DL + VRC + Low AR | -15.3% | Hit 96 immediately, correction came 4 weeks later |
| Feb 2020 | 2020-02-04 | 90.3 | DL | -12.5% | Continuation, SPY rallied 2.5% before COVID crash |
| Dec 2021 | 2021-12-09 | 100.0 | D/A + DL | -9.6% | Hit 100, SPY flat for 5 days then -10% over next 2 months |

**Common pattern:** Fragility accelerated quickly past 90. SPY was flat or UP for the first 5-10 trading days. The correction started 2-4 weeks after the signal, not immediately.

### False Alarms (markets recovered)

| Episode | Start | Peak Frag | Signals | Fwd 63d | Pattern |
|---------|-------|-----------|---------|---------|---------|
| Apr 2021 | 2021-04-20 | 100.0 | DL + Low AR + VRC | +4.9% | No D/A. Fragility elevated but SPY drifted up throughout. |
| May 2021 | 2021-05-20 | 87.8 | DL + SRD | +6.3% | Single signals flickering, never consolidated |
| Aug 2024 | 2024-08-16 | 83.7 | DL only | +7.4% | Never reached 90, single-signal episode |
| Oct 2024 | 2024-10-08 | 83.7 | Low AR only | +3.2% | Never reached 90, single-signal episode |

**Common pattern:** Fragility persisted in the 70-88 range but never accelerated past 90 (except Apr 2021, which lacked D/A). Single signal or DL + Low AR dominating. SPY drifted up throughout.

### Distinguishing Features

| Feature | Real Warning | False Alarm |
|---------|-------------|-------------|
| Peak fragility | 90+ (always) | 70-88 (usually) |
| D/A actively firing | Yes (3/3) | No (0/4) |
| Signal count at entry | 2-3 | 1-2 |
| SPY action during episode | Flat, then breaks | Drifts up |
| Time to correction | 2-4 weeks after entry | Never |

---

## Key Lessons

1. **Signal count changes during holding are noise.** Every real warning had signals flickering. Don't exit because one signal turns off.

2. **Fragility dropping because SPY fell is the signal WORKING.** The regime multiplier compresses as SPY moves away from highs. A drop from 90 to 70 while SPY is -5% is bullish for the trade, not a reason to exit.

3. **The "muddled middle" (70-88 persistent) is the false alarm zone.** If fragility settles here for 5+ days without accelerating past 90, the episode is more likely to resolve upward.

4. **Time is the exit, not signals.** Hold for 10-15 trading days (21d strategy) or 15-20 trading days (63d strategy), then exit regardless.

5. **Sample sizes are small.** 4-5 completed events for threshold crosses, 12-13 observations per persistence bucket. Size accordingly — quarter-Kelly at most.

6. **The 5d horizon is not tradeable as alpha.** Signal edges are 20-30 bps, which doesn't cover options transaction costs. Useful as a timing indicator within longer trades only.
