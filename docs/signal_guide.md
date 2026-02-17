# Risk Dashboard V2 — Signal Guide

The executive summary answers three questions about market fragility. Each question is backed by specific, measurable signals that flip ON or OFF. When a signal is ON, the dashboard explains exactly why.

---

## How to read the dashboard

The summary has three sections:

1. **Price Context Banner** — SPY price, 12-month return, extension vs 200-day moving average, drawdown from 52-week high, and a plain-English regime label (e.g. "Healthy uptrend", "Correction underway").

2. **Three Questions + Risk Dial** — The core of the dashboard. Each question groups related signals. A question shows CLEAR (no signals active), WATCH (one active), or WARNING (two or more active). The risk dial on the right synthesizes everything into a single 0–100 fragility score.

3. **Stored Energy** — Only appears when 2+ signals are active. Shows how much "fuel" is available for a drawdown: vol compression duration, calm streak length, and an estimated unwind range.

---

## The three questions

### 1. Is liquidity real?

These signals detect whether the market's apparent calm is genuine or artificially manufactured by systematic strategies (vol selling, risk parity, etc.). When index volatility is low but the underlying market structure is fragile, any shock gets amplified because there's no natural cushion.

#### Vol Suppression

- **What it measures:** Whether low index volatility is being driven by low diversification rather than genuine calm.
- **Trigger:** Absorption ratio below the 25th percentile AND realized volatility (22-day) below the 35th percentile.
- **In plain English:** The absorption ratio measures how much of sector variance is explained by a single factor. When it's very low, sectors are moving independently — which normally means diversification is working. But when that coincides with unusually low realized vol, it suggests systematic vol-selling strategies are artificially compressing index vol. The calm is real at the index level but manufactured, not organic.
- **What to do when active:** Be skeptical of low VIX readings. The "insurance" the market is providing (cheap puts) is cheap for a reason — the underlying structure is thinner than it looks.

#### VRP Compression

- **What it measures:** Whether the market is charging enough premium for bearing volatility risk.
- **Trigger:** Variance risk premium is negative (realized vol exceeded implied) OR VRP is below the 15th percentile of its history.
- **In plain English:** VRP = implied variance minus realized variance. Normally it's positive — you pay more for options than the vol that actually materializes. When VRP goes negative or very low, the market was recently surprised by more volatility than it expected. Options are priced too cheaply to be effective hedges.
- **What to do when active:** Hedging via options is likely mispriced. If you need protection, this is actually a reasonable time to buy it (cheap), but recognize that the market is in a state where it's underestimating risk.

---

### 2. Is everyone on the same side?

These signals detect crowded positioning and complacency. When everyone is long, leveraged, and unhedged, even a small catalyst can trigger a cascade as everyone tries to exit at the same time.

#### Breadth Divergence

- **What it measures:** Whether the index is being carried by a few large names while most sectors weaken.
- **Trigger:** SPY is within 5% of its 52-week high AND fewer than 55% of sector ETFs are above their 200-day moving average.
- **In plain English:** The index can keep making highs even as most of the market rolls over, because mega-cap names (tech, typically) mask the deterioration. This is a classic late-cycle pattern. The flag is still flying but the army is retreating.
- **What to do when active:** Don't trust the index level. Look at your individual positions — if they're in weaker sectors, the index's strength won't protect you. Consider whether your portfolio is concentrated in the same names propping up the index.

#### Extended Calm

- **What it measures:** How long the market has gone without a stress event.
- **Trigger:** Both complacency counters (days since 5% drawdown AND days since VIX > 28) are above the 70th percentile of history, OR either counter is above the 85th percentile.
- **In plain English:** Long calm streaks cause participants to gradually increase leverage, reduce hedges, and sell volatility. This isn't a timing signal — calm can persist for years. But it tells you the *magnitude* of the eventual unwind is growing. The longer the calm, the more positions have been built assuming it continues.
- **What to do when active:** This is about sizing, not direction. Recognize that when the break comes, it will likely be larger than "normal" because of the accumulated positioning. Keep hedges in place even when they feel wasteful.

#### Vol Compression

- **What it measures:** How long realized volatility has stayed below its historical median.
- **Trigger:** Realized vol (22-day Yang-Zhang) has been below its expanding median for more than 60 consecutive trading days.
- **In plain English:** When vol stays compressed for months, the entire ecosystem adapts. Risk models allow more leverage. Vol-targeting strategies scale up. Options sellers get more aggressive. VaR limits permit larger positions. All of this creates a coiled spring — the longer vol stays suppressed, the more violent the regime change when it arrives.
- **What to do when active:** The dashboard shows both the duration (days below median) and depth (how far below). Longer duration + deeper compression = bigger stored energy. This is the single best predictor of drawdown *magnitude* (not timing).

---

### 3. Are correlations stable?

These signals detect when the normal relationships between asset classes are breaking down. In a healthy market, stocks, bonds, credit, and rates move in well-understood patterns. When those patterns diverge, something is shifting beneath the surface.

#### Credit-Equity Divergence

- **What it measures:** Whether bond markets are pricing in risk that equity markets haven't acknowledged.
- **Trigger:** High-yield credit spread z-score > 0.75 (spreads widening) AND SPY's 21-day return is better than -2% (equity is flat or up).
- **In plain English:** Credit markets (especially high-yield bonds) tend to sniff out trouble 2–6 weeks before equities react. When HY spreads are widening but SPY is shrugging it off, credit is telling you something equity hasn't priced yet. Historically, this divergence resolves with equity catching down to credit, not credit recovering.
- **What to do when active:** This is one of the highest-conviction signals. If credit is widening while equity is stable, treat it as an early warning. Tighten stops, reduce exposure to high-beta names, and watch whether the divergence widens or resolves.

#### Rates-Equity Vol Gap

- **What it measures:** Whether volatility in the rates market (bonds) is elevated while equity vol remains calm.
- **Trigger:** MOVE index (bond vol) above the 70th percentile AND VIX below the 40th percentile. Falls back to MOVE > 100 and VIX < 18 if percentile data is unavailable.
- **In plain English:** Rates volatility transmits to equity volatility through dealer balance sheets. When bond dealers are stressed (high MOVE), they reduce risk capacity across all asset classes — eventually including equities. This gap tends to close by VIX rising to meet MOVE, not MOVE falling.
- **What to do when active:** The transmission mechanism is slower (weeks, not days), but persistent. If MOVE is elevated, cheap VIX is likely temporary. Not a great time to be selling equity vol or assuming equity calm will persist.

#### Vol Uncertainty

- **What it measures:** Whether the options market is pricing high uncertainty about the *path* of volatility, even if the level of vol is low.
- **Trigger:** VVIX/VIX ratio above the 80th percentile of its history (or above 7.5 absolute if percentile data is unavailable).
- **In plain English:** VVIX measures the volatility of VIX itself. A high VVIX/VIX ratio means: "VIX is low, but the market thinks VIX could move a lot from here." It's the options market saying "we don't trust this calm." This is a measure of explosive move potential in either direction — the market is pricing a regime change even if it hasn't happened yet.
- **What to do when active:** Directionally ambiguous — the break could go either way. But it means the current vol regime is unstable. Avoid strategies that need vol to stay range-bound (short straddles, calendar spreads). Favor convex positions.

---

## Risk dial

The fragility score combines signal count with price context:

```
fragility = (active_signals / 8) x 80 x regime_multiplier
```

Capped at 0–100. The regime multiplier (0.6x to 1.8x) amplifies or dampens based on where SPY is:

| Condition | Multiplier effect |
|-----------|-------------------|
| 12mo return > 25% | +0.25 (signals more dangerous at extended highs) |
| 12mo return > 15% | +0.10 |
| 12mo return < -5% | -0.15 (signals less dangerous in drawdowns) |
| Extension > 10% above 200d | +0.25 |
| Extension > 5% above 200d | +0.10 |
| Below 200d by > 2% | -0.15 |
| Within 2% of 52w high | +0.10 |
| Drawdown > 10% from high | -0.20 |

The logic: the same signals are *more* dangerous when the market is extended and near highs (further to fall, more complacent positioning) and *less* dangerous when the market is already correcting (positions already being unwound).

**Dial labels:**
- 0–20: Robust
- 20–40: Leaning robust
- 40–60: Neutral
- 60–80: Leaning fragile
- 80–100: Fragile

---

## What changed tracking

The dashboard saves signal states to `data/risk_dashboard_signal_state.json` at the end of each session. On the next load (from a different calendar day), it compares current vs previous states and shows which signals activated or deactivated. This lets you see at a glance what shifted overnight.

---

## Important caveats

- **Hit rates are placeholder estimates.** The "1-2 signals precede corrections ~30% of the time" text is a rough guess, not backtested. Phase 3 work includes running proper event studies on each signal.
- **These are fragility signals, not timing signals.** They tell you the market is *vulnerable*, not that a selloff is imminent. Fragile markets can stay fragile for months before anything happens.
- **The absorption ratio is display-only** in the scoring. It feeds the Vol Suppression signal but is not scored independently. Its chart is shown separately because the structural regime it reveals (concentrated vs diversified factor structure) is useful context even when no signals are active.
- **Sector ETF proxy.** Breadth currently uses 11 SPDR sector ETFs, not all ~500 S&P constituents. This makes the breadth signal coarser — a future upgrade will use full constituent breadth.
