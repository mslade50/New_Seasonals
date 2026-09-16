# Legend SPY/QQQ integration — September 16, 2026

The ETF-only research and existing IBKR runner are now connected. The default
producer uses raw SPY/QQQ RTH history; it never creates a Databento client or
requests a futures contract. The strategy version is `legend-etf-spy-qqq-v2`.
An old plan, paper proof, runtime version, or deployment manifest cannot silently
enable the changed strategy.

The retained original source is `research/legend_ema_backtest.py`. It supplied
the 75% body/range and inclusive no-EMA-touch setup. The runner retains 09:31
entry, causal target revisions at 09:46/10:01/10:16, and the 10:30 exit. The old
09:30-to-close research returns are not performance evidence for this timing.
The user was offered both timings; implementation used the existing runner's
timing as the stated default while awaiting a different preference.

The new signal evaluator uses a deterministic 20-calendar-day window, requires
200 pre-setup warmup bars and an exact exchange-calendar grid, and rejects
partial prior/entry sessions. It does not borrow futures contract-roll rules.
The runner re-fetches and compares the raw OHLC digest before loading any
qualified ETF into execution. Missing, stale, malformed, or revised inputs
block the session. Only SPY and QQQ can create new native trade contexts.

Read-only broker check on September 16 succeeded using the existing Primary
TWS. Complete 26-bar September 15 setups were available: SPY body/range
0.642857 and QQQ 0.700000, so neither qualified for September 16. The first
attempt tripped the existing broker clock-skew check; a repeat passed without
changing its threshold. No orders or paid data requests were made.

Historical signal comparison over 2012–August 2026 matched both implementations:
SPY 3,270 evaluated sessions / 132 signals; QQQ 3,245 / 183. There were zero
qualification, EMA, or ratio differences. Missing archived grids blocked
352 SPY and 377 QQQ sessions. Both evaluators used identical bounded histories;
this proves rule integration, not the original full-history return series.
Final-source evidence must be generated in the stable checkout for deployment.

The existing broker safety tests and 342-candidate execution replay are retained.
The deployment manifest now requires native candidate proof and includes the
new runtime source. Futures parity is still available for research but cannot
satisfy the new deployment gate. Shared account/symbol reservations, portfolio
capacity checks, dated live switches, and broker paper-proof checks remain.

Remaining live rollout requirements are actual shadow-session observations,
paper execution proof (including partial fills and OCA/timed exits), and the
shared-executor capacity producer/integration attestation. Automated unit and
historical tests do not substitute for those broker observations. No live date
or execution direction is enabled by this source change.
