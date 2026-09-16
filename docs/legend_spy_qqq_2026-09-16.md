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
SPY 3,477 evaluated sessions / 140 signals; QQQ 3,447 / 196. There were zero
qualification, EMA, or ratio differences. Missing archived grids blocked
145 SPY and 175 QQQ sessions. Both evaluators used identical bounded histories;
this proves rule integration, not the original full-history return series.
Final-source evidence is retained in the main workspace under
`artifacts/legend-spy-qqq-20260916/final_candidate_parity.json`. It passed
validation against the stable checkout, input hashes, and runtime.

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

## Installed shadow schedule

The stable checkout is `C:/Users/McKinley Slade/dev/new-seasonals-legend-runtime`.
All 132 unique Legend checks passed, including the native signal tests, retained
ETF research tests, and the 342-candidate execution regression. The historical
replay was run separately against the exact stable source. Ruff and workspace
hygiene checks passed. No additional dependency was installed.

The three Windows tasks are registered and Ready, starting September 17, 2026:
- `NewSeasonals-LegendETF-Signals`: trigger 08:45 ET, read-only ETF preparation.
- `NewSeasonals-LegendETF-Session`: trigger 09:28 ET, Primary shadow through exit.
- `NewSeasonals-LegendETF-Watchdog`: trigger 10:40 ET, shadow receipt check.

Their commands point to the stable checkout and contain no `-Live` flag. Windows
reported next runs at 08:45:45, 09:28:28, and 10:40:40 respectively while the
stored trigger boundaries are exactly 08:45:00, 09:28:00, and 10:40:00. They remain
before the required session preparation/observation deadlines.

The existing disabled runtime was backed up as
`runtime.env.pre-native-20260916.bak` beside `runtime.env`; only the strategy
version was updated. Live execution and both directions remain zero, live date
and account allowlist remain empty, and the paid data ceiling remains zero.
No old dry/live state or signal plan exists to interfere with the new version.
Tomorrow's 08:45 producer must create the first executable native plan.

To pause observations, disable these three named tasks. There are no Legend
orders to unwind from this rollout. Do not delete runtime evidence or use the
shared account's positions as a substitute for Legend-owned lot accounting.
