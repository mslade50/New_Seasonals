# Risk Agent (v0.2)

An independent **$200k paper sleeve** run each weekday morning, before the open, by a Claude agent. It reads every market dataset we hold in R2 (risk dashboard, ~110 ETFs, 25 futures roots, ETF option chains and IV history, breadth, put/call, macro calendar, seasonal ranks, earnings), forms a forward view, and manages a target book of ETFs, futures, ETF options and cash. It is **blind to the real book**. It delivers by email and a private-site tab (`risk-agent.html`). Nothing places live orders.

Owner decisions (2026-10-09): $200k paper sleeve; every ETF in R2 plus futures; email + private site delivery (no Sheets); option rule below.

v0.1 (Codex, 2026-10-08) lived outside git at `C:\Users\mckin\New_Seasonals\.claude\skills\risk-agent`. Its options checker math and its blindness principle were kept. Its packet adapter (hardcoded `data_hold`, 2.6 MB raw histories) was replaced.

## Rules

- **Option structures: no more than 5% of sleeve capital per structure.** The reference is `min(200k, NAV)`. A finite worst case is certified exactly on terminal payoff with Decimal arithmetic. For same-expiry ETF options, American exercise does not break the bound: an early assignment realises the short leg's intrinsic value while longs are worth at least theirs. A dividend reserve covers early call assignment.
- **Uncovered short calls** (no finite bound) trade only on a **stress budget**: the loss at `max(25%, 3 x the 99.9th-percentile historical up-move over the option's life)` must fit in the same 5%. In practice that means one or two units. This was pre-registered 2026-10-09 in `risk_agent_grammar.py` (`STRESS_MULT`, `STRESS_FLOOR`). Changing it needs a dated note here.
- **Calendars and diagonals** are `UNKNOWN`, not tradeable.
- **Initial sleeve controls (builder-set, tunable):**
  - 5-200 bps risk per ETF or future position, at the stop. A time-only exit sizes off 3 ATR.
  - Book risk at most 1500 bps in total.
  - Gross notional at most 3x NAV.
  - At most 12 open positions and at most 5 new per day.
  - Time exit at most 126 trading days.
- **Every open paper position gets a verdict every run** (hold, adjust or close). A decision can never silently drop a holding.
- **Survey before selecting.** `scratch/risk_agent_checks/<asof>/00_surface_map.md` must exist, and every `open` cites an evidence script inside that folder. This is the Daily Pitch disk gate, inherited for the same reason.
- **Blindness is enforced in code.** `risk_agent_universe.r2_key_allowed()` is the only R2 allowlist. Deny rules win, and the guard test pins both lists. The state builder never reads positions, fills, orders, sizing state, exposure state, other sleeves or other agents' ideas. The agent session is told not to open them either.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Sync | `risk_agent_data.py` | Allowed R2 objects to `data/risk_agent/cache/` |
| Grade | `scripts/grade_risk_agent.py` | Fills pending paper orders at the next session's raw bars, marks, exits, settles options. Writes `data/risk_agent_book.json` and `data/risk_agent_scoreboard.json` |
| State | `scripts/build_risk_agent_state.py` | `data/risk_agent_state.json` (compact, under ~250 KB) |
| Agent | `/risk-agent` skill (headless, pinned model) | `data/risk_agent_decision.json` + `scratch/risk_agent_checks/<asof>/` |
| Publish | `daily_risk_agent.py` | Validates (`risk_agent_grammar`), appends orders to the journal, emails, writes `data/risk_agent_today.json`, uploads R2 `risk_agent/today.json` + `risk_agent/journal.jsonl` |
| Check | `scripts/check_risk_agent_delivered.py` | Non-zero unless today has a decision or stand-down AND a sent receipt |

Runner: `scripts/run_risk_agent.bat` then `scripts/invoke_risk_agent.ps1`. Model and effort are pinned in the bat. There is no auto-retry, because a retry cannot tell "died before publishing" from "published, then the check failed".

**Schedule (owner, 2026-10-09): mornings on the trading desktop** (`DESKTOP-2KI41V6` on the tailnet, which also runs IB Gateway and the Pitch and Seasonal agents). It runs from that machine's `dev\New_Seasonals` checkout, the same as the other agent tasks, not from the pinned automation runtime.

| Task | Time (ET, weekdays) | What it does |
|---|---|---|
| `Risk Agent (paper)` | 06:30 | `scripts/run_risk_agent.bat`. After `premarket` (04:10) has corrected the dashboard, and after the Pitch (05:10) has mostly finished |
| `Risk Agent open fill` | 09:36 | `grade_risk_agent.py --open-fill`. Fills today's option orders at live IBKR quotes, so 0-1 DTE structures get a real entry |

The state records the dashboard export's `built_at`, so a decision is tied to the data version it saw. ETF and futures orders fill at today's open (MOO) on paper. The next morning's grade picks up those fills, stops and marks.

## Paper ledger

`data/risk_agent_journal.jsonl` is the source of truth. It is mirrored to R2 `risk_agent/journal.jsonl`, append-only, and never edited. Record kinds:

| kind | Written by | Meaning |
|---|---|---|
| `decision` | publish | The whole validated decision, model/effort, state hash |
| `order` | publish | One sized open / close / adjust, `status: pending` |
| `fill` | grader | ETF/future: MOO at next open, MOC at next close, LIMIT if touched within the window (a buy fills at min(limit, open) once low <= limit, a sell symmetric; a gap through the limit fills at the open). Option: next chain snapshot after the decision, long legs at ask, short at bid. Includes costs |
| `expire` | grader | LIMIT not filled in its window |
| `exit` | grader | Stop / target / time / option expiry. A bar touching both stop and target books the STOP. Stops fill at the worse of stop and open, plus 3 bps |
| `mark` | grader | Daily NAV, cash, per-position marks (ETF/future raw close; options chain mid, else intrinsic flagged `stale_mark`) |
| `stand_down` | publish | Data hold with reason |

**Option pricing.** The R2 chain (`options/positioning_history.parquet`) re-samples strikes every day as a percentage of spot and rolls its expiries (about 29-99 DTE, 2-3 expiries, roughly 30-100 strikes per underlying). An exact contract is therefore rarely quoted twice. `grade_risk_agent.quote_leg` prices a leg in this order:

1. The exact quote.
2. The same expiry, with IV interpolated in strike and priced with Black-Scholes. There is no extrapolation. The spread comes from the neighbouring strikes.
3. For marks, time exits and closes only, a surface fallback: IV at the leg's moneyness, then total variance interpolated across expiries, flat outside the quoted range.
4. Intrinsic, at expiry or when there is no snapshot. Only this case sets `stale_mark`.

Entry fills accept only 1 or 2, so a fill is always against a contract that was actually quoted. Every option record carries `quote_sources`. The decision validator prices only from exact quotes in the latest snapshot or live IBKR chain.

Costs: ETF 1 bp per side; futures $2.50 per contract per side; options $0.65 per contract. Futures mark on the yfinance continuous series. Roll gaps are a known paper artefact, flagged on the mark.

Replay (`risk_agent_ledger.replay()`) folds the journal into the open book, cash, NAV and realised P&L. The same function feeds the state builder, the grader and the validator `ctx`.

### Live IBKR quotes

`risk_agent_ibkr.py` prices options from TWS/Gateway in real time (weeklies, 0-1 DTE included). It is quotes only and blind: it opens the bare API socket (not `IB.connect()`, which syncs account state), requests contract details, chain params, market data and historical bars, and returns plain dicts. `tests/test_risk_agent_ibkr.py` greps its source for forbidden account/order method names.

- Env: `RISK_AGENT_IB_HOST` (127.0.0.1), `RISK_AGENT_IB_PORT` (7496; Gateway also 4001), `RISK_AGENT_IB_CLIENT_ID` (77).
- CLI: `python risk_agent_ibkr.py chain SPY --max-dte 14` merges into `data/risk_agent_chains_live.json` (per-underlying block replaced); `quote <conid>...`.
- `daily_risk_agent.py` loads that file (`--live-chains`), overlays it on the snapshot chains per quote key, and tags each leg `quote_source` live|snapshot plus `quote_ts` (small edit in `risk_agent_grammar._validate_open`). A live quote older than 30 minutes at publish is a warning.
- Option fill priority in `scripts/grade_risk_agent.py`: (a) legs carry RTH live quotes and the decision was published in that same session -> fill there (`ibkr_live`); (b) IBKR historical BID_ASK 09:30-09:35 ET of the fill session, long ask / short bid (`ibkr_hist`; bar open = avg bid, close = avg ask); (c) chain snapshot / interpolation. Daily option marks use the IBKR last-RTH-minute mid when available. `--no-ibkr` or `RISK_AGENT_NO_IBKR=1` disables it; unreachable IBKR costs about 2-8 s and the rest of a run is budgeted to 60 s.
- Pre-open decisions: run `python scripts/grade_risk_agent.py --open-fill` during RTH (about 09:35-10:30 ET). It fills every pending option order with asof before today from live quotes (`ibkr_live`, `quote_ts` recorded), needs no daily bar, is idempotent, and exits 0 doing nothing outside RTH or without IBKR. Needed because 0-1 DTE contracts may be gone before a next-morning historical lookup. Expiry-day settlement stays intrinsic on the raw close.

## Scoreboard

- NAV curve, total and annualised return, max drawdown, Sharpe once there are 20+ marks, versus SPY and versus T-bill cash.
- Per-position R and P&L.
- Hit rate of `forecast.p_win`.
- Brier score of the SPY `p_up` forecasts at 5 and 21 TD, and q10/q90 coverage.
- Every row is stamped with model and effort.

The scoreboard goes into the email footer and into the next run's state.

## Decision schema (`risk_agent.v2`)

Validated by `risk_agent_grammar.validate_decision`:

```
schema_version, asof, mode: decision|stand_down, reason (stand_down)
posture: {summary, net_beta, cash_pct}
forecasts: [{horizon_td in 1/5/10/21/63, p_up, q10_pct, q90_pct, basis}]   # 5 and 21 required
positions: [
  open:   {id: RA-<asof>-<n>, action: open, instrument, side, risk_bps, entry, exit,
           thesis, evidence{summary, n, script}, survived, what_kills_it,
           forecast{horizon_td, expected_return_pct, p_win}}
  hold|adjust|close: {id, action, reason, [exit], [entry]}
]
instrument: {type: etf, symbol} | {type: future, root, contract_month: YYYY-MM}
          | {type: option_structure, underlying, structure_qty, legs:[{right, strike, expiry, qty}]}
entry: MOO | MOC | LIMIT{limit, fill_window_td 1-5}   (options: CHAIN)
exit: {time_td 1-126, stop?, target?}
considered_and_rejected: [{idea, reason}]   # required for a decision
watchlist: [{idea, trigger, expires}]
```

## Known limits (2026-10-09)

- **Option chain coverage.** From the next collector run (`scripts/update_option_surface.py`) the chain holds the two nearest expiries with DTE >= 1 plus ~7, ~14, ~30 and ~60 DTE (no 90, no LEAPS); weeklies use a +/-3% minimum strike band. History before that run has nothing under about 29 DTE. `build_risk_agent_state.CHAIN_DTE_MIN` (5) still hides the 1-4 DTE expiries from the agent's chain quotes.
- IBIT has chains but no `master_prices` history, so it has no stress table. Bounded IBIT structures are fine; an uncovered short IBIT call is rejected.
- `macro_release_history` holds printed releases only. The forward view comes from `data/macro_events.csv` via `macro_calendar` (`events.schedule`).
- Futures mark on yfinance continuous front-month series. Roll gaps show up as P&L in paper.
- `ZC=F` opens equal the prior close about 22% of the time (a data-vendor artefact), so grain MOO fills are approximate.

## Aligned sites, change together

- `risk_agent_universe.py` (ETF/futures lists, multipliers, R2 allowlist), `risk_agent_grammar.py` (limits, schema), `.claude/skills/risk-agent/SKILL.md` (quotes the limits), this doc.
- Chain quote key `expiry|strike|right`: `risk_agent_grammar.chain_quote_key`, the state builder's `chains` block, the grader's option fills.
- Journal record kinds: `risk_agent_ledger.py`, `scripts/grade_risk_agent.py`, `daily_risk_agent.py`, `site/assets/risk-agent.js`.
- Private site tab: `site/risk-agent.html`, `site/assets/risk-agent.js`, `functions/risk-agent-today.js`, nav entry in `site/assets/common.js`.

## Guard tests

`tests/test_risk_agent_universe.py`, `tests/test_risk_agent_grammar.py`, `tests/test_risk_agent_ledger.py`, `tests/test_build_risk_agent_state.py`, `tests/test_daily_risk_agent.py`, `tests/js/test_risk_agent_tab.js`.
