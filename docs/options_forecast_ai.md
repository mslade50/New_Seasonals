# Options Forecast Lab and AI Tool Layer

## Product contract

The Forecast Lab turns a probabilistic market view into an auditable ranking of
listed option structures. It does not treat a target and probability as a
complete forecast.

A forecast must specify:

- underlying and reference level;
- event type (`touch_by` or `terminal_at_or_beyond`);
- target level, probability, and cutoff date;
- a conditional touch-time distribution for `touch_by` events;
- implied-volatility change when the forecast hits;
- no-hit underlying and implied-volatility states;
- risk budget and ranking objective.

The first browser implementation uses a 25% / 50% / 25% weighting across
early, middle, and late touch dates. Every assumption remains visible in the
UI. The production optimizer should accept a full discrete scenario grid.

## Deterministic optimizer

Pricing and ranking belong in normal code, not in model-generated prose.

1. Load synchronized, timestamped SPY, XSP, SPX/SPXW surfaces and underlying
   prices.
2. Normalize the economic target across products (for example, convert an SPY
   level to the contemporaneous equivalent SPX and XSP levels).
3. Keep only expiries alive through the forecast window and liquid strikes in
   a target-aware band.
4. Generate singles and defined-risk structures, including target-anchored put
   spreads, long puts, put-spread collars when portfolio context is supplied,
   and ratio-free alternatives where appropriate.
5. Reprice every leg on the scenario grid with the same rates, dividends,
   timestamp, and volatility-shift convention.
6. Include executable-side entry marks, commissions, spread cost, contract
   multiplier, discrete sizing, and settlement/exercise metadata.
7. Rank separately by expected P&L per dollar at risk, payoff if hit,
   robustness across touch timing, liquidity, and lowest defined loss. Do not
   collapse those objectives into an unexplained composite score.

Each result should include the input snapshot hash, quote timestamp, leg IDs,
scenario P&L vector, max loss, liquidity flags, and the reason it ranked where
it did. Results expire when their quote snapshot becomes stale.

## AI access pattern

Use an OpenAI Responses API agent with strict function tools over the
underlying data services. Do not let the model scrape the rendered private
site, calculate option values, or invent live prices.

Recommended read-only tools:

### `get_volatility_context`

Input: symbols and horizon. Output: IV level/rank/percentile, realized
volatility, IV/RV premium, term structure, skew, timestamps, and stale flags.

### `get_index_option_surfaces`

Input: `SPY`, `XSP`, `SPX`; earliest and latest expiry; target band. Output:
qualified contracts, bid/ask/mid, greeks, multiplier, exercise style,
settlement type, trading class, and market-data type.

### `rank_forecast_expressions`

Input: the complete forecast schema, objective, risk budget, account sizing
constraints, and a surface snapshot ID. Output: deterministic candidates and
their scenario vectors. This is the only component allowed to call the pricing
engine.

### `get_portfolio_hedge_context` (optional)

Input: account alias. Output: read-only beta, notional, sector concentration,
existing option delta/vega, and hedge budget. It must not return unrelated
account information.

The model's job is to translate natural language into the forecast schema,
identify missing assumptions, call the tools, compare the returned candidates,
and explain trade-offs. Structured outputs should enforce the final answer
shape.

## Execution boundary

Keep analysis and execution separate:

- AI tools are read-only by default.
- A candidate can be copied into the site's existing ticket only by a separate
  `stage_option_order(candidate_id)` action.
- Staging revalidates contract IDs, price freshness, quantity, defined max
  loss, account, and execution mode.
- The user sees a final ticket and must explicitly confirm it.
- The model never receives a direct live-order function.

Single-leg and credit-order support must remain dry-run until the local IBKR
agent has matching validation, risk caps, and explicit live authorization.

## SPY / XSP / SPX adapter work

The current local workbench qualifies stock/ETF underlyings and therefore
supports SPY. Cross-complex comparison needs the read-only IBKR adapter to:

- qualify XSP and SPX as index contracts rather than stocks;
- preserve SPX versus SPXW trading class and AM/PM settlement metadata;
- use the correct contract multiplier and settlement/exercise style;
- fetch all three products from one synchronized quote window;
- normalize targets from the reference underlying without relying on a fixed
  10:1 ratio;
- return partial results with explicit entitlement or quote failures.

This data change is read-only, but it should be tested against live contract
qualification before the site advertises cross-complex optimization.

## Training and evaluation

Do not fine-tune first. Start with tool calling, structured outputs, and a
small evaluation set. Fine-tuning is worth considering only after the tool
agent has repeated, measured failures that cannot be fixed through schemas,
examples, or deterministic validation.

Minimum evaluation cases:

- touch forecast versus terminal forecast;
- early versus late touch changing the best expiry;
- rich downside skew favoring a spread over a single put;
- SPX contract size exceeding the risk budget while SPY/XSP fits;
- stale or crossed quotes;
- dividend/early-exercise considerations on SPY;
- no-touch state making a superficially attractive long put negative EV;
- missing market-data entitlement for one complex;
- adversarial instruction attempting to bypass risk limits or submit an order.
