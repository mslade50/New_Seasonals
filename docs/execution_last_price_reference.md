# Market-order reference autofill — September 30, 2026

MKT, MOO and MOC entry tickets request the selected instrument's IBKR Last
through a protected read-only Pages endpoint. The reference refreshes while
the ticket is open. Symbol, currency, venue, contract month, account and entry
type changes invalidate prior responses and clear the old reference. Futures
prices come from the exact selected delivery month and canonical trading class.

The result must be an exact qualified contract with live data type 1, positive
finite Last and a quote observation no more than 30 seconds old. Midpoints,
previous closes, frozen data and delayed quotes are never silently substituted.
When Last is unavailable the reference is blank, with a manual-entry notice.
An operator can overwrite it; polling preserves that value until the instrument
or entry type changes. Switching back to LMT/STP LMT restores the typed limit
or trigger rather than converting the market reference into an order price.

The existing authenticated workbench query ring carries `mode=last_price`,
keeping this request out of the order-command path. An early return in the
existing `option_workbench.py` subprocess dispatches to `ticket_last_price.py`.
The helper connects read-only, requests Last, cancels its market-data subscription,
and disconnects. It never requests an options chain, writes volatility state,
or transmits an order. The existing agent already launches the subprocess for
each query, so neither ExecAgent nor today's day-trade service needs a restart.

The scoped installer preparation is in `broker_runtime/prepare_ticket_last_price.py`.
Installation preserves verified copies of the prior workbench source and source/
candidate SHA-256 receipts under the existing runtime backup directory and
ignored `artifacts/execution-last-price-20260930/`. The Pages release uses the
normal GitHub Actions R2 build and freshness gate. No order or execution journal,
account gate, exit monitoring or exit-size policy is changed.

Tests cover exact futures identity, Last versus midpoint/close, unavailable and
delayed quotes, symbol-change races, manual overrides, restoration of typed
limit prices, scoped runtime preparation, Access denial and query-id routing.
Native read-only quote and authenticated ticket checks complete rollout.

The runtime hook and helper were installed on September 30 without restarting
ExecAgent. Native read-only checks returned live SPY Last and MNQ December 2026
Last (qualified conId 815824267). The regression run passed 210 Python tests
and all 37 JavaScript suites. Rendered ticket checks verified reference autofill,
manual override handling and restoration of a typed limit price.

IBKR documents [market-data types](https://interactivebrokers.github.io/tws-api/market_data_type.html)
and [watchlist quote requests](https://interactivebrokers.github.io/tws-api/md_request.html).
