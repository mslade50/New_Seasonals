# Seasonal → order_staging integration spec

How `order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) should consume
the seasonal staging tabs written by `seasonal_order_staging.py` in this repo.

## Tabs (in the `Trade_Signals_Log` workbook)

| Tab | Written by | order_staging action |
|---|---|---|
| `Order_Staging` | `daily_scan.py` (Liquid) | submit (existing) |
| `Overflow` | `daily_scan.py` (Overflow) | submit (existing) |
| **`Seasonal`** | `seasonal_order_staging.py` | **submit (NEW)** — add this tab to the read+concat list, `Scan_Source='Seasonal'` |
| **`sznl_nostage`** | `seasonal_order_staging.py` | **do NOT read** — reference/manual tab (equity shorts + non-tradeable signals) |

The `Seasonal` tab uses the **same column schema** as `Order_Staging`/`Overflow`
(`save_staging_orders`), so it concatenates with no parsing changes — except for
one new `Order_Type` value (`MOO`). Columns: `Scan_Date, Symbol, SecType,
Exchange, Action, Quantity, Order_Type, Limit_Price, Manual_Limit,
Offset_ATR_Mult, TIF, Frozen_ATR, Signal_Close, Time_Exit_Date, Strategy_Ref,
Tgt_ATR_Mult, Stop_ATR_Mult, Use_Target, Use_Stop, Hold_Days, Trade_Direction,
Rank_252D, Risk_Amt, Risk_Bps, Scan_Source`.

## Entry instructions on the `Seasonal` tab

Two `Order_Type` values appear (set per instrument by the validated geography rule):

1. **`REL_OPEN`** (US single stocks + US-session equity ETFs) — *already supported.*
   Limit anchored to the T+1 open: `open − Offset_ATR_Mult·Frozen_ATR` for a long
   (`+` for a short), `TIF=DAY`. `Offset_ATR_Mult = 0.25`. Identical handling to
   the strategy book's REL_OPEN limit.

2. **`MOO`** (everything that gaps overnight — intl/commodity/bond/FX ETFs, GLD/TLT)
   — **NEW, must be added.** Submit a **Market-on-Open**: IBKR `Order(orderType="MKT",
   tif="OPG", action=Action, totalQuantity=Quantity)`. `Offset_ATR_Mult=0`,
   `Limit_Price=0`. No limit — it fills at the official open print.

## Bracket (both entry types)

Anchor the exit bracket to the **actual fill**, exactly like the strategy book:
- stop  = fill − `Stop_ATR_Mult·Frozen_ATR` (long; `+` for short), gated `Use_Stop`.
- target= fill + `Tgt_ATR_Mult·Frozen_ATR`  (long; `−` for short), gated `Use_Target`.
- time exit at `Time_Exit_Date` (= signal asof + `Hold_Days` business days).

`Frozen_ATR`, `Stop_ATR_Mult`, `Tgt_ATR_Mult` are all on the tab. Apply the
book-wide **stop-arming convention** (stop leg `goodAfterTime` = next session) the
same way the equity book does.

## Sizing

`Quantity` and `Risk_Amt` are already risk-sized scanner-side:
`risk$ = ACCOUNT_VALUE × risk_bps`, `shares = risk$ / |entry − stop|`, with
`risk_bps = 20` (or `13` in midterm years) and a **1% of ACCOUNT_VALUE aggregate
cap on the Seasonal tab** (pro-rata). order_staging should treat `Quantity` as
final, then apply its own **global 2.5% daily risk cap** on top of the combined
book (Order_Staging + Overflow + Seasonal) as usual.

## What stays manual

`sznl_nostage` rows never execute. They are:
- single-stock **equity shorts** — fully sized (so you can place them by hand if
  you want), tagged `Strategy_Ref … [eq-short]`, `Scan_Source='Seasonal_NoStage'`.
- **non-tradeable** signals (futures/index/FX/crypto, `SecType` ∈ {FUT, IND, CASH,
  CRYPTO}) — `Quantity=0`, `Order_Type='NONE'`, tagged `… [need-proxy]`. They become
  stageable once the proxy-ETF universe is promoted (seasonal handoff open item #4).

## Open follow-ups
- Promote the 22 proxy ETFs so the cross-asset sleeve's index/future signals map to
  tradeable ETFs and move from `sznl_nostage` to `Seasonal` (handoff #4).
- Decide whether seasonal positions share the equity book's `ACCOUNT_VALUE` risk
  pool or get a dedicated one (currently shared).
