# Google Sheets integration

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Google Sheets Integration

Tab layout in the `Trade_Signals_Log` workbook:
- `Order_Staging` — Liquid-tier signals (Limits, T+1 Open, Persistent GTC). Cleared + rewritten by every `daily_scan` run with `Scan_Source='Liquid'`.
- `Overflow` — Overflow-tier signals (same entry types, no MOC). Cleared + rewritten by `daily_scan --scope=overflow|all` with `Scan_Source='Overflow'`.
- Both staging tabs carry a `Manual_Limit` column (emitted empty by the scanner): type a price into it to pin that signal's entry — order_staging uses it verbatim as a LMT and anchors the bracket to it, skipping the gap clamp. Rows survive only until the next scan's clear+rewrite, so manual rows/pins must be added AFTER the ~4:47 AM ET scan and BEFORE order_staging runs (e.g. the 2026-07-06 TS/USO makeup rows via `scratch/stage_makeup_ts_uso.py`). Entry expiry is back-computed from `Time_Exit_Date` − (1 + `Hold_Days` − `Fill_Window_Days`) BDays, NOT from `Scan_Date`, so a makeup row can carry its true original schedule.
- `OLV_Exits` — vol-confirmed OLV stop exits (2026-07-20). Cleared+rewritten
  by BOTH bookend `daily_scan` runs (`stage_olv_vol_confirm_exits`); rows are
  per-LEG (stacked positions get one row per confirmed leg, keyed by
  `Time_Exit_Date`, with an `Entry_Date` audit column since 2026-07-30).
  Consumed by `olv_exit_moo.py` (OneDrive trading_ibkr) — the standalone
  pre-market task (weekdays 9:10 AM ET) that places rows with `Execute_On`
  == today as TRUE market-on-open SELLs (TIF=OPG, both accounts; PA sells
  the full matched PA leg). order_staging stopped consuming this tab
  2026-07-30 — its 9:31 post-open run could never deliver a real MOO.
- `moc_orders` — MOC entries from liquid tier only (`save_moc_orders` skips overflow rows). Currently vestigial: the strategy book has no Signal Close entries, so this tab is never written. Reactivates automatically if any strategy is set to `entry_type='Signal Close'`.
- `Seasonal` — tradeable seasonal-ideas tickets (longs + non-equity shorts). Written by `seasonal_order_staging.py` from `data/daily_seasonal_ideas.json`, `Scan_Source='Seasonal'`. Separate pipeline from the systematic book. Entry type per instrument (validated geography rule): US single stocks + US-session equity ETFs → `REL_OPEN` limit (0.25 ATR, DAY); everything that gaps overnight (intl/commodity/bond/FX ETFs, GLD/TLT) → `MOO` (market-on-open, `TIF=OPG`). Sizing: 20 bps/trade (13 bps in midterm years, `year%4==2`), 1% aggregate daily cap. order_staging must add `MOO` handling — see `docs/seasonal_order_staging_spec.md`.
- `sznl_nostage` — NOT auto-executed. Single-stock equity shorts (sized, tagged `[eq-short]`) + non-tradeable signals (futures/index/FX/crypto, `Quantity=0`, `Order_Type=NONE`, tagged `[need-proxy]` pending the proxy-ETF promotion). order_staging does not read this tab.
- `Pitch` — the Daily Pitch approval tab (2026-08-06). One row per LEG of the
  morning's three ideas, written by `daily_pitch.py` (clear+rewrite), with an
  empty `Approve` column. Typing exactly `Y` on EVERY leg of an idea is what
  authorizes `pitch_moo.py` (OneDrive) to place it at 9:05 / 9:32; blank, `N`
  and anything ambiguous place nothing, and a partly-approved multi-leg idea
  refuses the whole basket. `Manual_Only` rows are never auto-placed. The
  next morning's run reads the Approve cells into the journal BEFORE clearing
  the tab, which is the only window they can be captured in.
- `Trade_Signals_Log` (sheet1) — append-only signal history.
- `Portfolio` — open-positions snapshot from `daily_portfolio_report.py`.
- `execution`, `execution_2` — order_staging.py output for primary + small-account execution.

`daily_scan.py` writes both `Order_Staging` and `Overflow` via `save_staging_orders(..., tier_filter='Liquid'|'Overflow')`. The function clears+rewrites only the tier it's responsible for (so a `--scope=liquid` run never touches `Overflow`).

`order_staging.py` (in `C:\Users\McKinley Slade\OneDrive\trading_ibkr\`) reads BOTH tabs and concatenates with `Scan_Source` distinguishing tier. Applies the OVS 2-path gap-tier sizer + path-2 daily aggregate cap + global 2.5% daily risk cap before submitting to IBKR.

`verify_fills.py` updates Trade_Signals_Log with fill status post-close.

Auth: `gspread` with GCP service account from Streamlit secrets / `GCP_JSON` env var (GHA) / `credentials.json` (local).
