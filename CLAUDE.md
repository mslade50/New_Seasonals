# CLAUDE.md — Project Guide for New_Seasonals

Quantitative equity trading platform on Streamlit: a systematic strategy book (backtest, scan, order staging), a market-regime fragility dial, and agent products (Daily Pitch, posts, context brief).
**CLAUDE.md is not a changelog.** Incident notes, evidence and ship dates go in `docs/claude_ref/<subsystem>.md`, not here. Index: `docs/claude_ref/README.md`.
`trading_ibkr` means `C:\Users\McKinley Slade\OneDrive\trading_ibkr` (IBKR-bound, out of git).

## Silent-failure traps

- **yfinance MultiIndex.** Every multi-ticker download returns `(Price, Ticker)` columns. Every data function must handle it or it crashes silently:
```python
if isinstance(raw.columns, pd.MultiIndex):
    df = raw.xs(ticker, level='Ticker', axis=1)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.capitalize() for c in df.columns]
```
- **Dividend basis.** Compare a FROZEN dollar level (sheet limit, stop, ledger entry, live order) against RAW bars (`auto_adjust=False`); `verify_fills.py` does this. A level RECOMPUTED each run from the same series is safe on ADJUSTED bars; the engines rely on this and must not round the limit. Adding any ABSOLUTE dollar level to the engine path breaks this. Never converge the engine basis (adjusted) with `verify_fills` (raw). Guard: `tests/test_verify_fills_exdiv.py`.
- **`pages/` stays FLAT.** Streamlit discovers pages by scanning it. Pages import the root via `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`.
- **`pages/risk_dashboard_v2.py` is standalone.** It never imports `strategy_config`, `strat_backtester`, `daily_scan` or `indicators` (optional `SP500_TICKERS` from `abs_return_dispersion.py` with a fallback). `daily_risk_report.py` and `weekly_market_rundown.py` import its compute functions, so check both before deleting one.
- **`data/rd2_fragility.parquet` is append-only point-in-time.** Never do a full rewrite. New columns are DROPPED on append (shadow series get their own files). The AM correction may refresh only the last session's row. `rd2_fragility_ts.parquet` is research only, never a sizing fallback. Guard: `tests/test_fragility_append.py`.
- **Frozen stats JSON (A2 freeze).** Five live thresholds calibrate to the current dial. Do not adopt a re-scored `signal_horizon_stats.json` into the live path; replacements need a PIT re-validation.
- **Pre-registration.** Any NEW dial-conditioned control needs a written prereg (gates, decision rule, sensitivity) BEFORE the study runs.
- **The ledger replays TODAY's config over all history.** Its SIZE column is not what traded before the latest sizing change, so pre-change notional/exposure claims understate live. It is a rebuild, not a fill record.
- **Staleness rules differ on purpose; do not harmonize them:**
  - `daily_scan` fails OPEN to 1.0x on a dial older than 3 TD (`FRAG_STALE_TD`).
  - `dial_filters` entry gates fail CLOSED.
  - `exposure_leg` SKIPS the whole leg (`DIAL_STALE_TD`).
  - A stale or missing P/C state (`pc_fear.STALE_BD`) fails CLOSED to plain `frag_risk_bands`.
  - An existing but unreadable fragility parquet fails the risk report loudly.
  - The gap derate fails OPEN.
- **Nominal vs effective bps.** `strategy_config` dicts are NOMINAL. `GLOBAL_RISK_MULTIPLIER` (1.5) and the base-bps tilt scale them at import, so everything downstream sees scaled values. All bps in docs are nominal unless marked effective. `ACCOUNT_VALUE` (flat $750k) is the single sizing basis for scan, engine and reports, and for the event sleeve too; it is not live NLV.

## Environment outside the repo

- Production runs from a pinned local-primary worktree + venv via Windows Task Scheduler (`premarket`, `postclose`, etc.), NOT this checkout. Scheduled runs never update their own code. GitHub workflows are receipt-gated backups. See `docs/claude_ref/automation_and_r2.md`.
- Concurrent sessions work in this repo and commit directly to main. Commit promptly, re-check `git log`, never push blind.
- Some `trading_ibkr` scripts place LIVE orders (`order_staging.py`, `eq_order_entry.py`, `olv_exit_moo.py`, `event_moo.py`, `pitch_moo.py`, `radar_trail_sync.py`). Order staging is local and talks to TWS. Treat edits there as live-money changes.
- R2 (`seasonals-cache`, via `cache_io.py`) is canonical for `master_prices.parquet`, `earnings_calendar.parquet`, the intraday cache, `live_fills.parquet`, the event sleeve journal/state and the trend sleeve state. A stale local or repo copy must not overwrite them.

## Do NOT

- Rebuild an ML P(win) meta-labeling layer. Win rate and expectancy are decoupled in this book. `fragility_dial.md`
- Add a book-wide throttle/taper or dial-conditioned caps (PIT t=-0.23). The only evidenced dial hook is per-strategy `frag_risk_bands`. `fragility_dial.md`, `sizing.md`
- Re-add pooled per-direction caps (removed as redundant). `sizing.md`
- Add notional-denominated caps; ATR-risk caps and time exits are the control. Pitch sanity bounds are ATR risk too. `daily_pitch.md`
- Adopt a re-scored stats JSON. `fragility_dial.md`
- Revive the OVS dial tilt, put hedges, VXX proxy, 21d "fast confirm", trend-sleeve gate, >1.0x hi-frag boosts or sub-50 sizing ramps. `fragility_dial.md`
- Switch breadth collection to `marketsDiaryType=overview`. `fragility_dial.md`
- Converge the pitch's Wilder-14 ATR with the book's simple-mean ATR. `daily_pitch.md`
- Import `pitch_grammar`, `pitch_lab`, posts or context modules from the book. `daily_pitch.md`, `daily_posts_and_context.md`
- Re-add bull-equity names or a stop to the 3x Leader Gap Fade. Do not drop the 3x Bear Fade's < 65 leader exclusion. `strategies_3x_and_pilots.md`
- Drop the Monthly Weak Close 200d SMA gate. `strategies_3x_and_pilots.md`
- Add USO or UUP to the trend sleeve. `strategies_3x_and_pilots.md`
- Place a broker MOC as `MKT` with `tif='MOC'`; use native `orderType MOC`, `tif DAY`. `event_sleeve.md`
- Recompute, round or rescale radar numbers, or run `radar_trail_sync.py` without the upload step first. `radar.md`
- Pool the tails of a two-sided context price trigger (it must carry a `side_fn`). `daily_posts_and_context.md`
- Touch the exposure-leg raw-21d kill before its pre-registered replay runs. `fragility_dial.md`
- Tune sizing off overflow-tier backtest stats alone (survivorship). `ledger_and_fills.md`

## Routing table

Read the doc before changing a subsystem. Each doc holds the live rule, the history, and the "aligned sites, change together" list. All docs are in `docs/claude_ref/`.

| Subsystem | Doc | Guard tests |
|---|---|---|
| Repo tree, full critical rules, module boundaries, `daily_scan` CLI, ticker constants | `repo_structure.md` | `test_verify_fills_exdiv.py` |
| Fragility dial, P/C complacency, NYSE net highs, breadth, simple shadow, downside tables | `fragility_dial.md` | `test_fragility_append.py`, `test_pc_dial_signal.py`, `test_nyse_risk.py`, `test_collect_market_breadth.py`, `test_market_breadth_store.py`, `test_fragility_simple.py`, `test_risk_site_js.py` |
| Sizing: GRM, tilt, caps, OLV ladder, overlap clamp, frag bands, P/C fear bands, gap derate, step sequence | `sizing.md` | `test_base_bps_tilt.py`, `test_wcds_size_tiers.py`, `test_pooled_cap_sequential.py`, `test_frag_risk_bands.py`, `test_pc_fear_bands.py`, `test_cboe_putcall.py`, `test_gap_size_derate.py`, `test_earnings_size_override.py` |
| OVS: blackout, 2-path, scale-out, EOD-DD, cycle tilt | `ovs.md` | `test_ovs_scaleout.py`, `test_eod_dd.py` |
| OLV: vol-confirm stop, notional cap, capacity, T+3 window, retired book cap | `olv.md` | `test_olv_stop_and_cap.py`, `test_olv_fill_window.py`, `test_olv_exits.py` (trading_ibkr) |
| 3x fades, same-day de-rate, Leader Gap Fade, Monthly Weak Close, trend sleeve | `strategies_3x_and_pilots.md` | `test_same_day_derate.py`, `test_lev3x_leader_gap_fade.py`, `test_monthly_weak_close.py` |
| Event sleeve | `event_sleeve.md` | `test_event_sleeve.py`, `test_execution_report.py`, `test_event_site.py`, `test_event_moo.py` (trading_ibkr) |
| Daily Pitch | `daily_pitch.md` | `test_pitch_grammar.py`, `test_daily_pitch.py`, `test_pitch_grader.py`, `test_pitch_lab.py`, `test_pitch_delivery_check.py`, `test_pitch_moo.py` (trading_ibkr); fills approvals: `test_pitch_fills_approval.py`; site Pitch tab: `test_pitch_transport.py`, `tests/js/test_pitch_tab.js` |
| Daily Posts, Market Context | `daily_posts_and_context.md` | `test_daily_posts.py`, `test_context_engine.py`, `test_context_sender.py` |
| Ledger, stop arming/fill conventions, live fills store | `ledger_and_fills.md` | `test_fills_harvest.py` |
| Local-primary automation, R2, Sunday pipeline | `automation_and_r2.md` | `test_local_automation_powershell.py` |
| Retired GitHub-first schedule and radar digest (history only) | `automation_history.md` | none |
| Private site, shared Denali risk tab, trade log, hedge panel | `private_site.md` | `test_tradelog_site.py`, `test_risk_site_js.py`, `test_strategies_site.py`, `test_publish_sleeve_runtime_status.py`, `tests/js/test_strategies_tab.js`, `tests/js/test_sleeve_status.mjs` |
| Momentum radar staging + trail | `radar.md` | `tests/js/test_radar_tab.js`, `test_radar_transport.py`, `test_radar_trail_sync.py` + `test_stop_limit_entry.py` (trading_ibkr) |
| Google Sheets tabs | `sheets.md` | none |

Guard tests without a path are in `tests/`, except those marked trading_ibkr.

## Verification

- Python suite: `python -m pytest tests/ -q` (config in `pytest.ini`). A single guard: `python -m pytest tests/test_frag_risk_bands.py -q`.
- JS suite: `node scripts/test_js.mjs`.
- trading_ibkr tests run from that folder: `python -m pytest test_olv_exits.py -q`.
- Before finishing any change, run the guard tests for every subsystem it touches (routing table above). If you change a live rule, update its `docs/claude_ref/` doc in the same change.
