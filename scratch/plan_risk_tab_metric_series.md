# Plan: Risk tab — per-signal metric series + firing overlays vs SPY

Goal: the private site's risk page (site/risk.html + site/assets/risk.js) should show
the ACTUAL figures for each fragility signal's underlying metric (like the Streamlit
risk dashboard does), with the periods each signal fired overlaid against SPY price.
Default chart window = 1y, double-click zooms out to full history.

## Current state (context)

- `scripts/build_risk_json.py` runs in `deploy_site.yml` (best effort, ALWAYS exits 0
  — keep that contract), reuses `daily_risk_report.download_data()` +
  `compute_all_signals()` (10y of data), writes `data/site_risk.json`.
  `scripts/build_site.py` just copies it to `dist/data/risk.json` — NO change needed there.
- Each of the 7 signal dicts from `compute_all_signals()` already carries full
  historical series that are currently thrown away — only current on/off/badge/detail
  is serialized. The frontend renders signals as text cards plus one small
  SPY-vs-fragility chart (1y tail only).
- The Streamlit page already has the target visual: `chart_signal_overlay()` in
  `pages/risk_dashboard_v2.py:1945` (SPY top panel, Gantt-style signal strips bottom),
  and `_signal_periods()` compresses a boolean `signal_history` Series into
  `[(start, end), ...]` runs.

## Step 1 — Backend: extend `scripts/build_risk_json.py`

Add to the payload (all through the existing `_clean()`, rounded, best-effort):

1. **Full-history SPY series** — replace the `tail(252)` with the full ~10y
   `spy_close` (~2,500 points). Extend `fragility_series` the same way (or keep 1y —
   implementer's call, but if extended apply the same 1y-default treatment client-side).
2. **Per-signal block** under a new `signal_detail` key, one entry per signal:
   - `periods`: fired windows as `[[start, end], ...]` — reuse `_signal_periods`
     (importable via the `pages.risk_dashboard_v2` import path `daily_risk_report`
     already uses; the standalone rule only restricts what risk_dashboard_v2 IMPORTS,
     not who imports it).
   - `metric`: the underlying figure series + its threshold(s), per this map:

   | Signal | Series key in signal dict | Threshold line(s) |
   |---|---|---|
   | Distribution Dominance | `da_ratio` | 3.75 (fire), 6.0 (elevated) |
   | VIX Range Compression | `compression_pctile` | fires < 15 |
   | Defensive Leadership | `spread` (50d) | fires < -10pp |
   | Pre-FOMC Rally | none (periods only; `signal_dates` exists) | — |
   | Low Absorption Ratio | `ar_pctile` (and/or `ar_series`) | fires < 10th pctile |
   | Seasonal Rank Divergence | `spread` | fires > +10pp |
   | Dispersion | `composite_pctile` | fires > 85 |

   - `current`: latest metric value + the existing `summary` string (it already
     contains the human-readable "value vs threshold" text).
3. **Size control**: reindex every metric series to `spy_close.index` and ship ONE
   shared `dates` array at the top level; per-series arrays are values only (nulls
   for gaps). Round to 2-3 dp — keeps the JSON in the low hundreds of KB.

## Step 2 — Frontend: `site/assets/risk.js`

1. **Combined overlay chart** (top of the Signals section): port
   `chart_signal_overlay` to Plotly.js — SPY line in an upper subplot, one colored
   strip row per signal below (layout.shapes rects or fill-to-self traces, one legend
   entry per signal), shared x-axis.
2. **Per-signal cards**: keep the existing badge/detail card, add a chart per signal
   that has a metric series — SPY on the left axis (thin gray), metric on the right
   axis, dashed horizontal threshold line(s), translucent vertical bands
   (layout.shapes, yref:'paper') over each fired period. Show the current figure
   prominently in the card header (from `current`/`summary`). Pre-FOMC gets bands
   only, no metric trace.
3. **1y default, zoomable to full**: data spans the full 10y; set
   `layout.xaxis.range` to `[today - 365d, today]`. Compute the initial y-ranges in
   JS from the last-1y slice (Plotly autorange would otherwise fit the full history
   and dwarf the 1y view). Double-click then autoranges both axes to full history —
   same convention as the Streamlit dashboard. Verify `PLOT_CFG` in
   `site/assets/common.js` doesn't set `doubleClick: false`.
4. Apply the same 1y-default/zoom-out treatment to the existing SPY-vs-fragility
   chart once it carries full history.

## Step 3 — Verify

- `python scripts/build_risk_json.py` locally (heavy: ~10y yfinance pull), inspect
  `data/site_risk.json` for the new keys and sane size.
- `python scripts/build_site.py --no-signals --no-mtm` then
  `python -m http.server 8123 --directory dist` — check the risk page: bands line up
  with known episodes, 1y default renders, double-click zooms out.
- Backward compatibility: old payloads WITHOUT `signal_detail` must still render
  (guard every new section) — the page must never break when the builder fails,
  since risk.json is best-effort.

## Not in scope / cautions

- Don't touch `pages/risk_dashboard_v2.py` beyond importing its existing helpers
  (it stays standalone; no new imports INTO it).
- The fragility parquet append-only rule and the email pipeline are untouched — this
  only adds serialization to the site builder and rendering to the static page.
- Keep build_risk_json.py's exit-0 / best-effort contract intact.
