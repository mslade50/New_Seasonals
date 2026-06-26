# Seasonal Ideas — Session Handoff (2026-06-26)

Reference for picking this back up, on this machine or another. Read the
**cross-machine warning** first.

---

## ⚠️ Cross-machine warning (read before switching machines)

Almost everything from this session is **uncommitted and/or local-only**. If you
switch machines without acting, you lose it. Before moving:

1. **Commit + push the code** (the 9 new scripts + 2 modified files — see Git State
   below). Untracked files do **not** travel.
2. **The data backfills are local and gitignored or untracked** — they will **not**
   be on the other machine. Specifically:
   - `data/master_prices.parquet` had **11 proxy ETFs appended locally** (gitignored).
   - `data/proxy_extra_ranks.parquet` (the proxy seasonal ranks) is local only.
   - All `data/seasonal_*` and `data/seasonal_proxy_*` parquets are untracked local.
   - To reproduce the proxy analysis on the new machine, re-run
     `scratch/backfill_proxy_data.py` (re-downloads + rebuilds), then the backtests.
3. **`scratch/` is untracked** — the analysis scripts and the HTML report live there
   and won't travel unless you commit them or copy the folder.
4. **On R2 (these DO travel, shared bucket):** `seasonal_ideas_log.parquet` and
   `seasonal_ideas_outcomes.parquet` — the live forward log + scored outcomes.

---

## What we did this session (two workstreams)

### A. OLV all-time-high backtester fix — DONE, committed + pushed (`edafc6d`)
- **Bug:** `pages/strat_backtester.py` silently ignored the `use_ath` / "Today is ATH"
  filter that `daily_scan.py` enforces, so `52wh Breakout` booked phantom trades on
  new 52-week highs that weren't all-time highs (the LZB 2026-06-17 case that started
  this). These showed in the portfolio report + ledger but were never scanned/traded.
- **Fix:** ported the `use_ath` + `use_recent_ath` masks into strat_backtester's
  vectorized filter block. Rebuilding the ledger dropped 52wh Breakout from ~652 → 256
  trades (all removed rows were 52w-highs that weren't ATHs).
- Status: **committed and pushed.** Nothing outstanding here.

### B. Seasonal "Ideas" tab — performance tracking + backtest + analysis — BUILT, NOT committed
The Ideas tab (`daily_seasonal_ideas.py`) had no performance record and no backtest.
Built both, on one shared simulator, then did extensive analysis.

**Forward tracking (live going forward):**
- `seasonal_ideas_ledger.py` appends every emitted ticket to a log (+ R2). Hooked into
  `daily_seasonal_ideas.main()` so it logs automatically.
- `score_seasonal_ideas.py` scores matured ideas vs **raw** bars (verify_fills basis).

**Walk-forward backtest:**
- `backtest_seasonal_ideas.py` replays the candidate pipeline daily over 2010–2026,
  point-in-time, dedup (one open per ticker+direction), dumps candidate tickets for
  fast entry re-sims.
- `seasonal_ticket_sim.py` is the shared simulator (used by scorer AND backtest —
  parity). Supports market-on-open and limit (`open ∓ k·ATR`) entry.
- `resim_seasonal_entry.py` re-sims stored candidates under any entry in seconds.

**Live behavior change (in `seasonal_edge.py`):**
- Added a **0.667 all-years hit-rate gate** to `scan_seasonal_tickets` (drops the
  weakest-confirmation tickets). Affects both ticket channels; shared by live + backtest.

**Performance fix (in `seasonal_edge.py`):**
- Numpy-**vectorized** `seasonal_window_returns` + `expected_atr_move` (the hot loop).
  ~12× faster (per-asof 7s → 0.57s, full run ~1.3h → ~40 min). Verified
  candidate-identical to the old loop (0 mismatches across 61 dates; only float32→64
  noise). No numba/Rust needed.

---

## Key findings (the numbers)

**Complete tradeable book** (every instrument fills at IBKR, corrected entry rule):

| sleeve | entry | N | AvgR | PF | TotR | Sharpe | Sortino |
|---|---|---|---|---|---|---|---|
| US single stocks | limit | 3020 | 0.219 | 1.47 | +662 | 1.32 | 2.98 |
| US index ETFs | limit | 830 | 0.169 | 1.36 | +140 | 0.57 | 0.96 |
| international index ETFs | open | 1420 | 0.213 | 1.41 | +302 | 0.85 | 1.81 |
| commodities/bonds/FX | open | 2436 | 0.123 | 1.22 | +299 | 0.80 | 1.65 |
| **COMPLETE BOOK** | — | 7706 | 0.182 | 1.36 | **+1403** | **1.44** | 3.17 |
| **+ ex-midterm** | — | 5786 | 0.238 | 1.48 | +1377 | **1.63** | 5.29 |

1. **Entry rule = intraday-vs-gap, not stocks-vs-macro.** Limit (`open − 0.25 ATR`)
   only helps where the underlying trades during the US cash session: **US single
   stocks + US equity index/sector ETFs**. Everything that moves overnight/24h
   (international equity ETFs, commodity/bond ETFs, futures, FX) **gaps at the open and
   goes** → take **market-on-open**. (GLD/TLT are US-listed but their futures trade
   ~24h, so they gap like an intl ETF — it's the underlying's hours that matter.)
2. **Macro is tradeable via ETFs, and the ETFs BEAT the cash indices** (PF 1.33/Sharpe
   0.82 vs 1.16/0.39). European ETFs especially (EWG/EWQ/EWU/EWH/EWA all > their index)
   because a USD ETF on a foreign index carries equity × FX seasonal, which compound.
   Corrects my earlier wrong "40% of macro is untradeable" claim.
3. **Structural levers move the Sharpe; more signal-filtering does not.** Drop
   single-stock shorts (only losing cell) and downsize/skip **midterm years** (cycle 2 —
   the one real Sharpe lever; mirrors the OVS midterm tilt). A dose-response study of 5
   ex-ante confidence measures found only the all-years hit rate sorts trades, and even
   it improves per-trade quality without raising Sharpe (diversification trade-off).
4. **By horizon:** 21d is the workhorse (Sharpe 1.20 / 1.51 ex-midterm, 58% of total R,
   biggest avg R) but **lumpy from beta** — it holds ~5.6 concurrent correlated
   positions (vs 1.0 for 5d), so monthly σ is 2.5× and its deepest drawdown was the
   2018–19 selloff. 5d is smoothest but weakest. Combined book beats any single horizon
   (diversification across time frames). Chart: `scratch/horizon_curves.png`.

---

## Git state (precise, as of handoff)

- HEAD = `175f8ac` (= origin/main, clean push). The `daily_scan.py` Manual_Limit
  change committed as `175f8ac` is **not ours** — already handled, ignore.
- OLV fix `edafc6d` is in history (committed + pushed).

**Uncommitted, MINE, ready to commit:**

Modified (tracked):
- `daily_seasonal_ideas.py` — forward-log hook (logging only, no behavior change)
- `scripts/seasonal_edge.py` — **0.667 hit-rate gate (live behavior change)** +
  the numpy vectorization (perf only, candidate-identical)

New (untracked):
- `scripts/seasonal_ticket_sim.py`, `seasonal_ideas_ledger.py`,
  `score_seasonal_ideas.py`, `backtest_seasonal_ideas.py`, `resim_seasonal_entry.py`,
  `seasonal_sharpe.py`, `seasonal_time_in_market.py`, `enrich_seasonal_trades.py`
- `tests/test_seasonal_ticket_sim.py` (passing)

**Uncommitted, EXCLUDE from the code commit:**
- `data/daily_seasonal_ideas.json` + `.md` — rewritten by a local hook test run
  (asof 2026-06-24); pipeline-regenerated, not code.
- `data/*.parquet` (seasonal_*, proxy_*, proxy_extra_ranks) — local derived data.
- `scratch/` — analysis scripts + the HTML report (copy/commit separately if wanted).
- `data/master_prices.parquet` — gitignored, has the local +11-ETF backfill.
- `atr_seasonal_ranks.parquet` (tracked) — **deliberately NOT touched** (proxy ranks
  went to a separate local file).

**Recommended commits (when ready):**
1. Seasonal tracking + backtest feature — the 9 new files + `daily_seasonal_ideas.py`.
2. `seasonal_edge.py` — hit-rate gate + vectorization (its own commit so the live
   behavior change is isolated).

---

## How to reproduce on the new machine

After committing/pushing the code and pulling on the new machine:

```
# 1. caches: master_prices + earnings come from R2 automatically on most scripts.
#    The 11 proxy ETFs are NOT in R2 master_prices — re-backfill them:
python scratch/backfill_proxy_data.py        # (needs scratch/ copied over)

# 2. full daily backtest (~40 min after vectorization). Writes candidates +
#    backtest parquets used by every analysis script:
python scripts/backtest_seasonal_ideas.py --start 2010-01-01 --end 2026-02-25 \
    --cadence daily --channels seasonal,cross_asset --entry-mode t1_open

# 3. entry experiments are then seconds (no detector rerun) via the candidates parquet:
python scripts/resim_seasonal_entry.py --entry-mode limit --entry-atr-mult 0.25 --compare

# Risk-adjusted / time-in-market:
python scripts/seasonal_sharpe.py
python scripts/seasonal_time_in_market.py

# Forward log (auto-runs via the daily_seasonal_ideas hook); score matured ideas:
python scripts/score_seasonal_ideas.py
```

Note: most scripts set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is needed
locally (streamlit/protobuf clash) — run with that env var if imports fail.

---

## Outstanding decisions / next steps (none built unless noted)

1. **Commit + push the code** — do this before switching machines (priority).
2. **Schedule the scorer** — add `python scripts/score_seasonal_ideas.py` to
   `deploy_site.yml` so the live forward log updates daily.
3. **Wire the live-engine rules** (analysis-validated, not yet in the engine):
   - exclude single-stock shorts
   - midterm-year downsize/skip
   - the instrument-based entry convention (limit US equities / open everything else)
4. **Promote the 22 proxy ETFs to production** — add to the universe + rebuild ranks
   into the *tracked* `atr_seasonal_ranks.parquet` (separate from the local backfill).
   VXX excluded (history only from 2018).
5. **Size-by-confidence prototype** — scale position by hit rate / expected move rather
   than hard-filtering (the one path that might lift Sharpe without losing diversification).

---

## File map

**Committable code:** see Git State above.

**Scratch analysis (in `scratch/`, untracked):**
- `seasonal_ideas_report.html` — the full review report (open in browser).
- `backfill_proxy_data.py` — fetch 11 proxy ETF prices + build their ranks.
- `backtest_proxies.py` / `compare_proxies.py` — proxy backtest + vs-index comparison.
- `complete_book.py` — the complete tradeable book (corrected entry rule).
- `refined_hybrid.py`, `entry_by_geography.py`, `nonindex_macro_entry.py` — entry analyses.
- `sharpe_by_horizon.py` / `horizon_curves.py` — horizon breakdown + curves (PNG).
- `missed_macro_trades.py` — examples of trades the 0.25 ATR limit missed.
- `seasonal_segment_dedup.py`, `verify_vec_seasonal.py` — segmentation + vectorization checks.

**Key data (local/untracked unless noted):**
- `data/seasonal_ideas_candidates.parquet` — full-universe candidate tickets (re-sim source).
- `data/seasonal_proxy_candidates.parquet` — proxy candidate tickets.
- `data/seasonal_ideas_log.parquet` + `data/seasonal_ideas_outcomes.parquet` — forward
  log + outcomes (**also on R2**).
- `data/proxy_extra_ranks.parquet` — proxy ETF seasonal ranks (local only).
