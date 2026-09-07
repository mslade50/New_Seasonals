# Data and strategy fixes â€” 2026-09-06

Source changes on `codex/audit-fixes-data-20260906`, based on `e53c478e`. No scheduled process, broker/order, canonical dataset, Sheets table, R2 object, email, or deployment was changed by this work. The raw-bar rule for frozen dollar levels and adjusted-bar rule for recomputed relative levels remain intact.

| ID | Source correction | Proof |
|---|---|---|
| DS1 | ATR-seasonal labels carry their outcome dates; annual ranks exclude outcomes unknown at the annual cutoff. New parquet metadata identifies `annual-outcome-cutoff-v2`. | Six-horizon synthetic suffix invariance; all 96 real ticker/year/horizon comparisons exactly equal truncated-history results. |
| DS2 | Failed, empty, short, or internally broken basis repairs cannot splice the newly adjusted overlap into the old history. Other ticker updates continue with a degraded receipt. | Main-path short/empty/cliff/full-replacement fixtures. |
| DS3 | Publishing requires a readable prior earnings baseline; failed ticker requests retain prior events. Writes are atomic; upload failure raises; per-ticker degradation is recorded. | Missing/corrupt baseline never writes or uploads; partial provider failure preserves the failed ticker. New calendars require a reviewed local `--no-upload` initial build before publishing. |
| DS4 | Annual and full-period metrics include first-day PnL and initial capital in return/drawdown denominators; one-observation years remain visible. Single-strategy drawdowns also include starting capital. | Year-boundary reconciliation, first-day loss, and single-observation year. |
| DS5 | 3M/6M/12M use actual calendar months and an explicit optional as-of date; future entries are excluded. | 1/75/150/300-day trade cohorts count 2/3/4. |
| DS6 | Missing prices remain missing in dispersion, strategy cross-sectional returns, and selected risk calculations; dispersion excludes nonfinite/nonpositive endpoints. | Missing constituent is excluded from both returns and observed coverage. |
| DS7 | Rotation UI explicitly says next-session **close**, matching its existing close-only simulation and return timing. | The false next-open claim is removed. Actual next-open simulation requires open-price inputs and is not offered by this fix. |
| DS8 | Rebuild normalizer finds the OHLCV price level in either yfinance MultiIndex orientation. | Singleton `(Ticker, Price)` and `(Price, Ticker)` fixtures. |
| DS9 | Optional compounding sizes in entry-date order using only exits realized before entry; same-day exits do not finance entries. | Overlapping trades cannot borrow future gains; later entries can use settled gains. |
| B1 | Indicator identity includes price content, seasonal inputs, market context and VIX; version `v4-content`. | Same-date dependency/price revisions recompute; unchanged inputs reuse cache. |
| B2 | Cross-sectional identity includes exact universe membership and price content; cached matrix schemas are checked. | Same-size changed universe produces the same output as an isolated cold computation. |
| B3 | Coverage uses the requested universe, including absent names. Partial healthy inputs continue with per-strategy exceptions. Unknown symbols retain original dated staging obligations; they are never re-stamped as new signals. | Requested-vs-loaded receipt test and unavailable-row preservation test. A fully unusable price book still cannot mint new signals. |
| B4 | Shared `sheets_io.replace_worksheet_values` replaces cells and clears trailing values in one atomic batch with full readback. All scanner, Portfolio and verifier writers use it. Manual pins/read snapshots and finalized historical fill rows survive replacement. | Transport failure retains old table; response loss accepts only exact readback; same-length corruption and changed snapshots raise; frozen prices survive rescans. |
| B5 | Missing price feeds preserve prior fill status; healthy rows still verify. All-missing sources fail without changing the table. Receipts identify bar-model evidence separately from broker confirmation. | All-missing and partial-missing source fixtures. Empty/already-verified logs emit fresh no-work receipts. |
| B11 | Empty/nonfinite price context produces null values and `Insufficient data`, not a fabricated downtrend or NaN JSON. 252-session return uses 253 endpoints. | Empty/all-NaN/final-NaN/infinite cases serialize with strict JSON. |

Verification: **178 selected tests passed** (one existing Plotly/NumPy deprecation warning), including pivot precision, raw ex-dividend fill basis, ATR-seasonal parity, caps, fill-window behavior, overlays, and strict-producer contracts. The new source regressions run against unchanged audit-baseline source produced **27 failures and one pass**, demonstrating old failure paths without resetting any worktree. Evidence is under `artifacts/data-fixes/`: `baseline-regressions.log`, `verified-tests.log`, `seasonal_impact.json/csv`, and an isolated 50,745-row price sample. The fixed-clock price-island fixture uses the overnight `_today()` seam; its 12-row assertion no longer changes as wall time advances.

The real sample spans SPY, QQQ, IWM, EEM, EWZ, XLE, TLT and GLD, 2000-01-03 through 2026-09-04. Original master SHA-256 before/after reading: `e74582c50f52eacc2d93676344678e481f57f38ee4e363d2c49f871e246e0c67`. For 2026 the maximum 252-session rank change is 19.62 percentile points with 55 ticker/day-count threshold flips above 50; the 10-session maximum is 3.43 points with three flips. These are rank/threshold differences, **not trade counts or economic PnL estimates**. A representative cold-cache candidate/fill/size/PnL comparison is now complete (below). Full-book validation and controlled history promotion remain separate.

Receipts are strict JSON, locally atomic, schema version 1, and carry a UTC `generated_at`: `data/scan_coverage_{scope}_{bookend}.json`, `data/fill_verification_status.json`, `data/master_prices.parquet.status.json`, `data/earnings_calendar.parquet.status.json` (price/earnings follow a custom output path). The execution audit stream owns supervisor ingestion. Existing receipts after a failed early run must not be treated as fresh successes.

Sheets batch atomicity removes the clear/write gap; optimistic read prechecks are **not** a server-side compare-and-swap. Multiple writers still require orchestration leases. This matches Google's documented [atomic batch contract](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/batchUpdate) and [updateCells range-clearing behavior](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/request#UpdateCellsRequest). There was no live API mutation test.

Remaining boundaries: full-book corrected Seasonal replay and controlled history promotion; historical index/universe survivorship; true next-open rotation research (if desired); and broader optional-filter fallback policy owned by the root audit stream. No risk parameter or strategy threshold was changed.

## Representative cold-cache replay

Evidence: `artifacts/data-fixes/cold-replay-20260906-211102/replay_impact.json`; script `artifacts/replay_seasonal_impact.py`. Separate empty indicator-cache directories were used for the original and corrected ATR Seasonal rank algorithms. Both arms used the same source code, current risk settings, frozen hashed adjusted prices, fragility and P/C inputs. Network connections and uploads were denied. Runtime: 37.93 seconds.

Coverage: all 29 required price symbols (175,408 rows), full native universes of Weak Close Decent Sznls, Indices Oversold Bounce, Monday Dip and SPY QQQ MonFri Reversion (overlap control). Signals run from 2003-01-01 through the 2026-09-04 input close with 2000+ warmup. The three affected representatives use 5-session ATR Seasonal filters; the fourth holds overlap behavior constant. Unused ordinary seasonal-rank inputs were neutral in both arms and did not drive any selected strategy decision.

| Comparison | Original ranks | Corrected ranks | Change |
|---|---:|---:|---:|
| Candidate count | 1,634 | 1,634 | 5 identities removed, 5 added |
| Executed trade count | 1,023 | 1,023 | 3 identities removed, 3 added |
| Flat-$750k modeled PnL | $1,338,239 | $1,337,252 | -$987 |
| Compounded modeled PnL | $3,857,897 | $3,830,765 | -$27,132 |

Among 1,020 common trades, entry/exit prices match; flat sizing changes one share count/PnL, while compounded equity propagation changes 857 share counts and 852 PnLs. These are theoretical engine results, not broker fills or a forecast. Other strategy/overflow interactions and the longer Seasonal horizons are not validated by this representative subset. A broader frozen-input active-book comparison is in progress; no canonical rank/ledger files have been regenerated or promoted.
