# Build brief — "Hedge" scenario panel on the private site's Exec tab

Written 2026-08-25 for a builder agent. Read fully before touching anything.
Owner: McKinley. Display-only feature. **Nothing in this brief places, modifies,
schedules or sizes an order.** If you find yourself writing code that talks to
IBKR or to the execution broker's command path, stop — you have left scope.

## 0. Why this exists (context you need, not history for its own sake)

The Oversold Low Volume (OLV) strategy runs dip-buy legs with no resting stop
and no aggregate exposure cap. In Aug-2026 the OLV book reached ~80% of NAV
while the fragility dial sat at 89.5 — a combination with no precedent in the
ledger. McKinley decided to hedge such episodes **manually with index futures**
(he already runs MES) rather than carry automated sizing/trim machinery (a
0.5x band and an EOD trim were built 2026-08-24 and retired 2026-08-25; see
CLAUDE.md "Fragility Risk Bands" and "OLV Book Cap").

What was missing each time was the arithmetic: pull the live book, attribute
legs to strategies, mark them, apply each name's SPY beta, net off the futures
already held, convert "net OLV exposure back to X% of NAV" into a contract
count, and show a SPY-shock scenario. Doing that by hand took ~10 minutes of
scripts. This panel makes it a glance. The *decision* stays with the human.

Reference numbers from 2026-08-25 ~11:00 ET, useful to validate your output
(the book will have moved; the shapes and magnitudes should not):

| item | value |
|---|---|
| Primary OLV filled book, 14 legs, MTM | $602,784 = 80.4% of the $750k sizing NAV |
| Notional-weighted beta (63d / 252d) | 1.14 / 1.23 → SPY-equivalent $690k / $738k |
| Per-name betas (63d) | LUV 1.47, UNH −0.03, POWI 3.19, CINF −0.62, D −0.12, WWD 0.67, CMI 1.53, CRS 1.44, ON 3.49, MOD 3.27 |
| Index futures held | MESU6 −6 (mult 5, ~$38.2k/contract at index ~7,635) ≈ −$229k; DXU6 +2 (not an equity hedge) |
| Working OLV entries (unfilled GTD limits) | 9 orders, $353k |
| Target 50% NAV → contracts | (690k − 375k)/38.2k ≈ 8–10 MES total |

## 1. Scope

Three deliverables, in this order (each is independently useful; ship A even if
B slips):

**A. Nightly beta table** — `scripts/build_betas.py` → `data/betas.json`,
registered as a generated site input and copied to `dist/data/betas.json`.

**B. "Hedge" card on the Exec tab** — `site/assets/execution.js` (+ minimal
CSS in `site/assets/style.css`), rendered from the live book the tab already
polls every 4 s, plus `betas.json` and `assets/futures_specs.json`.

**C. Tests + docs** — Python tests for A, a node-vm JS test for B's pure
functions (same harness as `tests/test_risk_site_js.py`), and a CLAUDE.md
section.

### Non-goals (do not do these)
- No orders, no "apply hedge" button, no command to the execution broker, no
  `data-mutation` controls on the card. The card must render identically
  whether the agent is armed, dry-run or unknown.
- No changes to `strategy_config.py`, the scan, the engine, the ledger, or
  anything under `C:\Users\McKinley Slade\OneDrive\trading_ibkr`.
- No scheduling, no Task Scheduler entries, no email.
- No historical-episode analysis on the site (that stays research/agent work).
- Do not add a page under `pages/` (Streamlit dir must stay flat and this is
  not a Streamlit feature anyway).

## 2. Repo facts you will rely on (verified 2026-08-25)

Repo: `C:\Users\McKinley Slade\dev\New_Seasonals` (GitHub `main`; the site
deploys from GitHub Actions `deploy_site.yml`, 2x/day via `daily_screener`).
Read CLAUDE.md sections "Repo Structure", "Automated Pipeline", the private-site
notes, and "Local Task Scheduler" before starting. Key facts:

- **Deploy pipeline** (`.github/workflows/deploy_site.yml`): generator phase
  pulls inputs from R2 (`scripts/site_r2_pipeline.py pull --phase generator`),
  runs `build_trade_ledger.py --upload`, `daily_seasonal_ideas.py`,
  `scripts/build_atr_downside_stats.py` (best effort), `scripts/build_risk_json.py`
  (required), then `site_r2_pipeline.py publish-generated`; assembler phase
  pulls the exact bundle back and runs `scripts/build_site.py --production`,
  then `scripts/validate_site_freshness.py --out dist --require-r2-provenance`.
  **A new generated JSON must be (1) produced by a generator-phase step, (2)
  registered in `GENERATED_INPUTS` in `scripts/site_r2_pipeline.py`
  (pattern: `R2Input("atr_downside", "atr_downside_stats.json",
  "data/atr_downside_stats.json")`), and (3) copied into `dist/data/` by
  `build_site.py` (pattern: `data/site_risk.json` → `dist/data/risk.json`,
  ~line 2679).** Check whether `validate_site_freshness.py` enumerates
  payloads; if it does, register there too. Model the new step on the
  `build_atr_downside_stats.py` step: best-effort (exit 0 on failure, the card
  then shows "no beta table in this build").
- **Prices**: `data/master_prices.parquet` (long format: `ticker, date, Open,
  High, Low, Close, Volume`; ADJUSTED closes; ~6.8M rows — read with
  `columns=` and `filters=[("ticker","in",[...])]`, never whole). Present in
  the generator phase (pulled from R2). SPY is in it.
- **Universe constants**: `strategy_config.LIQUID_PLUS_COMMODITIES` (~190),
  `strategy_config.CSV_UNIVERSE` (~1060), `strategy_config.ACCOUNT_VALUE`
  (750000 — the primary account's fixed SIZING basis; do not replace it with
  live NLV for the primary).
- **Live book on the site**: `execution.js` polls the execution broker every
  4 s; `state.book.accounts[]` holds one entry per account with `key`
  ("primary"/"pa"), `label`, `nlv`, `positions[]`, `orders[]`, `fills[]`
  (shape from `book_snapshot.py`, OneDrive — read it; the site only sees the
  JSON). `acctBook()` returns the selected account's entry. `bookFresh()` /
  `bookAgeMs()` give staleness (BOOK_STALE_MS = 90 s). Panels are rendered by
  `renderPanels()` via `set(id, html)` into ids that live in
  `site/execution.html` (`modeBanner`, `connBar`, `positions`, `orders`,
  `closers`, `activity`). Follow that pattern: add an id, a `renderHedge()`
  and one line in `renderPanels()`.
  - position fields: `symbol, sec_type, position, avg_cost, market_price,
    market_value, unrealized_pnl, currency` (+ `expiry`, `multiplier`,
    `right`, `strike` for derivatives; `market_price` can be `null` when the
    snapshot degraded — fall back to `avg_cost`, flag it).
  - order fields: `symbol, sec_type, action, qty, order_type, lmt, aux, tif,
    status, good_after, good_till, parent_id, order_id, perm_id, order_ref,
    outside_rth, oca_group, oca_type, client_id`.
- **Order identity tag** (`order_ref`): `SYM|ACTION|Strategy Name|YYYY-MM-DD`
  with an optional 5th `|tranche` (OVS near/far; a literal `nan` shows up on
  some rows — treat any 5th field as opaque). Every leg of a bracket carries
  the PARENT's ref, so a filled long's still-working SELL exits are tagged
  `SYM|BUY|Strategy|date`. Strategy names are the `name` fields in
  `strategy_config.STRATEGY_BOOK` ("Oversold Low Volume", "Overbot Vol Spike",
  …) plus non-book tags ("Momentum_Radar", event sleeve rows are parent-only
  with no exits). Positions with no tagged working order are **Unattributed**.
- **Bracket anatomy**: parent entry (LMT, GTD) + children in one `oca_group`:
  TARGET (`LMT`), optional STOP (`STP`), TIME exit (`MKT` with `good_after` =
  the exit session, e.g. `20260903 15:59:00`), OVS also a PROFIT_TAKER LMT
  and an EOD-DD STP. A bracket whose parent BUY is still in `orders[]` with a
  working status is an **unfilled entry**, not a position. Working statuses:
  `Submitted`, `PreSubmitted`, `PendingSubmit`. Reference implementation of
  exactly this grouping: `discover()` in OneDrive `olv_book_cap.py` (read it;
  port the logic, do not import it — the site is JS).
- **Futures specs**: `site/assets/futures_specs.json` (loaded into
  `FUT_SPECS` by `execution.js` at init): `{"MES": {"multiplier": 5, ...},
  "ES": {...}, ...}`. For an equity-index future, SPY-equivalent notional =
  `position × multiplier × market_price` (the price IS the index level). Treat
  `MES, ES, MNQ, NQ, M2K, RTY, MYM, YM` as equity-index hedges (NQ/RTY/YM are
  imperfect SPY proxies — show them with their own β against SPY if you have
  it in betas.json, else 1.0 with a flag). Everything else (`DX`, rates,
  commodities) is listed but excluded from the equity hedge.
- **JS conventions**: vanilla JS, `"use strict"`, no build step, no external
  CDNs (Cloudflare Pages CSP); helpers in `site/assets/common.js`
  (`fetchJSONOrNull`, `fmt.*`, `esc`); Node-vm tests in `tests/test_risk_site_js.py`
  show how to load a JS file, stub `document`, and assert on rendered HTML —
  copy that harness. Tests must `skipif` when node is missing.
- **Python conventions**: type hints, pathlib, f-strings; tests under `tests/`;
  `py -3.10 -m pytest tests -q` must stay green (886+ tests, ~25 s).

## 3. Deliverable A — `scripts/build_betas.py` → `data/betas.json`

Purpose: a small, nightly table of SPY betas so the browser never touches
price history.

Inputs: `data/master_prices.parquet`, `strategy_config` (universes,
`ACCOUNT_VALUE`).

Universe: `set(CSV_UNIVERSE) | set(LIQUID_PLUS_COMMODITIES) | {"SPY","QQQ","IWM","DIA"}`.
(Everything the scanners can put in the book. If a held symbol is missing at
render time the card shows β = 1.00 with an "assumed" flag — see B.)

Method (keep it this simple; it is not a research surface):
- daily close-to-close returns from adjusted `Close`; SPY returns aligned on
  date;
- `beta_N` = OLS slope of ticker return on SPY return over the last N trading
  days (N = 63 and 252), requiring ≥ 20 paired observations, else `null`;
- `idio_vol63` = std of residuals over 63d (daily, decimal) — informational;
- `spy_last` = last SPY close; `asof` = last SPY date in the cache.

Output schema (`data/betas.json`, compact JSON):

```json
{
  "asof": "2026-08-24",
  "generated_utc": "2026-08-25T21:31:00Z",
  "method": "OLS slope of daily adjusted close returns vs SPY; null if < 20 obs",
  "spy_last": 763.47,
  "account_value": 750000,
  "tickers": {
    "LUV": {"beta63": 1.47, "beta252": 1.56, "idio_vol63": 0.021, "n63": 63, "n252": 252},
    "SPY": {"beta63": 1.0, "beta252": 1.0, "idio_vol63": 0.0, "n63": 63, "n252": 252}
  }
}
```

Wiring:
1. `deploy_site.yml`: add a best-effort step right after
   "Refresh ATR downside stats": `python scripts/build_betas.py` (use
   `continue-on-error`/the same shell guard the ATR step uses).
2. `scripts/site_r2_pipeline.py`: append `R2Input("betas", "betas.json",
   "data/betas.json")` to `GENERATED_INPUTS`. Mirror however `atr_downside`
   is treated for optional-ness (it is best effort; a missing file must not
   fail the deploy — check the `required` flag semantics on `R2Input`).
3. `scripts/build_site.py`: copy `data/betas.json` → `dist/data/betas.json`
   next to the `risk.json` copy; update the module docstring's output list.
4. `validate_site_freshness.py`: it has a `REQUIRED_PORTFOLIO_PAYLOADS`
   tuple and reads per-payload flags from `meta.json` (`meta.get("payloads")`).
   Do **not** add betas to the required tuple. If `build_site.py` stamps
   payload presence into `meta.json`, stamp `betas` there the same way so the
   card can tell "absent in this build" from "failed to load".
5. Commit `data/betas.json`? **No.** Generated payloads are gitignored BY
   NAME (`.gitignore` line ~90 lists `data/site_risk.json`; there is no
   `data/*.json` rule) — add a `data/betas.json` line to `.gitignore` in the
   same commit, or you will commit a 1 MB file on the first run.

Tests (`tests/test_build_betas.py`): synthetic two-ticker frame → known slope
(ticker = 2×SPY + noise ⇒ β≈2), SPY's own β == 1.0 exactly, `< 20` obs ⇒
`null`, output keys/schema, `account_value` == `strategy_config.ACCOUNT_VALUE`.

## 4. Deliverable B — the "Hedge" card

### 4.1 Placement and behaviour
- New `<div id="hedge">` mount. The Exec page's scaffold is NOT in
  `site/execution.html` (that file only loads the script); the panel divs
  are emitted by a template string inside `execution.js` (~line 190, where
  `<div id="closers" style="margin-top:14px"></div>` is written). Add the
  `hedge` div there between the `positions` and `orders` divs — it is a view
  of positions. Rendered by `renderHedge()` from `renderPanels()`, i.e.
  refreshed with the 4-s poll.
- Per selected account tab (Primary / PA), like every other panel.
- Reads: `acctBook()` (positions + orders + nlv), `HEDGE_BETAS` (loaded once
  at init via `fetchJSONOrNull("data/betas.json")`, may be null), `FUT_SPECS`.
- Staleness: if `!bookFresh()` render the card greyed with "book stale (Ns)" —
  same treatment the mode banner gives a stale book. Never hide it.
- No `data-mutation` attributes anywhere in the card. A one-line caption at
  the bottom: "Display only — nothing here sends orders."
- Degradation: no betas payload → all β = 1.00, banner "no beta table in this
  build (build_betas.py skipped)"; symbol missing from the table → β = 1.00
  and an "assumed" marker on that row; `market_price` null → mark at
  `avg_cost` with a marker.

### 4.2 Attribution (pure function, unit-tested)
`attributeBook(account, betas, futSpecs, opts) -> model`

For each equity position (`sec_type === "STK"`, `position !== 0`):
1. Collect that symbol's working orders whose `order_ref` parses as
   `SYM|ACTION|Strategy|date[|tranche]` with `SYM` equal to the position's
   symbol. Group by `oca_group` (fallback: the ref string).
2. A group whose **parent** is still working (an order with the same ref
   whose `action` is the entry side — BUY for a long book, SELL for a short —
   with a working status; link by `parent_id` first, then by ref) is an
   **unfilled entry**: record `{symbol, strategy, qty: parent.qty, lmt:
   parent.lmt}` under `workingEntries`; it is NOT exposure.
3. Otherwise it is a **filled leg**: `qty` = the TIME leg's `qty` (an `MKT`
   order with `good_after`), falling back to the TARGET `LMT` leg;
   `exitDate` = first 8 chars of `good_after` (`YYYYMMDD`) or null.
4. Clamp: legs for a symbol are assigned in order of `exitDate`; each takes
   `min(leg.qty, remaining held)`. Whatever held quantity is left after all
   tagged legs is an **Unattributed** leg for that symbol (strategy
   "Unattributed", exitDate null).
5. Mark = `market_price` (fallback `avg_cost`, flagged). Notional =
   `qty × mark × sign(position)`.
6. β = `betas.tickers[symbol][betaKey]` where `betaKey` is `"beta252"` by
   default with a toggle to `"beta63"` (a `<select>`; persist the choice in
   `localStorage` under `hedge.betaKey`, wrapped in try/catch). Missing → 1.0
   flagged.
7. SPY-equivalent = notional × β.

For each futures position (`sec_type === "FUT"`): if the root symbol is an
equity-index future (list in §2), SPY-equivalent = `position × multiplier ×
market_price × β_index` (β_index from betas.json for the proxy ETF — map
`MES/ES→SPY`, `MNQ/NQ→QQQ`, `M2K/RTY→IWM`, `MYM/YM→DIA` — else 1.0). Others go
to an "other futures" list, excluded from the equity total.

Model output (all numbers in dollars):
```
{
  navBasis: {kind: "sizing"|"live", value},       // primary: betas.account_value (750k); pa: book nlv
  byStrategy: [{strategy, legs, notionalLong, notionalShort, spyEquiv}],
  workingEntries: [{symbol, strategy, qty, lmt, notional}],
  equityLong, equityShort, equitySpyEquiv,           // totals across strategies
  futuresSpyEquiv, futures: [{symbol, position, multiplier, price, spyEquiv, counted}],
  netSpyEquiv,                                       // equitySpyEquiv + futuresSpyEquiv
  rolloff: [{date, remainingSpyEquiv}],              // next 15 sessions, cumulative as legs time-exit
  flags: [...]                                       // assumed betas, avg_cost marks, unparsed refs
}
```

### 4.3 Hedge math (pure function, unit-tested)
`hedgeTarget(model, targetPct, contract) -> {targetDollars, excess, contracts, perContract}`
- `targetDollars = targetPct × navBasis.value` (slider 0–150% in 5% steps,
  default **50%**, persisted in `localStorage` `hedge.targetPct`).
- Strategy scope: a checkbox list of strategies (default: only
  "Oversold Low Volume" checked, plus Unattributed unchecked) — the target
  applies to the SPY-equivalent of the CHECKED strategies, net of ALL counted
  index futures. This mirrors how McKinley framed it ("OLV exposure back to
  50% of NAV") while letting him widen to the whole book.
- `perContract = contract.multiplier × indexLevel` where `indexLevel` = the
  live `market_price` of a held contract of that root if present, else
  `betas.spy_last × 10` for ES/MES (document the ×10 approximation), and the
  contract selector offers MES (default) and ES.
- `excess = scopedSpyEquiv + futuresSpyEquiv − targetDollars` (positive ⇒
  need more short); `contracts = round(excess / perContract)` shown as
  "short N more MES" / "cover N MES" / "at target".

### 4.4 Scenario table
Rows: SPY −2%, −5%, −10%, +5%. Columns: market-driven P&L of the scoped
book **now** (`−shock × (scopedSpyEquiv + futuresSpyEquiv)`), and **at the
slider target** (`−shock × targetDollars`). One caption line: "Market
component only — idiosyncratic moves (single-name, sector clusters) are not
hedged by index futures; roughly two-thirds of OLV variance has been
idiosyncratic historically." Keep that sentence verbatim; it is the one
piece of research the card is allowed to assert.

### 4.5 Roll-off strip
A small inline bar/table: for each of the next 15 sessions, the scoped
SPY-equivalent remaining after legs whose `exitDate` ≤ that date have left
(time exits fire at 15:59 on `exitDate`). Sessions = weekdays; holidays are
NOT handled (say so in a tooltip, consistent with `addTradingDays` in the
same file). Purpose: shows when the hedge should come off.

### 4.6 Layout (top to bottom inside the card)
1. Header row: "Hedge (display only)" · nav basis (`$750,000 sizing` or
   `$66,557 live NLV`) · β window toggle (252d / 63d) · contract (MES / ES).
2. Strategy table: strategy · legs · long $ · short $ · β-wtd SPY-equiv ·
   checkbox (in scope). Totals row. Then "Index futures held" rows with their
   SPY-equiv and "counted" state, and an "Other futures (excluded)" line.
3. Target row: slider + numeric readout ("50% of NAV = $375,000"), then the
   verdict in large type: "short 3 more MES (9 total)" / "at target" /
   "cover 2 MES".
4. Scenario table (4.4).
5. Roll-off strip (4.5).
6. Working entries line: count and $ of unfilled entries in scope, with the
   note "not exposure yet — would push the book to $X if all fill".
7. Flags (assumed betas, avg-cost marks, unparsed refs) in small grey text.
8. Caption: "Display only — nothing here sends orders."

Style: reuse `.card`, `.cap`, existing table classes from `style.css`; add
at most a few rules (e.g. `.hedge-verdict`). Dark/light theme tokens as the
rest of the site.

## 5. Deliverable C — tests and docs

- `tests/test_build_betas.py` (see §3).
- `tests/test_execution_hedge_js.py`: node-vm harness copied from
  `tests/test_risk_site_js.py`. Load `execution.js` with the DOM stubbed,
  then call the pure functions directly (export them on `globalThis` or via
  `module.exports`-style guard like risk.js does — check how that test reaches
  functions). Fixture book (write it as JSON in the test): primary account,
  `nlv 812345`, positions LUV 4914 STK @ 40.81, UNH 396 STK @ 396.46, MESU6
  −6 FUT @ 7635 mult 5, DXU6 +2 FUT; orders: three LUV OLV brackets (SELL LMT
  + SELL MKT with good_after 20260901/02/03, qty 1069/1577/2268, refs
  `LUV|BUY|Oversold Low Volume|2026-08-1x`), two UNH OLV brackets (115/185),
  one UNH bracket with a working BUY parent (qty 292 @ 382.51 → unfilled
  entry), one untagged position remainder (UNH 396 − 300 = 96 → Unattributed).
  Betas fixture: LUV 1.5, UNH 0.4, SPY 1.0; spy_last 763.47; account_value
  750000. Assert: OLV legs = 5, OLV notional = 4914×40.81 + 300×396.46,
  Unattributed = 96 sh, workingEntries = [UNH 292], MES spyEquiv =
  −6×5×7635, `hedgeTarget(…, 0.5, MES)` contracts = round((OLV spyEquiv −
  229050 − 375000)/38175), roll-off after 20260903 excludes all LUV and UNH
  legs, stale book renders the grey state, no `data-mutation` attribute in
  the rendered HTML.
- CLAUDE.md: a short "Hedge panel (Exec tab, display-only)" section under the
  private-site notes: what it reads, what it never does, the β method, the
  `betas.json` contract, and the deploy/R2 registration. Add `betas.json` to
  the build_site docstring's output list.

## 6. Acceptance checklist

- [ ] `python scripts/build_betas.py` runs locally in < 60 s against the
      cached parquet and writes a valid `data/betas.json`; SPY β == 1.0.
- [ ] Deploy workflow has the step; `site_r2_pipeline.py` registers the file;
      `build_site.py` copies it; `validate_site_freshness.py` does not fail
      when it is absent.
- [ ] Card renders on Primary and PA with the live agent; numbers reconcile
      to the Positions panel (sum of attributed + unattributed shares ==
      position per symbol; never more).
- [ ] With betas.json absent the card still renders with β = 1.00 flags.
- [ ] Stale book → grey state; agent offline → grey state; no console errors.
- [ ] No `data-mutation` attributes; nothing in the card calls `sendCommand`
      / the broker.
- [ ] `py -3.10 -m pytest tests -q` green (including the new tests; JS test
      skips cleanly without node).
- [ ] Sanity against §0: with the 2026-08-25 book the card would show OLV
      ≈ $600k, β-wtd ≈ $690–740k, MES −6 ≈ −$229k, target 50% ⇒ ~8–10 MES
      total. If your numbers are far off, your attribution is wrong, not
      the brief.
- [ ] One commit on `main` (rebase onto origin first; do not commit
      `data/*.json` generated files), message in the repo's style, ending
      with the Co-Authored-By line if you are an agent.

## 7. Things that will bite you

- `order_ref` on OVS rows carries a 5th field (`near`/`far`) and some OLV rows
  carry a literal `|nan` — split on `|`, use fields [1] and [2], ignore [4].
- Children of a FILLED parent may show `parent_id` = 0 after a TWS restart;
  link by ref when `parent_id` finds nothing (OneDrive `olv_book_cap.discover`
  does exactly this).
- `market_price` is null when the snapshot degrades — never multiply by null;
  fall back to `avg_cost` and flag.
- The primary's NAV basis is the fixed $750k sizing anchor by convention
  (every cap in the book uses it); the PA's is live NLV. Do not "improve"
  this.
- `master_prices.parquet` is adjusted; that is correct for betas. Do not use
  `rd2_*` or ledger prices.
- Keep `pages/` flat; keep the site CSP-clean (no external scripts/fonts
  beyond what's already used).
- The Exec tab's mutation gating (`execMode()`, `syncMutationControls`) must be
  untouched; your card has no controls that gate.

## 8. Out of scope, deliberately

Automated hedging of any kind; the historical episode analysis (which
episodes had big OLV books at which dial — that lives in the agent
conversation and `scratch/`); anything in the OneDrive execution repo;
option hedges (rejected on evidence in RISK_DIALS_2026-07-16.md §4).
