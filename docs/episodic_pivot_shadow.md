# Episodic Pivot bot — shadow specification

Status: implemented as a local, research-only shadow workflow. Production and order staging are intentionally unavailable.

## Outcome and safety boundary

The process can:

1. import a full, timestamped TradingView premarket or after-hours CSV export;
2. nominate unusual movers using Pradeep Bonde's current discovery floor;
3. search Google and fetch the underlying source pages;
4. separate verified primary evidence from secondary coverage, stale stories, adverse events, and unresolved movers;
5. optionally enrich a candidate with a fresh, read-only IBKR snapshot;
6. calculate a deliberately non-executable liquidity/slippage research preview only when every gate passes; and
7. save a standalone HTML triage report plus hashed replay artifacts.

It cannot write to a broker, Google Sheets, R2, a live staging tab, a scheduler, or the private site. It is not in `STRATEGY_BOOK` or `daily_scan.py`. The policy constructor rejects `live_actions_enabled=True`. Every sizing record fixes `preview_only=true`, `executable=false`, `broker_route=NONE`, `order_submission_allowed=false`, and `production_eligible=false`; its schema is deliberately incompatible with the live order contract.

```text
TradingView full CSV export (premarket or after-hours)
        ↓
broad mover nomination
        ↓
Google URL discovery → actual-page fetch → timestamp/source/excerpt hash
        ↓
primary-source / causal-timing / trajectory-change triage
        ↓
optional fresh read-only IBKR enrichment
        ↓
hypothetical research sizing + standalone HTML report (never executable)
```

## What the Bonde research supports

Bonde's durable idea is not “buy a large gap.” An EP is a surprising new fact that can change the company's trajectory, accompanied by a large and unusual price/volume reaction, preferably after neglect and early in a new earnings or product cycle. The scan creates a research list; catalyst judgment decides whether the event is an EP. His self-reported historical counts are hypothesis-generating observations, not audited backtests. See the [original EP study](https://stockbee.blogspot.com/2007/02/episodic-pivots-and-idea-pickle.html), [March 2007 follow-up](https://stockbee.blogspot.com/2007/03/episodic-pivots.html), and [canonical 2010 explanation](https://stockbee.blogspot.com/2010/02/what-are-episodic-pivots-and-how-to.html).

The implementation preserves distinct setup lanes instead of blending different vintages into one score:

| Lane | Meaning in this process | Immediate preview? |
|---|---|---:|
| `CLASSIC_EP` | Timely, fetched, trajectory-changing business catalyst with verified regulator/primary authority and causal timing | Hypothetical sizing only, if every data gate passes |
| `EP9M_CATALYST` | Extraordinary-volume discovery with the same primary-evidence standard | Hypothetical sizing only, at lower modeled risk |
| `EP9M_STORY_WATCH` | Extraordinary volume without authoritative catalyst evidence | No |
| `STORY_EP_WATCH` | Theme, analyst action, promotion, or weakly classified narrative | No |
| `EXTENDED_GAP_DEP_CANDIDATE` | Same-day gap above 25%; a future DEP still requires a parent event, age, intact thesis, and delayed structure | No |
| `BEARISH_EP_RESEARCH` | Negative mover with relevant adverse evidence; no borrow/SSR/short-execution model exists | No |
| `CATALYST_WITH_FINANCING_RISK` | Positive catalyst accompanied by dilution/offering evidence | No |
| `CORPORATE_ACTION_OR_DISTRESS_REJECT` | Fixed-price deal, reverse split, bankruptcy, or unresolved corporate-action risk | No |

This follows Bonde's later distinction among classic EPs, stories, liquid/institutional events, EP9M, delayed reactions, and bearish variants in his [2024 Trade Factory presentation](https://impact.traders4acause.org/wp-content/uploads/2024/11/02-Pradeep-Trade-Factory-2.pdf). Bearish EPs are recorded in the research taxonomy but deliberately excluded from v0 execution because borrow, SSR, halt, and gap-through mechanics need their own model.

### Rules that changed over time

| Question | Earlier Bonde material | Later Bonde material | Frozen v0 choice |
|---|---|---|---|
| Entry time | Immediate after-hours/premarket buys in the [2014 process](https://stockbee.blogspot.com/2014/07/my-process-flow-for-episodic-pivots-ep.html) | Regular-session opening entry in later interviews; delayed entries increasingly emphasized | Research extended hours; hypothetical regular-session reference price only |
| Very large gaps | Even extreme earnings gaps could continue | [2024 gap study](https://stockbee.blogspot.com/2024/09/gaps-in-50-plus-moves.html) found >20% increasingly pullback-prone and sustained winners rarely began >25% | Warn at 20%; route >25% to delayed-entry watch |
| Stop | Gap low or prior two-day low | EP-day low, then fixed 2.5–4% variants | Prior two-day low plus 10 bps is the only fully known premarket template; alternatives must be tested separately |
| No-progress exit | Five sessions | Three sessions in later comments | Neither is automated in v0; both are named test variants |
| Analyst action | Usually a low-quality catalyst | Contextual exceptions when truly thesis-changing | Watch only; never auto-stage |
| Universe | Neglected, low float, often <25M shares | Also liquid institutional names and cap below roughly $10B | Cap/float/neglect rank the candidate; executable liquidity controls size |

The current discovery floor comes from Bonde's [July 2026 premarket workflow](https://stockbee.blogspot.com/2026/07/how-to-find-and-research-pre-market.html): at least a 2% or $0.90 move, at least 100,000 premarket shares, and price at least $1. The post's literal Boolean formula is ambiguous; the prose clearly makes price and volume mandatory for either move branch, which is how the code is grouped.

## Frozen v0 policy

### Discovery versus research sizing

Discovery is intentionally broad:

- absolute extended-hours move of at least 2% **or** $0.90, in either direction;
- same-session extended-hours volume at least 100,000 shares;
- price at least $1; and
- newest snapshot per symbol and target session retained under a deterministic candidate ID, then the top 25 sorted by session share and estimated dollar volume are researched.

The saved TradingView screens intentionally keep only the price, session-volume, primary-listing, and stock-type universe constraints. They must not contain a ±2% move filter: that would silently drop a $0.90 mover whose percentage change is below 2%. The exact Boolean OR is applied in versioned code after the full CSV export is validated.

A hypothetical sizing preview additionally requires a separate fresh IBKR observation and:

- live, tradeable, non-halted market data no older than 90 seconds;
- a known IBKR halt status, unique resolved stock contract, native primary exchange, and available SMART routing metadata;
- the latest premarket bar no older than 15 minutes and an IB quote previous-close within 0.5% of the raw daily-bar reference;
- bid, ask, displayed ask size, premarket VWAP, prior two-day low, and 63-day ADDV;
- gap at least 4% and no more than 25%;
- price at least $3;
- premarket dollar volume at least $1 million;
- spread no wider than 100 bps;
- ask no more than 2% above premarket VWAP;
- an initial stop no more than 20% below the ask;
- a fetched catalyst document with verified publication metadata that precedes the first recorded mover trigger;
- verified primary authority (regulator/SEC in v0; an issuer-domain identity map is not yet implemented); and
- structured evidence that the event can change the business trajectory, rather than phrase score alone.

The 4% stage floor and execution thresholds are conservative automation choices, not claims that Bonde published one universal entry formula.

### News evidence and catalyst score

Google is a discovery layer. A search result, title, or snippet cannot confirm an EP. The run must fetch an actual page and retain:

- original and canonical URL;
- publisher and source tier;
- publication and retrieval timestamps;
- normalized body excerpt and SHA-256 excerpt hash;
- catalyst class, materiality signals, and adverse flags.

Only verified regulator/SEC authority can establish automatic primary confirmation in v0. Business Wire, GlobeNewswire, and PR Newswire are labeled `ISSUER_WIRE_UNVERIFIED` until authorship is bound to a stable issuer identity. Reuters/AP/Bloomberg and similar outlets are secondary research evidence. One or several secondary pages can improve the report but cannot create a Classic preview. Retrieval is stamped only after fetch, parse, and hash complete; completion after the decision clock is rejected. Page publication metadata or an SEC accepted-at timestamp is required for automatic confirmation. A Google/search fallback timestamp remains research-only. The publication must also precede the first recorded mover trigger, with at most two seconds of clock skew.

The deterministic materiality rubric ranks regulatory approval, clinical data, raised guidance, earnings surprise, persistent demand/backlog, growth acceleration, quantified impact, and commercial milestones. Phrase score cannot by itself enable a preview. Earnings needs surprise plus a raise, acceleration, or persistence; contracts/products need quantified business impact; clinical/regulatory events need an actual endpoint or approval. Ordinary management changes and analyst actions remain watches. Dilution blocks direct long sizing while preserving research context. Fixed-price deals, reverse splits, and bankruptcy/distress are hard rejects. Guidance cuts, failed trials, investigations, restatements, and recalls can enter the separate bearish-research lane, which has no execution model.

Issuer references and event phrases must occur in the same sentence, or in an immediately following sentence explicitly beginning with “the company,” “it,” “management,” or “the board.” This closes ordinary multi-company-roundup leakage, but it is not a full issuer–event semantic extractor. A same-sentence roundup can still be ambiguous, so every local preview remains manual-review-only. A production proposal needs structured issuer evidence (for example SEC/IR identifiers) or a tested entity–event extractor before news classification can be considered sufficient.

Fetched pages are restricted to public HTTPS addresses on port 443. Every DNS answer must be globally routable, and the connection is pinned to a vetted IP while TLS still verifies the original hostname, closing the DNS-rebinding gap. Redirects and canonical URLs are revalidated, private/loopback/link-local destinations are rejected, cross-domain canonical metadata cannot elevate source authority, decoded response bodies are capped at 2 MB, and the body reader enforces an absolute deadline even against slow-drip responses. DNS and HTTP-header handling still rely on bounded connection/read-idle timeouts rather than a cancellable whole-request wall clock, so unattended scheduling remains forbidden until that final availability control is added.

Bonde's catalyst examples and rejection logic are documented in his [2007 catalyst list](https://stockbee.blogspot.com/2007/07/episodic-pivot-catalysts.html), [earnings-surprise notes](https://stockbee.blogspot.com/2007/04/trading-earnings-surprises.html), [2013 research workflow](https://stockbee.blogspot.com/2013/09/night-time-is-right-time.html), and [2014 catalyst hunt](https://stockbee.blogspot.com/2014/09/everyday-hunt-for-stocks-with-big.html).

Google Programmable Search is the preferred provider because it returns direct article URLs. It requires `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_ID`. Credential-free Google News RSS is available as a fallback, but wrapper links that do not resolve to an actual article correctly remain unconfirmed.

### Hypothetical sizing preview

The v0 sizing object is long-only and regular-hours-referential, but it is not an order: it contains no `Action`, `Quantity`, `Order_Type`, `TIF`, `Limit_Price`, `Approval`, `Execute_On`, or `Transmit` fields. Its reference entry is the ask plus modeled slippage, rounded up to the next cent; if that reference would cross the 25% maximum opening gap, no preview is created. The hypothetical stop is the prior two-session low minus 10 bps, rounded down. Any future execution design would have to re-resolve the contract and recheck halt, quote, gap, and liquidity independently; this workflow has no release mechanism.

Expected entry slippage is:

```text
half quoted spread
+ 8 bps base friction
+ 80 bps × sqrt(order notional / 63-day ADDV)
```

It must remain below 100 bps. Modeled risk per share includes the entry-to-stop distance, a 100 bps stressed exit cost, and an additional 2% event-gap stress. This explicitly avoids pretending that a stop guarantees the planned loss.

Maximum preview shares are the minimum of:

```text
risk budget / modeled risk per share
2% of account value / entry limit
0.25% of 63-day ADDV / entry limit
1% of observed premarket shares
25% of displayed ask size
25,000 absolute shares
```

Classic EP risk is 10 bps of the configured $750,000 research account; `EP9M_CATALYST` is 7.5 bps. Those are intentionally below Bonde's published personal-risk examples and should remain shadow assumptions until prospective fills support them.

After individual sizing, previews are scaled pro rata to a 50 bps aggregate modeled-risk cap and a 10% aggregate hypothetical-notional cap. These are research comparisons, not capital-allocation instructions.

## Historical study: what is and is not knowable

The local data can support an observed-panel event census, not a production backtest:

- daily adjusted OHLCV has broad history but current-universe survivorship bias;
- the current FMP company list is not point-in-time membership;
- earnings dates lack reliable before-open/after-close timestamps;
- earnings surprise fields are current-vintage and therefore excluded;
- there is no historical news archive with causal publication times;
- the 15-minute cache is regular-session and only about 197 liquid names; and
- there is no broad historical premarket quote/bar history.

The frozen daily proxy uses only prior-session eligibility at the open:

```text
at least 126 prior bars
prior close >= $3
prior 63-session ADDV >= $5M
event open / prior close - 1 >= 10%
prior 63-session return <= 20%
no earlier volume-confirmed event in the prior 126 sessions
```

Event-day volume at least 2× the prior 20-day average is explicitly labeled **ex-post confirmation**. It cannot justify an event-open fill. An earnings label only says the calendar date matched the event session or prior trading session; it does not establish the release time.

Before any outcome study, the census emits an anomaly table for invalid bars, >50% moves, and half/double price-basis cliffs. Known cache problems such as alternating split bases make this gate necessary. Diagnostic mode can calculate fixed 1/5/20/60-session observations, but it is labeled survivor-biased and non-tradeable; it is not a strategy P&L.

### Current local diagnostic result

The 2026-08-25 run over the current FMP operating-company slice found:

| Census stage | Count |
|---|---:|
| Companies with local history | 1,206 |
| Daily bars | 5,772,117 |
| Open-observable candidates | 5,934 |
| Ex-post 2× volume confirmed | 4,511 |
| First confirmed event in prior 126 sessions | 3,126 |
| Strict first-event plus neglect proxy | 2,470 |
| Strict events left clean for diagnostics | 2,413 |
| Strict events matched to an earnings date | 1,067 |

The anomaly table contains 33,831 zero/invalid legacy bars, 370 half/double cliffs, and 1,302 extreme-move flags. Fifty-seven strict events were excluded from outcomes because an anomaly was on or near the event.

The mechanical proxy was weak at ordinary horizons. From the next session's open, SPY-excess mean/median returns were -0.21%/-0.29% at five sessions, +0.27%/-0.21% at 20, and +2.83%/+0.17% at 60. Equal-weight event-date bootstrap 95% intervals were [-0.60%, +0.18%], [-0.59%, +0.77%], and [+0.84%, +3.82%], respectively; issuer-cluster intervals were [-1.14%, -0.16%], [-1.30%, +0.44%], and [+0.74%, +4.01%].

The one-time 2024–2026 evaluation slice was worse in the middle of the distribution: next-open SPY-excess mean/median was -0.77%/-1.34% at five sessions, -0.98%/-2.35% at 20, and +2.35%/-2.41% at 60. Because those results are now published, this slice is a **consumed holdout** and cannot support a future untouched-holdout claim after policy critique or tuning. The next genuinely untouched test must be prospective shadow data or a later frozen vintage. The positive long-horizon mean alongside a negative median is a right-tail signature, not a general candidate-level edge. It reinforces Bonde's central distinction: gap/volume is a nomination mechanism, while catalyst quality and a small number of outsized winners drive the concept. These figures still inherit every survivorship, timing, adjustment, and ex-post-volume limitation above.

The run lives under `artifacts/episodic_pivot/historical-diagnostic-20260825T014152Z/`; `diagnostic_clustered.csv` reports date- and issuer-clustered sensitivity rather than treating every row as independent.

## Running the shadow process

### 1. Export and validate the two TradingView screens

The current saved screens are [Premarket EP](https://www.tradingview.com/screener/yftOvM3e/) and [After-hours EP](https://www.tradingview.com/screener/Hqgnyp7Y/). Keep price at least $1, the corresponding extended-session volume at least 100,000, and the desired primary-listing/security-type universe. Do **not** apply a change-percent filter. Include the official session-specific Price, Change, Change %, and Volume columns, export the full CSV, and note the result count TradingView displays.

The importer is dry-run-only unless `--write-artifact` is supplied:

```powershell
python scripts/import_tradingview_ep.py `
  --input artifacts/episodic_pivot/imports/premarket.csv `
  --session premarket `
  --captured-at 2026-08-25T08:30:00-04:00 `
  --screen-id yftOvM3e `
  --reported-count 37

python scripts/import_tradingview_ep.py `
  --input artifacts/episodic_pivot/imports/premarket.csv `
  --session premarket `
  --captured-at 2026-08-25T08:30:00-04:00 `
  --screen-id yftOvM3e `
  --reported-count 37 `
  --write-artifact
```

After-hours captures map to the next actual NYSE session; Friday and pre–Good Friday exports therefore map through the weekend correctly. A header-only export with a displayed count of zero is a valid completed scan. Any count mismatch, duplicate symbol, malformed number, missing session-specific field, timezone-less capture, non-trading date, or out-of-window capture fails the whole import. TradingView rows are stamped `BROWSER_EXPORT`, `tradeable=false`, unknown halt state, and no bid/ask/VWAP/contract data. They can start research but can never create a sizing preview.

### 2. Run the research funnel

The research runner is also dry by default: it validates inputs but makes no network request and writes nothing.

```powershell
python scripts/run_episodic_pivot_shadow.py `
  --snapshot artifacts/episodic_pivot/imports/2026-08-25-premarket-HASH.json `
  --news-mode google-news `
  --target-session-date 2026-08-25
```

To perform the read-only network research and write local artifacts:

Credential-free Google News discovery:

```powershell
python scripts/run_episodic_pivot_shadow.py `
  --snapshot artifacts/episodic_pivot/ibkr_snapshot_YYYYMMDDTHHMMSSZ.json `
  --news-mode google-news `
  --allow-network `
  --run-research
```

Preferred Google Programmable Search:

```powershell
$env:GOOGLE_CSE_API_KEY = '<local secret>'
$env:GOOGLE_CSE_ID = '<local search engine id>'
python scripts/run_episodic_pivot_shadow.py `
  --snapshot artifacts/episodic_pivot/ibkr_snapshot_YYYYMMDDTHHMMSSZ.json `
  --news-mode google-cse `
  --allow-network `
  --run-research
```

Unverified fixture replay (classification/audit only; it cannot create a preview):

```powershell
python scripts/run_episodic_pivot_shadow.py `
  --snapshot tests/fixtures/ep_snapshot.json `
  --evidence tests/fixtures/ep_evidence.json `
  --as-of 2026-08-24T12:31:00Z `
  --target-session-date 2026-08-24 `
  --run-research
```

Verified replay after a network run refreshed the news evidence:

```powershell
python scripts/run_episodic_pivot_shadow.py `
  --snapshot artifacts/episodic_pivot/ibkr_snapshot_FRESH.json `
  --evidence artifacts/episodic_pivot/EP-RUN-SOURCE/evidence_by_symbol.json `
  --evidence-manifest artifacts/episodic_pivot/EP-RUN-SOURCE/manifest.json `
  --run-research
```

The source manifest must come from a network research run, its run directory must match its `run_id`, and its recorded SHA-256 must match `evidence_by_symbol.json`. Arbitrary offline JSON is stamped `UNVERIFIED_REPLAY` and cannot become preview-eligible. This provides a tamper-evident local provenance chain; it is not a signature against a malicious local operator.

Each run writes `manifest.json`, `candidates.json`, candidate-ID and symbol-keyed evidence files, `decisions.json`, `research_sizing_preview.json`, `research_sizing_preview.csv`, `report.md`, and a standalone `report.html`. The manifest hashes every artifact and records that publishing, staging, and broker contact did not occur. Rerunning identical offline inputs uses the same run ID. CSV cells are formula-escaped and HTML is escaped while JSON retains raw source text for audit.

### 3. Optional targeted IBKR enrichment

TradingView supplies broad discovery; IBKR supplies fresh executable-market facts only for the resulting bounded candidate list. The adapter is dry by default and imports `ib_insync` only after `--capture` is explicitly supplied:

```powershell
python scripts/capture_ep_premarket_ibkr.py `
  --symbols-from artifacts/episodic_pivot/imports/2026-08-25-premarket-HASH.json `
  --port 7497

python scripts/capture_ep_premarket_ibkr.py `
  --symbols-from artifacts/episodic_pivot/imports/2026-08-25-premarket-HASH.json `
  --port 7497 `
  --capture
```

Capture is restricted to 04:00–09:25 ET and connects with `readonly=True`. For each TradingView target it qualifies one USD stock contract, cross-checks the primary exchange, fetches raw daily and extended-hours bars, derives the first actual 5-minute trigger timestamp, requests a bounded live quote/halt watch, then cancels every subscription. It exposes no order API. If no target file is supplied, the older IBKR rank-limited scanner union remains available as a clearly labeled non-exhaustive fallback.

### 4. Daily shadow cadence (not scheduled)

- 7:45 PM ET: export after-hours, validate, and begin evidence collection for the next session.
- 6:30 AM ET: export premarket, merge by target session/ticker, and run the main research pass.
- 8:30 AM ET: export a fresh delta, enrich new/high-priority names through read-only IBKR, and reuse verified evidence where hashes/provenance match.
- 8:50 AM ET: freeze the HTML review report.

These are operator times only. No Task Scheduler task, GitHub cron, browser-cookie export, email, upload, or deployment has been created. A missing TradingView login must produce a fresh capture failure; yesterday's rows must never substitute for today's scan.

### 5. Historical census

```powershell
python scripts/study_episodic_pivot_history.py `
  --data-root 'C:\Users\McKinley Slade\dev\New_Seasonals\data' `
  --mode census
```

`--mode diagnostic` adds fixed-horizon observations only after the anomaly output has been reviewed. It does not make the study point-in-time or production-ready.

## Review decisions before any production proposal

The highest-value critiques are:

1. whether 4% is the right immediate-entry floor after the broader 2%/$0.90 nomination;
2. whether all >25% gaps should be delayed-entry watches or whether exceptional regulatory/earnings cases deserve an override;
3. prior-two-day-low versus a fixed 2.5–4% initial stop;
4. 10/7.5 bps risk and the 2% gap-stress haircut;
5. whether material contracts/products need a mandatory quantified-economics field;
6. whether a story/EP9M lane should ever become stageable; and
7. which Google provider and IBKR market-data subscriptions will be used.
8. which point-in-time company universe and IBKR batch size should replace scanner-only sampling before production.

A later production proposal should first accumulate prospective shadow decisions and quotes, compare preview limits with actual opening prints, validate spreads/depth/halts, and freeze the policy. Only then should a separate dedicated review tab and executor be designed with exact approval, date, freshness, deduplication, and absent-by-default activation controls. Nothing in this implementation performs that activation.
