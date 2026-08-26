# Episodic Pivot bot — shadow specification

Status: implemented as a local, research-only shadow workflow. The active Codex heartbeat runs twice on weekdays at 08:20 and 19:20 ET and delivers the night queue, morning report, or a fail-closed alert by email. Production and order staging are intentionally unavailable.

## Outcome and safety boundary

The process can:

1. import a full, timestamped TradingView premarket or after-hours CSV export;
2. nominate unusual movers using Pradeep Bonde's current discovery floor;
3. search Google and fetch the underlying source pages;
4. separate verified primary evidence from secondary coverage, stale stories, adverse events, and unresolved movers;
5. optionally enrich a candidate with a fresh, read-only IBKR snapshot;
6. calculate a deliberately non-executable liquidity/slippage research preview only when every gate passes; and
7. save a standalone HTML triage report plus hashed replay artifacts.

It cannot write to a broker, Google Sheets, R2, a live staging tab, or the private site. It is not in `STRATEGY_BOOK` or `daily_scan.py`. The policy constructor rejects `live_actions_enabled=True`. Every sizing record fixes `preview_only=true`, `executable=false`, `broker_route=NONE`, `order_submission_allowed=false`, and `production_eligible=false`; its schema is deliberately incompatible with the live order contract. The Codex heartbeat may create only local research artifacts and reports; it is not an activation path.

```text
TradingView full CSV export (premarket or after-hours)
        ↓
broad mover nomination
        ↓
fresh read-only IBKR enrichment → strict prior ATR% gate
        ↓
Google URL discovery → actual-page fetch → timestamp/source/excerpt hash
        ↓
primary-source / causal-timing / trajectory-change triage
        ↓
bounded fresh read-only quote recapture
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
- the latest premarket bar no older than 15 minutes and an IB quote previous-close within 0.5% of the adjusted daily-bar reference;
- at least 126 completed daily bars and a verified adjusted-basis prior ATR(14)% strictly greater than 4%;
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

Issuer references and event phrases must occur in the same issuer-bearing clause, or in an immediately following sentence explicitly beginning with “the company,” “it,” “management,” or “the board.” Report-of/while/whereas clause boundaries prevent a candidate from inheriting a named peer's event in common multi-company roundups, but this is not a full issuer–event semantic extractor. More complex sentences can still be ambiguous, so every local preview remains manual-review-only. A production proposal needs structured issuer evidence (for example SEC/IR identifiers) or a tested entity–event extractor before news classification can be considered sufficient.

Fetched pages are restricted to public HTTPS addresses on port 443. Every DNS answer must be globally routable, and the connection is pinned to a vetted IP while TLS still verifies the original hostname, closing the DNS-rebinding gap. Redirects and canonical URLs are revalidated, private/loopback/link-local destinations are rejected, cross-domain canonical metadata cannot elevate source authority, decoded response bodies are capped at 2 MB, and the pinned transport shares an absolute per-page deadline across connection, headers, redirects, and body reads. A failure is recorded as unresolved evidence; it cannot be promoted into a catalyst confirmation.

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
- ordinary historical news coverage is incomplete and many vendor timestamps lack a timezone;
- the 15-minute cache is regular-session and only about 197 liquid names; and
- there is no broad historical premarket quote/bar history.

The frozen daily proxy uses only prior-session eligibility at the open:

```text
at least 126 prior bars
prior close >= $3
prior 63-session ADDV >= $5M
event open / prior close - 1 >= 10%
simple ATR(14) through the prior session / prior close > 4% (strict, unrounded)
prior 63-session return <= 20%
no earlier volume-confirmed event in the prior 126 sessions
```

The 14 true ranges consume 15 completed source bars, and those bars must be
consecutive NYSE sessions. Any missing session, invalid bar, half/double, or
>=50% basis cliff in that input window fails the ATR gate closed. The event day
cannot enter ATR. A low-ATR 10%/2x-volume gap still resets the 126-session
novelty clock; otherwise the volatility filter would incorrectly relabel a later
repeat gap as the first EP.

Event-day volume at least 2× the prior 20-day average is explicitly labeled **ex-post confirmation**. It cannot justify an event-open fill. An earnings label only says the calendar date matched the event session or prior trading session; it does not establish the release time.

Before any outcome study, the census emits an anomaly table for invalid bars, >50% moves, and half/double price-basis cliffs. Known cache problems such as alternating split bases make this gate necessary. Diagnostic mode can calculate fixed 1/5/20/60-session observations, but it is labeled survivor-biased and non-tradeable; it is not a strategy P&L.

### Current local diagnostic result

The 2026-08-25 run over the current FMP operating-company slice found:

| Census stage | Count |
|---|---:|
| Companies with local history | 1,206 |
| Daily bars | 5,772,872 |
| Open-observable candidates | 5,934 |
| Ex-post 2× volume confirmed | 4,511 |
| First confirmed event in prior 126 sessions | 3,126 |
| Strict first-event plus neglect proxy, before ATR | 2,470 |
| Rejected: ATR source window unclean/missing | 20 |
| Rejected: ATR source sessions not consecutive | 0 |
| Rejected: clean prior ATR% <= 4% | 1,141 |
| Strict events after prior ATR% > 4% | 1,309 |
| Default diagnostic population after event-basis review | 1,300 |
| Event-day half/double closes retained only in the inclusion sensitivity | 9 |
| ATR-qualified events matched to an earnings date | 392 |

The anomaly table contains 35,080 rows: 33,831 zero/invalid legacy-bar flags,
370 half/double-cliff flags, and 1,302 extreme-move flags, with some overlap.
Nine ATR-qualified events closed near half/double the prior close. Price data
alone cannot distinguish a genuine doubling from a split-basis problem, so the
events remain in the eligible-event ledger but are excluded from the default
diagnostic population. A separately named sensitivity includes them. A
legitimate >=50% event-day gap that is not a half/double cliff therefore remains
eligible and auditable instead of being mechanically censored; an extreme move
inside the *prior* ATR source window still fails the ATR gate closed. The earlier
centered five-session flag was removed because a large move after an EP could
leak backward and change inclusion.

The available-case cohorts differ by horizon because recent events are
right-censored. In the 1,300-event basis-review-cleared population, next-open
SPY-excess mean/median was -0.12%/-0.13% at five sessions (N=1,295),
+0.98%/+0.01% at 20 (N=1,266), and +5.72%/+1.59% at 60 (N=1,252). On the
same balanced 1,252-event cohort, the corresponding figures were
-0.02%/-0.05%, +1.13%/+0.08%, and +5.72%/+1.59%. Equal-weight event-date
bootstrap 95% intervals on the available cohorts were [-0.81%, +0.50%],
[-0.53%, +1.90%], and [+2.04%, +7.53%]; issuer-cluster intervals were
[-1.24%, +0.09%], [-0.80%, +1.39%], and [+2.63%, +7.16%]. Including the nine
unresolved basis-review events changes the 20-session date-cluster mean from
+0.69% to +0.79%; both intervals cross zero. The later positive mean is a
descriptive right-tail observation, not evidence that the typical event works
or that 4% is the optimal volatility threshold.

The 2024–2026 rows are machine-labeled `CONSUMED_HOLDOUT_2024_2026`. After
excluding five basis-review rows in that slice, next-open SPY-excess
mean/median was -0.49%/-1.52% at five sessions (N=370), -0.14%/-3.12% at 20
(N=341), and +6.12%/-0.82% at 60 (N=327). On the balanced 327-event cohort the
five/20/60-session means were -0.17%, +0.39%, and +6.12%, with medians
-1.33%, -2.61%, and -0.82%. Including the five flagged rows moves the available
20-session mean from -0.14% back to +0.22%; the date- and issuer-cluster means
move from -0.44%/-0.42% to -0.08%/+0.03%, and every interval crosses zero.
This slice cannot support a future untouched-holdout claim. The next untouched
test must be prospective shadow data or a later frozen vintage. All figures
still inherit the survivorship, timing, adjustment, and ex-post-volume limits
above.

The rerun lives under
`artifacts/episodic_pivot/historical-diagnostic-20260825T233943Z/`.
`diagnostic_horizon_comparison.csv` reports available and balanced cohorts,
`diagnostic_clustered.csv` reports date- and issuer-clustered results without
the unresolved basis rows, and
`diagnostic_clustered_including_event_half_double_review.csv` is the explicitly
named nine-event inclusion sensitivity. Its candidate parquet SHA-256 is
`d6e13fc73e451bb61f1ec18dbf7361b09a9a4f5dc6e5a3af6a112ff59984f112`, the
same input hash frozen by the v6 historical-news manifest below.

### Expected cadence and descriptive winner traits

Counts below are strict, basis-review-cleared historical proxy events, not raw TradingView rows and not news-confirmed entries. The recent planning baseline is **about 2–3 names per calendar week and 10–12 per month**. From 2024 through 2026-08-14, the mean was 2.74 per week and 11.72 per month; the active-week median was two, the weekly P90 was seven, and 19% of calendar weeks had no event. The last two complete calendar years produced 110 and 125 events. The 2026 sample had already reached 140 by August 14, so capacity should be designed for at least 150–200 strict candidates per year rather than assuming 125 is a ceiling. These counts still inherit the current-universe survivor panel and ex-post event-volume confirmation. The eventual number of primary-news-confirmed EPs is not historically estimable; the prospective shadow log is intended to measure it.

“Winner” here means a frictionless next-open-to-close observation relative to SPY, not a strategy return. Among 1,266 events with a 20-session outcome, the top-decile cutoff was +22.64%; among 1,252 with a 60-session outcome, it was +40.70%. Only 53 events belonged to both top deciles. A separate durable cohort—positive at 20 sessions and top-decile at 60—contained 100 events and had median excess observations of +24.10% at 20 sessions and +61.10% at 60. The 20-session overall mean was almost entirely tail-driven; the 60-session positive location survived moderate tail trimming but remains descriptive.

| Trait median | All events | Top 20-session | Top 60-session | Durable |
|---|---:|---:|---:|---:|
| Opening gap | 13.85% | 14.53% | 15.11% | 15.23% |
| Event-day relative volume | 3.70× | 3.74× | 3.24× | 3.38× |
| Prior ATR(14)% | 5.75% | 6.86% | 7.04% | 6.79% |
| Prior 63-session return | -16.48% | -21.71% | -22.01% | -21.83% |
| Prior 63-session ADDV | $41.36M | $35.94M | $37.98M | $40.36M |

Higher prior ATR was the only measured trait whose winner enrichment repeated across the full, 2020+, and consumed 2024–2026 samples. Within the already-filtered population, durable-winner incidence rose from 3.5% in the lowest local ATR quartile to 12.1% in the highest. This is exploratory, outcome-selected evidence: it supports preserving continuous ATR as a research-priority feature, not raising the strict >4% gate or changing risk. Gap size, relative volume, liquidity, and prior trend did not discriminate consistently across eras.

The largest durable 60-session observations were VKTX on 2023-12-04 (+453.14% excess), IONQ on 2024-09-27 (+318.76%), IBRX on 2020-04-14 (+318.45%), SIRI on 2003-03-10 (+295.81%), and FSLY on 2020-05-07 (+233.97%). These are outliers, not expected returns. Event-date clustering is material: 2020-11-09 alone contributed seven durable events. The review report therefore displays same-day cluster context and treats a basket of apparent winners as correlated event risk.

Historical news cannot yet explain why those names won. There are zero point-in-time-identity-validated primary catalyst labels, every historical trajectory label is unresolved, and current-CIK/earnings-date matches are era-confounded. No historical news category is therefore a mandatory gate. Going forward, the shadow process freezes issuer identity, exact accepted/published time, source text hash, quantified surprise, novelty, causal timing, quote state, and subsequent 20/60-session paths. That prospective record is what can eventually distinguish fast continuation from durable rerating and test which catalyst traits actually matter.

The deterministic analysis can be rebuilt with:

```powershell
python scripts/analyze_episodic_pivot_process.py `
  --candidates artifacts/episodic_pivot/historical-diagnostic-20260825T233943Z/historical_candidates.parquet `
  --evidence artifacts/episodic_pivot/historical-news-20260825T233751Z/event_evidence.parquet `
  --output-dir artifacts/episodic_pivot/process-review-YYYYMMDD
```

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

After-hours captures map to the next actual NYSE session; Friday and pre–Good Friday exports therefore map through the weekend correctly. A header-only export with a displayed count of zero is a valid completed scan. Any count mismatch, duplicate symbol, malformed number, missing session-specific field, timezone-less capture, non-trading date, or out-of-window capture fails the whole import. TradingView rows are stamped `BROWSER_EXPORT`, `tradeable=false`, unknown halt state, and no bid/ask/VWAP/contract or ATR data. They can nominate research targets but can never create a sizing preview. The required daily order is TradingView discovery -> read-only IBKR enrichment -> prior ATR% gate -> news research.

### News research (only after IBKR enrichment)

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

### Required targeted IBKR enrichment before EP treatment

TradingView supplies broad discovery; IBKR supplies fresh executable-market facts only for the resulting bounded candidate list. The adapter is dry by default and imports `ib_insync` only after `--capture` is explicitly supplied:

```powershell
python scripts/capture_ep_premarket_ibkr.py `
  --symbols-from artifacts/episodic_pivot/imports/2026-08-24-after-hours-HASH.json `
  --symbols-from artifacts/episodic_pivot/imports/2026-08-25-premarket-HASH.json `
  --max-captured 150 `
  --port 7497

python scripts/capture_ep_premarket_ibkr.py `
  --symbols-from artifacts/episodic_pivot/imports/2026-08-24-after-hours-HASH.json `
  --symbols-from artifacts/episodic_pivot/imports/2026-08-25-premarket-HASH.json `
  --max-captured 150 `
  --port 7497 `
  --capture
```

`--symbols-from` may be repeated. The loader validates that every import targets the same NYSE session, preserves the newest symbol observation, records every contributing screen ID, and applies the broad 2%/$0.90 move rule before enforcing the 150-name capture bound. A mixed-session input or more than 150 qualifying targets fails closed rather than silently truncating the queue.

Capture is restricted to 04:00–09:25 ET and connects with `readonly=True`. For each TradingView target it qualifies one USD stock contract, cross-checks the primary exchange, fetches `ADJUSTED_LAST` daily bars plus raw extended-hours trades, derives the first actual 5-minute trigger timestamp, requests a bounded live quote/halt watch, then cancels every subscription. [IBKR documents](https://interactivebrokers.github.io/tws-api/historical_bars.html) `TRADES` as split-adjusted but not dividend-adjusted, so it is not accepted for the daily ATR path; an unverified basis creates `PRIOR_ATR_PRICE_BASIS_UNVERIFIED`. The capture requires at least 126 completed daily bars and computes the same simple ATR(14) used by the historical census. All 15 source bars must be consecutive NYSE sessions and pass the invalid/half-double/>=50% basis checks. A sizing preview is blocked when `100 * ATR(14) / previous close <= 4`, and missing ATR is `PRIOR_ATR_UNRESOLVED`, not evidence of low volatility. It exposes no order API. If no target file is supplied, the older IBKR rank-limited scanner union remains available as a clearly labeled non-exhaustive fallback.

### 4. Automation-ready shadow cadence

- **7:20 PM ET, Monday–Friday — night phase of `EP Night and Morning Shadow Process`:** use the signed-in Codex in-app browser to refresh the saved after-hours screen, verify its identity, required filter/column state, and displayed count, export the complete CSV, and import it with an exact timezone-aware capture time. The run stores a validated queue for the next NYSE session and emails the inline queue summary plus the normalized JSON attachment. It does not contact IBKR or news providers.
- **8:20 AM ET, Monday–Friday — morning phase of `EP Night and Morning Shadow Process`:** skip non-session days; refresh and validate the saved premarket screen in the in-app browser; export and import the complete CSV; merge it with the uniquely matching prior-night queue; apply the broad local mover rule; and capture at most 150 targets through read-only IBKR.
- **After the first morning capture:** block ATR-unresolved, unverified adjusted-basis, and prior ATR% <=4 names before the main network news pass. Research at most the configured 25 names, using Google Programmable Search when its local credentials exist and credential-free Google News otherwise.
- **Before the final morning report:** consume the network run's hashed `refresh_targets.json`, recapture only that researched subset through read-only IBKR, and replay the verified evidence against the fresh snapshot. The complete HTML report is sent as the email body; `report.html`, `report.md`, `research_sizing_preview.csv`, and `manifest.json` are attached. These are review artifacts, not order files.

The news-request budget is applied only after the ATR/basis gate. Low-ATR,
ATR-unresolved, and basis-unverified movers remain visible in the audit decisions with
`NEWS_RESEARCH_SKIPPED_PRIOR_ATR`, but they never consume a search request or
displace a >4% candidate. ATR-qualified names beyond the configured research cap
remain visible as `NEWS_RESEARCH_NOT_SELECTED_BY_CAP`.

The active thread-attached heartbeat is `EP Night and Morning Shadow Process` (`ep-after-hours-shadow-queue`). One recurrence carries both weekday phases. Email is the research-delivery channel; the Codex task retains only a minimal delivery status or failed-run notification and does not carry ticker or research content. The heartbeat does not run Git, commit, push, upload, publish, deploy, write Sheets, or access any broker order endpoint. The morning run may continue premarket-only when the prior-night queue is absent, but it must disclose that degraded coverage and may never substitute a stale queue. A missing TradingView login, saved-screen mismatch, missing required column, count mismatch, ambiguous download, mixed target date, IBKR connection/data failure, or capture outside its allowed session fails closed and triggers a failure email. A zero-result validated export is a successful empty scan and is still emailed.

Email delivery is dry-run by default and requires the explicit `--send` flag. Credentials are read from the explicitly supplied env file without being copied into the worktree. Recipients resolve in this order: `EP_RECIPIENTS`, `RECIPIENTS`, then the `EMAIL_USER` sender. Successful sends write a non-sensitive receipt containing the source digest and recipient count, but no address or password; a matching receipt prevents an accidental duplicate send. A successful receipt for a different artifact or recipient set requires an explicit reviewed `--resend`.

```powershell
python scripts/send_episodic_pivot_email.py `
  --kind night `
  --artifact artifacts/episodic_pivot/imports/2026-08-24-after-hours-HASH.json `
  --env-file 'C:\Users\McKinley Slade\dev\New_Seasonals\.env' `
  --send

python scripts/send_episodic_pivot_email.py `
  --kind morning `
  --artifact artifacts/episodic_pivot/EP-RUN-ID `
  --env-file 'C:\Users\McKinley Slade\dev\New_Seasonals\.env' `
  --send
```

The final run manifest hashes `refresh_targets.json` and stamps it with `record_type=EP_RESEARCH_QUOTE_REFRESH_TARGETS_V1`, `research_only=true`, `broker_route=NONE`, and `order_submission_allowed=false`. Names skipped by the ATR gate or research cap are excluded from that recapture list.

### 5. Historical census

```powershell
python scripts/study_episodic_pivot_history.py `
  --data-root 'C:\Users\McKinley Slade\dev\New_Seasonals\data' `
  --mode census
```

`--mode diagnostic` adds fixed-horizon observations only after the anomaly output has been reviewed. It does not make the study point-in-time or production-ready.

### Historical news-flow enrichment

The historical label functions receive ticker, event date, prior-session
features, and ATR state but no forward-return columns. SEC submissions are
issuer-batched and expanded through relevant historical chunks; FMP stock news
is queried in the same bounded date window for every event. The command's parent
process does load the outcome-bearing source dataframe, so the implementation is
**column-isolated, not process-isolated, blinding**. The freeze manifest records
that limitation explicitly.

```powershell
python scripts/enrich_episodic_pivot_history.py `
  --candidates artifacts/episodic_pivot/historical-diagnostic-RUN/historical_candidates.parquet `
  --data-root 'C:\Users\McKinley Slade\dev\New_Seasonals\data' `
  --env-file 'C:\Users\McKinley Slade\dev\New_Seasonals\.env'
```

After the provider cache is frozen, add `--cache-only` for a deterministic
rebuild that cannot contact either provider. The output separates
`blinded_events.parquet`, `document_ledger.parquet`, `event_evidence.parquet`,
provider failures, and `post_freeze_outcomes.parquet`. `label_freeze.json`
hashes every decision-bearing label file before outcomes are joined.

The candidate availability window is the prior regular close through 09:30 ET.
An exact SEC acceptance time receives a three-minute availability proxy.
NYSE opens and closes, including early closes, come from the exchange calendar.
`PREOPEN_SEC_ASSUMED_PUBLIC` means only that this assumed public time landed in
the window; it is not proof of dissemination or causation. A rule-based SEC type
can become primary only when the event-date ticker/CIK identity is validated by
an optional point-in-time crosswalk. The current FMP profile CIK is not enough.
Without that crosswalk, classifiable filings remain
`PREOPEN_SEC_ASSUMED_PUBLIC_IDENTITY_UNRESOLVED_CLASSIFIED` and the primary type
and trajectory remain unresolved. FMP headlines and text can identify secondary
context, but their timezone-naive timestamp keeps them `TIMING_UNRESOLVED` and
historical FMP direction is disabled. Known passive-holdings and law-firm
solicitation templates remain in the raw document ledger and raw-count columns
but are excluded from decision counts, event types, and catalyst decisions.
Issuer-bearing clauses are extracted before classification so a candidate does
not inherit another company's event merely because both appear in one sentence.
This remains a conservative heuristic, not a general semantic parser. Empty
provider results stay `COVERAGE_UNRESOLVED`.

The complete cache-only rebuild
`historical-news-20260825T233751Z` labeled all 1,309 ATR-qualified events and
retained 6,863 deduplicated source records:

| Historical evidence posture | Events |
|---|---:|
| Point-in-time-identity-validated primary disclosure | 0 |
| Classifiable pre-open SEC context; identity unresolved | 696 |
| Unclassified pre-open SEC context; identity unresolved | 86 |
| Date-matched flow; exact timing unresolved | 185 |
| Stale/post-open context only | 62 |
| Coverage unresolved | 280 |

Every SEC statistic below uses a **current-CIK-matched, historically
identity-unvalidated** link. Within that limitation, pre-open SEC context was
dominated by earnings/guidance (656 events, 50.1% of all candidates), followed
by other material filings (75), financing/dilution (18), M&A/strategic (12),
management/governance (10), and unclassified material agreements (10).

Issuer-clause-bound secondary FMP context was present for 674 events: 413 were
earnings/guidance, 196 unclassified, 16 M&A/strategic, 15
regulatory/clinical, 10 product/customer/contract, seven analyst actions, six
legal/investigation, four each financing/dilution and distress/restructuring,
and three management/governance. These are date-matched context categories,
not evidence that the story preceded or caused the gap. All 1,309 secondary
trajectory postures are deliberately `TRAJECTORY_UNRESOLVED`.

The provider audit recorded 1,113 SEC document records whose acceptance-plus-
three-minute proxy fell before the open, 1,309/1,309 FMP cache hits,
1,304/1,309 SEC cache hits, and five unresolved current-vintage CIK mappings.
The ledger retained 367 known low-signal legal solicitations and seven holdings
updates while excluding them from decision counts and event types; a residual
audit covering named law firms plus obvious class-action/contact/deadline/click-
through templates found zero standard-discovery rows. The event summary now
stores both raw and decision-only article counts, so a solicitation-heavy event
cannot appear richly corroborated after its documents are excluded. Schema v6
freezes the four label files before outcomes are joined.

After labels were frozen, the 630 basis-review-cleared,
current-CIK-matched earnings/guidance events with a 20-day outcome had a
date-cluster SPY-excess mean of +1.08% with a 95% interval of
[-0.27%, +2.43%] and an issuer-cluster mean of +0.91% with an interval of
[-0.43%, +2.24%]. The intervals cross zero and the identity link is
not point-in-time, so this is not a validated earnings-EP effect. Historical FMP
positive/adverse comparisons were removed after audit found reaction-story,
cross-company, and resolution/negation failures. Directional news efficacy must
be tested prospectively from timestamped shadow evidence. This census supports
the architecture for that test; it does not validate a mandatory gate, an
automatic trajectory signal, or any historical trade rule.

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
