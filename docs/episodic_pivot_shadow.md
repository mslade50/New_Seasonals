# Episodic Pivot bot — shadow specification

Status: implemented for research and local order previews only. Production is intentionally unavailable.

## Outcome and safety boundary

The process can:

1. capture a timestamped, read-only IBKR premarket snapshot;
2. nominate unusual movers using Pradeep Bonde's current premarket discovery floor;
3. search Google and fetch the underlying source pages;
4. distinguish a confirmed classic catalyst from an unconfirmed mover, EP9M/story watch, adverse event, or delayed-entry watch;
5. create a liquidity- and slippage-capped local order preview; and
6. save an evidence-complete run manifest for replay.

It cannot write to a broker, Google Sheets, R2, a live staging tab, a scheduler, or the private site. It is not in `STRATEGY_BOOK` or `daily_scan.py`. The policy constructor rejects `live_actions_enabled=True`; every preview has a blank approval field and `Live_Eligible=false`.

```text
IBKR read-only scanner sample
        ↓
broad mover nomination
        ↓
Google URL discovery → actual-page fetch → timestamp/source/excerpt hash
        ↓
Classic / EP9M catalyst / story watch / adverse / delayed-entry decision
        ↓
spread + impact + risk + ADDV + premarket volume + displayed-depth caps
        ↓
local DAY-limit staging preview (never executable)
```

## What the Bonde research supports

Bonde's durable idea is not “buy a large gap.” An EP is a surprising new fact that can change the company's trajectory, accompanied by a large and unusual price/volume reaction, preferably after neglect and early in a new earnings or product cycle. The scan creates a research list; catalyst judgment decides whether the event is an EP. His self-reported historical counts are hypothesis-generating observations, not audited backtests. See the [original EP study](https://stockbee.blogspot.com/2007/02/episodic-pivots-and-idea-pickle.html), [March 2007 follow-up](https://stockbee.blogspot.com/2007/03/episodic-pivots.html), and [canonical 2010 explanation](https://stockbee.blogspot.com/2010/02/what-are-episodic-pivots-and-how-to.html).

The implementation preserves distinct setup lanes instead of blending different vintages into one score:

| Lane | Meaning in this process | Immediate preview? |
|---|---|---:|
| `CLASSIC_EP` | Timely, fetched, material business catalyst from a primary/reputable source | Yes, if every execution gate passes |
| `EP9M_CATALYST` | Extraordinary-volume discovery with the same confirmed catalyst standard | Yes, at lower risk |
| `EP9M_STORY_WATCH` | Extraordinary volume without authoritative catalyst evidence | No |
| `STORY_EP_WATCH` | Theme, analyst action, promotion, or weakly classified narrative | No |
| `DELAYED_EP_WATCH` | Good catalyst but an initial gap above 25%, or another reason to seek a later structure | No |
| `ADVERSE_EVENT` | Offering/dilution, fixed-price deal, bankruptcy, reverse split, or investigation flag | No |

This follows Bonde's later distinction among classic EPs, stories, liquid/institutional events, EP9M, delayed reactions, and bearish variants in his [2024 Trade Factory presentation](https://impact.traders4acause.org/wp-content/uploads/2024/11/02-Pradeep-Trade-Factory-2.pdf). Bearish EPs are recorded in the research taxonomy but deliberately excluded from v0 execution because borrow, SSR, halt, and gap-through mechanics need their own model.

### Rules that changed over time

| Question | Earlier Bonde material | Later Bonde material | Frozen v0 choice |
|---|---|---|---|
| Entry time | Immediate after-hours/premarket buys in the [2014 process](https://stockbee.blogspot.com/2014/07/my-process-flow-for-episodic-pivots-ep.html) | Regular-session opening entry in later interviews; delayed entries increasingly emphasized | Research premarket; preview regular-hours DAY limit only |
| Very large gaps | Even extreme earnings gaps could continue | [2024 gap study](https://stockbee.blogspot.com/2024/09/gaps-in-50-plus-moves.html) found >20% increasingly pullback-prone and sustained winners rarely began >25% | Warn at 20%; route >25% to delayed-entry watch |
| Stop | Gap low or prior two-day low | EP-day low, then fixed 2.5–4% variants | Prior two-day low plus 10 bps is the only fully known premarket template; alternatives must be tested separately |
| No-progress exit | Five sessions | Three sessions in later comments | Neither is automated in v0; both are named test variants |
| Analyst action | Usually a low-quality catalyst | Contextual exceptions when truly thesis-changing | Watch only; never auto-stage |
| Universe | Neglected, low float, often <25M shares | Also liquid institutional names and cap below roughly $10B | Cap/float/neglect rank the candidate; executable liquidity controls size |

The current discovery floor comes from Bonde's [July 2026 premarket workflow](https://stockbee.blogspot.com/2026/07/how-to-find-and-research-pre-market.html): at least a 2% or $0.90 move, at least 100,000 premarket shares, and price at least $1. The post's literal Boolean formula is ambiguous; the prose clearly makes price and volume mandatory for either move branch, which is how the code is grouped.

## Frozen v0 policy

### Discovery versus staging

Discovery is intentionally broad:

- long-side move of at least 2% **or** $0.90;
- premarket volume at least 100,000 shares;
- price at least $1; and
- newest snapshot per symbol retained under a deterministic candidate ID, then the top 25 sorted by premarket share and dollar volume are researched.

A local preview additionally requires:

- live, tradeable, non-halted market data no older than 90 seconds;
- a known IBKR halt status, unique resolved stock contract, native primary exchange, SMART routing, and LMT support;
- the latest premarket bar no older than 15 minutes and an IB quote previous-close within 0.5% of the raw daily-bar reference;
- bid, ask, displayed ask size, premarket VWAP, prior two-day low, and 63-day ADDV;
- gap at least 4% and no more than 25%;
- price at least $3;
- premarket dollar volume at least $1 million;
- spread no wider than 100 bps;
- ask no more than 2% above premarket VWAP;
- an initial stop no more than 20% below the ask;
- a fetched catalyst document published no later than the decision time; and
- catalyst materiality score at least 3/5.

The 4% stage floor and execution thresholds are conservative automation choices, not claims that Bonde published one universal entry formula.

### News evidence and catalyst score

Google is a discovery layer. A search result, title, or snippet cannot confirm an EP. The run must fetch an actual page and retain:

- original and canonical URL;
- publisher and source tier;
- publication and retrieval timestamps;
- normalized body excerpt and SHA-256 excerpt hash;
- catalyst class, materiality signals, and adverse flags.

One primary/reputable fetched source, or two independent fetched secondary sources describing the same catalyst class within 12 hours, can establish source confirmation. SEC, FDA/government, and known issuer-wire domains receive primary priority. Reuters/AP/Bloomberg and similar outlets are reputable secondary confirmation. An unknown company-IR-looking domain stays secondary until an explicit issuer-domain map can prove ownership. Retrieval is stamped only after fetch, parse, and hash complete; completion after the decision clock is rejected. Publisher metadata gets at most two seconds of clock-skew tolerance and can never cross the 09:35 entry cutoff.

The deterministic materiality rubric scores regulatory approval, clinical data, raised guidance, earnings surprise, persistent demand/backlog, growth acceleration, quantified impact, and commercial milestones. It is deliberately conservative: ordinary management changes and analyst actions remain manual watches. Dilution, offerings, capped takeovers, bankruptcy, reverse splits, investigations, failed clinical/regulatory outcomes, guidance cuts, restatements, and recalls block staging.

Issuer references and event phrases must occur in the same sentence, or in an immediately following sentence explicitly beginning with “the company,” “it,” “management,” or “the board.” This closes ordinary multi-company-roundup leakage, but it is not a full issuer–event semantic extractor. A same-sentence roundup can still be ambiguous, so every local preview remains manual-review-only. A production proposal needs structured issuer evidence (for example SEC/IR identifiers) or a tested entity–event extractor before news classification can be considered sufficient.

Fetched pages are restricted to public HTTPS addresses on port 443. Every DNS answer must be globally routable, and the connection is pinned to a vetted IP while TLS still verifies the original hostname, closing the DNS-rebinding gap. Redirects and canonical URLs are revalidated, private/loopback/link-local destinations are rejected, cross-domain canonical metadata cannot elevate source authority, decoded response bodies are capped at 2 MB, and the body reader enforces an absolute deadline even against slow-drip responses. DNS and HTTP-header handling still rely on bounded connection/read-idle timeouts rather than a cancellable whole-request wall clock, so unattended scheduling remains forbidden until that final availability control is added.

Bonde's catalyst examples and rejection logic are documented in his [2007 catalyst list](https://stockbee.blogspot.com/2007/07/episodic-pivot-catalysts.html), [earnings-surprise notes](https://stockbee.blogspot.com/2007/04/trading-earnings-surprises.html), [2013 research workflow](https://stockbee.blogspot.com/2013/09/night-time-is-right-time.html), and [2014 catalyst hunt](https://stockbee.blogspot.com/2014/09/everyday-hunt-for-stocks-with-big.html).

Google Programmable Search is the preferred provider because it returns direct article URLs. It requires `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_ID`. Credential-free Google News RSS is available as a fallback, but wrapper links that do not resolve to an actual article correctly remain unconfirmed.

### Entry and sizing preview

The v0 preview is long-only, regular-hours-only, `LMT`, `DAY`, with no market fallback. Entry limit is the ask plus modeled entry slippage, rounded up to the next cent; if that limit would cross the 25% maximum opening gap, no preview is created. The initial stop is the prior two-session low minus 10 bps, rounded down. A future executor would have to re-resolve the same IBKR contract, recheck halt state and a fresh quote, refuse release when either the opening price or current ask is below the 4% activation floor or above the 25% ceiling, and cancel the unfilled entry at 09:35 ET. Those explicit fields prevent a stale DAY limit from buying a thesis-breaking gap-down, newly extended open, or much later fade.

Expected entry slippage is:

```text
half quoted spread
+ 8 bps base friction
+ 80 bps × sqrt(order notional / 63-day ADDV)
```

It must remain below 100 bps. Modeled risk per share includes the entry-to-stop distance, a 100 bps stressed exit cost, and an additional 2% event-gap stress. This explicitly avoids pretending that a stop guarantees the planned loss.

Final shares are the minimum of:

```text
risk budget / modeled risk per share
2% of account value / entry limit
0.25% of 63-day ADDV / entry limit
1% of observed premarket shares
25% of displayed ask size
25,000 absolute shares
```

Classic EP risk is 10 bps of the configured $750,000 research account; `EP9M_CATALYST` is 7.5 bps. Those are intentionally below Bonde's published personal-risk examples and should remain shadow assumptions until prospective fills support them.

After individual sizing, all previews are scaled pro rata to a 50 bps aggregate daily risk cap and a 10% aggregate-notional cap. This prevents a cluster day from turning individually small candidates into an uncontrolled basket.

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

### 1. Capture IBKR data

Use paper TWS/Gateway and a local environment containing the optional `ib_insync` package:

```powershell
python scripts/capture_ep_premarket_ibkr.py --port 7497
```

The adapter connects with `readonly=True`, round-robins the union of `TOP_PERC_GAIN`, `HOT_BY_VOLUME`, and `MOST_ACTIVE`, requests details only for the narrowed list, and saves a timestamped JSON file under `artifacts/episodic_pivot/`. It uses a bounded streaming watchlist request so missing IBKR halt telemetry remains `UNKNOWN`, resets ib_insync's default market-data type to unknown, waits for an explicit IBKR data-type callback, and then cancels every subscription. It has no order-submission method. Live quotes and the required exchange subscriptions still depend on the user's IBKR entitlements.

This capture is explicitly a **non-exhaustive scanner sample**. IBKR caps each API scan at 50 rows, so the union cannot prove it found every stock satisfying the 2% **or** $0.90 rule. Each payload records per-scan counts/ranks, union and truncation counts, errors, selection method, and `exchange_complete=false`. Before production review, the capture must add a defined-universe sweep in entitlement-sized quote batches, use scanners only as supplements, and report requested/resolved/live coverage. Raising the scanner cap alone would not solve the coverage problem.

### 2. Research and preview

Credential-free Google News discovery:

```powershell
python scripts/run_episodic_pivot_shadow.py `
  --snapshot artifacts/episodic_pivot/ibkr_snapshot_YYYYMMDDTHHMMSSZ.json `
  --news-mode google-news `
  --allow-network
```

Preferred Google Programmable Search:

```powershell
$env:GOOGLE_CSE_API_KEY = '<local secret>'
$env:GOOGLE_CSE_ID = '<local search engine id>'
python scripts/run_episodic_pivot_shadow.py `
  --snapshot artifacts/episodic_pivot/ibkr_snapshot_YYYYMMDDTHHMMSSZ.json `
  --news-mode google-cse `
  --allow-network
```

Unverified fixture replay (classification/audit only; it cannot create a preview):

```powershell
python scripts/run_episodic_pivot_shadow.py `
  --snapshot tests/fixtures/ep_snapshot.json `
  --evidence tests/fixtures/ep_evidence.json `
  --as-of 2026-08-24T12:31:00Z `
  --execute-on 2026-08-24
```

Verified replay after a network run refreshed the news evidence:

```powershell
python scripts/run_episodic_pivot_shadow.py `
  --snapshot artifacts/episodic_pivot/ibkr_snapshot_FRESH.json `
  --evidence artifacts/episodic_pivot/EP-RUN-SOURCE/evidence_by_symbol.json `
  --evidence-manifest artifacts/episodic_pivot/EP-RUN-SOURCE/manifest.json
```

The source manifest must come from a network research run, its run directory must match its `run_id`, and its recorded SHA-256 must match `evidence_by_symbol.json`. Arbitrary offline JSON is stamped `UNVERIFIED_REPLAY` and cannot become stageable. This provides a tamper-evident local provenance chain; it is not a signature against a malicious local operator.

Each run writes `manifest.json`, `candidates.json`, candidate-ID and symbol-keyed evidence files, `decisions.json`, `staging_preview.json`, `staging_preview.csv`, and `report.md`. The manifest hashes every artifact. Rerunning identical offline inputs uses the same run ID. A network run timestamps each fetch honestly and rechecks quote freshness after research; if research makes the quote stale, recapture IBKR data and replay the saved `evidence_by_symbol.json` with that network run's manifest rather than weakening the freshness gate. CSV cells are formula-escaped for safe review while JSON retains the raw source text for audit.

### 3. Historical census

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
