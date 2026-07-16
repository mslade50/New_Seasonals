# Risk Trade Console — Final Spec (2026-07-16)

Replaces the "Read" nuggets zone on the private site's risk page with one decisive, configuration-conditional block. Synthesized from the rtc_* probes (config-history, config-outcomes, structure-ev), the page inventory, the prior-evidence ledger, and three critiqued designs. All numbers cited here trace to `scratch/rtc_config_history.parquet`, `scratch/rtc_config_stats.json`, and `scratch/rtc_structure_ev.json` (all built 2026-07-16, history 2016-07-18 to 2026-07-15, 2,512 trading days).

## 1. Executive summary

The single most important empirical finding: **no signal configuration class supports an option-structure recommendation at honest model prices.** Every pre-registered class, priced adversarially (BS-European, entry IV = VIX3M, +0.40 vol/OTM% static skew, 5% haircut per side, held to expiry), is negative-EV for both the 3-month 30/10-delta put spread and 10-delta tail puts. The scariest-sounding class (2+ bearish signals active) saw its tail puts expire worthless in 17 of 17 episodes this decade. The least-negative cell is today's own class (Defensive Leadership near a 52-week high): spread EV -8% of debit with a CI of [-87%, +84%] on 13 episodes — "elevated downside" language, never "+EV" language.

So the honest console is a **conditional tail-probability display with earned no-trade lines**, not a trade recommender. What it CAN say with a straight face: the forward SPY distribution conditional on today's configuration class, with episode N, versus baseline — e.g. today ([DL fired within 5 sessions] near 52w high): 18 prior episodes, median 3-month return near baseline (+3.8%) but 10th percentile -11.2% vs -5.4%, P(>=10% drawdown) 24% vs 13%. The structure line renders as the reason NOT to buy premium. The template contains a verdict-flip gate — if a class's spread EV CI lower bound ever exceeds zero after a stats regeneration, the sentence McKinley asked for ("historically +EV for a 3m 30/10 put spread") emits — but under current frozen evidence that gate is unreachable, and he must sign off on that explicitly. This is the same verdict that killed the 2026-07-16 hedge block, now configuration-conditional, distribution-grounded, and structurally incapable of manufacturing a hedge.

Secondary finding: exact configurations are unusable as conditioning units (129 distinct configs, median occupancy ~5 days; Dispersion — the headline of McKinley's own example sentence — has 8 deduped episodes and can never anchor a card). The viable vocabulary is 6 coarse classes plus a quiet state.

## 2. Configuration -> outcome evidence table

Episode-deduped (21td cooldown; see the episode-rule fork in section 5), forward SPY from episode entry, baseline = all 2,512 overlapping day-windows. Returns in %.

| Class | N ep | fwd63 mean | fwd63 median | fwd63 p10 | P(DD>=5%) | P(DD>=10%) | Structure EV (spread, % of debit) |
|---|---|---|---|---|---|---|---|
| Baseline (all days) | 2512d | +3.9 | +4.7 | -5.4 | .30 | .13 | -60 [-89,-24] |
| Quiet (0 signals) | 59 | +3.6 | +4.2 | -6.0 | .28 | .11 | -65 [-91,-30] |
| Exactly 1 bearish | 51 | +2.5 | +4.4 | -4.7 | .38 | .15 | -77 [-100,-35] |
| **2+ bearish** | 26 | **+0.7** | +3.4 | -8.3 | **.44** | **.20** | -67 [-100,-3]; tail puts 0-for-17 |
| **DL any (today)** | 18 | +1.6 | +3.8 | **-11.2** | .29 | **.24** | -8 [-87,+84], hits 3/13 |
| AR any | 22 | +0.7 | +4.1 | -11.4 | **.50** | .18 | (pooled fallback only) |
| SRD any | 16 | +1.7 | +4.7 | -7.5 | .38 | .19 | (pooled fallback only) |
| DA any | 12 | +0.5 | +2.4 | -6.9 | .42 | .17 | P(VIX>=28) .50 vs .39; N at floor |
| VRC any | 19 | +2.7 | +3.5 | -3.4 | .44 | .11 | 21d drag only (-0.8% fwd21); 63d tails BELOW baseline |
| Pre-FOMC (excluded) | 43 | +3.8 | +4.6 | -3.7 | .29 | .12 | fwd21 +2.2% — positive edge, never counts bearish |

Killed for N<12 episodes: Dispersion (8), DA-near-high (11), DL+other (11), VRC+AR (11).

Caveats that bind every row: one decade of history containing **three** real drawdowns (2018Q4, 2020, 2022); signal fire histories recomputed from today's signal code and thresholds, chosen with hindsight — PIT storage cannot cure this; the previously cited BEAR_2plus t of -2.3/-2.5 was computed against an overlapping day-level baseline and does not exist in either JSON artifact — it must be recomputed at episode level against an episode-deduped baseline before it may render, and with ~14 classes screened post-hoc, one unadjusted |t|>2 is roughly the null expectation. The honest statistical headline is "nothing here is statistically distinct after multiplicity correction; these are conditional distributions, not proven edges" — and that sentence renders on the card.

## 3. Trade Console design

**Placement.** Replace the nuggets grid in-place (risk.js line ~42, between the price-context strip and the Signals h2). The ask's literal zone ("between Signals and the SPY chart") is empty in code; the nuggets are the text-heavy target, and verdict-above-evidence is the right UX (console = conclusion, Signals accordions = receipts). Flag the deviation to McKinley. The `nuggets` payload key STAYS (ideas.js:44-46 consumes it); risk.js renders nuggets only as fallback when `trade_console` is absent, so old payloads degrade to the old layout exactly.

**Class engine.** Frozen vocabulary, first match wins, declared as a constant beside `SIGNAL_METRICS` in `scripts/build_risk_json.py`, version-stamped ("class set v1, frozen 2026-07-16, screened post-hoc, one-shot"):

1. `BEAR_2PLUS` — >=2 of the 6 bearish signals on-or-recent-5td (BEAR_2plus and _HI collapsed: 94% day overlap, identical episodes).
2. `DL_TAIL` — DL on-or-recent (every 2016-26 episode was near a high; today's state).
3. `AR_DRAWDOWN` — AR on-or-recent.
4. `SRD_TAIL` — SRD on-or-recent; tail-percentile read only, no "reduce exposure" action (its median equals baseline).
5. `DA_SOFT` — DA on-or-recent; N=12 sits on the floor, renders with LOW SAMPLE badge and no verdict adverbs.
6. `VRC_CONTEXT` — context line only ("short-horizon drag; 3-month tails below baseline"), no action. Not a bearish class — its 63d distribution is benign.
7. `NONE` — everything else (~62% of days, not 46% as one draft claimed).

Pre-FOMC never counts bearish. Classes below 12 episodes never render; a lone Dispersion fire yields the NONE card plus one line: "Dispersion fired — only 8 prior episodes, too few to condition on."

**Card anatomy** (monotone, sizing_state-hero styling, four fixed slots):

1. *What fired*: signal names, recency ("fired 3 sessions ago, now off" — stats are measured from fire date, said when recency > 0), SPY context in plain words. No dial scores, no z anywhere.
2. *What that class preceded*: episode fwd63 **mean AND median** vs baseline, p10, P(DD>=5/10%), always with N, year span, and the episode-level t plus "does not survive a 14-class correction" when it doesn't. DL card additionally shows drop-best-episode (the -8% spread mean is hauled up by Jan-2020) and hit counts (3/13), never "both winners."
3. *Structure check*, gray MODEL-PRICED badge: cost as % of notional, EV as % of debit/premium with bootstrap CI, degenerate cells as counts ("expired worthless 17 of 17"), mandatory "held to expiry" qualifier plus the oracle-exit bound (perfect exits lift the spread to only +4%/+25%/+37% — the negative verdict is not a payoff-convention artifact, but say both numbers). Verdict clause: "neither structure has cleared cost at model prices in this class this decade." Flip gate: CI lower bound > 0 after a regeneration emits "historically +EV for [tenor/delta structure]" — currently unreachable.
4. *Action line*, always prefixed **"Historical read:"** — vocabulary limited to: no trade / smaller size / elevated left tail, premium still doesn't clear model cost. Derived from the distribution, never the structure EV.

Fixed footer, never collapsible: "N=… episodes; one decade, three real drawdowns; signal histories recomputed from today's code (lookahead); option EV is BS + static skew + 5%/side haircut — no real option chains; display-only — nothing here feeds sizing or staging. Stats vintage {built_utc}, class set v1."

**NONE-state copy** (earned silence, builder-formatted numbers only): "No bearish signals active or recent. In 59 quiet-state episodes SPY's next 3 months matched baseline (+3.6% vs +3.9%). Premium bought here bled -65% of debit at model prices [builder formats the true figure from the JSON — the -59.6%/-64.5% discrepancy between drafts is exactly why no number is hand-typed]. Historical read: nothing to do."

**Honesty layer** (binding rules): every rendered figure formatted at build time from the stats file — zero hand-typed numbers; CI on every mean and every EV; CI straddling zero forces sign-neutral phrasing; minimum tiers (N>=20 across >=6 years full display; 12-19 LOW SAMPLE badge, descriptive only; <12 silent); concentration guard (>40% of episodes in one year demotes a tier); one primary endpoint per class (episode fwd63 mean), everything else descriptive and never headline-promoted.

**Silence/degraded states — rendered, not invisible.** `trade_console.state` in {ok, degraded, silent}: stale stats (>180d) or class-fingerprint mismatch -> degraded (fired-line renders, structure block suppressed, reason shown); stale inputs (rd2/master_prices asof >3td — the rd2_spy_ohlc silent-staleness failure was real) -> `{state:'silent', reason:'…'}` rendered as one line; builder exception only -> key omitted (inner try/except + traceback, per-cause FAILED prints so the GHA log is diagnosable). risk.js additionally badges/suppresses when `asof` is >2 sessions older than today (skipped deploys fossilize "fired 3 days ago"). Never three compounding layers of identical invisibility.

**Payload.** New key `trade_console`, strings fully rendered at build time; risk.js does zero composition. Guard: `if (tc && tc.state)`, full card on `state==='ok'`. Schema: `{asof, state, reason?, class_id, class_set_version, headline, fired_line, dist_line, structure_line?, action_line, caveats[], n_ep, stats_built, stats_sha}`.

## 4. What the console must NEVER claim

- "+EV" for any structure unless the CI-lower-bound gate passes on committed, regenerated stats. Under current evidence: never.
- Absolute dollar EV or Sharpe of any structure; a ranking of spread vs wing (depends on skew shape we cannot observe); any option number without the MODEL-PRICED label — there are no historical chains, VIX3M is not SPY BS IV, the skew is static-linear (dominant error, ±30-50% on the 10-delta leg), and the 5% haircut is generous for far OTM (tail-put EV is biased UP as shown).
- Anything shaped like an order: no tickers-as-tickets, strikes, expiries, or quantities — tenor/delta words only; every action line carries the "Historical read:" prefix; the footer carries the display-only/nothing-staged disclaimer. The site is becoming trade-capable; this block must be uncopyable into order_staging.
- Statistical significance it doesn't have: no bare hit rates without N, no "historically/reliably" on LOW SAMPLE classes, day-level t-stats, or the unverified t=-2.3 until recomputed.
- Anything from a retired taxonomy: the fingerprint guard (hash of signal names + thresholds + precedence stamped into the stats file, compared against live `compute_all_signals` every build) makes the predecessor's retired-regime failure a hard, loud stop instead of a silent lie.
- Dispersion-anchored claims (8 episodes) — McKinley's own example sentence is the first thing this rule silences, and the spec says so.

## 5. Implementation plan

**Step 1 — unify the episode rule and regenerate (blocking).** The drag stats use 21td cooldown (BEAR_2plus N=26, fwd63 +0.7) while structure EV uses >=63td spacing (N=17, fwd63 +2.4, CI positive) — one card would cite both. Pin >=horizon (63td) spacing for everything, per the critique that 21td cooldown under a 63td endpoint overlaps windows ~3:1 and inflates any t. Write `scripts/build_trade_console_stats.py` (productionizes `scratch/rtc_config_stats.py` + `rtc_structure_ev.py`; `_NoOp` streamlit stub per `build_signal_horizon_stats.py`; includes the rd2_spy_ohlc splice check) as the SINGLE writer — this also resolves the parallel-agent write-collision on `rtc_structure_ev.json`. Add: P(fwd63<=-5%) (currently in neither JSON — the sample card's "23% vs 15%" was untraceable), medians, drop-best-episode, hit counts, episode-block t vs episode-deduped baseline, oracle bounds, and the class fingerprint + built_utc + git sha.

**Step 2 — commit the stats file at a tracked path.** `data/` is gitignored in general; verify with `git check-ignore` and either add an exception (the `sector_map.parquet` precedent) or place it under `scripts/`. In the GHA deploy checkout the file must exist or the console is permanently silent in prod while working locally.

**Step 3 — builder.** `build_trade_console()` in `scripts/build_risk_json.py`: inner try/except + traceback (vol_kpi pattern), classifies today from the already-computed signal fires using the IDENTICAL on-or-recent-5td rule as the stats script, reads the stats file directly (no transcribed constants), checks fingerprint/vintage/input-staleness, emits rendered strings. Whole-payload exit-0 contract preserved.

**Step 4 — tests.** `tests/test_trade_console.py`: class precedence and N-floor fallthrough; fingerprint mismatch -> degraded; stale stats -> degraded; regex asserts on the RENDERED payload strings (not source grep): no `/buy|sell|open|short/i` outside the "Historical read:" prefix, no "historically +EV" unless the CI gate boolean is set, footer disclaimer present; schema check; test FAILS when `built_utc` exceeds 400 days — the only kill criterion with an executor (scheduled re-runs and "+10 new episodes" triggers have no owner and were rejected as unenforced promises).

**Step 5 — frontend.** risk.js: `tradeConsoleHtml(tc)` at the nuggets slot; nuggets as explicit fallback only when the key is absent; `style.css` gets `.trade-console` cloned from the sizing-hero block. Local verify: `python scripts/build_site.py --no-signals` + http.server, eyeball today's DL_TAIL card against the JSONs.

**Step 6 — sign-off items for McKinley** (before shipping): (a) placement deviation (pre-Signals, not the literal empty zone); (b) nuggets disappear from risk.html only; (c) whether NONE-state renders a card or one line; (d) the headline concession: under frozen evidence this console will never emit a +EV trade — its deliverable is calibrated tail odds plus earned silence. If that's not acceptable, the honest alternative is no block at all, not a softer pricing model.

## 6. Rejected ideas

- **Dial-score z-tables / mean-Z verdict prose** — explicitly not wanted; it's what the nuggets did wrong.
- **Exact-configuration conditioning** — 129 configs, median 5 days occupancy; anecdote generator.
- **Dispersion or any N<12 class anchoring a card** — including the ask's own example.
- **SINGLE_SOFT as a bearish action class** — the data contradicts it (one-signal days: P(fwd63<=-5%) 3.6% vs 15.4% baseline).
- **Nightly EV re-estimation** — freeze policy A2; stats regenerate only on deliberate review via the one script.
- **Time-varying skew slope via ^SKEW** — uncalibrated; keep 0.40 fixed with a 0.30/0.50 sensitivity band.
- **Source-grep enforcement tests** — theater; assert on rendered strings.
- **Scheduled/quarterly kill criteria without an executor** — replaced by vintage tripwires the builder and test enforce every run.
- **Silent omission for all failure modes** — three layers of identical invisibility; degraded states render their reason.
- **Reusing `tests/backtest_put_hedge.py` pricing as-is** — it's flat-vol; the adversarial conventions live in `ca_overlays.py`.
