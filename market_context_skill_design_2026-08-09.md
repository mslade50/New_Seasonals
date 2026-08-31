# Market Context Skill — Design Spec (2026-08-09)

Handoff spec for a build agent. Status: approved DIRECTION from McKinley; implementation
details are recommendations except where marked **hard requirement**. Companion precedent
docs: `daily_pitch_agent_spec_2026-08-06.html` (this repo) and
`denali-dashboard/.claude/skills/scott-report-card/SKILL.md` (the delivery template).

---

## 1. Intent (verbatim from McKinley)

> Build a daily skill that traverses between the denali-dashboard repo and the
> New_Seasonals repo. Provide daily context through the Slack report-card channel which is
> unrelated to the existing portfolio — it does a lot of what the daily-pitch skill does,
> but rather than provide the exact trade it just provides the context. E.g. "the last 6
> August FOMC dates in midterm years have been negative, -0.5% on average, with the
> following session reversing that decline." It should be digging into the most minutiae
> possible to pull nuggets out that would otherwise be missed, and then report all relevant
> data points for the next session OR that provide context to the previous session (e.g.
> "SPY dropped 80 bps today following a 52w high the day before; historically that has led
> to 1% of outperformance the following week (t=3.01, n=90), but in midterm years that
> reverses to __"). Only macro assets — indices, currencies, futures, etc. Pulling out
> relevant pieces of information that could be helpful context for Scott the next day.

## 2. Product definition

One Slack post per evening (Sun–Thu) in the same channel as the Scott report card: the
**Market Context brief**. Two lanes:

1. **Tomorrow's tape** — every scheduled feature of the next session (FOMC, CPI, NFP,
   opex, quad witching, VIX expiry, Jackson Hole, month-end/turn-of-month, holiday
   adjacency, day-of-week/cycle-year cell), each with the most *specific defensible*
   conditional history: not "FOMC days average +X" but "August FOMC in midterm years:
   6 of last 6 negative, mean −0.5%, next session reversed the decline in 5 of 6."
2. **Today in context** — what just printed, conditioned on state: new 52w high/low,
   down-day-after-52w-high, z-score extremes, streaks, >2-ATR moves, cross-asset
   divergences, surprise vs consensus on today's releases — and what history says follows.

Every claim carries its numbers (n, mean, hit rate, t where meaningful) and its
conditioning cell. Minutiae is the point: the deterministic sweep enumerates broadly, the
agent drills the interesting cells further (interaction cells, era splits, the reversal
day, the path shape).

**Non-goals (hard requirement):**
- No trades, no legs, no entries/exits/sizing, no advisory verbs. This is context, not a
  pitch. The report-card banned-verb list applies (trim / cut / reduce / consider /
  recommend / watch / hedge / exit / fade / buy / sell / long / short as imperatives).
- Nothing about the Denali book. The existing report card owns positions; this product
  deliberately never reads `data.json`, D1, or HF state.
- No web research in v1. All claims are computed from local history. (This also makes the
  headless permission story clean — see §10.)
- Not interactive, no approval loop, no site tab.

## 3. Placement and cross-repo wiring

**All new code lives in New_Seasonals** — the data (`master_prices.parquet`,
`macro_events.csv`, `macro_release_history.parquet`), the study library (`pitch_lab`,
`scripts/seasonal_edge.py`), and the calendar machinery are all here, and the scheduled
task must run with this repo as cwd.

**denali-dashboard contributes only the delivery pattern and the channel**: the new Slack
sender is modeled on `denali-dashboard/scripts/send_report_card_slack.py` (Block Kit
inside one colored attachment, `to_mrkdwn`, retry policy, `--dry-run`, today-dated
freshness guard) but is **rewritten self-contained in New_Seasonals** — do not import
across repos. Copy the same `SLACK_WEBHOOK_URL` value into New_Seasonals `.env` (posting
to the same Incoming Webhook lands in the same channel).

Module boundary (hard requirement, house rule): nothing in the systematic book imports the
new modules, and the new modules never write to `data/pitch_*.json` or
`data/pitch_journal.jsonl`. New state lives under its own names.

## 4. Universe

Macro assets only. Define `CONTEXT_UNIVERSE` explicitly in the engine (single aligned
site), drawn from what `master_prices.parquet` already carries (1999→present, adjusted):

- **US indices:** SPY QQQ IWM DIA `^GSPC ^NDX ^IXIC ^DJI ^RUT ^NYA`
- **Global indices:** `^FTSE ^GDAXI ^FCHI ^N225 ^HSI ^AXJO ^KS11 ^MXX ^BVSP ^GSPTSE` EFA EEM FXI EWZ EWJ
- **Rates/credit:** TLT IEF LQD HYG `^TNX ^FVX ^IRX ^MOVE`
- **FX:** DX-Y.NYB UUP + the 27 `=X` pairs (EURUSD, JPY, GBPUSD, AUDUSD, USDMXN, USDZAR, …)
- **Futures (20):** ES NQ YM CL NG GC SI HG PL PA ZC ZS ZW SB KC CC CT LE HE LBS (`=F`)
- **Vol:** `^VIX ^VIX3M ^VVIX ^SKEW`
- **Crypto (context only):** BTC-USD ETH-USD

Sector ETFs enter only as breadth context (e.g. % above 200d), never as headline subjects.
Single names are out of scope entirely.

## 5. Core loop

Mirrors daily-pitch's survey-then-select shape and the report card's
deterministic-aggregator-then-agent shape.

```
run_market_context.bat (Task Scheduler, Sun–Thu ~18:30 ET)
  [0] freshness: data_provider auto-pull; python scripts/build_context_state.py  (fatal on failure)
  [1] claude -p "/market-context" --settings scripts\context_headless_settings.json
        Stage A  read data/context_state.json (sweep results, calendar, tape, novelty)
        Stage B  write scratch/context_checks/<date>/00_cell_map.md — verdict per candidate
        Stage C  drill-downs: ad-hoc .py in the day folder using pitch_lab/seasonal_edge
        Stage D  compose data/context_briefs/YYYY-MM-DD.md + .json
  [2] python scripts/send_context_slack.py       (inside the same claude run, exact command)
  [3] python scripts/check_context_delivered.py  (in the .bat; non-zero exit = loud miss)
```

### Stage [0] — deterministic engine: `scripts/build_context_state.py`

No LLM. Produces `data/context_state.json` containing:

- **`meta`** — asof session, prices-fresh boolean (see §10 freshness gate), cycle year
  (`year % 4` + label), month, day-of-month, weekday, is_month_end_window,
  sessions-to-month-end.
- **`calendar`** — reuse `build_pitch_state.build_calendar(today)` wholesale (events in
  [−5, +15] td with `td_ahead`, `next_by_type` for every `macro_calendar.EVENT_TYPES`).
- **`tape`** — per-ticker context vector over `CONTEXT_UNIVERSE`: copy
  `build_pitch_state._metrics_for` verbatim (ret_1d/5d/21d/63d/252d, rank_5/21/63d, z10,
  Wilder-14 ATR, atr_pct, rvol21_ann, dist_52w_high/low_pct, dist_sma200_pct, vol_vs_63d)
  plus extremes and breadth blocks.
- **`triggers`** — the fired cells from the trigger library (§6), each with **base stats
  already computed** via `pitch_lab`: `{trigger_id, subject, description, n, mean_pct,
  median_pct, hit, t, sign_p, worst_pct, best_pct, era_split, cycle_split,
  concentration_note, episode_dates_tail}`. Base stats use `fwd_ret` at h∈{1,5} on the
  subject (this is context, not a tradeable entry, so lag=0 forward returns are correct —
  state that in the brief's convention footnote once, not per item).
- **`releases_today`** — today's rows from
  `macro_releases.load_macro_releases(require_surprise=False)` with `surprise_label`
  (above/below/inline — never "beat", house vocabulary).
- **`novelty`** — per-trigger `{is_new, last_published, last_headline_number,
  materially_moved}` diffed against `data/context_flag_state.json` (mirror the
  report-card aggregator's novelty layer including a `delta_suppressed` escape hatch when
  the state file is missing/corrupt).

CLI: `--asof`, `--out`, `--no-state` (dry run, doesn't advance novelty baseline),
`--offline`.

### Stage B — cell map (hard requirement)

Before selecting anything, the skill writes
`scratch/context_checks/<YYYY-MM-DD>/00_cell_map.md`: every fired trigger and every
next-session calendar cell gets a written verdict — PUBLISH / DRILL / SKIP(reason) /
DEAD(n too small even for anecdote). Same anti-recall rationale as daily-pitch's surface
map: the map forces the model to actually read the sweep instead of pattern-matching to
famous cells. Missing map ⇒ nothing may publish (checked by `send_context_slack.py`).

### Stage C — drill-downs

For each DRILL verdict, write a real `.py` in the day folder and run it. This is where
"minutiae" lives — the engine computes base cells; the agent computes the interactions
and follow-ons that make a nugget worth reading:

- condition the base cell further (August FOMC → midterm-year August FOMC; day after 52w
  high → down ≥50 bps after 52w high → same, midterm years only),
- the next-session and next-week path (`pitch_lab.episode_paths`, `horizon_scan`),
- the reversal claim ("following session reversed the decline in k of n"),
- era stability (`era_split` at 2018), concentration (`cluster_note`),
- controls: same-span drift, all-days, local ±126 td (`local_control`) — the three-control
  convention from daily-pitch applies whenever a nugget's edge claim depends on it.

Budget: 3–8 drill scripts is normal. Do not silently truncate drill-downs to save tokens
(inherited hard rule).

### Stage D — compose and publish

Select 4–8 nuggets across both lanes (see §7 selection rules), write
`data/context_briefs/YYYY-MM-DD.md` + `.json`, run the Slack sender, advance
`data/context_flag_state.json`, append one record per published nugget to
`data/context_journal.jsonl` (own file, own kinds — never the pitch journal).

## 6. Trigger library (v1)

Implemented as a registry in the engine — one function per trigger returning fired/not +
the boolean mask over history, so `pitch_lab` computes stats uniformly. Extensible by
adding a function; keep the registry list and the SKILL.md's trigger inventory as an
**aligned-sites pair**.

**Event-anchored (next session or ≤3 td ahead):**
- E1 fomc_decision (+ splits: month-specific, cycle-year, hiking/cutting era)
- E2 cpi / E3 nfp / E4 ppi (+ splits: cycle-year, month, recent-surprise-direction via
  `macro_release_history` — e.g. "after two consecutive above-consensus CPIs")
- E5 fomc_minutes, E6 opex, E7 quad_witching, E8 vix_expiry, E9 jackson_hole,
  E10 election adjacency
- E11 month-end / turn-of-month window (final 3 td / first 2 td; split by above/below 20d
  MA — the month-end-crush finding from the denali memory is exactly this shape)
- E12 holiday adjacency (session before/after a market holiday)
- E13 bare calendar cells: day-of-week × month, first/last session of month, cycle-year
  day-of-year seasonal rank (`seasonal_edge.seasonal_window_returns` with
  `cycle_phase_filter`)

**Price-state (today's session, per subject ticker):**
- P1 new 52w high / P2 new 52w low (+ "first in ≥30/≥90 cd")
- P3 down ≥ X bps the session after a 52w high (X ∈ {50, 100}); mirror for lows
- P4 |z10| ≥ 2 ; P5 rank_5d or rank_21d ≤ 5 or ≥ 95
- P6 single-session move ≥ 2 ATR ; P7 streaks ≥ 5 consecutive up/down
- P8 close crossing the 200d MA (first cross in ≥63 td)
- P9 cross-asset one-liners: stocks/bonds same-direction days, dollar+gold both up,
  VIX up on an SPX up-day, 2s10s proxy moves via ^TNX−^FVX
- P10 vol-structure: ^VIX3M−^VIX inversion onset/exit ; ^VIX +N% day
- P11 breadth extremes: pct_above_sma200 crossing decile edges
- P12 release-surprise follow-through: today's CPI/NFP printed above/below consensus →
  historical next-1/5d conditional on same label (`load_macro_releases`)

**Interaction cells** are the agent's job (Stage C), not the engine's — the engine fires
base cells; the agent crosses them (P3 × midterm, E1 × August, P1 × E2-tomorrow, …).
Rationale: the interaction space is combinatorial and the model is better at choosing
which crossings are interesting than a grid is.

## 7. Statistical honesty contract (hard requirement)

Inherit the daily-pitch doctrine wholesale; restate in SKILL.md:

- **Every nugget carries n and its cell definition.** Mean/hit always; t only when n
  supports it (t on n=6 is noise — use the exact sign test, `pitch_lab.sign_test`, and
  report the record: "6 of 6 down, sign p=0.016").
- **Small-N doctrine:** a clean record + meaningful per-event magnitude + plausible
  mechanism is publishable as *context* — labeled. Three confidence tags, code-enforced
  coherence like `pitch_grammar._validate_evidence`:
  `[solid]` n≥50 and |t|≥2.5 and era-stable · `[suggestive]` n 15–50 or single-era ·
  `[anecdote]` n<15 — **max two anecdotes per brief**, and an anecdote may never be the
  headline.
- **Multiplicity is priced on the sweep:** the engine's trigger scan is a search — the
  brief's footnote states cells-scanned count each day, and any nugget whose only
  support is the swept p-value gets Benjamini–Hochberg (`seasonal_edge.benjamini_hochberg`,
  α=0.10) before it can be tagged `[solid]`. Pre-specified famous cells (FOMC drift,
  turn-of-month) are hypotheses, not searches — no correction, per the daily-pitch rule.
- **Era honesty:** any nugget whose sign flips across the 2018 era split must say so or be
  dropped. "Dead after 2018" is itself a publishable nugget when the cell is famous.
- **Concentration:** if the top 2 episodes carry the mean, say so (`cluster_note`).
- **Selection bias sentence** in the standing footer, modeled on
  `daily_seasonal_ideas.METHODOLOGY`: descriptive, post-selection, not out-of-sample.
- **Convention footnote** (standing, one line): forward returns are close-to-close from
  the signal close (h=1 next session, h=5 next week), full history 1999+ unless the cell
  says otherwise, adjusted prices, NYSE trading days.

**Selection rules for the 4–8 published nuggets:**
- ≥1 from each lane when both lanes fired; tomorrow's-tape lane leads if a top-tier event
  (FOMC/CPI/NFP) is next session.
- Rank by: relevance to the next session > specificity of the cell > strength
  (|t| or sign p) > novelty. A `[suggestive]` cell about tomorrow's exact setup beats a
  `[solid]` generic.
- **Quiet-tape contract:** no top-tier event and no fired price-state trigger → ship a
  3-line "QUIET TAPE" note (next scheduled events + one seasonal-position line) and stop.
  Never pad.

## 8. Novelty / repetition control

`data/context_flag_state.json` keyed by trigger fingerprint (trigger_id + subject +
cell-qualifiers). Rules:

- A nugget published in the last 5 td may not repeat unless (a) the event moved from
  "upcoming" to "next session" — one escalation allowed, must add new specificity — or
  (b) the headline number materially changed.
- Countdown re-tellings are banned ("3 days to FOMC… 2 days to FOMC…" with the same stat).
- `delta_suppressed: true` (missing/corrupt state) → publish but make no NEW/first-time
  claims.

## 9. Output contract

### `data/context_briefs/YYYY-MM-DD.md` (agent-written; parsed by the Slack sender)

```markdown
# Market Context — Tuesday 2026-08-11

**Headline:** <one sentence, one cell, one number>

## Tomorrow's tape
1. **<subject> — <cell>** [solid]
   <2–3 sentences: the stat with n/hit/t or sign p, the sharper split, the follow-on
   (next-session reversal, week path). No advice.>

## Today in context
1. **<subject> — <pattern>** [suggestive]
   ...

## Calendar
- <next 5 td of macro_events with times ET>

---
*Cells scanned: 214. Conventions: close-to-close fwd returns, 1999+, adjusted, NYSE td.
Descriptive post-selection statistics, not out-of-sample forecasts.*
```

Word budget **250–400 words** for the body. Prose rules inherited: no em dashes, no
"it's not X, it's Y", no AI throat-clearing, one fact appears once, all times ET.
The numbered-item head format is load-bearing for the sender parser — keep
`1. **Title** [tag]` exact, mirror the report-card sender's regex approach.

### `.json` sibling

`{asof, quiet, headline, nuggets:[{lane, trigger_id, subject, cell, tag, n, mean_pct,
hit, t, sign_p, era_note, concentration_note, drill_script, text}], cells_scanned,
calendar_next_5td, conventions}` — the journal record and the Slack sender's
authoritative source for `quiet`.

### Slack

`scripts/send_context_slack.py`, self-contained clone of the report-card sender's
mechanics: one colored attachment (pick a distinct standing color, e.g. `#6e40c9`
purple, so it never visually collides with the report card's severity colors), header
`Market Context — <weekday> <date>`, lane dividers, context-block footer, `to_mrkdwn`
conversion, webhook-first with bot-token fallback, `--dry-run` writing
`data/context_briefs/<stem>.slack.json`, today-dated freshness guard, retry only on
429/5xx. Gate on the `00_cell_map.md` existing for the date (see Stage B).

## 10. Runtime, scheduling, permissions

- **Task:** `Market Context Brief`, Sun–Thu **18:30 ET** (after the 18:00 report card so
  the two never contend for the interactive Claude session; report card typically
  finishes well inside 30 min — verify during shakeout). Interactive logon, run-as-user,
  `ExecutionTimeLimit=PT1H`, `StartWhenAvailable=true`. Ship a
  `scripts/register_market_context_task.ps1` modeled on `register_daily_pitch_task.ps1`,
  inert until an operator runs it. Use `powershell Get-Date` for the log-name datestamp,
  not wmic (removed on newer Win11).
- **Freshness gate (hard requirement):** `build_context_state.py` verifies the freshest
  `master_prices` bar equals the expected session (today Mon–Fri; Friday on Sunday runs).
  The PM price workflow crons at 20:30 UTC — in EDT that lands 16:30 ET (fine), **in EST
  it is 15:30 ET, before the close**, so the freshest bar at 18:30 may be intraday or
  yesterday depending on season. If stale: retry the R2 pull once after 10 min, then
  degrade — publish the Tomorrow's-tape lane only, with a `PRICES STALE (last bar <date>)`
  warning line, and suppress every Today-in-context nugget. Never compute today-pattern
  stats on a stale or partial bar.
- **Sunday runs:** next-session lane previews Monday; Today-in-context lane covers
  Friday's session and must label it "Friday in context" (or be skipped if already
  covered Thursday evening — novelty state handles this).
- **Headless invocation:** clone the denali pattern, not the daily-pitch one — this skill
  needs no web access, so a scoped allowlist is strictly better than
  `--permission-mode bypassPermissions`:
  `scripts/context_headless_settings.json` allowing `Skill`,
  `Bash(python scripts/build_context_state.py:*)`,
  `Bash(python scripts/send_context_slack.py:*)`,
  `Bash(python scratch/context_checks/:*)` (folder-prefix rule for drill scripts —
  **verify this prefix form matches in a manual run**; if it doesn't, fall back to
  bypassPermissions as daily-pitch does, acceptable here because the session ingests no
  untrusted web text), plus `Read, Write, Edit, Glob, TodoWrite`; deny `Read(**/.env*)`,
  credentials/token/key globs, `Edit(**/.env*)` (use Edit-form denies only — the
  Write-form denies in the report-card settings are dead rules; drop them). Omit Grep,
  WebSearch, WebFetch.
- **Model/effort pinned in the .bat** (env-overridable like `PITCH_MODEL`), stamped into
  journal records.
- **Log:** `scripts/logs/market_context_last_run.log` + dated log, same shape as
  daily-pitch.
- **House rule:** eyeball several manual runs (`--dry-run` sender) before registering the
  task, and run at least one full live post to Slack manually before scheduling.

## 11. Files to create / touch

**New_Seasonals (all new):**
- `.claude/skills/market-context/SKILL.md`
- `scripts/build_context_state.py` (engine + trigger registry)
- `scripts/send_context_slack.py`
- `scripts/check_context_delivered.py`
- `scripts/run_market_context.bat`, `scripts/register_market_context_task.ps1`
- `scripts/context_headless_settings.json`
- `data/context_flag_state.json`, `data/context_journal.jsonl` (created at first run)
- `data/context_briefs/` — **gitignored**; briefs stay local (add `data/context_briefs/`
  to `.gitignore` in phase 1)
- `tests/test_context_engine.py` (trigger masks on synthetic panels; stats coherence;
  tag/N validation)
- CLAUDE.md: new dated section with an **Aligned sites — change together** list
  (CONTEXT_UNIVERSE <-> SKILL.md inventory <-> trigger registry <-> sender parser <-> md
  skeleton) and a Guards line.

**New_Seasonals `.env` addition:** `SLACK_WEBHOOK_URL` (same value as denali-dashboard's).

**denali-dashboard:** no code changes. Optionally a one-line note in its CLAUDE.md/memory
that a second product posts to the report-card channel at 18:30.

## 12. Build plan

1. **Engine first.** `build_context_state.py` with ~8 triggers (E1–E4, E11, P1–P3),
   tests, run against history to sanity-check fired-cell counts. Eyeball the JSON.
2. **Skill + composer.** SKILL.md, cell map, drill-down loop, md/json outputs. Run
   manually 3–4 evenings; iterate on nugget selection quality (this is where the product
   lives or dies — the calibration examples in §13 are the bar).
3. **Slack sender + dry runs**, then one manual live post.
4. **Remaining triggers** (E5–E13, P4–P12), novelty layer, quiet contract.
5. **Schedule** via the ps1; watch a full week of logs.

## 13. Calibration examples (what good looks like)

- "**^GSPC — August FOMC, midterm years** [anecdote] — All 6 since 2002 closed down,
  mean −0.5% (sign p=0.016). The following session recovered the decline in 5 of 6.
  Full-sample FOMC-day drift is +0.2% (n=223), so the August-midterm cell runs against
  the base rate."
- "**SPY — down 80 bps the session after a 52w high** [solid] — Next-week return
  historically +1.0% vs +0.1% all-days (t=3.01, n=90, hit 63%). In midterm years the
  cell flips: −0.4% (n=17, suggestive). Era-stable pre/post 2018."
- A correct *rejection*, for contrast: turn-of-month long bias tagged famous-but-dead —
  "flat since 2013 (post-2013 mean +0.02%, t=0.3, n=132)" — published once as its own
  nugget, then novelty-blocked.

## 14. Resolved decisions (McKinley, 2026-08-09)

1. **Same channel** as the report card — reuse the existing `SLACK_WEBHOOK_URL` value.
2. **Briefs are not committed** — `data/context_briefs/` is gitignored, local-only.
3. **Cadence is Sun–Thu**, no Friday edition.
4. **Slack-only delivery** — the brief never lands in the daily-pitch email.
5. **No claims scoreboard** — the journal exists as an audit trail of what was claimed,
   nothing replays it.
