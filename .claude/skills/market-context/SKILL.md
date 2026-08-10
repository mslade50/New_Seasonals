---
name: market-context
description: Produce the evening Market Context brief - 4 to 8 statistical nuggets about the next session and the one that just closed, posted to Slack. Use when running the evening context brief (scheduled ~18:30 ET Sun-Thu, or on request), or when McKinley asks for market context, the context brief, or a rerun of tonight's brief.
---

# Market Context

One Slack post per evening: the most specific defensible history behind
everything tomorrow's session already has scheduled, and behind whatever just
printed today. Scott reads it cold, on a phone, with no follow-up questions.

Spec of record: `market_context_skill_design_2026-08-09.md` (repo root).
Sections 2, 3, 7 and 10 are hard requirements. Read it if anything here is
ambiguous.

## What this is not

- **Not a pitch.** No trades, no legs, no entries, exits or sizing, and no
  advisory verbs. `trim cut reduce consider recommend watch hedge exit fade
  buy sell long short` are banned as imperatives. If a sentence tells Scott
  what to do, it is the wrong sentence. The daily-pitch product exists for
  that and is a different product.
- **Not about the book.** This never reads `data.json`, D1, HedgeFacts or any
  position state. The denali report card owns positions; this one is
  deliberately position-blind, and that is what lets it say things the report
  card cannot.
- **Not a news summary.** No web research in v1. Every claim is computed from
  local history, and a claim you cannot compute does not go in.
- **Not a survey of what is interesting to you.** The engine enumerates; you
  choose from what it enumerated, in writing. See Stage B.

Minutiae is the product. "FOMC days average +0.2%" is worthless to Scott, who
knows that. "August FOMC in midterm years: 6 of 6 down, mean -0.5%, and the
following session recovered the decline in 5 of 6" is the job.

---

## Stage A. Read the sweep

```bash
python scripts/build_context_state.py
```

Writes three files. Read the first one WHOLE:

| file | how to read it |
|---|---|
| `data/context_state.json` | whole, cover to cover. meta, calendar, tape, `cells_index` |
| `data/context_cells.json` | by LOOKUP from your drill scripts, keyed by fingerprint. Never cover to cover |
| `data/context_tape.json` | by lookup, per ticker |

**Check `meta.prices_fresh` first.** False means the price cache is behind the
session the brief is written for, the entire price lane was suppressed, and
tonight is a Tomorrow's-tape-only brief carrying a `PRICES STALE (last bar
<date>)` line. Never compute a "today printed X" claim on a stale bar, and
never quietly write the brief as though the lane merely came up empty.

**Then read `warnings`, then `sweep`.** `sweep.cells_scanned` is the number
the footnote quotes. `sweep.dropped_by_cap` names every cell the engine
truncated; if something dropped there matters, recompute it in a drill script
rather than pretending the cap was the whole surface.

### What the sweep already did

Every cell in `cells_index` arrives with its base statistics computed: N,
mean, hit, t, the sign-test record, the all-days control, the edge over that
control, a pre/post-2018 era split, a four-way cycle-year split, a
`tag_hint`, and for event cells the same-month cell. You are not here to
recompute those. You are here to decide which of them matter and to drill the
ones that do into something specific enough to be worth Scott's attention.

### The one convention that governs everything

Every cell anchors on **today's analogue** and h=1 is **tomorrow**:

- an event landing on the next session anchors on the session BEFORE it, so
  h=1 is the event session's own close-to-close move
- a price state anchors on the session it printed, so h=1 is what the next
  session did afterwards

Forward returns are lag=0 close-to-close on purpose. This is context, not an
entry; the daily-pitch lag=1 rule exists because a pitch has to be tradeable.
State the convention once in the footnote and never per item.

## Stage B. The cell map, before you select anything

Write `scratch/context_checks/<RUN DATE>/00_cell_map.md` FIRST, before
choosing a single nugget.

**Three dates are in play and only one names files.** Tonight's run has a run
date (today), an asof session (`meta.asof_session`, the tape you just read)
and a next session (`meta.next_session`, the one you are previewing). On a
Sunday all three differ. The RUN DATE names the cell-map folder, the brief
files and the delivery check, because it is the only one that is unique per
brief. The next session is what the brief's title names. Every trigger group in `cells_index` gets a written
verdict, and so does every calendar entry inside the next five sessions:

| verdict | means |
|---|---|
| `PUBLISH` | goes in the brief roughly as the engine computed it |
| `DRILL` | interesting but not yet specific enough; needs a script (Stage C) |
| `SKIP(reason)` | examined and rejected, with the reason in the same line |
| `DEAD` | N too small to carry even an anecdote, or the cell is degenerate |

The dismissals matter as much as the picks. A trigger you do not look at has
to be visibly dismissed with a reason, never silently absent. `sweep` counts
what fired; the map is what proves you read it.

**No map, nothing publishes.** `send_context_slack.py` checks the file exists
for the brief's date and refuses to post without it. This is the same
anti-recall rule as the daily-pitch surface map, and it exists because the
alternative is writing tonight's brief from whatever you already believed
about the market this morning.

The map is also where you note the two engine hints you may not simply
inherit:

- `tag_hint` is a floor, not a grant. You may downgrade a cell. You may never
  upgrade past what the numbers support.
- `bh_pass` prices the SWEEP. A cell that arrived as a pre-specified famous
  hypothesis (FOMC drift, turn-of-month, pre-holiday drift) was not found by
  the search and does not owe it a correction. Say which it is in the map.

## Stage C. Drill

For every `DRILL` verdict, write a real `.py` in the day folder and run it.
This is where the product lives. The engine computed base cells; you compute
the crossings and follow-ons that make one worth reading.

Run them with the exact relative form, from the repo root:

```bash
python scratch/context_checks/<YYYY-MM-DD>/01_whatever.py
```

The unattended run's allowlist matches that command by PREFIX
(`scripts/context_headless_settings.json`). An absolute path or a quoted path
does not match it and the scheduled evening stalls on a permission prompt
nobody is there to answer.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
```

`pitch_lab` is the substrate and it is READ-ONLY: `load_prices`,
`close_panel`, `load_events`, `fwd_ret`, `declusters`, `local_control`,
`summarize`, `era_split`, `sign_test`, `cluster_note`, `horizon_scan`,
`episode_paths`. `scripts/seasonal_edge.py` adds `seasonal_window_returns`
(with `cycle_phase_filter`), `binom_p_greater` and `benjamini_hochberg`. Do
not re-derive any of it in the day folder.

Two conventions carry from `pitch_lab` and both have drawn blood before:
returns go into `summarize` as FRACTIONS and come out as PERCENT, and ATR is
Wilder-14.

What a drill is for:

1. **Condition the cell further.** August FOMC -> midterm-year August FOMC.
   Down-day-after-a-52w-high -> the same in midterm years. This is the single
   highest-value move available and it is the reason the engine stops at base
   cells: the interaction space is combinatorial and choosing well inside it
   is judgement, not a grid.
2. **The follow-on.** `episode_paths` and `horizon_scan`. Did the next
   session reverse the move? Where in the week does the effect sit? "5 of 6
   recovered it the next day" is a better sentence than a second mean.
3. **Era stability** (`era_split` at 2018) and **concentration**
   (`cluster_note`). If the top two episodes carry the mean, the brief says
   so or the nugget dies.
4. **Controls**, whenever the nugget's claim is an EDGE rather than a level:
   the instrument's own drift over the same span, all days full history, and
   the local +/-126td neighbourhood (`local_control`). A claim of "above its
   base rate" without a base rate is not a claim.

Budget: 3 to 8 drill scripts is a normal evening. Do not silently shrink this
stage to save tokens.

## Stage D. Select, compose, publish

### Selection

Ship **4 to 8** nuggets, and rank by:

1. relevance to the next session
2. specificity of the cell
3. strength (|t|, or the exact sign-test p at small N)
4. novelty

In that order. A `[suggestive]` cell about tomorrow's exact setup beats a
`[solid]` generic one, every time.

- At least one nugget from each lane when both lanes fired.
- Tomorrow's-tape leads when a top-tier event (FOMC, CPI, NFP) is the next
  session.
- **Quiet-tape contract**: no top-tier event and no fired price-state trigger
  means a 3-line QUIET TAPE note (next scheduled events, one seasonal-position
  line) and nothing else. Never pad to reach four.

### The honesty contract (hard requirement)

- **Every nugget carries N and its cell definition.** Always.
- **t only when N supports it.** At N under 15 a t-stat is noise; quote the
  record and its exact sign-test p instead ("6 of 6 down, sign p 0.016").
- **Three tags, and the budget is enforced**: `[solid]` N>=50, |t|>=2.5 and
  era-stable; `[suggestive]` N 15-50 or single-era; `[anecdote]` N<15, **at
  most two per brief, and an anecdote may never be the headline.**
- **Era honesty.** Any nugget whose sign flips across 2018 says so or is
  dropped. "Famous and dead since 2013" is itself a publishable nugget.
- **Concentration.** If the top two episodes carry the mean, say so.
- **Multiplicity.** The footnote quotes `sweep.cells_scanned`. A nugget whose
  only support is a swept p-value needs `bh_pass` before it can be tagged
  `[solid]`. Pre-specified cells are exempt, and the cell map says which.
- Small N is not a kill. A clean record with a real per-event magnitude and a
  plausible mechanism is publishable as context, labelled honestly.

### Novelty

`novelty.flags[<fingerprint>]` carries the repetition state.

- `repeat_blocked: true` means it published inside the last 5 sessions and
  the number has not moved. It does not go in again. One exception: an event
  escalating from "upcoming" to "next session" earns one re-telling, and it
  must add new specificity, not just a new countdown.
- Countdown re-tellings are banned outright. "3 days to CPI" then "2 days to
  CPI" with the same stat is the failure mode this rule exists for.
- `delta_suppressed: true` means the state file was missing or unreadable.
  Publish, but make no NEW or first-time claims tonight.

### Prose rules

House style, and Scott reads every word:

- No em dashes. No "it's not X, it's Y". No AI throat-clearing, no hedging
  filler, no restating the setup.
- One fact appears once. All times ET.
- Body budget **250 to 400 words**. The headline is one sentence, one cell,
  one number.
- Name the mechanism when there is one and say so when there is not. "No
  mechanism, pattern only" is an honest and useful line.

### Output

Write BOTH files to `data/context_briefs/`, named for the RUN date:

`YYYY-MM-DD.md` — the sender parses this, so the numbered head format
`1. **Title** [tag]` is load-bearing and exact:

```markdown
# Market Context — Tuesday 2026-08-11

**Headline:** <one sentence, one cell, one number>

## Tomorrow's tape
1. **<subject> — <cell>** [solid]
   <2 to 3 sentences: the stat with N and hit and t or sign p, the sharper
   split, the follow-on. No advice.>

## Today in context
1. **<subject> — <pattern>** [suggestive]
   ...

## Calendar
- <next 5 td of macro_events with times ET>

---
*Cells scanned: 214. Conventions: close-to-close forward returns from the
signal close, 1999+, adjusted, NYSE trading days. Descriptive post-selection
statistics, not out-of-sample forecasts.*
```

`YYYY-MM-DD.json` — the sender's authoritative source for `quiet`, and the
journal record:

```json
{"asof": "YYYY-MM-DD", "quiet": false, "prices_stale": false,
 "headline": "...", "cells_scanned": 214,
 "nuggets": [{"lane": "tomorrow|today", "trigger_id": "E:fomc_decision",
              "fingerprint": "E:fomc_decision|^GSPC|k1", "subject": "^GSPC",
              "cell": "August FOMC, midterm years", "tag": "anecdote",
              "n": 6, "mean_pct": -0.5, "hit": 0.0, "t": null,
              "sign_p": 0.016, "era_note": "...", "concentration_note": "...",
              "drill_script": "scratch/context_checks/.../fomc_august.py",
              "text": "..."}],
 "calendar_next_5td": [...], "conventions": "..."}
```

### Publish

```bash
python scripts/send_context_slack.py --dry-run   # writes .slack.json, no post
python scripts/send_context_slack.py             # posts
```

The sender gates on the cell map existing, on the brief being dated today,
and on the tag budget. Fix a rejection by fixing the brief, never by
loosening the gate. A successful post advances
`data/context_flag_state.json` and appends one record per nugget to
`data/context_journal.jsonl`.

## The trigger inventory

**Aligned pair**: this table and `PRICE_TRIGGERS` / the event sweep in
`scripts/build_context_state.py` change together. A trigger added there
without a line here is invisible to the cell map.

**Event lane** (anchored on the session k td before the event, k in 1..3):

| id | cell |
|---|---|
| `E:fomc_decision` `E:fomc_minutes` `E:fomc_intermeeting` | Fed dates |
| `E:cpi` `E:nfp` `E:ppi` | the top-tier prints |
| `E:opex` `E:quad_witching` `E:vix_expiry` | expiries |
| `E:jackson_hole` `E:election` | set pieces |
| `E:month_end` `E:turn_of_month` | last 3 / first 2 sessions of a month |
| `E:holiday_pre` `E:holiday_post` | either side of a market closure |
| `E:weekday_month` | the bare day-of-week x month cell, fires every day |
| `E:seasonal_doy` | same trading-day-of-year in prior years, all years and same cycle phase |

**Price lane** (anchored on the session the state printed; suppressed
entirely when `meta.prices_fresh` is false):

| id | cell |
|---|---|
| `P1/P1b` `P2/P2b` | first 52w high / low in 30+ or 90+ calendar days |
| `P3/P3b/P3c` | 50bp or 100bp reversal the session after a 52w extreme |
| `P4` | \|z10\| >= 2 |
| `P5/P5b` | 5d or 21d return in the top or bottom 5% of its year |
| `P6` | single session >= 2 ATR |
| `P7/P7b` | 5+ consecutive up or down closes |
| `P8` | first 200d MA cross in 63+ sessions |
| `P9..P9f` | stocks and bonds together, dollar and gold together, VIX up on an up day, 10y-5y curve moves |
| `P10/P10b/P10c` | VIX term structure inverting or un-inverting, VIX +10% |
| `P11/P11b` | breadth crossing 80% or 20% above the 200d |
| `P12:<event>_<label>` | today's macro print conditioned on above/below consensus |

## Universe

Macro only: US and global indices, rates and credit, FX, futures, vol, and
crypto for context. `CONTEXT_UNIVERSE` in the engine is the single aligned
site. Sector ETFs are breadth context only and are never a nugget subject.
Single names are out of scope entirely.

Two universe facts worth knowing before you write:

- Foreign CASH indices are excluded from the EVENT lane. ^N225 and ^FTSE
  close before an 08:30 ET print or a 14:00 ET decision, so their "event day"
  bar does not contain the event. They stay in the price lane, where their
  own session is the right clock.
- `^GSPTSE` is in the spec and absent from `master_prices`; `LBS=F` stopped
  printing in 2023. Both surface in `meta.universe_missing` or as stale.

## Standing down

If the price cache is broken AND the calendar has nothing scheduled, say so
and ship the Tomorrow's-tape lane with the stale warning. Never ship nothing:
silence is indistinguishable from a broken scheduled task. Never ship a
"today printed" claim computed on a stale bar.
