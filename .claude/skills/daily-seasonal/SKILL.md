---
name: daily-seasonal
description: Produce the Daily Seasonal - up to three falsified seasonal trade ideas across every rank outlier, cycle cell and calendar cell, with horizons from 3 to 63 sessions and exits developed from the pattern's shape. Use when running the morning seasonal run (scheduled 07:00 ET after the Daily Pitch, or on request), or when McKinley asks for seasonal ideas, the seasonal slate, or a rerun.
---

# Daily Seasonal

Deliver **up to three** seasonal trade ideas, found on the whole seasonal surface and
interrogated against the data before they reach McKinley. He reads them, says yes or no
per idea, and places orders. There is no conversation: the ideas must be finished when
they arrive.

The run is 07:00 ET on weekdays, after the Daily Pitch has published. Three is the full
slate. One or two ship when that is what survived, with the empty slots paid for; a
morning where nothing survives ships a stand-down. Never pad, never publish nothing.

Design of record: `docs/seasonal_agent_design_2026-09-30.md`. Where this file is silent,
the Daily Pitch skill (`.claude/skills/daily-pitch/SKILL.md`) governs.

## What this is

A falsification desk that owns the seasonal surface end to end. The board ranks cells;
this agent tries to kill trades. On 2026-09-30 all four board tickets died under the
pitch battery: WMT and GS were sector beta, TXN flipped sign at the tradeable entry, and
TRV was a real 26-year pattern, decayed since 2016, sitting in its losing regime branch.

## What this is not

- **Not the site board.** The board is a scanner and, for this agent, a candidate feed.
  It stays live on the site. A board ticket is a candidate, never evidence: every one is
  re-checked from scratch from the entry the order would actually fill at.
- **Not the systematic book.** A trade that is materially a St OS Sznl or Weak Close
  Decent Sznls signal (see `book.staged_signals`) is dead on arrival unless it adds a
  real twist (instrument, legs, side, or a meaningfully different window), and the
  write-up names the overlap.
- **Not the pitch.** An idea matching `history.pitch_recent_fingerprints` (what the
  pitch shipped inside 10 sessions) is dropped, or carries a `changed_since` sentence
  saying what materially changed. The pitch negative registry is read-only here; this
  product keeps its own.
- **Not a place orders happen.** Nothing is placed without McKinley's Y on the tab.

False positives are fine. He filters. Unchecked ideas are not fine.

## Stage A. State (deterministic, already code)

```bash
python scripts/build_seasonal_state.py
```

Writes `data/seasonal_state.json` and `data/seasonal_tape.json`. Read the state whole:

| key | what it carries |
|---|---|
| `asof`, `warnings` | read warnings FIRST; a stale cache changes what you may claim |
| `calendar`, `risk` | `events`, `next_by_type`, `cycle_year`; `fragility`, `pc_fear`, `signals`, `exposure_leg` |
| `book`, `earnings` | `staged_signals`, `event_sleeve`; report dates inside 63 sessions |
| `ranks` | `rank_date`, `horizons`, `n_tickers`, `outliers`, `by_class`, `by_sector` |
| `calendar_cells` | turn-of-month, holiday adjacency, weekday-of-month, macro-event offsets inside 10 sessions, month-of-year, cycle-year |
| `board` | today's site seasonal tickets (candidate feed only) |
| `history`, `watchlist`, `scoreboard` | `recent_fingerprints` (own), `pitch_recent_fingerprints`; own near-misses; graded record |
| `negative_registry`, `pitch_negative_registry` | own (writable), pitch (read-only) |

Each outlier carries `ticker, class, sector, ranks{5,10,21,63,126,252}, side,
agree_horizons, path_turn_td, cycle{phase,n,k,mean_atr}, all_years{n,k,mean_atr},
ext{r5_pct,r10_pct,r21_pct}, atr14, close, adv_usd_21d`. The `cycle` and `all_years`
stats are already entry-anchored at T+1 over the smallest agreeing horizon. They are a
screen. The check still recomputes them at the entry the idea will actually use.

Read the tape whole and SORT it. Do not look up the names you walked in with.

## The lab and the data map

Every check script starts from `pitch_lab.py` at the repo root:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
```

Conventions (its docstring is the authority): returns are FRACTIONS in and PERCENT out
of `summarize`; entry is signal on close D, MOC on D+lag, lag=1 is the real order; ATR
is Wilder-14; controls are own-drift, all-days and local +/-126td; N < 15 is judged by
the exact `sign_test`, not a t-stat. `battery()` is round 1; `horizon_scan` and
`episode_paths` are round 3. Do not re-derive any of this in the day folder. A reusable
helper gets promoted into `pitch_lab.py` with a test the same morning.

Seasonal machinery that already exists, and is not rebuilt:

| need | asset |
|---|---|
| ranks, 1,025 names, 6 horizons | `atr_seasonal_ranks.parquet` (repo root) |
| day-of-year windows, cycle cohorts, expected path | `scripts/seasonal_edge.py` (`seasonal_window_returns`, `expected_seasonal_path`, `cycle_phase`) |
| turn-of-month, holiday, weekday-month, cycle sweeps | `scripts/build_context_state.py` (`month_window_anchors`, `holiday_adjacent_anchors`, `sweep_seasonal_cells`) |
| event calendar 2000-2027 | `pitch_lab.load_events` (`data/macro_events.csv`) |

`month_window_anchors` counts the incomplete current month as a month end: drop the
current month from any month-end history. The rest of the data map is the pitch skill's
table and applies unchanged.

## Stage B. Survey the whole surface, then select from it

Write `scratch/seasonal_checks/<YYYY-MM-DD>/00_surface_map.md` FIRST. It enumerates the
whole surface and gives every cell a verdict. A dismissal needs a reason; a cell with no
line is a cell never looked at. No map, no publish: the gate reads the disk on the ideas
path and the stand-down path.

### B1. Four enumerations, each exhaustive

**1. Every rank outlier, by asset class AND by sector.** `ranks.by_class` and
`ranks.by_sector` are the checklist. Every class and every sector with outliers gets a
verdict line naming its outliers, their side, how many horizons agree, and CHECK or PASS
with a reason. A class with outliers and no verdict line fails the survey. Note
`path_turn_td` where the expected path turns inside 5 sessions: an entry at the turn and
an entry today are different trades.

**2. Every calendar cell inside 10 sessions, crossed with the ten asset classes.** Take
every entry in `calendar_cells`, not just the next one, and cross it with:

| class | proxies |
|---|---|
| US large | SPY, QQQ, ^GSPC, ^NDX |
| US small and breadth | IWM |
| rates | TLT, IEF, ^TNX |
| credit | HYG, LQD |
| gold and miners | GLD, GDX |
| other metals | SLV |
| energy | USO, UNG, DBC |
| dollar and FX | UUP, DX-Y.NYB |
| international | EFA, EEM, FXI |
| volatility | ^VIX, ^VIX3M, ^MOVE, SVXY |

You are not checking every cell. You are deciding in writing which deserve a check and
why the rest do not. "Turn-of-month x credit: not examined" is legal only when the next
words say why. Cycle year is a conditioner on every cell above, not a cell of its own.

**3. Every active watchlist entry**, with today's value of its trigger number. Trigger
moved: CHECK. Unchanged: PASS, still citing today's value. Expired: prune at the
post-publish rewrite.

**4. Every board ticket**, one verdict line each. The board is a candidate feed; a
ticket that deserves a check gets one from scratch, and a ticket that does not says why
(sector beta on its face, inside a staged book signal, entry already past the window).

### B2. Select 8 to 12 candidates from the map

Name the novelty axis on each:

| axis | what it means |
|---|---|
| `rank_outlier` | a name at a rank extreme with adjacent horizons agreeing |
| `cycle_cell` | a seasonal window conditioned on the presidential-cycle year |
| `calendar_cell` | turn-of-month, holiday adjacency, weekday-of-month, event offsets, month-of-year |
| `path_turn` | entry at the expected path's nadir or peak rather than today |
| `relative_value` | long X / short Y where only the legs were ever studied as seasonals |
| `instrument_translation` | a known seasonal through a better vehicle (futures, a cleaner ETF) |
| `inversion` | a documented seasonal that flips sign under a stated regime |
| `historical_analogue` | nearest-neighbour years to this one, with honest N |

Horizons run 3 to 63 sessions. Instruments: US equities, ETFs and futures (never reject
an idea for lacking an ETF wrapper). No options.

**Coverage floors. Requirements, not targets.**

- 8 to 12 candidates touching **at least four asset classes** from the table.
- **At least one non-equity vehicle** and **at least one short**.
- **At least one calendar-cell candidate and at least one rank-outlier candidate.**
  These are different search modes and both get opened.
- Every calendar cell inside 10 sessions and every class with outliers appears in the
  map with a verdict.

Coverage governs where you LOOK; what ships is a separate question. Standing down after
a complete survey is a fine morning; shipping three equity longs without having opened
rates, metals or FX is not.

Before spending a check, run the candidate against
`data/seasonal_agent_negative_registry.md` and, read-only,
`data/pitch_negative_registry.md`. A collision does not kill automatically, but the
write-up says what is different. Once `scoreboard` carries graded ideas, read its
per-axis and per-grade splits and note the read in the map; while the count is a
handful, say so and move on.

## Stage C. Falsification (the point of the whole thing)

Hand the candidates to checking agents whose brief is to **kill** them: two or three in
parallel, three or four candidates each, then one red-team pass over the survivors
together. Every script goes in `scratch/seasonal_checks/<YYYY-MM-DD>/`.

Each checker's prompt carries: the candidate block from the surface map verbatim with
its axis and cell; the paths to the map and the day folder; the adjacent registry
entries (both registries, not the whole files); the import boilerplate and conventions
line above; the five seasonal rules below with their worked examples; and the standing
brief, **your job is to kill this; a survivor is a failure to kill, not a success to
celebrate.** It returns per candidate a verdict (KILL / SURVIVES / NEAR-MISS), the
decisive numbers, its script paths, which substantive kill it was, and for every
NEAR-MISS **the number it turned on**.

### Round 1. The pitch battery

The pitch's round 1 unchanged (`pitch_lab.battery()`): pattern against own-drift and
all-days controls, N and worst window, era stability, registry collision, book overlap,
cost against edge, volatility events inside the window.

### The five seasonal rules. Every check answers all five.

These were written from the 2026-09-30 kills (`scratch/pitch_checks/2026-09-30/` sb1,
sb2, sb4, sb5). Each worked example is a real number from that morning.

**1. Entry-anchored windows only.** The historical window starts at the close the order
would fill at, never at the day-of-year close and never at lag 0. If the idea waits for
a path turn, measure from the turn. Example (`sb2_txn.py`): the board's TXN short read
"midterm 5/6 lower"; measured from the T+5 entry it was 1/6. The whole edge lived in
sessions the order could never own.

**2. Index and sector residual, with its own hit rate and sign test.** Report the return
net of SPY and net of the sector ETF, each with hit count and `sign_test` p. A zero
residual means the pattern is the index's or the sector's: pitch it as that seasonal or
kill it. Examples: WMT's +1.98% window was XLP's; WMT minus XLP was +0.05%, 3/6. GS was
XLF, residual 10/16. And the sector itself gets the same test: XLP minus SPY over its
window was -0.02%, 11/26 (`sb5_xlp.py`).

**3. Cycle cells show drop-best, drop-two-best and the non-cycle cohort.** A conditioner
that adds nothing has to be visible. Example (`sb1_trv.py`): TRV October in midterm
years +7.66%, 5/6; drop-two-best +4.32%, 3/4; non-midterm +4.00%, 16/20. The cycle added
size, the pattern stood without it, and removing the best years did not flip the sign:
that passes this rule. A cell where drop-best flips the sign is a kill (the registry
holds several).

**4. Regime branch.** Split history by SPY within 2% of its 52-week high at entry versus
not, and by the fragility dial where the PIT history allows (state which vintage). Say
which branch today sits in. Example (`sb4_redteam.py`): TRV near-high years +1.10%, 7 up
/ 5 down, which is its own drift; off-high years +8.05%, 14 up / 0 down. Today SPY was
near its high. The whole edge sat in the other branch: killed.

**5. Recency.** The last 10 years individually, each with its return and max adverse
excursion in ATR. A long record with a coin-flip last decade is grade C at best. Example
(`sb4_redteam.py`, `sb1_trv_dev_b.py`): TRV 2016-2025 went 6/4, sign p 0.61 against its
base rate, and the losers carried 5 to 7 ATR of MAE. The 26-year record was not today's
trade.

### Round 2 is mandatory for round-1 survivors

As the pitch: decluster and `cluster_note` concentration (top two years, one year);
definition neighbours (shift the window start +/-3 sessions, the horizon to its
neighbours, the rank threshold); era split pre/post 2018 plus the regime the mechanism
implies; gate attribution (run it without the rank or cycle gate; if the gate does not
move the result, nothing may be credited to it). A `b`/`c` suffix on the same script, or
a section inside it.

### Round 3. Develop the exit from the shape

A survivor is a pattern; a pitch is a trade. Every composed idea gets a development
script (`_dev` suffix), and its path goes in `evidence.dev_script`. The publisher
requires it.

1. **Horizon.** `pitch_lab.horizon_scan` across 1 to 63 sessions. `horizon_td` and
   `exit.time_td` come from this table. If the edge peaks at h=14 and fades by h=21, the
   idea is 14 sessions.
2. **Where the dip sits.** `pitch_lab.episode_paths` on winners and losers. If winners
   typically draw down before they pay, that decides both the entry (MOC now, or a
   close-anchored LIMIT k ATR lower, or wait for the path turn) and the stop.
3. **Entry form.** MOC against a close-anchored LIMIT, as WHOLE variants (fill rate plus
   conditional stats), never a marginal-fill decomposition.
4. **Stop from the adverse-excursion table.** Tabulate intraday MAE in ATR per year and
   the result under 0.8 / 1.0 / 1.3 / 1.6 / 2.0 / 2.5 / 3.0 ATR stops, then choose one
   of three forms and say why:
   - **time-only**, sized off a 3.0 ATR catastrophe distance (`stop_atr_for_sizing` >=
     3.0);
   - **a real stop** at a level the table supports;
   - **a trail** armed after a stated MFE (`exit.trail`).

   Never a default 2:1 ticket. Example (`sb1_trv_dev_b.py`): TRV's 0.8 ATR stop was
   touched in 19 of 26 years, 14 of them winners, and cut the mean from +4.84% to
   +1.23%; a 1.6 ATR stop left +0.79%; no stop kept +4.84%; a 3 ATR stop triggered 8/26.
   The shape asked for time-only with a catastrophe distance, which the board's fixed
   ticket cannot express.
5. **Loser paths.** `what_kills_it` quotes a number from `episode_paths` ("losers were
   down 1.5 ATR by day 6 and never recovered; a close below X by then says the window
   has failed"), never a generic risk.

### The red-team pass

One agent over the developed survivors together: leg-return correlation over the hold
window (two seasonal longs in one sector are one trade: merge or drop); overlap with the
systematic layers in the state (staged signals, event sleeve, the ledger; live broker
positions are deliberately absent); cost at the developed entry form; and the strongest
single argument against each idea. If that argument would convince you, it is a kill.

**Hard requirement.** Every delivered idea carries evidence computed fresh this morning
and a `survived` line naming a consideration that could have killed it and did not.

### Small N is not a kill

The pitch doctrine applies verbatim, and seasonals need it more than anything: a cycle
cell has one observation every four years. A 6-0 record is sign p 0.016. Quote the
record, its exact `sign_test` p and the per-event edge. No idea needs a t-stat; leave
`t_stat` null below 15. "Insufficient N", "not significant" and "t below 2" are illegal
kill reasons standing alone (the publisher prints `KILL-LINT:`). A small-N kill names a
substantive failure and quotes its number: no mechanism, a gate that does not filter,
definition fragility, sign instability across eras, a residual that is zero (rule 2),
the wrong regime branch today (rule 4), a dead last decade (rule 5), or cost.

Multiplicity corrections price a search: charge them to a sweep's best occupant, never
to a cell that arrived with a mechanism attached.

Kills are journaled with reasons, one footer line each. Do not quietly shrink this stage
to save tokens.

### When one or two survive

Ship them, without asking. The empty slots are paid for in a `short_slate` block, priced
exactly as in the pitch: **two named kills per empty slot**, at least 8 candidates over
at least 4 novelty axes and 4 asset classes, a reason of at least 120 characters, and 1
to 3 near-misses each with **the number it turned on**. Nothing else relaxes. A thin
sweep cannot fill this block: go back to stage B.

```json
{
  "asof": "YYYY-MM-DD",
  "ideas": [ ... one or two, full idea schema ... ],
  "short_slate": {
    "reason": "what the rest of the morning looked like and why it came back empty",
    "candidates_considered": 11,
    "axes": ["rank_outlier", "cycle_cell", "calendar_cell", "relative_value"],
    "asset_classes": ["us_large", "rates", "gold_miners", "energy"],
    "closest": [{"title": "Long TRV into late October",
      "decisive": "off-high branch +8.05% 14/0; near-high +1.10% 7/5",
      "why_died": "SPY within 2% of its 52w high today; 2016-2025 6/4, sign p 0.61",
      "script": "scratch/seasonal_checks/YYYY-MM-DD/c1_trv.py"}]
  },
  "killed": [{"title": "...", "reason": "...", "novelty_axis": "..."}]
}
```

### When nothing survives

Ship a stand-down: NO TRADES email led by the near-misses, an empty tab, a `stand_down`
journal record. The publisher enforces, as for the pitch: at least 8 candidates over at
least 4 axes, at least 4 asset classes, at least 6 named kills, a reason of at least 120
characters, a `checks_dir` holding real `.py` checks and `00_surface_map.md`, and 1 to 3
near-misses with their numbers. "Nothing worth trading" is a claim about the whole
surface.

The payload is the short-slate shape with `"ideas": []`, the block keyed `stand_down`,
and `"checks_dir": "scratch/seasonal_checks/YYYY-MM-DD"` inside it.

## Stage D. Compose

Pick the best three by risk and reward over their horizons, or however many survived.
Grade honestly:

| grade | meaning |
|---|---|
| A | N >= 50, abs(t) >= 2.5, holds across eras, verified fresh today |
| B | real pattern, N 15 to 50 or single era, verified fresh today |
| C | context: N < 15, cycle cell or analogue reasoning. **At most one per day** |

Most seasonals are B or C: 26 years of one October is N=26, and one cycle phase of it is
N=6. A cycle cell with N=6 competes for the C slot on edge size and mechanism, per the
small-N doctrine, and it may win the morning. A grade-C evidence line quotes the record
and its sign p ("5-1, sign p 0.11") plus the per-event edge.

### Sizing

Default **30 bps** of risk per idea. The agent may set 15 to 50 bps by conviction, under
these rules:

- **40 to 50 bps** only for grade A or B with a clean last-10-years record (rule 5), a
  positive residual against BOTH the index and the sector (rule 2), and today in the
  favourable regime branch (rule 4).
- **15 to 20 bps** for grade C, for any idea in the unfavourable regime branch, and for
  any idea whose mean edge is under 2x its catastrophe risk.
- Otherwise 30.

`stop_atr_for_sizing` is the real stop distance when the idea has one (>= 1.0), and the
catastrophe distance (>= 3.0) when the exit is time-only or trailed. The idea's `thesis`
or `evidence.summary` must say why the size is not 30 whenever it is not.

### Prose rules

- No em dashes. No "it's not X, it's Y". No AI throat-clearing or filler.
- The thesis is three to five sentences with a **variant perception** and **who is on
  the other side** (the flow behind the calendar: rebalancing, window dressing, tax
  timing, index events, insurers' cat-season books). "This window has been green" is not
  a mechanism.
- Evidence is numbers with N, control, residual, regime branch and era note. One table
  at most.
- `what_kills_it` is an observation inside the hold, quoting the loser paths.

### Schema

Write `data/seasonal_agent_ideas.json`:

```json
{
  "asof": "YYYY-MM-DD",
  "ideas": [
    {
      "title": "one line, no ticker soup",
      "grade": "A|B|C",
      "novelty_axis": "cycle_cell",
      "horizon_td": 21,
      "legs": [
        {"ticker": "XLE", "side": "LONG", "weight": 1.0},
        {"ticker": "SPY", "side": "SHORT", "weight": 1.0}
      ],
      "entry": {"type": "LIMIT", "anchor": "CLOSE", "atr_mult": -0.5, "fill_window_td": 3},
      "exit": {"time_td": 21, "time_order": "MOC", "target_atr": null, "stop_atr": null,
               "trail": {"arm_atr": 2.0, "trail_atr": 1.5}},
      "sizing": {"mode": "risk_bps", "risk_bps": 30, "stop_atr_for_sizing": 3.0},
      "thesis": "...",
      "evidence": {"summary": "...", "n": 26, "t_stat": null, "window": "2000-2025",
                   "control": "own drift and XLE minus SPY", "era_note": "...",
                   "table": [["cohort","N","avg","hit"], ["...","...","...","..."]],
                   "script": "scratch/seasonal_checks/YYYY-MM-DD/c3_xle.py",
                   "dev_script": "scratch/seasonal_checks/YYYY-MM-DD/c3_xle_dev.py"},
      "survived": "the consideration that could have killed it and did not",
      "what_kills_it": "...",
      "overlap": "what the book or sleeves already hold that correlates, or None",
      "changed_since": "only when re-pitching inside 10 td, or matching a pitch fingerprint"
    }
  ],
  "killed": [{"title": "...", "reason": "...", "novelty_axis": "..."}]
}
```

`trail` is optional. Vocabularies, and nothing else is legal:

```
novelty_axis rank_outlier | cycle_cell | calendar_cell | path_turn |
             relative_value | instrument_translation | inversion |
             historical_analogue
entry.type   MOO | MOC | LIMIT
             LIMIT also needs anchor (OPEN|CLOSE), atr_mult (signed),
             fill_window_td (1..10)
exit         time_td is ALWAYS present (1..horizon_td), time_order MOC|MOO,
             target_atr and stop_atr optional
exit.trail   optional {arm_atr, trail_atr}: trail at trail_atr ATR once MFE
             reaches arm_atr. Makes the idea MANUAL for auto-placement
horizon_td   3..63
sizing.mode  risk_bps (default 30, allowed 15..50) | nav_pct (index or carry
             constructions; say why in the thesis)
             stop_atr_for_sizing >= 1.0; >= 3.0 for time-only or trailed
             exits (the catastrophe distance)
             HARD CAPS (the validator fails the publish): risk_bps <= 100
             per idea, ATR risk <= 60 bps per idea and <= 150 bps across
             the slate, nav_pct <= 0.5, at most 4 legs per idea
legs         side LONG|SHORT; a futures leg adds sec_type "FUT", contract,
             proxy_ticker and multiplier
```

ATR is Wilder-14 on the traded instrument, never the book's simple mean. Placement
follows from the grammar: a CLOSE-anchored LIMIT places with its stop and target; MOO
and MOC place with a time exit only, and a price stop or target on them makes the idea
manual; an OPEN-anchored LIMIT goes in the post-open pass; any futures leg is manual;
**any `exit.trail` is manual.**

## Background-agent completion gate

Every checker, red-team agent and background task started in this run is joined before
publishing: keep their task IDs, wait for each to reach a terminal state, read every
result, and fold its evidence or failure into the verdict. A time-limited wait is not
completion. If an outer deadline nears, finish with a fully evidenced stand-down; never
publish unchecked ideas. The skill returns only after the publish command has finished
and its validation result has been read.

## Publish

```bash
python daily_pitch.py --product seasonal --ideas data/seasonal_agent_ideas.json --validate-only   # check
python daily_pitch.py --product seasonal --ideas data/seasonal_agent_ideas.json                   # ship
```

The publisher validates, sizes every leg, captures yesterday's Approve cells off the
**Seasonal Agent** tab before overwriting it, sends the **Daily Seasonal** email,
rewrites the tab, and appends to `data/seasonal_agent_journal.jsonl`. Iterate on
`--validate-only` until it is silent. `--dry-run --html-out preview.html` renders
without sending. Fix validation errors by fixing the idea, never by loosening the
grammar.

The gates read the disk, not the payload, exactly as the pitch's do:

- `scratch/seasonal_checks/<asof>/` exists and holds `00_surface_map.md` plus at least
  one `.py` check;
- every `evidence.script` and `evidence.dev_script` resolves to a file inside that
  folder (a path into another day's folder fails as stale);
- every composed idea carries `dev_script`;
- fewer than three composed ideas needs its `short_slate` block.

A directed idea (non-empty `directed_by`, McKinley's wording) skips the survey and
`dev_script` rules and none of the others.

## After publishing

1. Append any reusable kill to `data/seasonal_agent_negative_registry.md`: the cell, the
   rule it failed (1 to 5 above or a battery item), the number, and the script path.
   Never write to `data/pitch_negative_registry.md`.
2. Update `data/seasonal_agent_watchlist.json`: append today's near-misses and `closest`
   entries with title, cell, **the trigger number** ("SPY closes more than 2% below its
   52w high before the window opens"), script path, source and expiry (default 15 td; a
   dated seasonal may park to its next window). Remove entries that expired or fired
   today.
3. Leave every check in `scratch/seasonal_checks/<date>/` as the audit trail.
4. Grading runs on its own:

```bash
python scripts/grade_pitch_journal.py --product seasonal
```

It replays every idea, approved or declined, against its stated entry, exits and trail,
and rewrites `data/seasonal_agent_scoreboard.json`. That is where the product earns its
keep or gets retuned or killed.

## Standing down

If the price cache is broken or the rank file is stale (`ranks.rank_date` not the prior
session), the seasonal run does not ship on stale data. Say so and stop. A missed
morning delivers nothing; it never delivers stale ideas late in the session.
